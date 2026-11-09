"""One-way electromagnetic response driven by a frozen Li ESF snapshot.

This module is the deliberately one-way bridge used before enabling any
magnetosphere--ionosphere feedback.  The Li density closure is evaluated once
when :class:`FrozenLiDrive` is constructed.  Subsequent field-line right-hand
side evaluations receive only a prescribed, smoothly ramped electric-potential
drive; they cannot return FAC or otherwise modify/re-evaluate the Li state.

The electrostatic Li model advances the stream function ``psi = Phi / B``.
Consequently the dimensional electric potential supplied to an EM model is

``Phi [V] = B [T] * psi [m^2/s]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
LineState = TypeVar("LineState")
LineRhs = TypeVar("LineRhs")


def _immutable_float_array(
    name: str,
    values: ArrayLike,
    *,
    ndim: int | None = None,
) -> FloatArray:
    """Return an owning-independent array backed by immutable bytes.

    A normal ``array.setflags(write=False)`` can be reversed when the array
    owns its memory.  Reconstructing from a ``bytes`` buffer makes read-only
    storage part of the ownership chain, so clients cannot silently re-enable
    writes and alter a stored snapshot or drive.
    """

    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(
        array.shape
    )


@dataclass(frozen=True)
class LiSnapshot:
    """Deep-copied, immutable 2-D plasma-density snapshot from the Li model."""

    density_m3: FloatArray
    time_s: float = 0.0

    def __post_init__(self) -> None:
        density = _immutable_float_array("density_m3", self.density_m3, ndim=2)
        if np.any(density < 0.0):
            raise ValueError("density_m3 must be non-negative")
        time = float(self.time_s)
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("time_s must be non-negative and finite")
        object.__setattr__(self, "density_m3", density)
        object.__setattr__(self, "time_s", time)

    def density_for_closure(self) -> FloatArray:
        """Return a writable copy for the one permitted Li closure solve."""

        return np.array(self.density_m3, copy=True)


@dataclass(frozen=True)
class MagneticNormalization:
    """Convert between Li stream function and electric potential.

    ``psi`` has SI units of ``m^2/s`` and ``Phi`` has SI units of volts.  This
    first one-way adapter uses the Li model's scalar background-field
    magnitude.  Spatially varying ``B`` should be introduced with the later
    flux-tube geometry rather than hidden in a projector.
    """

    background_field_t: float

    def __post_init__(self) -> None:
        field = float(self.background_field_t)
        if not np.isfinite(field) or field <= 0.0:
            raise ValueError("background_field_t must be positive and finite")
        object.__setattr__(self, "background_field_t", field)

    def electric_potential_v(self, stream_function_m2_s: ArrayLike) -> FloatArray:
        """Return ``Phi [V] = B [T] * psi [m^2/s]``."""

        psi = _immutable_float_array(
            "stream_function_m2_s", stream_function_m2_s
        )
        return _immutable_float_array(
            "electric_potential_v", self.background_field_t * psi
        )

    def stream_function_m2_s(self, electric_potential_v: ArrayLike) -> FloatArray:
        """Return the inverse normalization ``psi = Phi / B``."""

        potential = _immutable_float_array(
            "electric_potential_v", electric_potential_v
        )
        return _immutable_float_array(
            "stream_function_m2_s", potential / self.background_field_t
        )


@dataclass(frozen=True)
class SmoothRamp:
    """C1-continuous switch-on envelope based on cubic smoothstep."""

    start_s: float = 0.0
    rise_time_s: float = 1.0

    def __post_init__(self) -> None:
        start = float(self.start_s)
        rise = float(self.rise_time_s)
        if not np.isfinite(start):
            raise ValueError("start_s must be finite")
        if not np.isfinite(rise) or rise <= 0.0:
            raise ValueError("rise_time_s must be positive and finite")
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "rise_time_s", rise)

    def value(self, time_s: float) -> float:
        """Evaluate an envelope that is zero, smoothstep, then one."""

        time = float(time_s)
        if not np.isfinite(time):
            raise ValueError("time_s must be finite")
        fraction = (time - self.start_s) / self.rise_time_s
        if fraction <= 0.0:
            return 0.0
        if fraction >= 1.0:
            return 1.0
        return float(fraction * fraction * (3.0 - 2.0 * fraction))

    __call__ = value


@dataclass(frozen=True)
class FrozenDriveSample:
    """Prescribed field-line drive at one time, with no feedback channel."""

    time_s: float
    envelope: float
    stream_function_m2_s: FloatArray
    electric_potential_v: FloatArray

    def __post_init__(self) -> None:
        time = float(self.time_s)
        envelope = float(self.envelope)
        if not np.isfinite(time):
            raise ValueError("time_s must be finite")
        if not np.isfinite(envelope) or not 0.0 <= envelope <= 1.0:
            raise ValueError("envelope must lie in [0, 1]")
        psi = _immutable_float_array(
            "stream_function_m2_s", self.stream_function_m2_s
        )
        potential = _immutable_float_array(
            "electric_potential_v", self.electric_potential_v
        )
        if psi.shape != potential.shape:
            raise ValueError("stream function and potential must have equal shapes")
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "envelope", envelope)
        object.__setattr__(self, "stream_function_m2_s", psi)
        object.__setattr__(self, "electric_potential_v", potential)


class PotentialClosure(Protocol):
    """Minimal protocol implemented by the existing Li potential solvers."""

    def solve(self, density_m3: FloatArray) -> tuple[FloatArray, Any]: ...


class FrozenLiDrive:
    """Cache one projected Li potential and expose a prescribed time drive."""

    def __init__(
        self,
        snapshot: LiSnapshot,
        potential_solver: PotentialClosure,
        *,
        normalization: MagneticNormalization,
        project_potential: Callable[[FloatArray], ArrayLike],
        ramp: SmoothRamp,
    ) -> None:
        if not isinstance(snapshot, LiSnapshot):
            raise TypeError("snapshot must be a LiSnapshot")

        # This is intentionally the only Li closure call in the object's
        # lifetime.  A writable copy protects the immutable stored snapshot
        # even from a legacy solver that modifies its work array in place.
        solution = potential_solver.solve(snapshot.density_for_closure())
        if not isinstance(solution, tuple) or len(solution) != 2:
            raise TypeError("potential_solver.solve must return (psi, diagnostics)")
        raw_psi, diagnostics = solution
        raw_psi_readonly = _immutable_float_array("Li potential", raw_psi, ndim=2)
        projected_psi = _immutable_float_array(
            "projected potential", project_potential(raw_psi_readonly)
        )

        self._snapshot = snapshot
        self._normalization = normalization
        self._ramp = ramp
        self._projected_psi_m2_s = projected_psi
        self._projected_phi_v = normalization.electric_potential_v(projected_psi)
        self._closure_diagnostics = diagnostics

    @property
    def snapshot(self) -> LiSnapshot:
        return self._snapshot

    @property
    def closure_diagnostics(self) -> Any:
        return self._closure_diagnostics

    @property
    def projected_stream_function_m2_s(self) -> FloatArray:
        return self._projected_psi_m2_s

    @property
    def projected_electric_potential_v(self) -> FloatArray:
        return self._projected_phi_v

    def at(self, time_s: float) -> FrozenDriveSample:
        """Return the cached spatial drive scaled by the smooth envelope."""

        envelope = self._ramp.value(time_s)
        return FrozenDriveSample(
            time_s=float(time_s),
            envelope=envelope,
            stream_function_m2_s=envelope * self._projected_psi_m2_s,
            electric_potential_v=envelope * self._projected_phi_v,
        )


class OneWayLine(Protocol[LineState, LineRhs]):
    """Field-line backend that accepts state plus an external drive only."""

    def rhs(self, state: LineState, drive: FrozenDriveSample) -> LineRhs: ...


class OneWayAlfvenExperiment:
    """Connect a frozen Li drive to a field-line RHS without a return path."""

    def __init__(
        self,
        line: OneWayLine[LineState, LineRhs],
        drive: FrozenLiDrive,
    ) -> None:
        self._line = line
        self._drive = drive

    def rhs(self, time_s: float, state: LineState) -> LineRhs:
        """Evaluate ``line.rhs(state, drive)``; no FAC or Li callback exists."""

        return self._line.rhs(state, self._drive.at(time_s))

