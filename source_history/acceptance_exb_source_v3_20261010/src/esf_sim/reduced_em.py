"""State algebra and the exact electrostatic limit of the reduced EM model.

This module is intentionally a thin adapter around :class:`ESFModel`.  With
zero electromagnetic coupling it calls the historical ``evaluate_rhs``
method directly.  No zero FAC array is manufactured and no alternative
elliptic solve is selected.  The old Li electrostatic implementation therefore
remains the regression oracle while the compound state needed by later
Alfven coupling is introduced independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .alfven_1d import AlfvenState
from ._model_impl import RhsDiagnostics


class ElectrostaticModel(Protocol):
    """Structural type supplied by the existing Li ESF model."""

    def evaluate_rhs(
        self, density: np.ndarray
    ) -> tuple[np.ndarray, RhsDiagnostics]: ...


@dataclass(frozen=True)
class CoupledEMState:
    """Density plus the reduced field-line state.

    A tendency uses the same container: its density member then has units of
    m^-3 s^-1 and its Alfven members carry their corresponding rates.
    """

    density_m3: np.ndarray
    alfven: AlfvenState

    def __post_init__(self) -> None:
        density = np.asarray(self.density_m3, dtype=np.float64)
        if density.ndim != 2 or not np.all(np.isfinite(density)):
            raise ValueError("density_m3 must be a finite two-dimensional array")
        velocity = np.asarray(self.alfven.velocity_m_s, dtype=np.float64)
        magnetic = np.asarray(self.alfven.magnetic_t, dtype=np.float64)
        if velocity.ndim != 1 or magnetic.shape != velocity.shape:
            raise ValueError("Alfven arrays must be one-dimensional and shape matched")
        if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(magnetic)):
            raise ValueError("Alfven arrays must be finite")


@dataclass(frozen=True)
class CoupledDiagnostics:
    """Diagnostics for an EM RHS evaluation in the electrostatic limit."""

    electrostatic: RhsDiagnostics
    coupling_gain: float = 0.0


def zeros_like_state(state: CoupledEMState) -> CoupledEMState:
    """Return a zero compound state without mutating ``state``."""

    return CoupledEMState(
        np.zeros_like(state.density_m3),
        AlfvenState(
            np.zeros_like(state.alfven.velocity_m_s),
            np.zeros_like(state.alfven.magnetic_t),
        ),
    )


def linear_combination(
    weights: tuple[float, ...] | list[float],
    states: tuple[CoupledEMState, ...] | list[CoupledEMState],
) -> CoupledEMState:
    """Form a compound-state linear combination for Runge--Kutta/AB methods."""

    if not states or len(weights) != len(states):
        raise ValueError("weights and states must have the same non-zero length")
    density_shape = states[0].density_m3.shape
    line_shape = states[0].alfven.velocity_m_s.shape
    for state in states:
        if state.density_m3.shape != density_shape:
            raise ValueError("all density arrays must have the same shape")
        if (
            state.alfven.velocity_m_s.shape != line_shape
            or state.alfven.magnetic_t.shape != line_shape
        ):
            raise ValueError("all Alfven arrays must have the same shape")
    density = np.zeros(density_shape, dtype=np.float64)
    velocity = np.zeros(line_shape, dtype=np.float64)
    magnetic = np.zeros(line_shape, dtype=np.float64)
    for weight, state in zip(weights, states, strict=True):
        density += float(weight) * state.density_m3
        velocity += float(weight) * state.alfven.velocity_m_s
        magnetic += float(weight) * state.alfven.magnetic_t
    return CoupledEMState(density, AlfvenState(velocity, magnetic))


class ESLimitAdapter:
    """Embed the existing Li model in a compound state at exactly zero gain."""

    def __init__(self, electrostatic_model: ElectrostaticModel):
        self.electrostatic_model = electrostatic_model

    def rhs(
        self, time_s: float, state: CoupledEMState
    ) -> tuple[CoupledEMState, CoupledDiagnostics]:
        del time_s  # the historical electrostatic RHS is autonomous
        density_rhs, diagnostics = self.electrostatic_model.evaluate_rhs(
            state.density_m3
        )
        tendency = CoupledEMState(
            density_rhs,
            AlfvenState(
                np.zeros_like(state.alfven.velocity_m_s),
                np.zeros_like(state.alfven.magnetic_t),
            ),
        )
        return tendency, CoupledDiagnostics(diagnostics, coupling_gain=0.0)

    def ssprk3_step(
        self, time_s: float, state: CoupledEMState, dt_s: float
    ) -> tuple[CoupledEMState, CoupledDiagnostics]:
        """Advance the compound state with the legacy SSPRK3 operation order."""

        k0, d0 = self.rhs(time_s, state)
        u1 = CoupledEMState(
            state.density_m3 + dt_s * k0.density_m3,
            AlfvenState(
                state.alfven.velocity_m_s + dt_s * k0.alfven.velocity_m_s,
                state.alfven.magnetic_t + dt_s * k0.alfven.magnetic_t,
            ),
        )
        k1, d1 = self.rhs(time_s + dt_s, u1)
        u2 = CoupledEMState(
            0.75 * state.density_m3
            + 0.25 * (u1.density_m3 + dt_s * k1.density_m3),
            AlfvenState(
                0.75 * state.alfven.velocity_m_s
                + 0.25
                * (u1.alfven.velocity_m_s + dt_s * k1.alfven.velocity_m_s),
                0.75 * state.alfven.magnetic_t
                + 0.25 * (u1.alfven.magnetic_t + dt_s * k1.alfven.magnetic_t),
            ),
        )
        k2, d2 = self.rhs(time_s + 0.5 * dt_s, u2)
        result = CoupledEMState(
            (state.density_m3 + 2.0 * (u2.density_m3 + dt_s * k2.density_m3))
            / 3.0,
            AlfvenState(
                (
                    state.alfven.velocity_m_s
                    + 2.0
                    * (u2.alfven.velocity_m_s + dt_s * k2.alfven.velocity_m_s)
                )
                / 3.0,
                (
                    state.alfven.magnetic_t
                    + 2.0 * (u2.alfven.magnetic_t + dt_s * k2.alfven.magnetic_t)
                )
                / 3.0,
            ),
        )
        diagnostic = max((d0, d1, d2), key=lambda d: d.electrostatic.cfl_rate)
        return result, diagnostic

    def ab3_predictor_corrector_step(
        self,
        time_s: float,
        state: CoupledEMState,
        previous: CoupledEMState,
        previous2: CoupledEMState,
        dt_s: float,
    ) -> tuple[CoupledEMState, CoupledDiagnostics]:
        """Apply the paper's legacy AB3 predictor-corrector operation order."""

        predictor = CoupledEMState(
            15.0 / 8.0 * state.density_m3
            - 5.0 / 4.0 * previous.density_m3
            + 3.0 / 8.0 * previous2.density_m3,
            AlfvenState(
                15.0 / 8.0 * state.alfven.velocity_m_s
                - 5.0 / 4.0 * previous.alfven.velocity_m_s
                + 3.0 / 8.0 * previous2.alfven.velocity_m_s,
                15.0 / 8.0 * state.alfven.magnetic_t
                - 5.0 / 4.0 * previous.alfven.magnetic_t
                + 3.0 / 8.0 * previous2.alfven.magnetic_t,
            ),
        )
        tendency, diagnostic = self.rhs(time_s, predictor)
        return CoupledEMState(
            state.density_m3 + dt_s * tendency.density_m3,
            AlfvenState(
                state.alfven.velocity_m_s + dt_s * tendency.alfven.velocity_m_s,
                state.alfven.magnetic_t + dt_s * tendency.alfven.magnetic_t,
            ),
        ), diagnostic
