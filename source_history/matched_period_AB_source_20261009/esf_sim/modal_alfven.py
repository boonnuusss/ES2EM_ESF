"""Transverse modal FAC mapping and a voltage-characteristic Alfven line.

This module supplies the missing dimensional bridge between a two-dimensional
Li-type ionospheric plane and a one-dimensional field-aligned transmission
line.  A scalar parallel vector potential is expanded as

``A_parallel(x, y, s) = sum_k A_k(s) q_k(x, y)``.

The real modes are exact eigenvectors of the *discrete* transverse operator
used by the ESF code: all stored x values are periodic (``numpy.roll``), while
equal extrapolation into the y ghost cells gives a homogeneous discrete
Neumann boundary.  Consequently

``j_parallel,k = lambda_k A_k / mu0``

with ``lambda_k`` the eigenvalue of ``-Laplacian_perp``.  The spatially
constant mode is deliberately excluded: a bare one-dimensional line current
is not FAC, and cannot enter the Li current-continuity equation without the
transverse Laplacian.

The field-line coordinate ``s`` increases from the ionospheric Li plane at
``s=0`` toward the magnetosphere.  For uniform Alfven speed the variables

``w_plus  = Phi + V_A A_parallel``  (travels toward increasing s),
``w_minus = Phi - V_A A_parallel``  (travels toward decreasing s)

obey independent advection equations.  Both characteristics have units of
volts.  At the left boundary a supplied ionospheric voltage fixes the incoming
characteristic through ``w_plus = 2 Phi_boundary - w_minus``.  At the remote
end the historical fixed-voltage condition remains available, while a passive
characteristic termination may instead impose ``w_minus = R w_plus``.  Here
``R=0`` is a matched/outgoing boundary and ``|R| <= 1`` cannot inject Alfvén
energy into the line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .alfven_1d import MU0


XKind = Literal["cos", "sin"]


def _positive_finite(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _finite_array(name: str, values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array.copy()


@dataclass(frozen=True, order=True)
class TransverseMode:
    """One real separable eigenmode of the transverse finite-difference grid."""

    x_wavenumber: int
    y_wavenumber: int
    x_kind: XKind = "cos"


class ModalBasis:
    """Orthonormal real transverse modes without the ``k_perp=0`` mode.

    Normalization uses the cell mean,

    ``mean(q_i * q_j) = delta_ij``.

    Therefore a physical area integral is represented by

    ``integral(f g dA) = domain_area * dot(f_coeff, g_coeff)``

    whenever both fields lie in the retained modal subspace.  The basis is
    stored in separable form (one x and one y vector per mode), avoiding a
    potentially large ``(mode, ny, nx)`` allocation.
    """

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        dx_m: float,
        dy_m: float,
        modes: Iterable[TransverseMode],
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        if self.nx < 2 or self.ny < 2:
            raise ValueError("transverse grid requires nx and ny >= 2")
        self.dx_m = _positive_finite("dx_m", dx_m)
        self.dy_m = _positive_finite("dy_m", dy_m)
        self.modes = tuple(modes)
        if not self.modes:
            raise ValueError("at least one non-constant transverse mode is required")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("transverse modes must be unique")

        x_components: list[np.ndarray] = []
        y_components: list[np.ndarray] = []
        eigenvalues: list[float] = []
        for mode in self.modes:
            x, lambda_x = self._x_component(mode)
            y, lambda_y = self._y_component(mode)
            eigenvalue = lambda_x + lambda_y
            if eigenvalue <= 0.0:
                raise ValueError("the spatially constant k_perp=0 mode is excluded")
            x_components.append(x)
            y_components.append(y)
            eigenvalues.append(eigenvalue)

        self._x_components = np.asarray(x_components, dtype=np.float64)
        self._y_components = np.asarray(y_components, dtype=np.float64)
        self.eigenvalues_m2 = np.asarray(eigenvalues, dtype=np.float64)

        gram_x = self._x_components @ self._x_components.T / self.nx
        gram_y = self._y_components @ self._y_components.T / self.ny
        gram = gram_x * gram_y
        if not np.allclose(gram, np.eye(self.mode_count), atol=2.0e-13, rtol=0.0):
            raise ValueError(
                "selected real modes are not mutually orthogonal on this grid; "
                "check sine/cosine aliases"
            )

    @classmethod
    def low_wavenumber(
        cls,
        *,
        nx: int,
        ny: int,
        dx_m: float,
        dy_m: float,
        max_x_wavenumber: int,
        max_y_wavenumber: int,
        include_sine: bool = True,
    ) -> "ModalBasis":
        """Build all separable modes in a low-wavenumber rectangle.

        The rectangle contains ``0 <= m_x <= max_x_wavenumber`` and
        ``0 <= m_y <= max_y_wavenumber``.  The joint zero mode is omitted.
        Sine modes at x wavenumber zero and at the even-grid Nyquist frequency
        do not exist and are omitted as well.
        """

        nx_i = int(nx)
        ny_i = int(ny)
        max_x = int(max_x_wavenumber)
        max_y = int(max_y_wavenumber)
        if max_x < 0 or max_x > nx_i // 2:
            raise ValueError("max_x_wavenumber lies outside the resolvable range")
        if max_y < 0 or max_y >= ny_i:
            raise ValueError("max_y_wavenumber lies outside the resolvable range")
        modes: list[TransverseMode] = []
        for ky in range(max_y + 1):
            for kx in range(max_x + 1):
                if kx != 0 or ky != 0:
                    modes.append(TransverseMode(kx, ky, "cos"))
                if (
                    include_sine
                    and kx > 0
                    and not (nx_i % 2 == 0 and kx == nx_i // 2)
                ):
                    modes.append(TransverseMode(kx, ky, "sin"))
        return cls(nx=nx_i, ny=ny_i, dx_m=dx_m, dy_m=dy_m, modes=modes)

    @property
    def mode_count(self) -> int:
        return len(self.modes)

    @property
    def cell_area_m2(self) -> float:
        return self.dx_m * self.dy_m

    @property
    def domain_area_m2(self) -> float:
        return self.nx * self.ny * self.cell_area_m2

    def _x_component(self, mode: TransverseMode) -> tuple[np.ndarray, float]:
        k = int(mode.x_wavenumber)
        if k < 0 or k > self.nx // 2:
            raise ValueError(f"invalid x wavenumber {k} for nx={self.nx}")
        if mode.x_kind not in ("cos", "sin"):
            raise ValueError("x_kind must be 'cos' or 'sin'")
        if mode.x_kind == "sin" and (
            k == 0 or (self.nx % 2 == 0 and k == self.nx // 2)
        ):
            raise ValueError("zero and even-grid Nyquist sine modes vanish")

        phase = 2.0 * np.pi * k * np.arange(self.nx, dtype=np.float64) / self.nx
        if mode.x_kind == "cos":
            vector = np.cos(phase)
            if k != 0 and not (self.nx % 2 == 0 and k == self.nx // 2):
                vector *= np.sqrt(2.0)
        else:
            vector = np.sqrt(2.0) * np.sin(phase)
        eigenvalue = 4.0 * np.sin(np.pi * k / self.nx) ** 2 / self.dx_m**2
        return vector, float(eigenvalue)

    def _y_component(self, mode: TransverseMode) -> tuple[np.ndarray, float]:
        k = int(mode.y_wavenumber)
        if k < 0 or k >= self.ny:
            raise ValueError(f"invalid y wavenumber {k} for ny={self.ny}")
        index = np.arange(self.ny, dtype=np.float64)
        vector = np.cos(np.pi * k * (index + 0.5) / self.ny)
        if k != 0:
            vector *= np.sqrt(2.0)
        eigenvalue = 4.0 * np.sin(0.5 * np.pi * k / self.ny) ** 2 / self.dy_m**2
        return vector, float(eigenvalue)

    def mode_field(self, mode_index: int) -> np.ndarray:
        """Return one normalized mode on the ``(ny, nx)`` grid."""

        index = int(mode_index)
        if index < 0 or index >= self.mode_count:
            raise IndexError("mode index out of range")
        return self._y_components[index, :, None] * self._x_components[index, None, :]

    def project(self, field: np.ndarray) -> np.ndarray:
        """Project a ``(ny, nx)`` field onto the retained normalized modes."""

        values = _finite_array("field", field, (self.ny, self.nx))
        return np.einsum(
            "my,mx,yx->m",
            self._y_components,
            self._x_components,
            values,
            optimize=True,
        ) / (self.nx * self.ny)

    def lift(self, coefficients: np.ndarray) -> np.ndarray:
        """Lift modal coefficients to a physical ``(ny, nx)`` field."""

        values = _finite_array("coefficients", coefficients, (self.mode_count,))
        return np.einsum(
            "m,my,mx->yx",
            values,
            self._y_components,
            self._x_components,
            optimize=True,
        )

    def negative_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Apply the exact periodic-x/equal-ghost-y ``-Laplacian``."""

        values = _finite_array("field", field, (self.ny, self.nx))
        x_part = (
            2.0 * values - np.roll(values, 1, axis=1) - np.roll(values, -1, axis=1)
        ) / self.dx_m**2
        y_previous = np.vstack((values[0:1], values[:-1]))
        y_next = np.vstack((values[1:], values[-1:]))
        y_part = (2.0 * values - y_previous - y_next) / self.dy_m**2
        return x_part + y_part

    def field_inner_product(self, left: np.ndarray, right: np.ndarray) -> float:
        """Return ``integral(left * right dA)`` using cell-area weights."""

        a = _finite_array("left", left, (self.ny, self.nx))
        b = _finite_array("right", right, (self.ny, self.nx))
        return float(self.cell_area_m2 * np.sum(a * b))

    def modal_inner_product(self, left: np.ndarray, right: np.ndarray) -> float:
        """Return the physical-area inner product from modal coefficients."""

        a = _finite_array("left", left, (self.mode_count,))
        b = _finite_array("right", right, (self.mode_count,))
        return float(self.domain_area_m2 * np.dot(a, b))

    def parallel_current_coefficients_a_m2(
        self, apar_coefficients_t_m: np.ndarray, *, mu0_h_m: float = MU0
    ) -> np.ndarray:
        """Map ``A_parallel`` coefficients [T m] to ``j_parallel`` [A/m2]."""

        apar = _finite_array(
            "apar_coefficients_t_m", apar_coefficients_t_m, (self.mode_count,)
        )
        mu0 = _positive_finite("mu0_h_m", mu0_h_m)
        return self.eigenvalues_m2 * apar / mu0

    def parallel_current_density_a_m2(
        self, apar_coefficients_t_m: np.ndarray, *, mu0_h_m: float = MU0
    ) -> np.ndarray:
        """Return the physical-grid FAC implied by modal ``A_parallel``."""

        return self.lift(
            self.parallel_current_coefficients_a_m2(
                apar_coefficients_t_m, mu0_h_m=mu0_h_m
            )
        )


@dataclass(frozen=True)
class ModalAlfvenState:
    """Cell-centred voltage characteristics, shaped ``(s_cells, modes)``."""

    w_plus_v: np.ndarray
    w_minus_v: np.ndarray

    def validated(self, cells: int, modes: int) -> "ModalAlfvenState":
        shape = (int(cells), int(modes))
        return ModalAlfvenState(
            _finite_array("w_plus_v", self.w_plus_v, shape),
            _finite_array("w_minus_v", self.w_minus_v, shape),
        )

    @classmethod
    def from_potential_and_apar(
        cls,
        potential_coefficients_v: np.ndarray,
        apar_coefficients_t_m: np.ndarray,
        *,
        alfven_speed_m_s: float,
    ) -> "ModalAlfvenState":
        potential = np.asarray(potential_coefficients_v, dtype=np.float64)
        apar = np.asarray(apar_coefficients_t_m, dtype=np.float64)
        if potential.ndim != 2 or apar.shape != potential.shape:
            raise ValueError("potential and A_parallel arrays must share a 2-D shape")
        if not np.all(np.isfinite(potential)) or not np.all(np.isfinite(apar)):
            raise ValueError("potential and A_parallel arrays must be finite")
        speed = _positive_finite("alfven_speed_m_s", alfven_speed_m_s)
        return cls(potential + speed * apar, potential - speed * apar)


@dataclass(frozen=True)
class BoundaryCurrentGradient:
    """Left-boundary ``dA_parallel/ds`` and ``dj_parallel/ds`` diagnostics."""

    d_apar_ds_t: np.ndarray
    modal_d_j_parallel_ds_a_m3: np.ndarray
    d_j_parallel_ds_a_m3: np.ndarray


@dataclass(frozen=True)
class LeftBoundaryResponse:
    """Characteristic and FAC response at the Li-plane end of the line."""

    outgoing_w_minus_v: np.ndarray
    incoming_w_plus_v: np.ndarray
    boundary_apar_coefficients_t_m: np.ndarray
    current_gradient: BoundaryCurrentGradient


class ModalAlfvenLine:
    """Uniform-``V_A`` characteristic finite-volume field-line solver.

    The grid is cell-centred on ``0 < s < length_m``.  ``w_plus`` moves from
    the ionosphere at the left toward increasing s, and ``w_minus`` moves back
    toward the ionosphere.  Periodic operation is available for convergence
    and propagation tests; voltage boundaries are used otherwise.
    """

    def __init__(
        self,
        *,
        length_m: float,
        cells: int,
        alfven_speed_m_s: float,
        mode_count: int,
        spatial_order: int = 2,
        periodic: bool = False,
    ) -> None:
        self.length_m = _positive_finite("length_m", length_m)
        self.cells = int(cells)
        self.mode_count = int(mode_count)
        self.alfven_speed_m_s = _positive_finite(
            "alfven_speed_m_s", alfven_speed_m_s
        )
        if self.cells < 4:
            raise ValueError("field line requires at least four cells")
        if self.mode_count < 1:
            raise ValueError("mode_count must be positive")
        if int(spatial_order) not in (1, 2):
            raise ValueError("spatial_order must be one or two")
        self.spatial_order = int(spatial_order)
        self.periodic = bool(periodic)

    @property
    def spacing_m(self) -> float:
        return self.length_m / self.cells

    @property
    def s_m(self) -> np.ndarray:
        return (np.arange(self.cells, dtype=np.float64) + 0.5) * self.spacing_m

    @property
    def maximum_stable_dt_s(self) -> float:
        return self.spacing_m / self.alfven_speed_m_s

    @property
    def alfven_conductance_s(self) -> float:
        """Return the characteristic Alfvén conductance ``1/(mu0 V_A)``."""

        return 1.0 / (MU0 * self.alfven_speed_m_s)

    def cfl_number(self, dt_s: float) -> float:
        """Return the dimensionless line CFL ``V_A dt / ds``."""

        dt = _positive_finite("dt_s", dt_s)
        return self.alfven_speed_m_s * dt / self.spacing_m

    def potential_coefficients_v(self, state: ModalAlfvenState) -> np.ndarray:
        state = state.validated(self.cells, self.mode_count)
        return 0.5 * (state.w_plus_v + state.w_minus_v)

    def apar_coefficients_t_m(self, state: ModalAlfvenState) -> np.ndarray:
        state = state.validated(self.cells, self.mode_count)
        return 0.5 * (state.w_plus_v - state.w_minus_v) / self.alfven_speed_m_s

    def left_outgoing_characteristic_v(self, state: ModalAlfvenState) -> np.ndarray:
        """Return ``w_minus`` leaving the line through the Li plane at ``s=0``."""

        state = state.validated(self.cells, self.mode_count)
        slope = self._slopes(state.w_minus_v)
        return (state.w_minus_v[0] - 0.5 * slope[0]).copy()

    def right_outgoing_characteristic_v(self, state: ModalAlfvenState) -> np.ndarray:
        """Return ``w_plus`` reconstructed to the remote boundary face."""

        state = state.validated(self.cells, self.mode_count)
        slope = self._slopes(state.w_plus_v)
        return (state.w_plus_v[-1] + 0.5 * slope[-1]).copy()

    def conductance_reflection_coefficient(
        self, pedersen_conductance_s: float
    ) -> float:
        """Return the voltage reflection coefficient of a passive load.

        A height-integrated Pedersen load obeys ``j_parallel =
        Sigma_P (-Laplacian_perp) Phi``.  Since the same transverse eigenvalue
        multiplies the Alfvén current, it cancels from the terminal relation,
        leaving ``R=(Sigma_A-Sigma_P)/(Sigma_A+Sigma_P)``.

        ``Sigma_P=Sigma_A`` is matched, zero conductance is open (``R=+1``),
        and infinite conductance approaches the historical zero-voltage
        reflection (``R=-1``).
        """

        conductance = float(pedersen_conductance_s)
        if not np.isfinite(conductance) or conductance < 0.0:
            raise ValueError(
                "pedersen_conductance_s must be non-negative and finite"
            )
        sigma_a = self.alfven_conductance_s
        return float((sigma_a - conductance) / (sigma_a + conductance))

    def right_incoming_characteristic_v(
        self,
        state: ModalAlfvenState,
        reflection_coefficient: np.ndarray | float,
    ) -> np.ndarray:
        """Return a passive remote incoming characteristic ``R w_plus``.

        Scalar and per-mode coefficients are accepted.  Restricting every
        coefficient to ``[-1, 1]`` makes the diagonal terminal contractive in
        the characteristic-energy norm.
        """

        reflection = np.asarray(reflection_coefficient, dtype=np.float64)
        if reflection.ndim == 0:
            reflection = np.full(self.mode_count, float(reflection))
        reflection = _finite_array(
            "right_reflection_coefficient", reflection, (self.mode_count,)
        )
        if np.any(np.abs(reflection) > 1.0 + 16.0 * np.finfo(float).eps):
            raise ValueError(
                "right_reflection_coefficient must lie in [-1, 1]"
            )
        return reflection * self.right_outgoing_characteristic_v(state)

    def left_incoming_characteristic_v(
        self,
        state: ModalAlfvenState,
        boundary_voltage_coefficients_v: np.ndarray,
    ) -> np.ndarray:
        """Return the ``w_plus`` imposed by a modal Li-boundary voltage."""

        boundary_voltage = _finite_array(
            "boundary_voltage_coefficients_v",
            boundary_voltage_coefficients_v,
            (self.mode_count,),
        )
        return 2.0 * boundary_voltage - self.left_outgoing_characteristic_v(state)

    @staticmethod
    def _minmod(*values: np.ndarray) -> np.ndarray:
        stack = np.stack(values)
        positive = np.all(stack > 0.0, axis=0)
        negative = np.all(stack < 0.0, axis=0)
        magnitude = np.min(np.abs(stack), axis=0)
        return np.where(positive, magnitude, np.where(negative, -magnitude, 0.0))

    def _slopes(self, values: np.ndarray) -> np.ndarray:
        if self.spatial_order == 1:
            return np.zeros_like(values)
        if self.periodic:
            backward = values - np.roll(values, 1, axis=0)
            forward = np.roll(values, -1, axis=0) - values
            return self._minmod(
                2.0 * backward, 0.5 * (backward + forward), 2.0 * forward
            )
        slopes = np.zeros_like(values)
        # One-sided boundary slopes preserve a linear characteristic exactly.
        # Setting these slopes to zero injects a spurious half-cell pulse when
        # the two prescribed endpoint voltages differ.
        slopes[0] = values[1] - values[0]
        slopes[-1] = values[-1] - values[-2]
        backward = values[1:-1] - values[:-2]
        forward = values[2:] - values[1:-1]
        slopes[1:-1] = self._minmod(
            2.0 * backward, 0.5 * (backward + forward), 2.0 * forward
        )
        return slopes

    def _boundary_voltage(self, name: str, values: np.ndarray | float | None) -> np.ndarray:
        if values is None:
            return np.zeros(self.mode_count, dtype=np.float64)
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            array = np.full(self.mode_count, float(array))
        return _finite_array(name, array, (self.mode_count,))

    def _advection_rhs(
        self,
        values: np.ndarray,
        *,
        speed_m_s: float,
        incoming: np.ndarray | None,
    ) -> np.ndarray:
        slope = self._slopes(values)
        speed = float(speed_m_s)
        if self.periodic:
            if speed > 0.0:
                left_face_flux = speed * np.roll(values + 0.5 * slope, 1, axis=0)
            else:
                left_face_flux = speed * (values - 0.5 * slope)
            return -(
                np.roll(left_face_flux, -1, axis=0) - left_face_flux
            ) / self.spacing_m

        if incoming is None:
            raise ValueError("a non-periodic characteristic requires an incoming state")
        face_flux = np.empty((self.cells + 1, self.mode_count), dtype=np.float64)
        if speed > 0.0:
            face_flux[0] = speed * incoming
            face_flux[1:] = speed * (values + 0.5 * slope)
        else:
            face_flux[:-1] = speed * (values - 0.5 * slope)
            face_flux[-1] = speed * incoming
        return -(face_flux[1:] - face_flux[:-1]) / self.spacing_m

    def rhs(
        self,
        state: ModalAlfvenState,
        *,
        left_voltage_v: np.ndarray | float | None = None,
        right_voltage_v: np.ndarray | float | None = None,
        right_reflection_coefficient: np.ndarray | float | None = None,
    ) -> ModalAlfvenState:
        """Return the characteristic method-of-lines derivative [V/s]."""

        state = state.validated(self.cells, self.mode_count)
        if self.periodic:
            if (
                left_voltage_v is not None
                or right_voltage_v is not None
                or right_reflection_coefficient is not None
            ):
                raise ValueError("periodic lines do not accept voltage boundaries")
            incoming_plus = incoming_minus = None
        else:
            left = self._boundary_voltage("left_voltage_v", left_voltage_v)
            incoming_plus = 2.0 * left - self.left_outgoing_characteristic_v(
                state
            )
            if right_reflection_coefficient is None:
                right = self._boundary_voltage("right_voltage_v", right_voltage_v)
                incoming_minus = (
                    2.0 * right - self.right_outgoing_characteristic_v(state)
                )
            else:
                if right_voltage_v is not None:
                    raise ValueError(
                        "right_voltage_v and right_reflection_coefficient are "
                        "mutually exclusive"
                    )
                incoming_minus = self.right_incoming_characteristic_v(
                    state, right_reflection_coefficient
                )
        return ModalAlfvenState(
            self._advection_rhs(
                state.w_plus_v,
                speed_m_s=self.alfven_speed_m_s,
                incoming=incoming_plus,
            ),
            self._advection_rhs(
                state.w_minus_v,
                speed_m_s=-self.alfven_speed_m_s,
                incoming=incoming_minus,
            ),
        )

    @staticmethod
    def _add(
        state: ModalAlfvenState, derivative: ModalAlfvenState, scale: float
    ) -> ModalAlfvenState:
        return ModalAlfvenState(
            state.w_plus_v + scale * derivative.w_plus_v,
            state.w_minus_v + scale * derivative.w_minus_v,
        )

    def step(
        self,
        state: ModalAlfvenState,
        dt_s: float,
        *,
        left_voltage_v: np.ndarray | float | None = None,
        right_voltage_v: np.ndarray | float | None = None,
        right_reflection_coefficient: np.ndarray | float | None = None,
    ) -> ModalAlfvenState:
        """Advance one SSPRK3 step with fixed voltages during the step."""

        state = state.validated(self.cells, self.mode_count)
        dt = _positive_finite("dt_s", dt_s)
        if self.cfl_number(dt) > 1.0 + 1.0e-12:
            raise ValueError("Alfven characteristic CFL exceeds one")
        kwargs = {
            "left_voltage_v": left_voltage_v,
            "right_voltage_v": right_voltage_v,
            "right_reflection_coefficient": right_reflection_coefficient,
        }
        stage1 = self._add(state, self.rhs(state, **kwargs), dt)
        euler2 = self._add(stage1, self.rhs(stage1, **kwargs), dt)
        stage2 = ModalAlfvenState(
            0.75 * state.w_plus_v + 0.25 * euler2.w_plus_v,
            0.75 * state.w_minus_v + 0.25 * euler2.w_minus_v,
        )
        euler3 = self._add(stage2, self.rhs(stage2, **kwargs), dt)
        return ModalAlfvenState(
            state.w_plus_v / 3.0 + 2.0 * euler3.w_plus_v / 3.0,
            state.w_minus_v / 3.0 + 2.0 * euler3.w_minus_v / 3.0,
        )

    def advance(
        self,
        state: ModalAlfvenState,
        *,
        duration_s: float,
        cfl: float = 0.8,
        left_voltage_v: np.ndarray | float | None = None,
        right_voltage_v: np.ndarray | float | None = None,
        right_reflection_coefficient: np.ndarray | float | None = None,
    ) -> ModalAlfvenState:
        """Advance for ``duration_s`` using a constant-CFL SSPRK3 sequence."""

        duration = float(duration_s)
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("duration_s must be non-negative and finite")
        cfl_value = float(cfl)
        if not np.isfinite(cfl_value) or not (0.0 < cfl_value <= 1.0):
            raise ValueError("cfl must lie in (0, 1]")
        result = state.validated(self.cells, self.mode_count)
        elapsed = 0.0
        base_dt = cfl_value * self.maximum_stable_dt_s
        while elapsed < duration - 1.0e-15:
            dt = min(base_dt, duration - elapsed)
            result = self.step(
                result,
                dt,
                left_voltage_v=left_voltage_v,
                right_voltage_v=right_voltage_v,
                right_reflection_coefficient=right_reflection_coefficient,
            )
            elapsed += dt
        return result

    def left_boundary_current_gradient(
        self,
        state: ModalAlfvenState,
        basis: ModalBasis,
        *,
        boundary_voltage_coefficients_v: np.ndarray,
        mu0_h_m: float = MU0,
    ) -> BoundaryCurrentGradient:
        """Construct ``d_s j_parallel`` at the ionospheric boundary.

        The outgoing ``w_minus`` is reconstructed to the left boundary face.
        Imposing the boundary voltage gives

        ``A_boundary = (Phi_boundary - w_minus,out) / V_A``.

        A half-cell one-sided difference then provides ``d_s A``.  Applying
        the transverse eigenvalue yields

        ``d_s j_parallel,k = lambda_k d_s A_k / mu0``.

        This is the volume-current divergence term required by the local Li
        closure; it is not a direct insertion of a 1-D line current.
        """

        if basis.mode_count != self.mode_count:
            raise ValueError("basis and field line must retain the same number of modes")
        state = state.validated(self.cells, self.mode_count)
        boundary_voltage = _finite_array(
            "boundary_voltage_coefficients_v",
            boundary_voltage_coefficients_v,
            (self.mode_count,),
        )
        mu0 = _positive_finite("mu0_h_m", mu0_h_m)
        outgoing = self.left_outgoing_characteristic_v(state)
        boundary_apar = (
            boundary_voltage - outgoing
        ) / self.alfven_speed_m_s
        cell_apar = 0.5 * (
            state.w_plus_v[0] - state.w_minus_v[0]
        ) / self.alfven_speed_m_s
        derivative = (cell_apar - boundary_apar) / (0.5 * self.spacing_m)
        current_gradient = basis.eigenvalues_m2 * derivative / mu0
        return BoundaryCurrentGradient(
            d_apar_ds_t=derivative,
            modal_d_j_parallel_ds_a_m3=current_gradient,
            d_j_parallel_ds_a_m3=basis.lift(current_gradient),
        )

    def left_boundary_response(
        self,
        state: ModalAlfvenState,
        basis: ModalBasis,
        *,
        boundary_voltage_coefficients_v: np.ndarray,
        mu0_h_m: float = MU0,
    ) -> LeftBoundaryResponse:
        """Return incoming voltage characteristic and dimensional FAC gradient.

        Positive derivatives point from the Li plane toward increasing ``s``.
        The derivative uses the boundary face and first cell centre, separated
        by ``ds/2``; this makes its orientation and sign explicit.
        """

        boundary_voltage = _finite_array(
            "boundary_voltage_coefficients_v",
            boundary_voltage_coefficients_v,
            (self.mode_count,),
        )
        outgoing = self.left_outgoing_characteristic_v(state)
        incoming = 2.0 * boundary_voltage - outgoing
        boundary_apar = 0.5 * (incoming - outgoing) / self.alfven_speed_m_s
        gradient = self.left_boundary_current_gradient(
            state,
            basis,
            boundary_voltage_coefficients_v=boundary_voltage,
            mu0_h_m=mu0_h_m,
        )
        return LeftBoundaryResponse(
            outgoing_w_minus_v=outgoing,
            incoming_w_plus_v=incoming,
            boundary_apar_coefficients_t_m=boundary_apar,
            current_gradient=gradient,
        )


__all__ = [
    "BoundaryCurrentGradient",
    "LeftBoundaryResponse",
    "ModalAlfvenLine",
    "ModalAlfvenState",
    "ModalBasis",
    "TransverseMode",
]
