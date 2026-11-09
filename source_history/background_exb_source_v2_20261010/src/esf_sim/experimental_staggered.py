"""Stage 3: evaluate ExB velocity directly at finite-volume faces."""

from __future__ import annotations

import numpy as np

from . import _model_impl
from .numerics import reconstruct_interfaces, rusanov_flux


_BASE_MODEL = _model_impl.ESFModel


def staggered_face_velocity(
    potential: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return normal ExB velocities on x and y faces."""

    psi_x_face = 0.5 * (potential + np.roll(potential, -1, axis=1))
    velocity_x_face = np.empty_like(potential)
    velocity_x_face[1:-1] = -(
        psi_x_face[2:] - psi_x_face[:-2]
    ) / (2.0 * dy)
    velocity_x_face[0] = -(psi_x_face[1] - psi_x_face[0]) / dy
    velocity_x_face[-1] = -(psi_x_face[-1] - psi_x_face[-2]) / dy

    velocity_y_face = np.zeros(
        (potential.shape[0] + 1, potential.shape[1]), dtype=np.float64
    )
    psi_y_face = 0.5 * (potential[:-1] + potential[1:])
    velocity_y_face[1:-1] = (
        np.roll(psi_y_face, -1, axis=1)
        - np.roll(psi_y_face, 1, axis=1)
    ) / (2.0 * dx)
    return velocity_x_face, velocity_y_face


def staggered_advection_rhs(
    density: np.ndarray,
    velocity_x_face: np.ndarray,
    velocity_y_face: np.ndarray,
    *,
    dx: float,
    dy: float,
    scheme: str,
    pdm_alpha: float,
) -> np.ndarray:
    left_x, right_x = reconstruct_interfaces(
        density,
        axis=1,
        periodic=True,
        scheme=scheme,
        pdm_alpha=pdm_alpha,
    )
    flux_x = rusanov_flux(velocity_x_face, left_x, right_x)
    divergence_x = (flux_x - np.roll(flux_x, 1, axis=1)) / dx

    left_y, right_y = reconstruct_interfaces(
        density,
        axis=0,
        periodic=False,
        scheme=scheme,
        pdm_alpha=pdm_alpha,
    )
    flux_y = rusanov_flux(velocity_y_face, left_y, right_y)
    divergence_y = (flux_y[1:] - flux_y[:-1]) / dy
    return -(divergence_x + divergence_y)


class StaggeredESFModel(_BASE_MODEL):
    def evaluate_rhs(self, density: np.ndarray):
        potential, potential_diagnostics = self.potential_solver.solve(density)
        velocity_x_face, velocity_y_face = staggered_face_velocity(
            potential, self.config.dx_m, self.config.dy_m
        )
        rhs = staggered_advection_rhs(
            density,
            velocity_x_face,
            velocity_y_face,
            dx=self.config.dx_m,
            dy=self.config.dy_m,
            scheme=self.config.scheme,
            pdm_alpha=self.config.pdm_alpha,
        )
        speed_x = np.maximum(
            np.abs(velocity_x_face),
            np.abs(np.roll(velocity_x_face, 1, axis=1)),
        )
        speed_y = np.maximum(
            np.abs(velocity_y_face[:-1]), np.abs(velocity_y_face[1:])
        )
        diagnostics = _model_impl.RhsDiagnostics(
            cfl_rate=float(
                np.max(speed_x / self.config.dx_m + speed_y / self.config.dy_m)
            ),
            maximum_speed_x=float(np.max(np.abs(velocity_x_face))),
            maximum_speed_y=float(np.max(np.abs(velocity_y_face))),
            potential=potential_diagnostics,
        )
        return rhs, diagnostics
