"""Allocation-conscious finite-volume kernels for production ESF runs.

The formulas are identical to :mod:`esf_sim._numerics_impl`. Direct slices
replace repeated ``np.take`` calls and scalar Rusanov flux is evaluated in its
algebraically equivalent upwind form. Historical reproduction runs therefore
retain their original operation ordering.
"""

from __future__ import annotations

import numpy as np

from ._numerics_impl import PDM7_LEFT_COEFFICIENTS, van_leer_slope


def _median3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))


def reconstruct_interfaces(
    quantity: np.ndarray,
    *,
    axis: int,
    periodic: bool,
    scheme: str,
    pdm_alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.moveaxis(np.asarray(quantity, dtype=np.float64), axis, -1)
    n = q.shape[-1]
    padding = [(0, 0)] * q.ndim
    padding[-1] = (4, 4)
    padded = np.pad(q, padding, mode="wrap" if periodic else "edge")
    start = 4 if periodic else 3
    length = n if periodic else n + 1

    def at(offset: int) -> np.ndarray:
        return padded[..., start + offset : start + offset + length]

    qi = at(0)
    qip1 = at(1)
    if scheme == "first_order":
        left, right = qi, qip1
    elif scheme == "van_leer":
        left = qi + 0.5 * van_leer_slope(qi - at(-1), qip1 - qi)
        right = qip1 - 0.5 * van_leer_slope(qip1 - qi, at(2) - qip1)
    elif scheme == "pdm7":
        if not 0.0 < pdm_alpha <= 1.0:
            raise ValueError("pdm_alpha must be in (0, 1]")
        c = PDM7_LEFT_COEFFICIENTS
        high_left = (
            c[0] * at(-3) + c[1] * at(-2) + c[2] * at(-1)
            + c[3] * qi + c[4] * qip1 + c[5] * at(2) + c[6] * at(3)
        )
        high_right = (
            c[0] * at(4) + c[1] * at(3) + c[2] * at(2)
            + c[3] * qip1 + c[4] * qi + c[5] * at(-1) + c[6] * at(-2)
        )
        inv_alpha = 1.0 / pdm_alpha
        left = _median3(inv_alpha * qi + (1.0 - inv_alpha) * at(-1), high_left, qi)
        right = _median3(
            inv_alpha * qip1 + (1.0 - inv_alpha) * at(2), high_right, qip1
        )
    else:
        raise ValueError(f"unknown reconstruction scheme: {scheme!r}")
    return np.moveaxis(left, -1, axis), np.moveaxis(right, -1, axis)


def _upwind_flux(velocity: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return velocity * np.where(velocity >= 0.0, left, right)


def advection_rhs(
    density: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    *,
    dx: float,
    dy: float,
    scheme: str,
    pdm_alpha: float = 0.2,
    periodic_y: bool = False,
) -> np.ndarray:
    n = np.asarray(density, dtype=np.float64)
    u = np.asarray(velocity_x, dtype=np.float64)
    v = np.asarray(velocity_y, dtype=np.float64)
    if n.shape != u.shape or n.shape != v.shape:
        raise ValueError("density and velocity arrays must have the same shape")

    lx, rx = reconstruct_interfaces(
        n, axis=1, periodic=True, scheme=scheme, pdm_alpha=pdm_alpha
    )
    fx = _upwind_flux(0.5 * (u + np.roll(u, -1, axis=1)), lx, rx)
    divergence = (fx - np.roll(fx, 1, axis=1)) / dx

    ly, ry = reconstruct_interfaces(
        n, axis=0, periodic=periodic_y, scheme=scheme, pdm_alpha=pdm_alpha
    )
    if periodic_y:
        fy = _upwind_flux(0.5 * (v + np.roll(v, -1, axis=0)), ly, ry)
        divergence += (fy - np.roll(fy, 1, axis=0)) / dy
    else:
        vface = np.empty((n.shape[0] + 1, n.shape[1]), dtype=np.float64)
        vface[1:-1] = 0.5 * (v[:-1] + v[1:])
        vface[0], vface[-1] = v[0], v[-1]
        fy = _upwind_flux(vface, ly, ry)
        divergence += (fy[1:] - fy[:-1]) / dy
    return -divergence


__all__ = ["advection_rhs", "reconstruct_interfaces"]
