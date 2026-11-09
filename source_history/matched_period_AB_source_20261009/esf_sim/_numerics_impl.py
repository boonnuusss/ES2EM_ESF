from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

PDM7_LEFT_COEFFICIENTS = np.asarray(
    [-1 / 140, 5 / 84, -101 / 420, 319 / 420, 107 / 210, -19 / 210, 1 / 105],
    dtype=np.float64,
)


def _median3(a: FloatArray, b: FloatArray, c: FloatArray) -> FloatArray:
    return a + b + c - np.minimum(np.minimum(a, b), c) - np.maximum(np.maximum(a, b), c)


def van_leer_slope(delta_minus: FloatArray, delta_plus: FloatArray) -> FloatArray:
    """Harmonic van Leer limited slope."""

    product = delta_minus * delta_plus
    denominator = delta_minus + delta_plus
    slope = np.zeros_like(product)
    valid = product > 0.0
    np.divide(2.0 * product, denominator, out=slope, where=valid & (denominator != 0.0))
    return slope


def reconstruct_interfaces(
    quantity: FloatArray,
    *,
    axis: int,
    periodic: bool,
    scheme: str,
    pdm_alpha: float = 0.2,
) -> tuple[FloatArray, FloatArray]:
    """Return left/right states at every interface along *axis*.

    Periodic dimensions have ``N`` interfaces, indexed by their left cell.
    Extrapolated dimensions have ``N + 1`` interfaces, including both domain
    boundaries.  Four ghost cells support the seventh-order stencil.
    """

    q = np.moveaxis(np.asarray(quantity, dtype=np.float64), axis, -1)
    n = q.shape[-1]
    pad_spec = [(0, 0)] * q.ndim
    pad_spec[-1] = (4, 4)
    padded = np.pad(q, pad_spec, mode="wrap" if periodic else "edge")

    if periodic:
        base = np.arange(4, 4 + n)
    else:
        base = np.arange(3, 4 + n)

    def take(offset: int) -> FloatArray:
        return np.take(padded, base + offset, axis=-1)

    qi = take(0)
    qip1 = take(1)

    if scheme == "first_order":
        left = qi
        right = qip1
    elif scheme == "van_leer":
        slope_i = van_leer_slope(qi - take(-1), qip1 - qi)
        slope_ip1 = van_leer_slope(qip1 - qi, take(2) - qip1)
        left = qi + 0.5 * slope_i
        right = qip1 - 0.5 * slope_ip1
    elif scheme == "pdm7":
        if not (0.0 < pdm_alpha <= 1.0):
            raise ValueError("pdm_alpha must be in (0, 1]")

        offsets_left = range(-3, 4)
        high_left = sum(c * take(o) for c, o in zip(PDM7_LEFT_COEFFICIENTS, offsets_left))

        # Equation (3.90): the right state is the mirrored seventh-order stencil.
        high_right = (
            (-1 / 140) * take(4)
            + (5 / 84) * take(3)
            - (101 / 420) * take(2)
            + (319 / 420) * take(1)
            + (107 / 210) * take(0)
            - (19 / 210) * take(-1)
            + (1 / 105) * take(-2)
        )

        inv_alpha = 1.0 / pdm_alpha
        pdm_left = inv_alpha * qi + (1.0 - inv_alpha) * take(-1)
        pdm_right = inv_alpha * qip1 + (1.0 - inv_alpha) * take(2)
        left = _median3(pdm_left, high_left, qi)
        right = _median3(pdm_right, high_right, qip1)
    else:
        raise ValueError(f"unknown reconstruction scheme: {scheme!r}")

    return np.moveaxis(left, -1, axis), np.moveaxis(right, -1, axis)


def rusanov_flux(velocity: FloatArray, left: FloatArray, right: FloatArray) -> FloatArray:
    """Scalar Rusanov flux, algebraically equal to an upwind flux."""

    return 0.5 * (velocity * (left + right) - np.abs(velocity) * (right - left))


def flux_divergence(
    density: FloatArray,
    velocity_x: FloatArray,
    velocity_y: FloatArray,
    *,
    dx: float,
    dy: float,
    scheme: str,
    pdm_alpha: float = 0.2,
    periodic_y: bool = False,
) -> FloatArray:
    """Finite-volume divergence of ``density * velocity``."""

    n = np.asarray(density, dtype=np.float64)
    u = np.asarray(velocity_x, dtype=np.float64)
    v = np.asarray(velocity_y, dtype=np.float64)
    if n.shape != u.shape or n.shape != v.shape:
        raise ValueError("density and velocity arrays must have the same shape")

    left_x, right_x = reconstruct_interfaces(
        n, axis=1, periodic=True, scheme=scheme, pdm_alpha=pdm_alpha
    )
    u_face = 0.5 * (u + np.roll(u, -1, axis=1))
    flux_x = rusanov_flux(u_face, left_x, right_x)
    div_x = (flux_x - np.roll(flux_x, 1, axis=1)) / dx

    left_y, right_y = reconstruct_interfaces(
        n, axis=0, periodic=periodic_y, scheme=scheme, pdm_alpha=pdm_alpha
    )
    if periodic_y:
        v_face = 0.5 * (v + np.roll(v, -1, axis=0))
        flux_y = rusanov_flux(v_face, left_y, right_y)
        div_y = (flux_y - np.roll(flux_y, 1, axis=0)) / dy
    else:
        v_face = np.empty((n.shape[0] + 1, n.shape[1]), dtype=np.float64)
        v_face[1:-1] = 0.5 * (v[:-1] + v[1:])
        v_face[0] = v[0]
        v_face[-1] = v[-1]
        flux_y = rusanov_flux(v_face, left_y, right_y)
        div_y = (flux_y[1:] - flux_y[:-1]) / dy

    return div_x + div_y


def advection_rhs(
    density: FloatArray,
    velocity_x: FloatArray,
    velocity_y: FloatArray,
    **kwargs: object,
) -> FloatArray:
    return -flux_divergence(density, velocity_x, velocity_y, **kwargs)

