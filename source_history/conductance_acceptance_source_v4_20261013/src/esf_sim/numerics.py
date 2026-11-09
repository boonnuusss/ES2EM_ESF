"""Finite-volume reconstructions with a monotonicity-preserving PDM closure."""

from __future__ import annotations

import numpy as np

from . import _numerics_impl as _impl


PDM7_LEFT_COEFFICIENTS = _impl.PDM7_LEFT_COEFFICIENTS
rusanov_flux = _impl.rusanov_flux
van_leer_slope = _impl.van_leer_slope
_raw_reconstruct_interfaces = _impl.reconstruct_interfaces


def reconstruct_interfaces(
    quantity: np.ndarray,
    *,
    axis: int,
    periodic: bool,
    scheme: str,
    pdm_alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct interface states and enforce the PDM local maximum principle."""

    left, right = _raw_reconstruct_interfaces(
        quantity,
        axis=axis,
        periodic=periodic,
        scheme=scheme,
        pdm_alpha=pdm_alpha,
    )
    if scheme != "pdm7":
        return left, right

    q = np.moveaxis(np.asarray(quantity, dtype=np.float64), axis, -1)
    if periodic:
        q_left = q
        q_right = np.roll(q, -1, axis=-1)
    else:
        q_left = np.concatenate((q[..., :1], q), axis=-1)
        q_right = np.concatenate((q, q[..., -1:]), axis=-1)
    lower = np.minimum(q_left, q_right)
    upper = np.maximum(q_left, q_right)
    lower = np.moveaxis(lower, -1, axis)
    upper = np.moveaxis(upper, -1, axis)
    return np.clip(left, lower, upper), np.clip(right, lower, upper)


_impl.reconstruct_interfaces = reconstruct_interfaces


def flux_divergence(*args, **kwargs):
    return _impl.flux_divergence(*args, **kwargs)


def advection_rhs(*args, **kwargs):
    return _impl.advection_rhs(*args, **kwargs)
