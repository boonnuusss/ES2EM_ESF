"""Maximum-principle-preserving wrapper for optimized finite-volume kernels."""

from __future__ import annotations

import numpy as np

from .optimized_numerics import reconstruct_interfaces as _raw_reconstruct


def reconstruct_interfaces(quantity, *, axis, periodic, scheme, pdm_alpha=0.2):
    left, right = _raw_reconstruct(
        quantity, axis=axis, periodic=periodic, scheme=scheme, pdm_alpha=pdm_alpha
    )
    if scheme != "pdm7":
        return left, right
    q = np.moveaxis(np.asarray(quantity, dtype=np.float64), axis, -1)
    if periodic:
        q_left, q_right = q, np.roll(q, -1, axis=-1)
    else:
        q_left = np.concatenate((q[..., :1], q), axis=-1)
        q_right = np.concatenate((q, q[..., -1:]), axis=-1)
    lower = np.moveaxis(np.minimum(q_left, q_right), -1, axis)
    upper = np.moveaxis(np.maximum(q_left, q_right), -1, axis)
    return np.clip(left, lower, upper), np.clip(right, lower, upper)


def _flux(velocity, left, right):
    return velocity * np.where(velocity >= 0.0, left, right)


def advection_rhs(
    density, velocity_x, velocity_y, *, dx, dy, scheme, pdm_alpha=0.2,
    periodic_y=False,
):
    n = np.asarray(density, dtype=np.float64)
    u = np.asarray(velocity_x, dtype=np.float64)
    v = np.asarray(velocity_y, dtype=np.float64)
    if n.shape != u.shape or n.shape != v.shape:
        raise ValueError("density and velocity arrays must have the same shape")
    lx, rx = reconstruct_interfaces(
        n, axis=1, periodic=True, scheme=scheme, pdm_alpha=pdm_alpha
    )
    fx = _flux(0.5*(u+np.roll(u, -1, axis=1)), lx, rx)
    divergence = (fx-np.roll(fx, 1, axis=1))/dx
    ly, ry = reconstruct_interfaces(
        n, axis=0, periodic=periodic_y, scheme=scheme, pdm_alpha=pdm_alpha
    )
    if periodic_y:
        fy = _flux(0.5*(v+np.roll(v, -1, axis=0)), ly, ry)
        divergence += (fy-np.roll(fy, 1, axis=0))/dy
    else:
        vface = np.empty((n.shape[0]+1, n.shape[1]))
        vface[1:-1] = 0.5*(v[:-1]+v[1:])
        vface[0] = v[0]
        vface[-1] = v[-1]
        fy = _flux(vface, ly, ry)
        divergence += (fy[1:]-fy[:-1])/dy
    return -divergence


__all__ = ["advection_rhs", "reconstruct_interfaces"]
