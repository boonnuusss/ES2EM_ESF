from __future__ import annotations

import numpy as np

from esf_sim.alfven_1d import AlfvenState
from esf_sim.config import SimulationConfig
from esf_sim.model import ESFModel, initial_density
from esf_sim.potential import PotentialDiagnostics
from esf_sim.reduced_em import (
    CoupledEMState,
    ESLimitAdapter,
    linear_combination,
    zeros_like_state,
)


class FakePotentialSolver:
    def __init__(self, potential: np.ndarray):
        self.potential = potential.copy()
        self.calls = 0

    def solve(self, density: np.ndarray):
        assert density.shape == self.potential.shape
        self.calls += 1
        return self.potential.copy(), PotentialDiagnostics(0, 0.0, False, False)


def tiny_config() -> SimulationConfig:
    return SimulationConfig(
        nx=9,
        ny=9,
        x_min_km=0.0,
        x_max_km=8.0,
        y_min_km=250.0,
        y_max_km=258.0,
        dx_km=1.0,
        dy_km=1.0,
        perturbation_wavelength_km=8.0,
        scheme="pdm7",
        pdm_alpha=0.4,
    )


def coupled(density: np.ndarray, cells: int = 7) -> CoupledEMState:
    return CoupledEMState(
        density.copy(),
        AlfvenState(np.zeros(cells), np.zeros(cells)),
    )


def test_zero_state_and_linear_combination_do_not_mutate_inputs():
    state = CoupledEMState(
        np.arange(12.0).reshape(3, 4),
        AlfvenState(np.arange(5.0), -np.arange(5.0)),
    )
    zero = zeros_like_state(state)
    doubled = linear_combination((1.5, 0.5), (state, state))
    assert np.count_nonzero(zero.density_m3) == 0
    assert np.count_nonzero(zero.alfven.velocity_m_s) == 0
    assert np.array_equal(doubled.density_m3, 2.0 * state.density_m3)
    assert np.array_equal(doubled.alfven.magnetic_t, 2.0 * state.alfven.magnetic_t)


def test_adapter_uses_exact_unforced_legacy_rhs_branch():
    config = tiny_config()
    model = ESFModel(config)
    yy, xx = np.meshgrid(np.arange(9.0), np.arange(9.0), indexing="ij")
    fake = FakePotentialSolver(2.0e6 * np.sin(2.0 * np.pi * xx / 9.0) + yy**2)
    model.potential_solver = fake
    density = initial_density(config)
    expected_rhs, expected_diagnostic = model.evaluate_rhs(density)
    state = coupled(density)
    actual, diagnostic = ESLimitAdapter(model).rhs(123.0, state)
    assert fake.calls == 2
    assert np.array_equal(actual.density_m3, expected_rhs)
    assert np.count_nonzero(actual.alfven.velocity_m_s) == 0
    assert np.count_nonzero(actual.alfven.magnetic_t) == 0
    assert diagnostic.electrostatic == expected_diagnostic
    assert diagnostic.coupling_gain == 0.0


def test_tau_alfven_over_tau_epb_zero_matches_one_legacy_ssprk3_step():
    config = tiny_config().with_updates(dt_s=0.01)
    density = initial_density(config)
    yy, xx = np.meshgrid(np.arange(9.0), np.arange(9.0), indexing="ij")
    potential = 500.0 * (np.sin(2.0 * np.pi * xx / 9.0) + 0.1 * yy)

    legacy = ESFModel(config)
    legacy.potential_solver = FakePotentialSolver(potential)
    expected, _ = legacy.ssprk3_step(density, config.dt_s)

    adapted_model = ESFModel(config)
    adapted_model.potential_solver = FakePotentialSolver(potential)
    state, _ = ESLimitAdapter(adapted_model).ssprk3_step(
        0.0, coupled(density), config.dt_s
    )
    assert np.array_equal(state.density_m3, expected)
    assert np.count_nonzero(state.alfven.velocity_m_s) == 0
    assert np.count_nonzero(state.alfven.magnetic_t) == 0


def test_es_limit_ab3_predictor_corrector_is_bitwise_legacy_equivalent():
    config = tiny_config().with_updates(dt_s=0.01)
    current = initial_density(config)
    previous = current * (1.0 - 2.0e-5)
    previous2 = current * (1.0 - 5.0e-5)
    yy, xx = np.meshgrid(np.arange(9.0), np.arange(9.0), indexing="ij")
    potential = 300.0 * (np.cos(2.0 * np.pi * xx / 9.0) - 0.2 * yy)

    legacy = ESFModel(config)
    legacy.potential_solver = FakePotentialSolver(potential)
    expected, _ = legacy.ab3_predictor_corrector_step(
        current, previous, previous2, config.dt_s
    )

    adapted_model = ESFModel(config)
    adapted_model.potential_solver = FakePotentialSolver(potential)
    actual, _ = ESLimitAdapter(adapted_model).ab3_predictor_corrector_step(
        0.0,
        coupled(current),
        coupled(previous),
        coupled(previous2),
        config.dt_s,
    )
    assert np.array_equal(actual.density_m3, expected)
    assert np.count_nonzero(actual.alfven.velocity_m_s) == 0
    assert np.count_nonzero(actual.alfven.magnetic_t) == 0
