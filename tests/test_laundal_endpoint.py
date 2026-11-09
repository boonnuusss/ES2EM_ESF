from __future__ import annotations

import numpy as np
import pytest

from esf_sim.laundal_endpoint import (
    DynamicEndpointParameters,
    DynamicEndpointState,
    FieldLineDisplacementState,
    InstantaneousFieldLineClosure,
    InstantaneousLaundalEndpoint,
    finite_alfven_to_instantaneous_slope_ratio,
)


OXYGEN_MASS = 2.66e-26
B0 = 5.0e-5


def make_endpoint(sigma_p=5.0, sigma_h=5.0):
    parameters = DynamicEndpointParameters.from_conductances(
        pedersen_conductance_s=sigma_p,
        hall_conductance_s=sigma_h,
        ion_mass_kg=OXYGEN_MASS,
        background_field_t=B0,
    )
    return InstantaneousLaundalEndpoint(parameters)


def test_parameter_conversion_recovers_requested_conductances():
    endpoint = make_endpoint(3.0, 7.0)
    parameters = endpoint.parameters
    assert parameters.pedersen_conductance_s == pytest.approx(3.0)
    assert parameters.hall_conductance_s == pytest.approx(7.0)
    assert parameters.ion_magnetization == pytest.approx(3.0 / 7.0)


def test_steady_dynamic_load_recovers_pedersen_hall_tensor():
    endpoint = make_endpoint(4.0, 2.5)
    sigma_p = endpoint.parameters.pedersen_conductance_s
    sigma_h = endpoint.parameters.hall_conductance_s

    ex_only = np.array([0.08, 0.0])
    ey_only = np.array([0.0, 0.08])
    assert endpoint.steady_current_a_m(ex_only) == pytest.approx(
        [sigma_p * ex_only[0], sigma_h * ex_only[0]], rel=2.0e-14
    )
    assert endpoint.steady_current_a_m(ey_only) == pytest.approx(
        [-sigma_h * ey_only[1], sigma_p * ey_only[1]], rel=2.0e-14
    )


def test_unforced_ion_motion_decays_and_rotates_exactly():
    endpoint = make_endpoint()
    initial = DynamicEndpointState(np.array([120.0, -35.0]))
    dt = 0.031
    result = endpoint.step(initial, np.zeros(2), dt)
    p = endpoint.parameters
    angle = p.gyrofrequency_s * dt
    expected = np.exp(-p.collision_frequency_s * dt) * np.array(
        [
            np.cos(angle) * 120.0 + np.sin(angle) * -35.0,
            -np.sin(angle) * 120.0 + np.cos(angle) * -35.0,
        ]
    )
    assert result.ion_velocity_m_s == pytest.approx(expected, rel=2.0e-15)


def test_step_field_shows_electron_first_then_ion_relaxation():
    endpoint = make_endpoint(5.0, 5.0)
    electric = np.array([0.05, 0.0])
    initial = endpoint.zero_state()
    current_initial = endpoint.current_a_m(initial, electric)
    current_steady = endpoint.steady_current_a_m(electric)

    # At t=0 the infinitely light electrons carry the transverse current;
    # ions subsequently acquire both Pedersen and Hall motion.
    assert current_initial[0] == pytest.approx(0.0, abs=1.0e-15)
    assert current_initial[1] > current_steady[1] > 0.0

    relaxation_time = 12.0 / endpoint.parameters.collision_frequency_s
    relaxed = endpoint.step(initial, electric, relaxation_time)
    assert endpoint.current_a_m(relaxed, electric) == pytest.approx(
        current_steady, rel=1.0e-5, abs=1.0e-8
    )


def test_kinetic_energy_rate_is_input_power_minus_collision_loss():
    endpoint = make_endpoint(3.0, 6.0)
    state = DynamicEndpointState(np.array([180.0, -70.0]))
    electric = np.array([0.035, -0.012])
    rhs = endpoint.rhs(state, electric).ion_velocity_m_s
    p = endpoint.parameters
    direct_rate = (
        p.surface_density_m2
        * p.ion_mass_kg
        * np.dot(state.ion_velocity_m_s, rhs)
    )
    input_power, collision_loss = endpoint.energy_rate_terms_w_m2(state, electric)
    assert direct_rate == pytest.approx(input_power - collision_loss, rel=2.0e-14)


def test_exact_step_converges_to_steady_state_without_gyro_cfl_constraint():
    endpoint = make_endpoint(2.0, 8.0)
    electric = np.array([0.01, -0.03])
    # Twenty-five collision times leave an analytic exp(-25) transient while
    # still exercising a step far beyond the ion gyroperiod/CFL scale.
    dt = 25.0 / endpoint.parameters.collision_frequency_s
    result = endpoint.step(endpoint.zero_state(), electric, dt)
    assert result.ion_velocity_m_s == pytest.approx(
        endpoint.steady_velocity(electric), rel=5.0e-9, abs=1.0e-8
    )


def test_zero_transit_field_line_is_linear_and_tracks_relative_footpoint_motion():
    closure = InstantaneousFieldLineClosure(length_m=500.0e3, background_field_t=B0)
    state = FieldLineDisplacementState(0.0)
    state = closure.step(
        state,
        magnetospheric_velocity_m_s=500.0,
        electron_velocity_m_s=125.0,
        dt_s=40.0,
    )
    assert state.relative_displacement_m == pytest.approx(15.0e3)
    assert closure.magnetic_perturbation_t(state) == pytest.approx(
        B0 * 15.0e3 / 500.0e3
    )
    s = np.linspace(0.0, closure.length_m, 7)
    profile = closure.displacement_profile_m(
        s,
        ionospheric_displacement_m=-4.0e3,
        magnetospheric_displacement_m=11.0e3,
    )
    assert np.diff(profile) == pytest.approx(np.full(6, 2.5e3))


def test_finite_alfven_line_converges_quadratically_to_laundal_terminal_limit():
    omega = 2.0 * np.pi / 300.0
    length = 500.0e3
    speeds = np.array([2.0e6, 4.0e6, 8.0e6])
    errors = np.array(
        [
            abs(
                finite_alfven_to_instantaneous_slope_ratio(
                    angular_frequency_s=omega,
                    length_m=length,
                    alfven_speed_m_s=speed,
                )
                - 1.0
            )
            for speed in speeds
        ]
    )
    orders = np.log2(errors[:-1] / errors[1:])
    assert np.all(orders > 1.99)
    assert errors[-1] < 3.0e-7
