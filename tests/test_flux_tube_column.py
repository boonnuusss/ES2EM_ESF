from __future__ import annotations

import numpy as np
import pytest

from esf_sim.flux_tube_column import dipole_chapman_column_profile


def test_dipole_chapman_column_has_documented_reference_values():
    altitude_km = np.array([250.0, 350.0, 438.0, 550.0, 650.0])
    profile = dipole_chapman_column_profile(
        altitude_km * 1000.0,
        chapman_peak_altitude_m=438_000.0,
        chapman_scale_height_m=70_000.0,
        earth_radius_m=6_371_000.0,
        lower_altitude_m=100_000.0,
        quadrature_order=128,
    )
    expected_gravity_km = np.array(
        [
            317.33664714993546,
            666.6653104147172,
            1327.253718258658,
            3527.827313219548,
            9639.779397734665,
        ]
    )
    expected_collision_km = np.array(
        [
            326.704187561715,
            750.140946061739,
            1928.053732447361,
            11668.428728209667,
            100052.88742973867,
        ]
    )
    np.testing.assert_allclose(
        profile.gravity_column_weight_m / 1000.0,
        expected_gravity_km,
        rtol=2.0e-13,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        profile.collision_column_weight_m / 1000.0,
        expected_collision_km,
        rtol=2.0e-13,
        atol=2.0e-10,
    )
    expected_density_column = np.array(
        [
            5.391143362945419e12,
            1.8943913077850714e17,
            1.3272537182586593e18,
            1.5821515661922747e18,
            1.2079709919590308e18,
        ]
    )
    expected_collision_column = np.array(
        [
            6.870658346306701e12,
            7.724251434701702e16,
            2.3700815003557014e17,
            1.6249583406211053e17,
            1.1396639693878611e17,
        ]
    )
    np.testing.assert_allclose(
        profile.background_density_column_m2,
        expected_density_column,
        rtol=3.0e-13,
    )
    np.testing.assert_allclose(
        profile.background_collision_column_m2_s,
        expected_collision_column,
        rtol=3.0e-13,
    )
    expected_effective_collision = np.array(
        [
            1.274434360905765,
            0.4077431839429691,
            0.1785703417327939,
            0.1027056051609425,
            0.09434530936372963,
        ]
    )
    np.testing.assert_allclose(
        profile.effective_collision_frequency_s,
        expected_effective_collision,
        rtol=3.0e-13,
    )
    # Dimensional identities: Cnu_col=N_col*nu_eff and
    # G_nu=nu_apex*H_nu.
    np.testing.assert_allclose(
        profile.background_collision_column_m2_s,
        profile.background_density_column_m2
        * profile.effective_collision_frequency_s,
        rtol=3.0e-15,
    )
    apex_collision = 26.7 * np.exp(-altitude_km * 1000.0 / 81_400.0)
    np.testing.assert_allclose(
        profile.collision_moment_m_s,
        apex_collision * profile.collision_column_weight_m,
        rtol=3.0e-15,
    )
    assert np.all(np.diff(profile.gravity_column_weight_m) > 0.0)
    assert np.all(np.diff(profile.collision_column_weight_m) > 0.0)
    for values in (
        profile.gravity_column_weight_m,
        profile.collision_column_weight_m,
        profile.background_density_column_m2,
        profile.background_collision_column_m2_s,
        profile.collision_moment_m_s,
        profile.effective_collision_frequency_s,
    ):
        assert not values.flags.writeable
    summary = profile.summary()
    assert summary["model"] == "dipole_chapman_flux_tube_columns"
    assert summary["first_gravity_column_weight_km"] == pytest.approx(
        expected_gravity_km[0]
    )
    assert summary["last_collision_column_weight_km"] == pytest.approx(
        expected_collision_km[-1]
    )


def test_dipole_chapman_quadrature_is_converged_and_validated():
    altitude = np.linspace(250_000.0, 650_000.0, 21)
    coarse = dipole_chapman_column_profile(
        altitude,
        chapman_peak_altitude_m=438_000.0,
        chapman_scale_height_m=70_000.0,
        quadrature_order=64,
    )
    fine = dipole_chapman_column_profile(
        altitude,
        chapman_peak_altitude_m=438_000.0,
        chapman_scale_height_m=70_000.0,
        quadrature_order=128,
    )
    np.testing.assert_allclose(
        coarse.gravity_column_weight_m,
        fine.gravity_column_weight_m,
        rtol=2.0e-13,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        coarse.collision_column_weight_m,
        fine.collision_column_weight_m,
        rtol=2.0e-13,
        atol=5.0e-5,
    )

    with pytest.raises(ValueError, match="exceed"):
        dipole_chapman_column_profile(
            np.array([100_000.0]),
            chapman_peak_altitude_m=438_000.0,
            chapman_scale_height_m=70_000.0,
        )
    with pytest.raises(ValueError, match="at least 16"):
        dipole_chapman_column_profile(
            np.array([250_000.0]),
            chapman_peak_altitude_m=438_000.0,
            chapman_scale_height_m=70_000.0,
            quadrature_order=8,
        )
