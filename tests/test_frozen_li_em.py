from __future__ import annotations

import numpy as np
import pytest

from esf_sim.frozen_li_em import (
    FrozenDriveSample,
    FrozenLiDrive,
    LiSnapshot,
    MagneticNormalization,
    OneWayAlfvenExperiment,
    SmoothRamp,
)


class FakePotentialSolver:
    def __init__(self) -> None:
        self.calls = 0
        self.received: list[np.ndarray] = []

    def solve(self, density_m3: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        self.calls += 1
        self.received.append(density_m3)
        # Deliberately modify the work array: the LiSnapshot must still not
        # change, which catches accidental sharing with legacy solvers.
        density_m3[0, 0] = -123.0
        psi = np.arange(density_m3.size, dtype=np.float64).reshape(density_m3.shape)
        return psi, {"solve_calls": self.calls}


def _make_drive(
    snapshot: LiSnapshot,
    solver: FakePotentialSolver,
    *,
    ramp: SmoothRamp | None = None,
) -> FrozenLiDrive:
    return FrozenLiDrive(
        snapshot,
        solver,
        normalization=MagneticNormalization(4.0e-5),
        project_potential=lambda psi: psi.mean(axis=0),
        ramp=ramp or SmoothRamp(start_s=2.0, rise_time_s=4.0),
    )


def test_li_snapshot_is_deep_copied_and_strongly_read_only() -> None:
    source = np.arange(12.0).reshape(3, 4) + 1.0
    snapshot = LiSnapshot(source, time_s=1700.0)

    source[:] = 999.0
    np.testing.assert_array_equal(snapshot.density_m3, np.arange(12.0).reshape(3, 4) + 1.0)
    assert not snapshot.density_m3.flags.writeable
    with pytest.raises(ValueError):
        snapshot.density_m3[0, 0] = 3.0
    with pytest.raises(ValueError):
        snapshot.density_m3.setflags(write=True)


def test_magnetic_normalization_uses_phi_equals_b_psi_in_si_units() -> None:
    normalization = MagneticNormalization(background_field_t=5.0e-5)
    psi_m2_s = np.array([-2.0e6, 0.0, 4.0e6])

    phi_v = normalization.electric_potential_v(psi_m2_s)

    np.testing.assert_allclose(phi_v, [-100.0, 0.0, 200.0])
    np.testing.assert_allclose(normalization.stream_function_m2_s(phi_v), psi_m2_s)
    assert not phi_v.flags.writeable


def test_smooth_ramp_is_c1_continuous_and_bounded() -> None:
    ramp = SmoothRamp(start_s=10.0, rise_time_s=8.0)
    assert ramp(9.0) == 0.0
    assert ramp(10.0) == 0.0
    assert ramp(14.0) == 0.5
    assert ramp(18.0) == 1.0
    assert ramp(30.0) == 1.0

    epsilon = 1.0e-5
    left_start_slope = (ramp(10.0) - ramp(10.0 - epsilon)) / epsilon
    right_start_slope = (ramp(10.0 + epsilon) - ramp(10.0)) / epsilon
    left_end_slope = (ramp(18.0) - ramp(18.0 - epsilon)) / epsilon
    right_end_slope = (ramp(18.0 + epsilon) - ramp(18.0)) / epsilon
    assert abs(left_start_slope - right_start_slope) < 1.0e-6
    assert abs(left_end_slope - right_end_slope) < 1.0e-6


def test_frozen_drive_closes_li_once_then_reuses_immutable_projection() -> None:
    density = np.arange(12.0).reshape(3, 4) + 10.0
    snapshot = LiSnapshot(density, time_s=1200.0)
    solver = FakePotentialSolver()
    projector_calls: list[np.ndarray] = []

    def project(psi: np.ndarray) -> np.ndarray:
        projector_calls.append(psi)
        return psi[0] - psi[-1]

    drive = FrozenLiDrive(
        snapshot,
        solver,
        normalization=MagneticNormalization(2.0e-5),
        project_potential=project,
        ramp=SmoothRamp(start_s=0.0, rise_time_s=10.0),
    )

    assert solver.calls == 1
    assert len(projector_calls) == 1
    assert solver.received[0].flags.writeable
    np.testing.assert_array_equal(snapshot.density_m3, density)
    assert drive.closure_diagnostics == {"solve_calls": 1}

    before = drive.projected_stream_function_m2_s.copy()
    zero = drive.at(0.0)
    half = drive.at(5.0)
    full = drive.at(10.0)
    late = drive.at(100.0)

    assert solver.calls == 1
    assert len(projector_calls) == 1
    np.testing.assert_array_equal(zero.stream_function_m2_s, 0.0 * before)
    np.testing.assert_allclose(half.stream_function_m2_s, 0.5 * before)
    np.testing.assert_array_equal(full.stream_function_m2_s, before)
    np.testing.assert_array_equal(late.stream_function_m2_s, before)
    np.testing.assert_allclose(full.electric_potential_v, 2.0e-5 * before)
    np.testing.assert_array_equal(drive.projected_stream_function_m2_s, before)
    np.testing.assert_array_equal(snapshot.density_m3, density)
    assert not full.stream_function_m2_s.flags.writeable
    assert not full.electric_potential_v.flags.writeable


class FakeLine:
    def __init__(self) -> None:
        self.calls: list[tuple[object, FrozenDriveSample]] = []

    def rhs(self, state: object, drive: FrozenDriveSample) -> tuple[str, object, float]:
        self.calls.append((state, drive))
        return "line_rhs", state, float(np.max(drive.electric_potential_v))


def test_one_way_experiment_passes_only_prescribed_drive_to_line_rhs() -> None:
    snapshot = LiSnapshot(np.ones((3, 4)))
    solver = FakePotentialSolver()
    drive = _make_drive(snapshot, solver, ramp=SmoothRamp(0.0, 2.0))
    line = FakeLine()
    experiment = OneWayAlfvenExperiment(line, drive)
    state = object()

    result = experiment.rhs(1.0, state)

    assert result[0] == "line_rhs"
    assert result[1] is state
    assert solver.calls == 1
    assert len(line.calls) == 1
    received_state, received_drive = line.calls[0]
    assert received_state is state
    assert isinstance(received_drive, FrozenDriveSample)
    assert received_drive.envelope == 0.5
    assert not hasattr(received_drive, "field_aligned_current")
    assert not hasattr(received_drive, "density_m3")

    # Repeated field-line evaluations never call back into the Li closure.
    experiment.rhs(2.0, state)
    experiment.rhs(4.0, state)
    assert solver.calls == 1
    np.testing.assert_array_equal(snapshot.density_m3, np.ones((3, 4)))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: LiSnapshot(np.array([[1.0, np.nan]])), "finite"),
        (lambda: LiSnapshot(np.array([[1.0, -1.0]])), "non-negative"),
        (lambda: MagneticNormalization(0.0), "positive"),
        (lambda: SmoothRamp(rise_time_s=0.0), "positive"),
    ],
)
def test_public_inputs_are_validated(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
