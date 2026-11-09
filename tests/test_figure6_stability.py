import numpy as np

from esf_sim.experimental_time import (
    _archive_stable_checkpoint,
    _load_checkpoint,
    convert_fixed_history_checkpoint,
)


def test_convert_fixed_history_checkpoint_assigns_exact_times(tmp_path):
    source = tmp_path / "fixed.npz"
    current = np.full((3, 4), 3.0)
    previous = np.full((3, 4), 2.0)
    previous2 = np.full((3, 4), 1.0)
    np.savez_compressed(
        source,
        density=current,
        density_previous=previous,
        density_previous2=previous2,
        history_length=3,
        time_s=2300.0,
        dt_s=0.0625,
        step=123,
    )
    destination = tmp_path / "variable.npz"
    convert_fixed_history_checkpoint(source, destination)
    states, times, dt_s, step = _load_checkpoint(destination)
    assert np.array_equal(states[0], current)
    assert np.array_equal(states[1], previous)
    assert np.array_equal(states[2], previous2)
    assert np.allclose(times, [2300.0, 2299.9375, 2299.875])
    assert dt_s == 0.0625
    assert step == 123


def test_stable_checkpoint_archive_is_separate_from_live_file(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "variable_history.npz"
    checkpoint.parent.mkdir(parents=True)
    np.savez_compressed(checkpoint, value=np.asarray([1.0]))
    archive = _archive_stable_checkpoint(checkpoint, 2300.0)
    assert archive.exists()
    assert archive != checkpoint
    assert "t002300.000" in archive.name
    with np.load(archive) as saved:
        assert float(saved["value"][0]) == 1.0
