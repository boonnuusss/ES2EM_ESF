"""Test migration-specific protection against accepting corrupt terminal data."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('reproduction_runner', Path(__file__).resolve().parents[1]/'scripts/reproduce.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _write_state(root, *, final_density=2.0, snapshot_time=60, cp_hash='expected'):
    root.mkdir(exist_ok=True)
    (root/'snapshots').mkdir(exist_ok=True)
    densities = {key:np.full((2,2),2.0) for key in ('baseline_density_m3','column_control_density_m3','coupled_density_m3')}
    state = densities | {'time_s':60.,'configuration_hash':cp_hash,'coupled_w_plus_v':np.zeros((1,2)),'coupled_w_minus_v':np.zeros((1,2))}
    np.savez(root/'checkpoint.npz',**state)
    final = dict(state)
    final['coupled_density_m3'] = np.full((2,2),final_density)
    np.savez(root/'final_states.npz',**final)
    np.savez(root/'snapshots/snapshot_000006.npz',**(state|{'time_s':snapshot_time}))


def test_terminal_audit_accepts_consistent_state(tmp_path):
    _write_state(tmp_path)
    assert runner.terminal_audit(tmp_path,60,'expected')['passed']


@pytest.mark.parametrize('overrides',[
    {'final_density':3.}, {'final_density':float('nan')},
    {'snapshot_time':50.}, {'cp_hash':'wrong'},
])
def test_terminal_audit_rejects_corruption(tmp_path,overrides):
    _write_state(tmp_path,**overrides)
    with pytest.raises((ValueError,AssertionError)):
        runner.terminal_audit(tmp_path,60,'expected')
