import numpy as np

from esf_sim.config import SimulationConfig
from esf_sim.model import initial_density as paper_initial_density
from esf_sim.optimized_alpha04_run_corrected import paper_exact_initial_density


def test_corrected_optimized_workflow_uses_paper_exact_initial_condition():
    config=SimulationConfig.from_json("configs/figure3.json").with_updates(
        pdm_alpha=0.4,t_end_s=5011.0
    )
    expected=paper_initial_density(config)
    actual=paper_exact_initial_density(config)
    assert np.array_equal(actual,expected)
    assert np.isclose(float(actual.max()),1.1e12)
    assert float(actual.min()) < 2e7
