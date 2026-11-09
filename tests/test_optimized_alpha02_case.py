from esf_sim.config import SimulationConfig
from esf_sim.model import initial_density


def test_alpha02_optimized_case_uses_paper_initial_state():
    config = SimulationConfig.from_json("configs/figure3.json").with_updates(
        pdm_alpha=0.2, t_end_s=5050.0
    )
    density = initial_density(config)
    assert density.shape == (401, 201)
    assert density.min() < 2e7
    assert density.max() == 1.1e12
