from pathlib import Path

from esf_sim.config import SimulationConfig
from esf_sim.figure6 import FIGURE6_CASES, case_config
from esf_sim.figure6_galerkin import GalerkinUpdatedPotentialSolver
from esf_sim.model import initial_density


def test_galerkin_update_keeps_current_multilevel_operator_consistent() -> None:
    base = SimulationConfig.from_json("configs/figure6.json")
    config = case_config(base, FIGURE6_CASES[0], Path("output/test_galerkin"))
    solver = GalerkinUpdatedPotentialSolver(config)
    density = initial_density(config)

    _, first = solver.solve(density)
    hierarchy = solver.hierarchy
    _, second = solver.solve(density * 1.000001)

    assert first.rebuilt_preconditioner
    assert not second.rebuilt_preconditioner
    assert solver.hierarchy is hierarchy
    assert second.iterations < 40
    assert second.relative_residual < 1.0e-7
