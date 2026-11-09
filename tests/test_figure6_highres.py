from pathlib import Path

import numpy as np

from esf_sim.config import SimulationConfig
from esf_sim.figure6 import FIGURE6_CASES, case_config
from esf_sim.figure6_highres import UpdatedHierarchyPotentialSolver
from esf_sim.model import initial_density


def test_updated_hierarchy_solver_reuses_coarsening_and_updates_fine_matrix() -> None:
    base = SimulationConfig.from_json("configs/figure6.json")
    case = FIGURE6_CASES[0]
    config = case_config(base, case, Path("output/test_figure6_highres"))
    solver = UpdatedHierarchyPotentialSolver(config)
    density = initial_density(config)

    potential1, diagnostics1 = solver.solve(density)
    hierarchy = solver.hierarchy
    potential2, diagnostics2 = solver.solve(density * 1.000001)

    assert diagnostics1.rebuilt_preconditioner
    assert not diagnostics2.rebuilt_preconditioner
    assert solver.hierarchy is hierarchy
    assert np.max(np.abs(potential1[[0, -1]])) == 0.0
    assert np.max(np.abs(potential2[[0, -1]])) == 0.0
    assert diagnostics2.relative_residual < 1.0e-7
