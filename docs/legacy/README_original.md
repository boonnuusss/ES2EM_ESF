# Independent reproduction of Li et al. (2021) Figure 2

This package independently solves the two-dimensional electrostatic ESF model and
compares first-order upwind, second-order van Leer, and seventh-order PDM finite-volume
transport. The ScienceDB arrays under data/reference are read only when building the
acceptance report; they never enter the potential solve or density update.

## Installation

Python 3.10 or later is required. The active best baseline uses Intel MKL PARDISO through `pypardiso` and applies equal-value extrapolation to the vertical potential boundary.

    python -m pip install -e ".[test]"

## Commands

Run the 256-by-256 slotted-cylinder test, elliptic manufactured solution, and temporal
accuracy test:

    python -m esf_sim validate

Run the nine fast reduced-electromagnetic checks and optionally archive the JSON
report:

    python -m esf_sim validate-em
    python -m esf_sim validate-em --output output/em_validation

`validate-em` uses analytic relations and synthetic small grids only. It does not
load a formal Li snapshot, invoke PARDISO, or start a long EPB simulation. The
command exits successfully only when all nine checks pass; every stage reports
its name, status, measured metric, and threshold in execution order.

The recommended checkpointed nonlinear reduced-EM experiment is the explicitly
exploratory 101-by-201 **v6 three-track** configuration:

    python -m esf_sim run-em-growth --config configs/em_growth_li_101x201_alpha04.json --em-config configs/em_growth_reduced_em_101x201_alpha04_matched_dipole_chapman.json

The command resumes its schema-v3 checkpoint by default; add `--no-resume` for
a deliberate fresh trajectory.  After completion (or after a staged checkpoint
has been promoted to a completed run), build the auditable comparison with:

    python -m esf_sim.em_growth_postprocess output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track

This configuration starts local-ES, column-ES, and column-EM from the same
analytic Chapman array.  It uses an energy-passive matched remote terminal,
independent dipole/Chapman particle and collision columns (`H_n` and `H_nu`),
summed two-hemisphere surface FAC, and conservative transport of `H_n n` in
both column tracks.  The report separates the geometry effect
`column-ES - local-ES`, the electromagnetic effect `column-EM - column-ES`,
and their total `column-EM - local-ES`.

The older `em_reduced_preliminary.json`, constant-625-km, fixed-voltage, and
matched-constant-column configurations are retained for regression and A/B
diagnosis only.  They must not be presented as the formal exploratory result:
625 km was the Alfvén grid spacing rather than a derived plasma column, the
fixed remote end has reflection coefficient `R=-1`, and a two-track comparison
confounds column geometry with EM feedback.  The rationale, archive schemas,
staged-run guidance, and remaining model limitations are recorded in
`docs/reduced_em_exploratory_run.md`.

Run or resume all three 201-by-401 ESF simulations, the relative 1e-12 sensitivity case,
and export Figure 2:

    python -m esf_sim reproduce-figure2 --config configs/figure2.json

To rebuild the figure and reports from existing snapshots:

    python -m esf_sim reproduce-figure2 --config configs/figure2.json --skip-simulation --skip-sensitivity

The active-baseline outputs are written under `output/best_baseline`: compressed density arrays, archived configuration files, JSON-lines solver diagnostics, checkpoints, morphology metrics, the author-data acceptance report, and SVG/PDF/PNG/TIFF figure files. PNG and TIFF are 600 dpi; text remains editable in SVG and PDF. Existing harmonic-baseline outputs under `output/` and the former point-centered Dirichlet baseline under `output/legacy_point_centered_dirichlet/` are retained.

## Active best baseline

The official `configs/figure2.json`, `configs/figure3.json`, and `configs/figure6.json` select the point-centered five-point realization of equation (2.21), centered logarithmic coefficient gradients, equal-value vertical potential ghosts, and Intel MKL PARDISO. The machine-readable selection record is `output/best_baseline/baseline_manifest.json`.

The former point-centered zero-potential boundary is available through the `*_legacy_point_centered_dirichlet.json` configurations. The older conservative harmonic-face implementation remains available through the `*_legacy_harmonic.json` configurations.

Run or resume Figure 3 and Figure 6 with:

    python -m esf_sim reproduce-figure3 --config configs/figure3.json
    python -m esf_sim.figure6 --config configs/figure6.json

## Reduced-EM validation scope

The electromagnetic branch is now a dimensional **2-D transverse + 1-D
field-aligned reduced-EM model**. Stages 1--7 are reported in this fixed order:

1. uniform-Alfven-speed wave accuracy and the Walen relation;
2. analytic reflection from a conductance load;
3. propagation and reflection across a non-uniform-Alfven-speed impedance jump;
4. lossless kinetic-plus-magnetic energy and Poynting/Joule power budgets;
5. the finite-length, high-Alfven-speed limit and Laundal dynamic endpoint;
6. the exact Li electrostatic limit;
7. a frozen-Li-snapshot one-way drive.

Stage 8 retains the bounded toy controller as an independently testable numerical
guard. Stage 9 is the physical feedback test: real transverse Fourier/DCT modes
carry a discrete perpendicular wavenumber and produce a field-aligned current.
The v6 column closure lifts the two mirrored hemispheres into a summed outward
surface current `J_surface` and solves

    div_perp(H_nu n nu_a grad(psi))
        = -H_n g d_x(n) + (B_a/m_i) J_surface.

It therefore does not divide surface FAC by the former artificial 625 km
averaging length.  The modal projection/lifting remains area-adjoint, the
constant transverse mode is excluded, and its Li-side power pairing is
`integral(Phi J_surface dA)`.  Direct Krylov, fixed-point, and modal-Woodbury
solvers implement the same dimensional closure.  The literal `gain=0` branch
still dispatches to the historical local electrostatic RHS and SSPRK3 step, so
legacy regression hashes and trajectories remain protected.

Stage 9 runs a short nonlinear 9x9x8 two-way experiment and checks positive
density, zero-net surface FAC, coupling convergence, solver agreement, and
modal/grid port-power pairing.  These tests do **not** establish a converged
101x201 or 201x401 electromagnetic EPB prediction.  The dipole/Chapman mapping
currently uses the scalar apex-volume factor `W=B_a/B`, not the full mapped
transverse metric tensor.  The dynamic Alfvén line has constant `V_A` (and its
associated simplified impedance), and the model has no common-mode parallel
mass transport.  Field-dependent Alfvén structure, metric/SBP refinement,
transverse/mode convergence, and parallel plasma redistribution remain required
before a production interpretation.

## Reproduction choices

- All internal quantities use SI units.
- The horizontal grid is periodic; the vertical density and potential boundaries use equal-value extrapolation. The grid is displayed after shifting 0--250 km to
  -125--125 km.
- The upper boundary is 650 km. The 600 km value appearing once in the Chinese thesis
  is inconsistent with the English paper, the 401-by-1-km grid, and the plotted axis.
- All independently simulated panels are saved at 3150 s. ScienceDB V2 stores the
  first-order author reference at 2800 s, van Leer at 3150 s, and alpha=0.2 PDM7 at
  3150 s; these source times are retained in the acceptance report.
- Fine-branch pixel alignment is not an acceptance condition because round-off-scale
  perturbations can select different nonlinear branches.
