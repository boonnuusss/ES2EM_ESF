# Reduced-EM exploratory EPB growth run

This document defines the current auditable route from the Li local
electrostatic model to a reduced electromagnetic EPB experiment. The
recommended calculation is the 101 x 201, `alpha=0.4`, **v6 three-track**
candidate. It is exploratory: it is neither a reproduction of Li et al.
Figure 3 nor a converged electromagnetic production result.

## Why the old fixed/625-km result is not the reference calculation

The completed historical two-track run is useful as a regression and failure
diagnostic, but its strong EPB suppression must not be interpreted as a robust
EM prediction. Three modelling choices were entangled:

1. Its remote voltage was fixed to zero. In characteristic variables this is
   a lossless short circuit, `w_minus=-w_plus` and `R=-1`, rather than an
   outgoing magnetospheric boundary. The 20,000 km, 1000 km/s line has a 40 s
   round-trip time, so the 3150 s Li plane repeatedly received undamped
   reflected waves.
2. The single `H=625 km` was copied from the 32-cell Alfvén-line spacing. It
   was not a plasma flux-tube column. Using it for both particle/gravity and
   collision columns discarded their very different altitude dependence and
   made the returned FAC/gravity near-cancellation calibration-dependent.
3. A local-ES trajectory was compared directly with a column-modified EM
   trajectory. That two-track difference mixed flux-tube geometry with
   electromagnetic feedback. Early variable-column experiments also did not
   consistently transport the conserved column `H_n n`.

The historical run reached 3150 s without a numerical crash, but produced a
FAC-to-gravity ratio near 0.979 and no visible EM bubble. This is a result for
that explicit reflecting, constant-625-km experiment only. Its configurations
and hashes remain supported for exact regression; they are not the recommended
long-run command.

## v6 matched two-column surface-FAC model

The current candidate keeps the endpoint-inclusive 101 x 201 Li grid, analytic
Chapman initial density, PDM7 transport with `alpha=0.4`, equal-value vertical
potential extrapolation, and the mimetic harmonic/log-RHS closure. The
field-line discretization remains one straight 20,000 km line with constant
`V_A=1000 km/s`, 32 finite volumes, second-order reconstruction, and 44
non-constant transverse Fourier/DCT modes. Modal Woodbury and PARDISO reuse
accelerate the coupled solve without changing its equations.

The remote terminal is energy-passive and matched, so `R=0`. A passive finite
conductance load remains available with
`right_boundary_model="conductance"` and conductance `Sigma_P`; its analytic
reflection coefficient is
`(Sigma_A-Sigma_P)/(Sigma_A+Sigma_P)`, where
`Sigma_A=1/(mu0 V_A)`.

For apex altitude `h_a`, the reduced dipole/Chapman mapping uses the scalar
apex-area volume factor `W=B_a/B` and two independent full-conjugate moments:

```text
H_n(h_a)  = integral W [n_C(s)/n_C(h_a)] ds
H_nu(h_a) = integral W [n_C(s) nu(s)/(n_C(h_a) nu(h_a))] ds.
```

`H_n` is the particle/gravity column weight. `H_nu` is the collision column
weight; equivalently, the collision moment is
`G_nu=H_nu*nu_a` and `nu_eff=G_nu/H_n`. With `R_E=6371 km`, a 100 km lower
integration altitude, and 128-point Gauss--Legendre quadrature, the reference
values are:

| apex altitude (km) | `H_n` (km) | `H_nu` (km) |
|---:|---:|---:|
| 250 | 317.337 | 326.704 |
| 350 | 666.665 | 750.141 |
| 438 | 1327.254 | 1928.054 |
| 550 | 3527.827 | 11668.429 |
| 650 | 9639.779 | 100052.887 |

Thus a single density-weighted length underestimates the collision moment by
factors of about 1.45, 3.31, and 10.38 at 438, 550, and 650 km.

The conservative column closure is

```text
div_perp(H_nu n nu_a grad(psi))
    = -H_n g d_x(n) + (B_a/m_i) J_surface,
```

where `J_surface=N_hemisphere*j_parallel` is the summed outward surface FAC in
A/m2. It is not divided by an artificial averaging length. The discrete
balance is `integral(J_surface dA)=0`, and the Li port power
`integral(Phi J_surface dA)` is paired directly with the Alfvén characteristic
power. Direct Krylov, fixed-point, and Woodbury paths use the same source.

Both column trajectories advance the actual conserved column:

```text
partial_t(H_n n) + div_perp(H_n n v_E) = 0,
```

then divide its conservative tendency by fixed `H_n(y)` to recover the apex
volume-density tendency.

## Three comparison tracks

The candidate sets `comparison_tracks="local_column_em"` and advances three
states from the same float64 Chapman density at every accepted macro time:

- **local-ES**: historical Li closure and
  `partial_t n + div(n v_E)=0`;
- **column-ES**: `H_n/H_nu` column closure and conservative `H_n n`
  transport, with `J_surface=0`;
- **column-EM**: the same column geometry and transport plus the self-consistent
  Alfvén/surface-FAC feedback.

This makes the reported decomposition unambiguous:

```text
geometry effect = column-ES - local-ES
EM effect       = column-EM - column-ES
total effect    = column-EM - local-ES.
```

The decomposition is reported for the final density norm and for the fitted
horizontal-structure growth rate. Fine nonlinear branch positions can still
differ, so the fields and time histories should be inspected alongside scalar
metrics.

## Run, resume, and post-process

Run or resume the v6 candidate:

```text
python -m esf_sim run-em-growth --config configs/em_growth_li_101x201_alpha04.json --em-config configs/em_growth_reduced_em_101x201_alpha04_matched_dipole_chapman.json
```

Resume is the default. Use `--no-resume` only when intentionally starting all
three tracks from `t=0`:

```text
python -m esf_sim run-em-growth --config configs/em_growth_li_101x201_alpha04.json --em-config configs/em_growth_reduced_em_101x201_alpha04_matched_dipole_chapman.json --no-resume
```

The configured output directory is
`output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track`.
Post-process a completed run with:

```text
python -m esf_sim.em_growth_postprocess output/em_growth/exploratory_alpha04_101x201_matched_dipole_chapman_three_track
```

The runner writes an atomic schema-v3 checkpoint containing the three density
tracks, the EM characteristics, the three amplitude histories, retained macro
step, and reduction count. Exact-time density snapshots and `final_states.npz`
use schema v2 and contain all three densities. The snapshot manifest rejects
gaps, mismatched hashes, and frames newer than a resumed stable checkpoint.
The postprocessor writes editable SVG/PDF, PNG, source NPZ, and JSON metrics with
separate geometry, EM, and total effects.

The old schema-v1/v2 paired checkpoints and snapshot schema v1 remain readable
only by `comparison_tracks="legacy_pair"`. The default legacy field is omitted
from the protected trajectory hash, preserving historical scalar-H hashes. A
legacy pair or earlier one-column checkpoint is intentionally rejected by the
v6 three-track runner; never copy or rename it into the new output directory.

## Time stepping and staged execution

The configured EM macro step is 0.25 s. With 625 km Alfvén cells and
`V_A=1000 km/s`, its independent line CFL is 0.4. The local-ES and column-ES
controls each have independent dyadic SSPRK3 subcycling: if only one control
violates its density CFL, that control restarts from the macro-step input and
reaches the same macro endpoint without adding an EM solve. An EM density/line
CFL rejection or a coupling failure rejects the entire trial and halves the
retained macro step. No state or history is committed until all three tracks
accept the same endpoint.

If a required control substep or EM macro step would fall below 1/64 s, the run
is marked `paused_minimum_dt` and preserves the last atomically consistent
checkpoint. Subcycling and rejected-step events are recorded in
`timestep_events.jsonl`.

Before a full 3150 s calculation, verify the fast EM suite and extend the same
v6 checkpoint in reviewed stages (for example 60 s, 300 s, then 3150 s):

```text
python -m esf_sim validate-em
```

Change physical or numerical trajectory controls only through the JSON; such
changes correctly invalidate the checkpoint hash. `t_end_s`, reporting
cadence, output path, and experiment label may be changed for a staged
extension because they do not alter already accepted steps.

## Current limitations

The v6 changes remove known constant-H, reflected-terminal, FAC-unit, and
column-continuity inconsistencies. They do not make the model complete:

- `W=B_a/B` is a scalar apex-volume approximation. A strict dipole mapping
  generally produces a transverse metric tensor in the collision operator and
  a projected gravity term.
- The dynamic Alfvén line uses constant `V_A` and simplified constant
  impedance; realistic field-aligned density, magnetic-field, and ionospheric
  load structure are absent.
- There is no common-mode parallel mass transport, parallel pressure balance,
  chemistry, or end-point particle exchange.
- One numerical line represents mirrored hemispheres; a full two-hemisphere
  metric/SBP implementation and its energy budget are not yet present.
- Transverse resolution, retained-mode, line-grid/CFL, lower-boundary, and
  column-model convergence studies remain required.

Consequently, the v6 output should be described as a controlled three-track
reduced-EM experiment. It cleanly identifies whether a difference comes from
flux-tube geometry or EM feedback, but it is not yet a production prediction of
equatorial plasma-bubble morphology.
