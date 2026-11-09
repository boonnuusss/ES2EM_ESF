# ESF production solver optimization

This branch is intended for predictive ESF simulations rather than exact
floating-point reproduction of Li et al. (2021). Historical Figure 2/3/6
drivers remain unchanged.

## Recommended algorithm

- Density flux: seventh-order upwind PDM reconstruction with the existing
  local maximum-principle guard, evaluated by the allocation-conscious kernel.
- Potential: point-centred equation (2.21) while the maximum cell Peclet number
  is below 0.75; one-way switch to a positive exponential-fitted conservative
  operator after that threshold.
- Linear algebra: cached five-point CSR topology and PARDISO symbolic reuse;
  the exact relative residual is checked after every solve.
- Time: true non-uniform third-order Adams-Bashforth using timestamped RHS
  history. CFL changes retain history. Two SSPRK3 steps are used only when an
  imported checkpoint has no RHS history.
- Step control: CFL safety factor 0.85, growth limited to 1.25 per accepted
  step, trial rejection on CFL or positivity failure.

The one-way potential switch is important. A centred drift stencil ceases to
be monotone when a local cell Peclet number exceeds one. Continuing to use it
can create a non-physical near-null mode, extreme ExB velocity, and a time-step
collapse even when the direct linear solve has a tiny algebraic residual.

## Commands

Benchmark a representative late checkpoint:

```powershell
$env:PYTHONPATH='.deps;src'
python -m esf_sim.optimization_benchmark `
  --config configs/figure3.json `
  --checkpoint output/best_baseline/figure3_v2_timeseries/extended_1000s_round2/runs/alpha_0p4/checkpoints/pdm7.npz `
  --output output/optimization/late_alpha04_benchmark.json
```

The compatibility production driver uses the midpoint predictor:

```powershell
python -m esf_sim.production --config configs/figure3.json `
  --checkpoint INPUT.npz --output OUTPUT.npz --duration 100
```

For new scientific runs, call `advance_checkpoint_ab3` from
`esf_sim.high_order_time`; this is the recommended third-order path. The old
midpoint predictor is retained as a more dissipative A/B fallback.

## Measured 201 x 401 late-state result

At alpha=0.4 and t=5011.125 s, the centred stencil has maximum cell Peclet
numbers 1.70 (x) and 1.13 (y). It produces a maximum speed of 14.7 km/s and a
safe dt of 0.0156 s. The exponential operator gives 1.47 km/s and a safe dt of
0.145 s on the identical density array.

The cached exponential potential agrees with the legacy implementation to a
relative L2 difference below 4e-14. The optimized PDM7 RHS agrees to about
6e-15. On this machine, the flux kernel is about 2.3x faster and the combined
projected physical-time throughput is about 12x higher. A real 20 s extension
completed with dt=0.064--0.125 s, no negative density, and one rejected step.

## Accuracy and scope

The exponential potential stencil is second-order consistent and monotone.
The density reconstruction remains seventh-order in smooth regions; its PDM
guard reduces order locally near extrema as required for non-oscillatory
transport. The non-uniform Adams-Bashforth weights integrate a quadratic RHS
exactly and demonstrate third-order convergence on a scalar manufactured ODE.

The operator switch changes late nonlinear morphology relative to the formal
paper-calibration baseline. Results should therefore record the switch time,
maximum Peclet number, and operator mode. Do not mix the optimized outputs into
pixel-level paper-reproduction acceptance tests.
