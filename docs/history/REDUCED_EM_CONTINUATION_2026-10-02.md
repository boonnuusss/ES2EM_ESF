# Reduced-EM continuation status — 2026-10-02

Production 101x201/k4 reached 5000 s; original production hash remains 8547522688d8127afb575a35b8affb857cc97886b0b3aeb301073b4f4d95a02d.
See REDUCED_EM_EPB_T5000_2026-09-30.md for the verified report, plots, energy limitations and 143-test regression result. EM still lacks clear Y bifurcation at 5000 s.

## Refined joint grid/modal study

201x401, (8,8) FAC maxima, 152 modes; same physics, unit gain, matched terminal, all three tracks.
Hash f7aecc67b3ddaae0c9c6325d0ef6fee2b6084060db301c8c4df5d6b9b52ad055.
Output: output/em_growth/resolution_201x401_k8_matched_dipole_chapman_three_track_20260930.
60 s and 300 s acceptance passed. 300 s final/checkpoint/last snapshot are exact. No timestep reductions, max coupling residual 4.511e-12, paired-power defect 6.170e-16. Sampled energy residual 4.442 J (8.971e-6 relative exchange).
Visual QA via view_image completed for postprocess_t300/em_growth_comparison.png.
At 300 s, fine-versus-coarse EM total density L2 difference is 0.000138633; horizontal perturbation L2 difference 0.00157415. EM 100–300 s rate is 3.195274e-4 vs 3.208565e-4 s^-1 coarse (0.414% lower). Pure EM-minus-column effect difference is 3.1054%. This is early-time sensitivity evidence, NOT mature bifurcation convergence.
Comparison: resolution_comparison_t300.json. Full 300 s terminal artifacts are preserved in archive_t300; plots and acceptance remain at their named t300 paths.

## Current next step

Resume refined checkpoint to 3150 s using configs/em_growth_li_201x401_alpha04_resolution.json and tmp/em_growth_resolution_201x401_k8_t3150.json. Never use --no-resume. Never resume production files with the refined hash. The hidden driver <ORIGINAL_CHECKOUT>/tmp/run_em_resolution_mature_20261002.py runs this continuation, terminal array audit, postprocess_t3150 and acceptance_t3150 automatically. stdout/stderr logs in refined output use extension_t3150_20261002 names.
Monitor every 30 minutes, not faster. status time is only a startup marker; checkpoint and diagnostics are the progress truth. Check for existing live processes before restarting. On interruption, resume the same config/checkpoint and allow runner reconciliation. Do not edit or delete existing outputs.

When complete, view_image the final plot; compare with coarse production snapshot_000315.npz/postprocess_t3150 at equal time; distinguish growth, amplitude, morphology, FAC and energy. If needed extend refined run to 5000 s with the same hash and archive t3150 first. Do not claim full grid/modal convergence from two levels or from 300 s agreement.
Only after resolution evidence proceed in user-requested order: physical E-region conductance, default-zero common upward drift (hash-compatible/tested, all tracks), length/VA sensitivity. Keep remaining physical limitations explicit. Do not flip FAC sign, use fixed-voltage shortcut, weaken feedback, git reset or delete user outputs.

Environment: PYTHONPATH=<ORIGINAL_CHECKOUT>/.deps;src; PYPARDISO_MKL_RT=<ORIGINAL_CHECKOUT>/.deps/Library/bin/mkl_rt.3.dll; prepend its Library/bin to PATH; MKL_NUM_THREADS=10. Working directory must be <ORIGINAL_CHECKOUT> (not the E: checkout).
