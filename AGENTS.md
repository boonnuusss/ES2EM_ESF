# Project handoff

Read `docs/HANDOFF.md`, `docs/CASES.md`, `docs/ACCEPTANCE.md`, and `docs/VALIDATION.md` before changing physics or claiming reproduction. `cases.json` is the executable catalogue. Keep historical evidence in `benchmarks/` and source snapshots in `source_history/` immutable. New results belong in ignored `output/`.

Run the smoke case and targeted regression checks before long experiments. Never automatically start or extend a production trajectory. Historical scripts and docs may mention old computer paths, PIDs, schedules and unfinished runs; those are historical observations, not current instructions. Use `scripts/reproduce.py`, not the historical recovery commands.

Preserve the local ES / column ES / column EM controls, physical FAC sign, unit feedback, separate particle/conductance columns and original acceptance policy. Report failures as failures. Do not weaken thresholds to claim reproduction. Configuration hash equality alone does not establish identical source or numerical environment. Track source SHA256 and actual dependencies too.
