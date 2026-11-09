# External inputs

Reduced-EM cases in `cases.json` use analytic Chapman initial conditions and do not require author MAT files or a historical checkpoint. They generate their own density, line state and diagnostics.

The original complete ES Figure 2/3/6 reporting pipelines require ScienceDB V2 author reference arrays. They are not included or silently downloaded. Obtain the dataset associated with Li et al. (2021), *Numerical considerations in the simulation of equatorial spread F*, verify its actual provenance and reuse terms, then place the files below. This migration did not independently verify a dataset DOI/download endpoint, so none is invented here.

| File in `data/reference/` | Expected MD5 from the existing code | Use |
|---|---|---|
| `numDensity.mat` | `F052A4FCEA234ADED59DA7023552D739` | ES figure comparison |
| `circAdvecData.mat` | `BF0542002E253F498C06498DFF54381E` | Author advection validation data |
| `ReadMe.txt` | `A922F4117B0CECFD29566CD26AF017CA` | Dataset variable definitions |

The small original `ReadMe.txt` is retained; it describes the arrays. `esf_sim.reference.verify_reference_files` verifies the above checksums. The full figure CLI can complete numerical propagation before failing on a missing reference file: check external inputs before launching it. Missing external-data tests should be identified explicitly, not counted as successful science acceptance.
