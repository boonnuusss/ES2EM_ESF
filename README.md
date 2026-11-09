# ES2EM_ESF

Lightweight, data-free reproduction of electrostatic ESF and exploratory reduced-electromagnetic EPB experiments. 中文交接说明见 [docs/HANDOFF.md](docs/HANDOFF.md)。

The repository contains source code, frozen historical Python sources, configurations, tests and small historical acceptance reports. Large density histories, checkpoints, movies, external `.mat` data and local environments are excluded. Initial states for the reduced-EM cases are analytic Chapman profiles, so these cases can start from scratch without downloading simulation data.

## Install

Use 64-bit Python 3.12 (the assembly machine used 3.12.3); Python >=3.10 is declared by the original package. Windows x64 is the tested migration target. On a different OS, first confirm that `pypardiso`/MKL can be installed and loaded. The smoke case uses SciPy without PARDISO, but installation still follows the original dependency declaration.

```powershell
git clone https://github.com/boonnuusss/ES2EM_ESF.git
cd ES2EM_ESF
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe scripts/reproduce.py list
.\.venv\Scripts\python.exe scripts/reproduce.py run smoke
```

Linux/macOS: substitute `.venv/bin/python` for `.\.venv\Scripts\python.exe`. Do not copy an old machine's `.deps`, DLL paths or virtual environment. Package ranges are not an exact lock; see [provenance](provenance/ASSEMBLY_ENVIRONMENT.json).

## Run a historical case

```powershell
# Inspect configuration, source selection and protected hash without simulation:
python scripts/reproduce.py check B --stage 60
# Explicitly compute B from its analytic initial state to 60 seconds:
python scripts/reproduce.py run B --stage 60
# Explicit mature-time reconstruction (may be expensive):
python scripts/reproduce.py run B --stage 3150
```

Activate the environment before the latter commands, or use its Python executable. Each case/stage has its own output directory. No command automatically advances from 60 to 300/3150/5000 s. `--resume` resumes only that case/stage directory; no historical checkpoint is shipped. A fresh 3150 s case recomputes the trajectory from zero, not from the separate 60 s directory. After an interrupted run, inspect any `.reproduction.lock` and active process before removing a stale lock.

The runner enforces one numerical thread, records the actual source hashes/environment, verifies terminal arrays, renders the comparison and exits nonzero on scientific acceptance failure. Historical reports in `benchmarks/` are reference evidence; they are not proof that a new machine passed. See [case catalogue](docs/CASES.md), [acceptance policy](docs/ACCEPTANCE.md) and [validation status](docs/VALIDATION.md).

## Electrostatic figures

The original Figure 2/3/6 code and configurations are retained. These full workflows require the authors' `data/reference/numDensity.mat` for the comparison report; this third-party file is not redistributed. Without it, do not run a complete figure workflow expecting acceptance to finish. The reduced-EM runner above does not need it. See [external data](docs/EXTERNAL_DATA.md) and the original [ES workflow](docs/legacy/README_original.md).

## Scientific scope

Always compare local ES, column ES and column EM. EM feedback is `column EM - column ES`; geometry is `column ES - local ES`. Historical mature cases recovered primary EM growth but still lacked a clear Li-like Y bifurcation. Two resolutions and early-time agreement do not establish convergence. Imposed drift and equivalent passive terminal conductance are sensitivity experiments, not a calibrated E-region/PRE model or quantitative prediction.

The code is a research snapshot. No software license has been selected by the author in this migration; publication does not imply an additional reuse license.
