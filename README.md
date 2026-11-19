# F1 Pit-Stop Timing

Senior seminar capstone, Fort Hays State University (Seminar for Software Engineering), 2026-2027.
Author: Paulen Chhun.

## Goal

Learn which lap a Formula 1 car should pit on, and test that decision inside a race simulator.

The project builds on TUM's open-source race simulation and its Virtual Strategy Engineer (VSE)
([TUMFTM/race-simulation](https://github.com/TUMFTM/race-simulation)). TUM's supervised VSE was
trained on what teams actually did, so it imitates teams. This project trains the pit-timing
decision on simulator-optimal labels instead: for sampled race situations, every candidate pit
lap is simulated and the one that gives the best finishing position becomes the label.

Scope: dry races only; pit timing only (compound choice uses TUM's existing network); two
regulation eras with separate parameters (2022-2025 and 2026). Reinforcement learning is out
of scope for now.

Tracks: Barcelona (high tyre wear), Monza (low wear), Singapore (frequent safety cars) and
Austin (in between).

## Status (October 2026)

| Work | Status |
|---|---|
| TUM simulator running locally, original source unchanged | Done |
| VSE decision point located and documented (`notes/vse_interface.md`) | Done |
| Fixed-lap policy plugged in for one driver | Done |
| Simulator vs real result, Austria 2019 (`notes/sim_vs_real_2019.md`) | Done |
| Pit-lap sweep, Austria 2019, 207 runs (`notes/pit_lap_sweep.md`) | Done |
| FastF1 workbooks for 2019 and 2022-2026 with a full data dictionary | Done |
| Own fitted parameters written into TUM parameter files (`notes/fitted_parameters.md`, `notes/fitted_parameters_study_circuits.md`) | Done for 2019: Austria and the four study circuits |
| Held-out check and quadratic wear where it predicts better (`notes/heldout_check.md`) | Done |
| Label generator: Monte Carlo per-lap scores for 14 one-stop drivers (`notes/label_generator.md`) | Done |
| Race situation sampling, methodology step 4: 860,000 situations from 80 driver-races (`notes/situation_sampling.md`) | Done for the four one-stop 2019 races |
| Retrained pit-timing network | Planned |
| Evaluation against the original VSE and real strategies | Planned |

## Layout

```
f1_to_excel.py          FastF1 -> one Excel workbook per season (dry races only)
download_data.command   macOS double-click wrapper: F1TV login check + download
data_pipeline/          results export and the 2019 simulator-vs-real comparison
integration/            our code that plugs into TUM (policies, experiment runner, sweep)
race-simulation/        TUM's simulator, unmodified snapshot (LGPL-3.0)
notes/                  findings, parameter mapping, environment setup, evidence logs
f1_data/                generated workbooks (not in git); ../f1-race-data/ is also searched
fastf1_cache/           FastF1 download cache (not in git)
```

## Setup

Two separate Python environments are required, because FastF1 needs Python 3.10+ and TUM's
pinned dependencies need Python 3.8. They exchange data only through files.

| Environment | Python | Used for |
|---|---|---|
| `.venv-data` | 3.10+ | FastF1, Excel export, analysis, parameter fitting |
| `.venv-tum` | 3.8 | everything in `race-simulation/` and `integration/` |

```sh
python3.10 -m venv .venv-data
.venv-data/bin/pip install fastf1 openpyxl pandas numpy matplotlib pyjwt

# TUM: see notes/environment.md for the exact Python 3.8 build used on macOS
.venv-tum/bin/pip install -r race-simulation/requirements.txt
```

Exact package versions are in `notes/requirements-data.lock.txt` and
`notes/requirements-tum.lock.txt`. On Apple Silicon, TUM's TensorFlow 2.2 cannot run under
Rosetta (no AVX); see `notes/environment.md` for the workaround and its limits.

On 64-bit Windows, build `.venv-tum` with the setup script below. Then use
`.venv-tum/Scripts/python.exe` in place of `.venv-tum/bin/python` in the commands in this file.

```sh
powershell -ExecutionPolicy Bypass -File integration/setup_windows.ps1
```

## Data

```sh
# one season, quick test on three races
.venv-data/bin/python f1_to_excel.py --years 2024 --rounds 1 2 3

# full dataset
.venv-data/bin/python f1_to_excel.py --years 2019 2022 2023 2024 2025 2026

# refresh only the legends of existing workbooks (no download)
.venv-data/bin/python f1_to_excel.py --relabel f1_data/F1_2025_dry_races.xlsx
```

- Output: `f1_data/F1_<year>_dry_races.xlsx`. Every column header has a hover note with its
  meaning, unit and use; the DataDictionary and Legend sheets explain every column and code.
- A race is excluded as WET/MIXED if any lap used intermediate or wet tyres, or if any rain was
  recorded during the race. Excluded races stay listed in the Races sheet with the reason.
- Every season except 2021 and 2022 needs an F1TV subscription token, 2019 included:
  `python -m fastf1 auth f1tv --authenticate`. Tokens expire after a few days.
- Rate limits: if races show FAILED, re-run the same command. Cached races load quickly.

## Simulator experiments

```sh
.venv-tum/bin/python integration/run_experiment.py baseline
.venv-tum/bin/python integration/run_experiment.py fixed --driver VER --pit-lap 25 --compound A3
.venv-tum/bin/python integration/run_experiment.py realstrategy
integration/run_sweep.sh

# label generator: every candidate pit lap, one driver (see notes/label_generator.md)
.venv-tum/bin/python integration/label_generator.py --driver VER --compound A3 --deterministic
.venv-tum/bin/python integration/label_generator.py --driver VER --compound A3 --runs 10
# Monte Carlo labels on the fitted files, then the per-lap score table
integration/run_mc_labels.sh 500 28
.venv-data/bin/python data_pipeline/summarise_labels.py --runs 500
# race situations for every driver, then the per-situation rows (see notes/situation_sampling.md)
.venv-tum/bin/python integration/run_situations.py --runs 200 --jobs 28
.venv-data/bin/python data_pipeline/build_situations.py integration/output/situations/*_2019_fitted_*_mc200

# our fitted parameters: fit from the workbook, write the fitted race file, then pass its path
# with --race (see notes/fitted_parameters.md)
.venv-data/bin/python data_pipeline/fit_parameters.py --year 2019 --round 9
.venv-data/bin/python data_pipeline/fit_traffic.py --year 2019 --round 9
.venv-tum/bin/python integration/write_fitted_pars.py --fit notes/evidence/fitted_parameters/Spielberg_2019_fit.json
.venv-tum/bin/python integration/run_experiment.py realstrategy --race integration/parameters/pars_Spielberg_2019_fitted.ini
.venv-tum/bin/python integration/label_generator.py --race integration/parameters/pars_Spielberg_2019_fitted.ini --driver VER --compound A3 --deterministic
.venv-data/bin/python data_pipeline/compare_fitted.py
.venv-tum/bin/python integration/tune_traffic.py --loser 0.3 0.8 1.33
# all five 2019 races (see notes/fitted_parameters_study_circuits.md)
.venv-data/bin/python data_pipeline/check_lap_model.py
.venv-tum/bin/python integration/check_races.py
# held-out check (see notes/heldout_check.md)
.venv-data/bin/python data_pipeline/heldout_tyres.py
.venv-tum/bin/python integration/heldout_traffic.py

# needs real TensorFlow, so not on Apple Silicon: TUM's stock simulator and neural-network VSEs
.venv-tum/bin/python -X utf8 race-simulation/main_racesim.py
.venv-tum/bin/python integration/demo_vse.py

.venv-tum/bin/python -m unittest integration.test_policy integration.test_label_generator integration.test_write_fitted_pars integration.test_heldout_traffic -v
.venv-data/bin/python -m unittest data_pipeline.test_comparison data_pipeline.test_fit_parameters data_pipeline.test_fit_traffic data_pipeline.test_heldout_tyres data_pipeline.test_build_situations -v
```

Any change to TUM's files is logged in `notes/tum_changes.md`.

## Credits and license

The race simulation in `race-simulation/` is by the Institute of Automotive Technology,
Technical University of Munich, released under LGPL-3.0. F1 timing data is accessed through
[FastF1](https://github.com/theOehrly/Fast-F1) and is not redistributed in this repository.
