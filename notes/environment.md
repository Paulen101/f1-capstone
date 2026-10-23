# Reproducing the week 2 work

Run commands from this project's root (`Main`). The two environments are deliberately separate:

| Environment | Interpreter | Purpose |
|---|---|---|
| `.venv-data` | Homebrew Python 3.10.18, arm64 | FastF1, Excel export and comparison |
| `.venv-tum` | Python 3.8.18, x86_64 under Rosetta | TUM and integration experiments |

Python 3.8 was downloaded from Astral's `python-build-standalone` release `20231002`, file `cpython-3.8.18+20231002-x86_64-apple-darwin-install_only.tar.gz`, unpacked under `.tools/python`, then used to create `.venv-tum`. Intel builds are needed for TUM's pinned 2020-era packages. Package wheels were downloaded with the permitted curl transport, then installed offline; direct pip/Git network access was unavailable in this execution environment.

TUM's original `requirements.txt` is unchanged and every pinned top-level version is installed, including NumPy 1.18.4 and TensorFlow 2.2.0. Full installed transitive versions are recorded in `requirements-tum.lock.txt` and `requirements-data.lock.txt`. The TensorFlow stack requires a compatible protobuf; this environment uses 3.20.3. Both environments pass `pip check`.

TensorFlow 2.2's Intel build requires AVX, which is unavailable under Rosetta on this machine. A normal import aborts the process. `integration/optional_tensorflow.py` delays this unused dependency for the baseline, `realstrategy`, and fixed-lap experiments. Any actual TensorFlow attribute request loads the real pinned package, which still cannot run here. Supervised/reinforcement inference is therefore unavailable on this Mac; neither is needed or used for the current work.

```sh
# All full races use the original Spielberg 2019 parameter file.
.venv-tum/bin/python integration/run_experiment.py baseline
.venv-tum/bin/python integration/run_experiment.py fixed --driver VER --pit-lap 25 --compound A3
.venv-tum/bin/python integration/run_experiment.py realstrategy

# Generated mapping and independent behavioral checks
.venv-tum/bin/python integration/build_parameter_mapping.py
.venv-tum/bin/python -m unittest integration.test_policy -v
.venv-data/bin/python -m unittest data_pipeline.test_comparison -v

# Results-only workaround; --offline works with the HTTP cache populated here.
.venv-data/bin/python data_pipeline/public_results.py --offline
.venv-data/bin/python data_pipeline/compare_2019.py \
  --workbook f1_data/F1_2019_results_only.xlsx \
  --result-time-format winner-and-gaps
```

The selected 2019 validation race has no downloaded lap/weather data. The original pipeline was attempted for rounds 8, 9 and 12; its generated `F1_2019_dry_races.xlsx` contains failed/excluded race rows, not a usable dry dataset. The separately generated `F1_2019_results_only.xlsx` contains 20 public result rows for round 9 and explicitly marks dry conditions unverified. It is not substituted into a dry-race fit.

Raw public results were obtained from `https://api.jolpi.ca/ergast/f1/2019/9/results.json` and loaded through FastF1 3.8.3's normal result parser. The cached response is retained under `fastf1_cache`; a source copy is in `notes/evidence/austria_2019_jolpica_results.json`. The observed winner-total/finisher-gap format is normalized explicitly, with both raw and normalized values preserved in the joined CSV.

F1's archive returned HTTP 401; FastF1's mirror returned 404 for the selected 2019 sessions. The maintainer confirms the authentication change and describes the new alpha release in [FastF1 issue 978](https://github.com/theOehrly/Fast-F1/issues/978). An F1TV Access subscription or higher and supported authentication, or a previously saved 2019 cache/workbook, is needed to complete lap/weather validation. No account login or subscription was attempted.

The user-provided `/Users/paul/Desktop/Seminar/fastf1_cache` was inspected read-only. Its parsed cache contains 2025 and 2026 directories; its HTTP cache contains 216 timing responses for 2025 and 144 for 2026, and no 2019 URLs. It was not modified or deleted. Those seasons remain separate from the 2019 validation work.
