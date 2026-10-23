# TUM change log

Upstream: https://github.com/TUMFTM/race-simulation, commit `96ef2c2021982217be008fe458df47c1a72da071`.

**No upstream files changed.** The LGPL-3.0 license and original requirements remain intact. The source was downloaded as an upstream archive because the sandbox could not perform a network Git clone; it is a source snapshot, not a Git checkout. `tum_upstream_manifest.json` records every original file's SHA-256 plus the downloaded archive's hash.

| Location | Function | Change and reason |
|---|---|---|
| `integration/run_experiment.py` | `main` | `# CAPSTONE HOOK` assigns a one-driver policy through public `Race.vse`; upstream code stays unchanged. |
| `integration/optional_tensorflow.py` | `defer_tensorflow` | Defers the installed TensorFlow import until a TensorFlow API is actually requested; allows deterministic simulations on an Apple Silicon CPU without AVX. No TensorFlow operations are replaced. |

The first stock import/run was attempted before integration changes; it aborted in TensorFlow's AVX guard (`evidence/baseline_stock_import_failure.log`). The first successful baseline used the same untouched TUM source and original strategy tuples with the explicit import-deferral wrapper. It did not execute either neural-network VSE. This is a wrapper-driven baseline, not a claim that TUM's original `main_racesim.py` runs unassisted on this Mac.
