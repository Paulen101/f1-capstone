"""Train the pit-timing network on labels that value waiting, by policy iteration (.venv-train). Step 6, round 2.

    .venv-train/Scripts/python.exe data_pipeline/train_stopping_network.py
    .venv-train/Scripts/python.exe data_pipeline/train_stopping_network.py --iterations 4 --name pit_network_v2

Start from a network (default pit_network_v1, trained on "pit now vs the best fixed later lap"). Each iteration:

  1. replay the current network on every stored Monte Carlo run (data_pipeline/stopping_labels.py): where would
     it stop if it decided from each lap on?
  2. new targets: pitting now minus following the current network from the next lap, in the same run
  3. train a fresh network on them (runs 0-159; same architecture and settings as train_pit_network.py)
  4. score the new network exactly on the held-out runs 160-199: replay it from the first lap

The network with the best held-out score (mean position, then race time) is saved as integration/models/
<name>.npz; every iteration's score, against the start network and the best fixed lap (chosen on the
training runs), goes to notes/evidence/pit_network/<name>.json.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.stopping_labels import compare, fixed_lap_pit, load_folders, policy_outcomes, targets  # noqa: E402
from data_pipeline.train_pit_network import (TIME_CLIP_S, VAL_FIRST_RUN, add_tyre_features, save,  # noqa: E402
                                             train)
from integration.pit_network import FEATURES, POSITION_TIE, PitNetwork, feature_vector  # noqa: E402

SITUATIONS = ROOT / "integration/output/situations"


def decisions(network: PitNetwork, x: np.ndarray, has: np.ndarray) -> np.ndarray:
    """pit_now() of integration/pit_network.py on every row, vectorised; rows without a situation stay out."""
    pred = np.zeros((len(x), 2))
    pred[has] = network.predict(x[has])
    pos, t = pred[:, 0], pred[:, 1]
    return has & ((pos < -POSITION_TIE) | ((np.abs(pos) <= POSITION_TIE) & (t <= 0.0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=Path, default=ROOT / "integration/models/pit_network_v1.npz")
    parser.add_argument("--name", default="pit_network_v2")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    started = time.perf_counter()
    folders = sorted(p for p in SITUATIONS.glob("*_2019_fitted_*_mc200") if (p / "situations.csv").is_file())
    rows = load_folders(folders)
    has = rows.HasSituation.to_numpy()
    situated = add_tyre_features(rows[has].reset_index())
    x = np.zeros((len(rows), len(FEATURES)))
    x[situated["index"].to_numpy()] = np.array([feature_vector(r) for r in situated.to_dict("records")])
    in_val = (rows.Run >= VAL_FIRST_RUN).to_numpy()
    print("%d driver-races, %d candidate rows, %d with a situation (%.0f s)"
          % (len(folders), len(rows), has.sum(), time.perf_counter() - started), flush=True)

    val_rows = rows[in_val].reset_index(drop=True)
    fixed = policy_outcomes(val_rows, fixed_lap_pit(rows, ~in_val)[in_val])
    network = PitNetwork(args.start)
    start_outcome = policy_outcomes(val_rows, decisions(network, x, has)[in_val])
    report = {"Name": args.name, "Start": str(args.start.relative_to(ROOT)), "DriverRaces": len(folders),
              "Settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "StartVsFixed": compare(start_outcome, fixed), "Iterations": []}
    print("start network - fixed lap: %+.3f places, %+.2f s" % (report["StartVsFixed"]["PosDiff"],
                                                                 report["StartVsFixed"]["TimeDiff_s"]))
    best_key, best_path = None, None
    for iteration in range(1, args.iterations + 1):
        pit = decisions(network, x, has)
        y_frame = targets(rows, pit)
        y = y_frame[["AdvantagePosition", "AdvantageTime_s"]].to_numpy()
        y[:, 1] = np.clip(y[:, 1], -TIME_CLIP_S, TIME_CLIP_S)
        usable = has & (rows.Lap < rows.LastLap).to_numpy() & ~np.isnan(y).all(1)
        tr, va = usable & ~in_val, usable & in_val
        model, mean, std, history = train(x[tr], y[tr], x[va], y[va], args)
        path = ROOT / "integration/models" / ("%s_iter%d.npz" % (args.name, iteration))
        save(model, mean, std, path)
        network = PitNetwork(path)
        outcome = policy_outcomes(val_rows, decisions(network, x, has)[in_val])
        entry = {"Iteration": iteration, "TrainRows": int(tr.sum()), "Epochs": len(history),
                 "MeanLaterStopLap": float(np.nanmean(y_frame.LaterStopLap[usable])),
                 "MeanStopLap": float(outcome.StopLap.mean()),
                 "VsFixed": compare(outcome, fixed), "VsStart": compare(outcome, start_outcome)}
        report["Iterations"].append(entry)
        print("iteration %d: - fixed lap %+.3f places %+.2f s;  - start network %+.3f places %+.2f s; mean stop lap %.1f"
              % (iteration, entry["VsFixed"]["PosDiff"], entry["VsFixed"]["TimeDiff_s"],
                 entry["VsStart"]["PosDiff"], entry["VsStart"]["TimeDiff_s"], entry["MeanStopLap"]), flush=True)
        key = (round(entry["VsFixed"]["PosDiff"], 4), entry["VsFixed"]["TimeDiff_s"])
        if best_key is None or key < best_key:
            best_key, best_path, report["Best"] = key, path, iteration
    final = ROOT / "integration/models" / (args.name + ".npz")
    final.write_bytes(best_path.read_bytes())
    for i in range(1, args.iterations + 1):
        (ROOT / "integration/models" / ("%s_iter%d.npz" % (args.name, i))).unlink()
    report["Seconds"] = round(time.perf_counter() - started)
    out = ROOT / "notes/evidence/pit_network" / (args.name + ".json")
    out.write_text(json.dumps(report, indent=1))
    print("best: iteration %d -> %s (%.0f s)" % (report["Best"], final.relative_to(ROOT), report["Seconds"]))


if __name__ == "__main__":
    main()
