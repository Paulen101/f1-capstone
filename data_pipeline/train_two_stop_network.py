"""Train a pit-timing network that may stop twice, by policy iteration on exact replays (.venv-train).

    .venv-train/Scripts/python.exe data_pipeline/train_two_stop_network.py --track Austin

Input: integration/output/two_stop/<Track>_2019_fitted_*_mc<runs>/ (integration/two_stop_generator.py).
The start policy is pit_network_v1 for the first stop and no second stop. Each iteration replays the current
network on every stored run (data_pipeline/two_stop_labels.py), builds targets for both stints (pitting now
minus following the network from the next lap), trains a fresh network on the training runs, and scores it
exactly on the held-out runs (the last fifth). The best iteration is saved as integration/models/<name>.npz;
the scores, against the start policy, the best fixed one-stop lap and the best fixed one- or two-stop plan
(both chosen on the training runs) and the hindsight bound, go to notes/evidence/pit_network/<name>.json.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.train_pit_network import TIME_CLIP_S, add_tyre_features, save, train  # noqa: E402
from data_pipeline.train_stopping_network import decisions  # noqa: E402
from data_pipeline.two_stop_labels import G1, best_plan, compare, load_folders, outcomes, targets  # noqa: E402
from integration.pit_network import FEATURES, PitNetwork, feature_vector  # noqa: E402

TWO_STOP = ROOT / "integration/output/two_stop"


def features(table: pd.DataFrame) -> np.ndarray:
    x = np.zeros((len(table), len(FEATURES)))
    has = table.HasSituation.to_numpy()
    situated = add_tyre_features(table[has].reset_index())
    x[situated["index"].to_numpy()] = np.array([feature_vector(r) for r in situated.to_dict("records")])
    return x


def hindsight(t1: pd.DataFrame, t2: pd.DataFrame) -> pd.DataFrame:
    """Best plan of each run knowing its future (upper bound)."""
    plans = pd.concat([t1[G1 + ["PitPosition", "PitRaceTime_s", "PitFinished", "PitLaps"]],
                       t2[G1 + ["PitPosition", "PitRaceTime_s", "PitFinished", "PitLaps"]]])
    f = plans[plans.PitFinished.astype(bool)].sort_values(["PitPosition", "PitRaceTime_s"])
    best = f.groupby(G1).first().reset_index()
    out = t1.drop_duplicates(G1)[G1].merge(best, on=G1, how="left")
    out["PitFinished"] = out.PitFinished.eq(True)
    return out


def summary(name, outcome, references):
    entry = {"Policy": name, "MeanStopLap": float(outcome.StopLap.mean()),
             "TwoStopShare": float((outcome.SecondStop < 10 ** 4).mean())}
    for ref_name, ref in references.items():
        entry["Vs " + ref_name] = compare(outcome, ref)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", default="Austin")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--start", type=Path, default=ROOT / "integration/models/pit_network_v1.npz")
    parser.add_argument("--name", default="pit_network_v3_two_stop")
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
    folders = sorted(p for p in TWO_STOP.glob("%s_2019_fitted_*_mc%d" % (args.track, args.runs))
                     if (p / "manifest.json").is_file())
    t1, t2 = load_folders(folders)
    x1, x2 = features(t1), features(t2)
    has1, has2 = t1.HasSituation.to_numpy(), t2.HasSituation.to_numpy()
    first_val = int(args.runs * 0.8)
    val1, val2 = (t1.Run >= first_val).to_numpy(), (t2.Run >= first_val).to_numpy()
    print("%d driver-races; %d first-stint and %d second-stint rows (%.0f s)"
          % (len(folders), len(t1), len(t2), time.perf_counter() - started), flush=True)

    def held_out(out):
        return out[out.Run >= first_val].reset_index(drop=True)

    fixed = best_plan(t1, t2, lambda t: (t.Run < first_val).to_numpy())
    refs = {name: held_out(outcomes(t1, t2, *pits)) for name, pits in fixed.items()}
    network = PitNetwork(args.start)
    pit1, pit2 = decisions(network, x1, has1), np.zeros(len(t2), bool)
    refs["start (v1, one stop)"] = held_out(outcomes(t1, t2, pit1, pit2))
    upper = hindsight(t1, t2)
    refs["hindsight"] = upper[upper.Run >= first_val].reset_index(drop=True)
    report = {"Name": args.name, "Track": args.track, "DriverRaces": len(folders), "HeldOutFirstRun": first_val,
              "Settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "References": [summary(n, refs[n], {"fixed one-stop": refs["fixed one-stop"]})
                             for n in ("fixed best plan", "start (v1, one stop)")],
              "Iterations": []}
    for r in report["References"]:
        print("%-22s - fixed one-stop %+.3f places; two-stop share %.2f"
              % (r["Policy"], r["Vs fixed one-stop"]["PosDiff"], r["TwoStopShare"]))
    print("hindsight - fixed one-stop %+.3f places" % compare(refs["hindsight"], refs["fixed one-stop"])["PosDiff"])

    best_key, best_path = None, None
    for iteration in range(1, args.iterations + 1):
        y1, y2, use1, use2 = targets(t1, t2, pit1, pit2)
        x = np.vstack([x1[use1], x2[use2]])
        y = np.vstack([y1[use1].to_numpy(), y2[use2].to_numpy()])
        y[:, 1] = np.clip(y[:, 1], -TIME_CLIP_S, TIME_CLIP_S)
        val = np.concatenate([val1[use1], val2[use2]])
        keep = ~np.isnan(y).all(1)
        model, mean, std, history = train(x[keep & ~val], y[keep & ~val], x[keep & val], y[keep & val], args)
        path = ROOT / "integration/models" / ("%s_iter%d.npz" % (args.name, iteration))
        save(model, mean, std, path)
        network = PitNetwork(path)
        pit1, pit2 = decisions(network, x1, has1), decisions(network, x2, has2)
        outcome = held_out(outcomes(t1, t2, pit1, pit2))
        entry = summary("iteration %d" % iteration, outcome, refs)
        entry.update({"Iteration": iteration, "TrainRows": int((keep & ~val).sum()), "Epochs": len(history)})
        report["Iterations"].append(entry)
        print("iteration %d: - fixed one-stop %+.3f, - fixed best plan %+.3f, - start %+.3f places; two-stop %.2f"
              % (iteration, entry["Vs fixed one-stop"]["PosDiff"], entry["Vs fixed best plan"]["PosDiff"],
                 entry["Vs start (v1, one stop)"]["PosDiff"], entry["TwoStopShare"]), flush=True)
        key = (round(entry["Vs fixed one-stop"]["PosDiff"], 4), entry["Vs fixed one-stop"]["TimeDiff_s"])
        if best_key is None or key < best_key:
            best_key, best_path, report["Best"] = key, path, iteration
    final = ROOT / "integration/models" / (args.name + ".npz")
    final.write_bytes(best_path.read_bytes())
    for i in range(1, args.iterations + 1):
        (ROOT / "integration/models" / ("%s_iter%d.npz" % (args.name, i))).unlink()
    report["Seconds"] = round(time.perf_counter() - started)
    (ROOT / "notes/evidence/pit_network" / (args.name + ".json")).write_text(json.dumps(report, indent=1))
    print("best: iteration %d -> %s (%.0f s)" % (report["Best"], final.relative_to(ROOT), report["Seconds"]))


if __name__ == "__main__":
    main()
