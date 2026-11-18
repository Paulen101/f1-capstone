"""Train a compound head on top of the two-stop network v4: v4 times the stops, the head picks the tyre (.venv-train).

    .venv-train/Scripts/python.exe data_pipeline/train_compound_head.py

Round 5 (network v5) learned lap and compound together and lost at Austin and Singapore what it gained at
Spielberg and Monza. Here v4's decisions stay as they raced: it decides on the stored compound when to stop and
whether to stop again. Only when it stops does the head choose between the stored compound and the other legal
one (integration/compound_runs.py).

Label, exact on the stored runs (data_pipeline/two_stop_labels.py): for every first-stint lap with a situation,
the outcome of a stop onto the other compound minus the outcome of a stop onto the stored compound followed by
v4's second-stint decisions, in the same run (position: both finished; time: also the same laps). The head sees
the situation with the other compound's tyre features and predicts both differences; it switches when
pit_now() says the other compound is better.

Training runs 0-39, held out 40-49 as in train_two_stop_network.py. On the held-out runs the head is compared
with v4, v5, a per-driver rule (always or never switch, chosen on the training runs; saved in the report for
integration/evaluate_pit_network.py) and the hindsight bound
(the better compound on v4's lap in every run). Output: integration/models/<name>.npz and
notes/evidence/pit_network/<name>.json.
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
from data_pipeline.train_pit_network import TIME_CLIP_S, save, train  # noqa: E402
from data_pipeline.train_stopping_network import decisions  # noqa: E402
from data_pipeline.train_two_stop_network import (COMPOUND_RUNS, TWO_STOP, features,  # noqa: E402
                                                  first_stint_decisions)
from data_pipeline.two_stop_labels import G1, advantage, compare, load_folders, outcomes, stop_values  # noqa: E402
from integration.pit_network import PitNetwork  # noqa: E402


def pairs(t1: pd.DataFrame) -> tuple:
    """(stored row, other row) indices of every (race, driver, run, lap) with exactly one other compound."""
    key = t1[G1 + ["Lap"]].astype(str).agg("|".join, axis=1)
    stored = pd.Series(np.flatnonzero(t1.Default.to_numpy(bool)), index=key[t1.Default.to_numpy(bool)].to_numpy())
    other_mask = ~t1.Default.to_numpy(bool)
    other = pd.Series(np.flatnonzero(other_mask), index=key[other_mask].to_numpy())
    if other.index.duplicated().any():
        raise ValueError("More than one other compound on a lap: the head chooses between two")
    return stored.loc[other.index].to_numpy(), other.to_numpy()


def switch(pit1: np.ndarray, stored: np.ndarray, other: np.ndarray, use_other: np.ndarray) -> np.ndarray:
    """Move v4's stop from the stored-compound row to the other-compound row where `use_other` (per pair)."""
    out = pit1.copy()
    move = pit1[stored] & use_other
    out[stored[move]] = False
    out[other[move]] = True
    return out


def pit_now_vec(pred: np.ndarray) -> np.ndarray:
    from integration.pit_network import POSITION_TIE
    pos, t = pred[:, 0], pred[:, 1]
    return (pos < -POSITION_TIE) | ((np.abs(pos) <= POSITION_TIE) & (t <= 0.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", nargs="+", default=["Austin", "Monza", "Singapore", "Spielberg"])
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--timing", type=Path, default=ROOT / "integration/models/pit_network_v4_two_stop.npz")
    parser.add_argument("--v5", type=Path, default=ROOT / "integration/models/pit_network_v5_compound.npz")
    parser.add_argument("--name", default="pit_compound_head_v1")
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
    folders = sorted(p for track in args.tracks for p in TWO_STOP.glob("%s_2019_fitted_*_mc%d" % (track, args.runs))
                     if (p / "manifest.json").is_file())
    t1, t2 = load_folders(folders, COMPOUND_RUNS)
    x1, x2 = features(t1), features(t2)
    has2 = t2.HasSituation.to_numpy()
    first_val = int(args.runs * 0.8)
    stored, other = pairs(t1)
    print("%d driver-races, %d laps with a compound choice (%.0f s)" % (len(folders), len(stored),
                                                                          time.perf_counter() - started), flush=True)

    timing = PitNetwork(args.timing)
    pit2 = decisions(timing, x2, has2)
    pit1 = first_stint_decisions(timing, x1, t1, stored_only=True)
    values = stop_values(t1, t2, pit2)
    y = advantage(values.iloc[other].reset_index(drop=True), values.iloc[stored].reset_index(drop=True)).to_numpy()
    y[:, 1] = np.clip(y[:, 1], -TIME_CLIP_S, TIME_CLIP_S)
    run = t1.Run.to_numpy()[other]
    usable = (t1.HasSituation.to_numpy() & (t1.Lap < t1.LastLap).to_numpy())[other] & ~np.isnan(y).all(1)
    tr, va = usable & (run < first_val), usable & (run >= first_val)
    model, mean, std, history = train(x1[other][tr], y[tr], x1[other][va], y[va], args)
    path = ROOT / "integration/models" / (args.name + ".npz")
    save(model, mean, std, path)
    head = PitNetwork(path)
    use_head = pit_now_vec(head.predict(x1[other]))

    # references on the held-out runs
    stored_compound = t1[t1.Default].groupby(["Race", "Driver"]).StopCompound.first().rename("StoredCompound")

    def held_out(p1, p2=pit2):
        out = outcomes(t1, t2, p1, p2).merge(stored_compound.reset_index(), on=["Race", "Driver"])
        out["Track"] = out.Race.str.extract(r"pars_(\w+?)_\d{4}")[0]
        return out[out.Run >= first_val].reset_index(drop=True)

    train_pairs = run < first_val
    gain = pd.DataFrame({"Race": t1.Race.to_numpy()[other], "Driver": t1.Driver.to_numpy()[other],
                         "Pos": y[:, 0], "T": y[:, 1], "AtStop": pit1[stored]})
    by_driver = gain[train_pairs & gain.AtStop].groupby(["Race", "Driver"])[["Pos", "T"]].mean()
    always = pit_now_vec(by_driver.to_numpy())
    rule = pd.Series(always, index=by_driver.index)
    rule_use = rule.reindex(pd.MultiIndex.from_frame(gain[["Race", "Driver"]])).fillna(False).to_numpy(bool)
    hindsight_use = np.nan_to_num(y[:, 0], nan=1.0) < 0
    hindsight_use |= (np.nan_to_num(y[:, 0], nan=1.0) == 0) & (np.nan_to_num(y[:, 1], nan=1.0) < 0)
    v5 = PitNetwork(args.v5)
    policies = {
        "v4 (stored compound)": (pit1, pit2),
        "v4 + per-driver rule": (switch(pit1, stored, other, rule_use), pit2),
        "v4 + head": (switch(pit1, stored, other, use_head), pit2),
        "v4 + hindsight compound": (switch(pit1, stored, other, hindsight_use), pit2),
        "v5": (first_stint_decisions(v5, x1, t1), decisions(v5, x2, has2)),
    }
    results = {name: held_out(*p) for name, p in policies.items()}
    base = results["v4 (stored compound)"]
    report = {"Name": args.name, "Timing": str(args.timing), "Rows": {"Train": int(tr.sum()), "Val": int(va.sum())},
              "Epochs": len(history), "Policies": []}
    for name, out in results.items():
        entry = {"Policy": name, "OtherCompoundShare": float((out.StopCompound != out.StoredCompound).mean()),
                 "TwoStopShare": float((out.SecondStop < 10 ** 4).mean()), "Vs v4": {}}
        for t in ["all"] + sorted(out.Track.unique()):
            a = out if t == "all" else out[out.Track == t]
            b = base if t == "all" else base[base.Track == t]
            entry["Vs v4"][t] = compare(a.reset_index(drop=True), b.reset_index(drop=True))
            entry.setdefault("OtherByTrack", {})[t] = float((a.StopCompound != a.StoredCompound).mean())
        report["Policies"].append(entry)
        print("%-26s other %.2f two-stop %.2f | - v4: " % (name, entry["OtherCompoundShare"], entry["TwoStopShare"])
              + "  ".join("%s %+.3f±%.3f" % (t, v["PosDiff"], v["PosDiff_SE"]) for t, v in entry["Vs v4"].items()),
              flush=True)
    report["PerDriverRule"] = {"%s %s" % (race.split("_")[1], driver): bool(v) for (race, driver), v in rule.items()}
    report["Seconds"] = round(time.perf_counter() - started)
    (ROOT / "notes/evidence/pit_network" / (args.name + ".json")).write_text(json.dumps(report, indent=1))
    print("saved %s (%.0f s)" % (path.relative_to(ROOT), report["Seconds"]))


if __name__ == "__main__":
    main()
