"""Compare the policies of integration/evaluate_pit_network.py (.venv-data or .venv-train).

    summarise_evaluation.py integration/output/evaluation/pit_network_v1_mc100

Every policy raced the same seeds, so each comparison is paired by (track, driver, run):

    position   finishing position difference, over runs where the driver finished under both policies
    race time  race-time difference, over runs where both completed the same number of laps (a lapped
               driver's time is that of their last completed lap, as in the training rows)

Negative means the first policy (the network, unless --base says otherwise) is better. Output next to
the input and in notes/evidence/pit_network/: <name>_summary.csv (all tracks and per track) and
<name>_by_driver.csv (one row per driver-race and comparison).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def paired(runs: pd.DataFrame, base: str, other: str) -> pd.DataFrame:
    keys = ["Track", "Driver", "Run"]
    a = runs[runs.Policy == base].set_index(keys)
    b = runs[runs.Policy == other].set_index(keys)
    both = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    finished = both.Finished_a & both.Finished_b
    same_laps = finished & (both.SimulatedLaps_a == both.SimulatedLaps_b)
    return pd.DataFrame({
        "PosDiff": np.where(finished, both.Position_a - both.Position_b, np.nan),
        "TimeDiff_s": np.where(same_laps, both.RaceTime_s_a - both.RaceTime_s_b, np.nan),
        "StopDiff": both.FirstStopLap_a - both.FirstStopLap_b,
    }, index=both.index).reset_index()


def describe(diffs: pd.DataFrame) -> dict:
    out = {}
    for column in ("PosDiff", "TimeDiff_s"):
        values = diffs[column].dropna()
        out[column] = values.mean()
        out[column + "_SE"] = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
        out[column + "_N"] = len(values)
    out["Better"] = (diffs.PosDiff < 0).mean()
    out["Worse"] = (diffs.PosDiff > 0).mean()
    out["MeanStopDiff_laps"] = diffs.StopDiff.mean()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--base", default="network")
    args = parser.parse_args()
    runs = pd.read_csv(args.folder / "runs.csv")
    runs["FirstStopLap"] = pd.to_numeric(runs.FirstStopLap, errors="coerce")
    name = args.folder.name
    others = [p for p in ["real", "tum", "fixed", "network"] if p != args.base and p in set(runs.Policy)]

    policies = runs.groupby("Policy").agg(
        MeanPosition=("Position", lambda s: s[runs.loc[s.index, "Finished"]].mean()),
        Finished=("Finished", "mean"), MeanFirstStop=("FirstStopLap", "mean"), MeanStops=("Stops", "mean"))
    print(policies.round(3).to_string(), "\n")

    summary, by_driver = [], []
    for other in others:
        diffs = paired(runs, args.base, other)
        for track, part in [("all", diffs)] + list(diffs.groupby("Track")):
            summary.append({"Comparison": "%s - %s" % (args.base, other), "Track": track, **describe(part)})
        for (track, driver), part in diffs.groupby(["Track", "Driver"]):
            by_driver.append({"Comparison": "%s - %s" % (args.base, other), "Track": track, "Driver": driver,
                              **describe(part)})
    summary, by_driver = pd.DataFrame(summary), pd.DataFrame(by_driver)
    significant = by_driver.assign(
        SigBetter=by_driver.PosDiff < -2 * by_driver.PosDiff_SE, SigWorse=by_driver.PosDiff > 2 * by_driver.PosDiff_SE,
        SigFaster=by_driver.TimeDiff_s < -2 * by_driver.TimeDiff_s_SE,
        SigSlower=by_driver.TimeDiff_s > 2 * by_driver.TimeDiff_s_SE)
    counts = significant.groupby("Comparison")[["SigBetter", "SigWorse", "SigFaster", "SigSlower"]].sum()
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False), "\n")
    print("driver-races clearly better / worse (2 SE):\n" + counts.to_string())

    evidence = ROOT / "notes/evidence/pit_network"
    evidence.mkdir(parents=True, exist_ok=True)
    for folder in (args.folder, evidence):
        summary.round(4).to_csv(folder / (name + "_summary.csv"), index=False)
        by_driver.round(4).to_csv(folder / (name + "_by_driver.csv"), index=False)
        policies.round(4).to_csv(folder / (name + "_policies.csv"))


if __name__ == "__main__":
    main()
