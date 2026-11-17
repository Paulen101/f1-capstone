"""Collect Monte Carlo pit-lap labels into per-lap scores and compare them with the real stops (.venv-data).

    summarise_labels.py --runs 500

Reads integration/output/labels/<Track>_2019_fitted_<driver>_mc<runs>/ (label_generator.py) and, where it
exists, the deterministic run of the same driver. Writes to notes/evidence/labels/:

    pit_lap_scores.csv   the training label: one row per driver and candidate pit lap, with the mean
                         position and race-time difference to the best lap (paired over runs with the same
                         seed) and their standard errors
    label_summary.csv    one row per driver: real stop, Monte Carlo label, laps within two standard errors,
                         deterministic label
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integration.write_fitted_pars import TUM_PARAMETERS, parse_pars  # noqa: E402

LABELS = ROOT / "integration/output/labels"
OUT = ROOT / "notes/evidence/labels"
YEAR = 2019


def real_stop(track: str, driver: str) -> int:
    pars = parse_pars((TUM_PARAMETERS / ("pars_%s_%d.ini" % (track, YEAR))).read_bytes().decode("utf-8"))
    return int(pars["driver_pars"][driver]["strategy_info"][1][0])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=500)
    args = parser.parse_args()
    scores, rows = [], []
    for folder in sorted(LABELS.glob("*_%d_fitted_*_mc%d" % (YEAR, args.runs))):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        track, driver = folder.name.split("_%d_fitted_" % YEAR)[0], manifest["driver"]
        summary = pd.read_csv(folder / "summary.csv")
        real = real_stop(track, driver)
        close = manifest["laps_within_2se_of_label"]
        scores.append(summary.assign(Track=track, Driver=driver, Compound=manifest["compound"], RealPitLap=real)[
            ["Track", "Driver", "Compound", "PitLap", "RealPitLap", "FinishedRuns", "MeanPosition",
             "PosDiffVsLabel", "PosDiffSE", "TimeDiffVsLabel_s", "TimeDiffSE_s"]])
        deterministic = folder.with_name(folder.name.replace("_mc%d" % args.runs, "_deterministic"))
        det = (json.loads((deterministic / "manifest.json").read_text(encoding="utf-8"))["label_pit_lap"]
               if (deterministic / "manifest.json").is_file() else None)
        rows.append({"Track": track, "Driver": driver, "RealPitLap": real,
                     "MonteCarloLabel": manifest["label_pit_lap"], "FastestPitLap": manifest["fastest_pit_lap"],
                     "Within2SE_First": min(close), "Within2SE_Last": max(close), "Within2SE_Laps": len(close),
                     "RealWithin2SE": real in close, "DeterministicLabel": det,
                     "FinishedRunsShare": round(summary.FinishedRuns.min() / manifest["runs_per_lap"], 3),
                     "Runs": manifest["runs_per_lap"]})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(scores).to_csv(OUT / "pit_lap_scores.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "label_summary.csv", index=False)
    print(table.to_string(index=False))
    mc = (table.MonteCarloLabel - table.RealPitLap).abs()
    det = (table.DeterministicLabel - table.RealPitLap).abs()
    print("Mean distance to the real stop: Monte Carlo %.1f laps, deterministic %.1f laps; real stop within "
          "2 SE of the label for %d of %d" % (mc.mean(), det.mean(), table.RealWithin2SE.sum(), len(table)))


if __name__ == "__main__":
    main()
