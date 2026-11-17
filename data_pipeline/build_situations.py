"""Turn Monte Carlo label runs into per-situation training rows (.venv-data). Methodology step 4.

    build_situations.py integration/output/labels/Spielberg_2019_fitted_VER_mc200 [more folders]

Each folder comes from label_generator.py --situations. A situation is (run, decision lap L): the race
state at L from situations.csv. In that run every candidate pit lap L' >= L shares the same history up
to L, so their outcomes (runs.csv) are paired continuations of the situation:

    pit now      outcome of candidate L in this run
    pit later    outcome of candidate L* in this run, where L* > L is the later lap that is best on
                 average over all runs (summary.csv: mean position, then mean race time). L* is chosen
                 from the average, never from this run's own future.
    advantage    pit now - pit later, in finishing position and in race time (negative: pitting now
                 is better)

Rows where the driver did not finish both continuations have no advantage. Race time is compared only
when both ran the full distance. Output: notes/evidence/situations/situations.csv, or --out.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def best_later_laps(summary: pd.DataFrame) -> dict:
    """For each candidate lap, the later candidate with the best mean position, then mean race time."""
    ranked = summary[summary.FinishedRuns > 0].sort_values("PitLap")
    laps = ranked.PitLap.tolist()
    later = {}
    for i, lap in enumerate(laps):
        rest = ranked.iloc[i + 1:]
        if len(rest):
            best = rest.sort_values(["MeanPosition", "MeanRaceTime_s", "PitLap"]).iloc[0]
            later[lap] = int(best.PitLap)
    return later


def build(folder: Path) -> pd.DataFrame:
    situations = pd.read_csv(folder / "situations.csv")
    runs = pd.read_csv(folder / "runs.csv")
    later = best_later_laps(pd.read_csv(folder / "summary.csv"))
    tot = int(runs.SimulatedLaps.max())
    outcome = runs.set_index(["PitLap", "Run"])[["Position", "RaceTime_s", "Finished", "SimulatedLaps"]]
    rows = situations[situations.Lap.isin(later)].copy()
    rows["LaterLap"] = rows.Lap.map(later)
    now = outcome.reindex(list(zip(rows.Lap, rows.Run))).reset_index(drop=True)
    then = outcome.reindex(list(zip(rows.LaterLap, rows.Run))).reset_index(drop=True)
    rows = rows.reset_index(drop=True)
    for name, part in (("PitNow", now), ("PitLater", then)):
        rows[name + "Position"] = part.Position
        rows[name + "RaceTime_s"] = part.RaceTime_s
        rows[name + "Finished"] = part.Finished.astype(bool)
    both = rows.PitNowFinished & rows.PitLaterFinished
    full = both & (now.SimulatedLaps >= tot) & (then.SimulatedLaps >= tot)
    rows["AdvantagePosition"] = np.where(both, rows.PitNowPosition - rows.PitLaterPosition, np.nan)
    rows["AdvantageTime_s"] = np.where(full, rows.PitNowRaceTime_s - rows.PitLaterRaceTime_s, np.nan).round(3)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=ROOT / "notes/evidence/situations/situations.csv")
    args = parser.parse_args()
    table = pd.concat([build(folder) for folder in args.folders], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print("%d situations from %d driver-races, %d with a position advantage, %d with a time advantage. Wrote %s"
          % (len(table), len(args.folders), table.AdvantagePosition.notna().sum(),
             table.AdvantageTime_s.notna().sum(), args.out))


if __name__ == "__main__":
    main()
