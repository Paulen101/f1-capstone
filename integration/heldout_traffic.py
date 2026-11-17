"""Leave-one-race-out check of the traffic values (.venv-tum).

    heldout_traffic.py --tracks Spielberg Catalunya Monza Singapore Austin

Each race is simulated with its fitted file, but with traffic values taken from the other races only:
    t_gap_overtake    TUM's value for this track x the median ratio measured / TUM at the other tracks
                      (keeps TUM's ranking of tracks, uses our level)
    min_t_dist, t_duel, t_overtake_loser
                      median of the other tracks' measured values (t_duel at least 0)
Compared with the same file using this race's own measured values ("own") and TUM's file ("tum").
Deterministic, real strategies, as in check_races.py.

Output: notes/evidence/fitted_parameters/heldout_traffic.csv.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integration.tune_traffic import EVIDENCE, count_passes, real_results, score, setup, simulate  # noqa: E402
from integration.race_files import TUM_PARAMETERS  # noqa: E402

YEAR = 2019


def pooled(traffic: dict, track: str) -> dict:
    """Traffic values for `track` from the other tracks only."""
    others = [v for t, v in traffic.items() if t != track]
    ratio = statistics.median(o["t_gap_overtake"]["measured"] / o["t_gap_overtake"]["tum"] for o in others)
    values = {"t_gap_overtake": round(traffic[track]["t_gap_overtake"]["tum"] * ratio, 3)}
    for key in ("min_t_dist", "t_duel", "t_overtake_loser"):
        values[key] = round(max(statistics.median(o[key]["measured"] for o in others), 0.0), 3)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", nargs="+", default=["Spielberg", "Catalunya", "Monza", "Singapore", "Austin"])
    parser.add_argument("--out", type=Path, default=EVIDENCE / "heldout_traffic.csv")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 8):
        parser.error("Run this script with .venv-tum (Python 3.8)")
    setup()
    import pandas as pd
    traffic = {t: json.loads((EVIDENCE / ("%s_%d_traffic.json" % (t, YEAR))).read_text(encoding="utf-8"))
               for t in args.tracks}
    rows = []
    for track in args.tracks:
        name = "%s_%d" % (track, YEAR)
        real = real_results(name)
        fitted = ROOT / "integration/parameters" / ("pars_%s_fitted.ini" % name)
        runs = {"tum": (TUM_PARAMETERS / ("pars_%s.ini" % name), {}),
                "own": (fitted, {}),
                "other races": (fitted, pooled({t: v["values"] for t, v in traffic.items()}, track))}
        for label, (path, values) in runs.items():
            race = simulate(path, values)
            row = {"Track": track, "Traffic": label, "RealPasses": traffic[track]["counts"]["passes"],
                   "t_gap_overtake": race.track.t_gap_overtake, "min_t_dist": race.race_pars["min_t_dist"],
                   "t_duel": race.race_pars["t_duel"], "t_overtake_loser": race.race_pars["t_overtake_loser"],
                   "status": int(race.result_status)}
            if race.result_status == 0:
                row.update(Passes=count_passes(race), **score(race, real))
            rows.append(row)
            print({k: v for k, v in row.items() if k != "Positions"}, flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Wrote %s" % args.out)


if __name__ == "__main__":
    main()
