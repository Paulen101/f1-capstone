"""Simulate each race with TUM's file and with our fitted files, and score them against the real race (.venv-tum).

    check_races.py --tracks Spielberg Catalunya Monza Singapore Austin

For every track (2019) it runs, deterministic and with realstrategy for every driver:
    tum        TUM's race file
    notraffic  our fit without the traffic group, if integration/output/parameters/pars_<Track>_2019_notraffic.ini
               exists (write_fitted_pars.py --groups pit tyres pace --out ...)
    fitted     integration/parameters/pars_<Track>_2019_fitted.ini
and compares passes, finishing order and race times with the real race. Needs <Track>_2019_results.csv and
<Track>_2019_traffic.json from the data pipeline.

Output: notes/evidence/fitted_parameters/race_checks.csv.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integration.tune_traffic import EVIDENCE, count_passes, real_results, score, setup, simulate  # noqa: E402
from integration.race_files import TUM_PARAMETERS  # noqa: E402

YEAR = 2019


def files(track: str) -> dict:
    candidates = {"tum": TUM_PARAMETERS / ("pars_%s_%d.ini" % (track, YEAR)),
                  "notraffic": ROOT / "integration/output/parameters" / ("pars_%s_%d_notraffic.ini" % (track, YEAR)),
                  "fitted": ROOT / "integration/parameters" / ("pars_%s_%d_fitted.ini" % (track, YEAR))}
    return {name: path for name, path in candidates.items() if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", nargs="+", default=["Spielberg", "Catalunya", "Monza", "Singapore", "Austin"])
    parser.add_argument("--out", type=Path, default=EVIDENCE / "race_checks.csv")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 8):
        parser.error("Run this script with .venv-tum (Python 3.8)")
    setup()
    import pandas as pd
    rows = []
    for track in args.tracks:
        name = "%s_%d" % (track, YEAR)
        real = real_results(name)
        real_passes = json.loads((EVIDENCE / ("%s_traffic.json" % name)).read_text(encoding="utf-8"))["counts"]["passes"]
        for label, path in files(track).items():
            row = {"Track": track, "Parameters": label,
                   "RealPasses": real_passes, "RealTop4": "-".join(real.sort_values("Position").Abbreviation[:4])}
            try:
                race = simulate(path, {})
            except RuntimeError as error:
                # TUM's Melbourne and Sakhir 2019 files burn slightly more than the 110 kg tank holds
                # (write_fitted_pars.py lowers the burn in our files); a car then runs dry on the last lap
                if "Unreasonable value" not in str(error):
                    raise
                row["status"] = "ran out of fuel"
                rows.append(row)
                print(row, flush=True)
                continue
            row["status"] = int(race.result_status)
            if race.result_status == 0:
                row.update(Passes=count_passes(race), **score(race, real))
            rows.append(row)
            print({k: v for k, v in row.items() if k != "Positions"}, flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Wrote %s" % args.out)


if __name__ == "__main__":
    main()
