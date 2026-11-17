"""Check TUM's and our race files as lap-time models on the real laps of each race (.venv-data).

    check_lap_model.py --rounds 9 5 14 15 19

For every clean lap: the lap time the file's formula gives for a car alone on track (pace, tyre and
fuel terms, no traffic) against the measured lap time. Free-air laps are the ones the fit used;
all clean laps include traffic, which the simulator adds itself.

Output: notes/evidence/fitted_parameters/lap_model_checks.csv.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.compare_fitted import predict  # noqa: E402
from data_pipeline.fit_parameters import add_gaps, find_workbook, free_air  # noqa: E402
from integration.race_files import tum_track  # noqa: E402
from integration.write_fitted_pars import TUM_PARAMETERS, compound_codes, parse_pars  # noqa: E402

EVIDENCE = ROOT / "notes/evidence/fitted_parameters"


def check(laps: pd.DataFrame, track: str, year: int) -> list:
    fit = json.loads((EVIDENCE / ("%s_%d_fit.json" % (track, year))).read_text(encoding="utf-8"))
    tum = parse_pars((TUM_PARAMETERS / fit["tum_file"]).read_bytes().decode("utf-8"))
    codes = compound_codes(fit, tum)
    files = {"tum": TUM_PARAMETERS / fit["tum_file"],
             "fitted": ROOT / "integration/parameters" / fit["tum_file"].replace(".ini", "_fitted.ini")}
    laps = add_gaps(laps)
    laps = laps[laps.Compound.isin(codes)]
    rows = []
    for name, path in files.items():
        pars = parse_pars(path.read_bytes().decode("utf-8"))
        for subset, data in (("free-air laps", free_air(laps)), ("all clean laps", laps[laps.CleanLap])):
            error = predict(pars, data, codes) - data.LapTime_s.values
            rows.append({"Track": track, "Parameters": name, "Laps": subset, "Count": len(data),
                         "RMSE_s": round(float(np.sqrt(np.mean(error ** 2))), 3),
                         "MeanError_s": round(float(error.mean()), 3)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--rounds", type=int, nargs="+", default=[9, 5, 14, 15, 19])
    args = parser.parse_args()
    workbook = find_workbook(args.year)
    races = pd.read_excel(workbook, sheet_name="Races")
    all_laps = pd.read_excel(workbook, sheet_name="Laps")
    rows = []
    for rnd in args.rounds:
        location = races[(races.Year == args.year) & (races.Round == rnd)].Location.iloc[0]
        laps = all_laps[(all_laps.Year == args.year) & (all_laps.Round == rnd)]
        rows += check(laps, tum_track(location), args.year)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(EVIDENCE / "lap_model_checks.csv", index=False)


if __name__ == "__main__":
    main()
