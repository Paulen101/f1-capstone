"""Held-out check of the lap-time fit at tyre ages it has not seen (.venv-data).

    heldout_tyres.py --rounds 9 5 14 15 19

A pit-lap label asks what happens if a driver stays out longer or shorter than in the real race, so
the tyre model has to hold beyond the tyre ages it was fitted on. For every stint, the free-air laps
in the last third of the stint's tyre-age range are held out; the model is fitted on the rest (same
method as fit_parameters.py) and predicts the held-out laps. Compared with:

    tum        TUM's race file as a lap-time formula (its values are fixed, not refitted)
    in-sample  our fit on all free-air laps, the held-out ones included (a lower bound)

Output: notes/evidence/fitted_parameters/heldout_tyres.csv (one row per race and model) and
heldout_tyres_by_extra_age.csv (held-out error by how far the tyre age lies beyond the training laps).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.compare_fitted import predict as tum_predict  # noqa: E402
from data_pipeline.fit_parameters import add_gaps, find_workbook, free_air, solve  # noqa: E402
from integration.race_files import tum_track  # noqa: E402
from integration.write_fitted_pars import TUM_PARAMETERS, compound_codes, parse_pars  # noqa: E402

EVIDENCE = ROOT / "notes/evidence/fitted_parameters"
HELD_OUT_SHARE = 1 / 3


def split(free: pd.DataFrame) -> pd.Series:
    """True for free-air laps in the last third of their stint's tyre-age range."""
    ages = free.groupby(["Driver", "Stint"]).TyreLife
    low, high = ages.transform("min"), ages.transform("max")
    return free.TyreLife > high - (high - low) * HELD_OUT_SHARE


def our_predict(fit: dict, laps: pd.DataFrame) -> np.ndarray:
    levels = laps.Driver.map(fit["levels"])
    k0 = laps.Compound.map({c: t["k0"] for c, t in fit["tyres"].items()})
    k1 = laps.Compound.map({c: t["k1"] for c, t in fit["tyres"].items()})
    return (levels - fit["trend"] * (laps.LapNumber - 1) + k0 + k1 * (laps.TyreLife - 1)).values


def metrics(error) -> dict:
    error = np.asarray(error)
    return {"RMSE_s": round(float(np.sqrt(np.mean(error ** 2))), 3), "MeanError_s": round(float(error.mean()), 3)}


def check(laps: pd.DataFrame, track: str, year: int):
    fit_file = json.loads((EVIDENCE / ("%s_%d_fit.json" % (track, year))).read_text(encoding="utf-8"))
    tum = parse_pars((TUM_PARAMETERS / fit_file["tum_file"]).read_bytes().decode("utf-8"))
    codes = compound_codes(fit_file, tum)
    free = free_air(add_gaps(laps))
    free = free[free.Compound.isin(codes)]
    held = split(free)
    train, test = free[~held], free[held]
    heldout_fit = solve(train)
    test = test[test.Driver.isin(heldout_fit["levels"]) & test.Compound.isin(heldout_fit["tyres"])]
    full_fit = solve(free)
    predictions = {"held-out fit": our_predict(heldout_fit, test), "in-sample fit": our_predict(full_fit, test),
                   "tum": tum_predict(tum, test, codes)}
    trained = {"held-out fit": len(train), "in-sample fit": len(free), "tum": 0}
    rows = [{"Track": track, "Model": name, "TrainLaps": trained[name], "TestLaps": len(test),
             **metrics(pred - test.LapTime_s.values)} for name, pred in predictions.items()]
    # how far beyond the oldest training lap of the same stint each test lap is
    oldest = train.groupby(["Driver", "Stint"]).TyreLife.max()
    extra = test.TyreLife.values - oldest.reindex(list(zip(test.Driver, test.Stint))).values
    by_age = pd.DataFrame({"Track": track, "ExtraAge": extra,
                           "Error_s": predictions["held-out fit"] - test.LapTime_s.values,
                           "TumError_s": predictions["tum"] - test.LapTime_s.values})
    wear = {c: {"held_out": round(heldout_fit["tyres"][c]["k1"], 4), "all": round(full_fit["tyres"][c]["k1"], 4)}
            for c in heldout_fit["tyres"]}
    return rows, by_age, wear


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--rounds", type=int, nargs="+", default=[9, 5, 14, 15, 19])
    args = parser.parse_args()
    workbook = find_workbook(args.year)
    races = pd.read_excel(workbook, sheet_name="Races")
    all_laps = pd.read_excel(workbook, sheet_name="Laps")
    rows, ages = [], []
    for rnd in args.rounds:
        location = races[(races.Year == args.year) & (races.Round == rnd)].Location.iloc[0]
        track = tum_track(location)
        r, a, wear = check(all_laps[(all_laps.Year == args.year) & (all_laps.Round == rnd)], track, args.year)
        rows += r
        ages.append(a)
        print("%-10s wear k1 held-out fit vs all laps: %s" % (track, wear))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    ages = pd.concat(ages)
    ages["ExtraAgeBin"] = pd.cut(ages.ExtraAge, [0, 3, 6, 10, np.inf], labels=["1-3", "4-6", "7-10", "11+"])
    by_age = ages.groupby("ExtraAgeBin", observed=True).agg(
        Laps=("Error_s", "size"), MeanError_s=("Error_s", "mean"),
        RMSE_s=("Error_s", lambda e: float(np.sqrt(np.mean(e ** 2)))),
        TumMeanError_s=("TumError_s", "mean")).round(3).reset_index()
    print(by_age.to_string(index=False))
    table.to_csv(EVIDENCE / "heldout_tyres.csv", index=False)
    by_age.to_csv(EVIDENCE / "heldout_tyres_by_extra_age.csv", index=False)


if __name__ == "__main__":
    main()
