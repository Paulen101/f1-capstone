"""Fit tyre, fuel/track trend, pace and pit-loss values for one race from the workbook (.venv-data).

    fit_parameters.py --year 2019 --round 9

Output in notes/evidence/fitted_parameters/, named by TUM's track name (Barcelona -> Catalunya):
<Track>_<year>_fit.json and <Track>_<year>_results.csv (the real classification, for the checks).
Writing the values into a TUM race file is a separate step: integration/write_fitted_pars.py.

Model for a free-air lap:
    lap time = driver level - trend * (lap - 1) + k0[compound] + k1[compound] * tyre age
               (+ k2[compound] * tyre age**2 with quadratic wear)
Tyre age is TyreLife - 1, so a new tyre has age 0 as in TUM. The hard compound is the reference
(k0 = 0). Quadratic wear is used when it predicts whole stints left out of the fit better than
linear wear, by at least QUAD_MIN_GAIN_S (stint_cv). Method and limits: notes/fitted_parameters.md
and notes/heldout_check.md.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integration.race_files import tum_track  # noqa: E402

DATA_DIRS = (ROOT / "f1_data", ROOT.parent / "f1-race-data")
ORDER = ["HARD", "MEDIUM", "SOFT"]   # hardest first
FREE_AIR_GAP_S = 2.0
REFERENCE_LAPS = 3                   # clean laps that give the normal lap time next to a pit stop
REFERENCE_WINDOW = 5                 # ...taken from at most this many laps before or after it
BOOTSTRAP_RUNS = 200
MIN_DRIVER_LAPS = 5
CV_FOLDS = 5                         # stint groups for choosing linear or quadratic wear
QUAD_MIN_GAIN_S = 0.003              # quadratic wear must lower the cross-validated RMSE by this much


def add_gaps(laps: pd.DataFrame) -> pd.DataFrame:
    """Add the time gap to the car ahead on track at the end and at the start of every lap."""
    laps = laps.sort_values("LapEndTime_s").copy()
    laps["GapAhead_s"] = laps.LapEndTime_s.diff()
    laps = laps.sort_values(["Driver", "LapNumber"])
    laps["GapAheadAtStart_s"] = laps.groupby("Driver").GapAhead_s.shift(1)
    return laps


def free_air(laps: pd.DataFrame, gap: float = FREE_AIR_GAP_S) -> pd.DataFrame:
    """Clean laps with no car within `gap` seconds ahead at either end of the lap.

    A car held up early in the race laps faster as the field spreads out. That hides tyre wear and
    inflates the trend, so held-up laps are left out.
    """
    clear = (laps.GapAhead_s.fillna(np.inf) > gap) & (laps.GapAheadAtStart_s.fillna(np.inf) > gap)
    return laps[laps.CleanLap & clear & laps.Compound.isin(ORDER)]


def design(laps: pd.DataFrame, drivers: list, compounds: list, ordered: bool, quad: bool = False):
    """Build the least-squares problem. Returns matrix, lower bounds and column names.

    With `quad`, wear also gets an age squared term (k2 >= 0: wear can only speed up with age),
    ordered like k1. Only used with `ordered`.
    """
    if quad and not ordered:
        raise ValueError("Quadratic wear is only fitted with the ordering rule")
    softer = lambda i: laps.Compound.isin(compounds[i:]).values.astype(float)
    age = laps.TyreLife.values.astype(float) - 1.0
    columns = [((laps.Driver == d).values.astype(float), -np.inf, "level_" + d) for d in drivers]
    columns.append((-(laps.LapNumber.values.astype(float) - 1.0), 0.0 if ordered else -np.inf, "trend"))
    if ordered:
        # Each softer compound is the next harder one plus a step that cannot go the wrong way:
        # no slower when new (k0 step <= 0) and no slower to wear (k1 step >= 0).
        columns.append((age, 0.0, "k1_step_0"))
        if quad:
            columns.append((age ** 2, 0.0, "k2_step_0"))
        for i in range(1, len(compounds)):
            columns.append((-softer(i), 0.0, "k0_step_%d" % i))
            columns.append((softer(i) * age, 0.0, "k1_step_%d" % i))
            if quad:
                columns.append((softer(i) * age ** 2, 0.0, "k2_step_%d" % i))
    else:
        for i, compound in enumerate(compounds):
            own = (laps.Compound == compound).values.astype(float)
            if i:
                columns.append((own, -np.inf, "k0_" + compound))
            columns.append((own * age, -np.inf, "k1_" + compound))
    return (np.column_stack([c[0] for c in columns]), np.array([c[1] for c in columns]),
            [c[2] for c in columns])


def solve(laps: pd.DataFrame, ordered: bool = True, quad: bool = False) -> dict:
    """Fit the lap-time model. `ordered` keeps softer compounds from being slower or longer-lived;
    `quad` adds a k2 * age**2 wear term."""
    drivers = sorted(laps.Driver.unique())
    compounds = [c for c in ORDER if (laps.Compound == c).any()]
    matrix, lower, names = design(laps, drivers, compounds, ordered, quad)
    target = laps.LapTime_s.values.astype(float)
    solution = lsq_linear(matrix, target, bounds=(lower, np.full(len(lower), np.inf)))
    value = dict(zip(names, solution.x))
    tyres = {}
    k0 = k1 = k2 = 0.0
    for i, compound in enumerate(compounds):
        if ordered:
            k0 -= value["k0_step_%d" % i] if i else 0.0
            k1 += value["k1_step_%d" % i]
            k2 += value.get("k2_step_%d" % i, 0.0)
        else:
            k0, k1 = value.get("k0_" + compound, 0.0), value["k1_" + compound]
        tyres[compound] = {"k0": float(k0), "k1": float(k1)}
        if quad:
            tyres[compound]["k2"] = float(k2)
    residual = target - matrix @ solution.x
    return {"trend": float(value["trend"]), "tyres": tyres,
            "levels": {d: float(value["level_" + d]) for d in drivers},
            "rmse": float(np.sqrt(np.mean(residual ** 2)))}


def bootstrap(laps: pd.DataFrame, runs: int = BOOTSTRAP_RUNS, seed: int = 0, quad: bool = False) -> dict:
    """Standard errors from resampling whole stints, because laps within a stint are not independent."""
    rng = np.random.default_rng(seed)
    stints = [group for _, group in laps.groupby(["Driver", "Stint"])]
    drivers = set(laps.Driver)
    keys = ("k0", "k1", "k2") if quad else ("k0", "k1")
    draws = []
    while len(draws) < runs:
        sample = pd.concat([stints[i] for i in rng.integers(len(stints), size=len(stints))])
        if set(sample.Driver) != drivers:
            continue                    # every driver needs a level
        fit = solve(sample, quad=quad)
        draws.append([fit["trend"]] + [fit["tyres"].get(c, {}).get(k, np.nan)
                                       for c in ORDER for k in keys])
    # Spread from the quartiles: a resample that leaves a value unidentified cannot blow it up.
    low, high = np.nanpercentile(np.array(draws), [25, 75], axis=0)
    spread = (high - low) / 1.349
    names = ["trend"] + ["%s_%s" % (c, k) for c in ORDER for k in keys]
    return dict(zip(names, spread))


def predict_free_air(fit: dict, laps: pd.DataFrame) -> np.ndarray:
    """Lap time of each lap from a solve() result. NaN for a driver or compound the fit does not know."""
    k = {name: laps.Compound.map({c: t.get(name, 0.0) for c, t in fit["tyres"].items()}) for name in ("k0", "k1", "k2")}
    age = laps.TyreLife - 1
    return (laps.Driver.map(fit["levels"]) - fit["trend"] * (laps.LapNumber - 1)
            + k["k0"] + k["k1"] * age + k["k2"] * age ** 2).values


def stint_cv(free: pd.DataFrame, folds: int = CV_FOLDS, seed: int = 0) -> dict:
    """Prediction error of linear and quadratic wear on whole stints left out of the fit.

    Stints are split into `folds` groups; each group is predicted from a fit on the others. Laps of a
    driver with no other stint in the training part cannot be predicted and are skipped.
    """
    stints = free[["Driver", "Stint"]].drop_duplicates()
    order = np.random.default_rng(seed).permutation(len(stints)) % folds
    fold = pd.Series(order, index=pd.MultiIndex.from_frame(stints)).reindex(
        list(zip(free.Driver, free.Stint))).values
    errors = {"lin": [], "quad": []}
    for k in range(folds):
        train, test = free[fold != k], free[fold == k]
        for model in errors:
            prediction = predict_free_air(solve(train, quad=model == "quad"), test)
            errors[model].extend((prediction - test.LapTime_s.values)[np.isfinite(prediction)])
    return {model: round(float(np.sqrt(np.mean(np.square(e)))), 4) for model, e in errors.items()}


def stint_sequences(laps: pd.DataFrame) -> dict:
    """Compound of every stint per driver, in order. The writer matches these to TUM's codes."""
    ordered = laps.dropna(subset=["Stint"]).sort_values(["Driver", "LapNumber"])
    return {driver: group.groupby("Stint", sort=True).Compound.first().tolist()
            for driver, group in ordered.groupby("Driver")}


def reference(laps: pd.DataFrame, lap_numbers) -> float:
    """Median of the first clean laps found in `lap_numbers`, which are already in search order."""
    clean = laps.loc[[n for n in lap_numbers if n in laps.index]]
    clean = clean[clean.CleanLap].head(REFERENCE_LAPS)
    return float(clean.LapTime_s.median()) if len(clean) else np.nan


def pit_stops(laps: pd.DataFrame) -> list:
    """Time lost on the in-lap and on the out-lap of every stop, against nearby clean laps.

    The in-lap is compared with the laps just before it (old tyres) and the out-lap with the laps
    just after it (new tyres), so the tyre change itself is not counted as pit loss.
    """
    stops = []
    for driver, own in laps.groupby("Driver"):
        own = own.set_index("LapNumber")
        for lap in own.index[own.IsInLap]:
            if lap + 1 not in own.index or not own.IsOutLap[lap + 1]:
                continue                # retired in the pits, or the race ended
            inlap, outlap = own.loc[lap], own.loc[lap + 1]
            before = reference(own, range(lap - 1, lap - 1 - REFERENCE_WINDOW, -1))
            after = reference(own, range(lap + 2, lap + 2 + REFERENCE_WINDOW))
            new_tyres = outlap.Compound != inlap.Compound or outlap.TyreLife <= inlap.TyreLife
            stops.append({"driver": driver, "in_lap": int(lap),
                          "green": bool(inlap.GreenFlag and outlap.GreenFlag),
                          "tyre_change": bool(new_tyres),
                          "in_lap_loss_s": round(inlap.LapTime_s - before, 3),
                          "out_lap_loss_s": round(outlap.LapTime_s - after, 3)})
    return stops


def usable(stop: dict) -> bool:
    return bool(stop["green"] and stop["tyre_change"]
                and np.isfinite(stop["in_lap_loss_s"]) and np.isfinite(stop["out_lap_loss_s"]))


def fit_race(laps: pd.DataFrame) -> dict:
    laps = add_gaps(laps)
    free = free_air(laps)
    per_driver = free.groupby("Driver").size()
    cv = stint_cv(free)
    wear_model = "quad" if cv["quad"] < cv["lin"] - QUAD_MIN_GAIN_S else "lin"
    quad = wear_model == "quad"
    fit = solve(free, quad=quad)
    errors = bootstrap(free, quad=quad)
    stops = pit_stops(laps)
    used = [s for s in stops if usable(s)]
    tyres = {}
    for compound, values in fit["tyres"].items():
        own = free[free.Compound == compound]
        tyres[compound] = {
            "k0_s": round(values["k0"], 4) + 0.0, "k1_s_per_lap": round(values["k1"], 4),
            "k0_se": round(errors[compound + "_k0"], 4), "k1_se": round(errors[compound + "_k1"], 4),
            "laps": int(len(own)), "stints": int(own.groupby(["Driver", "Stint"]).ngroups),
            "max_age": int(own.TyreLife.max() - 1)}
        if quad:
            tyres[compound]["k2_s_per_lap2"] = round(values["k2"], 6)
            tyres[compound]["k2_se"] = round(errors[compound + "_k2"], 6)
    free_form = solve(free, ordered=False)
    all_clean = solve(laps[laps.CleanLap & laps.Compound.isin(ORDER)], ordered=False)
    plain = lambda fit: {"trend_s_per_lap": round(fit["trend"], 4),
                         "tyres": {c: {k: round(v, 4) for k, v in t.items()} for c, t in fit["tyres"].items()}}
    return {
        "laps": {"clean": int(laps.CleanLap.sum()), "free_air": int(len(free)),
                 "free_air_gap_s": FREE_AIR_GAP_S, "rmse_s": round(fit["rmse"], 3)},
        "trend_s_per_lap": round(fit["trend"], 4), "trend_se": round(errors["trend"], 4),
        "wear_model": wear_model, "wear_cv_rmse_s": cv,
        "tyres": tyres,
        "driver_level_s": {d: round(v, 3) for d, v in fit["levels"].items()},
        "driver_free_air_laps": {d: int(n) for d, n in per_driver.items()},
        "drivers_below_min_laps": sorted(d for d in laps.Driver.unique()
                                         if per_driver.get(d, 0) < MIN_DRIVER_LAPS),
        "stints": stint_sequences(laps),
        "pit": {"stops_used": len(used),
                "in_lap_loss_s": round(float(np.median([s["in_lap_loss_s"] for s in used])), 3) if used else None,
                "out_lap_loss_s": round(float(np.median([s["out_lap_loss_s"] for s in used])), 3) if used else None,
                "stops": [{**s, "used": usable(s)} for s in stops]},
        "for_comparison": {"free_air_without_ordering": plain(free_form),
                           "all_clean_laps_without_ordering": plain(all_clean)}}


def find_workbook(year: int) -> Path:
    for folder in DATA_DIRS:
        path = folder / ("F1_%d_dry_races.xlsx" % year)
        if path.is_file():
            return path
    raise FileNotFoundError("No F1_%d_dry_races.xlsx in %s" % (year, ", ".join(map(str, DATA_DIRS))))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--round", type=int, default=9)
    parser.add_argument("--workbook", type=Path, help="default: F1_<year>_dry_races.xlsx in f1_data/ "
                                                      "or ../f1-race-data/")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    workbook = args.workbook or find_workbook(args.year)
    races = pd.read_excel(workbook, sheet_name="Races")
    race = races[(races.Year == args.year) & (races.Round == args.round)]
    if len(race) != 1 or race.iloc[0].Condition != "DRY" or not race.iloc[0].Included:
        parser.error("Round %d of %d is not an included dry race in %s" % (args.round, args.year, workbook))
    laps = pd.read_excel(workbook, sheet_name="Laps")
    laps = laps[(laps.Year == args.year) & (laps.Round == args.round)]
    track = tum_track(str(race.iloc[0].Location))
    result = {"race": {"year": args.year, "round": args.round, "event": race.iloc[0].Event,
                       "location": race.iloc[0].Location},
              "tum_file": "pars_%s_%d.ini" % (track, args.year), "workbook": workbook.name,
              **fit_race(laps)}
    out = args.out or ROOT / "notes" / "evidence" / "fitted_parameters" / ("%s_%d_fit.json" % (track, args.year))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    results = pd.read_excel(workbook, sheet_name="Results")
    results = results[(results.Year == args.year) & (results.Round == args.round)]
    results[["Abbreviation", "Position", "Laps", "Status", "ResultTime_s"]].to_csv(
        out.with_name(out.name.replace("_fit.json", "_results.csv")), index=False)
    print("Laps: %(clean)d clean, %(free_air)d free air; residual %(rmse_s).3f s" % result["laps"])
    print("Trend %.4f s/lap (SE %.4f)" % (result["trend_s_per_lap"], result["trend_se"]))
    for compound, t in result["tyres"].items():
        print("%-6s k0 %+.4f (SE %.4f)  k1 %.4f (SE %.4f)%s  %d laps, %d stints, age up to %d"
              % (compound, t["k0_s"], t["k0_se"], t["k1_s_per_lap"], t["k1_se"],
                 "  k2 %.5f (SE %.5f)" % (t["k2_s_per_lap2"], t["k2_se"]) if "k2_s_per_lap2" in t else "",
                 t["laps"], t["stints"], t["max_age"]))
    print("Wear model %s; stint cross-validation RMSE %s" % (result["wear_model"], result["wear_cv_rmse_s"]))
    print("Pit loss: in-lap %(in_lap_loss_s)s s, out-lap %(out_lap_loss_s)s s, %(stops_used)d stops"
          % result["pit"])
    print("Wrote %s" % out)


if __name__ == "__main__":
    main()
