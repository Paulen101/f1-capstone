"""Compare realstrategy runs on TUM's and on our fitted parameters with the real 2019 Austrian GP.

    .venv-data python data_pipeline/compare_fitted.py

Real results come from notes/evidence/sim_vs_real_2019.csv, which compare_2019.py wrote from the
workbook, so this needs no workbook. Every folder integration/output/realstrategy_Spielberg_2019_*
is included. Output: notes/evidence/fitted_parameters/sim_vs_real.csv (per driver) and
sim_vs_real_summary.csv (per parameter set).

If the 2019 workbook is found, it also checks each parameter file as a lap-time model: the lap time
TUM's formula gives for every real clean lap, without any traffic, against the measured lap time.
Output: lap_model_check.csv.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.compare_2019 import compare  # noqa: E402
from data_pipeline.fit_parameters import add_gaps, find_workbook, free_air  # noqa: E402
from integration.write_fitted_pars import TUM_PARAMETERS, compound_codes, parse_pars  # noqa: E402

FIT = ROOT / "notes/evidence/fitted_parameters/Spielberg_2019_fit.json"
FILES = {"tum": TUM_PARAMETERS / "pars_Spielberg_2019.ini",
         "fitted": ROOT / "integration/parameters/pars_Spielberg_2019_fitted.ini"}

TOTAL_LAPS = 71
OUTPUT = ROOT / "integration" / "output"
PREFIX = "realstrategy_Spielberg_2019_"


def predict(pars: dict, laps: pd.DataFrame, codes: dict) -> np.ndarray:
    """Lap time from TUM's formula for a car alone on track: pace, tyre and fuel terms only."""
    track = pars["track_pars"]
    times = []
    for lap in laps.itertuples():
        car = pars["car_pars"][pars["driver_pars"][lap.Driver]["team"]]
        tyre = pars["tireset_pars"][lap.Driver][codes[lap.Compound]]
        fuel = car["m_fuel"] - car["b_fuel_perlap"] * (lap.LapNumber - 1)
        times.append(track["t_q"] + track["t_gap_racepace"] + car["t_car"]
                     + pars["driver_pars"][lap.Driver]["t_driver"] + tyre["k_0"]
                     + tyre["k_1_lin"] * (lap.TyreLife - 1) + fuel * track["t_lap_sens_mass"])
    return np.array(times)


def lap_model_check() -> pd.DataFrame:
    import json
    laps = pd.read_excel(find_workbook(2019), sheet_name="Laps")
    laps = add_gaps(laps[(laps.Year == 2019) & (laps.Round == 9)])
    clean = laps[laps.CleanLap]
    fit = json.loads(FIT.read_text(encoding="utf-8"))
    rows = []
    for name, path in FILES.items():
        pars = parse_pars(path.read_bytes().decode("utf-8"))
        codes = compound_codes(fit, parse_pars(FILES["tum"].read_bytes().decode("utf-8")))
        for subset, data in (("free-air laps", free_air(laps)), ("all clean laps", clean)):
            error = predict(pars, data, codes) - data.LapTime_s.values
            rows.append({"Parameters": name, "Laps": subset, "Count": len(data),
                         "RMSE_s": round(float(np.sqrt(np.mean(error ** 2))), 3),
                         "MeanError_s": round(float(error.mean()), 3)})
    return pd.DataFrame(rows)


def main():
    joined = pd.read_csv(ROOT / "notes/evidence/sim_vs_real_2019.csv")
    real = joined[["Abbreviation", "Position_real", "Laps", "Status", "ResultTime_s"]].rename(
        columns={"Position_real": "Position"})
    folders = {"tum": OUTPUT / "realstrategy"}
    folders.update({path.name[len(PREFIX):]: path for path in sorted(OUTPUT.glob(PREFIX + "*"))})
    table = pd.DataFrame({"ActualPosition": real.set_index("Abbreviation").Position})
    summary = []
    for name, folder in folders.items():
        sim = pd.read_csv(folder / "classification.csv")
        merged = compare(sim, real, TOTAL_LAPS, "winner-and-gaps").set_index("Driver")
        errors = merged.TimeError_s.dropna()
        ranks = merged[["Position_sim", "Position_real"]].corr(method="spearman").iloc[0, 1]
        summary.append({"Parameters": name, "PositionMAE": round(merged.PositionError.abs().mean(), 3),
                        "ExactPositions": int((merged.PositionError == 0).sum()),
                        "RankCorrelation": round(ranks, 3),
                        "Winner": merged.index[merged.Position_sim == 1][0],
                        "TimeDrivers": len(errors), "TimeMAE_s": round(errors.abs().mean(), 3),
                        "TimeRMSE_s": round(float(np.sqrt((errors ** 2).mean())), 3),
                        "TimeMeanError_s": round(errors.mean(), 3)})
        table[name + "_Position"] = merged.Position_sim
        table[name + "_TimeError_s"] = merged.TimeError_s.round(3)
    summary = pd.DataFrame(summary)
    print(summary.to_string(index=False))
    out = ROOT / "notes/evidence/fitted_parameters"
    out.mkdir(parents=True, exist_ok=True)
    table.sort_values("ActualPosition").to_csv(out / "sim_vs_real.csv", index_label="Driver")
    summary.to_csv(out / "sim_vs_real_summary.csv", index=False)
    try:
        check = lap_model_check()
    except FileNotFoundError as missing:
        print("Lap-time model check skipped: %s" % missing)
    else:
        print(check.to_string(index=False))
        check.to_csv(out / "lap_model_check.csv", index=False)
    print("Wrote %s" % out)


if __name__ == "__main__":
    main()
