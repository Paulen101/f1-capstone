"""Compare a TUM CSV classification with workbook results; never tune parameters."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def compare(simulated, real, total_laps, time_format):
    if simulated["Driver"].duplicated().any() or real["Abbreviation"].duplicated().any():
        raise ValueError("Expected one result per driver in one race")
    if set(simulated.Driver) != set(real.Abbreviation):
        raise ValueError("Driver sets differ; do not silently drop unmatched drivers")
    merged = simulated.merge(real, left_on="Driver", right_on="Abbreviation",
                             suffixes=("_sim", "_real"), validate="one_to_one")
    merged["PositionError"] = merged.Position_sim - merged.Position_real
    winner_time = real.loc[real.Position == 1, "ResultTime_s"].iloc[0]
    if not np.isfinite(winner_time) or winner_time <= 0:
        raise ValueError("The winner needs a valid total race time")
    merged["ActualRaceTime_s"] = merged.ResultTime_s
    other_finishers = ((merged.Position_real != 1) & (merged.Status == "Finished")
                       & (merged.Laps == total_laps) & merged.ResultTime_s.notna())
    if time_format == "winner-and-gaps":
        if (merged.loc[other_finishers, "ResultTime_s"] > winner_time * 0.5).any():
            raise ValueError("ResultTime_s appears to contain totals, not gaps")
        merged.loc[other_finishers, "ActualRaceTime_s"] += winner_time
    elif time_format == "totals":
        if (merged.loc[other_finishers, "ResultTime_s"] < winner_time * 0.5).any():
            raise ValueError("ResultTime_s appears to contain gaps, not totals")
    else:
        raise ValueError("Specify totals or winner-and-gaps")
    comparable = ((merged.SimulatedLaps == total_laps) & (merged.Laps == total_laps)
                  & merged.DrivingAtEnd & merged.ResultTime_s.notna()
                  & merged.RaceTime_s.notna() & (merged.Status == "Finished"))
    merged["TimeComparable"] = comparable
    merged["TimeError_s"] = (merged.RaceTime_s - merged.ActualRaceTime_s).where(comparable)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--round", type=int, default=9, choices=[9],
                        help="This week 2 benchmark uses the Austrian GP")
    parser.add_argument("--result-time-format", required=True, choices=["totals", "winner-and-gaps"],
                        help="Verify the source Time field, then specify its interpretation")
    parser.add_argument("--simulation", type=Path, default=ROOT / "integration/output/realstrategy")
    parser.add_argument("--report", type=Path, default=ROOT / "notes/sim_vs_real_2019.md")
    args = parser.parse_args()
    manifest = json.loads((args.simulation / "manifest.json").read_text())
    if (manifest["mode"] != "realstrategy" or manifest["result_status"] != 0
            or manifest["race_file"] != "pars_Spielberg_2019.ini"
            or manifest["use_prob_infl"] or manifest["create_rand_events"]):
        raise ValueError("Expected a valid deterministic Spielberg 2019 realstrategy run")
    races = pd.read_excel(args.workbook, sheet_name="Races")
    race = races[(races.Year == 2019) & (races.Round == args.round)]
    if len(race) != 1:
        raise ValueError("Expected exactly one 2019 race in Races sheet")
    verified_dry = (race.iloc[0].get("Condition") == "DRY"
                    and bool(race.iloc[0].get("Included", False))
                    and race.iloc[0].get("LoadStatus") == "OK")
    real = pd.read_excel(args.workbook, sheet_name="Results")
    real = real[(real.Year == 2019) & (real.Round == args.round)].copy()
    sim = pd.read_csv(args.simulation / "classification.csv")
    total_laps = int(real.Laps.max())
    merged = compare(sim, real, total_laps, args.result_time_format).sort_values("Position_real")
    evidence = ROOT / "notes/evidence"
    evidence.mkdir(exist_ok=True)
    merged.to_csv(evidence / "sim_vs_real_2019.csv", index=False)
    time_errors = merged.TimeError_s.dropna()
    mae = merged.PositionError.abs().mean()
    winner_sim = merged.loc[merged.Position_sim == 1].iloc[0]
    winner_real = merged.loc[merged.Position_real == 1].iloc[0]
    duration_error = winner_sim.RaceTime_s - winner_real.ActualRaceTime_s
    relative = lambda path: str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path.resolve())
    lines = ["# Simulator vs reality: 2019 Austrian Grand Prix", "",
             "Dry-race verification: **%s**." % ("verified by workbook Races sheet" if verified_dry else "PENDING; public results only, no lap/weather verification"), "",
             "Source: `%s`, `Results`, `Year == 2019 and Round == %d`; all %d drivers. "
             "Simulation: `%s/classification.csv`, `realstrategy` for every driver, "
             "probabilistic influences and random events disabled. Original Spielberg 2019 parameters; no tuning."
             % (relative(args.workbook), args.round, len(merged), relative(args.simulation)), "",
             "- Finishing-position MAE: **%.3f places** (%d drivers)." % (mae, len(merged)),
             "- Exact finishing positions: **%d/%d**." % ((merged.PositionError == 0).sum(), len(merged)),
             "- Simulated winner: **%s**, %.3f s; actual winner: **%s**, %.3f s."
             % (winner_sim.Driver, winner_sim.RaceTime_s, winner_real.Driver, winner_real.ActualRaceTime_s),
             "- Race duration error (simulated winner minus actual winner): **%+.3f s**." % duration_error]
    if len(time_errors):
        lines.append("- Comparable full-distance driver times: **%d**; MAE **%.3f s**, RMSE **%.3f s**, mean signed error **%+.3f s**."
                     % (len(time_errors), time_errors.abs().mean(), np.sqrt((time_errors ** 2).mean()), time_errors.mean()))
    lines += ["", "| Driver | Actual position | Simulated position | Actual laps | Simulated laps | Time error (s) |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in merged.itertuples():
        error = "%+.3f" % row.TimeError_s if row.TimeComparable else "excluded"
        lines.append("| %s | %d | %d | %d | %d | %s |" %
                     (row.Driver, row.Position_real, row.Position_sim, row.Laps, row.SimulatedLaps, error))
    lines += ["", "Time comparisons include only drivers who finished the full scheduled distance in both datasets, "
              "with `Status == Finished` and finite times. Lapped drivers, retirements, and unequal distances are excluded "
              "from time metrics; they remain in the position metric. Time interpretation was explicitly selected as "
              "`%s`. The joined CSV retains raw `ResultTime_s` and normalized `ActualRaceTime_s`. "
              "For the public FastF1 3.8.3 export, the winner has 4921.822 s and Leclerc has a 2.724 s gap; "
              "the normalized Leclerc total is 4924.546 s, verified against Jolpica's `millis=4924546`. "
              "The installed FastF1 documentation describes totals, but these observed values require winner-plus-gap normalization."
              % args.result_time_format, "",
              "This race is a diagnostic, not evidence that the simulator reproduces reality. The finishing-order errors "
              "are substantial, even with the recorded pit strategies. Traffic, fitted pace and tire parameters, penalties, "
              "and race events can contribute; this comparison does not identify their individual effects. No parameters were tuned.", "",
              "Raw joined values: [evidence/sim_vs_real_2019.csv](evidence/sim_vs_real_2019.csv). "
              "Run logs and simulator validity manifests are saved with the experiment outputs.", ""]
    args.report.write_text("\n".join(lines))
    print("Position MAE %.3f; time MAE %.3f (%d full-distance drivers); winner duration error %+.3f s" %
          (mae, time_errors.abs().mean(), len(time_errors), duration_error))


if __name__ == "__main__":
    main()
