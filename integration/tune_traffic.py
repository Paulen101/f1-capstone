"""Run the race over a grid of traffic values and compare each run with the real race (.venv-tum).

    tune_traffic.py --race integration/parameters/pars_Spielberg_2019_fitted.ini

Deterministic, realstrategy for every driver, as in run_experiment.py. The values are changed in
memory only; no parameter file is written. For each run it reports the on-track passes (same
definition as data_pipeline/fit_traffic.py: cars next to each other in race position that swap
during a green lap in which neither pits, from lap 2) and the finishing order and race times
against the real result.

Output: notes/evidence/fitted_parameters/traffic_grid.csv.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
TUM = ROOT / "race-simulation"
REAL = ROOT / "notes/evidence/sim_vs_real_2019.csv"
TRAFFIC = ROOT / "notes/evidence/fitted_parameters/Spielberg_2019_traffic.json"
RACE_KEYS = ("min_t_dist", "t_duel", "t_overtake_loser")       # race_pars
TRACK_KEYS = ("t_gap_overtake",)                                 # track_pars


def setup():
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools" / "matplotlib"))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(TUM))
    from integration.optional_tensorflow import defer_tensorflow
    defer_tensorflow()


def simulate(source: Path, values: dict):
    """Run one deterministic realstrategy race with `values` replacing the file's traffic values."""
    import numpy as np
    from racesim.src.import_pars import import_pars
    from racesim.src.race import Race
    random.seed(0)
    np.random.seed(0)
    pars, paths = import_pars(use_print=False, use_vse=True, race_pars_file=str(source),
                             mcs_pars_file="pars_mcs.ini")
    for key, value in values.items():
        pars["race_pars" if key in RACE_KEYS else "track_pars"][key] = value
    pars["vse_pars"]["vse_type"] = dict.fromkeys(pars["race_pars"]["participants"], "realstrategy")
    race = Race(**pars, vse_paths=paths, use_prob_infl=False, create_rand_events=False)
    race.simulate_race()
    return race


def count_passes(race) -> int:
    """On-track passes between cars next to each other in race position, pit laps left out."""
    in_laps = [{row[0] for row in d.strategy_info[1:]} for d in race.drivers_list]
    passes = 0
    for lap in range(2, race.cur_lap + 1):
        order = sorted(range(race.no_drivers), key=lambda i: race.positions[lap - 1, i])
        for front, back in zip(order[:-1], order[1:]):
            if not (race.bool_driving[lap, front] and race.bool_driving[lap, back]):
                continue
            if {lap - 1, lap} & (in_laps[front] | in_laps[back]):
                continue
            passes += int(race.positions[lap, back] < race.positions[lap, front])
    return passes


def score(race, real) -> dict:
    import numpy as np
    import pandas as pd
    from data_pipeline.compare_2019 import compare
    sim = pd.DataFrame([{"Driver": d.initials, "Position": int(race.positions[-1, i]),
                         "SimulatedLaps": float(race.progress[i]),
                         "RaceTime_s": float(race.racetimes[int(race.progress[i]), i]),
                         "DrivingAtEnd": bool(race.bool_driving[-1, i])}
                        for i, d in enumerate(race.drivers_list)])
    merged = compare(sim, real, race.race_pars["tot_no_laps"], "winner-and-gaps").set_index("Driver")
    errors = merged.TimeError_s.dropna()
    top = merged.sort_values("Position_sim").index[:4].tolist()
    return {"PositionMAE": round(float(merged.PositionError.abs().mean()), 2),
            "ExactPositions": int((merged.PositionError == 0).sum()),
            "RankCorrelation": round(float(merged[["Position_sim", "Position_real"]]
                                           .corr(method="spearman").iloc[0, 1]), 3),
            "Top4": "-".join(top),
            "TimeMeanError_s": round(float(errors.mean()), 1),
            "TimeMAE_s": round(float(errors.abs().mean()), 1),
            "Positions": " ".join("%s%d" % (d, p) for d, p in merged.Position_sim.sort_values().items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race", default="integration/parameters/pars_Spielberg_2019_fitted.ini")
    parser.add_argument("--gap", type=float, nargs="+",
                        default=[2.01, 1.5, 1.0, 0.75, 0.5, 0.4, 0.315, 0.2, 0.1, 0.05],
                        help="t_gap_overtake values; TUM accepts only 0 < value < 5")
    parser.add_argument("--loser", type=float, nargs="+", help="t_overtake_loser values (default: TUM, measured)")
    parser.add_argument("--out", type=Path, default=ROOT / "notes/evidence/fitted_parameters/traffic_grid.csv")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 8):
        parser.error("Run this script with .venv-tum (Python 3.8)")
    setup()
    import pandas as pd
    from integration.race_files import resolve_race_file
    source = resolve_race_file(args.race)
    measured = {k: v["measured"] for k, v in json.loads(TRAFFIC.read_text(encoding="utf-8"))["values"].items()}
    tum = {k: v["tum"] for k, v in json.loads(TRAFFIC.read_text(encoding="utf-8"))["values"].items()}
    joined = pd.read_csv(REAL)
    real = joined[["Abbreviation", "Position_real", "Laps", "Status", "ResultTime_s"]].rename(
        columns={"Position_real": "Position"})
    rows = []
    sets = {"tum": {k: tum[k] for k in RACE_KEYS}, "measured": {k: measured[k] for k in RACE_KEYS}}
    losers = args.loser or sorted({tum["t_overtake_loser"], measured["t_overtake_loser"]})
    for (name, base), gap, loser in itertools.product(sets.items(), args.gap, losers):
        values = dict(base, t_gap_overtake=gap, t_overtake_loser=loser)
        race = simulate(source, values)
        row = {"duel_and_dist": name, **{k: round(v, 3) for k, v in values.items()},
               "status": int(race.result_status)}
        if race.result_status == 0:
            row.update(passes=count_passes(race), **score(race, real))
        rows.append(row)
        print({k: v for k, v in row.items() if k != "Positions"}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("Real race: 40 passes, top 4 VER-LEC-BOT-HAM. Wrote %s" % args.out)


if __name__ == "__main__":
    main()
