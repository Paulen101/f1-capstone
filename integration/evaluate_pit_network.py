"""Race the pit-timing network against TUM's VSE, the real strategy and the best fixed lap (.venv-tum).

    evaluate_pit_network.py --model integration/models/pit_network_v1.npz --runs 100 --jobs 28

For every driver of the fitted 2019 races it simulates the same Monte Carlo runs (same seeds: same random
safety cars, retirements and lap-time noise) once per policy for that driver; every other driver keeps their
real strategy, as in the training data:

    network   integration/pit_network.py decides each lap; one stop onto the driver's training stop compound
    network_tum  the same network for the timing, the compound from TUM's compound-choice network
              (integration/tum_compound_choice.py), as the project scope plans
    network_two  the network may also make a second stop (window and compound as in
              integration/two_stop_generator.py); needs a network trained with StopsMade
    network_choice  as network_two, and the network also chooses the first stop's compound among the legal
              ones (integration/compound_runs.py); a second stop only after a stop onto the stored compound
    real      the driver's real 2019 strategy (pit laps and compounds)
    tum       TUM's supervised VSE (nn_supervised_tirechange + compoundchoice) for this driver
    fixed     one stop on the Monte Carlo label lap (best mean position over the training runs), same compound

Seeds start at --seed (default 500000), far from the training runs' 0-199, so these are new random races.
Output in --out (default integration/output/evaluation/<model>_mc<runs>/): runs.csv (one row per simulated
race) and manifest.json. data_pipeline/summarise_evaluation.py compares the policies.
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integration.label_generator import (MAX_ATTEMPTS, complete_background_strategies, load, pick_label,  # noqa: E402
                                         run_seed, setup)
from integration.run_situations import stop_compounds  # noqa: E402
from integration.write_fitted_pars import parse_pars  # noqa: E402

POLICIES = ["network", "network_tum", "network_two", "network_choice", "real", "tum", "fixed"]
YEAR = 2019
SITUATIONS = ROOT / "integration/output/situations"


def race_file(track: str) -> Path:
    return ROOT / "integration/parameters" / ("pars_%s_%d_fitted.ini" % (track, YEAR))


def label_lap(track: str, driver: str, runs: int = 200):
    """Best fixed pit lap of the training Monte Carlo runs (label_generator.pick_label)."""
    path = SITUATIONS / ("%s_%d_fitted_%s_mc%d" % (track, YEAR, driver, runs)) / "summary.csv"
    rows = []
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            finished = int(row["FinishedRuns"])
            rows.append({"PitLap": int(row["PitLap"]), "FinishedRuns": finished,
                         "MeanPosition": float(row["MeanPosition"]) if finished else 0.0,
                         "MeanLaps": float(row["MeanLaps"]) if finished else 0.0,
                         "MeanRaceTime_s": float(row["MeanRaceTime_s"]) if finished else 0.0})
    return pick_label(rows)


def simulate(task):
    """All runs of one (track, driver, policy). Runs in a worker process."""
    settings, track, driver, policy, runs = task
    setup()
    import numpy as np
    from racesim.src.race import Race
    from integration.fixed_lap_policy import FixedLapPolicy, FixedLapVSE
    from integration.pit_network import NetworkVSE, PitNetwork
    source = race_file(track)
    args = argparse.Namespace(race=str(source), driver=driver, deterministic=False, events="sampled")
    _, pars, paths, _, _ = load(args)
    if policy == "real":
        complete_background_strategies(pars, target=None)   # a driver who really retired must now finish legally
    if policy == "tum":
        pars["vse_pars"]["vse_type"][driver] = "supervised"
    stop = stop_compounds(parse_pars(source.read_bytes().decode("utf-8")))[driver]
    team = pars["driver_pars"][driver]["team"]
    track_pars = pars["track_pars"]
    pit_loss = round(track_pars["t_pitdrive_inlap"] + track_pars["t_pitdrive_outlap"]
                     + track_pars["t_pit_tirechange_min"] + pars["car_pars"][team]["t_pit_tirechange_add"], 3)
    network = PitNetwork(settings["model"]) if policy.startswith("network") else None
    choice, second, others = None, None, None
    if policy in ("network_two", "network_choice"):
        from integration.two_stop_generator import second_stop_compounds
        second = second_stop_compounds(parse_pars(source.read_bytes().decode("utf-8")))[driver]
    if policy == "network_choice":
        from integration.compound_runs import other_compounds
        others = other_compounds(parse_pars(source.read_bytes().decode("utf-8")))[driver]
    if policy == "network_tum":
        from integration.tum_compound_choice import TumCompoundChoice
        choice = TumCompoundChoice(paths, track_pars["name"], pars["vse_pars"]["available_compounds"],
                                   pars["vse_pars"]["param_dry_compounds"])
    fixed_lap = label_lap(track, driver) if policy == "fixed" else None
    tot = pars["race_pars"]["tot_no_laps"]
    rows, invalid = [], 0
    for run in runs:
        for attempt in range(MAX_ATTEMPTS):
            seed = run_seed(settings["seed"], run, attempt)
            random.seed(seed)
            np.random.seed(seed % 2 ** 32)
            race = Race(**pars, vse_paths=paths, use_prob_infl=True, create_rand_events=True)
            # CAPSTONE HOOK: public property injection; upstream source stays unchanged.
            if policy.startswith("network"):
                race.vse = NetworkVSE(race.vse, race, driver, network, None if choice else stop,
                                      pars["vse_pars"]["location_cat"], pit_loss, compound_choice=choice,
                                      second_compound=second, min_stint=settings["min_stint"],
                                      other_compounds=others)
            elif policy == "fixed":
                race.vse = FixedLapVSE(race.vse, driver, FixedLapPolicy(fixed_lap, stop))
            race.simulate_race()
            if race.result_status == 0:
                break
            invalid += 1
        else:
            raise RuntimeError("No valid race in %d attempts: %s %s %s run %d"
                               % (MAX_ATTEMPTS, track, driver, policy, run))
        idx = next(i for i, d in enumerate(race.drivers_list) if d.initials == driver)
        stops = [(s[0], s[1]) for s in race.drivers_list[idx].strategy_info[1:]]
        laps = float(race.progress[idx])
        rows.append({"Track": track, "Driver": driver, "Policy": policy, "Run": run, "Seed": seed,
                     "Position": int(race.positions[-1, idx]), "SimulatedLaps": laps, "TotLaps": tot,
                     "RaceTime_s": round(float(race.racetimes[int(laps), idx]), 3),
                     "Finished": bool(race.bool_driving[-1, idx]), "Stops": len(stops),
                     "FirstStopLap": stops[0][0] if stops else "",
                     "StopPlan": " ".join("%d:%s" % s for s in stops),
                     # random events of this race (the same for every policy with this seed)
                     "SCPhases": sum(1 for p in race.fcy_data["phases"] if p[2] == "SC"),
                     "VSCPhases": sum(1 for p in race.fcy_data["phases"] if p[2] == "VSC"),
                     "FirstStopFlag": race.flagstates[stops[0][0]] if stops else ""})
    return rows, invalid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=ROOT / "integration/models/pit_network_v1.npz")
    parser.add_argument("--tracks", nargs="+", default=["Spielberg", "Monza", "Singapore", "Austin"])
    parser.add_argument("--drivers", nargs="+", help="default: every driver of the race")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=500000)
    parser.add_argument("--jobs", type=int, default=28)
    parser.add_argument("--chunk", type=int, default=25, help="runs per worker task")
    parser.add_argument("--min-stint", type=int, default=10, help="network_two: laps between stops and after the last")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 8):
        parser.error("Run this script with the .venv-tum Python (3.8)")
    out = args.out or ROOT / "integration/output/evaluation" / ("%s_mc%d" % (args.model.stem, args.runs))
    out.mkdir(parents=True, exist_ok=True)
    settings = {"model": str(args.model.resolve()), "seed": args.seed, "min_stint": args.min_stint}
    tasks = []
    for track in args.tracks:
        drivers = sorted(stop_compounds(parse_pars(race_file(track).read_bytes().decode("utf-8"))))
        for driver in args.drivers or drivers:
            if driver not in drivers:
                continue
            for policy in args.policies:
                for first in range(0, args.runs, args.chunk):
                    tasks.append((settings, track, driver, policy,
                                  list(range(first, min(first + args.chunk, args.runs)))))
    started = time.perf_counter()
    rows, invalid = [], 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, (part, bad) in enumerate(pool.map(simulate, tasks), 1):
            rows += part
            invalid += bad
            if i % 20 == 0 or i == len(tasks):
                print("%d/%d tasks (%.0f s)" % (i, len(tasks), time.perf_counter() - started), flush=True)
    with open(out / "runs.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {"model": str(args.model), "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                "tracks": args.tracks, "drivers": args.drivers, "policies": args.policies, "runs": args.runs,
                "seed": args.seed, "invalid_attempts": invalid, "races": len(rows),
                "seconds": round(time.perf_counter() - started)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote %s: %d races, %d invalid attempts re-run (%.0f s)"
          % (out / "runs.csv", len(rows), invalid, time.perf_counter() - started))


if __name__ == "__main__":
    main()
