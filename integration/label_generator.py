"""Simulate every candidate pit lap for one driver and pick the best one (use .venv-tum).

    label_generator.py --race pars_Spielberg_2019.ini --driver VER --compound A3 --deterministic
    label_generator.py --race pars_Spielberg_2019.ini --driver VER --compound A3 --runs 10

Output in integration/output/labels/<race>_<driver>_<mode>/:
    runs.csv       one row per simulated race
    summary.csv    one row per candidate pit lap
    manifest.json  settings, race-file hash, runtime, label

The label is a pit lap: best mean finishing position, then more laps completed, then lower
mean race time. Turning it into per-lap pit/stay training rows is a later step.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TUM = ROOT / "race-simulation"
MAX_ATTEMPTS = 5  # TUM's own Monte Carlo loop also re-runs races it marks invalid
ATTEMPT_SEED_STEP = 1000003


def run_seed(base: int, run: int, attempt: int = 0) -> int:
    """Every candidate lap uses the same seed for a given run, so they share random events."""
    return base + run + attempt * ATTEMPT_SEED_STEP


def clear_recorded_events(event_pars: dict) -> None:
    """Make TUM draw safety cars and retirements in each run; empty lists request that.

    A decision model cannot know future events, so labels should not depend on the real ones.
    """
    event_pars["fcy_data"]["phases"] = []
    event_pars["retire_data"]["retirements"] = []


def check_stop(stops: list, pit_lap: int, compound: str, finished: bool) -> bool:
    """Return whether the requested stop ran; raise if the simulator did anything else."""
    if not stops and not finished:
        return False  # retired before reaching the pit lap
    if len(stops) != 1 or stops[0][0] != pit_lap or stops[0][1] != compound:
        raise AssertionError("Expected one stop on lap %s onto %s, got %s" % (pit_lap, compound, stops))
    return True


def summarize(rows: list, tot_no_laps: int) -> list:
    """Reduce run rows to one row per pit lap. Retired runs have no finishing position."""
    by_lap = {}
    for row in rows:
        by_lap.setdefault(row["PitLap"], []).append(row)
    summary = []
    for pit_lap in sorted(by_lap):
        runs = by_lap[pit_lap]
        finished = [row for row in runs if row["Finished"]]
        if not finished:
            summary.append({"PitLap": pit_lap, "Runs": len(runs), "FinishedRuns": 0, "MeanPosition": "",
                            "StdPosition": "", "MeanLaps": "", "MeanRaceTime_s": "",
                            "FullDistanceRuns": 0, "MeanFullDistanceTime_s": ""})
            continue
        positions = [row["Position"] for row in finished]
        full = [row["RaceTime_s"] for row in finished if row["SimulatedLaps"] >= tot_no_laps]
        summary.append({
            "PitLap": pit_lap, "Runs": len(runs), "FinishedRuns": len(finished),
            "MeanPosition": round(statistics.mean(positions), 6),
            "StdPosition": round(statistics.stdev(positions), 6) if len(positions) > 1 else 0.0,
            "MeanLaps": round(statistics.mean(row["SimulatedLaps"] for row in finished), 6),
            "MeanRaceTime_s": round(statistics.mean(row["RaceTime_s"] for row in finished), 3),
            "FullDistanceRuns": len(full),
            "MeanFullDistanceTime_s": round(statistics.mean(full), 3) if full else ""})
    return summary


def pick_label(summary: list):
    """Best mean position; a lapped car's shorter race time must not win the tie-break."""
    candidates = [row for row in summary if row["FinishedRuns"]]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row["MeanPosition"], -row["MeanLaps"],
                                            row["MeanRaceTime_s"]))["PitLap"]


def pick_fastest(summary: list):
    """Lowest race time using only full-distance runs. Diagnostic: time and position can disagree."""
    candidates = [row for row in summary
                  if row["FinishedRuns"] and row["FullDistanceRuns"] * 2 >= row["FinishedRuns"]]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["MeanFullDistanceTime_s"])["PitLap"]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def simulate_run(race_class, pars, paths, driver, policy, policy_vse, seed, stochastic):
    """Simulate one race; return its target-driver outcome and whether TUM marked it valid."""
    import numpy as np
    # Seed before building the race: TUM draws random events in the constructor.
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    race = race_class(**pars, vse_paths=paths, use_prob_infl=stochastic, create_rand_events=stochastic)
    # CAPSTONE HOOK: public property injection; upstream source stays unchanged.
    race.vse = policy_vse(race.vse, driver, policy)
    race.simulate_race()
    if race.result_status != 0:
        return None, race.result_status
    idx = next(i for i, d in enumerate(race.drivers_list) if d.initials == driver)
    finished = bool(race.bool_driving[-1, idx])
    stops = [(stop[0], stop[1]) for stop in race.drivers_list[idx].strategy_info[1:]]
    laps = float(race.progress[idx])
    return {"Position": int(race.positions[-1, idx]), "SimulatedLaps": laps,
            "RaceTime_s": round(float(race.racetimes[int(laps), idx]), 3), "Finished": finished,
            "StopExecuted": check_stop(stops, policy.pit_lap, policy.compound, finished)}, 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race", default="pars_Spielberg_2019.ini",
                        help="TUM race file name, or the path of one of our fitted files")
    parser.add_argument("--driver", required=True)
    parser.add_argument("--compound", required=True, help="fixed compound for the single stop")
    parser.add_argument("--deterministic", action="store_true",
                        help="one run per lap, no randomness, events exactly as in the race file")
    parser.add_argument("--runs", type=int, default=10, help="Monte Carlo runs per candidate lap")
    parser.add_argument("--events", choices=["sampled", "recorded"], default="sampled",
                        help="Monte Carlo only: draw safety cars and retirements each run, or keep the "
                             "race file's real ones (random ones are still drawn where the file has none)")
    parser.add_argument("--first-lap", type=int, default=2)
    parser.add_argument("--last-lap", type=int, help="default: second-to-last lap of the race")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 8):
        parser.error("Run this script with the .venv-tum Python (3.8)")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    return parser, args


def main():
    parser, args = parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools" / "matplotlib"))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(TUM))
    from integration.optional_tensorflow import defer_tensorflow
    defer_tensorflow()
    from racesim.src.import_pars import import_pars
    from racesim.src.race import Race
    from integration.fixed_lap_policy import FixedLapPolicy, FixedLapVSE
    from integration.race_files import resolve_race_file

    source = resolve_race_file(args.race)
    stochastic = not args.deterministic
    runs = 1 if args.deterministic else args.runs
    events = "file" if args.deterministic else args.events
    pars, paths = import_pars(use_print=False, use_vse=True, race_pars_file=str(source),
                              mcs_pars_file="pars_mcs.ini")
    participants = pars["race_pars"]["participants"]
    if args.driver not in participants:
        parser.error("Unknown driver for this race: %s" % args.driver)
    if any(stint[1] in ("I", "W") for strategy in pars["vse_pars"]["real_strategy"].values()
           for stint in strategy):
        parser.error("Dry races only")
    if pars["vse_pars"]["real_strategy"][args.driver][0][1] == args.compound:
        parser.error("Choose a fixed compound different from the start compound")
    pars["vse_pars"]["vse_type"] = dict.fromkeys(participants, "realstrategy")
    if stochastic and args.events == "sampled":
        clear_recorded_events(pars["event_pars"])

    tot_no_laps = pars["race_pars"]["tot_no_laps"]
    last_lap = args.last_lap if args.last_lap is not None else tot_no_laps - 1
    if not 1 <= args.first_lap <= last_lap < tot_no_laps:
        parser.error("Pit laps must lie between 1 and %d" % (tot_no_laps - 1))

    mode = "deterministic" if args.deterministic else "mc%d" % runs
    out = args.out or ROOT / "integration" / "output" / "labels" / (
        "%s_%s_%s" % (source.stem.replace("pars_", ""), args.driver, mode))
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    invalid = 0
    started = time.perf_counter()
    for pit_lap in range(args.first_lap, last_lap + 1):
        policy = FixedLapPolicy(pit_lap, args.compound)
        for run in range(runs):
            for attempt in range(MAX_ATTEMPTS):
                seed = run_seed(args.seed, run, attempt)
                result, status = simulate_run(Race, pars, paths, args.driver, policy, FixedLapVSE,
                                              seed, stochastic)
                if result is not None:
                    break
                if not stochastic:
                    raise RuntimeError("TUM rejected the simulation: result_status=%s" % status)
                invalid += 1
            else:
                raise RuntimeError("No valid race in %d attempts: lap %d, run %d"
                                   % (MAX_ATTEMPTS, pit_lap, run))
            rows.append({"Race": source.name, "Driver": args.driver, "Compound": args.compound,
                         "PitLap": pit_lap, "Run": run, "Seed": seed, **result})
        print("lap %d/%d done (%.0f s)" % (pit_lap, last_lap, time.perf_counter() - started), flush=True)
    elapsed = time.perf_counter() - started

    summary = summarize(rows, tot_no_laps)
    label = pick_label(summary)
    fastest = pick_fastest(summary)
    write_csv(out / "runs.csv", rows)
    write_csv(out / "summary.csv", summary)
    manifest = {"race_file": source.name,
                "race_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "python": sys.version, "driver": args.driver, "compound": args.compound,
                "first_lap": args.first_lap, "last_lap": last_lap, "runs_per_lap": runs,
                "use_prob_infl": stochastic, "create_rand_events": stochastic, "events": events,
                "background_vse": "realstrategy", "seed": args.seed, "races_simulated": len(rows),
                "invalid_races_rerun": invalid, "runtime_s": round(elapsed, 1),
                "ms_per_race": round(elapsed / len(rows) * 1000, 1),
                "label_pit_lap": label, "fastest_pit_lap": fastest}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Label: pit on lap %s. Fastest: lap %s. %d races, %d invalid re-run, %.1f ms per race. Output: %s"
          % (label, fastest, len(rows), invalid, manifest["ms_per_race"], out))


if __name__ == "__main__":
    main()
