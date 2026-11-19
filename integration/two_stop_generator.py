"""Monte Carlo runs for one- and two-stop plans, with the race situation in both stints (.venv-tum).

    two_stop_generator.py --track Austin --runs 50 --jobs 28 [--drivers VER HAM]

For each driver and Monte Carlo run r (seed r, as in the one-stop runs) it simulates:

    one stop on L1                  every L1 from lap 2 to the second-to-last lap, onto C1
    two stops on L1 and L2          L1 >= --first-stop-min, L2 >= L1 + --min-stint, L2 <= laps - --min-stint;
                                    first onto C1, then onto C2

C1 is the compound of the one-stop runs (integration/run_situations.stop_compounds). C2 is the driver's real
last compound if they really stopped twice or more, otherwise the last compound most two-stoppers used.

Every plan in run r shares the same random race up to its first difference (same seed, nothing re-run), so a
lap-by-lap policy can be replayed exactly (data_pipeline/two_stop_labels.py). Situations are recorded where
the history is shared: first-stint situations (L1 = 0 in situations.csv) at every lap in the run that stops
latest; second-stint situations after a stop on L1, at every lap a second stop is allowed, in the run with
only that stop.

Output: integration/output/two_stop/<Track>_2019_fitted_<driver>_mc<runs>/ with runs.csv (L2 = 0: one stop),
situations.csv and manifest.json. Finished drivers are skipped, so the batch can be restarted.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integration.label_generator import MAX_ATTEMPTS, load, run_seed, setup  # noqa: E402
from integration.run_situations import stop_compounds  # noqa: E402
from integration.write_fitted_pars import parse_pars  # noqa: E402

OUT = ROOT / "integration/output/two_stop"
YEAR = 2019


def second_stop_compounds(pars: dict) -> dict:
    """Compound of each driver's second stop: real last compound after two or more stops, else the most
    common last compound of the drivers who did stop twice."""
    strategies = {d: v["strategy_info"] for d, v in pars["driver_pars"].items()}
    two = {d: s for d, s in strategies.items() if len(s) >= 3}
    common = Counter(s[-1][1] for s in two.values()).most_common(1)
    fallback = common[0][0] if common else None
    return {d: (strategies[d][-1][1] if d in two else fallback) for d in strategies}


def plans(tot: int, first_stop_min: int, min_stint: int) -> dict:
    """{L1: [L2, ...]} of the two-stop plans; every L1 has the one-stop plan as well."""
    out = {}
    for l1 in range(2, tot):
        out[l1] = [l2 for l2 in range(l1 + min_stint, tot - min_stint + 1)] if l1 >= first_stop_min else []
    return out


def simulate(task):
    """All runs of the plans starting with the first stops in `task`. Runs in a worker process."""
    settings, first_stops = task
    setup()
    import numpy as np
    from racesim.src.race import Race
    from integration.fixed_lap_policy import FixedLapVSE, PlannedStopsPolicy
    from integration.situations import features
    args = argparse.Namespace(race=settings["race"], driver=settings["driver"], deterministic=False,
                              events="sampled")
    _, pars, paths, _, _ = load(args)
    driver, c1, c2 = settings["driver"], settings["c1"], settings["c2"]
    tot = pars["race_pars"]["tot_no_laps"]
    team = pars["driver_pars"][driver]["team"]
    track = pars["track_pars"]
    pit_loss = round(track["t_pitdrive_inlap"] + track["t_pitdrive_outlap"] + track["t_pit_tirechange_min"]
                     + pars["car_pars"][team]["t_pit_tirechange_add"], 3)
    tree = plans(tot, settings["first_stop_min"], settings["min_stint"])
    rows, situations, invalid = [], [], 0
    for l1 in first_stops:
        for l2 in [0] + tree[l1]:
            stops = ((l1, c1),) if l2 == 0 else ((l1, c1), (l2, c2))
            policy = PlannedStopsPolicy(stops)
            record = []
            if l2 == 0 and tree[l1]:
                record += [(l1, lap) for lap in range(l1 + settings["min_stint"], tot - settings["min_stint"] + 1)]
            if l2 == 0 and l1 == tot - 1:
                record += [(0, lap) for lap in range(2, tot)]
            for run in range(settings["runs"]):
                for attempt in range(MAX_ATTEMPTS):
                    seed = run_seed(0, run, attempt)
                    random.seed(seed)
                    np.random.seed(seed % 2 ** 32)
                    race = Race(**pars, vse_paths=paths, use_prob_infl=True, create_rand_events=True)
                    # CAPSTONE HOOK: public property injection; upstream source stays unchanged.
                    race.vse = FixedLapVSE(race.vse, driver, policy)
                    race.simulate_race()
                    if race.result_status == 0:
                        break
                    invalid += 1
                else:
                    raise RuntimeError("No valid race in %d attempts: %s run %d" % (MAX_ATTEMPTS, stops, run))
                idx = next(i for i, d in enumerate(race.drivers_list) if d.initials == driver)
                made = [(s[0], s[1]) for s in race.drivers_list[idx].strategy_info[1:]]
                finished = bool(race.bool_driving[-1, idx])
                if made != list(stops) and (finished or made != list(stops)[:len(made)]):
                    raise AssertionError("Planned %s, simulated %s" % (stops, made))
                laps = float(race.progress[idx])
                rows.append({"Race": Path(settings["race"]).name, "Driver": driver, "Run": run, "Seed": seed,
                             "L1": l1, "L2": l2, "C1": c1, "C2": c2 if l2 else "",
                             "Position": int(race.positions[-1, idx]), "SimulatedLaps": laps,
                             "RaceTime_s": round(float(race.racetimes[int(laps), idx]), 3), "Finished": finished})
                for first, lap in record:
                    if race.bool_driving[lap - 1, idx]:
                        situations.append({"Run": run, "Seed": seed, "L1": first,
                                           **features(race, idx, lap, pars["vse_pars"]["param_dry_compounds"],
                                                      pars["vse_pars"]["location_cat"], pit_loss)})
    return rows, situations, invalid


def write_csv(path: Path, rows: list):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", default="Austin")
    parser.add_argument("--drivers", nargs="+", help="default: every driver of the race")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--first-stop-min", type=int, default=5)
    parser.add_argument("--min-stint", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=28)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 8):
        parser.error("Run this script with the .venv-tum Python (3.8)")
    race_file = ROOT / "integration/parameters" / ("pars_%s_%d_fitted.ini" % (args.track, YEAR))
    text = race_file.read_bytes().decode("utf-8")
    pars = parse_pars(text)
    c1s, c2s = stop_compounds(pars), second_stop_compounds(pars)
    tot = pars["race_pars"]["tot_no_laps"]
    tree = plans(tot, args.first_stop_min, args.min_stint)
    OUT.mkdir(parents=True, exist_ok=True)
    for driver in args.drivers or sorted(c1s):
        name = "%s_%d_fitted_%s_mc%d" % (args.track, YEAR, driver, args.runs)
        out = OUT / name
        if (out / "manifest.json").is_file():
            print("%s: already done" % name, flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        settings = {"race": str(race_file), "driver": driver, "c1": c1s[driver], "c2": c2s[driver],
                    "runs": args.runs, "first_stop_min": args.first_stop_min, "min_stint": args.min_stint}
        # balance the work: first stops with many second stops are spread over the workers
        firsts = sorted(tree, key=lambda l1: -len(tree[l1]))
        jobs = max(1, min(args.jobs, len(firsts)))
        tasks = [(settings, firsts[i::jobs]) for i in range(jobs)]
        started = time.perf_counter()
        rows, situations, invalid = [], [], 0
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for part_rows, part_situations, bad in pool.map(simulate, tasks):
                rows += part_rows
                situations += part_situations
                invalid += bad
        rows.sort(key=lambda r: (r["L1"], r["L2"], r["Run"]))
        situations.sort(key=lambda r: (r["L1"], r["Run"], r["Lap"]))
        write_csv(out / "runs.csv", rows)
        write_csv(out / "situations.csv", situations)
        manifest = {"race_file": race_file.name, "race_file_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "driver": driver, "c1": c1s[driver], "c2": c2s[driver], "runs": args.runs,
                    "first_stop_min": args.first_stop_min, "min_stint": args.min_stint,
                    "one_stop_plans": len(tree), "two_stop_plans": sum(len(v) for v in tree.values()),
                    "races": len(rows), "situations": len(situations), "invalid_races_rerun": invalid,
                    "seconds": round(time.perf_counter() - started)}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print("%s: %d races, %d situations, %d re-run (%.0f s)"
              % (name, len(rows), len(situations), invalid, time.perf_counter() - started), flush=True)


if __name__ == "__main__":
    main()
