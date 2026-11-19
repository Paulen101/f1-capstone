"""One-stop Monte Carlo runs onto every other legal compound, to choose lap and compound together (.venv-tum).

    compound_runs.py --tracks Austin Monza Singapore Spielberg --runs 50 --jobs 28

The two-stop runs (integration/two_stop_generator.py) stop onto one compound per driver: C1 of
integration/run_situations.stop_compounds. A driver has three dry compounds and must use two, so a single stop
may also go onto the dry compound that is neither the start compound nor C1. For each driver, Monte Carlo run
r (seed r, the same seeds as the two-stop runs) and every stop lap from 2 to the second-to-last lap, this
simulates one stop onto each such compound.

Nothing differs from the stored runs before the stop, so the first-stint situations of the two-stop folders
hold for these races as well and the replay stays exact (data_pipeline/two_stop_labels.py, `extra`).

Output: integration/output/compound_runs/<Track>_2019_fitted_<driver>_mc<runs>/runs.csv (the columns of the
two-stop runs.csv, L2 = 0) and manifest.json. Finished drivers are skipped, so the batch can be restarted.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
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
from integration.two_stop_generator import write_csv  # noqa: E402
from integration.write_fitted_pars import parse_pars  # noqa: E402

OUT = ROOT / "integration/output/compound_runs"
YEAR = 2019


def other_compounds(pars: dict) -> dict:
    """Legal one-stop compounds of each driver besides C1: dry compounds other than the start compound."""
    c1s = stop_compounds(pars)
    dry = pars["vse_pars"]["param_dry_compounds"]
    out = {}
    for driver, values in pars["driver_pars"].items():
        start = values["strategy_info"][0][1]
        out[driver] = [c for c in dry if c not in (start, c1s[driver])]
    return out


def simulate(task):
    """All runs of one driver's stops onto `compound` on the laps in `laps`. Runs in a worker process."""
    settings, driver, compound, laps = task
    setup()
    import numpy as np
    from racesim.src.race import Race
    from integration.fixed_lap_policy import FixedLapVSE, PlannedStopsPolicy
    args = argparse.Namespace(race=settings["race"], driver=driver, deterministic=False, events="sampled")
    _, pars, paths, _, _ = load(args)
    rows, invalid = [], 0
    for l1 in laps:
        policy = PlannedStopsPolicy(((l1, compound),))
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
                raise RuntimeError("No valid race in %d attempts: %s lap %d run %d" % (MAX_ATTEMPTS, driver, l1, run))
            idx = next(i for i, d in enumerate(race.drivers_list) if d.initials == driver)
            made = [(s[0], s[1]) for s in race.drivers_list[idx].strategy_info[1:]]
            finished = bool(race.bool_driving[-1, idx])
            if made != [(l1, compound)] and (finished or made):
                raise AssertionError("Planned stop on %d onto %s, simulated %s" % (l1, compound, made))
            laps_done = float(race.progress[idx])
            rows.append({"Race": Path(settings["race"]).name, "Driver": driver, "Run": run, "Seed": seed,
                         "L1": l1, "L2": 0, "C1": compound, "C2": "",
                         "Position": int(race.positions[-1, idx]), "SimulatedLaps": laps_done,
                         "RaceTime_s": round(float(race.racetimes[int(laps_done), idx]), 3), "Finished": finished})
    return driver, rows, invalid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", nargs="+", default=["Austin", "Monza", "Singapore", "Spielberg"])
    parser.add_argument("--drivers", nargs="+", help="default: every driver of the race")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--jobs", type=int, default=28)
    parser.add_argument("--chunks", type=int, default=4, help="worker tasks per driver and compound")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 8):
        parser.error("Run this script with the .venv-tum Python (3.8)")
    for track in args.tracks:
        race_file = ROOT / "integration/parameters" / ("pars_%s_%d_fitted.ini" % (track, YEAR))
        text = race_file.read_bytes().decode("utf-8")
        pars = parse_pars(text)
        others = other_compounds(pars)
        tot = pars["race_pars"]["tot_no_laps"]
        settings = {"race": str(race_file), "runs": args.runs}
        todo = [d for d in (args.drivers or sorted(others))
                if d in others and not (OUT / ("%s_%d_fitted_%s_mc%d" % (track, YEAR, d, args.runs))
                                        / "manifest.json").is_file()]
        tasks = [(settings, d, c, list(range(2, tot))[i::args.chunks])
                 for d in todo for c in others[d] for i in range(args.chunks)]
        started = time.perf_counter()
        results = {d: ([], 0) for d in todo}
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for driver, rows, bad in pool.map(simulate, tasks):
                results[driver] = (results[driver][0] + rows, results[driver][1] + bad)
        for driver in todo:
            rows, invalid = results[driver]
            rows.sort(key=lambda r: (r["C1"], r["L1"], r["Run"]))
            out = OUT / ("%s_%d_fitted_%s_mc%d" % (track, YEAR, driver, args.runs))
            out.mkdir(parents=True, exist_ok=True)
            write_csv(out / "runs.csv", rows)
            manifest = {"race_file": race_file.name,
                        "race_file_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "driver": driver, "compounds": others[driver], "runs": args.runs, "races": len(rows),
                        "invalid_races_rerun": invalid}
            (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
            print("%s %s: %s, %d races, %d re-run" % (track, driver, " ".join(others[driver]), len(rows), invalid),
                  flush=True)
        print("%s: %d drivers in %.0f s" % (track, len(todo), time.perf_counter() - started), flush=True)


if __name__ == "__main__":
    main()
