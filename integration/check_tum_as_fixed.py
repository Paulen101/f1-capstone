"""Is TUM's VSE result just its choice of stop lap and compound? (.venv-tum). Notes: pit_network.md, round 2.

    check_tum_as_fixed.py [--track Spielberg] [--races 30]

Takes one-stop races of the `tum` policy from integration/output/evaluation/pit_network_v1_mc100/runs.csv and
re-simulates each with the same seed and a FixedLapPolicy on TUM's stop lap and compound. Same finishing
position and race time means TUM's advantage is entirely which (lap, compound) it picks in each race, so a
policy can learn it from stored runs only if the runs hold every compound, not just one per driver.
"""
import argparse
import csv
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integration.label_generator import load, setup  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", default="Spielberg")
    parser.add_argument("--races", type=int, default=30)
    parser.add_argument("--runs", type=Path,
                        default=ROOT / "integration/output/evaluation/pit_network_v1_mc100/runs.csv")
    args = parser.parse_args()
    setup()
    import numpy as np
    from racesim.src.race import Race
    from integration.fixed_lap_policy import FixedLapPolicy, FixedLapVSE
    with open(args.runs, newline="") as stream:
        rows = [r for r in csv.DictReader(stream)
                if r["Track"] == args.track and r["Policy"] == "tum" and r["Stops"] == "1"][:args.races]
    same = 0
    for r in rows:
        driver, (lap, compound), seed = r["Driver"], r["StopPlan"].split(":"), int(r["Seed"])
        ns = argparse.Namespace(race=str(ROOT / "integration/parameters" / ("pars_%s_2019_fitted.ini" % args.track)),
                                driver=driver, deterministic=False, events="sampled")
        _, pars, paths, _, _ = load(ns)
        random.seed(seed)
        np.random.seed(seed % 2 ** 32)
        race = Race(**pars, vse_paths=paths, use_prob_infl=True, create_rand_events=True)
        race.vse = FixedLapVSE(race.vse, driver, FixedLapPolicy(int(lap), compound))
        race.simulate_race()
        idx = next(i for i, d in enumerate(race.drivers_list) if d.initials == driver)
        laps = float(race.progress[idx])
        position, time = int(race.positions[-1, idx]), round(float(race.racetimes[int(laps), idx]), 3)
        match = position == int(r["Position"]) and math.isclose(time, float(r["RaceTime_s"]), abs_tol=1e-3)
        same += match
        print("%s seed %d stop %s %s: TUM P%s %s s, fixed P%d %s s %s"
              % (driver, seed, lap, compound, r["Position"], r["RaceTime_s"], position, time,
                 "same" if match else "DIFFERENT"))
    print("%d of %d races identical" % (same, len(rows)))


if __name__ == "__main__":
    main()
