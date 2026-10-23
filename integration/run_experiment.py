"""Run baseline, fixed-lap, or realstrategy experiments using .venv-tum."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
TUM = ROOT / "race-simulation"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["baseline", "fixed", "realstrategy"])
    parser.add_argument("--race", default="pars_Spielberg_2019.ini")
    parser.add_argument("--driver", default="VER")
    parser.add_argument("--pit-lap", type=int, default=25)
    parser.add_argument("--compound", default="A3")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 8):
        parser.error("Run this script with .venv-tum/bin/python (Python 3.8)")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools" / "matplotlib"))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(TUM))
    from integration.optional_tensorflow import defer_tensorflow
    defer_tensorflow()
    import numpy as np
    from racesim.src.import_pars import import_pars
    from racesim.src.race import Race
    from integration.fixed_lap_policy import FixedLapPolicy, FixedLapVSE

    random.seed(0)
    np.random.seed(0)
    out = args.out or ROOT / "integration" / "output" / args.mode
    out.mkdir(parents=True, exist_ok=True)
    pars, paths = import_pars(use_print=True, use_vse=args.mode != "baseline",
                             race_pars_file=args.race, mcs_pars_file="pars_mcs.ini")
    if pars["race_pars"]["season"] != 2019:
        parser.error("Week 2 experiments are restricted to the 2019 validation season")
    if any(row[1] in ("I", "W") for d in pars["driver_pars"].values()
           for row in d["strategy_info"]):
        parser.error("Dry races only")
    if args.mode != "baseline":
        pars["vse_pars"]["vse_type"] = dict.fromkeys(pars["race_pars"]["participants"], "realstrategy")
    race = Race(**pars, vse_paths=paths, use_prob_infl=False, create_rand_events=False)
    adapter = None
    if args.mode == "fixed":
        if args.driver not in pars["race_pars"]["participants"]:
            parser.error("Unknown driver")
        driver = next(d for d in race.drivers_list if d.initials == args.driver)
        if driver.strategy_info[0][1] == args.compound:
            parser.error("Choose a fixed compound different from the start compound")
        policy = FixedLapPolicy(args.pit_lap, args.compound)
        policy.decide(1, race.race_pars["tot_no_laps"], True, 0)
        adapter = FixedLapVSE(race.vse, args.driver, policy)
        # CAPSTONE HOOK: public property injection; upstream source stays unchanged.
        race.vse = adapter
    race.simulate_race()
    if race.result_status != 0:
        raise RuntimeError("TUM rejected the simulation: result_status=%s" % race.result_status)
    race.export_results_as_csv(str(out))
    finish = []
    stops = []
    for idx, driver in enumerate(race.drivers_list):
        finish.append({"Driver": driver.initials, "Position": int(race.positions[-1, idx]),
                       "SimulatedLaps": float(race.progress[idx]),
                       "RaceTime_s": float(race.racetimes[int(race.progress[idx]), idx]),
                       "DrivingAtEnd": bool(race.bool_driving[-1, idx])})
        for lap, compound, age, fuel in driver.strategy_info[1:]:
            stops.append({"Driver": driver.initials, "InLap": lap, "OutLap": lap + 1,
                          "Compound": compound, "TyreAge": age, "FuelAdded_kg": fuel,
                          "InLapInfluence": "+".join(driver.lap_influences.get(lap, [])),
                          "OutLapInfluence": "+".join(driver.lap_influences.get(lap + 1, []))})
    write_csv(out / "classification.csv", sorted(finish, key=lambda row: row["Position"]))
    write_csv(out / "pitstops.csv", stops)
    if adapter:
        write_csv(out / "policy_decisions.csv", adapter.decisions)
        actual = [s for s in stops if s["Driver"] == args.driver]
        if len(actual) != 1 or actual[0]["InLap"] != args.pit_lap or actual[0]["Compound"] != args.compound:
            raise AssertionError("Actual stop did not match the policy")
        if "pitinlap" not in actual[0]["InLapInfluence"] or "pitoutlap" not in actual[0]["OutLapInfluence"]:
            raise AssertionError("Pit entry/exit were not executed by the simulator")
        print("Verified actual simulator pit stop:", actual[0])
    source = TUM / "racesim" / "input" / "parameters" / args.race
    manifest = {"mode": args.mode, "python": sys.version, "race_file": args.race,
                "race_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "result_status": int(race.result_status), "laps": race.cur_lap,
                "drivers": race.no_drivers, "use_prob_infl": False, "create_rand_events": False,
                "vse": "disabled (original strategy_info)" if args.mode == "baseline" else "realstrategy",
                "tensorflow_import": "deferred; neural-network VSEs unused",
                "fixed_policy": {"driver": args.driver, "in_lap": args.pit_lap, "compound": args.compound} if adapter else None}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Completed %s: %s laps, %s drivers; CSVs: %s" % (args.mode, race.cur_lap, race.no_drivers, out))


if __name__ == "__main__":
    main()
