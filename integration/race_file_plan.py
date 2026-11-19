"""Best one- and two-stop plan of each driver from the race file alone: tyre wear plus pit loss (any Python 3.8+).

    race_file_plan.py --tracks Austin Catalunya Monza Singapore Spielberg

For a track without Monte Carlo runs this is the cheapest plan there is: no simulation, only the driver's
degradation model in the fitted race file (integration/pit_network.degradation) and the time a stop costs
(pit-lane drive, tyre change, the team's extra change time and the cold-tyre penalty of the out-lap). Fuel
is the same for every plan and traffic is ignored, so only the tyre and stop costs differ between plans.

A stop on lap L means L laps on the start tyres (from their start age), then fresh tyres. The stop
compounds are those of the two-stop runs (integration/two_stop_generator.py): C1 for a single or first stop,
C2 for a second; the windows are those runs' windows as well (one stop on laps 2 to laps - 1; two stops
with the first from lap 5 and stints of at least 10 laps).

Used by the plausibility limits of integration/pit_network.NetworkVSE (`limits`, notes/pit_network.md).
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integration.pit_network import degradation  # noqa: E402
from integration.run_situations import stop_compounds  # noqa: E402
from integration.two_stop_generator import plans, second_stop_compounds  # noqa: E402
from integration.write_fitted_pars import parse_pars  # noqa: E402

YEAR = 2019


def stint_cost(tire_pars: dict, compound: str, start_age: float, laps: int) -> float:
    return sum(degradation(tire_pars, compound, start_age + a) for a in range(laps))


def stop_cost(pars: dict, driver: str) -> float:
    track = pars["track_pars"]
    team = pars["driver_pars"][driver]["team"]
    return (track["t_pitdrive_inlap"] + track["t_pitdrive_outlap"] + track["t_pit_tirechange_min"]
            + pars["car_pars"][team]["t_pit_tirechange_add"] + pars["tireset_pars"][driver]["t_add_coldtires"])


def driver_plan(pars: dict, driver: str, c1: str, c2: str, first_stop_min: int = 5, min_stint: int = 10) -> dict:
    """Best one-stop lap and best two-stop laps with their costs (s); TwoStopGain_s > 0: two stops are faster."""
    tire = pars["tireset_pars"][driver]
    start = pars["driver_pars"][driver]["strategy_info"][0]
    c0, age0 = start[1], start[2]
    tot = pars["race_pars"]["tot_no_laps"]
    stop = stop_cost(pars, driver)
    first = [stint_cost(tire, c0, age0, n) for n in range(tot + 1)]          # first[n]: n laps on start tyres
    fresh1 = [stint_cost(tire, c1, 0, n) for n in range(tot + 1)]
    fresh2 = [stint_cost(tire, c2, 0, n) for n in range(tot + 1)] if c2 else None
    one = {l1: first[l1] + stop + fresh1[tot - l1] for l1 in range(2, tot)}
    l1_best = min(one, key=one.get)
    result = {"Driver": driver, "Laps": tot, "C1": c1, "C2": c2 or "", "StopCost_s": round(stop, 3),
              "OneStopLap": l1_best, "OneStopCost_s": round(one[l1_best], 3),
              "TwoStopL1": None, "TwoStopL2": None, "TwoStopCost_s": None, "TwoStopGain_s": None}
    if fresh2 is not None:
        two = {(l1, l2): first[l1] + fresh1[l2 - l1] + fresh2[tot - l2] + 2 * stop
               for l1, l2s in plans(tot, first_stop_min, min_stint).items() for l2 in l2s}
        if two:
            best = min(two, key=two.get)
            result.update({"TwoStopL1": best[0], "TwoStopL2": best[1], "TwoStopCost_s": round(two[best], 3),
                           "TwoStopGain_s": round(one[l1_best] - two[best], 3)})
    return result


def race_plans(pars: dict, **kwargs) -> dict:
    c1s, c2s = stop_compounds(pars), second_stop_compounds(pars)
    return {d: driver_plan(pars, d, c1s[d], c2s[d], **kwargs) for d in sorted(c1s)}


def two_stops_pay(plan: dict) -> bool:
    """The race file says two stops are at least as fast as one."""
    return plan["TwoStopGain_s"] is not None and plan["TwoStopGain_s"] >= 0


def planned_stops(plan: dict) -> list:
    """Stop laps of the race-file plan: [L1] or [L1, L2]."""
    return [plan["TwoStopL1"], plan["TwoStopL2"]] if two_stops_pay(plan) else [plan["OneStopLap"]]


def limits(plan: dict, early: float, late: float, min_stint: int = 10) -> dict:
    """Plausibility limits for NetworkVSE: a stop no earlier than `early` and no later than `late` (fractions of
    the race) around each planned stop; a second stop only where two stops pay. Laps are clipped to the legal
    windows of the two-stop runs."""
    tot = plan["Laps"]
    stops = planned_stops(plan)
    def window(lap, lowest, highest):
        lo = max(lowest, int(math.floor(lap - early * tot)))
        hi = min(highest, int(math.ceil(lap + late * tot)))
        return [lo, max(lo, hi)]
    out = {"first": window(stops[0], 2, tot - 1), "second": None}
    if len(stops) == 2:
        out["second"] = window(stops[1], 2, tot - min_stint)
    return out


def deviations(runs_csv: Path, policy: str) -> dict:
    """{track: [first stop of `policy` minus the race-file plan's first stop, as a fraction of the race]} from the
    races of integration/evaluate_pit_network.py."""
    out, cache = {}, {}
    with open(runs_csv, newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Policy"] != policy or not row["FirstStopLap"]:
                continue
            track = row["Track"]
            if track not in cache:
                cache[track] = race_plans(load_track(track))
            plan = cache[track][row["Driver"]]
            out.setdefault(track, []).append((float(row["FirstStopLap"]) - planned_stops(plan)[0]) / plan["Laps"])
    return out


def percentile(values: list, q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q / 100.0
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def widths(devs: dict, held_out: str, low: float = 5.0, high: float = 95.0) -> dict:
    """Window widths for a held-out track, from the other tracks only: early = -(low percentile), late = high
    percentile of the pooled deviations, both at least 0."""
    pooled = [d for track, values in devs.items() if track != held_out for d in values]
    return {"early": round(max(0.0, -percentile(pooled, low)), 3), "late": round(max(0.0, percentile(pooled, high)), 3),
            "from_tracks": sorted(t for t in devs if t != held_out), "races": len(pooled)}


def load_track(track: str) -> dict:
    path = ROOT / "integration/parameters" / ("pars_%s_%d_fitted.ini" % (track, YEAR))
    return parse_pars(path.read_bytes().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracks", nargs="+", default=["Austin", "Catalunya", "Monza", "Singapore", "Spielberg"])
    parser.add_argument("--widths", type=Path, help="runs.csv of a seen network: write the per-track window widths")
    parser.add_argument("--policy", default="v6", help="--widths: the policy whose first stops set the widths")
    parser.add_argument("--out", type=Path, default=ROOT / "notes/evidence/pit_network/plausibility_widths.json")
    args = parser.parse_args()
    if args.widths:
        devs = deviations(args.widths, args.policy)
        result = {track: widths(devs, track) for track in args.tracks}
        args.out.write_text(json.dumps({"runs": str(args.widths), "policy": args.policy, "widths": result}, indent=1))
        for track, w in result.items():
            print("%-10s early %.3f late %.3f  (from %s, %d races)" % (track, w["early"], w["late"],
                                                                        " ".join(w["from_tracks"]), w["races"]))
        return
    for track in args.tracks:
        rows = list(race_plans(load_track(track)).values())
        gains = [r["TwoStopGain_s"] for r in rows if r["TwoStopGain_s"] is not None]
        print("%-10s laps %d  one-stop lap median %d  two stops faster for %d/%d drivers (gain median %+.1f s)"
              % (track, rows[0]["Laps"], sorted(r["OneStopLap"] for r in rows)[len(rows) // 2],
                 sum(g > 0 for g in gains), len(rows), sorted(gains)[len(gains) // 2]))


if __name__ == "__main__":
    main()
