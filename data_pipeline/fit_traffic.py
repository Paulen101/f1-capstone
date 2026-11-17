"""Measure TUM's traffic values for one race from the workbook (.venv-data).

    fit_traffic.py --year 2019 --round 9

Needs the lap-time fit from fit_parameters.py (free-air pace of every car) and TUM's race file
(DRS effect, DRS window, top speeds). Output: notes/evidence/fitted_parameters/<Location>_<year>_traffic.json.

Every value comes from pairs of cars next to each other in race position at the end of a lap, both
on a green-flag lap and neither pitting in that lap or the one before:

    min_t_dist        closest gap a following car keeps when it does not pass: 5th percentile of the
                      end-of-lap gap over close pairs (gap <= 1.5 s at the start) that did not swap
    t_duel            lap time lost by the car in front while a car is within the DRS window behind it
                      and nobody passes: median of its lap time minus its free-air lap time, with no
                      car within 2 s ahead of it
    t_overtake_loser  extra time lost by the car that is passed, on top of t_duel
    t_gap_overtake    TUM passes when the follower's advantage over the lap reaches this value. The
                      advantage is pace difference + DRS - gap at the start of the lap - TUM's
                      top-speed term. The threshold is chosen so that it predicts as many passes over
                      close pairs as really happened.

A lap's free-air time comes from the lap-time fit, so its error (about 0.4 s) blurs the pass
threshold. Method and limits: notes/fitted_parameters.md.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.fit_parameters import find_workbook  # noqa: E402
from integration.write_fitted_pars import TUM_PARAMETERS, parse_pars  # noqa: E402

CLOSE_GAP_S = 1.5         # pairs closer than this at the start of a lap can fight for position
CLEAR_AHEAD_S = 2.0       # the defending car counts as unhindered if nobody is this close ahead
MIN_DIST_QUANTILE = 0.05
BOOTSTRAP_RUNS = 200


def free_air_time(laps: pd.DataFrame, fit: dict) -> pd.Series:
    """Lap time of each car alone on track, from the lap-time fit. NaN for wet or unknown tyres."""
    levels = laps.Driver.map(fit["driver_level_s"])
    k0 = laps.Compound.map({c: t["k0_s"] for c, t in fit["tyres"].items()})
    k1 = laps.Compound.map({c: t["k1_s_per_lap"] for c, t in fit["tyres"].items()})
    return levels - fit["trend_s_per_lap"] * (laps.LapNumber - 1) + k0 + k1 * (laps.TyreLife - 1)


def pairs(laps: pd.DataFrame, fit: dict, top_speed: dict) -> pd.DataFrame:
    """One row per pair of cars next to each other in race position at the start of a lap."""
    laps = laps.assign(Excess_s=laps.LapTime_s - free_air_time(laps, fit))
    by_key = laps.set_index(["Driver", "LapNumber"])
    rows = []
    for lap in range(2, int(laps.LapNumber.max()) + 1):
        start = laps[laps.LapNumber == lap - 1].dropna(subset=["Position"]).sort_values("Position")
        order = start.Driver.tolist()
        ends = dict(zip(start.Driver, start.LapEndTime_s))
        for ahead_of_front, front, back in zip([None] + order[:-2], order[:-1], order[1:]):
            try:
                now = by_key.loc[[(front, lap), (back, lap)]]
                before = by_key.loc[[(front, lap - 1), (back, lap - 1)]]
            except KeyError:
                continue                                  # a car retired
            if not (now.GreenFlag.all() and before.GreenFlag.all()):
                continue
            if now.IsInLap.any() or now.IsOutLap.any() or before.IsInLap.any():
                continue
            f, b = now.iloc[0], now.iloc[1]
            gap_ahead = ends[front] - ends[ahead_of_front] if ahead_of_front else np.inf
            rows.append({"lap": lap, "front": front, "back": back,
                         "gap_start_s": ends[back] - ends[front],
                         "gap_end_s": b.LapEndTime_s - f.LapEndTime_s,
                         "front_gap_ahead_s": gap_ahead,
                         "pace_advantage_s": f.LapTime_s - f.Excess_s - (b.LapTime_s - b.Excess_s),
                         "front_excess_s": f.Excess_s, "back_excess_s": b.Excess_s,
                         "front_clean": bool(f.CleanLap), "back_clean": bool(b.CleanLap),
                         "top_speed_delta_kph": top_speed[back] - top_speed[front],
                         "passed": bool(b.Position < f.Position)})
    return pd.DataFrame(rows)


def advantage(p: pd.DataFrame, track: dict, race: dict) -> pd.Series:
    """The follower's lead over the front car after the lap, in TUM's terms, minus TUM's top-speed term."""
    drs = np.where(p.gap_start_s <= race["drs_window"], -track["t_drseffect"], 0.0)
    return (p.pace_advantage_s + drs - p.gap_start_s
            - track["t_gap_overtake_vel"] * p.top_speed_delta_kph)


def count_matching_threshold(adv: pd.Series, passes: int) -> float:
    """The threshold that predicts `passes` passes: midway between the passes-th and next largest value."""
    ranked = np.sort(adv.values)[::-1]
    if passes == 0:
        return float(ranked[0] + 0.01)
    if passes >= len(ranked):
        return float(ranked[-1])
    return float((ranked[passes - 1] + ranked[passes]) / 2)


def measure(p: pd.DataFrame, track: dict, race: dict) -> dict:
    close = p[p.gap_start_s <= CLOSE_GAP_S]
    held = close[~close.passed]
    duel = held[(held.gap_start_s <= race["drs_window"]) & held.front_clean
                & (held.front_gap_ahead_s > CLEAR_AHEAD_S)]
    t_duel = float(duel.front_excess_s.median())
    passed = p[p.passed & (p.front_gap_ahead_s > CLEAR_AHEAD_S)]
    return {"min_t_dist": float(held.gap_end_s.quantile(MIN_DIST_QUANTILE)),
            "t_duel": t_duel,
            "t_overtake_loser": float(passed.front_excess_s.median()) - t_duel,
            "t_gap_overtake": count_matching_threshold(advantage(close, track, race), int(close.passed.sum())),
            "counts": {"pairs": int(len(p)), "close_pairs": int(len(close)), "passes": int(p.passed.sum()),
                       "duel_laps": int(len(duel)), "passes_clear_ahead": int(len(passed))}}


def bootstrap(p: pd.DataFrame, track: dict, race: dict, runs: int = BOOTSTRAP_RUNS, seed: int = 0) -> dict:
    """Standard errors from resampling whole laps."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in p.groupby("lap")]
    draws = []
    for _ in range(runs):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append({k: v for k, v in measure(sample, track, race).items() if k != "counts"})
    return {k: float(np.std([d[k] for d in draws], ddof=1)) for k in draws[0]}


def pass_curve(p: pd.DataFrame, track: dict, race: dict) -> list:
    close = p[p.gap_start_s <= CLOSE_GAP_S].assign(adv=lambda d: advantage(d, track, race))
    bins = pd.cut(close.adv, [-np.inf, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, np.inf])
    table = close.groupby(bins, observed=False).passed.agg(["count", "sum"])
    return [{"advantage_s": str(b), "pairs": int(r["count"]), "passes": int(r["sum"])}
            for b, r in table.iterrows()]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--round", type=int, default=9)
    parser.add_argument("--race", default="pars_Spielberg_2019.ini", help="TUM race file name")
    parser.add_argument("--fit", type=Path, default=ROOT / "notes/evidence/fitted_parameters/Spielberg_2019_fit.json")
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    fit = json.loads(args.fit.read_text(encoding="utf-8"))
    pars = parse_pars((TUM_PARAMETERS / args.race).read_bytes().decode("utf-8"))
    track, race = pars["track_pars"], pars["race_pars"]
    top_speed = {d: v["vel_max"] for d, v in pars["driver_pars"].items()}
    laps = pd.read_excel(args.workbook or find_workbook(args.year), sheet_name="Laps")
    laps = laps[(laps.Year == args.year) & (laps.Round == args.round)]
    p = pairs(laps, fit, top_speed)
    result = measure(p, track, race)
    errors = bootstrap(p, track, race)
    tum = {"min_t_dist": race["min_t_dist"], "t_duel": race["t_duel"],
           "t_overtake_loser": race["t_overtake_loser"], "t_gap_overtake": track["t_gap_overtake"]}
    out = {"race": fit["race"], "tum_file": args.race,
           "values": {k: {"measured": round(result[k], 3), "se": round(errors[k], 3), "tum": tum[k]}
                      for k in tum},
           "counts": result["counts"], "pass_rate_by_advantage": pass_curve(p, track, race),
           "settings": {"close_gap_s": CLOSE_GAP_S, "clear_ahead_s": CLEAR_AHEAD_S,
                        "min_dist_quantile": MIN_DIST_QUANTILE, "bootstrap_runs": BOOTSTRAP_RUNS}}
    path = args.out or args.fit.with_name(args.fit.name.replace("_fit.json", "_traffic.json"))
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    p.to_csv(path.with_name(path.stem + "_pairs.csv"), index=False)
    for k, v in out["values"].items():
        print("%-17s measured %6.3f (SE %.3f)   TUM %6.3f" % (k, v["measured"], v["se"], v["tum"]))
    print("Counts: %s" % out["counts"])
    print("Wrote %s" % path)


if __name__ == "__main__":
    main()
