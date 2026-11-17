"""Write our fitted values into a copy of a TUM race parameter file (any Python 3.8+, no packages).

    write_fitted_pars.py --fit notes/evidence/fitted_parameters/Spielberg_2019_fit.json

Input: the JSON written by data_pipeline/fit_parameters.py and, for the traffic group, the JSON
written by data_pipeline/fit_traffic.py (default: <race>_traffic.json next to the fit). Output:
integration/parameters/<race>_fitted.ini and <race>_fitted.manifest.json. The TUM file is read,
never changed. Every value not listed in the manifest is identical to the TUM file.

Groups (--groups, default all):
    pit     pit-lane drive times on the in-lap and out-lap
    tyres   k_0 and k_1_lin (or tire_deg_model "quad", k_1_quad, k_2_quad) for every driver, fuel
            sensitivity, and the lap-time level
    pace    car and driver pace offsets (needs tyres)
    traffic min_t_dist, t_duel, t_overtake_loser and t_gap_overtake, measured from car pairs
Method and limits: notes/fitted_parameters.md.
"""
import argparse
import configparser
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
TUM_PARAMETERS = ROOT / "race-simulation" / "racesim" / "input" / "parameters"
OUT_DIR = ROOT / "integration" / "parameters"
GROUPS = ("pit", "tyres", "pace", "traffic")
RACE_TRAFFIC = ("min_t_dist", "t_duel", "t_overtake_loser")
TRACK_TRAFFIC = ("t_gap_overtake",)
MIN_STOPS = 5
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
STRING = re.compile(r'"[^"]*"')


def parse_pars(text: str) -> dict:
    """Parse a TUM race file the way import_pars() does: one JSON value per section."""
    parser = configparser.ConfigParser()
    parser.read_string(text)
    return {section.lower(): json.loads(parser.get(section, section.lower()))
            for section in parser.sections()}


def set_number(text: str, path: tuple, value) -> str:
    """Change the number (or, for a str value, the string) at `path` (section, key, key, ...) and
    leave every other character alone."""
    position = text.index("\n%s = " % path[0])
    for key in path[1:]:
        match = re.compile(r'"%s"\s*:\s*' % re.escape(key)).search(text, position)
        if not match:
            raise KeyError("No %s in %s" % (key, ".".join(path)))
        position = match.end()
    if isinstance(value, str):
        match = STRING.match(text, position)
        if not match:
            raise ValueError("%s is not a string" % ".".join(path))
        return text[:position] + json.dumps(value) + text[match.end():]
    match = NUMBER.match(text, position)
    if not match:
        raise ValueError("%s is not a number" % ".".join(path))
    return text[:position] + repr(value) + text[match.end():]


def changed_values(before, after, path=""):
    """List every leaf value that differs, as (path, before, after)."""
    if isinstance(before, dict) and isinstance(after, dict) and before.keys() == after.keys():
        return [item for key in before
                for item in changed_values(before[key], after[key], "%s.%s" % (path, key) if path else key)]
    return [] if before == after else [(path, before, after)]


def pit_values(fit: dict, pars: dict) -> dict:
    """Pit-lane drive times from the measured in-lap and out-lap losses.

    TUM adds t_pitdrive_inlap to the in-lap, and t_pitdrive_outlap + standstill (minimum + team
    addition) + cold tyres to the out-lap. Standstill and cold-tyre values have no source in our
    data and are kept, so they are taken off each measured out-lap loss.
    """
    stops = [stop for stop in fit["pit"]["stops"] if stop["used"]]
    if len(stops) < MIN_STOPS:
        raise ValueError("Need at least %d green-flag stops, got %d" % (MIN_STOPS, len(stops)))
    track = pars["track_pars"]
    outlap = []
    for stop in stops:
        team = pars["driver_pars"][stop["driver"]]["team"]
        standstill = track["t_pit_tirechange_min"] + pars["car_pars"][team]["t_pit_tirechange_add"]
        outlap.append(stop["out_lap_loss_s"] - standstill
                      - pars["tireset_pars"][stop["driver"]]["t_add_coldtires"])
    values = {("track_pars", "t_pitdrive_inlap"): round(statistics.median(s["in_lap_loss_s"] for s in stops), 3),
              ("track_pars", "t_pitdrive_outlap"): round(statistics.median(outlap), 3)}
    if min(values.values()) <= 0:
        raise ValueError("Pit drive times must be positive: %s" % values)
    return values


def compound_codes(fit: dict, pars: dict) -> dict:
    """Match our compound names to TUM's codes by comparing every driver's stint sequence."""
    codes = {}
    for driver, names in fit["stints"].items():
        tum = [stint[1] for stint in pars["driver_pars"][driver]["strategy_info"]]
        if len(tum) != len(names):
            continue                    # stint counts differ, e.g. a stop TUM does not record
        for name, code in zip(names, tum):
            if codes.setdefault(name, code) != code:
                raise ValueError("%s is both %s and %s in the TUM file" % (name, codes[name], code))
    if len(set(codes.values())) != len(codes) or set(fit["tyres"]) - set(codes):
        raise ValueError("Cannot match compounds to TUM codes: %s" % codes)
    return codes


def driver_levels(fit: dict, pars: dict) -> dict:
    """Fitted free-air level per driver. A driver with too few free-air laps takes the team-mate's
    level plus TUM's own gap between the two. If the team-mate is short of free-air laps too, the
    driver takes TUM's pace (t_car + t_driver) shifted by the field's median difference between our
    levels and TUM's pace."""
    levels = dict(fit["driver_level_s"])
    drivers = pars["driver_pars"]
    short = set(fit["drivers_below_min_laps"])
    tum_pace = {d: pars["car_pars"][drivers[d]["team"]]["t_car"] + drivers[d]["t_driver"] for d in drivers}
    if len(short) >= len(drivers):
        raise ValueError("No driver has enough free-air laps")
    shift = statistics.median(levels[d] - tum_pace[d] for d in drivers if d not in short)
    for driver in sorted(short):
        mates = [d for d in drivers if d != driver and drivers[d]["team"] == drivers[driver]["team"]
                 and d not in short]
        if mates:
            levels[driver] = levels[mates[0]] + drivers[driver]["t_driver"] - drivers[mates[0]]["t_driver"]
        else:
            levels[driver] = tum_pace[driver] + shift
    if set(levels) != set(drivers):
        raise ValueError("Drivers differ between the fit and the TUM file")
    return levels


def tyre_values(fit: dict, pars: dict, with_pace: bool) -> dict:
    """Tyre model, fuel sensitivity and lap-time level. These depend on each other and go together.

    Our model:  level - trend * (lap - 1) + k0 + k1 * age (+ k2 * age**2)
    TUM:        t_q + t_gap_racepace + t_car + t_driver + k_0 + k_1_lin * age + fuel mass * t_lap_sens_mass
    TUM has no track evolution, so the whole trend becomes fuel sensitivity. The fuel mass falls by
    b_fuel_perlap each lap from m_fuel, which gives level = t_q + t_gap_racepace + t_car + t_driver
    + m_fuel * t_lap_sens_mass.
    With quadratic wear (fit["wear_model"] == "quad") every driver's tire_deg_model becomes "quad" and
    k1, k2 go to TUM's k_1_quad, k_2_quad; k_1_lin is then unused by TUM and left as it was.
    """
    values = {}
    codes = compound_codes(fit, pars)
    quad = fit.get("wear_model", "lin") == "quad"
    for driver in pars["tireset_pars"]:
        if quad:
            values[("tireset_pars", driver, "tire_deg_model")] = "quad"
        for name, tyre in fit["tyres"].items():
            values[("tireset_pars", driver, codes[name], "k_0")] = round(tyre["k0_s"], 4)
            if quad:
                values[("tireset_pars", driver, codes[name], "k_1_quad")] = round(tyre["k1_s_per_lap"], 4)
                values[("tireset_pars", driver, codes[name], "k_2_quad")] = round(tyre["k2_s_per_lap2"], 6)
            else:
                values[("tireset_pars", driver, codes[name], "k_1_lin")] = round(tyre["k1_s_per_lap"], 4)
    cars = list(pars["car_pars"].values())
    if any((car["m_fuel"], car["b_fuel_perlap"]) != (cars[0]["m_fuel"], cars[0]["b_fuel_perlap"])
           for car in cars):
        raise ValueError("Cars differ in fuel load or consumption; one fuel sensitivity cannot fit all")
    burn = cars[0]["b_fuel_perlap"]
    laps = pars["race_pars"]["tot_no_laps"]
    if cars[0]["m_fuel"] < burn * laps:
        # TUM's Monza 2019 file: 110 kg at 2.076 kg/lap lasts 52.99 of 53 laps, so a race without a
        # safety car runs dry on the last lap and TUM stops it. Real cars finished: burn the most that
        # still covers the distance.
        burn = math.floor(cars[0]["m_fuel"] / laps * 1000) / 1000
        for team in pars["car_pars"]:
            values[("car_pars", team, "b_fuel_perlap")] = burn
    sensitivity = round(fit["trend_s_per_lap"] / burn, 5)
    values[("track_pars", "t_lap_sens_mass")] = sensitivity
    levels = driver_levels(fit, pars)
    base = pars["track_pars"]["t_q"] + cars[0]["m_fuel"] * sensitivity
    drivers = pars["driver_pars"]
    if with_pace:
        best = min(levels.values())
        values[("track_pars", "t_gap_racepace")] = round(best - base, 3)
        for team in pars["car_pars"]:
            team_best = min(levels[d] for d in drivers if drivers[d]["team"] == team)
            values[("car_pars", team, "t_car")] = round(team_best - best, 3)
            for driver in drivers:
                if drivers[driver]["team"] == team:
                    values[("driver_pars", driver, "t_driver")] = round(levels[driver] - team_best, 3)
    else:
        # Keep TUM's pace offsets and move the whole field so its mean level matches the fit.
        offsets = [levels[d] - pars["car_pars"][drivers[d]["team"]]["t_car"] - drivers[d]["t_driver"]
                   for d in drivers]
        values[("track_pars", "t_gap_racepace")] = round(statistics.mean(offsets) - base, 3)
    return values


def traffic_values(traffic: dict) -> dict:
    """Measured traffic values. TUM accepts t_gap_overtake only above 0.

    A duel cannot make a car faster, so a negative t_duel (noise around zero: the measurement's
    standard error is 0.06-0.2 s) is written as 0.
    """
    measured = {key: value["measured"] for key, value in traffic["values"].items()}
    measured["t_duel"] = max(measured["t_duel"], 0.0)
    if not 0.0 < measured["t_gap_overtake"] < 5.0 or min(measured.values()) < 0:
        raise ValueError("Traffic values outside TUM's range: %s" % measured)
    values = {("race_pars", key): round(measured[key], 3) for key in RACE_TRAFFIC}
    values.update({("track_pars", key): round(measured[key], 3) for key in TRACK_TRAFFIC})
    return values


def build(stock_text: str, fit: dict, groups: tuple = GROUPS, traffic: dict = None) -> tuple:
    """Return the fitted file text and the list of changed values."""
    if "pace" in groups and "tyres" not in groups:
        raise ValueError("The pace group needs the tyres group: both set the lap-time level")
    stock = parse_pars(stock_text)
    values = {}
    if "pit" in groups:
        values.update(pit_values(fit, stock))
    if "tyres" in groups:
        values.update(tyre_values(fit, stock, "pace" in groups))
    if "traffic" in groups:
        if traffic is None:
            raise ValueError("The traffic group needs the output of fit_traffic.py")
        values.update(traffic_values(traffic))
    values = {path: value if isinstance(value, str) else value + 0.0   # -0.0 becomes 0.0
              for path, value in values.items()}
    text = stock_text
    for path, value in values.items():
        text = set_number(text, path, value)
    after = parse_pars(text)
    changes = changed_values(stock, after)
    for path, value in values.items():
        node = after
        for key in path:
            node = node[key]
        if node != value:
            raise AssertionError("%s was not written" % ".".join(path))
    if not {path for path, _, _ in changes} <= {".".join(path) for path in values}:
        raise AssertionError("Unexpected changes: %s" % changes)
    return text, [{"parameter": path, "tum": old, "fitted": new} for path, old, new in changes]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race", help="TUM race file name (default: the one named in the fit)")
    parser.add_argument("--fit", type=Path, required=True, help="JSON from data_pipeline/fit_parameters.py")
    parser.add_argument("--traffic", type=Path, help="JSON from data_pipeline/fit_traffic.py "
                                                      "(default: <race>_traffic.json next to --fit)")
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--out", type=Path, help="default: integration/parameters/<race>_fitted.ini")
    args = parser.parse_args()
    fit = json.loads(args.fit.read_text(encoding="utf-8"))
    args.race = args.race or fit.get("tum_file", "pars_Spielberg_2019.ini")
    source = TUM_PARAMETERS / args.race
    traffic_file = args.traffic or args.fit.with_name(args.fit.name.replace("_fit.json", "_traffic.json"))
    traffic = json.loads(traffic_file.read_text(encoding="utf-8")) if "traffic" in args.groups else None
    text, changes = build(source.read_bytes().decode("utf-8"), fit, tuple(args.groups), traffic)
    out = args.out or OUT_DIR / (source.stem + "_fitted.ini")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(text.encode("utf-8"))
    manifest = {"race_file": out.name, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "tum_file": args.race, "tum_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "fit_file": args.fit.as_posix(),
                "fit_sha256": hashlib.sha256(args.fit.read_bytes()).hexdigest(),
                "traffic_file": traffic_file.as_posix() if traffic else None,
                "traffic_sha256": hashlib.sha256(traffic_file.read_bytes()).hexdigest() if traffic else None,
                "groups": args.groups, "changed": changes}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Wrote %s: %d values changed (%s)" % (out, len(changes), ", ".join(args.groups)))


if __name__ == "__main__":
    main()
