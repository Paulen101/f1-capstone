"""Build a complete field mapping and value inventory from one TUM .ini file."""
import argparse
import configparser
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each line: field | meaning/units | workbook source | estimation limitation.
SPECS = """
race_pars.season|Season used to select Monte Carlo parameters|Races.Year|2019 validation only; production eras are 2022–2025 and 2026 separately.
race_pars.tot_no_laps|Scheduled race distance, laps|TrackSummary.RaceLaps; Results.Laps; Laps.LapNumber|Observed max laps needs checking for shortened races.
race_pars.min_t_dist|Minimum following time gap, s|NO SOURCE|Lap-end positions alone cannot identify a following-gap floor.
race_pars.min_t_dist_sc|Minimum following gap behind SC, s|NO SOURCE|No continuous inter-car gap observations.
race_pars.t_duel|Time lost by battling drivers per lap, s|NO SOURCE|Cannot separate from traffic and intrinsic pace with these columns.
race_pars.t_overtake_loser|Time lost by overtaken driver, s|NO SOURCE|Lap-end position changes do not identify maneuver cost.
race_pars.use_drs|Enable DRS|NO SOURCE|Series/rules configuration; DRS activation is absent from the workbook.
race_pars.drs_window|DRS eligibility gap, s|NO SOURCE|Rules input, not a fitted race observation.
race_pars.drs_allow_lap|First lap allowing DRS|NO SOURCE|Rules input; must be era-specific.
race_pars.drs_sc_delay|Laps before DRS returns after SC|NO SOURCE|Rules input; must be era-specific.
race_pars.participants|Driver identifiers in the race|Results.Abbreviation; Laps.Driver|Check starters versus DNS, missing lap data, and identifier changes.
track_pars.name|TUM track identifier|Races.Location; Races.Event|Requires an explicit name/track-layout lookup.
track_pars.t_q|Best qualifying lap time, s|NO SOURCE|This workbook contains races only, not qualifying.
track_pars.t_gap_racepace|Offset from qualifying reference to best race pace, s|Laps.LapTime_s, CleanLap, TyreLife, Compound, Driver|Conditional fit only; qualifying reference and fuel model are missing.
track_pars.t_lap_sens_mass|Lap-time sensitivity to fuel mass, s/kg|NO SOURCE|No measured fuel mass; cannot separate fuel burn and tire wear from lap times alone.
track_pars.t_pit_tirechange_min|Minimum stationary tire-change time, s|NO SOURCE|PitLaneTime_s includes transit; it cannot isolate stationary service time.
track_pars.t_pitdrive_inlap|Extra pit-entry transit loss in green conditions, s|PitStops.InLapTime_s; Laps.LapTime_s, CleanLap; PitStops.TrackStatusInLap|Only combined in+out loss is robust; excludes standstill and cold-tire effects in TUM.
track_pars.t_pitdrive_outlap|Extra pit-exit transit loss in green conditions, s|PitStops.OutLapTime_s; Laps.LapTime_s, CleanLap|Same combined-loss limitation; fuel, compound, tire warmup, and traffic confound the split.
track_pars.t_pitdrive_inlap_fcy|Pit-entry transit loss under VSC, s|PitStops.InLapTime_s, UnderSCorVSC; Laps.VSC, SC, CleanLap|Use both in/out-lap status and matched VSC references; sparse observations.
track_pars.t_pitdrive_outlap_fcy|Pit-exit transit loss under VSC, s|PitStops.OutLapTime_s; Laps.VSC, SC, CleanLap|Use both laps; UnderSCorVSC alone does not distinguish VSC from SC.
track_pars.t_pitdrive_inlap_sc|Pit-entry transit loss under SC, s|PitStops.InLapTime_s, UnderSCorVSC; Laps.SC, VSC|SC queue and deployment phase matter; only conditional estimates.
track_pars.t_pitdrive_outlap_sc|Pit-exit transit loss under SC, s|PitStops.OutLapTime_s; Laps.SC, VSC|SC queue and deployment phase matter; do not mix with green stops.
track_pars.pits_aft_finishline|Whether pit box is after timing line|NO SOURCE|Circuit geometry, not reliably encoded by lap-level pit timestamps.
track_pars.t_loss_pergridpos|Start loss per grid position, s/position|Results.GridPosition; Laps.LapNumber, LapTime_s, Driver|Fit lap-1 residuals across drivers/races; start performance and traffic confound this.
track_pars.t_loss_firstlap|Standing-start first-lap loss, s|Laps.LapNumber, LapTime_s; Results.GridPosition|Separate from grid position, fuel, tire, and traffic effects.
track_pars.t_gap_overtake|Required relative pace advantage to pass, s|NO SOURCE|No pass-event or continuous gap data; cannot identify from position alone.
track_pars.t_gap_overtake_vel|Modify pass threshold using speed difference, s/(km/h)|NO SOURCE|Qualifying speed-trap measurements absent.
track_pars.t_drseffect|Lap-time benefit of DRS, s (negative)|NO SOURCE|DRS use is not included in this workbook.
track_pars.mult_t_lap_sc|SC lap-time multiplier|Laps.LapTime_s, SC, VSC, LapNumber, IsInLap, IsOutLap|Use fully neutralized laps and matched green baseline; SC boolean includes partial laps.
track_pars.mult_t_lap_fcy|VSC lap-time multiplier|Laps.LapTime_s, VSC, SC, LapNumber, IsInLap, IsOutLap|Separate VSC from SC and partial transitions; not a CleanLap-only pace fit.
car_pars.*.drivetype|Combustion or electric vehicle type|NO SOURCE|Configuration; combustion for this validation file.
car_pars.*.manufacturer|Team/manufacturer label|Results.TeamName; Laps.Team|Normalize team names to TUM identifiers; this file uses team names.
car_pars.*.t_car|Car/team intrinsic pace offset, s/lap|Laps.Team, Driver, LapTime_s, CleanLap, Compound, TyreLife|Fuel-correct first; car and driver offsets are not separately identifiable without constraints.
car_pars.*.m_fuel|Initial fuel mass, kg|NO SOURCE|Fuel load is not public in the workbook; regulatory maximum is not measured actual load.
car_pars.*.b_fuel_perlap|Fuel burned per green lap, kg/lap|NO SOURCE|Requires an external assumption or fuel model.
car_pars.*.energy|Initial electric energy, kWh|NO SOURCE|Unused/null in this combustion-car file.
car_pars.*.energy_perlap|Energy consumption, kWh/lap|NO SOURCE|Unused/null in this combustion-car file.
car_pars.*.mult_consumption_sc|SC fuel/energy use multiplier|NO SOURCE|No fuel/energy telemetry in workbook.
car_pars.*.mult_consumption_fcy|VSC fuel/energy use multiplier|NO SOURCE|No fuel/energy telemetry in workbook.
car_pars.*.auto_consumption_adjust|Adjust fuel consumption to remaining distance|NO SOURCE|Simulator configuration, not an observed parameter.
car_pars.*.t_pit_tirechange_add|Team stationary service excess above minimum, s|NO SOURCE|PitLaneTime_s includes transit and does not identify stationary team performance.
car_pars.*.t_pit_refuel_perkg|Stationary refueling time, s/kg|NO SOURCE|Null in this no-refueling F1 configuration.
car_pars.*.t_pit_charge_perkwh|Stationary charging time, s/kWh|NO SOURCE|Null in this F1 configuration.
car_pars.*.color|Team display color|NO SOURCE|Cosmetic configuration; not present in exported results.
tireset_pars.*.tire_deg_model|Tire degradation functional form|Laps.LapTime_s, TyreLife, Compound, CleanLap, Stint|Model selection, not direct observation; use fuel-corrected clean laps and holdout checks.
tireset_pars.*.mult_tiredeg_sc|Effective wear rate during SC|NO SOURCE|Neutralized-lap time is not a direct measurement of tire wear; retain explicit assumption.
tireset_pars.*.mult_tiredeg_fcy|Effective wear rate during VSC|NO SOURCE|No direct wear observation; retain explicit assumption.
tireset_pars.*.t_add_coldtires|Freshly mounted tire warmup penalty, s|Laps.LapTime_s, IsOutLap, Compound, TyreLife; PitStops.OutLapTime_s|Confounded with exit transit and fuel; conditional residual estimate only.
tireset_pars.*.*.k_0|Compound baseline pace offset, s|Laps.LapTime_s, Compound, TyreLife, CleanLap, Stint, Driver|Fuel-correct, constrain a reference compound, account for stint and driver offsets.
tireset_pars.*.*.k_1_lin|Linear degradation slope, s/tire-lap|Laps.LapTime_s, TyreLife, Compound, CleanLap, Stint, Driver|Fuel-correct first; traffic remains a confounder even for clean laps.
tireset_pars.*.*.k_1_quad|Linear term in quadratic degradation, s/tire-lap|Laps.LapTime_s, TyreLife, Compound, CleanLap, Stint, Driver|Fit jointly with k_2_quad; inactive when tire_deg_model is lin.
tireset_pars.*.*.k_2_quad|Quadratic degradation term, s/tire-lap²|Laps.LapTime_s, TyreLife, Compound, CleanLap, Stint, Driver|Needs a broad age range; inactive when tire_deg_model is lin.
driver_pars.*.carno|Driver race number|Laps.DriverNumber|Convert to integer and match abbreviations.
driver_pars.*.name|Full driver name|Results.FullName|Normalize spelling and accents.
driver_pars.*.initials|Driver abbreviation|Results.Abbreviation; Laps.Driver|Use exact identifier matching.
driver_pars.*.team|Team identifier|Results.TeamName; Laps.Team|Normalize to car_pars keys.
driver_pars.*.t_driver|Driver intrinsic pace offset, s/lap|Laps.Driver, Team, LapTime_s, Compound, TyreLife, CleanLap|Cannot separate from t_car without team/driver reference constraints; fuel-correct first.
driver_pars.*.strategy_info|Start and stop tuples [inlap, compound, initial tire age, fuel/energy added]|Stints.StartLap, Compound, TyreLifeAtStart; PitStops.InLap, CompoundAfter|Stop lap is InLap, not next stint StartLap. Start tire age ≈ TyreLifeAtStart−1; no-refuel value fixed at 0.
driver_pars.*.p_grid|Starting grid position|Results.GridPosition|Handle pit-lane starts and penalties explicitly.
driver_pars.*.t_teamorder|Team-order bias on pass threshold, s|NO SOURCE|Not identifiable from these exports.
driver_pars.*.vel_max|Qualifying maximum speed, km/h|NO SOURCE|Qualifying/speed-trap data not exported.
monte_carlo_pars.min_dist_sc|Minimum separation around SC phases, laps|NO SOURCE|Numerical constraint for lap-wise simulator, not a measured race feature.
monte_carlo_pars.min_dist_vsc|Minimum separation around VSC phases, laps|NO SOURCE|Numerical constraint for lap-wise simulator.
monte_carlo_pars.ref_driver|Assumed winner used in reference pre-simulation|Results.Abbreviation, Position|Can use actual winner for retrospective validation only; future use must avoid outcome leakage.
event_pars.fcy_data.phases|[start, end, SC/VSC, SC delay, SC duration] phase list|TrackStatus.SessionTime_s, Status; Laps.LapStartTime_s, LapNumber, SC, VSC|Align session time to race start; exact SC release/delay is not fully identifiable. Red flags unsupported by this phase schema.
event_pars.fcy_data.domain|Phase boundaries in progress or race time|NO SOURCE|Representation choice; prefer time after explicit timestamp alignment.
event_pars.retire_data.retirements|[driver, retirement time/progress] list|Results.Status, Laps, Abbreviation; Laps.LapNumber, LapStartTime_s, LapTime_s|Last completed lap gives only approximate retirement boundary; distinguish lapped finishers.
event_pars.retire_data.domain|Retirement boundary coordinate system|NO SOURCE|Representation choice; align to FCY time domain.
vse_pars.available_compounds|Compounds available to strategy engineer|Laps.Compound; Stints.Compound|Observed compounds do not prove availability; external nomination list and explicit legacy-code mapping required.
vse_pars.param_dry_compounds|Dry compounds with fitted parameters|Laps.Compound, CleanLap; Stints.Compound|Parameterize only supported dry compounds; never include I/W in this project policy.
vse_pars.location_cat|Pirelli tire-stress category (1 high, 3 low)|NO SOURCE|Requires an external track/nomination source.
vse_pars.base_strategy.*|Free-track optimized strategy tuples|NO SOURCE|Simulator optimization output, not a measured workbook parameter; optimization is not part of this timing-only dummy policy.
vse_pars.real_strategy.*|Recorded start/stop strategy tuples|Stints.StartLap, Compound, TyreLifeAtStart; PitStops.InLap, CompoundAfter|Historical replay input only. Exact initial tire age and compound codes need checking; fuel addition fixed at 0.
vse_pars.vse_type.*|Which VSE controls a driver|NO SOURCE|Experiment configuration; realstrategy background and one fixed-lap adapter.
"""


def fields(value, path):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from fields(child, path + (key,))
    else:
        # Lists are one parameter; their ordered element semantics are documented.
        yield path, value


def schema(path):
    parts = list(path)
    if parts[0] in ("car_pars", "driver_pars", "tireset_pars"):
        parts[1] = "*"
    if parts[0] == "tireset_pars" and len(parts) == 4:
        parts[2] = "*"
    if parts[:2] in (["vse_pars", "base_strategy"], ["vse_pars", "real_strategy"], ["vse_pars", "vse_type"]):
        parts[2] = "*"
    return ".".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ini", type=Path, default=ROOT / "race-simulation/racesim/input/parameters/pars_Spielberg_2019.ini")
    args = parser.parse_args()
    specs = dict((line.split("|", 1)[0], line.split("|")[1:])
                 for line in SPECS.strip().splitlines())
    config = configparser.ConfigParser()
    if not config.read(args.ini):
        raise FileNotFoundError(args.ini)
    rows, used = [], {}
    for section in config.sections():
        for parameter, raw in config[section].items():
            for path, value in fields(json.loads(raw), (parameter,)):
                key = schema(path)
                if key not in specs:
                    raise ValueError("Unmapped field: " + ".".join(path))
                meaning, source, limitation = specs[key]
                rows.append({"Parameter": ".".join(path), "Schema": key,
                             "Value": json.dumps(value), "Meaning": meaning,
                             "WorkbookColumns": source, "Limitations": limitation})
                used[key] = used.get(key, 0) + 1
    out = ROOT / "notes"
    out.mkdir(exist_ok=True)
    with (out / "parameter_inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Parameter mapping: Spielberg 2019", "",
             "Source: `race-simulation/racesim/input/parameters/pars_Spielberg_2019.ini`. "
             "All **%d concrete parameters** map to **%d field patterns** below. "
             "`*` expands across every team, driver, or compound key. Lists are shown as one parameter with ordered tuple semantics. "
             "The [full value inventory](parameter_inventory.csv) contains every concrete path and original value. "
             "The generator fails if any field lacks a mapping." % (len(rows), len(used)), "",
             "Workbook sources refer to the standard dry-race export (`F1_2019_dry_races.xlsx` for validation; "
             "2022–2025 and 2026 separate for production). These are proposed estimators, not claims that a parameter "
             "has already been fitted. **NO SOURCE** means the current workbook cannot supply the parameter. "
             "Do not fill gaps with invented measurements.", "",
             "| Parameter pattern | Count | Meaning / units | Workbook columns | Limitations / assumptions |",
             "|---|---:|---|---|---|"]
    for key in used:
        meaning, source, limitation = specs[key]
        lines.append("| `%s` | %d | %s | %s | %s |" % (key, used[key], meaning, source, limitation))
    lines += ["", "## Estimation rules", "",
              "- Fit intrinsic pace and tire degradation only on `Laps.CleanLap == True`. Correct fuel first using "
              "an explicitly sourced or assumed fuel model; stint summary lap times are uncorrected. CleanLap does not remove traffic or engine-mode changes.",
              "- The linear tire formula is `k_0 + k_1_lin * age`; the quadratic formula is "
              "`k_0 + k_1_quad * age + k_2_quad * age**2` (`helper_funcs/src/calc_tire_degradation.py`). "
              "This .ini selects `lin` for all drivers; stored quadratic coefficients are inactive.",
              "- Green pit loss is approximately `(InLapTime_s + OutLapTime_s) - 2 * matched_reference_clean_lap`. "
              "Match driver, nearby race phase, fuel and compound as far as possible; reject red flags, missing exits, "
              "and non-green in/out-laps. `PitLaneTime_s` is not this loss. The TUM sum of green in/out transit, minimum "
              "standstill and team standstill addition excludes other modeled effects such as tire warmup; the estimands are not identical.",
              "- Keep SC/VSC stops in the source tables and estimate them separately. Check the out-lap too; "
              "`PitStops.UnderSCorVSC` flags the in-lap only. Do not use neutralized laps for intrinsic pace/degradation fits.",
              "- Compound codes are legacy TUM identifiers, not current C1–C5 labels. In this file their hard-to-soft "
              "order is `A3, A4, A6`; verify the race-specific mapping against recorded stints before fitting. "
              "The dummy policy fixes `VER` to `A3` after its single stop; there is no compound optimizer.",
              "- `import_pars()` also merges season/all-season parameters from `pars_mcs.ini` (accident/failure "
              "probabilities, lap/start/pit variability, FCY probabilities). Those are outside this one-race-file "
              "inventory and mostly need multi-race evidence or external assumptions. The experiments disable random influences/events.", ""]
    (out / "parameter_mapping.md").write_text("\n".join(lines))
    print("Mapped %d parameters across %d field patterns; %d concrete NO SOURCE entries" %
          (len(rows), len(used), sum(row["WorkbookColumns"] == "NO SOURCE" for row in rows)))


if __name__ == "__main__":
    main()
