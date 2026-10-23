# Parameter mapping: Spielberg 2019

Source: `race-simulation/racesim/input/parameters/pars_Spielberg_2019.ini`. All **740 concrete parameters** map to **74 field patterns** below. `*` expands across every team, driver, or compound key. Lists are shown as one parameter with ordered tuple semantics. The [full value inventory](parameter_inventory.csv) contains every concrete path and original value. The generator fails if any field lacks a mapping.

Workbook sources refer to the standard dry-race export (`F1_2019_dry_races.xlsx` for validation; 2022–2025 and 2026 separate for production). These are proposed estimators, not claims that a parameter has already been fitted. **NO SOURCE** means the current workbook cannot supply the parameter. Do not fill gaps with invented measurements.

| Parameter pattern | Count | Meaning / units | Workbook columns | Limitations / assumptions |
|---|---:|---|---|---|
| `race_pars.season` | 1 | Season used to select Monte Carlo parameters | Races.Year | 2019 validation only; production eras are 2022–2025 and 2026 separately. |
| `race_pars.tot_no_laps` | 1 | Scheduled race distance, laps | TrackSummary.RaceLaps; Results.Laps; Laps.LapNumber | Observed max laps needs checking for shortened races. |
| `race_pars.min_t_dist` | 1 | Minimum following time gap, s | NO SOURCE | Lap-end positions alone cannot identify a following-gap floor. |
| `race_pars.min_t_dist_sc` | 1 | Minimum following gap behind SC, s | NO SOURCE | No continuous inter-car gap observations. |
| `race_pars.t_duel` | 1 | Time lost by battling drivers per lap, s | NO SOURCE | Cannot separate from traffic and intrinsic pace with these columns. |
| `race_pars.t_overtake_loser` | 1 | Time lost by overtaken driver, s | NO SOURCE | Lap-end position changes do not identify maneuver cost. |
| `race_pars.use_drs` | 1 | Enable DRS | NO SOURCE | Series/rules configuration; DRS activation is absent from the workbook. |
| `race_pars.drs_window` | 1 | DRS eligibility gap, s | NO SOURCE | Rules input, not a fitted race observation. |
| `race_pars.drs_allow_lap` | 1 | First lap allowing DRS | NO SOURCE | Rules input; must be era-specific. |
| `race_pars.drs_sc_delay` | 1 | Laps before DRS returns after SC | NO SOURCE | Rules input; must be era-specific. |
| `race_pars.participants` | 1 | Driver identifiers in the race | Results.Abbreviation; Laps.Driver | Check starters versus DNS, missing lap data, and identifier changes. |
| `track_pars.name` | 1 | TUM track identifier | Races.Location; Races.Event | Requires an explicit name/track-layout lookup. |
| `track_pars.t_q` | 1 | Best qualifying lap time, s | NO SOURCE | This workbook contains races only, not qualifying. |
| `track_pars.t_gap_racepace` | 1 | Offset from qualifying reference to best race pace, s | Laps.LapTime_s, CleanLap, TyreLife, Compound, Driver | Conditional fit only; qualifying reference and fuel model are missing. |
| `track_pars.t_lap_sens_mass` | 1 | Lap-time sensitivity to fuel mass, s/kg | NO SOURCE | No measured fuel mass; cannot separate fuel burn and tire wear from lap times alone. |
| `track_pars.t_pit_tirechange_min` | 1 | Minimum stationary tire-change time, s | NO SOURCE | PitLaneTime_s includes transit; it cannot isolate stationary service time. |
| `track_pars.t_pitdrive_inlap` | 1 | Extra pit-entry transit loss in green conditions, s | PitStops.InLapTime_s; Laps.LapTime_s, CleanLap; PitStops.TrackStatusInLap | Only combined in+out loss is robust; excludes standstill and cold-tire effects in TUM. |
| `track_pars.t_pitdrive_outlap` | 1 | Extra pit-exit transit loss in green conditions, s | PitStops.OutLapTime_s; Laps.LapTime_s, CleanLap | Same combined-loss limitation; fuel, compound, tire warmup, and traffic confound the split. |
| `track_pars.t_pitdrive_inlap_fcy` | 1 | Pit-entry transit loss under VSC, s | PitStops.InLapTime_s, UnderSCorVSC; Laps.VSC, SC, CleanLap | Use both in/out-lap status and matched VSC references; sparse observations. |
| `track_pars.t_pitdrive_outlap_fcy` | 1 | Pit-exit transit loss under VSC, s | PitStops.OutLapTime_s; Laps.VSC, SC, CleanLap | Use both laps; UnderSCorVSC alone does not distinguish VSC from SC. |
| `track_pars.t_pitdrive_inlap_sc` | 1 | Pit-entry transit loss under SC, s | PitStops.InLapTime_s, UnderSCorVSC; Laps.SC, VSC | SC queue and deployment phase matter; only conditional estimates. |
| `track_pars.t_pitdrive_outlap_sc` | 1 | Pit-exit transit loss under SC, s | PitStops.OutLapTime_s; Laps.SC, VSC | SC queue and deployment phase matter; do not mix with green stops. |
| `track_pars.pits_aft_finishline` | 1 | Whether pit box is after timing line | NO SOURCE | Circuit geometry, not reliably encoded by lap-level pit timestamps. |
| `track_pars.t_loss_pergridpos` | 1 | Start loss per grid position, s/position | Results.GridPosition; Laps.LapNumber, LapTime_s, Driver | Fit lap-1 residuals across drivers/races; start performance and traffic confound this. |
| `track_pars.t_loss_firstlap` | 1 | Standing-start first-lap loss, s | Laps.LapNumber, LapTime_s; Results.GridPosition | Separate from grid position, fuel, tire, and traffic effects. |
| `track_pars.t_gap_overtake` | 1 | Required relative pace advantage to pass, s | NO SOURCE | No pass-event or continuous gap data; cannot identify from position alone. |
| `track_pars.t_gap_overtake_vel` | 1 | Modify pass threshold using speed difference, s/(km/h) | NO SOURCE | Qualifying speed-trap measurements absent. |
| `track_pars.t_drseffect` | 1 | Lap-time benefit of DRS, s (negative) | NO SOURCE | DRS use is not included in this workbook. |
| `track_pars.mult_t_lap_sc` | 1 | SC lap-time multiplier | Laps.LapTime_s, SC, VSC, LapNumber, IsInLap, IsOutLap | Use fully neutralized laps and matched green baseline; SC boolean includes partial laps. |
| `track_pars.mult_t_lap_fcy` | 1 | VSC lap-time multiplier | Laps.LapTime_s, VSC, SC, LapNumber, IsInLap, IsOutLap | Separate VSC from SC and partial transitions; not a CleanLap-only pace fit. |
| `car_pars.*.drivetype` | 10 | Combustion or electric vehicle type | NO SOURCE | Configuration; combustion for this validation file. |
| `car_pars.*.manufacturer` | 10 | Team/manufacturer label | Results.TeamName; Laps.Team | Normalize team names to TUM identifiers; this file uses team names. |
| `car_pars.*.t_car` | 10 | Car/team intrinsic pace offset, s/lap | Laps.Team, Driver, LapTime_s, CleanLap, Compound, TyreLife | Fuel-correct first; car and driver offsets are not separately identifiable without constraints. |
| `car_pars.*.m_fuel` | 10 | Initial fuel mass, kg | NO SOURCE | Fuel load is not public in the workbook; regulatory maximum is not measured actual load. |
| `car_pars.*.b_fuel_perlap` | 10 | Fuel burned per green lap, kg/lap | NO SOURCE | Requires an external assumption or fuel model. |
| `car_pars.*.energy` | 10 | Initial electric energy, kWh | NO SOURCE | Unused/null in this combustion-car file. |
| `car_pars.*.energy_perlap` | 10 | Energy consumption, kWh/lap | NO SOURCE | Unused/null in this combustion-car file. |
| `car_pars.*.mult_consumption_sc` | 10 | SC fuel/energy use multiplier | NO SOURCE | No fuel/energy telemetry in workbook. |
| `car_pars.*.mult_consumption_fcy` | 10 | VSC fuel/energy use multiplier | NO SOURCE | No fuel/energy telemetry in workbook. |
| `car_pars.*.auto_consumption_adjust` | 10 | Adjust fuel consumption to remaining distance | NO SOURCE | Simulator configuration, not an observed parameter. |
| `car_pars.*.t_pit_tirechange_add` | 10 | Team stationary service excess above minimum, s | NO SOURCE | PitLaneTime_s includes transit and does not identify stationary team performance. |
| `car_pars.*.t_pit_refuel_perkg` | 10 | Stationary refueling time, s/kg | NO SOURCE | Null in this no-refueling F1 configuration. |
| `car_pars.*.t_pit_charge_perkwh` | 10 | Stationary charging time, s/kWh | NO SOURCE | Null in this F1 configuration. |
| `car_pars.*.color` | 10 | Team display color | NO SOURCE | Cosmetic configuration; not present in exported results. |
| `tireset_pars.*.tire_deg_model` | 20 | Tire degradation functional form | Laps.LapTime_s, TyreLife, Compound, CleanLap, Stint | Model selection, not direct observation; use fuel-corrected clean laps and holdout checks. |
| `tireset_pars.*.mult_tiredeg_sc` | 20 | Effective wear rate during SC | NO SOURCE | Neutralized-lap time is not a direct measurement of tire wear; retain explicit assumption. |
| `tireset_pars.*.mult_tiredeg_fcy` | 20 | Effective wear rate during VSC | NO SOURCE | No direct wear observation; retain explicit assumption. |
| `tireset_pars.*.t_add_coldtires` | 20 | Freshly mounted tire warmup penalty, s | Laps.LapTime_s, IsOutLap, Compound, TyreLife; PitStops.OutLapTime_s | Confounded with exit transit and fuel; conditional residual estimate only. |
| `tireset_pars.*.*.k_0` | 60 | Compound baseline pace offset, s | Laps.LapTime_s, Compound, TyreLife, CleanLap, Stint, Driver | Fuel-correct, constrain a reference compound, account for stint and driver offsets. |
| `tireset_pars.*.*.k_1_lin` | 60 | Linear degradation slope, s/tire-lap | Laps.LapTime_s, TyreLife, Compound, CleanLap, Stint, Driver | Fuel-correct first; traffic remains a confounder even for clean laps. |
| `tireset_pars.*.*.k_1_quad` | 60 | Linear term in quadratic degradation, s/tire-lap | Laps.LapTime_s, TyreLife, Compound, CleanLap, Stint, Driver | Fit jointly with k_2_quad; inactive when tire_deg_model is lin. |
| `tireset_pars.*.*.k_2_quad` | 60 | Quadratic degradation term, s/tire-lap² | Laps.LapTime_s, TyreLife, Compound, CleanLap, Stint, Driver | Needs a broad age range; inactive when tire_deg_model is lin. |
| `driver_pars.*.carno` | 20 | Driver race number | Laps.DriverNumber | Convert to integer and match abbreviations. |
| `driver_pars.*.name` | 20 | Full driver name | Results.FullName | Normalize spelling and accents. |
| `driver_pars.*.initials` | 20 | Driver abbreviation | Results.Abbreviation; Laps.Driver | Use exact identifier matching. |
| `driver_pars.*.team` | 20 | Team identifier | Results.TeamName; Laps.Team | Normalize to car_pars keys. |
| `driver_pars.*.t_driver` | 20 | Driver intrinsic pace offset, s/lap | Laps.Driver, Team, LapTime_s, Compound, TyreLife, CleanLap | Cannot separate from t_car without team/driver reference constraints; fuel-correct first. |
| `driver_pars.*.strategy_info` | 20 | Start and stop tuples [inlap, compound, initial tire age, fuel/energy added] | Stints.StartLap, Compound, TyreLifeAtStart; PitStops.InLap, CompoundAfter | Stop lap is InLap, not next stint StartLap. Start tire age ≈ TyreLifeAtStart−1; no-refuel value fixed at 0. |
| `driver_pars.*.p_grid` | 20 | Starting grid position | Results.GridPosition | Handle pit-lane starts and penalties explicitly. |
| `driver_pars.*.t_teamorder` | 20 | Team-order bias on pass threshold, s | NO SOURCE | Not identifiable from these exports. |
| `driver_pars.*.vel_max` | 20 | Qualifying maximum speed, km/h | NO SOURCE | Qualifying/speed-trap data not exported. |
| `monte_carlo_pars.min_dist_sc` | 1 | Minimum separation around SC phases, laps | NO SOURCE | Numerical constraint for lap-wise simulator, not a measured race feature. |
| `monte_carlo_pars.min_dist_vsc` | 1 | Minimum separation around VSC phases, laps | NO SOURCE | Numerical constraint for lap-wise simulator. |
| `monte_carlo_pars.ref_driver` | 1 | Assumed winner used in reference pre-simulation | Results.Abbreviation, Position | Can use actual winner for retrospective validation only; future use must avoid outcome leakage. |
| `event_pars.fcy_data.phases` | 1 | [start, end, SC/VSC, SC delay, SC duration] phase list | TrackStatus.SessionTime_s, Status; Laps.LapStartTime_s, LapNumber, SC, VSC | Align session time to race start; exact SC release/delay is not fully identifiable. Red flags unsupported by this phase schema. |
| `event_pars.fcy_data.domain` | 1 | Phase boundaries in progress or race time | NO SOURCE | Representation choice; prefer time after explicit timestamp alignment. |
| `event_pars.retire_data.retirements` | 1 | [driver, retirement time/progress] list | Results.Status, Laps, Abbreviation; Laps.LapNumber, LapStartTime_s, LapTime_s | Last completed lap gives only approximate retirement boundary; distinguish lapped finishers. |
| `event_pars.retire_data.domain` | 1 | Retirement boundary coordinate system | NO SOURCE | Representation choice; align to FCY time domain. |
| `vse_pars.available_compounds` | 1 | Compounds available to strategy engineer | Laps.Compound; Stints.Compound | Observed compounds do not prove availability; external nomination list and explicit legacy-code mapping required. |
| `vse_pars.param_dry_compounds` | 1 | Dry compounds with fitted parameters | Laps.Compound, CleanLap; Stints.Compound | Parameterize only supported dry compounds; never include I/W in this project policy. |
| `vse_pars.location_cat` | 1 | Pirelli tire-stress category (1 high, 3 low) | NO SOURCE | Requires an external track/nomination source. |
| `vse_pars.base_strategy.*` | 20 | Free-track optimized strategy tuples | NO SOURCE | Simulator optimization output, not a measured workbook parameter; optimization is not part of this timing-only dummy policy. |
| `vse_pars.real_strategy.*` | 20 | Recorded start/stop strategy tuples | Stints.StartLap, Compound, TyreLifeAtStart; PitStops.InLap, CompoundAfter | Historical replay input only. Exact initial tire age and compound codes need checking; fuel addition fixed at 0. |
| `vse_pars.vse_type.*` | 20 | Which VSE controls a driver | NO SOURCE | Experiment configuration; realstrategy background and one fixed-lap adapter. |

## Estimation rules

- Fit intrinsic pace and tire degradation only on `Laps.CleanLap == True`. Correct fuel first using an explicitly sourced or assumed fuel model; stint summary lap times are uncorrected. CleanLap does not remove traffic or engine-mode changes.
- The linear tire formula is `k_0 + k_1_lin * age`; the quadratic formula is `k_0 + k_1_quad * age + k_2_quad * age**2` (`helper_funcs/src/calc_tire_degradation.py`). This .ini selects `lin` for all drivers; stored quadratic coefficients are inactive.
- Green pit loss is approximately `(InLapTime_s + OutLapTime_s) - 2 * matched_reference_clean_lap`. Match driver, nearby race phase, fuel and compound as far as possible; reject red flags, missing exits, and non-green in/out-laps. `PitLaneTime_s` is not this loss. The TUM sum of green in/out transit, minimum standstill and team standstill addition excludes other modeled effects such as tire warmup; the estimands are not identical.
- Keep SC/VSC stops in the source tables and estimate them separately. Check the out-lap too; `PitStops.UnderSCorVSC` flags the in-lap only. Do not use neutralized laps for intrinsic pace/degradation fits.
- Compound codes are legacy TUM identifiers, not current C1–C5 labels. In this file their hard-to-soft order is `A3, A4, A6`; verify the race-specific mapping against recorded stints before fitting. The dummy policy fixes `VER` to `A3` after its single stop; there is no compound optimizer.
- `import_pars()` also merges season/all-season parameters from `pars_mcs.ini` (accident/failure probabilities, lap/start/pit variability, FCY probabilities). Those are outside this one-race-file inventory and mostly need multi-race evidence or external assumptions. The experiments disable random influences/events.
