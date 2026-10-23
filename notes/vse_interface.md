# VSE decision interface

Source: TUM `race-simulation`, upstream commit `96ef2c2021982217be008fe458df47c1a72da071`.
Read directly from `racesim/src/vse.py:VSE.decide_pitstop` and `racesim/src/race.py:Race.__handle_vse`.

`VSE.decide_pitstop(...) -> list` is the integration point. It receives 25 arguments and returns one entry per driver: `None` means stay out; a compound string means pit at the end of `cur_lap` and mount that compound. There is no separate pit boolean, tire-age return value, or fuel return value. Driver order must match `driver_initials`; TUM sorts its driver objects by car number, not finishing position.

| Argument | Meaning and timing |
|---|---|
| `driver_initials` | Driver identifiers, shared index order for all per-driver inputs |
| `cur_compounds` | Current TUM compound codes, e.g. A3/A4/A6 |
| `no_past_tirechanges` | Executed tire changes, excluding start stint |
| `tire_ages` | Effective degradation ages; TUM has already aged the cars for this lap |
| `positions_prevlap` | Positions at end of preceding lap |
| `pit_prevlap` | Whether driver is on an out-lap now |
| `cur_lap` | Current lap, and the in-lap if a stop is chosen |
| `tot_no_laps` | Scheduled distance |
| `fcy_types` | Active SC/VSC type, or None, for each driver |
| `fcy_start_end_progs` | Active neutralization start/end progress; handling caps the known active end at current lap |
| `bool_driving` | Current-lap active/retired mask |
| `bool_driving_prevlap` | Active mask at end of preceding lap |
| `racetimes_prevlap` | Cumulative times at end of preceding lap |
| `location` | TUM track identifier |
| `used_2compounds` | Whether each driver has used two distinct compounds |
| `cur_positions` | Provisional current positions after on-track interactions, before pit entry |
| `cur_racetimes_tmp` | Previous race times plus current provisional lap times, before pit entry |
| `t_pit_tirechange_min` | Track minimum stationary service time, seconds |
| `t_pit_tirechange_adds` | Per-driver team service additions, seconds |
| `t_pitdrive_inlap` | Green pit-entry transit loss, seconds |
| `t_pitdrive_outlap` | Green pit-exit transit loss, seconds |
| `t_pitdrive_inlap_fcy` | VSC pit-entry transit loss, seconds |
| `t_pitdrive_outlap_fcy` | VSC pit-exit transit loss, seconds |
| `t_pitdrive_inlap_sc` | SC pit-entry transit loss, seconds |
| `t_pitdrive_outlap_sc` | SC pit-exit transit loss, seconds |

Call chain: `Race.simulate_race()` → `Race.__simulate_lap()` → `Race.__handle_vse()` → `VSE.decide_pitstop()`. Within a lap, TUM processes previous-stop pit exits, lap pace, neutralizations, aging/fuel consumption, retirements, and on-track overtaking before the decision. It then handles pit entry and finalizes lap times. A non-None decision adds `[cur_lap, compound, 0, 0.0]` to `driver.strategy_info`. The stop appears as an in-lap on N and an out-lap on N+1; the location of the pit box relative to the timing line determines which lap receives stationary service time.

`VSE_REALSTRATEGY.make_decision()` looks up the recorded stop's lap and compound. It does not return the tire age stored in the recorded tuple. The main race loop therefore mounts a zero-age set even where the original recorded strategy specifies a used set. This is an upstream behavior, retained in the comparison.

The future timing model should emit pit/stay only. A separate fixed rule must supply the compound. Current-lap provisional values are available at this simulator decision point, but most positional features are previous-lap values. Training must match that timing and must not use future pit laps, final results, or later neutralization endings as features.

## Fixed-lap experiment

`integration/fixed_lap_policy.py` implements a single stop on N with a predetermined dry compound. `FixedLapVSE` delegates other drivers to TUM's `realstrategy` and replaces every decision for the selected driver, suppressing its originally recorded stop. It accepts only `realstrategy` background drivers because the original neural-network caches would otherwise need additional synchronization.

The hook is in `integration/run_experiment.py`, marked `# CAPSTONE HOOK`: assign the adapter through the public `race.vse` property after construction and before `simulate_race()`. No TUM source hook or monkeypatch is needed.

Executed using Python 3.8:

```sh
.venv-tum/bin/python integration/run_experiment.py baseline
.venv-tum/bin/python integration/run_experiment.py fixed --driver VER --pit-lap 25 --compound A3
.venv-tum/bin/python integration/run_experiment.py realstrategy
```

All three completed 71 laps with 20 drivers and TUM `result_status == 0`. Baseline used the original `.ini` `strategy_info` with VSE disabled and no numeric parameter edits. The selected event is the 2019 Austrian GP, a dry candidate; the workbook's tire/rainfall dry filter remains unverified because no 2019 timing/weather cache was available.

Evidence from the actual fixed run, `integration/output/fixed/pitstops.csv`:

```csv
Driver,InLap,OutLap,Compound,TyreAge,FuelAdded_kg,InLapInfluence,OutLapInfluence
VER,25,26,A3,0,0.0,pitinlap,pitoutlap
```

The same run's native `Spielberg_2019_lapinfluences.csv` marks VER `pitinlap` on 25 and `pitoutlap` on 26. A compact extract joining the native influence, lap-time and cumulative-time CSVs is saved as [evidence/fixed_lap_native_rows.csv](evidence/fixed_lap_native_rows.csv). The runner checks both flags and that VER has exactly this one stop. It fixes A3 throughout the experiment; compound optimization, pit order, and undercut/overcut logic are absent. A3 differs from VER's starting A4, so the completed race also passes TUM's two-compound plausibility check.

Logs: `notes/evidence/baseline.log`, `fixed.log`, `realstrategy.log`. Each output directory includes a `manifest.json` with validity status, settings, interpreter, and race-file hash. For the Apple Silicon import workaround, see [environment.md](environment.md).
