# Fitted parameters: Spielberg 2019

Methodology step 2, first race. Updated 2026-11-17. Data: `F1_2019_dry_races.xlsx`, round 9.

**Result:** tyre, fuel, pit-loss, pace and traffic values are fitted and written to `integration/parameters/pars_Spielberg_2019_fitted.ini`. The fitted values describe real lap times better than TUM's. With the traffic values measured too, the simulated race is also closer to the real one: position MAE 2.8 → 1.8, race-time error 8.2 s → 6.7 s. Before, TUM's pass threshold (2.01 s) allowed almost no passes; the data says about 0.3 s.

## How it is produced

```sh
.venv-data/bin/python data_pipeline/fit_parameters.py --year 2019 --round 9
.venv-data/bin/python data_pipeline/fit_traffic.py --year 2019 --round 9
.venv-tum/bin/python integration/write_fitted_pars.py --fit notes/evidence/fitted_parameters/Spielberg_2019_fit.json
```

1. `fit_parameters.py` reads the workbook and writes [evidence/fitted_parameters/Spielberg_2019_fit.json](evidence/fitted_parameters/Spielberg_2019_fit.json).
2. `fit_traffic.py` measures the traffic values from car pairs and writes [evidence/fitted_parameters/Spielberg_2019_traffic.json](evidence/fitted_parameters/Spielberg_2019_traffic.json), with every pair in `Spielberg_2019_traffic_pairs.csv`.
3. `write_fitted_pars.py` copies TUM's race file and changes only the fitted values (135 here). A manifest next to the file lists every change and the hashes of its inputs.

No TUM file is changed or added. `run_experiment.py` and `label_generator.py` take the fitted file's path through `--race`.

## Method

Model for one lap: `driver level − trend × (lap − 1) + k0[compound] + k1[compound] × tyre age`. A new tyre has age 0, as in TUM.

**Free-air laps only.** A lap is used if it is a clean lap and no car is within 2.0 s ahead at its start or end. 625 of 1,334 clean laps remain.

- The earlier v4 fit used all clean laps and gave negative medium and soft slopes. `fit_parameters.py` reproduces those v4 numbers exactly when it is run on all clean laps.
- The cause is traffic. Cars held up early in the race lap faster as the field spreads out, which hides tyre wear and inflates the trend.
- On free-air laps the trend is 0.0295 s/lap, almost exactly TUM's physical fuel effect (0.0294 s/lap). On all clean laps it was 0.036.

**Ordered compounds.** A softer compound may not be slower when new and may not wear more slowly. Without this rule the free-air fit still says medium and soft wear at half the rate of the hard and are slower when new. The reason is that medium and soft were used almost only in the first stint, so compound and race phase cannot be told apart at this race.

**Pit loss.** The in-lap is compared with the driver's clean laps just before it, the out-lap with the clean laps just after it. 22 green-flag tyre stops; MAG's drive-through is left out.

## Fitted values

| Quantity | Fitted | Standard error | TUM file |
|---|---:|---:|---:|
| Trend, s/lap | 0.0295 | 0.0042 | 0.0294 (fuel only) |
| Hard k1, s/lap | 0.0177 | 0.0046 | 0.0258 (median of 20 drivers) |
| Medium k1, s/lap | 0.0177 | 0.0046 | 0.010 (12 of 20 at the 0.01 floor) |
| Soft k1, s/lap | 0.0177 | 0.0042 | 0.010 (18 of 20 at the floor) |
| Medium k0, s vs hard | 0.000 | 0.010 | differs by driver |
| Soft k0, s vs hard | −0.056 | 0.104 | differs by driver |
| In-lap loss, s | 3.414 | | 2.691 |
| Out-lap loss, s | 17.806 | | 15.1 (drive + median standstill + cold tyres) |

Standard errors come from resampling whole stints 200 times.

**The three slopes are equal because the ordering rule is active, not because the data shows it.** The data supports one wear rate for this race, about 0.018 s/lap. It cannot separate the compounds.

How the values map to TUM parameters:

| Our value | TUM parameter | Rule |
|---|---|---|
| k0, k1 | `tireset_pars.<driver>.<A3/A4/A6>.k_0`, `k_1_lin` | same value for every driver; hard = A3, medium = A4, soft = A6, matched from the stint sequences |
| Trend | `track_pars.t_lap_sens_mass` | trend ÷ fuel burn per lap (1.549 kg). TUM has no track evolution, so fuel carries the whole trend |
| Driver level | `t_gap_racepace`, `car_pars.<team>.t_car`, `driver_pars.<driver>.t_driver` | fastest driver sets the level; a team's faster driver sets `t_car` |
| In-lap loss | `track_pars.t_pitdrive_inlap` | median |
| Out-lap loss | `track_pars.t_pitdrive_outlap` | median after taking off TUM's standstill and cold-tyre times, which have no source in our data |

PER has only 4 free-air laps, so he takes STR's level plus TUM's own gap between the two.

## Traffic values

TUM decides overtaking with four values. All four are measured from the workbook, from pairs of cars next to each other in race position at the start of a lap: both on a green lap, neither pitting in that lap or the one before. 1,231 pairs, 292 of them within 1.5 s, 40 on-track passes. A car's free-air lap time comes from the lap-time fit above.

| TUM value | What it does in TUM | How it is measured | Measured | SE | TUM |
|---|---|---|---:|---:|---:|
| `min_t_dist` | closest a following car can get | 5th percentile of the end-of-lap gap over close pairs that did not swap | 0.488 s | 0.023 | 0.5 |
| `t_duel` | time both cars lose while fighting | lap time of the front car above its free-air time, with a car within 1 s behind, nobody within 2 s ahead, no pass | 0.033 s | 0.085 | 0.3 |
| `t_overtake_loser` | extra time the passed car loses | same for the passed car on the lap it is passed, minus `t_duel` | 1.33 s | 0.128 | 0.3 |
| `t_gap_overtake` | follower passes if its lead after the lap would reach this | threshold on the follower's advantage (pace difference + DRS − start gap − TUM's top-speed term) that predicts as many passes over close pairs as really happened | 0.315 s | 0.085 | 2.01 |

Standard errors come from resampling whole laps 200 times.

- **`min_t_dist` agrees with TUM.** It is now backed by data.
- **`t_gap_overtake` is the value that mattered.** Real passes happened at small advantages: pass rate 21 % at −0.25 to 0 s, 43 % at 0 to 0.25 s, 100 % above 1 s. TUM's 2.01 s needs a car about 2.2 s/lap faster from 0.5 s behind, so in the simulator only 3 passes happen all race.
- **`t_duel` is close to zero once the front car has free air.** Without that condition the front car seems to lose about 1 s, but that is because it is itself stuck behind another car.
- **`t_overtake_loser` is probably too high.** A car is more likely to be passed on a lap where it is slow for other reasons, such as a mistake, and that loss is counted here too. 19 passes have a front car with free air.
- The pass threshold is blurred by the lap-time fit's error (about 0.4 s per lap), so it gets the number of passes right, not which pairs pass: at the measured threshold about half the predicted passes are real ones.
- DRS effect and window and the top-speed term are TUM's.

### Measuring rather than tuning

`integration/tune_traffic.py` reruns the race over a grid of values (in memory, no file written) and scores each run against the real result: [evidence/fitted_parameters/traffic_grid.csv](evidence/fitted_parameters/traffic_grid.csv).

| `t_gap_overtake` (others measured) | Passes (real 40) | Position MAE | Rank correlation | Time MAE |
|---:|---:|---:|---:|---:|
| 2.01 (TUM) | 4 | 3.2 | 0.764 | 9.5 s |
| 1.0 | 9 | 3.3 | 0.747 | 8.5 s |
| 0.5 | 23 | 2.0 | 0.872 | 13.8 s |
| **0.315 (measured)** | **37** | **1.8** | **0.907** | **6.7 s** |
| 0.2 | 54 | 1.6 | 0.907 | 7.1 s |
| 0.1 | 54 | 1.6 | 0.913 | 11.1 s |

Every threshold at or below about 0.5 s gives position MAE 1.4–2.0 instead of 2.7–3.5. Below that the result jumps from one value to the next: the top four reshuffle on margins of a single lap, because each run is one deterministic race. Tuning to the finishing order of one race would fit that noise, so the file uses the measured values without further tuning. They land in the good region on every score.

## Check 1: lap times without traffic

Each parameter file is used as a formula for the lap time of a car alone on track and compared with every real lap. Both files were fitted on this race.

| Laps | TUM RMSE | Fitted RMSE | TUM mean error | Fitted mean error |
|---|---:|---:|---:|---:|
| 625 free-air laps | 0.568 s | 0.395 s | −0.202 s | +0.005 s |
| 1,334 clean laps | 0.854 s | 0.680 s | −0.430 s | −0.187 s |

The fitted file is closer on both. TUM's values predict laps 0.2 s too fast in free air. The fitted file's −0.19 s on all clean laps is the average time real cars lost in traffic, which the simulator has to add itself.

## Check 2: simulated race against the real result

Deterministic, `realstrategy` for all drivers. Time metrics use the 4 drivers who finished the full distance.

| Parameters | Position MAE | Exact positions | Rank correlation | Time mean error | Winner (real: VER) |
|---|---:|---:|---:|---:|---|
| TUM | 2.8 | 2/20 | 0.779 | −8.2 s | LEC |
| + pit loss | 2.7 | 3/20 | 0.794 | −4.8 s | LEC |
| + pit loss and tyres (TUM pace) | 2.5 | 2/20 | 0.818 | +23.9 s | LEC |
| + pit loss, tyres and pace | 2.9 | 1/20 | 0.780 | +13.2 s | LEC |
| + pit loss, tyres, pace and traffic (the fitted file) | 1.8 | 4/20 | 0.907 | +1.9 s (5 drivers, MAE 6.7 s) | LEC |

**With the traffic values the fitted file is better on every measure.** Without them it was worse than TUM's file, for this reason:

- In the fit VER is the fastest car (0.15 s/lap faster than LEC), which agrees with the real race. In the simulator he starts second and follows LEC for the whole race, because TUM requires a 2.01 s/lap advantage to overtake (`t_gap_overtake`).
- While he follows, TUM adds `t_duel` (0.3 s) to both cars every lap. LEC's simulated laps are 0.2–0.45 s slower than his real ones for this reason, about 18 s over the race.
- In the real race VER dropped to seventh at the start and passed LEC on lap 69 with a much smaller pace advantage. The simulator can reproduce neither.
- With TUM's values LEC is the faster car and drives away, so the problem does not show.

With the measured traffic values the simulator makes 37 passes (real 40). The top five are LEC, VER, VET, BOT, HAM (real: VER, LEC, BOT, VET, HAM), and VER finishes 0.5 s behind LEC. He does not win: his real pass on lap 69 was by a margin a deterministic race does not reproduce. VER's bad start (seventh after lap 1) is not modelled either.

"TUM pace" in the third row is not a consistent set: TUM's per-driver tyre offsets carry part of each driver's pace, so its pace offsets do not fit our tyres (top cars end up 0.1–0.5 s/lap too slow).

## Check 3: pit-lap labels

One stop onto the hard tyre, in-laps 2–70, deterministic. "Fastest" is the stop with the lowest race time.

| Driver | Real stop | Fastest, TUM | Fastest, pit + tyres | Fastest, fitted without traffic | Fastest, fitted file | Within 2 s, fitted file |
|---|---:|---:|---:|---:|---:|---:|
| VER | 31 | 52 | 34 | 18 | 42 | 37–47 |
| HAM | 30 | 17 | 34 | 21 | 31 | 26–34 |
| LEC | 22 | 25 | 28 | 20 | 34 | 30–46 |
| BOT | 21 | 36 | 34 | 27 | 34 | 25–45 |
| Mean distance from the real stop | | 13.0 | 6.5 | 7.5 | 9.3 | |

All columns: [evidence/fitted_parameters/labels.csv](evidence/fitted_parameters/labels.csv).

- **VER's late optimum is gone.** With the fitted tyres the tyre model alone puts the best stop on lap 34 or 35, in the simulator and by hand calculation. TUM's values put it on lap 52. With pit + tyres, laps 29–42 are within 2 s of the best, which includes VER's real lap 31.
- **Without the traffic values, the best stops moved early (laps 18–27)**, because the front cars ran in a train and stopping early got a car into free air. The best lap was a single lap, not a window, for VER and HAM.
- **With the traffic values that effect is gone.** Cars can pass, so being stuck no longer drives the stop. Every driver now has a window of 8–21 laps within 2 s of the best.
- **The best stops are later than the real ones**, by 9 laps on average. HAM's real stop is inside his window; LEC's and BOT's real stops (laps 21–22) are early, probably a reaction to each other and to VER. The label moves one driver while the others keep their real strategy, so it does not capture that fight. With equal wear on all compounds the tyre model also has no reason to stop early.
- These are still deterministic. Monte Carlo runs are needed before they are used as labels.
- The real stops are not known to be optimal, so the distance to them is an indication only.

## Limits

- **One race, fitted and checked on the same race.** Nothing here shows the method works on a race it has not seen.
- **Compounds are not separated.** Equal wear rates make a soft first stint look as durable as a medium one. Real soft starters stopped on laps 21–25, medium starters on 25–35.
- **Linear wear only.** A late drop in tyre performance is not modelled.
- **Free-air pace is thin for some drivers:** PER 4 laps, GAS 10, GIO 15. SAI's level is probably too slow, because his second stint ended with damage.
- **Starting tyre age.** TUM's file starts used sets at age 2; the workbook says 3. The file's stint list was not changed.
- **Traffic values come from one race**, and `t_overtake_loser` is probably biased high. DRS values, the top-speed term and all safety-car pit values are TUM's.
- **The start is not modelled.** TUM starts from the grid with a fixed loss per grid position; real first-lap position changes (VER from second to seventh) are not reproduced.

## Decision

Labels are generated with the full fitted file (pit, tyres, pace, traffic), decided 2026-11-17. With the traffic values it is both the consistent fit and the one closest to the real race. `--groups` still writes any subset for comparison. The previous full fit without traffic is kept in the outputs as `fitted_notraffic`.

Reproduce the tables: write a file per group set with `--groups` and `--out integration/output/parameters/pars_Spielberg_2019_<name>.ini`, run `run_experiment.py realstrategy --race <file>` and `label_generator.py --race <file> --driver <D> --compound A3 --deterministic` for each, then `data_pipeline/compare_fitted.py`.
