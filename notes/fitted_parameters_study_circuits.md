# Fitted parameters: the study circuits in 2019

Methodology step 2 for the 2019 races at the four study circuits: Barcelona (TUM: Catalunya), Monza, Singapore and Austin. Same method as Spielberg, see [fitted_parameters.md](fitted_parameters.md). Updated 2026-11-17.

**Result:** every race now has a fitted file in `integration/parameters/`. As lap-time models the fitted files beat TUM's at every track. In the simulated race they match the real number of passes everywhere, while finishing order is about as good as TUM's: better at Spielberg and Austin, worse at Singapore. Everything is still fitted and checked on the same race.

TUM's 2019 files were probably tuned to these same races, so beating them on finishing order in-sample is not expected; the next check is a race the fit has not seen.

## How it is produced

```sh
# per race (rounds: Spielberg 9, Barcelona 5, Monza 14, Singapore 15, Austin 19)
.venv-data/bin/python data_pipeline/fit_parameters.py --year 2019 --round 14
.venv-data/bin/python data_pipeline/fit_traffic.py --year 2019 --round 14
.venv-tum/bin/python integration/write_fitted_pars.py --fit notes/evidence/fitted_parameters/Monza_2019_fit.json
# the same without the traffic group, for comparison (not in git)
.venv-tum/bin/python integration/write_fitted_pars.py --fit notes/evidence/fitted_parameters/Monza_2019_fit.json --groups pit tyres pace --out integration/output/parameters/pars_Monza_2019_notraffic.ini
# checks over all five races
.venv-data/bin/python data_pipeline/check_lap_model.py
.venv-tum/bin/python integration/check_races.py
```

Files are named by TUM's track name; `integration/race_files.py` maps workbook locations to it. `fit_parameters.py` also writes the real classification (`<Track>_2019_results.csv`) that the simulator checks compare with.

Changes made to run these races:

- **Drivers without free-air laps.** If a driver and the team-mate both have fewer than 5 free-air laps, the driver keeps TUM's pace (`t_car + t_driver`), shifted by the field's median difference between our levels and TUM's. Needed at Singapore, where VER had 3 free-air laps and ALB 4.
- **Negative `t_duel`** is written as 0. A duel cannot make a car faster; the measured values are noise around zero.
- **TUM's Monza file does not run as shipped.** Its two recorded VSC phases are 1.46 laps apart and TUM requires 1.5. The simulator only needs one phase per lap, so all runners lower that minimum in memory to the recorded gap if it is at least one lap (`allow_recorded_fcy`). No TUM file is changed; the run manifests record it.

## Fitted values

| Track | Free-air laps | Lap RMSE | Trend s/lap | k1 hard / medium / soft, s/lap | k0 medium / soft vs hard, s | In-lap / out-lap loss, s |
|---|---:|---:|---:|---|---|---|
| Spielberg | 625 | 0.39 | 0.030 | 0.018 / 0.018 / 0.018 | 0.00 / −0.06 | 3.4 / 17.8 |
| Barcelona | 508 | 0.55 | 0.058 | 0.038 / 0.038 / 0.038 | −0.49 / −0.56 | 4.0 / 19.5 |
| Monza | 434 | 0.47 | 0.053 | 0.030 / 0.030 / 0.045 | 0.00 / −0.17 | 4.5 / 20.4 |
| Singapore | 323 | 1.14 | 0.080 | 0.058 / 0.096 / 0.096 | 0.00 / 0.00 | 7.9 / 19.9 |
| Austin | 469 | 0.56 | 0.067 | 0.090 / 0.127 / 0.127 | −0.21 / −0.21 | 0.6 / 20.4 |

All values with standard errors: `evidence/fitted_parameters/<Track>_2019_fit.json`.

- **Compounds separate better than at Spielberg.** Barcelona gives medium and soft about 0.5 s faster than hard when new; Monza and Singapore give the soft or medium faster wear; Austin both. The ordering rule still binds for some pairs, and standard errors on k0 are 0.2–0.7 s.
- **Trends are about twice the fuel effect** (0.05–0.08 s/lap against about 0.03). Spielberg matched the fuel effect; at these races track evolution adds to it. TUM puts the whole trend into fuel sensitivity, which is right for lap times but means the car gets faster for the wrong reason.
- **Singapore is a poor fit.** Only 323 free-air laps, most top cars almost never in free air, residual 1.14 s. Its values are the least reliable.
- Austin's in-lap loss of 0.6 s is real: TUM's file has 0.38 s, the pit entry is close to the timing line.

## Traffic values

| Track | Passes | `t_gap_overtake` measured (TUM) | `min_t_dist` | `t_duel` | `t_overtake_loser` |
|---|---:|---|---:|---:|---:|
| Spielberg | 40 | 0.32 ± 0.09 (2.01) | 0.49 | 0.03 | 1.33 |
| Barcelona | 17 | 1.08 ± 0.17 (2.31) | 0.55 | −0.05 → 0 | 1.64 |
| Monza | 27 | 1.17 ± 0.06 (1.76) | 0.39 | 0.14 | 0.74 |
| Singapore | 23 | 2.57 ± 0.13 (3.75) | 0.58 | −0.03 → 0 | 2.56 ± 1.1 |
| Austin | 43 | 1.21 ± 0.30 (1.83) | 0.59 | −0.19 → 0 | 1.46 |

- **The pass threshold ranks the tracks like TUM does**, Singapore hardest, but is 30–85 % lower everywhere. Spielberg is the outlier.
- **`t_duel` is about zero at every track** once the defending car has free air.
- **`min_t_dist` stays near TUM's 0.5 s** (0.39–0.59).
- `t_overtake_loser` is large everywhere (0.7–2.6 s) and probably biased high, see the Spielberg note.

## Check 1: lap times without traffic

`data_pipeline/check_lap_model.py`, all in-sample: [evidence/fitted_parameters/lap_model_checks.csv](evidence/fitted_parameters/lap_model_checks.csv).

| Track | TUM RMSE, free-air laps | Fitted RMSE | TUM mean error | Fitted mean error |
|---|---:|---:|---:|---:|
| Spielberg | 0.57 s | 0.40 s | −0.20 s | 0.00 s |
| Barcelona | 1.56 s | 0.55 s | −1.35 s | 0.00 s |
| Monza | 0.82 s | 0.47 s | −0.25 s | 0.00 s |
| Singapore | 1.61 s | 1.14 s | −0.62 s | 0.00 s |
| Austin | 1.47 s | 0.56 s | −1.13 s | 0.00 s |

TUM's formula predicts laps that are too fast at every track, by more than a second at Barcelona and Austin. The simulator's traffic model has to absorb that.

## Check 2: simulated race against the real result

`integration/check_races.py`: deterministic, real strategies, recorded safety cars and retirements. [evidence/fitted_parameters/race_checks.csv](evidence/fitted_parameters/race_checks.csv).

| Track | Passes: real / TUM / ours | Position MAE: TUM / no traffic / ours | Rank correlation: TUM / ours | Race-time error: TUM / ours |
|---|---|---|---|---|
| Spielberg | 40 / 1 / 37 | 2.8 / 2.9 / 1.8 | 0.78 / 0.91 | −8 s / +2 s |
| Barcelona | 17 / 4 / 21 | 1.8 / 1.4 / 2.0 | 0.91 / 0.89 | −101 s / −54 s |
| Monza | 27 / 20 / 26 | 1.1 / 1.8 / 1.5 | 0.96 / 0.90 | +1 s / −6 s |
| Singapore | 23 / 13 / 22 | 1.2 / 2.3 / 2.3 | 0.95 / 0.85 | −132 s / −56 s |
| Austin | 43 / 44 / 47 | 2.2 / 0.9 / 1.0 | 0.85 / 0.95 | −15 s / −5 s |
| Mean | | 1.8 / 1.9 / 1.7 | 0.89 / 0.90 | |

- **Passes match at every track** with our values (21–47 against 17–43 real). TUM's files give too few at four of five.
- **Finishing order is about even with TUM's.** Clear gains at Spielberg and Austin, losses at Singapore and Monza. Single deterministic races move on small margins, as the Spielberg grid showed.
- **Race times at the safety-car races are far off for both files** (Barcelona, Singapore). TUM is 101 and 132 s too fast, ours about 55 s. The cause is in TUM's safety-car handling and is not fitted here.
- The traffic group alone does not always help finishing order (Barcelona 1.4 → 2.0), but it is what makes the pass counts right.

## Check 3: pit-lap labels

Deterministic, one stop onto the driver's real second compound, every candidate lap. Drivers: top finishers who really stopped once. [evidence/fitted_parameters/labels_study_circuits.csv](evidence/fitted_parameters/labels_study_circuits.csv).

| Track | Driver | Real stop | Best, TUM | Best, fitted | Within 2 s, fitted |
|---|---|---:|---:|---:|---|
| Monza | LEC | 20 | 41 | 30 | 30 |
| Monza | BOT | 27 | 28 | 30 | 30 |
| Monza | RIC | 30 | 25 | 30 | 28–30 |
| Singapore | VET | 19 | 19 | 22 | 19–37 |
| Singapore | LEC | 20 | 22 | 27 | 24–27 |
| Singapore | VER | 19 | 28 | 22 | 19–37 |
| Singapore | HAM | 26 | 36 | 30 | 30 |
| Austin | HAM | 24 | 30 | 22 | 22–24 |
| Austin | RIC | 21 | 39 | 20 | 20–21 |
| Austin | SAI | 18 | 25 | 22 | 22 |
| Mean distance from the real stop | | | 7.9 | 3.7 | |

- **The fitted files put the best stop much closer to the real one**: 3.7 laps on average, against 7.9 with TUM's files.
- **These labels use hindsight about safety cars.** A deterministic run replays the recorded safety cars, so stopping under one is cheap. At Monza all three drivers come out at lap 30, the lap of the second VSC; at Singapore HAM's best with TUM's file is lap 36, the first safety car. Monte Carlo runs with sampled events remove this, and are needed before any of these become training labels.
- Many windows are 1–3 laps wide, so the deterministic optimum is still sharp.
- **Barcelona has no labels.** Every top finisher stopped twice, partly under the safety car, and the label generator handles one stop only.

## Limits

- **Fitted and checked on the same races.** The next check is a race the fit has not seen.
- **Singapore's fit is weak** (few free-air laps, four drivers on fallback pace).
- **Two-stop races cannot be labelled yet.**
- **Linear wear, trend treated as fuel**, and TUM's safety-car model, as before.
