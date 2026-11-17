# Held-out check of the fitted parameters

Methodology step 2, check on data the fit has not seen. Five 2019 races: Spielberg, Barcelona (TUM: Catalunya), Monza, Singapore, Austin. Updated 2026-11-17.

Until now every check used the same race the values were fitted on. This note tests the two parts most likely to fail out of sample:

1. **Tyres at ages the fit has not seen.** A pit-lap label asks what happens if a driver stays out longer or shorter than in reality, so the tyre model must hold beyond the observed tyre ages.
2. **Traffic values at a race they were not measured on.** If they transfer, future races could use pooled values.

**Result:** the tyre and pace fit holds out of sample: worse than in-sample, still clearly better than TUM's files at every track. But at Barcelona and Monza the fitted wear rate depends almost entirely on the last laps of each stint, and without them late laps are predicted too fast. The traffic values do not transfer between tracks: pooled values work at three tracks and fail at Spielberg. They have to be measured per race.

```sh
.venv-data/bin/python data_pipeline/heldout_tyres.py
.venv-tum/bin/python integration/heldout_traffic.py
```

## 1. Tyre ages the fit has not seen

For every stint, the free-air laps in the last third of the stint's tyre-age range are held out. The lap-time model is fitted on the rest with the same method and predicts the held-out laps. [evidence/fitted_parameters/heldout_tyres.csv](evidence/fitted_parameters/heldout_tyres.csv).

| Track | Held-out laps | Held-out fit, RMSE / mean error | In-sample fit | TUM's file |
|---|---:|---|---|---|
| Spielberg | 208 | 0.42 / +0.02 s | 0.37 / +0.01 s | 0.53 / −0.17 s |
| Barcelona | 171 | 0.74 / −0.40 s | 0.53 / −0.10 s | 1.57 / −1.41 s |
| Monza | 164 | 0.66 / −0.18 s | 0.56 / −0.03 s | 0.88 / −0.19 s |
| Singapore | 124 | 1.45 / +0.10 s | 1.16 / −0.08 s | 1.70 / −0.85 s |
| Austin | 172 | 0.72 / +0.08 s | 0.65 / +0.04 s | 1.56 / −1.17 s |

"In-sample" is our fit on all free-air laps, the held-out ones included, so it is a lower bound. TUM's values are fixed, so its row is the same kind of test as ours.

- **The held-out fit beats TUM's file at every track**, and is 0.05–0.3 s worse than in-sample.
- **The wear rate is not stable at Barcelona and Monza.** Fitted wear on all laps vs without the last third of each stint:

  | Track | Hard k1, all laps → held-out fit | Medium | Soft |
  |---|---|---|---|
  | Spielberg | 0.018 → 0.015 | 0.018 → 0.020 | 0.018 → 0.020 |
  | Barcelona | 0.038 → 0.005 | 0.038 → 0.005 | 0.038 → 0.005 |
  | Monza | 0.030 → 0.004 | 0.030 → 0.016 | 0.045 → 0.041 |
  | Singapore | 0.058 → 0.087 | 0.096 → 0.087 | 0.096 → 0.087 |
  | Austin | 0.090 → 0.093 | 0.127 → 0.125 | 0.127 → 0.125 |

  At Barcelona and Monza the early part of a stint shows almost no wear and the late part a lot. That is the shape of a late drop-off, which a linear wear model cannot represent. The held-out fit then predicts the late laps too fast (Barcelona −0.40 s on average).
- **For labels this matters**: a wear rate learnt from early laps makes long stints look too cheap, and so pushes the best stop later. Barcelona and Monza need a non-linear wear term (TUM has `k_1_quad`, unused so far) before their labels are trusted.
- **Error by how far a lap lies beyond the training laps of the same stint** ([heldout_tyres_by_extra_age.csv](evidence/fitted_parameters/heldout_tyres_by_extra_age.csv)):

  | Laps beyond | Laps | Mean error | RMSE | TUM mean error |
  |---|---:|---:|---:|---:|
  | 1–3 | 352 | −0.07 s | 0.76 s | −0.75 s |
  | 4–6 | 253 | −0.08 s | 0.64 s | −0.74 s |
  | 7–10 | 149 | −0.24 s | 0.97 s | −0.67 s |
  | 11+ | 85 | +0.14 s | 1.12 s | −0.76 s |

  The error grows with distance, about 1 s RMSE from 7 laps beyond. Labels that move a stop more than about 7 laps from the real one rely on extrapolation.

## 2. Traffic values from the other races

Each race is simulated with its fitted file, but with traffic values from the other four races only:

- `t_gap_overtake` = TUM's value for this track × the median ratio of measured to TUM value at the other tracks. This keeps TUM's ranking of tracks and uses our level.
- `min_t_dist`, `t_duel`, `t_overtake_loser` = median of the other tracks (`t_duel` at least 0).

Deterministic, real strategies. [evidence/fitted_parameters/heldout_traffic.csv](evidence/fitted_parameters/heldout_traffic.csv).

| Track | Pass threshold: own / other races | Passes: real / own / other races / TUM | Position MAE: own / other races / TUM |
|---|---|---|---|
| Spielberg | 0.32 / 1.34 | 40 / 37 / 13 / 1 | 1.8 / 3.5 / 2.8 |
| Barcelona | 1.08 / 1.53 | 17 / 21 / 17 / 4 | 2.0 / 2.1 / 1.8 |
| Monza | 1.17 / 0.99 | 27 / 26 / 26 / 20 | 1.5 / 1.6 / 1.1 |
| Singapore | 2.57 / 2.11 | 23 / 22 / 35 / 13 | 2.3 / 2.4 / 1.2 |
| Austin | 1.21 / 1.04 | 43 / 47 / 54 / 44 | 1.0 / 0.9 / 2.2 |
| Mean | | pass error 2.6 / 10.2 / 14.0 | 1.7 / 2.1 / 1.8 |

- **Pooled values work at Barcelona, Monza and Austin**: pass counts and finishing order are close to the own-race values.
- **They fail at Spielberg**, whose measured threshold is far lower than its TUM value suggests (ratio 0.16 against 0.47–0.68 elsewhere), and overshoot at Singapore (35 passes against 23).
- **Conclusion: measure traffic values per race.** The workbooks have the data for every race we simulate, so this costs nothing. Pooled values are a fallback only.
- TUM's files win on finishing order at Monza and Singapore, both with own and pooled values. Single deterministic races move on small margins, and TUM's 2019 files were probably tuned to these races.

## What this changes

- Use own-race traffic values, as the fitted files already do.
- Before trusting Barcelona and Monza labels: add a non-linear wear term and repeat this check.
- Treat label stops more than about 7 laps from the driver's real stop as extrapolation.
- Still open: a check on another season. Our study-circuit races after 2022 have no TUM race file yet.
