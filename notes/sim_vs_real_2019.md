# Simulator vs reality: 2019 Austrian Grand Prix

Dry-race verification: **verified by workbook Races sheet**.

Source: `f1_data/F1_2019_dry_races.xlsx`, `Results`, `Year == 2019 and Round == 9`; all 20 drivers. Simulation: `integration/output/realstrategy/classification.csv`, `realstrategy` for every driver, probabilistic influences and random events disabled. Original Spielberg 2019 parameters; no tuning.

- Finishing-position MAE: **2.800 places** (20 drivers).
- Exact finishing positions: **2/20**.
- Simulated winner: **LEC**, 4903.682 s; actual winner: **VER**, 4921.822 s.
- Race duration error (simulated winner minus actual winner): **-18.140 s**.
- Comparable full-distance driver times: **4**; MAE **8.226 s**, RMSE **11.227 s**, mean signed error **-8.226 s**.

| Driver | Actual position | Simulated position | Actual laps | Simulated laps | Time error (s) |
|---|---:|---:|---:|---:|---:|
| VER | 1 | 2 | 71 | 71 | -7.175 |
| LEC | 2 | 1 | 71 | 71 | -20.864 |
| BOT | 3 | 3 | 71 | 71 | -0.761 |
| VET | 4 | 8 | 71 | 70 | excluded |
| HAM | 5 | 4 | 71 | 71 | -4.106 |
| NOR | 6 | 9 | 70 | 70 | excluded |
| GAS | 7 | 11 | 70 | 70 | excluded |
| SAI | 8 | 17 | 70 | 69 | excluded |
| RAI | 9 | 6 | 70 | 70 | excluded |
| GIO | 10 | 10 | 70 | 70 | excluded |
| PER | 11 | 13 | 70 | 70 | excluded |
| RIC | 12 | 7 | 70 | 70 | excluded |
| HUL | 13 | 15 | 70 | 69 | excluded |
| STR | 14 | 12 | 70 | 70 | excluded |
| ALB | 15 | 5 | 70 | 70 | excluded |
| GRO | 16 | 18 | 70 | 69 | excluded |
| KVY | 17 | 16 | 70 | 69 | excluded |
| RUS | 18 | 14 | 69 | 69 | excluded |
| MAG | 19 | 20 | 69 | 69 | excluded |
| KUB | 20 | 19 | 68 | 69 | excluded |

Time comparisons include only drivers who finished the full scheduled distance in both datasets, with `Status == Finished` and finite times. Lapped drivers, retirements, and unequal distances are excluded from time metrics; they remain in the position metric. Time interpretation was explicitly selected as `winner-and-gaps`. The joined CSV retains raw `ResultTime_s` and normalized `ActualRaceTime_s`. For the public FastF1 3.8.3 export, the winner has 4921.822 s and Leclerc has a 2.724 s gap; the normalized Leclerc total is 4924.546 s, verified against Jolpica's `millis=4924546`. The installed FastF1 documentation describes totals, but these observed values require winner-plus-gap normalization.

This race is a diagnostic, not evidence that the simulator reproduces reality. The finishing-order errors are substantial, even with the recorded pit strategies. Traffic, fitted pace and tire parameters, penalties, and race events can contribute; this comparison does not identify their individual effects. No parameters were tuned.

Raw joined values: [evidence/sim_vs_real_2019.csv](evidence/sim_vs_real_2019.csv). Run logs and simulator validity manifests are saved with the experiment outputs.
