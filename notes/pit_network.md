# Pit-timing network (methodology step 6)

Status: first network trained and raced in the simulator, 2026-11-18. Builds on the race situations of [situation_sampling.md](situation_sampling.md).

## What it does

At the end of every lap of the first stint the network sees the race situation and answers pit or stay, at the same moment TUM's VSE decides. It predicts the two pit-now advantages of the training rows: pitting now minus pitting at the best later lap, in finishing position and in race time (negative: pitting now is better). The driver pits when pitting now is predicted to be at least as good: position first, and a predicted position difference within ±0.1 places counts as a tie decided by race time (`pit_now` in `integration/pit_network.py`). It stops once, onto the same compound as in the training rows, and on the last possible lap if it has not stopped by then (TUM requires two compounds).

## Inputs

The 22 inputs are the situation features of [situation_sampling.md](situation_sampling.md) plus four tyre features from the race file:

| Input | Notes |
|---|---|
| race progress, laps remaining, tyre age, tyre age / race length | as TUM's network |
| relative compound (hard / medium / soft) | one-hot |
| position, gap ahead, gap behind | gaps capped at 30 s; no car ahead / behind flagged separately |
| car ahead within 1.5 s last lap, car behind pitted last lap | as TUM's network |
| safety car, VSC | one-hot flag state |
| TUM location category, pit loss (s) | track |
| **TyreDelta_s** | current tyres minus fresh stop tyres, one lap |
| **WearRate_s** | how much slower the current tyres get next lap |
| **StopWearRate_s** | mean per-lap wear of the stop compound over the laps left |
| **StayCost_s** | time the current tyres lose over the laps left compared with fresh stop tyres now |

The tyre features come from the driver's degradation model in the fitted race file (`tyre_features`), the same numbers the simulator uses. Without them the only track information would be the pit loss and TUM's location category.

## Training

`data_pipeline/train_pit_network.py` in a third environment, `.venv-train` (Python 3.10, PyTorch 2.5.1 CPU, numpy 2.2.6, pandas 2.3.3):

- 860,472 rows; Monte Carlo runs 0–159 of every driver-race for training (686,046 rows), runs 160–199 held out for validation (174,426 rows).
- Multilayer perceptron, 22 → 64 → 64 → 2, ReLU; mean squared error on both targets (position in places, race time in units of 10 s, clipped to ±60 s); targets that do not exist (retirement, unequal laps) are left out of the loss. Adam, learning rate 0.001, batches of 4,096, early stopping on the validation loss (patience 5).
- Mean squared error is minimized by the conditional mean, so the network learns the expected advantage given the situation, although each row's target is one random future.
- About 30 s on the laptop CPU (28 epochs). `--device auto` (default) trains on the RTX 5060 when the CUDA 12.8 build of PyTorch is installed in `.venv-train`; at this size the GPU is optional.
- The trained weights are saved as plain numpy arrays (`integration/models/pit_network_v1.npz`, 50 KB) and run in the simulator's Python 3.8 environment by a numpy forward pass (`PitNetwork`), so neither PyTorch nor TensorFlow is needed there. The training script checks that the numpy and PyTorch predictions agree (largest difference below 1e-3).

## Checks before racing it

- `integration/test_pit_network.py`: TUM's degradation formula is reproduced; the adapter stops exactly once, on the first lap the network says pit, or on the last possible lap; and **the situation the network sees during the race is identical to the one recorded after the race**, which is how the training rows were made (every input, every lap of a Monte Carlo Spielberg race).
- Validation (runs 160–199, other random futures of the same races):

| | Position MSE | Race-time MSE (s²) | Pit decision agrees with run average |
|---|---:|---:|---:|
| Constant (training mean) | 2.40 | 116.6 | |
| Lookup: training mean of the same race, driver and lap | 1.16 | 39.4 | |
| **Network** | **1.22** | **39.2** | **91.9 %** of 4,580 (race, driver, lap) cells |

  The network matches a lookup table that knows the exact race, driver and lap, using only the situation. Most of the remaining error is the randomness of each row's future, which no model can predict. Per track the decision agreement is 88 % (Spielberg) to 96 % (Austin).

- **A track it has never seen: no.** Trained on three tracks and tested on the fourth (`--holdout-track`, `notes/evidence/pit_network/pit_network_no<Track>.json`), the decision agreement drops to 37 % (Spielberg), 55 % (Monza), 77 % (Singapore) and 55 % (Austin): at Spielberg and Austin it pits on every lap, at Monza on almost none. With three tracks the network cannot learn what a track does; it recognises the four it was trained on. New tracks need more training tracks (Barcelona, the 2022–2025 seasons), not a bigger network.

## Racing it in the simulator

`integration/evaluate_pit_network.py --runs 100 --jobs 28`: every driver of the four races (80 driver-races), 100 new Monte Carlo races each (seeds from 500,000; training used 0–199), the same seeds for every policy, so the comparisons are paired. Only the target driver's policy changes; everyone else keeps their real strategy, as in the training data. 32,000 races, about 5 minutes, none rejected. `data_pipeline/summarise_evaluation.py` compares them ([evidence](evidence/pit_network/)):

| Policy | Mean finishing position | Mean first stop (lap) | Mean stops |
|---|---:|---:|---:|
| Real 2019 strategy | 9.52 | 21.3 | 1.36 |
| TUM's supervised VSE | 8.94 | 27.0 | 1.15 |
| Best fixed lap (Monte Carlo label) | 8.90 | 26.1 | 0.95 |
| **Network** | **8.95** | 24.9 | 0.95 |

(Stops below 1: retirements before the stop. 10.4 % of races end in a retirement under every policy.)

Paired differences, network minus the other policy (negative: the network is better), with standard errors:

| | All | Austin | Monza | Singapore | Spielberg |
|---|---:|---:|---:|---:|---:|
| **vs real**, places | **−0.57 ± 0.02** | +0.09 | −0.97 | −1.15 | −0.25 |
| vs real, race time (s) | **−4.5 ± 0.2** | +0.8 | −7.4 | −10.4 | −1.4 |
| **vs TUM's VSE**, places | **+0.01 ± 0.02** | +0.47 | +0.14 | −0.83 | +0.25 |
| vs TUM's VSE, race time (s) | **0.0 ± 0.2** | +5.8 | +1.4 | −9.1 | +1.7 |
| vs best fixed lap, places | +0.05 ± 0.01 | 0.00 | 0.00 | +0.16 | +0.03 |
| vs best fixed lap, race time (s) | 0.0 ± 0.1 | −0.2 | +0.1 | −0.2 | +0.2 |

Per driver-race (2 standard errors): better than the real strategy in 33 of 80, worse in 10; better than TUM's VSE in 21, worse in 30.

What this says:

- **It beats the real strategies** by about half a place and 4.5 s per race, most at Singapore and Monza. This is inside our simulator, which also produced the training labels, so it measures how well the network learned the simulator, not that it would have beaten the teams. The real strategies were also made against reacting rivals.
- **It ties TUM's VSE overall, but the tracks differ.** It is clearly better at Singapore (−0.83 places) and worse at Austin (+0.47), Spielberg (+0.25) and Monza (+0.14). Where it loses, TUM's VSE is doing something the network is not allowed to do: at Austin it makes **two stops in 64 %** of races; at Spielberg it chooses the compound (A6 or A4 in 56 % of races) where the network always mounts the driver's real second compound (A3). The network decides only *when* to make one stop.
- **It is no better than the best fixed lap.** It learned where the best stop is on average, but reacting to the race situation (safety cars, gaps) does not yet gain anything on top. Background drivers who cannot react, and a label that compares with one fixed later lap, give it little to react to.

## Next

1. Let TUM's compound-choice network pick the compound when our network pits (the project scope already says compound choice uses TUM's network); this removes the Spielberg handicap.
2. Two-stop decisions: situations in the second stint, needed for Austin and Barcelona.
3. Reacting background drivers (TUM's VSE for everyone else), so undercuts exist and reacting can pay.
4. More tracks before claiming anything about new tracks.

## Reproduce

```sh
.venv-train/Scripts/python.exe data_pipeline/train_pit_network.py                  # GPU if the CUDA build is installed
.venv-train/Scripts/python.exe data_pipeline/train_pit_network.py --holdout-track Monza --name pit_network_noMonza
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --runs 100 --jobs 28
.venv-data/Scripts/python.exe data_pipeline/summarise_evaluation.py integration/output/evaluation/pit_network_v1_mc100
```
