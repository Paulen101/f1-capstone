# Pit-timing network (methodology step 6)

Status: first network trained and raced in the simulator, with TUM's compound choice, 2026-11-18. Builds on the race situations of [situation_sampling.md](situation_sampling.md).

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

`integration/evaluate_pit_network.py --runs 100 --jobs 28`: every driver of the four races (80 driver-races), 100 new Monte Carlo races each (seeds from 500,000; training used 0–199), the same seeds for every policy, so the comparisons are paired (checked: with the same seed the safety cars, retirements and lap-time noise are identical under every policy until the target driver's decisions differ). Only the target driver's policy changes; everyone else keeps their real strategy, as in the training data. Five policies, 40,000 races, about 6 minutes, none rejected. `data_pipeline/summarise_evaluation.py` compares them ([evidence](evidence/pit_network/)):

| Policy | Mean finishing position | Mean first stop (lap) | Mean stops | Stop under SC/VSC |
|---|---:|---:|---:|---:|
| Real 2019 strategy | 9.52 | 21.3 | 1.36 | 5.4 % |
| TUM's supervised VSE | 8.94 | 27.0 | 1.15 | **11.2 %** |
| Best fixed lap (Monte Carlo label) | 8.90 | 26.1 | 0.95 | 3.9 % |
| Network, training compound | 8.95 | 24.9 | 0.95 | 4.3 % |
| **Network, TUM's compound choice** | **8.94** | 24.9 | 0.95 | 4.4 % |

(Stops below 1: retirements before the stop. 10.4 % of races end in a retirement under every policy. A safety car or VSC comes out in 57 % of races.)

### Compound from TUM's network

The project scope gives compound choice to TUM's network. `integration/tum_compound_choice.py` runs TUM's `nn_supervised_compoundchoice` on its own, with TUM's preprocessor and TUM's rule for walking down the ranking, told that one tyre change remains (our network makes one stop; with one change left TUM never mounts the current compound again). A test checks that its output equals TUM's own `VSE_SUPERVISED` pipeline. When our network asks "pit now?", the tyre features describe the compound TUM would mount on that lap.

It changes almost nothing: −0.005 ± 0.003 places and −0.16 s against the training compound. At Spielberg TUM's network mounts the hard (A3) in 66 % of stops and the medium (A4) in 34 %, never the soft our earlier reading blamed; at Singapore it picks the same compound as the training rows in every race. **Compound choice was not why TUM's VSE wins at Spielberg.**

### Against the other policies

Paired differences, network with TUM's compound choice minus the other policy (negative: the network is better), with standard errors:

| | All | Austin | Monza | Singapore | Spielberg |
|---|---:|---:|---:|---:|---:|
| **vs real**, places | **−0.58 ± 0.02** | +0.06 | −0.97 | −1.15 | −0.24 |
| vs real, race time (s) | **−4.6 ± 0.2** | +0.1 | −7.4 | −10.4 | −1.3 |
| **vs TUM's VSE**, places | **0.00 ± 0.02** | +0.43 | +0.15 | −0.83 | +0.26 |
| vs TUM's VSE, race time (s) | **−0.2 ± 0.2** | +5.0 | +1.4 | −9.1 | +1.8 |
| vs best fixed lap, places | +0.04 ± 0.01 | −0.04 | 0.00 | +0.16 | +0.05 |
| vs best fixed lap, race time (s) | −0.2 ± 0.1 | −0.9 | +0.2 | −0.2 | +0.4 |

Per driver-race (2 standard errors): better than the real strategy in 32 of 80, worse in 10; better than TUM's VSE in 20, worse in 31.

### Where TUM's VSE wins: safety cars

Split by whether the race had a safety car or VSC:

| Network − TUM's VSE, places | All | Austin | Monza | Singapore | Spielberg |
|---|---:|---:|---:|---:|---:|
| Green races only | **−0.31 ± 0.02** | +0.06 | −0.10 | −1.20 | −0.01 |
| Races with SC or VSC | **+0.27 ± 0.03** | +0.75 | +0.35 | −0.52 | +0.49 |

Without neutralisations the network is as good as or better than TUM's VSE at every track. With them, TUM's VSE gains up to three quarters of a place. It makes its stop under a safety car or VSC in 11.2 % of races; the network in 4.4 %, hardly more than the fixed lap (3.9 %).

The network has learned what a safety car is worth: switching the flag of green validation situations to SC moves its predicted advantage by −0.75 places and −5.9 s, close to the data (−0.65 places, −5.1 s for the same race, driver and lap). It does pit when a safety car comes in its window. What it does not do is **wait for one**. The training label compares pitting now with pitting at one fixed later lap (the best on average), so staying out is never credited with the chance of a cheap stop under a later safety car. TUM's VSE, imitating teams, stops later in general (lap 27.0 against 24.9), so in more races a safety car comes out while it is still on its first set.

### What this says

- **It beats the real strategies** by about half a place and 4.6 s per race, most at Singapore and Monza. This is inside our simulator, which also produced the training labels, so it measures how well the network learned the simulator, not that it would have beaten the teams. The real strategies were also made against reacting rivals.
- **It ties TUM's VSE overall**: better in green races (−0.31 places), worse in safety-car races (+0.27). At Austin TUM's VSE also makes **two stops in 64 %** of races, which the network cannot.
- **It is no better than the best fixed lap**, for the same reason: without a label that values waiting, reacting adds little.

## Next

1. **A label that values waiting** (optimal stopping). For every run every candidate lap's outcome is already known, and the history up to the stop is shared, so any lap-by-lap policy can be scored exactly on the stored runs. The value of staying out at lap L is then the expected outcome of deciding well from L + 1 on, not of one fixed lap; it can be built backwards from the last lap with the network itself (fitted Q-iteration). This credits staying out with the chance of a later safety car.
2. Two-stop decisions: situations in the second stint, needed for Austin and Barcelona.
3. Reacting background drivers (TUM's VSE for everyone else), so undercuts exist and reacting can pay.
4. More tracks before claiming anything about new tracks.

## Reproduce

```sh
.venv-train/Scripts/python.exe data_pipeline/train_pit_network.py                  # GPU if the CUDA build is installed
.venv-train/Scripts/python.exe data_pipeline/train_pit_network.py --holdout-track Monza --name pit_network_noMonza
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --runs 100 --jobs 28
.venv-data/Scripts/python.exe data_pipeline/summarise_evaluation.py integration/output/evaluation/pit_network_v1_mc100 --base network_tum
```
