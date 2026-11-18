# Pit-timing network (methodology step 6)

Status: first network trained and raced in the simulator, with TUM's compound choice; round 2 (labels that value waiting) and round 3 (two stops, Austin) done, 2026-11-18. Builds on the race situations of [situation_sampling.md](situation_sampling.md).

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

## Round 2: a label that values waiting

The first label compares pitting now with pitting on one fixed later lap. Staying out is then never credited with what a good decision later can gain, such as a cheap stop under a later safety car.

**Exact replay.** In every stored Monte Carlo run every candidate pit lap shares the same random race up to its stop (same seed; no race was re-run, which the loader checks). So any policy that decides lap by lap can be scored exactly on the stored runs: it stops on the first lap it says pit, and its outcome is that lap's stored outcome in that run (`data_pipeline/stopping_labels.py`).

**Policy iteration** (`data_pipeline/train_stopping_network.py`): replay the current network on every run; new target = pitting now minus *following the current network from the next lap*, in the same run; train a fresh network (runs 0–159); score it exactly on runs 160–199; repeat. Started from v1, four iterations, about 50 s on the GPU. The best iteration is saved as `integration/models/pit_network_v2.npz` ([report](evidence/pit_network/pit_network_v2.json)).

| Held-out runs, places vs best fixed lap | All | SC or VSC races | Green races |
|---|---:|---:|---:|
| Hindsight: best lap of each run, knowing the future (upper bound) | −0.80 ± 0.02 | −1.04 ± 0.03 | −0.42 ± 0.02 |
| Wait for SC: stop under SC/VSC from 5 laps before the fixed lap, else on it | −0.02 ± 0.01 | −0.04 ± 0.02 | 0.00 |
| Same, but wait 4 laps longer without one | +0.07 ± 0.02 | 0.00 ± 0.03 | +0.19 ± 0.02 |
| Network v1 | 0.00 ± 0.02 | −0.03 ± 0.03 | +0.04 ± 0.02 |
| **Network v2** (4 iterations) | −0.03 ± 0.02 | −0.07 ± 0.03 | +0.04 ± 0.02 |

(`data_pipeline/stopping_headroom.py`, [evidence](evidence/pit_network/stopping_headroom.csv).) Each iteration moved the stops later (mean lap 25.3 → 26.3) and gained a little, but only 0.03 places in total.

**In the simulator** (100 new races per driver-race, same seeds as before, [evidence](evidence/pit_network/comparison_v1_v2_mc100_summary.csv)): v2 with TUM's compound choice stops 1.9 laps later than v1 and is **−0.006 ± 0.010 places** against it, no measurable change. Against TUM's VSE it is still −0.33 places in green races and +0.27 in safety-car races.

**Why waiting cannot pay here.** Even a hand-made "wait for a safety car" rule gains at most 0.04 places on the stored runs: with one stop onto one fixed compound per driver, an early or late stop has nothing to gain from a cheap pit stop. Yet TUM's VSE gains up to half a place in safety-car races. `integration/check_tum_as_fixed.py` re-simulates TUM's races as fixed stops on TUM's lap and compound with the same seed: **30 of 30 Spielberg races are identical** (Austin: 18 of 20; the other two are retirements with the same position). TUM's advantage is therefore entirely *which lap and which compound* it picks in each race, for example a late stop onto softs (Spielberg: lap 41 on A6 in 29 % of races). Our stored runs hold one compound per driver, so no label built from them can learn that combination, however it values waiting.

## Round 3: two-stop decisions (Austin)

At Austin TUM's VSE makes two stops in 64 % of races and beat the one-stop network by 0.43 places. The network can now make an optional second stop.

**Runs** (`integration/two_stop_generator.py --track Austin --runs 50`): for every driver and Monte Carlo run (seeds 0–49, the same as the one-stop runs), every one-stop lap and every two-stop plan with stints of at least 10 laps: first stop from lap 5, second stop at least 10 laps later and at least 10 laps before the end. 582 plans × 50 runs = 29,100 races per driver, about 5 minutes; 20 drivers in 97 minutes, none re-run. The first stop goes onto the one-stop compound, the second onto the driver's real last compound if they really stopped twice, otherwise the last compound most two-stoppers used (the soft at Austin). Situations are recorded in both stints where the history is shared. The one-stop races and first-stint situations are identical to those of the earlier one-stop batch with the same seeds (checked on a 3-run pilot: 162 races and 157 situations).

**Situation features** (`integration/situations.py`) now describe the current stint: compound and tyre age since the last stop, and a new input `StopsMade` (0 on the first set, 1 after a stop). On the first set the network decides the first stop; after it, whether and when to stop again.

**Labels and training** (`data_pipeline/two_stop_labels.py`, `train_two_stop_network.py`): the exact replay of round 2 over two levels. In the second stint, stopping now is compared with following the network from the next lap, which may mean never stopping again; in the first stint a stop now is worth what following the network's second-stint decisions after it is worth. Started from v1 (no second stop), four iterations of policy iteration, 30 s on the GPU; runs 0–39 for training, 40–49 held out. Saved as `integration/models/pit_network_v3_two_stop.npz` ([report](evidence/pit_network/pit_network_v3_two_stop.json)).

| Held-out runs, places vs best fixed one-stop lap | Places | Two-stop share |
|---|---:|---:|
| Hindsight upper bound | −1.20 | |
| Best fixed one- or two-stop plan (chosen on training runs) | −0.61 ± 0.08 | 94 % |
| Start: v1, one stop | −0.07 ± 0.04 | 0 % |
| **Network v3** (iteration 4) | **−0.56 ± 0.07** | 94 % |

**In the simulator** (100 new races per Austin driver, the same seeds as the other policies, [evidence](evidence/pit_network/austin_two_stop_mc100_summary.csv)):

| Austin | Mean position | First stop | Second stop | Stops |
|---|---:|---:|---:|---:|
| Real 2019 strategy | 9.52 | 17.9 | | 1.60 |
| TUM's supervised VSE | 9.14 | 21.9 | 39.7 | 1.60 |
| Network v1, one stop, TUM's compound | 9.58 | 23.0 | | 0.95 |
| **Network v3, two stops allowed** | **9.04** | 17.5 | 38.8 | 1.88 |

| Network v3 minus | Places | Race time | Green races | SC/VSC races |
|---|---:|---:|---:|---:|
| Real 2019 strategy | **−0.49 ± 0.03** | **−7.1 ± 0.3 s** | −0.38 | −0.57 |
| **TUM's VSE** | **−0.11 ± 0.03** | **−2.1 ± 0.3 s** | −0.34 | +0.09 |
| Network v1 (one stop, TUM's compound) | −0.54 ± 0.03 | −7.4 ± 0.3 s | −0.40 | −0.66 |

It stops twice in 92 % of races (4 % are retirements before the first stop). Per driver (2 standard errors) it is better than the one-stop network for 15 of 20 drivers and worse for none, and better than TUM's VSE for 4 and worse for 6. **At Austin the network now beats TUM's VSE overall**, clearly in green races; with a safety car TUM's VSE is still slightly better (+0.09 ± 0.05), as in round 2.

## Next

1. Two-stop runs for the other tracks (Singapore: TUM two-stops in 18 % of races; Barcelona once its labels exist), then one network for all tracks with the second stop allowed everywhere.
2. Store every legal stop compound in the runs and choose lap and compound together (round 2: TUM's remaining safety-car edge).
3. Reacting background drivers (TUM's VSE for everyone else), so undercuts exist and reacting can pay.
4. More tracks before claiming anything about new tracks.

## Reproduce

```sh
.venv-train/Scripts/python.exe data_pipeline/train_pit_network.py                  # GPU if the CUDA build is installed
.venv-train/Scripts/python.exe data_pipeline/train_pit_network.py --holdout-track Monza --name pit_network_noMonza
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --runs 100 --jobs 28
.venv-data/Scripts/python.exe data_pipeline/summarise_evaluation.py integration/output/evaluation/pit_network_v1_mc100 --base network_tum
# round 2
.venv-train/Scripts/python.exe data_pipeline/train_stopping_network.py --iterations 4
.venv-train/Scripts/python.exe data_pipeline/stopping_headroom.py
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --model integration/models/pit_network_v2.npz --policies network network_tum --out integration/output/evaluation/pit_network_v2_mc100
.venv-tum/Scripts/python.exe integration/check_tum_as_fixed.py --track Spielberg --races 30
# round 3 (two stops, Austin)
.venv-tum/Scripts/python.exe integration/two_stop_generator.py --track Austin --runs 50 --jobs 28
.venv-train/Scripts/python.exe data_pipeline/train_two_stop_network.py --track Austin --iterations 4
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --model integration/models/pit_network_v3_two_stop.npz --tracks Austin --policies network_two --out integration/output/evaluation/pit_network_v3_austin_mc100
```
