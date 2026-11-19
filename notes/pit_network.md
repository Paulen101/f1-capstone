# Pit-timing network (methodology step 6)

Status: first network trained and raced in the simulator, with TUM's compound choice; round 2 (labels that value waiting) and round 3 (two stops, Austin) and round 4 (one two-stop network for all four tracks) round 5 (lap and compound together) round 6 (v4's timing with a compound head: the best policy so far) and round 7 (safety-car headroom: not worth training) and round 8 (Barcelona as a fifth track: v6 + compound head is now the best policy on five tracks) done, 2026-11-18. Builds on the race situations of [situation_sampling.md](situation_sampling.md).

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

**Labels and training** (`data_pipeline/two_stop_labels.py`, `train_two_stop_network.py --tracks Austin`): the exact replay of round 2 over two levels. In the second stint, stopping now is compared with following the network from the next lap, which may mean never stopping again; in the first stint a stop now is worth what following the network's second-stint decisions after it is worth. Started from v1 (no second stop), four iterations of policy iteration, 30 s on the GPU; runs 0–39 for training, 40–49 held out. Saved as `integration/models/pit_network_v3_two_stop.npz` ([report](evidence/pit_network/pit_network_v3_two_stop.json)).

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

## Round 4: one two-stop network for all four tracks

**Runs**: the round-3 generator for Singapore (762 plans per driver), Monza (486) and Spielberg (1,197), 50 runs each, 60 driver-races in 7.5 hours, none re-run and none failed. The second stop goes onto the soft (Singapore: A6 or A7; Monza: A4 or A6; Spielberg: A6, the last compound of the two drivers who stopped twice).

**Training** (`train_two_stop_network.py`, all four tracks by default): 233,000 first-stint and 2.94 million second-stint rows, the same policy iteration as round 3, 2 minutes on the GPU. Iteration 3 is best and is saved as `integration/models/pit_network_v4_two_stop.npz` ([report](evidence/pit_network/pit_network_v4_two_stop.json)).

| Held-out runs (40–49), places vs best fixed one-stop lap | Places | Two-stop share |
|---|---:|---:|
| Hindsight upper bound | −0.93 | |
| Best fixed one- or two-stop plan (chosen on training runs) | −0.15 ± 0.02 | 24 % |
| Start: v1, one stop | −0.01 ± 0.03 | 0 % |
| **Network v4** (iteration 3) | **−0.16 ± 0.03** | 28 % |

It matches the best fixed plan per driver. The fixed plans show why the overall gain is smaller than at Austin: a second stop only pays at Austin.

**In the simulator** (100 new races per driver-race, seeds from 500000, the same as every other policy; [evidence](evidence/pit_network/all_tracks_two_stop_mc100_summary.csv)):

| Mean position | Austin | Monza | Singapore | Spielberg | Two-stop share (A / M / S / S) |
|---|---:|---:|---:|---:|---|
| Real 2019 strategy | 9.52 | 9.52 | 9.52 | 9.52 | 60 / 37 / 36 / 9 % |
| TUM's supervised VSE | 9.14 | **8.41** | 9.20 | **9.02** | 64 / 0 / 18 / 0 % |
| Network v1, one stop, TUM's compound | 9.58 | 8.55 | 8.37 | 9.28 | 0 % |
| **Network v4, two stops allowed** | **9.05** | 8.57 | 8.26 | 9.29 | 88 / 3 / 0 / 3 % |

| Network v4 minus | Places | Race time | Green races | SC/VSC races |
|---|---:|---:|---:|---:|
| Real 2019 strategy | **−0.73 ± 0.02** | **−6.3 ± 0.2 s** | −0.74 | −0.72 |
| **TUM's VSE** | **−0.15 ± 0.02** | **−1.7 ± 0.1 s** | −0.43 | +0.08 |
| Network v1 (one stop, TUM's compound) | −0.15 ± 0.01 | −1.5 ± 0.1 s | −0.12 | −0.18 |
| Best fixed one-stop lap | −0.11 ± 0.01 | −1.7 ± 0.1 s | −0.07 | −0.14 |

| Network v4 minus TUM's VSE, by track | Places | Race time | Green | SC/VSC | Drivers better / worse (2 SE) |
|---|---:|---:|---:|---:|---|
| Austin | −0.10 ± 0.03 | −1.5 s | −0.29 | +0.07 | 4 / 8 |
| Monza | +0.16 ± 0.03 | +1.4 s | −0.11 | +0.38 | 0 / 6 |
| Singapore | −0.94 ± 0.05 | −9.3 s | −1.35 | −0.59 | 17 / 0 |
| Spielberg | +0.27 ± 0.03 | +2.2 s | +0.03 | +0.47 | 0 / 12 |

What this shows:

- **One network learns where a second stop pays.** It stops twice in 88 % of Austin races and almost never at the other three tracks. At Austin it equals the Austin-only v3 (v4 minus v3: +0.01 ± 0.02 places, +0.7 s), so training on all tracks costs nothing there.
- **Over all four tracks it now beats TUM's VSE** by 0.15 places and 1.7 s, where v1 tied it. It beats the real strategies for 40 of 80 driver-races and is worse for 2.
- **TUM's VSE still wins at Monza and Spielberg, and only in safety-car races** (+0.38 and +0.47 places; green races −0.11 and +0.03). Neither TUM nor the network two-stops there, so this is the round-2 finding again: TUM picks lap and compound together when a safety car comes out, and our runs hold one compound per driver.
- The network stops earlier than TUM at every track (first stop 16.4 vs 21.9 laps at Austin, 19.7 vs 27.1 at Singapore, 30.5 vs 33.9 at Spielberg).

## Round 5: choosing lap and compound together

Round 4 left TUM's VSE ahead only in safety-car races at Monza and Spielberg, where it picks lap and compound together. Every stored run so far stopped onto one compound per driver.

**Runs** (`integration/compound_runs.py`): a driver has three dry compounds and must use two, so a single stop can go onto one more compound than the stored one (the dry compound that is neither the start compound nor the stored stop compound; for example the soft A6 for most Spielberg drivers). For every driver and run (the same seeds 0–49), a stop on every lap onto that compound: 80 driver-races in 35 minutes, none re-run. Before the stop nothing differs from the stored runs, so the recorded first-stint situations apply to these races too. A check re-simulated 9 stored races with this code and the stored compound: all 9 identical.

**Replay with compound choice** (`data_pipeline/two_stop_labels.py`): the first-stint table gets one row per lap and compound. On each lap a policy predicts both compounds and takes the better one (`choose_compound`: best position advantage; within 0.1 places, best time). It pits if that one says pit. The forced last-lap stop and any second stop stay with the stored compound, because only those runs exist. The simulator adapter does the same (`NetworkVSE(other_compounds=...)`, `best_choice`), and a test checks that both rules agree.

**Training** (`train_two_stop_network.py`, default since this round): starts from v4 as it raced, on the stored compound only. Letting v4 choose between compounds from the start is worse than the fixed one-stop plan (+0.08 places): it was never trained on the other compound, picks it in 43 % of runs and loses its two-stop plans. Four iterations, 3 minutes on the GPU; iteration 3 is best and is saved as `integration/models/pit_network_v5_compound.npz` ([report](evidence/pit_network/pit_network_v5_compound.json)).

| Held-out runs (40–49), places vs best fixed one-stop lap on the stored compound | Places | Other compound |
|---|---:|---:|
| Hindsight upper bound | −1.00 | |
| Best fixed one-stop lap and compound | −0.07 ± 0.01 | 37 % |
| Best fixed plan of all (one or two stops, either compound) | −0.20 ± 0.02 | 35 % |
| Start: v4, stored compound | −0.16 ± 0.03 | 0 % |
| **Network v5** (iteration 3) | **−0.19 ± 0.04** | 35 % |

The offline gain over v4 is small (−0.03 ± 0.03 places), and the iterations swing (−0.06 to −0.19).

**In the simulator** (policy `network_choice`, 100 new races per driver-race, the same seeds as every other policy; [evidence](evidence/pit_network/all_tracks_compound_mc100_summary.csv)):

| Network v5 minus | Places | Race time | Green races | SC/VSC races |
|---|---:|---:|---:|---:|
| Real 2019 strategy | **−0.72 ± 0.02** | **−6.3 ± 0.2 s** | −0.74 | −0.69 |
| TUM's VSE | **−0.14 ± 0.02** | **−1.6 ± 0.2 s** | −0.43 | +0.11 |
| Network v4 (stored compound) | +0.02 ± 0.01 | +0.1 ± 0.1 s | +0.00 | +0.02 |

| By track | v5 − v4 places | v5 − v4 time | v5 − TUM places | v5 − TUM in SC/VSC races (v4) | Other compound: v5 / TUM | Two stops: v5 / v4 |
|---|---:|---:|---:|---:|---|---|
| Austin | +0.17 ± 0.02 | +2.2 s | +0.07 | +0.27 (+0.07) | 10 / 23 % | 73 / 88 % |
| Monza | −0.05 ± 0.02 | −0.6 s | +0.11 | +0.31 (+0.38) | 74 / 26 % | 1 / 3 % |
| Singapore | +0.09 ± 0.03 | +0.1 s | −0.85 | −0.49 (−0.59) | 0 / 12 % | 1 / 0 % |
| Spielberg | **−0.15 ± 0.02** | **−1.3 s** | +0.13 | +0.33 (+0.47) | 56 / 51 % | 1 / 3 % |

What this shows:

- **Choosing the compound helps where TUM's edge was.** At Spielberg the network now takes the other compound in 56 % of races (TUM: 51 %) and gains 0.15 places and 1.3 s on v4. At Monza it gains 0.05 places. In safety-car races TUM's lead shrinks from +0.47 to +0.33 places at Spielberg and from +0.38 to +0.31 at Monza, but it does not close.
- **Retraining costs as much elsewhere.** At Austin v5 makes two stops in 73 % of races instead of 88 %. A first stop onto the other compound (10 %) rules out a second stop, and the retrained timing is a little worse: +0.17 places. At Singapore, where it never changes compound, it is 0.09 places worse.
- **Overall: no change.** v5 is +0.02 ± 0.01 places against v4 and −0.14 against TUM's VSE (v4: −0.15). **v4 stays the best network**; v5 shows the compound gain is real but local.

## Round 6: v4's timing with a compound head

Round 5 retrained the whole network and lost at Austin and Singapore what the compound gained elsewhere. Here v4 stays exactly as it raced: it decides on the stored compound when to stop and whether to stop again. Only when it stops does a small second network, the compound head, choose the tyre.

**Head** (`data_pipeline/train_compound_head.py`, 1 minute on the GPU): the same 64-64 network shape. It sees the situation with the other compound's tyre features. Its label is exact on the stored runs: a stop onto the other compound minus a stop onto the stored compound followed by v4's second-stint decisions, in the same run and on the same lap. It is trained on all 233,000 first-stint laps of runs 0–39, and it switches when pit_now() says the other compound is better. Saved as `integration/models/pit_compound_head_v1.npz` ([report](evidence/pit_network/pit_compound_head_v1.json)). For comparison, a **per-driver rule** always or never switches, whichever was better on v4's stop laps in the training runs. It switches for 15 Monza drivers, 12 Spielberg drivers and 1 Singapore driver.

| Held-out runs (40–49), minus v4 | All | Austin | Monza | Singapore | Spielberg |
|---|---:|---:|---:|---:|---:|
| v4 + compound head | −0.04 ± 0.01 | 0 | −0.07 | 0 | −0.10 |
| v4 + per-driver rule | −0.05 ± 0.01 | 0 | −0.09 | −0.02 | −0.09 |
| v4 + better compound in hindsight (bound) | −0.07 ± 0.01 | 0 | −0.11 | −0.02 | −0.16 |
| v5 (round 5) | −0.03 ± 0.03 | +0.08 | −0.08 | −0.07 | −0.07 |

**In the simulator** (policies `network_head` and `network_rule`, 100 new races per driver-race, the same seeds as every other policy; [evidence](evidence/pit_network/all_tracks_head_mc100_summary.csv)):

| v4 + compound head minus | Places | Race time | Green races | SC/VSC races |
|---|---:|---:|---:|---:|
| Real 2019 strategy | **−0.78 ± 0.02** | **−6.8 ± 0.2 s** | −0.79 | −0.77 |
| **TUM's VSE** | **−0.20 ± 0.02** | **−2.2 ± 0.1 s** | −0.48 | +0.03 ± 0.03 |
| v4 (stored compound) | −0.05 ± 0.004 | −0.5 s | −0.05 | −0.05 |
| v5 (round 5) | −0.07 ± 0.01 | −0.6 s | −0.05 | −0.08 |
| v4 + per-driver rule | +0.007 ± 0.002 | +0.05 s | +0.005 | +0.009 |

| By track | − v4 | − TUM | − TUM, SC/VSC races | Drivers better / worse than TUM (2 SE) |
|---|---:|---:|---:|---|
| Austin | 0 | −0.10 | +0.07 | 4 / 8 |
| Monza | −0.07 ± 0.01 | +0.09 | +0.31 | 0 / 3 |
| Singapore | 0 | −0.94 | −0.59 | 17 / 0 |
| Spielberg | −0.13 ± 0.01 | +0.14 | +0.34 | 0 / 7 |

What this shows:

- **The head keeps every gain and loses nothing.** It changes nothing at Austin and Singapore and gains 0.13 places at Spielberg and 0.07 at Monza. That is 0.05 places and 0.5 s overall, better than v4 for 20 driver-races and worse for 2. **v4 + compound head is now the best policy:** 0.20 places and 2.2 s ahead of TUM's VSE and 0.78 places ahead of the real strategies.
- **A per-driver rule does as well.** It is even 0.007 places better than the head. With one stored compound and one other per driver, choosing the tyre comes down to "this driver should use the other tyre at this track". The head learns that and little more.
- **The compound was not TUM's safety-car edge.** The compound gains as much in green races as in safety-car races (−0.05 each). TUM still leads by +0.31 and +0.34 places in safety-car races at Monza and Spielberg. TUM's VSE makes its first stop under a safety car or VSC in 10–12 % of races; v4 does in 3–5 %. What remains is **when** to stop around a safety car. Round 2's labels could not reward waiting for one, because one compound per driver left nothing to gain. With both compounds now in the runs, that can be retested.

## Round 7: is safety-car timing worth training?

After round 6, TUM's VSE still leads in safety-car races at Monza (+0.31 places) and Spielberg (+0.34). In those races it stops under a safety car or VSC in 22 % and 18 % of races; v4 + compound head does in 6 % and 5 %. In green races TUM stops later (Monza lap 26 vs 24, Spielberg 36 vs 30), which keeps the window open for a safety car. Before training anything, `data_pipeline/safety_car_headroom.py` measures what reacting to safety cars could add, exactly on the stored runs with both compounds, against v4 + compound head on the held-out runs 40–49 (38 % of them have a safety car or VSC before the last lap; [table](evidence/pit_network/safety_car_headroom.csv), [rule grid](evidence/pit_network/safety_car_headroom_grid.json)):

| Minus v4 + compound head (held-out) | All races | SC/VSC races |
|---|---:|---:|
| Hindsight, any plan (bound for any policy) | −0.80 ± 0.04 | −1.17 ± 0.08 |
| Hindsight, changing only what it does on SC/VSC laps | −0.16 ± 0.02 | −0.45 ± 0.05 |
| … stop under an SC before the planned stop | −0.08 | −0.21 |
| … wait past the planned stop for a later SC | −0.06 | −0.17 |
| … second stop under an SC | −0.04 | −0.11 |
| Rule: stop at once under SC/VSC once progress ≥ p (best p = 0.2) | 0.00 | 0.00 |
| Rule: second stop under SC/VSC with ≥ r laps left (best: never) | 0.00 | 0.00 |
| Rule: when v4 says pit, wait up to k laps for an SC (best k = 1) | +0.01 ± 0.03 | −0.05 ± 0.06 |
| Rule: under SC/VSC, pit if v4's predicted advantage < θ (best θ = 0) | +0.02 ± 0.01 | +0.05 ± 0.02 |

Rule settings were chosen on the training runs 0–39.

What this shows:

- **The bound is real but needs hindsight.** Reaching −0.45 places in safety-car races means knowing *which* safety car is worth a stop (how long it lasts, where the car rejoins) or that one is coming. No rule that only knows the present gains anything on held-out runs. Waiting longer, or lowering the bar under a safety car, gets steadily worse (holding 10 laps: +0.57 places).
- **Per track it does not hold up either.** Waiting 1–4 laps looks slightly better at Austin and Spielberg (−0.02 to −0.09 ± 0.05 on held-out runs) and worse at Monza and Singapore. Choosing the wait per track on the training runs gives Austin −0.04, Monza +0.04, Singapore 0 and Spielberg +0.06 on the held-out runs: no gain.
- **v4 already knows about safety cars.** It sees the SC and VSC flags, and policy iteration trained it on exact labels for stopping under them. What these situation features can say about safety cars is already in it. **A separate safety-car round is not worth training.** TUM's remaining lead at Monza and Spielberg is a different trade-off (later stops that pay off when a safety car comes) that our replay does not find better on average. Testing that needs either more runs per driver or features describing the safety car itself (laps since it came out, the field's gaps under it).

## Round 8: Barcelona as a fifth track

Barcelona 2019 had no labels: every top finisher stopped twice. With the two-stop generator it can now be added.

**Race file fix.** The fitted Barcelona file burned 1.667 kg/lap over 66 laps, 110.02 kg from a 110 kg tank, so every Barcelona race without a safety car ran dry on the last lap and the simulator stopped with an error. `write_fitted_pars.py` already lowers the burn in that case (added for Monza in a8afaad), but the Barcelona file was written before that. Rewritten from the same fit: `b_fuel_perlap` 1.666, `t_lap_sens_mass` 0.03479 → 0.03481, `t_gap_racepace` 4.266 → 4.264; nothing else changed.

**Unseen-track test first.** Before any Barcelona training, v4 + compound head raced there (`catalunya_unseen_v4_mc100`, the same seeds as every evaluation): **+0.77 ± 0.04 places and +12.3 s behind TUM's VSE**, +0.88 places behind the real strategies, worse than TUM for 19 of 20 drivers. It stopped twice in 7 % of races (real: 78 %). The compound head made it worse there (+0.16 places against v4 alone). This is the leave-one-track-out finding again, now on a real fifth race.

**Runs**: `two_stop_generator.py --track Catalunya --runs 50` (967 plans per driver: 64 one-stop, 903 two-stop; 48,350 races per driver), 20 drivers in 2 h 55 min, none re-run; then `compound_runs.py --tracks Catalunya` (9 min). The second stop goes onto each driver's real last compound (most of the field stopped twice).

**Training**, as v4: `train_two_stop_network.py` on all five tracks, starting from v1, stored compound, 4 iterations (297,000 first-stint and 3.88 million second-stint rows, 3 minutes on the GPU). Iteration 4 is best; a 6-iteration run gives identical iterations 1–4 and worse 5–6 (−0.118 and −0.072 vs the fixed one-stop), so 4 stays. Saved as `integration/models/pit_network_v6_five_tracks.npz` ([report](evidence/pit_network/pit_network_v6_five_tracks.json)). Then the compound head on v6 (`train_compound_head.py --timing pit_network_v6_five_tracks.npz`, `integration/models/pit_compound_head_v2.npz`, [report](evidence/pit_network/pit_compound_head_v2.json); its log calls the timing network "v4").

| Held-out runs (40–49), places | All five tracks |
|---|---:|
| Best fixed plan minus best fixed one-stop | −0.149 ± 0.022 |
| v6 minus best fixed one-stop | −0.123 ± 0.034 |
| v6 minus best fixed plan | +0.026 ± 0.031 |
| v6 + head minus v6 | −0.016 ± 0.010 (Barcelona +0.038, Spielberg −0.087) |

**In the simulator** (policies `network_two` and `network_head`, 100 new races per driver-race, the same seeds as every other policy; races of real, TUM and v4 + head reused from rounds 6 and the unseen test; [summary](evidence/pit_network/all_five_tracks_mc100_summary.csv), [by driver](evidence/pit_network/all_five_tracks_mc100_by_driver.csv)):

| v6 + head minus | Places | Race time | Green races | SC/VSC races |
|---|---:|---:|---:|---:|
| Real 2019 strategy | **−0.62 ± 0.02** | **−5.1 s** | −0.63 | −0.61 |
| **TUM's VSE** | **−0.18 ± 0.02** | **−1.2 s** | −0.40 | +0.00 ± 0.02 |
| v4 + head (round 6; unseen at Barcelona) | −0.17 ± 0.01 | −1.6 s | −0.17 | −0.18 |
| v6 alone | −0.04 ± 0.003 | −0.4 s | −0.04 | −0.04 |

| By track | − TUM places | − TUM time | − TUM, SC/VSC | − v4 + head | Drivers better / worse than TUM (2 SE) | Two-stop share: v6+head / TUM / real |
|---|---:|---:|---:|---:|---|---|
| **Barcelona** | **−0.23 ± 0.03** | −0.2 ± 0.3 s | −0.30 | **−1.00 ± 0.04** (−12.5 s) | 6 / 5 | 70 / 19 / 78 % |
| Austin | −0.11 | −1.5 s | +0.08 | −0.01 | 4 / 7 | 92 / 64 / 60 % |
| Monza | +0.11 | +0.9 s | +0.31 | +0.02 | 0 / 3 | 1 / 0 / 37 % |
| Singapore | −0.82 | −6.9 s | −0.44 | **+0.12 ± 0.03** | 14 / 0 | 3 / 18 / 36 % |
| Spielberg | +0.16 | +1.6 s | +0.36 | +0.02 | 0 / 5 | 4 / 0 / 9 % |

What this shows:

- **Barcelona goes from the worst track to a win.** Trained on it, the network stops twice in 70 % of races (real 78 %) and is 1.00 places and 12.5 s better than the network that had never seen it, better for 15 of 20 drivers. It now beats TUM's VSE there by 0.23 places, in safety-car races too (−0.30); on race time the two are level (−0.2 ± 0.3 s). TUM stops twice in only 19 % of Barcelona races.
- **The other four tracks barely move.** Austin, Monza and Spielberg are within ±0.02 places of v4 + head. Singapore loses 0.12 places (TUM lead −0.94 → −0.82; 3 drivers worse, none better). On the old four tracks v6 + head is −0.16 ± 0.02 against TUM, v4 + head −0.20: one network for five tracks costs 0.04 places there.
- **Over five tracks it beats TUM's VSE by 0.18 places and 1.2 s** and the real strategies by 0.62 places. The safety-car split is unchanged in kind: clearly ahead in green races (−0.40), level with a safety car, where TUM still leads at Monza (+0.31) and Spielberg (+0.36).
- **The compound head again adds a little** (−0.04 places, at Spielberg and Monza) and costs nothing at Barcelona in the simulator (−0.006), although it was slightly worse there on the held-out runs.
- **A new track still needs its own runs.** Five tracks are not yet evidence that the network transfers; that needs a leave-one-out test on five (or more) tracks (next section).

### Leave one track out, on five tracks

Does the network work on a track it has never raced, and does a fifth training track help? For each of the four old tracks X, two two-stop networks were trained exactly like v4/v6 but without X: on the other three old tracks (`pit_network_v4_noX`) and on those plus Barcelona (`pit_network_v6_noX`). Each starts from the one-stop network trained without X (`pit_network_noX`, round 1), so nothing about X reaches it. For Barcelona, v4 itself never saw it. Each raced its held-out track (policy `network_two`, timing only, no compound head; 100 races per driver, the same seeds as every evaluation). About 25 minutes in total. [Table](evidence/pit_network/leave_one_track_out_five.csv).

| Places minus TUM's VSE (race time) | Seen (v6) | Unseen, trained on 3 tracks | Unseen, trained on 4 tracks (incl. Barcelona) |
|---|---:|---:|---:|
| Austin | −0.11 (−1.5 s) | +0.97 (+11.4 s) | +0.92 (+11.8 s) |
| Monza | +0.15 (+1.6 s) | **+0.53** (+5.4 s) | +0.99 (+7.5 s) |
| Singapore | −0.81 (−6.9 s) | +1.62 (+18.0 s) | **−0.78 (−7.1 s)** |
| Spielberg | +0.30 (+2.7 s) | +1.52 (+12.4 s) | +1.37 (+12.6 s) |
| Barcelona | −0.22 (−0.2 s) | | +0.61 (+9.2 s; v4) |
| **Old four tracks together** | | **+1.16 ± 0.02** | **+0.62 ± 0.02** |

| Unseen: two-stop share / mean first stop (lap) | 3 tracks | 4 tracks | Seen (v6) |
|---|---|---|---|
| Austin | 38 % / 12.4 | 94 % / **5.2** | 92 % / 16.3 |
| Monza | 0 % / 30.8 | 4 % / 15.2 | 5 % / 25.6 |
| Singapore | 94 % / **3.2** | 11 % / 18.3 | 3 % / 17.4 |
| Spielberg | 94 % / **5.5** | 85 % / 13.2 | 7 % / 27.3 |

What this shows:

- **On a track it has never raced the network is still clearly worse than TUM's VSE**, by 0.5 to 1.5 places, in green and safety-car races alike, at every track but one. TUM's VSE was trained on many seasons and tracks; ours on three or four.
- **The fifth track halves the average gap** (+1.16 → +0.62 places over the four old tracks), but almost all of that is Singapore: there the 4-track network is as good as the one trained on Singapore (−0.78 vs −0.81). At Austin and Spielberg it helps by 0.05–0.15 places, and at **Monza it is worse** (+0.53 → +0.99): from Barcelona it learns earlier stops (lap 15 vs 31) that are wrong at Monza.
- **The failures are implausible plans, not small timing errors**: a first stop on lap 3–5 and two stops where nobody needs them (Spielberg 85–94 %). On an unseen track the track inputs (location category, pit loss, tyre features) take values outside what it trained on, and the network extrapolates badly.
- **So a new track still needs its own runs.** Four or five tracks do not teach the network what a track does; whether it ever transfers is a question for many more tracks (2022–2025), not this data set. A cheaper safeguard for unseen tracks would be to keep its decisions within plausible windows (no stop before a minimum stint, two stops only where a fixed plan shows they pay); that has not been tried.

### Plausibility limits on unseen tracks, and the race-file plan

The unseen networks fail with implausible plans. Limits that need nothing but the new track's race file:

- **Race-file plan** (`integration/race_file_plan.py`): for every driver, the cheapest one-stop and two-stop plan by the race file's own degradation model plus the stop cost (pit-lane drive, tyre change, team's extra change time, cold-tyre out-lap). Fuel is the same for every plan and traffic is ignored, so no simulation is needed (milliseconds per track). The compounds and windows are those of the two-stop runs. It gets the stop count right at four of five tracks: two stops faster for 20/20 drivers at Austin (median +7.1 s), for 0/20 at Monza, Singapore and Spielberg (−9 to −17 s); Barcelona is borderline (4/20, median −2.1 s), where the simulator says two stops pay.
- **Limits** (`NetworkVSE(limits=...)`): each stop must fall in a window around the plan's stop, from `early` before to `late` after (fractions of the race); the network cannot stop before the window and is forced to stop at its end. A second stop only for drivers whose race-file plan has two stops.
- **Widths from the other tracks only** (`race_file_plan.py --widths`, [widths](evidence/pit_network/plausibility_widths.json)): the 5th and 95th percentile of the seen network v6's first stop minus the race-file plan's, pooled over the four tracks other than the held-out one. The seen network stops earlier than the plan, rarely later: early 0.20–0.26, late 0.02–0.06 of the race. (v6 trained on the held-out track too; its stops elsewhere are only a second-order leak.)

Each hidden-track network of the leave-one-out test raced again with limits (`network_limited`), and the race-file plan raced alone as fixed stops (`race_file_plan`); same seeds as every evaluation ([table](evidence/pit_network/plausibility_limits.csv)):

| Places minus TUM's VSE | Austin | Monza | Singapore | Spielberg | Barcelona | Old four together |
|---|---:|---:|---:|---:|---:|---:|
| Seen (v6) | −0.11 | +0.15 | −0.81 | +0.30 | −0.22 | −0.12 |
| Unseen, 3 tracks | +0.97 | +0.53 | +1.62 | +1.52 | | +1.16 |
| … with limits | +0.21 | +0.14 | +0.32 | +0.67 | | **+0.33** |
| Unseen, 4 tracks | +0.92 | +0.99 | −0.78 | +1.37 | +0.61 | +0.62 |
| … with limits | +0.88 | +0.88 | −0.85 | +0.72 | +0.50 | **+0.41** |
| **Race-file plan alone** | **−0.17** | +0.14 | −0.75 | +0.21 | **−0.23** | **−0.14** |

| Race-file plan minus | Places | Race time | Green | SC/VSC |
|---|---:|---:|---:|---:|
| **v6 + compound head, five tracks (best trained policy, seen)** | **+0.02 ± 0.01** | **−1.0 s** | +0.04 | +0.00 |
| TUM's VSE | −0.16 ± 0.01 | −2.2 s | −0.36 | +0.00 |
| Real 2019 strategy | −0.60 ± 0.02 | −6.1 s | −0.60 | −0.61 |

What this shows:

- **The limits help, but not enough.** They cut the unseen networks' loss to TUM from +1.16 to +0.33 places (3 tracks) and from +0.62 to +0.41 (4 tracks). The windows are wide on the early side (up to a quarter of the race, because the seen network often stops well before the plan), so at Austin and Monza the 4-track network still stops on lap 5 and 15. The limited networks still lose to TUM's VSE.
- **The race-file plan alone beats every unseen network and TUM's VSE**: −0.14 places over the old four tracks and −0.23 at Barcelona, without the network and without a single Monte Carlo run.
- **It is as good as the best trained network on the tracks the network trained on**: +0.02 ± 0.01 places and 1.0 s *faster* than v6 + compound head over five tracks, level in safety-car races. On these five tracks the network's gain over TUM comes from knowing the race file's tyre model, which the race-file plan uses directly. Its own contribution beyond that (reacting to the race as it unfolds) is about zero in position.
- **Caveat: the race-file plan knows the simulator's tyre model exactly.** The simulator runs on the same fitted degradation model the plan optimises, fitted on laps of the same 2019 race. On a real future race the tyre model would come from practice or an earlier year and be uncertain, and a plan built on it would be worse; TUM's VSE never sees the race file's tyres. The same holds, less directly, for our networks (their tyre inputs come from the race file). So this does not show that a trivial plan beats TUM in reality; it shows that **in this simulator, the network's lead over TUM is the tyre model, not learned race-craft.**

### A wrong tyre model: how much of the lead survives?

The race-file plan and the network's tyre inputs read the same degradation model the simulator races on. A real team only has an estimate. Here the strategy believes a wrong model while the simulator keeps the true one (`integration/tyre_uncertainty.py`, `evaluate_pit_network.py --tyre-error`, `NetworkVSE(believed_tires=...)`):

- per race and compound, shared by the whole field (one estimate, as from practice running): `k_0` + N(0, σ₀) s and every wear term × exp(N(0, σ_w))
- **moderate**: σ₀ = 0.3 s, σ_w = 0.3 (about the fit's own standard errors: k0 0.2–0.7 s, wear terms 20–50 %); **large**: σ₀ = 0.6 s, σ_w = 0.6 (the upper end; wear fitted without late-stint laps was off by far more, [heldout_check.md](heldout_check.md))
- drawn from (track, run, level), so both policies believe the same wrong model in the same race; TUM's VSE, the real strategies and the background drivers do not read the race file's tyres and are unchanged. A check re-raced 100 races with the exact model: positions, race times and stops identical to the earlier runs.

Policies: v6 + compound head and the race-file plan, 100 races per driver-race on all five tracks per level, same seeds ([table](evidence/pit_network/tyre_uncertainty.csv)):

| Minus TUM's VSE | Places | Race time | Green | SC/VSC | Minus real | Austin | Barcelona | Monza | Singapore | Spielberg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v6 + head, exact tyres | **−0.18 ± 0.02** | −1.2 s | −0.40 | +0.00 | −0.62 | −0.11 | −0.23 | +0.11 | −0.82 | +0.16 |
| race-file plan, exact | **−0.16 ± 0.01** | −2.2 s | −0.36 | +0.00 | −0.60 | −0.17 | −0.23 | +0.14 | −0.75 | +0.21 |
| v6 + head, moderate error | +0.27 ± 0.02 | +3.4 s | +0.03 | +0.48 | −0.17 | +0.37 | +0.26 | +0.19 | +0.16 | +0.39 |
| race-file plan, moderate | +0.19 ± 0.02 | +1.5 s | +0.00 | +0.34 | −0.25 | +0.23 | −0.11 | +0.33 | −0.15 | +0.65 |
| v6 + head, large error | +0.78 ± 0.02 | +7.3 s | +0.62 | +0.91 | +0.33 | +1.22 | +0.52 | +0.64 | +0.94 | +0.56 |
| race-file plan, large | +0.74 ± 0.02 | +6.8 s | +0.59 | +0.86 | +0.30 | +0.94 | +0.32 | +1.07 | +0.29 | +1.08 |

| Wrong minus exact tyres (same policy) | Moderate | Large |
|---|---:|---:|
| v6 + head | +0.45 ± 0.02 places, +4.4 s | +0.95 ± 0.02, +8.0 s |
| race-file plan | +0.35 ± 0.01, +3.7 s | +0.90 ± 0.02, +9.0 s |
| race-file plan minus v6 + head | −0.09 ± 0.02, −1.8 s | −0.04 ± 0.02, −0.8 s |

What this shows:

- **The lead over TUM's VSE does not survive a realistic tyre error.** With errors about the size of the fit's own standard errors, both policies lose about 0.4 places and fall behind TUM: the network by 0.27 places, the race-file plan by 0.19. In green races they are level with TUM; with a safety car TUM leads by 0.3–0.5. With large errors both are 0.75 places behind TUM and worse than the real strategies.
- **The network is not more robust than the plan; slightly less.** It loses 0.45 places to the error against the plan's 0.35, and the plan is ahead of it under both error levels (−0.09 and −0.04 places). The network learned to trust its tyre inputs; nothing in its training rewarded doubting them.
- **What the wrong model does**: at Austin the network's two-stop share falls from 92 % to 59 % (moderate) and 37 % (large), at Barcelona from 70 % to 46 % and 36 %; a wear rate believed too low makes long stints look cheap.
- **This completes the picture of rounds 6–8.** Our policies' lead over TUM in this simulator is the exact tyre model they share with it. TUM's VSE never had that model and is unaffected; against it, the honest comparison is the "moderate" row.

### Training for a wrong tyre model (network v7)

The network trusted its tyre inputs completely, because in training they were always exact. Here it trains on wrong ones: every stored run gets its own wrong tyre model at the **moderate** level (the error model above, drawn per track and run with a different key from the evaluation races, so no evaluation race repeats a training error). Only the inputs change; the runs and their outcomes are those of the true model, so the labels stay exact. The policy-iteration replay also decides on the wrong inputs, so it scores exactly what a strategy with a wrong model would get.

- `add_tyre_features(rows, tyre_error)` (`data_pipeline/train_pit_network.py`): the error is a pace offset plus a wear factor per compound, so each tyre input is linear in it. The inputs are split once into pace and wear parts and each run's error is applied with arrays; `data_pipeline/test_tyre_error_features.py` checks this against the simulator-side calculation (`tyre_uncertainty.believed_tires` + `tyre_features`) to 1e-9.
- **v7** (`train_two_stop_network.py --tyre-error moderate`, otherwise as v6: five tracks, start v1, stored compound, 4 iterations; iteration 2 best; [report](evidence/pit_network/pit_network_v7_robust.json)) and **compound head v3** (`train_compound_head.py --tyre-error moderate --timing pit_network_v7_robust.npz`; [report](evidence/pit_network/pit_compound_head_v3_robust.json)). On held-out runs with wrong inputs, v1 loses 0.25 places to the error (+0.26 vs the fixed one-stop instead of +0.01); v7 is 0.31 places better than v1 there.

Raced as v7 + head v3 (`network_head`) at every error level, the same seeds and the same error draws as the previous section ([table](evidence/pit_network/robust_training.csv)):

| Minus TUM's VSE (places) | Exact tyres | Moderate error | Large error |
|---|---:|---:|---:|
| v6 + head (trained on exact tyres) | **−0.18 ± 0.02** | +0.27 ± 0.02 | +0.78 ± 0.02 |
| **v7 + head (trained on moderate errors)** | −0.08 ± 0.02 | **−0.07 ± 0.02** | **−0.02 ± 0.02** |
| race-file plan | −0.16 ± 0.01 | +0.19 ± 0.02 | +0.74 ± 0.02 |
| v7 + head minus v6 + head | +0.10 ± 0.01 | **−0.34 ± 0.02** (−3.7 s) | **−0.80 ± 0.02** (−7.1 s) |
| v7 + head minus race-file plan | +0.08 ± 0.01 | **−0.26 ± 0.01** (−1.9 s) | **−0.76 ± 0.02** (−6.6 s) |

| v7 + head, moderate error, minus TUM | Places | Race time | Green | SC/VSC | Austin | Barcelona | Monza | Singapore | Spielberg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | −0.07 ± 0.02 | −0.4 s | −0.32 | +0.14 | +0.22 | −0.17 | +0.21 | −0.78 | +0.18 |

What this shows:

- **Training on wrong tyres makes the network almost blind to the error.** From exact tyres to large errors it moves by 0.06 places (v6: 0.96, race-file plan: 0.90). It was trained only at the moderate level and still holds at the large one.
- **With a realistic error it is the only policy ahead of TUM's VSE**: −0.07 ± 0.02 places, 0.34 ahead of v6 and 0.26 ahead of the race-file plan. Ahead in green races (−0.32), behind with a safety car (+0.14), as before.
- **The price is the exact-tyre case**: 0.10 places worse than v6 when the tyres are known exactly. It stops twice less often (Austin is where it loses: +0.22 against TUM, v6 −0.11), because a two-stop plan relies on believing a high wear rate.
- **What it probably learned**: to lean less on the tyre inputs and more on the track inputs (location category, pit loss), which identify the five tracks it trained on. That would make it robust on these tracks but no help on a new one; this has not been tested.
- **Overall**: with tyre knowledge that is as uncertain as our own fit, the robust network keeps a small lead over TUM's VSE (−0.07 places), where every policy trained or planned on exact tyres falls behind. That, not the −0.18 of v6 with exact tyres, is the fair headline against TUM.

### Is the robust network memorising the tracks?

The leave-one-out test again, now with the robust training: for each of the five tracks, a network trained like v7 (moderate tyre errors, 4 iterations) on the other four, starting from the one-stop network without that track (`pit_network_noX`; v1 for Barcelona), saved as `pit_network_v7_noX`. Each raced its hidden track with moderate and with exact tyres; the earlier exact-trained hidden-track networks (`pit_network_v6_noX`, v4 for Barcelona) raced it again with moderate tyres. About 20 minutes ([table](evidence/pit_network/leave_one_track_out_robust.csv)).

| Places minus TUM's VSE, moderate tyre error | Austin | Monza | Singapore | Spielberg | Barcelona | All five |
|---|---:|---:|---:|---:|---:|---:|
| Robust v7, track seen | +0.22 | +0.21 | −0.78 | +0.18 | −0.17 | **−0.07 ± 0.02** |
| Robust, track hidden | +1.04 | +0.75 | +0.45 | +0.84 | +0.29 | **+0.67 ± 0.02** |
| Exact-trained, track hidden | +1.38 | +0.98 | −0.21 | +1.30 | +0.45 | +0.78 ± 0.02 |
| Race-file plan | +0.23 | +0.33 | −0.15 | +0.65 | −0.11 | +0.19 ± 0.02 |

With exact tyres: robust hidden +0.63, exact-trained hidden +0.62, race-file plan −0.16, robust v7 seen −0.08.

What this shows:

- **Yes: the robust network's lead comes from knowing the tracks.** On a hidden track it is 0.67 places behind TUM's VSE, against 0.07 ahead on the tracks it trained on. Robust training made the hidden-track result only slightly less bad (+0.78 → +0.67 with a tyre error) and not at all with exact tyres (+0.63 vs +0.62).
- **It is robust, but robustly wrong.** The hidden-track networks barely react to the tyre error (+0.63 exact, +0.67 moderate), because they learned to lean on the tyre inputs less; without the track inputs it knows, nothing tells them what a track needs. They stop twice where it does not pay (Monza 31 %, Spielberg 45 %) and hardly where it does (Austin 12 %, Barcelona 0 %).
- **On a new track with uncertain tyres the race-file plan is our best option** (+0.19), still behind TUM's VSE. TUM's VSE, trained on many seasons and tracks, is the better strategy for a track neither of our approaches has data for.
- **The fair summary of the project so far**: on tracks with our own Monte Carlo runs, the robust network beats TUM's VSE by 0.07 ± 0.02 places with realistic tyre uncertainty (0.32 in green races; TUM is better with a safety car). It does not transfer to new tracks; more tracks are the only route to that.

## Next

1. More tracks (2022–2025 seasons): the only route to a network that transfers; each needs its own runs (about 3 hours per track).
2. Reacting background drivers (TUM's VSE for everyone else), so safety-car and undercut reactions can pay; keep a race-file-plan baseline and the tyre-error rows in every table.
3. If a new track has to be raced without runs: the race-file plan, not the network.

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
.venv-train/Scripts/python.exe data_pipeline/train_two_stop_network.py --tracks Austin --start integration/models/pit_network_v1.npz --compounds stored --name pit_network_v3_two_stop --iterations 4
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --model integration/models/pit_network_v3_two_stop.npz --tracks Austin --policies network_two --out integration/output/evaluation/pit_network_v3_austin_mc100
# round 4 (two stops, all tracks); then append its runs.csv to pit_network_v1_mc100/runs.csv as all_tracks_two_stop_mc100/runs.csv
.venv-tum/Scripts/python.exe integration/two_stop_generator.py --track Singapore --runs 50 --jobs 28        # and Monza, Spielberg
.venv-train/Scripts/python.exe data_pipeline/train_two_stop_network.py --start integration/models/pit_network_v1.npz --compounds stored --name pit_network_v4_two_stop --iterations 4
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --model integration/models/pit_network_v4_two_stop.npz --policies network_two --out integration/output/evaluation/pit_network_v4_mc100
.venv-data/Scripts/python.exe data_pipeline/summarise_evaluation.py integration/output/evaluation/all_tracks_two_stop_mc100 --base network_two
# round 5 (compound choice); then append its runs.csv to those of v1 and v4 as all_tracks_compound_mc100/runs.csv
.venv-tum/Scripts/python.exe integration/compound_runs.py --runs 50 --jobs 28
.venv-train/Scripts/python.exe data_pipeline/train_two_stop_network.py --iterations 4
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --model integration/models/pit_network_v5_compound.npz --policies network_choice --out integration/output/evaluation/pit_network_v5_mc100
.venv-data/Scripts/python.exe data_pipeline/summarise_evaluation.py integration/output/evaluation/all_tracks_compound_mc100 --base network_choice
# round 6 (compound head on v4); then append its runs.csv as all_tracks_head_mc100/runs.csv
.venv-train/Scripts/python.exe data_pipeline/train_compound_head.py
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --model integration/models/pit_network_v4_two_stop.npz --policies network_head network_rule --out integration/output/evaluation/pit_compound_head_v1_mc100
.venv-data/Scripts/python.exe data_pipeline/summarise_evaluation.py integration/output/evaluation/all_tracks_head_mc100 --base network_head
# round 7 (safety-car headroom)
.venv-train/Scripts/python.exe data_pipeline/safety_car_headroom.py
# round 8 (Barcelona); all_five_tracks_mc100/runs.csv = real, tum, network_head (renamed v4_head) of
# all_tracks_head_mc100 and catalunya_unseen_v4_mc100, plus pit_network_v6_mc100 (network_two -> v6, network_head -> v6_head)
.venv-tum/Scripts/python.exe integration/write_fitted_pars.py --fit notes/evidence/fitted_parameters/Catalunya_2019_fit.json
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --model integration/models/pit_network_v4_two_stop.npz --tracks Catalunya --policies real tum network_two network_head --out integration/output/evaluation/catalunya_unseen_v4_mc100
.venv-tum/Scripts/python.exe integration/two_stop_generator.py --track Catalunya --runs 50 --jobs 28
.venv-tum/Scripts/python.exe integration/compound_runs.py --tracks Catalunya --runs 50 --jobs 28
.venv-train/Scripts/python.exe data_pipeline/train_two_stop_network.py --tracks Austin Monza Singapore Spielberg Catalunya --start integration/models/pit_network_v1.npz --compounds stored --name pit_network_v6_five_tracks --iterations 4
.venv-train/Scripts/python.exe data_pipeline/train_compound_head.py --tracks Austin Monza Singapore Spielberg Catalunya --timing integration/models/pit_network_v6_five_tracks.npz --name pit_compound_head_v2
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --model integration/models/pit_network_v6_five_tracks.npz --tracks Spielberg Monza Singapore Austin Catalunya --policies network_two network_head --compound-head integration/models/pit_compound_head_v2.npz --out integration/output/evaluation/pit_network_v6_mc100
.venv-data/Scripts/python.exe data_pipeline/summarise_evaluation.py integration/output/evaluation/all_five_tracks_mc100 --base v6_head
# round 8, leave one track out: for X in Austin Monza Singapore Spielberg, SET v4 (other old tracks) and v6 (plus Catalunya)
.venv-train/Scripts/python.exe data_pipeline/train_two_stop_network.py --tracks <other tracks> --start integration/models/pit_network_noX.npz --compounds stored --name pit_network_SET_noX --iterations 4
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --model integration/models/pit_network_SET_noX.npz --tracks X --policies network_two --out integration/output/evaluation/loo_SET_noX_mc100
# round 8, plausibility limits and the race-file plan (Catalunya: --model pit_network_v4_two_stop.npz)
.venv-tum/Scripts/python.exe integration/race_file_plan.py
.venv-tum/Scripts/python.exe integration/race_file_plan.py --widths integration/output/evaluation/all_five_tracks_mc100/runs.csv --policy v6
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --model integration/models/pit_network_v6_noX.npz --tracks X --policies network_limited race_file_plan --out integration/output/evaluation/limits_v6_noX_mc100
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --model integration/models/pit_network_v4_noX.npz --tracks X --policies network_limited --out integration/output/evaluation/limits_v4_noX_mc100
# round 8, robust network v7 (trained on moderate tyre errors), raced at LEVEL exact, moderate and large
.venv-train/Scripts/python.exe data_pipeline/train_two_stop_network.py --tracks Austin Monza Singapore Spielberg Catalunya --start integration/models/pit_network_v1.npz --compounds stored --name pit_network_v7_robust --iterations 4 --tyre-error moderate
.venv-train/Scripts/python.exe data_pipeline/train_compound_head.py --tracks Austin Monza Singapore Spielberg Catalunya --timing integration/models/pit_network_v7_robust.npz --name pit_compound_head_v3_robust --tyre-error moderate
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --model integration/models/pit_network_v7_robust.npz --tracks Spielberg Monza Singapore Austin Catalunya --policies network_head --compound-head integration/models/pit_compound_head_v3_robust.npz --tyre-error LEVEL --out integration/output/evaluation/robust_v7_LEVEL_mc100
# round 8, robust leave one track out: for X in all five (start pit_network_noX, v1 for Catalunya)
.venv-train/Scripts/python.exe data_pipeline/train_two_stop_network.py --tracks <other four> --start integration/models/pit_network_noX.npz --compounds stored --name pit_network_v7_noX --iterations 4 --tyre-error moderate
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --model integration/models/pit_network_v7_noX.npz --tracks X --policies network_two --tyre-error LEVEL --out integration/output/evaluation/loo_robust_noX_LEVEL_mc100
# round 8, wrong tyre model (LEVEL moderate and large)
.venv-tum/Scripts/python.exe integration/evaluate_pit_network.py --model integration/models/pit_network_v6_five_tracks.npz --tracks Spielberg Monza Singapore Austin Catalunya --policies network_head race_file_plan --compound-head integration/models/pit_compound_head_v2.npz --tyre-error LEVEL --out integration/output/evaluation/tyre_error_LEVEL_mc100
```
