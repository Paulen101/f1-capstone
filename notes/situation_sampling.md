# Race situation sampling (methodology step 4)

Status: done for the four one-stop 2019 races, 2026-11-17 (results at the end). Builds on the Monte Carlo label generator ([label_generator.md](label_generator.md)).

## Goal

The pit-timing network decides lap by lap, like TUM's VSE: at the end of every lap it sees the race situation and answers pit or stay. Its training data therefore has to be rows of the form

> situation at lap L (tyre age, race progress, position, gaps, safety car, ...) → how good is pitting now, compared with pitting later?

The label generator so far produces one curve per driver and race: the mean outcome of pitting on each lap, from lap 1. Step 4 turns that into per-situation rows.

## How a situation is sampled

A situation is (race, target driver, Monte Carlo run r, decision lap L), with the target driver still on its first stint at L.

The label generator already contains these. In run r every candidate pit lap uses the same seed, so the race is identical for every candidate up to the lap the target driver stops: the same safety cars, retirements and lap-time noise. A situation at lap L in run r is therefore shared by all candidates L' ≥ L, and their outcomes are paired continuations of the same state:

- pit at L: the outcome of candidate L in run r
- stay at L and pit at L' > L: the outcome of candidate L' in run r

Nothing is restarted mid-race, so TUM needs no change. Because the future after L is drawn at random in each run, a model trained on many runs learns the expected outcome given the situation, not the outcome of one known future. The features describe only the state up to L, never later events.

## Features at each decision lap

Taken from the run whose candidate is the latest lap (its race has no target stop before then), at the same moment TUM's VSE decides: end of lap L, before pit entry. TUM's own timing network uses nine features; all are recorded so the new network can be compared like for like, plus a few that the per-lap scores need:

| Feature | TUM's network | Source |
|---|---|---|
| tyre age (and age relative to race length) | yes | driver tyre age |
| race progress, laps remaining | yes | lap / total laps |
| position | yes | positions at end of lap |
| relative compound (soft/medium/hard of the three available) | yes | current compound |
| safety car / VSC status | yes | flag state |
| remaining tyre changes | yes | one-stop: 1 until the stop |
| pursuer changed tyres last lap | yes | car behind's pit history |
| track | yes | race |
| car ahead within 1.5 s last lap | yes | race times |
| gap ahead, gap behind (s) | no | race times |
| pit loss of this track (s) | no | race file |

## Labels: a score for every candidate lap

As decided, the label is a score for every candidate lap, not a single best lap. For a situation at lap L in run r, the outcome (finishing position and race time) of every candidate L' ≥ L in that run is in the run's `runs.csv`; the built rows keep two of them, pit now and the best later lap:

- **pit-now advantage** = outcome(pit at L) − best expected outcome of pitting later. "Expected" is averaged over runs, not taken from run r alone, so it does not use run r's future.
- both measures are kept: finishing position (the project's label rule) and race time (tie-breaker, and smoother). Race time is compared when both continuations completed the same number of laps; a lapped driver's race time is that of their last completed lap.

The network then learns, per situation, the expected outcome of pitting now against waiting; at decision time it pits when pitting now is at least as good as the best later option.

## Diversity

The first labels used only one-stop top finishers, which covers few situations: always near the front, always the same start. To widen the situations:

- every driver of a race as the target, not only the top finishers (midfield gaps, traffic, being lapped)
- all candidate laps from the earliest legal stop to the end
- random safety cars and retirements in every run (already the case)
- races: the four fitted races with mostly one-stop strategies (Spielberg, Monza, Singapore, Austin); Barcelona needs two-stop labels first

## Size and cost

At 500 runs a driver-race is about 34,500 simulated races, about 5 minutes on 28 processes. All 20 drivers of four races is 80 driver-races, about 7 hours, giving roughly 80 × 500 × 40 ≈ 1.6 million situations. A first version can use 200 runs and fewer drivers.

## Known limits

- **Background drivers do not react.** Other drivers keep their real pit laps, so undercut/cover fights are missing. Fixing that needs background drivers on a reactive VSE (TUM's own), which is possible but slower.
- **One stop, fixed compound** per driver, as in the label generator.
- **Situations are only those that arise from the real starting grid and real background strategies**, varied by random events. Perturbing start positions or background strategies would widen them further.

## Results: 80 driver-races (2026-11-17)

`integration/run_situations.py --runs 200 --jobs 28`: all 20 drivers at Spielberg, Monza, Singapore and Austin 2019, our fitted race files, 200 runs, one stop onto the driver's real second compound (a driver who never changed compound takes the compound most others finished on). About 4 hours, no failures. `data_pipeline/build_situations.py` built the rows:

- **860,472 situations** from 4,580 (race, driver, decision lap) cells, about 188 runs each. The full table (130 MB) stays in `integration/output/situations/situations.csv`, which is too large for git; [evidence/situations/situation_summary.csv](evidence/situations/situation_summary.csv) has counts and mean advantages per track, race fifth and flag state.
- **Position advantage for 94 %** of rows; the rest are runs where the driver retired in one of the two continuations.
- **Race-time advantage for 87 %** of rows. Comparing only full-distance finishes gave 37 %, and 6 % for drivers in P16–20, because most of the field is lapped; comparing equal completed laps fixes that (84–91 % in every position band).

### Diversity

| | Pilot (VER Spielberg) | All 80 driver-races |
|---|---|---|
| Situations | 12,832 | 860,472 |
| In the lead | 64 % | 7.8 % |
| Every position 1–20 | no | yes, 4–6 % each for P2–P17, fewer for P18–20 |
| Safety car / VSC | | 4.4 % / 1.6 % |
| Car ahead within 1.5 s last lap | | 24 % |
| Car behind pitted last lap | | 2.3 % |
| Median gap ahead | | 4.2 s (5–95 %: 0.5–25 s) |
| Tyre age | | 2–71 laps |

### Do the rows make sense?

Advantage = pit now − pit at the best later lap; negative means pitting now is better. Green-flag situations:

| Race progress | Position | Race time | Pitting now gains places | Same place either way |
|---|---:|---:|---:|---:|
| 0–20 % | +1.99 | +18.1 s | 3 % | 24 % |
| 20–40 % | +0.52 | +4.5 s | 8 % | 53 % |
| 40–60 % | −0.02 | −0.7 s | 10 % | 76 % |
| 60–80 % | −0.14 | −1.8 s | 15 % | 78 % |
| 80–100 % | +0.18 | +1.8 s | 11 % | 67 % |

- **Early stops are clearly wrong** at every track (+10 s at Spielberg to +30 s at Austin in the first fifth), and the gain turns into a loss around half distance, as worn tyres get slower.
- **Safety cars make stopping cheaper.** For the same driver and lap, pitting under a safety car is 5.9 s (0.71 places) better than under green, and under a VSC 4.1 s (0.43 places).
- **The last fifth flips back** (+1.8 s, Spielberg +6.5 s). With few laps left new tyres cannot pay back the stop, so in the simulator the best one-stop is then as late as possible: the stop is forced by the two-compound rule, not needed for pace. The low wear at Spielberg also explains the late Monte Carlo labels there.
- **Position is a coarse signal**: in 59 % of rows both choices give the same place. Race time separates 57 % of rows that tie on position, so the network should be trained on both, as planned.
- **Checks:** no positions outside 1–20, no negative gaps, gaps missing only for the leader (ahead) and the last car (behind); tyre age is lap plus the start age (0–2 laps).

### What this does not cover

- Every situation is a driver on the first stint with one stop to go; second-stint and two-stop decisions are not sampled.
- The best later lap is chosen from 200-run means, so it is slightly optimistic (the luckiest of several close candidates): pitting now looks a little worse than it is, most where many later candidates are close.
- Background drivers still keep their real pit laps (no undercut fights), as in the labels.
