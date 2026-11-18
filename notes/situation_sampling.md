# Race situation sampling (methodology step 4): design

Status: design, 2026-11-17. Builds on the Monte Carlo label generator ([label_generator.md](label_generator.md)).

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

As decided, the label is a score for every candidate lap, not a single best lap. For a situation at lap L in run r, the row stores the outcome (finishing position and race time) of every candidate L' ≥ L in that run. From those:

- **pit-now advantage** = outcome(pit at L) − best expected outcome of pitting later. "Expected" is averaged over runs, not taken from run r alone, so it does not use run r's future.
- both measures are kept: finishing position (the project's label rule) and race time (tie-breaker, and smoother).

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
