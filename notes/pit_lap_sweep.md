# Pit-lap sweep: Austria 2019

Prepared 2026-10-23. This is the first version of the label generator: for one driver, try every in-lap for a single stop and record the race outcome, with all other drivers on their real strategies.

Setup: `integration/run_sweep.sh` (wraps `run_experiment.py fixed`), original `pars_Spielberg_2019.ini`, deterministic (no probabilistic influences, no random events), one stop onto A3 (hard), in-laps 2–70, drivers VER, LEC and HAM. 207 runs; every run passed the runner's check that the simulator executed exactly the requested stop. Raw rows: `integration/output/sweep/pit_lap_sweep.csv`. Figure: `figures/pit_lap_sweep_austria_2019.png`.

Runs were made on Linux (Python 3.8.20, same pinned packages) after confirming that the baseline, realstrategy and fixed-lap outputs there are byte-identical to this Mac's `integration/output/`.

## Results (full-distance runs only)

| Driver | Real in-lap | Fastest in-lap | Gain vs real | Finish at fastest | Best finish and its lap |
|---|---:|---:|---:|---|---|
| VER | 31 | 52 | −9.2 s | P2 (real P2) | P1, only when pitting on lap 23 (+22.8 s slower overall) |
| HAM | 30 | 17 | −20.1 s | P3 (real P4) | P2 on lap 11 |
| LEC | 22 | 25 | −0.2 s | P1 (real P1) | P1 from lap 9 to 33, except lap 30 |

Laps 2–8 (2–6 for LEC) put the driver a lap down (in-lap too early for a 60+ lap hard stint), so their race times are not comparable and are left out.

## What it shows

1. **The curve has the expected shape for LEC only.** LEC's real stop (lap 22) sits in a flat optimum (laps 16–29 all within 2 s). VER's fastest stop is lap 52, which no team would choose.
2. **VER's late optimum comes from the parameters.** In `pars_Spielberg_2019.ini`, VER's medium (A4) degrades at 0.01 s/lap and his hard (A3) at 0.0321 s/lap. With the hard tyre wearing three times faster than the medium, staying out is always cheaper. 32 of 60 driver-compound k1 values in this file sit at exactly 0.01, the file's minimum (30% across all 121 TUM files). That looks like a lower bound in TUM's fitting; it is an inference, not documented.
3. **Traffic makes the curve jagged.** HAM gains 20 s by pitting on lap 17 because it frees him from the BOT train; LEC loses 26 s by pitting on lap 30 or 34+ because he rejoins behind traffic. In the realstrategy run, 11 of 19 cars finish exactly 0.5 s (`min_t_dist`) behind the car ahead: the field forms trains that cannot pass.
4. **Minimum race time and best position disagree.** VER's only P1 (lap 23) is 22.8 s slower overall than his fastest stop, because he jumps LEC and then holds him up.

## Consequences for label generation

- Labels from these parameters would teach "pit as late as possible" for VER. The degradation parameters need refitting (our own fits) before any labels are generated.
- The objective (time vs position) must be fixed before labeling; this sweep already shows they pick different laps.
- Deterministic labels sit on traffic cliffs one lap apart. Monte Carlo averaging (TUM paper: 1,000 runs gives ±0.08 positions) is needed to smooth them, which multiplies cost by the run count.
- Cost here: about 1 s per run including Python start-up, so 69 laps × 1 driver ≈ 70 s deterministic. With Monte Carlo inside one process (TUM: about 0.1 s per race), 1,000 runs per lap is about 2 h per driver per race.
