# Label generator

`integration/label_generator.py` simulates every candidate pit lap for one driver in one race and picks the best lap. This is the first slice of methodology step 5. All other drivers stay on TUM's `realstrategy`, and the target driver makes one stop onto a fixed compound through `FixedLapVSE`. No TUM source is changed.

It runs all races in one Python process, unlike `run_sweep.sh`, which starts Python for every run.

## Label rule

For each candidate pit lap, the generator averages over the runs in which the target driver finished. Candidates are ranked by:

1. lowest mean finishing position
2. most laps completed, so a lapped car's shorter race time cannot win a tie
3. lowest mean race time

Runs where the target driver retired have no finishing position and are left out. `FinishedRuns` in `summary.csv` shows how many runs remain.

`fastest_pit_lap` in the manifest is a diagnostic only. It uses full-distance runs and shows where the time and position objectives disagree.

## Randomness

| Mode | Probabilistic influences | Safety cars and retirements |
|---|---|---|
| `--deterministic` | off | exactly as in the race file |
| `--runs N --events sampled` (default) | on | recorded events cleared; TUM draws new ones each run |
| `--runs N --events recorded` | on | the file's real events; TUM draws only where the file's list is empty |

`sampled` is the default because a decision model cannot know future events. Keeping the real safety cars would teach pit laps that rely on hindsight.

Every candidate lap uses the same seed for a given run number. TUM draws events in the `Race` constructor, before the stop can affect anything, so all candidate laps in a run face the same safety cars and retirements. Later random draws can still diverge once the stop changes the sequence of calls.

## Results, Austria 2019 (2026-11-06, Windows, stock TUM parameters)

**Deterministic check:** VER, LEC and HAM, in-laps 2–70 onto A3. All 207 rows (position, laps completed, race time) match `integration/output/sweep/pit_lap_sweep.csv` exactly. That file came from 207 separate `run_experiment.py` processes, so running everything in one process changes nothing.

| Driver | Label lap (best position) | Fastest lap | Matches `pit_lap_sweep.md` |
|---|---:|---:|---|
| VER | 23 | 52 | yes |
| LEC | 25 | 25 | yes |
| HAM | 11 | 17 | yes |

**Speed:** 134–152 ms per race deterministic and 114 ms per race in Monte Carlo, in one process. 69 candidate laps × 1,000 runs is therefore about 2.2 h per driver per race, as estimated in `pit_lap_sweep.md`.

**Monte Carlo, 10 runs, sampled events (VER):** the label moves to lap 45 and the fastest lap to 55. VER retires in run 0 at every candidate lap, because the shared seed gives the same retirement, which leaves 9 finished runs per lap. Laps 43–45 tie on mean position 1.67 (SD 0.50) and lap 31 has 1.89 (SD 0.33), so the gap is about one standard error. **Ten runs cannot tell these laps apart**, so lap 45 is not a meaningful label yet. Summary: [evidence/label_generator/Spielberg_2019_VER_mc10_summary.csv](evidence/label_generator/Spielberg_2019_VER_mc10_summary.csv).

**Reproducibility:** rerunning laps 20–25 with 3 runs reproduced the matching 18 rows of the 10-run output exactly.

## Monte Carlo labels on the fitted files (2026-11-17)

`integration/run_mc_labels.sh`: 500 runs per candidate lap, sampled safety cars and retirements, our fitted 2019 files, every one-stop top finisher at Spielberg, Monza, Singapore and Austin (14 driver-races, about 400,000 races, 28 processes). Collected with `data_pipeline/summarise_labels.py`:

- [evidence/labels/pit_lap_scores.csv](evidence/labels/pit_lap_scores.csv): **the training label as decided**, a score for every candidate lap: mean position and race-time difference to the best lap, paired over runs with the same seed, with standard errors
- [evidence/labels/label_summary.csv](evidence/labels/label_summary.csv): one row per driver

| Track | Driver | Real stop | Monte Carlo label | Within 2 SE of the label | Deterministic label |
|---|---|---:|---:|---|---:|
| Spielberg | VER | 31 | 28 | 22–41 | 42 |
| Spielberg | HAM | 30 | 37 | 31–39 | 31 |
| Spielberg | LEC | 22 | 37 | 31–40 | 31 |
| Spielberg | BOT | 21 | 37 | 34–41 | 34 |
| Monza | LEC | 20 | 22 | 21–28 | 30 |
| Monza | BOT | 27 | 23 | 22–26 | 30 |
| Monza | RIC | 30 | 21 | 20–22 | 28 |
| Singapore | VET | 19 | 14 | 13–16 | 22 |
| Singapore | LEC | 20 | 21 | 18–23 | 27 |
| Singapore | VER | 19 | 15 | 15–18 | 22 |
| Singapore | HAM | 26 | 18 | 15–20 | 18 |
| Austin | HAM | 24 | 23 | 22–25 | 22 |
| Austin | RIC | 21 | 22 | 20–23 | 20 |
| Austin | SAI | 18 | 22 | 19–24 | 22 |

- **The hindsight is gone.** The deterministic Monza labels sat on lap 28–30, the lap of the recorded VSC; with sampled events they move to 21–23.
- **The labels are simulator-optimal, not copies of the teams.** On average they are 5.7 laps from the real stop (deterministic: 5.5), and the real stop lies within two standard errors of the label for 3 of 14 drivers (VER Spielberg, HAM and RIC Austin).
- **The misses have a pattern.** At Spielberg the simulator prefers late stops (lap 37; real 21–30): the front-runners really stopped early to undercut or cover each other, and background drivers here cannot react. At Singapore it prefers early stops (laps 14–18; real 19–26).
- **Windows are 3–10 laps wide** at 500 runs, except VER at Spielberg (20 laps). The best lap by race time and by position differ by up to 10 laps (BOT Spielberg 27 vs 37); RIC and SAI at Austin are usually lapped, so they have no race-time optimum.
- **10–14 % of runs end with the driver retired** (random retirements); those runs have no position and are left out.

Problems found and fixed while running these:

- **Memory.** Every process reserved about 5.5 GB because the maths library prepared buffers for all 32 CPU threads; 16 workers ran out of memory and the pool hung. Workers now use one maths thread (81 MB each), and a crashed worker raises an error instead of hanging. `--jobs` gives byte-identical results to one process.
- **TUM's Monza file is short of fuel**: 110 kg at 2.076 kg/lap lasts 52.99 of 53 laps, so a race without a safety car ran dry on the last lap. The fitted file burns 2.075 kg/lap (`write_fitted_pars.py`).
- **Drivers who really retired** (SAI Monza, stopped soft to soft; VET Austin, no stop) finish in sampled runs on one compound, and TUM rejects such a race. Their last stop now goes onto the compound most others finished on (`complete_background_strategies`, recorded in the manifest).

## Known limits

- **Stock parameters:** results from TUM's original parameter files are for pipeline testing only. `pit_lap_sweep.md` shows they teach "pit as late as possible" for VER. Real labels need our fitted parameters (methodology step 2).
- **Background strategies:** other drivers keep their real pit laps even when `sampled` draws different safety cars, so they do not react to those events or to the target driver (no undercut fights). In `sampled` mode a driver who really retired keeps the stops made before retiring, completed to a second compound where needed.
- **One stop, fixed compound:** the stop count and compound are not chosen yet.
- **Whole-race labels:** every run starts at lap 1 and fixes the stop in advance. `--situations` records the race situation at every candidate lap for per-situation rows (methodology step 4, [situation_sampling.md](situation_sampling.md)).
- **Speed:** about 7–12 ms per race wall clock with 24–28 processes on this 32-thread laptop.
