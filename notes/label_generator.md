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

## Known limits

- **Stock parameters:** results from TUM's original parameter files are for pipeline testing only. `pit_lap_sweep.md` shows they teach "pit as late as possible" for VER. Real labels need our fitted parameters (methodology step 2).
- **Background strategies:** other drivers keep their real pit laps even when `sampled` draws different safety cars, so they do not react to those events. In `sampled` mode a driver who really retired also keeps only the stops made before retiring. This matters most at Singapore.
- **One stop, fixed compound:** the stop count and compound are not chosen yet.
- **Whole-race labels:** every run starts at lap 1 and fixes the stop in advance. Mid-race situation sampling (methodology step 4) and per-lap pit/stay rows are not built yet.
- **Single process:** parallel runs are not implemented. Seeds depend only on the run number, so adding them should not change results.
