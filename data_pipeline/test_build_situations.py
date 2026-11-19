"""Checks for the situation builder on a made-up label run (.venv-data)."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_pipeline.build_situations import best_later_laps, build, summarise


class BuildTests(unittest.TestCase):
    def test_best_later_lap_uses_mean_position_then_time(self):
        summary = pd.DataFrame({"PitLap": [10, 11, 12, 13], "FinishedRuns": [5, 5, 5, 0],
                                "MeanPosition": [2.0, 1.5, 1.5, None], "MeanRaceTime_s": [100.0, 99.0, 98.0, None]})
        self.assertEqual(best_later_laps(summary), {10: 12, 11: 12})

    def test_advantage_pairs_the_same_run(self):
        runs = pd.DataFrame({"PitLap": [10, 10, 11, 11], "Run": [0, 1, 0, 1], "Position": [3, 1, 2, 2],
                             "RaceTime_s": [100.0, 90.0, 98.0, 95.0], "Finished": [True] * 4,
                             "SimulatedLaps": [50.0] * 4})
        summary = pd.DataFrame({"PitLap": [10, 11], "FinishedRuns": [2, 2], "MeanPosition": [2.0, 2.0],
                                "MeanRaceTime_s": [95.0, 96.5]})
        situations = pd.DataFrame({"Run": [0, 1, 0, 1], "Lap": [10, 10, 11, 11], "TyreAge": [12, 12, 13, 13]})
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            runs.to_csv(folder / "runs.csv", index=False)
            summary.to_csv(folder / "summary.csv", index=False)
            situations.to_csv(folder / "situations.csv", index=False)
            rows = build(folder)
        self.assertEqual(len(rows), 2)                      # lap 11 has no later candidate
        self.assertEqual(rows.LaterLap.tolist(), [11, 11])
        self.assertEqual(rows.AdvantagePosition.tolist(), [1, -1])
        self.assertEqual(rows.AdvantageTime_s.tolist(), [2.0, -5.0])

    def test_race_time_needs_the_same_completed_laps(self):
        runs = pd.DataFrame({"PitLap": [10, 10, 10, 11, 11, 11], "Run": [0, 1, 2] * 2,
                             "Position": [15, 16, 17, 15, 17, 16], "Finished": [True] * 6,
                             "RaceTime_s": [100.0, 101.0, 102.0, 99.0, 98.0, 103.0],
                             "SimulatedLaps": [49.4, 49.9, 50.0, 49.7, 48.95, 50.0]})
        summary = pd.DataFrame({"PitLap": [10, 11], "FinishedRuns": [3, 3], "MeanPosition": [16.0, 16.0],
                                "MeanRaceTime_s": [101.0, 100.0]})
        situations = pd.DataFrame({"Run": [0, 1, 2], "Lap": [10, 10, 10]})
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            runs.to_csv(folder / "runs.csv", index=False)
            summary.to_csv(folder / "summary.csv", index=False)
            situations.to_csv(folder / "situations.csv", index=False)
            rows = build(folder)
        times = rows.AdvantageTime_s.tolist()
        self.assertEqual(times[0], 1.0)                     # both lapped once (49 laps)
        self.assertTrue(pd.isna(times[1]))                  # 49 vs 48 laps: not comparable
        self.assertEqual(times[2], -1.0)                    # both ran the full distance


    def test_summary_groups_by_track_phase_and_flag(self):
        table = pd.DataFrame({"Race": ["pars_Monza_2019_fitted.ini"] * 3, "Driver": ["LEC"] * 3,
                              "Lap": [5, 6, 50], "RaceProgress": [0.09, 0.11, 1.0], "FlagState": ["G"] * 3,
                              "Position": [1, 2, 3], "AdvantagePosition": [1.0, -1.0, 0.0],
                              "AdvantageTime_s": [2.0, None, -1.0]})
        summary = summarise(table)
        self.assertEqual(summary.RaceProgressFrom_pct.tolist(), [0, 80])
        self.assertEqual(summary.Situations.tolist(), [2, 1])
        self.assertEqual(summary.PitNowBetterPosition.tolist(), [0.5, 0.0])
        self.assertEqual(summary.PitNowBetterTime.tolist(), [0.0, 1.0])
        self.assertEqual(summary.TimeCompared.tolist(), [0.5, 1.0])



if __name__ == "__main__":
    unittest.main()
