"""Checks for the situation builder on a made-up label run (.venv-data)."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_pipeline.build_situations import best_later_laps, build


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


if __name__ == "__main__":
    unittest.main()
