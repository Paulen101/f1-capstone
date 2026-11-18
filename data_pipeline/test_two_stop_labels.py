"""Checks for the two-stop replay (.venv-train or .venv-data), on a hand-made driver-race.

Laps 2-5 can take the first stop; after a stop on lap 2 a second stop is allowed on lap 4 only.
Outcome (finishing position) of each plan in runs 0 and 1:

    plan       run 0  run 1
    2          8      9
    2 then 4   3      12
    3          6      5
    4          7      4
    5          9      6
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.two_stop_labels import TERMINAL, best_plan, load_folders, outcomes, targets

PLANS = {(2, 0): (8, 9), (2, 4): (3, 12), (3, 0): (6, 5), (4, 0): (7, 4), (5, 0): (9, 6)}


def write_folder(folder: Path):
    runs = [{"Race": "R", "Driver": "D", "Run": run, "Seed": run, "L1": l1, "L2": l2, "C1": "A3",
             "C2": "A6" if l2 else "", "Position": pos[run], "SimulatedLaps": 50.0,
             "RaceTime_s": 1000.0 + pos[run], "Finished": True}
            for (l1, l2), pos in PLANS.items() for run in (0, 1)]
    situations = [{"Run": run, "Seed": run, "L1": 0, "Lap": lap, "Position": 5} for run in (0, 1) for lap in range(2, 6)]
    situations += [{"Run": run, "Seed": run, "L1": 2, "Lap": 4, "Position": 5} for run in (0, 1)]
    pd.DataFrame(runs).to_csv(folder / "runs.csv", index=False)
    pd.DataFrame(situations).to_csv(folder / "situations.csv", index=False)


class TwoStopReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        write_folder(Path(cls.tmp.name))
        cls.t1, cls.t2 = load_folders([cls.tmp.name])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pit1(self, lap):
        return (self.t1.Lap == lap).to_numpy()

    def test_tables(self):
        self.assertEqual(self.t1.Lap.tolist(), [2, 3, 4, 5] * 2)
        self.assertEqual(self.t2.Lap.tolist(), [4, TERMINAL] * 2)
        self.assertEqual(self.t2.PitPosition.tolist(), [3, 8, 12, 9])

    def test_one_stop_and_two_stop_policies(self):
        never = np.zeros(len(self.t2), bool)
        always = self.t2.Lap.to_numpy() == 4
        self.assertEqual(outcomes(self.t1, self.t2, self.pit1(3), never).PitPosition.tolist(), [6, 5])
        self.assertEqual(outcomes(self.t1, self.t2, self.pit1(2), never).PitPosition.tolist(), [8, 9])
        out = outcomes(self.t1, self.t2, self.pit1(2), always)
        self.assertEqual(out.PitPosition.tolist(), [3, 12])
        self.assertEqual(out.SecondStop.tolist(), [4, 4])

    def test_targets(self):
        always = self.t2.Lap.to_numpy() == 4
        y1, y2, use1, use2 = targets(self.t1, self.t2, self.pit1(4), always)
        # second stop on 4 after a first on 2: (2,4) - (2, none)
        self.assertEqual(y2.AdvantagePosition[use2].tolist(), [3 - 8, 12 - 9])
        # first stint, lap 2: a stop now is worth (2,4); following from lap 3 stops on 4 -> (4, none)
        self.assertEqual(y1.AdvantagePosition[0], 3 - 7)
        self.assertEqual(y1.AdvantagePosition[4], 12 - 4)
        self.assertEqual(use1.tolist(), [True, True, True, False] * 2)   # lap 5 is the forced last lap

    def test_best_fixed_plans(self):
        policies = best_plan(self.t1, self.t2, lambda t: np.ones(len(t), bool))
        pit1, pit2 = policies["fixed one-stop"]
        self.assertEqual(outcomes(self.t1, self.t2, pit1, pit2).PitPosition.tolist(), [6, 5])     # lap 3: 5.5
        pit1, pit2 = policies["fixed best plan"]
        self.assertEqual(outcomes(self.t1, self.t2, pit1, pit2).PitPosition.tolist(), [6, 5])     # (2,4): 7.5


if __name__ == "__main__":
    unittest.main()
