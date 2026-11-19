"""Checks for the two-stop replay (.venv-train or .venv-data), on a hand-made driver-race.

Laps 2-5 can take the first stop; after a stop on lap 2 a second stop is allowed on lap 4 only.
Outcome (finishing position) of each plan in runs 0 and 1:

    plan       run 0  run 1
    2          8      9
    2 then 4   3      12
    3          6      5
    4          7      4
    5          9      6

With compound runs, a first stop may also go onto B (one stop only):

    B on lap   run 0  run 1
    2          5      9
    3          6      3
    4          2      10
    5          9      9
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.two_stop_labels import TERMINAL, best_plan, choose_compound, load_folders, outcomes, targets

PLANS = {(2, 0): (8, 9), (2, 4): (3, 12), (3, 0): (6, 5), (4, 0): (7, 4), (5, 0): (9, 6)}
OTHER = {2: (5, 9), 3: (6, 3), 4: (2, 10), 5: (9, 9)}


def write_folder(folder: Path):
    runs = [{"Race": "R", "Driver": "D", "Run": run, "Seed": run, "L1": l1, "L2": l2, "C1": "A3",
             "C2": "A6" if l2 else "", "Position": pos[run], "SimulatedLaps": 50.0,
             "RaceTime_s": 1000.0 + pos[run], "Finished": True}
            for (l1, l2), pos in PLANS.items() for run in (0, 1)]
    situations = [{"Run": run, "Seed": run, "L1": 0, "Lap": lap, "Position": 5} for run in (0, 1) for lap in range(2, 6)]
    situations += [{"Run": run, "Seed": run, "L1": 2, "Lap": 4, "Position": 5} for run in (0, 1)]
    pd.DataFrame(runs).to_csv(folder / "runs.csv", index=False)
    pd.DataFrame(situations).to_csv(folder / "situations.csv", index=False)


def write_extra(folder: Path):
    runs = [{"Race": "R", "Driver": "D", "Run": run, "Seed": run, "L1": l1, "L2": 0, "C1": "B", "C2": "",
             "Position": pos[run], "SimulatedLaps": 50.0, "RaceTime_s": 1000.0 + pos[run], "Finished": True}
            for l1, pos in OTHER.items() for run in (0, 1)]
    pd.DataFrame(runs).to_csv(folder / "runs.csv", index=False)
    (folder / "manifest.json").write_text("{}")


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



class CompoundChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "runs" / "R_D").mkdir(parents=True)
        (root / "extra" / "R_D").mkdir(parents=True)
        write_folder(root / "runs" / "R_D")
        write_extra(root / "extra" / "R_D")
        cls.t1, cls.t2 = load_folders([root / "runs" / "R_D"], root / "extra")
        cls.never = np.zeros(len(cls.t2), bool)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def pit1(self, lap, compound):
        return ((self.t1.Lap == lap) & (self.t1.StopCompound == compound)).to_numpy()

    def test_tables(self):
        self.assertEqual(self.t1.Lap.tolist(), [2, 2, 3, 3, 4, 4, 5, 5] * 2)
        self.assertEqual(self.t1.StopCompound.tolist(), ["A3", "B"] * 8)      # stored compound first
        self.assertTrue(self.t1.HasSituation.all())                            # same situation for both

    def test_outcomes_with_compound(self):
        out = outcomes(self.t1, self.t2, self.pit1(3, "B"), self.never)
        self.assertEqual(out.PitPosition.tolist(), [6, 3])
        self.assertEqual(out.StopCompound.tolist(), ["B", "B"])
        # never pitting: the forced last-lap stop goes onto the stored compound, not B
        out = outcomes(self.t1, self.t2, np.zeros(len(self.t1), bool), self.never)
        self.assertEqual(out.PitPosition.tolist(), [9, 6])
        self.assertEqual(out.StopCompound.tolist(), ["A3", "A3"])

    def test_targets_compare_with_the_next_lap(self):
        y1, _, use1, _ = targets(self.t1, self.t2, self.pit1(4, "A3"), self.never)
        # run 0: B on lap 2 is worth 5; following from lap 3 stops on 4 onto A3: 7
        self.assertEqual(y1.AdvantagePosition[1], 5 - 7)
        self.assertEqual(y1.AdvantagePosition[0], 8 - 7)
        self.assertEqual(y1.AdvantagePosition[3], 6 - 7)
        self.assertEqual(use1.tolist(), [True] * 6 + [False] * 2 + [True] * 6 + [False] * 2)

    def test_choose_compound(self):
        pred = np.zeros((len(self.t1), 2))
        pred[:, 0] = [0.0, -0.05, 0.0, -0.5, 0.0, 0.2, 0.0, 0.0] * 2    # B: tie on 2, better on 3, worse on 4
        pred[:, 1] = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 2       # the tie goes to the faster A3
        chosen = choose_compound(self.t1, pred, 0.1)
        self.assertEqual(chosen.tolist(), [True, False, False, True, True, False, True, False] * 2)

    def test_choice_equals_the_simulator_rule(self):
        from integration.pit_network import POSITION_TIE, best_choice
        rng = np.random.default_rng(1)
        pred = np.round(rng.normal(scale=0.2, size=(len(self.t1), 2)), 2)
        chosen = choose_compound(self.t1, pred, POSITION_TIE)
        for start in range(0, len(self.t1), 2):
            pair = [tuple(pred[start]), tuple(pred[start + 1])]
            self.assertEqual(chosen[start:start + 2].tolist(), [best_choice(pair) == 0, best_choice(pair) == 1])

    def test_compound_head_moves_the_stop_to_the_other_compound(self):
        from data_pipeline.train_compound_head import pairs, switch
        stored, other = pairs(self.t1)
        self.assertEqual(self.t1.StopCompound.to_numpy()[stored].tolist(), ["A3"] * 8)
        self.assertEqual(self.t1.Lap.to_numpy()[stored].tolist(), self.t1.Lap.to_numpy()[other].tolist())
        pit = self.pit1(3, "A3")                                # the timing network stops on lap 3
        moved = switch(pit, stored, other, np.ones(len(stored), bool))
        self.assertEqual(outcomes(self.t1, self.t2, moved, self.never).PitPosition.tolist(), [6, 3])   # B on 3
        kept = switch(pit, stored, other, np.zeros(len(stored), bool))
        self.assertEqual(outcomes(self.t1, self.t2, kept, self.never).PitPosition.tolist(), [6, 5])    # A3 on 3

    def test_best_fixed_plans(self):
        policies = best_plan(self.t1, self.t2, lambda t: np.ones(len(t), bool))
        expected = {"fixed one-stop": [6, 5], "fixed one-stop, any compound": [6, 3], "fixed best plan": [6, 3]}
        for name, positions in expected.items():
            self.assertEqual(outcomes(self.t1, self.t2, *policies[name]).PitPosition.tolist(), positions, name)


if __name__ == "__main__":
    unittest.main()
