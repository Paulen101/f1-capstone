"""Checks for the policy replay behind the waiting-aware labels (.venv-train or .venv-data)."""
import unittest

import numpy as np
import pandas as pd

from data_pipeline.stopping_labels import fixed_lap_pit, follow, policy_outcomes, targets


def table(runs):
    """runs: {run: [(lap, position, finished, has_situation), ...]} for one driver-race, laps ascending."""
    rows = []
    for run, laps in runs.items():
        for lap, position, finished, has in laps:
            rows.append({"Race": "R", "Driver": "D", "Run": run, "Lap": lap, "PitPosition": position,
                         "PitRaceTime_s": 1000.0 + position, "PitFinished": finished, "PitLaps": 50.0,
                         "HasSituation": has})
    rows = pd.DataFrame(rows)
    rows["LastLap"] = rows.Lap.max()
    return rows


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.rows = table({0: [(2, 5, True, True), (3, 3, True, True), (4, 4, True, True), (5, 6, True, True)],
                           1: [(2, 7, True, True), (3, 2, True, True), (4, 8, True, True), (5, 9, True, True)]})

    def test_stops_on_first_pit_lap_or_last_lap(self):
        pit = np.array([False, True, True, False, False, False, False, False])
        later = follow(self.rows, pit)
        self.assertEqual(later.StopLap.tolist(), [3, 3, 4, 5, 5, 5, 5, 5])
        self.assertEqual(later.PitPosition.tolist(), [3, 3, 4, 6, 9, 9, 9, 9])

    def test_target_is_pit_now_minus_following_from_next_lap(self):
        pit = np.array([False, True, False, False, False, True, False, False])
        y = targets(self.rows, pit)
        # run 0, lap 2: pit now P5; follow from lap 3: stops lap 3, P3 -> +2
        # lap 3: P3 now; follow from lap 4 never pits -> forced lap 5, P6 -> -3; lap 4: P4 - P6 -> -2
        self.assertEqual(y.AdvantagePosition.tolist()[:3], [2.0, -3.0, -2.0])
        self.assertTrue(np.isnan(y.AdvantagePosition[3]))            # last lap: nothing later
        # run 1, lap 2: P7 now; follow from 3 stops at 3 (P2) -> +5
        self.assertEqual(y.AdvantagePosition[4], 5.0)
        self.assertEqual(y.LaterStopLap[4], 3)

    def test_rows_without_situation_never_decide(self):
        rows = table({0: [(2, 5, False, True), (3, 20, False, False), (4, 20, False, False)]})
        later = follow(rows, np.array([False, True, True]))
        self.assertEqual(later.StopLap.tolist(), [4, 4, 4])
        self.assertTrue(targets(rows, np.array([False, True, True])).AdvantagePosition.isna().all())

    def test_policy_outcome_per_run(self):
        pit = np.array([False, False, True, False, True, False, False, False])
        out = policy_outcomes(self.rows, pit)
        self.assertEqual(out.Run.tolist(), [0, 1])
        self.assertEqual(out.StopLap.tolist(), [4, 2])
        self.assertEqual(out.PitPosition.tolist(), [4, 7])

    def test_fixed_lap_chosen_from_given_runs_only(self):
        only_run_1 = (self.rows.Run == 1).to_numpy()
        pit = fixed_lap_pit(self.rows, only_run_1)
        self.assertEqual(self.rows.Lap[pit].unique().tolist(), [3])   # run 1's best lap, used in every run


if __name__ == "__main__":
    unittest.main()
