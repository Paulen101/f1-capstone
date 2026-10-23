"""Verify gap normalization and handling of unequal race distances."""
import unittest

import pandas as pd

from data_pipeline.compare_2019 import compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.sim = pd.DataFrame({"Driver": ["VER", "LEC", "BOT"], "Position": [2, 1, 3],
                                 "SimulatedLaps": [71, 71, 70], "DrivingAtEnd": [True] * 3,
                                 "RaceTime_s": [4900.0, 4890.0, 4930.0]})
        self.real = pd.DataFrame({"Abbreviation": ["VER", "LEC", "BOT"], "Position": [1, 2, 3],
                                  "Laps": [71] * 3, "Status": ["Finished"] * 3,
                                  "ResultTime_s": [4921.822, 2.724, 18.960]})

    def test_winner_plus_gap_and_unequal_distance(self):
        result = compare(self.sim, self.real, 71, "winner-and-gaps").set_index("Driver")
        self.assertAlmostEqual(result.loc["LEC", "ActualRaceTime_s"], 4924.546)
        self.assertAlmostEqual(result.loc["LEC", "TimeError_s"], -34.546)
        self.assertFalse(result.loc["BOT", "TimeComparable"])
        self.assertTrue(pd.isna(result.loc["BOT", "TimeError_s"]))
        self.assertEqual(len(result), 3)

    def test_totals_are_not_added_to_winner_again(self):
        self.real.loc[1:, "ResultTime_s"] += 4921.822
        result = compare(self.sim, self.real, 71, "totals").set_index("Driver")
        self.assertAlmostEqual(result.loc["LEC", "ActualRaceTime_s"], 4924.546)
        with self.assertRaises(ValueError):
            compare(self.sim, self.real, 71, "winner-and-gaps")

    def test_incorrect_format_and_missing_driver_fail(self):
        with self.assertRaises(ValueError):
            compare(self.sim, self.real, 71, "totals")
        with self.assertRaises(ValueError):
            compare(self.sim.iloc[:2], self.real, 71, "winner-and-gaps")


if __name__ == "__main__":
    unittest.main()
