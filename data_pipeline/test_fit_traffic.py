"""Checks for the traffic measurement on made-up car pairs with known answers (.venv-data)."""
import unittest

import numpy as np
import pandas as pd

from data_pipeline.fit_traffic import advantage, count_matching_threshold, measure

TRACK = {"t_drseffect": -0.3, "t_gap_overtake_vel": -0.02, "t_gap_overtake": 2.0}
RACE = {"drs_window": 1.0}


def make_pairs():
    rows = []
    for lap in range(2, 42):
        gap = 0.4 + 0.02 * lap                       # 0.44 .. 1.22 s at the start of the lap
        passed = lap % 4 == 0
        rows.append({"lap": lap, "gap_start_s": gap, "gap_end_s": -1.0 if passed else 0.5 + 0.01 * lap,
                     "front_gap_ahead_s": np.inf, "pace_advantage_s": 0.8 if passed else 0.1,
                     "front_excess_s": 1.5 if passed else 0.2, "back_excess_s": 0.0,
                     "front_clean": True, "back_clean": True, "top_speed_delta_kph": 0.0,
                     "passed": passed})
    return pd.DataFrame(rows)


class TrafficTests(unittest.TestCase):
    def test_threshold_predicts_the_real_number_of_passes(self):
        adv = pd.Series([1.0, 0.6, 0.4, 0.2, -0.5])
        self.assertAlmostEqual(count_matching_threshold(adv, 2), 0.5)
        self.assertEqual(int((adv >= count_matching_threshold(adv, 2)).sum()), 2)

    def test_advantage_adds_drs_only_inside_the_window(self):
        p = pd.DataFrame({"pace_advantage_s": [0.5, 0.5], "gap_start_s": [0.8, 1.2],
                          "top_speed_delta_kph": [0.0, 10.0]})
        self.assertEqual([round(v, 3) for v in advantage(p, TRACK, RACE)], [0.0, -0.5])

    def test_known_values_are_recovered(self):
        result = measure(make_pairs(), TRACK, RACE)
        self.assertEqual(result["counts"]["passes"], 10)
        self.assertAlmostEqual(result["t_duel"], 0.2)
        self.assertAlmostEqual(result["t_overtake_loser"], 1.3)
        held = make_pairs().query("not passed")
        self.assertAlmostEqual(result["min_t_dist"], held.gap_end_s.quantile(0.05))
        close = make_pairs().query("gap_start_s <= 1.5")
        predicted = advantage(close, TRACK, RACE) >= result["t_gap_overtake"]
        self.assertEqual(int(predicted.sum()), 10)


if __name__ == "__main__":
    unittest.main()
