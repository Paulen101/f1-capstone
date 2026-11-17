"""Checks for the held-out tyre-age split (.venv-data)."""
import unittest

import pandas as pd

from data_pipeline.heldout_tyres import split


class SplitTests(unittest.TestCase):
    def test_last_third_of_each_stint_is_held_out(self):
        laps = pd.DataFrame({"Driver": ["AAA"] * 10 + ["BBB"] * 4, "Stint": [1] * 10 + [2] * 4,
                             "TyreLife": list(range(1, 11)) + [5, 6, 7, 8]})
        held = split(laps)
        self.assertEqual(laps.TyreLife[held & (laps.Driver == "AAA")].tolist(), [8, 9, 10])
        self.assertEqual(laps.TyreLife[held & (laps.Driver == "BBB")].tolist(), [8])


if __name__ == "__main__":
    unittest.main()
