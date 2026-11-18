"""Checks for the paired policy comparison (.venv-data)."""
import unittest

import pandas as pd

from data_pipeline.summarise_evaluation import describe, paired


def run(policy, run, position, laps=71.0, time=5000.0, finished=True, stop=30):
    return {"Track": "Spielberg", "Driver": "VER", "Policy": policy, "Run": run, "Position": position,
            "SimulatedLaps": laps, "RaceTime_s": time, "Finished": finished, "FirstStopLap": stop}


class PairedTests(unittest.TestCase):
    def test_pairs_by_run_and_skips_retirements(self):
        runs = pd.DataFrame([run("network", 0, 2, time=4990.0, stop=32), run("real", 0, 3, stop=30),
                             run("network", 1, 1), run("real", 1, 20, laps=10.0, finished=False)])
        diffs = paired(runs, "network", "real")
        self.assertEqual(diffs.PosDiff.dropna().tolist(), [-1])
        self.assertEqual(diffs.TimeDiff_s.dropna().tolist(), [-10.0])
        self.assertEqual(diffs.StopDiff.tolist(), [2, 0])

    def test_race_time_only_for_equal_laps(self):
        runs = pd.DataFrame([run("network", 0, 15, laps=70.0, time=4900.0), run("real", 0, 16, laps=71.0)])
        diffs = paired(runs, "network", "real")
        self.assertEqual(diffs.PosDiff.tolist(), [-1])
        self.assertTrue(diffs.TimeDiff_s.isna().all())

    def test_describe(self):
        diffs = pd.DataFrame({"PosDiff": [-1.0, 1.0, -2.0], "TimeDiff_s": [-3.0, 1.0, float("nan")],
                              "StopDiff": [1, 2, 3]})
        d = describe(diffs)
        self.assertAlmostEqual(d["PosDiff"], -2 / 3)
        self.assertEqual(d["TimeDiff_s_N"], 2)
        self.assertAlmostEqual(d["Better"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
