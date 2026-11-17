"""Checks for the parameter fit on made-up laps with known answers (.venv-data)."""
import unittest

import numpy as np
import pandas as pd

from data_pipeline.fit_parameters import add_gaps, free_air, pit_stops, solve

LEVELS = {"AAA": 70.0, "BBB": 70.4, "CCC": 71.0, "DDD": 71.3}
PLANS = {"AAA": [("SOFT", 18), ("HARD", 42)], "BBB": [("MEDIUM", 25), ("HARD", 35)],
         "CCC": [("HARD", 33), ("MEDIUM", 27)], "DDD": [("MEDIUM", 20), ("SOFT", 15), ("HARD", 25)]}


def make_laps(tyres, trend=0.03, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for driver, plan in PLANS.items():
        lap = 0
        for stint, (compound, length) in enumerate(plan, start=1):
            for life in range(1, length + 1):
                lap += 1
                k0, k1 = tyres[compound]
                rows.append({"Driver": driver, "Stint": stint, "LapNumber": lap, "Compound": compound,
                             "TyreLife": life,
                             "LapTime_s": LEVELS[driver] - trend * (lap - 1) + k0 + k1 * (life - 1)
                                          + rng.normal(0, noise)})
    return pd.DataFrame(rows)


class SolveTests(unittest.TestCase):
    def test_recovers_known_values(self):
        truth = {"HARD": (0.0, 0.02), "MEDIUM": (-0.3, 0.04), "SOFT": (-0.6, 0.08)}
        fit = solve(make_laps(truth, noise=0.05))
        self.assertAlmostEqual(fit["trend"], 0.03, delta=0.004)
        for compound, (k0, k1) in truth.items():
            self.assertAlmostEqual(fit["tyres"][compound]["k0"], k0, delta=0.08)
            self.assertAlmostEqual(fit["tyres"][compound]["k1"], k1, delta=0.006)
        for driver, level in LEVELS.items():
            self.assertAlmostEqual(fit["levels"][driver], level, delta=0.1)

    def test_ordering_stops_a_softer_tyre_from_lasting_longer(self):
        upside_down = {"HARD": (0.0, 0.05), "MEDIUM": (0.4, 0.01), "SOFT": (0.6, 0.0)}
        laps = make_laps(upside_down, noise=0.05)
        free = solve(laps, ordered=False)["tyres"]
        self.assertLess(free["SOFT"]["k1"], free["HARD"]["k1"])
        tyres = solve(laps)["tyres"]
        self.assertGreaterEqual(tyres["MEDIUM"]["k1"], tyres["HARD"]["k1"] - 1e-9)
        self.assertGreaterEqual(tyres["SOFT"]["k1"], tyres["MEDIUM"]["k1"] - 1e-9)
        self.assertLessEqual(tyres["MEDIUM"]["k0"], 1e-9)
        self.assertLessEqual(tyres["SOFT"]["k0"], tyres["MEDIUM"]["k0"] + 1e-9)


class FreeAirTests(unittest.TestCase):
    def test_laps_close_behind_another_car_are_left_out(self):
        laps = pd.DataFrame({
            "Driver": ["AAA"] * 3 + ["BBB"] * 3, "LapNumber": [1, 2, 3] * 2,
            "LapEndTime_s": [100.0, 170.0, 240.0, 101.0, 174.0, 248.0],
            "CleanLap": True, "Compound": "HARD"})
        kept = free_air(add_gaps(laps))
        # BBB is 1 s behind after lap 1 and 4 s after lap 2: only its lap 3 starts and ends clear.
        self.assertEqual(sorted(zip(kept.Driver, kept.LapNumber)),
                         [("AAA", 1), ("AAA", 2), ("AAA", 3), ("BBB", 3)])


class PitStopTests(unittest.TestCase):
    def make(self, life_after_stop):
        laps = pd.DataFrame({"Driver": "AAA", "LapNumber": range(1, 13), "LapTime_s": 70.0,
                             "CleanLap": True, "IsInLap": False, "IsOutLap": False, "GreenFlag": True,
                             "Compound": "MEDIUM", "TyreLife": range(1, 13)})
        laps.loc[5, ["LapTime_s", "CleanLap", "IsInLap"]] = [73.5, False, True]          # lap 6
        laps.loc[6, ["LapTime_s", "CleanLap", "IsOutLap"]] = [88.0, False, True]         # lap 7
        laps.loc[6:, "LapTime_s"] -= 1.0                                                 # faster after
        laps.loc[6:, "TyreLife"] = range(life_after_stop, life_after_stop + 6)
        laps.loc[6, "LapTime_s"] = 88.0
        return laps

    def test_losses_are_measured_against_each_side_of_the_stop(self):
        stop, = pit_stops(self.make(life_after_stop=1))
        self.assertEqual((stop["in_lap"], stop["green"], stop["tyre_change"]), (6, True, True))
        self.assertAlmostEqual(stop["in_lap_loss_s"], 3.5)
        self.assertAlmostEqual(stop["out_lap_loss_s"], 19.0)

    def test_drive_through_is_not_a_tyre_change(self):
        stop, = pit_stops(self.make(life_after_stop=7))
        self.assertFalse(stop["tyre_change"])


if __name__ == "__main__":
    unittest.main()
