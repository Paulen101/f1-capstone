"""Behavior checks for the single-driver timing adapter; Python 3.8 stdlib only."""
import unittest

from integration.fixed_lap_policy import FixedLapPolicy, FixedLapVSE


class BackgroundVSE:
    param_dry_compounds = ["A3", "A4", "A6"]
    vse_pars = {"vse_type": {"VER": "realstrategy", "HAM": "realstrategy"}}

    def decide_pitstop(self, **state):
        # A recorded stop for VER on 31 must be suppressed. HAM's stop remains.
        return ["A3" if state["cur_lap"] == 31 else None,
                "A3" if state["cur_lap"] == 25 else None]


class PolicyTests(unittest.TestCase):
    def test_exact_lap_and_one_stop(self):
        policy = FixedLapPolicy(25, "A3")
        self.assertIsNone(policy.decide(24, 71, True, 0))
        self.assertEqual(policy.decide(25, 71, True, 0), "A3")
        self.assertIsNone(policy.decide(26, 71, True, 0))
        self.assertIsNone(policy.decide(25, 71, True, 1))

    def test_retirement_and_invalid_schedule(self):
        self.assertIsNone(FixedLapPolicy(25, "A3").decide(25, 71, False, 0))
        for lap in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                FixedLapPolicy(lap, "A3")
        with self.assertRaises(ValueError):
            FixedLapPolicy(71, "A3").decide(1, 71, True, 0)
        with self.assertRaises(ValueError):
            FixedLapPolicy(25, "W")

    def test_adapter_preserves_background_and_suppresses_old_stop(self):
        adapter = FixedLapVSE(BackgroundVSE(), "VER", FixedLapPolicy(25, "A3"))
        state = dict(driver_initials=["VER", "HAM"], tot_no_laps=71,
                     bool_driving=[True, True], no_past_tirechanges=[0, 0])
        self.assertEqual(adapter.decide_pitstop(cur_lap=25, **state), ["A3", "A3"])
        state["no_past_tirechanges"][0] = 1
        self.assertEqual(adapter.decide_pitstop(cur_lap=31, **state), [None, None])
        self.assertEqual(adapter.decisions[0]["Lap"], 25)

    def test_compound_must_be_available(self):
        with self.assertRaises(ValueError):
            FixedLapVSE(BackgroundVSE(), "VER", FixedLapPolicy(25, "A1"))


if __name__ == "__main__":
    unittest.main()
