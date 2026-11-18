"""Checks for the pit-timing network adapter (.venv-tum; the simulator test runs one Spielberg race).

    .venv-tum/Scripts/python.exe -m unittest integration.test_pit_network
"""
import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from integration.pit_network import (FEATURES, NetworkVSE, PitNetwork, degradation, feature_vector, pit_now,
                                     tyre_features)

ROOT = Path(__file__).resolve().parents[1]
QUAD = {"tire_deg_model": "quad", "A3": {"k_0": 0.0, "k_1_quad": 0.02, "k_2_quad": 0.001},
        "A4": {"k_0": -0.5, "k_1_quad": 0.05, "k_2_quad": 0.001}}


class StubNetwork:
    """Pits from a given lap on (stay: +1 place, pit: -1 place), reading the lap from RaceProgress."""

    def __init__(self, tot_no_laps: int, pit_from: int):
        self.tot, self.pit_from = tot_no_laps, pit_from
        self.inputs = []

    def predict(self, x):
        x = np.atleast_2d(x)
        self.inputs.append(x[0])
        lap = round(x[0][FEATURES.index("RaceProgress")] * self.tot)
        return np.array([[-1.0, -1.0] if lap >= self.pit_from else [1.0, 1.0]])


class FeatureTests(unittest.TestCase):
    def test_degradation_matches_tum(self):
        from integration.label_generator import setup
        setup()
        from helper_funcs.src.calc_tire_degradation import calc_tire_degradation
        lin = {"tire_deg_model": "lin", "A3": {"k_0": 0.1, "k_1_lin": 0.03}}
        for pars, compound in ((QUAD, "A4"), (lin, "A3")):
            for age in (0, 5, 30):
                self.assertAlmostEqual(degradation(pars, compound, age),
                                       calc_tire_degradation(age, 1, compound, pars))

    def test_tyre_features(self):
        f = tyre_features(QUAD, "A4", "A3", age=10, laps_remaining=2)
        self.assertAlmostEqual(f["TyreDelta_s"], (-0.5 + 0.5 + 0.1) - 0.0)
        self.assertAlmostEqual(f["WearRate_s"], (0.05 * 11 + 0.121) - (0.05 * 10 + 0.1))
        self.assertAlmostEqual(f["StopWearRate_s"], (0.04 + 0.004) / 2)
        stay = (degradation(QUAD, "A4", 10) - 0.0) + (degradation(QUAD, "A4", 11) - degradation(QUAD, "A3", 1))
        self.assertAlmostEqual(f["StayCost_s"], stay)

    def test_missing_gaps_are_free_air(self):
        situation = {"RaceProgress": 0.5, "LapsRemaining": 30, "TyreAge": 30, "TyreAgeProgress": 0.5,
                     "RelCompound": 2, "Position": 1, "GapAhead_s": float("nan"), "GapBehind_s": 45.0,
                     "CloseAheadPrevLap": "False", "PursuerPittedPrevLap": True, "FlagState": "VSC",
                     "LocationCat": 2, "PitLoss_s": 20.0, "TyreDelta_s": 1.0, "WearRate_s": 0.1,
                     "StopWearRate_s": 0.05, "StayCost_s": 40.0}
        v = dict(zip(FEATURES, feature_vector(situation)))
        self.assertEqual((v["GapAhead_s"], v["NoCarAhead"]), (30.0, 1.0))
        self.assertEqual((v["GapBehind_s"], v["NoCarBehind"]), (30.0, 0.0))
        self.assertEqual((v["RelSoft"], v["RelHard"], v["FlagVSC"], v["FlagSC"]), (1.0, 0.0, 1.0, 0.0))
        self.assertEqual((v["CloseAheadPrevLap"], v["PursuerPittedPrevLap"]), (0.0, 1.0))

    def test_decision_rule(self):
        self.assertTrue(pit_now(-0.5, 10.0))       # gains places: pit even if slower
        self.assertFalse(pit_now(0.5, -10.0))      # loses places: stay even if faster
        self.assertTrue(pit_now(0.05, -1.0))       # tie in position: time decides
        self.assertFalse(pit_now(-0.05, 1.0))

    def test_numpy_network_round_trip(self):
        rng = np.random.default_rng(0)
        w0, b0, w1, b1 = rng.normal(size=(4, len(FEATURES))), rng.normal(size=4), rng.normal(size=(2, 4)), rng.normal(size=2)
        mean, std = rng.normal(size=len(FEATURES)), np.abs(rng.normal(size=len(FEATURES))) + 0.5
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "net.npz"
            np.savez(path, features=np.array(FEATURES), x_mean=mean, x_std=std, y_scale=np.array([1.0, 10.0]),
                     n_layers=np.array(2), w0=w0, b0=b0, w1=w1, b1=b1)
            x = rng.normal(size=(3, len(FEATURES)))
            expected = (np.maximum(((x - mean) / std) @ w0.T + b0, 0) @ w1.T + b1) * [1.0, 10.0]
            np.testing.assert_allclose(PitNetwork(path).predict(x), expected)


class SimulatorTests(unittest.TestCase):
    """One Monte Carlo Spielberg race with the adapter installed for VER."""

    @classmethod
    def setUpClass(cls):
        import argparse
        from integration.label_generator import load, setup
        setup()
        args = argparse.Namespace(race=str(ROOT / "integration/parameters/pars_Spielberg_2019_fitted.ini"),
                                  driver="VER", deterministic=False, events="sampled")
        cls.source, cls.pars, cls.paths, _, _ = load(args)
        cls.tot = cls.pars["race_pars"]["tot_no_laps"]

    def race(self, network, seed=3, compound_choice=None, second_compound=None):
        from racesim.src.race import Race
        random.seed(seed)
        np.random.seed(seed)
        race = Race(**self.pars, vse_paths=self.paths, use_prob_infl=True, create_rand_events=True)
        race.vse = NetworkVSE(race.vse, race, "VER", network, None if compound_choice else "A3",
                              self.pars["vse_pars"]["location_cat"], 20.0, compound_choice=compound_choice,
                              second_compound=second_compound)
        race.simulate_race()
        self.assertEqual(race.result_status, 0)
        return race, race.vse

    def stops(self, race):
        idx = next(i for i, d in enumerate(race.drivers_list) if d.initials == "VER")
        return [(s[0], s[1]) for s in race.drivers_list[idx].strategy_info[1:]]

    def test_pits_once_on_the_first_pit_decision(self):
        race, _ = self.race(StubNetwork(self.tot, pit_from=30))
        self.assertEqual(self.stops(race), [(30, "A3")])

    def test_forced_stop_on_the_last_possible_lap(self):
        race, _ = self.race(StubNetwork(self.tot, pit_from=10 ** 6))
        self.assertEqual(self.stops(race), [(self.tot - 1, "A3")])

    def test_live_situation_equals_the_recorded_one(self):
        """The training rows were recorded after the race; the network sees them during it. Same values."""
        network = StubNetwork(self.tot, pit_from=10 ** 6)
        race, vse = self.race(network)
        live = network.inputs
        self.assertEqual(len(live), self.tot - 1 - 2)      # laps 2 .. tot-2 asked; tot-1 is the forced stop
        for lap, x in zip(range(2, self.tot - 1), live):
            recorded = feature_vector(vse.situation(lap))
            for name, a, b in zip(FEATURES, x, recorded):
                self.assertTrue(math.isclose(a, b, abs_tol=1e-9), "lap %d %s: live %s, after %s" % (lap, name, a, b))

    def test_second_stop_after_min_stint(self):
        network = StubNetwork(self.tot, pit_from=30)
        race, vse = self.race(network, second_compound="A6")
        self.assertEqual(self.stops(race), [(30, "A3"), (40, "A6")])
        # the second-stint situation seen live equals the one recorded after the race
        live = dict(zip(FEATURES, network.inputs[-1]))
        recorded = dict(zip(FEATURES, feature_vector(vse.situation(40, "A6"))))
        self.assertEqual(live, recorded)
        self.assertEqual((live["StopsMade"], live["TyreAge"], live["RelSoft"]), (1.0, 10.0, 0.0))

    def test_no_second_stop_without_second_compound(self):
        race, _ = self.race(StubNetwork(self.tot, pit_from=30))
        self.assertEqual(self.stops(race), [(30, "A3")])

    def compound_choice(self):
        from integration.tum_compound_choice import TumCompoundChoice
        vse = self.pars["vse_pars"]
        return TumCompoundChoice(self.paths, self.pars["track_pars"]["name"], vse["available_compounds"],
                                 vse["param_dry_compounds"])

    def test_tum_compound_choice_equals_tum_vse(self):
        """Same network output as TUM's own VSE_SUPERVISED pipeline, for every compound and race phase."""
        from racesim.src.vse_supervised import VSE_SUPERVISED
        choice = self.compound_choice()
        tum = VSE_SUPERVISED(self.paths["supervised_preprocessor_cc"], self.paths["supervised_preprocessor_tc"],
                             self.paths["supervised_nnmodel_cc"], self.paths["supervised_nnmodel_tc"])
        interpreter = tum.nnmodel_cc["interpreter"]
        for compound in ("A3", "A4", "A6"):
            for progress in (0.1, 0.4, 0.7):
                tum.reset()
                tum.preprocess_features(tireageprogress_corr_zeroinchange=[0.3], raceprogress=progress, position=[5],
                                        rel_compound_num_nl=[choice.rel_compound(compound)], fcy_stat_nl=[0],
                                        remainingtirechanges_nl=[1], tirechange_pursuer=[0], location_cat=2,
                                        close_ahead_prevlap=[0], location="Spielberg", used_2compounds_nl=[False],
                                        no_avail_dry_compounds=3)
                interpreter.set_tensor(tum.nnmodel_cc["input_index"], tum.X_conv_cc[:1])
                interpreter.invoke()
                expected = interpreter.get_tensor(tum.nnmodel_cc["output_index"])[0]
                np.testing.assert_allclose(choice.probabilities(compound, progress), expected, rtol=1e-6)
                chosen = choice.choose(compound, progress)
                self.assertIn(chosen, ("A3", "A4", "A6"))
                self.assertNotEqual(chosen, compound)

    def test_stop_mounts_tum_compound(self):
        choice = self.compound_choice()
        race, vse = self.race(StubNetwork(self.tot, pit_from=30), compound_choice=choice)
        start = race.drivers_list[vse.idx].strategy_info[0][1]
        expected = choice.choose(start, 29 / self.tot)
        self.assertEqual(self.stops(race), [(30, expected)])
        self.assertEqual(vse.decisions[-1]["Compound"], expected)


if __name__ == "__main__":
    unittest.main()
