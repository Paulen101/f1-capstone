"""Checks for the fitted-parameter writer (any Python 3.8+)."""
import unittest

from integration.write_fitted_pars import (TUM_PARAMETERS, build, changed_values, compound_codes,
                                           parse_pars, set_number)

STOCK = (TUM_PARAMETERS / "pars_Spielberg_2019.ini").read_bytes().decode("utf-8")
NAMES = {"A3": "HARD", "A4": "MEDIUM", "A6": "SOFT"}


def make_fit(pars, extra_in=0.0, extra_out=0.0):
    """A fit that says exactly what the TUM file says, apart from the extra pit losses."""
    track, drivers = pars["track_pars"], pars["driver_pars"]
    stops = []
    for driver in drivers:
        standstill = track["t_pit_tirechange_min"] + pars["car_pars"][drivers[driver]["team"]]["t_pit_tirechange_add"]
        stops.append({"driver": driver, "used": True,
                      "in_lap_loss_s": track["t_pitdrive_inlap"] + extra_in,
                      "out_lap_loss_s": track["t_pitdrive_outlap"] + standstill
                                        + pars["tireset_pars"][driver]["t_add_coldtires"] + extra_out})
    stops.append({"driver": "MAG", "used": False, "in_lap_loss_s": 99.0, "out_lap_loss_s": 99.0})
    return {"pit": {"stops": stops},
            "stints": {d: [NAMES[stint[1]] for stint in drivers[d]["strategy_info"]] for d in drivers},
            "tyres": {"HARD": {"k0_s": 0.0, "k1_s_per_lap": 0.02},
                      "MEDIUM": {"k0_s": -0.1, "k1_s_per_lap": 0.03},
                      "SOFT": {"k0_s": -0.3, "k1_s_per_lap": 0.05}},
            "trend_s_per_lap": 0.031,
            "driver_level_s": {d: 70.0 + 0.1 * i for i, d in enumerate(sorted(drivers))},
            "drivers_below_min_laps": []}


class WriterTests(unittest.TestCase):
    def setUp(self):
        self.pars = parse_pars(STOCK)

    def test_tum_own_pit_losses_change_nothing(self):
        text, changes = build(STOCK, make_fit(self.pars), ("pit",))
        self.assertEqual(changes, [])
        self.assertEqual(text, STOCK)

    def test_extra_pit_loss_goes_to_the_right_lap(self):
        _, changes = build(STOCK, make_fit(self.pars, extra_in=0.5, extra_out=3.0), ("pit",))
        self.assertEqual({c["parameter"]: c["fitted"] for c in changes},
                         {"track_pars.t_pitdrive_inlap": 3.191, "track_pars.t_pitdrive_outlap": 14.396})

    def test_too_few_stops_fail(self):
        fit = make_fit(self.pars)
        fit["pit"]["stops"] = fit["pit"]["stops"][:4]
        with self.assertRaises(ValueError):
            build(STOCK, fit, ("pit",))

    def test_compounds_match_tum_codes(self):
        self.assertEqual(compound_codes(make_fit(self.pars), self.pars),
                         {"HARD": "A3", "MEDIUM": "A4", "SOFT": "A6"})

    def test_tyres_and_pace_reproduce_the_fitted_lap_time(self):
        fit = make_fit(self.pars)
        text, _ = build(STOCK, fit, ("tyres", "pace"))
        new = parse_pars(text)
        track = new["track_pars"]
        self.assertAlmostEqual(track["t_lap_sens_mass"] * 1.549, 0.031, places=4)
        for driver, level in fit["driver_level_s"].items():
            car = new["car_pars"][new["driver_pars"][driver]["team"]]
            first_lap_new_hard = (track["t_q"] + track["t_gap_racepace"] + car["t_car"]
                                  + new["driver_pars"][driver]["t_driver"]
                                  + car["m_fuel"] * track["t_lap_sens_mass"])
            self.assertAlmostEqual(first_lap_new_hard, level, places=2)
            self.assertEqual(new["tireset_pars"][driver]["A6"]["k_0"], -0.3)
            self.assertEqual(new["tireset_pars"][driver]["A4"]["k_1_lin"], 0.03)
            self.assertEqual(new["tireset_pars"][driver]["A4"]["k_1_quad"],
                             self.pars["tireset_pars"][driver]["A4"]["k_1_quad"])

    def test_tyres_alone_keep_tum_pace_offsets(self):
        text, changes = build(STOCK, make_fit(self.pars), ("tyres",))
        new = parse_pars(text)
        self.assertEqual(new["car_pars"], self.pars["car_pars"])
        self.assertEqual(new["driver_pars"], self.pars["driver_pars"])
        self.assertTrue(all(c["parameter"].split(".")[0] in ("track_pars", "tireset_pars") for c in changes))

    def test_driver_without_free_air_laps_follows_team_mate(self):
        fit = make_fit(self.pars)
        fit["drivers_below_min_laps"] = ["PER"]
        fit["driver_level_s"]["PER"] = 99.0
        new = parse_pars(build(STOCK, fit, ("tyres", "pace"))[0])["driver_pars"]
        tum_gap = self.pars["driver_pars"]["PER"]["t_driver"] - self.pars["driver_pars"]["STR"]["t_driver"]
        self.assertAlmostEqual(new["PER"]["t_driver"] - new["STR"]["t_driver"], tum_gap, places=3)

    def test_pace_needs_tyres(self):
        with self.assertRaises(ValueError):
            build(STOCK, make_fit(self.pars), ("pace",))

    def test_file_layout_is_kept(self):
        text, changes = build(STOCK, make_fit(self.pars, extra_out=3.0), ("pit", "tyres", "pace"))
        self.assertEqual(len(text.splitlines()), len(STOCK.splitlines()))
        differing = sum(a != b for a, b in zip(STOCK.splitlines(), text.splitlines()))
        self.assertEqual(differing, len(changes))

    def test_traffic_values_go_to_race_and_track(self):
        traffic = {"values": {"min_t_dist": {"measured": 0.488}, "t_duel": {"measured": 0.033},
                              "t_overtake_loser": {"measured": 1.33}, "t_gap_overtake": {"measured": 0.315}}}
        _, changes = build(STOCK, make_fit(self.pars), ("traffic",), traffic)
        self.assertEqual({c["parameter"]: c["fitted"] for c in changes},
                         {"race_pars.min_t_dist": 0.488, "race_pars.t_duel": 0.033,
                          "race_pars.t_overtake_loser": 1.33, "track_pars.t_gap_overtake": 0.315})

    def test_traffic_needs_its_measurement_and_a_positive_threshold(self):
        with self.assertRaises(ValueError):
            build(STOCK, make_fit(self.pars), ("traffic",))
        traffic = {"values": {"min_t_dist": {"measured": 0.5}, "t_duel": {"measured": 0.3},
                              "t_overtake_loser": {"measured": 0.3}, "t_gap_overtake": {"measured": 0.0}}}
        with self.assertRaises(ValueError):
            build(STOCK, make_fit(self.pars), ("traffic",), traffic)


class SetNumberTests(unittest.TestCase):
    def test_only_the_addressed_value_changes(self):
        text = set_number(STOCK, ("tireset_pars", "VER", "A4", "k_1_lin"), 0.5)
        self.assertEqual(changed_values(parse_pars(STOCK), parse_pars(text)),
                         [("tireset_pars.VER.A4.k_1_lin", 0.01, 0.5)])

    def test_similar_keys_are_not_touched(self):
        text = set_number(STOCK, ("track_pars", "t_pitdrive_inlap"), 3.5)
        self.assertEqual(changed_values(parse_pars(STOCK), parse_pars(text)),
                         [("track_pars.t_pitdrive_inlap", 2.691, 3.5)])

    def test_unknown_key_fails(self):
        with self.assertRaises(KeyError):
            set_number(STOCK, ("track_pars", "t_not_a_parameter"), 1.0)


if __name__ == "__main__":
    unittest.main()
