"""Behavior checks for label selection; Python 3.8 stdlib only, no simulator needed."""
import unittest

from integration.label_generator import (check_stop, clear_recorded_events, close_to_label,
                                         complete_background_strategies,
                                         paired_differences, pick_fastest, pick_label, run_seed,
                                         summarize)


def row(pit_lap, position, laps=71.0, time=5000.0, finished=True):
    return {"PitLap": pit_lap, "Position": position, "SimulatedLaps": laps,
            "RaceTime_s": time, "Finished": finished}


class SummaryTests(unittest.TestCase):
    def test_best_mean_position_wins(self):
        rows = [row(10, 1), row(10, 5), row(11, 2), row(11, 3)]
        self.assertEqual(pick_label(summarize(rows, 71)), 11)

    def test_tie_goes_to_lower_race_time(self):
        rows = [row(10, 1, time=5000.0), row(11, 1, time=4990.0)]
        self.assertEqual(pick_label(summarize(rows, 71)), 11)

    def test_lapped_car_does_not_win_tie_on_shorter_time(self):
        rows = [row(10, 4, laps=70.0, time=4950.0), row(11, 4, laps=71.0, time=4990.0)]
        self.assertEqual(pick_label(summarize(rows, 71)), 11)

    def test_retired_runs_are_left_out_of_position(self):
        rows = [row(10, 2), row(10, 20, laps=12.0, finished=False), row(11, 3)]
        summary = summarize(rows, 71)
        self.assertEqual(summary[0]["MeanPosition"], 2)
        self.assertEqual(summary[0]["Runs"], 2)
        self.assertEqual(summary[0]["FinishedRuns"], 1)
        self.assertEqual(pick_label(summary), 10)

    def test_lap_with_no_finishers_cannot_be_the_label(self):
        rows = [row(10, 20, laps=3.0, finished=False), row(11, 15)]
        self.assertEqual(pick_label(summarize(rows, 71)), 11)
        self.assertIsNone(pick_label(summarize(rows[:1], 71)))

    def test_fastest_ignores_laps_without_full_distance(self):
        rows = [row(10, 4, laps=70.0, time=4950.0), row(11, 2, time=4990.0), row(12, 3, time=4985.0)]
        self.assertEqual(pick_fastest(summarize(rows, 71)), 12)

    def test_position_and_fastest_can_disagree(self):
        rows = [row(23, 1, time=5010.0), row(52, 2, time=4990.0)]
        summary = summarize(rows, 71)
        self.assertEqual((pick_label(summary), pick_fastest(summary)), (23, 52))


class PairedTests(unittest.TestCase):
    def runs(self, pit_lap, positions, times):
        return [dict(row(pit_lap, p, time=t), Run=i) for i, (p, t) in enumerate(zip(positions, times))]

    def test_differences_are_paired_by_run(self):
        rows = self.runs(10, [1, 3, 5], [5000.0, 5010.0, 5020.0]) + self.runs(11, [2, 4, 6], [5001.0, 5011.0, 5021.0])
        paired = paired_differences(rows, 10, 71)
        self.assertEqual(paired[11]["PosDiffVsLabel"], 1)
        self.assertEqual(paired[11]["PosDiffSE"], 0)                # every run exactly one place worse
        self.assertEqual(paired[11]["TimeDiffVsLabel_s"], 1.0)
        self.assertEqual(paired[10]["PosDiffVsLabel"], 0)

    def test_laps_within_two_standard_errors_are_close(self):
        rows = (self.runs(10, [1, 1, 1, 1], [5000.0] * 4) + self.runs(11, [1, 2, 1, 1], [5000.0] * 4)
                + self.runs(12, [3, 3, 3, 3], [5000.0] * 4))
        self.assertEqual(close_to_label(paired_differences(rows, 10, 71)), [10, 11])

    def test_retired_runs_are_not_paired(self):
        rows = self.runs(10, [1, 1], [5000.0] * 2) + [dict(row(11, 20, laps=30.0, finished=False), Run=0),
                                                      dict(row(11, 2), Run=1)]
        self.assertEqual(paired_differences(rows, 10, 71)[11]["PairedRuns"], 1)


class BackgroundStrategyTests(unittest.TestCase):
    def pars(self, strategies):
        return {"vse_pars": {"real_strategy": strategies},
                "driver_pars": {d: {"strategy_info": [list(x) for x in s]} for d, s in strategies.items()}}

    def test_one_compound_finishers_get_the_most_common_last_compound(self):
        pars = self.pars({"AAA": [[0, "A4", 2, 0.0], [20, "A3", 0, 0.0]],
                          "BBB": [[0, "A4", 2, 0.0], [24, "A3", 0, 0.0]],
                          "CCC": [[0, "A6", 2, 0.0], [30, "A4", 0, 0.0]],
                          "SAI": [[0, "A6", 2, 0.0], [27, "A6", 29, 0.0]],
                          "VET": [[0, "A4", 2, 0.0]]})
        changes = complete_background_strategies(pars, "AAA")
        strategies = pars["vse_pars"]["real_strategy"]
        self.assertEqual(strategies["SAI"], [[0, "A6", 2, 0.0], [27, "A3", 0, 0.0]])
        self.assertEqual(strategies["VET"], [[0, "A4", 2, 0.0], [24, "A3", 0, 0.0]])
        self.assertEqual(pars["driver_pars"]["VET"]["strategy_info"], strategies["VET"])
        self.assertEqual([c["driver"] for c in changes], ["SAI", "VET"])

    def test_target_driver_and_complete_strategies_are_left_alone(self):
        pars = self.pars({"AAA": [[0, "A4", 2, 0.0]], "BBB": [[0, "A4", 2, 0.0], [24, "A3", 0, 0.0]]})
        self.assertEqual(complete_background_strategies(pars, "AAA"), [])
        self.assertEqual(pars["vse_pars"]["real_strategy"]["AAA"], [[0, "A4", 2, 0.0]])


class StopCheckTests(unittest.TestCase):
    def test_requested_stop_passes(self):
        self.assertTrue(check_stop([(25, "A3")], 25, "A3", True))

    def test_retiring_before_the_stop_is_allowed(self):
        self.assertFalse(check_stop([], 25, "A3", False))

    def test_finishing_without_the_stop_is_an_error(self):
        with self.assertRaises(AssertionError):
            check_stop([], 25, "A3", True)

    def test_wrong_lap_compound_or_count_is_an_error(self):
        for stops in ([(26, "A3")], [(25, "A4")], [(25, "A3"), (40, "A4")]):
            with self.assertRaises(AssertionError):
                check_stop(stops, 25, "A3", True)


class RandomnessTests(unittest.TestCase):
    def test_same_run_gets_same_seed_and_runs_differ(self):
        self.assertEqual(run_seed(0, 3), run_seed(0, 3))
        self.assertNotEqual(run_seed(0, 3), run_seed(0, 4))

    def test_retry_seed_does_not_collide_with_another_run(self):
        seeds = {run_seed(0, run, attempt) for run in range(1000) for attempt in range(5)}
        self.assertEqual(len(seeds), 5000)

    def test_clearing_events_requests_random_ones(self):
        events = {"fcy_data": {"phases": [[45.6, 52.0, "SC", None, None]], "domain": "progress"},
                  "retire_data": {"retirements": [["VER", 12.0]], "domain": "progress"}}
        clear_recorded_events(events)
        self.assertEqual(events["fcy_data"], {"phases": [], "domain": "progress"})
        self.assertEqual(events["retire_data"], {"retirements": [], "domain": "progress"})


if __name__ == "__main__":
    unittest.main()
