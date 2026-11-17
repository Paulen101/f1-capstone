"""Behavior checks for label selection; Python 3.8 stdlib only, no simulator needed."""
import unittest

from integration.label_generator import (check_stop, clear_recorded_events, pick_fastest, pick_label,
                                         run_seed, summarize)


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
