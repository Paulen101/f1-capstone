"""Checks for the leave-one-race-out traffic values (any Python 3.8+)."""
import unittest

from integration.heldout_traffic import pooled


def values(gap, tum_gap, dist=0.5, duel=0.0, loser=1.0):
    return {"t_gap_overtake": {"measured": gap, "tum": tum_gap}, "min_t_dist": {"measured": dist},
            "t_duel": {"measured": duel}, "t_overtake_loser": {"measured": loser}}


class PooledTests(unittest.TestCase):
    def test_own_race_is_never_used(self):
        traffic = {"A": values(0.1, 2.0, dist=9.0, loser=9.0), "B": values(1.0, 2.0, loser=1.0),
                   "C": values(1.5, 3.0, loser=2.0)}
        result = pooled(traffic, "A")
        self.assertAlmostEqual(result["t_gap_overtake"], 2.0 * 0.5)       # ratio 0.5 at both others
        self.assertAlmostEqual(result["min_t_dist"], 0.5)
        self.assertAlmostEqual(result["t_overtake_loser"], 1.5)

    def test_negative_duel_loss_becomes_zero(self):
        traffic = {"A": values(1.0, 2.0), "B": values(1.0, 2.0, duel=-0.2), "C": values(1.0, 2.0, duel=-0.1)}
        self.assertEqual(pooled(traffic, "A")["t_duel"], 0.0)


if __name__ == "__main__":
    unittest.main()
