import unittest

from sm9rrsfl.weighting import SuspicionWeightManager


class SuspicionWeightManagerTest(unittest.TestCase):
    def test_suspicious_client_is_downweighted_before_removal(self):
        manager = SuspicionWeightManager(
            ["client-0", "client-1", "client-2"],
            penalty_factor=0.5,
            remove_after=3,
        )

        first = manager.update(["client-0", "client-1", "client-2"], {"client-1"}, {"client-1"})
        self.assertLess(first.weights["client-1"], first.weights["client-0"])
        self.assertEqual(first.newly_removed, set())

        second = manager.update(["client-0", "client-1", "client-2"], {"client-1"}, {"client-1"})
        self.assertEqual(second.newly_removed, set())

        third = manager.update(["client-0", "client-1", "client-2"], {"client-1"}, {"client-1"})
        self.assertEqual(third.newly_removed, {"client-1"})
        self.assertEqual(third.true_positive_removed, 1)
        self.assertEqual(third.weights["client-1"], 0.0)

    def test_normal_client_weight_recovers(self):
        manager = SuspicionWeightManager(
            ["client-0", "client-1"],
            penalty_factor=0.5,
            recovery_factor=2.0,
            remove_after=3,
        )
        penalized = manager.update(["client-0", "client-1"], {"client-0"}, set())
        self.assertLess(penalized.weights["client-0"], 0.5)
        recovered = manager.update(["client-0", "client-1"], set(), set())
        self.assertGreaterEqual(recovered.weights["client-0"], penalized.weights["client-0"])


if __name__ == "__main__":
    unittest.main()
