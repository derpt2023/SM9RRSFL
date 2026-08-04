import unittest

from sm9rrsfl.weighting import SuspicionWeightManager


class SuspicionWeightManagerTest(unittest.TestCase):
    def test_suspicious_tag_requires_trace_certificate_before_revocation(self):
        manager = SuspicionWeightManager(
            ["tag-0", "tag-1", "tag-2"],
            penalty_factor=0.5,
            remove_after=3,
        )

        active = ["tag-0", "tag-1", "tag-2"]
        first = manager.update(active, {"tag-1"}, {"tag-1"})
        self.assertLess(first.weights["tag-1"], first.weights["tag-0"])
        self.assertEqual(first.trace_requested_tags, set())

        second = manager.update(active, {"tag-1"}, {"tag-1"})
        self.assertEqual(second.trace_requested_tags, set())

        third = manager.update(active, {"tag-1"}, {"tag-1"})
        self.assertEqual(third.trace_requested_tags, {"tag-1"})
        self.assertNotIn("tag-1", manager.revoked)
        self.assertEqual(third.weights["tag-1"], 0.0)

        manager.confirm_revocation("tag-1")
        self.assertIn("tag-1", manager.revoked)
        after_certificate = manager.update(active, set(), set())
        self.assertEqual(after_certificate.weights["tag-1"], 0.0)

    def test_normal_client_weight_recovers(self):
        manager = SuspicionWeightManager(
            ["tag-0", "tag-1"],
            penalty_factor=0.5,
            recovery_factor=2.0,
            remove_after=3,
        )
        penalized = manager.update(["tag-0", "tag-1"], {"tag-0"}, set())
        self.assertLess(penalized.weights["tag-0"], 0.5)
        recovered = manager.update(["tag-0", "tag-1"], set(), set())
        self.assertGreaterEqual(recovered.weights["tag-0"], penalized.weights["tag-0"])

    def test_ctol_trigger_is_zero_but_not_revoked_before_certificate(self):
        manager = SuspicionWeightManager(["tag-0", "tag-1"], remove_after=1)

        trigger = manager.update(["tag-0", "tag-1"], {"tag-0"}, {"tag-0"})
        self.assertEqual(trigger.weights["tag-0"], 0.0)
        self.assertNotIn("tag-0", manager.revoked)
        self.assertIn("tag-0", manager.pending_trace)
        self.assertAlmostEqual(sum(manager.weights.values()), 1.0)

    def test_single_indicator_suspicion_downweights_without_advancing_count(self):
        manager = SuspicionWeightManager(
            ["tag-0", "tag-1"],
            penalty_factor=0.5,
            remove_after=2,
        )

        for _ in range(3):
            result = manager.update(["tag-0", "tag-1"], {"tag-0"}, set())

        self.assertLess(result.weights["tag-0"], result.weights["tag-1"])
        self.assertEqual(manager.consecutive_suspicions["tag-0"], 0)
        self.assertEqual(result.trace_requested_tags, set())
        self.assertEqual(manager.pending_trace, set())

    def test_single_indicator_round_preserves_dual_indicator_evidence(self):
        manager = SuspicionWeightManager(
            ["tag-0", "tag-1"],
            remove_after=2,
        )

        first = manager.update(["tag-0", "tag-1"], {"tag-0"}, {"tag-0"})
        self.assertEqual(first.trace_requested_tags, set())
        self.assertEqual(manager.consecutive_suspicions["tag-0"], 1)

        single = manager.update(["tag-0", "tag-1"], {"tag-0"}, set())
        self.assertEqual(single.trace_requested_tags, set())
        self.assertEqual(manager.consecutive_suspicions["tag-0"], 1)

        third = manager.update(["tag-0", "tag-1"], {"tag-0"}, {"tag-0"})
        self.assertEqual(third.trace_requested_tags, {"tag-0"})
        self.assertEqual(manager.consecutive_suspicions["tag-0"], 2)

    def test_fully_normal_round_floor_halves_dual_indicator_evidence(self):
        manager = SuspicionWeightManager(
            ["tag-0", "tag-1"],
            remove_after=6,
        )

        for _ in range(5):
            manager.update(["tag-0", "tag-1"], {"tag-0"}, {"tag-0"})
        normal = manager.update(["tag-0", "tag-1"], set(), set())

        self.assertEqual(normal.trace_requested_tags, set())
        self.assertEqual(manager.consecutive_suspicions["tag-0"], 2)

        manager.update(["tag-0", "tag-1"], set(), set())
        self.assertEqual(manager.consecutive_suspicions["tag-0"], 1)

        manager.update(["tag-0", "tag-1"], set(), set())
        self.assertEqual(manager.consecutive_suspicions["tag-0"], 0)

    def test_count_increment_tags_must_be_suspicious(self):
        manager = SuspicionWeightManager(["tag-0", "tag-1"])

        with self.assertRaisesRegex(ValueError, "subset"):
            manager.update(["tag-0", "tag-1"], set(), {"tag-0"})

if __name__ == "__main__":
    unittest.main()
