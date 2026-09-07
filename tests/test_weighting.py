import unittest
from sm9rrsfl.weighting import SuspicionWeightManager, bounded_aggregation_coefficients
from sm9rrsfl.svd_detector import DetectionResult


class SuspicionWeightManagerTest(unittest.TestCase):
    def test_trace_requires_count_and_certificate(self):
        m = SuspicionWeightManager(["a", "b", "c"], remove_after=3)
        for _ in range(2):
            self.assertFalse(m.update(["a", "b", "c"], {"a"}, {"a"}).trace_requested_tags)
        d = m.update(["a", "b", "c"], {"a"}, {"a"})
        self.assertEqual(d.trace_requested_tags, {"a"})
        self.assertEqual(d.weights["a"], 0)
        self.assertNotIn("a", m.revoked)
        m.confirm_revocation("a")
        self.assertIn("a", m.revoked)
        with self.assertRaises(ValueError):
            m.confirm_revocation("b")

    def test_mild_suspicion_counts_and_penalizes_before_ctol(self):
        m = SuspicionWeightManager(["a", "b"], remove_after=3)
        d = m.update(["a", "b"], {"a"}, {"a"})
        self.assertEqual(m.evidence_counts["a"], 1)
        self.assertEqual(d.weights["a"], .1)
        self.assertFalse(d.trace_requested_tags)
        with self.assertRaises(ValueError):
            m.update(["a", "b"], set(), {"a"})
        with self.assertRaises(ValueError):
            m.update(["a", "b"], {"a"}, set(), immediate_revocation_tags={"a"})

    def test_severe_deviation_bypasses_ctol_but_not_certificate(self):
        m = SuspicionWeightManager(["a", "b"], remove_after=5)
        d = m.update(["a", "b"], {"a"}, {"a"}, immediate_revocation_tags={"a"})
        self.assertEqual(m.evidence_counts["a"], 1)
        self.assertEqual(d.trace_requested_tags, {"a"})
        self.assertEqual(d.weights["a"], 0.)
        self.assertNotIn("a", m.revoked)
        m.confirm_revocation("a")
        self.assertIn("a", m.revoked)

    def test_recovery_requires_trusted_evidence_and_is_bounded(self):
        m = SuspicionWeightManager(["a", "b"], recovery_factor=1.25)
        m.update(["a", "b"], {"a"}, {"a"})
        m.update(["a", "b"], set(), set())
        self.assertEqual(m.weights["a"], .1)
        self.assertEqual(m.evidence_counts["a"], .5)
        m.update(["a", "b"], set(), set(), recovery_tags={"a"})
        self.assertAlmostEqual(m.weights["a"], .28)
        self.assertEqual(m.evidence_counts["a"], .25)
        for _ in range(30):
            m.update(["a", "b"], set(), set(), recovery_tags={"a"})
        self.assertGreater(m.weights["a"], .999)
        self.assertLessEqual(m.weights["a"], 1.)
        self.assertEqual(m.evidence_counts["a"], .25 * 0.5 ** 30)

    def test_nonconsecutive_suspicion_preserves_fractional_decay_before_ctol(self):
        m = SuspicionWeightManager(["a", "b"], remove_after=3)
        m.update(["a", "b"], {"a"}, {"a"})
        m.update(["a", "b"], set(), set(), recovery_tags={"a"})
        self.assertEqual(m.evidence_counts["a"], .5)
        d = m.update(["a", "b"], {"a"}, {"a"})
        self.assertEqual(m.evidence_counts["a"], 1.5)
        self.assertFalse(d.trace_requested_tags)
        m.update(["a", "b"], set(), set(), recovery_tags={"a"})
        self.assertEqual(m.evidence_counts["a"], .75)
        d = m.update(["a", "b"], {"a"}, {"a"})
        self.assertEqual(m.evidence_counts["a"], 1.75)
        self.assertFalse(d.trace_requested_tags)
        d = m.update(["a", "b"], {"a"}, {"a"})
        self.assertEqual(m.evidence_counts["a"], 2.75)
        self.assertFalse(d.trace_requested_tags)
        d = m.update(["a", "b"], {"a"}, {"a"})
        self.assertEqual(m.evidence_counts["a"], 3.)
        self.assertEqual(d.trace_requested_tags, {"a"})
        self.assertEqual(d.weights["a"], 0.)

    def test_missing_or_uncalibrated_observation_does_not_decay(self):
        m = SuspicionWeightManager(["a", "b"], remove_after=3)
        m.update(["a", "b"], {"a"}, {"a"})
        m.update(["b"], set(), set())
        self.assertEqual(m.evidence_counts["a"], 1.)
        m.update(["a", "b"], {"a"}, set())
        self.assertEqual(m.evidence_counts["a"], 1.)

    def test_normal_count_decays_even_when_shock_freezes_history_and_recovery(self):
        m = SuspicionWeightManager(["a", "b", "c"], remove_after=3)
        m.update(["a", "b", "c"], {"a"}, {"a"})
        d = m.update(["a", "b", "c"], {"b", "c"}, {"b", "c"}, recovery_tags={"a"})
        self.assertTrue(d.history_frozen)
        self.assertEqual(m.weights["a"], .1)
        self.assertEqual(m.evidence_counts["a"], .5)

    def test_shock_freezes_history_but_does_not_defer_either_revocation_path(self):
        tags = list(map(str, range(100)))
        bad = set(tags[:80])
        for severe in (False, True):
            with self.subTest(severe=severe):
                m = SuspicionWeightManager(tags, remove_after=5 if severe else 1)
                d = m.update(tags, bad, bad, immediate_revocation_tags=bad if severe else set())
                self.assertTrue(d.history_frozen)
                self.assertEqual(d.trace_requested_tags, bad)
                self.assertTrue(all(d.weights[tag] == 0 for tag in bad))
                self.assertTrue(all(d.weights[tag] == 1 for tag in tags[80:]))

    def test_last_two_identities_are_not_exempt_from_ctol(self):
        m = SuspicionWeightManager(["a", "b"], remove_after=1)
        d = m.update(["a", "b"], {"a", "b"}, {"a", "b"})
        self.assertEqual(d.trace_requested_tags, {"a", "b"})
        for tag in d.trace_requested_tags:
            m.confirm_revocation(tag)
        self.assertEqual(m.revoked, {"a", "b"})

    def test_pending_revocation_cannot_recover_or_be_requested_twice(self):
        m = SuspicionWeightManager(["a", "b"], remove_after=3)
        m.update(["a", "b"], {"a"}, {"a"}, immediate_revocation_tags={"a"})
        d = m.update(["a", "b"], set(), set(), recovery_tags={"a"})
        self.assertEqual(d.weights["a"], 0.)
        self.assertEqual(m.weights["a"], .1)
        self.assertEqual(m.evidence_counts["a"], 1)
        d = m.update(["a", "b"], {"a"}, {"a"}, immediate_revocation_tags={"a"})
        self.assertFalse(d.trace_requested_tags)
        self.assertEqual(m.evidence_counts["a"], 1)

    def test_no_clean_reference_rejection_does_not_claim_measured_suspicion(self):
        m = SuspicionWeightManager(["late", "b"])
        d = m.update(["late", "b"], {"late"}, set())
        self.assertEqual(m.evidence_counts["late"], 0)
        self.assertFalse(d.trace_requested_tags)

    def test_survivors_cannot_absorb_unlimited_mass(self):
        tags = list(map(str, range(100)))
        normal = DetectionResult(True, "normal")
        rejected = DetectionResult(False, "strong", True)
        decisions = {t: normal if int(t) < 27 else rejected for t in tags}
        coeffs = bounded_aggregation_coefficients(
            tags, dict.fromkeys(tags, .01), dict.fromkeys(tags, 1.), decisions, 2.)
        self.assertAlmostEqual(sum(coeffs.values()), .54)
        self.assertAlmostEqual(sum(coeffs[t] for t in tags[:7]), .14)
        self.assertTrue(all(c <= .02 for c in coeffs.values()))

    def test_all_suspicious_does_not_renormalize_penalties_away(self):
        tags = ["a", "b"]
        coeffs = bounded_aggregation_coefficients(tags, dict.fromkeys(tags, .5),
                    dict.fromkeys(tags, .1),
                    dict.fromkeys(tags, DetectionResult(True, "mild", True)), 2.)
        self.assertEqual(sum(coeffs.values()), 0.)

    def test_clean_sample_weighted_fedavg_and_zero_fallback(self):
        tags = ["a", "b"]
        nominal = {"a": .8, "b": .2}
        normal = dict.fromkeys(tags, DetectionResult(True, "normal"))
        self.assertEqual(bounded_aggregation_coefficients(tags, nominal,
                          dict.fromkeys(tags, 1), normal, 2), nominal)
        rejected = dict.fromkeys(tags, DetectionResult(False, "strong"))
        self.assertEqual(sum(bounded_aggregation_coefficients(
            tags, nominal, dict.fromkeys(tags, 1), rejected, 2).values()), 0)

    def test_clip_factor_is_not_renormalized(self):
        d = {"a": DetectionResult(True, "normal", clip_factor=.25)}
        self.assertEqual(bounded_aggregation_coefficients(["a"], {"a": 1},
                         {"a": 1}, d, 2)["a"], .25)
