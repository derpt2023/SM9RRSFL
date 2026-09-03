import pickle
import unittest
from unittest import mock
from dataclasses import replace
import numpy as np

from sm9rrsfl.ours_policy import OursParameters, bounded_candidates
from sm9rrsfl.svd_detector import LongitudinalSVDDetector, _fit_normal


class NormalStateDetectorTest(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(11).normal(size=1030)
        self.detector = LongitudinalSVDDetector(
            window_size=7, expected_update_size=1030, matrix_offset=0, matrix_shape=(100, 10))

    def warmup(self, tag="a"):
        for r in range(1, 8):
            d = self.detector.evaluate(tag, self.x, round_id=r)
            self.assertFalse(d.would_flag)
            self.assertTrue(self.detector.commit(tag, admit_history=True))

    def test_identical_clean_updates_do_not_force_two_clusters(self):
        self.warmup()
        for r in range(8, 15):
            d = self.detector.evaluate("a", self.x, round_id=r)
            self.assertTrue(d.accepted)
            self.assertFalse(d.would_flag)
            self.assertEqual(d.normal_cluster_count, 1)
            self.detector.commit("a", admit_history=True)

    def test_sign_flip_is_visible_despite_identical_svd(self):
        self.warmup()
        d = self.detector.evaluate("a", -self.x, round_id=8)
        self.assertFalse(d.accepted)
        self.assertTrue(d.count_increment)
        self.assertTrue(d.immediate_revocation)
        self.assertGreater(d.signed_score, 4)
        self.assertGreater(d.class_score, 4)
        self.detector.commit("a", admit_history=False)

    def test_mild_deviation_counts_without_immediate_revocation(self):
        self.warmup()
        with mock.patch.object(self.detector, "_scores", return_value=np.array([2.5, 0., 0.])):
            d = self.detector.evaluate("a", self.x, round_id=8)
        self.assertTrue(d.accepted)
        self.assertTrue(d.would_flag)
        self.assertTrue(d.count_increment)
        self.assertFalse(d.immediate_revocation)
        self.assertFalse(d.history_eligible)

    def test_severe_spectral_only_deviation_no_longer_requires_corroboration(self):
        self.warmup()
        with mock.patch.object(self.detector, "_scores", return_value=np.array([5., 0., 0.])):
            d = self.detector.evaluate("a", self.x, round_id=8)
        self.assertFalse(d.accepted)
        self.assertTrue(d.immediate_revocation)
        self.assertTrue(d.count_increment)
        self.assertEqual(d.signed_score, 0.)
        self.assertEqual(d.class_score, 0.)

    def test_accumulated_drift_counts_as_suspicious_not_immediate(self):
        self.warmup()
        self.detector._states["a"].drift = 10.
        d = self.detector.evaluate("a", self.x, round_id=8)
        self.assertTrue(d.would_flag)
        self.assertTrue(d.count_increment)
        self.assertTrue(d.accepted)
        self.assertFalse(d.immediate_revocation)

    def test_warning_and_reject_boundaries_use_strict_exceedance(self):
        for score, flagged in ((2., False), (4., True)):
            with self.subTest(score=score):
                self.setUp()
                self.warmup()
                with mock.patch.object(self.detector, "_scores", return_value=np.array([score, 0., 0.])):
                    d = self.detector.evaluate("a", self.x, round_id=8)
                self.assertEqual(d.count_increment, flagged)
                self.assertFalse(d.immediate_revocation)
                self.assertTrue(d.accepted)

    def test_all_kmeans_clusters_are_normal_states(self):
        model = _fit_normal([[-2., 0.]] * 4 + [[2., 0.]] * 4, 2)
        self.assertEqual(len(model.centers), 2)
        self.assertTrue(np.isfinite(model.radii).all())

    def test_singleton_mode_falls_back_to_one(self):
        model = _fit_normal([[0., 0.]] * 6 + [[100., 0.]], 2)
        self.assertEqual(len(model.centers), 1)

    def test_evaluate_cannot_admit_history_and_commit_is_required(self):
        self.warmup()
        before = pickle.dumps(self.detector._states["a"].history)
        d = self.detector.evaluate("a", self.x, round_id=8)
        self.assertFalse(d.history_eligible)  # first of three confirmations
        self.assertEqual(before, pickle.dumps(self.detector._states["a"].history))
        with self.assertRaisesRegex(RuntimeError, "commit"):
            self.detector.evaluate("a", self.x, round_id=9)
        self.assertFalse(self.detector.commit("a", admit_history=False))

    def test_repeated_attacks_never_become_normal(self):
        self.warmup()
        anchor = pickle.dumps(self.detector._states["a"].anchor)
        history = pickle.dumps(self.detector._states["a"].history)
        for r in range(8, 25):
            d = self.detector.evaluate("a", -self.x, round_id=r)
            self.assertTrue(d.would_flag)
            self.assertFalse(self.detector.commit("a", admit_history=True))
        self.assertEqual(anchor, pickle.dumps(self.detector._states["a"].anchor))
        self.assertEqual(history, pickle.dumps(self.detector._states["a"].history))

    def test_three_normal_confirmations_eventually_admit_history(self):
        self.warmup()
        admitted = []
        for r in range(8, 11):
            self.detector.evaluate("a", self.x, round_id=r)
            admitted.append(self.detector.commit("a", admit_history=True))
        self.assertEqual(admitted, [False, False, True])

    def test_learning_rate_decay_is_not_itself_novelty(self):
        self.warmup()
        d = self.detector.evaluate("a", self.x * .01, round_id=8, learning_rate=.01)
        self.assertFalse(d.would_flag)

    def test_norm_clip_uses_only_clean_prefix(self):
        self.warmup()
        d = self.detector.evaluate("a", self.x * 100, round_id=8)
        self.assertAlmostEqual(d.clip_factor, .02)
        self.detector.commit("a", admit_history=True)
        self.assertAlmostEqual(self.detector._states["a"].norm_limit, np.linalg.norm(self.x))

    def test_clipped_but_accepted_updates_cannot_enter_trusted_history(self):
        self.warmup()
        before = pickle.dumps(self.detector._states["a"].history)
        for r in range(8, 13):
            d = self.detector.evaluate("a", self.x * 2.1, round_id=r)
            self.assertTrue(d.accepted)
            self.assertLess(d.clip_factor, 1.)
            self.assertFalse(d.history_eligible)
            self.assertFalse(self.detector.commit("a", admit_history=True))
        self.assertEqual(before, pickle.dumps(self.detector._states["a"].history))

    def test_late_tag_cannot_bootstrap_on_attacks(self):
        d = self.detector.evaluate("late", self.x, round_id=8)
        self.assertFalse(d.accepted)
        self.assertFalse(d.count_increment)
        self.assertFalse(d.immediate_revocation)
        self.assertFalse(self.detector.commit("late", admit_history=True))

    def test_round_order_shapes_and_nonfinite_rejected(self):
        self.warmup()
        for vector in (np.ones(12), np.full(1030, np.nan), np.ones((103, 10))):
            with self.assertRaises(ValueError):
                self.detector.evaluate("a", vector, round_id=8)
        with self.assertRaises(ValueError):
            self.detector.evaluate("a", self.x, round_id=7)

    def test_checkpoint_restores_exact_detection_state(self):
        self.warmup()
        restored = pickle.loads(pickle.dumps(self.detector))
        self.assertEqual(self.detector.evaluate("a", -self.x, round_id=8),
                         restored.evaluate("a", -self.x, round_id=8))

    def test_compact_state_and_forget(self):
        self.warmup()
        before = self.detector.memory_bytes()
        self.assertLess(before, 100000)
        self.detector.forget("a")
        self.assertLess(self.detector.memory_bytes(), before)

    def test_candidates_are_bounded_unique_and_cover_important_axes(self):
        candidates = bounded_candidates(12)
        self.assertEqual(len({tuple(c.items()) for c in candidates}), 12)
        for name in candidates[0]:
            self.assertGreater(len({c[name] for c in candidates}), 1, name)
        for p in candidates:
            OursParameters(**p).validate()
        self.assertEqual(len({tuple(c.items()) for c in bounded_candidates(36)}), 36)
        with self.assertRaises(ValueError):
            bounded_candidates(37)
        with self.assertRaises(ValueError):
            replace(OursParameters(), detector_reject_threshold=1.).validate()
