import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sm9rrsfl.fair_tuning import _training_health_reasons
from sm9rrsfl.fl import ExperimentConfig, RoundRecord
from sm9rrsfl.ours_policy import OursParameters
from sm9rrsfl.svd_detector import LongitudinalSVDDetector


class QuarantineDetectorTest(unittest.TestCase):
    def detector(self):
        rng = np.random.default_rng(56)
        x = rng.normal(size=1030)
        detector = LongitudinalSVDDetector(expected_update_size=len(x),
            matrix_offset=0, matrix_shape=(100, 10))
        for r in range(1, 8):
            detector.evaluate('a', x + rng.normal(scale=.05, size=len(x)), round_id=r)
            detector.commit('a', admit_history=True)
        return detector, x

    def test_training_points_fit_the_same_online_group_metric(self):
        detector, _ = self.detector()
        state = detector._states['a']
        for point in state.history:
            self.assertLessEqual(detector._scores(state.anchor, point).max(), 1. + 1e-12)

    def test_gradual_benign_drift_updates_live_history_without_moving_anchor(self):
        detector, x = self.detector()
        old = copy.deepcopy(detector._states['a'].anchor)
        direction = np.random.default_rng(57).normal(size=len(x))
        admissions = 0
        for r in range(8, 58):
            d = detector.evaluate('a', (x + .002 * (r - 7) * direction) * .99 ** r, round_id=r)
            self.assertFalse(d.would_flag)
            admissions += detector.commit('a', admit_history=True)
        state = detector._states['a']
        self.assertGreater(admissions, 30)
        np.testing.assert_array_equal(old.centers, state.anchor.centers)
        np.testing.assert_array_equal(old.scale, state.normal.scale)
        self.assertFalse(np.array_equal(old.centers, state.normal.centers))

    def test_amplitude_spike_is_not_diluted_and_convergence_is_allowed(self):
        detector, x = self.detector()
        d = detector.evaluate('a', x * 100, round_id=8)
        self.assertTrue(d.immediate_revocation)
        self.assertFalse(d.accepted)
        self.assertGreater(d.norm_score, detector.policy.detector_reject_threshold)
        detector.commit('a', admit_history=False)
        detector, x = self.detector()
        d = detector.evaluate('a', x * .01, round_id=8)
        self.assertFalse(d.would_flag)
        self.assertEqual(d.norm_score, 0.)

    def test_recovery_does_not_require_history_admission(self):
        detector, x = self.detector()
        for r in [8, 9]:
            with patch.object(detector, '_scores', return_value=np.array([2., 0., 0., 0.])):
                d = detector.evaluate('a', x, round_id=r)
            self.assertFalse(d.history_eligible)
            self.assertEqual(d.recovery_eligible, r == 9)
            detector.commit('a', admit_history=True)

    def test_benign_norm_rebound_after_convergence_is_not_an_attack(self):
        detector, x = self.detector()
        for r in range(8, 80):
            d = detector.evaluate('a', x * .1, round_id=r)
            self.assertFalse(d.would_flag)
            detector.commit('a', admit_history=True)
        rebound = detector.evaluate('a', x, round_id=80)
        self.assertFalse(rebound.would_flag)
        self.assertLess(rebound.norm_score, .1)

    def test_subthreshold_normal_envelope_does_not_accumulate_alarm(self):
        detector, x = self.detector()
        for r in range(8, 108):
            with patch.object(detector, '_scores', return_value=np.array([2., 0., 0., 0.])):
                d = detector.evaluate('a', x, round_id=r)
            self.assertFalse(d.would_flag)
            detector.commit('a', admit_history=True)


class TrainingHealthTest(unittest.TestCase):
    def result(self, method='sm9rrs', ratio=.8, false=0, loss=0., rounds=5):
        config = ExperimentConfig(method=method, num_clients=10, malicious_ratio=ratio)
        records = [RoundRecord(method, ratio, r, .8, .2, 1, 9, 0, 0, false, '',
                               honest_weight_loss=loss) for r in range(1, rounds + 1)]
        return SimpleNamespace(config=config, records=records,
                               malicious_clients=tuple(range(round(ratio * 10))))

    def test_all_honest_revoked_on_last_round_is_a_failure(self):
        result = self.result()
        result.records[-1] = replace(result.records[-1], false_positive_revocations=2)
        self.assertIn('all_honest_revoked', _training_health_reasons(result))

    def test_clean_revocations_and_sustained_starvation_are_failures(self):
        self.assertIn('clean_false_revocation_rate', _training_health_reasons(self.result(ratio=0, false=2)))
        self.assertIn('honest_weight_starvation', _training_health_reasons(self.result(loss=.995)))
        self.assertNotIn('honest_weight_starvation', _training_health_reasons(self.result(loss=.995, rounds=4)))

    def test_krum_per_client_deficits_are_not_total_weight_starvation(self):
        self.assertEqual(_training_health_reasons(self.result(method='krum', loss=.995)), ())
