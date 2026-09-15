import pickle
import unittest
from dataclasses import replace
from unittest import mock

import sm9rrsfl
import numpy as np

import cifar_ours_development_policy as development
from sm9rrsfl import fl
from sm9rrsfl.datasets import make_synthetic_mnist_like
from sm9rrsfl.ours_policy import OursParameters
from sm9rrsfl.svd_detector import LongitudinalSVDDetector
from sm9rrsfl.weighting import SuspicionWeightManager, bounded_aggregation_coefficients


POLICY = replace(OursParameters(), detector_distance_threshold=1.75,
                 detector_reject_threshold=6.0, detector_drift_allowance=1.25,
                 detector_history_threshold=1.0, detector_reference_budget=3.5)


class DevelopmentPolicyTest(unittest.TestCase):
    def setUp(self):
        self.vector = np.random.default_rng(31).normal(size=1030)

    def pair(self, tags=("a",)):
        detector = fl.LongitudinalSVDDetector(
            window_size=3, expected_update_size=1030, policy=POLICY,
            matrix_offset=0, matrix_shape=(100, 10))
        manager = fl.SuspicionWeightManager(tags, penalty_factor=.5)
        for rd in range(1, 4):
            for tag in tags:
                decision = detector.evaluate(tag, self.vector, round_id=rd)
                self.assertEqual(decision.reason, "clean_warmup")
                detector.commit(tag, admit_history=True)
        return detector, manager

    def decision(self, detector, score, tag="a", rd=4):
        with mock.patch.object(detector, "_scores", return_value=np.array([score, 0., 0., 0.])):
            return detector.evaluate(tag, self.vector, round_id=rd)

    def test_weak_update_has_zero_mass_without_permanent_penalty_or_history(self):
        with development.weak_quarantine_policy():
            detector, manager = self.pair()
            manager.weights["a"] = .4
            manager.evidence_counts["a"] = 2.
            old_history = pickle.dumps(detector._states["a"].history)
            decision = self.decision(detector, 1.5)
            self.assertEqual(decision.reason, "development_weak_quarantine")
            self.assertFalse(decision.accepted)
            self.assertTrue(decision.would_flag)
            self.assertFalse(decision.count_increment)
            self.assertFalse(decision.recovery_eligible)
            result = manager.update(["a"], {"a"}, set())
            self.assertEqual(result.suspicious_tags, {"a"})
            self.assertEqual(result.reliability_after_update["a"], .4)
            self.assertEqual(manager.evidence_counts["a"], 1.)
            self.assertFalse(result.trace_requested_tags)
            self.assertFalse(result.history_frozen)
            coefficients = bounded_aggregation_coefficients(
                ["a"], {"a": 1.}, result.weights, {"a": decision}, 2.)
            self.assertEqual(coefficients["a"], 0.)
            self.assertFalse(detector.commit("a", admit_history=True))
            self.assertEqual(old_history, pickle.dumps(detector._states["a"].history))
            self.assertFalse(detector.development_policy.weak_tags)

    def test_threshold_edges_preserve_strong_and_severe_evidence(self):
        for score, expected, counted, severe in (
                (1.25, "normal", False, False),
                (1.250001, "development_weak_quarantine", False, False),
                (1.75, "development_weak_quarantine", False, False),
                (1.750001, "suspicious", True, False),
                (6., "suspicious", True, False),
                (6.000001, "strong_novelty", True, True)):
            with self.subTest(score=score), development.weak_quarantine_policy():
                detector, manager = self.pair()
                decision = self.decision(detector, score)
                self.assertEqual(decision.reason, expected)
                self.assertEqual(decision.count_increment, counted)
                self.assertEqual(decision.immediate_revocation, severe)
                suspicious = {"a"} if decision.would_flag else set()
                result = manager.update(["a"], suspicious, {"a"} if counted else set(),
                                        immediate_revocation_tags={"a"} if severe else set())
                self.assertEqual(manager.weights["a"], .5 if counted else 1.)
                self.assertEqual(result.trace_requested_tags, {"a"} if severe else set())

    def test_drift_alarm_is_not_downgraded_to_weak(self):
        with development.weak_quarantine_policy():
            detector, manager = self.pair()
            detector._states["a"].drift = 20.
            decision = self.decision(detector, 1.5)
            self.assertEqual(decision.reason, "suspicious")
            self.assertTrue(decision.count_increment)
            self.assertFalse(decision.immediate_revocation)
            manager.evidence_counts["a"] = 2.
            result = manager.update(["a"], {"a"}, {"a"})
            self.assertEqual(result.trace_requested_tags, {"a"})

    def test_uncalibrated_rejection_keeps_original_penalty_and_frozen_count(self):
        with development.weak_quarantine_policy():
            detector, manager = self.pair()
            decision = detector.evaluate("late", self.vector, round_id=4)
            self.assertEqual(decision.reason, "insufficient_clean_history")
            manager.evidence_counts["late"] = 1.
            result = manager.update(["late"], {"late"}, set())
            self.assertEqual(result.weights["late"], .5)
            self.assertEqual(manager.evidence_counts["late"], 1.)
            self.assertTrue(result.history_frozen)

    def test_weak_majority_does_not_change_original_global_shock_rule(self):
        with development.weak_quarantine_policy():
            detector, manager = self.pair(("a", "b", "c"))
            self.decision(detector, 1.5, "a")
            self.decision(detector, 1.5, "b")
            self.decision(detector, .5, "c")
            manager.weights["c"] = .5
            result = manager.update(["a", "b", "c"], {"a", "b"}, set(), recovery_tags={"c"})
            self.assertFalse(result.history_frozen)
            self.assertEqual(manager.previous_suspicious, set())
            self.assertAlmostEqual(result.weights["c"], .6)

    def test_pickle_retains_shared_policy_after_fresh_context(self):
        with development.weak_quarantine_policy():
            detector, manager = self.pair()
            snapshot = pickle.dumps({"detector": detector, "weight_manager": manager})
        with development.weak_quarantine_policy():
            restored = pickle.loads(snapshot)
            detector, manager = restored["detector"], restored["weight_manager"]
            self.assertIs(detector.development_policy, manager.development_policy)
            self.decision(detector, 1.5)
            result = manager.update(["a"], {"a"}, set())
            self.assertEqual(result.weights["a"], 1.)
            self.assertFalse(result.trace_requested_tags)

    def test_context_restores_original_constructors_after_failure(self):
        original_krum = fl.krum
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with development.weak_quarantine_policy():
                self.assertIs(fl.krum, original_krum)
                raise RuntimeError("intentional")
        self.assertIs(fl.LongitudinalSVDDetector, LongitudinalSVDDetector)
        self.assertIs(fl.SuspicionWeightManager, SuspicionWeightManager)
        with self.assertRaisesRegex(RuntimeError, "require"):
            development.WeakQuarantineWeightManager()

    def test_production_round_checkpoint_resumes_variant_exactly(self):
        dataset = make_synthetic_mnist_like(train_samples=40, test_samples=20, seed=31)
        config = fl.ExperimentConfig(
            method="sm9rrs", num_clients=4, rounds=5, early_stop=False,
            malicious_ratio=0., crypto_mode="simulated", seed=31,
            detector_window=3, detector_distance_threshold=1.75,
            detector_reject_threshold=6., detector_drift_allowance=1.25,
            detector_history_threshold=1., detector_reference_budget=3.5)
        saved = {}

        def stop(state):
            if state["completed_round"] == 4:
                saved["snapshot"] = pickle.dumps(state)
                raise InterruptedError("test checkpoint boundary")

        with mock.patch.object(LongitudinalSVDDetector, "_scores", return_value=np.array([1.5, 0., 0., 0.])):
            with development.weak_quarantine_policy():
                full = fl.run_experiment(dataset, config)
            with development.weak_quarantine_policy():
                with self.assertRaises(InterruptedError):
                    fl.run_experiment(dataset, config, checkpoint_callback=stop)
            with development.weak_quarantine_policy():
                resumed = fl.run_experiment(dataset, config, resume_state=pickle.loads(saved["snapshot"]))
        self.assertEqual(full.records, resumed.records)
        self.assertEqual(full.records[-1].accepted_updates, 0)
        weak = [d for d in resumed.diagnostics if d.round >= 4]
        self.assertEqual(len(weak), 8)
        self.assertTrue(all(d.decision_reason == "development_weak_quarantine" for d in weak))
        self.assertTrue(all(d.aggregation_weight == 0. and d.weight_after_penalty_recovery == 1. for d in weak))
        self.assertTrue(all(not d.trace_requested and not d.history_admitted for d in weak))


if __name__ == "__main__":
    unittest.main()
