"""Scientific boundary and checkpoint tests for the explicit history ablation."""
import copy
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

import cifar_cnn_history_runtime as runtime
from sm9rrsfl.svd_detector import LongitudinalSVDDetector


class HistoryRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.original_commit = LongitudinalSVDDetector.commit
        self.update = np.linspace(-.02, .03, 120)

    def detector(self):
        return LongitudinalSVDDetector(window_size=20, expected_update_size=120)

    def step(self, detector, round_id, *, admit=True):
        decision = detector.evaluate("opaque-tag", self.update,
                                     round_id=round_id, learning_rate=1.)
        return decision, detector.commit("opaque-tag", admit_history=admit)

    def warm(self):
        detector = self.detector()
        for r in range(1, 25):
            self.step(detector, r)
        return detector

    def stable_reference(self, state):
        return pickle.dumps((state.history, state.normal, state.anchor, state.norm_limit))

    def test_original_is_exact_noop(self):
        a, b = self.detector(), self.detector()
        for r in range(1, 31):
            expected = self.step(a, r)
            with runtime.history_runtime("original", freeze_start_round=None):
                self.assertIs(LongitudinalSVDDetector.commit, self.original_commit)
                self.assertEqual(self.step(b, r), expected)
            self.assertEqual(pickle.dumps(a), pickle.dumps(b))

    def test_same_prefix_then_only_admission_changes_at_boundary(self):
        a, b = self.detector(), self.detector()
        for r in range(1, 25):
            expected = self.step(a, r)
            with runtime.history_runtime(runtime.FROZEN_VARIANT, freeze_start_round=25):
                self.assertEqual(self.step(b, r), expected)
            self.assertEqual(pickle.dumps(a), pickle.dumps(b))
        before = self.stable_reference(b._states["opaque-tag"])
        decision_a, admitted_a = self.step(a, 25)
        with runtime.history_runtime(runtime.FROZEN_VARIANT, freeze_start_round=25):
            decision_b, admitted_b = self.step(b, 25)
        self.assertEqual(decision_a, decision_b)
        self.assertTrue(decision_b.accepted)
        self.assertTrue(decision_b.history_eligible)
        self.assertTrue(admitted_a)
        self.assertFalse(admitted_b)
        state = b._states["opaque-tag"]
        self.assertEqual(before, self.stable_reference(state))
        self.assertEqual(state.last_round, 25)
        self.assertIsNone(state.pending)
        self.assertEqual(state.clean_streak, a._states["opaque-tag"].clean_streak)
        self.assertEqual(state.recovery_streak, a._states["opaque-tag"].recovery_streak)
        self.assertEqual(state.drift, a._states["opaque-tag"].drift)

    def test_frozen_reference_remains_unchanged_for_all_later_observations(self):
        detector = self.warm()
        before = self.stable_reference(detector._states["opaque-tag"])
        with runtime.history_runtime(runtime.FROZEN_VARIANT, freeze_start_round=25):
            for r in range(25, 151):
                _, admitted = self.step(detector, r)
                self.assertFalse(admitted)
                self.assertEqual(before, self.stable_reference(detector._states["opaque-tag"]))
        self.assertEqual(detector._states["opaque-tag"].last_round, 150)

    def test_safe_changing_input_updates_original_reference_but_not_frozen(self):
        original = self.warm()
        frozen = copy.deepcopy(original)
        before = self.stable_reference(frozen._states["opaque-tag"])
        old_normal = pickle.dumps(original._states["opaque-tag"].normal)
        self.update *= 1.01
        decision, admitted = self.step(original, 25)
        with runtime.history_runtime(runtime.FROZEN_VARIANT, freeze_start_round=25):
            frozen_decision, frozen_admitted = self.step(frozen, 25)
        self.assertEqual(decision, frozen_decision)
        self.assertTrue(admitted)
        self.assertFalse(frozen_admitted)
        self.assertNotEqual(old_normal, pickle.dumps(original._states["opaque-tag"].normal))
        self.assertNotEqual(before, self.stable_reference(original._states["opaque-tag"]))
        self.assertEqual(before, self.stable_reference(frozen._states["opaque-tag"]))

    def test_original_aggregation_guard_still_resets_clean_streak(self):
        detector = self.warm()
        with runtime.history_runtime(runtime.FROZEN_VARIANT, freeze_start_round=25):
            self.step(detector, 25, admit=False)
        self.assertEqual(detector._states["opaque-tag"].clean_streak, 0)
        self.assertGreater(detector._states["opaque-tag"].recovery_streak, 0)

    def test_rejection_drift_and_recovery_decisions_are_preserved(self):
        a = self.warm()
        b = copy.deepcopy(a)
        self.update *= 1000
        expected = self.step(a, 25, admit=False)
        with runtime.history_runtime(runtime.FROZEN_VARIANT, freeze_start_round=25):
            actual = self.step(b, 25, admit=False)
        self.assertFalse(actual[0].accepted)
        self.assertEqual(expected, actual)
        self.assertEqual(pickle.dumps(a), pickle.dumps(b))

    def test_checkpoint_uses_original_class_and_resume_reapplies_freeze(self):
        uninterrupted = self.warm()
        with runtime.history_runtime(runtime.FROZEN_VARIANT, freeze_start_round=25):
            for r in range(25, 41):
                self.step(uninterrupted, r)
            checkpoint = pickle.dumps(uninterrupted)
            for r in range(41, 51):
                self.step(uninterrupted, r)
        with runtime.history_runtime(runtime.FROZEN_VARIANT, freeze_start_round=25):
            resumed = pickle.loads(checkpoint)
            self.assertIs(type(resumed), LongitudinalSVDDetector)
            for r in range(41, 51):
                self.assertFalse(self.step(resumed, r)[1])
        self.assertEqual(pickle.dumps(resumed), pickle.dumps(uninterrupted))

    def test_pre_freeze_checkpoint_resumes_across_boundary(self):
        detector = self.warm()
        checkpoint = pickle.dumps(detector)
        with runtime.history_runtime(runtime.FROZEN_VARIANT, freeze_start_round=25):
            resumed = pickle.loads(checkpoint)
            self.assertFalse(self.step(resumed, 25)[1])
        self.assertEqual(resumed._states["opaque-tag"].last_round, 25)

    def test_actual_checkpoint_io_preserves_frozen_detector_and_variant_identity(self):
        from sm9rrsfl import experiments
        from sm9rrsfl.fl import ExperimentConfig
        config = ExperimentConfig(method="sm9rrs", detector_window=20, attack_start_round=25)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "round.pickle"
            for last_round in (24, 25):
                detector = self.warm()
                with runtime.history_runtime(runtime.FROZEN_VARIANT, freeze_start_round=25):
                    if last_round == 25:
                        self.step(detector, 25)
                    experiments._write_round_checkpoint(path, config, "H1-fingerprint",
                        {"detector": detector, "round": last_round},
                        runtime_seconds=1.25, peak_memory_mb=2.5)
                # H0 and H1 config intentionally match, but run identities do not.
                self.assertIsNone(experiments._load_round_checkpoint(
                    path, config, "H0-fingerprint")[0])
                with runtime.history_runtime(runtime.FROZEN_VARIANT, freeze_start_round=25):
                    loaded, seconds, memory = experiments._load_round_checkpoint(
                        path, config, "H1-fingerprint")
                    self.assertEqual((seconds, memory), (1.25, 2.5))
                    self.assertEqual(loaded["round"], last_round)
                    self.assertFalse(self.step(loaded["detector"], last_round + 1)[1])
                    self.assertFalse(self.step(detector, last_round + 1)[1])
                    self.assertEqual(pickle.dumps(loaded["detector"]), pickle.dumps(detector))

    def test_context_restores_after_exception_and_rejects_nested_context(self):
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with runtime.history_runtime(runtime.FROZEN_VARIANT, freeze_start_round=25):
                with self.assertRaisesRegex(RuntimeError, "nested"):
                    with runtime.history_runtime("original", freeze_start_round=None):
                        pass
                raise RuntimeError("sentinel")
        self.assertIs(LongitudinalSVDDetector.commit, self.original_commit)
        with runtime.history_runtime("original", freeze_start_round=None):
            pass

    def test_invalid_contract_and_warmup_freeze_are_rejected(self):
        for variant, start in [("unknown", 25), ("original", 25),
                               (runtime.FROZEN_VARIANT, None),
                               (runtime.FROZEN_VARIANT, True),
                               (runtime.FROZEN_VARIANT, 0),
                               (runtime.FROZEN_VARIANT, 25.0)]:
            with self.assertRaises(ValueError):
                with runtime.history_runtime(variant, freeze_start_round=start):
                    pass
        with runtime.history_runtime(runtime.FROZEN_VARIANT, freeze_start_round=20):
            with self.assertRaisesRegex(ValueError, "warmup"):
                self.step(self.detector(), 1)
        self.assertIs(LongitudinalSVDDetector.commit, self.original_commit)

    def test_pending_contract_and_forget_remain_original(self):
        detector = self.warm()
        with runtime.history_runtime(runtime.FROZEN_VARIANT, freeze_start_round=25):
            with self.assertRaisesRegex(RuntimeError, "pending"):
                detector.commit("opaque-tag", admit_history=True)
            detector.evaluate("opaque-tag", self.update, round_id=25)
            with self.assertRaisesRegex(RuntimeError, "previous decision"):
                detector.evaluate("opaque-tag", self.update, round_id=26)
            detector.forget("opaque-tag")
        self.assertNotIn("opaque-tag", detector._states)


if __name__ == "__main__":
    unittest.main()
