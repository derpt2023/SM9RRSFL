import json
import tempfile
import unittest
from dataclasses import replace, asdict
from pathlib import Path
from unittest import mock

from sm9rrsfl.datasets import make_synthetic_mnist_like, stratified_training_three_way_split
from sm9rrsfl.experiments import parse_args
from sm9rrsfl.fl import ExperimentResult, RoundRecord
from sm9rrsfl.ours_calibration import (
    OursCalibrationArtifact, OursCalibrationError, apply_ours_parameters,
    resolve_or_run_ours_calibration,
)


def args():
    return parse_args(["--dataset", "synthetic", "--methods", "sm9rrs", "--ratio-range", "0", ".8", "3",
                      "--num-clients", "10", "--rounds", "6", "--K", "3", "--attack", "alternating_minimization",
                      "--ours-parameter-mode", "auto", "--calibration-candidate-budget", "3",
                      "--no-early-stop", "--crypto-mode", "simulated", "--no-progress",
                      "--compute-backend", "numpy", "--jobs", "1"])


def fake_run(dataset, config):
    bad = config.malicious_ratio > 0
    records = [RoundRecord(config.method, config.malicious_ratio, r, .9, .1, 10, 0, 0, 0, 0, "",
                           attack_target_success_rate=.05 if bad else None,
                           attack_active=bad and r >= 5, honest_weight_loss=.0,
                           malicious_weight_mass=.01 if bad and r >= 5 else .0)
               for r in range(1, config.rounds + 1)]
    return ExperimentResult(config, records, .9, .1, config.rounds, (), ())


class OursCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.dataset = make_synthetic_mnist_like(train_samples=400, test_samples=40, seed=1)

    def test_deferred_space_needs_no_training_and_is_not_frozen(self):
        a = args()
        a.ours_calibration_selection_mode = "defer_to_unified_tuner"
        run = mock.Mock(side_effect=AssertionError("must not train"))
        split = stratified_training_three_way_split(self.dataset, seed=22)
        with tempfile.TemporaryDirectory() as tmp:
            main, artifact = resolve_or_run_ours_calibration(self.dataset, a, tmp, run,
                                                             split=split, split_seed=22)
            self.assertIs(main.x_train, split.calibration_dataset.x_train)
            self.assertIs(main.x_test, self.dataset.x_test)
            self.assertEqual(artifact.status, "candidate_space_only")
            self.assertIsNone(artifact.selected_parameters)
            self.assertEqual(len(artifact.candidate_results), 3)
            with self.assertRaises(OursCalibrationError):
                apply_ours_parameters(a, artifact)
            run.assert_not_called()

    def test_standalone_from_scratch_freezes_and_reuses_cache(self):
        a = args()
        calls = mock.Mock(side_effect=fake_run)
        with tempfile.TemporaryDirectory() as tmp:
            _, artifact = resolve_or_run_ours_calibration(self.dataset, a, tmp, calls)
            self.assertEqual(artifact.status, "frozen")
            self.assertGreater(calls.call_count, 0)
            apply_ours_parameters(a, artifact)
            # Resolver fingerprint depends on the declared candidate space,
            # not the chosen policy injected into the namespace afterwards.
            _, cached = resolve_or_run_ours_calibration(self.dataset, args(), tmp,
                                                       mock.Mock(side_effect=AssertionError))
            self.assertEqual(cached.artifact_fingerprint, artifact.artifact_fingerprint)
            self.assertTrue(set(artifact.calibration_seeds).isdisjoint({a.seed}))

    def test_official_test_cannot_change_selection_fingerprint(self):
        a = args()
        a.ours_calibration_selection_mode = "defer_to_unified_tuner"
        altered = replace(self.dataset, x_test=self.dataset.x_test + 100)
        with tempfile.TemporaryDirectory() as tmp:
            _, first = resolve_or_run_ours_calibration(self.dataset, a, tmp)
            _, second = resolve_or_run_ours_calibration(altered, a, tmp)
        self.assertEqual(first.calibration_fingerprint, second.calibration_fingerprint)

    def test_artifact_tampering_and_old_versions_are_rejected(self):
        a = args()
        a.ours_calibration_selection_mode = "defer_to_unified_tuner"
        with tempfile.TemporaryDirectory() as tmp:
            _, artifact = resolve_or_run_ours_calibration(self.dataset, a, tmp)
            payload = artifact.to_dict()
            payload["schema_version"] = 3
            with self.assertRaisesRegex(ValueError, "obsolete"):
                OursCalibrationArtifact.from_dict(payload)
            payload = artifact.to_dict()
            payload["algorithm_version"] = "ours-normal-states-v1"
            with self.assertRaisesRegex(ValueError, "obsolete"):
                OursCalibrationArtifact.from_dict(payload)
            payload = artifact.to_dict()
            payload["status"] = "frozen"
            with self.assertRaisesRegex(ValueError, "checksum"):
                OursCalibrationArtifact.from_dict(payload)

    def test_changed_training_or_protocol_does_not_reuse_artifact(self):
        a = args()
        a.ours_calibration_selection_mode = "defer_to_unified_tuner"
        with tempfile.TemporaryDirectory() as tmp:
            _, first = resolve_or_run_ours_calibration(self.dataset, a, tmp)
            _, changed = resolve_or_run_ours_calibration(
                replace(self.dataset, x_train=self.dataset.x_train + 1), a, tmp)
            a.lr *= 2
            _, protocol = resolve_or_run_ours_calibration(self.dataset, a, tmp)
        self.assertNotEqual(first.calibration_fingerprint, changed.calibration_fingerprint)
        self.assertNotEqual(first.calibration_fingerprint, protocol.calibration_fingerprint)

    def test_missing_asr_sources_fail_before_any_training(self):
        a = args()
        a.attack_target_count = 100
        run = mock.Mock(side_effect=AssertionError("must fail before training"))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(OursCalibrationError, "source-label samples"):
                resolve_or_run_ours_calibration(self.dataset, a, tmp, run)
        run.assert_not_called()

    def test_failed_selection_keeps_a_feasibility_report(self):
        def failed(dataset, config):
            result = fake_run(dataset, config)
            return replace(result, final_accuracy=0.0 if config.method == "sm9rrs" else .9)
        with tempfile.TemporaryDirectory() as tmp:
            from sm9rrsfl.fair_tuning import FairTuningError
            with self.assertRaises(FairTuningError):
                resolve_or_run_ours_calibration(self.dataset, args(), tmp, failed)
            text = (Path(tmp) / "ours_candidate_feasibility.csv").read_text()
            self.assertIn("clean_accuracy_drop", text)
            self.assertTrue((Path(tmp) / "ours_validation_results.csv").exists())
