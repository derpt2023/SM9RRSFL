"""Real miniature training, not a replay of historical experiment outputs."""
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sm9rrsfl.fair_tuning import OBJECTIVE_DEFAULTS, load_fair_tuning_config, run_fair_tuning, score_trial
from sm9rrsfl.datasets import make_synthetic_mnist_like, stratified_training_three_way_split
from sm9rrsfl.fl import ExperimentConfig, run_experiment
from sm9rrsfl.ours_policy import bounded_candidates


class NormalStateWorkflowTest(unittest.TestCase):
    def test_from_scratch_validation_final_html_and_resume(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root / "configs/fair_tuning.example.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            payload["shared_parameters"].update(
                dataset="synthetic", output_dir=str(output), download=False,
                # Use the production-length clean prefix. With just K=3 the
                # aggressive first candidate can exhaust all clients early;
                # that is a real invalid run, covered separately below.
                train_samples=400, test_samples=80, rounds=10, K=7,
                client_counts=[10], partitions=["iid"], ratio_range=[0, .4, 3],
                attack="alternating_minimization", attack_target_count=1, attack_start_round=9,
                calibration_candidate_budget=1, compute_backend="numpy", device="cpu",
                jobs=1, crypto_mode="simulated", progress=False)
            payload["tuning"].update(
                validation_seeds=[41], final_seeds=[141], trials_per_tunable_method=1,
                final_jobs=1,
                # Mechanics test on tiny synthetic data, NOT a performance
                # claim or a replacement for the production 0.05 clean gate.
                max_clean_accuracy_drop=1.0,
                method_spaces={"sm9rrs": "auto", "vert": {"vert_history_window": [3],
                               "vert_predict_epochs": [1]}, "alignins": {
                               "alignins_sparsity": [.3]}, "krum": {}, "ding13": {}, "fedavg": {}})
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps(payload))
            spec = load_fair_tuning_config(config)
            with mock.patch("sys.stdout", new=io.StringIO()):
                selected = run_fair_tuning(spec)
            self.assertEqual(len(selected), 6)
            report = json.loads((output / "best_parameters.json").read_text())
            self.assertTrue(report["training_data_contract"]["shared_training_arrays_across_phases"])
            contract = report["training_data_contract"]
            self.assertEqual(contract["train_samples"] + contract["calibration_samples"]
                             + contract["attack_auxiliary_samples"], 400)
            self.assertTrue(list(output.rglob("visualizations.html")))
            for filename in ("candidate_feasibility.csv", "validation_results.csv", "tuning_trials.csv"):
                self.assertTrue((output / filename).exists(), filename)
            with mock.patch("sys.stdout", new=io.StringIO()), mock.patch(
                    "sm9rrsfl.fair_tuning.run_measured_experiment", side_effect=AssertionError("must resume")):
                again = run_fair_tuning(spec)
            self.assertEqual(selected, again)

    def test_real_early_exhaustion_remains_invalid_for_selection(self):
        raw = make_synthetic_mnist_like(train_samples=400, test_samples=80, seed=42)
        data = stratified_training_three_way_split(raw, seed=20260810).calibration_dataset
        parameters = bounded_candidates(1)[0]
        parameters.update(detector_distance_threshold=.02, detector_reject_threshold=.04,
                          detector_history_threshold=.01, detector_drift_allowance=.01)
        config = ExperimentConfig(
            num_clients=10, rounds=6, detector_window=3, malicious_ratio=0.,
            attack="alternating_minimization", attack_start_round=5, attack_target_count=1,
            compute_backend="numpy", device="cpu", crypto_mode="simulated",
            lr=.05, seed=41, early_stop=False, **parameters)
        clean = run_experiment(data, config)
        attacked = run_experiment(data, replace(config, malicious_ratio=.2))
        self.assertLess(clean.stopped_round, config.rounds)
        self.assertEqual(len(clean.blacklisted_clients), config.num_clients)
        trial = score_trial("sm9rrs", "sm9rrs-001", parameters, [clean, attacked],
                            objective=OBJECTIVE_DEFAULTS)
        self.assertFalse(trial.valid)
        self.assertIn("round_completion_rate", trial.invalid_reasons)
        self.assertEqual(trial.score, float("-inf"))
