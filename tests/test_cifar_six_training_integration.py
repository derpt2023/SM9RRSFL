"""Exercise the new runtime through the real six-method CIFAR model path."""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import sm9rrsfl
import numpy as np
import torch

from cifar_ours_development_policy import weak_quarantine_policy
import run_cifar_six_from_scratch as runner
from sm9rrsfl.datasets import ImageDataset
from sm9rrsfl.fl import ExperimentConfig, run_experiment
from sm9rrsfl import torch_backend


class SixMethodTrainingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        rng = np.random.default_rng(194)
        cls.dataset = ImageDataset(
            rng.normal(0, .1, (20, 3, 32, 32)).astype(np.float32), np.arange(20) % 10,
            rng.normal(0, .1, (10, 3, 32, 32)).astype(np.float32), np.arange(10),
            name="cifar10", x_attack=rng.normal(0, .1, (20, 3, 32, 32)).astype(np.float32),
            y_attack=np.arange(20) % 10)
        cls.config = ExperimentConfig(
            num_clients=4, rounds=4, detector_window=3, attack_start_round=4,
            attack_target_count=2, malicious_ratio=.25, compute_backend="torch",
            device="cpu", crypto_mode="simulated", early_stop=False,
            lr=.005, batch_size=5, local_epochs=1, attack_epochs=1,
            attack_stealth_steps=1, attack_boost=5., seed=194,
            detector_distance_threshold=1.75, detector_reject_threshold=6.,
            detector_drift_allowance=1.25, detector_history_threshold=1.,
            detector_reference_budget=3.5, vert_history_window=3,
            vert_predict_epochs=1, vert_projection_dim=8)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_all_six_methods_train_under_the_original_optimizer(self):
        for method in ("sm9rrs", "vert", "alignins", "krum", "ding13", "fedavg"):
            with self.subTest(method=method):
                config = replace(self.config, method=method)
                checkpoints = []
                result = run_experiment(self.dataset, config, checkpoint_callback=checkpoints.append)
                self.assertEqual(result.stopped_round, 4)
                self.assertEqual(result.nonfinite_updates, 0)
                self.assertTrue(np.isfinite(checkpoints[-1]["params"]).all())
                self.assertTrue(all(np.isfinite(row.accuracy) for row in result.records))
                self.assertTrue(result.records[-1].attack_active)

    def test_weak_policy_and_original_training_resume_from_the_same_round_state(self):
        config = replace(self.config, method="sm9rrs")
        saved = {}
        def stop(state):
            if state["completed_round"] == 3:
                saved["state"] = pickle.dumps(state)
                raise InterruptedError("integration checkpoint")
        with weak_quarantine_policy():
            full = run_experiment(self.dataset, config)
        with weak_quarantine_policy():
            with self.assertRaises(InterruptedError):
                run_experiment(self.dataset, config, checkpoint_callback=stop)
        with weak_quarantine_policy():
            resumed = run_experiment(self.dataset, config, resume_state=pickle.loads(saved["state"]))
        self.assertEqual(full.records, resumed.records)
        self.assertEqual(full.diagnostics, resumed.diagnostics)

    def test_real_worker_keeps_optimizer_and_reuses_completed_training(self):
        config = replace(self.config, method="sm9rrs")
        task = {"task_id": "integration_worker", "phase": "validation", "method": "sm9rrs",
                "candidate": {"candidate_id": "sm9rrs-v8-003", "variant": "weak_quarantine",
                              "weak_threshold": 1.25},
                "config": asdict(config), "fingerprint": "integration"}
        repo = Path(runner.__file__).resolve().parent
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            runner.write_json(output / "manifest.json", {
                "spec": {"numerics": {"mode": "original_runtime_defaults"}}, "data_contract": {},
                "source_sha256": runner.source_hashes(repo)})
            runner.write_json(output / "validation_plan.json", {"tasks": [task]})
            args = SimpleNamespace(output=output, phase="validation", worker=task["task_id"],
                                   devices=["cpu"], data_dir=None)
            split = SimpleNamespace(calibration_dataset=self.dataset)
            with mock.patch.object(runner, "worker_environment", return_value={"environment": {}}), \
                    mock.patch.object(runner, "load_split", return_value=(split, {})), \
                    redirect_stdout(io.StringIO()):
                original_context = torch_backend.TorchTrainingContext
                runner.run_worker(args)
                self.assertIs(torch_backend.TorchTrainingContext, original_context)
                folder = output / "tasks" / task["task_id"]
                result = runner.checked_completed(output, task)
                self.assertEqual(result.stopped_round, config.rounds)
                self.assertFalse((folder / "numerical_stats.json").exists())
                self.assertEqual(json.loads((folder / "metrics.json").read_text())["nonfinite_updates"], 0)
                with mock.patch.object(runner.experiments, "run_measured_experiment",
                                       side_effect=AssertionError("completed training must be reused")):
                    runner.run_worker(args)
                self.assertTrue((folder / "rounds.csv").is_file())


if __name__ == "__main__":
    unittest.main()
