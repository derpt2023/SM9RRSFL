"""Single-client replays must retain the exact initial model and local recipe."""
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import numpy as np

import check_cifar_client_repeatability as check
from sm9rrsfl import fl
from sm9rrsfl.datasets import make_synthetic_mnist_like, stratified_training_three_way_split


class ClientRepeatabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.data = stratified_training_three_way_split(
            make_synthetic_mnist_like(train_samples=160, test_samples=20, seed=80),
            seed=81, train_fraction=.8, calibration_fraction=.1,
            attack_fraction=.1).calibration_dataset
        cls.config = fl.ExperimentConfig(
            method="sm9rrs", num_clients=8, rounds=100, detector_window=20,
            attack_start_round=25, local_epochs=1, batch_size=8, lr=.01,
            crypto_mode="simulated", compute_backend="torch", device="cpu", seed=403)
        cls.spec = fl.model_spec_for_dataset(cls.data)
        cls.params = fl.init_params(seed=403, spec=cls.spec)
        cls.partitions = fl.partition_clients(
            cls.data.y_train, 8, strategy=cls.config.partition,
            dirichlet_alpha=cls.config.dirichlet_alpha, seed=403)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_warmup_and_repeats_use_original_input_and_training_function(self):
        original = fl._local_train_client_delta
        before = self.params.copy()
        seen = []

        def record(*args, **kwargs):
            seen.append((kwargs["client_idx"], kwargs["round_id"], np.asarray(args[0]).copy(), kwargs["config"]))
            return original(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            with mock.patch.object(fl, "_local_train_client_delta", side_effect=record):
                check.run_client_replays(self.data, self.config, self.params, self.partitions[7],
                                         7, Path(temp) / "case", warmup=True, repeats=3)
        self.assertEqual([item[0] for item in seen], list(range(7)) + [7, 7, 7])
        for _, rd, params, config in seen:
            self.assertEqual(rd, 1)
            self.assertTrue(np.array_equal(params, before))
            self.assertEqual(config, self.config)
        self.assertTrue(np.array_equal(self.params, before))

    def test_invalid_indices_rejected_before_training_and_existing_output_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(fl, "_local_train_client_delta") as training:
                with self.assertRaises(ValueError):
                    check.run_client_replays(self.data, self.config, self.params, self.partitions[0],
                                             7, root / "bad", repeats=1)
                training.assert_not_called()
            existing = root / "existing"
            existing.mkdir()
            sentinel = existing / "environment.json"
            sentinel.write_text("original")
            with self.assertRaises(FileExistsError):
                check.run_client_replays(self.data, self.config, self.params, self.partitions[7],
                                         7, existing, repeats=1, environment_metadata={"new": True})
            self.assertEqual(sentinel.read_text(), "original")

    def test_strict_settings_are_explicit_and_preserve_tf32(self):
        parent = {"CUDA_VISIBLE_DEVICES": "7", "CUBLAS_WORKSPACE_CONFIG": None}
        child = check.strict_child_environment(parent)
        self.assertIsNone(parent["CUBLAS_WORKSPACE_CONFIG"])
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], "7")
        fake = SimpleNamespace(
            backends=SimpleNamespace(
                cudnn=SimpleNamespace(allow_tf32=True, deterministic=False, benchmark=True),
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
            use_deterministic_algorithms=mock.Mock())
        flags = {"cudnn_tf32": True, "matmul_tf32": False}
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "must start"):
                check.apply_strict_settings(fake, flags)
        fake.use_deterministic_algorithms.assert_not_called()
        with mock.patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": child["CUBLAS_WORKSPACE_CONFIG"]}):
            check.apply_strict_settings(fake, flags)
        fake.use_deterministic_algorithms.assert_called_once_with(True, warn_only=False)
        self.assertTrue(fake.backends.cudnn.deterministic)
        self.assertFalse(fake.backends.cudnn.benchmark)
        self.assertTrue(fake.backends.cudnn.allow_tf32)
        self.assertFalse(fake.backends.cuda.matmul.allow_tf32)

    def test_strict_environment_exception_does_not_hide_other_changes(self):
        baseline = {key: "recorded" for key in
                    ("python", "executable", "numpy", "torch", "cuda", "cudnn", "gpu", *check.FLAG_KEYS)}
        baseline.update(environment={"CUBLAS_WORKSPACE_CONFIG": None, "OMP_NUM_THREADS": "1"},
                        source_sha256={"sm9rrsfl/fl.py": "original-hash"})
        check.validate_baseline_environment(baseline, baseline)
        current = copy.deepcopy(baseline)
        current["environment"]["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        with self.assertRaisesRegex(ValueError, "CUBLAS_WORKSPACE_CONFIG"):
            check.validate_baseline_environment(current, baseline)
        check.validate_baseline_environment(current, baseline, strict=True)
        current["cudnn_tf32"] = "changed"
        with self.assertRaisesRegex(ValueError, "cudnn_tf32"):
            check.validate_baseline_environment(current, baseline, strict=True)

    def test_summary_quantifies_saved_update_differences(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = []
            for index, delta in enumerate((np.array([1., 2.], dtype=np.float32),
                                           np.array([1., 2.25], dtype=np.float32))):
                path = Path(temp) / f"delta_{index}.npy"
                np.save(path, delta, allow_pickle=False)
                runs.append({"delta": check.array_identity(delta), "delta_path": str(path),
                             "matches_A1_delta": index == 0})
            summary = check.summarize_group([{"status": "complete", "replays": {"runs": runs}}])
            self.assertFalse(summary["all_equal"])
            self.assertEqual(summary["max_abs_difference_from_first"], .25)
            self.assertEqual(summary["max_l2_difference_from_first"], .25)
            self.assertEqual(summary["matches_A1_delta_count"], 1)


if __name__ == "__main__":
    unittest.main()
