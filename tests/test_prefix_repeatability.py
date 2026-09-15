"""Numeric observations must preserve both runner paths and identify divergence."""
import contextlib
from copy import deepcopy
from dataclasses import asdict
import io
from pathlib import Path
import tempfile
import unittest

import torch

import check_cifar_prefix_repeatability as check
from sm9rrsfl import fl, experiments
from sm9rrsfl.datasets import make_synthetic_mnist_like, stratified_training_three_way_split


class PrefixRepeatabilityTest(unittest.TestCase):
    def test_both_runners_match_original_and_preserve_original_functions(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        original_run, original_measured = fl.run_experiment, experiments.run_experiment
        original_client = fl._local_train_client_delta
        try:
            data = stratified_training_three_way_split(
                make_synthetic_mnist_like(train_samples=120, test_samples=20, seed=81),
                seed=82, train_fraction=.8, calibration_fraction=.1,
                attack_fraction=.1).calibration_dataset
            config = fl.ExperimentConfig(
                method="sm9rrs", num_clients=4, rounds=4, detector_window=3,
                attack_start_round=4, local_epochs=1, batch_size=16, lr=.01,
                attack_epochs=1, attack_stealth_steps=1, attack_target_count=1,
                crypto_mode="simulated", compute_backend="torch", device="cpu",
                early_stop=False, seed=83)
            baseline = fl.run_experiment(data, config)
            reference = {r.round: {k: str(v) for k, v in asdict(r).items()} for r in baseline.records}
            with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
                root = Path(temp)
                a = check.run_case(data, config, root / "A", "measured", reference, "fixture")
                b = check.run_case(data, config, root / "B", "probe", reference, "fixture")
                self.assertTrue(check.compare_cases(a, b)["equal"])
                self.assertEqual(a["config"]["rounds"], 4)
                self.assertEqual(a["historical_metric_mismatches"], [])
                self.assertEqual(len(a["clients"]), 8)
                self.assertTrue((root / "A" / "params_round_002.npy").exists())
                # A exercises the production writer, whose intentional stop is
                # confined to this new diagnostic directory.
                self.assertEqual(len(list((root / "A" / "checkpoints").glob("*.pickle"))), 1)
                changed = deepcopy(b)
                changed["clients"][0]["delta"]["sha256"] = "different"
                diff = check.compare_cases(a, changed)
                self.assertFalse(diff["equal"])
                self.assertEqual(diff["first_client_difference"]["fields"], ["delta"])
                self.assertTrue(diff["first_client_difference"]["same_input_and_indices"])
                changed["clients"][0]["input"]["sha256"] = "different"
                self.assertFalse(check.compare_cases(a, changed)["first_client_difference"]["same_input_and_indices"])
                with self.assertRaises(FileExistsError):
                    check.run_case(data, config, root / "A", "probe", reference, "fixture")
            self.assertIs(fl.run_experiment, original_run)
            self.assertIs(experiments.run_experiment, original_measured)
            self.assertIs(fl._local_train_client_delta, original_client)
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
