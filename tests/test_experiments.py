import argparse
import io
import os
import unittest
from pathlib import Path

from sm9rrsfl.experiments import (
    ProgressReporter,
    _format_duration,
    default_output_dir,
    parse_args,
    resolve_parallel_jobs,
    resolve_output_dir,
)
from sm9rrsfl.fl import ExperimentConfig


class ExperimentOutputDirTest(unittest.TestCase):
    def test_sm9_mnist_default_keeps_main_output_directory(self):
        self.assertEqual(default_output_dir("mnist", "sm9"), Path("outputs/mnist"))

    def test_simulated_mnist_default_uses_separate_output_directory(self):
        self.assertEqual(
            default_output_dir("mnist", "simulated"),
            Path("outputs/mnist_simulated"),
        )

    def test_cifar10_default_uses_dataset_specific_directory(self):
        self.assertEqual(default_output_dir("cifar10", "sm9"), Path("outputs/cifar10"))
        self.assertEqual(
            default_output_dir("cifar10", "simulated"),
            Path("outputs/cifar10_simulated"),
        )

    def test_explicit_output_directory_overrides_mode_default(self):
        args = argparse.Namespace(
            dataset="mnist",
            crypto_mode="simulated",
            output_dir="outputs/custom",
        )

        self.assertEqual(resolve_output_dir(args), Path("outputs/custom"))

    def test_cifar10_clean_baseline_preset(self):
        args = parse_args(["--cifar10-clean-baseline", "--rounds", "5"])

        self.assertEqual(args.dataset, "cifar10")
        self.assertEqual(args.methods, ["fedavg"])
        self.assertEqual(args.ratios, [0.0])
        self.assertEqual(args.attack, "none")
        self.assertEqual(args.partitions, ["iid"])
        self.assertIsNone(args.train_samples)
        self.assertIsNone(args.test_samples)
        self.assertEqual(args.rounds, 5)
        self.assertEqual(resolve_output_dir(args), Path("outputs/cifar10_clean_baseline"))

    def test_cifar10_clean_baseline_uses_cifar_training_defaults(self):
        args = parse_args(["--cifar10-clean-baseline"])

        self.assertEqual(args.rounds, 300)
        self.assertEqual(args.local_epochs, 5)
        self.assertEqual(args.batch_size, 50)
        self.assertEqual(args.lr, 0.05)
        self.assertEqual(args.lr_decay, 0.99)

    def test_compute_backend_defaults_to_numpy(self):
        args = parse_args([])

        self.assertEqual(args.compute_backend, "numpy")
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.rounds, 30)
        self.assertEqual(args.local_epochs, 1)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.lr, 0.05)
        self.assertEqual(args.lr_decay, 1.0)

    def test_cifar10_uses_paper_training_defaults(self):
        args = parse_args(["--dataset", "cifar10"])

        self.assertEqual(args.rounds, 300)
        self.assertEqual(args.local_epochs, 5)
        self.assertEqual(args.batch_size, 50)
        self.assertEqual(args.lr, 0.05)
        self.assertEqual(args.lr_decay, 0.99)

    def test_explicit_training_arguments_override_dataset_defaults(self):
        args = parse_args(
            [
                "--dataset",
                "cifar10",
                "--rounds",
                "7",
                "--local-epochs",
                "2",
                "--batch-size",
                "16",
                "--lr",
                "0.02",
                "--lr-decay",
                "0.95",
            ]
        )

        self.assertEqual(args.rounds, 7)
        self.assertEqual(args.local_epochs, 2)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.lr, 0.02)
        self.assertEqual(args.lr_decay, 0.95)

    def test_gpu_backend_arguments_are_parsed(self):
        args = parse_args(["--compute-backend", "torch", "--device", "mps"])

        self.assertEqual(args.compute_backend, "torch")
        self.assertEqual(args.device, "mps")

    def test_eval_interval_and_sm9_workers_are_parsed(self):
        args = parse_args(["--eval-interval", "5", "--sm9-workers", "3"])

        self.assertEqual(args.eval_interval, 5)
        self.assertEqual(args.sm9_workers, 3)

    def test_sm9_workers_auto_is_bounded_by_client_count(self):
        args = parse_args(["--num-clients", "4", "--sm9-workers", "auto"])

        self.assertGreaterEqual(args.sm9_workers, 1)
        self.assertLessEqual(args.sm9_workers, min(4, os.cpu_count() or 1))

    def test_parallel_jobs_integer_is_capped_by_config_count(self):
        args = parse_args(["--jobs", "8"])

        self.assertEqual(resolve_parallel_jobs(args.jobs, object(), [object(), object()], args), 2)

    def test_progress_can_be_disabled(self):
        args = parse_args(["--no-progress"])

        self.assertTrue(args.no_progress)

    def test_progress_reporter_prints_eta(self):
        stream = io.StringIO()
        progress = ProgressReporter(total=2, stream=stream)
        progress.finish_config(ExperimentConfig(method="fedavg", malicious_ratio=0.0))
        progress.close()

        output = stream.getvalue()
        self.assertIn("1/2", output)
        self.assertIn("eta=", output)
        self.assertIn("complete", output)

    def test_format_duration(self):
        self.assertEqual(_format_duration(5), "5s")
        self.assertEqual(_format_duration(65), "1m05s")
        self.assertEqual(_format_duration(3661), "1h01m01s")


if __name__ == "__main__":
    unittest.main()
