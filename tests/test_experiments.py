import argparse
import unittest
from pathlib import Path

from sm9rrsfl.experiments import default_output_dir, parse_args, resolve_output_dir


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

    def test_compute_backend_defaults_to_numpy(self):
        args = parse_args([])

        self.assertEqual(args.compute_backend, "numpy")
        self.assertEqual(args.device, "auto")

    def test_gpu_backend_arguments_are_parsed(self):
        args = parse_args(["--compute-backend", "torch", "--device", "mps"])

        self.assertEqual(args.compute_backend, "torch")
        self.assertEqual(args.device, "mps")


if __name__ == "__main__":
    unittest.main()
