import argparse
import unittest
from pathlib import Path

from sm9rrsfl.experiments import default_output_dir, resolve_output_dir


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


if __name__ == "__main__":
    unittest.main()
