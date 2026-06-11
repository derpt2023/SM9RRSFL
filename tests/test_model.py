import unittest

import numpy as np

from sm9rrsfl.datasets import ImageDataset
from sm9rrsfl.model import (
    DEFAULT_SPEC,
    accuracy,
    describe_compute_backend,
    init_params,
    local_train_delta,
    model_spec_for_dataset,
)


class CIFARModelTest(unittest.TestCase):
    def test_cifar10_uses_dataset_specific_cnn(self):
        rng = np.random.default_rng(7)
        dataset = ImageDataset(
            x_train=rng.normal(0.0, 1.0, size=(4, 3, 32, 32)).astype(np.float32),
            y_train=np.array([0, 1, 2, 3], dtype=np.int64),
            x_test=rng.normal(0.0, 1.0, size=(2, 3, 32, 32)).astype(np.float32),
            y_test=np.array([0, 1], dtype=np.int64),
            name="cifar10",
            input_shape=(3, 32, 32),
            num_classes=10,
        )
        spec = model_spec_for_dataset(dataset)
        self.assertEqual(spec.architecture, "cifar10")
        self.assertEqual(spec.kernel_size, 5)
        self.assertEqual(spec.cifar_conv_filters, (64, 64))
        self.assertEqual(spec.cifar_hidden_dims, (384, 192))
        self.assertGreater(spec.parameter_size, 1_000_000)
        self.assertGreater(spec.parameter_size, DEFAULT_SPEC.parameter_size)

        params = init_params(seed=3, spec=spec)
        delta, stats = local_train_delta(
            params,
            dataset.x_train,
            dataset.y_train,
            lr=0.001,
            epochs=1,
            batch_size=2,
            seed=5,
            spec=spec,
        )

        self.assertEqual(delta.shape, params.shape)
        self.assertEqual(stats.samples, 4)
        self.assertTrue(np.isfinite(delta).all())
        self.assertGreaterEqual(accuracy(params + delta, dataset.x_test, dataset.y_test, spec=spec), 0.0)

    def test_auto_compute_backend_preserves_model_contract(self):
        rng = np.random.default_rng(11)
        dataset = ImageDataset(
            x_train=rng.normal(0.0, 1.0, size=(6, 1, 28, 28)).astype(np.float32),
            y_train=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
            x_test=rng.normal(0.0, 1.0, size=(3, 1, 28, 28)).astype(np.float32),
            y_test=np.array([0, 1, 2], dtype=np.int64),
            name="mnist",
            input_shape=(1, 28, 28),
            num_classes=10,
        )
        spec = model_spec_for_dataset(dataset)
        params = init_params(seed=9, spec=spec)

        delta, stats = local_train_delta(
            params,
            dataset.x_train,
            dataset.y_train,
            lr=0.001,
            epochs=1,
            batch_size=3,
            seed=12,
            spec=spec,
            compute_backend="auto",
            device="auto",
        )
        acc = accuracy(
            params + delta,
            dataset.x_test,
            dataset.y_test,
            spec=spec,
            compute_backend="auto",
            device="auto",
        )

        self.assertIn(describe_compute_backend("auto", "auto").split(":", 1)[0], {"numpy", "torch"})
        self.assertEqual(delta.shape, params.shape)
        self.assertEqual(stats.samples, 6)
        self.assertTrue(np.isfinite(delta).all())
        self.assertGreaterEqual(acc, 0.0)


if __name__ == "__main__":
    unittest.main()
