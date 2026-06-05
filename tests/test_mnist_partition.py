import unittest

import numpy as np

from sm9rrsfl.mnist import make_synthetic_mnist_like, partition_clients


class MnistPartitionTest(unittest.TestCase):
    def test_synthetic_dataset_shape(self):
        dataset = make_synthetic_mnist_like(train_samples=100, test_samples=20, seed=1)
        self.assertEqual(dataset.x_train.shape, (100, 784))
        self.assertEqual(dataset.y_test.shape, (20,))

    def test_dirichlet_partition_covers_all_samples(self):
        labels = np.arange(100) % 10
        parts = partition_clients(labels, 8, strategy="dirichlet", dirichlet_alpha=0.5, seed=2)
        merged = np.sort(np.concatenate(parts))
        self.assertTrue(np.array_equal(merged, np.arange(100)))
        self.assertTrue(all(len(part) > 0 for part in parts))


if __name__ == "__main__":
    unittest.main()
