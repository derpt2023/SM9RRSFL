import unittest
import importlib.util
from unittest import mock

import numpy as np

from sm9rrsfl.aggregation import fedavg, krum, torch_krum, torch_weighted_fedavg, weighted_fedavg


class KrumTest(unittest.TestCase):
    def test_krum_selects_update_near_honest_cluster(self):
        updates = np.array(
            [
                [0.0, 0.0],
                [0.1, -0.1],
                [-0.1, 0.1],
                [20.0, 20.0],
                [-25.0, 18.0],
            ],
            dtype=np.float32,
        )
        result = krum(updates, byzantine_count=2)
        self.assertIn(result.selected_index, {0, 1, 2})

    def test_weighted_fedavg_uses_weights(self):
        updates = np.array([[0.0, 0.0], [10.0, 20.0]], dtype=np.float32)
        result = weighted_fedavg(updates, [0.75, 0.25])
        self.assertTrue(np.allclose(result, [2.5, 5.0]))

    def test_fedavg_can_weight_by_sample_count(self):
        updates = np.array([[0.0, 0.0], [10.0, 20.0]], dtype=np.float32)
        result = fedavg(updates, sample_counts=[1, 3])
        self.assertTrue(np.allclose(result, [7.5, 15.0]))

    def test_weighted_fedavg_combines_defense_weights_and_sample_counts(self):
        updates = np.array([[0.0, 0.0], [10.0, 20.0]], dtype=np.float32)
        result = weighted_fedavg(updates, [1.0, 1.0], sample_counts=[1, 3])
        self.assertTrue(np.allclose(result, [7.5, 15.0]))

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_torch_weighted_fedavg_matches_numpy(self):
        updates = np.array([[0.0, 0.0], [10.0, 20.0]], dtype=np.float32)
        expected = weighted_fedavg(updates, [1.0, 1.0], sample_counts=[1, 3])
        result = torch_weighted_fedavg(updates, [1.0, 1.0], sample_counts=[1, 3], device="cpu")
        self.assertTrue(np.allclose(result, expected))

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_torch_krum_matches_numpy_selection(self):
        updates = np.array(
            [
                [0.0, 0.0],
                [0.1, -0.1],
                [-0.1, 0.1],
                [20.0, 20.0],
                [-25.0, 18.0],
            ],
            dtype=np.float32,
        )
        expected = krum(updates, byzantine_count=2)
        result = torch_krum(updates, byzantine_count=2, device="cpu")
        self.assertEqual(result.selected_index, expected.selected_index)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_mps_svd_helpers_use_numpy_fallback(self):
        from sm9rrsfl import torch_backend

        matrix = np.eye(3, dtype=np.float32)
        fake_device = type("FakeDevice", (), {"type": "mps"})()
        with mock.patch.object(torch_backend, "_resolve_device", return_value=fake_device):
            with mock.patch("torch.linalg.svd", side_effect=AssertionError("unexpected torch svd")):
                sigma, u0 = torch_backend.torch_top_singular_feature(matrix, device="mps")
            with mock.patch("torch.linalg.svdvals", side_effect=AssertionError("unexpected torch svdvals")):
                singulars = torch_backend.torch_singular_values_from_gram(matrix, device="mps")

        self.assertAlmostEqual(sigma, 1.0)
        self.assertEqual(u0.shape, (3,))
        self.assertTrue(np.allclose(singulars, [1.0, 1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
