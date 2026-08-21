import unittest
import importlib.util
from unittest import mock

import numpy as np

from sm9rrsfl.aggregation import fedavg, krum, torch_krum, torch_weighted_fedavg, weighted_fedavg


class KrumTest(unittest.TestCase):
    def test_numpy_svd_fallbacks_survive_lapack_nonconvergence(self):
        from sm9rrsfl import torch_backend

        rng = np.random.default_rng(19)
        matrix = rng.normal(size=(12, 4)).astype(np.float32)
        expected = np.linalg.svd(matrix.T @ matrix, compute_uv=False)

        with mock.patch(
            "numpy.linalg.svd",
            side_effect=np.linalg.LinAlgError("injected nonconvergence"),
        ):
            eig_fallback = torch_backend.numpy_singular_values_from_gram(matrix)
            sigma, direction = torch_backend.numpy_top_singular_feature(matrix)

        with mock.patch(
            "numpy.linalg.svd",
            side_effect=np.linalg.LinAlgError("injected nonconvergence"),
        ), mock.patch(
            "numpy.linalg.eigvalsh",
            side_effect=np.linalg.LinAlgError("injected eigen failure"),
        ):
            jacobi_fallback = torch_backend.numpy_singular_values_from_gram(matrix)

        self.assertTrue(np.allclose(eig_fallback, expected, rtol=1e-5, atol=1e-6))
        self.assertTrue(np.allclose(jacobi_fallback, expected, rtol=1e-5, atol=1e-6))
        self.assertTrue(np.isfinite(sigma))
        self.assertTrue(np.isfinite(direction).all())
        self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=5)

    def test_scaled_gram_singular_values_remain_finite_for_large_float32_input(self):
        from sm9rrsfl.torch_backend import numpy_singular_values_from_gram

        matrix = np.full((32, 4), 1e30, dtype=np.float32)
        singulars = numpy_singular_values_from_gram(matrix)

        self.assertTrue(np.isfinite(singulars).all())
        self.assertGreater(float(singulars[0]), 0.0)

    def test_top_singular_subspace_returns_true_singular_values_and_projector(self):
        from sm9rrsfl.torch_backend import numpy_top_singular_subspace

        matrix = np.diag([5.0, 3.0, 1.0]).astype(np.float32)
        singular_values, basis = numpy_top_singular_subspace(matrix, rank=2)

        # The detector needs sigma rather than the eigenvalues sigma**2 of
        # A.T@A, because it applies log(sigma + eps) and the q/(q+1) gap.
        np.testing.assert_allclose(singular_values, [5.0, 3.0, 1.0], atol=1e-7)
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-7)
        np.testing.assert_allclose(
            basis @ basis.T,
            np.diag([1.0, 1.0, 0.0]),
            atol=1e-7,
        )
        gap = (singular_values[1] - singular_values[2]) / singular_values[1]
        self.assertAlmostEqual(float(gap), 2.0 / 3.0)

    def test_top_singular_subspace_falls_back_from_eigh_to_true_svd(self):
        from sm9rrsfl.torch_backend import numpy_top_singular_subspace

        matrix = np.diag([5.0, 3.0, 1.0]).astype(np.float32)
        with mock.patch(
            "numpy.linalg.eigh",
            side_effect=np.linalg.LinAlgError("injected eigen failure"),
        ):
            singular_values, basis = numpy_top_singular_subspace(matrix, rank=2)

        np.testing.assert_allclose(singular_values, [5.0, 3.0, 1.0], atol=1e-7)
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-7)
        np.testing.assert_allclose(
            basis @ basis.T,
            np.diag([1.0, 1.0, 0.0]),
            atol=1e-7,
        )

    def test_top_singular_subspace_requires_q_plus_one_value(self):
        from sm9rrsfl.torch_backend import numpy_top_singular_subspace

        with self.assertRaisesRegex(ValueError, r"min\(matrix.shape\) - 1"):
            numpy_top_singular_subspace(np.eye(2, dtype=np.float32), rank=2)

    def test_torch_top_singular_subspace_matches_numpy_on_cpu(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        from sm9rrsfl.torch_backend import (
            numpy_top_singular_subspace,
            torch_top_singular_subspace,
        )

        matrix = np.diag([5.0, 3.0, 1.0]).astype(np.float32)
        expected_values, expected_basis = numpy_top_singular_subspace(matrix, rank=2)
        actual_values, actual_basis = torch_top_singular_subspace(
            matrix,
            rank=2,
            device="cpu",
        )

        np.testing.assert_allclose(actual_values, expected_values, atol=1e-6)
        np.testing.assert_allclose(
            actual_basis @ actual_basis.T,
            expected_basis @ expected_basis.T,
            atol=1e-6,
        )
        with self.assertRaisesRegex(ValueError, r"min\(matrix.shape\) - 1"):
            torch_top_singular_subspace(
                np.eye(2, dtype=np.float32),
                rank=2,
                device="cpu",
            )

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

    def test_streaming_weighted_fedavg_matches_stacked_path(self):
        from sm9rrsfl import aggregation

        updates = [
            np.array([0.0, 2.0], dtype=np.float32),
            np.array([10.0, 20.0], dtype=np.float32),
        ]
        with mock.patch.object(aggregation, "_STREAMING_AVERAGE_THRESHOLD_BYTES", 1):
            result = aggregation.weighted_fedavg(updates, [0.75, 0.25])

        self.assertTrue(np.allclose(result, [2.5, 6.5]))

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_torch_weighted_fedavg_matches_numpy(self):
        updates = np.array([[0.0, 0.0], [10.0, 20.0]], dtype=np.float32)
        expected = weighted_fedavg(updates, [1.0, 1.0], sample_counts=[1, 3])
        result = torch_weighted_fedavg(updates, [1.0, 1.0], sample_counts=[1, 3], device="cpu")
        self.assertTrue(np.allclose(result, expected))

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_torch_streaming_average_matches_numpy(self):
        from sm9rrsfl import torch_backend

        updates = [
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([10.0, 20.0], dtype=np.float32),
        ]
        expected = weighted_fedavg(updates, [1.0, 3.0])
        with mock.patch.object(
            torch_backend,
            "_STREAMING_TORCH_AVERAGE_THRESHOLD_BYTES",
            1,
        ):
            result = torch_weighted_fedavg(updates, [1.0, 3.0], device="cpu")

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

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
    def test_svd_helpers_fallback_when_torch_op_is_not_implemented(self):
        from sm9rrsfl import torch_backend

        matrix = np.eye(3, dtype=np.float32)
        with mock.patch("torch.linalg.svd", side_effect=NotImplementedError("missing svd op")):
            sigma, u0 = torch_backend.torch_top_singular_feature(matrix, device="cpu")
        with mock.patch("torch.linalg.svdvals", side_effect=NotImplementedError("missing svdvals op")):
            singulars = torch_backend.torch_singular_values_from_gram(matrix, device="cpu")

        self.assertAlmostEqual(sigma, 1.0)
        self.assertEqual(u0.shape, (3,))
        self.assertTrue(np.allclose(singulars, [1.0, 1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
