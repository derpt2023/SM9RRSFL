import unittest
from unittest import mock

import numpy as np

from sm9rrsfl.vert import VERTDefense


class VERTNumericsTest(unittest.TestCase):
    def defense(self, **kwargs):
        return VERTDefense(
            ["a", "b"], parameter_size=3, projection_dim=2,
            history_window=2, predict_epochs=1, **kwargs,
        )

    def torch_defense(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        defense = self.defense(compute_backend="torch", device="cpu")
        self.assertTrue(defense._ensure_torch_backend())
        return torch, defense

    def test_large_finite_cosine_matches_float64_reference(self):
        defense = self.defense()
        left = np.asarray([1e20, -2e20, 3e20], dtype=np.float32)
        right = np.asarray([-2e20, 1e20, 4e20], dtype=np.float32)
        expected = float(np.dot(left.astype(np.float64), right.astype(np.float64)) / (
            np.linalg.norm(left.astype(np.float64)) * np.linalg.norm(right.astype(np.float64))
        ))
        self.assertAlmostEqual(defense._cosine(left, right), expected, places=14)
        self.assertAlmostEqual(defense._cosine(left, left), 1.0, places=14)

    def test_torch_large_cosine_preserves_dtype_and_device(self):
        torch, defense = self.torch_defense()
        for dtype in (torch.float32, torch.float64):
            left = torch.tensor([1e20, -2e20, 3e20], dtype=dtype)
            right = torch.tensor([-2e20, 1e20, 4e20], dtype=dtype)
            actual = defense._torch_cosine(left, right)
            self.assertEqual(actual.dtype, dtype)
            self.assertEqual(actual.device, left.device)
            self.assertAlmostEqual(actual.item(), defense._cosine(left.numpy(), right.numpy()), places=6)
            self.assertAlmostEqual(defense._torch_cosine(left, left).item(), 1.0, places=6)

    def test_normal_cosine_and_original_product_epsilon_rule(self):
        torch, defense = self.torch_defense()
        cases = [
            ([1., 2., 3.], [4., 5., 6.]),
            ([0., 0., 0.], [1., 2., 3.]),
            ([1e-7, 0., 0.], [1e-7, 0., 0.]),
            # A small individual norm does not imply a small norm product.
            ([1e-15, 0., 0.], [1e6, 0., 0.]),
        ]
        for left_values, right_values in cases:
            left, right = np.asarray(left_values), np.asarray(right_values)
            product = np.linalg.norm(left) * np.linalg.norm(right)
            expected = 0.0 if product <= defense.eps else np.dot(left, right) / product
            self.assertAlmostEqual(defense._cosine(left, right), expected, places=14)
            actual = defense._torch_cosine(torch.tensor(left), torch.tensor(right))
            self.assertAlmostEqual(actual.item(), expected, places=14)

    def test_nonfinite_cosine_inputs_are_explicit_errors(self):
        torch, defense = self.torch_defense()
        for invalid in (float("nan"), float("inf"), -float("inf")):
            bad = np.asarray([invalid, 0., 1.], dtype=np.float32)
            zeros = np.zeros(3, dtype=np.float32)
            with self.assertRaisesRegex(FloatingPointError, "cosine input"):
                defense._cosine(zeros, bad)
            with self.assertRaisesRegex(FloatingPointError, "cosine input"):
                defense._torch_cosine(torch.tensor(zeros), torch.tensor(bad))

    def test_nonfinite_selection_scores_never_select_all_or_top_k(self):
        for options in ({}, {"top_k": 1}, {"malicious_ratio_prior": .5}):
            defense = self.defense(**options)
            for scores in ([float("nan")], [.9, float("inf")]):
                with self.assertRaisesRegex(FloatingPointError, "selection scores"):
                    defense._effective_top_k(scores)
                with self.assertRaisesRegex(FloatingPointError, "selection scores"):
                    defense._high_similarity_cluster_size(scores)

    def test_numpy_predictor_loss_and_gradient_failures_are_explicit(self):
        defense = self.defense()
        parameters = defense._initialize_trainable_parameters(round_id=3)
        gradients = {key: np.zeros_like(value) for key, value in parameters.items()}
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaisesRegex(FloatingPointError, "predictor loss"):
                defense._accumulate_training_gradient(
                    np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32),
                    np.full(2, 1e30, dtype=np.float32), parameters, gradients,
                )
        bad_gradients = {
            key: np.full_like(parameters[key], np.inf)
            for key in ("w1", "b1", "w2", "b2", "w3", "b3")
        }
        with mock.patch.object(defense, "_predict_backward", return_value=(np.zeros(2), bad_gradients)):
            with self.assertRaisesRegex(FloatingPointError, "predictor gradients"):
                defense._accumulate_training_gradient(
                    np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32),
                    np.zeros(2, dtype=np.float32), parameters, gradients,
                )

    def test_torch_predictor_loss_and_gradient_failures_are_explicit(self):
        torch, defense = self.torch_defense()
        defense._global_history = [np.zeros(3, dtype=np.float32)] * 2
        defense._client_history = [
            {"a": np.zeros(3, dtype=np.float32)},
            {"a": np.full(3, 1e30, dtype=np.float32)},
        ]

        def train():
            parameters = defense._initialize_torch_trainable_parameters(round_id=3)
            moment1 = {key: torch.zeros_like(value) for key, value in parameters.items()}
            moment2 = {key: torch.zeros_like(value) for key, value in parameters.items()}
            return defense._train_predictor_torch("a", parameters, moment1, moment2, 0)

        with self.assertRaisesRegex(FloatingPointError, "predictor loss for client a"):
            train()
        defense._client_history[1]["a"] = np.zeros(3, dtype=np.float32)

        def bad_gradients(loss, values, **kwargs):
            return tuple(torch.full_like(value, float("inf")) for value in values)

        with mock.patch.object(torch.autograd, "grad", side_effect=bad_gradients):
            with self.assertRaisesRegex(FloatingPointError, "predictor gradients for client a"):
                train()

    def test_nonfinite_updates_are_rejected_before_bootstrap(self):
        defense = self.defense()
        with self.assertRaisesRegex(ValueError, "VERT update contains"):
            defense.evaluate_round({"a": np.asarray([0., np.nan, 1.])}, round_id=1)


if __name__ == "__main__":
    unittest.main()
