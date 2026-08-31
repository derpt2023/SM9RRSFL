import inspect
import unittest

import numpy as np

from sm9rrsfl.alignins import (
    AlignInsDefense,
    aggregate_with_coefficients,
)


class AlignInsDefenseTest(unittest.TestCase):
    def test_tda_mpsa_and_population_z_score_follow_algorithm_one(self):
        defense = AlignInsDefense(
            alignins_sparsity=0.5,
            alignins_tda_radius=10.0,
            alignins_mpsa_radius=10.0,
        )
        updates = {
            "alice": np.array([2.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "bob": np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32),
            "carol": np.array([-2.0, -1.0, 0.0, 0.0], dtype=np.float32),
        }

        result = defense.evaluate_round(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            updates,
        )

        expected_tda = np.array(
            [2.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0), -2.0 / np.sqrt(5.0)]
        )
        expected_mpsa = np.array([1.0, 1.0, 0.0])
        expected_tda_mz = np.abs(
            (expected_tda - np.median(expected_tda)) / np.std(expected_tda)
        )
        expected_mpsa_mz = np.abs(
            (expected_mpsa - np.median(expected_mpsa)) / np.std(expected_mpsa)
        )
        np.testing.assert_allclose(
            list(result.tda_scores.values()), expected_tda, rtol=1e-6
        )
        np.testing.assert_allclose(
            list(result.mpsa_scores.values()), expected_mpsa, rtol=1e-6
        )
        np.testing.assert_allclose(
            list(result.tda_mz_scores.values()), expected_tda_mz, rtol=1e-6
        )
        np.testing.assert_allclose(
            list(result.mpsa_mz_scores.values()), expected_mpsa_mz, rtol=1e-6
        )

    def test_zero_population_standard_deviation_keeps_equal_scores(self):
        defense = AlignInsDefense(
            alignins_sparsity=1.0,
            alignins_tda_radius=1.0,
            alignins_mpsa_radius=1.0,
        )
        update = np.array([1.0, 2.0], dtype=np.float32)
        result = defense.evaluate_round(
            np.array([2.0, 1.0], dtype=np.float32),
            {"a": update, "b": update.copy()},
        )

        self.assertEqual(result.selected_clients, ("a", "b"))
        self.assertEqual(set(result.tda_mz_scores.values()), {0.0})
        self.assertEqual(set(result.mpsa_mz_scores.values()), {0.0})

    def test_filtering_and_clipped_coefficients_are_not_renormalized(self):
        defense = AlignInsDefense(
            alignins_sparsity=1.0,
            alignins_tda_radius=1.0,
            alignins_mpsa_radius=1.0,
        )
        updates = {
            "small": np.array([1.0, 0.0], dtype=np.float32),
            "large": np.array([3.0, 0.0], dtype=np.float32),
            "opposite": np.array([-1.0, 0.0], dtype=np.float32),
        }
        # Wide radii first isolate clipping semantics independently of the
        # exact outlier pattern in this deliberately tiny scenario.
        wide = AlignInsDefense(
            alignins_sparsity=1.0,
            alignins_tda_radius=10.0,
            alignins_mpsa_radius=10.0,
        ).evaluate_round(np.array([1.0, 0.0], dtype=np.float32), updates)

        self.assertAlmostEqual(wide.clip_norm, 1.0)
        self.assertAlmostEqual(wide.clip_factors["small"], 1.0)
        self.assertAlmostEqual(wide.clip_factors["large"], 1.0 / 3.0)
        self.assertAlmostEqual(wide.clip_factors["opposite"], 1.0)
        self.assertAlmostEqual(wide.aggregation_coefficients["small"], 1.0 / 3.0)
        self.assertAlmostEqual(wide.aggregation_coefficients["large"], 1.0 / 9.0)
        self.assertAlmostEqual(wide.aggregation_coefficients["opposite"], 1.0 / 3.0)
        self.assertAlmostEqual(
            sum(wide.aggregation_coefficients.values()),
            7.0 / 9.0,
        )
        aggregate = aggregate_with_coefficients(
            updates,
            wide.aggregation_coefficients,
        )
        np.testing.assert_allclose(aggregate, np.array([1.0 / 3.0, 0.0]))

        filtered = defense.evaluate_round(
            np.array([1.0, 0.0], dtype=np.float32),
            updates,
        )
        self.assertTrue(filtered.rejected_clients)
        self.assertTrue(
            all(
                filtered.aggregation_coefficients[client_id] == 0.0
                for client_id in filtered.rejected_clients
            )
        )

    def test_zero_norm_update_never_divides_by_zero(self):
        defense = AlignInsDefense(
            alignins_sparsity=1.0,
            alignins_tda_radius=10.0,
            alignins_mpsa_radius=10.0,
        )
        updates = {
            "zero": np.zeros(2, dtype=np.float32),
            "normal": np.array([2.0, 0.0], dtype=np.float32),
        }
        result = defense.evaluate_round(
            np.array([1.0, 0.0], dtype=np.float32),
            updates,
        )

        self.assertEqual(result.clip_factors["zero"], 1.0)
        self.assertTrue(
            all(np.isfinite(value) for value in result.aggregation_coefficients.values())
        )
        aggregate = aggregate_with_coefficients(
            updates,
            result.aggregation_coefficients,
        )
        np.testing.assert_allclose(
            aggregate,
            np.array([0.5, 0.0], dtype=np.float32),
        )

    def test_empty_intersection_produces_a_zero_update(self):
        defense = AlignInsDefense(
            alignins_sparsity=1.0,
            alignins_tda_radius=0.5,
            alignins_mpsa_radius=0.5,
        )
        updates = {
            "positive": np.array([1.0, 1.0], dtype=np.float32),
            "negative": np.array([-1.0, -1.0], dtype=np.float32),
        }
        result = defense.evaluate_round(
            np.array([1.0, 1.0], dtype=np.float32),
            updates,
        )

        self.assertEqual(result.selected_clients, tuple())
        self.assertEqual(result.rejected_clients, ("positive", "negative"))
        self.assertEqual(set(result.aggregation_coefficients.values()), {0.0})
        np.testing.assert_array_equal(
            aggregate_with_coefficients(
                updates,
                result.aggregation_coefficients,
            ),
            np.zeros(2, dtype=np.float32),
        )

    def test_client_names_and_mapping_order_do_not_supply_attack_truth(self):
        defense = AlignInsDefense(
            alignins_sparsity=0.5,
            alignins_tda_radius=0.8,
            alignins_mpsa_radius=0.8,
        )
        global_params = np.array([1.0, 1.0, 0.5, -0.5], dtype=np.float32)
        vectors = [
            np.array([1.0, 0.8, 0.2, -0.1], dtype=np.float32),
            np.array([0.9, 1.1, 0.1, -0.2], dtype=np.float32),
            np.array([-1.0, -1.0, 0.0, 0.2], dtype=np.float32),
        ]
        first = defense.evaluate_round(
            global_params,
            {"malicious-looking-name": vectors[0], "x": vectors[1], "y": vectors[2]},
        )
        second = defense.evaluate_round(
            global_params,
            {"renamed-0": vectors[0], "renamed-1": vectors[1], "renamed-2": vectors[2]},
        )

        self.assertEqual(
            tuple(first.tda_scores.values()), tuple(second.tda_scores.values())
        )
        self.assertEqual(
            tuple(first.mpsa_scores.values()), tuple(second.mpsa_scores.values())
        )
        self.assertEqual(
            tuple(first.aggregation_coefficients.values()),
            tuple(second.aggregation_coefficients.values()),
        )
        parameter_names = inspect.signature(
            AlignInsDefense.evaluate_round
        ).parameters
        self.assertNotIn("malicious_clients", parameter_names)
        self.assertNotIn("malicious_ratio", parameter_names)

    def test_parameter_and_input_validation(self):
        with self.assertRaises(ValueError):
            AlignInsDefense(alignins_sparsity=0.0)
        with self.assertRaises(ValueError):
            AlignInsDefense(alignins_tda_radius=-1.0)
        with self.assertRaises(ValueError):
            AlignInsDefense(alignins_mpsa_radius=0.0)
        defense = AlignInsDefense()
        with self.assertRaises(ValueError):
            defense.evaluate_round(np.ones(2), {})
        with self.assertRaises(ValueError):
            defense.evaluate_round(
                np.ones(2),
                {"a": np.ones(2), "b": np.ones(3)},
            )

    def test_torch_path_matches_numpy_and_stays_on_device(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        defense = AlignInsDefense(
            alignins_sparsity=0.5,
            alignins_tda_radius=10.0,
            alignins_mpsa_radius=10.0,
        )
        global_numpy = np.array([1.0, 0.5, -0.5, 0.25], dtype=np.float32)
        numpy_updates = {
            "a": np.array([1.0, 0.4, -0.2, 0.1], dtype=np.float32),
            "b": np.array([2.0, 1.0, -1.0, 0.5], dtype=np.float32),
            "c": np.array([-0.5, 0.1, 0.2, -0.1], dtype=np.float32),
        }
        torch_updates = {
            client_id: torch.tensor(update)
            for client_id, update in numpy_updates.items()
        }
        numpy_result = defense.evaluate_round(global_numpy, numpy_updates)
        torch_result = defense.evaluate_round(
            torch.tensor(global_numpy),
            torch_updates,
        )

        np.testing.assert_allclose(
            list(torch_result.tda_scores.values()),
            list(numpy_result.tda_scores.values()),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            list(torch_result.mpsa_scores.values()),
            list(numpy_result.mpsa_scores.values()),
        )
        np.testing.assert_allclose(
            list(torch_result.aggregation_coefficients.values()),
            list(numpy_result.aggregation_coefficients.values()),
            rtol=1e-5,
            atol=1e-6,
        )
        aggregate = aggregate_with_coefficients(
            torch_updates,
            torch_result.aggregation_coefficients,
        )
        self.assertTrue(torch.is_tensor(aggregate))
        self.assertEqual(aggregate.device, torch_updates["a"].device)
        expected = aggregate_with_coefficients(
            numpy_updates,
            numpy_result.aggregation_coefficients,
        )
        np.testing.assert_allclose(aggregate.cpu().numpy(), expected, rtol=1e-5)
        with self.assertRaises(TypeError):
            defense.evaluate_round(
                global_numpy,
                {"numpy": numpy_updates["a"], "torch": torch_updates["b"]},
            )
        with self.assertRaises(ValueError):
            defense.evaluate_round(
                torch.tensor(global_numpy),
                {
                    "float32": torch_updates["a"],
                    "float64": torch_updates["b"].to(dtype=torch.float64),
                },
            )


if __name__ == "__main__":
    unittest.main()
