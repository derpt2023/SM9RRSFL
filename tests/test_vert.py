import pickle
import unittest

import numpy as np

from sm9rrsfl.vert import VERTDefense


class VERTDefenseTest(unittest.TestCase):
    def test_two_bootstrap_rounds_then_selects_similarity_top_k(self):
        clients = [f"client-{index}" for index in range(4)]
        defense = VERTDefense(
            clients,
            parameter_size=8,
            history_window=3,
            projection_dim=4,
            predict_epochs=1,
            learning_rate=1e-3,
            top_k=1,
            seed=7,
        )
        first_updates = {
            client: np.full(8, index + 1, dtype=np.float32)
            for index, client in enumerate(clients)
        }
        first = defense.evaluate_round(first_updates, round_id=1)
        self.assertEqual(set(first.selected_clients), set(clients))
        defense.finalize_round(
            first_updates,
            np.mean(list(first_updates.values()), axis=0),
            first,
        )

        second_updates = {
            client: update * np.float32(0.9)
            for client, update in first_updates.items()
        }
        second = defense.evaluate_round(second_updates, round_id=2)
        self.assertEqual(set(second.selected_clients), set(clients))
        defense.finalize_round(
            second_updates,
            np.mean(list(second_updates.values()), axis=0),
            second,
        )

        third_updates = dict(second_updates)
        third_updates["client-3"] = -third_updates["client-3"]
        third = defense.evaluate_round(third_updates, round_id=3)
        self.assertEqual(len(third.selected_clients), 1)
        self.assertEqual(len(third.rejected_clients), 3)
        self.assertAlmostEqual(sum(third.weights.values()), 1.0)
        self.assertTrue(all(np.isfinite(score) for score in third.scores.values()))

    def test_state_is_checkpoint_picklable(self):
        defense = VERTDefense(
            ["client-0", "client-1"],
            parameter_size=4,
            projection_dim=2,
            predict_epochs=1,
            seed=5,
        )
        restored = pickle.loads(pickle.dumps(defense))
        result = restored.evaluate_round(
            {
                "client-0": np.ones(4, dtype=np.float32),
                "client-1": np.full(4, 2.0, dtype=np.float32),
            },
            round_id=1,
        )
        self.assertEqual(result.selected_clients, ("client-0", "client-1"))

    def test_default_selection_uses_kmeans_without_ratio(self):
        defense = VERTDefense(
            [f"client-{index}" for index in range(5)],
            parameter_size=4,
            projection_dim=2,
            predict_epochs=1,
            seed=11,
        )

        self.assertEqual(
            defense._effective_top_k([0.95, 0.92, 0.89, 0.20, 0.18]),
            3,
        )
        self.assertEqual(
            defense._effective_top_k([0.75, 0.75, 0.75, 0.75, 0.75]),
            5,
        )
        self.assertEqual(defense._effective_top_k([0.8, 0.1]), 1)
        self.assertIsNone(defense.malicious_ratio_prior)
        self.assertFalse(hasattr(defense, "malicious_ratio"))

    def test_prior_modes_share_scores_and_only_change_selection(self):
        clients = [f"client-{index}" for index in range(5)]
        defenses = [
            VERTDefense(
                clients,
                parameter_size=8,
                history_window=3,
                projection_dim=4,
                predict_epochs=1,
                top_k=1,
                seed=21,
            ),
            VERTDefense(
                clients,
                parameter_size=8,
                history_window=3,
                projection_dim=4,
                predict_epochs=1,
                malicious_ratio_prior=0.2,
                seed=21,
            ),
            VERTDefense(
                clients,
                parameter_size=8,
                history_window=3,
                projection_dim=4,
                predict_epochs=1,
                seed=21,
            ),
        ]
        update_rounds = [
            {
                client: np.linspace(index, index + 1.0, 8, dtype=np.float32)
                for index, client in enumerate(clients)
            },
            {
                client: np.linspace(
                    index + 0.1,
                    index + 1.1,
                    8,
                    dtype=np.float32,
                )
                for index, client in enumerate(clients)
            },
        ]
        for round_id, updates in enumerate(update_rounds, start=1):
            aggregate = np.mean(list(updates.values()), axis=0)
            for defense in defenses:
                result = defense.evaluate_round(updates, round_id=round_id)
                defense.finalize_round(updates, aggregate, result)

        current = dict(update_rounds[-1])
        current["client-4"] = -current["client-4"]
        fixed, ratio_prior, no_prior = [
            defense.evaluate_round(current, round_id=3) for defense in defenses
        ]

        self.assertEqual(fixed.scores, ratio_prior.scores)
        self.assertEqual(fixed.scores, no_prior.scores)
        self.assertEqual(len(fixed.selected_clients), 1)
        self.assertEqual(len(ratio_prior.selected_clients), 3)

    def test_selected_updates_use_uniform_fedavg_weights(self):
        clients = [f"client-{index}" for index in range(4)]
        defense = VERTDefense(
            clients,
            parameter_size=4,
            projection_dim=2,
            predict_epochs=1,
            top_k=2,
            seed=22,
        )
        for round_id in (1, 2):
            updates = {
                client: np.full(4, index + round_id, dtype=np.float32)
                for index, client in enumerate(clients)
            }
            result = defense.evaluate_round(updates, round_id=round_id)
            defense.finalize_round(
                updates,
                np.mean(list(updates.values()), axis=0),
                result,
            )

        result = defense.evaluate_round(updates, round_id=3)
        self.assertEqual(len(result.selected_clients), 2)
        self.assertEqual(set(result.weights.values()), {0.5})

    def test_predictor_is_fresh_each_round_and_has_linear_output(self):
        defense = VERTDefense(
            ["client-0", "client-1"],
            parameter_size=4,
            projection_dim=2,
            predict_epochs=1,
            seed=23,
        )
        first = defense._initialize_trainable_parameters(round_id=3)
        repeated = defense._initialize_trainable_parameters(round_id=3)
        next_round = defense._initialize_trainable_parameters(round_id=4)

        for name in first:
            np.testing.assert_array_equal(first[name], repeated[name])
        self.assertFalse(np.array_equal(first["w1"], next_round["w1"]))
        np.testing.assert_array_equal(first["a"], np.ones(4, dtype=np.float32))
        np.testing.assert_array_equal(first["b"], np.ones(4, dtype=np.float32))

        parameters = {name: np.zeros_like(value) for name, value in first.items()}
        parameters["b3"][:] = np.asarray([-1.0, 2.0], dtype=np.float32)
        output, _cache = defense._predict(
            np.asarray([0.5, -0.5], dtype=np.float32),
            parameters,
        )
        np.testing.assert_array_equal(
            output,
            np.asarray([-1.0, 2.0], dtype=np.float32),
        )

    def test_projector_output_is_raw_linear_feature(self):
        defense = VERTDefense(
            ["client-0", "client-1"],
            parameter_size=4,
            projection_dim=2,
            predict_epochs=1,
            seed=24,
        )
        feature = np.asarray([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)

        np.testing.assert_array_equal(
            defense._project_history_feature(feature),
            defense._linear_project(feature),
        )

    def test_explicit_ratio_prior_restores_legacy_dynamic_top_k(self):
        clients = [f"client-{index}" for index in range(10)]
        defense = VERTDefense(
            clients,
            parameter_size=4,
            projection_dim=2,
            predict_epochs=1,
            malicious_ratio_prior=0.6,
            seed=12,
        )
        scores = [1.0 - 0.01 * index for index in range(10)]

        self.assertEqual(defense._effective_top_k(scores), 3)
        self.assertEqual(defense._effective_top_k(scores[:7]), 2)

        no_attack_prior = VERTDefense(
            clients,
            parameter_size=4,
            projection_dim=2,
            predict_epochs=1,
            malicious_ratio_prior=0.0,
            seed=13,
        )
        self.assertEqual(no_attack_prior._effective_top_k(scores), 10)

    def test_fixed_top_k_and_ratio_prior_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            VERTDefense(
                ["client-0", "client-1"],
                parameter_size=4,
                projection_dim=2,
                predict_epochs=1,
                top_k=1,
                malicious_ratio_prior=0.5,
            )


if __name__ == "__main__":
    unittest.main()
