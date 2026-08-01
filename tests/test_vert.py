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
            malicious_ratio=0.5,
            history_window=3,
            projection_dim=4,
            predict_epochs=1,
            learning_rate=1e-3,
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
        # ceil((1 - 0.5) * 4) - 1 = 1, matching the paper's automatic
        # expected-honest-minus-one top-k rule.
        self.assertEqual(len(third.selected_clients), 1)
        self.assertEqual(len(third.rejected_clients), 3)
        self.assertAlmostEqual(sum(third.weights.values()), 1.0)
        self.assertTrue(all(np.isfinite(score) for score in third.scores.values()))

    def test_state_is_checkpoint_picklable(self):
        defense = VERTDefense(
            ["client-0", "client-1"],
            parameter_size=4,
            malicious_ratio=0.0,
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


if __name__ == "__main__":
    unittest.main()
