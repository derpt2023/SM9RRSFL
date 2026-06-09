import unittest

import numpy as np

from sm9rrsfl.aggregation import fedavg, krum, weighted_fedavg


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


if __name__ == "__main__":
    unittest.main()
