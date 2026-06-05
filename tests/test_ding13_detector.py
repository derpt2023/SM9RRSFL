import unittest

import numpy as np

from sm9rrsfl.ding13_detector import Ding13TrajectoryDetector
from sm9rrsfl.model import NUM_CLASSES, parameter_size


class Ding13DetectorTest(unittest.TestCase):
    def test_removes_consecutive_svd_outlier(self):
        clients = [f"client-{idx}" for idx in range(5)]
        detector = Ding13TrajectoryDetector(
            clients,
            contamination=0.2,
            remove_after=2,
            n_trees=80,
            sample_size=5,
            max_depth=4,
            seed=4,
        )
        malicious = {"client-4"}

        def update_for(client_idx, scale):
            update = np.zeros(parameter_size(), dtype=np.float32)
            offset = client_idx % NUM_CLASSES
            update[offset::NUM_CLASSES] = scale
            return update

        detector.evaluate_round(
            {client: update_for(idx, 0.01) for idx, client in enumerate(clients)},
            malicious,
            round_id=1,
        )
        first = detector.evaluate_round(
            {
                client: update_for(idx, 0.011 if client != "client-4" else 8.0)
                for idx, client in enumerate(clients)
            },
            malicious,
            round_id=2,
        )
        self.assertIn("client-4", first.outliers)

        second = detector.evaluate_round(
            {
                client: update_for(idx, 0.012 if client != "client-4" else 0.01)
                for idx, client in enumerate(clients)
            },
            malicious,
            round_id=3,
        )
        self.assertIn("client-4", second.newly_removed)
        self.assertEqual(second.true_positive_removed, 1)


if __name__ == "__main__":
    unittest.main()
