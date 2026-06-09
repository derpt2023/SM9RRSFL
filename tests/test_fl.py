import unittest

from sm9rrsfl.datasets import make_synthetic_mnist_like
from sm9rrsfl.fl import ExperimentConfig, run_experiment


class FederatedLoopTest(unittest.TestCase):
    def test_sm9rrs_zero_malicious_keeps_clients_active(self):
        dataset = make_synthetic_mnist_like(train_samples=80, test_samples=20, seed=10)
        result = run_experiment(
            dataset,
            ExperimentConfig(
                method="sm9rrs",
                malicious_ratio=0.0,
                num_clients=4,
                rounds=3,
                local_epochs=1,
                batch_size=16,
                crypto_mode="simulated",
                seed=10,
            ),
        )

        self.assertEqual(result.blacklisted_clients, tuple())
        self.assertEqual(result.records[-1].accepted_updates, 4)


if __name__ == "__main__":
    unittest.main()
