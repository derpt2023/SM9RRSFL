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

    def test_eval_interval_keeps_initial_and_final_records(self):
        dataset = make_synthetic_mnist_like(train_samples=80, test_samples=20, seed=11)
        result = run_experiment(
            dataset,
            ExperimentConfig(
                method="fedavg",
                malicious_ratio=0.0,
                num_clients=4,
                rounds=5,
                local_epochs=1,
                batch_size=16,
                eval_interval=2,
                early_stop=False,
                seed=11,
            ),
        )

        self.assertEqual([record.round for record in result.records], [0, 2, 4, 5])
        self.assertEqual(result.stopped_round, 5)

    def test_sm9rrs_worker_path_keeps_clients_active(self):
        dataset = make_synthetic_mnist_like(train_samples=80, test_samples=20, seed=12)
        result = run_experiment(
            dataset,
            ExperimentConfig(
                method="sm9rrs",
                malicious_ratio=0.0,
                num_clients=4,
                rounds=2,
                local_epochs=1,
                batch_size=16,
                crypto_mode="simulated",
                sm9_workers=2,
                seed=12,
            ),
        )

        self.assertEqual(result.blacklisted_clients, tuple())
        self.assertEqual(result.records[-1].accepted_updates, 4)


if __name__ == "__main__":
    unittest.main()
