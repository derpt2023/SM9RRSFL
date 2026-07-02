import unittest

from sm9rrsfl.datasets import make_synthetic_mnist_like
from sm9rrsfl.fl import ExperimentConfig, run_experiment


class FederatedLoopTest(unittest.TestCase):
    def test_invalid_krum_config_fails_before_training(self):
        dataset = make_synthetic_mnist_like(train_samples=20, test_samples=10, seed=14)
        config = ExperimentConfig(
            method="krum",
            malicious_ratio=0.8,
            num_clients=10,
            rounds=1,
            seed=14,
        )

        with self.assertRaisesRegex(ValueError, r"n=10, f=8, n-f-2=0"):
            run_experiment(dataset, config)

    def test_round_checkpoint_resume_matches_uninterrupted_run(self):
        dataset = make_synthetic_mnist_like(train_samples=80, test_samples=20, seed=13)
        config = ExperimentConfig(
            method="sm9rrs",
            malicious_ratio=0.25,
            num_clients=4,
            rounds=3,
            local_epochs=1,
            batch_size=16,
            attack_start_round=1,
            crypto_mode="simulated",
            early_stop=False,
            seed=13,
        )
        uninterrupted = run_experiment(dataset, config)
        saved = {}

        class SimulatedInterruption(Exception):
            pass

        def stop_after_first_round(state):
            saved["state"] = state
            if state["completed_round"] == 1:
                raise SimulatedInterruption

        with self.assertRaises(SimulatedInterruption):
            run_experiment(dataset, config, checkpoint_callback=stop_after_first_round)
        resumed = run_experiment(dataset, config, resume_state=saved["state"])

        self.assertEqual(
            [record.accuracy for record in resumed.records],
            [record.accuracy for record in uninterrupted.records],
        )
        self.assertEqual(resumed.blacklisted_clients, uninterrupted.blacklisted_clients)

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
        self.assertGreater(result.stage_timings.training_seconds, 0.0)
        self.assertGreater(result.stage_timings.evaluation_seconds, 0.0)
        self.assertGreaterEqual(result.stage_timings.hash_seconds, 0.0)

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
