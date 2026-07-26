import unittest
from unittest import mock

import numpy as np

from sm9rrsfl.datasets import make_synthetic_mnist_like
from sm9rrsfl.fl import ExperimentConfig, run_experiment


class FederatedLoopTest(unittest.TestCase):
    def test_nonfinite_client_update_is_rejected_before_aggregation(self):
        from sm9rrsfl import fl as fl_module

        dataset = make_synthetic_mnist_like(train_samples=80, test_samples=20, seed=20)
        config = ExperimentConfig(
            method="fedavg",
            malicious_ratio=0.0,
            num_clients=4,
            rounds=1,
            local_epochs=1,
            batch_size=16,
            early_stop=False,
            seed=20,
        )
        original = fl_module._local_train_client_delta

        def inject_nan(*args, **kwargs):
            delta, stats = original(*args, **kwargs)
            if kwargs["client_idx"] == 0:
                delta = delta.copy()
                delta[0] = np.nan
            return delta, stats

        with mock.patch.object(
            fl_module,
            "_local_train_client_delta",
            side_effect=inject_nan,
        ):
            result = run_experiment(dataset, config)

        self.assertEqual(result.records[-1].rejected_updates, 1)
        self.assertTrue(np.isfinite(result.final_accuracy))

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
            attack="sign_flip",
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

    def test_alternating_minimization_runs_inside_local_training(self):
        from sm9rrsfl import fl as fl_module

        dataset = make_synthetic_mnist_like(
            train_samples=40,
            test_samples=100,
            seed=131,
        )
        source_label = int(dataset.y_test[0])
        target_label = (source_label + 1) % dataset.num_classes
        config = ExperimentConfig(
            method="fedavg",
            malicious_ratio=0.5,
            num_clients=2,
            rounds=1,
            local_epochs=1,
            batch_size=16,
            lr=0.005,
            attack="alternating_minimization",
            attack_boost=2.0,
            attack_epochs=1,
            attack_stealth_steps=1,
            attack_distance_weight=1e-4,
            attack_source_label=source_label,
            attack_target_label=target_label,
            attack_target_count=1,
            attack_start_round=1,
            early_stop=False,
            seed=131,
        )

        with mock.patch.object(
            fl_module,
            "_poison_client_update",
            side_effect=AssertionError(
                "alternating minimization must not use post-hoc vector poisoning"
            ),
        ):
            result = run_experiment(dataset, config)

        self.assertEqual(result.records[-1].accepted_updates, 2)
        self.assertGreater(result.stage_timings.attack_seconds, 0.0)
        self.assertTrue(np.isfinite(result.final_accuracy))
        self.assertIsNotNone(result.records[-1].attack_target_success_rate)
        self.assertIsNotNone(result.records[-1].attack_target_confidence)
        self.assertGreaterEqual(result.records[-1].attack_target_success_rate, 0.0)
        self.assertLessEqual(result.records[-1].attack_target_success_rate, 1.0)
        self.assertGreaterEqual(result.records[-1].attack_target_confidence, 0.0)
        self.assertLessEqual(result.records[-1].attack_target_confidence, 1.0)

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

    def test_completed_sm9_task_checkpoint_contains_only_finalized_tombstone(self):
        dataset = make_synthetic_mnist_like(
            train_samples=80,
            test_samples=20,
            seed=101,
        )
        checkpoints = []
        run_experiment(
            dataset,
            ExperimentConfig(
                method="sm9rrs",
                malicious_ratio=0.0,
                num_clients=4,
                rounds=1,
                local_epochs=1,
                batch_size=16,
                crypto_mode="simulated",
                early_stop=False,
                seed=101,
            ),
            checkpoint_callback=checkpoints.append,
        )

        crypto_state = checkpoints[-1]["crypto_state"]
        self.assertIn(dataset.name, crypto_state.finalized_task_ids)
        self.assertNotIn(dataset.name, {task.task_id for task in crypto_state.tasks})

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

    def test_failed_trace_is_retryable_and_keeps_ctol_updates_rejected(self):
        from sm9rrsfl import fl as fl_module

        dataset = make_synthetic_mnist_like(train_samples=40, test_samples=20, seed=121)
        config = ExperimentConfig(
            method="sm9rrs",
            malicious_ratio=0.0,
            num_clients=2,
            rounds=1,
            local_epochs=1,
            batch_size=16,
            crypto_mode="simulated",
            suspicion_remove_after=1,
            early_stop=False,
            seed=121,
        )
        checkpoints = []
        with (
            mock.patch.object(
                fl_module.LongitudinalSVDDetector,
                "evaluate",
                return_value=mock.Mock(accepted=False, count_increment=True),
            ),
            mock.patch.object(
                fl_module,
                "_trace_and_archive",
                side_effect=ValueError("temporary trace failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "trace remains pending"),
        ):
            run_experiment(
                dataset,
                config,
                checkpoint_callback=checkpoints.append,
            )

        failed_state = checkpoints[-1]
        self.assertEqual(failed_state["completed_round"], 1)
        self.assertEqual(failed_state["records"][-1].accepted_updates, 0)
        self.assertEqual(failed_state["records"][-1].rejected_updates, 2)
        manager = failed_state["weight_manager"]
        self.assertEqual(len(manager.pending_trace), 2)
        self.assertEqual(sum(manager.weights.values()), 0.0)
        pending = failed_state["crypto_state"].pending_audits
        self.assertEqual(len(pending), 2)
        self.assertTrue(
            all(not item.evidence.model_update.flags.writeable for item in pending)
        )

        resumed_checkpoints = []
        resumed = run_experiment(
            dataset,
            config,
            resume_state=failed_state,
            checkpoint_callback=resumed_checkpoints.append,
        )
        self.assertEqual(len(resumed.blacklisted_clients), 2)
        terminal = resumed_checkpoints[-1]["crypto_state"]
        self.assertIn(dataset.name, terminal.finalized_task_ids)
        self.assertEqual(terminal.pending_audits, ())

    def test_revoking_last_member_closes_task_instead_of_reusing_old_ring(self):
        from sm9rrsfl import fl as fl_module

        dataset = make_synthetic_mnist_like(train_samples=20, test_samples=10, seed=122)
        checkpoints = []
        with mock.patch.object(
            fl_module.LongitudinalSVDDetector,
            "evaluate",
            return_value=mock.Mock(accepted=False, count_increment=True),
        ):
            result = run_experiment(
                dataset,
                ExperimentConfig(
                    method="sm9rrs",
                    malicious_ratio=0.0,
                    num_clients=1,
                    rounds=1,
                    local_epochs=1,
                    batch_size=16,
                    crypto_mode="simulated",
                    suspicion_remove_after=1,
                    early_stop=False,
                    seed=122,
                ),
                checkpoint_callback=checkpoints.append,
            )

        self.assertEqual(result.blacklisted_clients, ("client-0",))
        terminal = checkpoints[-1]["crypto_state"]
        self.assertIn(dataset.name, terminal.finalized_task_ids)
        self.assertNotIn(dataset.name, {task.task_id for task in terminal.tasks})

    def test_single_indicator_suspicion_does_not_trigger_trace(self):
        from sm9rrsfl import fl as fl_module

        dataset = make_synthetic_mnist_like(train_samples=20, test_samples=10, seed=123)
        with mock.patch.object(
            fl_module.LongitudinalSVDDetector,
            "evaluate",
            return_value=mock.Mock(accepted=False, count_increment=False),
        ):
            result = run_experiment(
                dataset,
                ExperimentConfig(
                    method="sm9rrs",
                    malicious_ratio=0.0,
                    num_clients=1,
                    rounds=2,
                    local_epochs=1,
                    batch_size=16,
                    crypto_mode="simulated",
                    suspicion_remove_after=1,
                    early_stop=False,
                    seed=123,
                ),
            )

        self.assertEqual(result.blacklisted_clients, tuple())
        self.assertEqual(result.records[-1].accepted_updates, 1)
        self.assertEqual(result.records[-1].rejected_updates, 0)


if __name__ == "__main__":
    unittest.main()
