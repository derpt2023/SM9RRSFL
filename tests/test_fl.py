import unittest
from unittest import mock

import numpy as np

from sm9rrsfl.datasets import (
    make_synthetic_mnist_like,
    stratified_training_three_way_split,
)
from sm9rrsfl.fl import ExperimentConfig, run_experiment
from sm9rrsfl.svd_detector import DetectionResult


def _composite_detection_result(*, anomalous, **overrides):
    values = dict(accepted=not anomalous, reason="strong_novelty" if anomalous else "normal",
                  would_flag=anomalous, count_increment=anomalous,
                  immediate_revocation=anomalous,
                  novelty_score=8.0 if anomalous else 0.0,
                  history_eligible=not anomalous, signed_score=8.0 if anomalous else 0.0)
    values.update(overrides)
    return DetectionResult(**values)


class FederatedLoopTest(unittest.TestCase):
    def test_alternating_attack_uses_training_auxiliary_not_official_test(self):
        from sm9rrsfl import fl as fl_module
        from sm9rrsfl.model import TrainStats, init_params, model_spec_for_dataset

        original = make_synthetic_mnist_like(
            train_samples=400,
            test_samples=80,
            seed=219,
        )
        dataset = stratified_training_three_way_split(
            original,
            seed=991,
        ).main_dataset
        config = ExperimentConfig(
            method="sm9rrs",
            attack="alternating_minimization",
            attack_source_label=5,
            attack_target_label=7,
            attack_target_count=1,
            seed=219,
        )
        target_indices = fl_module._select_attack_target_indices(dataset, config)
        captured = {}

        def fake_attack(global_vector, _x, _y, auxiliary_x, target_labels, **_kwargs):
            captured["auxiliary_x"] = np.asarray(auxiliary_x).copy()
            captured["target_labels"] = np.asarray(target_labels).copy()
            return np.zeros_like(global_vector), TrainStats(loss=0.0, samples=len(_y))

        spec = model_spec_for_dataset(dataset)
        params = init_params(seed=config.seed, spec=spec)
        with mock.patch.object(
            fl_module,
            "alternating_minimization_delta",
            side_effect=fake_attack,
        ):
            fl_module._alternating_minimization_client_delta(
                params,
                dataset,
                np.arange(8, dtype=np.int64),
                attack_target_indices=target_indices,
                client_idx=0,
                round_id=1,
                model_spec=spec,
                config=config,
                torch_context=None,
            )

        np.testing.assert_array_equal(
            captured["auxiliary_x"],
            dataset.x_attack[target_indices],
        )
        self.assertTrue(np.all(captured["target_labels"] == 7))

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
        self.assertEqual(result.records[-1].nonfinite_updates, 1)
        self.assertEqual(result.nonfinite_updates, 1)
        self.assertTrue(np.isfinite(result.final_accuracy))

    def test_vert_torch_backend_runs_on_cpu_when_no_accelerator_is_requested(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        dataset = make_synthetic_mnist_like(train_samples=80, test_samples=20, seed=211)
        config = ExperimentConfig(
            method="vert",
            malicious_ratio=0.0,
            num_clients=4,
            rounds=3,
            local_epochs=1,
            batch_size=16,
            compute_backend="torch",
            device="cpu",
            crypto_mode="simulated",
            early_stop=False,
            seed=211,
        )

        result = run_experiment(dataset, config)

        self.assertEqual(result.records[-1].round, 3)
        self.assertTrue(np.isfinite(result.final_accuracy))

    def test_alignins_torch_backend_runs_on_cpu_when_no_accelerator_is_requested(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")
        dataset = make_synthetic_mnist_like(train_samples=20, test_samples=10, seed=212)
        result = run_experiment(
            dataset,
            ExperimentConfig(
                method="alignins",
                malicious_ratio=0.0,
                num_clients=2,
                rounds=1,
                local_epochs=1,
                batch_size=8,
                compute_backend="torch",
                device="cpu",
                alignins_tda_radius=10.0,
                alignins_mpsa_radius=10.0,
                early_stop=False,
                seed=212,
            ),
        )

        self.assertEqual(result.records[-1].round, 1)
        self.assertEqual(result.records[-1].accepted_updates, 2)
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
        from sm9rrsfl import fl as fl_module

        dataset = make_synthetic_mnist_like(train_samples=80, test_samples=20, seed=13)
        config = ExperimentConfig(
            method="sm9rrs",
            malicious_ratio=0.25,
            num_clients=4,
            rounds=5,
            local_epochs=1,
            batch_size=16,
            attack="sign_flip",
            attack_start_round=0,
            detector_window=3,
            crypto_mode="simulated",
            early_stop=False,
            seed=13,
        )
        with mock.patch.object(
            fl_module,
            "_poison_client_update",
            wraps=fl_module._poison_client_update,
        ) as poison:
            uninterrupted = run_experiment(dataset, config)
        self.assertEqual(poison.call_count, 1)
        saved = {}

        class SimulatedInterruption(Exception):
            pass

        def stop_after_first_round(state):
            saved["state"] = state
            if state["completed_round"] == 1:
                raise SimulatedInterruption

        with self.assertRaises(SimulatedInterruption):
            run_experiment(dataset, config, checkpoint_callback=stop_after_first_round)
        self.assertEqual(saved["state"]["detector"].window_size, 3)
        self.assertEqual(
            {
                state.last_round
                for state in saved["state"]["detector"]._states.values()
            },
            {1},
        )
        resumed = run_experiment(dataset, config, resume_state=saved["state"])

        self.assertEqual(
            [record.accuracy for record in resumed.records],
            [record.accuracy for record in uninterrupted.records],
        )
        self.assertEqual(resumed.blacklisted_clients, uninterrupted.blacklisted_clients)
        self.assertEqual(resumed.diagnostics, uninterrupted.diagnostics)
        self.assertEqual(
            [record.attack_active for record in resumed.records],
            [False, False, False, False, False, True],
        )
        self.assertEqual(resumed.summary_dict()["effective_attack_start_round"], 5)
        malicious = set(resumed.malicious_clients)
        self.assertTrue(malicious)
        for diagnostic in resumed.diagnostics:
            self.assertEqual(
                diagnostic.attack_active,
                diagnostic.client_id in malicious and diagnostic.round >= 5,
            )

    def test_alternating_minimization_runs_inside_local_training(self):
        from sm9rrsfl import fl as fl_module

        dataset = make_synthetic_mnist_like(
            train_samples=400,
            test_samples=100,
            seed=131,
        )
        dataset = stratified_training_three_way_split(dataset, seed=131).main_dataset
        source_label = int(dataset.y_attack[0])
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

    def test_failed_trace_is_retryable_for_both_revocation_paths(self):
        from sm9rrsfl import fl as f
        dataset = make_synthetic_mnist_like(train_samples=60, test_samples=20, seed=121)
        for severe in (False, True):
            with self.subTest(severe=severe):
                config = ExperimentConfig(method="sm9rrs", malicious_ratio=0., num_clients=3,
                                          rounds=1, crypto_mode="simulated",
                                          suspicion_remove_after=5 if severe else 1,
                                          early_stop=False, seed=121)
                decisions = [_composite_detection_result(
                                anomalous=True, accepted=not severe, immediate_revocation=severe,
                                reason="strong_novelty" if severe else "suspicious"),
                             _composite_detection_result(anomalous=False),
                             _composite_detection_result(anomalous=False)]
                checkpoints = []
                with mock.patch.object(f.LongitudinalSVDDetector, "evaluate", side_effect=decisions), \
                     mock.patch.object(f.LongitudinalSVDDetector, "commit", return_value=False), \
                     mock.patch.object(f.ASVerifier, "verify_trace_result", return_value=False), \
                     self.assertRaisesRegex(RuntimeError, "trace remains pending"):
                    run_experiment(dataset, config, checkpoint_callback=checkpoints.append)
                state = checkpoints[-1]
                self.assertEqual(state["records"][-1].rejected_updates, 1)
                self.assertEqual(len(state["weight_manager"].pending_trace), 1)
                self.assertFalse(state["blacklisted"])
                self.assertEqual(len(state["crypto_state"].pending_audits), 1)
                self.assertFalse(state["crypto_state"].pending_audits[0].evidence.model_update.flags.writeable)
                resumed = run_experiment(dataset, config, resume_state=state)
                self.assertEqual(len(resumed.blacklisted_clients), 1)
                self.assertEqual(resumed.records[-1].blacklisted_clients, 1)
                self.assertTrue(resumed.diagnostics[0].revoked)
                self.assertFalse(resumed.diagnostics[0].trace_pending)

    def test_batch_revocation_retries_failed_certificate_then_closes_last_identity(self):
        from sm9rrsfl import fl as f
        dataset = make_synthetic_mnist_like(train_samples=60, test_samples=20, seed=128)
        config = ExperimentConfig(num_clients=3, malicious_ratio=0., rounds=4, eval_interval=3,
                                  suspicion_remove_after=5, crypto_mode="simulated", early_stop=False)
        original = f.ASVerifier.verify_trace_result
        calls = []

        def fail_first(verifier, evidence, trace_result):
            calls.append(evidence)
            return False if len(calls) == 1 else original(verifier, evidence, trace_result)

        checkpoints = []
        with mock.patch.object(f.LongitudinalSVDDetector, "evaluate",
                               return_value=_composite_detection_result(anomalous=True)), \
             mock.patch.object(f.LongitudinalSVDDetector, "commit", return_value=False), \
             mock.patch.object(f.ASVerifier, "verify_trace_result", autospec=True, side_effect=fail_first), \
             self.assertRaisesRegex(RuntimeError, "trace remains pending"):
            run_experiment(dataset, config, checkpoint_callback=checkpoints.append)
        state = checkpoints[-1]
        self.assertEqual(state["completed_round"], 1)
        self.assertEqual(state["records"][-1].round, 1)
        self.assertEqual(state["records"][-1].accepted_updates, 0)
        self.assertEqual(len(state["blacklisted"]), 2)
        self.assertEqual(len(state["crypto_state"].pending_audits), 1)
        resumed = run_experiment(dataset, config, resume_state=state)
        self.assertEqual(resumed.stopped_round, 1)
        self.assertEqual(resumed.records[-1].blacklisted_clients, 3)
        self.assertEqual(resumed.records[-1].false_positive_revocations, 3)
        self.assertTrue(all(d.revoked and not d.trace_pending for d in resumed.diagnostics))

    def test_mass_severe_detection_revokes_all_and_finalizes_without_stale_records(self):
        from sm9rrsfl import fl as f
        dataset = make_synthetic_mnist_like(train_samples=20, test_samples=10, seed=122)
        checkpoints = []
        config = ExperimentConfig(num_clients=2, malicious_ratio=0, rounds=4, crypto_mode="simulated",
                                  suspicion_remove_after=5, eval_interval=3, early_stop=False)
        with mock.patch.object(f.LongitudinalSVDDetector, "evaluate",
                               return_value=_composite_detection_result(anomalous=True)), \
             mock.patch.object(f.LongitudinalSVDDetector, "commit", return_value=False):
            result = run_experiment(dataset, config, checkpoint_callback=checkpoints.append)
        self.assertEqual(len(result.blacklisted_clients), 2)
        self.assertEqual(result.stopped_round, 1)
        self.assertEqual(result.records[-1].accepted_updates, 0)
        self.assertEqual(result.records[-1].false_positive_revocations, 2)
        self.assertTrue(all(d.immediate_revocation and d.revoked and d.count_after == 1.
                            for d in result.diagnostics))
        self.assertIn(dataset.name, checkpoints[-1]["crypto_state"].finalized_task_ids)
        resumed = run_experiment(dataset, config, resume_state=checkpoints[-1])
        self.assertEqual(resumed.records, result.records)
        self.assertEqual(resumed.diagnostics, result.diagnostics)
        self.assertEqual(resumed.blacklisted_clients, result.blacklisted_clients)

    def test_mild_suspicion_is_quarantined_before_ctol(self):
        from sm9rrsfl import fl as f
        dataset = make_synthetic_mnist_like(train_samples=20, test_samples=10, seed=123)
        decision = _composite_detection_result(anomalous=True, accepted=True, immediate_revocation=False,
                                               reason="suspicious")
        with mock.patch.object(f.LongitudinalSVDDetector, "evaluate", return_value=decision), \
             mock.patch.object(f.LongitudinalSVDDetector, "commit", return_value=False):
            result = run_experiment(dataset, ExperimentConfig(
                num_clients=1, malicious_ratio=0, rounds=1, crypto_mode="simulated", early_stop=False))
        d = result.diagnostics[0]
        self.assertFalse(d.aggregation_accepted)
        self.assertFalse(d.revoked)
        self.assertFalse(d.history_admitted)
        self.assertEqual(d.count_after, 1.)
        self.assertEqual(d.aggregation_weight, 0.)
        self.assertAlmostEqual(d.weight_after_penalty_recovery, .5)

    def test_normal_round_halves_suspicion_without_granting_reliability_recovery(self):
        from sm9rrsfl import fl as f
        dataset = make_synthetic_mnist_like(train_samples=20, test_samples=10, seed=126)
        decisions = [_composite_detection_result(anomalous=True, accepted=True, immediate_revocation=False,
                                                reason="suspicious"),
                     _composite_detection_result(anomalous=False, history_eligible=False)]
        with mock.patch.object(f.LongitudinalSVDDetector, "evaluate", side_effect=decisions), \
             mock.patch.object(f.LongitudinalSVDDetector, "commit", return_value=False):
            result = run_experiment(dataset, ExperimentConfig(
                num_clients=1, malicious_ratio=0, rounds=2, crypto_mode="simulated", early_stop=False))
        first, second = result.diagnostics
        self.assertEqual(second.count_before, 1.)
        self.assertEqual(second.count_after, .5)
        self.assertEqual(second.weight_after_penalty_recovery, first.weight_after_penalty_recovery)

    def test_ctol_round_excludes_mild_update_and_revokes_without_extra_delay(self):
        from sm9rrsfl import fl as f
        dataset = make_synthetic_mnist_like(train_samples=40, test_samples=20, seed=127)
        mild = _composite_detection_result(anomalous=True, accepted=True, immediate_revocation=False,
                                          reason="suspicious", novelty_score=2.5)
        normal = _composite_detection_result(anomalous=False)
        decisions = [mild, normal, normal, normal, mild, normal, mild, normal, mild, normal]
        with mock.patch.object(f.LongitudinalSVDDetector, "evaluate", side_effect=decisions), \
             mock.patch.object(f.LongitudinalSVDDetector, "commit", return_value=False):
            result = run_experiment(dataset, ExperimentConfig(
                num_clients=2, malicious_ratio=0, rounds=5, suspicion_remove_after=3,
                crypto_mode="simulated", early_stop=False))
        client = result.diagnostics[0].client_id
        rows = [d for d in result.diagnostics if d.client_id == client]
        self.assertEqual([d.count_after for d in rows], [1., .5, 1.5, 2.5, 3.])
        self.assertTrue(all(not d.revoked for d in rows[:-1]))
        self.assertEqual([d.aggregation_accepted for d in rows], [False, True, False, False, False])
        self.assertTrue(rows[-1].trace_requested)
        self.assertTrue(rows[-1].revoked)
        self.assertFalse(rows[-1].immediate_revocation)
        self.assertEqual(rows[-1].aggregation_weight, 0.)
        self.assertEqual(len(result.blacklisted_clients), 1)

    def test_vert_uses_two_bootstrap_rounds_then_filters(self):
        dataset = make_synthetic_mnist_like(
            train_samples=40,
            test_samples=10,
            seed=124,
        )
        result = run_experiment(
            dataset,
            ExperimentConfig(
                method="vert",
                malicious_ratio=0.5,
                num_clients=4,
                rounds=3,
                local_epochs=1,
                batch_size=8,
                attack="none",
                compute_backend="numpy",
                vert_projection_dim=16,
                vert_predict_epochs=1,
                vert_use_ratio_prior=True,
                early_stop=False,
                seed=124,
            ),
        )

        self.assertEqual(result.records[1].accepted_updates, 4)
        self.assertEqual(result.records[2].accepted_updates, 4)
        self.assertEqual(result.records[3].accepted_updates, 1)
        self.assertEqual(result.records[3].rejected_updates, 3)
        self.assertEqual(result.blacklisted_clients, tuple())

    def test_alignins_numpy_path_is_stateless_and_checkpointable(self):
        dataset = make_synthetic_mnist_like(
            train_samples=40,
            test_samples=10,
            seed=125,
        )
        checkpoints = []
        result = run_experiment(
            dataset,
            ExperimentConfig(
                method="alignins",
                malicious_ratio=0.0,
                num_clients=2,
                rounds=2,
                local_epochs=1,
                batch_size=8,
                attack="none",
                compute_backend="numpy",
                alignins_tda_radius=10.0,
                alignins_mpsa_radius=10.0,
                early_stop=False,
                seed=125,
            ),
            checkpoint_callback=checkpoints.append,
        )

        self.assertEqual(result.records[-1].accepted_updates, 2)
        self.assertEqual(result.records[-1].rejected_updates, 0)
        self.assertNotIn("alignins_defense", checkpoints[-1])
        self.assertTrue(
            all(
                np.isfinite(record.honest_weight_loss)
                and np.isfinite(record.malicious_weight_mass)
                for record in result.records
            )
        )

    def test_alignins_checkpoint_resume_matches_uninterrupted_run(self):
        dataset = make_synthetic_mnist_like(
            train_samples=40,
            test_samples=10,
            seed=127,
        )
        config = ExperimentConfig(
            method="alignins",
            malicious_ratio=0.5,
            num_clients=2,
            rounds=3,
            local_epochs=1,
            batch_size=8,
            attack="sign_flip",
            attack_start_round=1,
            compute_backend="numpy",
            alignins_tda_radius=10.0,
            alignins_mpsa_radius=10.0,
            early_stop=False,
            seed=127,
        )
        uninterrupted = run_experiment(dataset, config)
        saved = {}

        class SimulatedInterruption(Exception):
            pass

        def stop_after_first_round(state):
            if state["completed_round"] == 1:
                saved["state"] = state
                raise SimulatedInterruption

        with self.assertRaises(SimulatedInterruption):
            run_experiment(
                dataset,
                config,
                checkpoint_callback=stop_after_first_round,
            )
        self.assertNotIn("alignins_defense", saved["state"])
        resumed = run_experiment(dataset, config, resume_state=saved["state"])

        self.assertEqual(resumed.records, uninterrupted.records)
        self.assertEqual(resumed.blacklisted_clients, tuple())
        self.assertEqual(resumed.malicious_clients, uninterrupted.malicious_clients)

    def test_weight_diagnostics_use_fedavg_sample_count_baseline(self):
        from sm9rrsfl import fl as fl_module

        honest_loss, malicious_mass = fl_module._aggregation_weight_diagnostics(
            {"honest-small": 10, "honest-large": 30, "malicious": 60},
            {"malicious"},
            {"honest-small": 0.05, "honest-large": 0.25, "malicious": 0.10},
        )
        self.assertAlmostEqual(honest_loss, 0.25)
        self.assertAlmostEqual(malicious_mass, 0.10)

        clean_loss, clean_malicious_mass = (
            fl_module._aggregation_weight_diagnostics(
                {"small": 10, "large": 30, "largest": 60},
                set(),
                {"small": 0.10, "large": 0.30, "largest": 0.60},
            )
        )
        self.assertAlmostEqual(clean_loss, 0.0)
        self.assertAlmostEqual(clean_malicious_mass, 0.0)

    def test_fedavg_records_zero_honest_loss_and_actual_malicious_mass(self):
        dataset = make_synthetic_mnist_like(
            train_samples=20,
            test_samples=10,
            seed=128,
        )
        result = run_experiment(
            dataset,
            ExperimentConfig(
                method="fedavg",
                malicious_ratio=0.5,
                num_clients=2,
                rounds=1,
                local_epochs=1,
                batch_size=8,
                attack="none",
                compute_backend="numpy",
                early_stop=False,
                seed=128,
            ),
        )

        self.assertAlmostEqual(result.records[-1].honest_weight_loss, 0.0)
        self.assertAlmostEqual(result.records[-1].malicious_weight_mass, 0.5)


if __name__ == "__main__":
    unittest.main()
