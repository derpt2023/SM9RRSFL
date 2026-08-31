from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from sm9rrsfl.datasets import ImageDataset, stratified_training_three_way_split
from sm9rrsfl.fl import ExperimentResult, RoundRecord, malicious_client_count
from sm9rrsfl.ours_calibration import (
    OursCalibrationError,
    apply_ours_parameters,
    calibration_metadata,
    resolve_or_run_ours_calibration,
)


def _dataset(*, test_offset: float = 0.0) -> ImageDataset:
    labels = np.repeat(np.arange(3, dtype=np.int64), 40)
    values = np.arange(len(labels), dtype=np.float32).reshape(-1, 1, 1, 1)
    x_test = (
        np.arange(18, dtype=np.float32).reshape(-1, 1, 1, 1) + test_offset
    )
    y_test = np.repeat(np.arange(3, dtype=np.int64), 6)
    return ImageDataset(
        x_train=values,
        y_train=labels,
        x_test=x_test,
        y_test=y_test,
        name="toy",
        input_shape=(1, 1, 1),
        num_classes=3,
    )


def _args(**overrides):
    values = {
        "seed": 42,
        "detector_window": 4,
        "rounds": 6,
        "target_error": 0.1,
        "local_epochs": 1,
        "batch_size": 8,
        "lr": 0.05,
        "lr_decay": 0.99,
        "compute_backend": "numpy",
        "device": "cpu",
        "partitions": ["iid", "dirichlet"],
        "partition": "iid",
        "dirichlet_alpha": 0.5,
        "client_counts": [4],
        "num_clients": 4,
        "ratios": [0.0, 0.5],
        "attack": "sign_flip",
        "attack_scale": 5.0,
        "attack_boost": 10.0,
        "attack_epochs": 1,
        "attack_stealth_steps": 1,
        "attack_distance_weight": 1e-4,
        "attack_source_label": 1,
        "attack_target_label": 2,
        "attack_target_count": 1,
        "attack_start_round": 6,
        "dkg_threshold": 2,
        "dkg_nodes": 3,
        "sm9_workers": 1,
        "calibration_candidate_budget": 12,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeRunner:
    def __init__(self, *, fail_clean: bool = False):
        self.calls = []
        self.fail_clean = fail_clean

    def __call__(self, dataset, config):
        self.calls.append((dataset, config))
        if not config.detector_enforce:
            diagnostics = []
            for client_idx in range(config.num_clients):
                for round_id in range(1, config.rounds + 1):
                    # q=2 has the most stable spectral gap but slightly larger
                    # clean score blocks, allowing the selector to exercise all
                    # data-derived fields rather than returning constants.
                    gap = 0.08 + 0.12 * config.detector_subspace_dim
                    adjacent = (
                        0.30
                        + 0.04 * config.detector_subspace_dim
                        + 0.005 * client_idx
                        + 0.002 * round_id
                    )
                    anchor = adjacent * 0.8
                    diagnostics.append(
                        SimpleNamespace(
                            round=round_id,
                            client_id=f"client-{client_idx}",
                            task_tag=f"tag-{client_idx}",
                            spectral_gap=gap,
                            adjacent_score=adjacent,
                            anchor_score=anchor,
                            suspicious=False,
                            is_malicious=False,
                            aggregation_weight=1.0 / config.num_clients,
                        )
                    )
            records = [
                RoundRecord(
                    method="sm9rrs",
                    malicious_ratio=0.0,
                    round=0,
                    accuracy=0.2,
                    error=0.8,
                    accepted_updates=0,
                    rejected_updates=0,
                    blacklisted_clients=0,
                    true_positive_revocations=0,
                    false_positive_revocations=0,
                    krum_selected_client="",
                )
            ]
            return ExperimentResult(
                config=config,
                records=records,
                diagnostics=diagnostics,
                final_accuracy=0.2,
                final_error=0.8,
                stopped_round=config.rounds,
                malicious_clients=(),
                blacklisted_clients=(),
            )

        malicious_count = malicious_client_count(
            config.num_clients,
            config.malicious_ratio,
        )
        is_clean = malicious_count == 0
        q = config.detector_subspace_dim
        c_tol = config.suspicion_remove_after
        # q=2/C_tol=3 is the strongest attacked validation point.
        quality = 0.92 if (q, c_tol) == (2, 3) else 0.72 - 0.01 * c_tol
        final_accuracy = 0.88 if is_clean else quality
        attack_success = None if is_clean else (0.08 if (q, c_tol) == (2, 3) else 0.55)
        stopped_round = config.rounds
        accepted = config.num_clients
        if self.fail_clean and is_clean:
            stopped_round = config.rounds - 1
            accepted = config.num_clients // 2
        records = []
        for round_id in range(stopped_round + 1):
            initial = round_id == 0
            records.append(
                RoundRecord(
                    method="sm9rrs",
                    malicious_ratio=config.malicious_ratio,
                    round=round_id,
                    accuracy=0.2 if initial else final_accuracy,
                    error=0.8 if initial else 1.0 - final_accuracy,
                    accepted_updates=0 if initial else accepted,
                    rejected_updates=0,
                    blacklisted_clients=0 if is_clean else malicious_count,
                    true_positive_revocations=0 if is_clean else malicious_count,
                    false_positive_revocations=0,
                    krum_selected_client="",
                    attack_target_success_rate=attack_success,
                )
            )
        diagnostics = []
        for client_idx in range(config.num_clients):
            malicious = client_idx < malicious_count
            for round_id in range(1, stopped_round + 1):
                diagnostics.append(
                    SimpleNamespace(
                        round=round_id,
                        client_id=f"client-{client_idx}",
                        task_tag=f"tag-{client_idx}",
                        suspicious=(
                            malicious
                            and round_id >= config.attack_start_round
                            and not is_clean
                        ),
                        is_malicious=malicious,
                        aggregation_weight=1.0 / config.num_clients,
                    )
                )
        return ExperimentResult(
            config=config,
            records=records,
            diagnostics=diagnostics,
            final_accuracy=final_accuracy,
            final_error=1.0 - final_accuracy,
            stopped_round=stopped_round,
            malicious_clients=tuple(
                f"client-{index}" for index in range(malicious_count)
            ),
            blacklisted_clients=tuple(
                f"client-{index}" for index in range(malicious_count)
            ),
        )


class _CleanEnvelopeRunner(_FakeRunner):
    """Model the small-client finite-sample miss from the real smoke run."""

    clean_score_maximum = 1.0

    def __call__(self, dataset, config):
        if not config.detector_enforce or config.malicious_ratio > 0.0:
            return super().__call__(dataset, config)

        self.calls.append((dataset, config))
        unsafe = any(
            threshold <= self.clean_score_maximum
            for threshold in (
                config.detector_adjacent_threshold,
                config.detector_anchor_threshold,
                config.detector_drift_threshold,
            )
        )
        records = [
            RoundRecord(
                method="sm9rrs",
                malicious_ratio=0.0,
                round=round_id,
                accuracy=0.2 if round_id == 0 else 0.88,
                error=0.8 if round_id == 0 else 0.12,
                accepted_updates=0 if round_id == 0 else config.num_clients,
                rejected_updates=0,
                blacklisted_clients=0,
                true_positive_revocations=0,
                false_positive_revocations=0,
                krum_selected_client="",
            )
            for round_id in range(config.rounds + 1)
        ]
        diagnostics = []
        flagged_round = config.detector_window + 1
        for client_idx in range(config.num_clients):
            for round_id in range(1, config.rounds + 1):
                flagged = unsafe and client_idx == 0 and round_id == flagged_round
                if flagged:
                    weight = 0.01
                elif unsafe and round_id == flagged_round:
                    weight = 0.99 / max(1, config.num_clients - 1)
                else:
                    weight = 1.0 / config.num_clients
                score = self.clean_score_maximum if client_idx == 0 else 0.0
                diagnostics.append(
                    SimpleNamespace(
                        round=round_id,
                        client_id=f"client-{client_idx}",
                        task_tag=f"tag-{client_idx}",
                        suspicious=flagged,
                        is_malicious=False,
                        aggregation_weight=weight,
                        adjacent_score=score,
                        anchor_score=score,
                        cumulative_drift=score,
                    )
                )
        return ExperimentResult(
            config=config,
            records=records,
            diagnostics=diagnostics,
            final_accuracy=0.88,
            final_error=0.12,
            stopped_round=config.rounds,
            malicious_clients=(),
            blacklisted_clients=(),
        )


class _RevocationExhaustionRunner(_CleanEnvelopeRunner):
    """Stop early exactly as the real task does after revoking every signer."""

    def __init__(self, *, nonfinite: bool = False):
        super().__init__()
        self.nonfinite = nonfinite
        self.exhausted_clean_calls = 0

    def __call__(self, dataset, config):
        if not config.detector_enforce or config.malicious_ratio > 0.0:
            return super().__call__(dataset, config)
        unsafe = any(
            threshold <= self.clean_score_maximum
            for threshold in (
                config.detector_adjacent_threshold,
                config.detector_anchor_threshold,
                config.detector_drift_threshold,
            )
        )
        if not unsafe:
            return super().__call__(dataset, config)

        self.calls.append((dataset, config))
        self.exhausted_clean_calls += 1
        stopped_round = config.detector_window + 1
        records = []
        for round_id in range(stopped_round + 1):
            initial = round_id == 0
            exhausted = round_id == stopped_round
            records.append(
                RoundRecord(
                    method="sm9rrs",
                    malicious_ratio=0.0,
                    round=round_id,
                    accuracy=0.2 if initial else 0.4,
                    error=0.8 if initial else 0.6,
                    accepted_updates=(
                        0 if initial or exhausted else config.num_clients
                    ),
                    rejected_updates=config.num_clients if exhausted else 0,
                    blacklisted_clients=config.num_clients if exhausted else 0,
                    true_positive_revocations=0,
                    false_positive_revocations=(
                        config.num_clients if exhausted else 0
                    ),
                    krum_selected_client="",
                    nonfinite_updates=int(self.nonfinite and exhausted),
                )
            )
        diagnostics = []
        for client_idx in range(config.num_clients):
            for round_id in range(1, stopped_round + 1):
                exhausted = round_id == stopped_round
                score = self.clean_score_maximum if exhausted else 0.0
                diagnostics.append(
                    SimpleNamespace(
                        round=round_id,
                        client_id=f"client-{client_idx}",
                        task_tag=f"tag-{client_idx}",
                        suspicious=exhausted,
                        count_increment=exhausted,
                        revoked=exhausted,
                        is_malicious=False,
                        aggregation_weight=(
                            0.0 if exhausted else 1.0 / config.num_clients
                        ),
                        adjacent_score=score,
                        anchor_score=score,
                        cumulative_drift=score,
                    )
                )
        return ExperimentResult(
            config=config,
            records=records,
            diagnostics=diagnostics,
            final_accuracy=0.4,
            final_error=0.6,
            stopped_round=stopped_round,
            malicious_clients=(),
            blacklisted_clients=tuple(
                f"client-{client_idx}" for client_idx in range(config.num_clients)
            ),
            nonfinite_updates=int(self.nonfinite),
        )


class OursCalibrationTest(unittest.TestCase):
    def test_three_way_split_is_stratified_disjoint_and_test_safe(self):
        dataset = _dataset()
        split = stratified_training_three_way_split(dataset, seed=91)

        train = set(split.train_indices.tolist())
        calibration = set(split.calibration_indices.tolist())
        attack = set(split.attack_indices.tolist())
        self.assertFalse(train & calibration)
        self.assertFalse(train & attack)
        self.assertFalse(calibration & attack)
        self.assertEqual(train | calibration | attack, set(range(120)))
        self.assertEqual(len(train), 108)
        self.assertEqual(len(calibration), 6)
        self.assertEqual(len(attack), 6)
        self.assertIs(split.main_dataset.x_test, dataset.x_test)
        np.testing.assert_array_equal(
            split.main_dataset.x_attack,
            split.calibration_dataset.x_attack,
        )
        for label in range(3):
            self.assertEqual(np.count_nonzero(split.calibration_dataset.y_test == label), 2)
            self.assertEqual(np.count_nonzero(split.main_dataset.y_attack == label), 2)

    def test_calibration_derives_freezes_writes_and_reuses_parameters(self):
        args = _args()
        runner = _FakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            main_dataset, artifact = resolve_or_run_ours_calibration(
                _dataset(),
                args,
                tmp,
                run_fn=runner,
            )
            first_call_count = len(runner.calls)
            self.assertGreater(first_call_count, 0)
            self.assertEqual(len(main_dataset.y_train), 108)
            self.assertEqual(len(main_dataset.y_attack), 6)
            self.assertEqual(artifact.status, "complete")
            self.assertEqual(len(artifact.calibration_seeds), 2)
            self.assertEqual(len(set(artifact.calibration_seeds)), 2)
            self.assertNotIn(args.seed, artifact.calibration_seeds)
            self.assertEqual(artifact.selected_parameters.q, 2)
            self.assertEqual(artifact.selected_parameters.C_tol, 3)
            self.assertEqual(artifact.selected_parameters.C_max, 3)
            self.assertAlmostEqual(artifact.selected_parameters.penalty_factor, 0.1)
            self.assertAlmostEqual(
                artifact.selected_parameters.recovery_factor,
                0.1 ** -0.5,
            )
            self.assertGreater(artifact.selected_parameters.g0, 0.0)
            self.assertGreater(artifact.selected_parameters.theta_adj, 0.0)
            self.assertGreater(artifact.selected_parameters.theta_anc, 0.0)
            self.assertGreater(artifact.selected_parameters.h, 0.0)
            canonical = (
                Path(tmp)
                / ".ours_calibration"
                / artifact.calibration_fingerprint
                / "ours_parameters.json"
            )
            self.assertTrue(canonical.exists())
            self.assertTrue((Path(tmp) / "ours_calibration.json").exists())

            # Official test contents are not selection input.  A matching
            # training protocol therefore reuses the exact artifact without
            # invoking the runner, while preserving the new official test for
            # the main evaluation dataset.
            cached_dataset, cached = resolve_or_run_ours_calibration(
                _dataset(test_offset=10_000.0),
                args,
                tmp,
                run_fn=runner,
            )
            self.assertEqual(len(runner.calls), first_call_count)
            self.assertEqual(cached.artifact_fingerprint, artifact.artifact_fingerprint)
            self.assertGreater(float(cached_dataset.x_test.min()), 9_000.0)

    def test_apply_and_manifest_metadata_use_only_semantic_artifact_fields(self):
        args = _args()
        with tempfile.TemporaryDirectory() as tmp:
            _main_dataset, artifact = resolve_or_run_ours_calibration(
                _dataset(),
                args,
                tmp,
                run_fn=_FakeRunner(),
            )
        target = SimpleNamespace()
        self.assertIs(apply_ours_parameters(target, artifact), target)
        parameters = artifact.selected_parameters
        self.assertEqual(target.detector_subspace_dim, parameters.q)
        self.assertEqual(target.detector_gap_threshold, parameters.g0)
        self.assertEqual(target.suspicion_remove_after, parameters.C_tol)
        self.assertEqual(target.suspicion_count_max, parameters.C_max)
        self.assertEqual(target.detector_decision_rule, "any")

        metadata = calibration_metadata(artifact)
        self.assertEqual(
            metadata["artifact_fingerprint"],
            artifact.artifact_fingerprint,
        )
        self.assertNotIn("created_at_utc", metadata)
        self.assertNotIn("path", metadata)
        self.assertNotIn("runtime_seconds", metadata)

    def test_public_protocol_change_invalidates_cache(self):
        runner = _FakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            _main, first = resolve_or_run_ours_calibration(
                _dataset(),
                _args(lr=0.05),
                tmp,
                run_fn=runner,
            )
            first_count = len(runner.calls)
            _main, second = resolve_or_run_ours_calibration(
                _dataset(),
                _args(lr=0.025),
                tmp,
                run_fn=runner,
            )
        self.assertGreater(len(runner.calls), first_count)
        self.assertNotEqual(
            first.calibration_fingerprint,
            second.calibration_fingerprint,
        )

    def test_runtime_device_index_does_not_invalidate_numpy_calibration(self):
        runner = _FakeRunner()
        with tempfile.TemporaryDirectory() as tmp:
            _main, first = resolve_or_run_ours_calibration(
                _dataset(),
                _args(compute_backend="numpy", device="cuda:0"),
                tmp,
                run_fn=runner,
            )
            first_count = len(runner.calls)
            _main, second = resolve_or_run_ours_calibration(
                _dataset(),
                _args(compute_backend="numpy", device="cuda:1"),
                tmp,
                run_fn=runner,
            )

        self.assertEqual(len(runner.calls), first_count)
        self.assertEqual(
            first.calibration_fingerprint,
            second.calibration_fingerprint,
        )

    def test_corrupt_partial_artifact_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            _main, first = resolve_or_run_ours_calibration(
                _dataset(),
                _args(),
                tmp,
                run_fn=_FakeRunner(),
            )
            canonical = (
                Path(tmp)
                / ".ours_calibration"
                / first.calibration_fingerprint
                / "ours_parameters.json"
            )
            canonical.write_text('{"status":"partial"}', encoding="utf-8")
            rerun = _FakeRunner()
            _main, repaired = resolve_or_run_ours_calibration(
                _dataset(),
                _args(),
                tmp,
                run_fn=rerun,
            )

        self.assertGreater(len(rerun.calls), 0)
        self.assertEqual(repaired.status, "complete")
        self.assertEqual(
            repaired.artifact_fingerprint,
            first.artifact_fingerprint,
        )

    def test_calibration_fails_closed_when_clean_hard_gate_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(OursCalibrationError, "failed closed"):
                resolve_or_run_ours_calibration(
                    _dataset(),
                    _args(),
                    tmp,
                    run_fn=_FakeRunner(fail_clean=True),
                )
            self.assertFalse((Path(tmp) / "ours_calibration.json").exists())

    def test_scheme_b_defers_attacked_candidate_selection_to_unified_tuner(self):
        runner = _FakeRunner()
        args = _args(
            ours_calibration_selection_mode="defer_to_unified_tuner",
        )
        with tempfile.TemporaryDirectory() as tmp:
            _main, artifact = resolve_or_run_ours_calibration(
                _dataset(),
                args,
                tmp,
                run_fn=runner,
            )

        self.assertFalse(
            any(
                config.detector_enforce and config.malicious_ratio > 0.0
                for _dataset_arg, config in runner.calls
            )
        )
        self.assertEqual(
            artifact.objective_learning["status"],
            "deferred_to_unified_fair_tuner",
        )
        self.assertFalse(artifact.constraints["attacked_metrics_evaluated"])
        self.assertEqual(
            artifact.selected_candidate["selection_scope"],
            "provisional_clean_safe_base_for_unified_tuner",
        )

    def test_closed_loop_clean_envelope_repairs_small_client_discretization(self):
        runner = _CleanEnvelopeRunner()
        args = _args(
            detector_window=7,
            rounds=12,
            attack_start_round=9,
            partitions=["iid"],
            client_counts=[10],
            num_clients=10,
            ratios=[0.0, 0.2],
        )
        with tempfile.TemporaryDirectory() as tmp:
            _main, artifact = resolve_or_run_ours_calibration(
                _dataset(),
                args,
                tmp,
                run_fn=runner,
            )

        selected = artifact.selected_candidate
        trace = selected["clean_safety_trace"]
        self.assertGreaterEqual(len(trace), 2)
        self.assertFalse(trace[0]["valid"])
        self.assertIn("clean_suspicious_rate", trace[0]["invalid_reasons"])
        self.assertEqual(trace[0]["worst_clean_round_suspicious_rate"], 0.1)
        self.assertTrue(trace[-1]["valid"])
        self.assertEqual(selected["worst_clean_round_suspicious_rate"], 0.0)
        self.assertGreaterEqual(selected["worst_clean_round_ess_ratio"], 0.90)
        self.assertEqual(
            artifact.constraints["max_clean_round_suspicious_rate"],
            0.05,
        )
        self.assertEqual(artifact.constraints["min_clean_round_ess_ratio"], 0.90)
        self.assertGreater(artifact.selected_parameters.theta_adj, 1.0)
        self.assertGreater(artifact.selected_parameters.theta_anc, 1.0)
        self.assertGreater(artifact.selected_parameters.h, 1.0)

        attacked_configs = [
            config
            for _dataset_arg, config in runner.calls
            if config.detector_enforce and config.malicious_ratio > 0.0
        ]
        self.assertTrue(attacked_configs)
        self.assertTrue(
            all(
                config.detector_adjacent_threshold > 1.0
                and config.detector_anchor_threshold > 1.0
                and config.detector_drift_threshold > 1.0
                for config in attacked_configs
            )
        )

    def test_clean_false_revocation_task_exhaustion_is_refined(self):
        runner = _RevocationExhaustionRunner()
        args = _args(
            detector_window=7,
            rounds=12,
            attack_start_round=9,
            partitions=["iid"],
            client_counts=[10],
            num_clients=10,
            ratios=[0.0, 0.2],
        )
        with tempfile.TemporaryDirectory() as tmp:
            _main, artifact = resolve_or_run_ours_calibration(
                _dataset(),
                args,
                tmp,
                run_fn=runner,
            )

        self.assertGreater(runner.exhausted_clean_calls, 0)
        trace = artifact.selected_candidate["clean_safety_trace"]
        self.assertIn("incomplete_rounds", trace[0]["invalid_reasons"])
        self.assertIn(
            "clean_false_positive_rate",
            trace[0]["invalid_reasons"],
        )
        self.assertTrue(trace[-1]["valid"])
        self.assertEqual(artifact.selected_candidate["result_count"], 2)

    def test_nonfinite_task_exhaustion_remains_structural_failure(self):
        runner = _RevocationExhaustionRunner(nonfinite=True)
        args = _args(
            detector_window=7,
            rounds=12,
            attack_start_round=9,
            partitions=["iid"],
            client_counts=[10],
            num_clients=10,
            ratios=[0.0, 0.2],
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(OursCalibrationError, "nonfinite_updates"):
                resolve_or_run_ours_calibration(
                    _dataset(),
                    args,
                    tmp,
                    run_fn=runner,
                )

        clean_calls = [
            config
            for _dataset_arg, config in runner.calls
            if config.detector_enforce and config.malicious_ratio == 0.0
        ]
        self.assertEqual(len(clean_calls), args.calibration_candidate_budget)
        self.assertFalse(
            any(
                config.detector_enforce and config.malicious_ratio > 0.0
                for _dataset_arg, config in runner.calls
            )
        )


if __name__ == "__main__":
    unittest.main()
