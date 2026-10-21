"""Small fixtures exercise durability/metric semantics, never long training."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from sm9rrsfl import fl
from sm9rrsfl.datasets import ImageDataset
from scripts.experiments.cifar10 import attack_stability_worker as worker


def dataset():
    # 750 disjoint holdout source samples, one selected optimizer target.
    return ImageDataset(
        x_train=np.zeros((20, 1, 8, 8), np.float32),
        y_train=np.arange(20) % 10,
        x_test=np.zeros((750, 1, 8, 8), np.float32),
        y_test=np.full(750, 5),
        x_attack=np.zeros((5, 1, 8, 8), np.float32),
        y_attack=np.full(5, 5),
        name="mnist", num_classes=10)


def record(config, round_id):
    return fl.RoundRecord(
        method=config.method, malicious_ratio=config.malicious_ratio,
        round=round_id, accuracy=.6, error=.4, accepted_updates=4,
        rejected_updates=0, blacklisted_clients=0,
        true_positive_revocations=0, false_positive_revocations=0,
        krum_selected_client="", attack_target_success_rate=0.,
        attack_target_confidence=.1)


def state(config, round_id, records):
    return {"completed_round": round_id, "params": np.zeros(5, np.float32),
            "records": records, "nonfinite_updates": 0}


def fake_training(_dataset, config, *, resume_state=None, checkpoint_callback=None):
    rows = list(resume_state["records"]) if resume_state is not None else []
    start = resume_state["completed_round"] + 1 if resume_state is not None else 0
    for r in range(start, config.rounds + 1):
        rows.append(record(config, r))
        checkpoint_callback(state(config, r, list(rows)))
    return fl.ExperimentResult(
        config=config, records=rows, final_accuracy=.6, final_error=.4,
        stopped_round=config.rounds, malicious_clients=(), blacklisted_clients=())


class AttackStabilityWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = fl.ExperimentConfig(
            method="fedavg", rounds=3, num_clients=4, malicious_ratio=0.,
            attack_start_round=2, detector_window=1, early_stop=False,
            compute_backend="numpy", device="cpu", attack_target_count=1)
        self.request = {
            "output_dir": self.temp.name, "task_id": "clean", "fingerprint": "source-v1",
            "config": asdict(self.config), "profile": None, "device": "cpu",
            "deadline_epoch": float("inf"),
            "data_config": {"dataset": "cifar10", "data_dir": "unused",
                            "train_samples": 50000, "test_samples": 10000, "seed": 42,
                            "split_seed": 20261013, "validation_fraction": .15}}
        self.data_patch = mock.patch.object(
            worker, "_load_dataset", return_value=(dataset(), {"test_source": "train-only"}))
        self.metrics_patch = mock.patch.object(
            worker.RoundMetricEvaluator, "_target_metrics",
            side_effect=lambda vector, group, indices: (
                (1., .9) if group == "attack" else (.2, .3)))
        self.data_patch.start()
        self.metrics_patch.start()
        self.addCleanup(self.data_patch.stop)
        self.addCleanup(self.metrics_patch.stop)

    def path(self, filename):
        return Path(self.temp.name) / "clean" / filename

    def test_three_metric_denominators_remain_distinct_in_clean_control(self):
        evaluator = worker.RoundMetricEvaluator(dataset(), self.config)
        row = evaluator.evaluate(state(self.config, 0, [record(self.config, 0)]))
        self.assertEqual(row["legacy_target_count"], 1)
        self.assertEqual(row["training_target_count"], 1)
        self.assertEqual(row["source_transfer_count"], 750)
        self.assertEqual((row["legacy_asr"], row["training_target_success"],
                          row["source_transfer_asr"]), (0., 1., .2))
        manifest = evaluator.target_manifest()
        self.assertFalse(manifest["official_test_used_for_selection"])
        self.assertEqual(len(manifest["training_target_indices"]), 1)

    def test_task_fingerprint_excludes_only_runtime_placement_and_budget(self):
        identity = worker.task_identity(self.request)
        moved = {**self.request, "device": "mps", "deadline_epoch": 1,
                 "config": {**self.request["config"], "device": "mps"}}
        self.assertEqual(identity, worker.task_identity(moved))
        for changed in (
                {**self.request, "profile": {"id": "changed"}},
                {**self.request, "fingerprint": "new-source"},
                {**self.request, "config": {**self.request["config"], "lr": .001}},
                {**self.request, "data_config": {**self.request["data_config"], "split_seed": 2}}):
            self.assertNotEqual(identity["fingerprint"],
                                worker.task_identity(changed)["fingerprint"])

    def test_complete_task_is_idempotent_and_does_not_retrain(self):
        with mock.patch.object(fl, "run_experiment", side_effect=fake_training) as train:
            first = worker.execute_task(self.request)
            second = worker.execute_task(self.request)
        self.assertEqual(first, second)
        self.assertEqual(train.call_count, 1)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["completed_round"], 3)
        self.assertTrue(self.path("anchor_r0.npz").exists())
        self.assertEqual(set(np.load(self.path("anchor_r0.npz")).files), {"params"})
        with self.path("result.pickle").open("rb") as handle:
            self.assertIsInstance(pickle.load(handle), fl.ExperimentResult)

    def test_profile_change_cannot_overwrite_existing_artifacts(self):
        with mock.patch.object(fl, "run_experiment", side_effect=fake_training):
            worker.execute_task(self.request)
        original = self.path("status.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "identity differs"):
            worker.execute_task({**self.request, "profile": {"id": "another"}})
        self.assertEqual(self.path("status.json").read_bytes(), original)

    def test_budget_stop_keeps_last_completed_state_and_resumes(self):
        def interrupt(_dataset, config, **kwargs):
            callback = kwargs["checkpoint_callback"]
            callback(state(config, 0, [record(config, 0)]))
            self.request["deadline_epoch"] = 0
            callback(state(config, 1, [record(config, 0), record(config, 1)]))
            self.fail("deadline must interrupt after durable round one")
        with mock.patch.object(fl, "run_experiment", side_effect=interrupt):
            partial = worker.execute_task(self.request)
        self.assertEqual((partial["status"], partial["completed_round"]), ("partial", 1))
        self.assertFalse(self.path("result.pickle").exists())
        self.request["deadline_epoch"] = float("inf")
        with mock.patch.object(fl, "run_experiment", side_effect=fake_training) as train:
            final = worker.execute_task(self.request)
        self.assertEqual(final["status"], "complete")
        self.assertEqual(train.call_args.kwargs["resume_state"]["completed_round"], 1)
        rows = json.loads(self.path("metrics.json").read_text())["rounds"]
        self.assertEqual([r["round"] for r in rows], [0, 1, 2, 3])

    def test_resumed_elapsed_time_includes_prior_attempt_and_marks_unclean_resume(self):
        self.request["deadline_epoch"] = 0
        worker.execute_task(self.request)
        prior = json.loads(self.path("status.json").read_text())
        prior.update(status="running", elapsed_seconds=123.)
        self.path("status.json").write_text(json.dumps(prior))
        self.request["deadline_epoch"] = float("inf")
        with mock.patch.object(fl, "run_experiment", side_effect=fake_training):
            complete = worker.execute_task(self.request)
        self.assertTrue(complete["timing_has_unclean_resume"])
        self.assertEqual(complete["elapsed_before_seconds"], 123.)
        self.assertGreaterEqual(complete["elapsed_seconds"], 123.)
        self.assertGreaterEqual(complete["summary"]["runtime_seconds"], 123.)
        self.assertAlmostEqual(complete["elapsed_seconds"],
                               123. + complete["attempt_elapsed_seconds"], places=3)

    def test_honest_nonfinite_delta_stops_before_aggregation_and_restores_hooks(self):
        original_context = fl._maybe_torch_context
        original_attack = fl._alternating_minimization_client_delta

        def train(_dataset, config, **kwargs):
            kwargs["checkpoint_callback"](state(config, 0, [record(config, 0)]))
            fl._local_train_client_delta(None, None, None, client_idx=2, round_id=1)
            self.fail("nonfinite delta must stop the worker")

        with mock.patch.object(fl, "_local_train_client_delta", return_value=(
                np.array([np.nan]), SimpleNamespace(loss=0.1))):
            mocked_local = fl._local_train_client_delta
            with mock.patch.object(fl, "run_experiment", side_effect=train):
                outcome = worker.execute_task(self.request)
            self.assertIs(fl._local_train_client_delta, mocked_local)
        self.assertIs(fl._maybe_torch_context, original_context)
        self.assertIs(fl._alternating_minimization_client_delta, original_attack)
        self.assertEqual((outcome["status"], outcome["completed_round"]),
                         ("numerical_failure", 0))
        failure = json.loads(self.path("first_failure.json").read_text())
        self.assertEqual(failure["audit"]["stage"], "local_train_delta")
        self.assertEqual(failure["audit"]["client_id"], "client-2")
        self.assertFalse(self.path("result.pickle").exists())
        with mock.patch.object(fl, "run_experiment") as rerun:
            self.assertEqual(worker.execute_task(self.request), outcome)
            rerun.assert_not_called()

    def test_nonfinite_round_parameters_never_replace_last_good_checkpoint(self):
        def train(_dataset, config, **kwargs):
            callback = kwargs["checkpoint_callback"]
            callback(state(config, 0, [record(config, 0)]))
            bad = state(config, 1, [record(config, 0), record(config, 1)])
            bad["params"][0] = np.inf
            callback(bad)
        with mock.patch.object(fl, "run_experiment", side_effect=train):
            outcome = worker.execute_task(self.request)
        self.assertEqual(outcome["status"], "numerical_failure")
        with self.path("checkpoint.pickle").open("rb") as handle:
            saved = pickle.load(handle)
        self.assertEqual(saved["completed_round"], 0)
        self.assertTrue(np.isfinite(saved["params"]).all())

    def test_expired_budget_does_not_load_data_or_start_training(self):
        self.request["deadline_epoch"] = 0
        with mock.patch.object(worker, "_load_dataset") as loader:
            outcome = worker.execute_task(self.request)
        self.assertEqual(outcome["status"], "partial")
        loader.assert_not_called()

    def test_path_escape_and_unidentified_artifacts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "single safe"):
            worker.execute_task({**self.request, "task_id": "../escape"})
        self.path("stale.pickle").parent.mkdir()
        self.path("stale.pickle").write_bytes(b"unidentified")
        with self.assertRaisesRegex(ValueError, "unidentified"):
            worker.execute_task(self.request)

    def test_real_torch_training_and_extended_metrics_on_small_fixture(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional in the unit-test environment")
        self.metrics_patch.stop()
        self.request["config"].update(compute_backend="torch", rounds=1,
                                      attack_start_round=5, detector_window=3, local_epochs=1,
                                      batch_size=10, lr=.001)
        self.config = fl.ExperimentConfig(**self.request["config"])
        outcome = worker.execute_task(self.request)
        self.assertEqual(outcome["status"], "complete", outcome)
        self.assertEqual(outcome["completed_round"], 1)
        row = outcome["final_metrics"]
        self.assertEqual(row["source_transfer_count"], 750)
        for key in ("training_target_success", "training_target_confidence",
                    "source_transfer_asr", "source_transfer_confidence"):
            self.assertTrue(0. <= row[key] <= 1.)
        self.assertTrue(Path(outcome["checkpoint"]).is_file())

    def test_real_attack_audit_carries_round_and_client_and_scope_restores(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional in the unit-test environment")
        from sm9rrsfl.torch_backend import TorchTrainingContext
        self.metrics_patch.stop()
        original = TorchTrainingContext.alternating_minimization_delta_resident
        self.request["config"].update(
            compute_backend="torch", rounds=5, attack_start_round=5, detector_window=3,
            local_epochs=1, attack_epochs=1, batch_size=10, lr=.001,
            malicious_ratio=.5)
        self.request["profile"] = {
            "id": "toy-sgd", "optimizer": "sgd", "lr": None,
            "boost": 1., "stealth_steps": 1, "scope": "all"}
        outcome = worker.execute_task(self.request)
        self.assertEqual(outcome["status"], "complete", outcome)
        audits = [json.loads(line) for line in self.path("audit.jsonl").read_text().splitlines()]
        self.assertEqual(len(audits), 2)
        self.assertEqual({a["round"] for a in audits}, {5})
        self.assertTrue(all(a["client_id"] == f"client-{a['client_idx']}" for a in audits))
        self.assertTrue(all(a["status"] == "complete" for a in audits))
        self.assertIs(TorchTrainingContext.alternating_minimization_delta_resident, original)

    def test_round24_anchor_has_only_public_model_parameters(self):
        self.request["config"]["rounds"] = 25
        with mock.patch.object(fl, "run_experiment", side_effect=fake_training):
            result = worker.execute_task(self.request)
        self.assertEqual(result["status"], "complete")
        with np.load(self.path("anchor_r24.npz")) as saved:
            self.assertEqual(saved.files, ["params"])
            self.assertTrue(np.isfinite(saved["params"]).all())

    def test_anchor_write_failure_does_not_commit_new_checkpoint(self):
        self.request["config"]["rounds"] = 25
        original_atomic = worker._atomic
        checkpoint_before_failure = []

        def fail_anchor(path, writer):
            if Path(path).name == "anchor_r24.npz":
                checkpoint_before_failure.append(self.path("checkpoint.pickle").read_bytes())
                raise OSError("injected anchor write failure")
            return original_atomic(path, writer)

        with mock.patch.object(worker, "_atomic", side_effect=fail_anchor), \
                mock.patch.object(fl, "run_experiment", side_effect=fake_training):
            result = worker.execute_task(self.request)
        self.assertEqual((result["status"], result["completed_round"]), ("failed", 23))
        self.assertEqual(len(checkpoint_before_failure), 1)
        self.assertEqual(self.path("checkpoint.pickle").read_bytes(),
                         checkpoint_before_failure[0])
        with self.path("checkpoint.pickle").open("rb") as handle:
            checkpoint = pickle.load(handle)
        self.assertEqual(checkpoint["completed_round"], 23)
        self.assertFalse(self.path("anchor_r24.npz").exists())
        self.assertFalse(self.path("result.pickle").exists())
        failure = json.loads(self.path("first_failure.json").read_text())
        self.assertEqual(failure["last_completed_round"], 23)
        self.assertEqual(failure["error_type"], "OSError")
        # An extra metric row is explicitly recoverable; it is not committed
        # scientific state until its checkpoint and required anchor exist.
        rows = json.loads(self.path("metrics.json").read_text())["rounds"]
        self.assertEqual(rows[-1]["round"], 24)

    def test_keyboard_interrupt_keeps_complete_checkpoint_and_no_result(self):
        def train(_dataset, config, **kwargs):
            kwargs["checkpoint_callback"](state(config, 0, [record(config, 0)]))
            raise KeyboardInterrupt()
        with mock.patch.object(fl, "run_experiment", side_effect=train):
            result = worker.execute_task(self.request)
        self.assertEqual((result["status"], result["completed_round"]), ("partial", 0))
        self.assertEqual(result["error_type"], "KeyboardInterrupt")
        self.assertFalse(self.path("result.pickle").exists())


if __name__ == "__main__":
    unittest.main()
