"""New runtime identity, failure and budget boundaries, without CUDA/training."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import cifar_adaptive_runtime as runtime
from tests.test_cifar_six_pipeline import synthetic_run


class RuntimeFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name).resolve()
        self.manifest = {"fingerprint": "new-study", "source_sha256": {"synthetic": "test"},
                         "spec": {}, "data_contract": {"synthetic": True}}
        runtime.base.write_json(self.output / "manifest.json", self.manifest)
        self.tasks = [self.task(seed) for seed in (41, 42)]
        self.plan = runtime.save_plan(self.output, "block", self.tasks, self.manifest)
        self.args = SimpleNamespace(output=self.output, devices=["cuda:0"],
                                    data_dir=None, stop_at=None, plan=self.plan,
                                    worker=self.tasks[0]["task_id"])

    def task(self, seed):
        config = runtime.base.fl.ExperimentConfig(method="fedavg", seed=seed,
                    num_clients=100, rounds=4, early_stop=False, checkpoint_interval=1,
                    attack="alternating_minimization", malicious_ratio=0.,
                    attack_start_round=4, detector_window=3)
        raw = {"task_id": f"validation_fedavg-adaptive-0000_seed{seed}",
               "phase": "validation", "method": "fedavg",
               "candidate": {"candidate_id": "fedavg-adaptive-0000",
                             "variant": "original", "parameters": {}},
               "config": asdict(config)}
        return runtime.base.attach_fingerprints([raw], self.manifest)[0]

    def complete(self, task):
        folder = runtime.ensure_identity(self.output, task)
        result = synthetic_run(runtime.base.fl.ExperimentConfig(**task["config"]))
        runtime.base.experiments._write_completed_results_snapshot(folder, [result])
        return result

    def failure(self, task, *, kind="infrastructure_or_execution", exception="RuntimeError"):
        folder = runtime.ensure_identity(self.output, task)
        row = {"task_id": task["task_id"], "task_fingerprint": task["fingerprint"],
               "kind": kind, "exception": exception,
               "message": "synthetic failure", "execution_context": {"round": 2}}
        runtime.base.write_json(folder / "failure.json", row)
        return row

    def invoke_execute(self, tasks=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return runtime.execute(self.args, self.tasks if tasks is None else tasks,
                                   self.plan, **kwargs)


class IdentityTests(RuntimeFixture):
    def test_identity_creation_is_idempotent_and_changed_config_rejected(self):
        task = self.tasks[0]
        folder = runtime.ensure_identity(self.output, task)
        identity = (folder / "task.json").read_bytes()
        runtime.ensure_identity(self.output, task)
        self.assertEqual((folder / "task.json").read_bytes(), identity)
        changed = deepcopy(task)
        changed["config"]["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "immutable experiment identity"):
            runtime.ensure_identity(self.output, changed)
        self.assertEqual((folder / "task.json").read_bytes(), identity)

    def test_orphan_artifacts_are_not_relabelled_as_new_task(self):
        task = self.tasks[0]
        folder = self.output / "tasks" / task["task_id"]
        folder.mkdir(parents=True)
        (folder / "checkpoint.pickle").write_bytes(b"preserve")
        with self.assertRaisesRegex(ValueError, "orphan"):
            runtime.ensure_identity(self.output, task)
        self.assertFalse((folder / "task.json").exists())
        self.assertEqual((folder / "checkpoint.pickle").read_bytes(), b"preserve")

    def test_completed_snapshot_requires_exact_scientific_task(self):
        task = self.tasks[0]
        self.complete(task)
        self.assertIsNotNone(runtime.base.checked_completed(self.output, task))
        changed = deepcopy(task)
        changed["config"]["attack_boost"] *= 2
        with self.assertRaisesRegex(ValueError, "immutable task identity"):
            runtime.collect(self.output, [changed])

    def test_completed_snapshot_precedes_historical_failure_and_repairs_csv(self):
        task = self.tasks[0]
        self.complete(task)
        self.failure(task, kind="algorithm_numerical", exception="FloatingPointError")
        folder = self.output / "tasks" / task["task_id"]
        before = (folder / runtime.base.experiments.COMPLETED_RESULTS_SNAPSHOT).read_bytes()
        self.assertFalse((folder / "rounds.csv").exists())
        groups, statuses = runtime.collect(self.output, [task])
        self.assertEqual(statuses[0]["status"], "complete")
        self.assertTrue(statuses[0]["healthy"])
        self.assertEqual(len(groups[task["candidate"]["candidate_id"]]), 1)
        self.assertTrue((folder / "rounds.csv").is_file())
        self.assertEqual((folder / runtime.base.experiments.COMPLETED_RESULTS_SNAPSHOT).read_bytes(), before)
        self.assertTrue((folder / "failure.json").is_file())

    def test_wrong_task_failure_and_same_name_wrong_configuration_rejected(self):
        task = self.tasks[0]
        row = self.failure(task, kind="algorithm_numerical")
        path = self.output / "tasks" / task["task_id"] / "failure.json"
        row["task_id"] = "other-task"
        runtime.base.write_json(path, row)
        with self.assertRaisesRegex(ValueError, "different task"):
            runtime.terminal_failure(self.output, task)
        row["task_id"] = task["task_id"]
        runtime.base.write_json(path, row)
        changed = deepcopy(task)
        changed["config"]["seed"] += 1
        with self.assertRaises(ValueError):
            runtime.terminal_failure(self.output, changed)

    def test_new_source_hashes_keep_every_frozen_legacy_source(self):
        old = runtime.legacy.source_hashes(runtime.REPO)
        self.assertEqual(len(old), 43)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in runtime.NEW_SOURCES:
                (root / name).write_text("synthetic new source " + name)
            with mock.patch.object(runtime, "REPO", root), \
                    mock.patch.object(runtime.legacy, "source_hashes", return_value=old):
                hashes = runtime.source_hashes()
        self.assertEqual({name: hashes[name] for name in old}, old)
        self.assertEqual(set(hashes), set(old) | set(runtime.NEW_SOURCES))

    def test_immutable_plan_cannot_change_after_save(self):
        changed = deepcopy(self.tasks)
        changed[0]["config"]["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "immutable experiment identity"):
            runtime.save_plan(self.output, "block", changed, self.manifest)


class RoundMonitorTests(RuntimeFixture):
    def checkpoint_state(self, current=1, *, params=None, accuracy=.8):
        record = SimpleNamespace(accuracy=accuracy, attack_target_success_rate=.1,
                                 attack_target_confidence=.2, honest_weight_loss=0.,
                                 malicious_weight_mass=0.)
        return {"completed_round": current, "params": np.zeros(2) if params is None else params,
                "records": [record]}

    def fake_run(self, state):
        def execute(*args, **kwargs):
            kwargs["checkpoint_callback"](state)
            return "completed-result"
        return execute

    def test_budget_pause_happens_only_after_checkpoint_and_progress_write(self):
        task = self.tasks[0]
        folder = runtime.ensure_identity(self.output, task)
        checkpoint = folder / "saved-round.txt"
        state = self.checkpoint_state()
        with mock.patch.object(runtime.base.experiments, "run_experiment", side_effect=self.fake_run(state)):
            original = runtime.base.experiments.run_experiment
            with redirect_stdout(io.StringIO()):
                with runtime.round_monitor(task, folder, deadline=1.) as event:
                    with self.assertRaises(runtime.BudgetPause):
                        runtime.base.experiments.run_experiment(checkpoint_callback=lambda s: checkpoint.write_text(str(s["completed_round"])))
            self.assertIs(runtime.base.experiments.run_experiment, original)
        self.assertEqual(checkpoint.read_text(), "1")
        self.assertEqual(runtime.read_json(folder / "progress.json")["last_completed_round"], 1)
        self.assertEqual(event["last_completed_round"], 1)

    def test_last_round_finishes_even_if_deadline_elapsed(self):
        task = self.tasks[0]
        folder = runtime.ensure_identity(self.output, task)
        callback = mock.Mock()
        state = self.checkpoint_state(current=task["config"]["rounds"])
        with mock.patch.object(runtime.base.experiments, "run_experiment", side_effect=self.fake_run(state)), redirect_stdout(io.StringIO()):
            with runtime.round_monitor(task, folder, deadline=1.):
                result = runtime.base.experiments.run_experiment(checkpoint_callback=callback)
        self.assertEqual(result, "completed-result")
        callback.assert_called_once_with(state)

    def test_nonfinite_and_invalid_metrics_do_not_overwrite_good_checkpoint(self):
        task = self.tasks[0]
        folder = runtime.ensure_identity(self.output, task)
        for state in (self.checkpoint_state(params=np.array([np.nan])), self.checkpoint_state(accuracy=1.5)):
            checkpoint = mock.Mock()
            with mock.patch.object(runtime.base.experiments, "run_experiment", side_effect=self.fake_run(state)):
                with runtime.round_monitor(task, folder) as event:
                    with self.assertRaises(FloatingPointError) as caught:
                        runtime.base.experiments.run_experiment(checkpoint_callback=checkpoint)
            checkpoint.assert_not_called()
            self.assertEqual(event["last_completed_round"], 0)
            self.assertEqual(runtime.failure_class(caught.exception, event), "algorithm_numerical")

    def test_checkpoint_io_failure_is_operational_even_during_budget_pause(self):
        task = self.tasks[0]
        folder = runtime.ensure_identity(self.output, task)
        with mock.patch.object(runtime.base.experiments, "run_experiment", side_effect=self.fake_run(self.checkpoint_state())):
            with runtime.round_monitor(task, folder, deadline=1.) as event:
                with self.assertRaises(OSError) as caught:
                    runtime.base.experiments.run_experiment(checkpoint_callback=mock.Mock(side_effect=OSError("disk full")))
        self.assertEqual(runtime.failure_class(caught.exception, event), "infrastructure_or_execution")
        self.assertEqual(event["last_completed_round"], 0)

    def test_signal_saves_checkpoint_then_pauses_and_restores_handlers(self):
        task = self.tasks[0]
        folder = runtime.ensure_identity(self.output, task)
        original = signal.getsignal(signal.SIGTERM)
        checkpoint = mock.Mock()
        with mock.patch.object(runtime.base.experiments, "run_experiment", side_effect=self.fake_run(self.checkpoint_state())), redirect_stdout(io.StringIO()):
            with runtime.round_monitor(task, folder):
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                with self.assertRaises(runtime.BudgetPause):
                    runtime.base.experiments.run_experiment(checkpoint_callback=checkpoint)
        checkpoint.assert_called_once()
        self.assertEqual(signal.getsignal(signal.SIGTERM), original)

    def test_failure_classes_do_not_call_oom_or_setup_error_algorithm_evidence(self):
        self.assertEqual(runtime.failure_class(runtime.BudgetPause("budget"), {}), "budget_or_interrupt")
        self.assertEqual(runtime.failure_class(RuntimeError("CUDA out of memory"), {"round": 3}), "infrastructure_or_execution")
        self.assertEqual(runtime.failure_class(FloatingPointError("setup"), None), "infrastructure_or_execution")
        self.assertEqual(runtime.failure_class(FloatingPointError("bad round"), {"round": 3}), "algorithm_numerical")


class WorkerTests(RuntimeFixture):
    def test_completed_worker_reuses_snapshot_without_model_install_or_data_load(self):
        import cifar_resnet_gn
        task = self.tasks[0]
        self.complete(task)
        with mock.patch.object(runtime, "source_hashes", return_value=self.manifest["source_sha256"]), \
                mock.patch.object(cifar_resnet_gn, "install_runtime") as install, \
                mock.patch.object(runtime.base, "worker_environment") as environment, \
                mock.patch.object(runtime.base, "load_split") as data, redirect_stdout(io.StringIO()):
            self.assertEqual(runtime.worker(self.args), 0)
        install.assert_not_called()
        environment.assert_not_called()
        data.assert_not_called()

    def test_worker_rejects_modified_plan_fingerprint_before_loading_data(self):
        plan = runtime.read_json(self.plan)
        plan["tasks"][0]["config"]["lr"] *= 2
        runtime.base.write_json(self.plan, plan)
        with mock.patch.object(runtime.base, "load_split") as data:
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                runtime.worker(self.args)
        data.assert_not_called()

    def test_worker_source_change_is_operational_failure_before_data_load(self):
        task = self.tasks[0]
        runtime.ensure_identity(self.output, task)
        with mock.patch.object(runtime, "source_hashes", return_value={"wrong": "version"}), \
                mock.patch.object(runtime.base, "load_split") as data:
            with self.assertRaisesRegex(ValueError, "scientific source changed"):
                runtime.worker(self.args)
        data.assert_not_called()
        failure = runtime.terminal_failure(self.output, task)
        self.assertEqual(failure["kind"], "infrastructure_or_execution")


class SchedulerTests(RuntimeFixture):
    def proc(self, pid=8101, code=0):
        return SimpleNamespace(pid=pid, poll=lambda: code)

    def test_all_completed_reuses_snapshots_without_workers(self):
        for task in self.tasks:
            self.complete(task)
        with mock.patch.object(runtime.subprocess, "Popen") as start:
            statuses = self.invoke_execute()
        start.assert_not_called()
        self.assertTrue(all(row["status"] == "complete" for row in statuses))

    def test_deadline_already_elapsed_dispatches_nothing(self):
        with mock.patch.object(runtime.time, "time", return_value=11.), \
                mock.patch.object(runtime.subprocess, "Popen") as start:
            statuses = self.invoke_execute(deadline=10.)
        start.assert_not_called()
        self.assertTrue(all(row["status"] == "pending" for row in statuses))
        self.assertFalse((self.output / "tasks").exists())

    def test_deadline_is_rechecked_between_gpu_dispatches(self):
        self.args.devices = ["cuda:0", "cuda:1"]
        clock = [0.]
        def start(command, **kwargs):
            task = next(t for t in self.tasks if t["task_id"] == command[command.index("--worker") + 1])
            self.complete(task)
            clock[0] = 11.
            return self.proc()
        with mock.patch.object(runtime.time, "time", side_effect=lambda: clock[0]), \
                mock.patch.object(runtime.subprocess, "Popen", side_effect=start) as launched:
            statuses = self.invoke_execute(deadline=10.)
        self.assertEqual(launched.call_count, 1)
        self.assertEqual([row["status"] for row in statuses], ["complete", "pending"])

    def test_algorithm_failure_is_retained_without_retry_and_counts_as_evidence(self):
        for task in self.tasks:
            self.failure(task, kind="algorithm_numerical", exception="FloatingPointError")
        with mock.patch.object(runtime.subprocess, "Popen") as start:
            statuses = self.invoke_execute()
        start.assert_not_called()
        self.assertTrue(runtime.evidence_resolved(statuses))
        self.assertTrue(all(row["status"] == "failed" for row in statuses))

    def test_oom_retries_same_identity_once_then_success_is_reused(self):
        task = self.tasks[0]
        calls = []
        def start(command, **kwargs):
            calls.append(command)
            if len(calls) == 1:
                self.failure(task)
                return self.proc(code=1)
            self.complete(task)
            return self.proc(pid=8102)
        with mock.patch.object(runtime.subprocess, "Popen", side_effect=start):
            statuses = self.invoke_execute([task])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(statuses[0]["status"], "complete")
        self.assertTrue(runtime.evidence_resolved(statuses))
        with mock.patch.object(runtime.subprocess, "Popen") as start_again:
            self.invoke_execute([task])
        start_again.assert_not_called()

    def test_repeated_oom_stays_operational_blocker(self):
        task = self.tasks[0]
        def start(*args, **kwargs):
            self.failure(task)
            return self.proc(code=1)
        with mock.patch.object(runtime.subprocess, "Popen", side_effect=start) as launched:
            statuses = self.invoke_execute([task])
        self.assertEqual(launched.call_count, 2)
        self.assertFalse(runtime.evidence_resolved(statuses))

    def test_budget_pause_is_not_retried_or_counted_as_algorithm_evidence(self):
        task = self.tasks[0]
        def start(*args, **kwargs):
            self.failure(task, kind="budget_or_interrupt", exception="BudgetPause")
            return self.proc(code=75)
        with mock.patch.object(runtime.subprocess, "Popen", side_effect=start) as launched:
            statuses = self.invoke_execute([task])
        self.assertEqual(launched.call_count, 1)
        self.assertFalse(runtime.evidence_resolved(statuses))


if __name__ == "__main__":
    unittest.main()
