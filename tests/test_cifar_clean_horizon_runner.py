"""Bounded same-GPU serial dispatch and retained fresh diagnostic attempts."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
import signal
import tempfile
from pathlib import Path
from dataclasses import asdict
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import run_cifar_clean_horizon_panel as runner
import cifar_mechanism_runtime as observation
import cifar_deterministic_prefix_runtime as policy_observation
import cifar_cnn_history_runtime as history_observation
import cifar_clean_horizon_report as report

protocol, base, runtime = runner.protocol, runner.base, runner.runtime
GPU = "GPU-01234567-89ab-cdef-0123-456789abcdef"


class CleanHorizonRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.mechanism_output = self.root / "old-mechanism"
        self.output = self.root / "horizon80"
        (self.output / "task_plans").mkdir(parents=True)
        self.metadata = {"requested_device": "cuda:0", "environment": {"CUDA_VISIBLE_DEVICES": GPU},
            "actual_compute_device": {"name": "Synthetic 4090D", "uuid": GPU, "logical_device": "cuda:0"},
            "torch": {"version": "synthetic", "logical_cuda_devices": [
                {"logical_index": 0, "name": "Synthetic 4090D", "uuid": GPU}]},
            "nvidia": {"gpus": [{"uuid": GPU, "name": "Synthetic 4090D", "driver_version": "synthetic"}]}}
        self.prepared = SimpleNamespace(metadata=self.metadata)
        self.reference = {"mechanism_output": str(self.mechanism_output.resolve()),
            "execution_environment": protocol.prefix_report.matched.normalized_environment(self.metadata)}
        self.manifest = {"fingerprint": "runner-fixture", "same_gpu_uuid": GPU,
            "source_sha256": {"synthetic.py": "0" * 64}, "reference": self.reference,
            "data_contract": {"fixture": True}, "spec": {"fixture": True}}
        config = base.fl.ExperimentConfig(method="sm9rrs", malicious_ratio=0., num_clients=100,
            rounds=80, partition="dirichlet", seed=2026093001, compute_backend="torch", device="cuda:0",
            crypto_mode="simulated", detector_window=20, attack_start_round=25, checkpoint_interval=1,
            early_stop=False, batch_size=50, local_epochs=1, sm9_workers=1)
        self.tasks = []
        for repeat in (1, 2):
            for arm in ("H0", "H1"):
                task = {"task_id": f"horizon80_{arm}_repeat{repeat}", "repeat": repeat, "arm": arm,
                    "candidate": {"candidate_id": arm, "variant": "original" if arm == "H0" else "Ours-FrozenHistory-v1"},
                    "history_freeze_start_round": None if arm == "H0" else 25, "config": asdict(config),
                    "policy": "singleton_backward_cudnn_deterministic", "same_gpu_uuid": GPU}
                task["fingerprint"] = base.digest(task)
                self.tasks.append(task)
                folder = self.output / "tasks" / task["task_id"]
                folder.mkdir(parents=True)
                base.write_json(folder / "task.json", task)
        base.write_json(self.output / "manifest.json", self.manifest)
        base.write_json(self.output / "task_plans/clean_horizon.json", {
            "manifest_fingerprint": self.manifest["fingerprint"], "tasks": self.tasks})
        isolation = ExitStack()
        self.addCleanup(isolation.close)
        isolation.enter_context(mock.patch.object(protocol, "read_study", return_value=(self.manifest, self.tasks)))
        isolation.enter_context(mock.patch.object(protocol, "verify_reference"))
        isolation.enter_context(mock.patch.object(protocol, "source_hashes", return_value=self.manifest["source_sha256"]))
        self.args = SimpleNamespace(output=self.output, mechanism_output=self.mechanism_output, data_dir=None,
            worker=self.tasks[0]["task_id"], retry_failed=False, wait_seconds=600., poll_seconds=10.)
        self.gpu = {"index": 3, "uuid": GPU, "name": "Synthetic 4090D", "free_mib": 24000., "utilization": 0.}

    def readonly_guards(self):
        stack = ExitStack()
        for obj, name in ((base, "load_split"), (base, "write_json"), (base, "immutable_json"),
                          (base.fl, "run_experiment"), (base.experiments, "_load_round_checkpoint")):
            stack.enter_context(mock.patch.object(obj, name, side_effect=AssertionError("readonly path: " + name)))
        return stack

    def complete(self, task=None):
        task = task or self.tasks[0]
        folder = self.output / "tasks" / task["task_id"]
        (folder / "attempts/synthetic").mkdir(parents=True, exist_ok=True)
        value = protocol.seal_artifact({"status": "complete", "task_fingerprint": task["fingerprint"],
            "gpu_uuid": GPU, "environment": deepcopy(self.prepared.metadata), "observations": {},
            "wall_seconds": .1, "attempt": "synthetic", "fresh_start": True, "checkpoints_used": False,
            "completed_training_rounds": 80, "requested_training_rounds": 80})
        base.write_json(folder / "completed.json", value)
        return value

    def fake_worker(self, error=None, metadata=None, stopped_round=80):
        stack = ExitStack()
        stack.enter_context(mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}))
        stack.enter_context(mock.patch.object(base, "worker_environment", return_value=metadata or self.prepared.metadata))
        stack.enter_context(mock.patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))}))
        split = SimpleNamespace(calibration_dataset=object())
        stack.enter_context(mock.patch.object(base, "load_split", return_value=(split, self.manifest["data_contract"])))
        stack.enter_context(mock.patch.object(base.experiments, "_load_round_checkpoint", side_effect=AssertionError("old checkpoint")))
        observer = mock.MagicMock()
        observer.__enter__.return_value = observer
        observer.finish.return_value = {"synthetic": "control-boundary-only"}
        stack.enter_context(mock.patch.object(observation, "observe", return_value=observer))
        stack.enter_context(mock.patch.object(policy_observation, "source_evidence", return_value={"status": "synthetic"}))
        train = stack.enter_context(mock.patch.object(base.fl, "run_experiment", side_effect=error, return_value=SimpleNamespace(stopped_round=stopped_round)))
        return stack, train, observer, split

    def test_plan_only_is_readonly_without_reference_GPU_or_data(self):
        target, capture = self.output.parent / "not-created", io.StringIO()
        with self.readonly_guards(), mock.patch.object(runner.original, "gpu_inventory", side_effect=AssertionError("GPU")), \
                mock.patch.object(protocol, "audit_reference", side_effect=AssertionError("reference")), redirect_stdout(capture):
            self.assertEqual(runner.main(["--plan-only", "--output", str(target)]), 0)
        data = json.loads(capture.getvalue())
        self.assertEqual((data["fresh_processes"], data["clients_per_round"], data["maximum_client_training_calls"]), (4, 100, 32000))
        self.assertEqual((data["rounds_per_process"], data["maximum_training_rounds"], data["maximum_scoped_events_per_process"]), (80, 320, 160))
        self.assertTrue(data["numerical_policy_modified"])
        self.assertFalse(target.exists())

    def test_pending_summary_never_queries_GPU_loads_data_or_writes(self):
        before = protocol.evidence_hashes(self.output)
        with self.readonly_guards(), mock.patch.object(runner.original, "gpu_inventory", side_effect=AssertionError("GPU")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(["--summary", "--output", str(self.output)]), 2)
        self.assertEqual(before, protocol.evidence_hashes(self.output))
        self.assertFalse((self.output / "controller_runs").exists())

    def test_four_interleaved_fresh_processes_wait_on_same_UUID_then_run_strictly_serially(self):
        owner, starts, exits, active, events = self, [], [], set(), []
        by_id = {t["task_id"]: t for t in self.tasks}
        class Process:
            def __init__(self, command, **kwargs):
                owner.assertFalse(active)
                tid = command[command.index("--worker") + 1]
                owner.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], GPU)
                owner.assertTrue(kwargs["start_new_session"])
                owner.assertEqual(kwargs["cwd"], runner.REPO)
                owner.assertEqual(command[2], str(runner.REPO / "run_cifar_clean_horizon_panel.py"))
                owner.assertNotIn("--retry-failed", command)
                starts.append(tid)
                active.add(tid)
                self.tid, self.returncode, self.pid, self.polls = tid, None, 1000 + len(starts), 0
            def poll(self):
                self.polls += 1
                if self.polls == 1:
                    return None
                if self.returncode is None:
                    owner.complete(by_id[self.tid])
                    active.remove(self.tid)
                    exits.append(self.tid)
                    self.returncode = 0
                return self.returncode
        busy = {**self.gpu, "utilization": 70.}
        # Dispatch isolation: strict scientific validation has separate failure regressions below.
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}), \
                mock.patch.object(runner, "validate_artifact", side_effect=lambda output, task, manifest, artifact: artifact), \
                mock.patch.object(runner.original, "gpu_inventory", side_effect=[[busy], [self.gpu]] * 4) as inventory, \
                mock.patch.object(runner.subprocess, "Popen", side_effect=Process) as spawn, \
                mock.patch.object(runner.time, "sleep"), \
                runner.admission.waiting_gate(GPU, wait_seconds=600., poll_seconds=10., emit=events.append):
            self.assertTrue(runner.execute(self.args, self.manifest, self.tasks))
        self.assertEqual(starts, [t["task_id"] for t in self.tasks])
        self.assertEqual(exits, starts)
        self.assertEqual([(t["candidate"]["variant"], t["repeat"]) for t in self.tasks],
            [(v, r) for r in (1, 2) for v in ("original", "Ours-FrozenHistory-v1")])
        self.assertEqual((spawn.call_count, inventory.call_count), (4, 8))
        self.assertEqual([e["event"] for e in events], ["gpu_wait", "gpu_ready"] * 4)

    def test_gate_timeout_or_visibility_mismatch_never_launches_worker(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}), \
                mock.patch.object(runner.original, "gpu_inventory", return_value=[{**self.gpu, "free_mib": 1.}]), \
                mock.patch.object(runner.subprocess, "Popen") as spawn, \
                runner.admission.waiting_gate(GPU, wait_seconds=0., poll_seconds=10., emit=lambda _: None), \
                self.assertRaises(runner.admission.GPUWaitTimeout):
            runner.execute(self.args, self.manifest, self.tasks)
        spawn.assert_not_called()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-other"}), \
                mock.patch.object(runner.original, "gpu_inventory", return_value=[self.gpu]), \
                mock.patch.object(runner.subprocess, "Popen") as spawn, \
                runner.admission.waiting_gate(GPU, emit=lambda _: None), self.assertRaisesRegex(ValueError, "excluded"):
            runner.execute(self.args, self.manifest, self.tasks)
        spawn.assert_not_called()

    def test_completed_tasks_reuse_without_GPU_or_subprocess(self):
        for task in self.tasks:
            self.complete(task)
        with mock.patch.object(runner, "validate_artifact", side_effect=lambda output, task, manifest, artifact: artifact), \
                mock.patch.object(runner.original, "gpu_inventory", side_effect=AssertionError("GPU reuse")), \
                mock.patch.object(runner.subprocess, "Popen", side_effect=AssertionError("retraining")):
            self.assertTrue(runner.execute(self.args, self.manifest, self.tasks))

    def test_sealed_invalid_observations_block_reuse_before_GPU_or_launch(self):
        artifact = self.complete()
        self.assertEqual(protocol.load_completed(self.output, self.tasks[0]), artifact)
        with mock.patch.object(runner.original, "gpu_inventory") as inventory, \
                mock.patch.object(runner.subprocess, "Popen") as spawn, \
                self.assertRaisesRegex(ValueError, "observation.*schema|schema.*observation"):
            runner.execute(self.args, self.manifest, self.tasks)
        inventory.assert_not_called()
        spawn.assert_not_called()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}), \
                mock.patch.object(base, "worker_environment") as environment, \
                self.assertRaisesRegex(ValueError, "observation.*schema|schema.*observation"):
            runner.worker(self.args)
        environment.assert_not_called()

    def test_exit0_with_semantically_bad_completion_blocks_the_next_task(self):
        def finish_bad(*_args, **_kwargs):
            self.complete(self.tasks[0])
            return SimpleNamespace(returncode=0, poll=lambda: 0)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}), \
                mock.patch.object(runner.original, "gpu_inventory", return_value=[self.gpu]) as inventory, \
                mock.patch.object(runner.subprocess, "Popen", side_effect=finish_bad) as spawn, \
                self.assertRaisesRegex(ValueError, "observation.*schema|schema.*observation"):
            runner.execute(self.args, self.manifest, self.tasks)
        self.assertEqual((spawn.call_count, inventory.call_count), (1, 1))
        self.assertFalse((self.output / "tasks" / self.tasks[1]["task_id"] / "worker.log").exists())

    def test_resealed_nonfresh_or_checkpoint_artifact_blocks_reuse_before_dispatch(self):
        for field, value in (("fresh_start", False), ("checkpoints_used", True)):
            artifact = self.complete()
            artifact.pop("artifact_fingerprint")
            artifact[field] = value
            base.write_json(self.output / "tasks" / self.tasks[0]["task_id"] / "completed.json",
                            protocol.seal_artifact(artifact))
            with self.subTest(field=field), \
                    mock.patch.object(runner.original, "gpu_inventory") as inventory, \
                    mock.patch.object(runner.subprocess, "Popen") as spawn, \
                    self.assertRaisesRegex(ValueError, "not a fresh task"):
                runner.execute(self.args, self.manifest, self.tasks)
            inventory.assert_not_called()
            spawn.assert_not_called()

    def test_failed_process_and_exit0_missing_artifact_stop_without_retry(self):
        for code in (75, 0):
            process = SimpleNamespace(returncode=code, poll=lambda: code)
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}), \
                    mock.patch.object(runner.original, "gpu_inventory", return_value=[self.gpu]), \
                    mock.patch.object(runner.subprocess, "Popen", return_value=process) as spawn:
                self.assertFalse(runner.execute(self.args, self.manifest, self.tasks))
            self.assertEqual(spawn.call_count, 1)
        self.assertFalse((self.output / "tasks" / self.tasks[1]["task_id"] / "worker.log").exists())

    def test_explicit_retry_retains_previous_attempt_and_starts_without_checkpoint(self):
        folder = self.output / "tasks" / self.args.worker
        prior = folder / "attempts/previous/failure.json"
        prior.parent.mkdir(parents=True)
        base.write_json(prior, {"error": "retained"})
        original = prior.read_bytes()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}), self.readonly_guards(), \
                self.assertRaisesRegex(ValueError, "unsuccessful attempt"):
            runner.worker(self.args)
        self.args.retry_failed = True
        context, train, observer, split = self.fake_worker()
        with context, mock.patch.object(runner, "validate_artifact", side_effect=lambda output, task, manifest, artifact: artifact):
            self.assertEqual(runner.worker(self.args), 0)
        self.assertEqual(prior.read_bytes(), original)
        self.assertEqual(len(list((folder / "attempts").iterdir())), 2)
        self.assertIs(train.call_args.args[0], split.calibration_dataset)
        self.assertEqual(train.call_args.args[1].rounds, 80)
        self.assertEqual(asdict(train.call_args.args[1]), self.tasks[0]["config"])
        self.assertEqual(set(train.call_args.kwargs), {"checkpoint_callback"})
        callback = train.call_args.kwargs["checkpoint_callback"]
        state = {"completed_round": 0, "records": [SimpleNamespace(accuracy=.1)]}
        callback(state)
        observer.checkpoint.assert_called_once_with(state)
        observer.finish.assert_called_once_with(train.return_value)
        artifact = protocol.load_completed(self.output, self.tasks[0])
        self.assertEqual(artifact["completed_training_rounds"], 80)
        self.assertFalse(artifact["checkpoints_used"])
        self.assertFalse(list(folder.rglob("*.npz")))
        self.assertEqual(artifact["implementation_evidence"], {"status": "synthetic"})

    def test_new_worker_semantic_rejection_retains_failure_but_never_publishes_complete(self):
        context, train, _, _ = self.fake_worker()
        with context, self.assertRaisesRegex(ValueError, "observation.*schema|schema.*observation"):
            runner.worker(self.args)
        train.assert_called_once()
        folder = self.output / "tasks" / self.args.worker
        self.assertFalse((folder / "completed.json").exists())
        attempts = list((folder / "attempts").iterdir())
        self.assertEqual(len(attempts), 1)
        self.assertTrue((attempts[0] / "environment.json").is_file())
        self.assertFalse(list(folder.rglob("*.npz")))
        self.assertFalse((attempts[0] / "completed.json").exists())
        failure = runtime.read_json(attempts[0] / "failure.json")
        self.assertFalse(failure["algorithm_health_assessed"])
        self.assertEqual(failure["status"], "failed_clean_horizon_panel")
        self.assertRegex(failure["error"], "observation.*schema|schema.*observation")

    def test_worker_failure_is_retained_and_not_algorithm_health_failure(self):
        context, _, _, _ = self.fake_worker(error=RuntimeError("synthetic failure"))
        with context, self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            runner.worker(self.args)
        folder = self.output / "tasks" / self.args.worker
        self.assertFalse((folder / "completed.json").exists())
        failures = list((folder / "attempts").glob("*/failure.json"))
        self.assertEqual(len(failures), 1)
        failure = runtime.read_json(failures[0])
        self.assertFalse(failure["algorithm_health_assessed"])
        self.assertEqual(failure["status"], "failed_clean_horizon_panel")
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}), self.readonly_guards(), \
                self.assertRaisesRegex(ValueError, "unsuccessful attempt"):
            runner.worker(self.args)

    def test_history_intervention_is_outer_context_and_restores_before_finish(self):
        self.args.worker = self.tasks[1]["task_id"]
        original_commit = history_observation.LongitudinalSVDDetector.commit
        context, train, observer, _ = self.fake_worker()
        def training(*_args, **_kwargs):
            self.assertTrue(history_observation._active)
            self.assertIsNot(history_observation.LongitudinalSVDDetector.commit, original_commit)
            return SimpleNamespace(stopped_round=80)
        def finish(_result):
            self.assertFalse(history_observation._active)
            self.assertIs(history_observation.LongitudinalSVDDetector.commit, original_commit)
            observer.__exit__.assert_called_once()
            return {"synthetic": "context-order-only"}
        train.side_effect, observer.finish.side_effect = training, finish
        with context, mock.patch.object(runner, "validate_artifact", side_effect=lambda output, task, manifest, artifact: artifact):
            self.assertEqual(runner.worker(self.args), 0)
        self.assertIs(history_observation.LongitudinalSVDDetector.commit, original_commit)

    def test_early_algorithm_termination_keeps_actual_rounds_as_complete_evidence(self):
        # Detailed exhausted-run semantics are covered in report tests. Here the
        # controller must retain a validated scientific outcome without retry.
        context, _, observer, _ = self.fake_worker(stopped_round=26)
        observer.finish.return_value = {"terminal": {"stopped_round": 26,
            "requested_rounds": 80, "stop_reason": "all_honest_revoked"}}
        with context, mock.patch.object(report, "validate_observations"), \
                mock.patch.object(report, "validate_environment_policy"):
            self.assertEqual(runner.worker(self.args), 0)
        artifact = protocol.load_completed(self.output, self.tasks[0])
        self.assertEqual((artifact["completed_training_rounds"], artifact["requested_training_rounds"]), (26, 80))
        self.assertFalse(list((self.output / "tasks" / self.args.worker).rglob("failure.json")))

    def test_artifact_round_count_cannot_disagree_with_validated_observations(self):
        artifact = self.complete()
        artifact["observations"] = {"terminal": {"stopped_round": 29}}
        with mock.patch.object(report, "validate_observations"), \
                mock.patch.object(report, "validate_environment_policy"), \
                self.assertRaisesRegex(ValueError, "round counts differ"):
            runner.validate_artifact(self.output, self.tasks[0], self.manifest, artifact)

    def test_output_collisions_are_rejected_before_reference_or_GPU_access(self):
        for output in (self.mechanism_output, self.mechanism_output / "child", self.mechanism_output.parent):
            self.args.output = output
            with self.subTest(output=output), mock.patch.object(protocol, "audit_reference") as audit, \
                    mock.patch.object(runner.original, "gpu_inventory") as inventory, \
                    self.assertRaisesRegex(ValueError, "non-nested"):
                runner.run_parent(self.args)
            audit.assert_not_called()
            inventory.assert_not_called()

    def test_interruption_terminates_worker_group_then_kills_after_timeout(self):
        process = mock.Mock(pid=12345, returncode=None)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 15), -9]
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}), \
                mock.patch.object(runner.original, "gpu_inventory", return_value=[self.gpu]), \
                mock.patch.object(runner.subprocess, "Popen", return_value=process) as spawn, \
                mock.patch.object(runner.time, "sleep", side_effect=KeyboardInterrupt), \
                mock.patch.object(runner.os, "killpg") as kill, self.assertRaises(KeyboardInterrupt):
            runner.execute(self.args, self.manifest, self.tasks)
        self.assertEqual(kill.call_args_list, [mock.call(12345, signal.SIGTERM), mock.call(12345, signal.SIGKILL)])
        self.assertEqual(process.wait.call_count, 2)
        self.assertEqual(spawn.call_count, 1)

    def test_parent_SIGTERM_handler_is_restored_and_blocked_run_returns_summary(self):
        previous = signal.getsignal(signal.SIGTERM)
        def interrupt(*_):
            handler = signal.getsignal(signal.SIGTERM)
            self.assertIsNot(handler, previous)
            handler(signal.SIGTERM, None)
        with mock.patch.object(runner, "execute", side_effect=interrupt), \
                mock.patch.object(runner, "print_current_summary", return_value=2):
            self.assertEqual(runner.run_parent(self.args), 2)
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)
        records = [json.loads(line) for p in (self.output / "controller_runs").glob("*.jsonl") for line in p.read_text().splitlines()]
        self.assertTrue(any(r.get("exception") == "KeyboardInterrupt" and not r["algorithm_health_assessed"] for r in records))

    def test_parent_refuses_reference_migration_and_nonempty_unidentified_output(self):
        self.args.mechanism_output = self.output.parent / "other-prefix"
        with mock.patch.object(runner, "execute", side_effect=AssertionError("training")), \
                self.assertRaisesRegex(ValueError, "reference path"):
            runner.run_parent(self.args)
        self.args.mechanism_output = self.mechanism_output
        self.args.output = self.output.parent / "unidentified"
        self.args.output.mkdir()
        (self.args.output / "checkpoint.bin").write_bytes(b"preserve")
        with mock.patch.object(protocol, "audit_reference", side_effect=AssertionError("reference read")), \
                self.assertRaisesRegex(ValueError, "no valid"):
            runner.run_parent(self.args)
        self.assertEqual((self.args.output / "checkpoint.bin").read_bytes(), b"preserve")

    def test_worker_rejects_wrong_UUID_or_numerical_environment_before_training(self):
        for metadata, pattern in (({**self.metadata, "environment": {"CUDA_VISIBLE_DEVICES": "GPU-other"}}, "physical GPU"),
                                  ({**self.metadata, "torch": {**self.metadata["torch"], "version": "changed"}}, "numerical environment")):
            self.args.retry_failed = True
            context, train, _, _ = self.fake_worker(metadata=metadata)
            with context, self.assertRaisesRegex(ValueError, pattern):
                runner.worker(self.args)
            train.assert_not_called()
            self.assertFalse((self.output / "tasks" / self.args.worker / "completed.json").exists())

    def test_worker_rejects_multiple_visible_devices_before_data_or_training(self):
        context, train, _, _ = self.fake_worker()
        with context, mock.patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))}), \
                mock.patch.object(base, "load_split") as load, self.assertRaisesRegex(ValueError, "exactly one"):
            runner.worker(self.args)
        load.assert_not_called()
        train.assert_not_called()

    def test_changed_scientific_sources_prevent_publication_and_retain_failure(self):
        context, train, _, _ = self.fake_worker()
        with context, mock.patch.object(protocol, "source_hashes", return_value={"synthetic.py": "1" * 64}), \
                self.assertRaisesRegex(ValueError, "source changed"):
            runner.worker(self.args)
        train.assert_called_once()
        folder = self.output / "tasks" / self.args.worker
        self.assertFalse((folder / "completed.json").exists())
        failures = list((folder / "attempts").glob("*/failure.json"))
        self.assertEqual(len(failures), 1)
        self.assertIn("source changed", runtime.read_json(failures[0])["error"])

    def test_changed_reference_after_training_prevents_publication(self):
        context, train, _, _ = self.fake_worker()
        with context, mock.patch.object(protocol, "verify_reference", side_effect=ValueError("reference changed")), \
                self.assertRaisesRegex(ValueError, "reference changed"):
            runner.worker(self.args)
        train.assert_called_once()
        self.assertFalse((self.output / "tasks" / self.args.worker / "completed.json").exists())

    def test_new_parent_freezes_verified_reference_and_four_tasks_before_dispatch(self):
        self.args.output = self.root / "new-horizon"
        with mock.patch.object(protocol, "audit_reference", return_value=self.reference) as audit, \
                mock.patch.object(protocol, "reference_paths", return_value=[str(self.mechanism_output)]), \
                mock.patch.object(protocol, "build_manifest", return_value=self.manifest), \
                mock.patch.object(protocol, "build_tasks", return_value=self.tasks), \
                mock.patch.object(runner, "execute", return_value=False) as execute, \
                mock.patch.object(runner, "print_current_summary", return_value=2):
            self.assertEqual(runner.run_parent(self.args), 2)
        audit.assert_called_once_with(self.mechanism_output)
        execute.assert_called_once_with(self.args, self.manifest, self.tasks)
        plan = runtime.read_json(self.args.output / "task_plans/clean_horizon.json")
        self.assertEqual(plan["tasks"], self.tasks)
        self.assertTrue(all(t["config"]["rounds"] == 80 for t in plan["tasks"]))
        self.assertFalse(self.mechanism_output.exists())


if __name__ == "__main__":
    unittest.main()
