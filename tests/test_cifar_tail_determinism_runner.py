"""Bounded same-GPU serial dispatch and retained fresh diagnostic attempts."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
import signal
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import run_cifar_tail_determinism as runner
import cifar_tail_determinism_runtime as observation
import tests.test_cifar_tail_determinism_protocol as fixtures

protocol, base, runtime, GPU = runner.protocol, runner.base, runner.runtime, fixtures.GPU


class TailRunnerTests(fixtures.TailFixture):
    def setUp(self):
        super().setUp()
        self.args = SimpleNamespace(output=self.output, step_output=self.step_output, data_dir=None,
            worker=self.tasks[0]["task_id"], retry_failed=False, wait_seconds=600., poll_seconds=10.)
        self.gpu = {"index": 3, "uuid": GPU, "name": self.reference["execution_environment"]["actual_compute_device"]["name"],
                    "free_mib": 24000., "utilization": 0.}

    def fake_worker(self, error=None, metadata=None):
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
        observer.snapshots = {"tail": np.array([1., 2.], dtype=np.float32)}
        stack.enter_context(mock.patch.object(observation, "observe", return_value=observer))
        stack.enter_context(mock.patch.object(observation, "source_evidence", return_value={"status": "synthetic"}))
        train = stack.enter_context(mock.patch.object(base.fl, "run_experiment", side_effect=error))
        return stack, train, observer, split

    def test_plan_only_is_readonly_without_reference_GPU_or_data(self):
        target, capture = self.output.parent / "not-created", io.StringIO()
        with self.readonly_guards(), mock.patch.object(runner.original, "gpu_inventory", side_effect=AssertionError("GPU")), \
                mock.patch.object(protocol, "audit_reference", side_effect=AssertionError("reference")), redirect_stdout(capture):
            self.assertEqual(runner.main(["--plan-only", "--output", str(target)]), 0)
        data = json.loads(capture.getvalue())
        self.assertEqual((data["fresh_processes"], data["clients_per_process"], data["total_client_training_calls"]), (6, 85, 510))
        self.assertEqual(data["target_clients"], [19, 84])
        self.assertFalse(data["aggregation_executed"])
        self.assertTrue(data["numerical_policy_modified"])
        self.assertFalse(target.exists())

    def test_pending_summary_never_queries_GPU_loads_data_or_writes(self):
        before = protocol.evidence_hashes(self.output)
        with self.readonly_guards(), mock.patch.object(runner.original, "gpu_inventory", side_effect=AssertionError("GPU")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(["--summary", "--output", str(self.output)]), 2)
        self.assertEqual(before, protocol.evidence_hashes(self.output))
        self.assertFalse((self.output / "controller_runs").exists())

    def test_six_interleaved_fresh_processes_wait_on_same_UUID_then_run_strictly_serially(self):
        owner, starts, exits, active, events = self, [], [], set(), []
        by_id = {t["task_id"]: t for t in self.tasks}
        class Process:
            def __init__(self, command, **kwargs):
                owner.assertFalse(active)
                tid = command[command.index("--worker") + 1]
                owner.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], GPU)
                owner.assertTrue(kwargs["start_new_session"])
                owner.assertEqual(kwargs["cwd"], runner.REPO)
                owner.assertEqual(command[2], str(runner.REPO / "run_cifar_tail_determinism.py"))
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
                mock.patch.object(runner.original, "gpu_inventory", side_effect=[[busy], [self.gpu]] * 6) as inventory, \
                mock.patch.object(runner.subprocess, "Popen", side_effect=Process) as spawn, \
                mock.patch.object(runner.time, "sleep"), \
                runner.admission.waiting_gate(GPU, wait_seconds=600., poll_seconds=10., emit=events.append):
            self.assertTrue(runner.execute(self.args, self.manifest, self.tasks))
        self.assertEqual(starts, [t["task_id"] for t in self.tasks])
        self.assertEqual(exits, starts)
        self.assertEqual([(t["policy"], t["repeat"]) for t in self.tasks],
            [(p, r) for r in (1, 2, 3) for p in ("original", "tail_cudnn_deterministic")])
        self.assertEqual((spawn.call_count, inventory.call_count), (6, 12))
        self.assertEqual([e["event"] for e in events], ["gpu_wait", "gpu_ready"] * 6)

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

    def test_sealed_invalid_observations_or_arrays_block_reuse_before_GPU_or_launch(self):
        for arrays, reason in ((None, "observation schema"),
                               ({"bad": np.array([object()], dtype=object)}, "Object arrays")):
            with self.subTest(reason=reason):
                artifact = self.complete(arrays=arrays)
                # JSON seal, task identity, environment binding and NPZ SHA are valid.
                self.assertEqual(protocol.load_completed(self.output, self.tasks[0]), artifact)
                with mock.patch.object(runner.original, "gpu_inventory") as inventory, \
                        mock.patch.object(runner.subprocess, "Popen") as spawn, \
                        self.assertRaisesRegex(ValueError, reason):
                    runner.execute(self.args, self.manifest, self.tasks)
                inventory.assert_not_called()
                spawn.assert_not_called()
                with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}), \
                        mock.patch.object(base, "worker_environment") as environment, \
                        self.assertRaisesRegex(ValueError, reason):
                    runner.worker(self.args)
                environment.assert_not_called()

    def test_exit0_with_semantically_bad_completion_blocks_the_next_task(self):
        def finish_bad(*_args, **_kwargs):
            self.complete(self.tasks[0])
            return SimpleNamespace(returncode=0, poll=lambda: 0)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}), \
                mock.patch.object(runner.original, "gpu_inventory", return_value=[self.gpu]) as inventory, \
                mock.patch.object(runner.subprocess, "Popen", side_effect=finish_bad) as spawn, \
                self.assertRaisesRegex(ValueError, "observation schema"):
            runner.execute(self.args, self.manifest, self.tasks)
        self.assertEqual((spawn.call_count, inventory.call_count), (1, 1))
        self.assertFalse((self.output / "tasks" / self.tasks[1]["task_id"] / "worker.log").exists())

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
        self.assertEqual(train.call_args.args[1].rounds, 3)
        self.assertEqual(set(train.call_args.kwargs), {"checkpoint_callback"})
        self.assertIs(train.call_args.kwargs["checkpoint_callback"], observer.checkpoint)
        artifact = protocol.load_completed(self.output, self.tasks[0])
        self.assertFalse(artifact["training_round_completed"])
        self.assertFalse(artifact["checkpoints_used"])
        np.testing.assert_array_equal(protocol.load_arrays(self.output, self.tasks[0], artifact)["tail"], observer.snapshots["tail"])

    def test_new_worker_semantic_rejection_retains_snapshot_but_never_publishes_complete(self):
        context, train, _, _ = self.fake_worker()
        with context, self.assertRaisesRegex(ValueError, "observation schema"):
            runner.worker(self.args)
        train.assert_called_once()
        folder = self.output / "tasks" / self.args.worker
        self.assertFalse((folder / "completed.json").exists())
        attempts = list((folder / "attempts").iterdir())
        self.assertEqual(len(attempts), 1)
        self.assertTrue((attempts[0] / "snapshots.npz").is_file())
        self.assertFalse((attempts[0] / "completed.json").exists())
        failure = runtime.read_json(attempts[0] / "failure.json")
        self.assertFalse(failure["algorithm_health_assessed"])
        self.assertEqual(failure["status"], "failed_tail_determinism_probe")
        self.assertIn("observation schema", failure["error"])

    def test_strict_completion_accepts_full_snapshot_then_resealed_policy_tamper_blocks_reuse(self):
        from cifar_tail_determinism_report import ENVIRONMENT_FLAGS
        task = self.tasks[0]
        payload, arrays = fixtures.synthetic_step_observations(task, self.reference)
        profile = {"cudnn_enabled": True, "cudnn_benchmark_limit": 10, "cudnn_deterministic": False,
            "cudnn_benchmark": False, "cudnn_allow_tf32": True, "cuda_matmul_allow_tf32": True,
            "deterministic_algorithms": False, "deterministic_warn_only": False, "float32_matmul_precision": "high"}
        clients = payload["prefix"]["rounds"][0]["clients"]
        training = sum(len(c["minibatch_sizes_per_epoch"]) * c["epochs"] for c in clients)
        forwards = training + sum(len(e["batches"]) for e in payload["prefix"]["evaluations"])
        events = [{"client_id": target["client_id"],
            **{key: target["batches"][-1][key] for key in ("batch", "epoch", "batch_in_epoch", "samples")},
            "before": deepcopy(profile), "effective": deepcopy(profile), "restored": deepcopy(profile),
            "backward_completed": True} for target in payload["targets"]]
        payload["numerical_policy"] = {"schema": observation.POLICY_SCHEMA, "policy": task["policy"],
            "baseline": profile, "exit_profile": deepcopy(profile), "events": events,
            "expected_scope": observation.SCOPE, "other_flags_changed": False, "restored_on_exit": True,
            "scoped_changes": 0, "checks": {"forward_before": forwards, "forward_after": forwards,
                "non_target_backward_before": training - 2, "non_target_backward_after": training - 2,
                "post_flat": len(clients), "exit": 1}}
        artifact = self.complete(task, arrays=arrays)
        artifact.pop("artifact_fingerprint")
        artifact.update(observations=payload, training_round_completed=False,
            implementation_evidence={"status": "synthetic"})
        artifact["environment"]["torch"].update({key: profile[key] for key in ENVIRONMENT_FLAGS})
        # Older protocol fixtures omit optional collector flags. This check exercises
        # publication/reuse validation with the full numerical provenance of a worker.
        manifest = deepcopy(self.manifest)
        manifest["reference"]["execution_environment"] = protocol.prefix_report.matched.normalized_environment(artifact["environment"])
        base.write_json(self.output / "tasks" / task["task_id"] / "completed.json", protocol.seal_artifact(artifact))
        artifact = runner.validated_completed(self.output, task, manifest)
        self.assertEqual(artifact["observations"], payload)
        self.assertEqual(artifact["implementation_evidence"], {"status": "synthetic"})
        # Correct outer digest and array bytes must not bypass the scoped flag contract.
        artifact.pop("artifact_fingerprint")
        artifact["observations"]["numerical_policy"]["events"][0]["effective"]["cudnn_allow_tf32"] = not profile["cudnn_allow_tf32"]
        base.write_json(self.output / "tasks" / task["task_id"] / "completed.json", protocol.seal_artifact(artifact))
        with mock.patch.object(runner.original, "gpu_inventory") as inventory, \
                mock.patch.object(runner.subprocess, "Popen") as spawn, \
                self.assertRaisesRegex(ValueError, "effective backward flags"):
            runner.execute(self.args, manifest, self.tasks)
        inventory.assert_not_called()
        spawn.assert_not_called()

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
        self.assertEqual(failure["status"], "failed_tail_determinism_probe")
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}), self.readonly_guards(), \
                self.assertRaisesRegex(ValueError, "unsuccessful attempt"):
            runner.worker(self.args)

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
        self.args.step_output = self.output.parent / "other-prefix"
        with mock.patch.object(runner, "execute", side_effect=AssertionError("training")), \
                self.assertRaisesRegex(ValueError, "reference path"):
            runner.run_parent(self.args)
        self.args.step_output = self.step_output
        self.args.output = self.output.parent / "unidentified"
        self.args.output.mkdir()
        (self.args.output / "checkpoint.bin").write_bytes(b"preserve")
        with mock.patch.object(protocol, "audit_reference", side_effect=AssertionError("reference read")), \
                self.assertRaisesRegex(ValueError, "no valid"):
            runner.run_parent(self.args)
        self.assertEqual((self.args.output / "checkpoint.bin").read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
