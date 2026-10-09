"""Fresh-process serial dispatch, no implicit retry, and read-only CLI boundaries."""
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import run_cifar_prefix_probe as runner
import cifar_prefix_probe_runtime as observation
import tests.test_cifar_prefix_probe_protocol as fixtures

protocol, base, runtime, GPU = runner.protocol, runner.base, runner.runtime, fixtures.GPU


class PrefixRunnerTests(fixtures.PrefixFixture):
    def setUp(self):
        super().setUp()
        self.args = SimpleNamespace(output=self.output, worker=self.tasks[0]["task_id"],
            data_dir=None, retry_failed=False, min_free_memory_mib=16384., gpu="auto")
        self.inventory = [{"index": 3, "uuid": GPU, "name": self.reference["execution_environment"]["actual_compute_device"]["name"],
                           "free_mib": 24000., "utilization": 0.}]

    def all_files(self):
        return {str(p): p.read_bytes() for root in [self.output, *self.paths.values()]
                for p in root.rglob("*") if p.is_file()}

    def fake_worker_context(self, *, training_error=None, contract=None, environment=None):
        stack = ExitStack()
        metadata = self.worker_metadata() if environment is None else environment
        stack.enter_context(mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}))
        stack.enter_context(mock.patch.object(base, "worker_environment", return_value=metadata))
        stack.enter_context(mock.patch.object(protocol.history.threshold.timing.matched, "normalized_environment",
            return_value=self.reference["execution_environment"]))
        stack.enter_context(mock.patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))}))
        split = SimpleNamespace(calibration_dataset=object())
        stack.enter_context(mock.patch.object(base, "load_split", return_value=(split,
            self.manifest["data_contract"] if contract is None else contract)))
        stack.enter_context(mock.patch.object(base.experiments, "_load_round_checkpoint",
            side_effect=AssertionError("checkpoint read")))
        observer = mock.MagicMock()
        observer.__enter__.return_value = observer
        observer.finish.return_value = {"mock_observation": "worker boundary test"}
        stack.enter_context(mock.patch.object(observation, "observe", return_value=observer))
        train = stack.enter_context(mock.patch.object(base.fl, "run_experiment", side_effect=training_error,
            return_value=SimpleNamespace(stopped_round=3)))
        return stack, train, observer, split

    def test_plan_only_needs_no_reference_GPU_data_or_output_creation(self):
        target = self.output.parent / "not-created"
        capture = io.StringIO()
        with self.readonly_guards(), mock.patch.object(runner, "gpu_inventory", side_effect=AssertionError("GPU query")), \
                mock.patch.object(protocol, "audit_reference", side_effect=AssertionError("reference read")), \
                redirect_stdout(capture):
            self.assertEqual(runner.main(["--plan-only", "--output", str(target)]), 0)
        data = json.loads(capture.getvalue())
        self.assertEqual((data["tasks"], data["rounds_each"], data["total_training_rounds"]), (6, 3, 18))
        self.assertFalse(data["training_started"])
        self.assertFalse(data["numerical_policy_modified"])
        self.assertFalse(target.exists())

    def test_pending_summary_is_readonly_and_never_selects_GPU_or_trains(self):
        before, capture = self.all_files(), io.StringIO()
        with self.readonly_guards(), mock.patch.object(runner, "gpu_inventory", side_effect=AssertionError("GPU query")), \
                redirect_stdout(capture):
            self.assertEqual(runner.main(["--summary", "--output", str(self.output)]), 2)
        self.assertEqual(before, self.all_files())
        self.assertIn("CIFAR_PREFIX_PROBE_BEGIN", capture.getvalue())
        self.assertIn("CIFAR_PREFIX_PROBE_END", capture.getvalue())

    def test_GPU_selection_respects_explicit_UUID_mask_capacity_and_fixed_card(self):
        other = {**self.inventory[0], "index": 0, "uuid": "GPU-other", "free_mib": 30000.}
        rows = self.inventory + [other]
        name = self.inventory[0]["name"]
        self.assertEqual(runner.select_gpu(rows, "auto", name, 16000, mask=GPU)["uuid"], GPU)
        for mask in ("0", "0,1", "", "GPU-other"):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                runner.select_gpu(rows, GPU, name, 16000, mask=mask)
        for changed in ({"free_mib": 1.}, {"utilization": 6.}, {"name": "other model"}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                runner.select_gpu([{**self.inventory[0], **changed}], GPU, name, 16000)
        for minimum in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                runner.select_gpu(rows, "auto", name, minimum)

    def test_inventory_rejects_duplicate_UUID_and_nonfinite_capacity(self):
        valid = "3, " + GPU + ", synthetic, 24000, 0\n"
        for text in (valid + valid, valid.replace("24000", "NaN")):
            with mock.patch.object(runner.subprocess, "run", return_value=SimpleNamespace(stdout=text)), \
                    self.assertRaises(ValueError):
                runner.gpu_inventory()

    def test_all_six_workers_are_distinct_serial_processes_on_one_UUID(self):
        started, exited, pending = [], [], set()
        tasks_by_id = {t["task_id"]: t for t in self.tasks}
        owner = self
        class Process:
            def __init__(self, command, **kwargs):
                owner.assertFalse(pending, "a second worker started before the previous one exited")
                task_id = command[command.index("--worker") + 1]
                owner.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], GPU)
                owner.assertTrue(kwargs["start_new_session"])
                owner.assertEqual(kwargs["cwd"], runner.REPO)
                owner.assertNotIn("--checkpoint", command)
                started.append(task_id)
                pending.add(task_id)
                self.task_id, self.returncode, self.pid, self.polls = task_id, None, 1000 + len(started), 0
            def poll(self):
                self.polls += 1
                if self.polls == 1:
                    return None
                if self.returncode is None:
                    task = tasks_by_id[self.task_id]
                    base.write_json(owner.output / "tasks" / self.task_id / "completed.json", owner.completion(task))
                    pending.remove(self.task_id)
                    exited.append(self.task_id)
                    self.returncode = 0
                return self.returncode
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}), \
                mock.patch.object(runner, "gpu_inventory", return_value=self.inventory), \
                mock.patch.object(runner.subprocess, "Popen", side_effect=Process) as spawn, \
                mock.patch.object(runner.time, "sleep"):
            self.assertTrue(runner.execute(self.args, self.manifest, self.tasks))
        self.assertEqual(started, [t["task_id"] for t in self.tasks])
        self.assertEqual(exited, started)
        self.assertEqual(spawn.call_count, 6)

    def test_resume_cannot_override_narrower_ambient_GPU_visibility(self):
        before = self.all_files()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-other"}), \
                mock.patch.object(runner, "gpu_inventory", return_value=self.inventory), \
                mock.patch.object(runner.subprocess, "Popen") as spawn, \
                self.assertRaisesRegex(ValueError, "no idle compatible"):
            runner.execute(self.args, self.manifest, self.tasks)
        spawn.assert_not_called()
        self.assertEqual(before, self.all_files())

    def test_nonzero_worker_exit_stops_serial_panel_without_automatic_retry(self):
        process = SimpleNamespace(returncode=75, poll=lambda: 75)
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(runner, "gpu_inventory", return_value=self.inventory), \
                mock.patch.object(runner.subprocess, "Popen", return_value=process) as spawn:
            self.assertFalse(runner.execute(self.args, self.manifest, self.tasks))
        self.assertEqual(spawn.call_count, 1)
        self.assertFalse((self.output / "tasks" / self.tasks[1]["task_id"] / "worker.log").exists())

    def test_exit_zero_without_completion_is_blocked_and_completed_tasks_reuse_without_GPU(self):
        process = SimpleNamespace(returncode=0, poll=lambda: 0)
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(runner, "gpu_inventory", return_value=self.inventory), \
                mock.patch.object(runner.subprocess, "Popen", return_value=process) as spawn:
            self.assertFalse(runner.execute(self.args, self.manifest, self.tasks))
        self.assertEqual(spawn.call_count, 1)
        for task in self.tasks:
            base.write_json(self.output / "tasks" / task["task_id"] / "completed.json", self.completion(task))
        with mock.patch.object(runner, "gpu_inventory", side_effect=AssertionError("GPU on reuse")), \
                mock.patch.object(runner.subprocess, "Popen", side_effect=AssertionError("retraining")):
            self.assertTrue(runner.execute(self.args, self.manifest, self.tasks))

    def test_existing_failed_attempt_requires_explicit_fresh_retry_and_preserves_failure(self):
        folder = self.output / "tasks" / self.args.worker
        failed = folder / "attempts/previous/failure.json"
        failed.parent.mkdir(parents=True)
        base.write_json(failed, {"error": "retained-original"})
        original = failed.read_bytes()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}), self.readonly_guards(), \
                self.assertRaisesRegex(ValueError, "unsuccessful attempt"):
            runner.worker(self.args)
        self.args.retry_failed = True
        context, train, observer, split = self.fake_worker_context()
        with context:
            self.assertEqual(runner.worker(self.args), 0)
        train.assert_called_once()
        self.assertIs(train.call_args.args[0], split.calibration_dataset)
        config = train.call_args.args[1]
        self.assertEqual((config.rounds, config.device, config.malicious_ratio), (3, "cuda:0", 0.))
        self.assertEqual(set(train.call_args.kwargs), {"checkpoint_callback"})
        self.assertEqual(failed.read_bytes(), original)
        self.assertEqual(len(list((folder / "attempts").iterdir())), 2)
        result = protocol.load_completed(self.output, self.tasks[0])
        self.assertTrue(result["fresh_start"])
        self.assertFalse(result["checkpoints_used"])
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}), self.readonly_guards():
            self.assertEqual(runner.worker(self.args), 0)

    def test_worker_failure_keeps_attempt_and_does_not_become_algorithm_health_failure(self):
        context, train, _, _ = self.fake_worker_context(training_error=RuntimeError("synthetic infrastructure failure"))
        with context, self.assertRaisesRegex(RuntimeError, "infrastructure failure"):
            runner.worker(self.args)
        folder = self.output / "tasks" / self.args.worker
        self.assertFalse((folder / "completed.json").exists())
        failures = list((folder / "attempts").glob("*/failure.json"))
        self.assertEqual(len(failures), 1)
        failure = runtime.read_json(failures[0])
        self.assertEqual(failure["status"], "failed_prefix")
        self.assertFalse(failure["algorithm_health_assessed"])
        self.assertEqual(failure["task_fingerprint"], self.tasks[0]["fingerprint"])
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}), self.readonly_guards(), \
                self.assertRaisesRegex(ValueError, "unsuccessful attempt"):
            runner.worker(self.args)
        self.assertEqual(len(list((folder / "attempts").iterdir())), 1)

    def test_worker_rejects_bad_binding_data_or_numerical_environment_before_training(self):
        bad = self.worker_metadata()
        bad["actual_compute_device"]["uuid"] = "GPU-other"
        context, train, _, _ = self.fake_worker_context(environment=bad)
        with context, self.assertRaisesRegex(ValueError, "UUID"):
            runner.worker(self.args)
        train.assert_not_called()
        self.args.retry_failed = True
        context, train, _, _ = self.fake_worker_context(contract={"different": "split"})
        with context, self.assertRaisesRegex(ValueError, "data/split"):
            runner.worker(self.args)
        train.assert_not_called()
        context, train, _, _ = self.fake_worker_context()
        with context, mock.patch.object(protocol.history.threshold.timing.matched, "normalized_environment", return_value={}), \
                self.assertRaisesRegex(ValueError, "numerical environment"):
            runner.worker(self.args)
        train.assert_not_called()

    def test_parent_refuses_reference_migration_and_nonempty_unidentified_output(self):
        migrated = dict(self.paths, history=self.output.parent / "different-history")
        with mock.patch.object(runner, "execute", side_effect=AssertionError("training")), \
                self.assertRaisesRegex(ValueError, "reference locations"):
            runner.run_parent(self.args, migrated)
        self.args.gpu = "GPU-other"
        with mock.patch.object(runner, "execute", side_effect=AssertionError("training")), \
                self.assertRaisesRegex(ValueError, "cannot migrate"):
            runner.run_parent(self.args, self.paths)
        self.args.output = self.output.parent / "unidentified"
        self.args.output.mkdir()
        (self.args.output / "checkpoint.bin").write_bytes(b"preserve")
        with mock.patch.object(protocol, "audit_reference", side_effect=AssertionError("reference audit")), \
                self.assertRaisesRegex(ValueError, "no valid identity"):
            runner.run_parent(self.args, self.paths)
        self.assertEqual((self.args.output / "checkpoint.bin").read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
