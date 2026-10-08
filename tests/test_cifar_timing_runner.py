"""Independent dispatch, checkpoint and read-only reference boundaries for stage 2."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import run_cifar_timing_diagnostic as runner
from tests.test_cifar_six_pipeline import synthetic_run

base, runtime, protocol = runner.base, runner.runtime, runner.protocol


class RunnerFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "timing"
        self.output.mkdir()
        self.clean_output = self.root / "clean"
        self.matched_output = self.root / "matched"
        self.clean_output.mkdir()
        self.matched_output.mkdir()
        (self.clean_output / "untouched").write_text("original clean evidence")
        (self.matched_output / "untouched").write_text("original matched evidence")
        self.environment = {"actual_compute_device": {"name": "synthetic"}, "environment": {}}
        self.reference = {"execution_environment": self.environment}
        self.manifest = {"fingerprint": "synthetic-manifest", "reference": self.reference,
                         "spec": {"dataset": "synthetic"}, "data_contract": {"synthetic": True}}
        source = runtime.read_json(runner.clean.DEFAULT_CONFIG)["shared_parameters"]
        self.tasks = []
        for i in range(4):
            config = base.fl.ExperimentConfig(**{**source, "method": "sm9rrs", "seed": 2026093001,
                "malicious_ratio": .1 if i % 2 == 0 else .7,
                "partition": "iid" if i < 2 else "dirichlet"})
            self.tasks.append({"task_id": f"timing_A_{i}", "phase": "validation", "model": "v7_cnn",
                "method": "sm9rrs", "arm": "A", "candidate": {"candidate_id": "Ours014"},
                "config": asdict(config)})
        self.tasks = base.attach_fingerprints(self.tasks, self.manifest)
        base.write_json(self.output / "execution_environment.json", self.environment)
        self.args = SimpleNamespace(output=self.output, clean_output=self.clean_output,
            matched_output=self.matched_output, devices=["cuda:0", "cuda:1"], data_dir=None,
            worker=self.tasks[0]["task_id"])

    def complete(self, task, *, nonfinite=0):
        folder = runtime.ensure_identity(self.output, task)
        result = synthetic_run(base.fl.ExperimentConfig(**task["config"]), nonfinite=nonfinite)
        base.experiments._write_completed_results_snapshot(folder, [result])
        return result

    def failure(self, task, kind):
        folder = runtime.ensure_identity(self.output, task)
        base.write_json(folder / "failure.json", {"task_id": task["task_id"],
            "task_fingerprint": task["fingerprint"], "kind": kind, "message": "synthetic"})

    def setup_training(self, *, side_effect=None):
        """Patch only environmental and training boundaries, not the monitor."""
        task = self.tasks[0]
        folder = runtime.ensure_identity(self.output, task)
        self.args.devices = ["cuda:0"]
        calibration, official = object(), object()
        patches = [mock.patch.object(base, "worker_environment", return_value=self.environment),
            mock.patch.object(base, "check_environment"),
            mock.patch.object(base, "load_split", return_value=(SimpleNamespace(
                calibration_dataset=calibration, main_dataset=official), {"synthetic": True})),
            mock.patch("torch.cuda.reset_peak_memory_stats"),
            mock.patch("torch.cuda.max_memory_allocated", return_value=8 * 2**20),
            mock.patch.object(runner.clean, "observe", side_effect=AssertionError("clean observer forbidden")),
            mock.patch.object(base.experiments, "run_measured_experiment", side_effect=side_effect,
                return_value=synthetic_run(base.fl.ExperimentConfig(**task["config"])))]
        started = [patch.start() for patch in patches]
        for patch in patches:
            self.addCleanup(patch.stop)
        return task, folder, calibration, started[-1]


class WorkerTests(RunnerFixture):
    def test_worker_only_accepts_declared_task_explicit_device_and_frozen_environment(self):
        task = self.tasks[0]
        runtime.ensure_identity(self.output, task)
        self.args.devices = ["cuda:0"]
        with mock.patch.object(protocol, "read_study", return_value=(self.manifest, self.tasks)) as study, \
                mock.patch.object(runner, "run_task", return_value=0) as run:
            self.assertEqual(runner.worker(self.args), 0)
            study.assert_called_once_with(self.output, current_sources=True)
            self.assertEqual(run.call_args.args[2], task)
            self.args.worker = "other-stage-task"
            with self.assertRaisesRegex(ValueError, "declared"):
                runner.worker(self.args)
            self.args.worker = task["task_id"]
            self.args.devices = ["auto"]
            with self.assertRaisesRegex(ValueError, "explicit"):
                runner.worker(self.args)
            self.args.devices = ["cuda:0"]
            base.write_json(self.output / "execution_environment.json", {"changed": True})
            with self.assertRaisesRegex(ValueError, "environment"):
                runner.worker(self.args)
            self.assertEqual(run.call_count, 1)

    def test_worker_rejects_task_file_or_model_mismatch(self):
        task = self.tasks[0]
        folder = runtime.ensure_identity(self.output, task)
        self.args.devices = ["cuda:0"]
        with mock.patch.object(protocol, "read_study", return_value=(self.manifest, self.tasks)), \
                mock.patch.object(runner, "run_task", side_effect=AssertionError("training")):
            base.write_json(folder / "task.json", {**task, "model": "resnet18_gn2"})
            with self.assertRaisesRegex(ValueError, "task identity"):
                runner.worker(self.args)
            task["model"] = "resnet18_gn2"
            with self.assertRaisesRegex(ValueError, "original CNN"):
                runner.worker(self.args)

    def test_training_uses_calibration_identity_and_original_CNN_without_clean_observer(self):
        task, folder, calibration, train = self.setup_training()
        original = base.experiments.run_experiment
        previous_handlers = {s: runner.signal.getsignal(s) for s in (runner.signal.SIGINT, runner.signal.SIGTERM)}
        with mock.patch.object(runner.clean, "model_runtime", wraps=runner.clean.model_runtime) as model, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_task(self.args, self.manifest, task, folder), 0)
        model.assert_called_once_with("v7_cnn")
        self.assertIs(train.call_args.args[0], calibration)
        self.assertEqual(train.call_args.args[1].device, "cuda:0")
        self.assertEqual(train.call_args.kwargs["run_fingerprint"], task["fingerprint"])
        self.assertTrue(train.call_args.kwargs["retain_success_checkpoint"])
        self.assertEqual(asdict(train.call_args.kwargs["checkpoint_identity_config"]), task["config"])
        self.assertIs(base.experiments.run_experiment, original)
        self.assertEqual({s: runner.signal.getsignal(s) for s in previous_handlers}, previous_handlers)
        attempt = runtime.read_json(next((folder / "attempts").glob("*.json")))
        self.assertEqual(attempt["status"], "complete")
        self.assertEqual(attempt["cuda_peak_allocated_mib"], 8.)
        self.assertGreaterEqual(attempt["wall_seconds"], 0.)
        self.assertFalse((folder / "observations.json").exists())

    def test_completed_task_reuse_does_not_load_data_or_change_scientific_outputs(self):
        task = self.tasks[0]
        self.complete(task, nonfinite=1)
        folder = self.output / "tasks" / task["task_id"]
        before = {str(p): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
        with mock.patch.object(base, "load_split", side_effect=AssertionError("reload")), \
                mock.patch.object(base, "worker_environment", side_effect=AssertionError("GPU")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_task(self.args, self.manifest, task, folder), 0)
        self.assertEqual(before, {str(p): p.read_bytes() for p in folder.rglob("*") if p.is_file()})

    def test_corrupt_checkpoint_is_preserved_and_refuses_restart(self):
        task, folder, _, train = self.setup_training()
        identity = base.fl.ExperimentConfig(**task["config"])
        checkpoint = base.experiments._checkpoint_path(folder / "checkpoints", identity)
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"damaged checkpoint")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_task(self.args, self.manifest, task, folder), 1)
        self.assertEqual(checkpoint.read_bytes(), b"damaged checkpoint")
        train.assert_not_called()
        self.assertIn("refusing silent restart", runtime.read_json(folder / "failure.json")["message"])

    def test_changed_data_contract_refuses_training(self):
        task, folder, _, train = self.setup_training()
        self.manifest["data_contract"] = {"different": "data"}
        with redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_task(self.args, self.manifest, task, folder), 1)
        train.assert_not_called()
        self.assertIn("data contract", runtime.read_json(folder / "failure.json")["message"])

    def test_cuda_oom_records_cost_and_infrastructure_failure(self):
        task, folder, _, _ = self.setup_training(side_effect=RuntimeError("CUDA out of memory"))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_task(self.args, self.manifest, task, folder), 1)
        self.assertEqual(runtime.read_json(folder / "failure.json")["kind"], "infrastructure_oom")
        attempt = runtime.read_json(next((folder / "attempts").glob("*.json")))
        self.assertEqual(attempt["cuda_peak_allocated_mib"], 8.)
        self.assertEqual(attempt["status"], "failed")

    def test_sigterm_saves_round_before_pause_and_restores_handlers_for_resume(self):
        task, folder, _, train = self.setup_training()
        before = runner.signal.getsignal(runner.signal.SIGTERM)

        def fake_science(*args, **kwargs):
            runner.signal.getsignal(runner.signal.SIGTERM)(None, None)
            kwargs["checkpoint_callback"]({"completed_round": 1, "params": np.zeros(2), "records": []})

        def measured(*args, **kwargs):
            def durable(state):
                base.write_json(folder / "durable_round.json", {"round": state["completed_round"]})
            return base.experiments.run_experiment(*args, checkpoint_callback=durable)

        train.side_effect = measured
        with mock.patch.object(base.experiments, "run_experiment", side_effect=fake_science), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_task(self.args, self.manifest, task, folder), 75)
        self.assertEqual(runtime.read_json(folder / "durable_round.json")["round"], 1)
        self.assertEqual(runtime.read_json(folder / "progress.json")["last_completed_round"], 1)
        self.assertEqual(runtime.read_json(folder / "failure.json")["kind"], "budget_or_interrupt")
        self.assertEqual(runner.signal.getsignal(runner.signal.SIGTERM), before)
        train.side_effect = None
        with redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_task(self.args, self.manifest, task, folder), 0)
        self.assertEqual(len(list((folder / "attempts").glob("*.json"))), 2)

    def test_round_nonfinite_is_terminal_algorithm_failure(self):
        task, folder, _, train = self.setup_training()

        def fake_science(*args, **kwargs):
            kwargs["checkpoint_callback"]({"completed_round": 1, "params": np.array([np.nan]), "records": []})

        train.side_effect = lambda *args, **kwargs: base.experiments.run_experiment(*args)
        with mock.patch.object(base.experiments, "run_experiment", side_effect=fake_science), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_task(self.args, self.manifest, task, folder), 1)
        self.assertEqual(runtime.read_json(folder / "failure.json")["kind"], "algorithm_numerical")


class DispatchTests(RunnerFixture):
    def test_new_entry_and_oom_lane_not_reused(self):
        launched = []

        def launch(command, **kwargs):
            self.assertEqual(Path(command[2]).name, "run_cifar_timing_diagnostic.py")
            self.assertTrue(kwargs["start_new_session"])
            launched.append(command[-1])
            code = int(len(launched) == 1)
            if code:
                self.failure(self.tasks[0], "infrastructure_oom")
            return SimpleNamespace(pid=len(launched), poll=lambda: code)

        with mock.patch.object(runner.subprocess, "Popen", side_effect=launch), redirect_stdout(io.StringIO()):
            runner.execute(self.args, self.tasks)
        self.assertEqual(launched, ["cuda:0", "cuda:1", "cuda:1", "cuda:1"])

    def test_completed_unhealthy_and_numerical_failures_are_not_retrained(self):
        self.complete(self.tasks[0], nonfinite=1)
        for task in self.tasks[1:]:
            self.failure(task, "algorithm_numerical")
        with mock.patch.object(runner.subprocess, "Popen", side_effect=AssertionError("retrain")), \
                redirect_stdout(io.StringIO()) as output:
            runner.execute(self.args, self.tasks)
        self.assertIn("REUSE", output.getvalue())
        self.assertEqual(output.getvalue().count("RETAIN_NUMERICAL_FAILURE"), 3)

    def test_all_operational_lanes_block_preserves_queued_tasks(self):
        self.args.devices = ["cuda:0"]
        self.failure(self.tasks[0], "infrastructure_oom")
        with mock.patch.object(runner.subprocess, "Popen", return_value=SimpleNamespace(pid=1, poll=lambda: 1)) as launch, \
                redirect_stdout(io.StringIO()) as output:
            runner.execute(self.args, self.tasks)
        self.assertEqual(launch.call_count, 1)
        self.assertIn("EXECUTION_BLOCKED", output.getvalue())
        self.assertFalse((self.output / "tasks" / self.tasks[1]["task_id"]).exists())

    def test_interrupt_stops_worker_and_restores_controller_signal_handler(self):
        proc = mock.Mock(pid=7103)
        proc.poll.side_effect = [KeyboardInterrupt(), None]
        self.args.devices = ["cuda:0"]
        before = runner.signal.getsignal(runner.signal.SIGTERM)
        with mock.patch.object(runner.subprocess, "Popen", return_value=proc), \
                mock.patch.object(runner.os, "killpg") as kill, redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                runner.execute(self.args, self.tasks)
        kill.assert_called_once_with(7103, runner.signal.SIGTERM)
        proc.wait.assert_called_once()
        self.assertEqual(runner.signal.getsignal(runner.signal.SIGTERM), before)


class BoundaryTests(RunnerFixture):
    def test_all_pairwise_path_overlap_and_symlink_aliases_rejected_before_write(self):
        for output, old1, old2 in ((self.clean_output, self.clean_output, self.matched_output),
                (self.clean_output / "child", self.clean_output, self.matched_output),
                (self.root, self.clean_output, self.matched_output),
                (self.output, self.clean_output, self.clean_output / "child")):
            with self.assertRaisesRegex(ValueError, "separate"):
                runner.separate_outputs(output, old1, old2)
        link = self.root / "alias"
        link.symlink_to(self.clean_output, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "separate"):
            runner.separate_outputs(link, self.clean_output, self.matched_output)

    def test_plan_only_does_not_read_reference_load_data_probe_gpu_or_write(self):
        absent = self.root / "absent_output"
        with mock.patch.object(protocol, "audit_reference", side_effect=AssertionError("reference IO")), \
                mock.patch.object(base, "load_split", side_effect=AssertionError("download")), \
                mock.patch("run_cifar_six_with_progress.discover_gpus", side_effect=AssertionError("GPU")), \
                redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runner.main(["--plan-only", "--output", str(absent)]), 0)
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["tasks"], 26)
        self.assertEqual(payload["ours_tasks"], 18)
        self.assertEqual(payload["fedavg_tasks"], 8)
        self.assertEqual(payload["local_epochs"], 1)
        self.assertFalse(payload["official_test_used_for_selection"])
        self.assertFalse(absent.exists())

    def test_summary_cli_avoids_training_data_and_GPU(self):
        with mock.patch("cifar_timing_report.summarize", return_value={"status": "incomplete"}) as summary, \
                mock.patch("cifar_timing_report.print_summary"), \
                mock.patch.object(protocol, "audit_reference", side_effect=AssertionError("runner audit")), \
                mock.patch.object(base, "load_split", side_effect=AssertionError("data")), \
                mock.patch("run_cifar_six_with_progress.discover_gpus", side_effect=AssertionError("GPU")):
            self.assertEqual(runner.main(["--summary", "--output", str(self.output),
                "--clean-output", str(self.clean_output), "--matched-output", str(self.matched_output)]), 2)
        summary.assert_called_once_with(self.output, self.clean_output, self.matched_output)

    def test_parent_initialization_recovers_plan_without_writing_references(self):
        # Simulate the crash boundary just after the immutable manifest write.
        (self.output / "execution_environment.json").unlink()
        base.write_json(self.output / "manifest.json", self.manifest)
        before = {str(p): p.read_bytes() for source in (self.clean_output, self.matched_output)
                  for p in source.rglob("*") if p.is_file()}
        with mock.patch.object(protocol, "build_manifest", return_value=self.manifest), \
                mock.patch.object(protocol, "build_tasks", return_value=self.tasks), \
                mock.patch.object(protocol, "read_study", return_value=(self.manifest, self.tasks)), \
                mock.patch.object(runner, "execute"), \
                mock.patch("cifar_timing_report.summarize", return_value={"status": "incomplete"}), \
                mock.patch("cifar_timing_report.print_summary"):
            self.assertEqual(runner.run_parent(self.args, {}, self.reference), 2)
        self.assertEqual(runtime.read_json(self.output / "task_plans/timing.json")["tasks"], self.tasks)
        self.assertEqual(before, {str(p): p.read_bytes() for source in (self.clean_output, self.matched_output)
                  for p in source.rglob("*") if p.is_file()})

    def test_nonempty_unidentified_study_and_orphan_task_rejected(self):
        with mock.patch.object(protocol, "build_manifest", return_value=self.manifest), \
                mock.patch.object(runner, "execute", side_effect=AssertionError("training")):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                runner.run_parent(self.args, {}, self.reference)
        base.write_json(self.output / "manifest.json", self.manifest)
        folder = self.output / "tasks" / self.tasks[0]["task_id"]
        folder.mkdir(parents=True)
        (folder / "unknown_checkpoint.pickle").write_bytes(b"preserve")
        with mock.patch.object(protocol, "build_manifest", return_value=self.manifest), \
                mock.patch.object(protocol, "build_tasks", return_value=self.tasks), \
                mock.patch.object(protocol, "read_study", return_value=(self.manifest, self.tasks)), \
                mock.patch.object(runner, "execute", side_effect=AssertionError("training")):
            with self.assertRaisesRegex(ValueError, "orphan"):
                runner.run_parent(self.args, {}, self.reference)
        self.assertFalse((folder / "task.json").exists())


if __name__ == "__main__":
    unittest.main()
