"""Protocol, evidence, dispatch and recovery tests without CIFAR downloads/CUDA."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import run_cifar_diagnostic as runner
import cifar_diagnostic_report as report
from tests.test_cifar_six_pipeline import synthetic_run

base, runtime = runner.base, runner.runtime


class StudyFixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name)
        self.spec = runner.validate_spec(runtime.read_json(runner.DEFAULT_CONFIG))
        self.manifest = runner.build_manifest(self.spec, {"synthetic": True})
        self.tasks = runner.build_tasks(self.spec, self.manifest)
        base.write_json(self.output / "manifest.json", self.manifest)
        runtime.save_plan(self.output, "clean", self.tasks, self.manifest)
        self.args = SimpleNamespace(output=self.output, devices=["cuda:0", "cuda:1"], data_dir=None)

    def complete(self, task, accuracy=.60, nonfinite=0):
        folder = runtime.ensure_identity(self.output, task)
        result = synthetic_run(base.fl.ExperimentConfig(**task["config"]), accuracy=accuracy, nonfinite=nonfinite)
        base.experiments._write_completed_results_snapshot(folder, [result])
        base.write_json(folder / "observations.json", {"task_fingerprint": task["fingerprint"],
            "rounds": [{"round": r, "local_train_loss": 1. if r else None,
                "local_train_samples": 45000 if r else 0, "calibration_loss": 1.1,
                "calibration_samples": 2500, "cuda_peak_allocated_mib": 3500.,
                "round_wall_seconds": 4. if r else None} for r in range(151)]})
        (folder / "attempts").mkdir(exist_ok=True)
        base.write_json(folder / "attempts/test.json", {"task_fingerprint": task["fingerprint"],
            "status": "complete", "wall_seconds": 600.})
        return folder

    def failure(self, task, kind):
        folder = runtime.ensure_identity(self.output, task)
        base.write_json(folder / "failure.json", {"task_fingerprint": task["fingerprint"],
            "task_id": task["task_id"], "kind": kind, "message": "test failure"})

    def snapshot(self):
        return {str(p.relative_to(self.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.output.rglob("*") if p.is_file()}


class ProtocolTests(StudyFixture):
    def test_fixed_matrix_and_only_training_parameters_vary(self):
        self.assertEqual(len(self.tasks), 24)
        self.assertEqual(len({t["task_id"] for t in self.tasks}), 24)
        for task in self.tasks:
            self.assertEqual(task["phase"], "validation")
            self.assertEqual(task["purpose"], "development_panel")
            self.assertEqual(task["config"]["method"], "fedavg")
            self.assertEqual(task["config"]["malicious_ratio"], 0.)
            for k, v in task["config"].items():
                if k not in {"seed", "partition", "lr", "local_epochs", "lr_decay"}:
                    self.assertEqual(v, self.spec["shared_parameters"][k])
        self.assertEqual(runner.read_study(self.output, current_sources=True)[1], self.tasks)

    def test_no_silent_protocol_edits_or_formal_seed(self):
        for key, value in (("rounds", 100), ("malicious_ratio", .1), ("crypto_mode", "simulated"),
                           ("checkpoint_interval", 0), ("method", "sm9rrs")):
            spec = deepcopy(self.spec)
            spec["shared_parameters"][key] = value
            with self.assertRaises(ValueError):
                runner.validate_spec(spec)
        spec = deepcopy(self.spec)
        spec["seeds"][0] = 2026093011
        with self.assertRaises(ValueError):
            runner.validate_spec(spec)
        spec = deepcopy(self.spec)
        spec["settings"][4]["local_epochs"] = 3
        with self.assertRaises(ValueError):
            runner.validate_spec(spec)

    def test_plan_only_never_loads_data_probes_gpu_or_writes(self):
        before = self.snapshot()
        with mock.patch.object(base, "load_split", side_effect=AssertionError("download")), \
                mock.patch("run_cifar_six_with_progress.discover_gpus", side_effect=AssertionError("GPU")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(["--plan-only", "--output", str(self.output)]), 0)
        self.assertEqual(before, self.snapshot())

    def test_model_or_source_changes_invalidate_resume(self):
        plan = runtime.read_json(self.output / "task_plans/clean.json")
        plan["tasks"][0]["model"] = "resnet18_gn2"
        base.write_json(self.output / "task_plans/clean.json", plan)
        with self.assertRaisesRegex(ValueError, "plan"):
            runner.read_study(self.output)
        runtime.base.write_json(self.output / "task_plans/clean.json", {
            "manifest_fingerprint": self.manifest["fingerprint"], "tasks": self.tasks})
        with mock.patch.object(runner, "source_hashes", return_value={"changed": "source"}):
            with self.assertRaisesRegex(ValueError, "identity"):
                runner.read_study(self.output, current_sources=True)

    def test_frozen_sources_are_subset_of_diagnostic_sources(self):
        old = runtime.source_hashes()
        self.assertEqual(len(old), 49)
        self.assertEqual({k: runner.source_hashes()[k] for k in old}, old)

    def test_worker_always_uses_calibration_dataset(self):
        task = self.tasks[0]
        folder = runtime.ensure_identity(self.output, task)
        calibration, official = object(), object()
        self.args.devices = ["cuda:0"]
        result = synthetic_run(base.fl.ExperimentConfig(**task["config"]))
        with mock.patch.object(base, "worker_environment", return_value={}), \
                mock.patch.object(base, "check_environment"), \
                mock.patch.object(base, "load_split", return_value=(SimpleNamespace(
                    calibration_dataset=calibration, main_dataset=official), {"synthetic": True})), \
                mock.patch("torch.cuda.reset_peak_memory_stats"), \
                mock.patch("torch.cuda.max_memory_allocated", return_value=12), \
                mock.patch.object(base.experiments, "run_measured_experiment", return_value=result) as train, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_task(self.args, self.manifest, task, folder), 0)
        self.assertIs(train.call_args.args[0], calibration)
        self.assertEqual(train.call_args.kwargs["run_fingerprint"], task["fingerprint"])

    def test_orphan_task_not_relabelled(self):
        task = self.tasks[0]
        folder = self.output / "tasks" / task["task_id"]
        folder.mkdir(parents=True)
        (folder / "unknown.pickle").write_bytes(b"preserve")
        with self.assertRaisesRegex(ValueError, "orphan"):
            runtime.ensure_identity(self.output, task)
        self.assertFalse((folder / "task.json").exists())

    def test_interrupted_manifest_initialization_can_resume_without_reload(self):
        (self.output / "task_plans/clean.json").unlink()
        with mock.patch.object(base, "load_split", side_effect=AssertionError("reload")), \
                mock.patch.object(runner, "execute"), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_parent(self.args, self.spec), 2)
        self.assertEqual(runner.read_study(self.output)[1], self.tasks)

    def test_corrupt_checkpoint_is_not_silently_reinitialized(self):
        task = self.tasks[0]
        folder = runtime.ensure_identity(self.output, task)
        identity = base.fl.ExperimentConfig(**task["config"])
        checkpoint = base.experiments._checkpoint_path(folder / "checkpoints", identity)
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"damaged checkpoint")
        self.args.devices = ["cuda:0"]
        with mock.patch.object(base, "worker_environment", return_value={}), \
                mock.patch.object(base, "check_environment"), \
                mock.patch.object(base, "load_split", return_value=(object(), {"synthetic": True})), \
                mock.patch("torch.cuda.reset_peak_memory_stats"), \
                mock.patch.object(base.experiments, "run_measured_experiment", side_effect=AssertionError("restart")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_task(self.args, self.manifest, task, folder), 1)
        self.assertEqual(checkpoint.read_bytes(), b"damaged checkpoint")
        self.assertIn("refusing silent restart", runtime.read_json(folder / "failure.json")["message"])


class ReportTests(StudyFixture):
    def test_summary_is_readonly_and_never_imputes_missing(self):
        self.complete(self.tasks[0])
        self.failure(self.tasks[1], "infrastructure_oom")
        before = self.snapshot()
        with mock.patch.object(base, "load_split", side_effect=AssertionError("data")), \
                mock.patch.object(base.experiments, "run_measured_experiment", side_effect=AssertionError("train")):
            result = report.summarize(self.output)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(result["complete_tasks"], 1)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["decision"]["action"], "resolve_incomplete_or_invalid_evidence")
        self.assertNotIn("accuracy150", result["rows"][1])
        self.assertFalse(result["decision"]["candidates"][0]["selection_eligible"])

    def test_missing_loss_does_not_qualify(self):
        folder = self.complete(self.tasks[0])
        (folder / "observations.json").unlink()
        result = report.summarize(self.output)
        self.assertEqual(result["rows"][0]["status"], "invalid_evidence")

    def test_same_config_but_wrong_model_identity_rejected(self):
        folder = self.complete(self.tasks[0])
        task = deepcopy(self.tasks[0])
        task["model"] = "resnet18_gn2"
        base.write_json(folder / "task.json", task)
        self.assertEqual(report.summarize(self.output)["rows"][0]["status"], "invalid_evidence")

    def test_complete_healthy_winner_requires_matched_cnn_followup(self):
        for task in self.tasks:
            self.complete(task, .65 if task["candidate"]["candidate_id"] == "R4" else .60)
        result = report.summarize(self.output)
        self.assertEqual(result["complete_tasks"], 24)
        self.assertEqual(result["healthy_tasks"], 24)
        self.assertEqual(result["decision"]["best_healthy_resnet"], "R4")
        self.assertEqual(result["decision"]["action"], "needs_four_matched_cnn_runs_before_architecture_decision")
        self.assertEqual(result["decision"]["cnn_followup_parameters"]["lr_decay"], .995)
        self.assertFalse(result["decision"]["next_stage_started"])

    def test_R0_pairing_uses_each_seed_partition_and_inclusive_lines(self):
        for task in self.tasks:
            self.complete(task, .62 if task["candidate"]["candidate_id"] == "R0" else .60)
        result = report.summarize(self.output)
        self.assertTrue(result["decision"]["architecture_engineering_line_passed"])
        self.assertAlmostEqual(result["decision"]["paired_mean_gain_pp"], 2.)
        self.complete(self.tasks[4], .58)
        result = report.summarize(self.output)
        self.assertFalse(result["decision"]["architecture_engineering_line_passed"])

    def test_unhealthy_highest_accuracy_not_selected(self):
        for task in self.tasks:
            r4 = task["candidate"]["candidate_id"] == "R4"
            self.complete(task, .90 if r4 else .60, nonfinite=int(r4))
        result = report.summarize(self.output)
        self.assertEqual(result["healthy_tasks"], 20)
        self.assertEqual(result["decision"]["best_healthy_resnet"], "R0")
        self.assertFalse(result["decision"]["candidates"][-1]["selection_eligible"])

    def test_complete_snapshot_precedes_old_failure_without_repair_writes(self):
        self.complete(self.tasks[0])
        self.failure(self.tasks[0], "infrastructure_oom")
        before = self.snapshot()
        self.assertEqual(report.summarize(self.output)["rows"][0]["status"], "complete")
        self.assertEqual(before, self.snapshot())

    def test_numerical_failures_resolve_attempts_but_oom_does_not(self):
        for task in self.tasks:
            if task["candidate"]["candidate_id"] == "R4":
                self.failure(task, "algorithm_numerical")
            else:
                self.complete(task)
        result = report.summarize(self.output)
        self.assertEqual(result["status"], "resolved_with_numerical_failures")
        self.assertEqual(result["complete_tasks"], 20)
        self.assertEqual(result["decision"]["best_healthy_resnet"], "R0")
        self.failure(self.tasks[-1], "infrastructure_oom")
        self.assertEqual(report.summarize(self.output)["decision"]["action"], "resolve_incomplete_or_invalid_evidence")

    def test_summary_without_initialized_output_returns_copyable_error(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = runner.main(["--summary", "--output", str(self.output / "missing")])
        self.assertEqual(code, 2)
        self.assertIn("CIFAR_CLEAN_DIAGNOSTIC_END", stream.getvalue())
        self.assertFalse((self.output / "missing").exists())


class DispatchTests(StudyFixture):
    def test_controller_interrupt_terminates_and_joins_active_workers(self):
        proc = mock.Mock(pid=7101)
        proc.poll.side_effect = [KeyboardInterrupt(), None]
        self.args.devices = ["cuda:0"]
        with mock.patch.object(runner.subprocess, "Popen", return_value=proc), \
                mock.patch.object(runner.os, "killpg") as kill, redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                runner.execute(self.args, self.tasks[:2])
        kill.assert_called_once_with(7101, runner.signal.SIGTERM)
        proc.wait.assert_called_once()

    def test_oom_lane_is_not_reused_and_other_lane_continues(self):
        tasks = self.tasks[:3]
        launched = []

        def launch(command, **kwargs):
            task_id = command[command.index("--worker") + 1]
            device = command[-1]
            launched.append((task_id, device))
            code = int(len(launched) == 1)
            if code:
                self.failure(tasks[0], "infrastructure_oom")
            return SimpleNamespace(pid=len(launched), poll=lambda: code)

        with mock.patch.object(runner.subprocess, "Popen", side_effect=launch), redirect_stdout(io.StringIO()):
            runner.execute(self.args, tasks)
        self.assertEqual([d for _, d in launched], ["cuda:0", "cuda:1", "cuda:1"])

    def test_all_operational_failures_stop_dispatch_without_retry_loop(self):
        tasks = self.tasks[:3]
        launched = []

        def launch(command, **kwargs):
            index = len(launched)
            launched.append(command)
            self.failure(tasks[index], "infrastructure_or_execution")
            return SimpleNamespace(pid=index + 1, poll=lambda: 1)

        with mock.patch.object(runner.subprocess, "Popen", side_effect=launch), redirect_stdout(io.StringIO()):
            runner.execute(self.args, tasks)
        self.assertEqual(len(launched), 2)
        self.assertFalse((self.output / "tasks" / tasks[2]["task_id"]).exists())

    def test_completed_and_numerical_tasks_are_not_rerun(self):
        self.complete(self.tasks[0])
        self.failure(self.tasks[1], "algorithm_numerical")
        with mock.patch.object(runner.subprocess, "Popen", side_effect=AssertionError("rerun")), \
                redirect_stdout(io.StringIO()):
            runner.execute(self.args, self.tasks[:2])


if __name__ == "__main__":
    unittest.main()
