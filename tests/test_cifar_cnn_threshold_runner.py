"""Stage-3 integration boundaries; frozen scientific worker tests stay upstream."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import run_cifar_cnn_threshold_panel as runner
import tests.test_cifar_timing_runner as old_fixtures
from tests.test_cifar_six_pipeline import synthetic_run

base, runtime, protocol = runner.base, runner.runtime, runner.protocol


class ThresholdRunnerTests(unittest.TestCase):
    def setUp(self):
        self.fx = old_fixtures.RunnerFixture()
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.output = self.fx.root / "threshold"
        self.output.mkdir()
        self.reference = {"execution_environment": self.fx.environment}
        self.manifest = {"fingerprint": "threshold-manifest", "reference": self.reference,
                         "spec": {"dataset": "synthetic"}, "data_contract": {"synthetic": True}}
        self.tasks = []
        for point, warning, kappa, threshold in (("P0", 1.25, 1.25, 6.), ("P1", 1.5, 1.25, 6.),
                                               ("P2", 1.5, .85, 1.5), ("P3", 1.75, 1., 2.)):
            for partition in ("iid", "dirichlet"):
                for ratio in (0., .1, .7):
                    config = {**self.fx.tasks[0]["config"], "partition": partition, "malicious_ratio": ratio,
                        "detector_window": 20, "attack_start_round": 25, "detector_distance_threshold": warning,
                        "detector_drift_allowance": kappa, "detector_drift_threshold": threshold}
                    self.tasks.append({"task_id": f"threshold_{point}_{partition}_{ratio:g}", "phase": "validation",
                        "model": "v7_cnn", "method": "sm9rrs", "point": point,
                        "candidate": {"candidate_id": point}, "config": config})
        self.tasks = base.attach_fingerprints(self.tasks, self.manifest)
        base.write_json(self.output / "execution_environment.json", self.fx.environment)
        self.args = SimpleNamespace(output=self.output, timing_output=self.fx.output,
            clean_output=self.fx.clean_output, matched_output=self.fx.matched_output,
            data_dir=None, devices=["cuda:0", "cuda:1"], worker=self.tasks[0]["task_id"])

    def cli_paths(self):
        return ["--output", str(self.output), "--timing-output", str(self.args.timing_output),
                "--clean-output", str(self.args.clean_output), "--matched-output", str(self.args.matched_output)]

    def test_plan_only_is_24_new_fixed_runs_not_TPE_and_never_audits_or_trains(self):
        with mock.patch.object(protocol, "audit_reference", side_effect=AssertionError("reference IO")), \
                mock.patch.object(base, "load_split", side_effect=AssertionError("data")), \
                mock.patch("run_cifar_six_with_progress.discover_gpus", side_effect=AssertionError("GPU")), \
                redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runner.main(["--plan-only", *self.cli_paths()]), 0)
        plan = json.loads(stream.getvalue())
        self.assertEqual(plan["tasks"], 24)
        self.assertEqual([(p["warning"], p["kappa"], p["h"]) for p in plan["fixed_points"]],
                         [(1.25, 1.25, 6.), (1.5, 1.25, 6.), (1.5, .85, 1.5), (1.75, 1., 2.)])
        self.assertEqual((plan["K"], plan["attack_start_round"], plan["rounds"]), (20, 25, 150))
        self.assertTrue(plan["P0_is_six_new_runs"])
        self.assertEqual(plan["tpe_trials"], 0)
        self.assertFalse(plan["next_stage_automatic"])

    def test_worker_delegates_exact_new_identity_to_frozen_scientific_run_task(self):
        task = self.tasks[0]
        folder = runtime.ensure_identity(self.output, task)
        self.args.devices = ["cuda:1"]
        before = deepcopy(task)
        with mock.patch.object(protocol, "read_study", return_value=(self.manifest, self.tasks)) as study, \
                mock.patch.object(runner.timing, "run_task", return_value=75) as training:
            self.assertEqual(runner.worker(self.args), 75)
        study.assert_called_once_with(self.output, current_sources=True)
        training.assert_called_once_with(self.args, self.manifest, task, folder)
        self.assertEqual(before, task)
        self.args.worker = "old_timing_C"
        with mock.patch.object(protocol, "read_study", return_value=(self.manifest, self.tasks)), \
                mock.patch.object(runner.timing, "run_task", side_effect=AssertionError("wrong task")):
            with self.assertRaisesRegex(ValueError, "declared"):
                runner.worker(self.args)

    def test_real_inherited_run_task_keeps_calibration_checkpoint_identity_and_complete_reuse(self):
        task = self.tasks[0]
        runtime.ensure_identity(self.output, task)
        self.args.devices = ["cuda:1"]
        calibration = object()
        result = synthetic_run(base.fl.ExperimentConfig(**task["config"]), nonfinite=1)
        with mock.patch.object(protocol, "read_study", return_value=(self.manifest, self.tasks)), \
                mock.patch.object(base, "worker_environment", return_value=self.fx.environment), \
                mock.patch.object(base, "check_environment"), \
                mock.patch.object(base, "load_split", return_value=(SimpleNamespace(
                    calibration_dataset=calibration, main_dataset=object()), {"synthetic": True})), \
                mock.patch("torch.cuda.reset_peak_memory_stats"), \
                mock.patch("torch.cuda.max_memory_allocated", return_value=10), \
                mock.patch.object(runner.timing.clean, "observe", side_effect=AssertionError("clean observer")), \
                mock.patch.object(base.experiments, "run_measured_experiment", return_value=result) as train, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.worker(self.args), 0)
            self.assertEqual(runner.worker(self.args), 0)
        train.assert_called_once()
        self.assertIs(train.call_args.args[0], calibration)
        self.assertEqual(train.call_args.args[1].device, "cuda:1")
        self.assertEqual(train.call_args.kwargs["run_fingerprint"], task["fingerprint"])
        self.assertEqual(asdict(train.call_args.kwargs["checkpoint_identity_config"]), task["config"])
        self.assertTrue(train.call_args.kwargs["retain_success_checkpoint"])

    def test_new_entry_dispatches_all_P0_runs_and_pauses_failed_gpu(self):
        calls = []
        def launch(command, **kwargs):
            self.assertEqual(Path(command[2]).name, "run_cifar_cnn_threshold_panel.py")
            task_id = command[command.index("--worker") + 1]
            device = command[command.index("--devices") + 1]
            calls.append((task_id, device))
            if len(calls) == 1:
                task = self.tasks[0]
                base.write_json(self.output / "tasks" / task_id / "failure.json", {
                    "task_id": task_id, "task_fingerprint": task["fingerprint"], "kind": "infrastructure_oom"})
            return SimpleNamespace(pid=len(calls), poll=lambda: int(task_id == self.tasks[0]["task_id"]))
        with mock.patch.object(runner.subprocess, "Popen", side_effect=launch), \
                mock.patch.object(runner.time, "monotonic", return_value=30.), \
                mock.patch.object(runner.timing.clean, "progress") as progress, redirect_stdout(io.StringIO()):
            runner.execute(self.args, self.tasks)
        self.assertEqual(len(calls), 24)
        self.assertEqual(sum("_P0_" in task_id for task_id, _ in calls), 6)
        self.assertEqual([device for _, device in calls], ["cuda:0"] + ["cuda:1"] * 23)
        self.assertEqual(progress.call_args.args[-1], 24)
        with mock.patch.object(base, "checked_completed", return_value=None), \
                mock.patch.object(runtime, "terminal_failure", return_value={"kind": "algorithm_numerical"}), \
                mock.patch.object(runner.subprocess, "Popen", side_effect=AssertionError("numerical retry")), \
                redirect_stdout(io.StringIO()):
            runner.execute(self.args, self.tasks)

    def test_four_output_directories_reject_every_nested_or_linked_pair(self):
        paths = [self.output, self.args.timing_output, self.args.clean_output, self.args.matched_output]
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                for modified in (paths[i], paths[i] / "nested"):
                    aliases = list(paths)
                    aliases[j] = modified
                    with self.assertRaisesRegex(ValueError, "separate"):
                        runner.separate_outputs(*aliases)
        link = self.fx.root / "timing_alias"
        link.symlink_to(self.args.timing_output, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "separate"):
            runner.separate_outputs(link, *paths[1:])

    def test_parent_recovers_plan_initialization_without_mutating_three_references(self):
        references = [self.args.timing_output, self.args.clean_output, self.args.matched_output]
        before = {str(p): p.read_bytes() for root in references for p in root.rglob("*") if p.is_file()}
        base.write_json(self.output / "manifest.json", self.manifest)
        with mock.patch.object(protocol, "build_manifest", return_value=self.manifest), \
                mock.patch.object(protocol, "build_tasks", return_value=self.tasks), \
                mock.patch.object(protocol, "read_study", return_value=(self.manifest, self.tasks)), \
                mock.patch.object(runner, "execute"), \
                mock.patch("cifar_cnn_threshold_report.summarize", return_value={"status": "incomplete"}) as summarize, \
                mock.patch("cifar_cnn_threshold_report.print_summary"):
            self.assertEqual(runner.run_parent(self.args, {}, self.reference), 2)
        self.assertEqual(runtime.read_json(self.output / "task_plans/threshold.json")["tasks"], self.tasks)
        summarize.assert_called_once_with(self.output, *references)
        self.assertEqual(before, {str(p): p.read_bytes() for root in references for p in root.rglob("*") if p.is_file()})

    def test_summary_uses_new_compact_report_without_data_or_GPU(self):
        with mock.patch("cifar_cnn_threshold_report.summarize", return_value={"status": "complete"}) as summarize, \
                mock.patch("cifar_cnn_threshold_report.print_summary") as display, \
                mock.patch.object(base, "load_split", side_effect=AssertionError("data")), \
                mock.patch("run_cifar_six_with_progress.discover_gpus", side_effect=AssertionError("GPU")):
            self.assertEqual(runner.main(["--summary", *self.cli_paths()]), 0)
        summarize.assert_called_once_with(self.output, self.args.timing_output, self.args.clean_output, self.args.matched_output)
        display.assert_called_once_with({"status": "complete"})

    def test_reference_audit_failure_stops_before_any_GPU_probe(self):
        with mock.patch.object(protocol, "audit_reference", side_effect=ValueError("invalid old evidence")) as audit, \
                mock.patch("run_cifar_six_with_progress.discover_gpus", side_effect=AssertionError("GPU")), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "invalid old evidence"):
                runner.main(self.cli_paths())
        audit.assert_called_once_with(self.args.timing_output, self.args.clean_output, self.args.matched_output)

    def test_changed_existing_manifest_stops_before_GPU_or_new_training(self):
        base.write_json(self.output / "manifest.json", {"different": "identity"})
        with mock.patch.object(protocol, "audit_reference", return_value=self.reference), \
                mock.patch.object(protocol, "build_manifest", return_value=self.manifest), \
                mock.patch("run_cifar_six_with_progress.discover_gpus", side_effect=AssertionError("GPU")), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "identity changed"):
                runner.main(self.cli_paths())

    def test_interrupt_joins_new_worker_before_returning_control(self):
        proc = mock.Mock(pid=9341)
        proc.poll.side_effect = [KeyboardInterrupt(), None]
        self.args.devices = ["cuda:0"]
        previous = runner.signal.getsignal(runner.signal.SIGTERM)
        with mock.patch.object(runner.subprocess, "Popen", return_value=proc), \
                mock.patch.object(runner.os, "killpg") as kill, redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                runner.execute(self.args, self.tasks)
        kill.assert_called_once_with(proc.pid, runner.signal.SIGTERM)
        proc.wait.assert_called_once()
        self.assertEqual(runner.signal.getsignal(runner.signal.SIGTERM), previous)


if __name__ == "__main__":
    unittest.main()
