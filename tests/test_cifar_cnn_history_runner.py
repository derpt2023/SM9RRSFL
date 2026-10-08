"""New worker isolation and whole-run history context for the 12-task ablation."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import unittest

import run_cifar_cnn_history_panel as runner
from tests.test_cifar_cnn_history_protocol import HistoryFixture

base, runtime, protocol = runner.base, runner.runtime, runner.protocol


class HistoryRunnerTests(HistoryFixture):
    def setUp(self):
        super().setUp()
        self.args = SimpleNamespace(output=self.output, threshold_output=self.threshold_output,
            timing_output=self.timing_output, clean_output=self.clean_output, matched_output=self.matched_output,
            devices=["cuda:0", "cuda:1"], data_dir=None, worker=None)
        self.config_path = self.output / "test_config.json"
        base.write_json(self.config_path, self.spec)

    def cli_paths(self):
        return ["--config", str(self.config_path), "--output", str(self.output),
            "--threshold-output", str(self.threshold_output), "--timing-output", str(self.timing_output),
            "--clean-output", str(self.clean_output), "--matched-output", str(self.matched_output)]

    def test_plan_only_exposes_variant_and_clean_freeze_without_reference_data_or_GPU(self):
        with mock.patch.object(protocol, "audit_reference", side_effect=AssertionError("reference read")), \
                mock.patch.object(base, "load_split", side_effect=AssertionError("data")), \
                mock.patch("run_cifar_six_with_progress.discover_gpus", side_effect=AssertionError("GPU")), \
                redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runner.main(["--plan-only", *self.cli_paths()]), 0)
        plan = json.loads(stream.getvalue())
        self.assertEqual(plan["tasks"], 12)
        self.assertEqual(plan["arms"], protocol.ARMS)
        self.assertTrue(plan["freeze_applies_to_clean"])
        self.assertTrue(plan["both_arms_fresh"])
        self.assertEqual((plan["K"], plan["attack_start_round"]), (20, 25))
        self.assertFalse(plan["next_stage_automatic"])
        self.assertEqual(plan["tpe_trials"], 0)

    def test_entire_frozen_run_task_including_restore_is_inside_correct_variant_context(self):
        cls = runner.history_runtime.LongitudinalSVDDetector
        original = cls.commit
        self.args.devices = ["cuda:0"]
        for arm in ("H0", "H1"):
            task = next(t for t in self.tasks if t["arm"] == arm and t["config"]["malicious_ratio"] == 0)
            folder = runtime.ensure_identity(self.output, task)
            self.args.worker = task["task_id"]
            def inside(args, manifest, current, directory):
                self.assertEqual(current, task)
                self.assertEqual(manifest, self.manifest)
                self.assertEqual(directory, folder)
                self.assertTrue(runner.history_runtime._active)
                self.assertEqual(cls.commit is original, arm == "H0")
                self.assertNotEqual(current["fingerprint"], current["original_p0_fingerprint"])
                return 75
            with mock.patch.object(runner.timing, "run_task", side_effect=inside):
                self.assertEqual(runner.worker(self.args), 75)
            self.assertIs(cls.commit, original)
            self.assertFalse(runner.history_runtime._active)
        with mock.patch.object(runner.timing, "run_task", side_effect=ValueError("synthetic restore failure")):
            with self.assertRaisesRegex(ValueError, "restore failure"):
                runner.worker(self.args)
        self.assertIs(cls.commit, original)
        self.assertFalse(runner.history_runtime._active)

    def test_new_worker_dispatches_12_fresh_tasks_and_quarantines_operational_failure(self):
        launches = []
        def launch(command, **kwargs):
            self.assertEqual(Path(command[2]).name, "run_cifar_cnn_history_panel.py")
            task_id = command[command.index("--worker") + 1]
            device = command[command.index("--devices") + 1]
            launches.append((task_id, device))
            code = int(len(launches) == 1)
            if code:
                task = self.tasks[0]
                base.write_json(self.output / "tasks" / task_id / "failure.json", {
                    "task_id": task_id, "task_fingerprint": task["fingerprint"], "kind": "infrastructure_oom"})
            return SimpleNamespace(pid=len(launches), poll=lambda: code)
        with mock.patch.object(runner.subprocess, "Popen", side_effect=launch), \
                mock.patch.object(runner.time, "monotonic", return_value=30.), \
                mock.patch.object(runner.timing.clean, "progress") as progress, redirect_stdout(io.StringIO()):
            runner.execute(self.args, self.tasks)
        self.assertEqual(len(launches), 12)
        self.assertEqual(sum("_H0_" in task for task, _ in launches), 6)
        self.assertEqual(sum("_H1_" in task for task, _ in launches), 6)
        self.assertEqual([device for _, device in launches], ["cuda:0"] + ["cuda:1"] * 11)
        self.assertEqual(progress.call_args.args[-1], 12)

    def test_all_five_output_directories_and_symlink_aliases_are_isolated(self):
        paths = [self.output, self.threshold_output, self.timing_output, self.clean_output, self.matched_output]
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                altered = list(paths)
                altered[j] = paths[i] / "nested"
                with self.assertRaisesRegex(ValueError, "separate"):
                    runner.separate_outputs(*altered)
        alias = self.output / "reference_alias"
        alias.symlink_to(self.threshold_output, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "separate"):
            runner.separate_outputs(alias, *paths[1:])

    def test_parent_resumes_missing_plan_and_summary_without_writing_old_78_tasks(self):
        before = self.original_snapshot()
        (self.output / "task_plans/history.json").unlink()
        with mock.patch.object(runner, "execute") as execute, redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_parent(self.args, self.spec, self.reference), 2)
        execute.assert_called_once()
        self.assertEqual(len(execute.call_args.args[1]), 12)
        self.assertEqual(runtime.read_json(self.output / "task_plans/history.json")["tasks"], self.tasks)
        self.assertEqual(before, self.original_snapshot())
        self.assertTrue((self.output / "history_summary.json").exists())

    def test_summary_is_five_path_readonly_and_invalid_reference_stops_before_GPU(self):
        with mock.patch("cifar_cnn_history_report.summarize", return_value={"status": "incomplete"}) as summarize, \
                mock.patch("cifar_cnn_history_report.print_summary"), \
                mock.patch.object(base, "load_split", side_effect=AssertionError("data")), \
                mock.patch("run_cifar_six_with_progress.discover_gpus", side_effect=AssertionError("GPU")):
            self.assertEqual(runner.main(["--summary", *self.cli_paths()]), 2)
        summarize.assert_called_once_with(*(p.resolve() for p in (self.output, self.threshold_output,
            self.timing_output, self.clean_output, self.matched_output)))
        with mock.patch.object(protocol, "audit_reference", side_effect=ValueError("old evidence invalid")), \
                mock.patch("run_cifar_six_with_progress.discover_gpus", side_effect=AssertionError("GPU")), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "old evidence invalid"):
                runner.main(self.cli_paths())

    def test_interrupt_cleans_up_new_worker_and_restores_signal_handler(self):
        proc = mock.Mock(pid=9504)
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
