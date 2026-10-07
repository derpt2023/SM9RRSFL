"""Reference audit, four-task pairing and independent worker dispatch boundaries."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import run_cifar_matched_cnn as runner
import cifar_matched_cnn_report as report
from tests.test_cifar_diagnostic import StudyFixture

base, runtime, clean = runner.base, runner.runtime, runner.clean


class MatchedFixture(StudyFixture):
    def setUp(self):
        super().setUp()
        self.reference_output = self.output
        self.original_tasks = self.tasks
        for task in self.original_tasks:
            self.complete(task, .65 if task["candidate"]["candidate_id"] == "R3" else .60)
        self.environment = {"actual_compute_device": {"name": "synthetic", "compute_capability": [8, 9],
                                                      "total_memory_bytes": 24 * 2**30},
                            "environment": {}, "driver_versions": []}
        metadata = {k: v for k, v in self.environment.items() if k != "driver_versions"}
        for task in self.original_tasks:
            base.write_json(self.reference_output / "tasks" / task["task_id"] / "environment.json", metadata)
        base.write_json(self.reference_output / "execution_environment.json", self.environment)
        self.reference_fp = self.manifest["fingerprint"]
        self.reference = runner.audit_reference(self.reference_output, expected_fingerprint=self.reference_fp)
        patcher = mock.patch.object(runner, "REFERENCE_FINGERPRINT", self.reference_fp)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.spec = runtime.read_json(runner.DEFAULT_CONFIG)
        self.spec["reference_manifest_fingerprint"] = self.reference_fp
        # Separate study with independent cleanup; the original remains read-only.
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name)
        dataset = runtime.read_json(self.reference_output / "manifest.json")["spec"]["dataset"]
        self.manifest = runner.build_manifest(self.spec, self.reference, dataset)
        self.tasks = runner.build_tasks(self.manifest)
        base.write_json(self.output / "manifest.json", self.manifest)
        runtime.save_plan(self.output, "matched_cnn", self.tasks, self.manifest)
        base.write_json(self.output / "execution_environment.json", self.environment)
        self.args = SimpleNamespace(output=self.output, reference_output=self.reference_output,
                                    data_dir=None, devices=["cuda:0", "cuda:1"], worker=None)

    def reference_snapshot(self):
        return {str(p.relative_to(self.reference_output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.reference_output.rglob("*") if p.is_file()}


class ReferenceTests(MatchedFixture):
    def test_each_reference_task_environment_is_verified_and_hashed(self):
        task = self.original_tasks[16]
        relative = "tasks/" + task["task_id"] + "/environment.json"
        self.assertIn(relative, self.reference["evidence_sha256"])
        metadata = runtime.read_json(self.reference_output / relative)
        metadata["actual_compute_device"]["name"] = "different GPU"
        base.write_json(self.reference_output / relative, metadata)
        with self.assertRaisesRegex(ValueError, "numerical environment"):
            runner.audit_reference(self.reference_output, expected_fingerprint=self.reference_fp)

    def test_environment_normalization_matches_frozen_base_policy(self):
        metadata = {"environment": {"CUDA_VISIBLE_DEVICES": "GPU-a", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "OMP": "1"},
            "requested_device": "cuda:3", "actual_compute_device": {"name": "GPU", "uuid": "GPU-a", "logical_device": "cuda:3"},
            "nvidia": {"gpus": [{"driver_version": "test-driver"}]},
            "torch": {"version": "test", "logical_cuda_devices": [{"device": 3}], "logical_cuda_devices_status": "ok"}}
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            base.check_environment(Path(folder), metadata)
            saved = runtime.read_json(Path(folder) / "execution_environment.json")
        self.assertEqual(runner.normalized_environment(metadata), saved)

    def test_actual_reference_audit_and_summary_are_readonly(self):
        before = self.reference_snapshot()
        with mock.patch.object(base, "load_split", side_effect=AssertionError("data load")), \
                mock.patch.object(base.experiments, "run_measured_experiment", side_effect=AssertionError("training")):
            reference = runner.audit_reference(self.reference_output, expected_fingerprint=self.reference_fp)
            result = report.summarize(self.output, self.reference_output)
        self.assertEqual(reference, self.reference)
        self.assertEqual(before, self.reference_snapshot())
        self.assertTrue(result["reference_verified"])
        self.assertEqual(result["complete_tasks"], 0)

    def test_wrong_reference_fingerprint_or_unhealthy_reference_blocked(self):
        with self.assertRaisesRegex(ValueError, "reviewed"):
            runner.audit_reference(self.reference_output, expected_fingerprint="wrong")
        task = self.original_tasks[0]
        folder = self.reference_output / "tasks" / task["task_id"]
        results = base.experiments.load_completed_results_snapshot(folder)
        from dataclasses import replace
        base.experiments._write_completed_results_snapshot(folder, [replace(results[0], nonfinite_updates=1)])
        with self.assertRaisesRegex(ValueError, "24 complete healthy"):
            runner.audit_reference(self.reference_output, expected_fingerprint=self.reference_fp)

    def test_changed_reference_loss_file_prevents_cached_comparison(self):
        task = self.original_tasks[0]
        path = self.reference_output / "tasks" / task["task_id"] / "observations.json"
        rows = runtime.read_json(path)
        rows["rounds"][150]["calibration_loss"] += .001
        base.write_json(path, rows)
        for task in self.tasks:
            self.complete(task)
        result = report.summarize(self.output, self.reference_output)
        self.assertFalse(result["reference_verified"])
        self.assertEqual(result["decision"]["action"], "resolve_changed_or_invalid_reference")

    def test_reference_mutation_during_audit_rejected(self):
        real = runner.reference_hashes
        calls = []

        def hashes(*args):
            result = real(*args)
            if calls:
                result["observations.json"] = "concurrent edit"
            calls.append(True)
            return result

        with mock.patch.object(runner, "reference_hashes", side_effect=hashes):
            with self.assertRaisesRegex(ValueError, "changed while"):
                runner.audit_reference(self.reference_output, expected_fingerprint=self.reference_fp)

    def test_source_identity_keeps_old_53_files_unchanged(self):
        old = clean.source_hashes()
        self.assertEqual(len(old), 53)
        self.assertEqual({k: runner.source_hashes()[k] for k in old}, old)


class PairingTests(MatchedFixture):
    def test_four_CNN_tasks_match_every_R3_config_field(self):
        self.assertEqual(len(self.tasks), 4)
        for new, old in zip(self.tasks, self.reference["r3_tasks"]):
            self.assertEqual(new["config"], old["config"])
            self.assertEqual(new["model"], "v7_cnn")
            self.assertNotEqual(new["fingerprint"], old["fingerprint"])
            self.assertEqual(new["reference_task_fingerprint"], old["fingerprint"])
            self.assertEqual(new["config"]["local_epochs"], 2)
        self.assertEqual(runner.read_study(self.output, current_sources=True)[1], self.tasks)

    def test_schema_rejects_search_or_new_hyperparameter(self):
        for key, val in (("local_epochs", 3), ("model", "resnet18_gn2"), ("lr", .1)):
            spec = deepcopy(self.spec)
            spec["setting"][key] = val
            with self.assertRaises(ValueError):
                runner.validate_spec(spec)

    def test_plan_only_does_not_require_reference_or_GPU(self):
        with mock.patch.object(runner, "REFERENCE_FINGERPRINT", runtime.read_json(runner.DEFAULT_CONFIG)["reference_manifest_fingerprint"]), \
                mock.patch.object(runner, "audit_reference", side_effect=AssertionError("read reference")), \
                mock.patch("run_cifar_six_with_progress.discover_gpus", side_effect=AssertionError("GPU")), \
                redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runner.main(["--plan-only"]), 0)
        self.assertIn('"tasks": 4', stream.getvalue())

    def test_worker_only_delegates_C3_to_frozen_clean_training(self):
        task = self.tasks[0]
        runtime.ensure_identity(self.output, task)
        self.args.devices = ["cuda:0"]
        self.args.worker = task["task_id"]
        with mock.patch.object(clean, "run_task", return_value=0) as train:
            self.assertEqual(runner.worker(self.args), 0)
        self.assertEqual(train.call_args.args[2], task)
        self.assertEqual(train.call_args.args[1]["spec"]["dataset"]["train_samples"], 50000)
        self.args.worker = self.reference["r3_tasks"][0]["task_id"]
        with self.assertRaisesRegex(ValueError, "declared CNN"):
            runner.worker(self.args)

    def test_environment_change_rejected_before_worker_training(self):
        task = self.tasks[0]
        runtime.ensure_identity(self.output, task)
        self.args.devices = ["cuda:0"]
        self.args.worker = task["task_id"]
        base.write_json(self.output / "execution_environment.json", {"different": "GPU"})
        with mock.patch.object(clean, "run_task", side_effect=AssertionError("training")):
            with self.assertRaisesRegex(ValueError, "environment"):
                runner.worker(self.args)

    def test_output_overlap_and_orphans_refused(self):
        for path in (self.reference_output, self.reference_output / "child", self.reference_output.parent):
            self.args.output = path
            with self.assertRaisesRegex(ValueError, "separate"):
                runner.run_parent(self.args, self.spec, self.reference)
        self.args.output = self.output
        task = self.tasks[0]
        folder = self.output / "tasks" / task["task_id"]
        folder.mkdir(parents=True)
        (folder / "checkpoint.pickle").write_bytes(b"preserve")
        with mock.patch.object(runner, "execute", side_effect=AssertionError("training")):
            with self.assertRaisesRegex(ValueError, "orphan"):
                runner.run_parent(self.args, self.spec, self.reference)

    def test_initialization_resume_and_end_to_end_summary_do_not_write_reference(self):
        before = self.reference_snapshot()
        (self.output / "task_plans/matched_cnn.json").unlink()

        def finish(args, tasks):
            for task in tasks:
                self.complete(task, accuracy=.67)

        with mock.patch.object(runner, "execute", side_effect=finish), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_parent(self.args, self.spec, self.reference), 0)
        result = report.summarize(self.output, self.reference_output)
        self.assertEqual(result["healthy_tasks"], 4)
        self.assertAlmostEqual(result["decision"]["paired_mean_R3_minus_C3_pp"], -2.)
        self.assertEqual(before, self.reference_snapshot())
        with mock.patch.object(runner.subprocess, "Popen", side_effect=AssertionError("retrain")), redirect_stdout(io.StringIO()):
            runner.execute(self.args, self.tasks)


class DispatchTests(MatchedFixture):
    def test_new_entry_only_and_failed_lane_is_not_reused(self):
        launched = []

        def launch(command, **kwargs):
            self.assertEqual(Path(command[2]).name, "run_cifar_matched_cnn.py")
            task_id = command[command.index("--worker") + 1]
            self.assertIn("_C3_", task_id)
            launched.append(command[-1])
            code = int(len(launched) == 1)
            if code:
                self.failure(self.tasks[0], "infrastructure_oom")
            return SimpleNamespace(pid=len(launched), poll=lambda: code)

        with mock.patch.object(runner.subprocess, "Popen", side_effect=launch), redirect_stdout(io.StringIO()):
            runner.execute(self.args, self.tasks)
        self.assertEqual(launched, ["cuda:0", "cuda:1", "cuda:1", "cuda:1"])

    def test_interrupt_joins_active_worker(self):
        proc = mock.Mock(pid=7103)
        proc.poll.side_effect = [KeyboardInterrupt(), None]
        self.args.devices = ["cuda:0"]
        with mock.patch.object(runner.subprocess, "Popen", return_value=proc), \
                mock.patch.object(runner.os, "killpg") as kill, redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                runner.execute(self.args, self.tasks)
        kill.assert_called_once_with(7103, runner.signal.SIGTERM)
        proc.wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
