"""Recovery preserves validation evidence and never relaxes the original gate."""
from contextlib import redirect_stdout
import fcntl
import io
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import prepare_cifar_final_recovery as recovery
import run_cifar_six_from_scratch as runner
from sm9rrsfl import experiments, fl
from tests.test_cifar_six_pipeline import CONFIG, result_matrix, synthetic_run


def hashes(folder):
    return {str(path.relative_to(folder)): recovery.file_hash(path)
            for path in folder.rglob("*") if path.is_file()}


class FinalRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_temp = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.fixture_temp.name) / "source"
        cls.fixture.mkdir()
        (cls.fixture / ".runner.lock").touch()
        cls.spec = runner.load_spec(CONFIG)
        cls.manifest = runner.build_manifest(cls.spec, {"test": "no download"}, recovery.REPO)
        cls.tasks = runner.attach_fingerprints(runner.build_tasks(cls.spec, "validation"), cls.manifest)
        runner.write_json(cls.fixture / "manifest.json", cls.manifest)
        runner.write_json(cls.fixture / "validation_plan.json",
                          {"manifest_fingerprint": cls.manifest["fingerprint"], "tasks": cls.tasks})
        selected = {method: candidates[0]["candidate_id"] for method, candidates in cls.spec["candidates"].items()}
        cls.final_tasks = runner.attach_fingerprints(runner.build_tasks(cls.spec, "final", selected), cls.manifest)
        runner.write_json(cls.fixture / "final_plan.json", {"manifest_fingerprint": cls.manifest["fingerprint"],
                                                           "selected": selected, "tasks": cls.final_tasks})
        results = result_matrix(cls.spec, cls.tasks)
        for task in cls.tasks:
            folder = cls.fixture / "tasks" / task["task_id"]
            folder.mkdir(parents=True)
            runner.write_json(folder / "task.json", task)
            result = next(r for r in results[task["candidate"]["candidate_id"]]
                          if runner.semantic_config(r.config) == runner.semantic_config(task["config"]))
            experiments._write_completed_results_snapshot(folder, [result])
            (folder / "worker.log").write_text("original validation evidence\n")
        final_folder = cls.fixture / "tasks" / cls.final_tasks[0]["task_id"]
        final_folder.mkdir()
        experiments._write_completed_results_snapshot(final_folder,
            [synthetic_run(fl.ExperimentConfig(**cls.final_tasks[0]["config"]))])
        (final_folder / "checkpoints").mkdir()
        (final_folder / "worker.log").write_text("old formal run remains at source\n")
        (cls.fixture / "final_results").mkdir()
        (cls.fixture / "final_results" / "summary.csv").write_text("old official-test evidence\n")
        runner.write_json(cls.fixture / "execution_environment.json", {"numerical_mode": "original_runtime"})
        runner.write_json(cls.fixture / "validation_summary.json", {"stale": True})

    @classmethod
    def tearDownClass(cls):
        cls.fixture_temp.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.destination = self.root / "recovered"
        shutil.copytree(self.fixture, self.source)

    def make_failed(self, task):
        folder = self.source / "tasks" / task["task_id"]
        (folder / experiments.COMPLETED_RESULTS_SNAPSHOT).unlink()
        runner.write_json(folder / "failure.json", {"task_id": task["task_id"],
                          "exception": "FloatingPointError", "message": "retained original failure"})

    def test_default_audit_is_read_only_and_recomputes_selection(self):
        before = hashes(self.source)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(recovery.main(["--source", str(self.source)]), 0)
        self.assertEqual(hashes(self.source), before)
        report, summary, tasks = recovery.audit(self.source)
        self.assertEqual(report["validation_counts"], {"complete": 84, "failed": 0, "pending": 0})
        self.assertTrue(summary["ours_health_passed"])
        self.assertTrue(report["selection_changes"])
        self.assertEqual(report["old_final_artifacts"]["planned_tasks"], 180)
        self.assertEqual(report["old_final_artifacts"]["task_directories"], 1)
        self.assertEqual(report["old_final_artifacts"]["completed_snapshot_files"], 1)
        self.assertEqual(report["old_final_artifacts"]["checkpoint_directories"], 1)
        self.assertTrue(report["old_final_artifacts"]["final_results_exists"])
        self.assertEqual(tasks, self.tasks)
        self.assertFalse(self.destination.exists())

    def test_apply_copies_only_validation_and_preserves_source_and_task_identity(self):
        before = hashes(self.source)
        with recovery.source_lock(self.source):
            report, summary, tasks = recovery.audit(self.source)
            result = recovery.apply_copy(self.source, self.destination, report, summary, tasks)
        self.assertEqual(hashes(self.source), before)
        for name in ("final_plan.json", "final_results", "final_summary.json", "recovery_incomplete.json"):
            self.assertFalse((self.destination / name).exists())
        self.assertEqual({p.name for p in (self.destination / "tasks").iterdir()}, {t["task_id"] for t in tasks})
        self.assertEqual(recovery.read_json(self.destination / "manifest.json"), self.manifest)
        copied_results, statuses = runner.collect_results(self.destination, tasks)
        copied_selection = runner.select_validation(self.spec, copied_results, tasks)
        self.assertTrue(all(row["status"] == "complete" for row in statuses))
        self.assertEqual(copied_selection["selected"], report["new_selected"])
        new_plan = {"manifest_fingerprint": self.manifest["fingerprint"], "selected": report["new_selected"],
                    "tasks": runner.attach_fingerprints(runner.build_tasks(self.spec, "final", report["new_selected"]), self.manifest)}
        self.assertEqual(report["intended_final_plan_digest"], runner.digest(new_plan))
        self.assertEqual(recovery.read_json(self.destination / "validation_summary.json")["selected"], report["new_selected"])
        for record in result["copied_files"]:
            self.assertEqual(recovery.file_hash(self.destination / record["path"]), record["sha256"])
        self.assertEqual(result["copied_file_count"], len(result["copied_files"]))

    def test_original_baseline_failures_are_preserved_and_do_not_veto_ours(self):
        for task in self.tasks:
            if task["method"] == "vert":
                self.make_failed(task)
        report, summary, tasks = recovery.audit(self.source)
        self.assertEqual(report["validation_counts"], {"complete": 60, "failed": 24, "pending": 0})
        self.assertEqual(report["status"], "ready_to_copy_validation")
        self.assertEqual(summary["methods"]["vert"]["selection_status"], "fixed_fallback_unqualified")
        recovery.apply_copy(self.source, self.destination, report, summary, tasks)
        copied = recovery.read_json(self.destination / "validation_summary.json")
        failed = next(row for row in copied["tasks"] if row["status"] == "failed")
        self.assertTrue(Path(failed["failure_record"]).is_file())
        self.assertTrue(str(failed["failure_record"]).startswith(str(self.destination.resolve())))

    def test_original_final_only_runner_accepts_copy_and_freezes_expected_180_tasks(self):
        report, summary, tasks = recovery.audit(self.source)
        recovery.apply_copy(self.source, self.destination, report, summary, tasks)
        original_validation = hashes(self.destination / "tasks")
        args = SimpleNamespace(config=CONFIG, output=self.destination, devices=["cuda:0"],
                               data_dir=None, phase="final")

        def simulate_training(arguments, planned, output):
            self.assertEqual(arguments.phase, "final")
            self.assertEqual(len(planned), 180)
            self.assertTrue(all(task["phase"] == "final" for task in planned))
            self.assertEqual(output, self.destination.resolve())
            for task in planned:
                folder = output / "tasks" / task["task_id"]
                folder.mkdir(parents=True)
                runner.write_json(folder / "task.json", task)
                experiments._write_completed_results_snapshot(folder,
                    [synthetic_run(fl.ExperimentConfig(**task["config"]))])
            return 0

        with mock.patch.object(runner, "load_split", return_value=(object(), self.manifest["data_contract"])) as load, \
                mock.patch.object(runner, "execute_phase", side_effect=simulate_training) as execute, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_parent(args), 0)
        load.assert_called_once_with(self.spec, None)
        self.assertEqual(execute.call_count, 1)
        final_plan = recovery.read_json(self.destination / "final_plan.json")
        self.assertEqual(len(final_plan["tasks"]), 180)
        self.assertEqual(final_plan["selected"], report["new_selected"])
        self.assertEqual(runner.digest(final_plan), report["intended_final_plan_digest"])
        final_report = recovery.read_json(self.destination / "final_summary.json")
        self.assertEqual(final_report["status"], "completed")
        self.assertTrue((self.destination / "final_results" / "aggregate.csv").is_file())
        after = hashes(self.destination / "tasks")
        self.assertEqual({path: after[path] for path in original_validation}, original_validation)

    def test_failed_ours_never_bypasses_original_health_gate(self):
        for task in self.tasks:
            if task["method"] == "sm9rrs":
                self.make_failed(task)
        report, summary, tasks = recovery.audit(self.source)
        self.assertFalse(summary["ours_health_passed"])
        with self.assertRaisesRegex(ValueError, "blocked"):
            recovery.apply_copy(self.source, self.destination, report, summary, tasks)
        self.assertFalse(self.destination.exists())

    def test_damaged_snapshot_and_pending_task_block_apply(self):
        folder = self.source / "tasks" / self.tasks[0]["task_id"]
        snapshot = folder / experiments.COMPLETED_RESULTS_SNAPSHOT
        for damaged in (True, False):
            with self.subTest(damaged=damaged):
                if damaged:
                    snapshot.write_bytes(b"damaged")
                else:
                    snapshot.unlink()
                report, summary, tasks = recovery.audit(self.source)
                self.assertEqual(report["status"], "blocked")
                self.assertTrue(any(row.get("task_id") == self.tasks[0]["task_id"] for row in report["blockers"]))
                with self.assertRaises(ValueError):
                    recovery.apply_copy(self.source, self.destination, report, summary, tasks)

    def test_bad_completed_identity_or_failed_identity_is_not_treated_as_valid_failure(self):
        task = self.tasks[0]
        folder = self.source / "tasks" / task["task_id"]
        runner.write_json(folder / "task.json", {"wrong": "identity"})
        report, _, _ = recovery.audit(self.source)
        self.assertEqual(report["status"], "blocked")
        self.make_failed(task)
        report, _, _ = recovery.audit(self.source)
        self.assertEqual(report["status"], "blocked")

    def test_changed_source_or_manifest_is_rejected(self):
        with mock.patch.object(runner, "source_hashes", return_value={}):
            with self.assertRaisesRegex(ValueError, "manifest"):
                recovery.audit(self.source)
        manifest = recovery.read_json(self.source / "manifest.json")
        manifest["fingerprint"] = "wrong"
        runner.write_json(self.source / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "manifest"):
            recovery.audit(self.source)

    def test_arbitrary_old_plan_damage_is_not_misclassified_as_selection_change(self):
        old = recovery.read_json(self.source / "final_plan.json")
        old["tasks"][0]["fingerprint"] = "wrong"
        runner.write_json(self.source / "final_plan.json", old)
        with self.assertRaisesRegex(ValueError, "old final_plan"):
            recovery.audit(self.source)

    def test_validation_plan_identity_is_verified(self):
        plan = recovery.read_json(self.source / "validation_plan.json")
        plan["tasks"].pop()
        runner.write_json(self.source / "validation_plan.json", plan)
        with self.assertRaisesRegex(ValueError, "validation_plan"):
            recovery.audit(self.source)

    def test_identical_current_selection_refuses_unnecessary_recovery(self):
        report, _, _ = recovery.audit(self.source)
        selected = report["new_selected"]
        runner.write_json(self.source / "final_plan.json", {"manifest_fingerprint": self.manifest["fingerprint"],
                          "selected": selected, "tasks": runner.attach_fingerprints(
                              runner.build_tasks(self.spec, "final", selected), self.manifest)})
        report, _, _ = recovery.audit(self.source)
        self.assertFalse(report["selection_changes"])
        self.assertEqual(report["status"], "blocked")

    def test_source_lock_and_missing_lock_fail_without_writing_source(self):
        with (self.source / ".runner.lock").open("rb") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "active"):
                with recovery.source_lock(self.source):
                    self.fail("lock should be refused")
        (self.source / ".runner.lock").unlink()
        with self.assertRaises(FileNotFoundError):
            with recovery.source_lock(self.source):
                self.fail("missing lock should be refused")
        self.assertFalse((self.source / ".runner.lock").exists())

    def test_existing_and_nested_destinations_are_never_overwritten(self):
        self.destination.mkdir()
        (self.destination / "keep").write_text("user file")
        for dest in (self.destination, self.source, self.source / "nested"):
            with self.subTest(dest=dest), self.assertRaises(ValueError):
                recovery.check_destination(self.source, dest)
        self.assertEqual((self.destination / "keep").read_text(), "user file")

    def test_disk_space_and_copy_verification_fail_without_publishing(self):
        report, summary, tasks = recovery.audit(self.source)
        before = hashes(self.source)
        with mock.patch.object(recovery.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(ValueError, "disk space"):
                recovery.apply_copy(self.source, self.destination, report, summary, tasks)
        def bad_copy(source, target):
            target.write_bytes(b"bad copy")
        with mock.patch.object(recovery.shutil, "copy2", side_effect=bad_copy):
            with self.assertRaisesRegex(ValueError, "verification"):
                recovery.apply_copy(self.source, self.destination, report, summary, tasks)
        self.assertFalse(self.destination.exists())
        self.assertFalse(list(self.root.glob(".cifar-final-recovery-*")))
        self.assertEqual(hashes(self.source), before)

    def test_no_overwrite_even_if_destination_appears_during_copy(self):
        report, summary, tasks = recovery.audit(self.source)
        original_copy = shutil.copy2
        def race_copy(source, target):
            self.destination.mkdir(exist_ok=True)
            (self.destination / "keep").write_text("concurrent user file")
            return original_copy(source, target)
        with mock.patch.object(recovery.shutil, "copy2", side_effect=race_copy):
            with self.assertRaises(FileExistsError):
                recovery.apply_copy(self.source, self.destination, report, summary, tasks)
        self.assertEqual((self.destination / "keep").read_text(), "concurrent user file")
        self.assertFalse((self.destination / "manifest.json").exists())

    def test_symbolic_artifact_refuses_copy(self):
        (self.source / "tasks" / self.tasks[0]["task_id"] / "external").symlink_to(self.root)
        report, summary, tasks = recovery.audit(self.source)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            recovery.apply_copy(self.source, self.destination, report, summary, tasks)


if __name__ == "__main__":
    unittest.main()
