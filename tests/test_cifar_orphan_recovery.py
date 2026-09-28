"""Recovery of ENOSPC initialization remnants; no CUDA, data or training."""
from contextlib import redirect_stdout, redirect_stderr
import errno
import fcntl
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import repair_cifar_orphan_tasks as recovery
from tests import test_cifar_interactive as fixtures


class OrphanRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.InteractiveTests.setUpClass()

    def setUp(self):
        self.fixture = fixtures.InteractiveTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.output, self.config = self.fixture.output.resolve(), self.fixture.config.resolve()
        # Process listing is sandbox-restricted on the local macOS host. Feed
        # explicit process inventories; locks and all filesystem I/O are real.
        process_listing = mock.patch.object(recovery.subprocess, "run", return_value=SimpleNamespace(stdout=""))
        process_listing.start()
        self.addCleanup(process_listing.stop)
        (self.output / ".runner.lock").touch()
        (self.output / "tasks").mkdir()
        selected = {**fixtures.BASELINES, "sm9rrs": fixtures.OURS}
        self.plan = recovery.runner.final_plan(self.fixture.spec, self.fixture.manifest, selected)
        recovery.runner.write_json(self.output / "final_plan.json", self.plan)
        decision = {**recovery.controller.decision_basis(self.fixture.manifest, self.fixture.report),
                    "controller_source_sha256": recovery.controller.controller_hash(),
                    "created_at_utc": "2026-09-28T06:00:00+00:00"}
        recovery.runner.write_json(self.output / recovery.controller.DECISION_FILE, decision)

    def folder(self, index=0):
        path = self.output / "tasks" / self.plan["tasks"][index]["task_id"]
        path.mkdir(exist_ok=True)
        return path

    def orphan(self, index=0):
        path = self.folder(index)
        (path / "worker.log").write_text("OSError: [Errno 28] No space left on device\n")
        (path / ".task.json.123.tmp").write_text('{"task_id":')
        recovery.runner.write_json(path / "failure.json", {
            "task_id": self.plan["tasks"][index]["task_id"], "execution_context": None,
            "exception": "ValueError", "message": "task artifacts exist without their immutable identity"})
        return path

    def audit(self):
        return recovery.audit(self.output, self.config)

    def tree(self):
        return {str(p.relative_to(self.output)): p.read_bytes()
                for p in self.output.rglob("*") if p.is_file()}

    def invoke(self, *flags):
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            code = recovery.main(["--output", str(self.output), "--config", str(self.config), *flags])
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_is_read_only_and_only_identifies_preinit_remnants(self):
        self.orphan()
        (self.folder(1) / "worker.log").write_text("startup only\n")
        self.folder(2)
        before = self.tree()
        code, out, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertIn('"quarantinable_count": 1', out)
        self.assertIn('"initializable": 2', out)
        self.assertEqual(before, self.tree())
        self.assertFalse((self.output / "orphan_recovery").exists())

    def test_real_worker_rejects_remnants_then_reinitializes_after_quarantine(self):
        folder = self.orphan()
        task = self.plan["tasks"][0]
        args = SimpleNamespace(output=self.output, phase="final", worker=task["task_id"],
                               devices=["cuda:0"], data_dir=None)
        with self.assertRaisesRegex(ValueError, "without their immutable identity"):
            recovery.runner.run_worker(args)
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        report = recovery.quarantine(self.output, self.config, self.audit())
        backup = Path(report["quarantine_dir"]) / task["task_id"]
        self.assertFalse(folder.exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in backup.iterdir()})
        with mock.patch.object(recovery.runner, "worker_environment", side_effect=RuntimeError("reached_environment")), \
                mock.patch.object(recovery.runner, "load_split") as data:
            with self.assertRaisesRegex(RuntimeError, "reached_environment"):
                recovery.runner.run_worker(args)
        self.assertEqual(json.loads((folder / "task.json").read_text()), task)
        data.assert_not_called()

    def test_apply_preserves_valid_snapshot_checkpoint_and_all_frozen_metadata(self):
        self.orphan()
        folder = self.folder(1)
        task = self.plan["tasks"][1]
        recovery.runner.write_json(folder / "task.json", task)
        result = fixtures.synthetic_run(recovery.runner.fl.ExperimentConfig(**task["config"]))
        recovery.runner.experiments._write_completed_results_snapshot(folder, [result])
        (folder / "checkpoints").mkdir()
        (folder / "checkpoints" / "checkpoint.pickle").write_bytes(b"preserve checkpoint bytes")
        before = self.tree()
        code, out, err = self.invoke("--apply")
        self.assertEqual(code, 0, err)
        self.assertIn('"status": "quarantined"', out)
        for name, value in before.items():
            if not name.startswith("tasks/" + self.plan["tasks"][0]["task_id"] + "/"):
                self.assertEqual((self.output / name).read_bytes(), value, name)
        self.assertIsNotNone(recovery.runner.checked_completed(self.output, task))
        second = self.invoke("--apply")
        self.assertEqual(second[0], 0, second[2])
        self.assertIn('"status": "no_action"', second[1])
        self.assertEqual(len(list((self.output / "orphan_recovery").iterdir())), 1)

    def test_recovered_study_still_requires_y_and_keeps_frozen_014_plan(self):
        folder = self.orphan()
        frozen = {name: (self.output / name).read_bytes() for name in (
            "manifest.json", "validation_plan.json", "final_plan.json", "continuation_decision.json")}
        recovery.quarantine(self.output, self.config, self.audit())
        code, calls, text, _ = self.fixture.invoke("N\n")
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertIn("CONTINUATION_PROMPT", text)
        self.assertFalse(folder.exists())
        code, calls, text, _ = self.fixture.invoke("Y\n")
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 180)
        final = json.loads((self.output / "final_summary.json").read_text())
        self.assertTrue(final["full_execution_completed"])
        self.assertEqual(final["selected"]["sm9rrs"], fixtures.OURS)
        self.assertEqual(final["validation_final_metric_gate"]["status"], "unmet")
        for name, value in frozen.items():
            self.assertEqual((self.output / name).read_bytes(), value, name)

    def test_unknown_or_training_artifact_blocks_entire_apply_without_moving_safe_orphans(self):
        safe = self.orphan()
        folder = self.folder(1)
        for name in ("checkpoints", ".completed_results.pickle", "attempt_20260928.json",
                     "environment.json", "rounds.csv", "notes.txt"):
            with self.subTest(name=name):
                path = folder / name
                if name == "checkpoints":
                    path.mkdir()
                else:
                    path.write_text("evidence")
                before = self.tree()
                code, out, _ = self.invoke("--apply")
                self.assertEqual(code, 2)
                self.assertIn('"status": "blocked"', out)
                self.assertTrue(safe.exists())
                self.assertFalse((self.output / "orphan_recovery").exists())
                self.assertEqual(before, self.tree())
                path.rmdir() if path.is_dir() else path.unlink()

    def test_training_context_or_progress_in_orphan_is_not_reset(self):
        folder = self.orphan()
        failure_path = folder / "failure.json"
        failure = json.loads(failure_path.read_text())
        failure["execution_context"] = {"study_phase": "final"}
        recovery.runner.write_json(failure_path, failure)
        self.assertEqual(self.audit()["status"], "blocked")
        failure["execution_context"] = None
        recovery.runner.write_json(failure_path, failure)
        (folder / "worker.log").write_text("ROUND example round=1 accuracy=.5\n")
        self.assertEqual(self.audit()["status"], "blocked")

    def test_partial_json_allowed_but_complete_conflicting_identity_is_blocked(self):
        folder = self.orphan()
        (folder / "failure.json").write_text('{"task_id":')
        self.assertEqual(self.audit()["status"], "ready")
        recovery.runner.write_json(folder / ".task.json.123.tmp", {"wrong": "identity"})
        self.assertEqual(self.audit()["status"], "blocked")
        recovery.runner.write_json(folder / ".task.json.123.tmp", self.plan["tasks"][0])
        self.assertEqual(self.audit()["status"], "ready")
        recovery.runner.write_json(folder / "failure.json", {"task_id": "wrong", "execution_context": None})
        self.assertEqual(self.audit()["status"], "blocked")

    def test_existing_damaged_or_conflicting_identity_is_not_repaired(self):
        folder = self.orphan()
        for payload in ('{"task_id":', '{"wrong": 1}'):
            (folder / "task.json").write_text(payload)
            self.assertEqual(self.audit()["status"], "blocked")
            self.assertEqual(self.invoke("--apply")[0], 2)
            self.assertFalse((self.output / "orphan_recovery").exists())

    def test_symlinks_are_blocked(self):
        folder = self.orphan()
        path = folder / "failure.json"
        path.unlink()
        path.symlink_to(self.output / "manifest.json")
        self.assertEqual(self.audit()["status"], "blocked")
        path.unlink()
        task_path = folder / "task.json"
        task_path.symlink_to(self.output / "missing.json")
        self.assertEqual(self.audit()["status"], "blocked")

    def test_active_runner_and_orphan_worker_are_refused(self):
        self.orphan()
        with (self.output / ".runner.lock").open("rb") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.invoke("--apply")[0], 2)
        with mock.patch.object(recovery.subprocess, "run", return_value=SimpleNamespace(
                stdout=f"123 python run_cifar_six_relative_best.py --worker task --output {self.output}\n")):
            code, _, err = self.invoke("--apply")
            self.assertEqual(code, 2)
            self.assertIn("worker PID 123", err)
        self.assertFalse((self.output / "orphan_recovery").exists())

    def test_metadata_drift_after_audit_is_refused(self):
        folder = self.orphan()
        report = self.audit()
        (folder / "worker.log").write_text("changed while auditing")
        with self.assertRaisesRegex(ValueError, "study changed after audit"):
            recovery.quarantine(self.output, self.config, report)
        self.assertFalse((self.output / "orphan_recovery").exists())

    def test_unavailable_process_inventory_refuses_mutation(self):
        self.orphan()
        before = self.tree()
        with mock.patch.object(recovery.subprocess, "run", side_effect=PermissionError("ps unavailable")):
            self.assertEqual(self.invoke("--apply")[0], 2)
        self.assertEqual(before, self.tree())

    def test_interrupted_quarantine_keeps_every_original_and_can_resume(self):
        folders = [self.orphan(0), self.orphan(1)]
        contents = [{p.name: p.read_bytes() for p in f.iterdir()} for f in folders]
        real_rename = recovery.os.rename
        calls = []
        def interrupt(source, destination):
            calls.append(source)
            if len(calls) == 2:
                raise OSError(errno.ENOSPC, "injected rename failure")
            return real_rename(source, destination)
        with mock.patch.object(recovery.os, "rename", side_effect=interrupt):
            with self.assertRaises(OSError):
                recovery.quarantine(self.output, self.config, self.audit())
        batch = next((self.output / "orphan_recovery").iterdir())
        journal = json.loads((batch / "audit.json").read_text())
        self.assertEqual(len(journal["moved_tasks"]), 1)
        for i, folder in enumerate(folders):
            preserved = folder if folder.exists() else batch / folder.name
            self.assertEqual(contents[i], {p.name: p.read_bytes() for p in preserved.iterdir()})
        second = recovery.quarantine(self.output, self.config, self.audit())
        self.assertEqual(len(second["moved_tasks"]), 1)
        self.assertEqual(self.audit()["status"], "no_action")

    def test_intent_write_failure_moves_nothing(self):
        folder = self.orphan()
        before = {p.name: p.read_bytes() for p in folder.iterdir()}
        with mock.patch.object(recovery.runner, "write_json", side_effect=OSError(errno.ENOSPC, "full")):
            with self.assertRaises(OSError):
                recovery.quarantine(self.output, self.config, self.audit())
        self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})

    def test_tampered_manifest_plan_or_decision_is_refused(self):
        self.orphan()
        for name in ("manifest.json", "final_plan.json", "continuation_decision.json"):
            with self.subTest(name=name):
                path = self.output / name
                before = path.read_bytes()
                value = json.loads(before)
                value["unexpected"] = "changed"
                recovery.runner.write_json(path, value)
                self.assertEqual(self.invoke("--apply")[0], 2)
                self.assertFalse((self.output / "orphan_recovery").exists())
                path.write_bytes(before)


if __name__ == "__main__":
    unittest.main()
