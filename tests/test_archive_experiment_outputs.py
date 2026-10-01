import json
import fcntl
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import archive_experiment_outputs as archive


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / "outputs"
        self.root.mkdir()
        for name in archive.DEFAULT_KEEP:
            (self.root / name).mkdir()
            (self.root / name / "result").write_bytes(b"preserved")
        self.source = self.root / "older experiment"
        self.source.mkdir()
        (self.source / ".completed_results.pickle").write_bytes(b"result bytes")
        (self.source / "empty").mkdir()
        self.survey = patch.object(archive, "process_survey", return_value={
            "available": True, "possible_writers": []})
        self.survey.start()
        self.addCleanup(self.survey.stop)
        self.manifest = self.root / "old" / "manifest.json"

    def test_default_plan_has_no_writes(self):
        plan = archive.build_plan(self.root)
        self.assertEqual([r["name"] for r in plan["entries"]], [self.source.name])
        self.assertFalse((self.root / "old").exists())
        self.assertTrue(self.source.exists())

    def test_full_hash_and_rename_preserve_hidden_files_and_empty_dirs(self):
        expected = archive.inventory(self.source)
        result = archive.apply_plan(archive.build_plan(self.root), self.manifest)
        self.assertEqual(result["status"], "completed")
        self.assertFalse(self.source.exists())
        self.assertEqual(archive.inventory(self.root / "old" / self.source.name), expected)
        self.assertEqual(result["entries"][0]["inventory"], expected)
        self.assertTrue(any("sha256" in r for r in expected))
        for name in archive.DEFAULT_KEEP:
            self.assertEqual((self.root / name / "result").read_bytes(), b"preserved")

    def test_completed_manifest_is_idempotent(self):
        archive.apply_plan(archive.build_plan(self.root), self.manifest)
        loaded = json.loads(self.manifest.read_text())
        with patch.object(archive.os, "rename", side_effect=AssertionError("must not rename")):
            archive.apply_plan(loaded, self.manifest)

    def test_crash_after_rename_before_state_update_recovers(self):
        plan = archive.build_plan(self.root)
        original = archive._sync_directory
        def interrupted(path):
            if path == self.root and not self.source.exists():
                raise OSError("simulated power interruption")
            return original(path)
        with patch.object(archive, "_sync_directory", side_effect=interrupted):
            with self.assertRaises(OSError):
                archive.apply_plan(plan, self.manifest)
        loaded = json.loads(self.manifest.read_text())
        self.assertEqual(loaded["entries"][0]["status"], "rename_intent")
        self.assertFalse(self.source.exists())
        archive.apply_plan(loaded, self.manifest)
        self.assertEqual(json.loads(self.manifest.read_text())["status"], "completed")

    def test_crash_before_rename_recovers(self):
        with patch.object(archive.os, "rename", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                archive.apply_plan(archive.build_plan(self.root), self.manifest)
        self.assertTrue(self.source.exists())
        archive.apply_plan(json.loads(self.manifest.read_text()), self.manifest)
        self.assertFalse(self.source.exists())

    def test_collision_blocks_entire_batch(self):
        (self.root / "old" / self.source.name).mkdir(parents=True)
        with self.assertRaisesRegex(archive.ArchiveError, "collision"):
            archive.build_plan(self.root)
        self.assertTrue(self.source.exists())

    def test_collision_after_plan_blocks(self):
        plan = archive.build_plan(self.root)
        (self.root / "old" / self.source.name).mkdir(parents=True)
        with self.assertRaisesRegex(archive.ArchiveError, "both exist"):
            archive.apply_plan(plan, self.manifest)
        self.assertTrue(self.source.exists())

    def test_nested_symlink_blocks_before_move(self):
        (self.source / "link").symlink_to(self.root / archive.DEFAULT_KEEP[0])
        with self.assertRaisesRegex(archive.ArchiveError, "symlink"):
            archive.apply_plan(archive.build_plan(self.root), self.manifest)
        self.assertTrue(self.source.exists())

    def test_root_symlink_rejected(self):
        link = self.root.parent / "link"
        link.symlink_to(self.root)
        with self.assertRaisesRegex(archive.ArchiveError, "symlink"):
            archive.build_plan(link)

    def test_old_never_archived_recursively(self):
        (self.root / "old").mkdir()
        (self.root / "old" / "existing").write_bytes(b"saved")
        plan = archive.build_plan(self.root)
        self.assertNotIn("old", [r["name"] for r in plan["entries"]])
        archive.apply_plan(plan, self.manifest)
        self.assertEqual((self.root / "old" / "existing").read_bytes(), b"saved")

    def test_tampered_destination_rejected_on_resume(self):
        archive.apply_plan(archive.build_plan(self.root), self.manifest)
        (self.root / "old" / self.source.name / ".completed_results.pickle").write_bytes(b"changed")
        with self.assertRaisesRegex(archive.ArchiveError, "inventory mismatch"):
            archive.apply_plan(json.loads(self.manifest.read_text()), self.manifest)

    def test_unproven_destination_rejected(self):
        plan = archive.build_plan(self.root)
        (self.root / "old").mkdir()
        self.source.rename(self.root / "old" / self.source.name)
        with self.assertRaisesRegex(archive.ArchiveError, "unproven"):
            archive.apply_plan(plan, self.manifest)

    def test_running_experiment_blocks(self):
        with patch.object(archive, "process_survey", return_value={
                "available": True, "possible_writers": [{"pid": 999}]}):
            with self.assertRaisesRegex(archive.ArchiveError, "writers"):
                archive.apply_plan(archive.build_plan(self.root), self.manifest)
        self.assertTrue(self.source.exists())

    def test_unavailable_survey_requires_explicit_acknowledgement(self):
        with patch.object(archive, "process_survey", return_value={
                "available": False, "possible_writers": []}):
            plan = archive.build_plan(self.root)
            with self.assertRaisesRegex(archive.ArchiveError, "unavailable"):
                archive.apply_plan(plan, self.manifest)
            archive.apply_plan(plan, self.manifest, allow_unavailable_process_check=True)

    def test_missing_keep_rejected(self):
        with self.assertRaisesRegex(archive.ArchiveError, "preserved"):
            archive.build_plan(self.root, ["missing"])

    def test_held_runner_lock_blocks(self):
        lock = self.source / ".runner.lock"
        with lock.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(archive.ArchiveError, "runner lock"):
                archive.apply_plan(archive.build_plan(self.root), self.manifest)
        self.assertTrue(self.source.exists())

    def test_metadata_skipping_is_explicit(self):
        (self.root / ".DS_Store").write_bytes(b"new metadata")
        (self.root / "old").mkdir()
        (self.root / "old" / ".DS_Store").write_bytes(b"old metadata")
        with self.assertRaisesRegex(archive.ArchiveError, "collision"):
            archive.build_plan(self.root)
        plan = archive.build_plan(self.root, ignore_metadata=True)
        archive.apply_plan(plan, self.manifest)
        self.assertEqual((self.root / ".DS_Store").read_bytes(), b"new metadata")
        self.assertEqual((self.root / "old" / ".DS_Store").read_bytes(), b"old metadata")

    def test_rename_intent_and_inventory_durable_before_rename(self):
        def crash_before_move(source, target):
            saved = json.loads(self.manifest.read_text())
            self.assertEqual(saved["entries"][0]["status"], "rename_intent")
            self.assertEqual(saved["entries"][0]["inventory"], archive.inventory(source))
            raise OSError("crash before rename")
        with patch.object(archive.os, "rename", side_effect=crash_before_move):
            with self.assertRaises(OSError):
                archive.apply_plan(archive.build_plan(self.root), self.manifest)
        self.assertTrue(self.source.exists())

    def test_path_traversal_manifest_rejected(self):
        plan = archive.build_plan(self.root)
        plan["entries"][0]["name"] = "../outside"
        with self.assertRaisesRegex(archive.ArchiveError, "unsafe"):
            archive.apply_plan(plan, self.manifest)


if __name__ == "__main__":
    unittest.main()
