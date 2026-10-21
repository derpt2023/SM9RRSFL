import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_cifar_c7_replication as runner


class FixedProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol, cls.bases, cls.data = runner.load_replication()

    def test_exact_eight_task_matrix_and_order(self):
        rows = runner.task_specs(self.protocol, self.bases)
        self.assertEqual(
            [(r["config"]["method"], r["config"]["seed"],
              r["config"]["malicious_ratio"]) for r in rows],
            [("sm9rrs", 301, 0.0), ("vert", 301, 0.0),
             ("fedavg", 401, 0.8), ("sm9rrs", 401, 0.8), ("vert", 401, 0.8),
             ("fedavg", 401, 0.6), ("sm9rrs", 401, 0.6), ("vert", 401, 0.6)])
        self.assertEqual(len({r["task_id"] for r in rows}), 8)

    def test_original_method_configuration_changes_only_authorized_fields(self):
        for row in runner.task_specs(self.protocol, self.bases):
            config = row["config"]
            original = asdict(self.bases[config["method"]])
            different = {k for k in original if original[k] != config[k]}
            self.assertLessEqual(
                different, {"seed", "malicious_ratio", "partition", "sm9_workers"})
            self.assertEqual(config["num_clients"], 100)
            self.assertEqual(config["rounds"], 30)
            self.assertEqual(config["partition"], "dirichlet")
            self.assertEqual(config["sm9_workers"], 8)
            self.assertEqual(config["device"], "mps")
            self.assertEqual(config["attack_epochs"], 1)
            self.assertEqual(config["attack_start_round"], 25)

    def test_attack_profile_identical_across_methods_and_independent_dicts(self):
        rows = runner.task_specs(self.protocol, self.bases)
        self.assertTrue(all(r["profile"] is None for r in rows[:2]))
        for row in rows[2:]:
            self.assertEqual(row["profile"], runner.PROFILE)
        rows[2]["profile"]["boost"] = 1
        self.assertEqual(rows[3]["profile"]["boost"], 20)
        self.assertEqual(self.protocol["profile"], runner.PROFILE)

    def test_changed_seed_or_attack_profile_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for field, value in (
                    ("replication_seed", 402), ("rounds", 50),
                    ("num_clients", 50), ("sm9_workers", 1),
                    ("target_selection", "fixed_target_301"),
                    ("profile", {**runner.PROFILE, "boost": 10})):
                changed = {**self.protocol, field: value}
                path.write_text(json.dumps(changed))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    runner.load_replication(path)

    def test_output_never_overlaps_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for out in (
                    self.protocol["reference_dir"],
                    self.protocol["reference_dir"] + "/child",
                    "outputs", "../external_outputs"):
                path.write_text(json.dumps({**self.protocol, "output_dir": out}))
                with self.subTest(out=out), self.assertRaises(ValueError):
                    runner.load_replication(path)


class FrozenReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.ref = self.root / "outputs/reference"
        self.ref.mkdir(parents=True)
        self.entry = self.root / "run_cifar_c7_replication.py"
        self.entry.write_text("# independent controller\n")
        self.config_path = self.root / "replication.json"
        self.config_path.write_text('{"protocol":"fixed"}\n')
        report = self.root / "scripts/experiments/cifar10/cifar_c7_replication_report.py"
        report.parent.mkdir(parents=True)
        report.write_text("# frozen report\n")
        source = self.root / "upstream.py"
        source.write_text("# original science\n")
        body = {"sources": {"upstream.py": runner.sha(source)}, "protocol": "old"}
        self.old = {**body, "fingerprint": runner.digest(body)}
        runner.write_json(self.ref / "manifest.json", self.old)
        runner.write_json(self.ref / "recommended_attack.json",
                          {"profile": runner.PROFILE, "audit": {"effective": True}})
        for name in ("benchmark_result", "screen_brief", "report", "fedavg_selection"):
            runner.write_json(self.ref / f"{name}.json", {})
        self.task_names = ["clean_iid", "clean_dirichlet", "C5_fedavg_dir80"] + [
            f"C7_{method}_dir{ratio}" for ratio in (60, 80)
            for method in ("fedavg", "sm9rrs", "vert")]
        for name in self.task_names:
            task = self.ref / "tasks" / name
            identity = {"root_fingerprint": self.old["fingerprint"],
                        "config": {"seed": 301}, "unique": name}
            identity["fingerprint"] = runner.digest(identity)
            runner.write_json(task / "task.json", identity)
            runner.write_json(task / "status.json", {
                "fingerprint": identity["fingerprint"],
                "status": "complete", "completed_round": 30})
            for filename in ("metrics", "targets", "dataset_identity", "result"):
                runner.write_json(task / f"{filename}.json", {})
        self.protocol = {
            "reference_dir": "outputs/reference",
            "reference_fingerprint": self.old["fingerprint"]}
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(runner, "ROOT", self.root).start()
        mock.patch.object(runner, "__file__", str(self.entry)).start()

    def test_manifest_binds_old_sources_references_new_sources_and_task_specs(self):
        manifest = runner.build_manifest(
            self.protocol, [{"task_id": "new"}], {"seed": 42}, self.config_path)
        self.assertEqual(len(manifest["source_files"]), 4)
        self.assertEqual(len(manifest["reference_files"]), 60)
        self.assertEqual(manifest["task_specs"], [{"task_id": "new"}])
        self.assertEqual(manifest["backend"], "torch:mps")
        self.assertFalse(manifest["official_test_used_for_selection"])
        runner.verify_frozen(manifest)
        changed = copy.deepcopy(manifest)
        changed["task_specs"][0]["task_id"] = "other"
        self.assertNotEqual(
            runner.digest({k: v for k, v in changed.items() if k != "fingerprint"}),
            manifest["fingerprint"])

    def test_existing_science_change_rejected(self):
        (self.root / "upstream.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "Frozen source changed"):
            runner.reference_files(self.protocol)

    def test_new_source_and_reference_changes_detected_after_freezing(self):
        for changed in (self.entry, self.config_path,
                        self.ref / "tasks/clean_iid/metrics.json"):
            original = changed.read_bytes()
            manifest = runner.build_manifest(self.protocol, [], {}, self.config_path)
            changed.write_text('{"changed":true}\n')
            with self.subTest(path=changed), self.assertRaisesRegex(
                    ValueError, "Immutable"):
                runner.verify_frozen(manifest)
            changed.write_bytes(original)

    def test_failed_reference_rejected_even_with_full_round_count(self):
        status_path = self.ref / "tasks/clean_iid/status.json"
        status = runner.read_json(status_path)
        status["status"] = "numerical_failure"
        runner.write_json(status_path, status)
        with self.assertRaisesRegex(ValueError, "Incomplete reference"):
            runner.reference_files(self.protocol)

    def test_wrong_root_or_mutated_task_identity_rejected(self):
        path = self.ref / "tasks/clean_iid/task.json"
        original = runner.read_json(path)
        wrong_root = {**original, "root_fingerprint": "different"}
        runner.write_json(path, wrong_root)
        with self.assertRaisesRegex(ValueError, "root mismatch"):
            runner.reference_files(self.protocol)
        changed = copy.deepcopy(original)
        changed["config"]["seed"] = 401
        runner.write_json(path, changed)
        with self.assertRaisesRegex(ValueError, "identity corrupt"):
            runner.reference_files(self.protocol)

    def test_unproven_or_different_attack_selection_rejected(self):
        path = self.ref / "recommended_attack.json"
        for selected in (
                {"profile": runner.PROFILE, "audit": {"effective": False}},
                {"profile": {**runner.PROFILE, "boost": 10},
                 "audit": {"effective": True}}):
            runner.write_json(path, selected)
            with self.assertRaisesRegex(ValueError, "expected effective C7"):
                runner.reference_files(self.protocol)


class ControllerStateTests(unittest.TestCase):
    def controller(self):
        controller = object.__new__(runner.ReplicationController)
        controller.manifest = {"fingerprint": "new-root"}
        controller.specs = [{"task_id": "first"}, {"task_id": "second"}]
        controller.state = {"status": "running"}
        controller.snapshot = mock.Mock(return_value={
            "all_complete": False, "all_numerically_finite": False,
            "reference_integrity_valid": True, "all_comparisons_available": False})
        return controller

    def test_partial_task_interrupts_without_claiming_complete(self):
        controller = self.controller()
        controller.run_task = mock.Mock(return_value={"status": "partial"})
        with mock.patch("torch.backends.mps.is_available", return_value=True), \
                mock.patch.object(runner, "verify_frozen"):
            controller.run()
        self.assertEqual(controller.state["status"], "interrupted")
        self.assertEqual(controller.run_task.call_count, 1)

    def test_numerical_failure_retained_other_tasks_run_and_overall_not_complete(self):
        controller = self.controller()
        controller.run_task = mock.Mock(side_effect=[
            {"status": "numerical_failure"}, {"status": "complete"}])
        with mock.patch("torch.backends.mps.is_available", return_value=True), \
                mock.patch.object(runner, "verify_frozen"):
            controller.run()
        self.assertEqual(controller.run_task.call_count, 2)
        self.assertEqual(controller.state["status"], "finished_with_failures")

    def test_complete_requires_integrity_gated_report(self):
        controller = self.controller()
        controller.run_task = mock.Mock(return_value={"status": "complete"})
        with mock.patch("torch.backends.mps.is_available", return_value=True), \
                mock.patch.object(runner, "verify_frozen"):
            controller.run()
        self.assertEqual(controller.state["status"], "finished_with_failures")
        controller.snapshot.return_value = {
            "all_complete": True, "all_numerically_finite": True,
            "reference_integrity_valid": True, "all_comparisons_available": True}
        with mock.patch("torch.backends.mps.is_available", return_value=True), \
                mock.patch.object(runner, "verify_frozen"):
            controller.run()
        self.assertEqual(controller.state["status"], "complete")

    def test_reference_integrity_failure_blocks_training(self):
        controller = self.controller()
        controller.run_task = mock.Mock()
        controller.snapshot.return_value["reference_integrity_valid"] = False
        with mock.patch("torch.backends.mps.is_available", return_value=True), \
                self.assertRaisesRegex(ValueError, "Reference report integrity failed"):
            controller.run()
        controller.run_task.assert_not_called()

    def test_terminal_failure_cached_without_silent_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller()
            controller.out = Path(directory)
            request = {"task_id": "first", "scientific": "same"}
            controller.request = mock.Mock(return_value=request)
            runner.write_json(controller.out / "tasks/first/task.json", request)
            failure = {"status": "numerical_failure", "completed_round": 25,
                       "error": "nonfinite target gradient"}
            runner.write_json(controller.out / "tasks/first/status.json", failure)
            with mock.patch.object(runner, "verify_frozen"), \
                    mock.patch.object(runner, "task_identity", return_value=request), \
                    mock.patch.object(runner.subprocess, "Popen") as spawn:
                result = controller.run_task({"task_id": "first"})
            self.assertEqual(result, failure)
            spawn.assert_not_called()

    def test_mismatched_existing_task_cannot_resume_or_reuse_complete_status(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller()
            controller.out = Path(directory)
            controller.request = mock.Mock(return_value={"task_id": "first"})
            runner.write_json(controller.out / "tasks/first/task.json",
                              {"fingerprint": "old-experiment"})
            runner.write_json(controller.out / "tasks/first/status.json",
                              {"status": "complete", "completed_round": 30})
            with mock.patch.object(runner, "verify_frozen"), \
                    mock.patch.object(runner, "task_identity",
                                      return_value={"fingerprint": "new-experiment"}), \
                    mock.patch.object(runner.subprocess, "Popen") as spawn, \
                    self.assertRaisesRegex(ValueError, "Task identity mismatch"):
                controller.run_task({"task_id": "first"})
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
