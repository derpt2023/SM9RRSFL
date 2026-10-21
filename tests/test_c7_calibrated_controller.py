"""Protocol, selection and immutable-resume tests without model training."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_cifar_c7_calibrated_60r as runner


def measure(accuracy=.44, tail=.43, **changes):
    return {"valid": True, "final_accuracy": accuracy, "last10_accuracy": tail,
            "zero_aggregation_rounds": [], "max_false_positives": 0,
            "max_blacklist_size": 0, "max_nonfinite_updates": 0, **changes}


class CalibratedControllerTests(unittest.TestCase):
    def setUp(self):
        self.protocol, self.bases, self.data = runner.load_plan()
        self.reference = {"final_accuracy": .44, "last10_accuracy": .43}
        self.records = {c["id"]: measure() for rows in
                        self.protocol["calibration_candidates"].values() for c in rows}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def selection(self):
        return runner.select_candidates(self.protocol, self.records, self.reference)

    def test_calibration_is_eight_clean_seed301_tasks_with_no_attack_profile(self):
        tasks = runner.calibration_specs(self.protocol, self.bases)
        self.assertEqual(len(tasks), 8)
        self.assertEqual(sum(t["config"]["rounds"] for t in tasks), 240)
        self.assertEqual({t["config"]["method"] for t in tasks}, {"vert", "alignins"})
        for task in tasks:
            self.assertEqual((task["config"]["seed"], task["config"]["malicious_ratio"]), (301, 0))
            self.assertIsNone(task["profile"])
            self.assertEqual(task["stage"], "calibration")

    def test_final_is_six_clean_and_twelve_attacked_with_same_c7_and_fixed_public_parameters(self):
        tasks = runner.final_specs(self.protocol, self.bases, self.selection())
        self.assertEqual(len(tasks), 18)
        self.assertEqual(len({t["task_id"] for t in tasks}), 18)
        self.assertEqual(sum(t["config"]["rounds"] for t in tasks), 1080)
        for ratio in (0., .6, .8):
            group = [t for t in tasks if t["config"]["malicious_ratio"] == ratio]
            self.assertEqual({t["config"]["method"] for t in group}, set(runner.METHODS))
            for task in group:
                c = task["config"]
                self.assertEqual((c["seed"], c["rounds"], c["num_clients"]), (402, 60, 100))
                self.assertEqual((c["attack_start_round"], c["local_epochs"], c["batch_size"]), (25, 1, 50))
                self.assertEqual((c["partition"], c["dirichlet_alpha"], c["sm9_workers"]), ("dirichlet", .5, 8))
                self.assertEqual(task["profile"], runner.PROFILE if ratio else None)
        for method in runner.METHODS:
            configs = [dict(t["config"]) for t in tasks if t["config"]["method"] == method]
            for c in configs:
                c.pop("malicious_ratio")
            self.assertEqual(configs[0], configs[1])
            self.assertEqual(configs[0], configs[2])

    def test_ours_krum_fedavg_and_ding13_parameters_remain_the_existing_candidates(self):
        tasks = runner.final_specs(self.protocol, self.bases, self.selection())
        for task in tasks:
            method = task["config"]["method"]
            if method in ("vert", "alignins"):
                continue
            expected = asdict(self.bases[method])
            expected.update(seed=402, rounds=60, malicious_ratio=task["config"]["malicious_ratio"])
            self.assertEqual(task["config"], expected)

    def test_selection_prioritizes_qualified_tail_accuracy_then_final_then_id(self):
        self.records.update(V1=measure(accuracy=.45, tail=.42),
                            V2=measure(accuracy=.44, tail=.44),
                            V3=measure(accuracy=.45, tail=.44),
                            V4=measure(accuracy=.45, tail=.44))
        selected = self.selection()["selected"]
        self.assertEqual(selected["vert"]["candidate_id"], "V3")
        self.assertEqual(selected["alignins"]["candidate_id"], "A1")
        self.assertTrue(selected["vert"]["clean_qualified"])

    def test_attack_metrics_cannot_affect_clean_ranking(self):
        before = self.selection()["selected"]
        for number, record in enumerate(self.records.values()):
            record.update(training_target_success=number % 2,
                          source_transfer_asr=number / 10,
                          attack_target_confidence=1 - number / 10)
        after = self.selection()["selected"]
        self.assertEqual(before, after)
        self.assertFalse(self.selection()["attack_outcomes_used"])
        self.assertFalse(self.selection()["final_seed_results_used"])

    def test_health_failures_cannot_beat_an_eligible_lower_accuracy_candidate(self):
        self.records.update(V1=measure(.9, .9, zero_aggregation_rounds=[8]),
                            V2=measure(.9, .9, max_false_positives=11),
                            V3=measure(.9, .9, max_blacklist_size=100),
                            V4=measure(.43, .42))
        selection = self.selection()
        self.assertEqual(selection["selected"]["vert"]["candidate_id"], "V4")
        self.assertTrue(selection["selected"]["vert"]["clean_qualified"])
        failures = {r["id"]: r["failure_reasons"] for r in selection["candidate_evidence"]["vert"]}
        self.assertIn("empty_aggregation", failures["V1"])
        self.assertIn("clean_false_revocation_rate", failures["V2"])
        self.assertIn("all_clients_revoked", failures["V3"])

    def test_unqualified_fallback_is_explicit_and_preserves_all_failures(self):
        for candidate in ("V1", "V2", "V3", "V4"):
            self.records[candidate] = measure(.2, .2)
        self.records["V3"] = measure(.3, .3)
        selection = self.selection()
        chosen = selection["selected"]["vert"]
        self.assertEqual(chosen["candidate_id"], "V3")
        self.assertFalse(chosen["clean_qualified"])
        self.assertIn("final_clean_accuracy_drop", chosen["failure_reasons"])
        self.assertEqual(len(selection["candidate_evidence"]["vert"]), 4)

    def test_no_complete_finite_candidate_blocks_final_selection(self):
        for candidate in ("A1", "A2", "A3", "A4"):
            self.records[candidate] = {"valid": False, "errors": ["numerical_failure"]}
        with self.assertRaisesRegex(ValueError, "final stage blocked"):
            self.selection()

    def test_candidate_cannot_override_shared_training_or_attack_parameters(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["calibration_candidates"]["vert"][0]["overrides"]["lr"] = .1
        path = self.root / "config.json"
        path.write_text(json.dumps(protocol))
        with self.assertRaisesRegex(ValueError, "own defense parameters"):
            runner.load_plan(path)

    def test_wrong_seed_or_attack_profile_is_rejected(self):
        for field, value in (("final_seed", 401), ("final_rounds", 30),
                             ("profile", {**runner.PROFILE, "boost": 10})):
            protocol = {**self.protocol, field: value}
            path = self.root / "config.json"
            path.write_text(json.dumps(protocol))
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.load_plan(path)

    def test_stage_manifest_freezes_selection_and_provenance(self):
        source = self.root / "source.py"
        reference = self.root / "reference.json"
        selected = self.root / "selection.json"
        source.write_text("source")
        reference.write_text("reference")
        selected.write_text("selection")
        foundation = ({str(source): runner.sha(source)}, {str(reference): runner.sha(reference)})
        extra = {str(selected): runner.sha(selected)}
        manifest = runner.stage_manifest(self.protocol, [], self.data, "final", foundation, extra)
        self.assertEqual(manifest["fingerprint"], runner.digest({
            k: v for k, v in manifest.items() if k != "fingerprint"}))
        self.assertEqual(manifest["reference_files"][str(selected)], runner.sha(selected))
        runner.verify_frozen(manifest)
        for path in (source, reference, selected):
            original = path.read_text()
            path.write_text("tampered")
            with self.subTest(path=path), self.assertRaises(ValueError):
                runner.verify_frozen(manifest)
            path.write_text(original)

    def test_completed_stage_resume_does_not_rewrite_or_relaunch(self):
        spec = runner.calibration_specs(self.protocol, self.bases)[0]
        manifest = runner.stage_manifest(self.protocol, [spec], self.data, "final", ({}, {}))
        runner.write_json(self.root / "manifest.json", manifest)
        runner.write_json(self.root / "controller_status.json",
                          {"status": "complete", "elapsed_seconds": 123., "updated_utc": "unchanged"})
        runner.write_json(self.root / "report.json", {"saved": "unchanged"})
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        with mock.patch("scripts.experiments.cifar10.c7_calibrated_report.build_report",
                        return_value={"all_complete": True, "all_numerically_finite": True,
                                      "reference_integrity_valid": True}), \
                mock.patch.object(runner, "StageController") as controller:
            self.assertTrue(runner.run_stage(self.root, manifest, "log"))
            controller.assert_not_called()
        after = {p.name: p.read_bytes() for p in self.root.iterdir()}
        self.assertEqual(before, after)

    def test_completed_stage_corrupt_metrics_are_not_accepted(self):
        manifest = runner.stage_manifest(self.protocol, [], self.data, "final", ({}, {}))
        runner.write_json(self.root / "manifest.json", manifest)
        runner.write_json(self.root / "controller_status.json", {"status": "complete"})
        with mock.patch("scripts.experiments.cifar10.c7_calibrated_report.build_report",
                        return_value={"all_complete": False, "all_numerically_finite": False,
                                      "reference_integrity_valid": True}), \
                self.assertRaisesRegex(ValueError, "no longer"):
            runner.run_stage(self.root, manifest, "log")

    def test_terminal_failures_are_not_silently_retried(self):
        spec = runner.calibration_specs(self.protocol, self.bases)[0]
        manifest = runner.stage_manifest(self.protocol, [spec], self.data, "calibration", ({}, {}))
        controller = object.__new__(runner.StageController)
        controller.out, controller.manifest, controller.data = self.root, manifest, self.data
        request = {**spec, "data_config": self.data, "fingerprint": manifest["fingerprint"]}
        directory = self.root / "tasks" / spec["task_id"]
        runner.write_json(directory / "task.json", runner.task_identity(request))
        runner.write_json(directory / "status.json", {"status": "numerical_failure"})
        with mock.patch.object(runner.subprocess, "Popen") as spawn:
            self.assertEqual(controller.run_task(spec)["status"], "numerical_failure")
            spawn.assert_not_called()

    def test_missing_stage_identity_cannot_adopt_existing_tasks(self):
        (self.root / "tasks").mkdir()
        manifest = runner.stage_manifest(self.protocol, [], self.data, "calibration", ({}, {}))
        with self.assertRaisesRegex(ValueError, "without a stage identity"):
            runner.StageController(self.root, manifest, "log")

    def test_clean_measure_rejects_attack_tasks_before_reading_metrics(self):
        item = runner.final_specs(self.protocol, self.bases, self.selection())[6]
        with mock.patch("scripts.experiments.cifar10.c7_calibrated_report.task_summary") as read, \
                self.assertRaisesRegex(ValueError, "Only clean"):
            runner.clean_measure(self.root, item, "unused")
        read.assert_not_called()

    def test_changed_selection_policy_cannot_be_silently_ignored(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["selection"]["max_final_clean_accuracy_drop"] = .9
        with self.assertRaisesRegex(ValueError, "Selection rule"):
            runner.select_candidates(protocol, self.records, self.reference)

    def test_original_reference_directories_cannot_contain_new_output(self):
        for out in ("outputs/cifar10_attack_stability_v1/child",
                    "outputs/cifar10_c7_replication_v1/child", "outputs"):
            protocol = {**self.protocol, "output_dir": out}
            path = self.root / "config.json"
            path.write_text(json.dumps(protocol))
            with self.subTest(out=out), self.assertRaisesRegex(ValueError, "independent"):
                runner.load_plan(path)

    def test_explicit_calibration_failure_is_terminal_but_corrupt_identity_is_not(self):
        specs = runner.calibration_specs(self.protocol, self.bases)[:2]
        manifest = runner.stage_manifest(self.protocol, specs, self.data, "calibration", ({}, {}))
        report = {"reference_integrity_valid": True, "all_complete": False,
                  "all_numerically_finite": False, "tasks": {
                      s["task_id"]: {"valid_complete": i == 0, "integrity_errors": []}
                      for i, s in enumerate(specs)}}
        for i, item in enumerate(specs):
            runner.write_json(self.root / "tasks" / item["task_id"] / "status.json",
                              {"status": "complete" if i == 0 else "numerical_failure"})
        self.assertTrue(runner.stage_ready(self.root, manifest, report))
        final = {**manifest, "stage": "final"}
        self.assertFalse(runner.stage_ready(self.root, final, report))
        report["tasks"][specs[1]["task_id"]]["integrity_errors"].append("task identity mismatch")
        self.assertFalse(runner.stage_ready(self.root, manifest, report))

    def test_pending_candidate_keeps_calibration_pending(self):
        specs = runner.calibration_specs(self.protocol, self.bases)[:1]
        manifest = runner.stage_manifest(self.protocol, specs, self.data, "calibration", ({}, {}))
        report = {"reference_integrity_valid": True, "all_complete": False,
                  "all_numerically_finite": False, "tasks": {}}
        self.assertFalse(runner.stage_ready(self.root, manifest, report))

    def test_old_controller_complete_label_cannot_bypass_invalid_evidence(self):
        previous = {"source_files": {}, "reference_files": {},
                    "protocol": {"reference_dir": "unused"}}
        previous["fingerprint"] = runner.digest(previous)
        protocol = {**self.protocol, "reference_fingerprint": previous["fingerprint"]}
        with mock.patch.object(runner, "read_json", side_effect=[previous, {"status": "complete"}]), \
                mock.patch("scripts.experiments.cifar10.cifar_c7_six_report.build_report",
                           return_value={"all_complete": True, "all_numerically_finite": False}), \
                self.assertRaisesRegex(ValueError, "scientific evidence"):
            runner.frozen_foundation(protocol, runner.DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
