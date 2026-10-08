"""End-to-end immutable reference chain: original 24 -> matched 4 -> timing 26."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import cifar_timing_protocol as protocol
import cifar_timing_report as report
import run_cifar_timing_diagnostic as runner
from tests.test_cifar_diagnostic import StudyFixture

base, runtime, clean, matched = protocol.base, protocol.runtime, protocol.clean, protocol.matched


class TimingFixture(StudyFixture):
    def setUp(self):
        super().setUp()
        self.clean_output, self.clean_tasks = self.output, self.tasks
        self.clean_fp = self.manifest["fingerprint"]
        environment = {"actual_compute_device": {"name": "synthetic", "compute_capability": [8, 9],
                                                "total_memory_bytes": 24 * 2**30},
                       "environment": {}, "driver_versions": []}
        self.environment = environment
        self.metadata = {k: v for k, v in environment.items() if k != "driver_versions"}
        for task in self.tasks:
            accuracy = {"C0": .6164, "R3": .6160}.get(task["candidate"]["candidate_id"], .50)
            folder = self.complete(task, accuracy)
            base.write_json(folder / "environment.json", self.metadata)
        base.write_json(self.output / "execution_environment.json", environment)
        old_reference = matched.audit_reference(self.output, expected_fingerprint=self.clean_fp)
        self.patch_value(matched, "REFERENCE_FINGERPRINT", self.clean_fp)
        match_spec = runtime.read_json(matched.DEFAULT_CONFIG)
        match_spec["reference_manifest_fingerprint"] = self.clean_fp
        self.output = self.temporary_output()
        self.matched_output = self.output
        match_manifest = matched.build_manifest(match_spec, old_reference, self.manifest["spec"]["dataset"])
        self.match_fp = match_manifest["fingerprint"]
        self.match_tasks = matched.build_tasks(match_manifest)
        base.write_json(self.output / "manifest.json", match_manifest)
        runtime.save_plan(self.output, "matched_cnn", self.match_tasks, match_manifest)
        base.write_json(self.output / "execution_environment.json", environment)
        for task in self.match_tasks:
            folder = self.complete(task, .6167)
            base.write_json(folder / "environment.json", self.metadata)
        self.patch_value(protocol, "CLEAN_FINGERPRINT", self.clean_fp)
        self.patch_value(protocol, "MATCHED_FINGERPRINT", self.match_fp)
        self.reference = protocol.audit_reference(self.clean_output, self.matched_output)
        self.spec = runtime.read_json(protocol.DEFAULT_CONFIG)
        self.spec.update(clean_manifest_fingerprint=self.clean_fp, matched_manifest_fingerprint=self.match_fp)
        self.output = self.temporary_output()
        self.manifest = protocol.build_manifest(self.spec, self.reference)
        self.tasks = protocol.build_tasks(self.manifest)
        base.write_json(self.output / "manifest.json", self.manifest)
        runtime.save_plan(self.output, "timing", self.tasks, self.manifest)
        base.write_json(self.output / "execution_environment.json", environment)

    def patch_value(self, obj, name, value):
        patcher = mock.patch.object(obj, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def temporary_output(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return Path(temp.name)

    def references_snapshot(self):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for root in (self.clean_output, self.matched_output) for p in root.rglob("*") if p.is_file()}


class TimingProtocolTests(TimingFixture):
    def test_fixed_26_matrix_preserves_public_config_and_original_014(self):
        self.assertEqual(len(self.tasks), 26)
        self.assertEqual(len({t["task_id"] for t in self.tasks}), 26)
        self.assertEqual(len({t["fingerprint"] for t in self.tasks}), 26)
        self.assertEqual(sum(t["method"] == "sm9rrs" for t in self.tasks), 18)
        v7 = runtime.read_json(protocol.REPO / "configs/cifar10_six_relative_best_five_day_v7.json")
        original = next(c for c in v7["candidates"]["sm9rrs"] if c["candidate_id"] == "sm9rrs-v10-014")
        self.assertEqual(protocol.OURS_014, original["parameters"])
        for task in self.tasks:
            config = task["config"]
            old = next(t for t in self.clean_tasks if t["task_id"] == task["clean_reference_task_id"])
            allowed = {"method", "malicious_ratio", "detector_window", "attack_start_round"}
            if task["method"] == "sm9rrs":
                allowed |= set(protocol.OURS_014)
                self.assertEqual({k: config[k] for k in protocol.OURS_014}, protocol.OURS_014)
            self.assertEqual({k: v for k, v in config.items() if k not in allowed},
                             {k: v for k, v in old["config"].items() if k not in allowed})
            self.assertEqual(config["seed"], 2026093001)
            self.assertEqual(task["phase"], "validation")
            self.assertEqual(task["model"], "v7_cnn")
            self.assertEqual(config["rounds"], 150)
            self.assertEqual(config["crypto_mode"], "sm9")
        self.assertFalse(self.manifest["official_test_used_for_selection"])
        self.assertFalse(self.manifest["next_stage_automatic"])

    def test_paired_contrasts_change_only_declared_variable(self):
        for before, after, expected in (("A", "B", "attack_start_round"),
                                       ("B", "C", "detector_window"), ("FA12", "FA25", "attack_start_round")):
            for task in [t for t in self.tasks if t["arm"] == before]:
                pair = next(t for t in self.tasks if t["arm"] == after
                            and t["config"]["partition"] == task["config"]["partition"]
                            and t["config"]["malicious_ratio"] == task["config"]["malicious_ratio"])
                changed = {k for k in task["config"] if task["config"][k] != pair["config"][k]}
                self.assertEqual(changed, {expected})

    def test_real_reference_audit_and_empty_summary_read_only(self):
        before = self.references_snapshot()
        with mock.patch.object(base, "load_split", side_effect=AssertionError("data load")), \
                mock.patch.object(base.experiments, "run_measured_experiment", side_effect=AssertionError("training")):
            self.assertEqual(protocol.audit_reference(self.clean_output, self.matched_output), self.reference)
            summary = report.summarize(self.output, self.clean_output, self.matched_output)
        self.assertTrue(summary["reference_verified"])
        self.assertEqual(summary["complete_tasks"], 0)
        self.assertEqual(before, self.references_snapshot())
        self.assertEqual(len(self.reference["c0_tasks"]), 4)
        self.assertEqual(len(self.reference["clean_rows"]), 4)
        self.assertTrue(any("observations.json" in p for p in self.reference["clean_evidence_sha256"]))
        self.assertTrue(any("environment.json" in p for p in self.reference["matched_evidence_sha256"]))

    def test_read_study_checks_sources_plan_data_and_reference_identity(self):
        self.assertEqual(protocol.read_study(self.output, current_sources=True)[1], self.tasks)
        with mock.patch.object(protocol, "source_hashes", return_value={"changed": "source"}):
            with self.assertRaisesRegex(ValueError, "identity"):
                protocol.read_study(self.output, current_sources=True)
        for field, value in (("chosen_model", "resnet18_gn2"), ("matched_manifest_fingerprint", "other")):
            altered = deepcopy(self.manifest)
            altered["reference"][field] = value
            altered["fingerprint"] = base.digest({k: v for k, v in altered.items() if k != "fingerprint"})
            base.write_json(self.output / "manifest.json", altered)
            with self.assertRaisesRegex(ValueError, "reference identity"):
                protocol.read_study(self.output)
        base.write_json(self.output / "manifest.json", self.manifest)
        broken = deepcopy(self.tasks)
        broken[0]["model"] = "resnet18_gn2"
        base.write_json(self.output / "task_plans/timing.json", {
            "manifest_fingerprint": self.manifest["fingerprint"], "tasks": broken})
        with self.assertRaisesRegex(ValueError, "plan"):
            protocol.read_study(self.output)

    def test_reference_task_or_dataset_cannot_be_relabeled(self):
        manifest = deepcopy(self.manifest)
        manifest["reference"]["c0_tasks"][0]["config"]["lr"] = .10
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            protocol.build_tasks(manifest)
        manifest = deepcopy(self.manifest)
        manifest["data_contract"] = {**manifest["data_contract"], "different_data": True}
        manifest["fingerprint"] = base.digest({k: v for k, v in manifest.items() if k != "fingerprint"})
        base.write_json(self.output / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "dataset identity"):
            protocol.read_study(self.output)

    def test_spec_rejects_protocol_changes_and_formal_seed(self):
        for field, value in (("seed", 2026093011), ("ours_ratios", [.1]), ("extra", True)):
            spec = deepcopy(self.spec)
            spec[field] = value
            with self.assertRaises(ValueError):
                protocol.build_manifest(spec, self.reference)
        for field, value in (("local_epochs", 2), ("rounds", 100), ("lr", .01)):
            spec = deepcopy(self.spec)
            spec["public"][field] = value
            with self.assertRaises(ValueError):
                protocol.build_manifest(spec, self.reference)

    def test_original_sources_are_preserved_in_new_identity(self):
        old = matched.source_hashes()
        self.assertEqual(len(old), 55)
        self.assertEqual({k: protocol.source_hashes()[k] for k in old}, old)
        self.assertEqual(len(protocol.source_hashes()), 58)

    def test_mutated_old_reference_is_not_reused(self):
        path = self.clean_output / "tasks" / self.clean_tasks[0]["task_id"] / "observations.json"
        observations = runtime.read_json(path)
        observations["rounds"][150]["calibration_loss"] += .001
        base.write_json(path, observations)
        with self.assertRaisesRegex(ValueError, "complete/healthy"):
            protocol.audit_reference(self.clean_output, self.matched_output)

    def test_unhealthy_matched_result_blocks_stage_two(self):
        folder = self.matched_output / "tasks" / self.match_tasks[0]["task_id"]
        result = base.experiments.load_completed_results_snapshot(folder)[0]
        base.experiments._write_completed_results_snapshot(folder, [replace(result, nonfinite_updates=1)])
        with self.assertRaisesRegex(ValueError, "complete/healthy"):
            protocol.audit_reference(self.clean_output, self.matched_output)

    def test_matched_task_environment_mismatch_blocks_stage_two(self):
        metadata = deepcopy(self.metadata)
        metadata["actual_compute_device"]["name"] = "different"
        base.write_json(self.matched_output / "tasks" / self.match_tasks[0]["task_id"] / "environment.json", metadata)
        with self.assertRaisesRegex(ValueError, "numerical environment"):
            protocol.audit_reference(self.clean_output, self.matched_output)

    def test_concurrent_reference_mutation_blocks_stage_two(self):
        actual = protocol.matched_evidence_hashes
        calls = []
        def hashes(*args):
            evidence = actual(*args)
            if calls:
                evidence["changed"] = "during read"
            calls.append(True)
            return evidence
        with mock.patch.object(protocol, "matched_evidence_hashes", side_effect=hashes):
            with self.assertRaisesRegex(ValueError, "changed while"):
                protocol.audit_reference(self.clean_output, self.matched_output)

    def test_completed_26_task_controller_report_never_modifies_old_studies(self):
        before = self.references_snapshot()
        args = SimpleNamespace(output=self.output, clean_output=self.clean_output, matched_output=self.matched_output,
                               devices=["cuda:0"], data_dir=None)
        def finish(*_):
            for task in self.tasks:
                # Synthetic terminal snapshots only; this is not real training.
                self.complete(task)
        with mock.patch.object(runner, "execute", side_effect=finish), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_parent(args, self.spec, self.reference), 0)
        summary = report.summarize(self.output, self.clean_output, self.matched_output)
        self.assertEqual(summary["complete_tasks"], 26)
        self.assertEqual(summary["healthy_tasks"], 26)
        self.assertTrue(summary["reference_verified"])
        self.assertFalse(summary["decision"]["next_stage_started"])
        self.assertIsNone(summary["decision"]["selected_arm"])
        self.assertEqual(before, self.references_snapshot())


if __name__ == "__main__":
    unittest.main()
