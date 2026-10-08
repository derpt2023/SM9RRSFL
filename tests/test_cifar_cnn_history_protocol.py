"""History ablation identities and the read-only 78-task reference chain."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import unittest
from unittest import mock

import cifar_cnn_history_protocol as protocol
from tests.test_cifar_cnn_threshold_protocol import ThresholdFixture

base, runtime = protocol.base, protocol.runtime


class HistoryFixture(ThresholdFixture):
    def setUp(self):
        super().setUp()
        self.threshold_output, self.threshold_tasks = self.output, self.tasks
        self.threshold_manifest = self.manifest
        self.patch_value(protocol, "THRESHOLD_FINGERPRINT", self.manifest["fingerprint"])
        for task in self.tasks:
            key = (task["config"]["partition"], task["config"]["malicious_ratio"])
            accuracy, asr, fp = protocol.REVIEWED_P0[key] if task["arm"] == "P0" else (.60, .10, 0)
            nonfinite = int(task["arm"] == "P2" and key == ("dirichlet", .7))
            self.finish(task, accuracy, .10 if asr is None else asr, fp, nonfinite)
        self.reference = protocol.audit_reference(self.threshold_output, self.timing_output,
                                                   self.clean_output, self.matched_output)
        self.spec = runtime.read_json(protocol.DEFAULT_CONFIG)
        self.spec["threshold_manifest_fingerprint"] = protocol.THRESHOLD_FINGERPRINT
        self.output = self.temporary_output()
        self.manifest = protocol.build_manifest(self.spec, self.reference)
        self.tasks = protocol.build_tasks(self.manifest)
        base.write_json(self.output / "manifest.json", self.manifest)
        runtime.save_plan(self.output, "history", self.tasks, self.manifest)
        base.write_json(self.output / "execution_environment.json", self.environment)

    def finish(self, task, accuracy=.6, asr=.1, fp=0, nonfinite=0):
        folder = super().finish(task, accuracy, asr, fp, nonfinite)
        if task["candidate"].get("variant") == "Ours-FrozenHistory-v1":
            result = base.experiments.load_completed_results_snapshot(folder)[0]
            for diagnostic in result.diagnostics:
                if diagnostic.round >= task["history_freeze_start_round"]:
                    diagnostic.history_admitted = False
            base.experiments._write_completed_results_snapshot(folder, [result])
        return folder

    def original_snapshot(self):
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for root in (self.clean_output, self.matched_output, self.timing_output, self.threshold_output)
                for path in root.rglob("*") if path.is_file()}


class HistoryProtocolTests(HistoryFixture):
    def test_all_78_reference_tasks_audited_readonly_and_P2_failure_retained(self):
        before = self.original_snapshot()
        with mock.patch.object(base, "load_split", side_effect=AssertionError("download")), \
                mock.patch.object(base.experiments, "run_measured_experiment", side_effect=AssertionError("training")):
            actual = protocol.audit_reference(self.threshold_output, self.timing_output, self.clean_output, self.matched_output)
        self.assertEqual(actual, self.reference)
        self.assertEqual(actual["threshold_health_count"], 23)
        self.assertEqual([r["arm"] for r in actual["failure_rows"]], ["P2"])
        self.assertEqual(actual["upstream_reference"]["timing_health_count"], 21)
        self.assertEqual(len(actual["p0_rows"]), 6)
        self.assertTrue(all(r["healthy"] for r in actual["p0_rows"]))
        self.assertEqual(before, self.original_snapshot())

    def test_both_arms_are_fresh_identical_P0_configs_with_distinct_variant_identities(self):
        self.assertEqual(len(self.tasks), 12)
        self.assertEqual(len({task["fingerprint"] for task in self.tasks}), 12)
        for task in self.tasks:
            old = next(t for t in self.threshold_tasks if t["task_id"] == task["original_p0_task_id"])
            self.assertEqual(task["config"], old["config"])
            self.assertEqual(task["candidate"]["parameters"], old["candidate"]["parameters"])
            self.assertNotEqual(task["task_id"], old["task_id"])
            self.assertNotEqual(task["fingerprint"], old["fingerprint"])
            self.assertEqual(task["phase"], "validation")
            frozen = task["arm"] == "H1"
            self.assertEqual(task["ours_algorithm_modified"], frozen)
            self.assertEqual(task["history_freeze_start_round"], 25 if frozen else None)
            self.assertEqual(task["candidate"]["variant"], "Ours-FrozenHistory-v1" if frozen else "original")
        clean_H1 = [task for task in self.tasks if task["arm"] == "H1" and task["config"]["malicious_ratio"] == 0]
        self.assertEqual(len(clean_H1), 2)
        self.assertTrue(all(task["history_freeze_start_round"] == 25 for task in clean_H1))
        self.assertEqual(self.manifest["purpose"], "development_history_ablation")
        self.assertFalse(self.manifest["automatic_tpe"])
        self.assertFalse(self.manifest["formal_qualification_assessed"])
        self.assertFalse(self.manifest["next_stage_automatic"])

    def test_public_parameters_variant_freeze_and_development_seed_are_strict(self):
        for key, value in (("seed", 2026093011), ("automatic_tpe", True), ("both_arms_fresh", False),
                           ("freeze_applies_to_clean", False), ("chosen_candidate", "P3")):
            spec = deepcopy(self.spec)
            spec[key] = value
            with self.assertRaises(ValueError):
                protocol.validate_spec(spec)
        for mutate in (lambda s: s["public"].update(local_epochs=2),
                       lambda s: s["arms"][1].update(history_freeze_start_round=26),
                       lambda s: s["arms"][1].update(variant="original"),
                       lambda s: s["ours_parameters"].update(detector_distance_threshold=1.5)):
            spec = deepcopy(self.spec)
            mutate(spec)
            with self.assertRaises(ValueError):
                protocol.validate_spec(spec)

    def test_frozen_62_source_identities_are_subset_of_66(self):
        old = protocol.threshold.source_hashes()
        current = protocol.source_hashes()
        self.assertEqual((len(old), len(current)), (62, 66))
        self.assertEqual({key: current[key] for key in old}, old)

    def test_P0_health_or_reviewed_result_change_blocks_new_runs(self):
        task = next(t for t in self.threshold_tasks if t["arm"] == "P0")
        folder = self.threshold_output / "tasks" / task["task_id"]
        result = base.experiments.load_completed_results_snapshot(folder)[0]
        base.experiments._write_completed_results_snapshot(folder, [replace(result, nonfinite_updates=1)])
        with self.assertRaisesRegex(ValueError, "six original P0"):
            protocol.audit_reference(self.threshold_output, self.timing_output, self.clean_output, self.matched_output)
        changed = replace(result, final_accuracy=.5, records=[replace(r, accuracy=.5, error=.5) for r in result.records])
        base.experiments._write_completed_results_snapshot(folder, [changed])
        with self.assertRaisesRegex(ValueError, "reviewed history-ablation anchor"):
            protocol.audit_reference(self.threshold_output, self.timing_output, self.clean_output, self.matched_output)

    def test_missing_old_evidence_and_changed_task_environment_are_rejected(self):
        task = self.threshold_tasks[-1]
        folder = self.threshold_output / "tasks" / task["task_id"]
        metadata = deepcopy(self.metadata)
        metadata["actual_compute_device"]["name"] = "different GPU"
        base.write_json(folder / "environment.json", metadata)
        with self.assertRaisesRegex(ValueError, "numerical environment"):
            protocol.audit_reference(self.threshold_output, self.timing_output, self.clean_output, self.matched_output)
        (folder / base.experiments.COMPLETED_RESULTS_SNAPSHOT).unlink()
        with self.assertRaises(FileNotFoundError):
            protocol.audit_reference(self.threshold_output, self.timing_output, self.clean_output, self.matched_output)

    def test_reference_mutation_during_audit_is_blocked(self):
        real, calls = protocol.threshold_evidence_hashes, []
        def changed(*args):
            value = real(*args)
            if calls:
                value["concurrent mutation"] = "different"
            calls.append(True)
            return value
        with mock.patch.object(protocol, "threshold_evidence_hashes", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "changed while"):
                protocol.audit_reference(self.threshold_output, self.timing_output, self.clean_output, self.matched_output)

    def test_source_plan_variant_data_or_anchor_changes_cannot_resume(self):
        self.assertEqual(protocol.read_study(self.output, current_sources=True)[1], self.tasks)
        with mock.patch.object(protocol, "source_hashes", return_value={"changed": "hash"}):
            with self.assertRaisesRegex(ValueError, "identity changed"):
                protocol.read_study(self.output, current_sources=True)
        plan = runtime.read_json(self.output / "task_plans/history.json")
        plan["tasks"][6]["history_freeze_start_round"] = None
        base.write_json(self.output / "task_plans/history.json", plan)
        with self.assertRaisesRegex(ValueError, "task plan"):
            protocol.read_study(self.output)
        manifest = deepcopy(self.manifest)
        manifest["reference"]["p0_tasks"][0]["config"]["detector_window"] = 10
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            protocol.build_tasks(manifest)
        manifest = deepcopy(self.manifest)
        manifest["data_contract"] = {"different": "data"}
        manifest["fingerprint"] = base.digest({k: v for k, v in manifest.items() if k != "fingerprint"})
        base.write_json(self.output / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "dataset identity"):
            protocol.read_study(self.output)


if __name__ == "__main__":
    unittest.main()
