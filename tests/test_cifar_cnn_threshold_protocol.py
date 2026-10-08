"""Immutable 54-task evidence chain and the fresh 24-task fixed panel."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import cifar_cnn_threshold_protocol as protocol
from tests.test_cifar_timing_protocol import TimingFixture
from tests.test_cifar_six_pipeline import synthetic_run

base, runtime = protocol.base, protocol.runtime


class ThresholdFixture(TimingFixture):
    def setUp(self):
        super().setUp()
        self.timing_output, self.timing_tasks = self.output, self.tasks
        self.timing_manifest = self.manifest
        self.patch_value(protocol, "TIMING_FINGERPRINT", self.manifest["fingerprint"])
        for task in self.tasks:
            key = (task["config"]["partition"], task["config"]["malicious_ratio"])
            accuracy, asr, fp = protocol.REVIEWED_C[key] if task["arm"] == "C" else (.60, .10, 0)
            nonfinite = 0
            if task["arm"] in ("A", "B") and key[1] == 0:
                fp = 14
            if task["arm"] == "A" and key == ("iid", .7):
                nonfinite = 7
            self.finish(task, accuracy, .10 if asr is None else asr, fp, nonfinite)
        self.reference = protocol.audit_reference(self.timing_output, self.clean_output, self.matched_output)
        self.spec = runtime.read_json(protocol.DEFAULT_CONFIG)
        self.spec["timing_manifest_fingerprint"] = protocol.TIMING_FINGERPRINT
        self.output = self.temporary_output()
        self.manifest = protocol.build_manifest(self.spec, self.reference)
        self.tasks = protocol.build_tasks(self.manifest)
        base.write_json(self.output / "manifest.json", self.manifest)
        runtime.save_plan(self.output, "threshold", self.tasks, self.manifest)
        base.write_json(self.output / "execution_environment.json", self.environment)

    def finish(self, task, accuracy=.6, asr=.1, fp=0, nonfinite=0):
        folder = runtime.ensure_identity(self.output, task)
        result = synthetic_run(base.fl.ExperimentConfig(**task["config"]), accuracy, asr, nonfinite)
        diagnostics = []
        if task["method"] == "sm9rrs":
            for rd in range(1, 151):
                diagnostics.append(SimpleNamespace(round=rd, client_id="honest", is_malicious=False,
                    aggregation_accepted=True, aggregation_weight=.01, history_admitted=True,
                    revoked=False, attack_active=False))
                if result.malicious_clients:
                    diagnostics.append(SimpleNamespace(round=rd, client_id=result.malicious_clients[0], is_malicious=True,
                        aggregation_accepted=True, aggregation_weight=.01, history_admitted=True,
                        revoked=False, attack_active=rd >= task["config"]["attack_start_round"]))
        result = replace(result, diagnostics=diagnostics,
                         records=[replace(r, false_positive_revocations=fp if r.round >= 50 else 0) for r in result.records])
        base.experiments._write_completed_results_snapshot(folder, [result])
        base.write_json(folder / "environment.json", self.metadata)
        (folder / "attempts").mkdir(exist_ok=True)
        base.write_json(folder / "attempts/test.json", {"task_fingerprint": task["fingerprint"],
            "status": "complete", "wall_seconds": 100.})
        return folder

    def original_snapshot(self):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for root in (self.clean_output, self.matched_output, self.timing_output)
                for p in root.rglob("*") if p.is_file()}


class ThresholdProtocolTests(ThresholdFixture):
    def test_complete_reference_chain_retains_five_failures_and_does_not_train_or_write(self):
        before = self.original_snapshot()
        with mock.patch.object(base, "load_split", side_effect=AssertionError("download")), \
                mock.patch.object(base.experiments, "run_measured_experiment", side_effect=AssertionError("train")):
            observed = protocol.audit_reference(self.timing_output, self.clean_output, self.matched_output)
        self.assertEqual(observed, self.reference)
        self.assertEqual(observed["timing_health_count"], 21)
        self.assertEqual(len(observed["failure_rows"]), 5)
        self.assertEqual(len(observed["c_rows"]), 6)
        self.assertTrue(all(r["healthy"] for r in observed["c_rows"]))
        self.assertEqual(before, self.original_snapshot())

    def test_24_tasks_preserve_public_protocol_and_really_repeat_P0(self):
        self.assertEqual(len(self.tasks), 24)
        self.assertEqual(len({t["fingerprint"] for t in self.tasks}), 24)
        for candidate in protocol.CANDIDATES:
            tasks = [t for t in self.tasks if t["arm"] == candidate["id"]]
            self.assertEqual(len(tasks), 6)
            for task in tasks:
                old = next(t for t in self.timing_tasks if t["task_id"] == task["original_c_task_id"])
                changed = {k for k in task["config"] if task["config"][k] != old["config"][k]}
                self.assertLessEqual(changed, set(protocol.PARAMETER_FIELDS.values()))
                self.assertNotEqual(old["task_id"], task["task_id"])
                self.assertNotEqual(old["fingerprint"], task["fingerprint"])
                if candidate["id"] == "P0":
                    self.assertEqual(task["config"], old["config"])
                for field, actual in protocol.PARAMETER_FIELDS.items():
                    self.assertEqual(task["config"][actual], candidate[field])
                self.assertEqual(task["phase"], "validation")
                self.assertEqual(task["candidate"]["variant"], "original")
                self.assertEqual(task["config"]["crypto_mode"], "sm9")
        self.assertFalse(self.manifest["official_test_used_for_selection"])
        self.assertFalse(self.manifest["next_stage_automatic"])
        self.assertFalse(self.manifest["automatic_tpe"])
        self.assertFalse(self.manifest["numerical_policy_modified"])

    def test_only_predeclared_fixed_points_and_development_seed_are_permitted(self):
        for key, value in (("seed", 2026093011), ("automatic_tpe", True), ("p0_fresh_repeat", False),
                           ("detector_window", 10), ("attack_start_round", 26)):
            spec = deepcopy(self.spec)
            spec[key] = value
            with self.assertRaises(ValueError):
                protocol.validate_spec(spec)
        for key, value in (("lr", .01), ("local_epochs", 2), ("rounds", 163)):
            spec = deepcopy(self.spec)
            spec["public"][key] = value
            with self.assertRaises(ValueError):
                protocol.validate_spec(spec)
        spec = deepcopy(self.spec)
        spec["candidates"][2]["kappa"] = .9
        with self.assertRaises(ValueError):
            protocol.validate_spec(spec)

    def test_old_58_sources_remain_exact_subset_of_62(self):
        old = protocol.timing.source_hashes()
        self.assertEqual(len(old), 58)
        current = protocol.source_hashes()
        self.assertEqual(len(current), 62)
        self.assertEqual({k: current[k] for k in old}, old)

    def test_unhealthy_C_is_blocked_but_other_failed_conditions_remain_valid_evidence(self):
        task = next(t for t in self.timing_tasks if t["arm"] == "C")
        folder = self.timing_output / "tasks" / task["task_id"]
        result = base.experiments.load_completed_results_snapshot(folder)[0]
        base.experiments._write_completed_results_snapshot(folder, [replace(result, nonfinite_updates=1)])
        with self.assertRaisesRegex(ValueError, "six original C"):
            protocol.audit_reference(self.timing_output, self.clean_output, self.matched_output)

    def test_changed_anchor_measurements_missing_evidence_and_wrong_environment_are_blocked(self):
        task = next(t for t in self.timing_tasks if t["arm"] == "C")
        folder = self.timing_output / "tasks" / task["task_id"]
        snapshot = folder / base.experiments.COMPLETED_RESULTS_SNAPSHOT
        original_bytes = snapshot.read_bytes()
        result = base.experiments.load_completed_results_snapshot(folder)[0]
        changed = replace(result, final_accuracy=.5, records=[replace(r, accuracy=.5, error=.5) for r in result.records])
        base.experiments._write_completed_results_snapshot(folder, [changed])
        with self.assertRaisesRegex(ValueError, "reviewed development anchor"):
            protocol.audit_reference(self.timing_output, self.clean_output, self.matched_output)
        snapshot.write_bytes(original_bytes)
        metadata = deepcopy(self.metadata)
        metadata["actual_compute_device"]["name"] = "changed GPU"
        base.write_json(folder / "environment.json", metadata)
        with self.assertRaisesRegex(ValueError, "numerical environment"):
            protocol.audit_reference(self.timing_output, self.clean_output, self.matched_output)
        snapshot.unlink()
        with self.assertRaises(FileNotFoundError):
            protocol.audit_reference(self.timing_output, self.clean_output, self.matched_output)

    def test_concurrent_evidence_mutation_rejected(self):
        real = protocol.timing_evidence_hashes
        calls = []
        def changed(*args):
            hashes = real(*args)
            if calls:
                hashes["changed during read"] = "different"
            calls.append(True)
            return hashes
        with mock.patch.object(protocol, "timing_evidence_hashes", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "changed while"):
                protocol.audit_reference(self.timing_output, self.clean_output, self.matched_output)

    def test_manifest_source_reference_plan_and_data_mismatch_block_resume(self):
        self.assertEqual(protocol.read_study(self.output, current_sources=True)[1], self.tasks)
        with mock.patch.object(protocol, "source_hashes", return_value={"changed": "hash"}):
            with self.assertRaisesRegex(ValueError, "identity changed"):
                protocol.read_study(self.output, current_sources=True)
        altered = deepcopy(self.manifest)
        altered["data_contract"] = {**altered["data_contract"], "different": True}
        altered["fingerprint"] = base.digest({k: v for k, v in altered.items() if k != "fingerprint"})
        base.write_json(self.output / "manifest.json", altered)
        with self.assertRaisesRegex(ValueError, "dataset identity"):
            protocol.read_study(self.output)
        base.write_json(self.output / "manifest.json", self.manifest)
        plan = runtime.read_json(self.output / "task_plans/threshold.json")
        plan["tasks"][0]["model"] = "resnet18_gn2"
        base.write_json(self.output / "task_plans/threshold.json", plan)
        with self.assertRaisesRegex(ValueError, "task plan"):
            protocol.read_study(self.output)

    def test_original_C_cannot_be_relabelled_or_swapped(self):
        altered = deepcopy(self.manifest)
        altered["reference"]["c_tasks"][0]["config"]["local_epochs"] = 2
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            protocol.build_tasks(altered)
        altered = deepcopy(self.manifest)
        altered["reference"]["chosen_arm"] = "B"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            protocol.build_tasks(altered)

    def test_parent_finishes_only_24_new_tasks_without_mutating_original_54(self):
        import run_cifar_cnn_threshold_panel as runner
        import cifar_cnn_threshold_report as report
        before = self.original_snapshot()
        args = SimpleNamespace(output=self.output, timing_output=self.timing_output, clean_output=self.clean_output,
            matched_output=self.matched_output, devices=["cuda:0"], data_dir=None)
        def finish(*_):
            for task in self.tasks:
                self.finish(task)
        with mock.patch.object(runner, "execute", side_effect=finish), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run_parent(args, self.spec, self.reference), 0)
        summary = report.summarize(self.output, self.timing_output, self.clean_output, self.matched_output)
        self.assertEqual(summary["complete_tasks"], 24)
        self.assertTrue(summary["reference_verified"])
        self.assertFalse(summary["decision"]["next_stage_started"])
        self.assertIsNone(summary["decision"]["selected_candidate"])
        self.assertEqual(before, self.original_snapshot())


if __name__ == "__main__":
    unittest.main()
