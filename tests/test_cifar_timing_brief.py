"""Compact read-only transport retains timing evidence and audits clean A/B."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import unittest
from unittest import mock

import summarize_cifar_timing as brief
import tests.test_cifar_timing_report as fixtures


class TimingBriefTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures.TimingReportTests()
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        # The report fixture supplies real synthetic pickle/task evidence.
        # Only frozen study discovery is replaced; report/brief code is real.
        environment = {"actual_compute_device": {"name": "synthetic", "compute_capability": [8, 9]},
                       "environment": {}, "driver_versions": []}
        self.fx.environment = environment
        for ref in (self.fx.reference, self.fx.manifest["reference"]):
            ref.update(execution_environment=deepcopy(environment), clean_manifest_fingerprint="c" * 64,
                       matched_manifest_fingerprint="m" * 64)
        self.fx.protocol.audit_reference.return_value = deepcopy(self.fx.reference)
        brief.protocol.base.write_json(self.fx.output / "execution_environment.json", environment)
        patcher = mock.patch.object(brief.protocol, "read_study", side_effect=self.fx.protocol.read_study)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.args = ["--output", str(self.fx.output), "--clean-output", str(self.fx.clean),
                     "--matched-output", str(self.fx.matched)]

    def all_complete(self):
        self.fx.all_complete()
        metadata = {"actual_compute_device": {"name": "synthetic", "compute_capability": [8, 9]},
                    "environment": {}}
        for task in self.fx.tasks:
            brief.protocol.base.write_json(self.fx.output / "tasks" / task["task_id"] / "environment.json", metadata)

    def parse(self, text):
        lines = text.splitlines()
        self.assertEqual(lines[0], brief.BEGIN)
        self.assertEqual(lines[-1], brief.END)
        return [json.loads(line) for line in lines[1:-1]]

    def run_brief(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = brief.main(self.args)
        return code, stream.getvalue(), self.parse(stream.getvalue())

    def test_all_26_tasks_two_C0_references_and_audits_fit_compact_jsonl(self):
        self.all_complete()
        code, text, rows = self.run_brief()
        self.assertEqual(code, 0)
        self.assertEqual(sum(r["type"] == "task" for r in rows), 26)
        self.assertEqual(sum(r["type"] == "reference" for r in rows), 2)
        self.assertEqual(sum(r["type"] == "clean_pair_audit" for r in rows), 2)
        self.assertEqual(rows[0]["task_lines"], 26)
        self.assertEqual(rows[0]["healthy_tasks"], 26)
        self.assertLessEqual(len(text.encode("utf-8")), 35000)
        self.assertLessEqual(max(len(line.encode("utf-8")) for line in text.splitlines()), 4000)
        self.assertNotIn("coverage_by_round", text)
        self.assertNotIn('"values":', text)

    def test_original_group_counts_raw_small_overflow_and_zero_denominator_retained(self):
        task = self.fx.task()
        raw = 1.0000000000000002
        self.fx.complete(task, diagnostics=[], record_transform=lambda result: replace(result,
            records=[replace(r, malicious_weight_mass=raw, honest_weight_loss=.1) for r in result.records]))
        report = self.fx.summarize()
        before = deepcopy(report)
        rows = self.parse(brief.render(report))
        row = next(r for r in rows if r.get("id") == task["task_id"])
        original = next(r for r in report["rows"] if r["task_id"] == task["task_id"])
        for key, source in (("w1", original["mechanism_windows"]["first_round"]),
                            ("w5", original["mechanism_windows"]["first_five_rounds"]),
                            ("wa", original["full_attack_period"])):
            self.assertEqual(row[key]["M"], [source["client_diagnostics"]["malicious"][k] for k in brief.GROUP_FIELDS])
            self.assertEqual(row[key]["H"], [source["client_diagnostics"]["honest"][k] for k in brief.GROUP_FIELDS])
            self.assertEqual(row[key]["weights"][2], raw)
        self.assertEqual(row["w1"]["M"][2], 0)
        self.assertEqual(row["w1"]["M"][3], 70)
        self.assertIn("denominator zero means not computable", json.dumps(rows[1]))
        self.assertEqual(before, report)

    def test_complete_unhealthy_keeps_metrics_reasons_and_upstream_decision(self):
        self.all_complete()
        task = self.fx.task()
        self.fx.complete(task, accuracy=.4321, asr=.8765, nonfinite=1)
        code, _, rows = self.run_brief()
        self.assertEqual(code, 0)  # Transport completeness is not a new health gate.
        self.assertEqual(rows[0]["healthy_tasks"], 25)
        row = next(r for r in rows if r.get("id") == task["task_id"])
        self.assertEqual(row["acc"][-1], .4321)
        self.assertEqual(row["asr"], .8765)
        self.assertFalse(row["healthy"])
        self.assertIn("nonfinite_updates", row["reasons"])
        self.assertEqual(row["nf"], 1)
        decision = next(r for r in rows if r["type"] == "decision")
        self.assertEqual(decision["action"], "review_unhealthy_completed_runs_before_next_stage")
        self.assertIsNone(decision["selected_arm"])
        self.assertFalse(decision["next_stage_started"])

    def test_missing_and_early_termination_do_not_impute_final_results(self):
        task = self.fx.task()
        self.fx.complete(task, diagnostics=[], record_transform=lambda result: replace(result,
            stopped_round=14, records=result.records[:15]))
        code, _, rows = self.run_brief()
        self.assertEqual(code, 2)
        row = next(r for r in rows if r.get("id") == task["task_id"])
        self.assertEqual(row["status"], "terminal_incomplete")
        self.assertEqual(row["acc"], [None, None, None])
        self.assertIsNone(row["asr"])
        self.assertEqual(row["last"][0], 14)
        pending = next(r for r in rows if r.get("status") == "pending")
        self.assertEqual(pending["acc"], [None, None, None])
        self.assertIsNone(pending["healthy"])

    def test_source_reference_or_environment_failure_never_claims_valid_comparison(self):
        self.all_complete()
        for kind in ("source", "reference", "environment"):
            with self.subTest(kind=kind):
                self.fx.protocol.source_hashes.return_value = {"test.py": "hash"}
                self.fx.protocol.audit_reference.return_value = deepcopy(self.fx.reference)
                brief.protocol.base.write_json(self.fx.output / "execution_environment.json", self.fx.environment)
                if kind == "source":
                    self.fx.protocol.source_hashes.return_value = {"changed": "hash"}
                elif kind == "reference":
                    self.fx.protocol.audit_reference.return_value["chosen_setting"] = "other"
                else:
                    brief.protocol.base.write_json(self.fx.output / "execution_environment.json", {"different": True})
                code, _, rows = self.run_brief()
                self.assertEqual(code, 2)
                self.assertEqual(rows[0]["status"], "invalid_comparison_evidence")
                field = {"source": "source_matches_current", "reference": "reference_verified",
                         "environment": "execution_environment_compatible"}[kind]
                self.assertFalse(rows[0][field])

    def test_actual_report_and_pair_audit_are_readonly_without_GPU_data_or_training(self):
        self.all_complete()
        before = self.fx.snapshot()
        with mock.patch.object(brief.protocol.base, "load_split", side_effect=AssertionError("data load")), \
                mock.patch.object(brief.protocol.base.experiments, "run_measured_experiment", side_effect=AssertionError("training")), \
                mock.patch("run_cifar_six_with_progress.discover_gpus", side_effect=AssertionError("GPU probe")):
            self.assertEqual(self.run_brief()[0], 0)
        self.assertEqual(before, self.fx.snapshot())

    def test_equal_clean_scientific_records_are_zero_with_recorded_identity_hashes(self):
        self.all_complete()
        context = brief.audit_context(self.fx.output)
        self.assertEqual(context["status"], "audited")
        self.assertEqual(context["recorded_source_map_sha256"], brief.protocol.base.digest(self.fx.manifest["source_sha256"]))
        self.assertEqual(context["clean_manifest_fingerprint"], "c" * 64)
        self.assertEqual(context["matched_manifest_fingerprint"], "m" * 64)
        for audit in context["audits"]:
            self.assertTrue(audit["only_expected_attack_start_difference"])
            self.assertEqual(audit["config_differences"], {"attack_start_round": {"A": 12, "B": 25}})
            self.assertEqual(audit["differing_rounds"], 0)
            self.assertIsNone(audit["first_differing_round"])
            self.assertEqual(set(audit["round0"]["A"]), set(brief.PAIR_FIELDS))

    def test_earliest_scientific_difference_and_physical_devices_are_reported_without_cause(self):
        self.all_complete()
        task = self.fx.task("B", "dirichlet", 0.)
        def change(result):
            return replace(result, records=[replace(row, accepted_updates=99 if row.round == 2 else row.accepted_updates,
                accuracy=.604 if row.round >= 3 else row.accuracy) for row in result.records], final_accuracy=.604)
        self.fx.complete(task, record_transform=change)
        for arm, uuid in (("A", "GPU-A"), ("B", "GPU-B")):
            task = self.fx.task(arm, "dirichlet", 0.)
            metadata = {"requested_device": "cuda:0" if arm == "A" else "cuda:1", "environment": {},
                "actual_compute_device": {"name": "synthetic", "compute_capability": [8, 9],
                                          "uuid": uuid, "logical_device": "cuda:0"}}
            brief.protocol.base.write_json(self.fx.output / "tasks" / task["task_id"] / "environment.json", metadata)
        context = brief.audit_context(self.fx.output)
        row = next(r for r in context["audits"] if r["partition"] == "dirichlet")
        self.assertEqual(row["status"], "audited")
        self.assertEqual(row["differing_rounds"], 149)
        self.assertEqual(row["first_differing_round"], {"round": 2, "fields": {"accepted_updates": {"A": 100, "B": 99}}})
        self.assertTrue(all(row["environment_checks"].values()))
        self.assertEqual(row["recorded_devices"]["B"]["actual_compute_device"]["uuid"], "GPU-B")
        self.assertNotIn("cause", row)

    def test_pair_configuration_and_normalized_environment_mismatches_remain_explicit(self):
        task = self.fx.task("B", "iid", 0.)
        task["config"]["lr"] = .123
        self.all_complete()
        metadata = {"environment": {}, "actual_compute_device": {"name": "other", "compute_capability": [8, 9]}}
        brief.protocol.base.write_json(self.fx.output / "tasks" / task["task_id"] / "environment.json", metadata)
        row = brief.audit_context(self.fx.output)["audits"][0]
        self.assertEqual(row["status"], "evidence_mismatch")
        self.assertEqual(row["config_differences"]["lr"], {"A": .05, "B": .123})
        self.assertFalse(row["only_expected_attack_start_difference"])
        self.assertFalse(row["environment_checks"]["B_matches_root"])

    def test_missing_pair_environment_or_snapshot_is_not_auditable(self):
        self.all_complete()
        task = self.fx.task("A", "iid", 0.)
        (self.fx.output / "tasks" / task["task_id"] / "environment.json").unlink()
        task = self.fx.task("B", "dirichlet", 0.)
        (self.fx.output / "tasks" / task["task_id"] / brief.protocol.base.experiments.COMPLETED_RESULTS_SNAPSHOT).unlink()
        context = brief.audit_context(self.fx.output)
        self.assertEqual(context["status"], "unavailable_or_mismatched")
        self.assertEqual([r["status"] for r in context["audits"]], ["unavailable", "unavailable"])
        self.assertTrue(all("first_differing_round" not in r for r in context["audits"]))

    def test_read_exception_and_nonfinite_serialization_have_short_complete_error_markers(self):
        for failure in (RuntimeError("x" * 20000), None):
            with self.subTest(failure=type(failure).__name__):
                bad = {"status": "incomplete", "rows": [], "execution_hardware": {"bad": float("nan")}}
                with mock.patch.object(brief.original, "summarize", side_effect=failure, return_value=bad):
                    code, text, rows = self.run_brief()
                self.assertEqual(code, 2)
                self.assertEqual(rows[0]["status"], "brief_unavailable_or_invalid")
                self.assertLess(len(text.encode()), 5000)
                self.assertEqual(text.count(brief.BEGIN), 1)
                self.assertEqual(text.count(brief.END), 1)

    def test_manifest_change_between_reads_invalidates_combination_and_removes_pair_audits(self):
        self.all_complete()
        context = brief.audit_context(self.fx.output)
        context["audit_manifest_fingerprint"] = "changed"
        with mock.patch.object(brief, "audit_context", return_value=context):
            code, _, rows = self.run_brief()
        self.assertEqual(code, 2)
        self.assertEqual(rows[0]["status"], "invalid_comparison_evidence")
        self.assertIn("manifest changed", rows[0]["brief_error"])
        self.assertNotIn("clean_manifest_fingerprint", rows[0])
        self.assertNotIn("recorded_source_map_sha256", rows[0])
        self.assertFalse(any(r["type"] == "clean_pair_audit" for r in rows))


if __name__ == "__main__":
    unittest.main()
