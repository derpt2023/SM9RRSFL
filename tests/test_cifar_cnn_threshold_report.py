"""Bounded four-point threshold reporting tests without downloads or training."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import cifar_cnn_threshold_report as report
import tests.test_cifar_timing_report as timing_fixtures


class ThresholdReportTests(unittest.TestCase):
    def setUp(self):
        self.fx = timing_fixtures.TimingReportTests()
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        old_tasks = [deepcopy(t) for t in self.fx.tasks if t["arm"] == "C"]
        for task in old_tasks:
            self.fx.complete(task, accuracy=.60)
        c_rows = report.timing.collect(self.fx.output, old_tasks)
        self.timing_output = self.fx.output
        self.fx.output = self.fx.root / "threshold"
        self.fx.output.mkdir()
        report.base.write_json(self.fx.output / "execution_environment.json", self.fx.environment)
        self.fx.tasks = []
        for arm, values in (("P0", (1.25, 1.25, 6.)), ("P1", (1.5, 1.25, 6.)),
                            ("P2", (1.5, .85, 1.5)), ("P3", (1.75, 1., 2.))):
            for old in old_tasks:
                task = deepcopy(old)
                task["arm"] = arm
                task["task_id"] = old["task_id"].replace("C_", arm + "_", 1)
                task["candidate"]["candidate_id"] = arm
                task["config"].update(zip(report.THRESHOLD_FIELDS, values))
                task["fingerprint"] = report.base.digest({k: v for k, v in task.items() if k != "fingerprint"})
                self.fx.tasks.append(task)
        self.reference = {"timing_manifest_fingerprint": "timing-fingerprint",
            "clean_manifest_fingerprint": "clean-fingerprint", "matched_manifest_fingerprint": "matched-fingerprint",
            "c_rows": c_rows, "clean_rows": deepcopy(self.fx.reference["clean_rows"]),
            "execution_environment": deepcopy(self.fx.environment), "timing_health_count": 21, "failure_rows": []}
        self.manifest = {"fingerprint": "threshold-fingerprint", "spec": {"protocol": "threshold-test-protocol"},
            "reference": deepcopy(self.reference), "source_sha256": {"test.py": "hash"}}
        self.protocol = SimpleNamespace(read_study=mock.Mock(return_value=(self.manifest, self.fx.tasks)),
            audit_reference=mock.Mock(return_value=deepcopy(self.reference)),
            source_hashes=mock.Mock(return_value={"test.py": "hash"}))
        patcher = mock.patch.dict(sys.modules, {"cifar_cnn_threshold_protocol": self.protocol})
        patcher.start()
        self.addCleanup(patcher.stop)

    def task(self, arm="P0", partition="iid", ratio=.7):
        return self.fx.task(arm, partition, ratio)

    def summarize(self):
        return report.summarize(self.fx.output, self.timing_output, self.fx.clean, self.fx.matched)

    def all_complete(self):
        self.fx.all_complete()

    def parsed(self, result):
        stream = io.StringIO()
        with redirect_stdout(stream):
            report.print_summary(result)
        text = stream.getvalue()
        lines = text.splitlines()
        self.assertEqual(lines[0], "=== CIFAR_THRESHOLD_BEGIN ===")
        self.assertEqual(lines[-1], "=== CIFAR_THRESHOLD_END ===")
        return text, [json.loads(line) for line in lines[1:-1]]

    def test_full_panel_actual_counts_means_and_no_automatic_winner(self):
        self.all_complete()
        result = self.summarize()
        self.assertEqual(result["complete_tasks"], 24)
        self.assertEqual(result["healthy_tasks"], 24)
        for candidate in result["candidates"]:
            self.assertEqual((candidate["accuracy_n"], candidate["attack_asr_n"]), (6, 4))
            self.assertAlmostEqual(candidate["mean_accuracy150"], .6)
            self.assertAlmostEqual(candidate["mean_attack_asr150"], .2)
            self.assertTrue(candidate["eligible_for_review"])
        choice = result["decision"]
        self.assertEqual(choice["action"], "review_fixed_panel_before_any_TPE")
        self.assertIsNone(choice["selected_candidate"])
        self.assertFalse(choice["score_computed"])
        self.assertFalse(choice["formal_qualification_assessed"])
        self.assertFalse(choice["automatic_TPE_started"])
        self.assertIn("remains unlocated", choice["repeatability_note"])

    def test_complete_P0_health_flip_requires_manual_review_without_new_gate(self):
        self.all_complete()
        self.fx.complete(self.task(), accuracy=.61, nonfinite=1)
        result = self.summarize()
        self.assertEqual(result["complete_tasks"], 24)
        self.assertEqual(result["healthy_tasks"], 23)
        choice = result["decision"]
        self.assertEqual(choice["action"], "review_fixed_panel_before_any_TPE")
        self.assertTrue(choice["requires_manual_repeatability_review"])
        self.assertEqual(len(choice["P0_old_C_health_changes"]), 1)
        self.assertEqual(choice["P0_old_C_health_changes"][0]["ratio"], .7)
        self.assertFalse(result["candidates"][0]["eligible_for_review"])
        self.assertIsNone(choice["selected_candidate"])

    def test_unhealthy_high_accuracy_retains_metrics_and_is_not_ranked(self):
        self.all_complete()
        self.fx.complete(self.task("P2"), accuracy=.99, asr=.01, nonfinite=1)
        result = self.summarize()
        candidate = next(c for c in result["candidates"] if c["arm"] == "P2")
        self.assertEqual(candidate["complete"], 6)
        self.assertEqual(candidate["healthy"], 5)
        self.assertAlmostEqual(candidate["mean_accuracy150"], (.99 + .6 * 5) / 6)
        self.assertFalse(candidate["eligible_for_review"])
        self.assertEqual([c["arm"] for c in result["candidates"]], list(report.ARMS))
        self.assertIsNone(result["decision"]["selected_candidate"])

    def test_missing_task_and_numerical_failure_are_not_filled_with_zero(self):
        task = self.task("P1", ratio=.7)
        for item in self.fx.tasks:
            if item != task:
                self.fx.complete(item)
        result = self.summarize()
        candidate = next(c for c in result["candidates"] if c["arm"] == "P1")
        self.assertEqual((candidate["accuracy_n"], candidate["attack_asr_n"]), (5, 3))
        self.assertAlmostEqual(candidate["mean_accuracy150"], .6)
        self.assertEqual(result["decision"]["action"], "resolve_incomplete_or_invalid_evidence")
        folder = report.runtime.ensure_identity(self.fx.output, task)
        report.base.write_json(folder / "failure.json", {"task_id": task["task_id"],
            "task_fingerprint": task["fingerprint"], "kind": "algorithm_numerical", "message": "nonfinite"})
        result = self.summarize()
        self.assertEqual(result["status"], "resolved_with_failures")
        row = next(r for r in result["rows"] if r["task_id"] == task["task_id"])
        self.assertNotIn("accuracy150", row)
        _, parsed = self.parsed(result)
        row = next(r for r in parsed if r.get("id") == task["task_id"])
        self.assertEqual(row["acc"], [None, None, None])
        self.assertEqual(row["status"], "algorithm_numerical")

    def test_clean_utility_remains_separate_and_inclusive(self):
        task = self.task(ratio=0.)
        self.fx.complete(task, accuracy=.58)
        result = self.summarize()
        row = next(r for r in result["rows"] if r["task_id"] == task["task_id"])
        self.assertTrue(row["healthy"])
        self.assertTrue(row["clean_utility_vs_C0"]["within_3pp"])
        self.fx.complete(task, accuracy=.579)
        row = next(r for r in self.summarize()["rows"] if r["task_id"] == task["task_id"])
        self.assertTrue(row["healthy"])
        self.assertFalse(row["clean_utility_vs_C0"]["within_3pp"])

    def test_pair_directions_and_clean_background_are_separate_from_attack_ASR(self):
        self.all_complete()
        self.fx.complete(self.task("P1", "dirichlet", .1), accuracy=.65, asr=.15)
        self.fx.complete(self.task("P2", "dirichlet", .1), accuracy=.62, asr=.08)
        self.fx.complete(self.task("P3", "iid", 0.), accuracy=.61, asr=.12)
        result = self.summarize()
        first = next(p for p in result["comparisons"]["P1_minus_P0"]["pairs"] if p["partition"] == "dirichlet" and p["ratio"] == .1)
        self.assertAlmostEqual(first["accuracy_change_pp"], 5.)
        self.assertAlmostEqual(first["attack_asr_change_pp"], -5.)
        second = next(p for p in result["comparisons"]["P2_minus_P1"]["pairs"] if p["partition"] == "dirichlet" and p["ratio"] == .1)
        self.assertAlmostEqual(second["accuracy_change_pp"], -3.)
        self.assertAlmostEqual(second["attack_asr_change_pp"], -7.)
        clean = next(p for p in result["comparisons"]["P3_minus_P0"]["pairs"] if p["partition"] == "iid" and p["ratio"] == 0.)
        self.assertIsNone(clean["attack_asr_change_pp"])
        self.assertAlmostEqual(clean["clean_background_change_pp"], 2.)

    def test_readonly_summary_does_not_touch_old_or_new_files_or_train(self):
        self.all_complete()
        before = self.fx.snapshot()
        with mock.patch.object(report.base, "load_split", side_effect=AssertionError("download")), \
                mock.patch.object(report.base.experiments, "run_measured_experiment", side_effect=AssertionError("training")):
            self.summarize()
        self.assertEqual(before, self.fx.snapshot())
        self.protocol.audit_reference.assert_called_once_with(self.timing_output, self.fx.clean, self.fx.matched)

    def test_corrupt_identity_and_incomplete_snapshot_never_qualify(self):
        task = self.task()
        folder = self.fx.complete(task)
        wrong = deepcopy(task)
        wrong["arm"] = "P1"
        report.base.write_json(folder / "task.json", wrong)
        row = next(r for r in self.summarize()["rows"] if r["task_id"] == task["task_id"])
        self.assertEqual(row["status"], "invalid_evidence")
        report.base.write_json(folder / "task.json", task)
        self.fx.complete(task, diagnostics=[], record_transform=lambda r: replace(r, stopped_round=14, records=r.records[:15]))
        row = next(r for r in self.summarize()["rows"] if r["task_id"] == task["task_id"])
        self.assertEqual(row["status"], "terminal_incomplete")
        self.assertIsNone(row["accuracy150"])

    def test_changed_reference_environment_and_sources_block_all_comparisons(self):
        self.all_complete()
        for failure in ("reference", "environment", "source"):
            with self.subTest(failure=failure):
                self.protocol.audit_reference.return_value = deepcopy(self.reference)
                self.protocol.source_hashes.return_value = {"test.py": "hash"}
                report.base.write_json(self.fx.output / "execution_environment.json", self.fx.environment)
                if failure == "reference":
                    self.protocol.audit_reference.return_value["c_rows"][0]["accuracy150"] = .61
                elif failure == "environment":
                    report.base.write_json(self.fx.output / "execution_environment.json", {"different": "GPU"})
                else:
                    self.protocol.source_hashes.return_value = {"changed": "source"}
                result = self.summarize()
                self.assertEqual(result["status"], "invalid_comparison_evidence")
                self.assertEqual(result["comparisons"], {})
                self.assertEqual(result["reference_c_rows"], [])
                self.assertIsNone(result["decision"]["selected_candidate"])
                self.assertTrue(all(not candidate["eligible_for_review"] for candidate in result["candidates"]))
                self.assertTrue(all(candidate["accuracy_n"] == 6 for candidate in result["candidates"]))

    def test_missing_P0_client_observations_require_manual_mechanism_review(self):
        self.all_complete()
        task = self.task()
        self.fx.complete(task, diagnostics=[])
        result = self.summarize()
        self.assertEqual(result["complete_tasks"], 24)
        self.assertEqual(result["healthy_tasks"], 24)
        self.assertEqual(result["decision"]["action"], "review_fixed_panel_before_any_TPE")
        self.assertTrue(result["decision"]["requires_manual_mechanism_review"])
        issue = result["decision"]["mechanism_evidence_issues"][0]
        self.assertEqual(issue["task_id"], task["task_id"])
        self.assertEqual(issue["windows"][0]["remaining_groups_without_observations"], ["malicious", "honest"])
        self.assertFalse(result["decision"]["automatic_TPE_started"])

    def test_raw_denominators_health_and_float_values_survive_compact_transport(self):
        self.all_complete()
        task = self.task()
        diagnostics = [self.fx.diagnostic(25, str(i), True, accepted=i < 4, admitted=i < 3, start=25) for i in range(5)]
        self.fx.complete(task, diagnostics=diagnostics, record_transform=lambda r: replace(r,
            records=[replace(record, malicious_weight_mass=1.0000000000000002) for record in r.records]))
        result = self.summarize()
        _, rows = self.parsed(result)
        row = next(r for r in rows if r.get("id") == task["task_id"])
        self.assertEqual(row["w1"]["M"][0], 70)
        self.assertEqual(row["w1"]["M"][2], 5)
        self.assertEqual(row["w1"]["M"][3], 65)
        self.assertEqual(row["w1"]["M"][4:6], [4, 3])
        self.assertEqual(row["w1"]["weights"][2], 1.0000000000000002)
        self.assertEqual(row["thresholds"], [1.25, 1.25, 6.])

    def test_24_tasks_6_C_and_2_C0_fit_under_50KB_without_reference_identity_mix(self):
        self.all_complete()
        result = self.summarize()
        text, rows = self.parsed(result)
        self.assertLessEqual(len(text.encode()), 50000)
        self.assertEqual(sum(r["type"] == "task" for r in rows), 24)
        self.assertEqual(sum(r["type"] == "reference_C" for r in rows), 6)
        self.assertEqual(sum(r["type"] == "reference_C0" for r in rows), 2)
        self.assertEqual(sum(r["type"] == "comparison" for r in rows), 4)
        self.assertEqual(rows[0]["reference_timing_manifest_fingerprint"], "timing-fingerprint")
        self.assertEqual(rows[0]["reference_clean_manifest_fingerprint"], "clean-fingerprint")
        self.assertEqual(rows[0]["reference_matched_manifest_fingerprint"], "matched-fingerprint")
        self.assertNotIn("A->B changes onset", text)

    def test_invalid_study_has_copyable_error_without_creating_output(self):
        self.protocol.read_study.side_effect = ValueError("bad manifest")
        before = self.fx.snapshot()
        result = self.summarize()
        text, rows = self.parsed(result)
        self.assertEqual(rows[0]["status"], "unavailable_or_invalid_study")
        self.assertIn("bad manifest", text)
        self.assertEqual(before, self.fx.snapshot())


if __name__ == "__main__":
    unittest.main()
