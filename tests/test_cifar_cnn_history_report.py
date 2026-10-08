"""Synthetic snapshot audits for the fixed history intervention; never train."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest import mock

import cifar_cnn_history_protocol as protocol
import cifar_cnn_history_report as report
import tests.test_cifar_cnn_history_protocol as fixtures


class HistoryReportTests(unittest.TestCase):
    def setUp(self):
        self.fx = fixtures.HistoryFixture()
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)

    def task(self, arm="H1", partition="iid", ratio=.7):
        return next(t for t in self.fx.tasks if t["arm"] == arm and
                    t["config"]["partition"] == partition and t["config"]["malicious_ratio"] == ratio)

    def finish(self, task, **kwargs):
        folder = self.fx.finish(task, **kwargs)
        result = report.base.checked_completed(self.fx.output, task)
        diagnostics = []
        malicious = list(result.malicious_clients)
        for rd in range(1, 151):
            previous = result.records[rd - 1]
            groups = ((True, malicious[previous.true_positive_revocations:]),
                      (False, ["honest-" + str(i) for i in range(task["config"]["num_clients"] - len(malicious))]
                       [previous.false_positive_revocations:]))
            for is_malicious, identities in groups:
                for identity in identities:
                    diagnostics.append(SimpleNamespace(round=rd, client_id=identity, is_malicious=is_malicious,
                        aggregation_accepted=True, aggregation_weight=.01,
                        history_admitted=not (task["arm"] == "H1" and rd >= 25),
                        history_frozen=False, revoked=False,
                        attack_active=is_malicious and rd >= 25 and task["config"]["attack"] != "none"))
        report.base.experiments._write_completed_results_snapshot(folder, [replace(result, diagnostics=diagnostics)])
        return folder

    def all_complete(self):
        for task in self.fx.tasks:
            self.finish(task)

    def rewrite(self, task, transform):
        folder = self.fx.output / "tasks" / task["task_id"]
        result = report.base.checked_completed(self.fx.output, task)
        report.base.experiments._write_completed_results_snapshot(folder, [transform(result)])

    def summarize(self):
        return report.summarize(self.fx.output, self.fx.threshold_output, self.fx.timing_output,
                                self.fx.clean_output, self.fx.matched_output)

    def row(self, summary, task):
        return next(r for r in summary["rows"] if r["task_id"] == task["task_id"])

    def parsed(self, summary):
        stream = io.StringIO()
        with redirect_stdout(stream):
            report.print_summary(summary)
        text = stream.getvalue()
        lines = text.splitlines()
        self.assertEqual(lines[0], "=== CIFAR_HISTORY_BEGIN ===")
        self.assertEqual(lines[-1], "=== CIFAR_HISTORY_END ===")
        return text, [json.loads(line) for line in lines[1:-1]]

    def test_full_panel_checks_clean_and_attack_and_never_selects_or_starts(self):
        self.all_complete()
        result = self.summarize()
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["evidence_ready"])
        self.assertEqual((result["complete_tasks"], result["healthy_tasks"]), (12, 12))
        for row in result["rows"]:
            if row["arm"] == "H1":
                self.assertTrue(row["history_audit"]["verified"])
                self.assertEqual(row["history_audit"]["observed_admissions"], 0)
                self.assertEqual(row["history_audit"]["observed_verified_finite_updates"], 12600)
        for arm in result["candidates"]:
            self.assertEqual((arm["accuracy_n"], arm["attack_accuracy_n"], arm["attack_asr_n"]), (6, 4, 4))
            self.assertAlmostEqual(arm["mean_accuracy150"], .6)
            self.assertAlmostEqual(arm["mean_attack_accuracy150"], .6)
            self.assertAlmostEqual(arm["mean_attack_asr150"], .1)
            self.assertTrue(arm["eligible_for_review"])
        decision = result["decision"]
        self.assertEqual(decision["action"], "review_history_ablation_before_any_next_stage")
        self.assertIsNone(decision["selected_arm"])
        for field in ("score_computed", "formal_qualification_assessed", "next_stage_started", "automatic_TPE_started", "NoPermanent_started"):
            self.assertFalse(decision[field])

    def test_late_clean_admission_blocks_ready_despite_old_history_frozen_true(self):
        self.all_complete()
        task = self.task(ratio=0.)
        def admitted(result):
            for d in result.diagnostics:
                d.history_frozen = True
            result.diagnostics[-1].history_admitted = True
            return result
        self.rewrite(task, admitted)
        result = self.summarize()
        row = self.row(result, task)
        self.assertTrue(row["healthy"])
        self.assertEqual(row["history_audit"]["status"], "freeze_violated")
        self.assertEqual(row["history_audit"]["first_observed_admission"][0], 150)
        self.assertFalse(result["evidence_ready"])
        self.assertEqual(result["status"], "invalid_intervention_evidence")
        self.assertTrue(all(not c["eligible_for_review"] for c in result["candidates"]))

    def test_missing_non_boolean_and_true_admission_flags_are_distinct_evidence_failures(self):
        task = self.task()
        for mutation in ("missing", "integer", "true"):
            with self.subTest(mutation=mutation):
                self.finish(task)
                def corrupt(result):
                    d = next(d for d in result.diagnostics if d.round == 80)
                    if mutation == "missing":
                        del d.history_admitted
                    else:
                        d.history_admitted = 0 if mutation == "integer" else True
                    return result
                self.rewrite(task, corrupt)
                result = self.summarize()
                row = self.row(result, task)
                self.assertFalse(row["history_audit"]["verified"])
                self.assertFalse(result["evidence_ready"])
                if mutation != "true":
                    self.assertEqual(row["status"], "invalid_evidence")
                    self.assertEqual(row["history_audit"]["invalid_admission_flags"], 1)
                else:
                    self.assertEqual(row["history_audit"]["status"], "freeze_violated")

    def test_five_of_seventy_observations_keep_five_denominator_and_unobserved_gap(self):
        task = self.task()
        self.finish(task)
        def limited(result):
            return replace(result, diagnostics=[d for d in result.diagnostics if
                d.round == 25 and d.is_malicious and int(d.client_id) < 5])
        self.rewrite(task, limited)
        result = self.summarize()
        row = self.row(result, task)
        group = row["mechanism_windows"]["first_round"]["client_diagnostics"]["malicious"]
        self.assertEqual((group["original_clients"], group["observed_verified_finite_updates"],
                          group["unobserved_remaining_client_rounds"]), (70, 5, 65))
        self.assertEqual(group["aggregation_acceptance_rate"], 1.)
        self.assertEqual(group["history_admission_rate"], 0.)
        self.assertEqual(row["history_audit"]["status"], "incomplete_client_coverage")
        self.rewrite(task, lambda r: replace(r, diagnostics=[]))
        row = self.row(self.summarize(), task)
        self.assertIsNone(row["history_audit"]["observed_admission_rate"])
        self.assertIsNone(row["mechanism_windows"]["first_round"]["client_diagnostics"]["malicious"]["history_admission_rate"])

    def test_prior_revoked_absence_is_not_a_missing_observation(self):
        task = self.task()
        self.finish(task)
        def revoked(result):
            records = [replace(r, true_positive_revocations=70 if r.round >= 24 else 0) for r in result.records]
            diagnostics = [d for d in result.diagnostics if not (d.round >= 25 and d.is_malicious)]
            return replace(result, records=records, diagnostics=diagnostics)
        self.rewrite(task, revoked)
        row = self.row(self.summarize(), task)
        self.assertTrue(row["history_audit"]["verified"])
        group = row["history_audit"]["window"]["client_diagnostics"]["malicious"]
        self.assertEqual(group["remaining_client_rounds"], 0)
        self.assertEqual(group["status"], "no_remaining_clients")
        self.assertIsNone(group["history_admission_rate"])
        self.rewrite(task, lambda r: replace(r,
            records=[replace(record, false_positive_revocations=30 if record.round >= 24 else 0) for record in r.records],
            diagnostics=[d for d in r.diagnostics if d.round < 25]))
        row = self.row(self.summarize(), task)
        self.assertEqual(row["history_audit"]["status"], "not_exercised_no_post_freeze_observations")
        self.assertFalse(row["history_audit"]["verified"])
        self.assertIsNone(row["history_audit"]["observed_admission_rate"])

    def test_prefix_round25_and_old_repeat_audits_show_exact_fields_without_causal_claim(self):
        self.all_complete()
        task = self.task(partition="dirichlet", ratio=0.)
        self.rewrite(task, lambda r: replace(r, records=[replace(record,
            attack_target_confidence=.125 if record.round == 3 else .25 if record.round == 25 else record.attack_target_confidence)
            for record in r.records]))
        result = self.summarize()
        audit = next(a for a in result["record_pair_audits"] if a["partition"] == "dirichlet" and a["ratio"] == 0.)
        self.assertEqual(audit["H1_minus_H0_pre25"]["compared_rounds"], 25)
        self.assertEqual(audit["H1_minus_H0_pre25"]["first_differing_round"]["round"], 3)
        self.assertEqual(audit["H1_minus_H0_round25"]["first_differing_round"]["round"], 25)
        self.assertEqual(audit["H0_minus_old_P0"]["compared_rounds"], 151)
        self.assertTrue(result["evidence_ready"])
        self.assertIn("remains unlocated", result["decision"]["repeatability_note"])
        self.assertTrue(result["decision"]["requires_manual_repeatability_review"])
        issue = next(i for i in result["decision"]["record_repeatability_issues"] if i["partition"] == "dirichlet" and i["ratio"] == 0.)
        self.assertIn(["H1_minus_H0_pre25", "recorded_fields_differ", 3], issue["reasons"])
        self.assertIn(["H1_minus_H0_round25", "recorded_fields_differ", 25], issue["reasons"])

    def test_H0_missing_mechanism_coverage_also_blocks_ready_and_candidate_review(self):
        self.all_complete()
        task = self.task("H0", ratio=0.)
        self.rewrite(task, lambda r: replace(r, diagnostics=[d for d in r.diagnostics if d.round < 25]))
        result = self.summarize()
        self.assertTrue(result["history_intervention_verified"])
        self.assertFalse(result["mechanism_coverage_verified"])
        self.assertFalse(result["evidence_ready"])
        self.assertEqual(result["status"], "incomplete_mechanism_evidence")
        self.assertEqual((result["complete_tasks"], result["healthy_tasks"]), (12, 12))
        self.assertTrue(all(not c["eligible_for_review"] for c in result["candidates"]))
        self.assertIn(task["task_id"], result["decision"]["tasks_with_incomplete_mechanism_coverage"])

    def test_missing_reference_round_audit_is_manual_review_and_never_ready(self):
        self.all_complete()
        old = next(t for t in self.fx.threshold_tasks if t["arm"] == "P0")
        real = report.base.checked_completed
        def concurrent_missing(output, task):
            if output == self.fx.threshold_output and task == old:
                return None
            return real(output, task)
        # Simulate a reference disappearing after its initial immutable audit.
        with mock.patch.object(protocol, "audit_reference", return_value=self.fx.reference), \
                mock.patch.object(report.base, "checked_completed", side_effect=concurrent_missing):
            result = self.summarize()
        self.assertFalse(result["record_pair_audits_available"])
        self.assertFalse(result["evidence_ready"])
        self.assertTrue(result["decision"]["requires_manual_repeatability_review"])
        self.assertTrue(any(["H0_minus_old_P0", "unavailable", None] in issue["reasons"]
                            for issue in result["decision"]["record_repeatability_issues"]))

    def test_missing_task_and_numeric_failure_preserve_actual_n_without_zero_fill(self):
        missing = self.task(ratio=.1)
        for task in self.fx.tasks:
            if task != missing:
                self.finish(task)
        result = self.summarize()
        candidate = next(c for c in result["candidates"] if c["arm"] == "H1")
        self.assertEqual((candidate["accuracy_n"], candidate["attack_accuracy_n"], candidate["attack_asr_n"]), (5, 3, 3))
        self.assertAlmostEqual(candidate["mean_attack_asr150"], .1)
        folder = report.runtime.ensure_identity(self.fx.output, missing)
        report.base.write_json(folder / "failure.json", {"task_id": missing["task_id"], "task_fingerprint": missing["fingerprint"],
            "kind": "algorithm_numerical", "message": "nonfinite update"})
        result = self.summarize()
        self.assertEqual(result["status"], "resolved_with_failures")
        _, parsed = self.parsed(result)
        row = next(r for r in parsed if r.get("id") == missing["task_id"])
        self.assertEqual(row["acc"], [None, None, None])
        self.assertEqual(row["status"], "algorithm_numerical")

    def test_unhealthy_completed_retains_metrics_and_H0_health_flip_requires_review(self):
        self.all_complete()
        task = self.task("H0", ratio=0.)
        self.finish(task, accuracy=.62, fp=11)
        result = self.summarize()
        row = self.row(result, task)
        self.assertEqual(row["accuracy150"], .62)
        self.assertFalse(row["healthy"])
        self.assertEqual(result["healthy_tasks"], 11)
        self.assertTrue(result["decision"]["requires_manual_repeatability_review"])
        self.assertFalse(result["candidates"][0]["eligible_for_review"])
        self.assertTrue(row["clean_utility_vs_C0"]["within_3pp"])
        self.assertIsNone(result["decision"]["selected_arm"])

    def test_pair_direction_and_attack_only_means_remain_correct(self):
        self.all_complete()
        self.finish(self.task(ratio=.1), accuracy=.64, asr=.03)
        self.finish(self.task(ratio=0.), accuracy=.65, asr=.13)
        result = self.summarize()
        attacked = next(p for p in result["comparisons"]["H1_minus_H0"]["pairs"] if p["partition"] == "iid" and p["ratio"] == .1)
        self.assertAlmostEqual(attacked["accuracy_change_pp"], 4.)
        self.assertAlmostEqual(attacked["attack_asr_change_pp"], -7.)
        clean = next(p for p in result["comparisons"]["H1_minus_H0"]["pairs"] if p["partition"] == "iid" and not p["ratio"])
        self.assertIsNone(clean["attack_asr_change_pp"])
        self.assertAlmostEqual(clean["clean_background_change_pp"], 3.)
        candidate = result["candidates"][1]
        self.assertAlmostEqual(candidate["mean_accuracy150"], (.64 + .65 + 4 * .6) / 6)
        self.assertAlmostEqual(candidate["mean_attack_accuracy150"], (.64 + 3 * .6) / 4)
        self.assertAlmostEqual(candidate["mean_attack_asr150"], (.03 + 3 * .1) / 4)

    def test_source_reference_root_and_task_environment_changes_block_readiness(self):
        self.all_complete()
        for mode in ("source", "reference", "environment", "task_environment"):
            with self.subTest(mode=mode):
                if mode == "source":
                    patcher = mock.patch.object(protocol, "source_hashes", return_value={"different": "hash"})
                elif mode == "reference":
                    changed = deepcopy(self.fx.reference)
                    changed["p0_rows"][0]["accuracy150"] = .1
                    patcher = mock.patch.object(protocol, "audit_reference", return_value=changed)
                else:
                    path = self.fx.output / "execution_environment.json" if mode == "environment" else self.fx.output / "tasks" / self.task()["task_id"] / "environment.json"
                    original = path.read_bytes()
                    value = report.runtime.read_json(path)
                    value["actual_compute_device"]["name"] = "different GPU"
                    report.base.write_json(path, value)
                    patcher = mock.patch.object(protocol, "audit_reference", return_value=self.fx.reference)
                with patcher:
                    result = self.summarize()
                if mode in ("environment", "task_environment"):
                    path.write_bytes(original)
                self.assertEqual(result["status"], "invalid_comparison_evidence")
                self.assertFalse(result["evidence_ready"])
                self.assertEqual(result["comparisons"], {})
                self.assertEqual(result["reference_p0_rows"], [])
                self.assertTrue(all(not c["eligible_for_review"] for c in result["candidates"]))
                self.assertEqual(result["complete_tasks"], 12)

    def test_task_identity_and_partial_snapshot_do_not_become_complete(self):
        task = self.task()
        folder = self.finish(task)
        wrong = deepcopy(task)
        wrong["candidate"]["variant"] = "original"
        report.base.write_json(folder / "task.json", wrong)
        self.assertEqual(self.row(self.summarize(), task)["status"], "invalid_evidence")
        report.base.write_json(folder / "task.json", task)
        self.rewrite(task, lambda r: replace(r, stopped_round=30, records=r.records[:31], diagnostics=[d for d in r.diagnostics if d.round <= 30]))
        row = self.row(self.summarize(), task)
        self.assertEqual(row["status"], "terminal_incomplete")
        self.assertIsNone(row["accuracy150"])
        self.assertFalse(row["history_audit"]["verified"])

    def test_summary_is_readonly_and_no_train_download_or_missing_cost_imputation(self):
        self.all_complete()
        task = self.task()
        (self.fx.output / "tasks" / task["task_id"] / "attempts/test.json").unlink()
        def snapshot():
            return {**self.fx.original_snapshot(), **{str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.fx.output.rglob("*") if p.is_file()}}
        before = snapshot()
        with mock.patch.object(report.base, "load_split", side_effect=AssertionError("download")), \
                mock.patch.object(report.base.experiments, "run_measured_experiment", side_effect=AssertionError("train")):
            result = self.summarize()
        self.assertEqual(before, snapshot())
        row = self.row(result, task)
        self.assertIsNone(row["worker_wall_seconds"])
        self.assertFalse(row["worker_cost_exact"])
        self.assertTrue(result["evidence_ready"])

    def test_all_tasks_references_counts_and_raw_tolerance_fit_under_50KB(self):
        self.all_complete()
        task = self.task()
        self.rewrite(task, lambda r: replace(r, records=[replace(record, malicious_weight_mass=1.0000000000000002) for record in r.records]))
        result = self.summarize()
        text, rows = self.parsed(result)
        self.assertLessEqual(len(text.encode()), 50000)
        self.assertEqual(sum(r["type"] == "task" for r in rows), 12)
        self.assertEqual(sum(r["type"] == "reference_P0" for r in rows), 6)
        self.assertEqual(sum(r["type"] == "reference_C0" for r in rows), 2)
        self.assertEqual(sum(r["type"] == "record_pair_audit" for r in rows), 6)
        self.assertEqual(rows[0]["reference_threshold_manifest_fingerprint"], self.fx.threshold_manifest["fingerprint"])
        actual = next(r for r in rows if r.get("id") == task["task_id"])
        self.assertEqual(actual["w1"]["weights"][2], 1.0000000000000002)
        self.assertEqual(actual["wh"]["M"][2], 70 * 126)
        self.assertEqual(actual["hf"][2:7], [12600, 12600, 0, 0, 0.])
        self.assertIn("legacy client history_frozen", text)
        self.assertNotIn("A->B changes onset", text)

    def test_invalid_manifest_produces_copyable_error(self):
        with mock.patch.object(protocol, "read_study", side_effect=ValueError("corrupt identity")):
            result = self.summarize()
        text, rows = self.parsed(result)
        self.assertEqual(rows[0]["status"], "unavailable_or_invalid_study")
        self.assertIn("corrupt identity", text)
