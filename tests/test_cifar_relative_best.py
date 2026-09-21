"""Two inclusive final metric gaps, independent best peers and real lifecycle."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import cifar_relative_best_gate as gate
import run_cifar_six_relative_best as runner
from run_cifar_six_with_progress import runner_for_spec
from tests.test_cifar_six_pipeline import result_matrix, synthetic_run
from tests.test_cifar_six_630_integration import task_statuses
from tests.test_experiment_reporting import make_study
import experiment_reporting as reporting


class RelativeBestTests(unittest.TestCase):
    def setUp(self):
        self.spec = runner.load_spec(runner.DEFAULT_CONFIG)
        self.spec["candidates"] = {m: cs[:2] for m, cs in self.spec["candidates"].items()}
        self.spec["run_budget"] = runner.run_budget(self.spec)
        self.tasks = runner.build_tasks(self.spec, "validation")
        self.results = result_matrix(self.spec, self.tasks)
        self.ours = [c["candidate_id"] for c in self.spec["candidates"]["sm9rrs"]]
        for cid in self.results:
            self.change(cid)

    def change(self, cid, *, accuracy=.8, asr=.1, nonfinite=0):
        self.results[cid] = [synthetic_run(r.config, accuracy=accuracy, asr=asr, nonfinite=nonfinite)
                             for r in self.results[cid]]

    def report(self):
        return gate.select_validation(self.spec, self.results, self.tasks)

    def test_two_point_asr_equality_passes_and_larger_fails_even_float32(self):
        for value, passed in ((.115, True), (.12, True), (float(np.float32(.12)), True), (.125, False)):
            with self.subTest(value=value):
                for cid in self.ours:
                    self.change(cid, asr=value)
                report = self.report()
                self.assertEqual(report["status"] == "qualified_for_final", passed)
                self.assertEqual(report["ours_candidate_targets"][self.ours[0]]["asr_passed_task_count"], 24 if passed else 0)

    def test_two_point_accuracy_equality_and_one_more_wrong_sample(self):
        for value, passed in ((.785, True), (.78, True), (float(np.float32(.78)), True), (.7796, False)):
            with self.subTest(value=value):
                for cid in self.ours:
                    self.change(cid, accuracy=value)
                report = self.report()
                self.assertEqual(report["status"] == "qualified_for_final", passed)
                self.assertEqual(report["ours_candidate_targets"][self.ours[0]]["accuracy_passed_task_count"], 30 if passed else 0)

    def test_accuracy_compares_six_method_maximum_not_only_vert(self):
        cid = self.spec["candidates"]["ding13"][0]["candidate_id"]
        self.change(cid, accuracy=.83)
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_target_development")
        target = report["ours_candidate_targets"][self.ours[0]]
        self.assertTrue(target["asr_minimum_target_passed"])
        self.assertEqual(target["scenarios"][0]["accuracy_maximum"]["best_methods"], ["ding13"])

    def test_best_accuracy_and_asr_can_be_different_methods(self):
        self.change(self.spec["candidates"]["ding13"][0]["candidate_id"], accuracy=.82)
        self.change(self.spec["candidates"]["krum"][0]["candidate_id"], asr=.08)
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        row = next(r for r in report["ours_target"]["scenarios"] if "asr_minimum" in r)
        self.assertEqual(row["accuracy_maximum"]["best_methods"], ["ding13"])
        self.assertEqual(row["asr_minimum"]["best_methods"], ["krum"])
        self.assertEqual(row["accuracy_maximum"]["gap"], .02)
        self.assertEqual(row["asr_minimum"]["gap"], .02)

    def test_single_task_failure_is_not_hidden_in_seed_means(self):
        for cid in self.ours:
            index = next(i for i, r in enumerate(self.results[cid]) if r.config.malicious_ratio)
            old = self.results[cid][index]
            self.results[cid][index] = replace(old, records=old.records[:-1] +
                [replace(old.records[-1], attack_target_success_rate=.125)])
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertEqual(report["ours_candidate_targets"][self.ours[0]]["asr_passed_task_count"], 23)

    def test_clean_accuracy_is_checked_but_clean_asr_remains_diagnostic(self):
        for cid in self.ours:
            self.results[cid] = [replace(run, records=run.records[:-1] +
                [replace(run.records[-1], attack_target_success_rate=.99)])
                if run.config.malicious_ratio == 0 else run for run in self.results[cid]]
        self.assertEqual(self.report()["status"], "qualified_for_final")
        for cid in self.ours:
            self.results[cid] = [synthetic_run(run.config, accuracy=.7796, asr=.99)
                if run.config.malicious_ratio == 0 else run for run in self.results[cid]]
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertEqual(report["ours_candidate_targets"][self.ours[0]]["accuracy_passed_task_count"], 24)

    def test_process_metrics_and_old_v6_asr_verdict_cannot_veto_final_pass(self):
        for cid in self.ours:
            self.change(cid, asr=.12)
            self.results[cid] = [replace(run, records=[replace(r, accuracy=.1, error=.9, attack_target_success_rate=.99)
                if run.config.attack_start_round <= r.round < 100 else r for r in run.records])
                if run.config.malicious_ratio else run for run in self.results[cid]]
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertFalse(report["promotion_basis"]["paired_vert_target_required_when_scorable"])
        row = next(r for r in report["ours_target"]["scenarios"] if "asr_minimum" in r)
        self.assertEqual(row["process_diagnostics"]["peak_asr"], .99)

    def test_health_failure_and_fixed_baseline_selection_are_retained(self):
        first, second = [c["candidate_id"] for c in self.spec["candidates"]["vert"]]
        self.change(second, accuracy=.1, asr=0.)
        report = self.report()
        self.assertEqual(report["selected"]["vert"], first)
        self.assertEqual(report["status"], "qualified_for_final")
        for cid in self.ours:
            self.change(cid, nonfinite=1)
        self.assertEqual(self.report()["status"], "needs_ours_development")

    def test_unhealthy_complete_reference_still_sets_best(self):
        cid = self.spec["candidates"]["ding13"][0]["candidate_id"]
        self.change(cid, accuracy=.83, nonfinite=1)
        report = self.report()
        self.assertEqual(report["methods"]["ding13"]["selection_status"], "best_scored_unqualified")
        self.assertEqual(report["status"], "needs_ours_target_development")
        row = report["ours_candidate_targets"][self.ours[0]]["scenarios"][0]
        self.assertFalse(row["accuracy_maximum"]["references"]["ding13"]["task_healthy"])

    def test_missing_reference_is_explicit_and_other_tasks_still_compare(self):
        cid = self.spec["candidates"]["ding13"][0]["candidate_id"]
        self.results[cid].pop()
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertFalse(report["ours_target"]["full_target_passed"])
        self.assertEqual(report["methods"]["ding13"]["selection_status"], "fixed_fallback_unqualified")
        absent = [r for r in report["ours_target"]["scenarios"] if r["accuracy_maximum"]["missing_methods"]]
        self.assertEqual(len(absent), 1)
        self.change(cid, accuracy=.83)
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertEqual(report["ours_candidate_targets"][self.ours[0]]["accuracy_passed_task_count"], 1)

    def test_no_references_never_claims_complete_comparison(self):
        self.results = {cid: runs for cid, runs in self.results.items() if cid in self.ours}
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertFalse(report["ours_target"]["full_target_passed"])
        self.assertEqual(report["ours_target"]["relative_target_status"], "unassessed")

    def test_formal_diagnostics_have_same_policy_and_larger_accuracy_sample_count(self):
        selected = {m: cs[0]["candidate_id"] for m, cs in self.spec["candidates"].items()}
        tasks = runner.build_tasks(self.spec, "final", selected)
        results = result_matrix(self.spec, tasks)
        target = gate.describe_final_targets(self.spec, selected, results, tasks)
        self.assertEqual(target["scenarios"][0]["accuracy_maximum"]["sample_count"], 10000)
        self.assertEqual(target["role"], "descriptive_only_not_execution_or_health_status")
        results[selected["sm9rrs"]].pop()
        self.assertEqual(gate.describe_final_targets(self.spec, selected, results, tasks)["status"], "incomplete")

    def test_lifecycle_promotes_new_rule_only_and_html_explains_two_gaps(self):
        for passed in (True, False):
            with self.subTest(passed=passed), tempfile.TemporaryDirectory() as directory:
                for cid in self.ours:
                    self.change(cid, asr=.12 if passed else .125)
                root = Path(directory)
                config = root / "spec.json"
                runner.write_json(config, self.spec)
                phases = []
                def execute(args, tasks, output):
                    phases.append(tasks[0]["phase"])
                    return 0
                def collect(output, tasks):
                    groups = self.results if tasks[0]["phase"] == "validation" else result_matrix(self.spec, tasks)
                    return groups, task_statuses(tasks, groups)
                args = SimpleNamespace(config=config, output=root / "study", data_dir=None, devices=["cuda:0"], phase="all")
                with mock.patch.object(runner, "load_split", return_value=(object(), {})), \
                        mock.patch.object(runner, "execute_phase", side_effect=execute), \
                        mock.patch.object(runner, "collect_results", side_effect=collect), \
                        mock.patch.object(runner.experiments, "write_result_files"), redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.run_parent(args), 0)
                self.assertEqual(phases, ["validation", "final"] if passed else ["validation"])
                self.assertEqual((args.output / "final_plan.json").exists(), passed)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_study(root)
            data = reporting.aggregate_study(reporting.load_completed_study(root))
            data["manifest"]["spec"].update(asr_target=gate.ASR_TARGET, accuracy_target=gate.ACCURACY_TARGET,
                selection_metrics=gate.SELECTION_METRICS, performance_target=gate.PERFORMANCE_TARGET)
            data["final_summary"]["validation_final_metric_gate"] = {"status": "passed"}
            html = reporting._html(data, [], mean_page=True)
            self.assertIn("恰好相等也通过", html)
            self.assertIn("两项最优值可来自不同方法", html)
            self.assertNotIn("相对 VERT 目标", html)
            self.assertNotIn("须严格小于", html)
        self.assertEqual(Path(runner_for_spec(runner.REPO, self.spec)).name, "run_cifar_six_relative_best.py")


if __name__ == "__main__":
    unittest.main()
