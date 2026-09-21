"""Relative-ASR policy, strict sample-count boundary and lifecycle tests."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import cifar_relative_asr_gate as gate
import run_cifar_six_relative_asr as runner
from run_cifar_six_with_progress import runner_for_spec
from tests.test_cifar_six_pipeline import result_matrix, synthetic_run
from tests.test_cifar_six_630_integration import task_statuses
from tests.test_experiment_reporting import make_study
import experiment_reporting as reporting


class RelativeASRTests(unittest.TestCase):
    def setUp(self):
        self.spec = runner.load_spec(runner.DEFAULT_CONFIG)
        self.spec["candidates"] = {m: cs[:2] for m, cs in self.spec["candidates"].items()}
        self.spec["run_budget"] = runner.run_budget(self.spec)
        self.tasks = runner.build_tasks(self.spec, "validation")
        self.results = result_matrix(self.spec, self.tasks)
        self.ours = [c["candidate_id"] for c in self.spec["candidates"]["sm9rrs"]]

    def change(self, cid, asr=.2, accuracy=.8, nonfinite=0):
        self.results[cid] = [synthetic_run(r.config, asr=asr, accuracy=accuracy, nonfinite=nonfinite)
                             for r in self.results[cid]]

    def all_rates(self, value=.2):
        for cid in self.results:
            self.change(cid, asr=value)

    def report(self):
        return gate.select_validation(self.spec, self.results, self.tasks)

    def test_high_absolute_asr_can_pass_if_all_selected_methods_match(self):
        self.all_rates()
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertTrue(report["ours_asr_minimum_target_passed"])
        self.assertFalse(report["promotion_basis"]["absolute_asr_target_required"])
        self.assertNotIn("ours_absolute_target_passed", report)

    def test_strict_boundary_including_float32_serialization(self):
        for value, expected in ((.1, True), (.105, True), (.11, False), (.115, False),
                                (float(np.float32(.11)), False)):
            with self.subTest(value=value):
                self.all_rates(float(np.float32(.1)))
                for cid in self.ours:
                    self.change(cid, asr=value)
                report = self.report()
                self.assertEqual(report["status"] == "qualified_for_final", expected)
                target = report["ours_candidate_targets"][self.ours[0]]
                self.assertEqual(target["asr_passed_task_count"], 24 if expected else 0)

    def test_one_seed_failure_cannot_be_hidden_by_scenario_mean(self):
        for cid in self.ours:
            i = next(i for i, r in enumerate(self.results[cid]) if r.config.malicious_ratio)
            old = self.results[cid][i]
            self.results[cid][i] = replace(old, records=old.records[:-1] +
                [replace(old.records[-1], attack_target_success_rate=.06)])
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertEqual(report["ours_candidate_targets"][self.ours[0]]["asr_passed_task_count"], 23)

    def test_tad_complete_unhealthy_task_is_still_a_reference(self):
        self.all_rates()
        cid = self.spec["candidates"]["ding13"][0]["candidate_id"]
        self.change(cid, asr=.05, nonfinite=1)
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertEqual(report["methods"]["ding13"]["selection_status"], "best_scored_unqualified")
        row = next(r for r in report["ours_candidate_targets"][self.ours[0]]["scenarios"] if "asr_minimum" in r)
        self.assertEqual(row["asr_minimum"]["best_methods"], ["ding13"])
        self.assertFalse(row["asr_minimum"]["references"]["ding13"]["task_healthy"])

    def test_incomplete_tad_retains_other_29_references(self):
        self.all_rates()
        cid = self.spec["candidates"]["ding13"][0]["candidate_id"]
        self.results[cid].pop()
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertEqual(report["methods"]["ding13"]["selection_status"], "fixed_fallback_unqualified")
        self.assertEqual(report["final_metric_gate"]["per_task_reference_counts"]["ding13"], 29)
        self.assertFalse(report["ours_target"]["full_target_passed"])
        absent = [r for r in report["ours_target"]["scenarios"] if r.get("asr_minimum", {}).get("missing_methods")]
        self.assertEqual(len(absent), 1)
        self.assertEqual(absent[0]["asr_minimum"]["missing_methods"], ["ding13"])
        self.change(cid, asr=.05)
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertEqual(report["ours_candidate_targets"][self.ours[0]]["asr_passed_task_count"], 1)

    def test_only_fixed_selected_candidate_is_used(self):
        self.all_rates()
        a, b = [c["candidate_id"] for c in self.spec["candidates"]["vert"]]
        self.change(b, accuracy=.2, asr=0.)
        report = self.report()
        self.assertEqual(report["selected"]["vert"], a)
        self.assertEqual(report["status"], "qualified_for_final")

    def test_health_and_paired_accuracy_remain_required(self):
        self.all_rates()
        for cid in self.ours:
            self.change(cid, accuracy=.779)
        report = self.report()
        self.assertTrue(report["ours_health_passed"])
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertTrue(report["ours_candidate_targets"][self.ours[0]]["asr_minimum_target_passed"])
        for cid in self.ours:
            self.change(cid, nonfinite=1)
        self.assertFalse(self.report()["ours_health_passed"])

    def test_accuracy_keeps_vert_two_point_limit_not_best_method_one_point(self):
        self.all_rates()
        fedavg = self.spec["candidates"]["fedavg"][0]["candidate_id"]
        self.results[fedavg] = [synthetic_run(run.config, accuracy=.82, asr=.2)
                               if run.config.malicious_ratio else run for run in self.results[fedavg]]
        for accuracy, expected in ((.785, True), (.78, True), (.779, False)):
            with self.subTest(accuracy=accuracy):
                for cid in self.ours:
                    self.change(cid, accuracy=accuracy)
                report = self.report()
                self.assertEqual(report["status"] == "qualified_for_final", expected)
                self.assertTrue(report["ours_candidate_targets"][self.ours[0]]["asr_minimum_target_passed"])

    def test_only_last_round_controls_performance_despite_bad_earlier_metrics(self):
        self.all_rates()
        for cid in self.ours:
            self.results[cid] = [replace(run, records=[
                replace(r, accuracy=.1, error=.9, attack_target_success_rate=.99)
                if run.config.attack_start_round <= r.round < 100 else r
                for r in run.records]) if run.config.malicious_ratio else run
                for run in self.results[cid]]
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        target = report["ours_target"]
        self.assertTrue(target["asr_minimum_target_passed"])
        row = next(r for r in target["scenarios"] if "asr_minimum" in r)
        self.assertEqual(row["windows"]["final"]["ours_asr"], .2)
        self.assertEqual(row["process_diagnostics"]["peak_asr"], .99)

    def test_all_missing_baselines_do_not_become_full_six_way_pass(self):
        self.results = {cid: runs for cid, runs in self.results.items() if cid in self.ours}
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertEqual(report["ours_target"]["status"], "partially_assessed")
        self.assertFalse(report["ours_target"]["full_target_passed"])
        self.assertEqual(report["ours_target"]["relative_target_status"], "unassessed")

    def test_identity_mismatch_and_duplicate_are_not_missing_baselines(self):
        cid = self.spec["candidates"]["ding13"][0]["candidate_id"]
        self.results[cid].append(self.results[cid][0])
        with self.assertRaisesRegex(ValueError, "duplicate or mismatched"):
            self.report()

    def test_config_and_plan_only_preserve_protocol_and_budget(self):
        current = runner.load_spec(runner.DEFAULT_CONFIG)
        old = json.loads((runner.REPO / "configs/cifar10_six_final_metrics_v5.json").read_text())
        for key in ("candidates", "shared_parameters", "dataset", "objective", "gates", "validation", "final", "fallback_candidates"):
            self.assertEqual(current[key], old[key])
        self.assertNotIn("max_asr", current["performance_target"])
        self.assertEqual(Path(runner_for_spec(runner.REPO, current)).name, "run_cifar_six_relative_asr.py")
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output:
            destination = Path(directory) / "not_created"
            self.assertEqual(runner.main(["--plan-only", "--output", str(destination)]), 0)
            self.assertFalse(destination.exists())
        plan = json.loads(output.getvalue())
        self.assertEqual(plan["schema_version"], 6)
        self.assertEqual(plan["validation_runs"], 750)
        self.assertEqual(plan["final_runs"], 180)

    def test_runner_promotes_only_on_new_relative_target_and_reports_html_policy(self):
        for passed in (True, False):
            with self.subTest(passed=passed), tempfile.TemporaryDirectory() as directory:
                self.all_rates()
                if not passed:
                    for cid in self.ours:
                        self.change(cid, asr=.21)
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
            data["manifest"]["spec"].update(asr_target=gate.ASR_TARGET, selection_metrics=gate.SELECTION_METRICS,
                performance_target=self.spec["performance_target"])
            data["final_summary"]["validation_final_metric_gate"] = {"status": "passed"}
            html = reporting._html(data, [], mean_page=True)
            self.assertIn("须严格小于 1 个百分点", html)
            self.assertIn("恰好相等不通过", html)
            self.assertIn("不设5%的绝对上限", html)


if __name__ == "__main__":
    unittest.main()
