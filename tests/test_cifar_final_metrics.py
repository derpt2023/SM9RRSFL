"""Final endpoint policy regressions; no data download or GPU training."""
from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import cifar_final_metric_gate as gate
import run_cifar_six_final_metrics as runner
from run_cifar_six_with_progress import runner_for_spec
from reanalyze_cifar_final_metrics import read_run, reanalyze
from tests.test_cifar_six_pipeline import result_matrix, synthetic_run
from tests.test_cifar_six_630_integration import task_statuses
from tests.test_experiment_reporting import make_study
import experiment_reporting as reporting


class FinalMetricTests(unittest.TestCase):
    def setUp(self):
        self.spec = runner.load_spec(runner.DEFAULT_CONFIG)
        # Preserve each six-method arm, with two alternatives for ranking tests.
        self.spec["candidates"] = {m: cs[:2] for m, cs in self.spec["candidates"].items()}
        self.tasks = runner.build_tasks(self.spec, "validation")
        self.results = result_matrix(self.spec, self.tasks)
        self.ours = [c["candidate_id"] for c in self.spec["candidates"]["sm9rrs"]]

    def report(self):
        return gate.select_validation(self.spec, self.results, self.tasks)

    def trajectory(self, cid, early_acc=.8, early_asr=.05, final_acc=.8, final_asr=.05, nonfinite=0):
        runs = []
        for old in self.results[cid]:
            run = synthetic_run(old.config, accuracy=early_acc, asr=early_asr, nonfinite=nonfinite)
            records = run.records[:-1] + [replace(run.records[-1], accuracy=final_acc,
                error=1 - final_acc, attack_target_success_rate=final_asr)]
            runs.append(replace(run, records=records, final_accuracy=final_acc, final_error=1-final_acc))
        self.results[cid] = runs

    def test_process_peaks_tail_and_early_accuracy_cannot_veto_final_pass(self):
        for cid in self.ours:
            self.trajectory(cid, early_acc=.2, early_asr=1.)
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertTrue(report["ours_final_target_passed"])
        row = next(r for r in report["ours_target"]["scenarios"] if "process_diagnostics" in r)
        self.assertEqual(row["process_diagnostics"]["peak_asr"], 1.)
        self.assertGreater(row["process_diagnostics"]["tail_mean_asr"], .2)
        trial = next(t for t in report["trials"] if t["candidate_id"] == report["selected"]["sm9rrs"])
        self.assertAlmostEqual(trial["attack_success_rate"], .05)
        self.assertAlmostEqual(trial["robust_accuracy"], .8)
        self.assertAlmostEqual(trial["worst_attack_success_rate"], .05)

    def test_one_bad_final_scenario_is_not_hidden_in_seed_means(self):
        for cid in self.ours:
            index = next(i for i, r in enumerate(self.results[cid]) if r.config.malicious_ratio)
            old = self.results[cid][index]
            self.results[cid][index] = replace(old, records=old.records[:-1] +
                [replace(old.records[-1], attack_target_success_rate=.051)])
        report = self.report()
        self.assertTrue(report["ours_health_passed"])
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertNotIn("sm9rrs", report["selected"])
        for target in report["ours_candidate_targets"].values():
            self.assertEqual([r["reason"] for r in target["absolute_failures"]], ["final:absolute_asr"])

    def test_final_accuracy_gap_still_blocks(self):
        for cid in self.ours:
            self.trajectory(cid, final_acc=.779)
        report = self.report()
        self.assertEqual(report["status"], "needs_ours_target_development")
        self.assertTrue(report["ours_candidate_targets"][self.ours[0]]["absolute_target_passed"])
        self.assertEqual(report["ours_candidate_targets"][self.ours[0]]["relative_target_status"], "unmet")

    def test_score_ranking_follows_endpoint_including_failed_baselines(self):
        a, b = [c["candidate_id"] for c in self.spec["candidates"]["vert"]]
        self.trajectory(a, early_acc=.99, early_asr=0., final_acc=.79, final_asr=.2, nonfinite=1)
        self.trajectory(b, early_acc=.2, early_asr=1., final_acc=.8, final_asr=.05, nonfinite=1)
        report = self.report()
        self.assertEqual(report["selected"]["vert"], b)
        self.assertEqual(report["methods"]["vert"]["selection_status"], "best_scored_unqualified")
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertFalse(report["ours_target"]["reference_health_qualified"])

    def test_healthy_baseline_beats_higher_failed_score(self):
        a, b = [c["candidate_id"] for c in self.spec["candidates"]["vert"]]
        self.trajectory(a, final_acc=.79, final_asr=.05)
        self.trajectory(b, final_acc=.99, final_asr=0., nonfinite=1)
        self.assertEqual(self.report()["selected"]["vert"], a)

    def test_early_nonfinite_ours_still_fails_health(self):
        for cid in self.ours:
            self.trajectory(cid, final_asr=0., nonfinite=1)
        report = self.report()
        self.assertFalse(report["ours_health_passed"])
        self.assertEqual(report["status"], "needs_ours_development")

    def test_missing_tad_retains_fixed_unqualified_fallback(self):
        cid = self.spec["candidates"]["ding13"][0]["candidate_id"]
        self.results[cid].pop()
        report = self.report()
        self.assertEqual(report["status"], "qualified_for_final")
        self.assertEqual(report["methods"]["ding13"]["selection_status"], "fixed_fallback_unqualified")
        self.assertIsNone(report["methods"]["ding13"]["raw_score"])

    def test_formal_comparison_uses_final_only_and_does_not_hide_missing(self):
        for cid in self.ours:
            self.trajectory(cid, early_asr=1.)
        selected = self.report()["selected"]
        report = gate.describe_final_targets(self.spec, selected, self.results, self.tasks)
        self.assertEqual(report["status"], "passed")
        self.results[selected["vert"]].pop()
        report = gate.describe_final_targets(self.spec, selected, self.results, self.tasks)
        self.assertEqual(report["status"], "incomplete")

    def test_new_protocol_routes_separately_and_keeps_candidate_budget(self):
        spec = runner.load_spec(runner.DEFAULT_CONFIG)
        self.assertEqual(len(runner.build_tasks(spec, "validation")), 750)
        self.assertEqual(Path(runner_for_spec(runner.REPO, spec)).name, "run_cifar_six_final_metrics.py")
        self.assertEqual(Path(runner_for_spec(runner.REPO, {"schema_version": 4})).name, "run_cifar_six_mnist_gate.py")
        old = json.loads((runner.REPO / "configs/cifar10_six_mnist_gate_compact_plus_v4.json").read_text())
        for key in ("candidates", "shared_parameters", "objective", "gates", "validation", "final", "fallback_candidates"):
            self.assertEqual(spec[key], old[key])
        self.assertNotEqual(spec["output_dir"], old["output_dir"])

    def test_reanalysis_refuses_writes_inside_source_or_existing_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "study"
            source.mkdir()
            for output in (source, source / "audit", source.parent):
                with self.assertRaisesRegex(ValueError, "separate"):
                    reanalyze(source, output)
            output = Path(directory) / "audit"
            output.mkdir()
            (output / "keep").write_text("unchanged")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                reanalyze(source, output)

    def test_html_reports_final_asr_and_diagnostic_only_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_study(root)
            data = reporting.aggregate_study(reporting.load_completed_study(root))
            data["manifest"]["spec"].update(selection_metrics=gate.SELECTION_METRICS,
                performance_target=self.spec["performance_target"])
            data["final_summary"]["validation_final_metric_gate"] = {"status": "passed"}
            html = reporting._html(data, [], mean_page=True)
            self.assertIn("最终 ASR 均值 ± seed SD (%)", html)
            self.assertIn("30.00 ± 20.00", html)
            self.assertIn("过程均值、末尾窗口和峰值仅作诊断", html)
            self.assertNotIn("攻击窗口 ASR 均值 ± seed SD (%)", html)

    def test_runner_starts_formal_only_after_endpoint_pass_and_generates_final_summary(self):
        self.spec["run_budget"] = runner.run_budget(self.spec)
        for passed in (True, False):
            with self.subTest(passed=passed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = root / "spec.json"
                runner.write_json(config, self.spec)
                for cid in self.ours:
                    self.trajectory(cid, early_asr=.9, final_asr=.05 if passed else .051)
                # A scorable failed reference does not independently veto entry.
                for candidate in self.spec["candidates"]["vert"]:
                    self.trajectory(candidate["candidate_id"], nonfinite=1)
                phases = []

                def execute(args, tasks, output):
                    phases.append(tasks[0]["phase"])
                    return 0

                def collect(output, tasks):
                    groups = self.results if tasks[0]["phase"] == "validation" else result_matrix(self.spec, tasks)
                    return groups, task_statuses(tasks, groups)

                args = SimpleNamespace(config=config, output=root / "study", data_dir=None,
                                       devices=["cuda:0"], phase="all")
                with mock.patch.object(runner, "load_split", return_value=(object(), {})), \
                     mock.patch.object(runner, "execute_phase", side_effect=execute), \
                     mock.patch.object(runner, "collect_results", side_effect=collect), \
                     mock.patch.object(runner.experiments, "write_result_files"), redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.run_parent(args), 0)
                self.assertEqual(phases, ["validation", "final"] if passed else ["validation"])
                self.assertEqual((args.output / "final_plan.json").exists(), passed)
                if passed:
                    final = json.loads((args.output / "final_summary.json").read_text())
                    self.assertIn("validation_final_metric_gate", final)
                    self.assertEqual(final["ours_target"]["selection_metrics"], gate.SELECTION_METRICS)
                    self.assertEqual(final["process_target_diagnostics"]["role"], "diagnostic_only")
                    self.assertEqual(final["methods"]["vert"]["validation_selection_status"], "best_scored_unqualified")
                    self.assertFalse(final["formal_results_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
