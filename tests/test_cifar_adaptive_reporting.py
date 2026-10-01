from copy import deepcopy
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cifar_adaptive_reporting as reporting
import run_cifar_six_from_scratch as base
from tests.test_cifar_six_pipeline import synthetic_run


class AdaptiveReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.spec = json.loads(Path("configs/cifar10_six_relative_best_five_day_v7.json").read_text())
        self.spec["schema_version"] = 8
        self.spec["shared_parameters"].update(rounds=150, detector_window=10, attack_start_round=12)
        self.spec["final"] = {"seeds": [4101], "scenarios": [
            {"partition": partition, "malicious_ratio": ratio}
            for partition in ("iid", "dirichlet") for ratio in (0., .4)]}
        self.spec["candidates"] = {method: [dict(rows[0], candidate_id=f"{method}-b000-d000")]
                                     for method, rows in self.spec["candidates"].items()}
        self.selected = {method: rows[0]["candidate_id"] for method, rows in self.spec["candidates"].items()}
        self.tasks = base.build_tasks(self.spec, "final", self.selected)
        self.tasks = [dict(task, fingerprint=f"task-{index}") for index, task in enumerate(self.tasks)]
        self.results, self.statuses = {}, []
        for task in self.tasks:
            run = synthetic_run(base.fl.ExperimentConfig(**task["config"]), accuracy=.8, asr=.1,
                                nonfinite=1 if task["method"] == "vert" else 0)
            cid = task["candidate"]["candidate_id"]
            self.results.setdefault(cid, []).append(run)
            self.statuses.append({"task_id": task["task_id"], "candidate_id": cid, "method": task["method"],
                                  "status": "complete", "error": None, "metrics": base.metrics(run)})
            folder = self.output / "tasks" / task["task_id"]
            folder.mkdir(parents=True)
            (folder / "task.json").write_text(json.dumps(task))
        self.validation = {"status": "needs_ours_target_development", "selected": self.selected,
                           "methods": {"sm9rrs": {"selection_status": "manual_best_healthy"}}}

    def write(self, *, plots=False):
        if plots:
            return reporting.write_final_report(self.output, self.spec, self.selected,
                self.results, self.tasks, self.statuses, self.validation, {"answer": "Y"})
        with patch.object(reporting, "_plots", return_value=[]):
            return reporting.write_final_report(self.output, self.spec, self.selected,
                self.results, self.tasks, self.statuses, self.validation, {"answer": "Y"})

    def read_rows(self, name):
        with (self.output / "final_results" / name).open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_real_plots_single_seed_health_failures_and_repeat(self):
        budget = {"public_tpe_proposals": 0, "actual_search_budget": [{"fully_attempted_waves": 1}],
                  "cross_public_condition_search_budgets_may_differ": True}
        (self.output / "search_summary.json").write_text(json.dumps(budget))
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.output.rglob("task.json")}
        summary = self.write(plots=True)
        self.assertEqual(summary["search_summary"], budget)
        self.assertTrue(summary["full_execution_completed"])
        self.assertEqual(summary["status"], "completed_with_health_failures")
        self.assertEqual(summary["methods"]["vert"]["overall"]["final_accuracy"], .8)
        self.assertEqual(summary["methods"]["vert"]["healthy_runs"], 0)
        self.assertEqual(summary["required_final_round"], 150)
        self.assertEqual(summary["ours_final_metric_gate"]["role"],
                         "descriptive_only_no_parameter_reselection_or_health_override")
        destination = self.output / "final_results"
        self.assertEqual(len(list(destination.glob("*.png"))), 4)
        self.assertEqual(len(list(destination.glob("*.svg"))), 4)
        self.assertIn("SD不可用", (destination / "visualizations.html").read_text())
        self.assertIn("以Ours相对优势为目标搜索", (destination / "visualizations.html").read_text())
        for row in self.read_rows("aggregate.csv"):
            self.assertEqual(row["n"], "1")
            self.assertEqual(row["final_accuracy_sd"], "")
        self.assertEqual(len(self.read_rows("rounds.csv")), len(self.tasks) * 151)
        first = (destination / "aggregate.csv").read_bytes()
        repeat = self.write()
        self.assertEqual(repeat, summary)
        self.assertEqual((destination / "aggregate.csv").read_bytes(), first)
        self.assertTrue(all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items()))
        audit = json.loads((destination / "data_audit.json").read_text())
        self.assertTrue(audit["complete_matrix_verified"])
        self.assertIn("cifar_adaptive_reporting.py", audit["report_source_sha256"])
        self.assertIn("search_summary.json", audit["source_sha256"])

    def test_missing_task_stays_blank_disables_method_overall(self):
        cid = self.selected["ding13"]
        missing = self.results[cid].pop()
        status = next(row for row in self.statuses if row["task_id"] == next(
            t["task_id"] for t in self.tasks if t["method"] == "ding13" and
            t["config"]["partition"] == missing.config.partition and t["config"]["malicious_ratio"] == missing.config.malicious_ratio))
        status.update(status="failed", metrics=None, error="numeric failure at round 30")
        (self.output / "tasks" / status["task_id"] / "failure.json").write_text(json.dumps({
            "task_id": status["task_id"], "exception": "FloatingPointError", "message": "confidence at round 30",
            "execution_context": {"last_completed_round": 29}}))
        summary = self.write()
        self.assertEqual(summary["completed_tasks"], len(self.tasks) - 1)
        self.assertEqual(summary["report_status"], "completed_partial")
        self.assertIsNone(summary["methods"]["ding13"]["overall"])
        detail = next(row for row in summary["tasks"] if row["task_id"] == status["task_id"])
        self.assertEqual(detail["last_completed_round"], 29)
        self.assertIn("FloatingPointError", detail["health_reasons"][0])
        row = next(row for row in self.read_rows("aggregate.csv") if row["method"] == "ding13"
                   and row["partition"] == missing.config.partition and float(row["malicious_ratio"]) == missing.config.malicious_ratio)
        self.assertEqual(row["n"], "0")
        self.assertEqual(row["expected_n"], "1")
        self.assertEqual(row["final_accuracy_mean"], "")
        self.assertEqual(row["final_asr_mean"], "")
        self.assertFalse(summary["ours_final_metric_gate"]["full_target_passed"])

    def test_partial_result_never_uses_last_old_round(self):
        cid = self.selected["sm9rrs"]
        run = self.results[cid][-1]
        self.results[cid][-1] = replace(run, stopped_round=29, records=run.records[:30])
        summary = self.write()
        self.assertEqual(summary["ours_final_metric_gate"]["status"], "incomplete")
        row = next(row for row in summary["tasks"] if row["method"] == "sm9rrs" and not row["final_round_available"])
        self.assertEqual(row["last_completed_round"], 29)
        self.assertIsNone(row["final_accuracy"])
        self.assertIsNone(row["final_asr"])
        self.assertIsNone(summary["methods"]["sm9rrs"]["overall"])

    def test_corrupt_finished_result_or_missing_result_rejected(self):
        cid = self.selected["sm9rrs"]
        original = self.results[cid][0]
        for run in (replace(original, final_accuracy=.9),
                    replace(original, records=original.records[:-1]),
                    replace(original, records=[replace(r, accuracy=float("nan")) if r.round == 50 else r for r in original.records])):
            self.results[cid][0] = run
            with self.assertRaises(reporting.ReportIntegrityError):
                self.write()
        self.results[cid].pop(0)
        with self.assertRaisesRegex(reporting.ReportIntegrityError, "result is missing"):
            self.write()

    def test_task_identity_duplicate_foreign_config_and_matrix_rejected(self):
        original = deepcopy(self.tasks)
        self.tasks.pop()
        with self.assertRaisesRegex(reporting.ReportIntegrityError, "matrix"):
            self.write()
        self.tasks = deepcopy(original)
        self.tasks[0]["config"]["lr"] = .09
        with self.assertRaisesRegex(reporting.ReportIntegrityError, "mismatched"):
            self.write()
        self.tasks = deepcopy(original)
        cid = self.selected["sm9rrs"]
        self.results[cid].append(self.results[cid][0])
        with self.assertRaisesRegex(reporting.ReportIntegrityError, "duplicate"):
            self.write()
        self.results[cid].pop()
        path = self.output / "tasks" / self.tasks[0]["task_id"] / "task.json"
        path.write_text("{}")
        with self.assertRaisesRegex(reporting.ReportIntegrityError, "task.json"):
            self.write()

    def test_pending_is_incomplete_not_attempted(self):
        cid = self.selected["fedavg"]
        self.results[cid].pop()
        status = [s for s in self.statuses if s["method"] == "fedavg"][-1]
        status.update(status="pending", metrics=None)
        summary = self.write()
        self.assertFalse(summary["all_scheduled_tasks_attempted"])
        self.assertEqual(summary["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
