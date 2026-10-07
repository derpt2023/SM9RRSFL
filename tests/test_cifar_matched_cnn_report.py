"""Matched CNN evidence and decision tests without data downloads or CUDA."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import cifar_matched_cnn_report as report
import run_cifar_diagnostic as old_runner
from tests.test_cifar_six_pipeline import synthetic_run

base, runtime = report.base, report.runtime


class MatchedReportTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.output = Path(tmp.name) / "matched"
        self.reference_output = Path(tmp.name) / "clean"
        self.output.mkdir()
        self.reference_output.mkdir()
        spec = runtime.read_json(old_runner.DEFAULT_CONFIG)
        old_manifest = old_runner.build_manifest(spec, {"synthetic": True})
        r3_tasks = [t for t in old_runner.build_tasks(spec, old_manifest)
                    if t["candidate"]["candidate_id"] == "R3"]
        self.tasks = deepcopy(r3_tasks)
        for task in self.tasks:
            task["task_id"] = task["task_id"].replace("R3", "C3")
            task["candidate"]["candidate_id"] = "C3"
            task["model"] = "v7_cnn"
            task["fingerprint"] = base.digest({k: v for k, v in task.items() if k != "fingerprint"})
        rows = []
        for setting, accuracy, seconds in (("C0", .50, 100.), ("R3", .62, 400.)):
            for partition, seed in sorted(report.PAIRS):
                rows.append({"setting": setting, "partition": partition, "seed": seed,
                    "status": "complete", "healthy": True, "accuracy150": accuracy,
                    "worker_wall_seconds": seconds, "unfinished_attempts": 0})
        self.environment = {"actual_compute_device": {"name": "synthetic GPU"}, "torch": {"version": "test"}}
        self.reference = {"manifest_fingerprint": "old-test-fingerprint", "data_contract": {"synthetic": True},
            "execution_environment": self.environment, "evidence_sha256": {"test": "hash"},
            "r3_tasks": r3_tasks, "rows": rows, "selection": {}, "complete_tasks": 24, "healthy_tasks": 24}
        self.manifest = {"fingerprint": "test-fingerprint", "reference": deepcopy(self.reference),
            "spec": {"protocol": "cifar-v8-cnn-e2-match-v1", "reference_manifest_fingerprint": "old-test-fingerprint"},
            "source_sha256": {"synthetic.py": "hash"}}
        self.runner = SimpleNamespace(read_study=mock.Mock(return_value=(self.manifest, self.tasks)),
            source_hashes=mock.Mock(return_value={"synthetic.py": "hash"}),
            audit_reference=mock.Mock(return_value=deepcopy(self.reference)))
        patcher = mock.patch.dict(sys.modules, {"run_cifar_matched_cnn": self.runner})
        patcher.start()
        self.addCleanup(patcher.stop)
        base.write_json(self.output / "execution_environment.json", self.environment)

    def complete(self, task, accuracy=.60, nonfinite=0):
        folder = runtime.ensure_identity(self.output, task)
        result = synthetic_run(base.fl.ExperimentConfig(**task["config"]), accuracy=accuracy, nonfinite=nonfinite)
        base.experiments._write_completed_results_snapshot(folder, [result])
        base.write_json(folder / "observations.json", {"task_fingerprint": task["fingerprint"],
            "rounds": [{"round": r, "local_train_loss": 1. if r else None,
                "local_train_samples": 45000 if r else 0, "calibration_loss": 1.1,
                "calibration_samples": 2500, "cuda_peak_allocated_mib": 3500.} for r in range(151)]})
        (folder / "attempts").mkdir(exist_ok=True)
        base.write_json(folder / "attempts/test.json", {"task_fingerprint": task["fingerprint"],
            "status": "complete", "wall_seconds": 200.})
        return folder

    def all_complete(self, accuracy=.60):
        for task in self.tasks:
            self.complete(task, accuracy)

    def summarize(self):
        return report.summarize(self.output, self.reference_output)

    def snapshot(self):
        return {str(p.relative_to(self.output.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.output.parent.rglob("*") if p.is_file()}

    def test_summary_is_readonly_and_reports_actual_n_without_imputation(self):
        self.complete(self.tasks[0])
        before = self.snapshot()
        with mock.patch.object(base, "load_split", side_effect=AssertionError("no data")), \
                mock.patch.object(base.experiments, "run_measured_experiment", side_effect=AssertionError("no training")):
            result = self.summarize()
        self.assertEqual(before, self.snapshot())
        self.assertEqual(result["complete_tasks"], 1)
        self.assertEqual(result["healthy_tasks"], 1)
        self.assertEqual(result["status"], "incomplete")
        self.assertNotIn("accuracy150", result["rows"][1])
        self.assertFalse(result["decision"]["paired_comparison_available"])

    def test_healthy_pairs_and_inclusive_mean_engineering_line(self):
        self.all_complete()
        result = self.summarize()
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["reference_verified"])
        choice = result["decision"]
        self.assertTrue(choice["architecture_engineering_line_passed"])
        self.assertAlmostEqual(choice["paired_mean_R3_minus_C3_pp"], 2.)
        self.assertAlmostEqual(choice["paired_mean_C3_minus_C0_pp"], 10.)
        self.assertEqual(choice["R3_over_C3_total_worker_time_ratio"], 2.)
        self.assertFalse(choice["next_stage_started"])
        self.assertFalse(choice["formal_qualification_assessed"])
        self.runner.audit_reference.assert_called_once_with(self.reference_output, expected_fingerprint="old-test-fingerprint")

    def test_each_seed_partition_bound_is_inclusive(self):
        for task, accuracy in zip(self.tasks, (.63, .59, .59, .59)):
            self.complete(task, accuracy)
        choice = self.summarize()["decision"]
        self.assertAlmostEqual(choice["paired_min_R3_minus_C3_pp"], -1.)
        self.assertAlmostEqual(choice["paired_mean_R3_minus_C3_pp"], 2.)
        self.assertTrue(choice["architecture_engineering_line_passed"])
        self.complete(self.tasks[0], .631)
        self.assertFalse(self.summarize()["decision"]["architecture_engineering_line_passed"])

    def test_unhealthy_completion_preserves_metrics_and_blocks_recommendation(self):
        self.all_complete()
        self.complete(self.tasks[0], .99, nonfinite=1)
        result = self.summarize()
        self.assertEqual(result["complete_tasks"], 4)
        self.assertEqual(result["healthy_tasks"], 3)
        self.assertEqual(result["rows"][0]["accuracy150"], .99)
        self.assertEqual(result["decision"]["action"], "review_unhealthy_completed_runs_before_model_decision")
        self.assertIsNone(result["decision"]["architecture_engineering_line_passed"])

    def test_wrong_task_identity_is_not_usable_evidence(self):
        folder = self.complete(self.tasks[0])
        bad = deepcopy(self.tasks[0])
        bad["model"] = "resnet18_gn2"
        base.write_json(folder / "task.json", bad)
        result = self.summarize()
        self.assertEqual(result["rows"][0]["status"], "invalid_evidence")
        self.assertEqual(result["complete_tasks"], 0)

    def test_modified_reference_hash_blocks_cached_pairing(self):
        self.all_complete()
        self.runner.audit_reference.return_value["evidence_sha256"]["test"] = "changed"
        result = self.summarize()
        self.assertFalse(result["reference_verified"])
        self.assertEqual(result["reference_rows"], [])
        self.assertEqual(result["decision"]["action"], "resolve_changed_or_invalid_reference")
        self.assertFalse(result["decision"]["paired_comparison_available"])

    def test_reference_audit_failure_is_copyable_and_does_not_fall_back_to_cache(self):
        self.runner.audit_reference.side_effect = ValueError("damaged reference snapshot")
        result = self.summarize()
        stream = io.StringIO()
        with redirect_stdout(stream):
            report.print_summary(result)
        self.assertIn("damaged reference snapshot", stream.getvalue())
        self.assertIn("CIFAR_CNN_MATCH_BEGIN", stream.getvalue())
        self.assertIn("CIFAR_CNN_MATCH_END", stream.getvalue())
        self.assertFalse(result["reference_verified"])

    def test_environment_mismatch_and_missing_environment_block_comparison(self):
        self.all_complete()
        base.write_json(self.output / "execution_environment.json", {"different": "GPU"})
        self.assertFalse(self.summarize()["execution_environment_compatible"])
        (self.output / "execution_environment.json").unlink()
        choice = self.summarize()["decision"]
        self.assertEqual(choice["action"], "resolve_missing_or_incompatible_execution_environment")

    def test_source_mismatch_blocks_model_decision(self):
        self.all_complete()
        self.runner.source_hashes.return_value = {"changed": "source"}
        result = self.summarize()
        self.assertFalse(result["source_matches_current"])
        self.assertEqual(result["decision"]["action"], "resolve_changed_source_identity")

    def test_missing_cost_does_not_discard_accuracy_or_claim_exact_cost(self):
        self.all_complete()
        (self.output / "tasks" / self.tasks[0]["task_id"] / "attempts/test.json").unlink()
        result = self.summarize()
        self.assertEqual(result["complete_tasks"], 4)
        self.assertIsNone(result["rows"][0]["worker_wall_seconds"])
        self.assertTrue(result["decision"]["paired_comparison_available"])
        self.assertFalse(result["decision"]["worker_cost_exact"])
        self.assertIsNone(result["decision"]["R3_over_C3_total_worker_time_ratio"])
        self.assertEqual(result["decision"]["action"], "review_missing_cost_evidence_before_model_decision")

    def test_unfinished_attempt_never_counts_as_exact_cost(self):
        self.all_complete()
        folder = self.output / "tasks" / self.tasks[0]["task_id"]
        base.write_json(folder / "attempts/abandoned.json", {"task_fingerprint": self.tasks[0]["fingerprint"],
            "status": "running"})
        result = self.summarize()
        self.assertEqual(result["rows"][0]["unfinished_attempts"], 1)
        self.assertIsNone(result["decision"]["R3_over_C3_total_worker_time_ratio"])

    def test_attempt_wrong_fingerprint_is_invalid_evidence(self):
        folder = self.complete(self.tasks[0])
        base.write_json(folder / "attempts/test.json", {"task_fingerprint": "other task", "status": "complete", "wall_seconds": 12})
        self.assertEqual(self.summarize()["rows"][0]["status"], "invalid_evidence")

    def test_resolved_numerical_failure_has_no_paired_recommendation(self):
        for task in self.tasks[:-1]:
            self.complete(task)
        task = self.tasks[-1]
        folder = runtime.ensure_identity(self.output, task)
        base.write_json(folder / "failure.json", {"task_id": task["task_id"],
            "task_fingerprint": task["fingerprint"], "kind": "algorithm_numerical", "message": "nonfinite"})
        result = self.summarize()
        self.assertEqual(result["status"], "resolved_with_numerical_failures")
        self.assertEqual(result["complete_tasks"], 3)
        self.assertEqual(result["decision"]["action"], "resolve_incomplete_or_invalid_evidence")

    def test_completed_snapshot_precedes_historical_execution_failure(self):
        task = self.tasks[0]
        folder = self.complete(task)
        base.write_json(folder / "failure.json", {"task_id": task["task_id"],
            "task_fingerprint": task["fingerprint"], "kind": "infrastructure_oom", "message": "old failure"})
        before = self.snapshot()
        self.assertEqual(self.summarize()["rows"][0]["status"], "complete")
        self.assertEqual(before, self.snapshot())

    def test_missing_loss_evidence_does_not_qualify(self):
        folder = self.complete(self.tasks[0])
        (folder / "observations.json").unlink()
        self.assertEqual(self.summarize()["rows"][0]["status"], "invalid_evidence")

    def test_reference_unfinished_attempt_prevents_exact_cost(self):
        self.all_complete()
        reference = self.manifest["reference"]
        next(r for r in reference["rows"] if r["setting"] == "R3")["unfinished_attempts"] = 1
        self.runner.audit_reference.return_value = deepcopy(reference)
        choice = self.summarize()["decision"]
        self.assertFalse(choice["worker_cost_exact"])
        self.assertIsNone(choice["R3_over_C3_total_worker_time_ratio"])

    def test_duplicate_reference_pair_does_not_become_model_recommendation(self):
        self.all_complete()
        reference = self.manifest["reference"]
        reference["rows"][0] = deepcopy(reference["rows"][1])
        self.runner.audit_reference.return_value = deepcopy(reference)
        self.assertEqual(self.summarize()["decision"]["action"], "resolve_incomplete_or_invalid_evidence")

    def test_invalid_study_returns_copyable_error(self):
        self.runner.read_study.side_effect = ValueError("manifest corrupt")
        result = self.summarize()
        self.assertEqual(result["status"], "unavailable_or_invalid_study")
        self.assertFalse(result["training_started_by_summary"])
        stream = io.StringIO()
        with redirect_stdout(stream):
            report.print_summary(result)
        self.assertIn("CIFAR_CNN_MATCH_END", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
