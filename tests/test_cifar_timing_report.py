"""Scientific denominator, integrity and missing-data tests for timing reports."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import cifar_timing_report as report
import run_cifar_diagnostic as clean_runner
from tests.test_cifar_six_pipeline import synthetic_run

base, runtime = report.base, report.runtime


class TimingReportTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.output, self.clean, self.matched = (self.root / name for name in ("timing", "clean", "matched"))
        self.output.mkdir()
        self.clean.mkdir()
        self.matched.mkdir()
        config = runtime.read_json(clean_runner.DEFAULT_CONFIG)["shared_parameters"]
        self.tasks = []
        for arm, method, k, start, ratios in (("A", "sm9rrs", 10, 12, (0., .1, .7)),
                ("B", "sm9rrs", 10, 25, (0., .1, .7)), ("C", "sm9rrs", 20, 25, (0., .1, .7)),
                ("FA12", "fedavg", 10, 12, (.1, .7)), ("FA25", "fedavg", 10, 25, (.1, .7))):
            for partition in ("iid", "dirichlet"):
                for ratio in ratios:
                    task = {"task_id": f"{arm}_{partition}_{int(ratio * 100)}", "arm": arm, "model": "v7_cnn",
                        "method": method, "phase": "validation", "candidate": {"candidate_id": arm},
                        "config": {**config, "method": method, "seed": report.DEV_SEED, "partition": partition,
                                   "malicious_ratio": ratio, "detector_window": k, "attack_start_round": start}}
                    task["fingerprint"] = base.digest(task)
                    self.tasks.append(task)
        self.environment = {"actual_compute_device": {"name": "synthetic"}}
        self.reference = {"execution_environment": self.environment, "chosen_model": "v7_cnn", "chosen_setting": "C0",
            "clean_rows": [{"setting": "C0", "partition": partition, "seed": seed, "status": "complete", "healthy": True,
                            "accuracy150": .61, "background_5_to_7": .10}
                           for partition in ("iid", "dirichlet") for seed in (2026093001, 2026093002)]}
        self.manifest = {"fingerprint": "timing-fingerprint", "spec": {"protocol": "cifar-cnn-timing-diagnostic-v1"},
            "source_sha256": {"test.py": "hash"}, "reference": deepcopy(self.reference)}
        self.protocol = SimpleNamespace(read_study=mock.Mock(return_value=(self.manifest, self.tasks)),
            source_hashes=mock.Mock(return_value={"test.py": "hash"}),
            audit_reference=mock.Mock(return_value=deepcopy(self.reference)))
        patcher = mock.patch.dict(sys.modules, {"cifar_timing_protocol": self.protocol})
        patcher.start()
        self.addCleanup(patcher.stop)
        base.write_json(self.output / "execution_environment.json", self.environment)

    def task(self, arm="A", partition="iid", ratio=.7):
        return next(t for t in self.tasks if t["arm"] == arm and t["config"]["partition"] == partition
                    and t["config"]["malicious_ratio"] == ratio)

    def diagnostic(self, rd, identity, malicious, *, accepted=True, admitted=True, revoked=False, start=12):
        return SimpleNamespace(round=rd, client_id=identity, is_malicious=malicious,
            aggregation_accepted=accepted, aggregation_weight=.01 if accepted else 0.,
            history_admitted=admitted, revoked=revoked, attack_active=malicious and rd >= start)

    def complete(self, task, accuracy=.60, asr=None, nonfinite=0, diagnostics=None, record_transform=None):
        config = base.fl.ExperimentConfig(**task["config"])
        asr = (.20 if config.malicious_ratio else .10) if asr is None else asr
        result = synthetic_run(config, accuracy=accuracy, asr=asr, nonfinite=nonfinite)
        start = config.attack_start_round
        if diagnostics is None:
            diagnostics = []
            if config.method == "sm9rrs":
                for rd in range(start, start + 5):
                    if result.malicious_clients:
                        diagnostics.append(self.diagnostic(rd, result.malicious_clients[0], True, start=start))
                    diagnostics.append(self.diagnostic(rd, "honest-0", False, start=start))
        result = replace(result, diagnostics=diagnostics)
        if record_transform:
            result = record_transform(result)
        folder = runtime.ensure_identity(self.output, task)
        base.experiments._write_completed_results_snapshot(folder, [result])
        (folder / "attempts").mkdir(exist_ok=True)
        base.write_json(folder / "attempts/first.json", {"task_fingerprint": task["fingerprint"],
            "status": "complete", "wall_seconds": 120.})
        return folder

    def summarize(self):
        return report.summarize(self.output, self.clean, self.matched)

    def row(self, task):
        return next(r for r in self.summarize()["rows"] if r["task_id"] == task["task_id"])

    def all_complete(self):
        for task in self.tasks:
            self.complete(task)

    def snapshot(self):
        return {str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.root.rglob("*") if p.is_file()}

    def test_sparse_malicious_observations_use_five_not_seventy_as_denominator(self):
        task = self.task()
        diagnostics = [self.diagnostic(12, str(i), True, accepted=i < 4, admitted=i < 3) for i in range(5)]
        self.complete(task, diagnostics=diagnostics)
        group = self.row(task)["mechanism_windows"]["first_round"]["client_diagnostics"]["malicious"]
        self.assertEqual(group["original_clients"], 70)
        self.assertEqual(group["observed_verified_finite_updates"], 5)
        self.assertEqual(group["aggregation_accepted"], 4)
        self.assertEqual(group["aggregation_acceptance_rate"], .8)
        self.assertEqual(group["history_admission_rate"], .6)
        self.assertEqual(group["unobserved_remaining_client_rounds"], 65)

    def test_revoked_before_attack_distinguished_from_unobserved_remaining(self):
        task = self.task()
        diagnostics = [self.diagnostic(12, str(i), True) for i in range(5)]
        def revoked(result):
            return replace(result, records=[replace(r, true_positive_revocations=65 if r.round >= 11 else 0)
                                            for r in result.records])
        self.complete(task, diagnostics=diagnostics, record_transform=revoked)
        row = self.row(task)
        group = row["mechanism_windows"]["first_round"]["client_diagnostics"]["malicious"]
        self.assertEqual(row["pre_attack_designated_malicious_revoked"], 65)
        self.assertEqual(group["prior_revoked_client_rounds"], 65)
        self.assertEqual(group["remaining_client_rounds"], 5)
        self.assertEqual(group["unobserved_remaining_client_rounds"], 0)

    def test_absent_observations_do_not_become_zero_rejection_or_acceptance_rates(self):
        task = self.task()
        self.complete(task, diagnostics=[])
        group = self.row(task)["mechanism_windows"]["first_round"]["client_diagnostics"]["malicious"]
        self.assertEqual(group["status"], "no_observed_records")
        self.assertEqual(group["unobserved_remaining_client_rounds"], 70)
        self.assertIsNone(group["aggregation_acceptance_rate"])
        self.assertIsNone(group["history_admission_rate"])

    def test_fedavg_has_no_client_rates_but_retains_round_weight_metrics(self):
        task = self.task(arm="FA12")
        self.complete(task, record_transform=lambda result: replace(result,
            records=[replace(r, malicious_weight_mass=.7, honest_weight_loss=.1) for r in result.records]))
        window = self.row(task)["mechanism_windows"]["first_five_rounds"]
        self.assertEqual(window["client_diagnostics"]["status"], "not_available_for_fedavg")
        self.assertIsNone(window["client_diagnostics"]["malicious"])
        self.assertEqual(window["malicious_weight_mass"]["values"], [.7] * 5)

    def test_first_five_weight_values_and_history_are_windowed_without_clamping(self):
        task = self.task()
        raw = 1.0000000000000002
        values = [raw, .8, .6, .4, .2]
        diagnostics = [self.diagnostic(11, "0", True)]
        diagnostics += [self.diagnostic(rd, "0", True, admitted=rd < 14) for rd in range(12, 17)]
        diagnostics += [self.diagnostic(rd, "honest-0", False, admitted=rd != 13) for rd in range(12, 17)]
        self.complete(task, diagnostics=diagnostics, record_transform=lambda result: replace(result,
            records=[replace(r, malicious_weight_mass=values[r.round - 12] if 12 <= r.round <= 16 else 0.,
                             honest_weight_loss=.1) for r in result.records]))
        window = self.row(task)["mechanism_windows"]["first_five_rounds"]
        self.assertEqual(window["malicious_weight_mass"]["values"], values)
        self.assertEqual(window["malicious_weight_mass"]["max"], raw)
        self.assertAlmostEqual(window["malicious_weight_mass"]["mean"], .6)
        self.assertEqual(window["client_diagnostics"]["malicious"]["history_admitted"], 2)
        self.assertEqual(window["client_diagnostics"]["malicious"]["observed_verified_finite_updates"], 5)
        self.assertEqual(window["client_diagnostics"]["honest"]["history_admitted"], 4)
        self.assertEqual(window["client_diagnostics"]["honest"]["observed_verified_finite_updates"], 5)

    def test_honest_revocation_fraction_uses_original_honest_population(self):
        task = self.task()
        self.complete(task, record_transform=lambda result: replace(result,
            records=[replace(r, false_positive_revocations=3 if r.round >= 16 else 0) for r in result.records]))
        row = self.row(task)
        self.assertEqual(row["original_honest_clients"], 30)
        self.assertEqual(row["false_positive_revocations"], 3)
        self.assertEqual(row["false_positive_revocation_rate"], .1)
        self.assertEqual(row["mechanism_windows"]["first_five_rounds"]["new_honest_revocations"], 3)

    def test_full_attack_history_counts_include_later_pollution_without_round_lists(self):
        task = self.task()
        diagnostics = [self.diagnostic(rd, "0", True, admitted=rd >= 17) for rd in range(12, 19)]
        self.complete(task, diagnostics=diagnostics)
        row = self.row(task)
        early = row["mechanism_windows"]["first_five_rounds"]["client_diagnostics"]["malicious"]
        full = row["full_attack_period"]
        self.assertEqual(early["history_admitted"], 0)
        self.assertEqual(full["expected_rounds"], 139)
        self.assertEqual(full["observed_rounds"], 139)
        self.assertEqual(full["client_diagnostics"]["malicious"]["observed_verified_finite_updates"], 7)
        self.assertEqual(full["client_diagnostics"]["malicious"]["history_admitted"], 2)
        self.assertEqual(full["client_diagnostics"]["malicious"]["history_admission_rate"], 2 / 7)
        self.assertNotIn("coverage_by_round", full["client_diagnostics"]["malicious"])
        self.assertNotIn("values", full["malicious_weight_mass"])

    def test_full_attack_summary_is_not_claimed_for_clean_control(self):
        task = self.task(ratio=0.)
        self.complete(task)
        self.assertEqual(self.row(task)["full_attack_period"], {"status": "not_applicable_clean"})

    def test_readonly_missing_evidence_actual_n_and_no_loss_observer_requirement(self):
        task = self.task()
        folder = self.complete(task)
        self.assertFalse((folder / "observations.json").exists())
        before = self.snapshot()
        with mock.patch.object(base, "load_split", side_effect=AssertionError("data")), \
                mock.patch.object(base.experiments, "run_measured_experiment", side_effect=AssertionError("train")):
            result = self.summarize()
        self.assertEqual(before, self.snapshot())
        self.assertEqual(result["complete_tasks"], 1)
        self.assertEqual(result["healthy_tasks"], 1)
        missing = next(r for r in result["rows"] if r["status"] == "pending")
        self.assertNotIn("accuracy150", missing)
        self.assertEqual(result["decision"]["action"], "resolve_incomplete_or_invalid_execution_evidence")

    def test_complete_unhealthy_retains_accuracy_and_asr_without_qualifying(self):
        self.all_complete()
        task = self.task()
        self.complete(task, accuracy=.65, asr=.3, nonfinite=1)
        result = self.summarize()
        row = next(r for r in result["rows"] if r["task_id"] == task["task_id"])
        self.assertEqual(result["complete_tasks"], 26)
        self.assertEqual(result["healthy_tasks"], 25)
        self.assertEqual(row["accuracy150"], .65)
        self.assertEqual(row["attack_asr150"], .3)
        self.assertFalse(row["healthy"])
        self.assertEqual(result["decision"]["action"], "review_unhealthy_completed_runs_before_next_stage")

    def test_numerical_interruption_is_retained_without_fabricated_final_metrics(self):
        task = self.task()
        folder = runtime.ensure_identity(self.output, task)
        base.write_json(folder / "failure.json", {"task_id": task["task_id"], "task_fingerprint": task["fingerprint"],
            "kind": "algorithm_numerical", "message": "nonfinite at round 18"})
        base.write_json(folder / "progress.json", {"last_completed_round": 17})
        row = self.row(task)
        self.assertEqual(row["status"], "algorithm_numerical")
        self.assertEqual(row["last_completed_round"], 17)
        self.assertNotIn("accuracy150", row)

    def test_terminal_early_snapshot_does_not_mislabel_last_round_as_final150(self):
        task = self.task()
        self.complete(task, diagnostics=[], record_transform=lambda result: replace(result,
            stopped_round=14, records=result.records[:15]))
        row = self.row(task)
        self.assertEqual(row["status"], "terminal_incomplete")
        self.assertIsNone(row["accuracy150"])
        self.assertEqual(row["last_observed_accuracy"], .60)
        self.assertEqual(row["mechanism_windows"]["first_five_rounds"]["malicious_weight_mass"]["n"], 3)

    def test_unreached_attack_window_has_unknown_population_coverage(self):
        task = self.task(arm="B")
        self.complete(task, diagnostics=[], record_transform=lambda result: replace(result,
            stopped_round=14, records=result.records[:15]))
        window = self.row(task)["mechanism_windows"]["first_five_rounds"]
        self.assertEqual(window["rounds_observed"], [])
        self.assertIsNone(window["malicious_weight_mass"]["mean"])
        group = window["client_diagnostics"]["malicious"]
        self.assertIsNone(group["remaining_client_rounds"])
        self.assertIsNone(group["unobserved_remaining_client_rounds"])
        self.assertIsNone(group["aggregation_acceptance_rate"])

    def test_bad_identity_rejected_without_repair(self):
        task = self.task()
        folder = self.complete(task)
        bad = deepcopy(task)
        bad["arm"] = "B"
        base.write_json(folder / "task.json", bad)
        before = self.snapshot()
        self.assertEqual(self.row(task)["status"], "invalid_evidence")
        self.assertEqual(before, self.snapshot())

    def test_changed_reference_source_and_environment_prevent_comparison(self):
        self.all_complete()
        self.protocol.audit_reference.return_value["chosen_setting"] = "C3"
        result = self.summarize()
        self.assertFalse(result["reference_verified"])
        self.assertIsNone(result["comparisons"])
        self.assertEqual(result["reference_rows"], [])
        self.protocol.audit_reference.return_value = deepcopy(self.reference)
        self.protocol.source_hashes.return_value = {"changed": "source"}
        self.assertEqual(self.summarize()["decision"]["action"], "resolve_changed_source_identity")
        self.protocol.source_hashes.return_value = {"test.py": "hash"}
        base.write_json(self.output / "execution_environment.json", {"other": "GPU"})
        self.assertEqual(self.summarize()["decision"]["action"], "resolve_missing_or_incompatible_execution_environment")

    def test_pairing_uses_partition_ratio_seed_and_separates_clean_asr(self):
        self.all_complete()
        self.complete(self.task("A", "iid", .1), accuracy=.50, asr=.6)
        self.complete(self.task("B", "iid", .1), accuracy=.55, asr=.4)
        self.complete(self.task("FA12", "iid", .1), accuracy=.40, asr=.8)
        self.complete(self.task("FA25", "iid", .1), accuracy=.42, asr=.7)
        result = self.summarize()
        pairs = result["comparisons"]["A_to_B"]
        self.assertEqual(pairs["paired_n"], 6)
        self.assertEqual(pairs["attacked_paired_n"], 4)
        self.assertAlmostEqual(pairs["mean_attacked_asr_change_pp"], -5.)
        joint = next(p for p in result["comparisons"]["attack_start_joint_review"]
                     if p["partition"] == "iid" and p["malicious_ratio"] == .1)
        self.assertAlmostEqual(joint["Ours_ASR_change_pp"], -20.)
        self.assertAlmostEqual(joint["FedAvg_ASR_change_pp"], -10.)
        self.assertEqual(result["decision"]["action"], "review_timing_and_mechanism")
        self.assertIsNone(result["decision"]["selected_arm"])
        self.assertFalse(result["decision"]["next_stage_started"])

    def test_clean_utility_inclusive_3pp_is_separate_from_base_health(self):
        task = self.task("A", "iid", 0.)
        self.complete(task, accuracy=.58)
        row = self.row(task)
        self.assertTrue(row["healthy"])
        self.assertAlmostEqual(row["clean_utility_vs_C0"]["accuracy_drop_pp"], 3.)
        self.assertTrue(row["clean_utility_vs_C0"]["within_3pp"])
        self.complete(task, accuracy=.579)
        row = self.row(task)
        self.assertTrue(row["healthy"])
        self.assertFalse(row["clean_utility_vs_C0"]["within_3pp"])

    def test_own_arm_clean_background_and_accuracy_differences(self):
        self.complete(self.task("B", "dirichlet", 0.), accuracy=.59, asr=.12)
        task = self.task("B", "dirichlet", .7)
        self.complete(task, accuracy=.43, asr=.62)
        control = self.row(task)["attack_vs_same_arm_clean"]
        self.assertAlmostEqual(control["attack_accuracy_loss_pp"], 16.)
        self.assertAlmostEqual(control["attack_final_minus_clean_background_pp"], 50.)
        self.assertEqual(control["clean_background_5_to_7"], .12)

    def test_damaged_attempt_provenance_invalidates_and_unfinished_cost_is_not_exact(self):
        task = self.task()
        folder = self.complete(task)
        base.write_json(folder / "attempts/first.json", {"task_fingerprint": "wrong", "status": "complete", "wall_seconds": 1})
        self.assertEqual(self.row(task)["status"], "invalid_evidence")
        base.write_json(folder / "attempts/first.json", {"task_fingerprint": task["fingerprint"], "status": "running"})
        self.assertFalse(self.row(task)["worker_cost_exact"])

    def test_duplicate_client_record_and_excessive_weight_mass_rejected(self):
        task = self.task()
        d = self.diagnostic(12, "0", True)
        self.complete(task, diagnostics=[d, d])
        self.assertEqual(self.row(task)["status"], "invalid_evidence")
        self.complete(task, record_transform=lambda result: replace(result,
            records=[replace(r, malicious_weight_mass=1.01) for r in result.records]))
        self.assertEqual(self.row(task)["status"], "invalid_evidence")

    def test_history_admission_cannot_be_claimed_for_rejected_update(self):
        task = self.task()
        self.complete(task, diagnostics=[self.diagnostic(12, "0", True, accepted=False, admitted=True)])
        self.assertEqual(self.row(task)["status"], "invalid_evidence")

    def test_copyable_error_markers_and_two_reference_controls(self):
        result = self.summarize()
        self.assertEqual(len(result["reference_rows"]), 2)
        self.protocol.audit_reference.side_effect = ValueError("reference damaged")
        stream = io.StringIO()
        with redirect_stdout(stream):
            report.print_summary(self.summarize())
        self.assertIn("reference damaged", stream.getvalue())
        self.assertIn("CIFAR_TIMING_BEGIN", stream.getvalue())
        self.assertIn("CIFAR_TIMING_END", stream.getvalue())
        self.protocol.read_study.side_effect = ValueError("manifest damaged")
        self.assertEqual(self.summarize()["status"], "unavailable_or_invalid_study")


if __name__ == "__main__":
    unittest.main()
