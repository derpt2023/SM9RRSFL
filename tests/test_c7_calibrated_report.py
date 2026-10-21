"""Synthetic evidence tests for the independent calibrated C7 report."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from sm9rrsfl.fl import ExperimentConfig
from scripts.experiments.cifar10 import c7_calibrated_report as report


DATA = {"dataset": "cifar10", "data_dir": "/fixed/data", "seed": 42,
        "split_seed": 20261013, "train_samples": 50000,
        "test_samples": 10000, "validation_fraction": .15}


def spec(method="fedavg", ratio=.8, seed=402, rounds=60, name=None):
    config = replace(ExperimentConfig(), method=method, malicious_ratio=ratio, seed=seed,
                     rounds=rounds, num_clients=100, partition="dirichlet", dirichlet_alpha=.5,
                     local_epochs=1, batch_size=50, lr=.01, lr_decay=.99, detector_window=20,
                     attack="alternating_minimization", attack_start_round=25,
                     attack_source_label=5, attack_target_label=7, attack_target_count=1,
                     attack_epochs=1, attack_distance_weight=.0001, eval_interval=1,
                     early_stop=False, compute_backend="torch", device="mps")
    return {"task_id": name or f"{method}_{ratio}_{seed}_{rounds}", "config": asdict(config),
            "profile": copy.deepcopy(report.PROFILE) if ratio else None}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def panel(root, specs):
    body = {"protocol": {"name": "synthetic"}, "source_files": {}, "reference_files": {},
            "task_specs": specs, "data_config": DATA}
    manifest = {**body, "fingerprint": report._digest(body)}
    write(root / "manifest.json", manifest)
    for item in specs:
        write_task(root, item, manifest["fingerprint"])
    return manifest


def write_task(root, item, root_fingerprint):
    config = {k: v for k, v in item["config"].items() if k != "device"}
    body = {"protocol": "cifar_attack_stability_task_v1", "root_fingerprint": root_fingerprint,
            "config": config, "profile": item["profile"], "data_config": DATA}
    identity = {**body, "fingerprint": report._digest(body)}
    path = root / "tasks" / item["task_id"]
    targets = {**report.COUNTS, "training_target_indices": [900],
               "legacy_validation_indices": [2], "transfer_validation_indices": list(range(750)),
               "official_test_used_for_selection": False, "training_target_is_optimizer_input": True,
               "transfer_target_is_optimizer_input": False}
    rows = []
    for rid in range(config["rounds"] + 1):
        attack = config["malicious_ratio"] > 0 and rid >= 25
        rows.append({
            "round": rid, "method": config["method"], "malicious_ratio": config["malicious_ratio"],
            "attack_active": attack, "accuracy": .5 - (.1 if attack else 0),
            "training_target_success": float(attack), "training_target_confidence": .9 if attack else .1,
            "legacy_asr": float(attack), "legacy_confidence": .8 if attack else .1,
            "source_transfer_asr": .8 if attack else .1,
            "source_transfer_confidence": .8 if attack else .1,
            "honest_weight_loss": 0., "malicious_weight_mass": 0.,
            "accepted_updates": 100 if rid else 0, "rejected_updates": 0,
            "blacklisted_clients": 0, "false_positive_revocations": 0,
            "true_positive_revocations": 0, "nonfinite_updates": 0,
            "cumulative_nonfinite_updates": 0, **report.COUNTS,
        })
    status = {"task_id": item["task_id"], "status": "complete", "completed_round": config["rounds"],
              "requested_rounds": config["rounds"], "fingerprint": identity["fingerprint"],
              "elapsed_seconds": 100., "nonfinite_updates": 0, "final_metrics": rows[-1]}
    malicious = [f"client-{i}" for i in range(round(100 * config["malicious_ratio"]))]
    dataset = {"arrays": {k: "a" * 64 for k in report.ARRAYS}, "contract": {
        "train_samples": 40000, "calibration_samples": 7500, "attack_auxiliary_samples": 2500,
        "split_seed": 20261013, "official_test_used_for_selection": False}}
    for name, value in {
        "task.json": identity, "status.json": status, "targets.json": targets,
        "dataset_identity.json": dataset,
        "metrics.json": {"fingerprint": identity["fingerprint"],
                         "official_test_used_for_selection": False, "rounds": rows},
        "result.json": {**status, "profile": item["profile"], "metric_scope": targets,
                        "summary": {"malicious_clients": ",".join(malicious)}},
    }.items():
        write(path / name, value)
    if malicious:
        audits = [{"task_id": item["task_id"], "profile_id": "C7", "optimizer": "adam",
                   "scope": "weights", "lr": .001, "boost": 20., "stealth_steps": 10,
                   "round": rid, "client_id": "client-0", "status": "complete",
                   "normal_steps": 10, "target_steps": 1, "target_skipped_zero_loss": 0}
                  for rid in range(25, config["rounds"] + 1)]
        (path / "audit.jsonl").write_text("".join(json.dumps(a) + "\n" for a in audits))
    return path


class CalibratedReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.items = [spec(ratio=0), spec()]
        self.manifest = panel(self.root, self.items)

    def mutate(self, item, filename, change):
        path = self.root / "tasks" / item["task_id"] / filename
        value = json.loads(path.read_text())
        change(value)
        write(path, value)

    def mutate_rows(self, item, change):
        self.mutate(item, "metrics.json", lambda v: change(v["rounds"]))
        rows = json.loads((self.root / "tasks" / item["task_id"] / "metrics.json").read_text())["rounds"]
        for file in ("status.json", "result.json"):
            self.mutate(item, file, lambda v: v.update(final_metrics=rows[-1]))

    def task(self, item=None):
        return report.task_summary(self.root, item or self.items[1], self.manifest["fingerprint"])

    def test_sixty_round_windows_and_matched_clean(self):
        output = report.build_report(self.root)
        self.assertTrue(output["all_complete"])
        self.assertTrue(output["all_numerically_finite"])
        task = output["tasks"][self.items[1]["task_id"]]
        self.assertEqual(task["attack_window_rounds"], list(range(25, 61)))
        self.assertEqual(task["last10_rounds"], list(range(51, 61)))
        self.assertEqual(task["actual_attack_exposure"]["unique_completed_client_rounds"], 36)
        pair = output["clean_comparisons"][0]
        self.assertTrue(pair["available"])
        self.assertAlmostEqual(pair["final_accuracy_loss"], .1)
        self.assertAlmostEqual(pair["last10_target_confidence_gain"], .8)
        self.assertTrue(pair["attack_window_attack_effectiveness"]["effective"])

    def test_thirty_round_calibration_horizon(self):
        item = spec(method="vert", ratio=0, seed=301, rounds=30)
        manifest = panel(self.root / "calibration", [item])
        task = report.task_summary(self.root / "calibration", item, manifest["fingerprint"])
        self.assertTrue(task["complete"])
        self.assertEqual(task["last10_rounds"], list(range(21, 31)))
        self.assertEqual(task["attack_window_rounds"], list(range(25, 31)))

    def test_final_window_uses_last_ten_not_last_six(self):
        item = self.items[1]
        self.mutate_rows(item, lambda rows: [
            row.update(accuracy=.2 if row["round"] <= 54 else .4)
            for row in rows if row["round"] >= 51])
        task = self.task()
        self.assertTrue(task["complete"])
        self.assertAlmostEqual(task["last10_mean_accuracy"], .32)
        self.assertGreater(task["late_round_trends"]["51_60"]["accuracy"]["ols_slope_per_round"], 0)

    def test_identity_binding_cannot_be_bypassed_with_recomputed_hash(self):
        def change(value):
            value["config"]["vert_top_k"] = 99
            value["fingerprint"] = report._digest({k: v for k, v in value.items() if k != "fingerprint"})
        self.mutate(self.items[1], "task.json", change)
        self.assertFalse(self.task()["complete"])
        self.assertIn("declared", self.task()["integrity_errors"][0])

    def test_wrong_root_and_result_status_disagreement(self):
        task = report.task_summary(self.root, self.items[1], "wrong-root")
        self.assertFalse(task["complete"])
        self.mutate(self.items[1], "result.json", lambda v: v.update(completed_round=59))
        self.assertFalse(self.task()["complete"])

    def test_missing_round_and_denominator_reject_completed_metrics(self):
        item = self.items[1]
        self.mutate_rows(item, lambda rows: rows[29].update(source_transfer_count=749))
        self.assertFalse(self.task()["complete"])
        write_task(self.root, item, self.manifest["fingerprint"])
        self.mutate_rows(item, lambda rows: rows.pop(29))
        self.assertFalse(self.task()["complete"])

    def test_missing_health_field_is_reported_as_invalid_not_a_reader_crash(self):
        self.mutate_rows(self.items[1], lambda rows: rows[29].pop("accepted_updates"))
        task = self.task()
        self.assertFalse(task["complete"])
        self.assertTrue(task["integrity_errors"])
        self.assertFalse(task["health"]["complete_observed_health_pass"])

    def test_nan_metric_rejected_and_positive_counter_not_stable(self):
        item = self.items[1]
        self.mutate_rows(item, lambda rows: rows[29].update(accuracy=float("nan")))
        self.assertFalse(self.task()["complete"])
        write_task(self.root, item, self.manifest["fingerprint"])
        def counters(rows):
            rows[25]["nonfinite_updates"] = 1
            for row in rows[25:]:
                row["cumulative_nonfinite_updates"] = 1
        self.mutate_rows(item, counters)
        for file in ("status.json", "result.json"):
            self.mutate(item, file, lambda v: v.update(nonfinite_updates=1))
        task = self.task()
        self.assertTrue(task["complete"])
        self.assertFalse(task["numerically_finite"])

    def test_partial_has_no_final_metrics_and_preserves_checkpoint_prefix(self):
        item = self.items[1]
        self.mutate(item, "status.json", lambda v: v.update(status="partial", completed_round=29))
        task = self.task()
        self.assertFalse(task["complete"])
        self.assertNotIn("final_accuracy", task)
        self.assertEqual(task["observed_metric_points"], 30)
        self.assertEqual(task["uncommitted_metric_rows"], 31)
        self.assertEqual(task["integrity_errors"], [])

    def test_duplicate_attack_attempts_are_flagged_without_erasing_results(self):
        path = self.root / "tasks" / self.items[1]["task_id"] / "audit.jsonl"
        content = path.read_text()
        path.write_text(content + content.splitlines()[0] + "\n")
        task = self.task()
        audit = task["actual_attack_exposure"]
        self.assertTrue(task["complete"])
        self.assertEqual(audit["totals"]["call_attempts"], 37)
        self.assertEqual(audit["unique_completed_client_rounds"], 36)
        self.assertFalse(audit["completed_run_counts_without_duplicate_attempts"])

    def test_audit_profile_and_nonmalicious_client_are_invalid(self):
        path = self.root / "tasks" / self.items[1]["task_id"] / "audit.jsonl"
        content = path.read_text()
        for patch in ({"boost": 10}, {"client_id": "client-99"}):
            rows = [json.loads(line) for line in content.splitlines()]
            rows[0].update(patch)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertFalse(self.task()["complete"])

    def test_missing_audit_does_not_claim_zero_calls(self):
        path = self.root / "tasks" / self.items[1]["task_id"] / "audit.jsonl"
        path.unlink()
        task = self.task()
        self.assertTrue(task["complete"])
        self.assertFalse(task["actual_attack_exposure"]["available"])
        self.assertNotIn("totals", task["actual_attack_exposure"])

    def test_health_failed_method_remains_in_six_rankings(self):
        items = [spec(method=m) for m in report.METHODS]
        root = self.root / "six"
        panel(root, items)
        item = next(x for x in items if x["config"]["method"] == "ding13")
        path = root / "tasks" / item["task_id"]
        metrics = json.loads((path / "metrics.json").read_text())
        for row in metrics["rounds"][10:]:
            row.update(false_positive_revocations=20, blacklisted_clients=20)
        write(path / "metrics.json", metrics)
        for filename in ("status.json", "result.json"):
            value = json.loads((path / filename).read_text())
            value["final_metrics"] = metrics["rounds"][-1]
            write(path / filename, value)
        output = report.build_report(root)
        self.assertTrue(output["all_complete"])
        self.assertFalse(output["all_observed_health_pass"])
        self.assertTrue(output["all_six_method_comparisons_available"])
        ding = output["tasks"][item["task_id"]]
        self.assertTrue(ding["health"]["pre_attack_all_honest_revoked"])
        self.assertIn("ding13", [v["method"] for v in output["six_method_rankings"][0]["ordering"]])

    def test_empty_aggregation_and_clean_false_revocation_are_visible(self):
        item = self.items[0]
        def failure(rows):
            rows[8]["accepted_updates"] = 0
            for row in rows[15:]:
                row.update(false_positive_revocations=11, blacklisted_clients=11)
        self.mutate_rows(item, failure)
        task = self.task(item)
        self.assertTrue(task["complete"])
        self.assertEqual(task["health"]["empty_aggregation_rounds"], [8])
        self.assertIn("clean_false_revocation_rate", task["health"]["observed_failure_reasons"])

    def test_same_seed_target_or_data_mismatch_invalidates_pair(self):
        self.mutate(self.items[1], "targets.json", lambda v: v.update(training_target_indices=[901]))
        self.mutate(self.items[1], "result.json",
                    lambda v: v["metric_scope"].update(training_target_indices=[901]))
        output = report.build_report(self.root)
        self.assertFalse(output["all_complete"])
        self.assertFalse(output["clean_comparisons"][0]["available"])

    def test_manifest_reference_mutation_and_data_configuration_binding(self):
        referenced = self.root / "reference.json"
        referenced.write_text("original")
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["reference_files"] = {str(referenced): "0" * 64}
        manifest["fingerprint"] = report._digest({k: v for k, v in manifest.items() if k != "fingerprint"})
        write(self.root / "manifest.json", manifest)
        output = report.build_report(self.root)
        self.assertFalse(output["reference_integrity_valid"])
        self.assertFalse(output["all_complete"])

    def test_clean_pair_requires_same_candidate(self):
        items = [spec(method="vert", ratio=0), spec(method="vert")]
        items[0]["config"]["vert_top_k"] = 20
        items[1]["config"]["vert_top_k"] = 40
        root = self.root / "different"
        panel(root, items)
        output = report.build_report(root)
        self.assertTrue(output["all_complete"])
        self.assertFalse(output["clean_comparisons"][0]["available"])


if __name__ == "__main__":
    unittest.main()
