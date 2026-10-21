"""Synthetic, public-JSON tests for the independent fixed C7 station report."""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from sm9rrsfl.fl import ExperimentConfig
from scripts.experiments.cifar10 import c7_station_report as report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def spec(method="fedavg", ratio=.6, phase="validation", partition="iid", seed=None):
    config = asdict(replace(ExperimentConfig(), method=method, malicious_ratio=ratio,
                           rounds=150, num_clients=100, seed=seed or (301 if phase == "validation" else 401),
                           partition=partition, dirichlet_alpha=.5, local_epochs=1,
                           batch_size=50, lr=.01, lr_decay=.99, detector_window=20,
                           attack="alternating_minimization", attack_start_round=25,
                           attack_source_label=5, attack_target_label=7, attack_target_count=1,
                           attack_epochs=1, attack_boost=20., attack_distance_weight=.0001,
                           eval_interval=1, early_stop=False, compute_backend="torch", device="cuda"))
    key = report._digest(report._semantic(config))
    return {"phase": phase, "candidate_id": method + "-fixed", "method": method,
            "config": config, "profile": copy.deepcopy(report.PROFILE) if ratio else None,
            "task_dir": f"c7_runtime/{phase}/{key}"}


def panel(root, specs, completed=True):
    body = {"protocol": {"name": "synthetic fixed C7", "selection": "none"},
            "source_files": {}, "task_specs": specs, "official_test_used_for_selection": False,
            "fixed_parameters_before_validation": True}
    manifest = {**body, "fingerprint": report._digest(body)}
    write(root / "protocol_manifest.json", manifest)
    phases = {}
    for phase in sorted({s["phase"] for s in specs}):
        run = {"tuning_phase": phase,
               "tuning_context": {"protocol_fingerprint": manifest["fingerprint"],
                                  "c7_runtime_protocol": "cifar_c7_station_runtime_v1",
                                  "c7_attack_profile": copy.deepcopy(report.PROFILE),
                                  "evaluation_scope": report.SCOPES[phase],
                                  "parameter_selection": "fixed_before_validation"},
               "candidates": [{"config": s["config"], "candidate_id": s["candidate_id"], "method": s["method"]}
                              for s in specs if s["phase"] == phase]}
        run["fingerprint"] = report._digest(run)
        write(root / ".tuning_state" / phase / run["fingerprint"] / "run_manifest.json", run)
        phases[phase] = {"status": "complete", "fingerprint": run["fingerprint"]}
        if completed:
            for item in [s for s in specs if s["phase"] == phase]:
                write_task(root, item, run["fingerprint"])
    write(root / "controller_status.json", {"fingerprint": manifest["fingerprint"], "phases": phases})
    return manifest, phases


def write_task(root, item, phase_fingerprint):
    config, phase = item["config"], item["phase"]
    evaluation, source = (7500, 750) if phase == "validation" else (10000, 1000)
    shape = {"x_train": [40000, 3072], "y_train": [40000], "x_test": [evaluation, 3072],
             "y_test": [evaluation], "x_attack": [2500, 3072], "y_attack": [2500]}
    data = {"arrays": {name: {"shape": size, "sha256": "a" * 64, "dtype": "<f4"}
                       for name, size in shape.items()},
            "evaluation_count": evaluation, "source_transfer_count": source, "attack_auxiliary_count": 2500}
    body = {"protocol": "cifar_c7_station_runtime_v1", "stage": phase,
            "task_id": phase + "_" + item["method"], "task_key": report._digest(report._semantic(config)),
            "semantic_config": report._semantic(config), "run_fingerprint": phase_fingerprint,
            "profile": copy.deepcopy(report.PROFILE), "attack_enabled": config["malicious_ratio"] > 0,
            "evaluation_scope": report.SCOPES[phase], "data_identity": data}
    identity = {**body, "fingerprint": report._digest(body), "config": copy.deepcopy(config)}
    targets = {"stage": phase, "evaluation_scope": report.SCOPES[phase], "evaluation_count": evaluation,
               "training_target_indices": [7], "legacy_evaluation_indices": [0],
               "source_transfer_indices": list(range(source)), "training_target_count": 1,
               "legacy_target_count": 1, "source_transfer_count": source,
               "training_target_is_optimizer_input": True, "transfer_target_is_optimizer_input": False,
               "official_test_used_for_selection": False}
    rows = []
    count = round(100 * config["malicious_ratio"])
    for rid in range(151):
        attack = rid >= 25 and bool(count)
        rows.append({"round": rid, "method": config["method"], "malicious_ratio": config["malicious_ratio"],
                     "attack_active": attack, "accuracy": .4 if attack else .5,
                     "attack_target_success_rate": float(attack), "legacy_asr": float(attack),
                     "attack_target_confidence": .9 if attack else .1,
                     "legacy_confidence": .9 if attack else .1,
                     "training_target_success": float(attack), "training_target_confidence": .9 if attack else .1,
                     "source_transfer_asr": .8 if attack else .1, "source_transfer_confidence": .8 if attack else .1,
                     "training_target_count": 1, "legacy_target_count": 1, "source_transfer_count": source,
                     "honest_weight_loss": 0., "malicious_weight_mass": 0.,
                     "accepted_updates": 100 if rid else 0, "rejected_updates": 0,
                     "blacklisted_clients": 0, "false_positive_revocations": 0, "true_positive_revocations": 0,
                     "nonfinite_updates": 0, "cumulative_nonfinite_updates": 0,
                     "attack_calls_this_round": count if attack else 0})
    status = {"status": "complete", "task_id": body["task_id"], "task_key": body["task_key"],
              "stage": phase, "fingerprint": identity["fingerprint"], "run_fingerprint": phase_fingerprint,
              "requested_rounds": 150, "completed_round": 150, "elapsed_seconds": 12.,
              "nonfinite_updates": 0, "final_metrics": rows[-1]}
    result = {**status, "metric_scope": targets, "malicious_clients": [f"client-{i}" for i in range(count)],
              "blacklisted_clients": [], "attack_seconds": 1. if count else 0.,
              "summary": {"final_accuracy": rows[-1]["accuracy"],
                          "final_attack_target_success_rate": rows[-1]["legacy_asr"],
                          "final_attack_target_confidence": rows[-1]["legacy_confidence"]}}
    directory = root / item["task_dir"]
    for name, value in {"task.json": identity, "status.json": status, "result.json": result,
                        "targets.json": targets,
                        "metrics.json": {"fingerprint": identity["fingerprint"], "stage": phase,
                                         "evaluation_scope": report.SCOPES[phase],
                                         "official_test_used_for_selection": False, "rounds": rows}}.items():
        write(directory / name, value)
    (directory / "attack_audit.jsonl").write_text("")
    return directory


class StationReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.items = [spec(ratio=0), spec(), spec(phase="final", ratio=0), spec(phase="final")]
        self.manifest, self.phases = panel(self.root, self.items)

    def mutate(self, item, filename, change):
        path = self.root / item["task_dir"] / filename
        value = json.loads(path.read_text())
        change(value)
        write(path, value)

    def mutate_rows(self, item, change):
        self.mutate(item, "metrics.json", lambda value: change(value["rounds"]))
        rows = json.loads((self.root / item["task_dir"] / "metrics.json").read_text())["rounds"]
        for filename in ("status.json", "result.json"):
            self.mutate(item, filename, lambda value: value.update(final_metrics=rows[-1]))

    def task(self, item=None):
        item = item or self.items[1]
        return report.task_summary(self.root, item, self.phases[item["phase"]]["fingerprint"])

    def test_stage_denominators_windows_and_clean_pairs(self):
        data = report.build_report(self.root)
        self.assertTrue(data["all_complete"], data)
        self.assertTrue(data["all_numerically_finite"])
        self.assertFalse(data["official_test_used_for_selection"])
        self.assertTrue(data["final_evaluation_uses_official_test"])
        validation = data["tasks"][self.items[1]["task_dir"]]
        final = data["tasks"][self.items[3]["task_dir"]]
        self.assertEqual(validation["metric_denominators"]["evaluation_count"], 7500)
        self.assertEqual(final["metric_denominators"]["evaluation_count"], 10000)
        self.assertEqual(final["metric_denominators"]["source_transfer_count"], 1000)
        self.assertEqual(final["windows"]["all_attack"]["rounds"], list(range(25, 151)))
        self.assertEqual(final["windows"]["tail10"]["rounds"], list(range(141, 151)))
        self.assertEqual(final["actual_attack_exposure"]["committed_attack_calls"], 60 * 126)
        self.assertTrue(all(pair["available"] for pair in data["clean_comparisons"]))
        self.assertTrue(data["clean_comparisons"][0]["windows"]["tail10"]["fedavg_attack_effectiveness_confirmed"])

    def test_partial_never_gets_final_window_or_success(self):
        item = self.items[1]
        self.mutate(item, "status.json", lambda value: value.update(status="running", completed_round=29))
        task = self.task()
        self.assertFalse(task["valid_complete"])
        self.assertEqual(task["windows"], {})
        self.assertEqual(task["observed_metric_points"], 30)
        self.assertEqual(task["uncommitted_metric_points"], 121)
        data = report.build_report(self.root)
        self.assertFalse(data["all_complete"])
        self.assertFalse(data["clean_comparisons"][0]["available"])
        self.assertEqual(next(g for g in data["seed_aggregates"] if g["stage"] == "validation" and g["malicious_ratio"])["observed_n"], 0)

    def test_tail_ten_and_all_attack_have_distinct_denominators(self):
        item = self.items[1]
        self.mutate_rows(item, lambda rows: [row.update(accuracy=.2 if row["round"] <= 144 else .4)
                                            for row in rows[141:]])
        task = self.task()
        self.assertTrue(task["valid_complete"])
        self.assertAlmostEqual(task["windows"]["tail10"]["mean"]["accuracy"], .32)
        self.assertAlmostEqual(task["windows"]["all_attack"]["mean"]["accuracy"], (122 * .4 + 4 * .2) / 126)
        self.assertEqual(task["windows"]["final"]["mean"]["accuracy"], .4)

    def test_failed_task_keeps_first_failure_and_observed_prefix(self):
        item = self.items[1]
        self.mutate(item, "status.json", lambda value: value.update(status="numerical_failure", completed_round=24,
                                                                   error="nonfinite gradient", error_type="WorkerNumericalFailure"))
        write(self.root / item["task_dir"] / "first_failure.json", {"stage": "target_gradient", "round": 25})
        task = self.task()
        self.assertFalse(task["numerically_finite"])
        self.assertEqual(task["first_failure"]["round"], 25)
        self.assertEqual(task["observed_metric_points"], 25)
        self.assertFalse(report.build_report(self.root)["all_complete"])

    def test_missing_point_nan_and_wrong_denominator_reject_complete(self):
        for mutation in ("point", "nan", "count", "fraction", "legacy"):
            item = self.items[1]
            write_task(self.root, item, self.phases[item["phase"]]["fingerprint"])
            def change(rows):
                if mutation == "point": rows.pop(44)
                elif mutation == "nan": rows[44]["training_target_confidence"] = float("nan")
                elif mutation == "count": rows[44]["source_transfer_count"] = 1000
                elif mutation == "fraction": rows[44]["source_transfer_asr"] = .12345
                else: rows[44]["legacy_asr"] = 0.
            self.mutate_rows(item, change)
            with self.subTest(mutation=mutation):
                self.assertFalse(self.task()["valid_complete"])
                self.assertTrue(self.task()["integrity_errors"])

    def test_95_percent_false_revocation_not_washed_by_old_gate(self):
        item = spec(method="ding13", ratio=.6)
        manifest, phases = panel(self.root / "ding", [item])
        directory = self.root / "ding" / item["task_dir"]
        metrics = json.loads((directory / "metrics.json").read_text())
        for row in metrics["rounds"][19:]:
            row.update(false_positive_revocations=38, true_positive_revocations=60,
                       blacklisted_clients=98, accepted_updates=2, attack_calls_this_round=0)
        write(directory / "metrics.json", metrics)
        for filename in ("status.json", "result.json"):
            value = json.loads((directory / filename).read_text())
            value["final_metrics"] = metrics["rounds"][-1]
            if filename == "result.json":
                value.update(blacklisted_clients=[f"client-{i}" for i in range(98)], attack_seconds=0.)
            write(directory / filename, value)
        data = report.build_report(self.root / "ding")
        task = data["tasks"][item["task_dir"]]
        self.assertTrue(task["valid_complete"], task)
        self.assertTrue(task["health"]["original_v14_health_pass"])
        self.assertFalse(task["health"]["complete_observed_health_pass"])
        self.assertEqual(task["health"]["false_revocation_rate"], .95)
        self.assertEqual(task["actual_attack_exposure"]["committed_attack_calls"], 0)
        self.assertEqual(data["seed_aggregates"][0]["observed_n"], 1)
        self.assertEqual(data["seed_aggregates"][0]["health_failed_seeds_included"], [301])

    def test_missing_audit_not_zero_and_default_does_not_scan(self):
        item = self.items[1]
        path = self.root / item["task_dir"] / "attack_audit.jsonl"
        path.unlink()
        exposure = self.task()["actual_attack_exposure"]
        self.assertFalse(exposure["audit_available"])
        self.assertNotIn("audit_attempt_records", exposure)
        self.assertEqual(exposure["committed_attack_calls"], 7560)
        path.write_text("intentionally malformed, default is metadata-only\n")
        self.assertTrue(self.task()["valid_complete"])
        task = report.task_summary(self.root, item, self.phases["validation"]["fingerprint"], include_attempt_audit=True)
        self.assertFalse(task["valid_complete"])

    def test_device_change_is_not_scientific_identity_change(self):
        item = copy.deepcopy(self.items[1])
        item["config"].update(device="cuda:7", sm9_workers=8)
        self.assertTrue(self.task(item)["valid_complete"])
        self.mutate(item, "task.json", lambda value: value["config"].update(device="cuda:1", sm9_workers=1))
        self.assertTrue(self.task(item)["valid_complete"])

    def test_full_audit_attempts_not_confused_with_committed_calls(self):
        item = self.items[1]
        directory = self.root / item["task_dir"]
        identity = json.loads((directory / "task.json").read_text())
        record = {"fingerprint": identity["fingerprint"], "task_key": identity["task_key"],
                  "stage": "validation", "round": 25, "client_idx": 0, "status": "complete",
                  "attempt_id": "a", "profile_id": "C7", "optimizer": "adam", "lr": .001,
                  "boost": 20., "stealth_steps": 10, "scope": "weights", "normal_steps": 10,
                  "target_steps": 1}
        (directory / "attack_audit.jsonl").write_text(json.dumps(record) + "\n" +
            json.dumps({**record, "attempt_id": "b"}) + "\n")
        rows = [{"round": 25, "attack_calls_this_round": 1}]
        exposure = report._exposure(directory, rows, item["config"], identity, True)
        self.assertEqual(exposure["audit_attempt_records"], 2)
        self.assertEqual(exposure["audit_unique_completed_client_rounds"], 1)
        self.assertEqual(exposure["audit_repeated_client_rounds"], [[25, 0]])
        self.assertEqual(exposure["committed_attack_calls"], 1)
        self.assertFalse(report.task_summary(self.root, item, self.phases["validation"]["fingerprint"],
                                            include_attempt_audit=True)["valid_complete"])
        record["boost"] = 5.
        (directory / "attack_audit.jsonl").write_text(json.dumps(record) + "\n")
        self.assertFalse(report.task_summary(self.root, item, self.phases["validation"]["fingerprint"],
                                            include_attempt_audit=True)["valid_complete"])

    def test_wrong_stage_profile_and_summary_reject(self):
        for mutation in ("stage", "profile", "summary"):
            item = self.items[1]
            write_task(self.root, item, self.phases["validation"]["fingerprint"])
            if mutation == "summary":
                self.mutate(item, "result.json", lambda value: value["summary"].update(final_accuracy=.1))
            else:
                def change(value):
                    if mutation == "stage": value["stage"] = "final"
                    else: value["profile"]["boost"] = 5.
                    value["fingerprint"] = report._digest({k: v for k, v in value.items() if k not in ("fingerprint", "config")})
                self.mutate(item, "task.json", change)
            with self.subTest(mutation=mutation): self.assertFalse(self.task()["valid_complete"])

    def test_phase_manifest_binding_and_missing_sidecars(self):
        path = self.root / ".tuning_state" / "validation" / self.phases["validation"]["fingerprint"] / "run_manifest.json"
        value = json.loads(path.read_text())
        value["tuning_context"]["protocol_fingerprint"] = "bad"
        write(path, value)
        data = report.build_report(self.root)
        self.assertFalse(data["reference_integrity_valid"])
        self.assertFalse(data["all_complete"])
        self.assertTrue(data["stages"]["final"]["all_complete"])
        (self.root / self.items[3]["task_dir"] / "targets.json").unlink()
        self.assertFalse(report.build_report(self.root)["stages"]["final"]["all_complete"])

    def test_pending_without_phase_state_is_legal(self):
        root = self.root / "pending"
        manifest, _ = panel(root, [spec()], completed=False)
        write(root / "controller_status.json", {"fingerprint": manifest["fingerprint"], "phases": {}})
        data = report.build_report(root)
        self.assertEqual(data["report_integrity_errors"], [])
        self.assertFalse(data["all_complete"])
        self.assertEqual(data["status_counts"], {"not_started": 1})

    def test_healthy_final_is_not_blocked_by_validation_health_failure(self):
        item = self.items[0]
        self.mutate_rows(item, lambda rows: [row.update(false_positive_revocations=11, blacklisted_clients=11)
                                            for row in rows[20:]])
        self.mutate(item, "result.json", lambda value: value.update(blacklisted_clients=[f"client-{i}" for i in range(11)]))
        data = report.build_report(self.root)
        self.assertTrue(data["all_complete"], data)
        self.assertFalse(data["all_observed_health_pass"])
        self.assertTrue(data["stages"]["final"]["all_complete"])
        self.assertTrue(data["stages"]["final"]["all_observed_health_pass"])

    def test_same_stage_seed_targets_cannot_be_mixed(self):
        item = self.items[1]
        self.mutate(item, "targets.json", lambda value: value.update(training_target_indices=[8]))
        targets = json.loads((self.root / item["task_dir"] / "targets.json").read_text())
        self.mutate(item, "result.json", lambda value: value.update(metric_scope=targets))
        data = report.build_report(self.root)
        self.assertFalse(data["stages"]["validation"]["all_complete"])
        self.assertTrue(data["stages"]["final"]["all_complete"])

    def test_source_drift_invalidates_without_rewriting_results(self):
        value = copy.deepcopy(self.manifest)
        value["source_files"] = {"sm9rrsfl/fl.py": "0" * 64}
        value["fingerprint"] = report._digest({k: v for k, v in value.items() if k != "fingerprint"})
        write(self.root / "protocol_manifest.json", value)
        before = (self.root / self.items[1]["task_dir"] / "metrics.json").read_bytes()
        data = report.build_report(self.root)
        self.assertFalse(data["reference_integrity_valid"])
        self.assertFalse(data["all_complete"])
        self.assertEqual(before, (self.root / self.items[1]["task_dir"] / "metrics.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
