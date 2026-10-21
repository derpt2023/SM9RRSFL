"""Supplement safety gates and six-method reports use synthetic public artifacts."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import run_cifar_c7_six_methods as runner
from scripts.experiments.cifar10 import cifar_c7_six_report as report
from tests.test_cifar_c7_replication_report import write_task


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol, self.bases, self.data = runner.load_six_methods()

    def test_exact_six_tasks_and_original_first_candidates(self):
        rows = runner.task_specs(self.protocol, self.bases)
        self.assertEqual([row["task_id"] for row in rows], [
            f"C7_{method}_dir{ratio}_s401" for ratio in (80, 60)
            for method in ("alignins", "krum", "ding13")])
        self.assertEqual(sum(row["config"]["rounds"] for row in rows), 180)
        for row in rows:
            c = row["config"]
            self.assertEqual((c["num_clients"], c["seed"], c["local_epochs"],
                              c["batch_size"], c["attack_start_round"], c["sm9_workers"]),
                             (100, 401, 1, 50, 25, 8))
            self.assertEqual((c["partition"], c["dirichlet_alpha"]), ("dirichlet", .5))
            self.assertEqual(row["profile"], runner.PROFILE)
            if c["method"] == "alignins":
                self.assertEqual((c["alignins_sparsity"], c["alignins_tda_radius"],
                                  c["alignins_mpsa_radius"]), (.8, .05, .1))

    def test_changed_profile_seed_or_candidate_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            for field, value in (("seed", 301), ("rounds", 50),
                                 ("profile", {**runner.PROFILE, "boost": 10}),
                                 ("candidate_ids", {**self.protocol["candidate_ids"],
                                                    "alignins": "alignins-002"})):
                path.write_text(json.dumps({**self.protocol, field: value}))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    runner.load_six_methods(path)

    def test_reference_cannot_be_output_or_its_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            for output in ("outputs", self.protocol["reference_dir"],
                           self.protocol["reference_dir"] + "/new",
                           "outputs/cifar10_attack_stability_v1/new"):
                path.write_text(json.dumps({**self.protocol, "output_dir": output}))
                with self.subTest(output=output), self.assertRaises(ValueError):
                    runner.load_six_methods(path)

    def test_dry_run_does_not_freeze_or_launch_pending_reference(self):
        with mock.patch.object(runner, "read_json", return_value={"status": "running"}), \
                mock.patch.object(runner, "load_six_methods",
                                  return_value=(self.protocol, self.bases, self.data)), \
                mock.patch.object(runner, "build_manifest") as freeze, \
                mock.patch.object(runner, "SixMethodController") as controller, \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runner.main(["--dry-run"]), 0)
            value = json.loads(output.getvalue())
            self.assertFalse(value["reference_ready"])
            self.assertIsNone(value["fingerprint"])
            freeze.assert_not_called()
            controller.assert_not_called()

    def test_real_reference_gate_rejects_running_controller(self):
        previous = {"source_files": {}, "reference_files": {}, "task_specs": []}
        previous["fingerprint"] = runner.digest(previous)
        protocol = {**self.protocol, "reference_fingerprint": previous["fingerprint"]}
        with mock.patch.object(runner, "read_json", side_effect=[previous, {"status": "running"}]), \
                self.assertRaisesRegex(ValueError, "must finish"):
            runner.reference_files(protocol)

    def test_complete_label_cannot_bypass_invalid_reference_metrics(self):
        previous = {"source_files": {}, "reference_files": {}, "task_specs": [],
                    "protocol": {"reference_dir": "unused"}}
        previous["fingerprint"] = runner.digest(previous)
        protocol = {**self.protocol, "reference_fingerprint": previous["fingerprint"]}
        with mock.patch.object(runner, "read_json", side_effect=[previous, {"status": "complete"}]), \
                mock.patch(
                    "scripts.experiments.cifar10.cifar_c7_replication_report.build_report",
                    return_value={"all_complete": True, "all_numerically_finite": False}), \
                self.assertRaisesRegex(ValueError, "valid and finite"):
            runner.reference_files(protocol)

    def test_new_source_and_old_evidence_remain_frozen(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("new.py", "old.json"):
                (root / name).write_text("original")
            manifest = {"source_files": {"new.py": runner.sha(root / "new.py")},
                        "reference_files": {"old.json": runner.sha(root / "old.json")}}
            with mock.patch.object(runner, "ROOT", root):
                runner.verify_frozen(manifest)
                for name in ("new.py", "old.json"):
                    (root / name).write_text("changed")
                    with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Immutable"):
                        runner.verify_frozen(manifest)
                    (root / name).write_text("original")

    def test_progress_counts_supplement_only_and_labels_estimate(self):
        controller = object.__new__(runner.SixMethodController)
        controller.out = Path("/tmp/no-written-six-progress")
        controller.specs = runner.task_specs(self.protocol, self.bases)
        controller.state = {"planned_rounds": 180, "status": "running"}
        controller.last_display, controller.start = 0., 1.
        controller.elapsed_before, controller.live = 0., False
        with mock.patch.object(runner, "read_json", return_value={}), \
                mock.patch.object(controller, "save"), redirect_stdout(io.StringIO()) as output:
            controller.display(True)
        self.assertIn("rounds 0/180 | tasks 0/6", output.getvalue())
        self.assertIn("methods differ", output.getvalue())

    def test_requests_have_no_inherited_time_limit_and_unique_identity(self):
        controller = object.__new__(runner.SixMethodController)
        controller.out = Path("/tmp/test-six-methods")
        controller.data = self.data
        controller.manifest = {"fingerprint": "new-six-protocol"}
        rows = runner.task_specs(self.protocol, self.bases)
        requests = [controller.request(row) for row in rows]
        self.assertTrue(all("deadline_epoch" not in request for request in requests))
        self.assertEqual(len({runner.task_identity(r)["fingerprint"] for r in requests}), 6)
        self.assertTrue(all(r["device"] == "mps" for r in requests))

    def test_terminal_failures_are_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            controller = object.__new__(runner.SixMethodController)
            controller.out, controller.data = Path(temp), self.data
            controller.manifest = {"fingerprint": "new-six-protocol"}
            spec = runner.task_specs(self.protocol, self.bases)[0]
            request = controller.request(spec)
            directory = controller.out / "tasks" / spec["task_id"]
            runner.write_json(directory / "task.json", runner.task_identity(request))
            runner.write_json(directory / "status.json", {"status": "numerical_failure"})
            with mock.patch.object(runner, "verify_frozen"), \
                    mock.patch.object(runner.subprocess, "Popen") as spawn:
                self.assertEqual(controller.run_task(spec)["status"], "numerical_failure")
                spawn.assert_not_called()


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out, self.ref = Path(self.temp.name) / "new", Path(self.temp.name) / "reference"
        reference = self.write_panel(self.ref, report.REFERENCES, {"reference_dir": "unused"})
        self.write_panel(self.out, report.TASKS,
                         {"reference_fingerprint": reference["fingerprint"]})
        runner.write_json(self.ref / "controller_status.json", {"status": "complete"})
        self.patch = mock.patch.object(report, "replication_report", return_value={
            "all_complete": True, "all_numerically_finite": True,
            "reference_integrity_valid": True, "all_comparisons_available": True})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def write_panel(self, root, specifications, protocol):
        specs = []
        for task_id, spec in specifications.items():
            directory = write_task(root, task_id, spec)
            identity = runner.read_json(directory / "task.json")
            specs.append({"task_id": task_id, "config": identity["config"],
                          "profile": identity["profile"]})
        body = {"task_specs": specs, "data_config": identity["data_config"], "protocol": protocol}
        manifest = {**body, "fingerprint": runner.digest(body)}
        runner.write_json(root / "manifest.json", manifest)
        for task_id, spec in specifications.items():
            directory = root / "tasks" / task_id
            identity = runner.read_json(directory / "task.json")
            identity["root_fingerprint"] = manifest["fingerprint"]
            identity["fingerprint"] = runner.digest({k: v for k, v in identity.items() if k != "fingerprint"})
            metrics = runner.read_json(directory / "metrics.json")
            metrics["fingerprint"] = identity["fingerprint"]
            for row in metrics["rounds"]:
                row["blacklisted_clients"] = 0
            status = runner.read_json(directory / "status.json")
            status.update(fingerprint=identity["fingerprint"], final_metrics=metrics["rounds"][-1])
            runner.write_json(directory / "task.json", identity)
            runner.write_json(directory / "metrics.json", metrics)
            runner.write_json(directory / "status.json", status)
            runner.write_json(directory / "result.json", {**status, "summary": {
                "malicious_clients": ",".join(f"client-{i}" for i in range(round(spec[1]*100)))}})
            runner.write_json(directory / "dataset_identity.json",
                              {"arrays": {"x_train": "same"}, "contract": {"split_seed": 42}})
        return manifest

    def build(self):
        return report.build_report(self.out, self.ref)

    def mutate_rows(self, key, change):
        directory = self.out / "tasks" / key
        metrics = runner.read_json(directory / "metrics.json")
        change(metrics["rounds"])
        runner.write_json(directory / "metrics.json", metrics)
        for filename in ("status.json", "result.json"):
            status = runner.read_json(directory / filename)
            status["final_metrics"] = metrics["rounds"][-1]
            runner.write_json(directory / filename, status)

    def test_full_report_has_six_method_rankings_and_scope_limits(self):
        result = self.build()
        self.assertTrue(result["all_complete"])
        self.assertTrue(result["all_numerically_finite"])
        self.assertTrue(result["all_six_method_comparisons_available"])
        self.assertEqual(len(result["tasks"]), 6)
        self.assertEqual(len(result["reference_tasks"]), 6)
        self.assertEqual(len(result["ours_pairwise_comparisons"]), 10)
        self.assertTrue(all(len(r["ordering"]) == 6 for r in result["six_method_rankings"]))
        self.assertTrue(result["all_observed_health_pass"])
        krum = result["tasks"]["C7_krum_dir80_s401"]
        self.assertFalse(krum["method_context"]["original_condition_holds"])
        self.assertFalse(result["seed401_matched_clean_available"])

    def test_partial_task_blocks_rankings_but_preserves_observed_health(self):
        key = "C7_ding13_dir80_s401"
        directory = self.out / "tasks" / key
        status = runner.read_json(directory / "status.json")
        status.update(status="partial", completed_round=5)
        runner.write_json(directory / "status.json", status)
        metrics = runner.read_json(directory / "metrics.json")
        metrics["rounds"] = metrics["rounds"][:6]
        metrics["rounds"][-1]["false_positive_revocations"] = 20
        metrics["rounds"][-1]["blacklisted_clients"] = 20
        runner.write_json(directory / "metrics.json", metrics)
        result = self.build()
        self.assertFalse(result["all_complete"])
        self.assertFalse(result["all_six_method_comparisons_available"])
        task = result["tasks"][key]
        self.assertIn("all_honest_revoked", task["health"]["observed_failure_reasons"])
        self.assertNotIn("final_accuracy", task)

    def test_unhealthy_low_asr_is_not_hidden_or_promoted_to_health_pass(self):
        key = "C7_ding13_dir80_s401"
        def revoke(rows):
            for row in rows[3:]:
                row.update(false_positive_revocations=20, blacklisted_clients=20,
                           training_target_success=0., source_transfer_asr=0., legacy_asr=0.)
        self.mutate_rows(key, revoke)
        result = self.build()
        task = result["tasks"][key]
        self.assertTrue(task["valid_complete"])
        self.assertTrue(task["numerically_finite"])
        self.assertFalse(task["health"]["complete_observed_health_pass"])
        self.assertTrue(task["health"]["attacked_false_revocation_rate_exceeds_0_10"])
        self.assertFalse(result["all_observed_health_pass"])
        self.assertTrue(result["all_six_method_comparisons_available"])

    def test_attack_audit_missing_does_not_claim_zero_and_retries_are_explicit(self):
        key = "C7_ding13_dir80_s401"
        self.assertFalse(self.build()["tasks"][key]["actual_attack_exposure"]["available"])
        row = {"task_id": key, "profile_id": "C7", "optimizer": "adam", "scope": "weights",
               "lr": .001, "boost": 20, "stealth_steps": 10, "round": 25,
               "client_id": "client-0", "status": "complete", "normal_steps": 7,
               "target_steps": 1, "target_skipped_zero_loss": 0}
        path = self.out / "tasks" / key / "audit.jsonl"
        path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        exposure = self.build()["tasks"][key]["actual_attack_exposure"]
        self.assertTrue(exposure["available"])
        self.assertEqual(exposure["totals"]["call_attempts"], 2)
        self.assertEqual(exposure["totals"]["target_steps"], 2)
        self.assertEqual(exposure["unique_client_rounds"], 1)
        self.assertFalse(exposure["completed_run_counts_without_duplicate_attempts"])
        self.assertEqual(exposure["duplicate_client_round_pairs"],
                         [{"round": 25, "client_id": "client-0"}])

    def test_wrong_attack_audit_profile_invalidates_task(self):
        key = "C7_krum_dir60_s401"
        path = self.out / "tasks" / key / "audit.jsonl"
        path.write_text(json.dumps({"task_id": key, "profile_id": "C5", "round": 25}) + "\n")
        task = self.build()["tasks"][key]
        self.assertFalse(task["valid_complete"])
        self.assertIn("audit identity/profile", task["integrity_errors"][0])

    def test_data_or_target_mismatch_blocks_same_seed_comparison(self):
        key = "C7_alignins_dir60_s401"
        for filename, mutation in (
                ("dataset_identity.json", lambda value: value["arrays"].update(x_train="changed")),
                ("targets.json", lambda value: value.update(training_target_indices=[999]))):
            path = self.out / "tasks" / key / filename
            original = path.read_text()
            value = json.loads(original)
            mutation(value)
            path.write_text(json.dumps(value))
            with self.subTest(filename=filename):
                result = self.build()
                self.assertFalse(result["reference_integrity_valid"])
                self.assertFalse(result["all_six_method_comparisons_available"])
            path.write_text(original)

    def test_candidate_mutation_rejected_even_with_valid_task_fingerprint(self):
        directory = self.out / "tasks" / "C7_alignins_dir60_s401"
        identity = runner.read_json(directory / "task.json")
        identity["config"]["alignins_sparsity"] = .95
        identity["fingerprint"] = runner.digest({k: v for k, v in identity.items() if k != "fingerprint"})
        runner.write_json(directory / "task.json", identity)
        for filename in ("metrics.json", "status.json", "result.json"):
            value = runner.read_json(directory / filename)
            value["fingerprint"] = identity["fingerprint"]
            runner.write_json(directory / filename, value)
        result = self.build()
        task = result["tasks"]["C7_alignins_dir60_s401"]
        self.assertFalse(task["valid_complete"])
        self.assertIn("declared manifest", task["integrity_errors"][0])

    def test_late_trend_is_descriptive_and_does_not_force_wanted_ranking(self):
        key = "C7_alignins_dir60_s401"
        def improve(rows):
            for row in rows[25:]:
                row["accuracy"] = .5 + .02 * (row["round"] - 25)
        self.mutate_rows(key, improve)
        result = self.build()
        trend = result["tasks"][key]["late_round_trends"]["27_30"]["accuracy"]
        self.assertAlmostEqual(trend["ols_slope_per_round"], .02)
        self.assertAlmostEqual(trend["end_minus_start"], .06)
        rank = next(r for r in result["six_method_rankings"]
                    if r["malicious_ratio"] == .6 and r["metric"] == "final_accuracy")
        self.assertEqual(rank["ordering"][0]["method"], "alignins")
        self.assertIn("not_estimated", trend["statistical_significance"])

    def test_result_mismatch_and_reference_pending_are_explicit_failures(self):
        path = self.out / "tasks/C7_krum_dir60_s401/result.json"
        value = runner.read_json(path)
        value["completed_round"] = 29
        runner.write_json(path, value)
        self.assertFalse(self.build()["all_complete"])
        runner.write_json(self.ref / "controller_status.json", {"status": "running"})
        self.assertFalse(self.build()["reference_integrity_valid"])


if __name__ == "__main__":
    unittest.main()
