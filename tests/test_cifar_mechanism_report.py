"""Actual CPU paired histories plus corruption/causal-boundary report regressions."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from sm9rrsfl import fl
from tests.test_cifar_client_step_probe_runtime import dataset
from cifar_cnn_history_runtime import history_runtime
import cifar_mechanism_runtime as runtime
import cifar_mechanism_report as report


class MechanismReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = fl.ExperimentConfig(method="sm9rrs", malicious_ratio=0., num_clients=3,
            rounds=30, partition="iid", seed=2201, compute_backend="torch", device="cpu",
            crypto_mode="simulated", detector_window=20, attack_start_round=25, attack_target_count=2,
            checkpoint_interval=1, early_stop=False, batch_size=5, local_epochs=1, sm9_workers=1)
        cls.observed, cls.base_tasks = {}, {}
        for arm in report.ARMS:
            variant, freeze = ("original", None) if arm == "H0" else ("Ours-FrozenHistory-v1", 25)
            task = {"task_id": "cpu-" + arm, "fingerprint": "cpu-" + arm, "arm": arm,
                "candidate": {"variant": variant}, "history_freeze_start_round": freeze,
                "policy": "singleton_backward_cudnn_deterministic", "config": asdict(cls.config)}
            with history_runtime(variant, freeze_start_round=freeze):
                with runtime.observe(task) as observer:
                    result = fl.run_experiment(dataset(), cls.config, checkpoint_callback=observer.checkpoint)
            cls.base_tasks[arm], cls.observed[arm] = task, observer.finish(result)

    def setUp(self):
        self.output = Path("/nonexistent/mechanism-read-only")
        self.uuid = "GPU-test-physical"
        baseline = self.observed["H0"]["numerical_policy"]["baseline"]
        self.metadata = {"actual_compute_device": {"name": "synthetic GPU", "uuid": self.uuid, "logical_device": "cuda:0"},
            "environment": {"CUDA_VISIBLE_DEVICES": self.uuid},
            "nvidia": {"gpus": [{"name": "synthetic GPU", "uuid": self.uuid, "driver_version": "test"}]},
            "torch": {"version": "test", **{k: baseline[k] for k in report.tail.ENVIRONMENT_FLAGS},
                "logical_cuda_devices": [{"logical_index": 0, "name": "synthetic GPU", "uuid": self.uuid}]}}
        self.tasks, self.artifacts = [], {}
        for repeat in (1, 2):
            for arm in report.ARMS:
                task = deepcopy(self.base_tasks[arm])
                task.update(task_id=arm + "_" + str(repeat), fingerprint=arm + "_" + str(repeat), repeat=repeat)
                self.tasks.append(task)
                payload = deepcopy(self.observed[arm])
                payload.update(task_id=task["task_id"], task_fingerprint=task["fingerprint"])
                self.artifacts[task["task_id"]] = {"status": "complete", "task_fingerprint": task["fingerprint"],
                    "artifact_fingerprint": hashlib.sha256(task["task_id"].encode()).hexdigest(),
                    "gpu_uuid": self.uuid, "environment": deepcopy(self.metadata), "observations": payload,
                    "fresh_start": True, "checkpoints_used": False, "completed_training_rounds": 30,
                    "requested_training_rounds": 30, "wall_seconds": .1}
        old = deepcopy(self.observed["H0"])
        for name in ("rounds", "evaluations", "checkpoints", "actual_batches", "singleton_batches"):
            old[name] = [r for r in old[name] if r["round"] <= 3]
        for row in old["rounds"]:
            for client in row["clients"]:
                client.pop("raw_update", None)
                client.pop("update_finite", None)
        anchors = [{"task": {"task_id": "old_" + str(r), "policy": "singleton_backward_cudnn_deterministic"},
                    "observations": deepcopy(old)} for r in (1, 2, 3)]
        self.manifest = {"fingerprint": "manifest", "same_gpu_uuid": self.uuid,
            "source_sha256": {"frozen_science.py": "a" * 64}, "reference": {
                "execution_environment": report.prefix.matched.normalized_environment(self.metadata),
                "deterministic_prefix_anchors": anchors}}
        self.protocol = SimpleNamespace(read_study=mock.Mock(return_value=(self.manifest, self.tasks)),
            load_completed=mock.Mock(side_effect=lambda output, task: self.artifacts.get(task["task_id"])),
            source_hashes=mock.Mock(return_value=self.manifest["source_sha256"]),
            evidence_hashes=mock.Mock(return_value={"evidence": "stable"}), verify_reference=mock.Mock())
        patcher = mock.patch.dict(sys.modules, {"cifar_mechanism_protocol": self.protocol})
        patcher.start()
        self.addCleanup(patcher.stop)

    def payload(self, arm="H0", repeat=2):
        return self.artifacts[arm + "_" + str(repeat)]["observations"]

    def test_actual_cpu_panel_complete_and_intended_freeze_is_not_context_mismatch(self):
        value = report.summarize(self.output)
        self.assertEqual((value["status"], value["complete_tasks"], value["available_pairs"]), ("complete", 4, 4), value)
        self.assertTrue(value["decision"]["within_arm_all_observations_equal"])
        self.assertTrue(value["decision"]["paired_preintervention_conditions_equal"])
        self.assertEqual(value["decision"]["conclusion"], "paired_clean_mechanism_evidence_ready_for_review")
        self.assertFalse(value["decision"]["formal_qualification_assessed"])
        self.assertFalse(value["decision"]["performance_improvement_assessed"])
        phase = value["rows"][1]["diagnostics"]["phases"][-1]
        self.assertEqual(phase["active_client_rounds"], 15)
        self.assertEqual(phase["history_admitted"], 0)
        self.assertEqual(phase["unchanged_history_and_live_normal_commits"], phase["actual_commit_calls"])

    def test_round25_training_difference_blocks_common_preconditions(self):
        a, b = deepcopy(self.observed["H1"]), deepcopy(self.observed["H0"])
        a["rounds"][24]["clients"][0]["raw_update"]["sha256"] = "0" * 64
        a["rounds"][24]["clients"][0]["update"]["sha256"] = "0" * 64
        pair = report.compare_observations(a, b, cross_arm=True)
        self.assertFalse(pair["common_through_round24_and_round25_precommit_conditions_equal"])
        self.assertEqual(pair["first_common_precondition_mismatch"]["round"], 25)

    def test_round25_weight_history_guard_is_not_the_declared_intervention(self):
        a, b = deepcopy(self.observed["H1"]), deepcopy(self.observed["H0"])
        a["rounds"][24]["diagnostics"][0]["history_frozen"] = True
        pair = report.compare_observations(a, b, cross_arm=True)
        self.assertFalse(pair["common_through_round24_and_round25_precommit_conditions_equal"])
        self.assertEqual(pair["first_common_precondition_mismatch"]["first"]["field"], "history_frozen")

    def test_round26_detector_difference_is_allowed_but_within_arm_still_requires_equality(self):
        a, b = deepcopy(self.observed["H1"]), deepcopy(self.observed["H0"])
        event = next(e for e in a["history_events"] if e["round"] == 26)
        event["decision"]["norm_score"] += 1
        pair = report.compare_observations(a, b, cross_arm=True)
        self.assertTrue(pair["common_through_round24_and_round25_precommit_conditions_equal"])
        self.assertTrue(pair["round26_local_training_equal_before_first_affected_detection"])
        self.assertGreater(pair["post_intervention_differing_boundaries"], 0)
        repeat = report.compare_observations(a, deepcopy(self.observed["H1"]))
        self.assertFalse(repeat["equal"])
        self.assertEqual(repeat["first_divergence"]["round"], 26)

    def test_round26_singleton_divergence_blocks_paired_mechanism_interpretation(self):
        # A valid differing observed gradient at this still-common local stage
        # must not be attributed to history, which is first read by detector26.
        for repeat in (1, 2):
            batch = next(b for b in self.payload("H1", repeat)["singleton_batches"] if b["round"] == 26)
            batch["gradients"]["sha256"] = "0" * 64
        value = report.summarize(self.output)
        self.assertEqual(value["status"], "complete", value)
        self.assertTrue(value["decision"]["within_arm_all_observations_equal"])
        self.assertFalse(value["decision"]["paired_preintervention_conditions_equal"])
        self.assertFalse(value["decision"]["round26_local_training_equal_before_first_affected_detection"])
        self.assertEqual(value["decision"]["conclusion"], "common_conditions_differ_no_paired_causal_interpretation")
        pair = next(p for p in value["pairs"] if p["kind"] == "cross_arm")
        self.assertTrue(pair["common_through_round24_and_round25_precommit_conditions_equal"])
        self.assertEqual(pair["first_round26_local_training_mismatch"]["stage"], "singleton.gradients")
        self.assertEqual(pair["first_round26_local_training_mismatch"]["scope"], "round26_local_common_condition")

    def test_actual_singleton_gradient_precedes_returned_client_update(self):
        a, b = deepcopy(self.observed["H0"]), deepcopy(self.observed["H0"])
        a["singleton_batches"][0]["gradients"]["sha256"] = "0" * 64
        a["rounds"][0]["clients"][0]["update"]["sha256"] = "0" * 64
        value = report.compare_observations(a, b)
        self.assertEqual(value["first_divergence"]["stage"], "singleton.gradients")
        self.assertIsNone(value["first_divergence"]["magnitude"])

    def test_missing_history_policy_or_active_coverage_is_invalid(self):
        edits = [lambda p: p["history_events"].pop(), lambda p: p["actual_batches"].pop(),
            lambda p: p["rounds"][24]["clients"].pop(),
            lambda p: p["history_events"][75]["commit"]["after"].update(history_size=999),
            lambda p: p["numerical_policy"].update(restored_on_exit=False),
            lambda p: p["checkpoints"][25].update(blacklisted=["client-0"])]
        original = deepcopy(self.payload())
        for edit in edits:
            with self.subTest(edit=edit):
                self.artifacts["H0_2"]["observations"] = deepcopy(original)
                edit(self.payload())
                value = report.summarize(self.output)
                self.assertEqual(value["complete_tasks"], 3)
                self.assertEqual(value["decision"]["conclusion"], "incomplete_or_invalid_evidence")

    def test_missing_or_nonfresh_artifact_is_not_a_health_failure(self):
        self.artifacts.pop("H0_2")
        value = report.summarize(self.output)
        row = next(r for r in value["rows"] if r["task_id"] == "H0_2")
        self.assertEqual(row["status"], "missing")
        self.assertNotIn("diagnostics", row)
        self.artifacts["H1_2"]["fresh_start"] = False
        self.assertEqual(report.summarize(self.output)["complete_tasks"], 2)

    def test_source_reference_race_removes_success_claim(self):
        self.protocol.evidence_hashes.side_effect = [{"x": "before"}, {"x": "after"}]
        self.assertEqual(report.summarize(self.output)["status"], "invalid")
        self.protocol.evidence_hashes.side_effect = None
        self.protocol.verify_reference.side_effect = ValueError("old source changed")
        self.assertEqual(report.summarize(self.output)["status"], "invalid")

    def test_summary_never_loads_data_or_runs_training_and_is_compact_public_json(self):
        with mock.patch.object(report.prefix.timing.base, "load_split", side_effect=AssertionError("data read")), \
             mock.patch.object(fl, "run_experiment", side_effect=AssertionError("training")):
            value = report.summarize(self.output)
        output = io.StringIO()
        with redirect_stdout(output):
            report.print_summary(value)
        lines = output.getvalue().splitlines()
        self.assertEqual((lines[0], lines[-1]), ("=== CIFAR_MECHANISM_BEGIN ===", "=== CIFAR_MECHANISM_END ==="))
        records = [json.loads(s) for s in lines[1:-1]]
        self.assertEqual(len(records), 11)
        self.assertEqual(sum(r["type"] == "task" for r in records), 4)
        self.assertEqual(sum(r["type"] == "pair" for r in records), 4)
        self.assertNotIn("task_tag", output.getvalue())
        self.assertNotIn("CUDA_VISIBLE_DEVICES", output.getvalue())
        self.assertLess(len(output.getvalue().encode()), 65000)

    def test_revoked_client_changes_actual_denominators_and_short_health_failure_is_retained(self):
        from tests.test_cifar_mechanism_runtime import MechanismRuntimeTests, execute, task_for
        task = task_for()
        with MechanismRuntimeTests.force_revocations(False):
            _, payload, _ = execute(task)
        report.validate_observations(payload, task)
        diagnostic = report.mechanism_diagnostics(payload, task)
        self.assertEqual(diagnostic["per_round"][21]["active_client_rounds"], 3)
        self.assertEqual(diagnostic["per_round"][22]["active_client_rounds"], 2)
        self.assertEqual(diagnostic["phases"][-1]["active_client_rounds"], 10)
        self.assertFalse(diagnostic["health"]["healthy"])
        self.assertIn("clean_false_revocation_rate", diagnostic["health"]["reasons"])
        self.assertEqual(diagnostic["revocations"]["paths"]["immediate_only"], 1)
        self.assertTrue(diagnostic["revocation_consistency"]["consistent"])

    def test_actual_early_all_revoked_is_complete_execution_with_unavailable_later_phases(self):
        from tests.test_cifar_mechanism_runtime import MechanismRuntimeTests, execute, task_for
        source_task = task_for()
        with MechanismRuntimeTests.force_revocations(True):
            _, payload, _ = execute(source_task)
        report.validate_observations(payload, source_task)
        for task in self.tasks:
            task["config"] = deepcopy(source_task["config"])
            actual = deepcopy(payload)
            actual.update(task_id=task["task_id"], task_fingerprint=task["fingerprint"])
            self.artifacts[task["task_id"]].update(observations=actual, completed_training_rounds=22)
        value = report.summarize(self.output)
        self.assertEqual((value["status"], value["complete_tasks"], value["thirty_round_tasks"]), ("complete", 4, 0), value)
        self.assertEqual(value["decision"]["conclusion"], "completed_early_health_failure_without_full_thirty_round_comparison")
        phase = value["rows"][0]["diagnostics"]["phases"][-1]
        self.assertFalse(phase["complete"])
        self.assertEqual(phase["available_rounds"], 0)
        self.assertIsNone(phase["history_admission_rate"])
        self.assertIn("all_honest_revoked", value["rows"][0]["diagnostics"]["health"]["reasons"])

    def test_nonfinite_updates_are_completed_unhealthy_with_zero_actual_detector_denominator(self):
        from sm9rrsfl import torch_backend as backend
        from tests.test_cifar_mechanism_runtime import execute, task_for
        original = backend.TorchTrainingContext.local_train_delta_resident
        def nonfinite(context, *args, **kwargs):
            delta, stats = original(context, *args, **kwargs)
            delta[0] = float("nan")
            return delta, stats
        source_task = task_for()
        with mock.patch.object(backend.TorchTrainingContext, "local_train_delta_resident", new=nonfinite):
            _, payload, _ = execute(source_task)
        report.validate_observations(payload, source_task)
        forged = deepcopy(payload)
        for row in forged["rounds"]:
            row["record"]["nonfinite_updates"] = 0
        for cp in forged["checkpoints"]:
            cp["record"]["nonfinite_updates"] = 0
        forged["terminal"]["nonfinite_updates"] = 0
        with self.assertRaisesRegex(ValueError, "nonfinite count differs"):
            report.validate_observations(forged, source_task)
        for task in self.tasks:
            task["config"] = deepcopy(source_task["config"])
            actual = deepcopy(payload)
            actual.update(task_id=task["task_id"], task_fingerprint=task["fingerprint"])
            self.artifacts[task["task_id"]].update(observations=actual)
        value = report.summarize(self.output)
        self.assertEqual((value["status"], value["complete_tasks"]), ("complete", 4), value)
        diagnostic = value["rows"][0]["diagnostics"]
        self.assertFalse(diagnostic["health"]["healthy"])
        self.assertEqual(diagnostic["health"]["nonfinite_updates"], 90)
        self.assertIn("nonfinite_updates", diagnostic["health"]["reasons"])
        for row in diagnostic["per_round"]:
            self.assertEqual(row["active_client_rounds"], 3)
            self.assertEqual(row["observed_verified_finite_updates"], 0)
            self.assertEqual(row["unobserved_active_client_rounds"], 3)
            self.assertIsNone(row["history_admission_rate"])
            self.assertEqual(row["actual_commit_calls"], 0)

    def test_one_actual_nonfinite_update_cannot_be_hidden_by_consistent_forged_counters(self):
        from sm9rrsfl import torch_backend as backend
        from tests.test_cifar_mechanism_runtime import execute, task_for
        original = backend.TorchTrainingContext.local_train_delta_resident
        calls = []
        def one_nonfinite(context, *args, **kwargs):
            delta, stats = original(context, *args, **kwargs)
            if not calls:
                delta[0] = float("nan")
            calls.append(1)
            return delta, stats
        task = task_for()
        with mock.patch.object(backend.TorchTrainingContext, "local_train_delta_resident", new=one_nonfinite):
            _, payload, _ = execute(task)
        report.validate_observations(payload, task)
        self.assertEqual(payload["terminal"]["nonfinite_updates"], 1)
        self.assertEqual(report.mechanism_diagnostics(payload, task)["per_round"][0]["observed_verified_finite_updates"], 2)
        payload["rounds"][0]["record"]["nonfinite_updates"] = 0
        payload["checkpoints"][1]["record"]["nonfinite_updates"] = 0
        payload["terminal"]["nonfinite_updates"] = 0
        with self.assertRaisesRegex(ValueError, "nonfinite count differs"):
            report.validate_observations(payload, task)

    def test_finite_updates_require_actual_aggregation_and_matching_vector_dtype(self):
        payload = deepcopy(self.observed["H0"])
        payload["rounds"][0].update(aggregation_executed=False, aggregate=None, aggregate_order=[])
        with self.assertRaisesRegex(ValueError, "aggregation execution differs"):
            report.validate_observations(payload, self.base_tasks["H0"])
        payload = deepcopy(self.observed["H0"])
        payload["rounds"][0]["aggregate"]["dtype"] = "<f8"
        with self.assertRaisesRegex(ValueError, "aggregate shape/dtype"):
            report.validate_observations(payload, self.base_tasks["H0"])

    def test_pruned_or_numeric_boolean_detector_decisions_are_rejected(self):
        for field, replacement in (("norm_score", None), ("would_flag", 0)):
            with self.subTest(field=field):
                payload = deepcopy(self.observed["H0"])
                event = payload["history_events"][0]
                decisions = [event["decision"], event["after_evaluate"]["pending"]["decision"],
                             event["commit"]["before"]["pending"]["decision"]]
                for decision in decisions:
                    if replacement is None:
                        decision.pop(field, None)
                    else:
                        decision[field] = replacement
                with self.assertRaisesRegex(ValueError, "detector decision"):
                    report.validate_observations(payload, self.base_tasks["H0"])

    def test_absent_all_task_evidence_does_not_claim_health_assessed(self):
        self.artifacts.clear()
        value = report.summarize(self.output)
        self.assertFalse(value["decision"]["short_clean_health_assessed"])
        self.assertEqual(value["complete_tasks"], 0)

    def test_first_difference_respects_all_detector_evaluations_before_commits(self):
        a, b = deepcopy(self.observed["H0"]), deepcopy(self.observed["H0"])
        a["history_events"][72]["commit"]["after"]["history"]["sha256"] = "0" * 64
        a["history_events"][73]["decision"]["norm_score"] += 1
        pair = report.compare_observations(a, b)
        self.assertEqual(pair["first_divergence"]["stage"], "detector.decision")
        self.assertEqual(pair["first_divergence"]["client_id"], "client-1")
        a["rounds"][24]["coefficients"]["by_client"]["client-0"] = .1
        a["history_events"][73]["decision"]["norm_score"] -= 1
        self.assertEqual(report.compare_observations(a, b)["first_divergence"]["stage"], "coefficients")

    def test_aggregate_order_is_after_coefficients_and_blacklist_before_is_before_training(self):
        a, b = deepcopy(self.observed["H0"]), deepcopy(self.observed["H0"])
        a["rounds"][24]["aggregate_order"].reverse()
        a["rounds"][24]["coefficients"]["by_client"]["client-0"] = .1
        self.assertEqual(report.compare_observations(a, b)["first_divergence"]["stage"], "coefficients")
        a["rounds"][24]["blacklisted_before"] = ["client-0"]
        a["rounds"][24]["clients"][0]["model_input"]["sha256"] = "0" * 64
        self.assertEqual(report.compare_observations(a, b)["first_divergence"]["stage"], "blacklisted_before")
        a["rounds"][24]["blacklisted_before"] = []
        a["rounds"][24]["clients"][0]["model_input"] = deepcopy(b["rounds"][24]["clients"][0]["model_input"])
        a["rounds"][24]["blacklisted_after"] = ["client-0"]
        self.assertEqual(report.compare_observations(a, b)["first_divergence"]["stage"], "blacklisted_after")

    def test_wrong_declared_round_count_environment_or_physical_gpu_is_invalid(self):
        original = deepcopy(self.artifacts["H0_2"])
        changes = [("completed_training_rounds", True), ("requested_training_rounds", True),
                   ("requested_training_rounds", 29), ("gpu_uuid", "GPU-other"), ("wall_seconds", float("nan"))]
        for field, changed in changes:
            with self.subTest(field=field, changed=changed):
                self.artifacts["H0_2"] = deepcopy(original)
                self.artifacts["H0_2"][field] = changed
                self.assertEqual(report.summarize(self.output)["complete_tasks"], 3)
        self.artifacts["H0_2"] = deepcopy(original)
        self.artifacts["H0_2"]["environment"]["torch"]["cudnn_allow_tf32"] = not self.metadata["torch"]["cudnn_allow_tf32"]
        self.assertEqual(report.summarize(self.output)["complete_tasks"], 3)

    def test_file_evidence_stays_unchanged_and_concurrent_edit_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for task in self.tasks:
                (root / (task["task_id"] + ".json")).write_text(json.dumps(self.artifacts[task["task_id"]]))
            def evidence(output):
                return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.json")}
            def load(output, task):
                return json.loads((root / (task["task_id"] + ".json")).read_text())
            self.protocol.evidence_hashes.side_effect = evidence
            self.protocol.load_completed.side_effect = load
            before = evidence(root)
            self.assertEqual(report.summarize(root)["status"], "complete")
            self.assertEqual(before, evidence(root))
            def race(output, task):
                value = load(output, task)
                p = root / (task["task_id"] + ".json")
                p.write_text(p.read_text() + "\n")
                return value
            self.protocol.load_completed.side_effect = race
            self.assertEqual(report.summarize(root)["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
