"""Eighty-round CPU observations, old30 prefix comparisons and bounded reports."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import cifar_clean_horizon_report as report
import cifar_mechanism_report as old_report
from sm9rrsfl import fl
from sm9rrsfl.svd_detector import LongitudinalSVDDetector
from tests.test_cifar_mechanism_runtime import execute, task_for


class CleanHorizonReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks_by_arm, cls.observed, cls.old = {}, {}, {}
        for arm in report.ARMS:
            task = task_for(arm, rounds=80)
            _, payload, _ = execute(task)
            cls.tasks_by_arm[arm], cls.observed[arm] = task, payload
            oldtask = task_for(arm, rounds=30)
            _, oldpayload, _ = execute(oldtask)
            cls.old[arm] = {"task": oldtask, "observations": oldpayload, "artifact_fingerprint": "f" * 64}

    def setUp(self):
        self.output = Path("/nonexistent/clean-horizon-read-only")
        self.uuid = "GPU-test-physical"
        baseline = self.observed["H0"]["numerical_policy"]["baseline"]
        self.metadata = {"actual_compute_device": {"name": "synthetic GPU", "uuid": self.uuid, "logical_device": "cuda:0"},
            "environment": {"CUDA_VISIBLE_DEVICES": self.uuid},
            "nvidia": {"gpus": [{"name": "synthetic GPU", "uuid": self.uuid, "driver_version": "test"}]},
            "torch": {"version": "test", **{k: baseline[k] for k in report.tail.ENVIRONMENT_FLAGS},
                "logical_cuda_devices": [{"logical_index": 0, "name": "synthetic GPU", "uuid": self.uuid}]}}
        self.tasks, self.artifacts, self.anchors = [], {}, []
        for repeat in (1, 2):
            for arm in report.ARMS:
                task = deepcopy(self.tasks_by_arm[arm])
                task.update(task_id=arm + "_" + str(repeat), fingerprint=arm + "_" + str(repeat), repeat=repeat)
                self.tasks.append(task)
                payload = deepcopy(self.observed[arm])
                payload.update(task_id=task["task_id"], task_fingerprint=task["fingerprint"])
                self.artifacts[task["task_id"]] = {"status": "complete", "task_fingerprint": task["fingerprint"],
                    "artifact_fingerprint": hashlib.sha256(task["task_id"].encode()).hexdigest(),
                    "gpu_uuid": self.uuid, "environment": deepcopy(self.metadata), "observations": payload,
                    "fresh_start": True, "checkpoints_used": False, "completed_training_rounds": 80,
                    "requested_training_rounds": 80, "wall_seconds": .1}
                anchor = deepcopy(self.old[arm])
                anchor["task"].update(task_id="old_" + arm + "_" + str(repeat), repeat=repeat)
                self.anchors.append(anchor)
        self.manifest = {"fingerprint": "manifest", "same_gpu_uuid": self.uuid,
            "source_sha256": {"frozen_science.py": "a" * 64}, "reference": {
                "execution_environment": report.prefix.matched.normalized_environment(self.metadata)}}
        self.protocol = SimpleNamespace(read_study=mock.Mock(return_value=(self.manifest, self.tasks)),
            load_completed=mock.Mock(side_effect=lambda output, task: self.artifacts.get(task["task_id"])),
            source_hashes=mock.Mock(return_value=self.manifest["source_sha256"]),
            evidence_hashes=mock.Mock(return_value={"evidence": "stable"}), verify_reference=mock.Mock(),
            load_mechanism_anchors=mock.Mock(return_value=self.anchors))
        patcher = mock.patch.dict(sys.modules, {"cifar_clean_horizon_protocol": self.protocol})
        patcher.start()
        self.addCleanup(patcher.stop)

    def payload(self, arm="H0", repeat=2):
        return self.artifacts[arm + "_" + str(repeat)]["observations"]

    def test_actual80_cpu_full_matrix_reproduces_both_old30_same_arm_anchors(self):
        value = report.summarize(self.output)
        self.assertEqual((value["status"], value["complete_tasks"], value["full_horizon_tasks"], value["available_pairs"]), ("complete", 4, 4, 4), value)
        self.assertTrue(value["decision"]["full_horizon_coverage"])
        self.assertTrue(value["decision"]["prefix30_reproduced"])
        self.assertTrue(value["decision"]["within_arm_all_observations_equal"])
        self.assertTrue(value["decision"]["paired_preintervention_conditions_equal"])
        self.assertEqual(sum(len(row["historical_context"]["comparisons"]) for row in value["rows"]), 8)
        self.assertEqual(value["decision"]["requested_training_rounds"], 80)
        self.assertFalse(value["decision"]["h1_adoption_authorized"])
        self.assertEqual(old_report.ROUNDS, 30)
        self.assertEqual(report.ROUNDS, 80)
        for r in value["rows"]:
            self.assertEqual(len(r["trajectory"]), 81)
            self.assertEqual(len(r["diagnostics"]["per_round"]), 80)
            self.assertEqual([(p["start"], p["end"]) for p in r["diagnostics"]["phases"]][-2:], [(31, 51), (52, 80)])

    def test_same_arm_old30_divergence_preserves_completion_but_blocks_causal_claim(self):
        for anchor in self.anchors:
            if anchor["task"]["arm"] == "H0":
                anchor["observations"]["singleton_batches"][0]["gradients"]["sha256"] = "0" * 64
        value = report.summarize(self.output)
        self.assertEqual(value["status"], "complete")
        self.assertFalse(value["decision"]["prefix30_reproduced"])
        self.assertEqual(value["decision"]["conclusion"], "common_conditions_differ_no_paired_causal_interpretation")
        row = next(r for r in value["rows"] if r["arm"] == "H0")
        self.assertEqual(row["historical_context"]["comparisons"][0]["first_divergence"]["stage"], "singleton.gradients")

    def test_old30_scientific_config_exceptions_only_horizon_and_device(self):
        anchor = next(a for a in self.anchors if a["task"]["arm"] == "H0")
        anchor["task"]["config"]["device"] = "cuda:0"
        self.assertTrue(report.historical_context(self.observed["H0"], self.tasks_by_arm["H0"], self.anchors)["equal"])
        anchor["task"]["config"]["lr"] *= 2
        value = report.historical_context(self.observed["H0"], self.tasks_by_arm["H0"], self.anchors)
        self.assertFalse(value["equal"])
        self.assertIn("configuration differs", value["comparisons"][0]["comparison_error"])

    def test_late_singleton_repeat_divergence_and_round26_gate_remain_active(self):
        e = next(e for e in self.payload()["singleton_batches"] if e["round"] == 70)
        e["gradients"]["sha256"] = "0" * 64
        value = report.summarize(self.output)
        self.assertEqual(value["pairs"][0]["first_divergence"]["round"], 70)
        self.assertEqual(value["decision"]["conclusion"], "within_arm_repeat_divergence")
        for repeat in (1, 2):
            e = next(e for e in self.payload("H1", repeat)["singleton_batches"] if e["round"] == 26)
            e["gradients"]["sha256"] = "0" * 64
        value = report.summarize(self.output)
        self.assertFalse(value["decision"]["paired_preintervention_conditions_equal"])

    @staticmethod
    def force_at(round_id, all_clients):
        from dataclasses import replace
        original = LongitudinalSVDDetector.evaluate
        altered = []
        def evaluate(detector, tag, update, *, round_id=None, learning_rate=1.):
            decision = original(detector, tag, update, round_id=round_id, learning_rate=learning_rate)
            if round_id == target and (all_clients or not altered):
                altered.append(tag)
                decision = replace(decision, accepted=False, reason="strong_novelty", would_flag=True,
                    count_increment=True, history_eligible=False, immediate_revocation=True)
                state = detector._states[tag]
                state.pending = (*state.pending[:3], decision)
            return decision
        target = round_id
        return mock.patch.object(LongitudinalSVDDetector, "evaluate", new=evaluate)

    def test_late_revocation_retains_prefix30_and_actual_remaining_denominators(self):
        task = self.tasks_by_arm["H1"]
        with self.force_at(52, False):
            _, payload, _ = execute(task)
        report.validate_observations(payload, task)
        self.assertTrue(report.historical_context(payload, task, self.anchors)["equal"])
        diagnostic = report.mechanism_diagnostics(payload, task)
        self.assertEqual(diagnostic["per_round"][51]["active_client_rounds"], 3)
        self.assertEqual(diagnostic["per_round"][52]["active_client_rounds"], 2)
        phase = diagnostic["phases"][-1]
        self.assertEqual(phase["nominal_available_client_rounds"], 29 * 3)
        self.assertEqual(phase["permanently_excluded_client_rounds"], 28)
        self.assertEqual(phase["active_client_rounds"], 59)
        self.assertEqual(phase["unavailable_requested_rounds"], 0)
        self.assertFalse(diagnostic["health"]["healthy"])
        self.assertEqual(diagnostic["revocations"]["round_blocks"]["late_52_80"], 1)
        self.assertEqual(diagnostic["first_event_cases"]["first_late_revocation"]["event_round"], 52)
        for t in self.tasks:
            if t["arm"] == "H1":
                p = deepcopy(payload)
                p.update(task_id=t["task_id"], task_fingerprint=t["fingerprint"])
                self.artifacts[t["task_id"]]["observations"] = p
        value = report.summarize(self.output)
        self.assertEqual(value["status"], "complete")
        self.assertEqual(value["decision"]["conclusion"], "h1_bounded_clean_health_failure_do_not_adopt")
        self.assertFalse(value["decision"]["h1_observed_health_allows_further_review"])

    def test_all_revoked_early_has_missing_late_windows_not_fabricated_counts(self):
        task = self.tasks_by_arm["H0"]
        with self.force_at(22, True):
            _, payload, _ = execute(task)
        report.validate_observations(payload, task)
        diagnostic = report.mechanism_diagnostics(payload, task)
        self.assertFalse(diagnostic["health"]["healthy"])
        for phase in diagnostic["phases"][-2:]:
            self.assertEqual(phase["available_rounds"], 0)
            self.assertEqual(phase["unavailable_requested_rounds"], phase["requested_rounds"])
            self.assertEqual(phase["nominal_available_client_rounds"], 0)
            self.assertEqual(phase["permanently_excluded_client_rounds"], 0)
            self.assertFalse(phase["complete"])
            self.assertIsNone(phase["history_admission_rate"])
            self.assertEqual(phase["novelty_quantiles"]["n"], 0)
            self.assertIsNone(phase["novelty_quantiles"]["max"])
        self.assertFalse(report.historical_context(payload, task, self.anchors)["available"])

    def test_actual_nonfinite_updates_have_zero_detector_denominator_and_health_failure(self):
        from sm9rrsfl import torch_backend as backend
        original = backend.TorchTrainingContext.local_train_delta_resident
        def nonfinite(context, *args, **kwargs):
            delta, stats = original(context, *args, **kwargs)
            delta[0] = float("nan")
            return delta, stats
        task = self.tasks_by_arm["H0"]
        with mock.patch.object(backend.TorchTrainingContext, "local_train_delta_resident", new=nonfinite):
            _, payload, _ = execute(task)
        report.validate_observations(payload, task)
        diagnostic = report.mechanism_diagnostics(payload, task)
        self.assertEqual(diagnostic["health"]["nonfinite_updates"], 240)
        self.assertFalse(diagnostic["health"]["healthy"])
        self.assertEqual(diagnostic["phases"][-1]["observed_verified_finite_updates"], 0)
        self.assertIsNone(diagnostic["phases"][-1]["drift_quantiles"]["max"])
        payload["terminal"]["nonfinite_updates"] = 0
        with self.assertRaisesRegex(ValueError, "nonfinite count"):
            report.validate_observations(payload, task)

    def test_near_removal_margin_is_descriptive_and_threshold_equality_not_warning(self):
        payload = deepcopy(self.observed["H0"])
        # Diagnostics helper is pure; this isolated scalar fixture tests exact
        # descriptive boundaries, independent of authenticating full artifacts.
        d = payload["rounds"][51]["diagnostics"][0]
        limit = self.tasks_by_arm["H0"]["config"]["suspicion_remove_after"]
        d["count_after"] = limit - 1
        d["novelty_score"] = self.tasks_by_arm["H0"]["config"]["detector_distance_threshold"]
        value = report.mechanism_diagnostics(payload, self.tasks_by_arm["H0"])["per_round"][51]
        self.assertEqual(value["near_removal_observations"], 1)
        self.assertEqual(value["near_removal_count_margin_quantiles"]["max"], -1)
        self.assertEqual(value["triggers"]["warning_only"], 0)

    def test_evidence_race_missing_task_and_wrong_horizon_fail_closed(self):
        original = deepcopy(self.artifacts["H0_2"])
        self.artifacts["H0_2"]["requested_training_rounds"] = 30
        self.assertEqual(report.summarize(self.output)["complete_tasks"], 3)
        self.artifacts["H0_2"] = original
        self.protocol.evidence_hashes.side_effect = [{"e": "before"}, {"e": "after"}]
        self.assertEqual(report.summarize(self.output)["status"], "invalid")
        self.protocol.evidence_hashes.side_effect = None
        self.artifacts.clear()
        value = report.summarize(self.output)
        self.assertEqual(value["complete_tasks"], 0)
        self.assertFalse(value["decision"]["short_clean_health_assessed"])

    def test_summary_has11_finite_json_records_bounded_size_and_no_training(self):
        with mock.patch.object(report.prefix.timing.base, "load_split", side_effect=AssertionError("data")), \
             mock.patch.object(fl, "run_experiment", side_effect=AssertionError("training")):
            value = report.summarize(self.output)
        output = io.StringIO()
        with redirect_stdout(output):
            report.print_summary(value)
        lines = output.getvalue().splitlines()
        self.assertEqual((lines[0], lines[-1]), ("=== CIFAR_CLEAN_HORIZON_BEGIN ===", "=== CIFAR_CLEAN_HORIZON_END ==="))
        records = [json.loads(s) for s in lines[1:-1]]
        self.assertEqual(len(records), 11)
        self.assertEqual(sum(r["type"] == "task" for r in records), 4)
        self.assertLess(len(output.getvalue().encode()), 200000)
        self.assertNotIn('"task_tag"', output.getvalue())
        self.assertNotIn("CUDA_VISIBLE_DEVICES", output.getvalue())
        self.assertNotIn("full_thirty", output.getvalue())


if __name__ == "__main__":
    unittest.main()
