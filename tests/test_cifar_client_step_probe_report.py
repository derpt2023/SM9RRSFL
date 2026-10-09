"""Local-boundary evidence, actual retained-array distances and read-only failure."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import cifar_client_step_probe_report as report
import tests.test_cifar_client_step_probe_runtime as runtime_tests


class ClientStepReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = runtime_tests.ClientStepProbeRuntimeTests
        fixture.setUpClass()
        cls.original_payload = deepcopy(fixture.payload)
        cls.original_arrays = {key: value.copy() for key, value in fixture.snapshots.items()}
        cls.original_task = deepcopy(fixture.task)

    def setUp(self):
        self.output = Path("/nonexistent/read-only-step-test")
        self.uuid = "GPU-synthetic-physical"
        self.metadata = {"actual_compute_device": {"name": "synthetic GPU", "uuid": self.uuid, "logical_device": "cuda:0"},
            "environment": {"CUDA_VISIBLE_DEVICES": self.uuid}, "nvidia": {"gpus": [{"name": "synthetic GPU", "uuid": self.uuid, "driver_version": "test"}]},
            "torch": {"version": "test", "logical_cuda_devices": [{"logical_index": 0, "name": "synthetic GPU", "uuid": self.uuid}]}}
        self.tasks, self.artifacts, self.arrays = [], {}, {}
        for repeat in (1, 2, 3):
            task = deepcopy(self.original_task)
            task.update(task_id="step_repeat" + str(repeat), fingerprint=str(repeat) * 64, repeat=repeat)
            self.tasks.append(task)
            payload = deepcopy(self.original_payload)
            for item in (payload, payload["prefix"]):
                item.update(task_id=task["task_id"], task_fingerprint=task["fingerprint"])
            self.arrays[task["task_id"]] = {key: value.copy() for key, value in self.original_arrays.items()}
            self.artifacts[task["task_id"]] = {"status": "complete", "task_fingerprint": task["fingerprint"],
                "artifact_fingerprint": str(repeat + 3) * 64, "gpu_uuid": self.uuid,
                "environment": deepcopy(self.metadata), "observations": payload,
                "fresh_start": True, "checkpoints_used": False, "wall_seconds": 1.25}
        self.manifest = {"fingerprint": "manifest", "same_gpu_uuid": self.uuid,
            "source_sha256": {"original.py": "a" * 64}, "reference": {
                "execution_environment": report.matched.normalized_environment(self.metadata),
                "anchors": [{"observations": deepcopy(self.original_payload["prefix"])} for _ in range(3)]}}
        self.protocol = SimpleNamespace(
            read_study=mock.Mock(return_value=(self.manifest, self.tasks)),
            load_completed=mock.Mock(side_effect=lambda output, task: self.artifacts.get(task["task_id"])),
            load_arrays=mock.Mock(side_effect=lambda output, task, artifact: self.arrays[task["task_id"]]),
            source_hashes=mock.Mock(return_value=self.manifest["source_sha256"]),
            evidence_hashes=mock.Mock(return_value={"synthetic-complete": "stable"}),
            verify_reference=mock.Mock())
        patcher = mock.patch.dict(sys.modules, {"cifar_client_step_probe_protocol": self.protocol})
        patcher.start()
        self.addCleanup(patcher.stop)

    def payload(self, repeat=2):
        return self.artifacts["step_repeat" + str(repeat)]["observations"]

    def validate(self, repeat=2):
        key = "step_repeat" + str(repeat)
        return report.validate_observations(self.payload(repeat), self.tasks[repeat - 1], self.arrays[key])

    def changed_tail_array(self, stage, *, repeat=2, target_index=0):
        payload = self.payload(repeat)
        target = payload["targets"][target_index]
        key = target["tail_snapshots"][stage]
        value = self.arrays["step_repeat" + str(repeat)][key]
        value.flat[0] += np.float32(.125)
        fp = report.tensor_fingerprint(value)
        payload["snapshots"][key] = fp
        if stage == "final_delta":
            target["final_delta"] = fp
            payload["prefix"]["rounds"][0]["clients"][target_index]["update"] = fp
        elif stage == "loss":
            target["batches"][-1][stage] = {"value": float(value), "tensor": fp}
        else:
            target["batches"][-1][stage] = fp
        return value

    def pair(self, summary, earlier=1, later=2):
        return next(p for p in summary["pairs"] if (p["earlier_repeat"], p["later_repeat"]) == (earlier, later))

    def test_real_runtime_payload_is_accepted_and_three_exact_pairs_are_available(self):
        self.validate()
        summary = report.summarize(self.output)
        self.assertEqual((summary["status"], summary["complete_tasks"]), ("complete", 3))
        self.assertTrue(summary["all_pairs_equal"])
        self.assertTrue(summary["all_historical_contexts_reproduced"])
        self.assertEqual({(p["earlier_repeat"], p["later_repeat"]) for p in summary["pairs"]}, {(1, 2), (1, 3), (2, 3)})
        self.assertEqual(summary["decision"]["completed_training_rounds"], 0)
        for key in ("health_assessed", "formal_qualification_assessed", "automatic_next_stage", "cuda_cause_identified", "root_cause_identified"):
            self.assertFalse(summary["decision"][key])

    def test_actual_ten_parameter_cifar_forward_payload_passes_strict_validation(self):
        fixture = runtime_tests.ClientStepProbeRuntimeTests
        spec = runtime_tests.ModelSpec(input_shape=(1, 8, 8), num_classes=10, architecture="cifar10",
            cifar_conv_filters=(2, 3), cifar_hidden_dims=(7, 5))
        with mock.patch.object(runtime_tests.fl, "model_spec_for_dataset", return_value=spec):
            with runtime_tests.runtime.observe(fixture.task) as observer:
                runtime_tests.fl.run_experiment(fixture.data, fixture.config, checkpoint_callback=observer.checkpoint)
        value = observer.finish()
        report.validate_observations(value, fixture.task, observer.snapshots)
        comparison = report.compare_observations(value, value, observer.snapshots, observer.snapshots)
        self.assertTrue(comparison["all_recorded_equal"])
        self.assertTrue(all(len(tail["per_parameter"]) == 10 for tail in comparison["tail_comparisons"]))

    def test_tail_gradient_difference_is_quantified_without_inventing_operator_cause(self):
        self.changed_tail_array("gradients")
        self.validate()
        summary = report.summarize(self.output)
        pair = self.pair(summary)
        first = pair["first_divergence"]
        self.assertEqual(first["stage"], "gradients")
        self.assertEqual(first["batch"], 1)
        self.assertEqual(first["magnitude_scope"], "saved last minibatch")
        self.assertGreater(first["magnitude"]["max_abs"], 0.)
        tail = pair["tail_comparisons"][0]
        self.assertGreater(tail["per_parameter"][0]["gradients"]["numerically_unequal_fraction"], 0.)
        self.assertTrue(all(p["gradients"]["max_abs"] == 0. for p in tail["per_parameter"][1:]))
        self.assertFalse(summary["decision"]["root_cause_identified"])

    def test_non_tail_first_difference_has_unknown_magnitude_even_when_tail_is_saved(self):
        batch = self.payload()["targets"][0]["batches"][0]
        batch["gradients"] = {**batch["gradients"], "sha256": "f" * 64}
        self.validate()
        first = self.pair(report.summarize(self.output))["first_divergence"]
        self.assertEqual((first["stage"], first["batch"]), ("gradients", 0))
        self.assertIsNone(first["magnitude"])
        self.assertIn("no full snapshot", first["magnitude_scope"])

    def test_actual_indices_and_logits_precede_gradients_and_equal_mean_loss_is_not_equal_update(self):
        self.changed_tail_array("logits")
        self.changed_tail_array("gradients")
        self.changed_tail_array("final_delta")
        target = self.payload()["targets"][0]
        labels_key = target["tail_snapshots"]["labels"]
        self.arrays["step_repeat2"][labels_key].flat[0] += 1
        labels_fp = report.tensor_fingerprint(self.arrays["step_repeat2"][labels_key])
        self.payload()["snapshots"][labels_key] = labels_fp
        target["batches"][-1]["labels"] = labels_fp
        self.validate()
        summary = report.summarize(self.output)
        self.assertEqual(self.pair(summary)["first_divergence"]["stage"], "logits")
        self.assertEqual(summary["rows"][0]["targets"][0]["mean_loss"], summary["rows"][1]["targets"][0]["mean_loss"])
        self.assertFalse(summary["all_pairs_equal"])
        first = self.payload()["targets"][0]["batches"][0]
        first["dataset_indices"] = {**first["dataset_indices"], "sha256": "e" * 64}
        self.assertEqual(self.pair(report.summarize(self.output))["first_divergence"]["stage"], "dataset_indices")

    def test_numerical_metrics_use_earlier_reference_and_handle_zero_norm_without_infinity(self):
        a, b = np.array([3., 4.], np.float32), np.array([0., 4.], np.float32)
        m = report.magnitude(a, b)
        self.assertEqual((m["max_abs"], m["relative_l2"], m["numerically_unequal_fraction"]), (3., .75, .5))
        self.assertIsNone(report.magnitude(a, np.zeros_like(a))["relative_l2"])
        self.assertEqual(report.magnitude(np.zeros_like(a), np.zeros_like(a))["relative_l2"], 0.)
        json.dumps(m, allow_nan=False)

    def test_snapshot_missing_corrupt_or_nonfinite_is_never_a_complete_pair(self):
        for problem in ("missing", "content", "nonfinite"):
            with self.subTest(problem=problem):
                originals = {k: a.copy() for k, a in self.arrays["step_repeat2"].items()}
                key = self.payload()["targets"][0]["tail_snapshots"]["gradients"]
                if problem == "missing":
                    self.arrays["step_repeat2"].pop(key)
                else:
                    self.arrays["step_repeat2"][key].flat[0] = np.nan if problem == "nonfinite" else 55.
                summary = report.summarize(self.output)
                self.assertEqual(summary["rows"][1]["status"], "invalid_evidence")
                self.assertIsNone(self.pair(summary)["all_recorded_equal"])
                self.arrays["step_repeat2"] = originals

    def test_incomplete_coverage_broken_chain_loss_and_stop_identity_are_rejected(self):
        original = deepcopy(self.payload())
        changes = (
            lambda p: p["targets"][0]["batches"].pop(),
            lambda p: p["prefix"]["rounds"][0]["clients"].pop(),
            lambda p: p["targets"][0]["batches"][1]["pre_parameters"].update(sha256="b" * 64),
            lambda p: p["targets"][0]["batches"][-1]["loss"].update(value=55.),
            lambda p: p["stop"].update(before_aggregation=False),
            lambda p: p["prefix"]["evaluations"][0].update(batches=[]),
            lambda p: p["targets"][0]["parameter_layout"][0].update(offset=1),
        )
        for change in changes:
            with self.subTest(change=change):
                self.artifacts["step_repeat2"]["observations"] = deepcopy(original)
                change(self.payload())
                self.assertEqual(report.summarize(self.output)["rows"][1]["status"], "invalid_evidence")

    def test_historical_input_context_mismatch_keeps_values_and_does_not_claim_cause(self):
        self.manifest["reference"]["anchors"][0]["observations"]["data"]["x_train"]["sha256"] = "c" * 64
        summary = report.summarize(self.output)
        self.assertEqual(summary["status"], "complete")
        self.assertFalse(summary["all_historical_contexts_reproduced"])
        self.assertIn("data", summary["rows"][0]["context_mismatches"])
        self.assertFalse(summary["decision"]["root_cause_identified"])

    def test_non_target_historical_update_mismatch_is_an_explicit_context_problem(self):
        payload = deepcopy(self.payload())
        payload["targets"] = payload["targets"][1:]
        payload["prefix"]["rounds"][0]["clients"][0]["update"]["sha256"] = "d" * 64
        context = report.historical_context(payload, self.manifest["reference"]["anchors"])
        self.assertFalse(context["historical_context_reproduced"])
        self.assertIn("client-0.non_target_update", context["context_mismatches"])

    def test_missing_artifact_or_bad_environment_and_freshness_are_not_equal(self):
        original = deepcopy(self.artifacts["step_repeat2"])
        self.artifacts.pop("step_repeat2")
        self.assertEqual(report.summarize(self.output)["complete_tasks"], 2)
        for field, value in (("gpu_uuid", "GPU-other"), ("fresh_start", False), ("checkpoints_used", True), ("wall_seconds", float("nan"))):
            self.artifacts["step_repeat2"] = deepcopy(original)
            self.artifacts["step_repeat2"][field] = value
            self.assertEqual(report.summarize(self.output)["rows"][1]["status"], "invalid_evidence")
        self.artifacts["step_repeat2"] = deepcopy(original)
        self.artifacts["step_repeat2"]["environment"]["torch"]["version"] = "other"
        self.assertIsNone(report.summarize(self.output)["all_pairs_equal"])

    def test_source_reference_or_read_race_invalidates_report_without_partial_success(self):
        self.protocol.evidence_hashes.side_effect = [{"file": "before"}, {"file": "after"}]
        value = report.summarize(self.output)
        self.assertEqual(value["status"], "invalid")
        self.assertNotIn("pairs", value)
        self.protocol.evidence_hashes.side_effect = None
        self.protocol.verify_reference.side_effect = ValueError("old prefix/reference changed")
        self.assertEqual(report.summarize(self.output)["status"], "invalid")

    def test_summary_is_readonly_and_compact_output_has_three_tasks_three_pairs(self):
        with mock.patch("builtins.open", side_effect=AssertionError("file access outside protocol")), \
                mock.patch.object(report.prefix_report.timing.base, "load_split", side_effect=AssertionError("dataset")), \
                mock.patch.object(report.prefix_report.timing.base.fl, "run_experiment", side_effect=AssertionError("training")):
            value = report.summarize(self.output)
        capture = io.StringIO()
        with redirect_stdout(capture):
            report.print_summary(value)
        output = capture.getvalue()
        lines = output.splitlines()
        self.assertEqual(lines[0], "=== CIFAR_CLIENT_STEP_PROBE_BEGIN ===")
        self.assertEqual(lines[-1], "=== CIFAR_CLIENT_STEP_PROBE_END ===")
        rows = [json.loads(line) for line in lines[1:-1]]
        self.assertEqual(sum(r["type"] == "task" for r in rows), 3)
        self.assertEqual(sum(r["type"] == "pair" for r in rows), 3)
        self.assertEqual(len(rows), 9)
        self.assertLess(len(output.encode()), 50000)
        self.assertNotIn("task_tag", output)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", output)


if __name__ == "__main__":
    unittest.main()
