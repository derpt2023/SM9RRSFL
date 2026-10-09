"""Independent per-target inference and immutable, complete policy evidence."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import cifar_tail_determinism_report as report
import tests.test_cifar_tail_determinism_runtime as runtime_tests


class TailReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = runtime_tests.TailDeterminismRuntimeTests
        cls.fixture.setUpClass()

    def setUp(self):
        self.output = Path("/nonexistent/readonly-tail-report")
        self.uuid = "GPU-synthetic-physical"
        profile = self.fixture.payloads["original"]["numerical_policy"]["baseline"]
        self.metadata = {"actual_compute_device": {"name": "synthetic GPU", "uuid": self.uuid, "logical_device": "cuda:0"},
            "environment": {"CUDA_VISIBLE_DEVICES": self.uuid},
            "nvidia": {"gpus": [{"name": "synthetic GPU", "uuid": self.uuid, "driver_version": "test"}]},
            "torch": {"version": "test", **{k: profile[k] for k in report.ENVIRONMENT_FLAGS},
                "logical_cuda_devices": [{"logical_index": 0, "name": "synthetic GPU", "uuid": self.uuid}]}}
        self.tasks, self.artifacts, self.arrays = [], {}, {}
        for repeat in (1, 2, 3):
            for policy in report.POLICIES:
                task = deepcopy(self.fixture.tasks[policy])
                task.update(task_id=policy + "_r" + str(repeat), repeat=repeat,
                    fingerprint=hashlib.sha256((policy + str(repeat)).encode()).hexdigest())
                self.tasks.append(task)
                payload = deepcopy(self.fixture.payloads[policy])
                for item in (payload, payload["prefix"]):
                    item.update(task_id=task["task_id"], task_fingerprint=task["fingerprint"])
                self.arrays[task["task_id"]] = {k: a.copy() for k, a in self.fixture.arrays[policy].items()}
                self.artifacts[task["task_id"]] = {"status": "complete", "task_fingerprint": task["fingerprint"],
                    "artifact_fingerprint": hashlib.sha256((task["task_id"] + "artifact").encode()).hexdigest(),
                    "gpu_uuid": self.uuid, "environment": deepcopy(self.metadata), "observations": payload,
                    "fresh_start": True, "checkpoints_used": False, "training_round_completed": False, "wall_seconds": .25}
        original = deepcopy(self.fixture.payloads["original"])
        self.manifest = {"fingerprint": "manifest", "same_gpu_uuid": self.uuid,
            "source_sha256": {"frozen_science.py": "a" * 64}, "reference": {
                "execution_environment": report.step.matched.normalized_environment(self.metadata),
                "anchors": [{"observations": deepcopy(original["prefix"])} for _ in range(3)],
                "step_anchors": [{"observations": deepcopy(original)} for _ in range(3)]}}
        self.protocol = SimpleNamespace(
            read_study=mock.Mock(return_value=(self.manifest, self.tasks)),
            load_completed=mock.Mock(side_effect=lambda output, task: self.artifacts.get(task["task_id"])),
            load_arrays=mock.Mock(side_effect=lambda output, task, artifact: self.arrays[task["task_id"]]),
            source_hashes=mock.Mock(return_value=self.manifest["source_sha256"]),
            evidence_hashes=mock.Mock(return_value={"evidence": "stable"}), verify_reference=mock.Mock())
        patcher = mock.patch.dict(sys.modules, {"cifar_tail_determinism_protocol": self.protocol})
        patcher.start()
        self.addCleanup(patcher.stop)

    def key(self, policy="original", repeat=2):
        return policy + "_r" + str(repeat)

    def payload(self, policy="original", repeat=2):
        return self.artifacts[self.key(policy, repeat)]["observations"]

    def alter(self, stage="gradients", *, policy="original", repeat=2, target_index=0, amount=.125):
        key = self.key(policy, repeat)
        payload = self.payload(policy, repeat)
        target = payload["targets"][target_index]
        array_key = target["tail_snapshots"][stage]
        array = self.arrays[key][array_key]
        array.flat[0] += np.float32(amount)
        fp = report.step.tensor_fingerprint(array)
        payload["snapshots"][array_key] = fp
        if stage == "final_delta":
            target["final_delta"] = fp
            payload["prefix"]["rounds"][0]["clients"][target_index]["update"] = fp
        elif stage == "loss":
            target["batches"][-1][stage] = {"value": float(array), "tensor": fp}
        else:
            target["batches"][-1][stage] = fp

    def reproduce(self, policy="original", targets=(0, 1)):
        for repeat in (2, 3):
            for target in targets:
                for stage in ("gradients", "post_parameters", "final_delta"):
                    self.alter(stage, policy=policy, repeat=repeat, target_index=target, amount=repeat * .125)

    def conclusions(self, value=None):
        value = report.summarize(self.output) if value is None else value
        return [client["conclusion"] for client in value["decision"]["clients"]]

    def test_actual_cpu_two_policies_pass_both_validators_and_exact_nine_pairs(self):
        for policy in report.POLICIES:
            report.validate_observations(self.fixture.payloads[policy], self.fixture.tasks[policy], self.fixture.arrays[policy])
        value = report.summarize(self.output)
        self.assertEqual((value["status"], value["complete_tasks"]), ("complete", 6))
        self.assertEqual(sum(p["kind"] == "within_policy" for p in value["pairs"]), 6)
        self.assertEqual(sum(p["kind"] == "cross_policy" for p in value["pairs"]), 3)
        self.assertEqual({(p["earlier_repeat"], p["later_repeat"]) for p in value["pairs"] if p["kind"] == "within_policy"}, set(report.step.PAIRS))
        self.assertEqual(self.conclusions(value), ["original_variability_not_reproduced_in_this_panel"] * 2)
        for field in ("health_assessed", "formal_qualification_assessed", "performance_improvement_assessed", "automatic_next_stage", "operator_cause_identified"):
            self.assertFalse(value["decision"][field])
        self.assertEqual(value["decision"]["completed_training_rounds"], 0)

    def test_original_three_pair_tail_differences_and_stable_intervention_support_only_current_scope(self):
        self.reproduce()
        value = report.summarize(self.output)
        self.assertEqual(self.conclusions(value), ["supports_suppression_of_observed_variability_in_this_scope"] * 2)
        self.assertTrue(all(c["original_three_pairs_reproduce_tail_gradient_first_difference"] for c in value["decision"]["clients"]))
        self.assertFalse(value["decision"]["operator_cause_identified"])

    def test_intervention_differences_are_insufficient_and_targets_are_independent(self):
        self.reproduce()
        self.reproduce("tail_cudnn_deterministic", targets=(1,))
        self.assertEqual(self.conclusions(), ["supports_suppression_of_observed_variability_in_this_scope",
            "scoped_policy_insufficient_for_observed_repeatability"])

    def test_original_no_difference_cannot_credit_a_different_intervention(self):
        self.reproduce("tail_cudnn_deterministic")
        self.assertEqual(self.conclusions(), ["original_variability_not_reproduced_in_this_panel"] * 2)

    def test_target84_changed_pre_boundary_does_not_erase_target19_result(self):
        self.reproduce()
        self.alter("logits", policy="tail_cudnn_deterministic", target_index=1)
        value = report.summarize(self.output)
        self.assertEqual(value["status"], "complete")
        self.assertEqual(self.conclusions(value), ["supports_suppression_of_observed_variability_in_this_scope",
            "context_mismatch_no_causal_interpretation"])
        self.assertTrue(value["decision"]["clients"][1]["context_review_reasons"])

    def test_old_target_pre_boundary_mismatch_blocks_that_target_only_and_old_delta_is_allowed(self):
        self.reproduce()
        anchor = self.manifest["reference"]["step_anchors"][0]["observations"]
        anchor["targets"][0]["final_delta"]["sha256"] = "b" * 64
        self.assertEqual(self.conclusions(), ["supports_suppression_of_observed_variability_in_this_scope"] * 2)
        anchor["targets"][1]["batches"][-1]["logits"]["sha256"] = "c" * 64
        self.assertEqual(self.conclusions(), ["supports_suppression_of_observed_variability_in_this_scope",
            "context_mismatch_no_causal_interpretation"])

    def test_shared_data_or_non_target_update_mismatch_blocks_both_but_keeps_measurements(self):
        self.reproduce()
        self.manifest["reference"]["anchors"][0]["observations"]["data"]["x_train"]["sha256"] = "d" * 64
        value = report.summarize(self.output)
        self.assertEqual(value["status"], "complete")
        self.assertEqual(self.conclusions(value), ["context_mismatch_no_causal_interpretation"] * 2)
        a, b = deepcopy(self.payload()), deepcopy(self.payload())
        a["targets"] = a["targets"][1:]
        b["targets"] = b["targets"][1:]
        a["prefix"]["rounds"][0]["clients"][0]["update"]["sha256"] = "f" * 64
        self.assertIn("client-0.non_target_update", report.context_differences(a, b))

    def test_non_tail_first_difference_has_no_magnitude_and_is_not_a_reproduced_tail(self):
        self.reproduce()
        self.payload()["targets"][0]["batches"][0]["gradients"]["sha256"] = "a" * 64
        value = report.summarize(self.output)
        pair = value["pairs"][0]
        self.assertEqual(pair["clients"][0]["first_divergence"]["batch"], 0)
        self.assertIsNone(pair["clients"][0]["first_divergence"]["magnitude"])
        self.assertFalse(value["decision"]["clients"][0]["original_three_pairs_reproduce_tail_gradient_first_difference"])
        self.assertEqual(self.conclusions(value)[0], "context_mismatch_no_causal_interpretation")

    def test_changed_pattern_after_backward_is_reviewed_without_invented_support(self):
        for repeat in (2, 3):
            self.alter("final_delta", repeat=repeat, amount=repeat * .125)
        value = report.summarize(self.output)
        self.assertEqual(self.conclusions(value)[0], "original_difference_pattern_changed_requires_review")

    def test_missing_or_invalid_policy_is_not_complete_or_equal(self):
        original = deepcopy(self.payload("tail_cudnn_deterministic"))
        alterations = [lambda p: p.pop("numerical_policy"),
            lambda p: p["numerical_policy"]["events"].pop(),
            lambda p: p["numerical_policy"]["events"][0]["effective"].update(cudnn_allow_tf32=False),
            lambda p: p["numerical_policy"]["events"][0]["restored"].update(cudnn_deterministic=True),
            lambda p: p["numerical_policy"]["checks"].update(forward_after=999),
            lambda p: p["numerical_policy"].update(policy="original")]
        for alter in alterations:
            with self.subTest(alter=alter):
                self.artifacts[self.key("tail_cudnn_deterministic")]["observations"] = deepcopy(original)
                alter(self.payload("tail_cudnn_deterministic"))
                value = report.summarize(self.output)
                self.assertEqual(value["complete_tasks"], 5)
                self.assertEqual(self.conclusions(value), ["incomplete_or_invalid_evidence"] * 2)

    def test_environment_and_observer_baseline_are_cross_checked(self):
        payload = self.payload()
        metadata = deepcopy(self.metadata)
        metadata["torch"]["cudnn_allow_tf32"] = not metadata["torch"]["cudnn_allow_tf32"]
        with self.assertRaisesRegex(ValueError, "baseline differs"):
            report.validate_environment_policy(payload, metadata)
        metadata = deepcopy(self.metadata)
        metadata["torch"].pop("deterministic_warn_only")
        with self.assertRaisesRegex(ValueError, "missing"):
            report.validate_environment_policy(payload, metadata)
        metadata = deepcopy(self.metadata)
        metadata["torch"]["cudnn_deterministic"] = 0
        with self.assertRaisesRegex(ValueError, "baseline differs"):
            report.validate_environment_policy(payload, metadata)
        self.artifacts[self.key()]["environment"]["torch"]["version"] = "other"
        self.assertEqual(report.summarize(self.output)["complete_tasks"], 5)

    def test_new_baseline_flag_not_in_old_collector_cannot_silently_differ(self):
        self.reproduce()
        policy = self.payload("tail_cudnn_deterministic")["numerical_policy"]
        for profile in [policy["baseline"], policy["exit_profile"],
                        *[event[phase] for event in policy["events"] for phase in ("before", "effective", "restored")]]:
            profile["cudnn_benchmark_limit"] = 987
        value = report.summarize(self.output)
        self.assertEqual(value["status"], "complete")
        self.assertEqual(self.conclusions(value), ["context_mismatch_no_causal_interpretation"] * 2)
        self.assertTrue(any("numerical_policy.baseline" in p["context_mismatches"] for p in value["pairs"]))

    def test_missing_or_contradictory_completed_round_flag_is_invalid(self):
        self.artifacts[self.key()]["training_round_completed"] = True
        self.assertEqual(report.summarize(self.output)["complete_tasks"], 5)
        self.artifacts[self.key()].pop("training_round_completed")
        self.assertEqual(report.summarize(self.output)["complete_tasks"], 5)

    def test_bad_array_incomplete_task_or_identity_never_becomes_equality(self):
        original = deepcopy(self.artifacts[self.key()])
        self.artifacts.pop(self.key())
        self.assertEqual(report.summarize(self.output)["complete_tasks"], 5)
        for field, value in (("fresh_start", False), ("checkpoints_used", True), ("gpu_uuid", "other"), ("wall_seconds", float("nan")), ("task_fingerprint", "other")):
            self.artifacts[self.key()] = deepcopy(original)
            self.artifacts[self.key()][field] = value
            self.assertEqual(report.summarize(self.output)["complete_tasks"], 5)
        self.artifacts[self.key()] = original
        key = self.payload()["targets"][0]["tail_snapshots"]["gradients"]
        self.arrays[self.key()][key].flat[0] = np.nan
        self.assertEqual(report.summarize(self.output)["complete_tasks"], 5)

    def test_sources_reference_and_read_race_invalidate_whole_output(self):
        self.protocol.evidence_hashes.side_effect = [{"file": "before"}, {"file": "after"}]
        value = report.summarize(self.output)
        self.assertEqual(value["status"], "invalid")
        self.assertNotIn("pairs", value)
        self.protocol.evidence_hashes.side_effect = None
        self.protocol.source_hashes.side_effect = [{"source": "before"}, {"source": "after"}]
        self.assertEqual(report.summarize(self.output)["status"], "invalid")
        self.protocol.source_hashes.side_effect = None
        self.protocol.verify_reference.side_effect = ValueError("reference changed")
        self.assertEqual(report.summarize(self.output)["status"], "invalid")

    def test_optional_installed_source_is_deduplicated_and_never_binary_proof(self):
        source = {"torch_version": "2.3.0a0-custom", "torch_git_version": "a" * 40,
            "source_root": "/native/path/not/printed", "files": {
                "tools/autograd/derivatives.yaml": {"status": "unavailable", "path": "/native/path"},
                "aten/src/ATen/native/Convolution.cpp": {"status": "read", "sha256": "b" * 64,
                    "pattern_checks": {"backward_reads_current_deterministic_flags": True}}}}
        for artifact in self.artifacts.values():
            artifact["implementation_evidence"] = deepcopy(source)
        value = report.summarize(self.output)
        self.assertEqual(value["status"], "complete")
        self.assertEqual(len(value["implementation_evidence"]), 1)
        evidence = next(iter(value["implementation_evidence"].values()))
        self.assertFalse(evidence["binary_equivalence_verified"])
        self.assertFalse(evidence["executed_algorithm_identified"])
        self.assertEqual(evidence["files"]["tools/autograd/derivatives.yaml"]["status"], "unavailable")
        self.assertEqual(len({row["implementation_evidence_id"] for row in value["rows"]}), 1)
        self.assertNotIn("/native/path", json.dumps(value))

    def test_serialized_complete_json_and_npz_paths_are_read_only_and_race_checked(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for task in self.tasks:
                key = task["task_id"]
                folder = root / key
                folder.mkdir()
                (folder / "completed.json").write_text(json.dumps(self.artifacts[key]), encoding="utf8")
                np.savez(folder / "snapshots.npz", **self.arrays[key])
            def read_arrays(output, task, artifact):
                with np.load(root / task["task_id"] / "snapshots.npz", allow_pickle=False) as values:
                    return {k: values[k] for k in values.files}
            def evidence(output):
                return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in root.rglob("*") if path.is_file()}
            self.protocol.load_completed.side_effect = lambda output, task: json.loads((root / task["task_id"] / "completed.json").read_text())
            self.protocol.load_arrays.side_effect = read_arrays
            self.protocol.evidence_hashes.side_effect = evidence
            before = evidence(root)
            value = report.summarize(root)
            self.assertEqual(value["status"], "complete")
            self.assertEqual(evidence(root), before)
            def raced(*args):
                arrays = read_arrays(*args)
                path = root / self.tasks[0]["task_id"] / "completed.json"
                path.write_text(path.read_text() + "\n")
                return arrays
            self.protocol.load_arrays.side_effect = raced
            self.assertEqual(report.summarize(root)["status"], "invalid")

    def test_readonly_compact_panel_keeps_six_tasks_nine_pairs_and_nonzero_blocks_under_40kb(self):
        self.reproduce()
        with mock.patch("builtins.open", side_effect=AssertionError("unexpected filesystem access")), \
                mock.patch.object(report.step.prefix_report.timing.base, "load_split", side_effect=AssertionError("data")), \
                mock.patch.object(report.step.prefix_report.timing.base.fl, "run_experiment", side_effect=AssertionError("training")):
            value = report.summarize(self.output)
        output = io.StringIO()
        with redirect_stdout(output):
            report.print_summary(value)
        lines = output.getvalue().splitlines()
        self.assertEqual((lines[0], lines[-1]), ("=== CIFAR_TAIL_DETERMINISM_BEGIN ===", "=== CIFAR_TAIL_DETERMINISM_END ==="))
        rows = [json.loads(line) for line in lines[1:-1]]
        self.assertEqual((sum(r["type"] == "task" for r in rows), sum(r["type"] == "pair" for r in rows)), (6, 9))
        self.assertEqual(len(rows), 18)
        self.assertLess(len(output.getvalue().encode()), 40000)
        self.assertNotIn("task_tag", output.getvalue())
        self.assertNotIn("CUDA_VISIBLE_DEVICES", output.getvalue())
        task = next(r for r in rows if r["type"] == "task" and r["policy"] == "tail_cudnn_deterministic")
        self.assertEqual(task["numerical_policy"]["events"][0][5:8], [False, True, False])
        pair = next(r for r in rows if r["type"] == "pair")
        self.assertTrue(pair["tail_comparisons"][0]["numerically_changed_parameters"])


if __name__ == "__main__":
    unittest.main()
