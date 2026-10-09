"""Complete prefix evidence and first-round-only cross-policy conditions."""
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

import cifar_deterministic_prefix_report as report
import cifar_deterministic_prefix_runtime as runtime
import cifar_tail_determinism_runtime as old_runtime
from sm9rrsfl import fl
from sm9rrsfl.datasets import ImageDataset
from tests.test_cifar_client_step_probe_runtime import dataset


class DeterministicPrefixReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = dataset()
        cls.data = ImageDataset(raw.x_train[:-1], raw.y_train[:-1], raw.x_test, raw.y_test,
                                name=raw.name, num_classes=raw.num_classes)
        cls.config = fl.ExperimentConfig(method="sm9rrs", malicious_ratio=0., num_clients=3,
            rounds=3, partition="iid", seed=2201, compute_backend="torch", device="cpu",
            crypto_mode="simulated", detector_window=20, attack_start_round=25, attack_target_count=2,
            checkpoint_interval=1, early_stop=False, batch_size=5, local_epochs=1, sm9_workers=1)
        cls.base_task = {"task_id": "prefix-cpu", "fingerprint": "cpu-test", "config": asdict(cls.config)}
        cls.observed, cls.old = {}, {}
        for policy in report.POLICIES:
            task = {**deepcopy(cls.base_task), "policy": policy}
            with runtime.observe(task) as observer:
                result = fl.run_experiment(cls.data, cls.config, checkpoint_callback=observer.checkpoint)
            cls.observed[policy] = observer.finish(result)
            old_policy = "original" if policy == "original" else "tail_cudnn_deterministic"
            old_task = {**deepcopy(cls.base_task), "policy": old_policy, "target_clients": [0, 1], "stop_after_client": 1}
            with old_runtime.observe(old_task) as observer:
                fl.run_experiment(cls.data, cls.config, checkpoint_callback=observer.checkpoint)
            cls.old[old_policy] = observer.finish()

    def setUp(self):
        self.output = Path("/nonexistent/read-only-deterministic-prefix")
        self.uuid = "GPU-synthetic-physical"
        profile = self.observed["original"]["numerical_policy"]["baseline"]
        self.metadata = {"actual_compute_device": {"name": "synthetic GPU", "uuid": self.uuid, "logical_device": "cuda:0"},
            "environment": {"CUDA_VISIBLE_DEVICES": self.uuid},
            "nvidia": {"gpus": [{"name": "synthetic GPU", "uuid": self.uuid, "driver_version": "test"}]},
            "torch": {"version": "test", **{k: profile[k] for k in report.tail.ENVIRONMENT_FLAGS},
                "logical_cuda_devices": [{"logical_index": 0, "name": "synthetic GPU", "uuid": self.uuid}]}}
        self.tasks, self.artifacts = [], {}
        for repeat in (1, 2, 3):
            for policy in report.POLICIES:
                key = self.key(policy, repeat)
                task = {**deepcopy(self.base_task), "task_id": key, "policy": policy, "repeat": repeat,
                        "fingerprint": hashlib.sha256(key.encode()).hexdigest()}
                self.tasks.append(task)
                payload = deepcopy(self.observed[policy])
                payload.update(task_id=key, task_fingerprint=task["fingerprint"])
                self.artifacts[key] = {"status": "complete", "task_fingerprint": task["fingerprint"],
                    "artifact_fingerprint": hashlib.sha256((key + "artifact").encode()).hexdigest(),
                    "gpu_uuid": self.uuid, "environment": deepcopy(self.metadata), "observations": payload,
                    "fresh_start": True, "checkpoints_used": False, "completed_training_rounds": 3, "wall_seconds": .25}
        anchors = [{"task": {"task_id": "old_" + policy + str(repeat), "policy": policy, "repeat": repeat},
                    "observations": deepcopy(self.old[policy]), "artifact_fingerprint": "f" * 64}
                   for repeat in (1, 2, 3) for policy in ("original", "tail_cudnn_deterministic")]
        self.manifest = {"fingerprint": "manifest", "same_gpu_uuid": self.uuid,
            "source_sha256": {"frozen_science.py": "a" * 64}, "reference": {
                "execution_environment": report.prefix.matched.normalized_environment(self.metadata), "tail_anchors": anchors}}
        self.protocol = SimpleNamespace(
            read_study=mock.Mock(return_value=(self.manifest, self.tasks)),
            load_completed=mock.Mock(side_effect=lambda output, task: self.artifacts.get(task["task_id"])),
            source_hashes=mock.Mock(return_value=self.manifest["source_sha256"]),
            evidence_hashes=mock.Mock(return_value={"evidence": "stable"}), verify_reference=mock.Mock())
        patcher = mock.patch.dict(sys.modules, {"cifar_deterministic_prefix_protocol": self.protocol})
        patcher.start()
        self.addCleanup(patcher.stop)

    def key(self, policy="original", repeat=2):
        return policy + "_r" + str(repeat)

    def payload(self, policy="original", repeat=2):
        return self.artifacts[self.key(policy, repeat)]["observations"]

    def changed(self, fp, label):
        fp["sha256"] = hashlib.sha256(label.encode()).hexdigest()

    def reproduce(self, policy="original"):
        for repeat in (2, 3):
            payload = self.payload(policy, repeat)
            self.changed(payload["singleton_batches"][0]["gradients"], "grad" + str(repeat))
            self.changed(payload["rounds"][0]["clients"][0]["update"], "update" + str(repeat))

    def test_actual_full_cpu_payload_and_old_tail_intervention_are_compatible(self):
        for policy in report.POLICIES:
            task = {**deepcopy(self.base_task), "policy": policy}
            report.validate_observations(self.observed[policy], task)
            self.assertEqual(len(self.observed[policy]["singleton_batches"]), 6)
        value = report.summarize(self.output)
        self.assertEqual((value["status"], value["complete_tasks"], value["available_pairs"]), ("complete", 6, 9))
        self.assertTrue(all(p["equal"] for p in value["pairs"]))
        self.assertTrue(all(row["historical_context"]["round1_singleton_pre_reproduced"] for row in value["rows"]))
        self.assertTrue(all(row["historical_context"]["deterministic_effect_reproduced"] is True for row in value["rows"] if row["policy"] != "original"))
        self.assertEqual(value["decision"]["conclusion"], "scoped_policy_three_round_equal_control_variability_not_reproduced")
        for field in ("health_assessed", "performance_improvement_assessed", "formal_qualification_assessed", "automatic_next_stage", "operator_cause_identified"):
            self.assertFalse(value["decision"][field])

    def test_singleton_gradient_is_reported_before_its_resulting_client_update(self):
        self.reproduce()
        value = report.summarize(self.output)
        first = value["pairs"][0]["first_divergence"]
        self.assertEqual((first["round"], first["client_id"], first["batch"], first["stage"]), (1, "client-0", 1, "singleton.gradients"))
        self.assertIsNone(first["magnitude"])
        self.assertEqual(value["pairs"][0]["pipeline"]["first_divergence"]["stage"], "client_update")
        self.assertEqual(value["decision"]["original_pairs_different"], 3)
        self.assertEqual(value["decision"]["conclusion"], "supports_observed_three_round_reproducibility_for_scoped_policy")

    def test_earlier_non_singleton_client_update_takes_precedence(self):
        a, b = deepcopy(self.payload()), deepcopy(self.payload())
        # Client 1 singleton happens after client 0's resulting update.
        self.changed(a["singleton_batches"][1]["gradients"], "later-gradient")
        self.changed(a["rounds"][0]["clients"][0]["update"], "earlier-update")
        value = report.compare_observations(a, b)
        self.assertEqual((value["first_divergence"]["stage"], value["first_divergence"]["client_id"]), ("client_update", "client-0"))

    def test_post_intervention_round2_inputs_are_propagation_and_do_not_block_interpretation(self):
        self.reproduce()
        for repeat in (1, 2, 3):
            payload = self.payload(report.POLICIES[1], repeat)
            replacement = deepcopy(payload["rounds"][0]["post_model"])
            self.changed(replacement, "B-first-round-global-model")
            payload["rounds"][0]["post_model"] = deepcopy(replacement)
            payload["checkpoints"][1]["model"] = deepcopy(replacement)
            for evaluation in payload["evaluations"]:
                if evaluation["round"] == 1:
                    evaluation["model_input"] = deepcopy(replacement)
            for client in payload["rounds"][1]["clients"]:
                client["model_input"] = deepcopy(replacement)
        value = report.summarize(self.output)
        self.assertEqual(value["status"], "complete")
        self.assertEqual(value["decision"]["conclusion"], "supports_observed_three_round_reproducibility_for_scoped_policy")
        self.assertEqual(value["decision"]["context_review_reasons"], [])
        self.assertTrue(all(p["later_round_input_differences_are_propagation"] for p in value["pairs"]))

    def test_singleton_difference_without_pipeline_difference_still_prevents_B_repeatability(self):
        self.reproduce()
        payload = self.payload(report.POLICIES[1])
        self.changed(payload["singleton_batches"][4]["gradients"], "B-round3-singleton")
        value = report.summarize(self.output)
        pair = next(p for p in value["pairs"] if p["kind"] == "within_policy" and p["earlier_policy"] == report.POLICIES[1])
        self.assertTrue(pair["pipeline"]["equal"])
        self.assertFalse(pair["equal"])
        self.assertEqual(pair["first_divergence"]["round"], 3)
        self.assertEqual(value["decision"]["action"], "review_first_divergence_before_any_extension")

    def test_pipeline_diagnostic_difference_prevents_B_repeatability_even_with_same_models(self):
        self.reproduce()
        self.payload(report.POLICIES[1])["rounds"][2]["diagnostics"][0]["novelty_score"] += .1
        value = report.summarize(self.output)
        self.assertEqual(value["status"], "complete")
        self.assertFalse(value["decision"]["scoped_policy_all_three_round_observations_equal"])
        self.assertEqual(value["decision"]["conclusion"], "scoped_policy_did_not_reproduce_three_round_trajectory")

    def test_changed_r1_pre_or_historical_context_blocks_causal_interpretation(self):
        self.reproduce()
        self.changed(self.payload(report.POLICIES[1])["singleton_batches"][0]["logits"], "changed-forward")
        value = report.summarize(self.output)
        self.assertEqual(value["status"], "complete")
        self.assertEqual(value["decision"]["conclusion"], "context_mismatch_no_causal_interpretation")
        self.assertTrue(value["decision"]["context_review_reasons"])

    def test_old_random_target_deltas_are_not_required_and_old_B_effect_is_separate_diagnostic(self):
        self.reproduce()
        anchors = self.manifest["reference"]["tail_anchors"]
        for anchor in anchors:
            self.changed(anchor["observations"]["targets"][0]["final_delta"], "old-other-delta")
        value = report.summarize(self.output)
        self.assertEqual(value["status"], "complete")
        self.assertEqual(value["decision"]["conclusion"], "supports_observed_three_round_reproducibility_for_scoped_policy")
        self.assertTrue(value["decision"]["historical_deterministic_effect_requires_review"])

    def test_unobserved_earlier_target_minibatches_are_not_claimed_as_new_actual_evidence(self):
        self.reproduce()
        self.changed(self.manifest["reference"]["tail_anchors"][0]["observations"]["targets"][0]["batches"][0]["logits"], "old-earlier")
        value = report.summarize(self.output)
        self.assertTrue(value["rows"][0]["historical_context"]["round1_singleton_pre_reproduced"])
        self.assertIn("earlier target minibatches were not observed", value["rows"][0]["historical_context"]["scope"])

    def test_full_new_baseline_profile_is_checked_across_fresh_and_historical(self):
        self.reproduce()
        policy = self.payload(report.POLICIES[1])["numerical_policy"]
        for profile in [policy["baseline"], policy["exit_profile"],
                        *[event[phase] for event in policy["events"] for phase in ("before", "effective", "restored")]]:
            profile["cudnn_benchmark_limit"] = 999
        value = report.summarize(self.output)
        self.assertEqual(value["status"], "complete")
        self.assertEqual(value["decision"]["conclusion"], "context_mismatch_no_causal_interpretation")

    def test_missing_or_corrupt_policy_stage_and_wrong_class_shape_are_invalid(self):
        original = deepcopy(self.payload())
        edits = [lambda p: p["singleton_batches"].pop(),
            lambda p: p["actual_batches"].pop(),
            lambda p: p["numerical_policy"]["events"][0]["effective"].update(cudnn_deterministic=True),
            lambda p: p["numerical_policy"]["checks"].update(forward_before=0),
            lambda p: p["singleton_batches"][0]["logits"].update(shape=[1, 999]),
            lambda p: p["singleton_batches"][0]["gradients"].update(dtype="<f8"),
            lambda p: p["singleton_batches"][0]["loss"].update(value=float("nan")),
            lambda p: p["rounds"].pop()]
        for edit in edits:
            with self.subTest(edit=edit):
                self.artifacts[self.key()]["observations"] = deepcopy(original)
                edit(self.payload())
                value = report.summarize(self.output)
                self.assertEqual(value["complete_tasks"], 5)
                self.assertEqual(value["decision"]["conclusion"], "incomplete_or_invalid_evidence")

    def test_missing_invalid_artifact_environment_freshness_or_round_count_is_not_equal(self):
        original = deepcopy(self.artifacts[self.key()])
        self.artifacts.pop(self.key())
        self.assertEqual(report.summarize(self.output)["complete_tasks"], 5)
        for field, value in (("fresh_start", False), ("checkpoints_used", True), ("gpu_uuid", "other"),
                             ("completed_training_rounds", True), ("completed_training_rounds", 2), ("wall_seconds", float("nan"))):
            self.artifacts[self.key()] = deepcopy(original)
            self.artifacts[self.key()][field] = value
            self.assertEqual(report.summarize(self.output)["complete_tasks"], 5)
        self.artifacts[self.key()] = deepcopy(original)
        self.artifacts[self.key()].pop("completed_training_rounds")
        self.assertEqual(report.summarize(self.output)["complete_tasks"], 5)
        self.artifacts[self.key()] = deepcopy(original)
        self.artifacts[self.key()]["environment"]["torch"]["cudnn_allow_tf32"] = not self.metadata["torch"]["cudnn_allow_tf32"]
        self.assertEqual(report.summarize(self.output)["complete_tasks"], 5)

    def test_source_reference_and_read_race_remove_all_successful_pairs(self):
        self.protocol.evidence_hashes.side_effect = [{"file": "before"}, {"file": "after"}]
        value = report.summarize(self.output)
        self.assertEqual(value["status"], "invalid")
        self.assertNotIn("pairs", value)
        self.protocol.evidence_hashes.side_effect = None
        self.protocol.source_hashes.side_effect = [{"file": "before"}, {"file": "after"}]
        self.assertEqual(report.summarize(self.output)["status"], "invalid")
        self.protocol.source_hashes.side_effect = None
        self.protocol.verify_reference.side_effect = ValueError("old evidence changed")
        self.assertEqual(report.summarize(self.output)["status"], "invalid")

    def test_serialized_result_paths_are_read_only_and_change_during_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for task in self.tasks:
                (root / (task["task_id"] + ".json")).write_text(json.dumps(self.artifacts[task["task_id"]]))
            def evidence(output):
                return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.glob("*.json")}
            def load(output, task):
                return json.loads((root / (task["task_id"] + ".json")).read_text())
            self.protocol.load_completed.side_effect = load
            self.protocol.evidence_hashes.side_effect = evidence
            before = evidence(root)
            self.assertEqual(report.summarize(root)["status"], "complete")
            self.assertEqual(before, evidence(root))
            def raced(output, task):
                value = load(output, task)
                path = root / (task["task_id"] + ".json")
                path.write_text(path.read_text() + "\n")
                return value
            self.protocol.load_completed.side_effect = raced
            self.assertEqual(report.summarize(root)["status"], "invalid")

    def test_summary_is_readonly_and_compact_with_six_tasks_nine_pairs(self):
        self.reproduce()
        with mock.patch("builtins.open", side_effect=AssertionError("unexpected filesystem access")), \
                mock.patch.object(report.prefix.timing.base, "load_split", side_effect=AssertionError("dataset")), \
                mock.patch.object(fl, "run_experiment", side_effect=AssertionError("training")):
            value = report.summarize(self.output)
        output = io.StringIO()
        with redirect_stdout(output):
            report.print_summary(value)
        lines = output.getvalue().splitlines()
        self.assertEqual((lines[0], lines[-1]), ("=== CIFAR_DETERMINISTIC_PREFIX_BEGIN ===", "=== CIFAR_DETERMINISTIC_PREFIX_END ==="))
        rows = [json.loads(line) for line in lines[1:-1]]
        self.assertEqual((len(rows), sum(r["type"] == "task" for r in rows), sum(r["type"] == "pair" for r in rows)), (18, 6, 9))
        self.assertLess(len(output.getvalue().encode()), 50000)
        self.assertNotIn("task_tag", output.getvalue())
        self.assertNotIn("CUDA_VISIBLE_DEVICES", output.getvalue())
        task = next(row for row in rows if row["type"] == "task" and row["policy"] != "original")
        self.assertEqual(task["numerical_policy"]["events"][0][6:9], [False, True, False])
        self.assertEqual(len(task["trajectory"]), 4)


if __name__ == "__main__":
    unittest.main()
