"""Serialized old90/prefix6/step3 provenance and six distinct scoped-policy tasks."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import cifar_tail_determinism_protocol as protocol
import tests.test_cifar_client_step_probe_protocol as prior
from tests.test_cifar_prefix_probe_report import fingerprint

step, report, base, runtime = protocol.step, protocol.step_report, protocol.base, protocol.runtime
GPU = prior.GPU


def synthetic_step_observations(task, prefix_reference):
    """Small numeric arrays with full 85-client/22-minibatch coverage, no training."""
    prefix = deepcopy(prefix_reference["anchors"][0]["observations"])
    prefix.update(task_id=task["task_id"], task_fingerprint=task["fingerprint"], configuration=deepcopy(task["config"]))
    prefix["rounds"] = [{"round": 1, "clients": prefix["rounds"][0]["clients"][:85]}]
    prefix["evaluations"] = [e for e in prefix["evaluations"] if e["round"] == 0]
    prefix["checkpoints"] = prefix["checkpoints"][:1]
    targets, arrays = [], {}
    layout = [{"name": name, "shape": [2], "size": 2, "offset": 2 * i}
              for i, name in enumerate(protocol.PARAMETER_NAMES)]
    for index in protocol.TARGET_CLIENTS:
        client = prefix["rounds"][0]["clients"][index]
        target = {k: deepcopy(client[k]) for k in ("client_id", "samples", "model_input", "training_seed",
            "learning_rate", "epochs", "batch_size", "stats")}
        target.update(parameter_layout=deepcopy(layout), batches=[], tail_snapshots={})
        count = len(client["minibatch_sizes_per_epoch"])
        last_pre = np.arange(20, dtype=np.float32) / 10
        grad = np.ones(20, dtype=np.float32)
        grad[[0, 2]] += np.float32(task["repeat"] * .125)
        post = last_pre - np.float32(.05) * grad
        delta = post - np.arange(20, dtype=np.float32)
        tails = {"local_indices": np.array([3], dtype=np.int64),
            "dataset_indices": np.array([19 + index], dtype=np.int64),
            "features": np.full((1, 3, 32, 32), .2, dtype=np.float32),
            "pre_parameters": last_pre, "logits": np.arange(10, dtype=np.float32).reshape(1, 10),
            "labels": np.array([index % 10], dtype=np.int64), "loss": np.array(.25, dtype=np.float32),
            "gradients": grad, "post_parameters": post, "final_delta": delta}
        previous = client["model_input"]
        for batch_index, samples in enumerate(client["minibatch_sizes_per_epoch"]):
            batch = {"batch": batch_index, "epoch": 0, "batch_in_epoch": batch_index, "samples": samples}
            for stage in report.STAGES:
                if batch_index == count - 1:
                    fp = report.tensor_fingerprint(tails[stage])
                elif stage == "pre_parameters":
                    fp = previous
                elif stage == "post_parameters" and batch_index == count - 2:
                    fp = report.tensor_fingerprint(last_pre)
                else:
                    shape = ([samples] if stage in ("local_indices", "dataset_indices", "labels") else
                        [samples, 3, 32, 32] if stage == "features" else [samples, 10] if stage == "logits" else
                        [] if stage == "loss" else [20])
                    fp = fingerprint(f"{index}/{batch_index}/{stage}", shape,
                        "<i8" if stage in ("local_indices", "dataset_indices", "labels") else "<f4")
                batch[stage] = {"value": .25, "tensor": fp} if stage == "loss" else fp
            previous = batch["post_parameters"]
            target["batches"].append(batch)
        for stage, array in tails.items():
            key = f"client_{index}_{stage}"
            arrays[key] = array
            target["tail_snapshots"][stage] = key
        target["final_delta"] = report.tensor_fingerprint(delta)
        client["update"] = deepcopy(target["final_delta"])
        targets.append(target)
    payload = {"schema": report.SCHEMA, "task_id": task["task_id"], "task_fingerprint": task["fingerprint"],
        "configuration": deepcopy(task["config"]), "target_client_indices": [19, 84], "stop_after_client": 84,
        "stop": {"round": 1, "after_client": "client-84", "before_aggregation": True,
                 "completed_rounds": 0, "crypto_finalized": False, "next_client_not_trained": "client-85"},
        "prefix": prefix, "targets": targets,
        "snapshots": {key: report.tensor_fingerprint(value) for key, value in arrays.items()}}
    report.validate_observations(payload, task, arrays)
    return payload, arrays


class TailFixture(prior.StepFixture):
    @classmethod
    def setUpClass(cls):
        prior.StepFixture.setUpClass.__func__(cls)
        cls.prefix_reference = deepcopy(cls.reference)
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.step_output = Path(temporary.name) / "step"
        (cls.step_output / "task_plans").mkdir(parents=True)
        cls.step_manifest = step.build_manifest(cls.prefix_reference)
        cls.step_tasks = step.build_tasks(cls.step_manifest)
        base.write_json(cls.step_output / "manifest.json", cls.step_manifest)
        base.write_json(cls.step_output / "task_plans/steps.json", {
            "manifest_fingerprint": cls.step_manifest["fingerprint"], "tasks": cls.step_tasks})
        for task in cls.step_tasks:
            folder = cls.step_output / "tasks" / task["task_id"]
            attempt = folder / "attempts/synthetic"
            attempt.mkdir(parents=True)
            base.write_json(folder / "task.json", task)
            payload, arrays = synthetic_step_observations(task, cls.prefix_reference)
            path = attempt / "snapshots.npz"
            np.savez(path, **arrays)
            artifact = step.seal_artifact({"status": "complete", "task_fingerprint": task["fingerprint"],
                "gpu_uuid": GPU, "environment": deepcopy(cls.prepared.metadata), "observations": payload,
                "wall_seconds": .1, "attempt": "synthetic", "fresh_start": True, "checkpoints_used": False,
                "training_round_completed": False, "snapshot_file": {"path": "attempts/synthetic/snapshots.npz",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})
            base.write_json(folder / "completed.json", artifact)
        cls.reference = protocol.audit_reference(cls.step_output)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "tail"
        (self.output / "task_plans").mkdir(parents=True)
        self.manifest = protocol.build_manifest(self.reference)
        self.tasks = protocol.build_tasks(self.manifest)
        base.write_json(self.output / "manifest.json", self.manifest)
        base.write_json(self.output / "task_plans/steps.json", {
            "manifest_fingerprint": self.manifest["fingerprint"], "tasks": self.tasks})
        for task in self.tasks:
            folder = self.output / "tasks" / task["task_id"]
            folder.mkdir(parents=True)
            base.write_json(folder / "task.json", task)

    @contextmanager
    def changed_step(self, repeat=2):
        task = self.step_tasks[repeat - 1]
        folder = self.step_output / "tasks" / task["task_id"]
        artifact_path, npz_path = folder / "completed.json", folder / "attempts/synthetic/snapshots.npz"
        original_artifact, original_npz = artifact_path.read_bytes(), npz_path.read_bytes()
        value = runtime.read_json(artifact_path)
        arrays = step.load_arrays(self.step_output, task, value)
        def save():
            np.savez(npz_path, **arrays)
            value["snapshot_file"]["sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
            value.pop("artifact_fingerprint", None)
            base.write_json(artifact_path, step.seal_artifact(value))
        try:
            yield task, value, arrays, save
        finally:
            artifact_path.write_bytes(original_artifact)
            npz_path.write_bytes(original_npz)


class TailProtocolTests(TailFixture):
    def test_actual_99_task_reference_chain_is_audited_readonly(self):
        before = self.reference_hashes(), step.evidence_hashes(self.step_output)
        with self.readonly_guards():
            reference = protocol.audit_reference(self.step_output)
            protocol.verify_reference(reference)
        self.assertEqual(reference, self.reference)
        self.assertEqual(len(reference["step_anchors"]), 3)
        self.assertEqual(reference["anchors"], self.prefix_reference["anchors"])
        self.assertEqual(before, (self.reference_hashes(), step.evidence_hashes(self.step_output)))

    def test_six_interleaved_fresh_tasks_keep_every_original_config_field(self):
        self.assertEqual([(t["policy"], t["repeat"]) for t in self.tasks],
            [(p, r) for r in (1, 2, 3) for p in protocol.POLICIES])
        self.assertEqual(len({t["fingerprint"] for t in self.tasks}), 6)
        self.assertEqual(len({t["task_id"] for t in self.tasks}), 6)
        for task in self.tasks:
            self.assertEqual(task["config"], self.step_tasks[0]["config"])
            self.assertEqual((task["config"]["rounds"], task["target_clients"], task["stop_after_client"]), (3, [19, 84], 84))
            self.assertEqual(task["same_gpu_uuid"], GPU)
        self.assertEqual(self.manifest["policy_scope"]["tail_cudnn_deterministic"]["boundaries"], protocol.TAIL_SCOPE)
        self.assertFalse(self.manifest["original_numerical_policy"])
        for key in ("aggregation_executed", "training_round_completed", "checkpoints_reused",
                    "health_assessed", "formal_qualification_assessed", "next_stage_automatic"):
            self.assertFalse(self.manifest[key])

    def test_source_map_preserves_frozen78_and_adds_exact_four(self):
        old, new = step.source_hashes(), protocol.source_hashes()
        self.assertEqual((len(old), len(new)), (78, 82))
        self.assertEqual({k: new[k] for k in old}, old)
        self.assertEqual(set(new) - set(old), set(protocol.NEW_SOURCES))

    def test_sealed_manifest_policy_order_and_current_source_changes_reject_resume(self):
        self.assertEqual(protocol.read_study(self.output, current_sources=True)[1], self.tasks)
        with mock.patch.object(protocol, "source_hashes", return_value={"changed": "hash"}), \
                self.assertRaisesRegex(ValueError, "identity changed"):
            protocol.read_study(self.output, current_sources=True)
        changed = deepcopy(self.manifest)
        changed["execution_order"].reverse()
        changed["fingerprint"] = base.digest({k: v for k, v in changed.items() if k != "fingerprint"})
        base.write_json(self.output / "manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            protocol.read_study(self.output)
        base.write_json(self.output / "manifest.json", self.manifest)
        plan = runtime.read_json(self.output / "task_plans/steps.json")
        plan["tasks"][1]["policy"] = "original"
        base.write_json(self.output / "task_plans/steps.json", plan)
        with self.assertRaisesRegex(ValueError, "task plan"):
            protocol.read_study(self.output)

    def test_output_separation_covers_step_prefix_and_all_five_old_studies(self):
        roots = [self.step_output, self.prefix_output, *self.paths.values()]
        for root in roots:
            for output in (root, root / "new-tail", root.parent):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "non-nested"):
                    protocol.separate_outputs(output, self.step_output, roots[1:])

    def test_changed_original_NPZ_is_rejected_even_if_summary_is_mocked_complete(self):
        summary = report.summarize(self.step_output)
        with self.changed_step() as (task, value, arrays, save):
            path = self.step_output / "tasks" / task["task_id"] / value["snapshot_file"]["path"]
            path.write_bytes(path.read_bytes() + b"changed")
            with mock.patch.object(report, "summarize", return_value=summary), self.assertRaisesRegex(ValueError, "SHA mismatch"):
                protocol.audit_reference(self.step_output)
            with self.assertRaisesRegex(ValueError, "three-step evidence changed"):
                protocol.verify_reference(self.reference)

    def test_resealed_non_tail_gradient_difference_does_not_match_reviewed_pattern(self):
        with self.changed_step() as (_, value, _, save):
            value["observations"]["targets"][0]["batches"][0]["gradients"]["sha256"] = "f" * 64
            save()
            self.assertEqual(report.summarize(self.step_output)["status"], "complete")
            with self.assertRaisesRegex(ValueError, "client19 tail gradient"):
                protocol.audit_reference(self.step_output)

    def test_resealed_other_parameter_change_is_rejected_from_actual_arrays(self):
        with self.changed_step() as (_, value, arrays, save):
            target = value["observations"]["targets"][0]
            key = target["tail_snapshots"]["gradients"]
            arrays[key][4] += np.float32(.125)  # conv2_w in this small ten-block model.
            fp = report.tensor_fingerprint(arrays[key])
            value["observations"]["snapshots"][key] = target["batches"][-1]["gradients"] = fp
            save()
            self.assertEqual(report.summarize(self.step_output)["status"], "complete")
            with self.assertRaisesRegex(ValueError, "beyond conv1"):
                protocol.audit_reference(self.step_output)

    def test_step_anchor_observation_or_environment_edit_is_not_accepted_as_new_reference(self):
        for edit in (lambda r: r["step_anchors"][0]["observations"]["stop"].update(round=2),
                     lambda r: r["execution_environment"].update(unrecorded=True)):
            changed = deepcopy(self.reference)
            edit(changed)
            with self.assertRaisesRegex(ValueError, "reference identity or observations changed"):
                protocol.verify_reference(changed)

    def test_original90_evidence_change_remains_blocking(self):
        path = self.paths["history"] / "execution_environment.json"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaises(ValueError):
                protocol.verify_reference(self.reference)
        finally:
            path.write_bytes(original)

    def test_reference_read_race_is_rejected(self):
        real, calls = step.evidence_hashes, []
        def unstable(output):
            calls.append(1)
            return real(output) if len(calls) < 4 else {"changed": "during-read"}
        with mock.patch.object(step, "evidence_hashes", side_effect=unstable), \
                self.assertRaisesRegex(ValueError, "changed during the reference audit"):
            protocol.audit_reference(self.step_output)

    def test_new_completion_uses_original_safe_NPZ_identity_and_policy_fingerprint(self):
        artifact = self.complete(self.tasks[0])
        self.assertEqual(protocol.load_completed(self.output, self.tasks[0]), artifact)
        np.testing.assert_array_equal(protocol.load_arrays(self.output, self.tasks[0], artifact)["tail"], [1., 2.])
        wrong = deepcopy(artifact)
        wrong["task_fingerprint"] = self.tasks[1]["fingerprint"]
        wrong.pop("artifact_fingerprint")
        base.write_json(self.output / "tasks" / self.tasks[0]["task_id"] / "completed.json", protocol.seal_artifact(wrong))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            protocol.load_completed(self.output, self.tasks[0])


if __name__ == "__main__":
    unittest.main()
