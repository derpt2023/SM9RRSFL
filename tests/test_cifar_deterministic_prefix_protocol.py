"""Immutable full-prefix policies with a serialized 105-task upstream chain."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import cifar_deterministic_prefix_protocol as protocol
import cifar_tail_determinism_runtime as policy_runtime
import tests.test_cifar_tail_determinism_protocol as prior
import tests.test_cifar_prefix_probe_protocol as prefix_fixture
import tests.test_cifar_prefix_clients as client_fixture

tail, report, base, runtime = protocol.tail, protocol.tail_report, protocol.base, protocol.runtime
GPU = prior.GPU
PROFILE = {"cudnn_enabled": True, "cudnn_deterministic": False, "cudnn_benchmark": False,
    "cudnn_allow_tf32": True, "cuda_matmul_allow_tf32": True, "deterministic_algorithms": False,
    "deterministic_warn_only": False, "float32_matmul_precision": "high", "cudnn_benchmark_limit": 10}


def add_tail_policy(payload, task):
    effective = {**PROFILE, "cudnn_deterministic": task["policy"] == "tail_cudnn_deterministic"}
    events = [{"client_id": target["client_id"],
        **{key: target["batches"][-1][key] for key in ("batch", "epoch", "batch_in_epoch", "samples")},
        "before": deepcopy(PROFILE), "effective": deepcopy(effective), "restored": deepcopy(PROFILE),
        "backward_completed": True} for target in payload["targets"]]
    clients = payload["prefix"]["rounds"][0]["clients"]
    training = sum(len(c["minibatch_sizes_per_epoch"]) * c["epochs"] for c in clients)
    evaluation = sum(len(e["batches"]) for e in payload["prefix"]["evaluations"])
    payload["numerical_policy"] = {"schema": policy_runtime.POLICY_SCHEMA, "policy": task["policy"],
        "baseline": deepcopy(PROFILE), "exit_profile": deepcopy(PROFILE), "events": events,
        "expected_scope": policy_runtime.SCOPE, "other_flags_changed": False, "restored_on_exit": True,
        "scoped_changes": 2 if task["policy"] == "tail_cudnn_deterministic" else 0,
        "checks": {"forward_before": training + evaluation, "forward_after": training + evaluation,
            "non_target_backward_before": training - 2, "non_target_backward_after": training - 2,
            "post_flat": len(clients), "exit": 1}}
    return payload


class DeterministicPrefixFixture(prior.TailFixture):
    @classmethod
    def setUpClass(cls):
        original_metadata, original_payload = prefix_fixture.PrefixFixture.worker_metadata, client_fixture.payload
        def metadata(instance):
            value = original_metadata(instance)
            value["torch"].update({key: PROFILE[key] for key in report.ENVIRONMENT_FLAGS})
            return value
        def prefix_payload(task):
            value = original_payload(task)
            # Earlier generic fixtures also used client2/3 as singleton examples;
            # this reference has only the two clients observed in the real panel.
            for index, samples in ((2, 50), (3, 100)):
                part = value["partition"]["clients"][index]
                part.update(samples=samples, indices=prior.fingerprint("fixed-part" + str(index), [samples], "<i8"))
                for rd in value["rounds"]:
                    client = rd["clients"][index]
                    client.update(samples=samples, minibatch_sizes_per_epoch=[50] * (samples // 50),
                        epoch_indices=[prior.fingerprint("fixed-order" + str(index) + str(rd["round"]), [samples], "<i8")])
                    client["stats"]["samples"] = samples
            return value
        with mock.patch.object(prefix_fixture.PrefixFixture, "worker_metadata", metadata), \
                mock.patch.object(client_fixture, "payload", prefix_payload):
            prior.TailFixture.setUpClass.__func__(cls)
        cls.tail_reference = deepcopy(cls.reference)
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.tail_output = Path(temporary.name) / "tail"
        (cls.tail_output / "task_plans").mkdir(parents=True)
        cls.tail_manifest = tail.build_manifest(cls.tail_reference)
        cls.tail_tasks = tail.build_tasks(cls.tail_manifest)
        base.write_json(cls.tail_output / "manifest.json", cls.tail_manifest)
        base.write_json(cls.tail_output / "task_plans/steps.json", {
            "manifest_fingerprint": cls.tail_manifest["fingerprint"], "tasks": cls.tail_tasks})
        for task in cls.tail_tasks:
            folder = cls.tail_output / "tasks" / task["task_id"]
            attempt = folder / "attempts/synthetic"
            attempt.mkdir(parents=True)
            base.write_json(folder / "task.json", task)
            numeric_task = deepcopy(task)
            if task["policy"] == "tail_cudnn_deterministic":
                numeric_task["repeat"] = 4  # Different from A, identical across B's three runs.
            payload, arrays = prior.synthetic_step_observations(numeric_task, cls.prefix_reference)
            add_tail_policy(payload, task)
            report.validate_observations(payload, task, arrays)
            report.validate_environment_policy(payload, cls.prepared.metadata)
            path = attempt / "snapshots.npz"
            np.savez(path, **arrays)
            artifact = tail.seal_artifact({"status": "complete", "task_fingerprint": task["fingerprint"],
                "gpu_uuid": GPU, "environment": deepcopy(cls.prepared.metadata), "observations": payload,
                "wall_seconds": .1, "attempt": "synthetic", "fresh_start": True, "checkpoints_used": False,
                "training_round_completed": False, "snapshot_file": {"path": "attempts/synthetic/snapshots.npz",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})
            base.write_json(folder / "completed.json", artifact)
        cls.reference = protocol.audit_reference(cls.tail_output)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "deterministic-prefix"
        (self.output / "task_plans").mkdir(parents=True)
        self.manifest = protocol.build_manifest(self.reference)
        self.tasks = protocol.build_tasks(self.manifest)
        base.write_json(self.output / "manifest.json", self.manifest)
        base.write_json(self.output / "task_plans/prefix.json", {
            "manifest_fingerprint": self.manifest["fingerprint"], "tasks": self.tasks})
        for task in self.tasks:
            folder = self.output / "tasks" / task["task_id"]
            folder.mkdir(parents=True)
            base.write_json(folder / "task.json", task)

    def complete(self, task=None):
        task = task or self.tasks[0]
        folder = self.output / "tasks" / task["task_id"]
        (folder / "attempts/synthetic").mkdir(parents=True, exist_ok=True)
        value = protocol.seal_artifact({"status": "complete", "task_fingerprint": task["fingerprint"],
            "gpu_uuid": GPU, "environment": deepcopy(self.prepared.metadata), "observations": {},
            "wall_seconds": .1, "attempt": "synthetic", "fresh_start": True, "checkpoints_used": False,
            "completed_training_rounds": 3})
        base.write_json(folder / "completed.json", value)
        return value

    @contextmanager
    def changed_tail(self, index=3):
        task = self.tail_tasks[index]
        folder = self.tail_output / "tasks" / task["task_id"]
        json_path, npz_path = folder / "completed.json", folder / "attempts/synthetic/snapshots.npz"
        original_json, original_npz = json_path.read_bytes(), npz_path.read_bytes()
        value = runtime.read_json(json_path)
        arrays = tail.load_arrays(self.tail_output, task, value)
        def save():
            np.savez(npz_path, **arrays)
            value["snapshot_file"]["sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
            value.pop("artifact_fingerprint", None)
            base.write_json(json_path, tail.seal_artifact(value))
        try:
            yield task, value, arrays, save
        finally:
            json_path.write_bytes(original_json)
            npz_path.write_bytes(original_npz)


class DeterministicPrefixProtocolTests(DeterministicPrefixFixture):
    def test_actual105_task_chain_and_all_raw_tail_arrays_are_audited_readonly(self):
        before = self.reference_hashes(), tail.step.evidence_hashes(self.step_output), tail.evidence_hashes(self.tail_output)
        with self.readonly_guards():
            actual = protocol.audit_reference(self.tail_output)
            protocol.verify_reference(actual)
        self.assertEqual(actual, self.reference)
        self.assertEqual(len(actual["tail_anchors"]), 6)
        self.assertEqual(actual["step_anchors"], self.tail_reference["step_anchors"])
        self.assertEqual(before, (self.reference_hashes(), tail.step.evidence_hashes(self.step_output), tail.evidence_hashes(self.tail_output)))

    def test_six_full_three_round_tasks_keep_config_and_interleave_one_factor(self):
        self.assertEqual([(t["policy"], t["repeat"]) for t in self.tasks],
            [(p, r) for r in (1, 2, 3) for p in protocol.POLICIES])
        self.assertEqual(len({t["fingerprint"] for t in self.tasks}), 6)
        self.assertEqual(len({t["task_id"] for t in self.tasks}), 6)
        for task in self.tasks:
            self.assertEqual(task["config"], self.tail_tasks[0]["config"])
            self.assertEqual((task["partition"], task["config"]["num_clients"], task["config"]["rounds"]), ("dirichlet", 100, 3))
            self.assertNotIn("stop_after_client", task)
            self.assertNotIn("target_clients", task)
            self.assertEqual(task["same_gpu_uuid"], GPU)
        self.assertEqual(sum(t["config"]["rounds"] * t["config"]["num_clients"] for t in self.tasks), 1800)
        scope = self.manifest["policy_scope"][protocol.POLICIES[1]]
        self.assertEqual((scope["expected_events_per_task"], scope["rounds"]), (6, [1, 2, 3]))
        self.assertTrue(self.manifest["aggregation_executed"])
        self.assertFalse(self.manifest["original_numerical_policy"])
        for key in ("checkpoints_reused", "health_assessed", "formal_qualification_assessed", "next_stage_automatic"):
            self.assertFalse(self.manifest[key])

    def test_source_map_preserves_all82_and_adds_only_four_new_modules(self):
        old, new = tail.source_hashes(), protocol.source_hashes()
        self.assertEqual((len(old), len(new)), (82, 86))
        self.assertEqual({k: new[k] for k in old}, old)
        self.assertEqual(set(new) - set(old), set(protocol.NEW_SOURCES))

    def test_frozen_scope_order_task_plan_and_default_live_sources_block_resume(self):
        self.assertEqual(protocol.read_study(self.output)[1], self.tasks)
        with mock.patch.object(protocol, "source_hashes", return_value={"changed": "sha"}), \
                self.assertRaisesRegex(ValueError, "identity changed"):
            protocol.read_study(self.output)
        changed = deepcopy(self.manifest)
        changed["policy_scope"][protocol.POLICIES[1]]["expected_events_per_task"] = 2
        changed["fingerprint"] = base.digest({k: v for k, v in changed.items() if k != "fingerprint"})
        base.write_json(self.output / "manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            protocol.read_study(self.output)
        base.write_json(self.output / "manifest.json", self.manifest)
        plan = runtime.read_json(self.output / "task_plans/prefix.json")
        plan["tasks"].reverse()
        base.write_json(self.output / "task_plans/prefix.json", plan)
        with self.assertRaisesRegex(ValueError, "task plan"):
            protocol.read_study(self.output)

    def test_output_cannot_equal_contain_or_be_nested_in_any_reference(self):
        roots = [self.tail_output, self.step_output, self.prefix_output, *self.paths.values()]
        for root in roots:
            for path in (root, root / "new", root.parent):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "non-nested"):
                    protocol.separate_outputs(path, self.tail_output, roots[1:])

    def test_raw_tail_NPZ_identity_is_required_even_if_old_summary_claims_success(self):
        summary = report.summarize(self.tail_output)
        with self.changed_tail() as (task, value, _, _):
            path = self.tail_output / "tasks" / task["task_id"] / value["snapshot_file"]["path"]
            path.write_bytes(path.read_bytes() + b"changed")
            with mock.patch.object(report, "summarize", return_value=summary), self.assertRaisesRegex(ValueError, "SHA mismatch"):
                protocol.audit_reference(self.tail_output)
            with self.assertRaisesRegex(ValueError, "six-tail evidence changed"):
                protocol.verify_reference(self.reference)

    def test_resealed_variable_B_arrays_cannot_pass_a_mocked_successful_summary(self):
        summary = report.summarize(self.tail_output)
        with self.changed_tail() as (_, value, arrays, save):
            payload = value["observations"]
            target = payload["targets"][0]
            key = target["tail_snapshots"]["gradients"]
            arrays[key][0] += np.float32(.125)
            fp = report.step.tensor_fingerprint(arrays[key])
            payload["snapshots"][key] = target["batches"][-1]["gradients"] = fp
            save()
            self.assertEqual(report.summarize(self.tail_output)["status"], "complete")
            with mock.patch.object(report, "summarize", return_value=summary), self.assertRaisesRegex(ValueError, "actual archived tail arrays"):
                protocol.audit_reference(self.tail_output)

    def test_policy_flag_leak_and_missing_old_environment_fields_are_rejected(self):
        summary = report.summarize(self.tail_output)
        with self.changed_tail() as (_, value, _, save):
            value["observations"]["numerical_policy"]["events"][0]["restored"]["cudnn_deterministic"] = True
            save()
            with mock.patch.object(report, "summarize", return_value=summary), self.assertRaisesRegex(ValueError, "backward flags"):
                protocol.audit_reference(self.tail_output)
        with self.changed_tail() as (_, value, _, save):
            value["environment"]["torch"].pop("deterministic_warn_only")
            save()
            with mock.patch.object(report, "summarize", return_value=summary), self.assertRaisesRegex(ValueError, "numerical environment differs"):
                protocol.audit_reference(self.tail_output)

    def test_reference_anchor_or_original90_evidence_change_is_not_rebound(self):
        changed = deepcopy(self.reference)
        changed["tail_anchors"][0]["observations"]["stop"]["round"] = 2
        with self.assertRaisesRegex(ValueError, "reference identity or observations changed"):
            protocol.verify_reference(changed)
        path = self.paths["history"] / "execution_environment.json"
        old = path.read_bytes()
        try:
            path.write_bytes(old + b"\n")
            with self.assertRaises(ValueError):
                protocol.verify_reference(self.reference)
        finally:
            path.write_bytes(old)

    def test_source_or_evidence_read_race_is_rejected(self):
        real, calls = tail.evidence_hashes, []
        def unstable(output):
            calls.append(1)
            return real(output) if len(calls) < 4 else {"changed": "during-read"}
        with mock.patch.object(tail, "evidence_hashes", side_effect=unstable), \
                self.assertRaisesRegex(ValueError, "changed during the reference audit"):
            protocol.audit_reference(self.tail_output)

    def test_new_completion_is_json_only_and_bound_to_policy_task_and_physical_GPU(self):
        value = self.complete()
        self.assertEqual(protocol.load_completed(self.output, self.tasks[0]), value)
        self.assertFalse(list(self.output.rglob("*.npz")))
        folder = self.output / "tasks" / self.tasks[0]["task_id"]
        for edit in (lambda v: v.update(task_fingerprint=self.tasks[1]["fingerprint"]),
                     lambda v: v.update(gpu_uuid="GPU-other")):
            altered = deepcopy(value)
            edit(altered)
            altered.pop("artifact_fingerprint")
            base.write_json(folder / "completed.json", protocol.seal_artifact(altered))
            with self.assertRaises(ValueError):
                protocol.load_completed(self.output, self.tasks[0])


if __name__ == "__main__":
    unittest.main()
