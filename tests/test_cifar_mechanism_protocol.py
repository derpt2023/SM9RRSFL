"""Serialized 111-task provenance and fixed clean mechanism task identities."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cifar_mechanism_protocol as protocol
import tests.test_cifar_deterministic_prefix_protocol as prior
import cifar_deterministic_prefix_runtime as observed

base, runtime, prefix = protocol.base, protocol.runtime, protocol.prefix


def full_prefix_payload(task, reference):
    old_policy = "original" if task["policy"] == "original" else "tail_cudnn_deterministic"
    tail = next(a["observations"] for a in reference["tail_anchors"]
                if a["task"]["policy"] == old_policy and a["task"]["repeat"] == task["repeat"])
    value = deepcopy(reference["anchors"][0]["observations"])
    value.update(task_id=task["task_id"], task_fingerprint=task["fingerprint"], configuration=deepcopy(task["config"]))
    value["actual_batches"], value["singleton_batches"] = [], []
    events, training = [], 0
    baseline = deepcopy(prior.PROFILE)
    for row in value["rounds"]:
        for client in row["clients"]:
            sizes = client["minibatch_sizes_per_epoch"] * client["epochs"]
            training += len(sizes)
            value["actual_batches"].append({"round": row["round"],
                **{k: client[k] for k in ("client_id", "samples", "epochs", "batch_size")},
                "batch_sizes": sizes, "forward_count": len(sizes), "backward_count": len(sizes)})
            if 1 not in sizes:
                continue
            target = next(t for t in tail["targets"] if t["client_id"] == client["client_id"])
            singleton = {**deepcopy(target["batches"][-1]), "round": row["round"],
                "client_id": client["client_id"], "parameter_layout": deepcopy(target["parameter_layout"])}
            value["singleton_batches"].append(singleton)
            client["update"] = deepcopy(target["final_delta"])
            events.append({**{k: singleton[k] for k in observed.IDENTITY},
                "before": deepcopy(baseline), "restored": deepcopy(baseline),
                "effective": {**baseline, "cudnn_deterministic": task["policy"] != "original"},
                "backward_completed": True})
    evaluation = sum(len(e["batches"]) for e in value["evaluations"])
    value["numerical_policy"] = {"schema": observed.POLICY_SCHEMA, "policy": task["policy"],
        "baseline": baseline, "exit_profile": deepcopy(baseline), "events": events,
        "expected_scope": observed.SCOPE, "other_flags_changed": False, "restored_on_exit": True,
        "scoped_changes": 6 if task["policy"] != "original" else 0,
        "checks": {"forward_before": training + evaluation, "forward_after": training + evaluation,
            "non_target_backward_before": training - 6, "non_target_backward_after": training - 6,
            "post_flat": 300, "exit": 1}}
    protocol.deterministic_report.validate_observations(value, task)
    return value


class MechanismFixture(prior.DeterministicPrefixFixture):
    @classmethod
    def setUpClass(cls):
        prior.DeterministicPrefixFixture.setUpClass.__func__(cls)
        cls.deterministic_reference = deepcopy(cls.reference)
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.deterministic_output = Path(temporary.name) / "deterministic-prefix"
        (cls.deterministic_output / "task_plans").mkdir(parents=True)
        cls.deterministic_manifest = prefix.build_manifest(cls.deterministic_reference)
        cls.deterministic_tasks = prefix.build_tasks(cls.deterministic_manifest)
        base.write_json(cls.deterministic_output / "manifest.json", cls.deterministic_manifest)
        base.write_json(cls.deterministic_output / "task_plans/prefix.json", {
            "manifest_fingerprint": cls.deterministic_manifest["fingerprint"], "tasks": cls.deterministic_tasks})
        for task in cls.deterministic_tasks:
            folder = cls.deterministic_output / "tasks" / task["task_id"]
            folder.mkdir(parents=True)
            base.write_json(folder / "task.json", task)
            payload = full_prefix_payload(task, cls.deterministic_reference)
            artifact = prefix.seal_artifact({"status": "complete", "task_fingerprint": task["fingerprint"],
                "gpu_uuid": prior.GPU, "environment": deepcopy(cls.prepared.metadata), "observations": payload,
                "wall_seconds": .1, "fresh_start": True, "checkpoints_used": False, "completed_training_rounds": 3})
            base.write_json(folder / "completed.json", artifact)
        cls.reference = protocol.audit_reference(cls.deterministic_output)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "mechanism"
        (self.output / "task_plans").mkdir(parents=True)
        self.manifest = protocol.build_manifest(self.reference)
        self.tasks = protocol.build_tasks(self.manifest)
        base.write_json(self.output / "manifest.json", self.manifest)
        base.write_json(self.output / "task_plans/mechanism.json", {
            "manifest_fingerprint": self.manifest["fingerprint"], "tasks": self.tasks})
        for task in self.tasks:
            folder = self.output / "tasks" / task["task_id"]
            folder.mkdir(parents=True)
            base.write_json(folder / "task.json", task)

    @contextmanager
    def changed_prefix(self, index=3):
        task = self.deterministic_tasks[index]
        path = self.deterministic_output / "tasks" / task["task_id"] / "completed.json"
        original = path.read_bytes()
        value = runtime.read_json(path)
        def save():
            value.pop("artifact_fingerprint", None)
            base.write_json(path, prefix.seal_artifact(value))
        try:
            yield task, value, save
        finally:
            path.write_bytes(original)


class MechanismProtocolTests(MechanismFixture):
    def test_real_serialized111_chain_is_recomputed_without_writes(self):
        paths = protocol.reference_paths(self.reference)
        before = {p: protocol.evidence_hashes(Path(p)) for p in paths}
        with self.readonly_guards():
            actual = protocol.audit_reference(self.deterministic_output)
        self.assertEqual(actual, self.reference)
        self.assertEqual(len(actual["deterministic_prefix_anchors"]), 6)
        self.assertEqual(before, {p: protocol.evidence_hashes(Path(p)) for p in paths})
        self.assertEqual(protocol.read_study(self.output)[1], self.tasks)

    def test_four_tasks_one_history_factor_and_same_numeric_policy(self):
        self.assertEqual([(t["arm"], t["repeat"]) for t in self.tasks],
                         [("H0", 1), ("H1", 1), ("H0", 2), ("H1", 2)])
        self.assertEqual(len({t["fingerprint"] for t in self.tasks}), 4)
        original = self.deterministic_tasks[1]["config"]
        for task in self.tasks:
            self.assertEqual(task["config"], {**original, "rounds": 30})
            self.assertEqual(task["policy"], protocol.POLICY)
            self.assertEqual(task["same_gpu_uuid"], prior.GPU)
            self.assertEqual(task["history_freeze_start_round"], 25 if task["arm"] == "H1" else None)
        self.assertEqual(self.manifest["maximum_training_rounds"], 120)
        self.assertEqual(self.manifest["maximum_client_training_calls"], 12000)
        self.assertFalse(self.manifest["checkpoints_reused"])
        self.assertFalse(self.manifest["automatic_next_stage"])

    def test_source86_frozen_and_only_five_new_scientific_modules(self):
        old, new = prefix.source_hashes(), protocol.source_hashes()
        self.assertEqual((len(old), len(new)), (86, 91))
        self.assertEqual({k: new[k] for k in old}, old)
        self.assertEqual(set(new) - set(old), set(protocol.NEW_SOURCES))

    def test_old_sources_configuration_and_task_order_cannot_be_rebound(self):
        with mock.patch.object(protocol, "source_hashes", return_value={"changed": "hash"}), \
                self.assertRaisesRegex(ValueError, "identity changed"):
            protocol.read_study(self.output)
        changed = deepcopy(self.manifest)
        changed["rounds"] = 150
        changed["fingerprint"] = base.digest({k: v for k, v in changed.items() if k != "fingerprint"})
        base.write_json(self.output / "manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            protocol.read_study(self.output)
        base.write_json(self.output / "manifest.json", self.manifest)
        plan = runtime.read_json(self.output / "task_plans/mechanism.json")
        plan["tasks"].reverse()
        base.write_json(self.output / "task_plans/mechanism.json", plan)
        with self.assertRaisesRegex(ValueError, "task plan"):
            protocol.read_study(self.output)

    def test_all_reference_output_collisions_are_rejected(self):
        paths = protocol.reference_paths(self.reference)
        for root in paths:
            for path in (Path(root), Path(root) / "nested", Path(root).parent):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "non-nested"):
                    protocol.separate_outputs(path, paths[0], paths[1:])

    def test_actual_mechanism_plan_bytes_are_part_of_read_protection(self):
        before = protocol.evidence_hashes(self.output)
        self.assertIn("task_plans/mechanism.json", before)
        self.assertNotIn("task_plans/prefix.json", before)
        path = self.output / "task_plans/mechanism.json"
        path.write_bytes(path.read_bytes() + b"\n")
        self.assertNotEqual(before, protocol.evidence_hashes(self.output))

    def test_resealed_B_variability_is_rejected_from_actual_boundaries(self):
        with self.changed_prefix() as (_, value, save):
            value["observations"]["singleton_batches"][0]["gradients"]["sha256"] = "f" * 64
            save()
            with self.assertRaisesRegex(ValueError, "deterministic effect"):
                protocol.audit_reference(self.deterministic_output)
        with self.changed_prefix() as (_, value, save):
            value["observations"]["singleton_batches"][2]["gradients"]["sha256"] = "f" * 64
            save()
            with self.assertRaisesRegex(ValueError, "actual archived prefix"):
                protocol.audit_reference(self.deterministic_output)

    def test_missing_fresh_flag_restoration_or_environment_fails_before_launch(self):
        for edit, pattern in (
            (lambda v: v.update(fresh_start=False), "non-fresh"),
            (lambda v: v.update(completed_training_rounds=2), "three rounds"),
            (lambda v: v["observations"]["numerical_policy"].update(restored_on_exit=False), "restoration"),
            (lambda v: v["environment"]["torch"].pop("deterministic_warn_only"), "environment differs")):
            with self.subTest(pattern=pattern), self.changed_prefix() as (_, value, save):
                edit(value)
                save()
                with self.assertRaisesRegex(ValueError, pattern):
                    protocol.audit_reference(self.deterministic_output)

    def test_frozen_reference_bytes_and_embedded_anchor_are_verified(self):
        with self.changed_prefix() as (_, value, save):
            value["wall_seconds"] += 1
            save()
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                protocol.verify_reference(self.reference)
        changed = deepcopy(self.reference)
        changed["deterministic_prefix_anchors"][0]["observations"]["initial_model"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "identity or observations changed"):
            protocol.verify_reference(changed)

    def test_read_race_on_current_reference_cannot_be_certified(self):
        real, calls = prefix.evidence_hashes, []
        def unstable(output):
            calls.append(1)
            return real(output) if len(calls) == 1 else {"changed": "during-read"}
        with mock.patch.object(prefix, "evidence_hashes", side_effect=unstable), \
                self.assertRaisesRegex(ValueError, "changed during reference audit"):
            protocol.audit_reference(self.deterministic_output)


if __name__ == "__main__":
    unittest.main()
