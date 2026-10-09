"""Immutable step identities anchored to six real serialized synthetic prefixes."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import cifar_client_step_probe_protocol as protocol
import tests.test_cifar_prefix_probe_protocol as prefix_fixture
import tests.test_cifar_prefix_clients as clients_fixture

base, runtime = protocol.base, protocol.runtime
GPU = prefix_fixture.GPU


class StepFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prefix_fixture.PrefixFixture.setUpClass.__func__(cls)
        prepared = clients_fixture.SerializedClientReaderTests()
        prepared.fx, prepared.reference, prepared.paths = cls.fx, cls.reference, cls.paths
        prepared.setUp()
        cls.addClassCleanup(prepared.doCleanups)
        cls.prepared, cls.prefix_output = prepared, prepared.output
        # Establish exactly the reviewed r1 pattern in all three Dir repeats.
        for item in prepared.tasks:
            if item["partition"] != "dirichlet":
                continue
            path = prepared.output / "tasks" / item["task_id"] / "completed.json"
            artifact = runtime.read_json(path)
            observations = artifact["observations"]
            for index, samples in ((19, 701), (84, 301)):
                observations["partition"]["clients"][index].update(samples=samples,
                    indices=clients_fixture.observations.fingerprint("indices" + str(index), [samples], "<i8"))
                for rd in observations["rounds"]:
                    client = rd["clients"][index]
                    client.update(samples=samples, minibatch_sizes_per_epoch=[50] * (samples // 50) + [1],
                        epoch_indices=[clients_fixture.observations.fingerprint("epoch" + str(index) + str(rd["round"]), [samples], "<i8")])
                    client["stats"]["samples"] = samples
                observations["rounds"][0]["clients"][index]["update"] = clients_fixture.observations.fingerprint(
                    "updated-" + str(index) + "-repeat" + str(item["repeat"]))
            artifact.pop("artifact_fingerprint")
            base.write_json(path, protocol.prefix.seal_artifact(artifact))
        cls.reference = protocol.audit_reference(cls.prefix_output)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "step"
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

    def readonly_guards(self):
        return self.prepared.guards()

    def reference_hashes(self):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for root in [self.prefix_output, *self.paths.values()] for p in root.rglob("*") if p.is_file()}

    def complete(self, task=None, *, arrays=None):
        task = task or self.tasks[0]
        folder = self.output / "tasks" / task["task_id"]
        attempt = folder / "attempts/synthetic"
        attempt.mkdir(parents=True, exist_ok=True)
        snapshot = attempt / "snapshots.npz"
        np.savez(snapshot, **({"tail": np.array([1., 2.], dtype=np.float32)} if arrays is None else arrays))
        value = protocol.seal_artifact({"status": "complete", "task_fingerprint": task["fingerprint"],
            "gpu_uuid": GPU, "environment": deepcopy(self.prepared.metadata),
            "observations": {}, "wall_seconds": .1, "attempt": "synthetic",
            "fresh_start": True, "checkpoints_used": False,
            "snapshot_file": {"path": "attempts/synthetic/snapshots.npz", "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}})
        base.write_json(folder / "completed.json", value)
        return value


class StepProtocolTests(StepFixture):
    def test_six_prefix_and_90_upstream_reference_chain_is_audited_readonly(self):
        before = self.reference_hashes()
        with self.readonly_guards():
            actual = protocol.audit_reference(self.prefix_output)
            protocol.verify_reference(actual)
        self.assertEqual(actual, self.reference)
        self.assertEqual(len(actual["anchors"]), 3)
        self.assertEqual(before, self.reference_hashes())

    def test_three_fresh_tasks_preserve_whole_config_and_stop_before_client85(self):
        self.assertEqual(len(self.tasks), 3)
        self.assertEqual(len({t["fingerprint"] for t in self.tasks}), 3)
        old = self.reference["anchors"][0]["task"]
        for task in self.tasks:
            self.assertEqual(task["config"], old["config"])
            self.assertEqual(task["target_clients"], [19, 84])
            self.assertEqual(task["stop_after_client"], 84)
            self.assertEqual(task["same_gpu_uuid"], GPU)
            self.assertNotEqual(task["fingerprint"], old["fingerprint"])
        self.assertEqual(self.manifest["stop_boundary"], "before_round1_client85_local_training")
        for field in ("aggregation_executed", "training_round_completed", "checkpoints_reused", "health_assessed", "next_stage_automatic"):
            self.assertFalse(self.manifest[field])
        self.assertTrue(self.manifest["original_numerical_policy"])

    def test_new_source_map_preserves_all_72_old_sources(self):
        old, current = protocol.prefix.source_hashes(), protocol.source_hashes()
        self.assertEqual((len(old), len(current)), (72, 78))
        self.assertEqual({k: current[k] for k in old}, old)

    def test_changed_source_stop_contract_and_task_plan_cannot_resume(self):
        self.assertEqual(protocol.read_study(self.output, current_sources=True)[1], self.tasks)
        with mock.patch.object(protocol, "source_hashes", return_value={"changed": "hash"}), \
                self.assertRaisesRegex(ValueError, "identity changed"):
            protocol.read_study(self.output, current_sources=True)
        changed = deepcopy(self.manifest)
        changed["stop_after_client"] = 19
        changed["fingerprint"] = base.digest({k: v for k, v in changed.items() if k != "fingerprint"})
        base.write_json(self.output / "manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            protocol.read_study(self.output)
        base.write_json(self.output / "manifest.json", self.manifest)
        plan = runtime.read_json(self.output / "task_plans/steps.json")
        plan["tasks"][0]["config"]["batch_size"] = 51
        base.write_json(self.output / "task_plans/steps.json", plan)
        with self.assertRaisesRegex(ValueError, "task plan"):
            protocol.read_study(self.output)

    def test_reference_artifact_or_old90_evidence_change_blocks_resume(self):
        paths = [self.prefix_output / "tasks" / self.reference["anchors"][0]["task"]["task_id"] / "completed.json",
                 self.paths["history"] / "execution_environment.json"]
        for path in paths:
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"\n")
                with self.subTest(path=path), self.assertRaises(ValueError):
                    protocol.read_study(self.output, current_sources=True)
            finally:
                path.write_bytes(original)
        changed = deepcopy(self.reference)
        changed["anchors"][0]["observations"]["rounds"][0]["clients"][19]["stats"]["loss"] = .123
        with self.assertRaisesRegex(ValueError, "observations changed"):
            protocol.verify_reference(changed)

    def test_output_must_not_overlap_any_of_six_reference_studies(self):
        roots = [self.prefix_output, *self.paths.values()]
        protocol.separate_outputs(self.output, self.prefix_output, self.paths.values())
        for root in roots:
            for output in (root, root / "new-step", root.parent):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "non-nested"):
                    protocol.separate_outputs(output, self.prefix_output, self.paths.values())

    def test_completion_snapshot_is_bound_and_missing_is_not_reused(self):
        task = self.tasks[0]
        self.assertIsNone(protocol.load_completed(self.output, task))
        good = self.complete()
        self.assertEqual(protocol.load_completed(self.output, task), good)
        np.testing.assert_array_equal(protocol.load_arrays(self.output, task, good)["tail"], np.array([1., 2.], dtype=np.float32))
        snapshot = self.output / "tasks" / task["task_id"] / good["snapshot_file"]["path"]
        snapshot.write_bytes(snapshot.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            protocol.load_completed(self.output, task)

    def test_snapshot_absolute_traversal_and_symlinks_are_rejected(self):
        task = self.tasks[0]
        good = self.complete()
        for path in ("/tmp/snapshots.npz", "attempts/../snapshots.npz", "snapshots.npz"):
            bad = deepcopy(good)
            bad["snapshot_file"]["path"] = path
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "relative path"):
                protocol.load_arrays(self.output, task, bad)
        folder = self.output / "tasks" / task["task_id"]
        snapshot = folder / good["snapshot_file"]["path"]
        real = snapshot.with_name("real.npz")
        snapshot.rename(real)
        snapshot.symlink_to(real)
        with self.assertRaisesRegex(ValueError, "symlink"):
            protocol.load_arrays(self.output, task, good)

    def test_snapshot_object_arrays_and_oversized_member_count_are_rejected(self):
        task = self.tasks[0]
        value = self.complete(arrays={"bad": np.array([object()], dtype=object)})
        with self.assertRaisesRegex(ValueError, "Object arrays"):
            protocol.load_arrays(self.output, task, value)
        value = self.complete(arrays={"a" + str(i): np.zeros(1) for i in range(65)})
        with self.assertRaisesRegex(ValueError, "bounded diagnostic"):
            protocol.load_arrays(self.output, task, value)

    def test_task_directory_or_tasks_parent_symlink_cannot_escape_study(self):
        task = self.tasks[0]
        good = self.complete()
        folder = self.output / "tasks" / task["task_id"]
        external = self.output.parent / "external-task"
        folder.rename(external)
        folder.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink|outside|escapes"):
            protocol.load_arrays(self.output, task, good)
        folder.unlink()
        external.rename(folder)
        tasks = self.output / "tasks"
        external_tasks = self.output.parent / "external-tasks"
        tasks.rename(external_tasks)
        tasks.symlink_to(external_tasks, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink|outside|escapes"):
            protocol.load_arrays(self.output, task, good)


if __name__ == "__main__":
    unittest.main()
