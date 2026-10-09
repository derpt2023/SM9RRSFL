"""Independent probe identities, frozen 90-task provenance, and read-only replay."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cifar_prefix_probe_protocol as protocol
import tests.test_cifar_history_diagnose as integration

base, runtime = protocol.base, protocol.runtime
GPU = "GPU-01234567-89ab-cdef-0123-456789abcdef"


class PrefixFixture(unittest.TestCase):
    """Build actual synthetic upstream snapshots once, never start real training."""
    @classmethod
    def setUpClass(cls):
        integration.HistoryDiagnoseIntegrationTests.setUpClass.__func__(cls)
        cls.paths = dict(zip(protocol.REFERENCE_NAMES, (cls.fx.output, cls.fx.threshold_output,
            cls.fx.timing_output, cls.fx.clean_output, cls.fx.matched_output)))
        cls.reference = protocol.audit_reference(cls.paths)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "probe"
        self.output.mkdir()
        (self.output / "task_plans").mkdir()
        self.manifest = protocol.build_manifest(deepcopy(self.reference), GPU)
        self.tasks = protocol.build_tasks(self.manifest)
        base.write_json(self.output / "manifest.json", self.manifest)
        base.write_json(self.output / "task_plans/prefix.json", {
            "manifest_fingerprint": self.manifest["fingerprint"], "tasks": self.tasks})
        for task in self.tasks:
            (self.output / "tasks" / task["task_id"]).mkdir(parents=True)
            base.write_json(self.output / "tasks" / task["task_id"] / "task.json", task)

    def readonly_guards(self):
        stack = ExitStack()
        for obj, name in ((base, "load_split"), (base, "worker_environment"),
                          (base.experiments, "run_measured_experiment"),
                          (base.experiments, "_load_round_checkpoint")):
            stack.enter_context(mock.patch.object(obj, name, side_effect=AssertionError("forbidden " + name)))
        return stack

    def evidence(self):
        return {name: protocol.evidence_hashes(path) for name, path in self.paths.items()}

    def completion(self, task=None):
        task = task or self.tasks[0]
        metadata = self.worker_metadata()
        return protocol.seal_artifact({"status": "complete", "task_fingerprint": task["fingerprint"],
            "gpu_uuid": GPU, "environment": metadata, "observations": {}, "wall_seconds": .1,
            "attempt": "synthetic-attempt", "fresh_start": True, "checkpoints_used": False})

    def worker_metadata(self):
        metadata = deepcopy(self.fx.metadata)
        metadata["environment"]["CUDA_VISIBLE_DEVICES"] = GPU
        metadata["actual_compute_device"].update(uuid=GPU, logical_device="cuda:0")
        name = metadata["actual_compute_device"]["name"]
        metadata["nvidia"] = {"gpus": [{"uuid": GPU, "name": name, "driver_version": "synthetic"}]}
        metadata.setdefault("torch", {})["logical_cuda_devices"] = [
            {"logical_index": 0, "name": name, "uuid": GPU}]
        return metadata


class PrefixProtocolTests(PrefixFixture):
    def test_actual_90_task_chain_is_readonly_and_failed_health_is_retained(self):
        before = self.evidence()
        with self.readonly_guards():
            reference = protocol.audit_reference(self.paths)
            protocol.verify_reference(reference)
        self.assertEqual(reference, self.reference)
        self.assertEqual(before, self.evidence())
        self.assertEqual(reference["forensics_header"]["complete_tasks"], 12)
        self.assertEqual(reference["forensics_header"]["healthy_tasks"], 11)
        self.assertEqual(len(reference["h0_tasks"]), 2)
        self.assertEqual(set(reference["evidence_sha256"]), set(protocol.REFERENCE_NAMES))
        for prefix in reference["historical_prefix"].values():
            self.assertEqual([r["round"] for r in prefix], [0, 1, 2, 3])

    def test_six_fresh_tasks_change_only_original_H0_round_limit(self):
        self.assertEqual(len(self.tasks), 6)
        self.assertEqual(len({t["fingerprint"] for t in self.tasks}), 6)
        self.assertEqual({(t["partition"], t["repeat"]) for t in self.tasks},
                         {(p, r) for p in ("iid", "dirichlet") for r in (1, 2, 3)})
        for task in self.tasks:
            old = next(t for t in self.reference["h0_tasks"] if t["task_id"] == task["original_h0_task_id"])
            differences = {k for k in set(task["config"]) | set(old["config"])
                           if task["config"].get(k) != old["config"].get(k)}
            self.assertEqual(differences, {"rounds"})
            self.assertEqual((old["config"]["rounds"], task["config"]["rounds"]), (150, 3))
            self.assertEqual(task["original_h0_config"], old["config"])
            self.assertEqual(task["original_h0_fingerprint"], old["fingerprint"])
            self.assertNotEqual(task["fingerprint"], old["fingerprint"])
            self.assertEqual(task["same_gpu_uuid"], GPU)
            self.assertEqual(task["config"]["seed"], 2026093001)
            self.assertEqual(task["config"]["malicious_ratio"], 0.)
        for flag in ("original_numerical_policy", "original_ours_algorithm",
                     "fresh_process_per_repeat", "serial_same_physical_gpu"):
            self.assertTrue(self.manifest[flag])
        for flag in ("health_assessed", "formal_qualification_assessed", "next_stage_automatic", "checkpoints_reused"):
            self.assertFalse(self.manifest[flag])

    def test_old_66_source_identities_are_exact_subset_of_new_72(self):
        before, current = protocol.history.source_hashes(), protocol.source_hashes()
        self.assertEqual((len(before), len(current)), (66, 72))
        self.assertEqual({k: current[k] for k in before}, before)
        self.assertEqual(set(current) - set(before), set(protocol.NEW_SOURCES))

    def test_sealed_manifest_policy_task_plan_and_live_source_changes_block_resume(self):
        self.assertEqual(protocol.read_study(self.output, current_sources=True)[1], self.tasks)
        with mock.patch.object(protocol, "source_hashes", return_value={"changed": "hash"}):
            with self.assertRaisesRegex(ValueError, "identity changed"):
                protocol.read_study(self.output, current_sources=True)
            self.assertEqual(protocol.read_study(self.output, current_sources=False)[1], self.tasks)
        changed = deepcopy(self.manifest)
        changed["spec"]["rounds"] = 4
        changed["fingerprint"] = base.digest({k: v for k, v in changed.items() if k != "fingerprint"})
        base.write_json(self.output / "manifest.json", changed)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            protocol.read_study(self.output)
        base.write_json(self.output / "manifest.json", self.manifest)
        plan = runtime.read_json(self.output / "task_plans/prefix.json")
        plan["tasks"][0]["repeat"] = 4
        base.write_json(self.output / "task_plans/prefix.json", plan)
        with self.assertRaisesRegex(ValueError, "task plan"):
            protocol.read_study(self.output)

    def test_original_H0_configuration_or_fingerprint_cannot_be_repurposed(self):
        for key, value in (("local_epochs", 2), ("malicious_ratio", .1), ("attack_start_round", 22),
                           ("seed", 2026093011), ("detector_window", 10)):
            manifest = deepcopy(self.manifest)
            manifest["reference"]["h0_tasks"][0]["config"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                protocol.build_tasks(manifest)
        manifest = deepcopy(self.manifest)
        manifest["reference"]["h0_tasks"][0]["fingerprint"] = "different"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            protocol.build_tasks(manifest)
        reference = deepcopy(self.reference)
        reference["history_manifest_fingerprint"] = "unreviewed"
        with self.assertRaisesRegex(ValueError, "history reference"):
            protocol.build_manifest(reference, GPU)

    def test_all_five_upstream_evidence_sets_are_bound_even_on_resume(self):
        for name, root in self.paths.items():
            path = root / "execution_environment.json"
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"\n")
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "evidence changed"):
                    protocol.read_study(self.output, current_sources=True)
            finally:
                path.write_bytes(original)
        task = self.reference["h0_tasks"][0]
        path = self.paths["history"] / "tasks" / task["task_id"] / base.experiments.COMPLETED_RESULTS_SNAPSHOT
        original = path.read_bytes()
        try:
            path.unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                protocol.verify_reference(self.reference)
        finally:
            path.write_bytes(original)

    def test_reference_changes_during_audit_are_rejected(self):
        real, calls = protocol.evidence_hashes, []
        def change_after_first_pass(root):
            result = real(root)
            if len(calls) >= 5:
                result["simulated-concurrent-change"] = "different"
            calls.append(root)
            return result
        with mock.patch.object(protocol, "evidence_hashes", side_effect=change_after_first_pass), \
                self.assertRaisesRegex(ValueError, "changed during"):
            protocol.audit_reference(self.paths)

    def test_independent_output_rejects_equal_nested_parent_and_symlink_alias(self):
        protocol.separate_outputs(self.output, self.paths)
        for root in self.paths.values():
            for output in (root, root / "nested-probe", root.parent):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "non-nested"):
                    protocol.separate_outputs(output, self.paths)
        alias = self.output.parent / "reference-alias"
        alias.symlink_to(self.paths["history"], target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "non-nested"):
            protocol.separate_outputs(alias / "probe", self.paths)

    def test_completion_reuse_requires_identity_digest_and_exact_GPU_mask(self):
        task = self.tasks[0]
        folder = self.output / "tasks" / task["task_id"]
        # A long-run result/checkpoint is not a short-probe completion artifact.
        (folder / base.experiments.COMPLETED_RESULTS_SNAPSHOT).write_bytes(b"unused old result")
        self.assertIsNone(protocol.load_completed(self.output, task))
        good = self.completion(task)
        base.write_json(folder / "completed.json", good)
        self.assertEqual(protocol.load_completed(self.output, task), good)
        for field, value in (("task_fingerprint", "other"), ("gpu_uuid", "GPU-other"), ("status", "failed")):
            bad = {k: v for k, v in good.items() if k != "artifact_fingerprint"}
            bad[field] = value
            base.write_json(folder / "completed.json", protocol.seal_artifact(bad))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity mismatch"):
                protocol.load_completed(self.output, task)
        bad = deepcopy(good)
        bad["wall_seconds"] = 99.
        base.write_json(folder / "completed.json", bad)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            protocol.load_completed(self.output, task)
        bad = {k: v for k, v in deepcopy(good).items() if k != "artifact_fingerprint"}
        bad["environment"]["environment"]["CUDA_VISIBLE_DEVICES"] = "0"
        base.write_json(folder / "completed.json", protocol.seal_artifact(bad))
        with self.assertRaisesRegex(ValueError, "physical GPU"):
            protocol.load_completed(self.output, task)

    def test_GPU_binding_uses_available_UUIDs_and_rejects_missing_or_contradictory_inventory(self):
        metadata = self.worker_metadata()
        protocol.validate_gpu_environment(metadata, GPU)
        # Old Torch legitimately omits UUID; exact full-UUID mask and matching
        # one-device logical/physical inventory remain explicit binding evidence.
        metadata["actual_compute_device"]["uuid"] = None
        metadata["torch"]["logical_cuda_devices"][0].pop("uuid")
        protocol.validate_gpu_environment(metadata, GPU)
        mutations = (
            lambda m: m["actual_compute_device"].update(uuid="GPU-other"),
            lambda m: m["actual_compute_device"].update(logical_device="cuda:1"),
            lambda m: m["torch"]["logical_cuda_devices"][0].update(uuid="GPU-other"),
            lambda m: m["torch"].update(logical_cuda_devices=[]),
            lambda m: m["nvidia"].update(gpus=[]),
            lambda m: m["nvidia"]["gpus"][0].update(name="other"),
        )
        for mutate in mutations:
            changed = deepcopy(metadata)
            mutate(changed)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                protocol.validate_gpu_environment(changed, GPU)


if __name__ == "__main__":
    unittest.main()
