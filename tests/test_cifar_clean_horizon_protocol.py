"""Fixed horizon identities and provenance around serialized 111-task ancestry.

The old fixture is composed, not inherited. Only the numerical reader boundary
is synthesized: the old manifest, task plans, sealed four-task anchors and
ancestor files are real JSON/NPZ evidence handled by their original protocols.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cifar_clean_horizon_protocol as protocol
from tests import test_cifar_mechanism_protocol as prior

base, runtime, mechanism = protocol.base, protocol.runtime, protocol.mechanism


class CleanHorizonProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prior.MechanismFixture.setUpClass()
        cls.addClassCleanup(prior.MechanismFixture.doClassCleanups)
        cls.fixture = prior.MechanismFixture()
        cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.doCleanups)
        cls.mechanism_output = cls.fixture.output
        cls.mechanism_manifest = cls.fixture.manifest
        cls.mechanism_tasks = cls.fixture.tasks
        cls.artifacts = {}
        for task in cls.mechanism_tasks:
            artifact = mechanism.seal_artifact({"status": "complete", "task_fingerprint": task["fingerprint"],
                "gpu_uuid": prior.prior.GPU, "environment": deepcopy(cls.fixture.prepared.metadata),
                "observations": {"public_synthetic_observation": task["task_id"]}, "wall_seconds": .1,
                "fresh_start": True, "checkpoints_used": False, "completed_training_rounds": 30,
                "requested_training_rounds": 30})
            cls.artifacts[task["task_id"]] = artifact
            base.write_json(cls.mechanism_output / "tasks" / task["task_id"] / "completed.json", artifact)
        detail = {"type": "paired_details", "schema": "synthetic-reviewed-reader-boundary", "same_local_updates": True}
        cls.detail_sha = base.digest(detail)
        cls.records = [{"type": "header", "status": "complete", "manifest_fingerprint": cls.mechanism_manifest["fingerprint"],
            "same_gpu_uuid": cls.mechanism_manifest["same_gpu_uuid"], "source_count": len(cls.mechanism_manifest["source_sha256"]),
            "source_map_sha256": base.digest(cls.mechanism_manifest["source_sha256"]), "reader_sha256": protocol.reader.reader_hashes(),
            "source_reference_and_evidence_verified": True, "input_evidence_unchanged": True,
            "within_arm_all_observations_equal": True, "round26_local_training_equal": True,
            "detailed_pair_repeats_equal": True, "complete_tasks": 4, "available_pairs": 4},
            {"type": "legend"}, {**detail, "repeat": 1, "h0_task_id": cls.mechanism_tasks[0]["task_id"],
                "h1_task_id": cls.mechanism_tasks[1]["task_id"], "paired_detail_sha256": cls.detail_sha},
            {"type": "repeat_confirmation", "repeat": 2, "paired_detail_sha256": cls.detail_sha,
                "detailed_pair_equal_to_repeat1": True, "artifact_fingerprints": {
                    t["arm"]: cls.artifacts[t["task_id"]]["artifact_fingerprint"] for t in cls.mechanism_tasks if t["repeat"] == 2}},
            {"type": "decision", "automatic_next_stage": False}]
        # This initial audit really recomputes the original serialized ancestry.
        with mock.patch.object(protocol, "EXPECTED_MECHANISM", cls.mechanism_manifest["fingerprint"]), \
             mock.patch.object(protocol, "EXPECTED_DETAIL", cls.detail_sha), \
             mock.patch.object(protocol.reader, "diagnose", return_value=deepcopy(cls.records)):
            cls.reference = protocol.audit_reference(cls.mechanism_output)

    def setUp(self):
        for patcher in (mock.patch.object(protocol, "EXPECTED_MECHANISM", self.mechanism_manifest["fingerprint"]),
                        mock.patch.object(protocol, "EXPECTED_DETAIL", self.detail_sha),
                        mock.patch.object(protocol.reader, "diagnose", return_value=deepcopy(self.records)),
                        mock.patch.object(mechanism, "verify_reference")):
            patcher.start()
            self.addCleanup(patcher.stop)
        # The full old chain was checked above; the per-test mock avoids doing
        # the identical ancestor audit for each pure identity/corruption case.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "horizon"
        (self.output / "task_plans").mkdir(parents=True)
        self.manifest = protocol.build_manifest(deepcopy(self.reference))
        self.tasks = protocol.build_tasks(self.manifest)
        base.write_json(self.output / "manifest.json", self.manifest)
        base.write_json(self.output / "task_plans/clean_horizon.json", {
            "manifest_fingerprint": self.manifest["fingerprint"], "tasks": self.tasks})
        for task in self.tasks:
            folder = self.output / "tasks" / task["task_id"]
            folder.mkdir(parents=True)
            base.write_json(folder / "task.json", task)

    @contextmanager
    def changed_anchor(self, index=0):
        task = self.mechanism_tasks[index]
        path = self.mechanism_output / "tasks" / task["task_id"] / "completed.json"
        original = path.read_bytes()
        value = runtime.read_json(path)
        def save():
            value.pop("artifact_fingerprint", None)
            base.write_json(path, mechanism.seal_artifact(value))
        try:
            yield task, value, save
        finally:
            path.write_bytes(original)

    def reseal_manifest(self, value):
        value["fingerprint"] = base.digest({k: v for k, v in value.items() if k != "fingerprint"})
        base.write_json(self.output / "manifest.json", value)

    def test_real_serialized_anchor_audit_is_readonly_and_reader_is_recomputed(self):
        before = mechanism.evidence_hashes(self.mechanism_output)
        with self.fixture.readonly_guards():
            actual = protocol.audit_reference(self.mechanism_output)
        self.assertEqual(actual, self.reference)
        self.assertEqual(before, mechanism.evidence_hashes(self.mechanism_output))
        protocol.reader.diagnose.assert_called_once_with(self.mechanism_output.resolve())
        self.assertEqual(actual["reference_task_count"], 115)
        self.assertEqual(len(actual["mechanism_artifact_fingerprints"]), 4)
        self.assertEqual(protocol.read_study(self.output)[1], self.tasks)

    def test_four_fresh_eighty_round_tasks_change_only_horizon_and_keep_one_history_factor(self):
        self.assertEqual([(t["arm"], t["repeat"]) for t in self.tasks], [("H0", 1), ("H1", 1), ("H0", 2), ("H1", 2)])
        self.assertEqual(len({t["fingerprint"] for t in self.tasks}), 4)
        for task, old in zip(self.tasks, self.mechanism_tasks):
            self.assertEqual(task["config"], {**old["config"], "rounds": 80})
            self.assertEqual(task["same_gpu_uuid"], old["same_gpu_uuid"])
            self.assertEqual(task["candidate"], old["candidate"])
            self.assertEqual(task["policy"], protocol.POLICY)
            self.assertEqual(task["history_freeze_start_round"], 25 if task["arm"] == "H1" else None)
            self.assertEqual(task["original_mechanism_task_fingerprint"], old["fingerprint"])
            self.assertNotEqual(task["fingerprint"], old["fingerprint"])
        self.assertEqual((self.manifest["rounds"], self.manifest["maximum_training_rounds"],
                          self.manifest["maximum_client_training_calls"]), (80, 320, 32000))
        self.assertTrue(self.manifest["fresh_process_per_task"])
        self.assertTrue(self.manifest["serial_same_physical_gpu"])
        for field in ("checkpoints_reused", "automatic_next_stage", "formal_qualification_assessed"):
            self.assertIs(self.manifest[field], False)
        self.assertEqual(self.manifest["late_windows"], [[31, 51], [52, 80]])

    def test_source_map_keeps_all_ninety_three_and_adds_only_three_modules(self):
        frozen = {**mechanism.source_hashes(), **protocol.reader.reader_hashes()}
        current = protocol.source_hashes()
        self.assertEqual((len(frozen), len(current)), (93, 96))
        self.assertEqual({name: current[name] for name in frozen}, frozen)
        self.assertEqual(set(current) - set(frozen), set(protocol.NEW_SOURCES))
        self.assertEqual(self.reference["frozen_source_sha256"], frozen)

    def test_current_source_identity_change_blocks_existing_study(self):
        changed = {**self.manifest["source_sha256"], "run_cifar_clean_horizon_panel.py": "0" * 64}
        with mock.patch.object(protocol, "source_hashes", return_value=changed), \
             self.assertRaisesRegex(ValueError, "source/configuration identity changed"):
            protocol.read_study(self.output)

    def test_old_source_or_reader_sha_change_is_not_rebound_by_new_manifest(self):
        for name in (next(iter(mechanism.source_hashes())), "diagnose_cifar_mechanism.py"):
            with self.subTest(name=name):
                reference = deepcopy(self.reference)
                reference["frozen_source_sha256"][name] = "0" * 64
                with self.assertRaisesRegex(ValueError, "frozen 93 source identity changed"):
                    protocol.build_manifest(reference)

    def test_wrong_reviewed_mechanism_or_detail_digest_blocks_specification(self):
        for field in ("mechanism_manifest_fingerprint", "reviewed_detail_sha256"):
            reference = deepcopy(self.reference)
            reference[field] = "0" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "reviewed thirty-round"):
                protocol.build_manifest(reference)

    def test_task_configuration_policy_gpu_and_history_cannot_change(self):
        changes = (
            lambda t: t["config"].update(rounds=80), lambda t: t["config"].update(seed=7),
            lambda t: t["config"].update(malicious_ratio=.1), lambda t: t["config"].update(batch_size=49),
            lambda t: t.update(policy="original"), lambda t: t.update(same_gpu_uuid="GPU-other"),
            lambda t: t.update(history_freeze_start_round=26),
            lambda t: t["candidate"].update(variant="NoPermanent"),
        )
        for change in changes:
            with self.subTest(change=change):
                manifest = deepcopy(self.manifest)
                change(manifest["reference"]["mechanism_tasks"][1])
                with self.assertRaises(ValueError):
                    protocol.build_tasks(manifest)

    def test_horizon_or_task_order_cannot_be_resealed_as_an_accepted_protocol(self):
        changed = deepcopy(self.manifest)
        changed["rounds"] = 150
        self.reseal_manifest(changed)
        with self.assertRaisesRegex(ValueError, "source/configuration identity changed"):
            protocol.read_study(self.output)
        base.write_json(self.output / "manifest.json", self.manifest)
        plan = runtime.read_json(self.output / "task_plans/clean_horizon.json")
        plan["tasks"].reverse()
        base.write_json(self.output / "task_plans/clean_horizon.json", plan)
        with self.assertRaisesRegex(ValueError, "task plan differs"):
            protocol.read_study(self.output)

    def test_all_ten_reference_roots_are_separate_and_non_nested(self):
        roots = protocol.reference_paths(self.reference)
        self.assertEqual(len(roots), 10)
        for root in roots:
            for output in (Path(root), Path(root) / "nested", Path(root).parent):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "non-nested"):
                    protocol.separate_outputs(output, roots[0], roots[1:])
        self.assertNotEqual(protocol.DEFAULT_OUTPUT, protocol.DEFAULT_MECHANISM)

    def test_horizon_plan_and_artifacts_are_hashed_without_checkpoint_reads(self):
        task = self.tasks[0]
        folder = self.output / "tasks" / task["task_id"]
        artifact = protocol.seal_artifact({"status": "complete", "task_fingerprint": task["fingerprint"],
            "observations": {}, "fresh_start": True, "checkpoints_used": False})
        base.write_json(folder / "completed.json", artifact)
        (folder / "checkpoint.pt").write_bytes(b"private not evidence")
        before = protocol.evidence_hashes(self.output)
        self.assertIn("task_plans/clean_horizon.json", before)
        self.assertIn("tasks/" + task["task_id"] + "/completed.json", before)
        self.assertFalse(any("checkpoint" in name for name in before))
        plan = self.output / "task_plans/clean_horizon.json"
        plan.write_bytes(plan.read_bytes() + b"\n")
        self.assertNotEqual(before, protocol.evidence_hashes(self.output))

    def test_pasted_success_cannot_replace_missing_or_nonfresh_actual_anchor(self):
        for field, value in (("fresh_start", False), ("checkpoints_used", True), ("completed_training_rounds", 29)):
            with self.subTest(field=field), self.changed_anchor() as (_, artifact, save):
                artifact[field] = value
                save()
                with self.assertRaisesRegex(ValueError, "missing or nonfresh full mechanism anchor"):
                    protocol.audit_reference(self.mechanism_output)
        with self.changed_anchor() as (task, _, _):
            path = self.mechanism_output / "tasks" / task["task_id"] / "completed.json"
            path.unlink()
            with self.assertRaisesRegex(ValueError, "missing or nonfresh full mechanism anchor"):
                protocol.audit_reference(self.mechanism_output)

    def test_detail_actual_content_digest_not_just_repeated_claim_is_checked(self):
        records = deepcopy(self.records)
        records[2]["same_local_updates"] = False
        protocol.reader.diagnose.return_value = records
        with self.assertRaisesRegex(ValueError, "detail digest does not match actual content"):
            protocol.audit_reference(self.mechanism_output)

    def test_detail_success_flags_and_actual_source_reader_gpu_identity_are_required(self):
        changes = (
            lambda rows: rows[0].update(round26_local_training_equal=False),
            lambda rows: rows[0].update(source_map_sha256="0" * 64),
            lambda rows: rows[0].update(same_gpu_uuid="GPU-other"),
            lambda rows: rows[0].update(reader_sha256={}),
            lambda rows: rows[3].update(paired_detail_sha256="0" * 64),
            lambda rows: rows[3].update(detailed_pair_equal_to_repeat1=False),
            lambda rows: rows.pop(),
        )
        for change in changes:
            with self.subTest(change=change):
                rows = deepcopy(self.records)
                change(rows)
                protocol.reader.diagnose.return_value = rows
                with self.assertRaises(ValueError):
                    protocol.audit_reference(self.mechanism_output)

    def test_each_snapshot_map_race_blocks_reference_certification(self):
        baseline = protocol._snapshot(self.mechanism_output, self.mechanism_manifest["reference"])
        for field in ("sources", "readers", "mechanism", "upstream"):
            changed = deepcopy(baseline)
            changed[field] = {"changed": "during-read"}
            with self.subTest(field=field), mock.patch.object(protocol, "_snapshot", side_effect=[baseline, changed]), \
                 self.assertRaisesRegex(ValueError, "changed during horizon audit"):
                protocol.audit_reference(self.mechanism_output)

    def test_current_old_source_failure_is_propagated_even_with_successful_reader(self):
        with mock.patch.object(mechanism, "read_study", side_effect=ValueError("old scientific source changed")), \
             self.assertRaisesRegex(ValueError, "old scientific source changed"):
            protocol.audit_reference(self.mechanism_output)

    def test_changed_reference_anchor_digest_blocks_verify_and_prefix_loading(self):
        with self.changed_anchor() as (_, artifact, save):
            artifact["wall_seconds"] += .1
            save()
            with self.assertRaisesRegex(ValueError, "frozen mechanism or upstream reference changed"):
                protocol.verify_reference(self.reference)
            with self.assertRaisesRegex(ValueError, "mechanism anchor changed or disappeared"):
                protocol.load_mechanism_anchors(self.reference)
        self.assertEqual(len(protocol.load_mechanism_anchors(self.reference)), 4)


if __name__ == "__main__":
    unittest.main()
