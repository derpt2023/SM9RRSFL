"""Read-only mechanism-reader integration, provenance and corruption checks.

The CPU observations are composed from the established report fixture rather
than inheriting its TestCase and silently rediscovering all its old tests.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import diagnose_cifar_mechanism as reader
from tests import test_cifar_mechanism_report as cpu_fixture


class MechanismDiagnoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cpu_fixture.MechanismReportTests.setUpClass()

    def setUp(self):
        self.fixture = cpu_fixture.MechanismReportTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.output = self.fixture.output
        self.tasks = self.fixture.tasks
        self.artifacts = self.fixture.artifacts
        self.manifest = self.fixture.manifest
        self.manifest["fingerprint"] = reader.EXPECTED_MANIFEST
        real_protocol = reader.protocol
        self.protocol = self.fixture.protocol
        self.protocol.runtime = SimpleNamespace(read_json=mock.Mock(return_value=self.manifest))
        self.protocol.base = real_protocol.base
        self.protocol.prefix = real_protocol.prefix
        self.protocol.DEFAULT_OUTPUT = self.output
        self.references = {name: {"manifest.json": "f" * 64} for name in (
            "deterministic_prefix", "tail", "step", "prefix", "old_history", "old_threshold",
            "old_timing", "old_clean", "old_matched")}
        for patcher in (
            mock.patch.object(reader, "protocol", self.protocol),
            mock.patch.object(reader, "reader_hashes", return_value={"reader.py": "b" * 64}),
            mock.patch.object(reader, "reference_evidence_hashes", return_value=self.references),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def payload(self, arm="H0", repeat=2):
        return self.artifacts[arm + "_" + str(repeat)]["observations"]

    def summary(self):
        value = reader.report.summarize(self.output)
        self.assertEqual(value["status"], "complete", value)
        return value

    def capture_main(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = reader.main(["--output", str(self.output)])
        lines = output.getvalue().splitlines()
        self.assertEqual(lines[0], "=== CIFAR_MECHANISM_DETAILS_BEGIN ===")
        self.assertEqual(lines[-1], "=== CIFAR_MECHANISM_DETAILS_END ===")
        return code, [json.loads(line) for line in lines[1:-1]], output.getvalue()

    def test_real_cpu_observations_produce_two_equal_detailed_pairs(self):
        records = reader.diagnose(self.output)
        self.assertEqual([r["type"] for r in records], ["header", "legend", "paired_details", "repeat_confirmation", "decision"])
        header = records[0]
        self.assertEqual(header["complete_tasks"], 4)
        self.assertEqual(header["reference_studies_guarded"], 9)
        self.assertTrue(header["detailed_pair_repeats_equal"])
        self.assertTrue(header["source_reference_and_evidence_verified"])
        self.assertTrue(header["input_evidence_unchanged"])
        paired = records[2]
        self.assertTrue(paired["round25"]["common_precommit_equal"])
        self.assertEqual(paired["round25"]["H1"]["admitted"], 0)
        self.assertTrue(paired["round26"]["local_training_equality"]["equal"])
        self.assertEqual(paired["round26"]["coverage"]["H0"]["active_clients"], 3)
        self.assertEqual(len(paired["rounds27_30"]), 4)
        self.assertTrue(records[3]["detailed_pair_equal_to_repeat1"])
        self.assertEqual(records[3]["paired_detail_sha256"], paired["paired_detail_sha256"])
        self.assertEqual(records[3]["artifact_fingerprints"], {
            arm: self.artifacts[arm + "_2"]["artifact_fingerprint"] for arm in ("H0", "H1")})
        self.protocol.read_study.assert_called_with(self.output, current_sources=True)
        self.assertFalse(records[-1]["adopt_frozen_history"])

    def test_cli_has_five_public_json_records_and_never_trains_queries_gpu_or_writes(self):
        from sm9rrsfl import fl
        import numpy as np
        import torch
        import subprocess
        with ExitStack() as stack:
            for target, name in ((fl, "run_experiment"), (reader.protocol.base, "load_split"),
                                 (subprocess, "run"), (subprocess, "Popen"), (torch, "load"),
                                 (torch.cuda, "is_available"), (np, "load"),
                                 (Path, "write_text"), (Path, "write_bytes"), (Path, "mkdir")):
                stack.enter_context(mock.patch.object(target, name, side_effect=AssertionError(name + " forbidden")))
            code, records, text = self.capture_main()
        self.assertEqual(code, 0, text)
        self.assertEqual(len(records), 5)
        for flag in ("training_started", "experiment_files_written", "checkpoints_opened", "gpu_queried"):
            self.assertIs(records[0][flag], False)
        self.assertFalse(records[-1]["automatic_next_stage"])

    def test_missing_artifact_is_not_fabricated_as_unhealthy_completion(self):
        self.artifacts.pop("H0_2")
        code, records, text = self.capture_main()
        self.assertEqual(code, 2, text)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "unavailable_or_invalid_evidence")
        self.assertNotIn("healthy", records[0])
        self.assertNotIn("paired_details", text)

    def test_all_artifacts_missing_has_no_fabricated_zero_values(self):
        self.artifacts.clear()
        code, records, text = self.capture_main()
        self.assertEqual(code, 2, text)
        self.assertEqual(len(records), 1)
        self.assertNotIn("round26", records[0])
        self.assertNotIn("source_reference_and_evidence_verified", records[0])

    def test_wrong_reviewed_manifest_is_rejected_before_detail_computation(self):
        self.manifest["fingerprint"] = "wrong-manifest"
        with mock.patch.object(reader.details, "analyze_pair", side_effect=AssertionError("not eligible")):
            with self.assertRaisesRegex(ValueError, "reviewed clean mechanism manifest"):
                reader.diagnose(self.output)

    def test_summary_wrong_source_map_count_or_uuid_is_rejected(self):
        original = self.summary()
        for field, value in (("source_map_sha256", "0" * 64), ("source_count", 91), ("same_gpu_uuid", "GPU-other")):
            with self.subTest(field=field):
                summary = deepcopy(original)
                summary[field] = value
                with self.assertRaisesRegex(ValueError, "summary source map or physical GPU"):
                    reader._reviewed_summary(summary, self.manifest, self.tasks)

    def test_current_frozen_source_validation_failure_is_propagated(self):
        self.protocol.read_study.side_effect = ValueError("mechanism source/configuration identity changed")
        with self.assertRaisesRegex(ValueError, "source/configuration identity"):
            reader.diagnose(self.output)
        self.protocol.read_study.assert_called_once_with(self.output, current_sources=True)

    def test_within_arm_gradient_divergence_blocks_detailed_interpretation(self):
        batch = next(b for b in self.payload()["singleton_batches"] if b["round"] == 27)
        batch["gradients"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "repeatability or paired common conditions"):
            reader.diagnose(self.output)

    def test_round25_common_training_divergence_blocks_detailed_interpretation(self):
        for repeat in (1, 2):
            batch = next(b for b in self.payload("H1", repeat)["singleton_batches"] if b["round"] == 25)
            batch["gradients"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "repeatability or paired common conditions"):
            reader.diagnose(self.output)

    def test_round26_still_common_local_training_divergence_blocks_interpretation(self):
        for repeat in (1, 2):
            batch = next(b for b in self.payload("H1", repeat)["singleton_batches"] if b["round"] == 26)
            batch["gradients"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "repeatability or paired common conditions"):
            reader.diagnose(self.output)

    def test_unreproduced_historical_prefix_blocks_interpretation(self):
        for anchor in self.manifest["reference"]["deterministic_prefix_anchors"]:
            anchor["observations"]["initial_model"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "repeatability or paired common conditions"):
            reader.diagnose(self.output)

    def test_missing_review_flag_or_incorrect_pair_identity_is_not_assumed_true(self):
        original = self.summary()
        mutations = (
            lambda s: s["decision"].pop("round26_local_training_equal_before_first_affected_detection"),
            lambda s: s["pairs"][0].update(later_repeat=3),
            lambda s: s["pairs"][0].update(available=False),
            lambda s: s["rows"][0].update(status="missing"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                summary = deepcopy(original)
                mutation(summary)
                with self.assertRaises(ValueError):
                    reader._reviewed_summary(summary, self.manifest, self.tasks)

    def test_artifact_disappearing_after_verified_summary_is_rejected(self):
        summary = self.summary()
        self.artifacts.pop("H0_1")
        with mock.patch.object(reader.report, "summarize", return_value=summary):
            with self.assertRaisesRegex(ValueError, "artifact changed or disappeared after summary"):
                reader.diagnose(self.output)

    def test_artifact_fingerprint_changing_after_verified_summary_is_rejected(self):
        summary = self.summary()
        self.artifacts["H0_1"]["artifact_fingerprint"] = "0" * 64
        with mock.patch.object(reader.report, "summarize", return_value=summary):
            with self.assertRaisesRegex(ValueError, "artifact changed or disappeared after summary"):
                reader.diagnose(self.output)

    def test_reader_source_race_removes_success_claim(self):
        reader.reader_hashes.side_effect = [{"reader.py": "before"}, {"reader.py": "after"}]
        with self.assertRaisesRegex(ValueError, "sources changed during read"):
            reader.diagnose(self.output)

    def test_scientific_source_race_after_summary_removes_success_claim(self):
        sources = self.manifest["source_sha256"]
        self.protocol.source_hashes.side_effect = [sources, sources, sources, {"frozen_science.py": "0" * 64}]
        with self.assertRaisesRegex(ValueError, "sources changed during read"):
            reader.diagnose(self.output)

    def test_mechanism_evidence_race_after_summary_removes_success_claim(self):
        before = {"evidence": "stable"}
        self.protocol.evidence_hashes.side_effect = [before, before, before, {"evidence": "changed"}]
        with self.assertRaisesRegex(ValueError, "sources changed during read"):
            reader.diagnose(self.output)

    def test_upstream_evidence_race_after_summary_removes_success_claim(self):
        changed = deepcopy(self.references)
        changed["step"]["tasks/client/attempts/one/snapshots.npz"] = "changed"
        reader.reference_evidence_hashes.side_effect = [self.references, changed]
        with self.assertRaisesRegex(ValueError, "sources changed during read"):
            reader.diagnose(self.output)

    def test_upstream_reference_validation_failure_has_no_partial_success(self):
        self.protocol.verify_reference.side_effect = ValueError("old NPZ evidence changed")
        code, records, text = self.capture_main()
        self.assertEqual(code, 2, text)
        self.assertEqual(len(records), 1)
        self.assertNotIn("paired_details", text)

    def test_nonfinite_stored_decision_is_invalid_evidence(self):
        event = next(e for e in self.payload()["history_events"] if e["round"] == 26)
        event["decision"]["novelty_score"] = float("nan")
        code, records, text = self.capture_main()
        self.assertEqual(code, 2, text)
        self.assertEqual(len(records), 1)
        self.assertNotIn("NaN", text)

    def test_nonfinite_detail_result_is_rejected_before_printing_records(self):
        with mock.patch.object(reader.details, "analyze_pair", return_value={"invalid": float("inf")}):
            code, records, text = self.capture_main()
        self.assertEqual(code, 2, text)
        self.assertEqual(len(records), 1)
        self.assertNotIn("Infinity", text)
        self.assertNotIn("paired_details", text)

    def test_unequal_repeated_numeric_details_are_rejected(self):
        with mock.patch.object(reader.details, "analyze_pair", side_effect=[{"delta": 1}, {"delta": 2}]):
            with self.assertRaisesRegex(ValueError, "detailed paired results differ"):
                reader.diagnose(self.output)

    def test_complete_unhealthy_summary_is_not_changed_to_missing_or_full_health(self):
        summary = self.summary()
        for row in summary["rows"]:
            row["diagnostics"]["health"]["healthy"] = False
            row["diagnostics"]["health"]["reasons"] = ["clean_false_revocation_rate"]
        reader._reviewed_summary(summary, self.manifest, self.tasks)
        with mock.patch.object(reader.report, "summarize", return_value=summary):
            records = reader.diagnose(self.output)
        self.assertEqual(records[0]["complete_tasks"], 4)
        self.assertNotIn("healthy", records[0])
        self.assertFalse(records[-1]["full_protocol_health_assessed"])

    def test_unused_secret_metadata_and_payload_are_never_exported(self):
        secret = "SECRET_ONLY_INPUT_NEVER_EXPORT_19aebb"
        self.manifest["reference"]["execution_environment"]["secret_key"] = secret
        for artifact in self.artifacts.values():
            artifact["environment"]["secret_key"] = secret
            artifact["observations"]["unused_private_material"] = secret
        code, _, text = self.capture_main()
        self.assertEqual(code, 0, text)
        for prohibited in (secret, "secret_key", "unused_private_material", "task_tag", "CUDA_VISIBLE_DEVICES"):
            self.assertNotIn(prohibited, text)

    def test_missing_study_cli_preserves_boundaries_without_creating_directory(self):
        self.protocol.runtime.read_json.side_effect = FileNotFoundError("missing manifest.json")
        with mock.patch.object(Path, "mkdir", side_effect=AssertionError("mkdir forbidden")), \
             mock.patch.object(Path, "write_text", side_effect=AssertionError("write forbidden")):
            code, records, text = self.capture_main()
        self.assertEqual(code, 2, text)
        self.assertEqual(len(records), 1)
        self.assertIn("missing manifest.json", records[0]["error"])


class MechanismReferenceCoverageTests(unittest.TestCase):
    def test_reference_hash_dispatch_uses_study_helpers_and_each_old_path(self):
        prefix = reader.protocol.prefix
        reference = {"deterministic_prefix_output": "/read/prefix-deterministic", "tail_output": "/read/tail",
            "step_output": "/read/step", "prefix_output": "/read/prefix",
            "prefix_manifest": {"reference": {"paths": {name: "/read/old-" + name
                for name in reader.old_protocol.REFERENCE_NAMES}}}}
        with mock.patch.object(prefix, "evidence_hashes", return_value={"deterministic": "d"}) as deterministic, \
             mock.patch.object(prefix.tail, "evidence_hashes", return_value={"tail": "t"}) as tail, \
             mock.patch.object(prefix.tail.step, "evidence_hashes", return_value={"step": "s"}) as step, \
             mock.patch.object(reader.old_report, "evidence_hashes", return_value={"prefix": "p"}) as original, \
             mock.patch.object(reader.old_protocol, "evidence_hashes", return_value={"old": "o"}) as old:
            values = reader.reference_evidence_hashes(reference)
        deterministic.assert_called_once_with(Path(reference["deterministic_prefix_output"]))
        tail.assert_called_once_with(Path(reference["tail_output"]))
        step.assert_called_once_with(Path(reference["step_output"]))
        original.assert_called_once_with(Path(reference["prefix_output"]))
        self.assertEqual(old.call_args_list, [mock.call(Path(path))
            for path in reference["prefix_manifest"]["reference"]["paths"].values()])
        self.assertEqual(len(values), 9)

    def test_real_reference_hashes_guard_old_npz_and_exclude_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            studies = {name: root / name for name in ("deterministic_prefix", "tail", "step", "prefix",
                *["old_" + name for name in reader.old_protocol.REFERENCE_NAMES])}
            for path in studies.values():
                path.mkdir()
                (path / "manifest.json").write_text("{}")
                (path / "execution_environment.json").write_text("{}")
                (path / "checkpoints").mkdir()
                (path / "checkpoints/private_checkpoint.pt").write_bytes(b"not scientific evidence")
            npzs = []
            for name in ("tail", "step"):
                npz = studies[name] / "tasks/one/attempts/one/snapshots.npz"
                npz.parent.mkdir(parents=True)
                npz.write_bytes(b"public prior snapshot evidence")
                npzs.append(npz)
            reference = {name + "_output": str(studies[name]) for name in ("deterministic_prefix", "tail", "step", "prefix")}
            reference["prefix_manifest"] = {"reference": {"paths": {name: str(studies["old_" + name])
                for name in reader.old_protocol.REFERENCE_NAMES}}}
            before = reader.reference_evidence_hashes(reference)
            self.assertEqual(len(before), 9)
            key = "tasks/one/attempts/one/snapshots.npz"
            for name in ("tail", "step"):
                self.assertEqual(before[name][key], hashlib.sha256(b"public prior snapshot evidence").hexdigest())
            self.assertFalse(any("checkpoint" in key for mapping in before.values() for key in mapping))
            for npz in npzs:
                npz.write_bytes(b"changed public evidence")
            after = reader.reference_evidence_hashes(reference)
            self.assertNotEqual(after["tail"][key], before["tail"][key])
            self.assertNotEqual(after["step"][key], before["step"][key])
            for name in reader.old_protocol.REFERENCE_NAMES:
                self.assertIn("execution_environment.json", before["old_" + name])


if __name__ == "__main__":
    unittest.main()
