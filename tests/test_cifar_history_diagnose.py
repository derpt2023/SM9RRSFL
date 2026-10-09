"""Read-only history forensics: exact observations and a synthetic 90-task chain."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import diagnose_cifar_history as entry
import tests.test_cifar_cnn_history_protocol as fixtures
from tests.test_cifar_six_pipeline import synthetic_run

base, runtime, protocol = entry.base, entry.runtime, entry.protocol


def diagnostic(round_id, identity, *, malicious=False, tag="opaque", frozen=False, revoked=False):
    return base.fl.ClientDiagnosticRecord(
        round=round_id, client_id=identity, task_tag=tag, is_malicious=malicious,
        decision_reason="strong_novelty" if revoked else "normal", suspicious=revoked,
        count_increment=revoked, weight_before=1., weight_after_penalty_recovery=.1 if revoked else 1.,
        aggregation_weight=0. if revoked else .01, count_before=0., count_after=3. if revoked else 0.,
        trace_requested=revoked, trace_pending=False, revoked=revoked,
        novelty_score=99. if revoked else .25, anchor_score=.2, signed_score=.1,
        class_score=.1, cumulative_drift=0., clip_factor=1., aggregation_accepted=not revoked,
        history_eligible=not revoked, history_admitted=not (frozen or revoked),
        history_frozen=False, immediate_revocation=revoked,
        trusted_history_size=min(round_id - 1, 20), normal_cluster_count=0 if round_id <= 20 else 1,
        attack_active=malicious and round_id >= 25, recovery_eligible=False, norm_score=.1)


def complete_diagnostics(result, task):
    """Populate every online client, including each actual revocation-round observation."""
    malicious = list(result.malicious_clients)
    honest = ["honest-" + str(i) for i in range(result.config.num_clients - len(malicious))]
    diagnostics = []
    for rd in range(1, 151):
        previous, current = result.records[rd - 1], result.records[rd]
        for is_malicious, identities, prior, now in (
            (True, malicious, previous.true_positive_revocations, current.true_positive_revocations),
            (False, honest, previous.false_positive_revocations, current.false_positive_revocations),
        ):
            for index in range(prior, len(identities)):
                identity = identities[index]
                diagnostics.append(diagnostic(rd, identity, malicious=is_malicious,
                    tag=task["task_id"] + ":" + identity,
                    frozen=task["arm"] == "H1" and rd >= 25, revoked=index < now))
    last = result.records[-1]
    blacklist = tuple(malicious[:last.true_positive_revocations] + honest[:last.false_positive_revocations])
    records = [replace(r, blacklisted_clients=r.true_positive_revocations + r.false_positive_revocations,
        accepted_updates=result.config.num_clients - r.true_positive_revocations - r.false_positive_revocations)
        for r in result.records]
    return replace(result, diagnostics=diagnostics, blacklisted_clients=blacklist, records=records)


class FieldComparisonTests(unittest.TestCase):
    def test_exact_tail_difference_is_separate_from_discrete_accuracy_change(self):
        earlier = {1: SimpleNamespace(mass=.6977333333333333, accuracy=.232),
                   2: SimpleNamespace(mass=.6977333333333333, accuracy=.232)}
        later = {1: SimpleNamespace(mass=.6977333333333332, accuracy=.232),
                 2: SimpleNamespace(mass=.6977333333333333, accuracy=.2324)}
        result = entry.compare_fields(later, earlier, ("mass", "accuracy"))
        mass, accuracy = result["changed_fields"]
        self.assertEqual(mass[:3], ["mass", 1, [1, .6977333333333332, .6977333333333333]])
        self.assertIsNone(mass[3])
        self.assertGreater(mass[4], 0.)
        self.assertEqual(accuracy[3], [2, .2324, .232])
        self.assertEqual(result["matched_observations"], 2)

    def test_raw_tiny_overshoot_and_boolean_difference_are_not_rounded_away(self):
        raw = 1.0000000000000002
        result = entry.compare_fields({1: SimpleNamespace(mass=raw, revoked=True)},
            {1: SimpleNamespace(mass=1., revoked=False)}, ("mass", "revoked"))
        encoded = json.loads(json.dumps(result, allow_nan=False))
        self.assertEqual(encoded["changed_fields"][0][2][1], raw)
        self.assertEqual(encoded["changed_fields"][1][3], [1, True, False])
        self.assertIsNone(encoded["changed_fields"][1][4])

    def test_missing_keys_and_empty_observations_are_unavailable(self):
        result = entry.compare_fields({1: SimpleNamespace(value=1)}, {}, ("value",))
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["later_only_n"], 1)
        self.assertEqual(entry.compare_fields({}, {}, ("value",))["status"], "unavailable")

    def test_nonfinite_comparison_is_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonfinite"):
                entry.compare_fields({1: SimpleNamespace(value=value)},
                    {1: SimpleNamespace(value=0.)}, ("value",))

    def test_client_prefix_matches_public_identity_and_ignores_random_task_tags(self):
        config = base.fl.ExperimentConfig(method="sm9rrs", malicious_ratio=0., rounds=150,
                                         num_clients=100, attack_start_round=25, detector_window=20)
        earlier = replace(synthetic_run(config), diagnostics=[diagnostic(rd, "honest-" + str(i), tag="A")
                                                              for rd in range(1, 26) for i in range(100)])
        later = replace(earlier, diagnostics=[replace(d, task_tag="B", history_admitted=d.round < 25)
                                             for d in earlier.diagnostics])
        result = entry.pair_diagnostics(later, earlier)
        self.assertEqual(result["clients_pre25"]["status"], "audited")
        self.assertEqual(result["clients_pre25"]["matched_observations"], 2400)
        self.assertEqual(result["clients_pre25"]["changed_fields"], [])
        changed = replace(later, diagnostics=[replace(d, norm_score=.2) if (d.round, d.client_id) == (2, "honest-0") else d
                                             for d in later.diagnostics])
        first = entry.pair_diagnostics(changed, earlier)["clients_pre25"]["changed_fields"][0]
        self.assertEqual(first[:3], ["norm_score", 1, [(2, "honest-0"), .2, .1]])

    def test_symmetric_missing_prefix_clients_are_unavailable_despite_equal_keys(self):
        config = base.fl.ExperimentConfig(method="sm9rrs", malicious_ratio=0., rounds=150,
                                         num_clients=100, attack_start_round=25, detector_window=20)
        result = replace(synthetic_run(config), diagnostics=[diagnostic(rd, "honest-0") for rd in range(1, 25)])
        compared = entry.pair_diagnostics(result, result)["clients_pre25"]
        self.assertEqual(compared["changed_fields"], [])
        self.assertEqual(compared["status"], "unavailable")
        self.assertIn("incomplete", compared["reason"])
        for side in ("later", "earlier"):
            self.assertEqual(compared["coverage"][side], {"observed": 24, "remaining": 2400, "unobserved": 2376})

    def test_environment_uses_only_allowlisted_historical_values_and_never_probes(self):
        metadata = {"requested_device": "cuda:5", "secret": "DO_NOT_PRINT",
            "actual_compute_device": {"logical_device": "cuda:5", "uuid": "GPU-historic", "name": "4090D",
                                      "compute_capability": [8, 9], "secret": "DO_NOT_PRINT"},
            "torch": {"deterministic_algorithms": False, "version": "historic", "secret": "DO_NOT_PRINT"},
            "environment": {"PYTHONHASHSEED": None, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                            "API_KEY": "DO_NOT_PRINT"}}
        saved = deepcopy(metadata)
        with mock.patch.object(base, "worker_environment", side_effect=AssertionError("CUDA probe")):
            result = entry.recorded_environment(metadata)
        self.assertEqual(result["device"]["uuid"], "GPU-historic")
        self.assertFalse(result["torch"]["deterministic_algorithms"])
        self.assertEqual(result["torch"]["version"], "historic")
        self.assertNotIn("DO_NOT_PRINT", json.dumps(result))
        self.assertEqual(metadata, saved)


class HistoryDiagnoseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = fixtures.HistoryFixture()
        cls.fx.setUp()
        cls.addClassCleanup(cls.fx.doCleanups)
        # Upgrade only synthetic references, then freeze their real new evidence hashes.
        for task in cls.fx.threshold_tasks:
            if task["arm"] != "P0":
                continue
            folder = cls.fx.threshold_output / "tasks" / task["task_id"]
            result = base.checked_completed(cls.fx.threshold_output, task)
            base.experiments._write_completed_results_snapshot(folder, [complete_diagnostics(result, task)])
        cls.fx.reference = protocol.audit_reference(cls.fx.threshold_output, cls.fx.timing_output,
                                                     cls.fx.clean_output, cls.fx.matched_output)
        cls.fx.manifest = protocol.build_manifest(cls.fx.spec, cls.fx.reference)
        cls.fx.tasks = protocol.build_tasks(cls.fx.manifest)
        cls.fx.output = cls.fx.temporary_output()
        base.write_json(cls.fx.output / "manifest.json", cls.fx.manifest)
        runtime.save_plan(cls.fx.output, "history", cls.fx.tasks, cls.fx.manifest)
        base.write_json(cls.fx.output / "execution_environment.json", cls.fx.environment)
        cls.fx.patch_value(entry, "EXPECTED_MANIFEST", cls.fx.manifest["fingerprint"])
        for task in cls.fx.tasks:
            failed = task["arm"] == "H1" and task["config"]["partition"] == "dirichlet" and not task["config"]["malicious_ratio"]
            folder = cls.fx.finish(task, fp=49 if failed else 0)
            result = base.checked_completed(cls.fx.output, task)
            base.experiments._write_completed_results_snapshot(folder, [complete_diagnostics(result, task)])

    def paths(self):
        return (self.fx.output, self.fx.threshold_output, self.fx.timing_output,
                self.fx.clean_output, self.fx.matched_output)

    def all_hashes(self):
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for root in self.paths() for path in root.rglob("*") if path.is_file()}

    def readonly_guards(self):
        stack = ExitStack()
        for obj, name in ((base, "load_split"), (base, "worker_environment"),
                          (base.experiments, "run_measured_experiment"),
                          (base.experiments, "_load_round_checkpoint")):
            stack.enter_context(mock.patch.object(obj, name, side_effect=AssertionError("forbidden " + name)))
        return stack

    def test_real_90_task_chain_is_readonly_and_preserves_failed_health(self):
        before = self.all_hashes()
        with self.readonly_guards():
            records = entry.diagnose(*self.paths())
        self.assertEqual(before, self.all_hashes())
        header = records[0]
        self.assertEqual((header["status"], header["complete_tasks"], header["healthy_tasks"]), ("complete", 12, 11))
        self.assertEqual(header["recorded_source_count"], 66)
        self.assertTrue(header["input_evidence_unchanged"])
        self.assertTrue(header["source_reference_environment_verified"])
        for key in ("training_started", "experiment_files_written", "checkpoints_opened"):
            self.assertFalse(header[key])
        self.assertEqual(sum(r.get("type") == "prefix_pair" for r in records), 12)
        environments = next(r for r in records if r.get("type") == "recorded_environments")
        self.assertEqual(len(environments["tasks"]), 12)
        self.assertEqual(len(environments["profiles"]), 1)
        mechanisms = [r for r in records if "health" in r]
        self.assertEqual(len(mechanisms), 12)
        failed = next(r for r in mechanisms if r["arm"] == "H1" and r["p"] == "dirichlet" and not r["ratio"])
        self.assertFalse(failed["health"]["healthy"])
        self.assertEqual(failed["health"]["FP"], 49)
        self.assertIn("clean_false_revocation_rate", failed["health"]["reasons"])
        self.assertFalse(records[-1]["automatic_next_stage"])
        # Exercise the compact CLI serializer using the audited result from this exact read.
        output = io.StringIO()
        with mock.patch.object(entry, "diagnose", return_value=records), self.readonly_guards(), redirect_stdout(output):
            self.assertEqual(entry.main([]), 0)
        encoded = output.getvalue()
        self.assertTrue(encoded.startswith("=== CIFAR_HISTORY_FORENSICS_BEGIN ===\n"))
        self.assertTrue(encoded.endswith("=== CIFAR_HISTORY_FORENSICS_END ===\n"))
        for line in encoded.splitlines()[1:-1]:
            json.loads(line)
        self.assertLess(len(encoded.encode()), 50000)
        type(self).complete_output_bytes = len(encoded.encode())

    def test_reference_changes_and_missing_completed_evidence_are_rejected(self):
        old = self.fx.reference["p0_tasks"][0]
        path = self.fx.threshold_output / "tasks" / old["task_id"] / base.experiments.COMPLETED_RESULTS_SNAPSHOT
        original = path.read_bytes()
        try:
            result = base.checked_completed(self.fx.threshold_output, old)
            base.experiments._write_completed_results_snapshot(path.parent, [replace(result, final_accuracy=.4)])
            with self.readonly_guards(), self.assertRaises((ValueError, FileNotFoundError)):
                entry.diagnose(*self.paths())
            path.unlink()
            with self.readonly_guards(), self.assertRaises((ValueError, FileNotFoundError)):
                entry.diagnose(*self.paths())
        finally:
            path.write_bytes(original)

    def test_wrong_manifest_and_unreviewed_identity_are_rejected(self):
        path = self.fx.output / "manifest.json"
        original = path.read_bytes()
        try:
            altered = deepcopy(self.fx.manifest)
            altered["fingerprint"] = "corrupt"
            base.write_json(path, altered)
            with self.readonly_guards(), self.assertRaisesRegex(ValueError, "fingerprint"):
                entry.diagnose(*self.paths())
        finally:
            path.write_bytes(original)
        with mock.patch.object(entry, "EXPECTED_MANIFEST", "another-study"), self.readonly_guards():
            with self.assertRaisesRegex(ValueError, "reviewed"):
                entry.diagnose(*self.paths())

    def test_concurrent_input_change_is_rejected_after_reading(self):
        actual, calls = entry.evidence_hashes, []
        def changed(*args):
            value = actual(*args)
            if calls:
                value["changed during read"] = "changed"
            calls.append(True)
            return value
        with mock.patch.object(entry, "evidence_hashes", side_effect=changed), self.readonly_guards():
            with self.assertRaisesRegex(ValueError, "changed while"):
                entry.diagnose(*self.paths())

    def test_reader_version_changing_during_read_is_rejected(self):
        with mock.patch.object(entry, "reader_hashes", side_effect=[{"reader": "before"}, {"reader": "after"}]), \
                self.readonly_guards(), self.assertRaisesRegex(ValueError, "changed while"):
            entry.diagnose(*self.paths())

    def test_cli_error_is_short_framed_exit_two_without_training_or_cuda(self):
        flags = ("--output", "--threshold-output", "--timing-output", "--clean-output", "--matched-output")
        argv = [item for flag, path in zip(flags, self.paths()) for item in (flag, str(path))]
        output = io.StringIO()
        with mock.patch.object(entry.protocol, "read_study", side_effect=ValueError("missing manifest")), \
                self.readonly_guards(), redirect_stdout(output):
            self.assertEqual(entry.main(argv), 2)
        lines = output.getvalue().splitlines()
        self.assertEqual(lines[0], "=== CIFAR_HISTORY_FORENSICS_BEGIN ===")
        self.assertEqual(lines[-1], "=== CIFAR_HISTORY_FORENSICS_END ===")
        header = json.loads(lines[1])
        self.assertEqual(header["status"], "unavailable_or_invalid_evidence")
        self.assertIn("missing manifest", header["error"])
        self.assertFalse(header["training_started"])
        self.assertLess(len(output.getvalue()), 1500)


if __name__ == "__main__":
    unittest.main()
