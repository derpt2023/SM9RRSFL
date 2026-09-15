"""Offline audit fixtures: candidate identity and recorded decision attribution."""
import csv
import contextlib
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sm9rrsfl import fl
from sm9rrsfl.experiments import write_result_files


def recorded_result(*, ratio, warning, seed=402):
    config = fl.ExperimentConfig(
        method="sm9rrs", num_clients=20, rounds=6, detector_window=3,
        attack_start_round=4, malicious_ratio=ratio, seed=seed,
        detector_distance_threshold=warning, suspicion_remove_after=3,
        compute_backend="torch", device="cuda:7", partition="dirichlet")
    identities = [f"client-{i}" for i in range(config.num_clients)]
    malicious = set(fl._choose_malicious(identities, ratio, seed))
    leaking = set(sorted(malicious)[:2])
    revoked = set()
    counts = dict.fromkeys(identities, 0.)
    diagnostics, records = [], []
    records.append(fl.RoundRecord("sm9rrs", ratio, 0, .5, .5, 0, 0, 0, 0, 0, "",
                                  attack_target_success_rate=.1, attack_target_confidence=.1))
    for rd in range(1, 7):
        active = [identity for identity in identities if identity not in revoked]
        round_rows = []
        for identity in active:
            attack = bool(malicious and rd >= 4)
            immediate = not malicious and identity == "client-0" and rd == 4
            drift_only = not malicious and identity == "client-2" and rd >= 4
            suspicious = rd >= 4 and (
                identity in malicious - leaking if malicious else identity in {"client-0", "client-1", "client-2"})
            before = counts[identity]
            counts[identity] = before + 1 if suspicious else before * .5
            removed = bool(suspicious and (immediate or counts[identity] >= 3))
            if removed:
                revoked.add(identity)
            score = 7. if immediate else 1. if drift_only else warning + .5 if suspicious else .8
            coefficient = 0. if suspicious else 1. / (12 if attack else 20)
            item = fl.ClientDiagnosticRecord(
                round=rd, client_id=identity, task_tag="private-tag-" + identity,
                is_malicious=identity in malicious,
                decision_reason="strong_novelty" if immediate else "suspicious" if suspicious else "normal",
                suspicious=suspicious, count_increment=suspicious,
                weight_before=1., weight_after_penalty_recovery=.5 if suspicious else 1.,
                aggregation_weight=coefficient, count_before=before, count_after=counts[identity],
                trace_requested=removed, trace_pending=False, revoked=removed,
                novelty_score=score, anchor_score=score, signed_score=score,
                class_score=.5, cumulative_drift=7. if drift_only else 0., clip_factor=1.,
                aggregation_accepted=not suspicious, history_eligible=not suspicious,
                history_admitted=not suspicious, history_frozen=False,
                immediate_revocation=immediate, trusted_history_size=min(rd - 1, 3),
                normal_cluster_count=1, attack_active=attack and identity in malicious,
                recovery_eligible=not suspicious, norm_score=.2)
            round_rows.append(item)
        diagnostics.extend(round_rows)
        weights = {d.client_id: d.aggregation_weight for d in round_rows}
        honest_loss, malicious_mass = fl._aggregation_weight_diagnostics(
            {identity: 1 for identity in identities}, malicious, weights)
        accepted = sum(d.aggregation_weight > 0 for d in round_rows)
        asr = .1 if rd < 4 else .4 if malicious else .1
        accuracy = .5 if rd < 4 else .4 if malicious else .49
        records.append(fl.RoundRecord(
            "sm9rrs", ratio, rd, accuracy, 1 - accuracy, accepted,
            len(active) - accepted, len(revoked), len(revoked & malicious),
            len(revoked - malicious), "", attack_target_success_rate=asr,
            attack_target_confidence=.2, attack_active=bool(malicious and rd >= 4),
            malicious_weight_mass=malicious_mass, honest_weight_loss=honest_loss))
    return fl.ExperimentResult(config, records, records[-1].accuracy,
                               records[-1].error, 6, tuple(sorted(malicious)),
                               tuple(sorted(revoked)), diagnostics=diagnostics)


def write_fixture(root, mapped):
    results = [result for _, result in mapped]
    entries = [{"candidate_id": cid, "method": result.config.method,
                "config": asdict(replace(result.config, device="cuda:3", sm9_workers=4))}
               for cid, result in reversed(mapped)]
    manifest = {"tuning_phase": "validation", "dataset": {"name": "cifar10"},
                "tuning_context": {}, "candidates": entries,
                "configs": [entry["config"] for entry in entries]}
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
    manifest["fingerprint"] = fingerprint
    state = root / ".tuning_state" / "validation" / fingerprint
    state.mkdir(parents=True)
    (state / "run_manifest.json").write_text(json.dumps(manifest))
    (root / "tuning_progress.json").write_text(json.dumps({"phases": {"validation": {
        "status": "complete", "fingerprint": fingerprint, "total": len(results),
        "completed": len(results)}}}))
    write_result_files(state, results)
    rows = [{"candidate_id": cid, **result.summary_dict()} for cid, result in mapped]
    with (root / "validation_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return state


class OursAuditTest(unittest.TestCase):
    def setUp(self):
        import audit_cifar_ours_failures as audit
        self.audit = audit
        self.clean = recorded_result(ratio=0., warning=1.25)
        self.attack = recorded_result(ratio=.5, warning=1.75)

    def test_config_mapping_does_not_depend_on_candidate_order_or_cuda_index(self):
        # Same scenario, different defense; a scenario-only join would collide.
        other = replace(self.attack, config=replace(self.attack.config,
                                                    detector_distance_threshold=2.5))
        mapped = [("sm9rrs-003", self.attack), ("sm9rrs-005", other),
                  ("sm9rrs-001", self.clean)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root, mapped)
            _, loaded = self.audit.load_recorded_results(root)
        by_id = dict(loaded)
        self.assertEqual(set(by_id), {cid for cid, _ in mapped})
        self.assertEqual(by_id["sm9rrs-003"].config.detector_distance_threshold, 1.75)
        self.assertEqual(by_id["sm9rrs-005"].config.detector_distance_threshold, 2.5)

    def test_corrupt_snapshot_cannot_silently_fall_back_to_ambiguous_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = write_fixture(root, [("sm9rrs-003", self.attack)])
            (state / ".completed_results.pickle").write_bytes(b"not a pickle")
            with self.assertRaises((ValueError, FileNotFoundError, RuntimeError)):
                self.audit.load_recorded_results(root)

    def test_summary_csv_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root, [("sm9rrs-003", self.attack)])
            path = root / "validation_results.csv"
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["final_accuracy"] = "0.99"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                self.audit.load_recorded_results(root)

    def test_report_attributes_revocations_and_counts_updates_without_training(self):
        mapped = [("sm9rrs-001", self.clean), ("sm9rrs-003", self.attack)]
        with mock.patch.object(fl, "run_experiment", side_effect=AssertionError("must not train")):
            report = self.audit.build_audit(mapped)
        summary, tables = report["summary"], report["tables"]
        self.assertEqual(summary["attack_cases"], 1)
        self.assertEqual(summary["clean_cases"], 1)
        self.assertEqual(summary["accepted_malicious"]["all"], 6)
        self.assertEqual(summary["accepted_malicious"]["early"], 6)
        self.assertEqual(summary["accepted_malicious"]["first_attack_round"], 2)
        events = tables["false_revocation_events"]
        self.assertEqual(len(events), 3)
        self.assertEqual(summary["false_revocations_by_path"], {
            "immediate_revocation": 1, "count_threshold": 2, "unknown": 0})
        self.assertEqual(summary["false_revocation_score_evidence"], {
            "novelty_only": 2, "drift_only": 1})
        self.assertEqual(tables["clean_scenarios"][0][
            "first_false_revocation_rate_above_10pct_round"], 6)
        self.assertEqual(tables["attack_scenarios"][0]["peak_asr"], .4)
        self.assertNotIn("private-tag-", json.dumps(report))
        self.assertEqual(len(tables["attack_scenarios"]), 1)

    def test_missing_client_diagnostics_are_not_reported_as_no_leakage(self):
        changed = replace(self.attack, diagnostics=self.attack.diagnostics[:-1])
        with self.assertRaises(ValueError):
            self.audit.build_audit([("sm9rrs-001", self.clean), ("sm9rrs-003", changed)])

    def test_duplicate_client_diagnostics_are_rejected(self):
        changed = replace(self.attack, diagnostics=self.attack.diagnostics + [self.attack.diagnostics[0]])
        with self.assertRaises(ValueError):
            self.audit.build_audit([("sm9rrs-001", self.clean), ("sm9rrs-003", changed)])

    def test_report_writes_only_fresh_output_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            source, output = root / "source", root / "audit"
            write_fixture(source, [("sm9rrs-001", self.clean), ("sm9rrs-003", self.attack)])
            before = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in source.rglob("*") if p.is_file()}
            self.audit.main(["--source-output", str(source), "--output-dir", str(output)])
            summary_path = output / "summary.json"
            summary_before = summary_path.read_bytes()
            self.assertNotIn("private-tag-", summary_before.decode())
            self.assertEqual(json.loads(summary_before)["attack_cases"], 1)
            self.assertTrue((output / "false_revocation_history.csv").is_file())
            with mock.patch.object(self.audit, "load_recorded_results") as load:
                with self.assertRaises(ValueError):
                    self.audit.main(["--source-output", str(source), "--output-dir", str(output)])
                with self.assertRaises(ValueError):
                    self.audit.main(["--source-output", str(source), "--output-dir", str(source / "new")])
                load.assert_not_called()
            self.assertEqual(summary_path.read_bytes(), summary_before)
            after = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source.rglob("*") if p.is_file()}
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
