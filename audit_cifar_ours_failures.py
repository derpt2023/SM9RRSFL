#!/usr/bin/env python3
"""Read-only audit of completed Ours validation results; never runs training.

Only load this project's own trusted local result snapshot. Candidate identity
comes from the full manifest configuration, never the unlabelled diagnostics CSV.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import fields
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import sm9rrsfl  # Set the package's CPU thread environment before importing NumPy.
import numpy as np
from sm9rrsfl.experiments import load_completed_results_snapshot
from sm9rrsfl.fl import ExperimentConfig
from sm9rrsfl.tuning_resume import runtime_config_key


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty required table: {Path(path).name}")
    return rows


def _csv_text(value):
    return "" if value is None else str(value)


def _csv_config(row):
    values = {}
    for field in fields(ExperimentConfig):
        if field.name not in row:
            raise ValueError(f"missing config column: {field.name}")
        raw = row[field.name]
        default = getattr(ExperimentConfig(), field.name)
        if isinstance(default, bool):
            if raw not in ("True", "False"):
                raise ValueError(f"invalid boolean config column: {field.name}")
            values[field.name] = raw == "True"
        else:
            values[field.name] = type(default)(raw)
    return values


def _unique_config_rows(rows, label):
    result = {}
    for row in rows:
        key = runtime_config_key(_csv_config(row))
        if key in result:
            raise ValueError(f"duplicate full configuration in {label}")
        result[key] = row
    return result


def load_recorded_results(source):
    """Return (manifest, [(candidate_id, ExperimentResult), ...]) after checks.

    Source is the tuning output root containing tuning_progress.json. Failure
    never falls back to training, datasets, or a candidate-ambiguous CSV.
    """
    source = Path(source).expanduser().resolve()
    phase = json.loads((source / "tuning_progress.json").read_text())["phases"]["validation"]
    fingerprint = phase["fingerprint"]
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(ch not in "0123456789abcdef" for ch in fingerprint)):
        raise ValueError("invalid validation fingerprint")
    if phase.get("status") != "complete" or phase.get("completed") != phase.get("total"):
        raise ValueError("validation phase must be complete")
    state = source / ".tuning_state" / "validation" / fingerprint
    manifest = json.loads((state / "run_manifest.json").read_text())
    payload = dict(manifest)
    payload.pop("fingerprint", None)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("fingerprint") != fingerprint or digest != fingerprint:
        raise ValueError("manifest fingerprint mismatch")
    if manifest.get("tuning_phase") != "validation":
        raise ValueError("source is not a validation manifest")
    planned = {}
    for entry in manifest["candidates"]:
        config = entry["config"]
        key = runtime_config_key(config)
        if key in planned:
            raise ValueError("candidate configurations are not unique")
        if entry.get("method") != config.get("method") or not entry.get("candidate_id"):
            raise ValueError("candidate identity is inconsistent")
        planned[key] = entry["candidate_id"]
    config_keys = [runtime_config_key(config) for config in manifest["configs"]]
    if len(set(config_keys)) != len(config_keys) or set(config_keys) != set(planned):
        raise ValueError("manifest configs and candidates disagree")
    if len(planned) != phase["total"]:
        raise ValueError("manifest and progress counts disagree")
    results = load_completed_results_snapshot(state)
    if results is None:
        raise ValueError("completed result snapshot missing/incompatible; diagnostics cannot be recovered safely from global CSV")
    summaries = _unique_config_rows(_read_csv(state / "summary.csv"), "summary.csv")
    validations = _unique_config_rows(_read_csv(source / "validation_results.csv"), "validation_results.csv")
    if set(summaries) != set(planned) or set(validations) != set(planned):
        raise ValueError("summary/validation configurations differ from manifest")
    seen = set()
    mapped = []
    for result in results:
        key = runtime_config_key(result.config)
        if key not in planned or key in seen:
            raise ValueError("snapshot contains an unknown or duplicate full configuration")
        seen.add(key)
        summary, validation = summaries[key], validations[key]
        if validation.get("candidate_id") != planned[key]:
            raise ValueError("validation candidate label disagrees with full manifest configuration")
        if any(validation.get(k) != value for k, value in summary.items()):
            raise ValueError("validation and source summary disagree")
        for name, value in result.summary_dict().items():
            if name in ("device", "sm9_workers"):
                continue
            if summary.get(name) != _csv_text(value):
                raise ValueError(f"snapshot/summary mismatch in {name}")
        rounds = [record.round for record in result.records]
        if (result.config.eval_interval != 1 or result.stopped_round != result.config.rounds
                or rounds != list(range(result.config.rounds + 1))):
            raise ValueError("completed snapshot must contain every configured round")
        if sum(record.nonfinite_updates for record in result.records) != result.nonfinite_updates:
            raise ValueError("snapshot nonfinite total mismatch")
        if not math.isclose(result.records[-1].accuracy, result.final_accuracy, abs_tol=1e-12):
            raise ValueError("snapshot final accuracy mismatch")
        mapped.append((planned[key], result))
    if seen != set(planned):
        raise ValueError("completed snapshot is missing planned configurations")
    return manifest, sorted(mapped, key=lambda pair: (pair[0], runtime_config_key(pair[1].config)))


def _identity(candidate, result):
    config = result.config
    return {"candidate_id": candidate, "partition": config.partition,
            "dirichlet_alpha": config.dirichlet_alpha, "num_clients": config.num_clients,
            "malicious_ratio": config.malicious_ratio, "seed": config.seed}


def _checked_diagnostics(result):
    """Check recorded coverage, identities and aggregation/revocation totals."""
    if not result.diagnostics:
        raise ValueError("selected Ours result lacks client diagnostics")
    by_round = defaultdict(list)
    seen = set()
    malicious = set(result.malicious_clients)
    for diagnostic in result.diagnostics:
        key = (diagnostic.round, diagnostic.client_id)
        if key in seen or not 1 <= diagnostic.round <= result.stopped_round:
            raise ValueError("duplicate/out-of-range client diagnostic")
        seen.add(key)
        if diagnostic.is_malicious != (diagnostic.client_id in malicious):
            raise ValueError("diagnostic malicious label mismatch")
        if diagnostic.aggregation_accepted != (diagnostic.aggregation_weight > 0):
            raise ValueError("diagnostic aggregation acceptance mismatch")
        numeric = SCORE_FIELDS + ("aggregation_weight", "count_before", "count_after")
        if any(not math.isfinite(getattr(diagnostic, name)) for name in numeric):
            raise ValueError("nonfinite diagnostic score/state in selected result")
        by_round[diagnostic.round].append(diagnostic)
    previous_blacklisted = 0
    revoked = set()
    initial_clients = {d.client_id for d in by_round[1]}
    for record in result.records:
        if record.round == 0:
            continue
        diagnostics = by_round[record.round]
        expected = result.config.num_clients - previous_blacklisted - record.nonfinite_updates
        if len(diagnostics) != expected:
            raise ValueError(f"incomplete diagnostics at round {record.round}: expected {expected}, found {len(diagnostics)}")
        observed = {d.client_id for d in diagnostics}
        if observed & revoked:
            raise ValueError("diagnostics include a previously revoked client")
        if result.records[1].nonfinite_updates == 0 and not observed <= initial_clients:
            raise ValueError("diagnostics introduce an unknown client identity")
        if sum(d.aggregation_accepted for d in diagnostics) != record.accepted_updates:
            raise ValueError("diagnostic accepted count disagrees with round metrics")
        if not math.isclose(sum(d.aggregation_weight for d in diagnostics if d.is_malicious),
                            record.malicious_weight_mass, abs_tol=1e-8):
            raise ValueError("diagnostic malicious weight disagrees with round metrics")
        revoked.update(d.client_id for d in diagnostics if d.revoked)
        if (len(revoked - malicious) != record.false_positive_revocations
                or len(revoked & malicious) != record.true_positive_revocations
                or len(revoked) != record.blacklisted_clients):
            raise ValueError("diagnostic permanent revocations disagree with round metrics")
        previous_blacklisted = record.blacklisted_clients
    if revoked != set(result.blacklisted_clients):
        raise ValueError("diagnostics disagree with final blacklist")
    return by_round


SCORE_FIELDS = ("novelty_score", "anchor_score", "signed_score", "class_score",
                "norm_score", "cumulative_drift", "clip_factor")
DIAGNOSTIC_FIELDS = ("round", "client_id", "decision_reason", "suspicious", "count_increment",
                     "count_before", "count_after", "immediate_revocation", "trace_requested",
                     "trace_pending", "revoked", "aggregation_accepted", "aggregation_weight",
                     "history_eligible", "history_admitted", "history_frozen") + SCORE_FIELDS


def _diagnostic_row(diagnostic, config):
    row = {key: getattr(diagnostic, key) for key in DIAGNOSTIC_FIELDS}
    novelty = diagnostic.novelty_score > config.detector_distance_threshold
    drift = diagnostic.cumulative_drift > config.detector_drift_threshold
    row.update(warning_threshold=config.detector_distance_threshold,
               reject_threshold=config.detector_reject_threshold,
               drift_threshold=config.detector_drift_threshold,
               reference_budget=config.detector_reference_budget,
               remove_after=config.suspicion_remove_after,
               score_evidence="both" if novelty and drift else "novelty_only" if novelty else "drift_only" if drift else "neither")
    return row


def _distribution(values):
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None}
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite diagnostic score in selected result")
    return {"count": len(values), "min": float(values.min()), "p50": float(np.quantile(values, .5)),
            "p90": float(np.quantile(values, .9)), "p99": float(np.quantile(values, .99)),
            "max": float(values.max())}


def build_audit(mapped, attack_candidate="sm9rrs-003", clean_candidate="sm9rrs-001"):
    """Build small summary and separate CSV-ready tables from verified results."""
    tables = {name: [] for name in ("attack_scenarios", "attack_rounds", "accepted_malicious_distributions",
                                  "accepted_malicious_examples", "clean_scenarios",
                                  "false_revocation_events", "false_revocation_history")}
    paths = Counter()
    evidence_counts = Counter()
    accepted_counts = Counter(first_attack_round=0, early=0, all=0)
    attack_cases = clean_cases = 0
    for candidate, result in mapped:
        config = result.config
        attack_case = candidate == attack_candidate and config.malicious_ratio > 0 and config.attack != "none"
        clean_case = candidate == clean_candidate and config.malicious_ratio == 0
        if not (attack_case or clean_case):
            continue
        if config.method != "sm9rrs":
            raise ValueError("selected candidate is not Ours")
        diagnostics = _checked_diagnostics(result)
        identity = _identity(candidate, result)
        if attack_case:
            attack_cases += 1
            start = config.attack_start_round or config.detector_window + 2
            records = [record for record in result.records if record.round >= start]
            if not records or any(record.attack_target_success_rate is None for record in records):
                raise ValueError("missing attack-period metrics")
            prefix = next((r for r in result.records if r.round == start - 1), None)
            if prefix is None:
                raise ValueError("missing preattack accuracy")
            early = [r for r in records if r.round < start + 5]
            tail = records[-10:]
            peak = max(r.attack_target_success_rate for r in records)
            recovery = next((r.round for r in records if r.accuracy >= prefix.accuracy), None)
            scenario = {**identity, "attack_start_round": start, "attack_rounds": len(records),
                        "pre_attack_accuracy": prefix.accuracy, "first_attack_accuracy": records[0].accuracy,
                        "attack_accuracy_mean": fmean(r.accuracy for r in records),
                        "attack_accuracy_min": min(r.accuracy for r in records),
                        "tail10_accuracy_mean": fmean(r.accuracy for r in tail),
                        "final_accuracy": records[-1].accuracy, "first_recovery_to_pre_attack_round": recovery,
                        "attack_asr_mean": fmean(r.attack_target_success_rate for r in records),
                        "tail10_asr_mean": fmean(r.attack_target_success_rate for r in tail),
                        "final_asr": records[-1].attack_target_success_rate, "peak_asr": peak,
                        "peak_asr_rounds": ",".join(str(r.round) for r in records if r.attack_target_success_rate == peak),
                        "early5_malicious_weight_mass_max": max(r.malicious_weight_mass for r in early),
                        "nonfinite_updates": result.nonfinite_updates}
            tables["attack_scenarios"].append(scenario)
            accepted_counts["early"] += sum(d.is_malicious and d.aggregation_accepted
                                           for r in early for d in diagnostics[r.round])
            for record in records:
                ds = diagnostics[record.round]
                tables["attack_rounds"].append({**identity, "round": record.round, "accuracy": record.accuracy,
                    "asr": record.attack_target_success_rate, "accepted_updates": record.accepted_updates,
                    "rejected_updates": record.rejected_updates, "nonfinite_updates": record.nonfinite_updates,
                    "accepted_malicious": sum(d.is_malicious and d.aggregation_accepted for d in ds),
                    "accepted_honest": sum(not d.is_malicious and d.aggregation_accepted for d in ds),
                    "malicious_weight_mass": record.malicious_weight_mass, "honest_weight_loss": record.honest_weight_loss,
                    "false_positive_revocations": record.false_positive_revocations,
                    "true_positive_revocations": record.true_positive_revocations})
            for scope, selected_rounds in (("first_attack_round", [start]), ("all_attack_rounds", [r.round for r in records])):
                accepted = [d for rd in selected_rounds for d in diagnostics[rd] if d.is_malicious and d.aggregation_accepted]
                accepted_counts["first_attack_round" if scope == "first_attack_round" else "all"] += len(accepted)
                for field in SCORE_FIELDS:
                    tables["accepted_malicious_distributions"].append({**identity, "scope": scope, "field": field,
                        **_distribution([getattr(d, field) for d in accepted]),
                        "warning_threshold": config.detector_distance_threshold,
                        "reject_threshold": config.detector_reject_threshold,
                        "drift_threshold": config.detector_drift_threshold,
                        "reference_budget": config.detector_reference_budget})
                for rank, diagnostic in enumerate(sorted(accepted, key=lambda d: (-d.aggregation_weight, d.round, d.client_id))[:10], 1):
                    tables["accepted_malicious_examples"].append({**identity, "scope": scope, "rank_by_weight": rank,
                                                                **_diagnostic_row(diagnostic, config)})
        if clean_case:
            clean_cases += 1
            client_history = defaultdict(list)
            revoked = set()
            case_paths = Counter()
            for rd in sorted(diagnostics):
                for diagnostic in diagnostics[rd]:
                    prior = client_history[diagnostic.client_id]
                    if diagnostic.revoked and diagnostic.client_id not in revoked:
                        revoked.add(diagnostic.client_id)
                        # The immediate route takes precedence when both conditions hold.
                        if diagnostic.trace_requested and diagnostic.immediate_revocation:
                            path = "immediate_revocation"
                        elif diagnostic.trace_requested and diagnostic.count_increment and diagnostic.count_after >= config.suspicion_remove_after:
                            path = "count_threshold"
                        else:
                            path = "unknown"
                        case_paths[path] += 1
                        paths[path] += 1
                        row = {**identity, "revocation_path": path, **_diagnostic_row(diagnostic, config)}
                        evidence_counts[row["score_evidence"]] += 1
                        tables["false_revocation_events"].append(row)
                        for earlier in prior[-5:]:
                            tables["false_revocation_history"].append({**identity, "event_round": rd,
                                "revocation_path": path, **_diagnostic_row(earlier, config)})
                    prior.append(diagnostic)
            crossing = next((r.round for r in result.records if r.false_positive_revocations / config.num_clients > .1), None)
            tables["clean_scenarios"].append({**identity, "first_false_revocation_rate_above_10pct_round": crossing,
                "final_false_revocations": len(revoked), "final_false_revocation_rate": len(revoked) / config.num_clients,
                "immediate_revocation": case_paths["immediate_revocation"], "count_threshold": case_paths["count_threshold"],
                "unknown": case_paths["unknown"], "final_accuracy": result.final_accuracy,
                "nonfinite_updates": result.nonfinite_updates})
    if not attack_cases or not clean_cases:
        raise ValueError("selected attack/clean candidate has no matching cases")
    ranking = sorted(tables["attack_scenarios"], key=lambda row: (-row["attack_asr_mean"], -row["peak_asr"]))[:5]
    brief_fields = ("candidate_id", "partition", "dirichlet_alpha", "num_clients", "malicious_ratio", "seed",
                    "round", "client_id", "revocation_path", "score_evidence", "decision_reason", "suspicious",
                    "count_increment", "count_before", "count_after", "immediate_revocation", "trace_requested",
                    "aggregation_weight", "warning_threshold", "reject_threshold", "drift_threshold", "remove_after") + SCORE_FIELDS
    def brief(row):
        return {key: row[key] for key in brief_fields if key in row}
    event_samples = []
    sampled_paths = set()
    identity_fields = ("candidate_id", "partition", "dirichlet_alpha", "num_clients", "malicious_ratio", "seed", "client_id")
    for event in tables["false_revocation_events"]:
        if event["revocation_path"] in sampled_paths:
            continue
        sampled_paths.add(event["revocation_path"])
        prior = [row for row in tables["false_revocation_history"]
                 if row["event_round"] == event["round"]
                 and all(row[key] == event[key] for key in identity_fields)]
        event_samples.append({**brief(event), "preceding_history": [brief(row) for row in prior[-3:]]})
    leakage_samples = sorted((row for row in tables["accepted_malicious_examples"]
                              if row["scope"] == "first_attack_round"),
                             key=lambda row: -row["aggregation_weight"])[:3]
    return {"summary": {"attack_candidate": attack_candidate, "clean_candidate": clean_candidate,
                        "attack_cases": attack_cases, "clean_cases": clean_cases, "diagnostics_coverage": "checked",
                        "false_revocations_by_path": {name: paths[name] for name in ("immediate_revocation", "count_threshold", "unknown")},
                        "false_revocation_score_evidence": dict(evidence_counts), "accepted_malicious": dict(accepted_counts),
                        "top_leaking_scenarios": ranking, "clean_scenarios": tables["clean_scenarios"][:6],
                        "false_revocation_examples": event_samples,
                        "accepted_malicious_examples": [brief(row) for row in leakage_samples],
                        "definitions": {"first_attack_round": "first attack round only", "early": "first five attack rounds (or all available if fewer)",
                                        "all": "accepted client-round observations over the full attack period",
                                        "tail": "last 10 recorded attack rounds", "history": "up to five preceding observations of the same client",
                                        "recovery": "first accuracy reaching preattack accuracy; does not imply sustained recovery",
                                        "scores": "novelty is composite; signed/class/norm have no separately recorded admission thresholds",
                                        "examples": "top aggregation weights per scenario/scope; ranking does not establish individual causal impact"}},
            "tables": tables}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, default=Path(__file__).resolve().parent / "outputs/cifar10_v7_target_fair_tuning")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--attack-candidate", default="sm9rrs-003")
    parser.add_argument("--clean-candidate", default="sm9rrs-001")
    args = parser.parse_args(argv)
    source = args.source_output.expanduser().resolve()
    output = (args.output_dir or Path(__file__).resolve().parent / "outputs" /
              ("cifar10_ours_failure_audit_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).expanduser().resolve()
    if output == source or source in output.parents or output.exists():
        raise ValueError("output must be a fresh directory outside the original source")
    print("AUDIT_LOADING recorded validation snapshot", flush=True)
    manifest, mapped = load_recorded_results(source)
    print(f"AUDIT_MAPPING_VERIFIED results={len(mapped)}", flush=True)
    audit = build_audit(mapped, args.attack_candidate, args.clean_candidate)
    audit["summary"].update(source_fingerprint=manifest["fingerprint"], result_count=len(mapped), output_dir=str(output))
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in audit["tables"].items():
        with (output / (name + ".csv")).open("w", newline="", encoding="utf-8") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(audit["summary"], indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print("AUDIT_SUMMARY " + json.dumps(audit["summary"], ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
