#!/usr/bin/env python3
"""Read-only follow-up: pre-intervention differences and revocation mechanisms.

This entry does not train, open a checkpoint, probe CUDA, change numerical
policy, or write into any experiment directory. Redirect stdout for transport.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import cifar_cnn_history_protocol as protocol
import cifar_cnn_history_report as history
import summarize_cifar_timing as brief
import cifar_history_forensics as forensic

base, runtime, timing = history.base, history.runtime, history.timing
EXPECTED_MANIFEST = "3c6566877fe4ea96ec9a6d464828c6d4c050937e1b1cf7dea82b2eb4db640eed"
DISPLAY_TOLERANCE = 1e-12
RECORD_FIELDS = brief.PAIR_FIELDS
CLIENT_FIELDS = ("decision_reason", "suspicious", "count_increment", "weight_before",
    "weight_after_penalty_recovery", "aggregation_weight", "count_before", "count_after",
    "trace_requested", "trace_pending", "revoked", "novelty_score", "anchor_score",
    "signed_score", "class_score", "cumulative_drift", "clip_factor", "aggregation_accepted",
    "history_eligible", "history_admitted", "history_frozen", "immediate_revocation",
    "trusted_history_size", "normal_cluster_count", "recovery_eligible", "norm_score")
TORCH_FIELDS = ("version", "cuda_version", "cudnn_version", "deterministic_algorithms",
    "deterministic_warn_only", "cudnn_deterministic", "cudnn_benchmark", "cuda_matmul_allow_tf32",
    "cudnn_allow_tf32", "float32_matmul_precision", "num_threads", "num_interop_threads")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_fields(later, earlier, fields):
    """Compare matched keyed records; retain exact values and show small differences separately."""
    keys = sorted(set(later) | set(earlier))
    if set(later) != set(earlier):
        return {"status": "unavailable", "reason": "observation keys differ",
                "later_n": len(later), "earlier_n": len(earlier),
                "later_only_n": len(set(later) - set(earlier)),
                "earlier_only_n": len(set(earlier) - set(later))}
    if not keys:
        return {"status": "unavailable", "reason": "no observations"}
    changes = []
    for field in fields:
        count, first, above, maximum, max_key = 0, None, None, None, None
        for key in keys:
            left, right = getattr(later[key], field), getattr(earlier[key], field)
            if any(_number(x) and not math.isfinite(x) for x in (left, right)):
                raise ValueError("nonfinite comparison field: " + field)
            if left == right:
                continue
            count += 1
            example = [key, left, right]
            first = first or example
            numeric = _number(left) and _number(right)
            delta = abs(left - right) if numeric else None
            if numeric and (maximum is None or delta > maximum):
                maximum, max_key = delta, key
            # This display distinction is never a health gate or acceptance tolerance.
            if above is None and (not numeric or delta > DISPLAY_TOLERANCE):
                above = example
        if count:
            changes.append([field, count, first, above, maximum, max_key])
    return {"status": "audited", "matched_observations": len(keys), "changed_fields": changes}


def pair_diagnostics(later, earlier):
    left_records, right_records = timing._records(later), timing._records(earlier)
    # Existing validation checks identities, population, duplicate rows and counts.
    left_diagnostics = timing._diagnostics(later, left_records)
    right_diagnostics = timing._diagnostics(earlier, right_records)
    def clients(result):
        return {(d.round, d.client_id): d for d in result.diagnostics if 1 <= d.round <= 24}
    client_comparison = compare_fields(clients(later), clients(earlier), CLIENT_FIELDS)
    coverage = {}
    for name, result, records, diagnostics in (("later", later, left_records, left_diagnostics),
            ("earlier", earlier, right_records, right_diagnostics)):
        window = timing.mechanism_window(result, records, diagnostics, 1, 24)
        groups = [window["client_diagnostics"][g] for g in ("malicious", "honest")]
        observed = sum(g["observed_verified_finite_updates"] for g in groups)
        remaining = (sum(g["remaining_client_rounds"] for g in groups)
            if all(g["remaining_client_rounds"] is not None for g in groups) else None)
        coverage[name] = {"observed": observed, "remaining": remaining,
            "unobserved": remaining - observed if remaining is not None else None}
        if remaining is None or observed != remaining or not window["complete_round_records"]:
            client_comparison.update(status="unavailable", reason="incomplete pre25 client coverage")
    client_comparison["coverage"] = coverage
    return {
        "records_pre25": compare_fields({r: left_records[r] for r in range(25)},
            {r: right_records[r] for r in range(25)}, RECORD_FIELDS),
        "records_round25": compare_fields({25: left_records[25]}, {25: right_records[25]}, RECORD_FIELDS),
        "clients_pre25": client_comparison,
    }


def recorded_environment(metadata):
    """Only allowlisted historic provenance; no current-machine substitute or credentials."""
    device = metadata.get("actual_compute_device", {})
    return {"requested_device": metadata.get("requested_device"),
        "device": {k: device.get(k) for k in ("logical_device", "uuid", "name", "compute_capability")},
        "torch": {k: metadata.get("torch", {}).get(k) for k in TORCH_FIELDS},
        "environment": {k: metadata.get("environment", {}).get(k) for k in (
            "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG",
            "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


def evidence_hashes(output, tasks):
    paths = [output / x for x in ("manifest.json", "task_plans/history.json", "execution_environment.json")]
    for task in tasks:
        folder = output / "tasks" / task["task_id"]
        paths.extend(folder / name for name in ("task.json", "environment.json", base.experiments.COMPLETED_RESULTS_SNAPSHOT))
        paths.extend(sorted((folder / "attempts").glob("*.json")))
    return {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def reader_hashes():
    return {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("diagnose_cifar_history.py", "cifar_history_forensics.py")}


def diagnose(output, threshold_output, timing_output, clean_output, matched_output):
    readers_before = reader_hashes()
    paths = list(map(lambda x: Path(x).resolve(), (output, threshold_output, timing_output, clean_output, matched_output)))
    output, threshold_output, timing_output, clean_output, matched_output = paths
    from run_cifar_cnn_history_panel import separate_outputs
    separate_outputs(*paths)
    manifest, tasks = protocol.read_study(output, current_sources=True)
    if manifest["fingerprint"] != EXPECTED_MANIFEST:
        raise ValueError("this follow-up requires the reviewed 12-task history study")
    reference = manifest["reference"]
    old_manifest, old_tasks = protocol.threshold.read_study(threshold_output, current_sources=True)
    before = evidence_hashes(output, tasks)
    before_old = protocol.threshold_evidence_hashes(threshold_output, old_tasks)
    summary = history.summarize(*paths)
    required = ("reference_verified", "source_matches_current", "execution_environment_compatible",
        "task_environments_compatible", "history_intervention_verified", "mechanism_coverage_verified",
        "record_pair_audits_available", "evidence_ready")
    if (summary.get("status") != "complete" or summary.get("complete_tasks") != 12
            or any(summary.get(key) is not True for key in required)
            or summary.get("manifest_fingerprint") != manifest["fingerprint"]):
        raise ValueError("complete, verified history evidence is required; received " + str(summary.get("status")))

    results, environments, analyses, failures = {}, {}, [], []
    for task in tasks:
        ident = task["task_id"]
        result = base.checked_completed(output, task)
        if result is None:
            raise ValueError("history snapshot disappeared: " + ident)
        results[ident] = result
        environments[ident] = recorded_environment(runtime.read_json(output / "tasks" / ident / "environment.json"))
        analysis = forensic.analyze_task(result, task)
        if analysis["status"] != "valid":
            failures.append({"task_id": ident, "status": analysis["status"],
                             "error": brief.short_error(analysis.get("error", "invalid forensic evidence"))})
        analyses.append(analysis)
    old_results = {}
    for task in reference["p0_tasks"]:
        result = base.checked_completed(threshold_output, task)
        if result is None:
            raise ValueError("reference P0 snapshot disappeared: " + task["task_id"])
        old_results[task["task_id"]] = result

    pairs = []
    for partition, seed, ratio in history.PAIR_KEYS:
        def select(pool, arm):
            matches = [t for t in pool if t["arm"] == arm and
                (t["config"]["partition"], t["config"]["seed"], t["config"]["malicious_ratio"]) == (partition, seed, ratio)]
            if len(matches) != 1:
                raise ValueError("missing unique paired task")
            return matches[0]
        h0, h1, p0 = select(tasks, "H0"), select(tasks, "H1"), select(reference["p0_tasks"], "P0")
        if not h0["config"] == h1["config"] == p0["config"]:
            raise ValueError("paired scientific configurations differ")
        for label, later, earlier in (("H1-H0", results[h1["task_id"]], results[h0["task_id"]]),
                ("H0-P0", results[h0["task_id"]], old_results[p0["task_id"]])):
            pair = {"type": "prefix_pair", "label": label, "partition": partition, "ratio": ratio,
                    "seed": seed, **pair_diagnostics(later, earlier)}
            if any(pair[k]["status"] != "audited" for k in ("records_pre25", "records_round25", "clients_pre25")):
                failures.append({"pair": [label, partition, ratio], "status": "incomplete_prefix_observations"})
            pairs.append(pair)

    if (before != evidence_hashes(output, tasks)
            or before_old != protocol.threshold_evidence_hashes(threshold_output, old_tasks)
            or protocol.source_hashes() != manifest["source_sha256"]
            or readers_before != reader_hashes()):
        raise ValueError("evidence or scientific sources changed while being read")
    header = {"type": "header", "schema": "cifar-history-forensics-v1", "status": "complete" if not failures else "incomplete_forensic_evidence",
        "manifest_fingerprint": manifest["fingerprint"], "reference_threshold_manifest_fingerprint": old_manifest["fingerprint"],
        "recorded_source_count": len(manifest["source_sha256"]), "source_map_sha256": base.digest(manifest["source_sha256"]),
        "complete_tasks": summary["complete_tasks"], "healthy_tasks": summary["healthy_tasks"],
        "source_reference_environment_verified": True, "input_evidence_unchanged": True,
        "unchanged_check_scope": "history12 and threshold24 files plus reader/source files throughout this read; upstream54 checked by original nested reference audits",
        "history_evidence_sha256": base.digest(before), "threshold_evidence_sha256": base.digest(before_old),
        "reader_sha256": readers_before,
        "training_started": False, "experiment_files_written": False, "checkpoints_opened": False}
    records = [header, {**forensic.COMPACT_LEGEND,
        "changed_fields": ["field", "exact_difference_count", "first_exact[key,later,earlier]",
            "first_above_display_tolerance[key,later,earlier]", "max_abs_difference", "max_abs_key"],
        "display_tolerance": DISPLAY_TOLERANCE,
        "display_tolerance_role": "display aid only; original values retained, no health/algorithm tolerance changed",
        "prefix_keys": "round for scalar records; [round,client_id] for client diagnostics; task_tag excluded",
        "prefix_scope": "round0..24 and separate round25; client diagnostics only1..24, because history_admitted differs intentionally at25",
        "limits": "no per-client update/model/RNG snapshots are inferred from scalar agreement; CPU set-sum demonstration explains only a possible diagnostic tail-difference mechanism, not Dirichlet accuracy differences"}]
    profiles, devices = [], []
    for ident, env in environments.items():
        profile = {k: env[k] for k in ("torch", "environment")}
        if profile not in profiles:
            profiles.append(profile)
        devices.append([ident, env["requested_device"], env["device"], profiles.index(profile)])
    records.append({"type": "recorded_environments", "profiles": profiles,
        "device_fields": ["task_id", "requested_device", "recorded_actual_device", "profile_index"],
        "tasks": devices, "scope": "historic files only; no GPU probe or current flags substituted"})
    records.extend(pairs)
    for task, analysis in zip(tasks, analyses):
        detailed = task["config"]["malicious_ratio"] == .7 or (task["config"]["partition"] == "dirichlet" and not task["config"]["malicious_ratio"])
        records.append(forensic.compact_task_analysis(analysis, detailed=detailed))
    records.append({"type": "decision", "action": "review_prefix_and_revocation_evidence_before_any_new_training",
        "adopt_frozen_history": False, "automatic_next_stage": False, "failures": failures,
        "causal_limit": "freezing is not a successful repair in the reviewed panel; quantify neither the Dirichlet causal effect nor a CUDA cause from these observational traces"})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=protocol.DEFAULT_OUTPUT)
    parser.add_argument("--threshold-output", type=Path, default=protocol.DEFAULT_THRESHOLD)
    parser.add_argument("--timing-output", type=Path, default=protocol.DEFAULT_TIMING)
    parser.add_argument("--clean-output", type=Path, default=protocol.DEFAULT_CLEAN)
    parser.add_argument("--matched-output", type=Path, default=protocol.DEFAULT_MATCHED)
    args = parser.parse_args(argv)
    try:
        records = diagnose(args.output, args.threshold_output, args.timing_output, args.clean_output, args.matched_output)
        lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":"), allow_nan=False) for r in records]
        code = 0 if records[0]["status"] == "complete" else 2
    except Exception as exc:
        lines = [json.dumps({"type": "header", "status": "unavailable_or_invalid_evidence",
            "error": brief.short_error(exc), "training_started": False, "experiment_files_written": False})]
        code = 2
    print("=== CIFAR_HISTORY_FORENSICS_BEGIN ===", flush=True)
    for line in lines:
        print(line)
    print("=== CIFAR_HISTORY_FORENSICS_END ===", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
