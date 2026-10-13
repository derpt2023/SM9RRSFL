"""Read-only four-point CNN threshold panel; review evidence, never select a winner."""
from copy import deepcopy
import json
from pathlib import Path
from statistics import fmean

import cifar_timing_report as timing
import summarize_cifar_timing as brief

runtime, base = timing.runtime, timing.base
ARMS = ("P0", "P1", "P2", "P3")
PAIR_KEYS = [(p, timing.DEV_SEED, ratio) for p in ("iid", "dirichlet") for ratio in (0., .1, .7)]
THRESHOLD_FIELDS = ("detector_distance_threshold", "detector_drift_allowance", "detector_drift_threshold")
PAIR_FIELDS = ("partition", "ratio", "available", "both_healthy", "accuracy_change_pp",
               "attack_asr_change_pp", "clean_background_change_pp", "false_positive_revocations_change", "health_changed")


def _pairs(rows, arm):
    grouped = {}
    for row in rows:
        if row.get("arm") == arm:
            key = (row["partition"], row["seed"], row["malicious_ratio"])
            grouped.setdefault(key, []).append(row)
    return grouped


def compare(later_rows, later_arm, earlier_rows, earlier_arm):
    later, earlier = _pairs(later_rows, later_arm), _pairs(earlier_rows, earlier_arm)
    pairs = []
    for partition, seed, ratio in PAIR_KEYS:
        key = (partition, seed, ratio)
        left, right = later.get(key, []), earlier.get(key, [])
        identifiable = len(left) == len(right) == 1
        a, b = (left[0], right[0]) if identifiable else ({}, {})
        available = identifiable and a.get("status") == b.get("status") == "complete"
        pairs.append({"partition": partition, "seed": seed, "ratio": ratio,
            "later_task_id": a.get("task_id"), "earlier_task_id": b.get("task_id"),
            "available": available,
            "both_healthy": bool(available and a.get("healthy") is True and b.get("healthy") is True),
            "later_healthy": a.get("healthy"), "earlier_healthy": b.get("healthy"),
            "accuracy_change_pp": 100 * (a["accuracy150"] - b["accuracy150"]) if available else None,
            "attack_asr_change_pp": 100 * (a["attack_asr150"] - b["attack_asr150"]) if available and ratio else None,
            "clean_background_change_pp": 100 * (a["background_5_to_7"] - b["background_5_to_7"]) if available and not ratio else None,
            "false_positive_revocations_change": a["false_positive_revocations"] - b["false_positive_revocations"] if available else None,
            "health_changed": (a["healthy"] != b["healthy"]) if available else None})
    return {"later": later_arm, "earlier": earlier_arm, "paired_n": sum(p["available"] for p in pairs),
            "expected_pairs": 6, "healthy_paired_n": sum(p["both_healthy"] for p in pairs), "pairs": pairs}


def candidate_summaries(rows, *, evidence_ready=True):
    summaries = []
    for arm in ARMS:
        group = [r for r in rows if r.get("arm") == arm]
        complete = [r for r in group if r.get("status") == "complete"]
        attack = [r for r in complete if r["malicious_ratio"]]
        healthy = sum(r.get("healthy") is True for r in complete)
        exactly_paired = set(_pairs(group, arm)) == set(PAIR_KEYS) and all(len(v) == 1 for v in _pairs(group, arm).values())
        summaries.append({"arm": arm, "complete": len(complete), "expected": 6, "healthy": healthy,
            "accuracy_n": len(complete), "accuracy_expected_n": 6,
            "mean_accuracy150": fmean(r["accuracy150"] for r in complete) if complete else None,
            "attack_asr_n": len(attack), "attack_asr_expected_n": 4,
            "mean_attack_asr150": fmean(r["attack_asr150"] for r in attack) if attack else None,
            "eligible_for_review": evidence_ready and exactly_paired and len(complete) == healthy == 6,
            "clean_utility": [{"partition": r["partition"], "seed": r["seed"],
                "available": r.get("clean_utility_vs_C0", {}).get("available", False),
                "accuracy_drop_pp": r.get("clean_utility_vs_C0", {}).get("accuracy_drop_pp"),
                "within_3pp": r.get("clean_utility_vs_C0", {}).get("within_3pp")}
                for r in complete if not r["malicious_ratio"]]})
    return summaries


def decision(rows, comparisons, *, reference_verified, source_matches, environment_compatible):
    result = {"selected_candidate": None, "score_computed": False,
        "formal_qualification_assessed": False, "next_stage_started": False, "automatic_TPE_started": False,
        "clean_utility_is_separate_diagnostic": True,
        "repeatability_note": "the numerical path behind the earlier clean A/B difference remains unlocated; P0 versus old C is a repeated development condition, not a single-variable causal estimate or evidence of identified algorithm regression",
        "reference_C_role": "healthy development anchor only, not formal qualification or a formal Score"}
    if not reference_verified:
        return {**result, "action": "resolve_changed_or_invalid_reference"}
    if not source_matches:
        return {**result, "action": "resolve_changed_source_identity"}
    if not environment_compatible:
        return {**result, "action": "resolve_missing_or_incompatible_execution_environment"}
    repeat = comparisons["P0_minus_old_C"]
    changes = [p for p in repeat["pairs"] if p["health_changed"] is True]
    mechanism_issues = []
    for row in rows:
        if row.get("status") != "complete":
            continue
        windows = {**row.get("mechanism_windows", {}), "full_attack_period": row.get("full_attack_period", {})}
        unavailable = []
        for name, window in windows.items():
            if window.get("status") == "not_applicable_clean":
                continue
            clients = window.get("client_diagnostics", {})
            groups = [group for group in ("malicious", "honest") if
                      clients.get(group, {}).get("remaining_client_rounds", 0) and
                      clients.get(group, {}).get("observed_verified_finite_updates") == 0]
            if clients.get("status") != "available" or groups:
                unavailable.append({"window": name, "diagnostic_status": clients.get("status"),
                                    "remaining_groups_without_observations": groups})
        if unavailable:
            mechanism_issues.append({"task_id": row["task_id"], "windows": unavailable})
    result.update(P0_reference_pairs_complete=repeat["paired_n"] == 6,
        P0_old_C_health_changes=[{k: p[k] for k in ("partition", "seed", "ratio", "later_healthy", "earlier_healthy")} for p in changes],
        requires_manual_repeatability_review=bool(changes),
        requires_manual_mechanism_review=bool(mechanism_issues), mechanism_evidence_issues=mechanism_issues,
        mechanism_note="absent observations do not imply rejection or corrupted evidence; review coverage before inferring detection or history behavior",
        unhealthy_tasks=[r["task_id"] for r in rows if r.get("status") == "complete" and r.get("healthy") is not True])
    valid_matrix = (len(rows) == 24 and all(set(_pairs(rows, arm)) == set(PAIR_KEYS)
                    and all(len(v) == 1 for v in _pairs(rows, arm).values()) for arm in ARMS))
    if not valid_matrix or any(r.get("status") != "complete" for r in rows):
        return {**result, "action": "resolve_incomplete_or_invalid_evidence"}
    return {**result, "action": "review_fixed_panel_before_any_TPE",
        "review_note": "inspect all four declared points, failed health, clean utility, process counts, and P0 repeatability; no new numerical tolerance, winner, or search authorization is inferred"}


def summarize(output, timing_output, clean_output, matched_output):
    import cifar_cnn_threshold_protocol as protocol
    output = Path(output)
    try:
        manifest, tasks = protocol.read_study(output, current_sources=False)
    except Exception as exc:
        return {"status": "unavailable_or_invalid_study", "output": str(output), "error": str(exc),
                "training_started_by_summary": False}
    reference = manifest["reference"]
    errors = {}
    verified = False
    try:
        if protocol.audit_reference(Path(timing_output), Path(clean_output), Path(matched_output)) != reference:
            raise ValueError("reference evidence differs from the frozen threshold-panel reference")
        verified = True
    except Exception as exc:
        errors["reference_error"] = str(exc)
    environment = None
    try:
        environment = runtime.read_json(output / "execution_environment.json")
    except Exception as exc:
        errors["environment_error"] = str(exc)
    compatible = environment is not None and environment == reference["execution_environment"]
    sources = False
    try:
        sources = protocol.source_hashes() == manifest["source_sha256"]
    except Exception as exc:
        errors["source_error"] = str(exc)
    rows = timing.collect(output, tasks)
    task_by_id = {t["task_id"]: t for t in tasks}
    for row in rows:
        row["thresholds"] = [task_by_id[row["task_id"]]["config"][field] for field in THRESHOLD_FIELDS]
    ready = verified and compatible and sources
    comparisons = {}
    if ready:
        timing.add_clean_controls(rows, reference["clean_rows"])
        comparisons = {"P1_minus_P0": compare(rows, "P1", rows, "P0"),
            "P2_minus_P1": compare(rows, "P2", rows, "P1"),
            "P3_minus_P0": compare(rows, "P3", rows, "P0"),
            "P0_minus_old_C": compare(rows, "P0", reference["c_rows"], "C")}
    complete = sum(r["status"] == "complete" for r in rows)
    healthy = sum(r["status"] == "complete" and r.get("healthy") is True for r in rows)
    resolved = all(r["status"] in ("complete", "terminal_incomplete", "algorithm_numerical") for r in rows)
    return {"status": "invalid_comparison_evidence" if not ready else "complete" if complete == 24 else
            "resolved_with_failures" if resolved else "incomplete", "output": str(output),
        "protocol": manifest["spec"]["protocol"], "manifest_fingerprint": manifest["fingerprint"],
        "reference_timing_manifest_fingerprint": reference.get("timing_manifest_fingerprint"),
        "reference_clean_manifest_fingerprint": reference.get("clean_manifest_fingerprint"),
        "reference_matched_manifest_fingerprint": reference.get("matched_manifest_fingerprint"),
        "recorded_source_count": len(manifest["source_sha256"]),
        "recorded_source_map_sha256": base.digest(manifest["source_sha256"]),
        "reference_verified": verified, "source_matches_current": sources,
        "execution_environment_compatible": compatible,
        "execution_hardware": environment.get("actual_compute_device") if isinstance(environment, dict) else None,
        "complete_tasks": complete, "expected_tasks": 24, "healthy_tasks": healthy,
        "reference_timing_healthy_tasks": reference.get("timing_health_count"),
        "reference_timing_failures": [{k: r[k] for k in ("task_id", "healthy", "reasons") if k in r}
                                      for r in reference.get("failure_rows", [])] if verified else [],
        "base_health_rule_unchanged": True, "evaluation_split": "2500 calibration samples; official test unused for selection",
        "training_started_by_summary": False, "rows": rows,
        "reference_c_rows": reference["c_rows"] if ready else [],
        "clean_reference_rows": [r for r in reference["clean_rows"] if r["seed"] == timing.DEV_SEED] if ready else [],
        "candidates": candidate_summaries(rows, evidence_ready=ready), "comparisons": comparisons,
        "decision": decision(rows, comparisons, reference_verified=verified, source_matches=sources,
                             environment_compatible=compatible), **errors}


def print_summary(report):
    """Bound transport size by removing duplicate labels, never rounding measurements."""
    header = {"type": "header", "schema": "cifar-threshold-brief-v1",
              **{k: v for k, v in report.items() if k not in
                 ("rows", "reference_c_rows", "clean_reference_rows", "candidates", "comparisons", "decision")}}
    for key in ("error", "reference_error", "source_error", "environment_error"):
        if key in header:
            header[key] = brief.short_error(header[key])
    header.update(task_lines=len(report.get("rows", [])), old_C_lines=len(report.get("reference_c_rows", [])),
                  C0_lines=len(report.get("clean_reference_rows", [])))
    legend = deepcopy(brief.LEGEND)
    legend["schema"] = "cifar-threshold-brief-v1"
    legend["task"]["thresholds"] = "[warning, drift_allowance_kappa, drift_threshold_h]"
    legend["pair_fields"] = list(PAIR_FIELDS)
    legend["semantics"] = [s for s in legend["semantics"] if not s.startswith("A->B")]
    legend["semantics"] += ["P1-P0 changes warning; P2-P1 jointly changes kappa and h; P3-P0 jointly changes warning/kappa/h",
        "P0-old C repeats the recorded condition; the earlier numerical path is not identified; health changes require manual review",
        "compact windows retain n/mean/max and raw observation counts, not every per-round trajectory value"]
    records = [header, legend]
    for row in report.get("clean_reference_rows", []):
        records.append({"type": "reference_C0", "partition": row["partition"], "seed": row["seed"],
            "healthy": row.get("healthy"), "acc150": row.get("accuracy150"), "bg": row.get("background_5_to_7")})
    for row in report.get("reference_c_rows", []):
        compact = brief.compact_task(row)
        compact["type"] = "reference_C"
        for key in ("cost", "last", "clean", "m", "arm", "K", "start", "exposure"):
            compact.pop(key, None)
        records.append(compact)
    for row in report.get("rows", []):
        compact = brief.compact_task(row)
        compact["thresholds"] = row["thresholds"]
        records.append(compact)
    for candidate in report.get("candidates", []):
        records.append({"type": "candidate", **candidate})
    for label, comparison in report.get("comparisons", {}).items():
        records.append({"type": "comparison", "label": label,
            **{k: v for k, v in comparison.items() if k != "pairs"},
            "pairs": [[p[field] for field in PAIR_FIELDS] for p in comparison["pairs"]]})
    if "decision" in report:
        records.append({"type": "decision", **report["decision"]})
    # Pre-serialize so invalid numeric values never leave a truncated success-looking body.
    lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) for record in records]
    print("=== CIFAR_THRESHOLD_BEGIN ===", flush=True)
    for line in lines:
        print(line)
    print("=== CIFAR_THRESHOLD_END ===", flush=True)
