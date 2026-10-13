"""Read-only paired history intervention evidence; no selection or training."""
from copy import deepcopy
import json
from pathlib import Path
from statistics import fmean

import cifar_cnn_threshold_report as threshold
import cifar_timing_report as timing
import run_cifar_matched_cnn as matched
import summarize_cifar_timing as brief

runtime, base = timing.runtime, timing.base
ARMS = ("H0", "H1")
FREEZE_START = 25
PAIR_KEYS = threshold.PAIR_KEYS
PAIR_FIELDS = threshold.PAIR_FIELDS


def history_audit(result, task):
    """Audit actual admission flags, including clean and every post-freeze round."""
    audit = {"variant": task["candidate"]["variant"],
        "declared_freeze_start_round": task["history_freeze_start_round"],
        "audited_start_round": FREEZE_START, "audited_end_round": 150,
        "verified": False, "observed_admissions": None, "observed_admission_rate": None}
    if result is None:
        return {**audit, "status": "snapshot_unavailable"}
    try:
        invalid, admitted = [], []
        for d in result.diagnostics:
            if FREEZE_START <= d.round <= 150:
                value = getattr(d, "history_admitted", None)
                if type(value) is not bool:
                    invalid.append([d.round, d.client_id])
                elif value:
                    admitted.append([d.round, d.client_id])
        audit.update(invalid_admission_flags=len(invalid), first_invalid_flag=invalid[0] if invalid else None,
                     observed_admissions=len(admitted), first_observed_admission=admitted[0] if admitted else None)
        if invalid:
            raise ValueError("missing or non-boolean history_admitted in post-freeze client diagnostics")
        records = timing._records(result)
        diagnostics = timing._diagnostics(result, records)
        window = timing.mechanism_window(result, records, diagnostics, FREEZE_START, 126)
        clients = window["client_diagnostics"]
        groups = [clients[name] for name in ("malicious", "honest")]
        observed = sum(g["observed_verified_finite_updates"] for g in groups)
        known = all(g["remaining_client_rounds"] is not None for g in groups)
        remaining = sum(g["remaining_client_rounds"] for g in groups) if known else None
        gap = remaining - observed if remaining is not None else None
        coverage = known and window["complete_round_records"] and gap == 0
        audit.update(window=window, observed_verified_finite_updates=observed,
            remaining_client_rounds=remaining, unobserved_remaining_client_rounds=gap,
            complete_coverage=coverage, observed_admission_rate=len(admitted) / observed if observed else None)
        if task["arm"] == "H0":
            audit["status"] = "original_no_freeze_expected"
        elif admitted:
            audit["status"] = "freeze_violated"
        elif not window["complete_round_records"]:
            audit["status"] = "incomplete_round_records"
        elif not coverage:
            audit["status"] = "incomplete_client_coverage"
        elif not observed:
            audit["status"] = "not_exercised_no_post_freeze_observations"
        else:
            audit.update(status="verified_no_admissions", verified=True)
    except Exception as exc:
        audit.update(status="invalid_diagnostic_evidence", error=str(exc))
    return audit


def _task_evidence(output, tasks, rows, reference_environment):
    """Read snapshots and each recorded environment; never repair absent evidence."""
    records = {}
    for task, row in zip(tasks, rows):
        row["variant"] = task["candidate"]["variant"]
        row["history_freeze_start_round"] = task["history_freeze_start_round"]
        row["task_fingerprint"] = task["fingerprint"]
        row["task_environment_compatible"] = None
        result = None
        try:
            result = base.checked_completed(output, task)
            if result is not None:
                metadata = runtime.read_json(output / "tasks" / task["task_id"] / "environment.json")
                row["task_environment_compatible"] = matched.normalized_environment(metadata) == reference_environment
                row["recorded_device"] = metadata.get("requested_device")
                records[task["task_id"]] = timing._records(result)
        except Exception as exc:
            row["additional_evidence_error"] = str(exc)
            row["task_environment_compatible"] = False
        row["history_audit"] = history_audit(result, task)
    return records


def _record_differences(later, earlier, rounds):
    differences = []
    for rd in rounds:
        if rd not in later or rd not in earlier:
            return {"status": "unavailable", "compared_rounds": 0,
                    "error": "requested round records are incomplete"}
        changed = {}
        for field in brief.PAIR_FIELDS:
            values = [getattr(later[rd], field), getattr(earlier[rd], field)]
            if any(isinstance(value, (int, float)) and not isinstance(value, bool) and
                   not timing.finite(value) for value in values):
                return {"status": "unavailable", "compared_rounds": 0,
                        "error": "nonfinite comparison field at round " + str(rd) + ": " + field}
            if values[0] != values[1]:
                changed[field] = values
        if changed:
            differences.append({"round": rd, "fields": changed})
    return {"status": "audited", "compared_rounds": len(rounds), "differing_rounds": len(differences),
            "first_differing_round": differences[0] if differences else None}


def record_pair_audits(tasks, records, old_tasks, old_records):
    """Compare recorded scalar fields only; absence of differences is not bitwise equality."""
    output = []
    for partition, seed, ratio in PAIR_KEYS:
        key = (partition, seed, ratio)
        audit = {"partition": partition, "seed": seed, "ratio": ratio}
        try:
            pair = {}
            for arm, pool in (("H0", tasks), ("H1", tasks), ("P0", old_tasks)):
                matches = [t for t in pool if t["arm"] == arm and
                    (t["config"]["partition"], t["config"]["seed"], t["config"]["malicious_ratio"]) == key]
                if len(matches) != 1:
                    raise ValueError("expected exactly one task per paired condition")
                pair[arm] = matches[0]
            same_config = pair["H1"]["config"] == pair["H0"]["config"] == pair["P0"]["config"]
            if not same_config:
                raise ValueError("paired scientific configurations differ")
            h0, h1, p0 = records.get(pair["H0"]["task_id"], {}), records.get(pair["H1"]["task_id"], {}), old_records.get(pair["P0"]["task_id"], {})
            audit.update(status="audited", identical_scientific_configs=True,
                H1_minus_H0_pre25=_record_differences(h1, h0, list(range(25))),
                H1_minus_H0_round25=_record_differences(h1, h0, [25]),
                H0_minus_old_P0=_record_differences(h0, p0, list(range(151))))
        except Exception as exc:
            audit.update(status="unavailable", error=str(exc))
        output.append(audit)
    return output


def candidate_summaries(rows, *, evidence_ready):
    summaries = []
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        complete = [r for r in group if r["status"] == "complete"]
        attacked = [r for r in complete if r["malicious_ratio"]]
        healthy = sum(r.get("healthy") is True for r in complete)
        paired = threshold._pairs(group, arm)
        exact = set(paired) == set(PAIR_KEYS) and all(len(v) == 1 for v in paired.values())
        summaries.append({"arm": arm, "complete": len(complete), "expected": 6, "healthy": healthy,
            "accuracy_n": len(complete), "accuracy_expected_n": 6,
            "mean_accuracy150": fmean(r["accuracy150"] for r in complete) if complete else None,
            "attack_accuracy_n": len(attacked), "attack_accuracy_expected_n": 4,
            "mean_attack_accuracy150": fmean(r["accuracy150"] for r in attacked) if attacked else None,
            "attack_asr_n": len(attacked), "attack_asr_expected_n": 4,
            "mean_attack_asr150": fmean(r["attack_asr150"] for r in attacked) if attacked else None,
            "eligible_for_review": evidence_ready and exact and len(complete) == healthy == 6,
            "clean_utility": [{"partition": r["partition"], "seed": r["seed"],
                **{k: r.get("clean_utility_vs_C0", {}).get(k) for k in ("available", "accuracy_drop_pp", "within_3pp")}}
                for r in complete if not r["malicious_ratio"]]})
    return summaries


def _decision(rows, comparisons, audits, *, provenance, evidence_ready, errors):
    repeat = comparisons.get("H0_minus_old_P0", {})
    changes = [p for p in repeat.get("pairs", []) if p["health_changed"] is True]
    repeatability_issues = []
    for audit in audits:
        reasons = []
        if audit["status"] != "audited":
            reasons.append(["pair", "unavailable", None])
        else:
            for name in ("H1_minus_H0_pre25", "H1_minus_H0_round25", "H0_minus_old_P0"):
                value = audit[name]
                if value["status"] != "audited":
                    reasons.append([name, "unavailable", None])
                elif value["differing_rounds"]:
                    reasons.append([name, "recorded_fields_differ", value["first_differing_round"]["round"]])
        if reasons:
            repeatability_issues.append({"partition": audit["partition"], "ratio": audit["ratio"], "reasons": reasons})
    result = {"selected_arm": None, "score_computed": False, "formal_qualification_assessed": False,
        "next_stage_started": False, "automatic_TPE_started": False, "NoPermanent_started": False,
        "performance_gate_changed": False, "clean_utility_is_separate_diagnostic": True,
        "requires_manual_repeatability_review": bool(changes or repeatability_issues),
        "record_repeatability_issues": repeatability_issues,
        "H0_old_P0_health_changes": [{k: p[k] for k in ("partition", "ratio", "later_healthy", "earlier_healthy")} for p in changes],
        "unhealthy_tasks": [r["task_id"] for r in rows if r["status"] == "complete" and r.get("healthy") is not True],
        "history_evidence_issues": [{"task_id": r["task_id"], "status": r["history_audit"]["status"]}
            for r in rows if (r["arm"] == "H1" and not r["history_audit"]["verified"]) or
                (r["arm"] == "H0" and (not r["history_audit"].get("complete_coverage") or
                 not r["history_audit"].get("observed_verified_finite_updates")))],
        "repeatability_note": "the earlier numerical path remains unlocated; repeated-condition differences are not an identified algorithm regression, a significance test, or a universal noise bound",
        "intervention_note": "freeze retains each client's history/live normal at the end of round 24, not its K20 anchor; round25 evaluation/aggregation precedes the first suppressed commit, so training effects can begin at round26 for identical prefixes",
        "diagnostic_limit": "verified means actual admission flags satisfy the frozen source contract, not that every stored normal array was independently compared; unobserved clients are never treated as zero admissions or rejected"}
    gaps = [r["task_id"] for r in rows if r["status"] == "complete" and
            (not r["history_audit"].get("complete_coverage") or not r["history_audit"].get("observed_verified_finite_updates"))]
    result.update(requires_manual_mechanism_review=bool(gaps or result["history_evidence_issues"]),
                  tasks_with_incomplete_mechanism_coverage=gaps)
    if not provenance:
        return {**result, "action": "resolve_invalid_source_reference_or_environment", "evidence_errors": sorted(errors)}
    if len(rows) != 12 or any(r["status"] != "complete" for r in rows):
        return {**result, "action": "resolve_incomplete_or_invalid_execution_evidence"}
    if not evidence_ready:
        return {**result, "action": "review_unverified_history_intervention_evidence"}
    return {**result, "action": "review_history_ablation_before_any_next_stage"}


def summarize(output, threshold_output, timing_output, clean_output, matched_output):
    import cifar_cnn_history_protocol as protocol
    output = Path(output)
    try:
        manifest, tasks = protocol.read_study(output, current_sources=False)
    except Exception as exc:
        return {"status": "unavailable_or_invalid_study", "output": str(output), "error": str(exc),
                "training_started_by_summary": False}
    reference, errors = manifest["reference"], {}
    verified, sources, environment = False, False, None
    try:
        if protocol.audit_reference(Path(threshold_output), Path(timing_output), Path(clean_output), Path(matched_output)) != reference:
            raise ValueError("reference evidence differs from the frozen history-study reference")
        verified = True
    except Exception as exc:
        errors["reference_error"] = str(exc)
    try:
        sources = protocol.source_hashes() == manifest["source_sha256"]
    except Exception as exc:
        errors["source_error"] = str(exc)
    try:
        environment = runtime.read_json(output / "execution_environment.json")
    except Exception as exc:
        errors["environment_error"] = str(exc)
    compatible = environment is not None and environment == reference["execution_environment"]
    rows = timing.collect(output, tasks)
    records = _task_evidence(output, tasks, rows, reference["execution_environment"])
    task_environments = all(r["task_environment_compatible"] is not False for r in rows)
    if not task_environments:
        errors["task_environment_error"] = "one or more snapshot environments are missing or incompatible"
    provenance = verified and sources and compatible and task_environments
    freeze_verified = len([r for r in rows if r["arm"] == "H1"]) == 6 and all(
        r["history_audit"]["verified"] for r in rows if r["arm"] == "H1")
    complete = sum(r["status"] == "complete" for r in rows)
    healthy = sum(r["status"] == "complete" and r.get("healthy") is True for r in rows)
    mechanism_verified = len(rows) == 12 and all(r["history_audit"].get("complete_coverage") and
        r["history_audit"].get("observed_verified_finite_updates") for r in rows)
    comparisons, audits = {}, []
    if provenance:
        timing.add_clean_controls(rows, reference["clean_rows"])
        comparisons = {"H1_minus_H0": threshold.compare(rows, "H1", rows, "H0"),
                       "H0_minus_old_P0": threshold.compare(rows, "H0", reference["p0_rows"], "P0")}
        old_records = {}
        for task in reference["p0_tasks"]:
            try:
                result = base.checked_completed(Path(threshold_output), task)
                if result is not None:
                    old_records[task["task_id"]] = timing._records(result)
            except Exception as exc:
                errors["pair_record_error"] = str(exc)
        audits = record_pair_audits(tasks, records, reference["p0_tasks"], old_records)
    record_audits_available = len(audits) == 6 and all(a["status"] == "audited" and
        all(a[k]["status"] == "audited" for k in ("H1_minus_H0_pre25", "H1_minus_H0_round25", "H0_minus_old_P0"))
        for a in audits)
    evidence_ready = provenance and freeze_verified and mechanism_verified and record_audits_available and complete == len(rows) == 12
    resolved = all(r["status"] in ("complete", "terminal_incomplete", "algorithm_numerical") for r in rows)
    return {"status": "invalid_comparison_evidence" if not provenance else
            "invalid_intervention_evidence" if complete == 12 and not freeze_verified else
            "incomplete_mechanism_evidence" if complete == 12 and not (mechanism_verified and record_audits_available) else
            "complete" if complete == 12 else "resolved_with_failures" if resolved else "incomplete",
        "output": str(output), "protocol": manifest["spec"]["protocol"], "manifest_fingerprint": manifest["fingerprint"],
        **{"reference_" + key: reference.get(key) for key in
           ("threshold_manifest_fingerprint", "timing_manifest_fingerprint", "clean_manifest_fingerprint", "matched_manifest_fingerprint")},
        "recorded_source_count": len(manifest["source_sha256"]), "recorded_source_map_sha256": base.digest(manifest["source_sha256"]),
        "reference_verified": verified, "source_matches_current": sources,
        "execution_environment_compatible": compatible, "task_environments_compatible": task_environments,
        "execution_hardware": environment.get("actual_compute_device") if isinstance(environment, dict) else None,
        "complete_tasks": complete, "expected_tasks": 12, "healthy_tasks": healthy,
        "reference_threshold_healthy_tasks": reference.get("threshold_health_count"),
        "reference_threshold_failures": reference.get("failure_rows", []) if verified else [],
        "history_intervention_verified": freeze_verified, "mechanism_coverage_verified": mechanism_verified,
        "record_pair_audits_available": record_audits_available, "evidence_ready": evidence_ready,
        "base_health_rule_unchanged": True, "training_started_by_summary": False,
        "evaluation_split": "2500 calibration samples; official test unused for selection",
        "rows": rows, "reference_p0_rows": reference["p0_rows"] if provenance else [],
        "clean_reference_rows": [r for r in reference["clean_rows"] if r["seed"] == timing.DEV_SEED] if provenance else [],
        "candidates": candidate_summaries(rows, evidence_ready=evidence_ready), "comparisons": comparisons,
        "record_pair_audits": audits,
        "decision": _decision(rows, comparisons, audits, provenance=provenance, evidence_ready=evidence_ready, errors=errors), **errors}


def print_summary(report):
    excluded = ("rows", "reference_p0_rows", "clean_reference_rows", "candidates", "comparisons", "record_pair_audits", "decision")
    header = {"type": "header", "schema": "cifar-history-brief-v1", **{k: v for k, v in report.items() if k not in excluded}}
    for key in list(header):
        if "error" in key:
            header[key] = brief.short_error(header[key])
    header.update(task_lines=len(report.get("rows", [])), old_P0_lines=len(report.get("reference_p0_rows", [])),
                  C0_lines=len(report.get("clean_reference_rows", [])))
    legend = deepcopy(brief.LEGEND)
    legend["schema"] = "cifar-history-brief-v1"
    legend["task"].update(variant="explicit task candidate variant; freeze=25 only for H1, including clean",
        hf="[audit_status,verified,observed,remaining,unobserved,admitted,admission_rate,invalid_flags,first_invalid,first_admitted]",
        wh="all scheduled rounds25..150, including clean, with the same window fields", env="recorded task environment matches frozen normalized reference")
    legend["pair_fields"] = list(PAIR_FIELDS)
    legend["record_pair_fields"] = list(brief.PAIR_FIELDS)
    legend["record_pair_values"] = "exact [later,earlier]; pre25=round0..24; round25 is separate; old P0=round0..150; only round-record fields, not model state or client diagnostics"
    legend["semantics"] = [s for s in legend["semantics"] if not s.startswith("A->B")]
    legend["semantics"] += ["H1 freezes history deque/live normal from commit25, retaining round24 state rather than restoring K20 anchor; H0 is original",
        "legacy client history_frozen is the weight manager's guard, not this intervention; use task variant/freeze and actual history_admitted",
        "H0-oldP0 repeated-condition numerical differences remain unexplained; no statistical significance or automatic next stage is inferred",
        "compact windows retain raw observation counts and n/mean/max; individual trajectory values remain in original snapshots"]
    records = [header, legend]
    for row in report.get("clean_reference_rows", []):
        records.append({"type": "reference_C0", "partition": row["partition"], "seed": row["seed"],
            "healthy": row.get("healthy"), "acc150": row.get("accuracy150"), "bg": row.get("background_5_to_7")})
    for row in report.get("reference_p0_rows", []):
        compact = brief.compact_task(row)
        compact["type"] = "reference_P0"
        records.append(compact)
    for row in report.get("rows", []):
        compact = brief.compact_task(row)
        audit = row["history_audit"]
        compact.update(variant=row["variant"], freeze=row["history_freeze_start_round"], fp=row["task_fingerprint"],
            env=row["task_environment_compatible"], device=row.get("recorded_device"),
            hf=[audit.get(k) for k in ("status", "verified", "observed_verified_finite_updates", "remaining_client_rounds",
                "unobserved_remaining_client_rounds", "observed_admissions", "observed_admission_rate", "invalid_admission_flags",
                "first_invalid_flag", "first_observed_admission")])
        if "window" in audit:
            compact["wh"] = brief.compact_window(audit["window"])
        for key, source in (("history_error", audit), ("additional_evidence_error", row)):
            value = source.get("error" if key == "history_error" else key)
            if value is not None:
                compact[key] = brief.short_error(value)
        records.append(compact)
    records.extend({"type": "candidate", **candidate} for candidate in report.get("candidates", []))
    for label, comparison in report.get("comparisons", {}).items():
        records.append({"type": "comparison", "label": label,
            **{k: v for k, v in comparison.items() if k != "pairs"},
            "pairs": [[p[field] for field in PAIR_FIELDS] for p in comparison["pairs"]]})
    records.extend({"type": "record_pair_audit", **audit} for audit in report.get("record_pair_audits", []))
    if "decision" in report:
        records.append({"type": "decision", **report["decision"]})
    lines = [json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) for row in records]
    print("=== CIFAR_HISTORY_BEGIN ===", flush=True)
    for line in lines:
        print(line)
    print("=== CIFAR_HISTORY_END ===", flush=True)
