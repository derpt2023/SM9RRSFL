"""Read-only evidence for the CNN K/attack-start diagnostic; no candidate selection."""
import json
from pathlib import Path
from statistics import fmean

import cifar_adaptive_runtime as runtime

base = runtime.base
DEV_SEED = 2026093001


def finite(value):
    import math
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _count(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("invalid " + name)
    return value


def _probability(value, name, tolerance=0.):
    if not finite(value) or not 0 <= value <= 1 + tolerance:
        raise ValueError("invalid " + name)
    return value


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _records(result):
    records = result.records
    if not records or [r.round for r in records] != list(range(records[-1].round + 1)):
        raise ValueError("missing, duplicated or out-of-order round records")
    if result.stopped_round != records[-1].round or result.stopped_round > result.config.rounds:
        raise ValueError("snapshot stopped round does not match its records")
    if result.final_accuracy != records[-1].accuracy:
        raise ValueError("snapshot final accuracy does not match its last record")
    malicious_count = len(result.malicious_clients)
    if len(set(result.malicious_clients)) != malicious_count:
        raise ValueError("duplicated original malicious identities")
    if malicious_count != round(result.config.num_clients * result.config.malicious_ratio):
        raise ValueError("original malicious identity count differs from the declared ratio")
    previous_tp = previous_fp = 0
    for r in records:
        _probability(r.accuracy, "accuracy")
        _probability(r.attack_target_success_rate, "source-to-target rate")
        _probability(r.malicious_weight_mass, "malicious weight mass", 1e-9)
        _probability(r.honest_weight_loss, "honest weight loss", 1e-9)
        tp = _count(r.true_positive_revocations, "true-positive revocation count")
        fp = _count(r.false_positive_revocations, "false-positive revocation count")
        if not previous_tp <= tp <= malicious_count or not previous_fp <= fp <= result.config.num_clients - malicious_count:
            raise ValueError("revocation counters are inconsistent with the original client population")
        previous_tp, previous_fp = tp, fp
    return {r.round: r for r in records}


def _diagnostics(result, records):
    indexed, seen = {}, set()
    malicious = set(result.malicious_clients)
    if result.config.method != "sm9rrs":
        if result.diagnostics:
            raise ValueError("FedAvg unexpectedly contains SM9 client diagnostics")
        return indexed
    for d in result.diagnostics:
        key = (d.round, d.client_id)
        if key in seen or d.round < 1 or d.round not in records:
            raise ValueError("duplicated or out-of-range client diagnostic")
        seen.add(key)
        if d.is_malicious != (d.client_id in malicious):
            raise ValueError("client diagnostic malicious label disagrees with the original identities")
        for attr in ("is_malicious", "aggregation_accepted", "history_admitted", "revoked", "attack_active"):
            if type(getattr(d, attr)) is not bool:
                raise ValueError("invalid diagnostic flag " + attr)
        _probability(d.aggregation_weight, "client aggregation coefficient", 1e-9)
        if d.aggregation_accepted != (d.aggregation_weight > 0):
            raise ValueError("aggregation acceptance flag disagrees with its coefficient")
        if d.history_admitted and (not d.aggregation_accepted or d.revoked):
            raise ValueError("history admission contradicts aggregation acceptance or revocation")
        start = result.config.attack_start_round or result.config.detector_window + 2
        if d.attack_active != (d.is_malicious and d.round >= start and result.config.attack != "none"):
            raise ValueError("diagnostic attack-active flag disagrees with the declared attack window")
        indexed.setdefault(d.round, []).append(d)
    return indexed


def mechanism_window(result, records, diagnostics, start, length):
    """Denominators are observed verified finite updates, never the nominal population."""
    wanted = list(range(start, start + length))
    available = [rd for rd in wanted if rd in records]
    window = {"rounds_requested": wanted, "rounds_observed": available,
        "complete_round_records": len(available) == length,
        "window_role": "attack" if result.config.malicious_ratio else "same_scheduled_rounds_without_attack"}
    for field in ("malicious_weight_mass", "honest_weight_loss"):
        values = [getattr(records[rd], field) for rd in available]
        window[field] = {"values": values, "n": len(values),
                         "mean": fmean(values) if values else None, "max": max(values) if values else None}
    if result.config.method != "sm9rrs":
        window["client_diagnostics"] = {"status": "not_available_for_fedavg",
            "malicious": None, "honest": None,
            "note": "FedAvg has round aggregation-weight metrics but no SM9 per-client diagnostic records"}
        return window
    group_stats = {}
    for name, malicious_flag, total, counter in (
        ("malicious", True, len(result.malicious_clients), "true_positive_revocations"),
        ("honest", False, result.config.num_clients - len(result.malicious_clients), "false_positive_revocations")):
        coverage = []
        observed = []
        for rd in available:
            previous = records.get(rd - 1)
            prior_revoked = getattr(previous, counter) if previous is not None else None
            active = total - prior_revoked if prior_revoked is not None else None
            subset = [d for d in diagnostics.get(rd, []) if d.is_malicious == malicious_flag]
            if active is not None and len(subset) > active:
                raise ValueError("more client diagnostics than clients remaining before this round")
            observed.extend(subset)
            coverage.append({"round": rd, "prior_revoked_clients": prior_revoked,
                "clients_remaining_before_round": active, "observed_verified_finite_updates": len(subset),
                "unobserved_remaining_clients": active - len(subset) if active is not None else None})
        observed_n = len(observed)
        accepted = sum(d.aggregation_accepted for d in observed)
        admitted = sum(d.history_admitted for d in observed)
        prior_known = bool(coverage) and all(c["prior_revoked_clients"] is not None for c in coverage)
        remaining = sum(c["clients_remaining_before_round"] for c in coverage) if prior_known else None
        group_stats[name] = {"original_clients": total,
            "nominal_client_rounds_requested": total * length,
            "nominal_client_rounds_with_round_records": total * len(available),
            "prior_revoked_client_rounds": sum(c["prior_revoked_clients"] for c in coverage) if prior_known else None,
            "remaining_client_rounds": remaining,
            "observed_verified_finite_updates": observed_n,
            "unobserved_remaining_client_rounds": remaining - observed_n if remaining is not None else None,
            "status": "not_applicable_no_original_clients" if total == 0 else
                      "observed" if observed_n else
                      "no_remaining_clients" if available and remaining == 0 else "no_observed_records",
            "aggregation_accepted": accepted, "aggregation_acceptance_rate": _rate(accepted, observed_n),
            "history_admitted": admitted, "history_admission_rate": _rate(admitted, observed_n),
            "history_admitted_among_accepted_rate": _rate(sum(d.history_admitted and d.aggregation_accepted for d in observed), accepted),
            "revoked_observed_events": sum(d.revoked for d in observed), "coverage_by_round": coverage}
    total_observed = sum(g["observed_verified_finite_updates"] for g in group_stats.values())
    window["client_diagnostics"] = {"status": "available" if total_observed else "no_observed_records",
        **group_stats,
        "denominator": "observed verified finite online client updates in the requested rounds, grouped by original malicious identity",
        "unobserved_is_not_counted_as_rejected": True}
    if available:
        first, last = records.get(start - 1), records[available[-1]]
        window["new_honest_revocations"] = (last.false_positive_revocations - first.false_positive_revocations
                                               if first is not None else None)
        window["new_designated_malicious_revocations"] = (last.true_positive_revocations - first.true_positive_revocations
                                                           if first is not None else None)
    return window


def _attempts(folder, task):
    attempts = [runtime.read_json(p) for p in sorted((folder / "attempts").glob("*.json"))]
    if any(a.get("task_fingerprint") != task["fingerprint"] for a in attempts):
        raise ValueError("attempt provenance has the wrong task fingerprint")
    for attempt in attempts:
        if attempt.get("status") not in ("running", "complete", "failed"):
            raise ValueError("invalid worker attempt status")
        if attempt["status"] != "running" or "wall_seconds" in attempt:
            if not finite(attempt.get("wall_seconds")) or attempt["wall_seconds"] < 0:
                raise ValueError("invalid worker elapsed time")
    unfinished = sum(a["status"] == "running" for a in attempts)
    return {"worker_wall_seconds": sum(a.get("wall_seconds", 0) for a in attempts) if attempts else None,
        "unfinished_attempts": unfinished if attempts else None,
        "worker_cost_exact": bool(attempts) and unfinished == 0,
        "attempt_provenance": "complete" if attempts and not unfinished else "unfinished" if attempts else "missing",
        "cuda_peak_allocated_mib": max((a["cuda_peak_allocated_mib"] for a in attempts
            if finite(a.get("cuda_peak_allocated_mib"))), default=None)}


def full_attack_period(result, records, diagnostics, start):
    """Keep later history contamination visible without printing 139 per-round lists."""
    if not result.config.malicious_ratio:
        return {"status": "not_applicable_clean"}
    window = mechanism_window(result, records, diagnostics, start, result.config.rounds - start + 1)
    requested = window.pop("rounds_requested")
    observed = window.pop("rounds_observed")
    window.update(start_round=start, end_round=result.config.rounds,
                  expected_rounds=len(requested), observed_rounds=len(observed),
                  last_observed_round=observed[-1] if observed else None)
    for field in ("malicious_weight_mass", "honest_weight_loss"):
        window[field].pop("values")
    for group in ("malicious", "honest"):
        if window["client_diagnostics"][group] is not None:
            window["client_diagnostics"][group].pop("coverage_by_round")
    return window


def collect(output, tasks):
    rows = []
    for task in tasks:
        config = task["config"]
        start = config["attack_start_round"] or config["detector_window"] + 2
        folder = output / "tasks" / task["task_id"]
        row = {"task_id": task["task_id"], "arm": task["arm"], "method": config["method"],
            "partition": config["partition"], "seed": config["seed"], "malicious_ratio": config["malicious_ratio"],
            "detector_window": config["detector_window"], "attack_start_round": start,
            "scheduled_attack_rounds": config["rounds"] - start + 1 if config["malicious_ratio"] else 0,
            "status": "pending", "healthy": None}
        try:
            if folder.exists() and any(folder.iterdir()) and runtime.read_json(folder / "task.json") != task:
                raise ValueError("task identity differs from the declared timing diagnostic task")
            result = base.checked_completed(output, task)
            if result is None:
                failure = runtime.terminal_failure(output, task)
                if failure:
                    row.update(status=failure["kind"], error=failure.get("message"), exception=failure.get("exception"))
                progress = folder / "progress.json"
                if progress.exists():
                    row["last_completed_round"] = runtime.read_json(progress).get("last_completed_round")
                log = folder / "worker.log"
                if log.exists():
                    with log.open("rb") as handle:
                        handle.seek(max(0, log.stat().st_size - 2000))
                        row["log_tail"] = handle.read().decode("utf-8", errors="replace")[-1200:]
            else:
                records, metric = _records(result), base.metrics(result)
                diagnostics = _diagnostics(result, records)
                last = result.records[-1]
                final = records.get(150)
                honest = config["num_clients"] - len(result.malicious_clients)
                previous_attack = records.get(start - 1)
                windows = {"first_round": mechanism_window(result, records, diagnostics, start, 1),
                           "first_five_rounds": mechanism_window(result, records, diagnostics, start, 5)}
                row.update(status="complete" if result.stopped_round == 150 else "terminal_incomplete",
                    snapshot_present=True, stopped_round=result.stopped_round,
                    healthy=metric["healthy"], reasons=metric["reasons"],
                    accuracy50=records[50].accuracy if 50 in records else None,
                    accuracy100=records[100].accuracy if 100 in records else None,
                    accuracy150=final.accuracy if final is not None else None,
                    source_5_to_7_final=final.attack_target_success_rate if final is not None else None,
                    attack_asr150=final.attack_target_success_rate if final is not None and config["malicious_ratio"] else None,
                    background_5_to_7=final.attack_target_success_rate if final is not None and not config["malicious_ratio"] else None,
                    last_observed_accuracy=last.accuracy, last_observed_source_5_to_7=last.attack_target_success_rate,
                    original_malicious_clients=len(result.malicious_clients), original_honest_clients=honest,
                    false_positive_revocations=last.false_positive_revocations,
                    false_positive_revocation_rate=_rate(last.false_positive_revocations, honest),
                    true_positive_revocations=last.true_positive_revocations,
                    pre_attack_designated_malicious_revoked=(previous_attack.true_positive_revocations if previous_attack is not None else None),
                    pre_attack_honest_revoked=(previous_attack.false_positive_revocations if previous_attack is not None else None),
                    nonfinite_updates=result.nonfinite_updates, measured_runtime_seconds=result.runtime_seconds,
                    mechanism_windows=windows,
                    full_attack_period=full_attack_period(result, records, diagnostics, start), **_attempts(folder, task))
        except Exception as exc:
            row.update(status="invalid_evidence", healthy=None, error=str(exc))
        rows.append(row)
    return rows


def add_clean_controls(rows, reference_rows):
    c0 = {(r["partition"], r["seed"]): r for r in reference_rows if r["setting"] == "C0"}
    own = {(r["arm"], r["partition"], r["seed"]): r for r in rows
           if r["method"] == "sm9rrs" and r["malicious_ratio"] == 0}
    for row in rows:
        if row["status"] != "complete":
            continue
        clean = c0.get((row["partition"], row["seed"]))
        if row["method"] == "sm9rrs" and not row["malicious_ratio"]:
            usable = clean is not None and clean["status"] == "complete" and finite(clean.get("accuracy150"))
            gap = 100 * (clean["accuracy150"] - row["accuracy150"]) if usable else None
            row["clean_utility_vs_C0"] = {"available": usable, "accuracy_drop_pp": gap,
                "C0_accuracy150": clean["accuracy150"] if usable else None,
                "within_3pp": gap <= 3 + 1e-10 if usable else None,
                "role": "separate clean utility diagnostic; base health and formal gates are unchanged"}
        elif row["method"] == "sm9rrs":
            control = own.get((row["arm"], row["partition"], row["seed"]))
            usable = control is not None and control["status"] == "complete"
            row["attack_vs_same_arm_clean"] = {"available": usable,
                "control_task_id": control["task_id"] if control else None,
                "control_healthy": control.get("healthy") if control else None,
                "attack_accuracy_loss_pp": 100 * (control["accuracy150"] - row["accuracy150"]) if usable else None,
                "attack_final_minus_clean_background_pp": 100 * (row["attack_asr150"] - control["background_5_to_7"]) if usable else None,
                "clean_background_5_to_7": control["background_5_to_7"] if usable else None,
                "role": "descriptive same-arm clean control; malicious participation differs"}


def comparisons(rows):
    keyed = {(r["arm"], r["partition"], r["seed"], r["malicious_ratio"]): r for r in rows}
    output = {}
    for label, earlier, later, ratios in (("A_to_B", "A", "B", (0., .1, .7)),
                                          ("B_to_C", "B", "C", (0., .1, .7)),
                                          ("FA12_to_FA25", "FA12", "FA25", (.1, .7))):
        paired = []
        for partition in ("iid", "dirichlet"):
            for ratio in ratios:
                a, b = (keyed.get((arm, partition, DEV_SEED, ratio)) for arm in (earlier, later))
                usable = (a is not None and b is not None and a["status"] == b["status"] == "complete")
                paired.append({"partition": partition, "seed": DEV_SEED, "malicious_ratio": ratio,
                    "available": usable, "both_healthy": bool(usable and a["healthy"] and b["healthy"]),
                    "later_minus_earlier_accuracy_pp": 100 * (b["accuracy150"] - a["accuracy150"]) if usable else None,
                    "later_minus_earlier_source_5_to_7_pp": 100 * (b["source_5_to_7_final"] - a["source_5_to_7_final"]) if usable else None,
                    "source_to_target_role": "attack ASR" if ratio else "clean background confusion"})
        usable_pairs = [p for p in paired if p["available"]]
        # Clean background confusion and attack ASR must never be averaged together.
        attacked = [p for p in usable_pairs if p["malicious_ratio"]]
        output[label] = {"pairs": paired, "paired_n": len(usable_pairs), "expected_pairs": len(paired),
            "healthy_paired_n": sum(p["both_healthy"] for p in usable_pairs),
            "attacked_paired_n": len(attacked),
            "mean_attacked_accuracy_change_pp": fmean(p["later_minus_earlier_accuracy_pp"] for p in attacked) if attacked else None,
            "mean_attacked_asr_change_pp": fmean(p["later_minus_earlier_source_5_to_7_pp"] for p in attacked) if attacked else None}
    joint = []
    for ours in output["A_to_B"]["pairs"]:
        if not ours["malicious_ratio"]:
            continue
        fedavg = next(p for p in output["FA12_to_FA25"]["pairs"] if
                      (p["partition"], p["malicious_ratio"]) == (ours["partition"], ours["malicious_ratio"]))
        joint.append({"partition": ours["partition"], "seed": DEV_SEED, "malicious_ratio": ours["malicious_ratio"],
            "both_methods_available": ours["available"] and fedavg["available"],
            "Ours_ASR_change_pp": ours["later_minus_earlier_source_5_to_7_pp"],
            "FedAvg_ASR_change_pp": fedavg["later_minus_earlier_source_5_to_7_pp"],
            "Ours_accuracy_change_pp": ours["later_minus_earlier_accuracy_pp"],
            "FedAvg_accuracy_change_pp": fedavg["later_minus_earlier_accuracy_pp"]})
    output["attack_start_joint_review"] = joint
    output["interpretation"] = {"A_to_B": "K=10 unchanged; attack start 12 to 25 also reduces exposure from 139 to 126 rounds",
        "B_to_C": "attack start=25 unchanged; K changes 10 to 20",
        "limitations": "one development seed; process metrics are diagnostic only; lower ASR alone does not establish repaired detection or select a winner"}
    return output


def decision(rows, *, reference_verified, environment_compatible, source_matches):
    result = {"next_stage_started": False, "formal_qualification_assessed": False,
              "selected_arm": None, "performance_gate_changed": False}
    if not reference_verified:
        return {**result, "action": "resolve_changed_or_invalid_reference"}
    if not source_matches:
        return {**result, "action": "resolve_changed_source_identity"}
    if not environment_compatible:
        return {**result, "action": "resolve_missing_or_incompatible_execution_environment"}
    if len(rows) != 26 or any(r["status"] != "complete" for r in rows):
        return {**result, "action": "resolve_incomplete_or_invalid_execution_evidence"}
    if any(r["healthy"] is not True for r in rows):
        return {**result, "action": "review_unhealthy_completed_runs_before_next_stage"}
    missing = [r["task_id"] for r in rows if r["method"] == "sm9rrs" and
               any(w["client_diagnostics"]["status"] != "available" for w in r["mechanism_windows"].values())]
    if missing:
        return {**result, "action": "review_missing_client_diagnostic_evidence", "tasks_without_observations": missing}
    return {**result, "action": "review_timing_and_mechanism",
            "note": "review A/B and B/C with FedAvg exposure controls, clean utility, coverage and health; no automatic model or detector selection"}


def summarize(output, clean_output, matched_output):
    import cifar_timing_protocol as protocol
    output, clean_output, matched_output = Path(output), Path(clean_output), Path(matched_output)
    try:
        manifest, tasks = protocol.read_study(output, current_sources=False)
    except Exception as exc:
        return {"status": "unavailable_or_invalid_study", "output": str(output), "error": str(exc),
                "training_started_by_summary": False}
    reference = manifest["reference"]
    reference_verified, reference_error = False, None
    try:
        if protocol.audit_reference(clean_output, matched_output) != reference:
            raise ValueError("reference evidence differs from the immutable timing-study reference")
        reference_verified = True
    except Exception as exc:
        reference_error = str(exc)
    rows = collect(output, tasks)
    environment, environment_error = None, None
    try:
        environment = runtime.read_json(output / "execution_environment.json")
    except Exception as exc:
        environment_error = str(exc)
    compatible = environment is not None and environment == reference["execution_environment"]
    sources, source_error = False, None
    try:
        sources = protocol.source_hashes() == manifest["source_sha256"]
    except Exception as exc:
        source_error = str(exc)
    ready = reference_verified and compatible and sources
    reference_rows = [r for r in reference["clean_rows"] if r["seed"] == DEV_SEED] if reference_verified else []
    if ready:
        add_clean_controls(rows, reference_rows)
    complete = sum(r["status"] == "complete" for r in rows)
    healthy = sum(r["status"] == "complete" and r["healthy"] is True for r in rows)
    resolved = all(r["status"] in ("complete", "terminal_incomplete", "algorithm_numerical") for r in rows)
    result = {"status": "invalid_comparison_evidence" if not ready else "complete" if complete == 26 else
                        "resolved_with_failures" if resolved else "incomplete",
        "output": str(output), "protocol": manifest["spec"]["protocol"], "manifest_fingerprint": manifest["fingerprint"],
        "reference_verified": reference_verified, "source_matches_current": sources,
        "execution_environment_compatible": compatible,
        "execution_hardware": environment.get("actual_compute_device") if isinstance(environment, dict) else None,
        "complete_tasks": complete, "expected_tasks": 26, "healthy_tasks": healthy,
        "snapshot_tasks": sum(r.get("snapshot_present") is True for r in rows),
        "base_health_rule_unchanged": True, "clean_utility_rule": "separate <= 3 pp final-accuracy drop against paired C0; diagnostic only",
        "evaluation_split": "2500 calibration samples; official test unused for selection",
        "client_diagnostic_coverage": "SM9 verified finite updates only, from round 1; prior revoked clients are absent in later rounds; unobserved remaining clients are not inferred to be rejected",
        "pre_attack_revocation_semantics": "designated malicious identities revoked before attack start had not yet attacked; this count is not evidence of recognizing an active attack",
        "malicious_weight_mass_semantics": "sum of actual aggregation coefficients of original malicious clients, not a renormalized fraction of surviving total mass; raw values retained within 1e-9 tolerance",
        "training_started_by_summary": False, "reference_rows": reference_rows, "rows": rows,
        "comparisons": comparisons(rows) if ready else None,
        "decision": decision(rows, reference_verified=reference_verified, environment_compatible=compatible, source_matches=sources)}
    for key, value in (("reference_error", reference_error), ("environment_error", environment_error), ("source_error", source_error)):
        if value is not None:
            result[key] = value
    return result


def print_summary(report):
    print("=== CIFAR_TIMING_BEGIN ===", flush=True)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "reference_rows", "comparisons", "decision")},
                     ensure_ascii=False, indent=2, allow_nan=False))
    for row in report.get("reference_rows", []):
        print("REFERENCE_TASK " + json.dumps(row, ensure_ascii=False, allow_nan=False))
    for row in report.get("rows", []):
        print("TASK " + json.dumps(row, ensure_ascii=False, allow_nan=False))
    if report.get("comparisons") is not None:
        print("COMPARISONS " + json.dumps(report["comparisons"], ensure_ascii=False, allow_nan=False))
    if "decision" in report:
        print("DECISION " + json.dumps(report["decision"], ensure_ascii=False, allow_nan=False))
    print("=== CIFAR_TIMING_END ===", flush=True)
