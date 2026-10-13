"""Validation-only promotion policy for the independent 630-run CIFAR study.

Training and health requirements remain those of the original six-method
runner. This module selects validation candidates and controls promotion only;
it never reads formal results, changes any optimizer, or starts training.
"""
from __future__ import annotations

from collections import Counter
import math
from statistics import fmean

import run_cifar_six_from_scratch as base
from sm9rrsfl.calibration_policy import weighted_score


TOLERANCE = 1e-12


def _check_validation_plan(spec, tasks):
    shared = spec["shared_parameters"]
    if shared["rounds"] != 100 or shared["attack_start_round"] != 25:
        raise ValueError("mean dual gate requires final round 100 and attacked ASR rounds 25..100")
    validation = spec["validation"]
    scenarios = validation["scenarios"]
    ratios = {s["malicious_ratio"] for s in scenarios}
    pairs = {(s["partition"], s["malicious_ratio"]) for s in scenarios}
    if (len(validation["seeds"]) != 3 or len(set(validation["seeds"])) != 3
            or len(scenarios) != 10 or len(ratios) != 5 or 0 not in ratios
            or pairs != {(p, r) for p in ("iid", "dirichlet") for r in ratios}
            or any(not 0 <= ratio < 1 for ratio in ratios)
            or set(validation["seeds"]) & set(spec["final"]["seeds"])):
        raise ValueError("mean dual gate requires 3 independent validation seeds and 2 partitions x 5 ratios")
    expected = base.build_tasks(spec, "validation")
    if len(expected) != 630 or len(tasks) != len(expected):
        raise ValueError("mean dual gate requires exactly 630 validation tasks")
    wanted = {t["task_id"]: t for t in expected}
    actual = {}
    for task in tasks:
        identity = {k: v for k, v in task.items() if k != "fingerprint"}
        task_id = task.get("task_id")
        if task_id in actual or wanted.get(task_id) != identity:
            raise ValueError("gate input differs from the declared validation-only task plan")
        actual[task_id] = task
    return expected


SCORABLE_HEALTH_FAILURES = frozenset({
    "nonfinite_updates", "clean_accuracy_drop", "all_clients_revoked",
    "all_honest_revoked", "clean_false_revocation_rate", "honest_weight_starvation",
})
SCORE_FIELDS = ("clean_accuracy", "robust_accuracy", "attack_success_rate", "honest_weight_loss")


def _unit_interval(value, upper=1.):
    try:
        return value is not None and math.isfinite(value) and 0 <= value <= upper
    except TypeError:
        return False


def _run_audit(run):
    """Reject absent/invalid source metrics before legacy pessimistic sentinels."""
    reasons = set()
    records = run.records
    if (run.stopped_round != 100 or [r.round for r in records] != list(range(101))):
        reasons.add("incomplete_rounds")
    if not _unit_interval(run.final_accuracy) or not records or any(
            not _unit_interval(r.accuracy) for r in records):
        reasons.add("invalid_accuracy")
    elif abs(run.final_accuracy - records[-1].accuracy) > TOLERANCE:
        reasons.add("final_accuracy_record_mismatch")
    if any(not _unit_interval(getattr(r, key, None), upper=1. + 1e-9) for r in records
           for key in ("honest_weight_loss", "malicious_weight_mass")):
        reasons.add("invalid_weight_metrics")
    if any(r.attack_target_confidence is not None and not _unit_interval(r.attack_target_confidence)
           for r in records):
        reasons.add("invalid_attack_confidence")
    attacked = [r for r in records if r.round >= 25] if run.config.malicious_ratio > 0 else []
    if run.config.malicious_ratio > 0 and (len(attacked) != 76 or any(
            not _unit_interval(r.attack_target_success_rate) for r in attacked)):
        reasons.add("invalid_attack_metrics")
    # The original health policy stays authoritative. Malformed health fields
    # cannot be interpreted as an algorithm failure or a usable scored run.
    try:
        reasons.update(base.metrics(run)["reasons"])
    except (TypeError, ValueError, AttributeError, OverflowError):
        reasons.add("invalid_health_metrics")
    return {"healthy": not reasons, "reasons": sorted(reasons),
            "scorable": not (reasons - SCORABLE_HEALTH_FAILURES),
            "final_accuracy": run.final_accuracy if _unit_interval(run.final_accuracy) else None,
            "attack_mean_asr": (fmean(r.attack_target_success_rate for r in attacked)
                                if attacked and "invalid_attack_metrics" not in reasons else None)}


def _task_metric_rows(runs, expected_tasks):
    """One final accuracy and one attacked-round mean ASR per declared task."""
    by_config = {base.digest(base.semantic_config(run.config)): run for run in runs}
    rows = []
    for task in expected_tasks:
        run = by_config.get(base.digest(base.semantic_config(task["config"])))
        if run is None:
            continue
        metric = _run_audit(run)
        rows.append({"task_id": task["task_id"], "partition": run.config.partition,
                     "malicious_ratio": run.config.malicious_ratio, "seed": run.config.seed,
                     "accuracy_round": run.config.rounds, "final_accuracy": metric["final_accuracy"],
                     "asr_round_start": 25 if run.config.malicious_ratio > 0 else None,
                     "asr_round_end": 100 if run.config.malicious_ratio > 0 else None,
                     "attack_mean_asr": metric["attack_mean_asr"], "healthy": metric["healthy"],
                     "health_reasons": metric["reasons"], "scorable": metric["scorable"]})
    return rows


def select_validation(spec, results_by_candidate, tasks):
    """Choose healthy Ours with joint best means or both confirmed VERT gaps.

    Each baseline prefers its highest common-Score healthy candidate. If none
    is healthy, its highest raw common Score among complete, measurable failed
    candidates is selected and remains unqualified. A fixed identity is used
    only when no complete candidate can be scored. The selected VERT, including
    a scored health failure, supplies the near-VERT reference with explicit
    evidence quality. No incomplete run or formal-seed result can be scored.
    """
    expected = _check_validation_plan(spec, tasks)
    settings = spec.get("mean_dual_gate", {})
    accuracy_limit = settings.get("near_vert_accuracy_gap", .005)
    asr_limit = settings.get("near_vert_asr_gap", .01)
    if not _unit_interval(accuracy_limit) or not _unit_interval(asr_limit):
        raise ValueError("near-VERT limits must be finite nonnegative proportions")
    planned = {}
    for task in expected:
        planned.setdefault(task["candidate"]["candidate_id"], []).append(task)
    usable, identity_errors, measured = {}, {}, {}
    for cid, candidate_tasks in planned.items():
        runs = results_by_candidate.get(cid, [])
        observed = Counter(base.digest(base.semantic_config(r.config)) for r in runs)
        wanted = Counter(base.digest(base.semantic_config(t["config"])) for t in candidate_tasks)
        if observed != wanted:
            identity_errors[cid] = "incomplete_or_mismatched_validation_results"
            usable[cid], measured[cid] = [], []
            if candidate_tasks[0]["method"] == "fedavg":
                # A missing/foreign attacked result must not erase independent,
                # correctly identified clean controls and relax Ours' health.
                usable[cid] = [run for run in runs if run.config.malicious_ratio == 0
                               and observed[base.digest(base.semantic_config(run.config))] == 1
                               and wanted[base.digest(base.semantic_config(run.config))] == 1
                               and _run_audit(run)["healthy"]]
            continue
        measured[cid] = _task_metric_rows(runs, candidate_tasks)
        # Do not let legacy aggregation encounter malformed source metrics or
        # manufacture a finite Score from defaults for absent data.
        usable[cid] = [run for run in runs if _run_audit(run)["scorable"]]
    report = base.select_validation(spec, usable, tasks)
    original_ours = report["selected"].get("sm9rrs")
    trials = {row["candidate_id"]: row for row in report["trials"]}
    candidate_rows = {}
    for cid in planned:
        trial = trials[cid]
        rows = measured[cid]
        attacked = [r for r in rows if r["malicious_ratio"] > 0]
        if cid in identity_errors or len(usable[cid]) != 30:
            reasons = ({identity_errors[cid]} if cid in identity_errors else
                       {reason for row in rows for reason in row["health_reasons"]})
            trial["invalid_reasons"] = ",".join(sorted(reasons))
            trial["result_count"] = len(results_by_candidate.get(cid, []))
            report["methods"][trial["method"]]["candidate_failures"][cid] = trial["invalid_reasons"]
        reasons = set(filter(None, trial["invalid_reasons"].split(",")))
        scorable = bool(len(rows) == 30 and len(attacked) == 24
                        and all(r["scorable"] for r in rows)
                        and not (reasons - SCORABLE_HEALTH_FAILURES)
                        and all(_unit_interval(trial.get(key)) for key in SCORE_FIELDS)
                        and _unit_interval(trial.get("worst_attack_success_rate")))
        raw_score = weighted_score(trial, spec["objective"]) if scorable else None
        if raw_score is not None and not math.isfinite(raw_score):
            raise ValueError("scorable candidate unexpectedly has nonfinite common Score")
        eligible = bool(trial["valid"] and scorable)
        mean_accuracy = fmean(r["final_accuracy"] for r in rows) if scorable else None
        mean_asr = fmean(r["attack_mean_asr"] for r in attacked) if scorable else None
        trial.update(raw_score=raw_score, scorable=scorable, health_qualified=eligible)
        candidate_rows[cid] = {"candidate_id": cid, "method": trial["method"], "eligible": eligible,
                               "health_qualified": eligible, "scorable": scorable,
                               "invalid_reasons": trial["invalid_reasons"], "common_score": trial.get("score"),
                               "raw_score": raw_score,
                               "observed_result_count": len(results_by_candidate.get(cid, [])),
                               "matched_validation_tasks": len(rows), "attacked_tasks": len(attacked),
                               "mean_accuracy": mean_accuracy, "mean_asr": mean_asr, "tasks": rows}

    def rank(cid):
        return (candidate_rows[cid]["raw_score"], -trials[cid]["worst_attack_success_rate"], cid)

    ours = [cid for cid, row in candidate_rows.items() if row["method"] == "sm9rrs" and row["eligible"]]
    available, unavailable = {}, {}
    for method in base.ALL_METHODS[1:]:
        info = report["methods"][method]
        healthy = [cid for cid, row in candidate_rows.items() if row["method"] == method and row["eligible"]]
        scored = [cid for cid, row in candidate_rows.items() if row["method"] == method and row["scorable"]]
        cid = max(healthy or scored, key=rank) if healthy or scored else spec["fallback_candidates"][method]
        row = candidate_rows[cid]
        status = "eligible_score_selection" if healthy else "best_scored_unqualified" if scored else "fixed_fallback_unqualified"
        report["selected"][method] = cid
        info.update(selected_candidate=cid, selection_status=status,
                    selection_rule="healthy_score_then_complete_score_then_fixed",
                    health_qualified=row["eligible"], scorable=row["scorable"], raw_score=row["raw_score"],
                    selection_raw_score=row["raw_score"],
                    comparison_available=row["eligible"], scored_comparison_available=row["scorable"],
                    selected_without_valid_validation=not row["eligible"],
                    validation_selected_candidate_failures=row["invalid_reasons"],
                    scorable_failed_candidate_selected=status == "best_scored_unqualified")
        if row["scorable"]:
            available[method] = {"candidate_id": cid, "selection_rule": status,
                                 "health_qualified": row["eligible"], "scorable": True,
                                 "raw_score": row["raw_score"], "health_reasons": row["invalid_reasons"],
                                 "evidence_quality": "healthy" if row["eligible"] else "complete_scored_health_failure",
                                 "mean_accuracy": row["mean_accuracy"], "mean_asr": row["mean_asr"]}
        else:
            unavailable[method] = {"status": "unassessed", "reason": "no_complete_scorable_validation_candidate",
                                   "fallback_candidate": cid, "fallback_qualified": False,
                                   "blocks_promotion": False, "candidate_failures": info["candidate_failures"]}
    comparators = ours + [item["candidate_id"] for item in available.values()]
    best_accuracy = max((candidate_rows[cid]["mean_accuracy"] for cid in comparators), default=None)
    best_asr = min((candidate_rows[cid]["mean_asr"] for cid in comparators), default=None)
    vert = available.get("vert")
    targets, near_targets = {}, {}
    for cid in ours:
        row = candidate_rows[cid]
        accuracy_gap = best_accuracy - row["mean_accuracy"]
        asr_gap = row["mean_asr"] - best_asr
        differences = [{"candidate_id": ref, "method": candidate_rows[ref]["method"],
                        "reference_health_qualified": candidate_rows[ref]["health_qualified"],
                        "accuracy_difference_ours_minus_reference": row["mean_accuracy"] - candidate_rows[ref]["mean_accuracy"],
                        "asr_difference_ours_minus_reference": row["mean_asr"] - candidate_rows[ref]["mean_asr"]}
                       for ref in comparators]
        joint = accuracy_gap <= TOLERANCE and asr_gap <= TOLERANCE
        vert_accuracy_gap = vert["mean_accuracy"] - row["mean_accuracy"] if vert else None
        vert_asr_gap = row["mean_asr"] - vert["mean_asr"] if vert else None
        near = bool(vert and vert_accuracy_gap <= accuracy_limit + TOLERANCE
                    and vert_asr_gap <= asr_limit + TOLERANCE)
        route = "joint_best" if joint else "near_vert" if near else None
        near_targets[cid] = {"status": "passed" if near else "unmet" if vert else "unassessed",
                             "accuracy_gap_to_vert": vert_accuracy_gap, "asr_gap_to_vert": vert_asr_gap,
                             "reference_health_qualified": vert["health_qualified"] if vert else None}
        targets[cid] = {"status": "passed" if route else "unmet", "role": "selection_and_promotion_gate",
                        "mean_accuracy": row["mean_accuracy"], "mean_asr": row["mean_asr"],
                        "accuracy_gap_to_best": accuracy_gap, "asr_gap_to_best": asr_gap,
                        "joint_best_passed": joint, "near_vert_passed": near, "pass_route": route,
                        "near_vert": near_targets[cid], "differences": differences}
    joint_qualified = [cid for cid in ours if targets[cid]["joint_best_passed"]]
    near_qualified = [cid for cid in ours if targets[cid]["near_vert_passed"]]
    qualified = [cid for cid in ours if targets[cid]["status"] == "passed"]
    preferred = joint_qualified or near_qualified
    selected = max(preferred, key=rank) if preferred else None
    pass_route = targets[selected]["pass_route"] if selected else None
    unqualified_references = [method for method, info in available.items() if not info["health_qualified"]]
    scope = ("all_six_methods" if len(available) == 5 else "ours_and_available_qualified_baselines"
             if available else "ours_candidates_only_no_qualified_baselines")
    if unqualified_references:
        scope = "ours_and_scorable_selected_baselines_including_health_failures"
    audit = {"status": "passed" if selected else "unmet" if ours else "no_healthy_ours_candidate",
             "role": "selection_and_promotion_gate", "validation_only": True,
             "pass_route": pass_route, "selected_pass_route": pass_route,
             "near_vert_reference": vert,
             "definition": {"accuracy": "equal mean of final round-100 accuracy across all 30 validation tasks",
                            "asr": "equal mean across 24 attacked tasks of each task's mean ASR over rounds 25..100 inclusive",
                            "accuracy_task_count": 30, "asr_task_count": 24, "asr_round_count_per_task": 76,
                            "clean_task_asr_included": False, "numeric_tolerance": TOLERANCE,
                            "acceptance": "joint_best_or_near_vert",
                            "both_optima_must_be_attained_by_one_candidate": True,
                            "joint_optima_required_for_joint_route_only": True,
                            "tie_breaker": "joint-best tier first, then near-VERT tier; original common Score, lower worst ASR, candidate id"},
             "comparison_scope": scope,
             "six_method_dual_optimality_assessed": len(available) == 5 and not unqualified_references and bool(ours),
             "six_method_observed_means_compared": len(available) == 5 and bool(ours),
             "comparison_includes_unqualified_references": bool(unqualified_references),
             "unqualified_reference_methods": unqualified_references,
             "eligible_ours_candidates": ours, "qualified_ours_candidates": qualified,
             "joint_qualified_ours_candidates": joint_qualified, "near_vert_qualified_ours_candidates": near_qualified,
             "comparator_candidates": comparators, "available_baselines": available,
             "unavailable_baselines": unavailable, "best_mean_accuracy": best_accuracy,
             "best_mean_asr": best_asr, "selected_candidate": selected,
             "original_common_score_ours_candidate": original_ours,
             "candidate_rows": candidate_rows, "ours_candidate_targets": targets,
             "source": "declared validation task results; no formal/test results used"}
    report["ours_target"] = report["ours_candidate_targets"].get(selected, {
        "status": "incomplete", "role": "descriptive_only_not_selection_or_promotion"})
    report["mean_dual_gate"] = audit
    report["near_vert_gate"] = {
        "status": (near_targets[selected]["status"] if selected else "unassessed" if not vert else
                   "no_healthy_ours_candidate" if not ours else "unmet"),
        "validation_only": True, "reference_candidate": vert["candidate_id"] if vert else None,
        "reference_health_qualified": vert["health_qualified"] if vert else None,
        "reference_scorable": bool(vert),
        "accuracy_gap_limit": accuracy_limit, "asr_gap_limit": asr_limit,
        "selected_candidate": selected, "candidate_targets": near_targets,
        "reference_selection_rule": report["methods"]["vert"]["selection_status"],
        "unavailable_reference_does_not_imply_pass": True}
    report["best_healthy_score_candidate"] = original_ours
    report["ours_health_passed"] = bool(ours)
    report["ours_mean_dual_passed"] = bool(selected)
    report["ours_joint_mean_dual_passed"] = pass_route == "joint_best"
    report["ours_relative_performance_passed"] = bool(selected)
    report["pass_route"] = pass_route
    report["status"] = "qualified_for_final" if selected else "needs_ours_dual_development" if ours else "needs_ours_development"
    report["promotion_basis"].update(relative_performance_required=True,
                                     relative_performance_rule="joint_best_or_near_vert",
                                     near_vert_accuracy_gap=accuracy_limit, near_vert_asr_gap=asr_limit,
                                     unavailable_baselines_are_unassessed_not_passed=True)
    report["selection_rule"] = "ours_healthy_joint_best_then_near_vert_then_common_score; baselines_healthy_score_then_complete_score_then_fixed"
    info = report["methods"]["sm9rrs"]
    info["selected_candidate"] = selected
    info["selection_status"] = ("healthy_mean_dual_selection" if pass_route == "joint_best" else
                                "healthy_near_vert_selection" if selected else
                                "no_qualified_relative_performance_ours_candidate" if ours else "no_eligible_ours_candidate")
    info.update(health_qualified=bool(selected), scorable=bool(selected),
                raw_score=candidate_rows[selected]["raw_score"] if selected else None,
                selection_raw_score=candidate_rows[selected]["raw_score"] if selected else None,
                pass_route=pass_route, selected_without_valid_validation=False)
    if selected:
        report["selected"]["sm9rrs"] = selected
    else:
        report["selected"].pop("sm9rrs", None)
    return report
