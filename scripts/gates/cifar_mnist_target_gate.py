"""MNIST-strength, validation-only promotion for the expanded CIFAR study.

Absolute ASR and paired VERT limits are checked per seed and scenario, never
on seed means. Baseline health failures remain visible and do not prevent
selection of a complete, measurable baseline by the declared common Score.
No previous-study or formal result can substitute for a declared validation
configuration. This module does not train models or modify their algorithms.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import math
from statistics import fmean

import run_cifar_six_from_scratch as base
from cifar_mean_dual_gate import (SCORABLE_HEALTH_FAILURES, SCORE_FIELDS,
                                  TOLERANCE, _run_audit, _task_metric_rows,
                                  _unit_interval)
from sm9rrsfl.calibration_policy import weighted_score
from sm9rrsfl.performance_target import PerformanceTarget, scenario_key


def _check_validation_plan(spec, tasks):
    """Match the entire expanded plan, not a subset or an earlier study."""
    shared = spec["shared_parameters"]
    if shared["rounds"] != 100 or shared["attack_start_round"] != 25:
        raise ValueError("MNIST-target CIFAR gate requires rounds 0..100 and attack rounds 25..100")
    validation = spec["validation"]
    seeds, scenarios = validation["seeds"], validation["scenarios"]
    ratios = {s["malicious_ratio"] for s in scenarios}
    pairs = {(s["partition"], s["malicious_ratio"]) for s in scenarios}
    if (len(seeds) != 3 or len(set(seeds)) != 3
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(scenarios) != 10 or len(ratios) != 5 or 0 not in ratios
            or pairs != {(p, r) for p in ("iid", "dirichlet") for r in ratios}
            or any(not _unit_interval(r) or r == 1 for r in ratios)
            or set(seeds) & set(spec["final"]["seeds"])):
        raise ValueError("gate requires 3 independent validation seeds and 2 partitions x 5 ratios")
    if set(spec["candidates"]) != set(base.ALL_METHODS):
        raise ValueError("exactly six candidate method families are required")
    ids = []
    for method, candidates in spec["candidates"].items():
        if not candidates:
            raise ValueError("each method must have declared candidates")
        for candidate in candidates:
            ids.append(candidate["candidate_id"])
            if set(candidate["parameters"]) - base.METHOD_TUNABLE_PARAMETERS[method]:
                raise ValueError("candidate cannot alter shared or another method's parameters")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate ids")
    fallback = {m: cs[0]["candidate_id"] for m, cs in spec["candidates"].items() if m != "sm9rrs"}
    if spec["fallback_candidates"] != fallback:
        raise ValueError("baseline fallback must be each first declared candidate")
    policy = PerformanceTarget.parse(spec["performance_target"])
    if policy != PerformanceTarget():
        raise ValueError("MNIST targets require accuracy_gap=.02, asr_gap=.01, max_asr=.05, max_peak_asr=.20, tail_rounds=10")
    strict = spec.get("promotion", {}).get("require_mean_dual_best", False)
    if type(strict) is not bool:
        raise ValueError("promotion.require_mean_dual_best must be boolean")
    objective = spec["objective"]
    if (set(objective) != {"clean_accuracy_weight", "robust_accuracy_weight", "attack_success_weight", "honest_weight_loss_weight"}
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in objective.values())
            or not math.isclose(sum(objective.values()), 1.)):
        raise ValueError("four finite nonnegative common Score weights summing to one are required")
    if not _unit_interval(spec["gates"]["max_clean_accuracy_drop"]):
        raise ValueError("invalid clean utility threshold")
    expected = base.build_tasks(spec, "validation")
    wanted = {task["task_id"]: task for task in expected}
    if len(wanted) != len(expected) or len(tasks) != len(expected):
        raise ValueError("gate requires the entire declared validation task plan")
    actual = set()
    for task in tasks:
        identity = {k: v for k, v in task.items() if k != "fingerprint"}
        task_id = task.get("task_id")
        if task_id in actual or wanted.get(task_id) != identity:
            raise ValueError("gate input differs from the declared validation-only task plan")
        actual.add(task_id)
    return expected, policy, strict


def _candidate_evidence(spec, results_by_candidate, expected):
    """Reuse the existing health/Score policy after validating raw evidence."""
    planned = {}
    for task in expected:
        planned.setdefault(task["candidate"]["candidate_id"], []).append(task)
    usable, identity_errors, measured = {}, {}, {}
    expected_count = len(spec["validation"]["seeds"]) * len(spec["validation"]["scenarios"])
    expected_attacked = sum(t["malicious_ratio"] > 0 for t in spec["validation"]["scenarios"]) * len(spec["validation"]["seeds"])
    for cid, candidate_tasks in planned.items():
        runs = results_by_candidate.get(cid, [])
        observed = Counter(base.digest(base.semantic_config(r.config)) for r in runs)
        wanted = Counter(base.digest(base.semantic_config(t["config"])) for t in candidate_tasks)
        if observed != wanted:
            identity_errors[cid] = "incomplete_or_mismatched_validation_results"
            usable[cid], measured[cid] = [], []
            if candidate_tasks[0]["method"] == "fedavg":
                # Independent clean controls survive a missing attacked result.
                usable[cid] = [run for run in runs if run.config.malicious_ratio == 0
                               and observed[base.digest(base.semantic_config(run.config))] == 1
                               and wanted[base.digest(base.semantic_config(run.config))] == 1
                               and _run_audit(run)["healthy"]]
            continue
        measured[cid] = _task_metric_rows(runs, candidate_tasks)
        usable[cid] = [run for run in runs if _run_audit(run)["scorable"]]
    report = base.select_validation(spec, usable, expected)
    trials = {row["candidate_id"]: row for row in report["trials"]}
    candidate_rows = {}
    for cid in planned:
        trial, rows = trials[cid], measured[cid]
        attacked = [row for row in rows if row["malicious_ratio"] > 0]
        if cid in identity_errors or len(usable[cid]) != expected_count:
            reasons = ({identity_errors[cid]} if cid in identity_errors else
                       {reason for row in rows for reason in row["health_reasons"]})
            trial["invalid_reasons"] = ",".join(sorted(reasons))
            trial["result_count"] = len(results_by_candidate.get(cid, []))
            report["methods"][trial["method"]]["candidate_failures"][cid] = trial["invalid_reasons"]
        reasons = set(filter(None, trial["invalid_reasons"].split(",")))
        scorable = bool(len(rows) == expected_count and len(attacked) == expected_attacked
                        and all(row["scorable"] for row in rows)
                        and not (reasons - SCORABLE_HEALTH_FAILURES)
                        and all(_unit_interval(trial.get(key)) for key in SCORE_FIELDS)
                        and _unit_interval(trial.get("worst_attack_success_rate")))
        raw_score = weighted_score(trial, spec["objective"]) if scorable else None
        if raw_score is not None and not math.isfinite(raw_score):
            raise ValueError("scorable candidate has nonfinite common Score")
        eligible = bool(trial["valid"] and scorable)
        trial.update(raw_score=raw_score, scorable=scorable, health_qualified=eligible)
        candidate_rows[cid] = {
            "candidate_id": cid, "method": trial["method"], "eligible": eligible,
            "health_qualified": eligible, "scorable": scorable,
            "invalid_reasons": trial["invalid_reasons"], "common_score": trial.get("score"),
            "raw_score": raw_score, "observed_result_count": len(results_by_candidate.get(cid, [])),
            "matched_validation_tasks": len(rows), "attacked_tasks": len(attacked),
            "mean_accuracy": fmean(row["final_accuracy"] for row in rows) if scorable else None,
            "mean_asr": fmean(row["attack_mean_asr"] for row in attacked) if scorable else None,
            "tasks": rows}
    return report, trials, candidate_rows


def _paired_target(ours, vert, policy, *, reference_health_qualified):
    """Audit absolute and relative requirements separately on complete runs.

    Health is assessed elsewhere. A complete nonfinite-update baseline may
    still supply valid observed metrics; it is never relabeled as healthy.
    """
    reference = {scenario_key(run): run for run in vert} if vert is not None else None
    rows = []
    absolute_failures, relative_failures = [], []
    for run in sorted(ours, key=scenario_key):
        key = scenario_key(run)
        peer = reference[key] if reference is not None else None
        attacked = run.config.malicious_ratio > 0
        records = run.records[run.config.attack_start_round:] if attacked else run.records[-1:]
        peer_records = (peer.records[peer.config.attack_start_round:] if attacked else peer.records[-1:]) if peer else None
        windows = {"clean_final": (records, peer_records)} if not attacked else {
            "attack_mean": (records, peer_records),
            "tail_mean": (records[-policy.tail_rounds:], peer_records[-policy.tail_rounds:] if peer else None),
            "final": (records[-1:], peer_records[-1:] if peer else None)}
        row = {"scenario": list(key), "windows": {}, "absolute_failures": [], "relative_failures": []}
        if attacked:
            peak = max(record.attack_target_success_rate for record in records)
            row["ours_peak_asr"] = peak
            if peak > policy.max_peak_asr + TOLERANCE:
                row["absolute_failures"].append("peak_asr")
        for name, (ours_records, vert_records) in windows.items():
            values = {"ours_accuracy": fmean(record.accuracy for record in ours_records)}
            if vert_records:
                values["vert_accuracy"] = fmean(record.accuracy for record in vert_records)
                values["accuracy_gap_to_vert"] = values["vert_accuracy"] - values["ours_accuracy"]
                if values["accuracy_gap_to_vert"] > policy.accuracy_gap + TOLERANCE:
                    row["relative_failures"].append(name + ":accuracy_gap")
            if attacked:
                values["ours_asr"] = fmean(record.attack_target_success_rate for record in ours_records)
                if values["ours_asr"] > policy.max_asr + TOLERANCE:
                    row["absolute_failures"].append(name + ":absolute_asr")
                if vert_records:
                    values["vert_asr"] = fmean(record.attack_target_success_rate for record in vert_records)
                    values["asr_gap_to_vert"] = values["ours_asr"] - values["vert_asr"]
                    if values["asr_gap_to_vert"] > policy.asr_gap + TOLERANCE:
                        row["relative_failures"].append(name + ":asr_gap")
            row["windows"][name] = values
        row["failures"] = row["absolute_failures"] + row["relative_failures"]
        absolute_failures.extend({"scenario": list(key), "reason": reason} for reason in row["absolute_failures"])
        relative_failures.extend({"scenario": list(key), "reason": reason} for reason in row["relative_failures"])
        rows.append(row)
    absolute = not absolute_failures
    relative_status = "unassessed" if reference is None else "unmet" if relative_failures else "passed"
    full_pass = absolute and relative_status == "passed"
    target_qualified = absolute and relative_status != "unmet"
    return {"status": "passed" if full_pass else "partially_assessed" if target_qualified else "unmet",
            "role": "selection_and_promotion_gate", "policy": asdict(policy),
            "absolute_target_passed": absolute, "absolute_status": "passed" if absolute else "unmet",
            "relative_target_status": relative_status, "full_target_passed": full_pass,
            "target_qualified_for_promotion": target_qualified,
            "reference_health_qualified": reference_health_qualified,
            "reference_scorable": reference is not None,
            "missing_reference_is_not_a_pass": True,
            "absolute_failures": absolute_failures, "relative_failures": relative_failures,
            "scenarios": rows, "structurally_complete": reference is not None}


def select_validation(spec, results_by_candidate, tasks):
    """Select new validation candidates and gate entry to all six formal arms."""
    expected, policy, strict = _check_validation_plan(spec, tasks)
    report, trials, candidate_rows = _candidate_evidence(spec, results_by_candidate, expected)
    original_ours = report["selected"].get("sm9rrs")

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
                    selection_raw_score=row["raw_score"], comparison_available=row["eligible"],
                    scored_comparison_available=row["scorable"], selected_without_valid_validation=not row["eligible"],
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
    targets = {}
    for cid, row in candidate_rows.items():
        if row["method"] != "sm9rrs":
            continue
        if not row["eligible"]:
            targets[cid] = {"status": "ineligible", "health_qualified": False,
                            "invalid_reasons": row["invalid_reasons"], "full_target_passed": False,
                            "target_qualified_for_promotion": False, "promotion_qualified": False}
            continue
        audit = _paired_target(results_by_candidate[cid], results_by_candidate[vert["candidate_id"]] if vert else None,
                               policy, reference_health_qualified=vert["health_qualified"] if vert else None)
        joint = (best_accuracy - row["mean_accuracy"] <= TOLERANCE
                 and row["mean_asr"] - best_asr <= TOLERANCE)
        audit.update(health_qualified=True, mean_accuracy=row["mean_accuracy"], mean_asr=row["mean_asr"],
                     reference_candidate=vert["candidate_id"] if vert else None,
                     mean_dual_best_passed=joint, accuracy_gap_to_best=best_accuracy - row["mean_accuracy"],
                     asr_gap_to_best=row["mean_asr"] - best_asr,
                     promotion_qualified=audit["target_qualified_for_promotion"] and (joint or not strict))
        targets[cid] = audit
    qualified = [cid for cid in ours if targets[cid]["promotion_qualified"]]
    selected = max(qualified, key=lambda cid: (targets[cid]["mean_dual_best_passed"], *rank(cid))) if qualified else None
    route = ("mnist_target_with_paired_vert" if targets[selected]["full_target_passed"] else
             "mnist_absolute_target_without_scorable_vert") if selected else None
    unqualified_references = [method for method, info in available.items() if not info["health_qualified"]]
    scope = ("all_six_methods_healthy" if len(available) == 5 and not unqualified_references else
             "ours_and_scorable_selected_baselines_including_health_failures" if unqualified_references else
             "ours_and_available_healthy_selected_baselines" if available else "healthy_ours_candidates_only")
    selected_target = targets.get(selected, {"status": "unmet" if ours else "ineligible",
                                            "role": "selection_and_promotion_gate",
                                            "full_target_passed": False, "promotion_qualified": False})
    audit = {"status": "passed" if selected else "unmet" if ours else "no_healthy_ours_candidate",
             "role": "selection_and_promotion_gate", "validation_only": True, "policy": asdict(policy),
             "require_mean_dual_best": strict, "pass_route": route, "selected_pass_route": route,
             "selected_candidate": selected, "selected_target": selected_target,
             "absolute_target_passed": selected_target.get("absolute_target_passed", False),
             "relative_target_status": selected_target.get("relative_target_status", "unassessed"),
             "full_target_passed": selected_target.get("full_target_passed", False),
             "reference_candidate": vert["candidate_id"] if vert else None,
             "reference_health_qualified": vert["health_qualified"] if vert else None,
             "eligible_ours_candidates": ours, "qualified_ours_candidates": qualified,
             "candidate_rows": candidate_rows, "ours_candidate_targets": targets,
             "comparison_scope": scope, "available_baselines": available, "unavailable_baselines": unavailable,
             "comparator_candidates": comparators, "best_mean_accuracy": best_accuracy, "best_mean_asr": best_asr,
             "comparison_includes_unqualified_references": bool(unqualified_references),
             "unqualified_reference_methods": unqualified_references,
             "six_method_observed_means_compared": len(available) == 5 and bool(ours),
             "six_method_dual_optimality_assessed": len(available) == 5 and not unqualified_references and bool(ours),
             "definition": {"target_scope": "each validation seed and scenario separately",
                            "accuracy": "paired VERT: clean final, attack mean, tail mean, and final accuracy",
                            "asr": "absolute and paired VERT: attack mean, tail mean, and final ASR; absolute peak ASR",
                            "accuracy_task_count": 30, "asr_task_count": 24, "asr_round_count_per_task": 76,
                            "observed_mean_accuracy": "equal mean of 30 final-round accuracies",
                            "observed_mean_asr": "equal mean of 24 per-run attack-window means",
                            "mean_dual_role": "additional_promotion_requirement" if strict else "selection_preference_only",
                            "numeric_tolerance": TOLERANCE,
                            "tie_breaker": "target-qualified candidates; observed mean dual best first; common Score, lower worst ASR, candidate id"},
             "source": "declared new validation task results only; previous-study and formal results are not selection inputs"}
    report.update(mnist_target_gate=audit, ours_target=selected_target, ours_candidate_targets=targets,
                  best_healthy_score_candidate=original_ours, ours_health_passed=bool(ours),
                  ours_mnist_target_passed=bool(selected and selected_target["full_target_passed"]),
                  ours_absolute_target_passed=bool(selected and selected_target["absolute_target_passed"]),
                  ours_relative_performance_passed=bool(selected and selected_target["relative_target_status"] == "passed"),
                  ours_mean_dual_passed=bool(selected and selected_target["mean_dual_best_passed"]),
                  ours_joint_mean_dual_passed=bool(selected and selected_target["mean_dual_best_passed"]),
                  pass_route=route, status="qualified_for_final" if selected else
                  "needs_ours_target_development" if ours else "needs_ours_development",
                  selection_rule="ours_healthy_mnist_target_then_mean_dual_preference_then_common_score; baselines_healthy_score_then_complete_score_then_fixed")
    report["promotion_basis"].update(performance_target_required=True,
                                     absolute_asr_target_required=True,
                                     paired_vert_target_required_when_scorable=True,
                                     relative_performance_required=bool(vert),
                                     require_mean_dual_best=strict,
                                     unavailable_baselines_are_unassessed_not_passed=True,
                                     baseline_eligibility_required=False)
    info = report["methods"]["sm9rrs"]
    info.update(selected_candidate=selected, selection_status="healthy_mnist_target_selection" if selected else
                "no_target_qualified_ours_candidate" if ours else "no_eligible_ours_candidate",
                health_qualified=bool(selected), scorable=bool(selected),
                raw_score=candidate_rows[selected]["raw_score"] if selected else None,
                selection_raw_score=candidate_rows[selected]["raw_score"] if selected else None,
                pass_route=route, selected_without_valid_validation=False)
    if selected:
        report["selected"]["sm9rrs"] = selected
    else:
        report["selected"].pop("sm9rrs", None)
    return report
