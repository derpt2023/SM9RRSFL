"""Version 5: final-round Accuracy/ASR selection; unchanged full-run health.

Historical v4 modules remain byte-identical for immutable validation evidence.
Time averages, tails and peaks are diagnostic only, including Score tie breaks.
"""
from dataclasses import asdict
from statistics import fmean

import run_cifar_six_from_scratch as base
import cifar_mnist_target_gate as legacy
from cifar_mean_dual_gate import TOLERANCE
from sm9rrsfl.calibration_policy import weighted_score
from sm9rrsfl.performance_target import scenario_key

SELECTION_METRICS = {
    "accuracy": "final_round", "asr": "final_round", "round": 100,
    "performance_gate": "final_round_only", "process_metrics": "diagnostic_only",
    "replicate_summary": "equal_mean_of_final_values",
    "honest_weight_loss": "unchanged_time_and_scenario_mean",
}


def _check_validation_plan(spec, tasks):
    if spec.get("schema_version") != 5 or spec.get("selection_metrics") != SELECTION_METRICS:
        raise ValueError("schema 5 requires explicit final-round selection and diagnostic-only process metrics")
    return legacy._check_validation_plan(spec, tasks)


def _candidate_evidence(spec, results_by_candidate, expected):
    # Reuse only evidence integrity / health checks; replace every Accuracy/ASR
    # score term and ASR tie-break before selecting any method.
    report, trials, candidates = legacy._candidate_evidence(spec, results_by_candidate, expected)
    for cid, row in candidates.items():
        trial = trials[cid]
        if not row["scorable"]:
            continue
        runs = results_by_candidate[cid]
        attacked = [r for r in runs if r.config.malicious_ratio > 0]
        trial["process_diagnostics"] = {k: trial[k] for k in
            ("robust_accuracy", "attack_success_rate", "worst_attack_success_rate", "raw_score")}
        trial["robust_accuracy"] = fmean(r.records[-1].accuracy for r in attacked)
        trial["attack_success_rate"] = fmean(r.records[-1].attack_target_success_rate for r in attacked)
        trial["worst_attack_success_rate"] = max(r.records[-1].attack_target_success_rate for r in attacked)
        trial["raw_score"] = weighted_score(trial, spec["objective"])
        trial["score"] = trial["raw_score"] if row["eligible"] else None
        trial["selection_metrics"] = dict(SELECTION_METRICS)
        row.update(raw_score=trial["raw_score"], common_score=trial["score"],
                   mean_asr=trial["attack_success_rate"], selection_metrics=dict(SELECTION_METRICS))
        by_key = {(r.config.partition, r.config.malicious_ratio, r.config.seed): r for r in runs}
        for task in row["tasks"]:
            run = by_key[task["partition"], task["malicious_ratio"], task["seed"]]
            task["final_asr"] = run.records[-1].attack_target_success_rate
            task["asr_selection_round"] = run.config.rounds
            task["attack_mean_asr_role"] = "diagnostic_only"
    healthy = [cid for cid, row in candidates.items() if row["method"] == "sm9rrs" and row["eligible"]]
    if healthy:
        report["selected"]["sm9rrs"] = max(healthy, key=lambda cid:
            (candidates[cid]["raw_score"], -trials[cid]["worst_attack_success_rate"], cid))
    return report, trials, candidates


def _paired_target(ours, vert, policy, *, reference_health_qualified):
    reference = {scenario_key(r): r for r in vert} if vert is not None else None
    rows, absolute_failures, relative_failures = [], [], []
    for run in sorted(ours, key=scenario_key):
        key = scenario_key(run)
        peer = reference[key] if reference is not None else None
        final = run.records[-1]
        attacked = run.config.malicious_ratio > 0
        name = "final" if attacked else "clean_final"
        values = {"ours_accuracy": final.accuracy}
        row = {"scenario": list(key), "selection_round": run.config.rounds,
               "windows": {name: values}, "absolute_failures": [], "relative_failures": []}
        if peer is not None:
            values.update(vert_accuracy=peer.records[-1].accuracy,
                          accuracy_gap_to_vert=peer.records[-1].accuracy - final.accuracy)
            if values["accuracy_gap_to_vert"] > policy.accuracy_gap + TOLERANCE:
                row["relative_failures"].append(name + ":accuracy_gap")
        if attacked:
            values["ours_asr"] = final.attack_target_success_rate
            if values["ours_asr"] > policy.max_asr + TOLERANCE:
                row["absolute_failures"].append("final:absolute_asr")
            if peer is not None:
                values.update(vert_asr=peer.records[-1].attack_target_success_rate,
                              asr_gap_to_vert=values["ours_asr"] - peer.records[-1].attack_target_success_rate)
                if values["asr_gap_to_vert"] > policy.asr_gap + TOLERANCE:
                    row["relative_failures"].append("final:asr_gap")
            history = run.records[run.config.attack_start_round:]
            row["process_diagnostics"] = {
                "role": "diagnostic_only",
                "attack_mean_accuracy": fmean(r.accuracy for r in history),
                "attack_mean_asr": fmean(r.attack_target_success_rate for r in history),
                "tail_mean_asr": fmean(r.attack_target_success_rate for r in history[-policy.tail_rounds:]),
                "peak_asr": max(r.attack_target_success_rate for r in history),
            }
        row["failures"] = row["absolute_failures"] + row["relative_failures"]
        for failures, key_name in ((absolute_failures, "absolute_failures"), (relative_failures, "relative_failures")):
            failures.extend({"scenario": list(key), "reason": reason} for reason in row[key_name])
        rows.append(row)
    absolute = not absolute_failures
    relative = "unassessed" if reference is None else "unmet" if relative_failures else "passed"
    full = absolute and relative == "passed"
    qualified = absolute and relative != "unmet"
    return {"status": "passed" if full else "partially_assessed" if qualified else "unmet",
            "role": "selection_and_promotion_gate", "policy": asdict(policy),
            "selection_metrics": dict(SELECTION_METRICS),
            "absolute_target_passed": absolute, "absolute_status": "passed" if absolute else "unmet",
            "relative_target_status": relative, "full_target_passed": full,
            "target_qualified_for_promotion": qualified,
            "reference_health_qualified": reference_health_qualified,
            "reference_scorable": reference is not None, "missing_reference_is_not_a_pass": True,
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
    route = ("final_target_with_paired_vert" if targets[selected]["full_target_passed"] else
             "final_absolute_target_without_scorable_vert") if selected else None
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
                            "accuracy": "paired VERT: final round only in every scenario",
                            "asr": "absolute and paired VERT: final round only; process metrics diagnostic only",
                            "accuracy_task_count": 30, "asr_task_count": 24, "asr_round_count_per_task": 1,
                            "observed_mean_accuracy": "equal mean of 30 final-round accuracies",
                            "observed_mean_asr": "equal mean of 24 final-round ASRs",
                            "mean_dual_role": "additional_promotion_requirement" if strict else "selection_preference_only",
                            "numeric_tolerance": TOLERANCE,
                            "tie_breaker": "target-qualified candidates; observed mean dual best first; common Score, lower worst ASR, candidate id"},
             "source": "declared new validation task results only; previous-study and formal results are not selection inputs"}
    report.update(final_metric_gate=audit, ours_target=selected_target, ours_candidate_targets=targets,
                  best_healthy_score_candidate=original_ours, ours_health_passed=bool(ours),
                  ours_final_target_passed=bool(selected and selected_target["full_target_passed"]),
                  ours_absolute_target_passed=bool(selected and selected_target["absolute_target_passed"]),
                  ours_relative_performance_passed=bool(selected and selected_target["relative_target_status"] == "passed"),
                  ours_mean_dual_passed=bool(selected and selected_target["mean_dual_best_passed"]),
                  ours_joint_mean_dual_passed=bool(selected and selected_target["mean_dual_best_passed"]),
                  pass_route=route, status="qualified_for_final" if selected else
                  "needs_ours_target_development" if ours else "needs_ours_development",
                  selection_rule="ours_healthy_final_target_then_final_dual_preference_then_final_score; baselines_healthy_score_then_complete_score_then_fixed")
    report["promotion_basis"].update(performance_target_required=True,
                                     absolute_asr_target_required=True,
                                     paired_vert_target_required_when_scorable=True,
                                     relative_performance_required=bool(vert),
                                     require_mean_dual_best=strict,
                                     unavailable_baselines_are_unassessed_not_passed=True,
                                     baseline_eligibility_required=False)
    info = report["methods"]["sm9rrs"]
    info.update(selected_candidate=selected, selection_status="healthy_final_target_selection" if selected else
                "no_target_qualified_ours_candidate" if ours else "no_eligible_ours_candidate",
                health_qualified=bool(selected), scorable=bool(selected),
                raw_score=candidate_rows[selected]["raw_score"] if selected else None,
                selection_raw_score=candidate_rows[selected]["raw_score"] if selected else None,
                pass_route=route, selected_without_valid_validation=False)
    if selected:
        report["selected"]["sm9rrs"] = selected
    else:
        report["selected"].pop("sm9rrs", None)
    report["selection_metrics"] = dict(SELECTION_METRICS)
    audit["selection_metrics"] = dict(SELECTION_METRICS)
    return report


def describe_final_targets(spec, selected, results, tasks):
    """Final-test comparison only; never changes selection or execution health."""
    expected = {(c["partition"], c["dirichlet_alpha"], c["num_clients"], c["malicious_ratio"], c["seed"])
                for c in (t["config"] for t in tasks)}
    peers = {method: [r for r in results.get(cid, []) if base.metrics(r)["healthy"]]
             for method, cid in selected.items()}
    ours = peers.get("sm9rrs", [])
    ours_keys = {scenario_key(r) for r in ours}
    comparisons = {}
    for method in base.ALL_METHODS[1:]:
        reference = peers.get(method, [])
        keys = {scenario_key(r) for r in reference}
        if ours_keys != expected or keys != expected or not expected:
            comparisons[method] = {"status": "incomplete", "reference_method": method,
                "missing_ours": [list(k) for k in sorted(expected - ours_keys)],
                "missing_reference": [list(k) for k in sorted(expected - keys)],
                "full_target_passed": False}
        else:
            comparison = _paired_target(ours, reference,
                legacy.PerformanceTarget.parse(spec["performance_target"]), reference_health_qualified=True)
            # _paired_target uses VERT-labelled fields for validation. Relabel
            # them for the descriptive all-baseline comparison.
            for row in comparison["scenarios"]:
                for values in row["windows"].values():
                    for key in list(values):
                        if "vert" in key:
                            values[key.replace("vert", "reference")] = values.pop(key)
            comparison.update(reference_method=method, role="descriptive_only_not_execution_or_health_status")
            comparisons[method] = comparison
    statuses = [c["status"] for c in comparisons.values()]
    return {"status": "incomplete" if "incomplete" in statuses else
            "passed" if all(s == "passed" for s in statuses) else "unmet",
            "role": "descriptive_only_not_execution_or_health_status",
            "selection_metrics": dict(SELECTION_METRICS), "comparisons": comparisons,
            "unhealthy_runs_retained_in_observed_tables": True}
