"""v7: final Accuracy and ASR each within two percentage points of the best.

The best Accuracy and lowest ASR can belong to different fixed baselines.
Health, final-round Score, baseline fallback and mean-dual preference stay fixed.
"""
from copy import deepcopy
from fractions import Fraction
from statistics import fmean

import cifar_relative_asr_gate as previous
import run_cifar_six_from_scratch as base
from cifar_mean_dual_gate import TOLERANCE, _run_audit
from sm9rrsfl.performance_target import scenario_key

SELECTION_METRICS = dict(previous.SELECTION_METRICS)
ASR_TARGET = {**previous.ASR_TARGET, "max_gap": .02, "comparison": "less_than_or_equal"}
ACCURACY_TARGET = {**ASR_TARGET, "mode": "within_selected_six_method_maximum",
                   "scope": "each_task_final_round"}
PERFORMANCE_TARGET = {"accuracy_gap": .02, "asr_gap": .02, "tail_rounds": 10}


def validate_policy(spec):
    if (spec.get("schema_version") != 7 or spec.get("selection_metrics") != SELECTION_METRICS
            or spec.get("asr_target") != ASR_TARGET or spec.get("accuracy_target") != ACCURACY_TARGET
            or spec.get("performance_target") != PERFORMANCE_TARGET):
        raise ValueError("v7 requires final Accuracy/ASR gaps to selected-method best <= .02")


def accuracy_fraction(run, sample_count):
    value = run.records[-1].accuracy
    hits = round(value * sample_count)
    return (Fraction(hits, sample_count) if abs(value - hits / sample_count) <= 5e-8
            else Fraction(str(value)))


def evaluation_count(spec, phase):
    data = spec["dataset"]
    return (round(data["train_samples"] * data["validation_fraction"])
            if phase == "validation" else data["test_samples"])


def task_target(ours, references, missing, *, reference_candidates, accuracy_samples):
    rows, accuracy_failures, asr_failures = [], [], []
    for run in sorted(ours, key=scenario_key):
        key, final = scenario_key(run), run.records[-1]
        attacked = run.config.malicious_ratio > 0
        peers = {"sm9rrs": run, **{m: group[key] for m, group in references.items() if key in group}}
        absent = [m for m in base.ALL_METHODS if m not in peers]
        evidence = {m: {"candidate": reference_candidates[m], "task_healthy": _run_audit(peer)["healthy"],
                        "health_reasons": _run_audit(peer)["reasons"]} for m, peer in peers.items()}
        common = {"available_methods": list(peers), "missing_methods": absent,
                  "six_method_comparison_complete": not absent,
                  "missing_reasons": {m: missing.get(m, {}).get(key, ["missing_complete_task"]) for m in absent}}
        accuracies = {m: accuracy_fraction(peer, accuracy_samples) for m, peer in peers.items()}
        maximum = max(accuracies.values())
        accuracy_gap = maximum - accuracies["sm9rrs"]
        accuracy_pass = accuracy_gap <= Fraction(str(ACCURACY_TARGET["max_gap"]))
        values = {"ours_accuracy": final.accuracy}
        row = {"scenario": list(key), "selection_round": 100,
               "windows": {"final" if attacked else "clean_final": values},
               "accuracy_maximum": {**common, "passed": accuracy_pass, "maximum_accuracy": float(maximum),
                   "gap": float(accuracy_gap), "sample_count": accuracy_samples,
                   "best_methods": [m for m, value in accuracies.items() if value == maximum],
                   "references": {m: {**evidence[m], "observed_final_accuracy": peer.records[-1].accuracy,
                                      "comparison_accuracy": float(accuracies[m])} for m, peer in peers.items()}},
               "failures": []}
        if not accuracy_pass:
            reason = "final:accuracy_gap_to_maximum_exceeds_limit" if attacked else "clean_final:accuracy_gap_to_maximum_exceeds_limit"
            row["failures"].append(reason)
            accuracy_failures.append({"scenario": list(key), "reason": reason})
        if attacked:
            rates = {m: previous.asr_fraction(peer) for m, peer in peers.items()}
            minimum = min(rates.values())
            asr_gap = rates["sm9rrs"] - minimum
            asr_pass = asr_gap <= Fraction(str(ASR_TARGET["max_gap"]))
            values["ours_asr"] = final.attack_target_success_rate
            row["asr_minimum"] = {**common, "passed": asr_pass, "minimum_asr": float(minimum),
                "gap": float(asr_gap), "best_methods": [m for m, rate in rates.items() if rate == minimum],
                "references": {m: {**evidence[m], "observed_final_asr": peer.records[-1].attack_target_success_rate,
                                    "comparison_asr": float(rates[m])} for m, peer in peers.items()}}
            if not asr_pass:
                reason = "final:asr_gap_to_minimum_exceeds_limit"
                row["failures"].append(reason)
                asr_failures.append({"scenario": list(key), "reason": reason})
            history = run.records[run.config.attack_start_round:]
            row["process_diagnostics"] = {"role": "diagnostic_only",
                "attack_mean_accuracy": fmean(r.accuracy for r in history),
                "attack_mean_asr": fmean(r.attack_target_success_rate for r in history),
                "tail_mean_asr": fmean(r.attack_target_success_rate for r in history[-10:]),
                "peak_asr": max(r.attack_target_success_rate for r in history)}
        rows.append(row)
    attack_rows = [r for r in rows if "asr_minimum" in r]
    accuracy_pass = bool(rows) and not accuracy_failures
    asr_pass = bool(attack_rows) and not asr_failures
    complete = bool(rows) and all(r["accuracy_maximum"]["six_method_comparison_complete"] for r in rows)
    qualified = accuracy_pass and asr_pass
    return {"status": "passed" if qualified and complete else "partially_assessed" if qualified else "unmet",
        "role": "selection_and_promotion_gate", "policy": dict(PERFORMANCE_TARGET),
        "accuracy_target": dict(ACCURACY_TARGET), "asr_target": dict(ASR_TARGET),
        "selection_metrics": dict(SELECTION_METRICS), "accuracy_maximum_target_passed": accuracy_pass,
        "asr_minimum_target_passed": asr_pass, "target_qualified_for_promotion": qualified,
        "full_target_passed": qualified and complete, "six_method_comparison_complete": complete,
        "structurally_complete": complete, "missing_reference_is_not_a_pass": True,
        "relative_target_status": "unmet" if not qualified else "passed" if complete else "unassessed",
        "accuracy_maximum_failures": accuracy_failures, "asr_minimum_failures": asr_failures,
        "accuracy_passed_task_count": sum(r["accuracy_maximum"]["passed"] for r in rows),
        "accuracy_task_count": len(rows), "asr_passed_task_count": sum(r["asr_minimum"]["passed"] for r in attack_rows),
        "asr_task_count": len(attack_rows), "scenarios": rows}


def select_validation(spec, results_by_candidate, tasks):
    validate_policy(spec)
    # Reuse only health, final Score, fixed baselines and the mean-dual audit.
    # Replace the old performance verdict entirely, including failed old gates.
    compatibility = deepcopy(spec)
    compatibility.update(schema_version=6, asr_target=dict(previous.ASR_TARGET),
                         performance_target={"accuracy_gap": .02, "asr_gap": .01, "tail_rounds": 10})
    compatibility.pop("accuracy_target", None)
    report = previous.select_validation(compatibility, results_by_candidate, tasks)
    audit = report["final_metric_gate"]
    candidates = audit["candidate_rows"]
    trials = {r["candidate_id"]: r for r in report["trials"]}
    baseline_ids = {m: report["selected"][m] for m in base.ALL_METHODS[1:]}
    references, missing = previous.selected_task_evidence(baseline_ids, results_by_candidate, tasks)
    targets = {}
    for cid, candidate in candidates.items():
        if candidate["method"] != "sm9rrs":
            continue
        if not candidate["eligible"]:
            targets[cid] = {"status": "ineligible", "health_qualified": False,
                "invalid_reasons": candidate["invalid_reasons"], "full_target_passed": False,
                "target_qualified_for_promotion": False, "promotion_qualified": False}
            continue
        target = task_target(results_by_candidate[cid], references, missing,
            reference_candidates={"sm9rrs": cid, **baseline_ids}, accuracy_samples=evaluation_count(spec, "validation"))
        joint = (audit["best_mean_accuracy"] - candidate["mean_accuracy"] <= TOLERANCE
                 and candidate["mean_asr"] - audit["best_mean_asr"] <= TOLERANCE)
        target.update(health_qualified=True, mean_accuracy=candidate["mean_accuracy"], mean_asr=candidate["mean_asr"],
            mean_dual_best_passed=joint, accuracy_gap_to_best=audit["best_mean_accuracy"] - candidate["mean_accuracy"],
            asr_gap_to_best=candidate["mean_asr"] - audit["best_mean_asr"],
            promotion_qualified=target["target_qualified_for_promotion"] and
                (joint or not spec["promotion"]["require_mean_dual_best"]))
        targets[cid] = target
    qualified = [cid for cid, target in targets.items() if target["promotion_qualified"]]
    chosen = max(qualified, key=lambda cid: (targets[cid]["mean_dual_best_passed"], candidates[cid]["raw_score"],
                 -trials[cid]["worst_attack_success_rate"], cid)) if qualified else None
    target = targets.get(chosen, {"status": "unmet", "full_target_passed": False,
        "accuracy_maximum_target_passed": False, "asr_minimum_target_passed": False,
        "relative_target_status": "unassessed"})
    route = ("final_relative_best_full_comparison" if target["full_target_passed"] else
             "final_relative_best_partial_comparison") if chosen else None
    report["selected"] = {**baseline_ids, **({"sm9rrs": chosen} if chosen else {})}
    report.update(ours_target=target, ours_candidate_targets=targets,
        ours_final_target_passed=bool(chosen and target["full_target_passed"]),
        ours_accuracy_maximum_target_passed=bool(chosen and target["accuracy_maximum_target_passed"]),
        ours_asr_minimum_target_passed=bool(chosen and target["asr_minimum_target_passed"]),
        ours_relative_performance_passed=bool(chosen and target["relative_target_status"] == "passed"),
        ours_mean_dual_passed=bool(chosen and target["mean_dual_best_passed"]),
        ours_joint_mean_dual_passed=bool(chosen and target["mean_dual_best_passed"]),
        pass_route=route, status="qualified_for_final" if chosen else
        "needs_ours_target_development" if report["ours_health_passed"] else "needs_ours_development",
        selection_rule="healthy_final_accuracy_and_asr_near_selected_best_then_mean_dual_preference_then_score",
        accuracy_target=dict(ACCURACY_TARGET), asr_target=dict(ASR_TARGET))
    report["promotion_basis"].update(absolute_asr_target_required=False, paired_vert_target_required_when_scorable=False,
        selected_minimum_asr_target_required=True, selected_maximum_accuracy_target_required=True,
        relative_performance_required=True)
    report["methods"]["sm9rrs"].update(selected_candidate=chosen,
        selection_status="healthy_final_relative_best_selection" if chosen else
        "no_target_qualified_ours_candidate" if report["ours_health_passed"] else "no_eligible_ours_candidate",
        health_qualified=bool(chosen), scorable=bool(chosen),
        raw_score=candidates[chosen]["raw_score"] if chosen else None,
        selection_raw_score=candidates[chosen]["raw_score"] if chosen else None, pass_route=route)
    for key in ("reference_candidate", "reference_health_qualified", "absolute_target_status", "asr_comparison_scope"):
        audit.pop(key, None)
    audit.update(status="passed" if chosen else "unmet" if report["ours_health_passed"] else "no_healthy_ours_candidate",
        policy=dict(PERFORMANCE_TARGET), accuracy_target=dict(ACCURACY_TARGET), asr_target=dict(ASR_TARGET),
        selected_candidate=chosen, selected_target=target, qualified_ours_candidates=qualified,
        ours_candidate_targets=targets, pass_route=route, selected_pass_route=route,
        accuracy_maximum_target_passed=target["accuracy_maximum_target_passed"],
        asr_minimum_target_passed=target["asr_minimum_target_passed"],
        relative_target_status=target["relative_target_status"], full_target_passed=target["full_target_passed"],
        comparison_scope="ours_and_per_task_available_fixed_selected_baselines_including_scorable_health_failures",
        per_task_reference_counts={m: len(runs) for m, runs in references.items()})
    audit["definition"].pop("strict_boundary", None)
    audit["definition"].update(accuracy="per task: available selected-method maximum final Accuracy minus Ours <= .02",
        asr="per attacked task: Ours final ASR minus available selected-method minimum <= .02",
        boundary="sample-count recovery and exact rational difference; equality passes",
        numeric_tolerance="none in the two performance gaps; mean-dual preference retains legacy tolerance")
    return report


def describe_final_targets(spec, selected, results, tasks):
    references, missing = previous.selected_task_evidence({m: cid for m, cid in selected.items() if m != "sm9rrs"}, results, tasks)
    ours = [r for r in results.get(selected.get("sm9rrs"), []) if _run_audit(r)["scorable"]]
    target = task_target(ours, references, missing, reference_candidates=selected, accuracy_samples=evaluation_count(spec, "final"))
    expected = sum(t["method"] == "sm9rrs" for t in tasks)
    if len(ours) != expected or not expected:
        target.update(status="incomplete", full_target_passed=False, target_qualified_for_promotion=False)
    target.update(role="descriptive_only_not_execution_or_health_status",
                  unhealthy_runs_retained_in_observed_tables=True, expected_ours_tasks=expected)
    return target
