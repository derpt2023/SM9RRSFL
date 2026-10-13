"""v6: final ASR within strictly one percentage point of selected methods.

Keep v5's final Score, independent health, paired VERT requirements and fixed
baseline selection. Missing baseline tasks are explicit, never imputed; their
other complete, numerically scorable runs remain usable per-task references.
"""
from copy import deepcopy
from fractions import Fraction
from statistics import fmean

import cifar_final_metric_gate as previous
import run_cifar_six_from_scratch as base
from cifar_mean_dual_gate import TOLERANCE, _run_audit
from sm9rrsfl.performance_target import scenario_key

SELECTION_METRICS = dict(previous.SELECTION_METRICS)
ASR_TARGET = {
    "mode": "within_selected_six_method_minimum", "max_gap": .01,
    "comparison": "strictly_less_than", "scope": "each_attacked_task_final_round",
    "reference_candidates": "fixed_by_validation_score_or_declared_fallback",
    "include_ours": True, "include_scorable_health_failures": True,
    "missing_baselines": "compare_available_and_mark_incomplete", "absolute_cap": None,
    "boundary_arithmetic": "recover_discrete_sample_rates_then_exact_rational_comparison",
}


def validate_policy(spec):
    if (spec.get("schema_version") != 6 or spec.get("selection_metrics") != SELECTION_METRICS
            or spec.get("asr_target") != ASR_TARGET):
        raise ValueError("v6 requires declared final-round relative ASR policy")
    if spec["performance_target"] != {"accuracy_gap": .02, "asr_gap": .01, "tail_rounds": 10}:
        raise ValueError("v6 retains paired VERT limits and diagnostic tail length, without an absolute ASR cap")


def asr_fraction(run):
    """ASR uses exactly attack_target_count samples (fl rejects a short pool).

    Torch may serialize k/n as float32. Recover that count only within float32
    rounding error; otherwise preserve the supplied value without rounding.
    """
    value = run.records[-1].attack_target_success_rate
    count = run.config.attack_target_count
    hits = round(value * count)
    return Fraction(hits, count) if abs(value - hits / count) <= 5e-8 else Fraction(str(value))


def selected_task_evidence(selected, results, tasks):
    wanted = {m: {} for m in selected}
    for task in tasks:
        method = task["method"]
        if selected.get(method) == task["candidate"]["candidate_id"]:
            wanted[method][base.digest(base.semantic_config(task["config"]))] = task
    available, missing = {}, {}
    for method, cid in selected.items():
        seen, runs, failures = set(), {}, {}
        for run in results.get(cid, []):
            identity = base.digest(base.semantic_config(run.config))
            if identity not in wanted[method] or identity in seen:
                raise ValueError("duplicate or mismatched reference task: " + cid)
            seen.add(identity)
            audit = _run_audit(run)
            key = scenario_key(run)
            if audit["scorable"] and run.records[-1].attack_target_success_rate is not None:
                runs[key] = run
            else:
                failures[key] = audit["reasons"] or ["missing_final_asr"]
        available[method], missing[method] = runs, failures
    return available, missing


def task_target(ours, references, missing, policy, *, vert_scorable, reference_candidates):
    rows, asr_failures, relative_failures = [], [], []
    for run in sorted(ours, key=scenario_key):
        key, final = scenario_key(run), run.records[-1]
        attacked = run.config.malicious_ratio > 0
        values = {"ours_accuracy": final.accuracy}
        row = {"scenario": list(key), "selection_round": 100,
               "windows": {"final" if attacked else "clean_final": values},
               "asr_minimum_failures": [], "relative_failures": []}
        vert = references.get("vert", {}).get(key) if vert_scorable else None
        if vert is not None:
            values.update(vert_accuracy=vert.records[-1].accuracy,
                          accuracy_gap_to_vert=vert.records[-1].accuracy - final.accuracy)
            if values["accuracy_gap_to_vert"] > policy["accuracy_gap"] + TOLERANCE:
                row["relative_failures"].append("final:accuracy_gap" if attacked else "clean_final:accuracy_gap")
        if attacked:
            peers = {"sm9rrs": run, **{m: group[key] for m, group in references.items() if key in group}}
            rates = {method: asr_fraction(peer) for method, peer in peers.items()}
            minimum = min(rates.values())
            gap = rates["sm9rrs"] - minimum
            passed = gap < Fraction(str(ASR_TARGET["max_gap"]))
            absent = [m for m in base.ALL_METHODS if m not in peers]
            values["ours_asr"] = final.attack_target_success_rate
            row["asr_minimum"] = {
                "passed": passed, "minimum_asr": float(minimum), "gap": float(gap),
                "best_methods": [m for m, rate in rates.items() if rate == minimum],
                "available_methods": list(peers), "missing_methods": absent,
                "six_method_comparison_complete": not absent,
                "missing_reasons": {m: missing.get(m, {}).get(key, ["missing_complete_task"]) for m in absent},
                "references": {m: {"candidate": reference_candidates[m],
                    "observed_final_asr": peer.records[-1].attack_target_success_rate,
                    "comparison_asr": float(rates[m]), "task_healthy": _run_audit(peer)["healthy"],
                    "health_reasons": _run_audit(peer)["reasons"]} for m, peer in peers.items()},
            }
            if not passed:
                row["asr_minimum_failures"].append("final:asr_gap_to_minimum_not_strictly_below_limit")
            if vert is not None:
                vert_gap = asr_fraction(run) - asr_fraction(vert)
                values.update(vert_asr=vert.records[-1].attack_target_success_rate,
                              asr_gap_to_vert=float(vert_gap))
                if vert_gap > Fraction(str(policy["asr_gap"])):
                    row["relative_failures"].append("final:asr_gap")
            history = run.records[run.config.attack_start_round:]
            row["process_diagnostics"] = {"role": "diagnostic_only",
                "attack_mean_accuracy": fmean(r.accuracy for r in history),
                "attack_mean_asr": fmean(r.attack_target_success_rate for r in history),
                "tail_mean_asr": fmean(r.attack_target_success_rate for r in history[-policy["tail_rounds"]:]),
                "peak_asr": max(r.attack_target_success_rate for r in history)}
        row["failures"] = row["asr_minimum_failures"] + row["relative_failures"]
        for failures, name in ((asr_failures, "asr_minimum_failures"), (relative_failures, "relative_failures")):
            failures.extend({"scenario": list(key), "reason": reason} for reason in row[name])
        rows.append(row)
    attack_rows = [r for r in rows if "asr_minimum" in r]
    asr_pass = bool(attack_rows) and not asr_failures
    reference_complete = bool(attack_rows) and all(r["asr_minimum"]["six_method_comparison_complete"] for r in attack_rows)
    vert_complete = bool(rows) and all("vert_accuracy" in next(iter(r["windows"].values())) for r in rows)
    relative = "unmet" if relative_failures else "passed" if vert_complete else "unassessed"
    qualified = asr_pass and not relative_failures
    full = qualified and reference_complete and vert_complete
    return {"status": "passed" if full else "partially_assessed" if qualified else "unmet",
            "role": "selection_and_promotion_gate", "policy": dict(policy), "asr_target": dict(ASR_TARGET),
            "selection_metrics": dict(SELECTION_METRICS), "asr_minimum_target_passed": asr_pass,
            "absolute_target_status": "not_required", "relative_target_status": relative,
            "full_target_passed": full, "target_qualified_for_promotion": qualified,
            "six_method_comparison_complete": reference_complete, "missing_reference_is_not_a_pass": True,
            "asr_minimum_failures": asr_failures, "relative_failures": relative_failures,
            "scenarios": rows, "structurally_complete": reference_complete and vert_complete,
            "asr_passed_task_count": sum(r["asr_minimum"]["passed"] for r in attack_rows),
            "asr_task_count": len(attack_rows)}


def select_validation(spec, results_by_candidate, tasks):
    validate_policy(spec)
    # This immutable helper supplies final Score, health and fixed baselines.
    # Its former performance verdict is replaced in full below, not inherited.
    compatibility = deepcopy(spec)
    compatibility["schema_version"] = 5
    report = previous.select_validation(compatibility, results_by_candidate, tasks)
    audit = report["final_metric_gate"]
    candidates = audit["candidate_rows"]
    trials = {r["candidate_id"]: r for r in report["trials"]}
    baseline_ids = {m: report["selected"][m] for m in base.ALL_METHODS[1:]}
    references, missing = selected_task_evidence(baseline_ids, results_by_candidate, tasks)
    targets = {}
    for cid, candidate in candidates.items():
        if candidate["method"] != "sm9rrs":
            continue
        if not candidate["eligible"]:
            targets[cid] = {"status": "ineligible", "health_qualified": False,
                "invalid_reasons": candidate["invalid_reasons"], "full_target_passed": False,
                "target_qualified_for_promotion": False, "promotion_qualified": False}
            continue
        target = task_target(results_by_candidate[cid], references, missing, spec["performance_target"],
            vert_scorable="vert" in audit["available_baselines"],
            reference_candidates={"sm9rrs": cid, **baseline_ids})
        joint = (audit["best_mean_accuracy"] - candidate["mean_accuracy"] <= TOLERANCE and
                 candidate["mean_asr"] - audit["best_mean_asr"] <= TOLERANCE)
        target.update(health_qualified=True, mean_accuracy=candidate["mean_accuracy"], mean_asr=candidate["mean_asr"],
            reference_candidate=baseline_ids["vert"],
            reference_health_qualified=report["methods"]["vert"]["health_qualified"],
            reference_scorable="vert" in audit["available_baselines"],
            mean_dual_best_passed=joint, accuracy_gap_to_best=audit["best_mean_accuracy"]-candidate["mean_accuracy"],
            asr_gap_to_best=candidate["mean_asr"]-audit["best_mean_asr"],
            promotion_qualified=target["target_qualified_for_promotion"] and
                (joint or not spec["promotion"]["require_mean_dual_best"]))
        targets[cid] = target
    qualified = [cid for cid, t in targets.items() if t["promotion_qualified"]]
    chosen = max(qualified, key=lambda cid: (targets[cid]["mean_dual_best_passed"], candidates[cid]["raw_score"],
        -trials[cid]["worst_attack_success_rate"], cid)) if qualified else None
    target = targets.get(chosen, {"status": "unmet", "full_target_passed": False,
        "asr_minimum_target_passed": False, "relative_target_status": "unassessed"})
    route = ("final_relative_asr_full_comparison" if target["full_target_passed"] else
             "final_relative_asr_partial_comparison") if chosen else None
    report["selected"] = {**baseline_ids, **({"sm9rrs": chosen} if chosen else {})}
    report.pop("ours_absolute_target_passed", None)
    report.update(ours_target=target, ours_candidate_targets=targets,
        ours_final_target_passed=bool(chosen and target["full_target_passed"]),
        ours_asr_minimum_target_passed=bool(chosen and target["asr_minimum_target_passed"]),
        ours_relative_performance_passed=bool(chosen and target["relative_target_status"] == "passed"),
        ours_mean_dual_passed=bool(chosen and target["mean_dual_best_passed"]),
        ours_joint_mean_dual_passed=bool(chosen and target["mean_dual_best_passed"]),
        pass_route=route, status="qualified_for_final" if chosen else
        "needs_ours_target_development" if report["ours_health_passed"] else "needs_ours_development",
        selection_rule="ours_healthy_final_asr_near_selected_minimum_then_final_dual_preference_then_score; unchanged_baseline_selection",
        asr_target=dict(ASR_TARGET))
    report["promotion_basis"].update(absolute_asr_target_required=False, selected_minimum_asr_target_required=True)
    report["methods"]["sm9rrs"].update(selected_candidate=chosen,
        selection_status="healthy_final_relative_asr_selection" if chosen else
        "no_target_qualified_ours_candidate" if report["ours_health_passed"] else "no_eligible_ours_candidate",
        health_qualified=bool(chosen), scorable=bool(chosen),
        raw_score=candidates[chosen]["raw_score"] if chosen else None,
        selection_raw_score=candidates[chosen]["raw_score"] if chosen else None, pass_route=route)
    audit.pop("absolute_target_passed", None)
    audit.update(status="passed" if chosen else "unmet" if report["ours_health_passed"] else "no_healthy_ours_candidate",
        policy=dict(spec["performance_target"]), asr_target=dict(ASR_TARGET),
        selected_candidate=chosen, selected_target=target, qualified_ours_candidates=qualified,
        ours_candidate_targets=targets, pass_route=route, selected_pass_route=route,
        asr_minimum_target_passed=target["asr_minimum_target_passed"],
        absolute_target_status="not_required", relative_target_status=target["relative_target_status"],
        full_target_passed=target["full_target_passed"],
        asr_comparison_scope="ours_and_per_task_available_selected_baselines_including_scorable_health_failures",
        per_task_reference_counts={m: len(runs) for m, runs in references.items()})
    audit["definition"].update(asr="per attacked task: Ours final ASR minus available selected-method minimum < .01",
        absent_baseline_task="excluded only for this task; comparison incomplete, promotion not vetoed",
        strict_boundary="discrete sample rates with exact rational difference; equality fails")
    return report


def describe_final_targets(spec, selected, results, tasks):
    references, missing = selected_task_evidence({m: cid for m, cid in selected.items() if m != "sm9rrs"}, results, tasks)
    ours = [r for r in results.get(selected.get("sm9rrs"), []) if _run_audit(r)["scorable"]]
    target = task_target(ours, references, missing, spec["performance_target"],
        vert_scorable=True, reference_candidates=selected)
    expected = sum(t["method"] == "sm9rrs" for t in tasks)
    if len(ours) != expected or not expected:
        target.update(status="incomplete", full_target_passed=False, target_qualified_for_promotion=False)
    target["role"] = "descriptive_only_not_execution_or_health_status"
    target["unhealthy_runs_retained_in_observed_tables"] = True
    return target
