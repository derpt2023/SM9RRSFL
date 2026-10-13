"""Independent schema-8 endpoint selection; legacy scientific modules unchanged.

The shared protocol is frozen within each public-parameter block. Only complete,
identity-matched validation evidence can qualify Ours. Complete numerical health
failures remain baseline references, and missing baseline tasks stay explicit.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import math
from statistics import fmean

import run_cifar_six_from_scratch as base
from cifar_mean_dual_gate import SCORABLE_HEALTH_FAILURES, TOLERANCE
from cifar_relative_asr_gate import asr_fraction
from cifar_relative_best_gate import ACCURACY_TARGET, ASR_TARGET, accuracy_fraction, evaluation_count
from sm9rrsfl.calibration_policy import weighted_score
from sm9rrsfl.performance_target import scenario_key

OBJECTIVE = {"clean_accuracy_weight": .25, "robust_accuracy_weight": .5,
             "attack_success_weight": .2, "honest_weight_loss_weight": .05}
PERFORMANCE_TARGET = {"accuracy_gap": .02, "asr_gap": .02, "tail_rounds": 10}


def selection_metrics(round_count):
    return {"accuracy": "final_round", "asr": "final_round", "round": round_count,
            "performance_gate": "final_round_only", "process_metrics": "diagnostic_only",
            "replicate_summary": "equal_mean_of_final_values",
            "honest_weight_loss": "unchanged_time_and_scenario_mean"}


def _unit(value):
    try:
        return value is not None and math.isfinite(value) and 0 <= value <= 1
    except TypeError:
        return False


def run_audit(run, round_count=None):
    """Original full-process health plus strict, dynamic endpoint integrity."""
    reasons = set()
    expected_round = run.config.rounds if round_count is None else round_count
    if run.config.rounds != expected_round:
        reasons.add("wrong_declared_round_count")
    try:
        reasons.update(base.metrics(run)["reasons"])
    except (TypeError, ValueError, AttributeError, OverflowError):
        reasons.add("invalid_health_metrics")
    if (run.stopped_round != expected_round
            or [r.round for r in run.records] != list(range(expected_round + 1))):
        reasons.add("incomplete_rounds")
    if not _unit(run.final_accuracy) or not run.records:
        reasons.add("invalid_accuracy")
    elif abs(run.final_accuracy - run.records[-1].accuracy) > TOLERANCE:
        reasons.add("final_accuracy_record_mismatch")
    if run.config.malicious_ratio:
        attacked = [r for r in run.records if r.round >= run.config.attack_start_round]
        if (len(attacked) != expected_round - run.config.attack_start_round + 1
                or any(not _unit(r.attack_target_success_rate) for r in attacked)):
            reasons.add("invalid_attack_metrics")
    return {"healthy": not reasons, "scorable": not (reasons - SCORABLE_HEALTH_FAILURES),
            "reasons": sorted(reasons)}


def _check_validation_plan(spec, tasks):
    if spec.get("schema_version") != 8:
        raise ValueError("adaptive gate requires schema_version 8")
    shared, validation = spec["shared_parameters"], spec["validation"]
    if shared["rounds"] != 150:
        raise ValueError("schema 8 qualification requires final round 150")
    if ("selection_metrics" in spec
            and spec["selection_metrics"] != selection_metrics(shared["rounds"])):
        raise ValueError("selection_metrics must explicitly describe the final round 150")
    if shared["attack_start_round"] != shared["detector_window"] + 2:
        raise ValueError("attack onset must be the common clean window plus two")
    seeds, scenarios = validation["seeds"], validation["scenarios"]
    ratios = {s["malicious_ratio"] for s in scenarios}
    if (len(seeds) != 2 or len(set(seeds)) != 2
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(spec["final"]["seeds"]) != 1
            or any(type(seed) is not int or seed < 0 for seed in spec["final"]["seeds"])
            or set(seeds) & set(spec["final"]["seeds"])
            or len(scenarios) != 10 or len(ratios) != 5 or 0 not in ratios
            or any(not _unit(r) or r == 1 for r in ratios)
            or {(s["partition"], s["malicious_ratio"]) for s in scenarios}
            != {(p, r) for p in ("iid", "dirichlet") for r in ratios}):
        raise ValueError("two independent validation seeds and ten paired scenarios required")
    if spec["objective"] != OBJECTIVE or spec["gates"]["max_clean_accuracy_drop"] != .03:
        raise ValueError("schema 8 retains the fixed Score and 3pp clean utility gate")
    if spec["performance_target"] != PERFORMANCE_TARGET:
        raise ValueError("both endpoint gaps must be inclusive 2pp")
    if type(spec.get("promotion", {}).get("require_mean_dual_best", False)) is not bool:
        raise ValueError("mean-dual requirement must be boolean")
    if set(spec["candidates"]) != set(base.ALL_METHODS):
        raise ValueError("six method families required")
    ids = []
    for method, candidates in spec["candidates"].items():
        if not candidates:
            raise ValueError("each method needs at least one candidate")
        for candidate in candidates:
            ids.append(candidate["candidate_id"])
            if (set(candidate["parameters"]) - base.METHOD_TUNABLE_PARAMETERS[method]
                    or "detector_window" in candidate["parameters"]
                    or candidate.get("variant") != "original"):
                raise ValueError("candidate must preserve the shared protocol and original algorithm")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate id")
    expected = base.build_tasks(spec, "validation")
    wanted = {t["task_id"]: t for t in expected}
    if len(tasks) != len(expected):
        raise ValueError("entire validation task matrix required")
    seen = set()
    for task in tasks:
        identity = {k: v for k, v in task.items() if k not in ("fingerprint", "public_trial_id")}
        if task["task_id"] in seen or identity != wanted.get(task["task_id"]):
            raise ValueError("validation task identity mismatch or duplicate")
        seen.add(task["task_id"])
    return expected


def selected_task_evidence(selected, results, tasks, *, round_count):
    wanted = {m: {} for m in selected}
    for task in tasks:
        method = task["method"]
        if selected.get(method) == task["candidate"]["candidate_id"]:
            wanted[method][base.digest(base.semantic_config(task["config"]))] = task
    references, missing = {}, {}
    for method, cid in selected.items():
        seen, available, failures = set(), {}, {}
        for run in results.get(cid, []):
            identity = base.digest(base.semantic_config(run.config))
            if identity not in wanted[method] or identity in seen:
                raise ValueError("duplicate or foreign reference task: " + cid)
            seen.add(identity)
            key, audit = scenario_key(run), run_audit(run, round_count)
            if audit["scorable"]:
                available[key] = run
            else:
                failures[key] = audit["reasons"]
        for identity, task in wanted[method].items():
            if identity not in seen:
                config = base.fl.ExperimentConfig(**task["config"])
                key = (config.partition, config.dirichlet_alpha, config.num_clients,
                       config.malicious_ratio, config.seed)
                failures[key] = ["missing_complete_task"]
        references[method], missing[method] = available, failures
    return references, missing


def select_validation(spec, results_by_candidate, tasks):
    expected = _check_validation_plan(spec, tasks)
    rounds = spec["shared_parameters"]["rounds"]
    planned = {}
    for task in expected:
        planned.setdefault(task["candidate"]["candidate_id"], []).append(task)
    if set(results_by_candidate) - set(planned):
        raise ValueError("undeclared candidate evidence")
    usable, evidence = {}, {}
    for cid, group in planned.items():
        runs = results_by_candidate.get(cid, [])
        wanted = Counter(base.digest(base.semantic_config(t["config"])) for t in group)
        observed = Counter(base.digest(base.semantic_config(r.config)) for r in runs)
        if any(key not in wanted or count != 1 for key, count in observed.items()):
            raise ValueError("duplicate or foreign validation result: " + cid)
        audits = [run_audit(r, rounds) for r in runs]
        complete = observed == wanted
        scorable = complete and all(a["scorable"] for a in audits)
        usable[cid] = [r for r, a in zip(runs, audits) if a["scorable"]]
        evidence[cid] = {"scorable": scorable, "audits": audits, "complete": complete}
    compatibility = deepcopy(spec)
    # Legacy helper requires this descriptive field. It is never a schema-8
    # promotion condition or an objective for choosing attack strength.
    compatibility["gates"].setdefault("fedavg_min_attack_mean_asr", .5)
    compatibility["fallback_candidates"] = {m: cs[0]["candidate_id"] for m, cs in spec["candidates"].items() if m != "sm9rrs"}
    report = base.select_validation(compatibility, usable, expected)
    report["attack_effectiveness"].update(role="diagnostic_only_not_selection_or_promotion",
        threshold_source="legacy_v7_fedavg_min_attack_mean_asr_0.5")
    candidates = {}
    for trial in report["trials"]:
        cid = trial["candidate_id"]
        runs, proof = results_by_candidate.get(cid, []), evidence[cid]
        reasons = set(filter(None, trial.get("invalid_reasons", "").split(",")))
        reasons.update(reason for audit in proof["audits"] for reason in audit["reasons"])
        if not proof["complete"]:
            reasons.add("incomplete_candidate")
        scorable = bool(proof["scorable"] and not (reasons - SCORABLE_HEALTH_FAILURES)
                        and _unit(trial.get("honest_weight_loss")))
        if scorable:
            attacked = [r for r in runs if r.config.malicious_ratio]
            trial["process_diagnostics"] = {k: trial.get(k) for k in
                ("robust_accuracy", "attack_success_rate", "worst_attack_success_rate", "score")}
            trial.update(clean_accuracy=fmean(r.records[-1].accuracy for r in runs if not r.config.malicious_ratio),
                         robust_accuracy=fmean(r.records[-1].accuracy for r in attacked),
                         attack_success_rate=fmean(r.records[-1].attack_target_success_rate for r in attacked),
                         worst_attack_success_rate=max(r.records[-1].attack_target_success_rate for r in attacked))
            raw = weighted_score(trial, spec["objective"])
        else:
            raw = None
        eligible = bool(scorable and trial["valid"] and not reasons)
        trial.update(valid=eligible, health_qualified=eligible, scorable=scorable,
                     invalid_reasons=",".join(sorted(reasons)), raw_score=raw,
                     score=raw if eligible else None, selection_metrics=selection_metrics(rounds))
        candidates[cid] = {"candidate_id": cid, "method": trial["method"],
            "eligible": eligible, "health_qualified": eligible, "scorable": scorable,
            "raw_score": raw, "invalid_reasons": trial["invalid_reasons"],
            "mean_accuracy": fmean(r.records[-1].accuracy for r in runs) if scorable else None,
            "mean_asr": trial.get("attack_success_rate") if scorable else None,
            "worst_asr": trial.get("worst_attack_success_rate") if scorable else None,
            "tasks": [{"scenario": list(scenario_key(r)), **audit} for r, audit in zip(runs, proof["audits"])]}
    def rank(cid):
        row = candidates[cid]
        return row["raw_score"], -row["worst_asr"], cid
    selected = {}
    for method in base.ALL_METHODS:
        ids = [c["candidate_id"] for c in spec["candidates"][method]]
        healthy = [cid for cid in ids if candidates[cid]["eligible"]]
        scored = [cid for cid in ids if candidates[cid]["scorable"]]
        chosen = max(healthy or scored, key=rank) if healthy or scored else ids[0]
        if method != "sm9rrs":
            selected[method] = chosen
        report["methods"][method].update(selected_candidate=chosen if method != "sm9rrs" else None,
            selection_status="eligible_score_selection" if healthy else "best_scored_unqualified" if scored else "fixed_fallback_unqualified",
            health_qualified=bool(healthy), scorable=candidates[chosen]["scorable"],
            raw_score=candidates[chosen]["raw_score"])
    references, missing = selected_task_evidence(selected, results_by_candidate, expected, round_count=rounds)
    healthy_ours = [cid for cid, row in candidates.items() if row["method"] == "sm9rrs" and row["eligible"]]
    comparison_ids = healthy_ours + [cid for cid in selected.values() if candidates[cid]["scorable"]]
    best_acc = max((candidates[cid]["mean_accuracy"] for cid in comparison_ids), default=None)
    best_asr = min((candidates[cid]["mean_asr"] for cid in comparison_ids), default=None)
    targets = {}
    for cid, row in candidates.items():
        if row["method"] != "sm9rrs":
            continue
        if not row["eligible"]:
            targets[cid] = {"status": "ineligible", "promotion_qualified": False,
                            "health_qualified": False, "invalid_reasons": row["invalid_reasons"]}
            continue
        target = task_target(results_by_candidate[cid], references, missing,
            reference_candidates={"sm9rrs": cid, **selected},
            accuracy_samples=evaluation_count(spec, "validation"), round_count=rounds)
        joint = best_acc - row["mean_accuracy"] <= TOLERANCE and row["mean_asr"] - best_asr <= TOLERANCE
        target.update(health_qualified=True, mean_dual_best_passed=joint,
            mean_accuracy=row["mean_accuracy"], mean_asr=row["mean_asr"],
            promotion_qualified=target["target_qualified_for_promotion"] and
                (joint or not spec.get("promotion", {}).get("require_mean_dual_best", False)))
        targets[cid] = target
    qualified = [cid for cid, target in targets.items() if target["promotion_qualified"]]
    chosen = max(qualified, key=lambda cid: (targets[cid]["mean_dual_best_passed"], *rank(cid))) if qualified else None
    best_healthy = max(healthy_ours, key=rank) if healthy_ours else None
    if chosen:
        selected["sm9rrs"] = chosen
    target = targets.get(chosen, {"status": "unmet", "full_target_passed": False,
        "accuracy_maximum_target_passed": False, "asr_minimum_target_passed": False,
        "relative_target_status": "unassessed"})
    report.update(selected=selected, ours_candidate_targets=targets, ours_target=target,
        ours_health_passed=bool(healthy_ours), best_healthy_ours_candidate=best_healthy,
        ours_final_target_passed=bool(chosen and target["full_target_passed"]),
        status="qualified_for_final" if chosen else "needs_ours_target_development" if healthy_ours else "needs_ours_development",
        selection_metrics=selection_metrics(rounds), accuracy_target=dict(ACCURACY_TARGET), asr_target=dict(ASR_TARGET))
    report["methods"]["sm9rrs"].update(selected_candidate=chosen, health_qualified=bool(chosen),
        scorable=bool(chosen), raw_score=candidates[chosen]["raw_score"] if chosen else None,
        selection_status="healthy_final_relative_best_selection" if chosen else
            "no_target_qualified_ours_candidate" if healthy_ours else "no_eligible_ours_candidate")
    report["promotion_basis"].update(relative_performance_required=True, final_round=rounds,
                                     missing_baseline_tasks_do_not_independently_block=True)
    report["final_metric_gate"] = {"status": "passed" if chosen else "unmet" if healthy_ours else "no_healthy_ours_candidate",
        "selected_candidate": chosen, "eligible_ours_candidates": sorted(healthy_ours),
        "qualified_ours_candidates": sorted(qualified), "best_healthy_ours_candidate": best_healthy,
        "candidate_rows": candidates, "ours_candidate_targets": targets, "selected_target": target,
        "best_mean_accuracy": best_acc, "best_mean_asr": best_asr, "selection_metrics": selection_metrics(rounds)}
    return report


def task_target(ours, references, missing, *, reference_candidates, accuracy_samples, round_count=150):
    if len({scenario_key(run) for run in ours}) != len(ours):
        raise ValueError("duplicate Ours scenario in target evidence")
    for run in ours:
        if not run_audit(run, round_count)["scorable"]:
            raise ValueError("Ours target requires complete numerical endpoint evidence")
    rows, accuracy_failures, asr_failures = [], [], []
    for run in sorted(ours, key=scenario_key):
        key, final = scenario_key(run), run.records[-1]
        attacked = run.config.malicious_ratio > 0
        peers = {"sm9rrs": run, **{m: group[key] for m, group in references.items() if key in group}}
        absent = [m for m in base.ALL_METHODS if m not in peers]
        evidence = {m: {"candidate": reference_candidates[m], "task_healthy": run_audit(peer, round_count)["healthy"],
                        "health_reasons": run_audit(peer, round_count)["reasons"]} for m, peer in peers.items()}
        common = {"available_methods": list(peers), "missing_methods": absent,
                  "six_method_comparison_complete": not absent,
                  "missing_reasons": {m: missing.get(m, {}).get(key, ["missing_complete_task"]) for m in absent}}
        accuracies = {m: accuracy_fraction(peer, accuracy_samples) for m, peer in peers.items()}
        maximum = max(accuracies.values())
        accuracy_gap = maximum - accuracies["sm9rrs"]
        accuracy_pass = accuracy_gap <= Fraction(str(ACCURACY_TARGET["max_gap"]))
        values = {"ours_accuracy": final.accuracy}
        row = {"scenario": list(key), "selection_round": round_count,
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
            rates = {m: asr_fraction(peer) for m, peer in peers.items()}
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
        "selection_metrics": selection_metrics(round_count), "accuracy_maximum_target_passed": accuracy_pass,
        "asr_minimum_target_passed": asr_pass, "target_qualified_for_promotion": qualified,
        "full_target_passed": qualified and complete, "six_method_comparison_complete": complete,
        "structurally_complete": complete, "missing_reference_is_not_a_pass": True,
        "relative_target_status": "unmet" if not qualified else "passed" if complete else "unassessed",
        "accuracy_maximum_failures": accuracy_failures, "asr_minimum_failures": asr_failures,
        "accuracy_passed_task_count": sum(r["accuracy_maximum"]["passed"] for r in rows),
        "accuracy_task_count": len(rows), "asr_passed_task_count": sum(r["asr_minimum"]["passed"] for r in attack_rows),
        "asr_task_count": len(attack_rows), "scenarios": rows}


def describe_final_targets(spec, selected, results, tasks):
    """Describe frozen formal results without selecting new parameters."""
    rounds = spec["shared_parameters"]["rounds"]
    references, missing = selected_task_evidence(selected, results, tasks, round_count=rounds)
    ours = list(references.pop("sm9rrs", {}).values())
    ours_missing = missing.pop("sm9rrs", {})
    target = task_target(ours, references, missing, reference_candidates=selected,
                         accuracy_samples=evaluation_count(spec, "final"), round_count=rounds)
    target.update(role="descriptive_only_not_execution_or_health_status",
                  missing_ours_tasks=[{"scenario": list(key), "reasons": value}
                                      for key, value in ours_missing.items()])
    if ours_missing:
        target.update(status="incomplete", full_target_passed=False,
                      target_qualified_for_promotion=False,
                      accuracy_maximum_target_passed=False, asr_minimum_target_passed=False,
                      structurally_complete=False, six_method_comparison_complete=False)
    return target
