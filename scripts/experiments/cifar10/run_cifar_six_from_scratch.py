#!/usr/bin/env python3
"""Start an independent, resumable CIFAR-10 six-method study from raw data.

Validation never uses the official test set. A failed method is recorded and
cannot stop other validation workers or masquerade as a qualifying reference.
Formal evaluation starts when an Ours candidate passes the declared health
checks. Baseline failures and descriptive Accuracy/ASR comparisons cannot veto
that transition; unavailable baselines retain a predeclared fallback identity.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
from statistics import fmean, pstdev
import subprocess
import sys
import threading
from time import monotonic
import traceback

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import sm9rrsfl  # Install CPU thread defaults before loading numerical libraries.
import numpy as np
from sm9rrsfl import experiments, fl
from sm9rrsfl.datasets import load_image_dataset, stratified_training_three_way_split
from sm9rrsfl.fair_tuning import (ALL_METHODS, METHOD_TUNABLE_PARAMETERS,
                                 _matched_fedavg_clean_accuracy, _training_health_reasons,
                                 score_trial)
from sm9rrsfl.numerics import numerical_environment
from sm9rrsfl.ours_calibration import split_metadata
from sm9rrsfl.performance_target import PerformanceTarget, evaluate_target, scenario_key


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    experiments._durable_replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def semantic_config(config):
    row = asdict(config) if isinstance(config, fl.ExperimentConfig) else dict(config)
    for key in ("device", "sm9_workers"):
        row.pop(key, None)
    return row


def source_hashes(repo):
    paths = sorted((repo / "sm9rrsfl").glob("*.py")) + [repo / name for name in (
        "run_cifar_six_from_scratch.py", "cifar_ours_development_policy.py",
        "run_experiments_from_config.py")]
    return {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


@contextmanager
def file_lock(path, nonblocking=False):
    with Path(path).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError as exc:
            raise ValueError("another runner owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def immutable_json(path, payload):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"immutable experiment identity changed: {path}; use a new output")
    else:
        write_json(path, payload)


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    if spec.get("schema_version") != 2 or set(spec["candidates"]) != set(ALL_METHODS):
        raise ValueError("schema_version=2 and exactly six methods are required")
    data = spec["dataset"]
    if (data["name"] != "cifar10" or data["train_samples"] != 50000
            or data["test_samples"] != 10000 or data["validation_fraction"] != .05):
        raise ValueError("this entry point requires full CIFAR-10 and the 45k/2500/2500 split")
    base = fl.ExperimentConfig(**spec["shared_parameters"])
    if set(spec["shared_parameters"]) != set(asdict(base)):
        raise ValueError("shared_parameters must explicitly freeze every ExperimentConfig field")
    if (base.rounds != 100 or base.num_clients != 100 or base.early_stop
            or base.eval_interval != 1 or base.checkpoint_interval != 1
            or base.crypto_mode != "sm9" or base.compute_backend != "torch"
            or base.attack != "alternating_minimization" or base.attack_start_round != 25
            or base.detector_window != 20 or base.attack_boost != 5.
            or base.attack_epochs != 1 or base.attack_stealth_steps != 1
            or base.lr != .05 or base.lr_decay != .99 or base.local_epochs != 1
            or base.batch_size != 50 or base.attack_target_count != 200
            or base.attack_source_label != 5 or base.attack_target_label != 7
            or base.attack_distance_weight != .0001 or base.dirichlet_alpha != .5):
        raise ValueError("the v2 full-round, fixed shared training/attack protocol differs")
    gates, numerics = spec["gates"], spec["numerics"]
    if gates["max_nonfinite_updates"] != 0 or gates["min_round_completion_rate"] != 1.:
        raise ValueError("complete rounds and zero nonfinite updates are mandatory")
    if numerics != {"mode": "original_runtime", "shared_backtracking": False}:
        raise ValueError("v2 preserves original numerical execution without step backtracking")
    if spec["promotion"] != {"required_healthy_methods": ["sm9rrs"],
            "performance_target_is_gate": False, "attack_effectiveness_is_gate": False,
            "missing_clean_reference": "report_unassessed_without_baseline_veto"}:
        raise ValueError("v2 requires Ours-only health promotion and descriptive comparisons")
    fallback = {m: cs[0]["candidate_id"] for m, cs in spec["candidates"].items() if m != "sm9rrs"}
    if spec["fallback_candidates"] != fallback:
        raise ValueError("baseline fallback must be fixed to each first declared candidate")
    objective = spec["objective"]
    expected_weights = {"clean_accuracy_weight", "robust_accuracy_weight",
                        "attack_success_weight", "honest_weight_loss_weight"}
    if (set(objective) != expected_weights or any(not math.isfinite(v) or v < 0 for v in objective.values())
            or not math.isclose(sum(objective.values()), 1.)):
        raise ValueError("four finite objective weights summing to one are required")
    PerformanceTarget.parse(spec["performance_target"])
    if not 0 <= gates["max_clean_accuracy_drop"] <= 1 or not 0 < gates["fedavg_min_attack_mean_asr"] <= 1:
        raise ValueError("invalid clean utility or attack-effectiveness threshold")
    for phase in ("validation", "final"):
        seeds, scenarios = spec[phase]["seeds"], spec[phase]["scenarios"]
        if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("phase seeds must be distinct nonnegative integers")
        keys = [(s["partition"], s["malicious_ratio"]) for s in scenarios]
        if (not keys or len(keys) != len(set(keys)) or not any(r == 0 for _, r in keys)
                or not any(r > 0 for _, r in keys)
                or any(p not in ("iid", "dirichlet") or not 0 <= r < 1 for p, r in keys)):
            raise ValueError("each phase needs distinct valid clean and attacked scenarios")
        if any((p, 0.) not in keys for p, r in keys if r > 0):
            raise ValueError("every attacked partition needs its matched clean control")
    if set(spec["validation"]["seeds"]) & set(spec["final"]["seeds"]):
        raise ValueError("validation and formal seeds must be disjoint")
    ids = []
    for method, candidates in spec["candidates"].items():
        if len(candidates) != (6 if method in ("sm9rrs", "vert", "alignins") else 1):
            raise ValueError("v2 uses six candidates per tunable method and one per fixed baseline")
        for candidate in candidates:
            ids.append(candidate["candidate_id"])
            if not re.fullmatch(method + r"-v8-\d{3}", candidate["candidate_id"]):
                raise ValueError("candidate id must identify its method")
            if set(candidate["parameters"]) - METHOD_TUNABLE_PARAMETERS[method]:
                raise ValueError("candidate cannot alter shared or another method's parameters")
            config = replace(base, method=method, **candidate["parameters"])
            config.validate()
            if candidate["variant"] not in ("original", "weak_quarantine"):
                raise ValueError("unknown development variant")
            if candidate["variant"] == "weak_quarantine" and (method != "sm9rrs"
                    or not config.detector_drift_allowance <= candidate["weak_threshold"] < config.detector_distance_threshold):
                raise ValueError("weak quarantine is Ours-only and below its strong threshold")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate ids")
    return spec


def build_tasks(spec, phase, selected=None):
    tasks = []
    for method in ALL_METHODS:
        candidates = spec["candidates"][method]
        if phase == "final":
            if not selected or method not in selected:
                continue
            candidates = [c for c in candidates if c["candidate_id"] == selected[method]]
            if len(candidates) != 1:
                raise ValueError("formal candidate is not a declared validation candidate")
        for candidate in candidates:
            for seed in spec[phase]["seeds"]:
                for scenario in spec[phase]["scenarios"]:
                    values = {**spec["shared_parameters"], **candidate["parameters"], **scenario,
                              "method": method, "seed": seed}
                    config = fl.ExperimentConfig(**values)
                    if fl.experiment_config_error(config):
                        raise ValueError(fl.experiment_config_error(config))
                    task_id = (f"{phase}_{candidate['candidate_id']}_{scenario['partition']}_"
                               f"ratio{scenario['malicious_ratio']:g}_seed{seed}")
                    tasks.append({"task_id": task_id, "phase": phase, "method": method,
                                  "candidate": candidate, "config": asdict(config)})
    return tasks


def load_split(spec, data_dir=None):
    data = spec["dataset"]
    original = load_image_dataset("cifar10", str(data_dir or data["data_dir"]),
                                  download=data["download"], train_limit=50000,
                                  test_limit=10000, seed=data["split_seed"])
    split = stratified_training_three_way_split(original, seed=data["split_seed"],
                                                train_fraction=.9, calibration_fraction=.05,
                                                attack_fraction=.05)
    contract = split_metadata(split, data["split_seed"])
    for name, dataset in (("validation", split.calibration_dataset), ("final", split.main_dataset)):
        contract[name + "_arrays"] = {field: hashlib.sha256(np.ascontiguousarray(getattr(dataset, field)).tobytes()).hexdigest()
                                     for field in ("x_train", "y_train", "x_test", "y_test", "x_attack", "y_attack")}
    return split, contract


def attach_fingerprints(tasks, manifest):
    return [{**task, "fingerprint": digest({"manifest": manifest["fingerprint"], "task": task})}
            for task in tasks]


def build_manifest(spec, contract, repo):
    manifest = {"schema_version": 2, "spec": spec, "data_contract": contract,
                "source_sha256": source_hashes(repo), "old_results_required": False,
                "final_selection_uses_test_data": False,
                "shared_numerical_optimizer_changed": False,
                "baseline_algorithms_modified_by_runner": False,
                "promotion_requires": "ours_health_only",
                "failure_policy": "retain per-task failures; do not qualify invalid candidates or abort unrelated validation runs"}
    manifest["fingerprint"] = digest(manifest)
    return manifest


def checked_completed(output, task):
    folder = output / "tasks" / task["task_id"]
    if not (folder / experiments.COMPLETED_RESULTS_SNAPSHOT).exists():
        return None
    if not (folder / "task.json").exists() or json.loads((folder / "task.json").read_text()) != task:
        raise ValueError("completed snapshot has no matching immutable task identity")
    results = experiments.load_completed_results_snapshot(folder)
    if results is None or len(results) != 1 or semantic_config(results[0].config) != semantic_config(task["config"]):
        raise ValueError("completed result has the wrong configuration or damaged snapshot")
    return results[0]


def metrics(result):
    reasons = list(_training_health_reasons(result))
    rows = result.records
    if result.stopped_round != result.config.rounds or [r.round for r in rows] != list(range(result.config.rounds + 1)):
        reasons.append("incomplete_rounds")
    if result.nonfinite_updates or any(r.nonfinite_updates for r in rows):
        reasons.append("nonfinite_updates")
    if not rows or any(not math.isfinite(r.accuracy) or not 0 <= r.accuracy <= 1 for r in rows):
        reasons.append("invalid_accuracy")
    if any(not math.isfinite(value) or not 0 <= value <= 1 + 1e-9 for r in rows
           for value in (r.honest_weight_loss, r.malicious_weight_mass)):
        reasons.append("invalid_weight_metrics")
    if any(r.attack_target_confidence is not None and (not math.isfinite(r.attack_target_confidence)
            or not 0 <= r.attack_target_confidence <= 1) for r in rows):
        reasons.append("invalid_attack_confidence")
    attack = [r for r in rows if r.round >= result.config.attack_start_round] if result.config.malicious_ratio else []
    if attack and any(r.attack_target_success_rate is None or not math.isfinite(r.attack_target_success_rate)
                      or not 0 <= r.attack_target_success_rate <= 1 for r in attack):
        reasons.append("invalid_attack_metrics")
    rates = [r.attack_target_success_rate for r in attack]
    valid_rates = bool(rates) and "invalid_attack_metrics" not in reasons
    return {"healthy": not reasons, "reasons": sorted(set(reasons)), "final_accuracy": result.final_accuracy,
            "attack_mean_accuracy": fmean(r.accuracy for r in attack) if attack else None,
            "attack_mean_asr": fmean(rates) if valid_rates else None,
            "attack_peak_asr": max(rates) if valid_rates else None,
            "nonfinite_updates": result.nonfinite_updates, "stopped_round": result.stopped_round,
            "runtime_seconds": result.runtime_seconds}


def collect_results(output, tasks):
    results, statuses = {}, []
    for task in tasks:
        error = None
        try:
            result = checked_completed(output, task)
        except Exception as exc:
            result, error = None, str(exc)
        folder = output / "tasks" / task["task_id"]
        row = {"task_id": task["task_id"], "candidate_id": task["candidate"]["candidate_id"],
               "method": task["method"], "partition": task["config"]["partition"],
               "ratio": task["config"]["malicious_ratio"], "seed": task["config"]["seed"],
               "status": "complete" if result is not None else "failed" if (folder / "failure.json").exists() or error else "pending",
               "metrics": metrics(result) if result is not None else None, "error": error,
               "failure_record": str(folder / "failure.json") if (folder / "failure.json").exists() else None}
        statuses.append(row)
        if result is not None:
            results.setdefault(task["candidate"]["candidate_id"], []).append(result)
    return results, statuses


def paired_all_baselines(ours, baselines, policy, expected):
    comparisons = {}
    def relabel(value):
        if isinstance(value, dict):
            return {k.replace("vert", "reference"): relabel(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relabel(v) for v in value]
        return value.replace("vert:", "reference:") if isinstance(value, str) else value
    for method in ALL_METHODS[1:]:
        comparison = relabel(evaluate_target(ours, baselines.get(method, []), policy,
                                            expected_scenarios=expected))
        comparison["reference_method"] = method
        comparisons[method] = comparison
    complete = all(c["structurally_complete"] for c in comparisons.values())
    return {"status": "passed" if complete and all(c["status"] == "passed" for c in comparisons.values()) else "unmet" if complete else "incomplete",
            "structurally_complete": complete, "comparisons": comparisons,
            "worst_normalized_excess": max(c["worst_normalized_excess"] for c in comparisons.values()),
            "mean_normalized_excess": fmean(c["mean_normalized_excess"] for c in comparisons.values())}


def clean_utility_audit(runs, controls, maximum_drop):
    """Report missing controls explicitly; only measured healthy controls can fail utility."""
    rows = []
    for run in runs:
        if run.config.malicious_ratio != 0:
            continue
        key = (run.config.partition, run.config.dirichlet_alpha, run.config.num_clients, run.config.seed)
        reference = controls.get(key)
        value = float(run.final_accuracy)
        drop = max(0., reference - value) if reference is not None and math.isfinite(value) else None
        rows.append({"scenario": list(key), "reference_accuracy": reference,
                     "candidate_accuracy": value, "drop": drop,
                     "status": "unassessed" if drop is None else "passed" if drop <= maximum_drop + 1e-12 else "failed"})
    return {"status": "failed" if any(r["status"] == "failed" for r in rows) else
                       "passed" if rows and all(r["status"] == "passed" for r in rows) else "unassessed",
            "maximum_drop": maximum_drop, "scenarios": rows,
            "missing_reference_policy": "reported_unassessed_without_baseline_veto"}


def select_validation(spec, results_by_candidate, tasks):
    expected_count = len(spec["validation"]["seeds"]) * len(spec["validation"]["scenarios"])
    expected = {(t["config"]["partition"], t["config"]["dirichlet_alpha"], t["config"]["num_clients"],
                 t["config"]["malicious_ratio"], t["config"]["seed"]) for t in tasks}
    fedavg_id = spec["candidates"]["fedavg"][0]["candidate_id"]
    healthy_controls = [r for r in results_by_candidate.get(fedavg_id, [])
                        if r.config.malicious_ratio == 0 and metrics(r)["healthy"]]
    controls_error = None
    try:
        controls = _matched_fedavg_clean_accuracy({"fedavg-001": healthy_controls})
    except ValueError as exc:
        controls, controls_error = {}, str(exc)
    expected_clean = {(key[0], key[1], key[2], key[4]) for key in expected if key[3] == 0}
    missing_controls = [list(key) for key in sorted(expected_clean - set(controls))]
    trials, rows, selected, method_status, healthy_selected = {}, [], {}, {}, {}
    for method in ALL_METHODS:
        eligible = []
        for candidate in spec["candidates"][method]:
            cid = candidate["candidate_id"]
            runs = results_by_candidate.get(cid, [])
            utility = clean_utility_audit(runs, controls, spec["gates"]["max_clean_accuracy_drop"])
            if len(runs) != expected_count or {scenario_key(r) for r in runs} != expected:
                rows.append({"method": method, "candidate_id": cid, "valid": False,
                             "invalid_reasons": "incomplete_candidate", "result_count": len(runs),
                             "clean_utility": utility})
                continue
            # The same fixed Score ranks all eligible methods. Relative targets
            # never select Ours or qualify an unhealthy baseline.
            trial = score_trial(method, cid, candidate["parameters"], runs, objective=spec["objective"],
                                clean_accuracy_reference=None,
                                min_round_completion_rate=1., max_nonfinite_updates=0)
            trials[cid] = trial
            row = trial.row()
            row["variant"] = candidate["variant"]
            row["clean_utility"] = utility
            measured_drops = [item["drop"] for item in utility["scenarios"] if item["drop"] is not None]
            row["worst_clean_accuracy_drop"] = max(measured_drops) if measured_drops and utility["status"] != "unassessed" else None
            row["observed_worst_clean_accuracy_drop"] = max(measured_drops) if measured_drops else None
            extra = {reason for r in runs for reason in metrics(r)["reasons"]}
            if method in ("sm9rrs", "vert", "alignins") and utility["status"] == "failed":
                extra.add("clean_accuracy_drop")
            reasons = sorted(set(trial.invalid_reasons) | extra)
            row["valid"] = trial.valid and not extra
            row["invalid_reasons"] = ",".join(reasons)
            row["score"] = trial.score if row["valid"] else None
            # Legacy Score uses pessimistic sentinels when a metric is absent;
            # artifact tables use null instead, alongside the original evidence.
            if "accuracy_metrics" in reasons:
                row["clean_accuracy"] = row["robust_accuracy"] = None
            if "attack_metrics" in reasons:
                row["attack_success_rate"] = row["worst_attack_success_rate"] = None
            if "honest_weight_metrics" in reasons:
                row["honest_weight_loss"] = None
            rows.append(row)
            if row["valid"]:
                eligible.append(trial)
        if eligible:
            best = max(eligible, key=lambda t: (t.score, -t.worst_attack_success_rate, t.candidate_id))
            selected[method] = healthy_selected[method] = best.candidate_id
            selection_status = "eligible_score_selection"
        elif method != "sm9rrs":
            selected[method] = spec["fallback_candidates"][method]
            selection_status = "fixed_fallback_unqualified"
        else:
            selection_status = "no_eligible_ours_candidate"
        method_status[method] = {"status": "valid" if eligible else "no_eligible_candidate",
                                 "selection_status": selection_status,
                                 "comparison_available": bool(eligible),
                                 "selected_candidate": selected.get(method),
                                 "candidate_failures": {r["candidate_id"]: r["invalid_reasons"] for r in rows if r["method"] == method and not r["valid"]}}
    baselines = {m: results_by_candidate[cid] for m, cid in healthy_selected.items() if m != "sm9rrs"}
    policy = PerformanceTarget.parse(spec["performance_target"])
    candidates = {row["candidate_id"]: paired_all_baselines(results_by_candidate[row["candidate_id"]], baselines, policy, expected)
                  for row in rows if row["method"] == "sm9rrs" and row["valid"]}
    for audit in candidates.values():
        audit["role"] = "descriptive_only_not_selection_or_promotion"
    attack_rows = []
    # Evaluate every healthy fixed FedAvg attacked run even if another FedAvg
    # scenario failed. Missing runs are explicit, never treated as zero ASR.
    for run in results_by_candidate.get(fedavg_id, []):
        if run.config.malicious_ratio > 0 and metrics(run)["healthy"]:
            value = metrics(run)["attack_mean_asr"]
            attack_rows.append({"scenario": list(scenario_key(run)), "mean_asr": value,
                                "passed": value is not None and value >= spec["gates"]["fedavg_min_attack_mean_asr"]})
    expected_attack = sum(key[3] > 0 for key in expected)
    attack_effective = len(attack_rows) == expected_attack and expected_attack > 0 and all(r["passed"] for r in attack_rows)
    ours_audit = candidates.get(selected.get("sm9rrs"), {"status": "incomplete", "role": "descriptive_only_not_selection_or_promotion"})
    qualified = "sm9rrs" in healthy_selected
    return {"status": "qualified_for_final" if qualified else "needs_ours_development",
            "promotion_basis": {"required_method": "sm9rrs", "independent_health_required": True,
                                "clean_utility_against_available_healthy_controls_required": True,
                                "missing_healthy_clean_references": missing_controls,
                                "missing_clean_references_are_unassessed_not_passed": True,
                                "baseline_eligibility_required": False,
                                "relative_performance_required": False},
            "ours_health_passed": qualified,
            "all_six_methods_healthy": len(healthy_selected) == len(ALL_METHODS),
            "six_method_comparison_available": len(healthy_selected) == len(ALL_METHODS),
            "selected": selected, "methods": method_status, "trials": rows,
            "selection_rule": "fixed_common_score_for_eligible_candidates_else_predeclared_baseline_fallback",
            "ours_target": ours_audit, "ours_candidate_targets": candidates,
            "attack_effectiveness": {"passed": attack_effective, "role": "descriptive_only_not_promotion",
                                     "reference": "fedavg", "expected_attacked_runs": expected_attack,
                                     "observed_healthy_attacked_runs": len(attack_rows),
                                     "minimum_mean_asr": spec["gates"]["fedavg_min_attack_mean_asr"], "scenarios": attack_rows},
            "clean_reference_error": controls_error, "missing_healthy_clean_references": missing_controls,
            "official_test_used_for_selection": False, "automatic_parameter_search_continues": False}


@contextmanager
def round_observer(task_id, event_context=None, on_round=None):
    event_context = {} if event_context is None else event_context
    original = experiments.run_experiment
    def observed(*args, **kwargs):
        resume = kwargs.get("resume_state") or {}
        event_context["round"] = int(resume.get("completed_round", 0)) + 1
        event_context["last_completed_round"] = int(resume.get("completed_round", 0))
        callback = kwargs.get("checkpoint_callback")
        def progress(state):
            if not np.isfinite(np.asarray(state["params"])).all():
                raise FloatingPointError(f"non-finite global model at round {state['completed_round']}")
            if state.get("records"):
                last = state["records"][-1]
                for field in ("accuracy", "attack_target_success_rate", "attack_target_confidence",
                              "honest_weight_loss", "malicious_weight_mass"):
                    value = getattr(last, field)
                    if value is not None and (not math.isfinite(value) or not 0 <= value <= 1 + 1e-9):
                        raise FloatingPointError(f"invalid global {field} at round {state['completed_round']}")
            event_context["last_completed_round"] = state["completed_round"]
            if on_round is not None:
                on_round(state)
            if callback is not None:
                callback(state)
            event_context["round"] = state["completed_round"] + 1
            if state.get("records"):
                last = state["records"][-1]
                print(f"ROUND {task_id} round={state['completed_round']} accuracy={last.accuracy:.6f} nonfinite={last.nonfinite_updates}", flush=True)
        kwargs["checkpoint_callback"] = progress
        return original(*args, **kwargs)
    experiments.run_experiment = observed
    try:
        yield
    finally:
        experiments.run_experiment = original


def check_environment(output, metadata):
    comparable = json.loads(json.dumps(metadata))
    comparable.pop("requested_device", None)
    for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        comparable["environment"].pop(key, None)
    # Allocation identity is recorded in each task, while compatibility checks
    # allow moving to another physical card of the same numerical capability.
    actual = comparable.get("actual_compute_device", {})
    for key in ("logical_device", "uuid"):
        actual.pop(key, None)
    if "torch" in comparable:
        comparable["torch"].pop("logical_cuda_devices", None)
        comparable["torch"].pop("logical_cuda_devices_status", None)
    nvidia = comparable.pop("nvidia", {})
    comparable["driver_versions"] = sorted({r["driver_version"] for r in nvidia.get("gpus", [])})
    with file_lock(output / ".environment.lock"):
        immutable_json(output / "execution_environment.json", comparable)


def worker_environment(device):
    """Read provenance after choosing the requested GPU; leave numerical flags unchanged."""
    import torch
    torch.cuda.set_device(device)
    torch.cuda.init()
    properties = torch.cuda.get_device_properties(device)
    metadata = numerical_environment(device)
    metadata["actual_compute_device"] = {"logical_device": device, "name": properties.name,
                                         "uuid": str(properties.uuid) if hasattr(properties, "uuid") else None,
                                         "compute_capability": [properties.major, properties.minor],
                                         "total_memory_bytes": int(properties.total_memory)}
    return metadata


def repair_completed(output, task, result):
    """Finish the durable commit if a process exited after its result pickle."""
    folder = output / "tasks" / task["task_id"]
    if any(not (folder / name).exists() for name in
           ("summary.csv", "rounds.csv", "sm9rrs_diagnostics.csv", "summary.json")):
        experiments.write_result_files(folder, [result])
    if not (folder / "metrics.json").exists():
        write_json(folder / "metrics.json", metrics(result))
    experiments.finalize_config_checkpoint(folder / "checkpoints", fl.ExperimentConfig(**task["config"]), task["fingerprint"])


def run_worker(args):
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    plan = json.loads((output / f"{args.phase}_plan.json").read_text())
    tasks = [t for t in plan["tasks"] if t["task_id"] == args.worker]
    if len(tasks) != 1:
        raise ValueError("unknown task identity")
    task, spec = tasks[0], manifest["spec"]
    folder = output / "tasks" / task["task_id"]
    folder.mkdir(parents=True, exist_ok=True)
    try:
        if source_hashes(Path(__file__).resolve().parent) != manifest["source_sha256"]:
            raise ValueError("source code changed after this study was planned")
        if not (folder / "task.json").exists() and any(p.name != "worker.log" for p in folder.iterdir()):
            raise ValueError("task artifacts exist without their immutable identity")
        immutable_json(folder / "task.json", task)
        identity = fl.ExperimentConfig(**task["config"])
        complete = checked_completed(output, task)
        if complete is not None:
            repair_completed(output, task, complete)
            print("ALREADY_COMPLETED " + task["task_id"], flush=True)
            return
        metadata = worker_environment(args.devices[0])
        check_environment(output, metadata)
        write_json(folder / "environment.json", metadata)
        split, contract = load_split(spec, args.data_dir)
        if contract != manifest["data_contract"]:
            raise ValueError("CIFAR data digests or split differ from the immutable study")
        dataset = split.calibration_dataset if args.phase == "validation" else split.main_dataset
        config = replace(identity, device=args.devices[0])
        from cifar_ours_development_policy import weak_quarantine_policy
        variant = task["candidate"]
        policy = weak_quarantine_policy(variant["weak_threshold"]) if variant["variant"] == "weak_quarantine" else nullcontext()
        attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        event_context = {"attempt": attempt, "task_id": task["task_id"], "method": task["method"],
                         "study_phase": args.phase, "candidate_id": variant["candidate_id"],
                         "numerical_mode": "original_runtime", "shared_backtracking": False}
        def save_progress(state):
            # Only observe the original execution. Earlier attempt records and
            # checkpoints remain present if a later process resumes this task.
            last = state["records"][-1] if state.get("records") else None
            write_json(folder / f"attempt_{attempt}.json", {**event_context,
                       "last_completed_round": state["completed_round"],
                       "last_round": asdict(last) if last is not None else None})
        write_json(folder / f"attempt_{attempt}.json", event_context)
        with policy, round_observer(task["task_id"], event_context, save_progress):
            result = experiments.run_measured_experiment(dataset, config, checkpoint_dir=folder / "checkpoints",
                run_fingerprint=task["fingerprint"], retain_success_checkpoint=True,
                checkpoint_identity_config=identity)
        experiments.write_result_files(folder, [result])
        write_json(folder / "metrics.json", metrics(result))
        experiments.finalize_config_checkpoint(folder / "checkpoints", identity, task["fingerprint"])
        print("COMPLETED " + task["task_id"], flush=True)
    except BaseException as exc:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        failure = {"task_id": task["task_id"], "time_utc": stamp, "exception": type(exc).__name__,
                   "message": str(exc), "traceback": traceback.format_exc(),
                   "execution_context": locals().get("event_context"),
                   "nonfinite_updates_total": None,
                   "metrics_note": "Failed attempt has no complete-run metrics; unknown values are not imputed as zero."}
        write_json(folder / "failure.json", failure)
        write_json(folder / f"failure_{stamp}.json", failure)
        raise


def execute_phase(args, tasks, output):
    available, stopping, processes, lock = queue.Queue(), threading.Event(), set(), threading.Lock()
    started = monotonic()
    for device in args.devices:
        available.put(device)
    def launch(task):
        device = available.get()
        try:
            if stopping.is_set():
                return 130
            complete = checked_completed(output, task)
            if complete is not None:
                repair_completed(output, task, complete)
                print("ALREADY_COMPLETED " + task["task_id"], flush=True)
                return 0
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", task["task_id"],
                       "--phase", task["phase"], "--output", str(output), "--devices", device]
            if args.data_dir:
                command += ["--data-dir", str(args.data_dir.resolve())]
            folder = output / "tasks" / task["task_id"]
            folder.mkdir(parents=True, exist_ok=True)
            child_env = dict(os.environ)
            print(f"START {task['task_id']} device={device}", flush=True)
            with (folder / "worker.log").open("a", encoding="utf-8") as log:
                with lock:
                    if stopping.is_set():
                        return 130
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, bufsize=1, env=child_env)
                    processes.add(process)
                with process:
                    try:
                        for line in process.stdout:
                            log.write(line)
                            log.flush()
                            print(f"[{task['task_id']}] {line.rstrip()}", flush=True)
                        return process.wait()
                    finally:
                        with lock:
                            processes.discard(process)
        finally:
            available.put(device)
    failed = 0
    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        futures = [pool.submit(launch, task) for task in tasks]
        try:
            for future in as_completed(futures):
                try:
                    failed += int(future.result() != 0)
                except Exception as exc:
                    failed += 1
                    print(f"WORKER_FAILED {type(exc).__name__}: {exc}", flush=True)
                # Avoid rereading every large diagnostics pickle after every
                # completion. Full integrity checks run once at phase end.
                complete = sum((output / "tasks" / t["task_id"] / experiments.COMPLETED_RESULTS_SNAPSHOT).exists() for t in tasks)
                durations = []
                for task in tasks:
                    path = output / "tasks" / task["task_id"] / "metrics.json"
                    if path.exists():
                        try:
                            value = json.loads(path.read_text()).get("runtime_seconds")
                            if value is not None and math.isfinite(value) and value > 0:
                                durations.append(value)
                        except (OSError, ValueError):
                            pass
                eta = fmean(durations) * (len(tasks) - complete) / len(args.devices) if durations else None
                write_json(output / f"{tasks[0]['phase']}_progress.json", {"completed_snapshots": complete,
                           "total": len(tasks), "worker_failures_this_invocation": failed,
                           "elapsed_seconds_this_invocation": monotonic() - started,
                           "rough_remaining_seconds": eta,
                           "estimate_basis": "mean of completed full runs divided by active GPU lanes; method costs and retries can differ"})
                print(f"PROGRESS {complete}/{len(tasks)} rough_remaining_seconds={round(eta) if eta is not None else 'unknown'}", flush=True)
        except BaseException:
            stopping.set()
            for future in futures:
                future.cancel()
            with lock:
                for process in processes:
                    process.terminate()
            raise
    return failed


def summarize_final(spec, selected, results, statuses, tasks, selection_details=None):
    peers = {method: results.get(cid, []) for method, cid in selected.items()}
    healthy_peers = {method: [r for r in runs if metrics(r)["healthy"]] for method, runs in peers.items()}
    expected = {(t["config"]["partition"], t["config"]["dirichlet_alpha"], t["config"]["num_clients"],
                 t["config"]["malicious_ratio"], t["config"]["seed"]) for t in tasks}
    target = paired_all_baselines(healthy_peers.get("sm9rrs", []), healthy_peers,
                                 PerformanceTarget.parse(spec["performance_target"]), expected)
    target["role"] = "descriptive_only_not_execution_or_health_status"
    health = [{"task_id": row["task_id"], "reasons": row["metrics"]["reasons"] if row["metrics"] else [row["status"]]}
              for row in statuses if row["metrics"] is None or not row["metrics"]["healthy"]]
    methods = {}
    for method in ALL_METHODS:
        wanted = sum(t["method"] == method for t in tasks)
        observed = [row for row in statuses if row["method"] == method]
        full = bool(wanted) and len(observed) == wanted and all(row["metrics"] is not None for row in observed)
        healthy = full and all(row["metrics"]["healthy"] for row in observed)
        selection = (selection_details or {}).get(method, {})
        methods[method] = {"status": "complete_healthy" if healthy else "complete_unhealthy" if full else "incomplete_or_failed",
                           "selected_candidate": selected.get(method), "expected_runs": wanted,
                           "completed_runs": sum(row["metrics"] is not None for row in observed),
                           "healthy_runs": sum(bool(row["metrics"] and row["metrics"]["healthy"]) for row in observed),
                           "comparison_available": healthy,
                           "validation_selection_status": selection.get("selection_status", "unknown"),
                           "selected_without_valid_validation": selection.get("selection_status") == "fixed_fallback_unqualified"}
    full_execution = len(statuses) == len(tasks) and all(row["metrics"] is not None for row in statuses)
    all_attempted = len(statuses) == len(tasks) and all(row["status"] in ("complete", "failed") for row in statuses)
    status = "completed" if full_execution and not health else "completed_with_health_failures" if full_execution else "completed_with_task_failures" if all_attempted else "incomplete"
    clean_controls = _matched_fedavg_clean_accuracy({"fedavg-001": healthy_peers.get("fedavg", [])})
    ours_utility = clean_utility_audit(peers.get("sm9rrs", []), clean_controls, spec["gates"]["max_clean_accuracy_drop"])
    return {"status": status, "full_execution_completed": full_execution,
            "all_scheduled_tasks_attempted": all_attempted,
            "ours_independent_health_passed": methods["sm9rrs"]["status"] == "complete_healthy",
            "ours_clean_utility": ours_utility,
            "six_method_comparison_available": all(m["comparison_available"] for m in methods.values()),
            "ours_target": target, "tasks": statuses, "health_failures": health, "selected": selected,
            "methods": methods, "official_test_used_for_selection": False, "parameters_reselected": False}


def final_aggregate(tasks, results):
    """Include every planned method/scenario, including zero completed seeds."""
    rows = []
    all_results = [run for runs in results.values() for run in runs]
    keys = sorted({(t["method"], t["config"]["partition"], t["config"]["malicious_ratio"]) for t in tasks})
    for method, partition, ratio in keys:
        wanted = sorted({t["config"]["seed"] for t in tasks if (t["method"], t["config"]["partition"], t["config"]["malicious_ratio"]) == (method, partition, ratio)})
        group = [r for r in all_results if (r.config.method, r.config.partition, r.config.malicious_ratio) == (method, partition, ratio)]
        observed = sorted(r.config.seed for r in group)
        failures = sorted({reason for r in group for reason in metrics(r)["reasons"]})
        complete = observed == wanted
        values = [r.final_accuracy for r in group if math.isfinite(r.final_accuracy)]
        rates = [r.records[-1].attack_target_success_rate for r in group if r.records and r.records[-1].attack_target_success_rate is not None
                 and math.isfinite(r.records[-1].attack_target_success_rate)] if ratio else []
        rows.append({"method": method, "partition": partition, "malicious_ratio": ratio,
                     "expected_runs": len(wanted), "completed_runs": len(group),
                     "expected_seeds": json.dumps(wanted), "completed_seeds": json.dumps(observed),
                     "missing_seeds": json.dumps(sorted(set(wanted) - set(observed))),
                     "status": "incomplete" if not complete else "failed" if failures else "complete_healthy",
                     "failure_reasons": ",".join(failures),
                     "nonfinite_updates_total": sum(r.nonfinite_updates for r in group) if complete else None,
                     "observed_completed_runs_nonfinite_updates": sum(r.nonfinite_updates for r in group) if group else None,
                     "healthy_completed_runs": sum(metrics(r)["healthy"] for r in group),
                     "observed_final_accuracy_mean": fmean(values) if values else None,
                     "observed_final_accuracy_std": pstdev(values) if len(values) > 1 else 0. if values else None,
                     "observed_final_asr_mean": fmean(rates) if rates else None})
    return rows


def run_parent(args):
    repo = Path(__file__).resolve().parent
    spec = load_spec(args.config)
    output = (args.output or repo / spec["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".runner.lock", nonblocking=True):
        if not (output / "manifest.json").exists() and any(p.name != ".runner.lock" for p in output.iterdir()):
            raise ValueError("refusing a nonempty output without its immutable manifest")
        split, contract = load_split(spec, args.data_dir)
        del split
        manifest = build_manifest(spec, contract, repo)
        immutable_json(output / "manifest.json", manifest)
        validation = attach_fingerprints(build_tasks(spec, "validation"), manifest)
        immutable_json(output / "validation_plan.json", {"manifest_fingerprint": manifest["fingerprint"], "tasks": validation})
        print(f"SIX_METHOD_OUTPUT {output}\nVALIDATION_RUNS {len(validation)} rounds=100 devices={','.join(args.devices)}", flush=True)
        if args.phase != "final":
            execute_phase(args, validation, output)
        results, statuses = collect_results(output, validation)
        report = select_validation(spec, results, validation)
        report["tasks"] = statuses
        write_json(output / "validation_summary.json", report)
        print("VALIDATION_STATUS " + report["status"], flush=True)
        if report["status"] != "qualified_for_final":
            print("FINAL_NOT_STARTED: no healthy Ours candidate; validation results and baseline failures retained.", flush=True)
            return 0
        if args.phase == "validation":
            return 0
        final = attach_fingerprints(build_tasks(spec, "final", report["selected"]), manifest)
        immutable_json(output / "final_plan.json", {"manifest_fingerprint": manifest["fingerprint"],
                                                    "selected": report["selected"], "tasks": final})
        print(f"FINAL_RUNS {len(final)} parameters_frozen=true", flush=True)
        execute_phase(args, final, output)
        results, statuses = collect_results(output, final)
        final_report = summarize_final(spec, report["selected"], results, statuses, final, report["methods"])
        write_json(output / "final_summary.json", final_report)
        completed = [r for runs in results.values() for r in runs]
        final_dir = output / "final_results"
        final_dir.mkdir(parents=True, exist_ok=True)
        if completed:
            experiments.write_result_files(final_dir, completed)
        aggregate = final_aggregate(final, results)
        with (final_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
            writer.writeheader()
            writer.writerows(aggregate)
        print("FINAL_STATUS " + final_report["status"], flush=True)
        return 0


def main():
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=repo / "configs/cifar10_six_original_v2.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--phase", choices=("all", "validation", "final"), default="all")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len(args.devices) != len(set(args.devices)) or any(not re.fullmatch(r"cuda:\d+", d) for d in args.devices):
        parser.error("provide distinct logical CUDA devices, e.g. --devices cuda:0 cuda:1")
    if args.worker:
        if args.output is None or args.phase == "all" or len(args.devices) != 1:
            parser.error("worker requires output, one device and a concrete phase")
        run_worker(args)
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
