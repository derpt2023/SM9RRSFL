#!/usr/bin/env python3
"""Expanded CIFAR validation with MNIST performance targets and baseline fallback.

This separate v4 entry preserves immutable v2/v3 source identities and outputs.
Worker/scheduler bodies intentionally retain the original execution protocol.
Only validation candidates, selection and promotion change; no optimizer changes.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
from statistics import fmean
import subprocess
import sys
import threading
from time import monotonic
import traceback

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import run_cifar_six_from_scratch as base
from run_cifar_six_from_scratch import (
    ALL_METHODS, METHOD_TUNABLE_PARAMETERS, PerformanceTarget, experiments, fl,
    json_safe, write_json, digest, semantic_config, file_lock, immutable_json,
    build_tasks, load_split, attach_fingerprints, checked_completed, collect_results,
    worker_environment, check_environment, round_observer, repair_completed, metrics,
    summarize_final as summarize_original_final, final_aggregate,
)
from cifar_mnist_target_gate import select_validation
from cifar_expanded_candidates import run_budget

REPO = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO / "configs/cifar10_six_mnist_gate_v4.json"


def source_hashes(repo):
    hashes = base.source_hashes(repo)
    for name in ("run_cifar_six_630.py", "cifar_mean_dual_gate.py",
                 "run_cifar_six_mnist_gate.py", "cifar_mnist_target_gate.py",
                 "cifar_expanded_candidates.py"):
        hashes[name] = hashlib.sha256((repo / name).read_bytes()).hexdigest()
    return hashes


def build_manifest(spec, contract, repo):
    manifest = base.build_manifest(spec, contract, repo)
    manifest.pop("fingerprint")
    manifest.update(schema_version=4, source_sha256=source_hashes(repo),
                    promotion_requires="ours_health_and_mnist_scenario_targets",
                    validation_run_count=len(build_tasks(spec, "validation")), final_run_count=180,
                    formal_weights_inherited_from_validation=False)
    manifest["fingerprint"] = digest(manifest)
    return manifest


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    if spec.get("schema_version") != 4 or set(spec["candidates"]) != set(ALL_METHODS):
        raise ValueError("schema_version=4 and exactly six methods are required")
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
    promotion = spec["promotion"]
    if (type(promotion.get("require_mean_dual_best")) is not bool or
            {k: v for k, v in promotion.items() if k != "require_mean_dual_best"} != {
                "required_healthy_methods": ["sm9rrs"], "performance_target_is_gate": True,
                "attack_effectiveness_is_gate": False,
                "missing_clean_reference": "report_unassessed_without_baseline_veto"}):
        raise ValueError("v4 requires Ours health/targets, explicit mean-best policy and no baseline health veto")
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
        if not candidates or (not METHOD_TUNABLE_PARAMETERS[method] and len(candidates) != 1):
            raise ValueError("each method needs candidates; fixed algorithms retain one genuine configuration")
        configurations = set()
        for candidate in candidates:
            ids.append(candidate["candidate_id"])
            if not re.fullmatch(method + r"-v10-\d{3}", candidate["candidate_id"]):
                raise ValueError("candidate id must identify its method")
            if set(candidate["parameters"]) - METHOD_TUNABLE_PARAMETERS[method]:
                raise ValueError("candidate cannot alter shared or another method's parameters")
            config = replace(base, method=method, **candidate["parameters"])
            config.validate()
            identity = digest(semantic_config(config))
            if identity in configurations:
                raise ValueError("duplicate effective candidate parameters within a method")
            configurations.add(identity)
            if candidate["variant"] not in ("original", "weak_quarantine"):
                raise ValueError("unknown development variant")
            if candidate["variant"] == "weak_quarantine" and (method != "sm9rrs"
                    or not config.detector_drift_allowance <= candidate["weak_threshold"] < config.detector_distance_threshold):
                raise ValueError("weak quarantine is Ours-only and below its strong threshold")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate ids")
    validate_expanded_protocol(spec)
    return spec


def validate_expanded_protocol(spec):
    if spec.get("run_budget") != run_budget(spec):
        raise ValueError("declared run budget differs from the actual candidates and task matrix")
    policy = {"accuracy_gap": .02, "asr_gap": .01, "max_asr": .05,
              "max_peak_asr": .2, "tail_rounds": 10}
    if spec["performance_target"] != policy:
        raise ValueError("v4 freezes the actual MNIST per-scenario performance targets")
    if spec["objective"] != {"clean_accuracy_weight": .25, "robust_accuracy_weight": .5,
                              "attack_success_weight": .2, "honest_weight_loss_weight": .05}:
        raise ValueError("v4 retains the shared CIFAR Score; target changes do not change its weights")
    for phase, seeds, ratios in (
        ("validation", {1001, 1002, 1003}, {0., .1, .3, .5, .7}),
        ("final", {1101, 1102, 1103}, {0., .2, .4, .6, .8}),
    ):
        expected = {(partition, ratio) for partition in ("iid", "dirichlet") for ratio in ratios}
        if (set(spec[phase]["seeds"]) != seeds or
                {(s["partition"], s["malicious_ratio"]) for s in spec[phase]["scenarios"]} != expected):
            raise ValueError(f"v4 {phase} must retain its three seeds and ten declared scenarios")
    if any(c["variant"] != "original" for cs in spec["candidates"].values() for c in cs):
        raise ValueError("v4 searches original policies; detector development needs a separate protocol")


def validation_blockers(output, tasks, statuses):
    """Do not turn missing evidence or a resource failure into an algorithm verdict."""
    blockers = []
    by_id = {row["task_id"]: row for row in statuses}
    for task in tasks:
        row = by_id.get(task["task_id"])
        reason = None
        if row is None or row["status"] == "pending":
            reason = "validation task is pending"
        elif row.get("error"):
            reason = "invalid completed snapshot: " + row["error"]
        elif row["status"] == "failed":
            folder = output / "tasks" / task["task_id"]
            try:
                identity = json.loads((folder / "task.json").read_text())
                failure = json.loads((folder / "failure.json").read_text())
                if (identity != task or failure.get("task_id") != task["task_id"]
                        or not isinstance(failure.get("exception"), str)
                        or not isinstance(failure.get("message"), str)):
                    raise ValueError("failed validation task has no matching identity/failure record")
                message = failure["message"].lower()
                context = failure.get("execution_context")
                if (failure["exception"] in {"OutOfMemoryError", "MemoryError", "KeyboardInterrupt", "SystemExit"}
                        or any(text in message for text in (
                            "out of memory", "disk quota", "no space left", "source code changed",
                            "data digests or split differ", "immutable experiment identity changed",
                            "no matching immutable task identity", "without their immutable identity",
                            "invalid device ordinal", "driver initialization", "cuda initialization",
                            "device-side assert", "illegal memory access", "driver version",
                        ))):
                    reason = "resource, interruption or experiment-identity failure: " + failure["message"]
                elif (not isinstance(context, dict) or context.get("task_id") != task["task_id"]
                        or context.get("candidate_id") != task["candidate"]["candidate_id"]
                        or context.get("study_phase") != "validation"):
                    reason = "failure has no matching training context; environment/data setup may be incomplete"
                elif not (failure["exception"] in {"FloatingPointError", "OverflowError", "ZeroDivisionError", "LinAlgError", "_LinAlgError"}
                          or (failure["exception"] in {"RuntimeError", "ValueError"}
                              and any(text in message for text in (
                                  "nonfinite", "non-finite", "nan", "infinity", "singular",
                                  "ill-conditioned", "failed to converge",
                              )))):
                    reason = "unclassified execution failure requires diagnosis; not an algorithm health verdict: " + failure["exception"] + ": " + failure["message"]
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                reason = str(exc)
        elif row["status"] != "complete" or row.get("metrics") is None:
            reason = "validation task has no verifiable result"
        if reason:
            blockers.append({"task_id": task["task_id"], "reason": reason})
    return blockers


def final_plan(spec, manifest, selected):
    if set(selected) != set(ALL_METHODS):
        raise ValueError("formal evaluation must include all six methods")
    tasks = attach_fingerprints(build_tasks(spec, "final", selected), manifest)
    if len(tasks) != 180:
        raise ValueError("formal evaluation must schedule exactly 180 tasks")
    return {"manifest_fingerprint": manifest["fingerprint"], "selected": selected, "tasks": tasks}


def summarize_final(spec, selected, results, statuses, tasks, selection_details=None):
    report = summarize_original_final(spec, selected, results, statuses, tasks, selection_details)
    for method, info in report["methods"].items():
        selection = (selection_details or {}).get(method, {})
        source = selection.get("selection_status", "unknown")
        info["selected_without_valid_validation"] = source in {
            "best_scored_unqualified", "fixed_fallback_unqualified"}
        info["validation_selection_raw_score"] = selection.get("selection_raw_score")
        info["validation_selected_candidate_failures"] = selection.get("candidate_failures", {}).get(
            selected.get(method), "")
        info["scorable_failed_candidate_selected"] = source == "best_scored_unqualified"
    return report


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


def run_parent(args):
    spec = load_spec(args.config)
    output = (args.output or REPO / spec["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".runner.lock", nonblocking=True):
        if not (output / "manifest.json").exists() and any(p.name != ".runner.lock" for p in output.iterdir()):
            raise ValueError("refusing a nonempty output without its immutable manifest")
        split, contract = load_split(spec, args.data_dir)
        del split
        manifest = build_manifest(spec, contract, REPO)
        immutable_json(output / "manifest.json", manifest)
        validation = attach_fingerprints(build_tasks(spec, "validation"), manifest)
        immutable_json(output / "validation_plan.json", {"manifest_fingerprint": manifest["fingerprint"], "tasks": validation})
        frozen = None
        if (output / "final_plan.json").exists():
            frozen = json.loads((output / "final_plan.json").read_text())
            if frozen != final_plan(spec, manifest, frozen.get("selected", {})):
                raise ValueError("frozen final plan differs from its manifest or declared candidates")
        print(f"SIX_METHOD_OUTPUT {output}\nVALIDATION_RUNS {len(validation)} rounds=100 devices={','.join(args.devices)}", flush=True)
        if args.phase != "final" and frozen is None:
            execute_phase(args, validation, output)
        else:
            print("VALIDATION_REUSE auditing saved validation; no validation training", flush=True)
        results, statuses = collect_results(output, validation)
        blockers = validation_blockers(output, validation, statuses)
        if blockers:
            report = {"status": "validation_evidence_incomplete", "tasks": statuses, "blockers": blockers,
                      "official_test_used_for_selection": False}
            write_json(output / "validation_summary.json", report)
            print("VALIDATION_STATUS validation_evidence_incomplete", flush=True)
            print("FINAL_NOT_STARTED: missing, damaged or resource-failed validation evidence; resume validation after resolving the reported cause.", flush=True)
            return 2
        report = select_validation(spec, results, validation)
        report["tasks"] = statuses
        write_json(output / "validation_summary.json", report)
        print("VALIDATION_STATUS " + report["status"], flush=True)
        print("VALIDATION_SELECTION " + json.dumps(json_safe({
            method: {"candidate": info.get("selected_candidate"),
                     "selection_status": info.get("selection_status"),
                     "health_qualified": info.get("health_qualified"),
                     "raw_score": info.get("selection_raw_score")}
            for method, info in report["methods"].items()
        }), ensure_ascii=False), flush=True)
        gate = report["mnist_target_gate"]
        print("OURS_MNIST_TARGET_GATE " + json.dumps(json_safe({
            key: gate.get(key) for key in ("status", "comparison_scope", "selected_candidate",
                "qualified_ours_candidates", "selected_pass_route", "relative_target_status",
                "absolute_target_passed", "require_mean_dual_best")
        }), ensure_ascii=False), flush=True)
        if report["status"] != "qualified_for_final":
            if frozen is not None:
                raise ValueError("frozen final plan is no longer justified by its immutable validation evidence")
            print("FINAL_NOT_STARTED: no healthy Ours candidate meets the declared per-scenario targets and optional mean-best requirement; validation evidence retained.", flush=True)
            return 0
        planned = final_plan(spec, manifest, report["selected"])
        if frozen is not None and frozen != planned:
            raise ValueError("validation selection differs from the frozen final plan; refusing to mix formal results")
        if args.phase == "validation":
            return 0
        immutable_json(output / "final_plan.json", planned)
        final = planned["tasks"]
        print(f"FINAL_RUNS {len(final)} parameters_frozen=true", flush=True)
        execute_phase(args, final, output)
        results, statuses = collect_results(output, final)
        final_report = summarize_final(spec, report["selected"], results, statuses, final, report["methods"])
        final_report["validation_mnist_target_gate"] = {
            key: value for key, value in gate.items()
            if key not in ("candidate_rows", "ours_candidate_targets")}
        final_report["validation_ours_target"] = report["ours_target"]
        final_report["formal_results_used_for_selection"] = False
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
        print("FINAL_EXECUTION_SUMMARY " + json.dumps({
            "all_scheduled_tasks_attempted": final_report["all_scheduled_tasks_attempted"],
            "full_execution_completed": final_report["full_execution_completed"],
            "ours_independent_health_passed": final_report["ours_independent_health_passed"],
            "health_failure_tasks": len(final_report["health_failures"]),
            "report": str(output / "final_summary.json"),
        }), flush=True)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--phase", choices=("all", "validation", "final"), default="all")
    parser.add_argument("--plan-only", action="store_true",
                        help="validate protocol and print budget/targets without data, GPUs or output writes")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.plan_only:
        if args.worker:
            parser.error("--plan-only cannot be combined with --worker")
        spec = load_spec(args.config)
        output = (args.output or REPO / spec["output_dir"]).resolve()
        print(json.dumps({"schema_version": 4, "output_dir": str(output),
            "candidate_counts": {m: len(cs) for m, cs in spec["candidates"].items()},
            "validation_runs": len(build_tasks(spec, "validation")), "final_runs": 180,
            "performance_target": spec["performance_target"], "promotion": spec["promotion"],
            "objective": spec["objective"], "formal_results_used_for_selection": False},
            ensure_ascii=False, indent=2))
        return 0
    if (not args.devices or len(set(args.devices)) != len(args.devices)
            or any(not re.fullmatch(r"cuda:\d+", device) for device in args.devices)):
        parser.error("devices must be explicit CUDA ordinals; use the progress wrapper for auto selection")
    if args.worker:
        if args.output is None or args.phase == "all" or len(args.devices) != 1:
            parser.error("internal worker requires an output, a concrete phase and exactly one CUDA device")
        run_worker(args)
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
