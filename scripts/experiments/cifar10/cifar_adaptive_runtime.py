"""Isolated, checkpointed execution for the ResNet/TPE experiment.

No legacy scientific module is edited. The new model is installed only inside
new worker processes, and its source is included in the new immutable identity.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import run_cifar_six_from_scratch as base
import run_cifar_six_relative_best as legacy

REPO = Path(__file__).resolve().parent
NEW_SOURCES = ("run_cifar_adaptive.py", "cifar_adaptive_runtime.py",
               "cifar_adaptive_search.py", "cifar_adaptive_gate.py",
               "cifar_adaptive_reporting.py", "cifar_resnet_gn.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    result = legacy.source_hashes(REPO)
    result.update({name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                   for name in NEW_SOURCES})
    return result


def build_manifest(spec, contract):
    import cifar_resnet_gn
    payload = {"schema_version": 8, "spec": spec, "data_contract": contract,
               "source_sha256": source_hashes(),
               "model": cifar_resnet_gn.protocol_descriptor(),
               "formal_results_used_for_selection": False,
               "formal_weights_inherited_from_validation": False,
               "experiment_type": "ours_favorable_conditions_selected_on_two_development_seeds"}
    return {**payload, "fingerprint": base.digest(payload)}


def attach_tasks(spec, phase, manifest, selected=None):
    return base.attach_fingerprints(base.build_tasks(spec, phase, selected), manifest)


def save_plan(output, name, tasks, manifest):
    folder = Path(output) / "task_plans"
    folder.mkdir(exist_ok=True)
    path = folder / (name + ".json")
    base.immutable_json(path, {"manifest_fingerprint": manifest["fingerprint"], "tasks": tasks})
    return path


class BudgetPause(RuntimeError):
    """A saved round hit the declared search budget; not algorithm evidence."""


@contextmanager
def round_monitor(task, folder, deadline=None):
    original = base.experiments.run_experiment
    event = {"task_id": task["task_id"], "study_phase": task["phase"],
             "candidate_id": task["candidate"]["candidate_id"], "last_completed_round": 0}
    cancelled = [False]
    handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        handlers[signum] = signal.signal(signum, lambda *_: cancelled.__setitem__(0, True))

    def observed(*args, **kwargs):
        previous = kwargs.get("checkpoint_callback")
        resume = kwargs.get("resume_state") or {}
        event["last_completed_round"] = int(resume.get("completed_round", 0))

        def progress(state):
            current = state["completed_round"]
            event["round"] = current
            if not np.isfinite(np.asarray(state["params"])).all():
                raise FloatingPointError(f"non-finite global model at round {current}")
            if state.get("records"):
                row = state["records"][-1]
                for field in ("accuracy", "attack_target_success_rate", "attack_target_confidence",
                              "honest_weight_loss", "malicious_weight_mass"):
                    value = getattr(row, field)
                    if value is not None and (not math.isfinite(value) or not 0 <= value <= 1 + 1e-9):
                        raise FloatingPointError(f"invalid global {field} at round {current}")
            # Persist the completed round before any budget/interrupt exception.
            if previous is not None:
                previous(state)
            event["last_completed_round"] = current
            base.write_json(folder / "progress.json", event)
            print(f"ROUND {task['task_id']} round={current}/{task['config']['rounds']}", flush=True)
            if current < task["config"]["rounds"] and (cancelled[0] or (deadline and time.time() >= deadline)):
                raise BudgetPause("checkpoint saved; interrupted or search time budget exhausted")
        kwargs["checkpoint_callback"] = progress
        return original(*args, **kwargs)
    base.experiments.run_experiment = observed
    try:
        yield event
    finally:
        base.experiments.run_experiment = original
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def failure_class(exc, event):
    if isinstance(exc, BudgetPause):
        return "budget_or_interrupt"
    if isinstance(exc, FloatingPointError) and event and event.get("round") is not None:
        return "algorithm_numerical"
    return "infrastructure_or_execution"


def worker(args):
    import cifar_resnet_gn
    output = args.output.resolve()
    manifest = read_json(output / "manifest.json")
    plan_path = args.plan.resolve()
    if plan_path.parent != output / "task_plans":
        raise ValueError("worker plan must be inside this study's task_plans directory")
    plan = read_json(plan_path)
    if plan["manifest_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("worker plan belongs to a different experiment")
    matches = [t for t in plan["tasks"] if t["task_id"] == args.worker]
    if len(matches) != 1:
        raise ValueError("unknown or duplicate task")
    task = matches[0]
    raw = {k: v for k, v in task.items() if k != "fingerprint"}
    if base.attach_fingerprints([raw], manifest)[0] != task:
        raise ValueError("task fingerprint differs from its immutable contents")
    folder = output / "tasks" / task["task_id"]
    folder.mkdir(parents=True, exist_ok=True)
    event = None
    try:
        if source_hashes() != manifest["source_sha256"]:
            raise ValueError("scientific source changed; preserve this study and use its original version")
        # Parent creates identity before launching. Never manufacture identity
        # for an orphan checkpoint or training result.
        if read_json(folder / "task.json") != task:
            raise ValueError("task identity mismatch")
        identity = base.fl.ExperimentConfig(**task["config"])
        complete = base.checked_completed(output, task)
        if complete is not None:
            base.repair_completed(output, task, complete)
            print("ALREADY_COMPLETED " + task["task_id"], flush=True)
            return 0
        cifar_resnet_gn.install_runtime()
        metadata = base.worker_environment(args.devices[0])
        base.check_environment(output, metadata)
        base.write_json(folder / "environment.json", metadata)
        split, contract = base.load_split(manifest["spec"], args.data_dir)
        if contract != manifest["data_contract"]:
            raise ValueError("data contents or train/validation/test split changed")
        dataset = split.calibration_dataset if task["phase"] == "validation" else split.main_dataset
        config = replace(identity, device=args.devices[0])
        with round_monitor(task, folder, args.stop_at) as event:
            result = base.experiments.run_measured_experiment(
                dataset, config, checkpoint_dir=folder / "checkpoints",
                run_fingerprint=task["fingerprint"], retain_success_checkpoint=True,
                checkpoint_identity_config=identity)
        base.experiments.write_result_files(folder, [result])
        base.write_json(folder / "metrics.json", base.metrics(result))
        base.experiments.finalize_config_checkpoint(folder / "checkpoints", identity, task["fingerprint"])
        print("COMPLETED " + task["task_id"], flush=True)
        return 0
    except BaseException as exc:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        failure = {"task_id": task["task_id"], "task_fingerprint": task["fingerprint"], "time_utc": stamp,
                   "exception": type(exc).__name__, "message": str(exc),
                   "kind": failure_class(exc, event), "execution_context": event,
                   "traceback": traceback.format_exc()}
        base.write_json(folder / ("failure_" + stamp + ".json"), failure)
        base.write_json(folder / "failure.json", failure)
        if isinstance(exc, BudgetPause):
            print("CHECKPOINT_PAUSED " + task["task_id"], flush=True)
            return 75
        raise


def terminal_failure(output, task):
    path = Path(output) / "tasks" / task["task_id"] / "failure.json"
    if not path.exists():
        return None
    if read_json(path.parent / "task.json") != task:
        raise ValueError("failure cache has a different immutable task identity")
    row = read_json(path)
    if row.get("task_id") != task["task_id"] or row.get("task_fingerprint") != task["fingerprint"]:
        raise ValueError("failure belongs to a different task")
    return row


def collect(output, tasks):
    """Completed snapshots take precedence over retained historical failures."""
    groups, statuses = {}, []
    for task in tasks:
        result = base.checked_completed(output, task)
        row = {"task_id": task["task_id"], "method": task["method"],
               "candidate_id": task["candidate"]["candidate_id"]}
        if result is not None:
            base.repair_completed(output, task, result)
            groups.setdefault(row["candidate_id"], []).append(result)
            row.update(status="complete", healthy=base.metrics(result)["healthy"])
        else:
            failure = terminal_failure(output, task)
            row.update(status="failed" if failure else "pending", failure=failure)
        statuses.append(row)
    return groups, statuses


def evidence_resolved(statuses):
    return all(row["status"] == "complete" or
               (row.get("failure") or {}).get("kind") == "algorithm_numerical"
               for row in statuses)


def ensure_identity(output, task):
    folder = Path(output) / "tasks" / task["task_id"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "task.json"
    if not path.exists() and any(folder.iterdir()):
        raise ValueError(f"orphan task artifacts require inspection: {folder}")
    base.immutable_json(path, task)
    return folder


def execute(args, tasks, plan, *, deadline=None, heartbeat=None):
    """One worker per GPU; budget stops dispatch and checkpoints in-flight work.

    Model/numerical failures count as attempted evidence. OOM/IO/environment
    failures are retried once per invocation and remain operational blockers.
    """
    output = args.output.resolve()
    pending = []
    for task in tasks:
        result = base.checked_completed(output, task)
        if result is not None:
            base.repair_completed(output, task, result)
            print("REUSE " + task["task_id"], flush=True)
        elif (terminal_failure(output, task) or {}).get("kind") != "algorithm_numerical":
            pending.append(task)
    active, retry, finished = {}, {}, len(tasks) - len(pending)
    free = list(args.devices)
    last_display = 0.
    try:
        while pending or active:
            budget_expired = bool(deadline and time.time() >= deadline)
            while pending and free and not budget_expired:
                if deadline and time.time() >= deadline:
                    budget_expired = True
                    break
                task, device = pending.pop(0), free.pop(0)
                folder = ensure_identity(output, task)
                command = [sys.executable, "-u", str(REPO / "run_cifar_adaptive.py"),
                           "--worker", task["task_id"], "--output", str(output),
                           "--plan", str(plan.resolve()), "--devices", device]
                if args.data_dir:
                    command += ["--data-dir", str(args.data_dir.resolve())]
                if deadline:
                    command += ["--stop-at", repr(deadline)]
                log = (folder / "worker.log").open("a", encoding="utf-8")
                try:
                    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True, cwd=REPO)
                except BaseException:
                    log.close()
                    raise
                active[proc.pid] = (proc, task, device, log)
                print(f"START {task['task_id']} device={device}", flush=True)
            for pid, (proc, task, device, log) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                del active[pid]
                free.append(device)
                kind = (terminal_failure(output, task) or {}).get("kind")
                if code not in (0, 75) and kind != "algorithm_numerical" and retry.get(task["task_id"], 0) < 1 and not budget_expired:
                    retry[task["task_id"]] = 1
                    pending.append(task)
                    print("RETRY_SAME_CONFIG " + task["task_id"], flush=True)
                else:
                    finished += 1
                    print(f"WORKER_EXIT {task['task_id']} code={code} kind={kind}", flush=True)
            if heartbeat:
                heartbeat()
            if time.monotonic() - last_display >= 15:
                print(f"PROGRESS settled={finished}/{len(tasks)} active={len(active)} queued={len(pending)} budget_expired={budget_expired}", flush=True)
                last_display = time.monotonic()
            if budget_expired and not active:
                break
            if active:
                time.sleep(1)
        _, statuses = collect(output, tasks)
        return statuses
    except BaseException:
        for proc, _, _, _ in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        # Cooperative workers save their current round. If this takes longer,
        # the last previously saved round remains the recoverable boundary.
        end = time.monotonic() + 30
        for proc, _, _, log in active.values():
            try:
                proc.wait(timeout=max(.1, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            log.close()
        raise
