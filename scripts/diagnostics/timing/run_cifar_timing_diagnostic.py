#!/usr/bin/env python3
"""Stage 2 only: 26 CNN E1 detector-window/attack-start diagnostics.

The two completed clean studies are read-only references. This entry does not
run parameter search, formal experiments, or subsequent diagnostic stages.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import run_cifar_diagnostic as clean
import cifar_timing_protocol as protocol

base, runtime = clean.base, clean.runtime
REPO = Path(__file__).resolve().parent


def separate_outputs(output, clean_output, matched_output):
    """Reject aliases, nested references, and symlinks into an older study."""
    paths = [Path(p).resolve() for p in (output, clean_output, matched_output)]
    for i, left in enumerate(paths):
        for right in paths[i + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("timing output and both references must be separate sibling studies")


def worker(args):
    manifest, tasks = protocol.read_study(args.output, current_sources=True)
    matches = [task for task in tasks if task["task_id"] == args.worker]
    if len(matches) != 1 or len(args.devices) != 1 or args.devices == ["auto"]:
        raise ValueError("worker requires one declared timing task and one explicit device")
    if runtime.read_json(args.output / "execution_environment.json") != manifest["reference"]["execution_environment"]:
        raise ValueError("timing worker environment differs from the frozen reference")
    task = matches[0]
    if task["model"] != "v7_cnn" or task["phase"] != "validation":
        raise ValueError("timing workers require the original CNN on the calibration split")
    folder = args.output / "tasks" / task["task_id"]
    if runtime.read_json(folder / "task.json") != task:
        raise ValueError("timing worker task identity mismatch")
    with base.file_lock(folder / ".worker.lock", nonblocking=True):
        return run_task(args, manifest, task, folder)


def run_task(args, manifest, task, folder):
    if base.checked_completed(args.output, task) is not None:
        print("ALREADY_COMPLETED " + task["task_id"], flush=True)
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    attempts = folder / "attempts"
    attempts.mkdir(exist_ok=True)
    started = time.monotonic()
    attempt = {"task_fingerprint": task["fingerprint"], "device": args.devices[0],
               "started_at": datetime.now(timezone.utc).isoformat(), "status": "running"}
    path = attempts / (stamp + ".json")
    base.write_json(path, attempt)
    event, cuda = {}, None
    try:
        metadata = base.worker_environment(args.devices[0])
        base.check_environment(args.output, metadata)
        base.write_json(folder / "environment.json", metadata)
        import torch
        torch.cuda.reset_peak_memory_stats(args.devices[0])
        cuda = torch.cuda
        split, contract = base.load_split(manifest["spec"], args.data_dir)
        if contract != manifest["data_contract"]:
            raise ValueError("CIFAR data contract changed")
        identity = base.fl.ExperimentConfig(**task["config"])
        config = replace(identity, device=args.devices[0])
        checkpoint = base.experiments._checkpoint_path(folder / "checkpoints", identity)
        if checkpoint.exists():
            state, _, _ = base.experiments._load_round_checkpoint(checkpoint, identity, task["fingerprint"])
            if state is None:
                raise ValueError("damaged or mismatched checkpoint; refusing silent restart")
        # No clean-loss observer: adversarial local work and revoked clients do
        # not follow that observer's clean FedAvg sample-count contract.
        with clean.model_runtime("v7_cnn"), runtime.round_monitor(task, folder) as event:
            result = base.experiments.run_measured_experiment(split.calibration_dataset, config,
                checkpoint_dir=folder / "checkpoints", run_fingerprint=task["fingerprint"],
                retain_success_checkpoint=True, checkpoint_identity_config=identity)
        base.experiments.write_result_files(folder, [result])
        base.write_json(folder / "metrics.json", base.metrics(result))
        base.experiments.finalize_config_checkpoint(folder / "checkpoints", identity, task["fingerprint"])
        attempt["status"] = "complete"
        print("COMPLETED " + task["task_id"], flush=True)
        return 0
    except BaseException as exc:
        kind = runtime.failure_class(exc, event)
        if isinstance(exc, KeyboardInterrupt):
            kind = "budget_or_interrupt"
        if "out of memory" in str(exc).lower() and ("cuda" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"):
            kind = "infrastructure_oom"
        failure = {"task_id": task["task_id"], "task_fingerprint": task["fingerprint"],
            "kind": kind, "exception": type(exc).__name__, "message": str(exc),
            "execution_context": event, "traceback": traceback.format_exc()}
        base.write_json(folder / ("failure_" + stamp + ".json"), failure)
        base.write_json(folder / "failure.json", failure)
        attempt.update(status="failed", kind=kind)
        print("FAILED " + json.dumps(failure, ensure_ascii=False), flush=True)
        return 75 if isinstance(exc, (runtime.BudgetPause, KeyboardInterrupt)) else 1
    finally:
        if cuda is not None:
            try:
                attempt["cuda_peak_allocated_mib"] = cuda.max_memory_allocated(args.devices[0]) / 2**20
            except Exception as exc:
                attempt["cuda_peak_measurement_error"] = str(exc)
        attempt.update(wall_seconds=time.monotonic() - started,
                       ended_at=datetime.now(timezone.utc).isoformat())
        base.write_json(path, attempt)


def execute(args, tasks):
    pending, finished = [], 0
    for task in tasks:
        if base.checked_completed(args.output, task) is not None:
            print("REUSE " + task["task_id"], flush=True)
            finished += 1
        elif (runtime.terminal_failure(args.output, task) or {}).get("kind") == "algorithm_numerical":
            print("RETAIN_NUMERICAL_FAILURE " + task["task_id"], flush=True)
            finished += 1
        else:
            pending.append(task)
    active, free, last_display = {}, list(args.devices), 0.
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        while pending or active:
            while pending and free:
                task, device = pending.pop(0), free.pop(0)
                folder = runtime.ensure_identity(args.output, task)
                command = [sys.executable, "-u", str(REPO / "run_cifar_timing_diagnostic.py"),
                    "--worker", task["task_id"], "--output", str(args.output), "--devices", device]
                if args.data_dir:
                    command += ["--data-dir", str(args.data_dir.resolve())]
                log = (folder / "worker.log").open("a", encoding="utf-8")
                try:
                    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True, cwd=REPO)
                except BaseException:
                    log.close()
                    raise
                active[proc.pid] = (proc, task, device, log)
                config = task["config"]
                print(f"START {task['task_id']} model=v7_cnn method={config['method']} "
                      f"K={config['detector_window']} attack_start={config['attack_start_round']} "
                      f"device={device}", flush=True)
            for pid, (proc, task, device, log) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                del active[pid]
                kind = (runtime.terminal_failure(args.output, task) or {}).get("kind") if code else None
                if code and kind != "algorithm_numerical":
                    print(f"DEVICE_PAUSED {device} kind={kind}; inspect before resume", flush=True)
                else:
                    free.append(device)
                finished += 1
                print(f"WORKER_EXIT {task['task_id']} code={code} kind={kind}", flush=True)
            if time.monotonic() - last_display >= 15:
                clean.progress(args.output, active, finished, len(pending), len(tasks))
                last_display = time.monotonic()
            if pending and not active and not free:
                print("EXECUTION_BLOCKED no usable lanes; checkpoints retained", flush=True)
                break
            if active:
                time.sleep(1)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        for proc, _, _, _ in active.values():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        end = time.monotonic() + 30
        for proc, _, _, log in active.values():
            try:
                proc.wait(timeout=max(.1, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            log.close()


def run_parent(args, spec, reference):
    separate_outputs(args.output, args.clean_output, args.matched_output)
    expected = protocol.build_manifest(spec, reference)
    args.output.mkdir(parents=True, exist_ok=True)
    with base.file_lock(args.output / ".runner.lock", nonblocking=True):
        path = args.output / "manifest.json"
        if not path.exists() and any(p.name != ".runner.lock" for p in args.output.iterdir()):
            raise ValueError("nonempty timing output has no matching manifest")
        base.immutable_json(path, expected)
        tasks = protocol.build_tasks(expected)
        runtime.save_plan(args.output, "timing", tasks, expected)
        protocol.read_study(args.output, current_sources=True)
        base.immutable_json(args.output / "execution_environment.json", reference["execution_environment"])
        for task in tasks:
            runtime.ensure_identity(args.output, task)
        try:
            execute(args, tasks)
        except KeyboardInterrupt:
            print("INTERRUPTED use the same command to resume", flush=True)
        from cifar_timing_report import summarize, print_summary
        report = summarize(args.output, args.clean_output, args.matched_output)
        base.write_json(args.output / "timing_summary.json", report)
        print_summary(report)
        return 0 if report["status"] == "complete" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=protocol.DEFAULT_CONFIG)
    parser.add_argument("--clean-output", type=Path, default=protocol.DEFAULT_CLEAN)
    parser.add_argument("--matched-output", type=Path, default=protocol.DEFAULT_MATCHED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--devices", nargs="+", default=["auto"])
    parser.add_argument("--max-gpus", type=int, default=6)
    parser.add_argument("--min-free-memory-mib", type=float, default=16384.)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Read only; audit all three studies without GPU/data/training")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.clean_output, args.matched_output = args.clean_output.resolve(), args.matched_output.resolve()
    if args.summary:
        from cifar_timing_report import summarize, print_summary
        args.output = (args.output or protocol.DEFAULT_OUTPUT).resolve()
        separate_outputs(args.output, args.clean_output, args.matched_output)
        report = summarize(args.output, args.clean_output, args.matched_output)
        print_summary(report)
        return 0 if report["status"] == "complete" else 2
    if args.worker:
        if args.output is None:
            parser.error("worker requires --output")
        args.output = args.output.resolve()
        return worker(args)
    spec = protocol.validate_spec(runtime.read_json(args.config))
    args.output = (args.output or REPO / spec["output_dir"]).resolve()
    separate_outputs(args.output, args.clean_output, args.matched_output)
    if args.plan_only:
        print(json.dumps({"protocol": protocol.PROTOCOL, "tasks": 26, "ours_tasks": 18,
            "fedavg_tasks": 8, "model": "v7_cnn", "lr": .05, "local_epochs": 1, "lr_decay": .99,
            "seeds": [2026093001], "partitions": ["iid", "dirichlet"], "rounds": 150,
            "ours_arms": [{"id": "A", "K": 10, "attack_start_round": 12},
                          {"id": "B", "K": 10, "attack_start_round": 25},
                          {"id": "C", "K": 20, "attack_start_round": 25}],
            "ours_malicious_ratios": [0., .1, .7], "fedavg_attack_start_rounds": [12, 25],
            "fedavg_malicious_ratios": [.1, .7], "evaluation_split": "calibration_dataset",
            "official_test_used_for_selection": False, "references_audited": False,
            "reference_audit_required_before_training": True, "output": str(args.output),
            "training_started": False, "next_stage_automatic": False}, indent=2))
        return 0
    if args.max_gpus < 1:
        parser.error("--max-gpus must be positive")
    if not math.isfinite(args.min_free_memory_mib) or args.min_free_memory_mib < 0:
        parser.error("--min-free-memory-mib must be finite and nonnegative")
    print("AUDITING_REFERENCES original 24+4 tasks read only; no repeated training", flush=True)
    reference = protocol.audit_reference(args.clean_output, args.matched_output)
    if (args.output / "manifest.json").exists():
        if runtime.read_json(args.output / "manifest.json") != protocol.build_manifest(spec, reference):
            raise ValueError("timing source/config/reference identity changed; preserve this output")
    from run_cifar_six_with_progress import discover_gpus, select_devices
    recorded = reference["execution_environment"].get("actual_compute_device")
    if not recorded:
        raise ValueError("reference has no recorded GPU identity")
    devices, skipped = select_devices(discover_gpus(REPO), args.devices, recorded=recorded,
                                     min_free_memory_mib=args.min_free_memory_mib)
    args.devices = devices[:min(args.max_gpus, 26)]
    print("DEVICES " + json.dumps({"selected": args.devices, "skipped": skipped,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}), flush=True)
    print("STAGE_2 CNN E1 only; 18 Ours + 8 FedAvg; no automatic later stages", flush=True)
    return run_parent(args, spec, reference)


if __name__ == "__main__":
    raise SystemExit(main())
