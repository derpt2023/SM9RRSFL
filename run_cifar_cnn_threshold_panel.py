#!/usr/bin/env python3
"""Stage 3: four fixed Ours threshold points, 24 independent CNN E1 runs.

P0 is a new six-task repeat of timing condition C. This entry never starts TPE,
formal experiments, or subsequent panels. All earlier studies stay read-only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import run_cifar_timing_diagnostic as timing
import cifar_cnn_threshold_protocol as protocol

base, runtime = timing.base, timing.runtime
REPO = Path(__file__).resolve().parent


def separate_outputs(output, timing_output, clean_output, matched_output):
    """All four studies must be distinct and pairwise non-nested, including links."""
    paths = [Path(p).resolve() for p in (output, timing_output, clean_output, matched_output)]
    for i, left in enumerate(paths):
        for right in paths[i + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError("threshold output and all three references must be separate sibling studies")


def worker(args):
    manifest, tasks = protocol.read_study(args.output, current_sources=True)
    matches = [task for task in tasks if task["task_id"] == args.worker]
    if len(matches) != 1 or len(args.devices) != 1 or args.devices == ["auto"]:
        raise ValueError("worker requires one declared threshold task and one explicit device")
    if runtime.read_json(args.output / "execution_environment.json") != manifest["reference"]["execution_environment"]:
        raise ValueError("threshold worker environment differs from the frozen reference")
    task = matches[0]
    if task["model"] != "v7_cnn" or task["phase"] != "validation" or task["config"]["method"] != "sm9rrs":
        raise ValueError("threshold workers require original CNN Ours on the calibration split")
    folder = args.output / "tasks" / task["task_id"]
    if runtime.read_json(folder / "task.json") != task:
        raise ValueError("threshold worker task identity mismatch")
    with base.file_lock(folder / ".worker.lock", nonblocking=True):
        # Preserve the frozen CNN training, calibration evaluation, environment
        # checks, per-round checkpoint identity, cost and failure implementation.
        return timing.run_task(args, manifest, task, folder)


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
                command = [sys.executable, "-u", str(REPO / "run_cifar_cnn_threshold_panel.py"),
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
                print(f"START {task['task_id']} point={task['candidate']['candidate_id']} "
                      f"warning={config['detector_distance_threshold']} "
                      f"kappa={config['detector_drift_allowance']} h={config['detector_drift_threshold']} "
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
                timing.clean.progress(args.output, active, finished, len(pending), len(tasks))
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
    separate_outputs(args.output, args.timing_output, args.clean_output, args.matched_output)
    expected = protocol.build_manifest(spec, reference)
    args.output.mkdir(parents=True, exist_ok=True)
    with base.file_lock(args.output / ".runner.lock", nonblocking=True):
        path = args.output / "manifest.json"
        if not path.exists() and any(p.name != ".runner.lock" for p in args.output.iterdir()):
            raise ValueError("nonempty threshold output has no matching manifest")
        base.immutable_json(path, expected)
        tasks = protocol.build_tasks(expected)
        runtime.save_plan(args.output, "threshold", tasks, expected)
        protocol.read_study(args.output, current_sources=True)
        base.immutable_json(args.output / "execution_environment.json", reference["execution_environment"])
        for task in tasks:
            runtime.ensure_identity(args.output, task)
        try:
            execute(args, tasks)
        except KeyboardInterrupt:
            print("INTERRUPTED use the same command to resume", flush=True)
        from cifar_cnn_threshold_report import summarize, print_summary
        report = summarize(args.output, args.timing_output, args.clean_output, args.matched_output)
        base.write_json(args.output / "threshold_summary.json", report)
        print_summary(report)
        return 0 if report["status"] == "complete" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=protocol.DEFAULT_CONFIG)
    parser.add_argument("--timing-output", type=Path, default=protocol.DEFAULT_TIMING)
    parser.add_argument("--clean-output", type=Path, default=protocol.DEFAULT_CLEAN)
    parser.add_argument("--matched-output", type=Path, default=protocol.DEFAULT_MATCHED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--devices", nargs="+", default=["auto"])
    parser.add_argument("--max-gpus", type=int, default=6)
    parser.add_argument("--min-free-memory-mib", type=float, default=16384.)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Read-only compact report; no GPU/data/training")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.timing_output = args.timing_output.resolve()
    args.clean_output, args.matched_output = args.clean_output.resolve(), args.matched_output.resolve()
    if args.summary:
        from cifar_cnn_threshold_report import summarize, print_summary
        args.output = (args.output or protocol.DEFAULT_OUTPUT).resolve()
        separate_outputs(args.output, args.timing_output, args.clean_output, args.matched_output)
        report = summarize(args.output, args.timing_output, args.clean_output, args.matched_output)
        print_summary(report)
        return 0 if report["status"] == "complete" else 2
    if args.worker:
        if args.output is None:
            parser.error("worker requires --output")
        args.output = args.output.resolve()
        return worker(args)
    spec = protocol.validate_spec(runtime.read_json(args.config))
    args.output = (args.output or REPO / spec["output_dir"]).resolve()
    separate_outputs(args.output, args.timing_output, args.clean_output, args.matched_output)
    if args.plan_only:
        print(json.dumps({"protocol": protocol.PROTOCOL, "tasks": 24, "method": "sm9rrs",
            "fixed_points": [{"id": name, "warning": warning, "kappa": kappa, "h": threshold}
                             for name, warning, kappa, threshold in (
                                 ("P0", 1.25, 1.25, 6.), ("P1", 1.5, 1.25, 6.),
                                 ("P2", 1.5, .85, 1.5), ("P3", 1.75, 1., 2.))],
            "model": "v7_cnn", "lr": .05, "local_epochs": 1, "lr_decay": .99,
            "K": 20, "attack_start_round": 25, "rounds": 150, "seed": 2026093001,
            "partitions": ["iid", "dirichlet"], "malicious_ratios": [0., .1, .7],
            "P0_is_six_new_runs": True, "prior_C_is_read_only_repeat_reference": True,
            "search_algorithm": "none_fixed_panel", "tpe_trials": 0, "next_stage_automatic": False,
            "evaluation_split": "calibration_dataset", "official_test_used_for_selection": False,
            "references_audited": False, "reference_audit_required_before_training": True,
            "output": str(args.output), "training_started": False}, indent=2))
        return 0
    if args.max_gpus < 1:
        parser.error("--max-gpus must be positive")
    if not math.isfinite(args.min_free_memory_mib) or args.min_free_memory_mib < 0:
        parser.error("--min-free-memory-mib must be finite and nonnegative")
    print("AUDITING_REFERENCES original 24+4+26 tasks read only; P0 is six new runs", flush=True)
    reference = protocol.audit_reference(args.timing_output, args.clean_output, args.matched_output)
    if (args.output / "manifest.json").exists():
        if runtime.read_json(args.output / "manifest.json") != protocol.build_manifest(spec, reference):
            raise ValueError("threshold source/config/reference identity changed; preserve this output")
    from run_cifar_six_with_progress import discover_gpus, select_devices
    recorded = reference["execution_environment"].get("actual_compute_device")
    if not recorded:
        raise ValueError("reference has no recorded GPU identity")
    devices, skipped = select_devices(discover_gpus(REPO), args.devices, recorded=recorded,
                                     min_free_memory_mib=args.min_free_memory_mib)
    args.devices = devices[:min(args.max_gpus, 24)]
    print("DEVICES " + json.dumps({"selected": args.devices, "skipped": skipped,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}), flush=True)
    print("STAGE_3 four fixed threshold points; 24 new Ours tasks; no TPE or automatic later stages", flush=True)
    return run_parent(args, spec, reference)


if __name__ == "__main__":
    raise SystemExit(main())
