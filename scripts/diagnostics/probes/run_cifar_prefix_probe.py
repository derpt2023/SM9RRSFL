#!/usr/bin/env python3
"""Six fresh, serial original-Ours prefixes on one physical GPU, three rounds each."""
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

import cifar_prefix_probe_protocol as protocol

base, runtime = protocol.base, protocol.runtime
REPO = Path(__file__).resolve().parent


def gpu_inventory():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True, timeout=10)
    rows = []
    for line in result.stdout.splitlines():
        index, uuid, name, free, utilization = [part.strip() for part in line.split(",")]
        row = {"index": int(index), "uuid": uuid, "name": name,
               "free_mib": float(free), "utilization": float(utilization)}
        if not uuid.startswith("GPU-") or not all(math.isfinite(row[k]) for k in ("free_mib", "utilization")):
            raise ValueError("invalid nvidia-smi identity or capacity data")
        rows.append(row)
    if len({r["uuid"] for r in rows}) != len(rows):
        raise ValueError("duplicate NVIDIA GPU UUIDs")
    return rows


def select_gpu(rows, requested, expected_name, min_free, *, mask=None):
    if not math.isfinite(min_free) or min_free < 0:
        raise ValueError("minimum GPU memory must be finite and nonnegative")
    allowed = None
    if mask is not None:
        tokens = mask.split(",")
        if not tokens or any(not token.startswith("GPU-") for token in tokens):
            raise ValueError("prefix probe requires unset CUDA_VISIBLE_DEVICES or full GPU UUIDs, not ambiguous numeric masks")
        allowed = set(tokens)
    eligible = [r for r in rows if (allowed is None or r["uuid"] in allowed)
        and r["name"] == expected_name and r["free_mib"] >= min_free and r["utilization"] <= 5]
    if requested != "auto":
        eligible = [r for r in eligible if r["uuid"] == requested]
    if not eligible:
        raise ValueError("no idle compatible GPU has enough free memory; no training started")
    return sorted(eligible, key=lambda r: (-r["free_mib"], r["index"]))[0]


def worker(args):
    manifest, tasks = protocol.read_study(args.output, current_sources=True, verify_reference=False)
    task = next((t for t in tasks if t["task_id"] == args.worker), None)
    if task is None or os.environ.get("CUDA_VISIBLE_DEVICES") != manifest["same_gpu_uuid"]:
        raise ValueError("worker task or physical GPU binding differs from the immutable plan")
    folder = args.output / "tasks" / task["task_id"]
    with base.file_lock(folder / ".worker.lock", nonblocking=True):
        if protocol.load_completed(args.output, task) is not None:
            print("REUSE " + task["task_id"], flush=True)
            return 0
        attempts = folder / "attempts"
        if attempts.exists() and any(attempts.iterdir()) and not args.retry_failed:
            raise ValueError("existing unsuccessful attempt: inspect it before explicitly using --retry-failed")
        attempt = attempts / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        attempt.mkdir(parents=True, exist_ok=False)
        base.immutable_json(attempt / "started.json", {"task_fingerprint": task["fingerprint"],
            "gpu_uuid": task["same_gpu_uuid"], "fresh_start": True})
        started = time.monotonic()
        try:
            metadata = base.worker_environment("cuda:0")
            import torch
            if torch.cuda.device_count() != 1:
                raise ValueError("worker must see exactly one physical GPU")
            protocol.validate_gpu_environment(metadata, task["same_gpu_uuid"])
            if protocol.history.threshold.timing.matched.normalized_environment(metadata) != manifest["reference"]["execution_environment"]:
                raise ValueError("worker numerical environment differs from the original reviewed H0")
            base.immutable_json(attempt / "environment.json", metadata)
            split, contract = base.load_split(manifest["spec"], args.data_dir)
            if contract != manifest["data_contract"]:
                raise ValueError("CIFAR data/split contract changed")
            from cifar_prefix_probe_runtime import observe
            config = replace(base.fl.ExperimentConfig(**task["config"]), device="cuda:0")
            # No old checkpoint or model is loaded. The callback reads in-memory
            # round state only; all SM9 signing and verification stay real.
            with observe(task) as observer:
                def checkpoint(state):
                    observer.checkpoint(state)
                    last = state["records"][-1]
                    print(f"PREFIX {task['task_id']} round={state['completed_round']}/3 accuracy={last.accuracy:.6f}", flush=True)
                result = base.fl.run_experiment(split.calibration_dataset, config, checkpoint_callback=checkpoint)
                observations = observer.finish(result)
            if protocol.source_hashes() != manifest["source_sha256"]:
                raise ValueError("source changed while prefix was running")
            payload = protocol.seal_artifact({"status": "complete", "task_fingerprint": task["fingerprint"],
                "gpu_uuid": task["same_gpu_uuid"], "environment": metadata,
                "observations": observations, "wall_seconds": time.monotonic() - started,
                "attempt": attempt.name, "fresh_start": True, "checkpoints_used": False})
            base.immutable_json(attempt / "completed.json", payload)
            base.immutable_json(folder / "completed.json", payload)
            print("COMPLETED " + task["task_id"], flush=True)
            return 0
        except BaseException as exc:
            base.immutable_json(attempt / "failure.json", {"status": "failed_prefix", "task_fingerprint": task["fingerprint"],
                "exception": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
                "wall_seconds": time.monotonic() - started, "algorithm_health_assessed": False})
            raise


def execute(args, manifest, tasks):
    """Wait for each fresh worker before spawning the next; UUID never changes."""
    for task in tasks:
        folder = args.output / "tasks" / task["task_id"]
        if protocol.load_completed(args.output, task) is not None:
            print("REUSE " + task["task_id"], flush=True)
            continue
        attempts = folder / "attempts"
        if attempts.exists() and any(attempts.iterdir()) and not args.retry_failed:
            raise ValueError("unsuccessful prefix retained; inspect before --retry-failed: " + task["task_id"])
        # Recheck the same pinned card before every task; never silently migrate.
        select_gpu(gpu_inventory(), manifest["same_gpu_uuid"],
            manifest["reference"]["execution_environment"]["actual_compute_device"]["name"],
            args.min_free_memory_mib, mask=os.environ.get("CUDA_VISIBLE_DEVICES"))
        command = [sys.executable, "-u", str(REPO / "run_cifar_prefix_probe.py"),
                   "--worker", task["task_id"], "--output", str(args.output)]
        if args.data_dir:
            command += ["--data-dir", str(args.data_dir.resolve())]
        if args.retry_failed:
            command.append("--retry-failed")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=manifest["same_gpu_uuid"])
        print(f"START {task['task_id']} gpu_uuid={manifest['same_gpu_uuid']} rounds=3 fresh_process=true", flush=True)
        with (folder / "worker.log").open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    cwd=REPO, env=environment, start_new_session=True)
            try:
                last, started = 0., time.monotonic()
                while proc.poll() is None:
                    if time.monotonic() - last >= 15:
                        print(f"PROGRESS active={task['task_id']} elapsed_seconds={int(time.monotonic() - started)} log={folder / 'worker.log'}", flush=True)
                        last = time.monotonic()
                    time.sleep(1)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
            print(f"WORKER_EXIT {task['task_id']} code={proc.returncode}", flush=True)
            if proc.returncode or protocol.load_completed(args.output, task) is None:
                print("PREFIX_BLOCKED inspect " + str(folder / "worker.log"), flush=True)
                return False
    return True


def run_parent(args, paths):
    protocol.separate_outputs(args.output, paths)
    if (args.output / "manifest.json").exists():
        manifest, tasks = protocol.read_study(args.output, current_sources=True)
        if {k: str(Path(v).resolve()) for k, v in paths.items()} != manifest["reference"]["paths"]:
            raise ValueError("reference locations differ from the frozen probe")
        if args.gpu not in ("auto", manifest["same_gpu_uuid"]):
            raise ValueError("resume cannot migrate to a different physical GPU")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("nonempty probe output has no valid identity")
        print("VERIFYING_REFERENCE 90 completed tasks; read only", flush=True)
        reference = protocol.audit_reference(paths)
        selected = select_gpu(gpu_inventory(), args.gpu,
            reference["execution_environment"]["actual_compute_device"]["name"], args.min_free_memory_mib,
            mask=os.environ.get("CUDA_VISIBLE_DEVICES"))
        manifest = protocol.build_manifest(reference, selected["uuid"])
        tasks = protocol.build_tasks(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    with base.file_lock(args.output / ".runner.lock", nonblocking=True):
        base.immutable_json(args.output / "manifest.json", manifest)
        (args.output / "task_plans").mkdir(exist_ok=True)
        base.immutable_json(args.output / "task_plans/prefix.json", {"manifest_fingerprint": manifest["fingerprint"], "tasks": tasks})
        for task in tasks:
            (args.output / "tasks" / task["task_id"]).mkdir(parents=True, exist_ok=True)
            base.immutable_json(args.output / "tasks" / task["task_id"] / "task.json", task)
        previous = signal.getsignal(signal.SIGTERM)
        def stop(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        try:
            execute(args, manifest, tasks)
        except KeyboardInterrupt:
            print("INTERRUPTED attempt retained; inspect before explicit retry", flush=True)
        finally:
            signal.signal(signal.SIGTERM, previous)
        from cifar_prefix_probe_report import summarize, print_summary
        report = summarize(args.output)
        print_summary(report)
        return 0 if report["status"] == "complete" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=protocol.DEFAULT_OUTPUT)
    parser.add_argument("--history-output", type=Path, default=protocol.history.DEFAULT_OUTPUT)
    for name in ("threshold", "timing", "clean", "matched"):
        parser.add_argument("--" + name + "-output", type=Path, default=getattr(protocol.history, "DEFAULT_" + name.upper()))
    parser.add_argument("--gpu", default="auto", help="auto or one full NVIDIA GPU UUID (not a logical CUDA index)")
    parser.add_argument("--min-free-memory-mib", type=float, default=16384.)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--summary", action="store_true", help="read-only; no GPU/data loading")
    parser.add_argument("--retry-failed", action="store_true", help="explicitly create new attempts; preserve previous failures")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    paths = {k: getattr(args, k + "_output").resolve() for k in protocol.REFERENCE_NAMES}
    if args.plan_only and (args.summary or args.worker):
        parser.error("plan-only cannot be combined with summary or worker")
    if args.plan_only:
        protocol.separate_outputs(args.output, paths)
        print(json.dumps({"protocol": protocol.PROTOCOL, "tasks": 6, "rounds_each": 3,
            "total_training_rounds": 18, "repeats_each_partition": 3, "partitions": ["iid", "dirichlet"],
            "original_ours": True, "malicious_ratio": 0., "seed": 2026093001,
            "single_gpu_serial": True, "numerical_policy_modified": False,
            "training_started": False, "next_stage_automatic": False}, indent=2))
        return 0
    if args.summary:
        from cifar_prefix_probe_report import summarize, print_summary
        report = summarize(args.output)
        print_summary(report)
        return 0 if report["status"] == "complete" else 2
    if args.worker:
        return worker(args)
    return run_parent(args, paths)


if __name__ == "__main__":
    raise SystemExit(main())
