#!/usr/bin/env python3
"""Four fresh eighty-round clean H0/H1 horizon runs on the recorded physical GPU."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
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

import cifar_clean_horizon_protocol as protocol
import run_cifar_prefix_probe as original
import resume_cifar_prefix_probe as admission

base, runtime = protocol.base, protocol.runtime
REPO = Path(__file__).resolve().parent


def print_current_summary(output):
    from cifar_clean_horizon_report import summarize, print_summary
    value = summarize(output)
    print_summary(value)
    return 0 if value["status"] == "complete" else 2


def validate_artifact(output, task, manifest, artifact):
    """Validate complete scientific observations before publication, reuse or the next launch."""
    from cifar_clean_horizon_report import validate_observations, validate_environment_policy
    if (artifact.get("status") != "complete" or artifact.get("task_fingerprint") != task["fingerprint"]
            or artifact.get("gpu_uuid") != task["same_gpu_uuid"]
            or artifact.get("fresh_start") is not True or artifact.get("checkpoints_used") is not False):
        raise ValueError("completed mechanism run is not a fresh task on the frozen GPU")
    if protocol.prefix_report.matched.normalized_environment(artifact["environment"]) != manifest["reference"]["execution_environment"]:
        raise ValueError("completed mechanism run numerical environment differs")
    if not protocol.prefix_report.timing.finite(artifact.get("wall_seconds")) or artifact["wall_seconds"] < 0:
        raise ValueError("invalid completed mechanism run elapsed time")
    validate_observations(artifact["observations"], task)
    validate_environment_policy(artifact["observations"], artifact["environment"])
    rounds = artifact.get("completed_training_rounds")
    if (type(rounds) is not int or not 1 <= rounds <= task["config"]["rounds"]
            or type(artifact.get("requested_training_rounds")) is not int
            or artifact["requested_training_rounds"] != task["config"]["rounds"]):
        raise ValueError("mechanism execution round evidence differs from the requested horizon")
    observed_rounds = artifact["observations"].get("terminal", {}).get("stopped_round")
    if type(observed_rounds) is not int or observed_rounds != rounds:
        raise ValueError("artifact and mechanism observation round counts differ")
    return artifact


def validated_completed(output, task, manifest):
    artifact = protocol.load_completed(output, task)
    return None if artifact is None else validate_artifact(output, task, manifest, artifact)


def worker(args):
    manifest, tasks = protocol.read_study(args.output, current_sources=True)
    task = next((task for task in tasks if task["task_id"] == args.worker), None)
    if task is None or os.environ.get("CUDA_VISIBLE_DEVICES") != manifest["same_gpu_uuid"]:
        raise ValueError("worker identity or fixed physical GPU binding differs")
    folder = args.output / "tasks" / task["task_id"]
    with base.file_lock(folder / ".worker.lock", nonblocking=True):
        if validated_completed(args.output, task, manifest) is not None:
            print("REUSE " + task["task_id"], flush=True)
            return 0
        attempts = folder / "attempts"
        if attempts.exists() and any(attempts.iterdir()) and not args.retry_failed:
            raise ValueError("unsuccessful attempt retained; inspect before --retry-failed")
        attempt = attempts / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        attempt.mkdir(parents=True, exist_ok=False)
        base.immutable_json(attempt / "started.json", {"task_fingerprint": task["fingerprint"],
            "gpu_uuid": task["same_gpu_uuid"], "fresh_start": True})
        started = time.monotonic()
        try:
            metadata = base.worker_environment("cuda:0")
            import torch
            from cifar_mechanism_runtime import observe
            from cifar_deterministic_prefix_runtime import source_evidence
            from cifar_cnn_history_runtime import history_runtime
            if torch.cuda.device_count() != 1:
                raise ValueError("worker must see exactly one physical GPU")
            original.protocol.validate_gpu_environment(metadata, task["same_gpu_uuid"])
            if protocol.prefix_report.matched.normalized_environment(metadata) != manifest["reference"]["execution_environment"]:
                raise ValueError("worker numerical environment differs from original H0")
            base.immutable_json(attempt / "environment.json", metadata)
            implementation = source_evidence()
            base.immutable_json(attempt / "implementation_evidence.json", implementation)
            split, contract = base.load_split(manifest["spec"], args.data_dir)
            if contract != manifest["data_contract"]:
                raise ValueError("CIFAR data/split contract changed")
            config = replace(base.fl.ExperimentConfig(**task["config"]), device="cuda:0")
            print(f"CLEAN_HORIZON_CONTEXT clean Dirichlet; clients=100 rounds={config.rounds} "
                  f"variant={task['candidate']['variant']} policy={task['policy']}", flush=True)
            with history_runtime(task["candidate"]["variant"], freeze_start_round=task["history_freeze_start_round"]), observe(task) as observer:
                def checkpoint(state):
                    observer.checkpoint(state)
                    last = state["records"][-1]
                    print(f"CLEAN_HORIZON {task['task_id']} round={state['completed_round']}/{config.rounds} accuracy={last.accuracy:.6f}", flush=True)
                result = base.fl.run_experiment(split.calibration_dataset, config, checkpoint_callback=checkpoint)
            observations = observer.finish(result)
            if protocol.source_hashes() != manifest["source_sha256"]:
                raise ValueError("source changed while mechanism panel was running")
            protocol.verify_reference(manifest["reference"])
            payload = protocol.seal_artifact({"status": "complete", "task_fingerprint": task["fingerprint"],
                "gpu_uuid": task["same_gpu_uuid"], "environment": metadata, "observations": observations,
                "implementation_evidence": implementation,
                "wall_seconds": time.monotonic() - started, "attempt": attempt.name,
                "fresh_start": True, "checkpoints_used": False,
                "completed_training_rounds": result.stopped_round, "requested_training_rounds": config.rounds})
            validate_artifact(args.output, task, manifest, payload)
            base.immutable_json(attempt / "completed.json", payload)
            base.immutable_json(folder / "completed.json", payload)
            print(f"COMPLETED_CLEAN_HORIZON {task['task_id']} completed_training_rounds={result.stopped_round} requested_training_rounds={config.rounds}", flush=True)
            return 0
        except BaseException as exc:
            base.immutable_json(attempt / "failure.json", {"status": "failed_clean_horizon_panel",
                "task_fingerprint": task["fingerprint"], "exception": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(), "wall_seconds": time.monotonic() - started,
                "algorithm_health_assessed": False})
            raise


def execute(args, manifest, tasks):
    for task in tasks:
        folder = args.output / "tasks" / task["task_id"]
        if validated_completed(args.output, task, manifest) is not None:
            print("REUSE " + task["task_id"], flush=True)
            continue
        attempts = folder / "attempts"
        if attempts.exists() and any(attempts.iterdir()) and not args.retry_failed:
            raise ValueError("unsuccessful attempt retained; inspect before --retry-failed: " + task["task_id"])
        original.select_gpu(original.gpu_inventory(), manifest["same_gpu_uuid"],
            manifest["reference"]["execution_environment"]["actual_compute_device"]["name"],
            admission.MIN_FREE_MIB, mask=os.environ.get("CUDA_VISIBLE_DEVICES"))
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", task["task_id"],
                   "--output", str(args.output)]
        if args.data_dir:
            command += ["--data-dir", str(args.data_dir.resolve())]
        if args.retry_failed:
            command.append("--retry-failed")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=manifest["same_gpu_uuid"])
        print(f"START {task['task_id']} gpu_uuid={manifest['same_gpu_uuid']} clients=100 rounds={task['config']['rounds']} variant={task['candidate']['variant']} policy={task['policy']} fresh_process=true", flush=True)
        with (folder / "worker.log").open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
                                    env=environment, start_new_session=True)
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
            if proc.returncode or validated_completed(args.output, task, manifest) is None:
                print("CLEAN_HORIZON_BLOCKED inspect " + str(folder / "worker.log"), flush=True)
                return False
    return True


def run_parent(args):
    admission.validate_wait(args.wait_seconds, args.poll_seconds)
    protocol.separate_outputs(args.output, args.mechanism_output)
    if (args.output / "manifest.json").exists():
        manifest, tasks = protocol.read_study(args.output, current_sources=True)
        if manifest["reference"]["mechanism_output"] != str(args.mechanism_output.resolve()):
            raise ValueError("thirty-round mechanism reference path differs from frozen study")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("nonempty output has no valid mechanism-panel identity")
        print("VERIFYING_REFERENCE four thirty-round clean mechanism runs and the complete 115-task reference chain; read only", flush=True)
        reference = protocol.audit_reference(args.mechanism_output)
        protocol.separate_outputs(args.output, args.mechanism_output, protocol.reference_paths(reference))
        manifest = protocol.build_manifest(reference)
        tasks = protocol.build_tasks(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    with base.file_lock(args.output / ".runner.lock", nonblocking=True):
        base.immutable_json(args.output / "manifest.json", manifest)
        (args.output / "task_plans").mkdir(exist_ok=True)
        base.immutable_json(args.output / "task_plans/clean_horizon.json", {"manifest_fingerprint": manifest["fingerprint"], "tasks": tasks})
        for task in tasks:
            (args.output / "tasks" / task["task_id"]).mkdir(parents=True, exist_ok=True)
            base.immutable_json(args.output / "tasks" / task["task_id"] / "task.json", task)
        controls = args.output / "controller_runs"
        controls.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        with (controls / (stamp + ".jsonl")).open("x", encoding="utf-8") as control:
            def emit(value):
                line = json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), **value}, separators=(",", ":"))
                print("CLEAN_HORIZON_CONTROL " + line, flush=True)
                control.write(line + "\n")
                control.flush()
            emit({"event": "start", "manifest_fingerprint": manifest["fingerprint"],
                "same_gpu_uuid": manifest["same_gpu_uuid"], "wait_seconds": args.wait_seconds,
                "poll_seconds": args.poll_seconds, "retry_failed_requested": args.retry_failed})
            previous = signal.getsignal(signal.SIGTERM)
            def stop(*_):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, stop)
            try:
                with admission.waiting_gate(manifest["same_gpu_uuid"], wait_seconds=args.wait_seconds,
                        poll_seconds=args.poll_seconds, emit=emit):
                    execute(args, manifest, tasks)
            except (Exception, KeyboardInterrupt) as exc:
                emit({"event": "blocked", "exception": type(exc).__name__, "error": str(exc),
                      "completed_tasks_retained": True, "algorithm_health_assessed": False})
            finally:
                signal.signal(signal.SIGTERM, previous)
        return print_current_summary(args.output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=protocol.DEFAULT_OUTPUT)
    parser.add_argument("--mechanism-output", type=Path, default=protocol.DEFAULT_MECHANISM)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--wait-seconds", type=float, default=600.)
    parser.add_argument("--poll-seconds", type=float, default=10.)
    parser.add_argument("--retry-failed", action="store_true", help="only after inspecting retained failures; fresh attempt, no checkpoint reuse")
    parser.add_argument("--summary", action="store_true", help="read-only; no GPU, data or training")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output, args.mechanism_output = args.output.resolve(), args.mechanism_output.resolve()
    if sum(bool(x) for x in (args.summary, args.plan_only, args.worker)) > 1:
        parser.error("summary, plan-only and worker cannot be combined")
    if args.plan_only:
        protocol.separate_outputs(args.output, args.mechanism_output)
        print(json.dumps({"protocol": protocol.PROTOCOL, "fresh_processes": 4,
            "clients_per_round": 100, "rounds_per_process": 80, "maximum_training_rounds": 320,
            "maximum_client_training_calls": 32000, "partition": "dirichlet", "malicious_ratio": 0.,
            "numerical_policy_modified": True,
            "policies": ["singleton_backward_cudnn_deterministic"],
            "variants": ["H0", "H1"], "history_freeze_start_round": 25,
            "order": "H0repeat1,H1repeat1,H0repeat2,H1repeat2",
            "modified_scope": "all actual single-sample minibatch backward calls; all are final in this frozen clean partition",
            "maximum_scoped_events_per_process": 160, "same_original_gpu_uuid": True,
            "call_count_limit": "revoked clients stop contributing; all-honest revocation may stop execution early",
            "training_started": False, "health_assessed": False, "automatic_next_stage": False}, indent=2))
        return 0
    if args.summary:
        return print_current_summary(args.output)
    if args.worker:
        return worker(args)
    try:
        return run_parent(args)
    except (Exception, KeyboardInterrupt) as exc:
        print("CLEAN_HORIZON_REFUSED " + json.dumps({"exception": type(exc).__name__, "error": str(exc),
            "task_state": "inspect current summary; original evidence retained"}), flush=True)
        print_current_summary(args.output)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
