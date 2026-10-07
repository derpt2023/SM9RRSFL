#!/usr/bin/env python3
"""Stage 1 only: 24 clean FedAvg architecture/learning-setting diagnostics."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
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

import cifar_adaptive_runtime as runtime
import cifar_resnet_gn as resnet
from cifar_diagnostic_observer import observe

base = runtime.base
REPO = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO / "configs/cifar10_clean_diagnostic_v1.json"
PROTOCOL = "cifar-v8-clean-diagnostic-v1"
SETTINGS = [
    {"id": name, "model": model, "lr": lr, "local_epochs": epochs, "lr_decay": decay}
    for name, model, lr, epochs, decay in (
        ("C0", "v7_cnn", .05, 1, .99), ("R0", "resnet18_gn2", .05, 1, .99),
        ("R1", "resnet18_gn2", .02, 1, .99), ("R2", "resnet18_gn2", .10, 1, .99),
        ("R3", "resnet18_gn2", .05, 2, .99), ("R4", "resnet18_gn2", .05, 1, .995))]
NEW_SOURCES = ("run_cifar_diagnostic.py", "cifar_diagnostic_observer.py",
               "cifar_diagnostic_report.py", "run_cifar_six_with_progress.py")


def validate_spec(spec):
    if (spec.get("protocol") != PROTOCOL or spec.get("schema_version") != 1
            or spec.get("settings") != SETTINGS
            or spec.get("seeds") != [2026093001, 2026093002]
            or spec.get("partitions") != ["iid", "dirichlet"]):
        raise ValueError("stage 1 requires the fixed six-setting, two-seed, two-partition matrix")
    shared = spec["shared_parameters"]
    config = base.fl.ExperimentConfig(**shared)
    config.validate()
    if set(shared) != set(asdict(config)):
        raise ValueError("all ExperimentConfig fields must be explicit")
    fixed = dict(method="fedavg", malicious_ratio=0., num_clients=100, rounds=150,
        early_stop=False, compute_backend="torch", crypto_mode="sm9", batch_size=50,
        lr=.05, local_epochs=1, lr_decay=.99, eval_interval=1, checkpoint_interval=1,
        dirichlet_alpha=.5, attack="alternating_minimization", attack_start_round=12,
        detector_window=10, attack_boost=5., attack_epochs=1, attack_stealth_steps=1,
        attack_distance_weight=.0001, attack_source_label=5, attack_target_label=7,
        attack_target_count=200)
    if any(shared[k] != value for k, value in fixed.items()):
        raise ValueError("stage 1 must retain the declared clean 150-round public protocol")
    data = spec["dataset"]
    if any(data.get(k) != v for k, v in dict(name="cifar10", train_samples=50000,
            test_samples=10000, split_seed=20260917, validation_fraction=.05).items()):
        raise ValueError("stage 1 uses the existing CIFAR 45k/2500/2500 split")
    return spec


def source_hashes():
    hashes = runtime.source_hashes()
    for name in NEW_SOURCES:
        hashes[name] = hashlib.sha256((REPO / name).read_bytes()).hexdigest()
    return hashes


def build_manifest(spec, contract):
    payload = {"protocol": PROTOCOL, "spec": spec, "data_contract": contract,
        "source_sha256": source_hashes(), "evaluation_split": "calibration_dataset",
        "purpose": "development_panel", "official_test_used_for_selection": False,
        "model_contracts": {"v7_cnn": {"architecture": "original_cifar_cnn", "parameter_count": 1756426},
                            "resnet18_gn2": resnet.protocol_descriptor()},
        "loss_contract": "sample-weighted client minibatch training loss; calibration global-model CE",
        "next_stage_automatic": False}
    return {**payload, "fingerprint": base.digest(payload)}


def build_tasks(spec, manifest):
    tasks = []
    for setting in spec["settings"]:
        for seed in spec["seeds"]:
            for partition in spec["partitions"]:
                params = {k: setting[k] for k in ("lr", "local_epochs", "lr_decay")}
                config = base.fl.ExperimentConfig(**{**spec["shared_parameters"], **params,
                                                      "seed": seed, "partition": partition})
                config.validate()
                tasks.append({"task_id": f"clean_{setting['id']}_{partition}_seed{seed}",
                    "phase": "validation", "purpose": "development_panel", "method": "fedavg",
                    "model": setting["model"], "candidate": {"candidate_id": setting["id"],
                    "variant": "original", "parameters": params}, "config": asdict(config)})
    return base.attach_fingerprints(tasks, manifest)


def read_study(output, *, current_sources=False):
    manifest = runtime.read_json(output / "manifest.json")
    payload = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if base.digest(payload) != manifest["fingerprint"] or manifest["protocol"] != PROTOCOL:
        raise ValueError("diagnostic manifest fingerprint/protocol mismatch")
    validate_spec(manifest["spec"])
    if current_sources and manifest != build_manifest(manifest["spec"], manifest["data_contract"]):
        raise ValueError("diagnostic source/model identity changed; preserve this output")
    tasks = build_tasks(manifest["spec"], manifest)
    plan = runtime.read_json(output / "task_plans" / "clean.json")
    if plan != {"manifest_fingerprint": manifest["fingerprint"], "tasks": tasks}:
        raise ValueError("diagnostic task plan does not match its manifest")
    return manifest, tasks


def model_runtime(name):
    if name == "resnet18_gn2":
        return resnet.runtime()
    if name == "v7_cnn":
        if resnet.runtime_installed():
            raise ValueError("CNN worker must not inherit an active ResNet adapter")
        return nullcontext()
    raise ValueError("unknown diagnostic model")


def worker(args):
    output = args.output.resolve()
    manifest, tasks = read_study(output, current_sources=True)
    matches = [t for t in tasks if t["task_id"] == args.worker]
    if len(matches) != 1 or len(args.devices) != 1 or args.devices == ["auto"]:
        raise ValueError("worker requires a declared task and one explicit device")
    task = matches[0]
    folder = output / "tasks" / task["task_id"]
    if runtime.read_json(folder / "task.json") != task:
        raise ValueError("worker task identity mismatch")
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
    event = {}
    try:
        metadata = base.worker_environment(args.devices[0])
        base.check_environment(args.output, metadata)
        base.write_json(folder / "environment.json", metadata)
        import torch
        torch.cuda.reset_peak_memory_stats()
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
        with model_runtime(task["model"]), runtime.round_monitor(task, folder) as event, observe(task, folder):
            result = base.experiments.run_measured_experiment(split.calibration_dataset, config,
                checkpoint_dir=folder / "checkpoints", run_fingerprint=task["fingerprint"],
                retain_success_checkpoint=True, checkpoint_identity_config=identity)
        base.experiments.write_result_files(folder, [result])
        base.write_json(folder / "metrics.json", base.metrics(result))
        base.experiments.finalize_config_checkpoint(folder / "checkpoints", identity, task["fingerprint"])
        attempt["status"] = "complete"
        attempt["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        print("COMPLETED " + task["task_id"], flush=True)
        return 0
    except BaseException as exc:
        kind = runtime.failure_class(exc, event)
        if "out of memory" in str(exc).lower() and ("cuda" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"):
            kind = "infrastructure_oom"
        failure = {"task_id": task["task_id"], "task_fingerprint": task["fingerprint"],
            "kind": kind, "exception": type(exc).__name__, "message": str(exc),
            "execution_context": event, "traceback": traceback.format_exc()}
        base.write_json(folder / ("failure_" + stamp + ".json"), failure)
        base.write_json(folder / "failure.json", failure)
        attempt.update(status="failed", kind=kind)
        print("FAILED " + json.dumps(failure, ensure_ascii=False), flush=True)
        return 75 if isinstance(exc, runtime.BudgetPause) else 1
    finally:
        attempt.update(wall_seconds=time.monotonic() - started,
                       ended_at=datetime.now(timezone.utc).isoformat())
        base.write_json(path, attempt)


def progress(output, active, finished, queued, total):
    print(f"PROGRESS settled={finished}/{total} active={len(active)} queued={queued}", flush=True)
    for proc, task, device, _ in active.values():
        path = output / "tasks" / task["task_id"] / "progress.json"
        row = runtime.read_json(path) if path.exists() else {}
        rd = row.get("last_completed_round", 0)
        print(f"  {device} {task['task_id']} round={rd}/150 pid={proc.pid}", flush=True)


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
    free, active = list(args.devices), {}
    last_display = 0.
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        while pending or active:
            while pending and free:
                task, device = pending.pop(0), free.pop(0)
                folder = runtime.ensure_identity(args.output, task)
                command = [sys.executable, "-u", str(REPO / "run_cifar_diagnostic.py"),
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
                print(f"START {task['task_id']} model={task['model']} device={device}", flush=True)
            for pid, (proc, task, device, log) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                del active[pid]
                kind = (runtime.terminal_failure(args.output, task) or {}).get("kind") if code else None
                # Do not keep assigning work to a device that just failed operationally.
                if code and kind != "algorithm_numerical":
                    print(f"DEVICE_PAUSED {device} kind={kind}; resume after inspection", flush=True)
                else:
                    free.append(device)
                finished += 1
                print(f"WORKER_EXIT {task['task_id']} code={code} kind={kind}", flush=True)
            if time.monotonic() - last_display >= 15:
                progress(args.output, active, finished, len(pending), len(tasks))
                last_display = time.monotonic()
            if pending and not active and not free:
                print("EXECUTION_BLOCKED no usable lanes; outputs/checkpoints retained", flush=True)
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


def run_parent(args, spec):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with base.file_lock(output / ".runner.lock", nonblocking=True):
        path = output / "manifest.json"
        if path.exists():
            saved = runtime.read_json(path)
            if saved != build_manifest(spec, saved["data_contract"]):
                raise ValueError("configuration/source identity changed; preserve this study")
            # Recover interruption between the two immutable initialization writes.
            runtime.save_plan(output, "clean", build_tasks(spec, saved), saved)
            manifest, tasks = read_study(output, current_sources=True)
            if spec != manifest["spec"]:
                raise ValueError("configuration changed; preserve this study")
        else:
            if any(p.name != ".runner.lock" for p in output.iterdir()):
                raise ValueError("nonempty output has no diagnostic manifest")
            print("PREPARING_DATA full CIFAR split/hash; training has not started", flush=True)
            split, contract = base.load_split(spec, args.data_dir)
            del split
            manifest = build_manifest(spec, contract)
            base.immutable_json(path, manifest)
            tasks = build_tasks(spec, manifest)
            runtime.save_plan(output, "clean", tasks, manifest)
        for task in tasks:
            runtime.ensure_identity(output, task)
        from cifar_diagnostic_report import summarize, print_summary
        try:
            execute(args, tasks)
        except KeyboardInterrupt:
            print("INTERRUPTED use the same command to resume", flush=True)
        report = summarize(output)
        base.write_json(output / "diagnostic_summary.json", report)
        print_summary(report)
        return 0 if report["status"] == "complete" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--devices", nargs="+", default=["auto"])
    parser.add_argument("--max-gpus", type=int, default=6)
    parser.add_argument("--min-free-memory-mib", type=float, default=16384.)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Read only; no data download, GPU probe, training or output writes")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.summary:
        from cifar_diagnostic_report import summarize, print_summary
        output = (args.output or REPO / "outputs/cifar_v8_diagnostic_v1/clean").resolve()
        report = summarize(output)
        print_summary(report)
        return 0 if report["status"] == "complete" else 2
    if args.worker:
        if args.output is None:
            parser.error("worker requires --output")
        args.output = args.output.resolve()
        return worker(args)
    spec = validate_spec(runtime.read_json(args.config))
    args.output = (args.output or REPO / spec["output_dir"]).resolve()
    if args.plan_only:
        print(json.dumps({"protocol": PROTOCOL, "settings": spec["settings"], "tasks": 24,
            "seeds": spec["seeds"], "partitions": spec["partitions"], "rounds": 150,
            "evaluation_split": "calibration_dataset", "official_test_used_for_selection": False,
            "training_started": False, "output": str(args.output)}, indent=2))
        return 0
    if args.max_gpus < 1:
        parser.error("--max-gpus must be positive")
    # Verify a resumed study before any GPU initialization.
    if (args.output / "manifest.json").exists():
        saved = runtime.read_json(args.output / "manifest.json")
        if saved != build_manifest(spec, saved["data_contract"]):
            raise ValueError("configuration/source identity changed; preserve this study")
        if (args.output / "task_plans/clean.json").exists():
            read_study(args.output, current_sources=True)
    from run_cifar_six_with_progress import discover_gpus, select_devices
    recorded_path = args.output / "execution_environment.json"
    recorded = (runtime.read_json(recorded_path).get("actual_compute_device")
                if recorded_path.exists() else None)
    devices, skipped = select_devices(discover_gpus(REPO), args.devices,
        recorded=recorded, min_free_memory_mib=args.min_free_memory_mib)
    args.devices = devices[:args.max_gpus]
    print("DEVICES " + json.dumps({"selected": args.devices, "skipped": skipped,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}), flush=True)
    print("STAGE_1 clean FedAvg only; 24 tasks; no automatic later stages", flush=True)
    return run_parent(args, spec)


if __name__ == "__main__":
    raise SystemExit(main())
