#!/usr/bin/env python3
"""Four CNN E2 controls paired with the completed stage-1 R3; no later stages.

The original diagnostic sources remain frozen. Workers reuse its exact
training/observation implementation in separate processes with new identities.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import run_cifar_diagnostic as clean
import cifar_diagnostic_report as clean_report

base, runtime = clean.base, clean.runtime
REPO = Path(__file__).resolve().parent
PROTOCOL = "cifar-v8-cnn-e2-match-v1"
DEFAULT_CONFIG = REPO / "configs/cifar10_cnn_e2_match_v1.json"
DEFAULT_REFERENCE = REPO / "outputs/cifar_v8_diagnostic_v1/clean"
DEFAULT_OUTPUT = REPO / "outputs/cifar_v8_diagnostic_v1/cnn_e2_match"
SETTING = {"id": "C3", "model": "v7_cnn", "lr": .05, "local_epochs": 2, "lr_decay": .99}
REFERENCE_FINGERPRINT = "96cac3f9131837d9e21f7b4a1bda7775c35d8d60f8acbf5dc3937c46d4f8e3c5"
NEW_SOURCES = ("run_cifar_matched_cnn.py", "cifar_matched_cnn_report.py")


def validate_spec(spec):
    expected = {"schema_version", "protocol", "output_dir", "reference_manifest_fingerprint",
                "reference_setting", "setting"}
    if (set(spec) != expected or spec["schema_version"] != 1 or spec["protocol"] != PROTOCOL
            or spec["reference_manifest_fingerprint"] != REFERENCE_FINGERPRINT
            or spec["reference_setting"] != "R3" or spec["setting"] != SETTING):
        raise ValueError("this follow-up fixes four CNN E2 controls for the reviewed stage-1 R3")
    return spec


def source_hashes():
    hashes = clean.source_hashes()
    for name in NEW_SOURCES:
        hashes[name] = hashlib.sha256((REPO / name).read_bytes()).hexdigest()
    return hashes


def reference_hashes(output, tasks):
    paths = [output / name for name in ("manifest.json", "task_plans/clean.json", "execution_environment.json")]
    for task in tasks:
        folder = output / "tasks" / task["task_id"]
        paths.extend(folder / name for name in ("task.json", base.experiments.COMPLETED_RESULTS_SNAPSHOT,
                                                "observations.json", "environment.json"))
        paths.extend(sorted((folder / "attempts").glob("*.json")))
    return {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def normalized_environment(metadata):
    """Pure equivalent of the frozen base.check_environment comparison policy."""
    comparable = deepcopy(metadata)
    comparable.pop("requested_device", None)
    for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        comparable["environment"].pop(key, None)
    for key in ("logical_device", "uuid"):
        comparable.get("actual_compute_device", {}).pop(key, None)
    if "torch" in comparable:
        for key in ("logical_cuda_devices", "logical_cuda_devices_status"):
            comparable["torch"].pop(key, None)
    nvidia = comparable.pop("nvidia", {})
    comparable["driver_versions"] = sorted({r["driver_version"] for r in nvidia.get("gpus", [])})
    return comparable


def audit_reference(output, expected_fingerprint=REFERENCE_FINGERPRINT):
    """Read original snapshots, not diagnostic_summary.json; never repair them."""
    output = Path(output).resolve()
    manifest, tasks = clean.read_study(output, current_sources=True)
    if manifest["fingerprint"] != expected_fingerprint:
        raise ValueError("reference is not the reviewed 24-task stage-1 study")
    before = reference_hashes(output, tasks)
    environment = runtime.read_json(output / "execution_environment.json")
    for task in tasks:
        metadata = runtime.read_json(output / "tasks" / task["task_id"] / "environment.json")
        if normalized_environment(metadata) != environment:
            raise ValueError("reference task numerical environment mismatch: " + task["task_id"])
    report = clean_report.summarize(output)
    decision = report.get("decision", {})
    if (report.get("status") != "complete" or report.get("complete_tasks") != 24
            or report.get("healthy_tasks") != 24 or not report.get("source_matches_current")
            or decision.get("best_healthy_resnet") != "R3"
            or decision.get("action") != "needs_four_matched_cnn_runs_before_architecture_decision"):
        raise ValueError("reference must retain 24 complete healthy tasks and the audited R3 winner")
    if before != reference_hashes(output, tasks):
        raise ValueError("reference files changed while being read")
    r3 = [t for t in tasks if t["candidate"]["candidate_id"] == "R3"]
    for task in r3:
        if (task["model"] != "resnet18_gn2"
                or any(task["config"][k] != SETTING[k] for k in ("lr", "local_epochs", "lr_decay"))):
            raise ValueError("R3 public parameters differ from the fixed CNN pairing")
    return {"manifest_fingerprint": manifest["fingerprint"], "data_contract": manifest["data_contract"],
        "execution_environment": environment,
        "evidence_sha256": before, "r3_tasks": r3,
        "rows": [r for r in report["rows"] if r["setting"] in ("C0", "R3")],
        "selection": decision, "complete_tasks": 24, "healthy_tasks": 24}


def build_manifest(spec, reference, dataset):
    validate_spec(spec)
    if reference["manifest_fingerprint"] != spec["reference_manifest_fingerprint"]:
        raise ValueError("reference fingerprint differs from the fixed follow-up configuration")
    payload = {"protocol": PROTOCOL, "spec": {**spec, "dataset": dataset},
        "reference": reference, "data_contract": reference["data_contract"],
        "source_sha256": source_hashes(), "model": {"architecture": "original_cifar_cnn", "parameter_count": 1756426},
        "evaluation_split": "calibration_dataset", "purpose": "development_paired_control",
        "official_test_used_for_selection": False, "next_stage_automatic": False}
    return {**payload, "fingerprint": base.digest(payload)}


def build_tasks(manifest):
    tasks = []
    originals = manifest["reference"]["r3_tasks"]
    expected_pairs = {(p, s) for p in ("iid", "dirichlet") for s in (2026093001, 2026093002)}
    if len(originals) != 4 or {(t["config"]["partition"], t["config"]["seed"]) for t in originals} != expected_pairs:
        raise ValueError("reference must contain the four distinct R3 seed/partition pairs")
    for original in originals:
        raw = {k: v for k, v in original.items() if k != "fingerprint"}
        if base.attach_fingerprints([raw], {"fingerprint": manifest["reference"]["manifest_fingerprint"]})[0] != original:
            raise ValueError("R3 reference task fingerprint is inconsistent")
        config = deepcopy(original["config"])
        if (original["candidate"]["candidate_id"] != "R3" or original["model"] != "resnet18_gn2"
                or config["method"] != "fedavg" or config["malicious_ratio"] != 0 or config["rounds"] != 150
                or any(config[k] != SETTING[k] for k in ("lr", "local_epochs", "lr_decay"))):
            raise ValueError("only clean R3 E2 is eligible for paired CNN controls")
        base.fl.ExperimentConfig(**config).validate()
        tasks.append({"task_id": f"clean_C3_{config['partition']}_seed{config['seed']}",
            "phase": "validation", "purpose": "development_paired_control", "method": "fedavg",
            "model": "v7_cnn", "candidate": {"candidate_id": "C3", "variant": "original",
                "parameters": {k: SETTING[k] for k in ("lr", "local_epochs", "lr_decay")}},
            "config": config, "reference_task_id": original["task_id"],
            "reference_task_fingerprint": original["fingerprint"]})
    return base.attach_fingerprints(tasks, manifest)


def read_study(output, *, current_sources=False):
    output = Path(output)
    manifest = runtime.read_json(output / "manifest.json")
    payload = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if manifest.get("protocol") != PROTOCOL or base.digest(payload) != manifest["fingerprint"]:
        raise ValueError("matched CNN manifest fingerprint/protocol mismatch")
    validate_spec({k: v for k, v in manifest["spec"].items() if k != "dataset"})
    if (manifest["reference"]["manifest_fingerprint"] != manifest["spec"]["reference_manifest_fingerprint"]
            or manifest["data_contract"] != manifest["reference"]["data_contract"]):
        raise ValueError("paired reference/data identity mismatch")
    if current_sources and source_hashes() != manifest["source_sha256"]:
        raise ValueError("matched CNN source identity changed; preserve the study")
    tasks = build_tasks(manifest)
    plan = runtime.read_json(output / "task_plans/matched_cnn.json")
    if plan != {"manifest_fingerprint": manifest["fingerprint"], "tasks": tasks}:
        raise ValueError("matched CNN plan differs from the declared study")
    return manifest, tasks


def worker(args):
    manifest, tasks = read_study(args.output, current_sources=True)
    matches = [t for t in tasks if t["task_id"] == args.worker]
    if len(matches) != 1 or len(args.devices) != 1 or args.devices == ["auto"]:
        raise ValueError("worker requires one declared CNN control and one explicit device")
    if runtime.read_json(args.output / "execution_environment.json") != manifest["reference"]["execution_environment"]:
        raise ValueError("paired worker environment differs from the frozen reference")
    task = matches[0]
    folder = args.output / "tasks" / task["task_id"]
    if runtime.read_json(folder / "task.json") != task:
        raise ValueError("paired worker task identity mismatch")
    with base.file_lock(folder / ".worker.lock", nonblocking=True):
        # This exact frozen implementation dispatches CNN, validates the same
        # dataset/environment, observes losses, and resumes durable checkpoints.
        return clean.run_task(args, manifest, task, folder)


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
                command = [sys.executable, "-u", str(REPO / "run_cifar_matched_cnn.py"),
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
                print(f"START {task['task_id']} model=v7_cnn local_epochs=2 device={device}", flush=True)
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
    # Never initialize a child inside the reference, or vice versa.
    if args.output == args.reference_output or args.output in args.reference_output.parents or args.reference_output in args.output.parents:
        raise ValueError("paired output and original reference must be separate sibling studies")
    source_manifest = runtime.read_json(args.reference_output / "manifest.json")
    expected = build_manifest(spec, reference, source_manifest["spec"]["dataset"])
    args.output.mkdir(parents=True, exist_ok=True)
    with base.file_lock(args.output / ".runner.lock", nonblocking=True):
        path = args.output / "manifest.json"
        if not path.exists() and any(p.name != ".runner.lock" for p in args.output.iterdir()):
            raise ValueError("nonempty paired output has no matching manifest")
        base.immutable_json(path, expected)
        tasks = build_tasks(expected)
        runtime.save_plan(args.output, "matched_cnn", tasks, expected)
        read_study(args.output, current_sources=True)
        # Require the same normalized numerical environment as R3, allowing
        # logical/physical CUDA indices to change under the existing policy.
        base.immutable_json(args.output / "execution_environment.json", reference["execution_environment"])
        for task in tasks:
            runtime.ensure_identity(args.output, task)
        try:
            execute(args, tasks)
        except KeyboardInterrupt:
            print("INTERRUPTED use the same command to resume", flush=True)
        from cifar_matched_cnn_report import summarize, print_summary
        report = summarize(args.output, args.reference_output)
        base.write_json(args.output / "matched_summary.json", report)
        print_summary(report)
        return 0 if report["status"] == "complete" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-output", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--devices", nargs="+", default=["auto"])
    parser.add_argument("--max-gpus", type=int, default=4)
    parser.add_argument("--min-free-memory-mib", type=float, default=16384.)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Read only; audit both studies without GPU/data/training")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.reference_output = args.reference_output.resolve()
    if args.summary:
        from cifar_matched_cnn_report import summarize, print_summary
        result = summarize((args.output or DEFAULT_OUTPUT).resolve(), args.reference_output)
        print_summary(result)
        return 0 if result["status"] == "complete" else 2
    if args.worker:
        if args.output is None:
            parser.error("worker requires --output")
        args.output = args.output.resolve()
        return worker(args)
    spec = validate_spec(runtime.read_json(args.config))
    args.output = (args.output or REPO / spec["output_dir"]).resolve()
    if args.plan_only:
        print(json.dumps({"protocol": PROTOCOL, "tasks": 4, "setting": SETTING,
            "seeds": [2026093001, 2026093002], "partitions": ["iid", "dirichlet"], "rounds": 150,
            "reference_setting": "R3", "reference_manifest_fingerprint": REFERENCE_FINGERPRINT,
            "reference_audited": False, "reference_audit_required_before_training": True,
            "output": str(args.output), "training_started": False, "next_stage_automatic": False}, indent=2))
        return 0
    if args.max_gpus < 1:
        parser.error("--max-gpus must be positive")
    print("AUDITING_REFERENCE original 24 tasks read only; no repeated training", flush=True)
    reference = audit_reference(args.reference_output, expected_fingerprint=spec["reference_manifest_fingerprint"])
    source_manifest = runtime.read_json(args.reference_output / "manifest.json")
    if (args.output / "manifest.json").exists():
        if runtime.read_json(args.output / "manifest.json") != build_manifest(spec, reference, source_manifest["spec"]["dataset"]):
            raise ValueError("paired source/config/reference identity changed; preserve this output")
    from run_cifar_six_with_progress import discover_gpus, select_devices
    recorded = reference["execution_environment"].get("actual_compute_device")
    if not recorded:
        raise ValueError("reference has no recorded GPU identity")
    devices, skipped = select_devices(discover_gpus(REPO), args.devices, recorded=recorded,
                                     min_free_memory_mib=args.min_free_memory_mib)
    args.devices = devices[:min(args.max_gpus, 4)]
    print("DEVICES " + json.dumps({"selected": args.devices, "skipped": skipped,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}), flush=True)
    print("CNN_E2_MATCH four clean controls only; original 24 tasks stay read-only", flush=True)
    return run_parent(args, spec, reference)


if __name__ == "__main__":
    raise SystemExit(main())
