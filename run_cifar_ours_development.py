#!/usr/bin/env python3
"""Run eight isolated Ours development experiments, with resumable checkpoints.

This is a new development experiment, not a replacement for historical tuning.
Both policies use the same strict CUDA settings and training-derived holdout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
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
import subprocess
import sys
import threading
import traceback

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import sm9rrsfl  # Set thread defaults before importing NumPy or Torch.
from sm9rrsfl import experiments, fl
from sm9rrsfl.fair_tuning import _training_health_reasons
from sm9rrsfl.calibration_policy import TRAINING_HEALTH_CONSTRAINTS
import diagnose_cifar_nonfinite as probe
from check_cifar_client_repeatability import apply_strict_settings, strict_child_environment
from check_cifar_prefix_repeatability import environment


VARIANTS = ("control_003", "weak_quarantine")
SCENARIOS = (("iid", 0.0), ("dirichlet", 0.0), ("dirichlet", 0.5), ("dirichlet", 0.7))
NOTE = ("Ours-only development experiment; one new development seed, four scenarios. "
        "These results do not replace original tuning records or establish formal test performance.")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    experiments._durable_replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def semantic_config(config):
    values = asdict(config) if isinstance(config, fl.ExperimentConfig) else dict(config)
    for key in ("device", "sm9_workers"):
        values.pop(key, None)
    return values


def validate_paths(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("development output must be separate from, and not contain, the original tuning output")
    return source, output


def source_hashes(repo):
    files = sorted((repo / "sm9rrsfl").glob("*.py")) + [repo / name for name in (
        "run_cifar_ours_development.py", "cifar_ours_development_policy.py",
        "diagnose_cifar_nonfinite.py", "check_cifar_prefix_repeatability.py",
        "check_cifar_client_repeatability.py", "run_experiments_from_config.py")]
    return {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def build_tasks(source_configs, *, seed=601):
    """Use four recorded 003 configurations; runtime device is assigned later."""
    if seed in (401, 402, 403, 501, 502, 503) or seed < 0:
        raise ValueError("use a new development seed, not validation or formal-test seeds")
    tasks = []
    for partition, ratio in SCENARIOS:
        original = source_configs[(partition, ratio)]
        if (original.method != "sm9rrs" or original.rounds != 100 or original.early_stop
                or original.eval_interval != 1 or original.partition != partition
                or original.malicious_ratio != ratio or original.seed != 401
                or original.detector_distance_threshold != 1.75
                or original.suspicion_remove_after != 3
                or original.compute_backend not in ("torch", "auto") or not original.device.startswith("cuda")):
            raise ValueError("source must be the recorded full 100-round CUDA Ours-003 protocol")
        # Checkpoint frequency and worker count are execution controls only.
        config = replace(original, seed=seed, device="cuda:0", sm9_workers=1,
                         checkpoint_interval=1)
        for variant in VARIANTS:
            task_id = f"{variant}_{partition}_ratio{ratio:g}_seed{seed}"
            tasks.append({"task_id": task_id, "variant": variant, "config": asdict(config),
                          "source_config": asdict(original),
                          "source_candidate": "sm9rrs-003", "weak_threshold": 1.25})
    return tasks


def build_manifest(source, seed, repo):
    configs, source_manifest = {}, None
    for partition, ratio in SCENARIOS:
        config, manifest, _ = probe.select_recorded_run(source, "sm9rrs-003", partition, ratio, 401)
        if source_manifest is not None and manifest != source_manifest:
            raise ValueError("recorded scenarios do not share one source manifest")
        source_manifest = manifest
        configs[(partition, ratio)] = config
    manifest = {
        "schema_version": 1, "note": NOTE, "source": str(source),
        "source_fingerprint": source_manifest["fingerprint"],
        "dataset": source_manifest["dataset"],
        "tuning_context": source_manifest["tuning_context"],
        "source_sha256": source_hashes(repo),
        "variant_definitions": {
            "control_003": "Original 003 detector and weighting policy.",
            "weak_quarantine": "After warmup, exclude otherwise accepted updates with novelty score >1.25. Do not admit trusted history or recover reliability; preserve reliability and decay existing count by .5. Original 003 strong evidence, drift, revocation and shock rules remain in force."},
        "strict_settings": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "deterministic": True,
                            "warn_only": False, "cudnn_deterministic": True,
                            "cudnn_benchmark": False, "TF32": "unchanged; recorded and checked across tasks"},
        "development_gates": {"max_control_relative_clean_accuracy_drop": 0.03,
                              "max_nonfinite_updates": 0, "required_rounds": 100,
                              "max_mean_tail_final_asr": 0.1, "max_peak_asr": 0.3},
        "training_health_constraints": dict(TRAINING_HEALTH_CONSTRAINTS),
        "tasks": build_tasks(configs, seed=seed),
    }
    manifest["fingerprint"] = digest(manifest)
    for task in manifest["tasks"]:
        task["fingerprint"] = digest({"manifest_fingerprint": manifest["fingerprint"],
                                      "task_id": task["task_id"], "config": semantic_config(task["config"]),
                                      "variant": task["variant"]})
    return manifest


def ensure_manifest(output, expected):
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError("existing experiment manifest differs; use a new output for changed code or protocol")
    else:
        leftovers = [p for p in output.iterdir() if p.name != ".runner.lock"]
        if leftovers:
            raise ValueError("refusing to reuse a nonempty output without its immutable manifest")
        write_json(path, expected)


@contextmanager
def file_lock(path, *, nonblocking=False):
    with Path(path).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError as exc:
            raise ValueError("another runner is using this output") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def ensure_task_identity(task_dir, task):
    path = task_dir / "task.json"
    if path.exists():
        if json.loads(path.read_text()) != task:
            raise ValueError("task directory identity/variant/fingerprint differs")
    else:
        if any(path.name != "worker.log" for path in task_dir.iterdir()):
            raise ValueError("task artifacts exist without their variant identity; refusing to infer identity")
        write_json(path, task)


def checked_completed(task_dir, task):
    snapshot_path = task_dir / experiments.COMPLETED_RESULTS_SNAPSHOT
    if not snapshot_path.exists():
        return None
    identity_path = task_dir / "task.json"
    if not identity_path.exists() or json.loads(identity_path.read_text()) != task:
        raise ValueError("completed result lacks its matching task variant and fingerprint")
    results = experiments.load_completed_results_snapshot(task_dir)
    if results is None or len(results) != 1:
        raise ValueError(f"unreadable completed snapshot: {snapshot_path}; retain it for investigation")
    result = results[0]
    if semantic_config(result.config) != semantic_config(task["config"]):
        raise ValueError("completed result does not match its task configuration")
    return result


def check_execution_environment(output, metadata):
    """Allow CUDA remapping; require equal numerical and software settings."""
    comparable = {key: value for key, value in metadata.items()
                  if key not in ("device", "source_sha256", "environment")}
    comparable["environment"] = {key: value for key, value in metadata["environment"].items()
                                 if key not in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
    with file_lock(output / ".environment.lock"):
        path = output / "execution_environment.json"
        if path.exists():
            if json.loads(path.read_text()) != comparable:
                raise ValueError("worker environment differs from this experiment; resume with the same environment")
        else:
            write_json(path, comparable)


def result_metrics(result):
    records = result.records
    attack = [r for r in records if r.round >= (result.config.attack_start_round or result.config.detector_window + 2)]
    if result.config.malicious_ratio == 0:
        attack = []
    rates = [r.attack_target_success_rate for r in attack]
    rates_valid = bool(rates) and all(v is not None and math.isfinite(v) and 0 <= v <= 1 for v in rates)
    complete = (result.stopped_round == result.config.rounds
                and [r.round for r in records] == list(range(result.config.rounds + 1)))
    reasons = list(_training_health_reasons(result))
    if not complete:
        reasons.append("incomplete_rounds")
    if result.nonfinite_updates or sum(r.nonfinite_updates for r in records):
        reasons.append("nonfinite_updates")
    if (not records or not math.isfinite(result.final_accuracy)
            or any(not math.isfinite(r.accuracy) or not 0 <= r.accuracy <= 1 for r in records)):
        reasons.append("invalid_accuracy")
    if attack and not rates_valid:
        reasons.append("missing_or_invalid_attack_metrics")
    if any(not math.isfinite(value) or not 0 <= value <= 1 + 1e-9
           for r in records for value in (r.honest_weight_loss, r.malicious_weight_mass)):
        reasons.append("invalid_weight_metrics")
    honest = result.config.num_clients - len(result.malicious_clients)
    final = records[-1] if records else None
    mean = lambda values: sum(values) / len(values) if values else None
    return {
        "completed": complete, "health_pass": not reasons, "health_reasons": sorted(set(reasons)),
        "stopped_round": result.stopped_round, "nonfinite_updates": result.nonfinite_updates,
        "final_accuracy": result.final_accuracy,
        "attack_mean_accuracy": mean([r.accuracy for r in attack]),
        "attack_mean_asr": mean(rates) if rates_valid else None,
        "attack_peak_asr": max(rates) if rates_valid else None,
        "attack_tail10_asr": mean(rates[-10:]) if rates_valid else None,
        "attack_final_asr": rates[-1] if rates_valid else None,
        "mean_honest_weight_loss": mean([r.honest_weight_loss for r in records if r.round > 0]),
        "attack_mean_honest_weight_loss": mean([r.honest_weight_loss for r in attack]),
        "final_false_positive_revocations": final.false_positive_revocations if final else None,
        "max_false_revocation_rate": max((r.false_positive_revocations / honest for r in records), default=0.0) if honest else None,
        "runtime_seconds": result.runtime_seconds,
    }


def summarize(output, manifest):
    rows = []
    for task in manifest["tasks"]:
        task_dir = output / "tasks" / task["task_id"]
        try:
            result = checked_completed(task_dir, task)
            metrics = result_metrics(result) if result is not None else None
            error = None
        except Exception as exc:
            metrics, error = None, str(exc)
        rows.append({"task_id": task["task_id"], "variant": task["variant"],
                     "partition": task["config"]["partition"],
                     "malicious_ratio": task["config"]["malicious_ratio"],
                     "metrics": metrics, "error": error,
                     "failure_record": str(task_dir / "failure.json") if (task_dir / "failure.json").exists() else None})
    paired = []
    for partition, ratio in SCENARIOS:
        group = {r["variant"]: r["metrics"] for r in rows
                 if r["partition"] == partition and r["malicious_ratio"] == ratio}
        left, right = group.get("control_003"), group.get("weak_quarantine")
        if left is None or right is None:
            continue
        paired.append({"partition": partition, "malicious_ratio": ratio,
                       "candidate_minus_control": {key: right[key] - left[key] for key in (
                           "final_accuracy", "attack_mean_accuracy", "attack_mean_asr", "attack_tail10_asr",
                           "attack_final_asr", "attack_peak_asr", "mean_honest_weight_loss",
                           "final_false_positive_revocations") if right[key] is not None and left[key] is not None},
                       "control_relative_clean_drop_pass": (left["final_accuracy"] - right["final_accuracy"] <= .03 + 1e-12)
                       if ratio == 0 else None})
    complete = all(r["metrics"] is not None and r["metrics"]["completed"] for r in rows)
    candidate = [r for r in rows if r["variant"] == "weak_quarantine"]
    candidate_healthy = all(r["metrics"] is not None and r["metrics"]["health_pass"] for r in candidate)
    retention = complete and all(p["control_relative_clean_drop_pass"] for p in paired if p["malicious_ratio"] == 0)
    attacked = [r["metrics"] for r in candidate if r["malicious_ratio"] > 0]
    target = complete and all(m is not None and all(m[k] is not None and m[k] <= limit + 1e-12
                   for k, limit in (("attack_mean_asr", .1), ("attack_tail10_asr", .1),
                                    ("attack_final_asr", .1), ("attack_peak_asr", .3))) for m in attacked)
    report = {"note": NOTE, "manifest_fingerprint": manifest["fingerprint"],
              "status": "complete" if complete else "incomplete", "tasks": rows, "paired": paired,
              "all_tasks_health_pass": complete and all(r["metrics"]["health_pass"] for r in rows),
              "candidate_health_pass": candidate_healthy,
              "candidate_control_relative_clean_retention_pass": retention,
              "candidate_attack_targets_met_in_development": target,
              "formal_feasibility_assessed": False,
              "clean_reference_note": "The .03 retention check uses concurrent Ours-003, not the original tuning clean reference.",
              "next_step": "Review these eight runs before any multi-seed validation; no automatic promotion to formal comparison."}
    write_json(output / "development_summary.json", report)
    return report


@contextmanager
def round_progress(task_id):
    original = experiments.run_experiment
    def observed(*args, **kwargs):
        callback = kwargs.get("checkpoint_callback")
        def progress(state):
            if callback is not None:
                callback(state)
            if state.get("records"):
                last = state["records"][-1]
                print(f"ROUND task={task_id} round={state['completed_round']} "
                      f"accuracy={last.accuracy:.6f} nonfinite={last.nonfinite_updates}", flush=True)
        kwargs["checkpoint_callback"] = progress
        return original(*args, **kwargs)
    experiments.run_experiment = observed
    try:
        yield
    finally:
        experiments.run_experiment = original


def run_worker(args):
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    matches = [task for task in manifest["tasks"] if task["task_id"] == args.worker]
    if len(matches) != 1:
        raise ValueError("unknown task id")
    task = matches[0]
    task_dir = output / "tasks" / task["task_id"]
    task_dir.mkdir(parents=True, exist_ok=True)
    try:
        if manifest["source_sha256"] != source_hashes(Path(__file__).resolve().parent):
            raise ValueError("source files changed after the experiment was planned")
        ensure_task_identity(task_dir, task)
        completed = checked_completed(task_dir, task)
        identity_config = fl.ExperimentConfig(**task["config"])
        if completed is not None:
            experiments.write_result_files(task_dir, [completed])
            experiments.finalize_config_checkpoint(task_dir / "checkpoints", identity_config, task["fingerprint"])
            print("ALREADY_COMPLETED " + task["task_id"], flush=True)
            return
        config = replace(identity_config, device=args.devices[0])
        import torch
        before = environment(config.device)
        apply_strict_settings(torch, before)
        metadata = environment(config.device)
        check_execution_environment(output, metadata)
        write_json(task_dir / "environment.json", metadata)
        recorded, source_manifest, _ = probe.select_recorded_run(
            Path(manifest["source"]), "sm9rrs-003", config.partition, config.malicious_ratio, 401)
        if (asdict(recorded) != task["source_config"]
                or source_manifest["fingerprint"] != manifest["source_fingerprint"]
                or source_manifest["dataset"] != manifest["dataset"]
                or source_manifest["tuning_context"] != manifest["tuning_context"]):
            raise ValueError("recorded source changed after planning")
        print("VERIFYING_RECORDED_DATA " + task["task_id"], flush=True)
        dataset = probe.load_verified_data(source_manifest, args.data_dir)
        print("DATA_DIGESTS_MATCH " + task["task_id"], flush=True)
        from cifar_ours_development_policy import weak_quarantine_policy
        context = weak_quarantine_policy(threshold=task["weak_threshold"]) if task["variant"] == "weak_quarantine" else nullcontext()
        with context, round_progress(task["task_id"]):
            result = experiments.run_measured_experiment(
                dataset, config, checkpoint_dir=task_dir / "checkpoints",
                run_fingerprint=task["fingerprint"], retain_success_checkpoint=True,
                checkpoint_identity_config=identity_config)
        experiments.write_result_files(task_dir, [result])
        write_json(task_dir / "metrics.json", result_metrics(result))
        experiments.finalize_config_checkpoint(task_dir / "checkpoints", identity_config, task["fingerprint"])
        print("COMPLETED " + task["task_id"], flush=True)
    except BaseException as exc:
        failure = {"task_id": task["task_id"], "time_utc": datetime.now(timezone.utc).isoformat(),
                   "exception": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(task_dir / "failure.json", failure)
        write_json(task_dir / ("failure_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f") + ".json"), failure)
        raise


def run_parent(args):
    source, output = validate_paths(args.source, args.output)
    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".runner.lock", nonblocking=True):
        manifest = build_manifest(source, args.seed, Path(__file__).resolve().parent)
        ensure_manifest(output, manifest)
        print("DEVELOPMENT_OUTPUT " + str(output), flush=True)
        print(f"PLANNED_RUNS {len(manifest['tasks'])} rounds=100 seed={args.seed} devices={','.join(args.devices)}", flush=True)
        available = queue.Queue()
        for device in args.devices:
            available.put(device)
        processes, process_lock, stopping = set(), threading.Lock(), threading.Event()
        def launch(task):
            device = available.get()
            try:
                if stopping.is_set():
                    return 130
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                           "--worker", task["task_id"], "--output", str(output), "--devices", device]
                if args.data_dir:
                    command += ["--data-dir", str(args.data_dir.resolve())]
                task_dir = output / "tasks" / task["task_id"]
                task_dir.mkdir(parents=True, exist_ok=True)
                print(f"START {task['task_id']} device={device}", flush=True)
                with (task_dir / "worker.log").open("a", encoding="utf-8") as log:
                    with process_lock:
                        if stopping.is_set():
                            return 130
                        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                   text=True, bufsize=1, env=strict_child_environment(os.environ))
                        processes.add(process)
                    with process:
                        try:
                            for line in process.stdout:
                                log.write(line)
                                log.flush()
                                print(f"[{task['task_id']}] {line.rstrip()}", flush=True)
                            return process.wait()
                        finally:
                            with process_lock:
                                processes.discard(process)
            finally:
                available.put(device)
        failed = 0
        with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
            futures = [pool.submit(launch, task) for task in manifest["tasks"]]
            try:
                for future in as_completed(futures):
                    try:
                        failed += int(future.result() != 0)
                    except Exception as exc:
                        failed += 1
                        print(f"WORKER_LAUNCH_FAILED {type(exc).__name__}: {exc}", flush=True)
                    summarize(output, manifest)
            except BaseException:
                stopping.set()
                for future in futures:
                    future.cancel()
                with process_lock:
                    for process in processes:
                        process.terminate()
                raise
            finally:
                report = summarize(output, manifest)
        print("DEVELOPMENT_RESULT " + json.dumps({key: report[key] for key in (
            "status", "all_tasks_health_pass", "candidate_health_pass", "candidate_control_relative_clean_retention_pass",
            "candidate_attack_targets_met_in_development")}), flush=True)
        print("SUMMARY " + str(output / "development_summary.json"), flush=True)
        return 1 if failed or report["status"] != "complete" else 0


def main():
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=repo / "outputs/cifar10_v7_target_fair_tuning")
    parser.add_argument("--output", type=Path, default=repo / "outputs/cifar10_ours_development_v1_seed601")
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--seed", type=int, default=601)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len(set(args.devices)) != len(args.devices) or any(not re.fullmatch(r"cuda:\d+", device) for device in args.devices):
        parser.error("--devices must contain distinct logical CUDA devices, e.g. cuda:0 cuda:1")
    if args.worker:
        if len(args.devices) != 1:
            parser.error("each worker uses one CUDA device")
        run_worker(args)
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
