#!/usr/bin/env python3
"""Resume an existing prefix probe, waiting for its original GPU admission gate.

This parent-only adapter leaves all 72 frozen source files and task identities
unchanged. Workers still use run_cifar_prefix_probe.py. It changes only how the
controller handles temporary memory/utilization rejection before a worker.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import run_cifar_prefix_probe as original
import cifar_prefix_probe_report as report

protocol = original.protocol
MIN_FREE_MIB = 16384.
MAX_UTILIZATION = 5.
_active = False


class GPUWaitTimeout(RuntimeError):
    """The next worker was not launched within the bounded admission wait."""


def validate_wait(wait_seconds, poll_seconds):
    if not math.isfinite(wait_seconds) or not 0 <= wait_seconds <= 600:
        raise ValueError("wait-seconds must be finite and between 0 and 600")
    if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
        raise ValueError("poll-seconds must be positive and no greater than 60")


def gate_observation(rows, requested, expected_name, min_free, *, mask=None):
    """Distinguish resource pressure from incompatible or missing GPU evidence."""
    if requested == "auto" or not requested.startswith("GPU-") or min_free != MIN_FREE_MIB:
        raise ValueError("resume requires the original explicit UUID and 16384 MiB threshold")
    if mask is not None:
        tokens = mask.split(",")
        if any(not token.startswith("GPU-") for token in tokens):
            raise ValueError("ambiguous CUDA_VISIBLE_DEVICES; use full GPU UUIDs or leave unset")
        if requested not in tokens:
            raise ValueError("the original GPU is excluded by CUDA_VISIBLE_DEVICES; no migration allowed")
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError("invalid GPU inventory")
    uuids = [r.get("uuid") for r in rows]
    if any(not isinstance(value, str) or not value.startswith("GPU-") for value in uuids) or len(set(uuids)) != len(uuids):
        raise ValueError("invalid or duplicate UUID in GPU inventory")
    matches = [r for r in rows if r["uuid"] == requested]
    if len(matches) != 1:
        raise ValueError("the original GPU UUID is absent from the inventory; no migration allowed")
    row = matches[0]
    if row.get("name") != expected_name:
        raise ValueError("the original GPU model differs from the frozen reference")
    for key in ("free_mib", "utilization"):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid GPU capacity measurement: " + key)
    if row["utilization"] > 100:
        raise ValueError("invalid GPU utilization measurement")
    failed = []
    if row["free_mib"] < MIN_FREE_MIB:
        failed.append("free_memory_below_16384_mib")
    if row["utilization"] > MAX_UTILIZATION:
        failed.append("utilization_above_5_percent")
    return {"gpu_uuid": requested, "physical_index": row.get("index"), "name": row["name"],
        "free_mib": row["free_mib"], "utilization_percent": row["utilization"],
        "required_free_mib": MIN_FREE_MIB, "maximum_utilization_percent": MAX_UTILIZATION,
        "failed_checks": failed, "eligible": not failed}


@contextmanager
def waiting_gate(gpu_uuid, *, wait_seconds=600., poll_seconds=10., emit=None):
    """Wrap only the parent selector; original workers and admission rule remain."""
    global _active
    validate_wait(wait_seconds, poll_seconds)
    if _active:
        raise RuntimeError("GPU waiting contexts cannot be nested")
    emit = emit or (lambda value: print("GPU_GATE " + json.dumps(value, separators=(",", ":")), flush=True))
    selector = original.select_gpu

    def select(rows, requested, expected_name, min_free, *, mask=None):
        if requested != gpu_uuid:
            raise ValueError("resume attempted to change the frozen GPU UUID")
        started = time.monotonic()
        deadline = started + wait_seconds
        first_sample = True

        def timeout(observation, sampled_at):
            now = time.monotonic()
            emit({"event": "gpu_wait_timeout", "elapsed_seconds": now - started,
                  "sample_age_seconds": now - sampled_at, "wait_limit_seconds": wait_seconds,
                  "next_worker_started": False, **observation})
            raise GPUWaitTimeout("original GPU did not meet the unchanged admission gate before timeout; completed tasks retained")

        while True:
            observation = gate_observation(rows, requested, expected_name, min_free, mask=mask)
            sampled_at = time.monotonic()
            elapsed = sampled_at - started
            if not first_sample and sampled_at >= deadline:
                timeout(observation, sampled_at)
            if observation["eligible"]:
                # Always use the frozen selector's decision, never a substitute.
                selected = selector(rows, requested, expected_name, min_free, mask=mask)
                emit({"event": "gpu_ready", "elapsed_seconds": elapsed, **observation})
                return selected
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timeout(observation, sampled_at)
            emit({"event": "gpu_wait", "elapsed_seconds": elapsed,
                  "wait_limit_seconds": wait_seconds, "next_worker_started": False, **observation})
            try:
                time.sleep(min(poll_seconds, remaining))
                if time.monotonic() >= deadline:
                    timeout(observation, sampled_at)
                rows = original.gpu_inventory()
                first_sample = False
            except KeyboardInterrupt:
                emit({"event": "gpu_wait_interrupted", "gpu_uuid": gpu_uuid,
                      "next_worker_started": False, "completed_tasks_retained": True})
                raise

    _active = True
    original.select_gpu = select
    try:
        yield
    finally:
        original.select_gpu = selector
        _active = False


class ControlAudit:
    """A new controller sidecar; never overwrite any scientific evidence."""
    def __init__(self, output):
        folder = Path(output) / "controller_resumes"
        folder.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        self.path = folder / (stamp + "_" + str(os.getpid()) + ".jsonl")
        self.handle = self.path.open("x", encoding="utf-8")

    def emit(self, value):
        row = {"time_utc": datetime.now(timezone.utc).isoformat(), **value}
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        print("PREFIX_RESUME " + line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self):
        self.handle.close()


def print_current_summary(output):
    value = report.summarize(output)
    report.print_summary(value)
    return 0 if value["status"] == "complete" else 2


def resume(args):
    validate_wait(args.wait_seconds, args.poll_seconds)
    output = Path(args.output).resolve()
    # Requires the original complete plan; cannot create a fresh study or new GPU
    # assignment. The original reader also checks all 90 upstream evidence files.
    manifest, tasks = protocol.read_study(output, current_sources=True)
    reviewed = report.summarize(output)
    if reviewed["status"] not in ("complete", "incomplete_or_invalid_evidence") or any(
            r["status"] not in ("complete", "missing") for r in reviewed.get("rows", [])):
        raise ValueError("existing probe evidence is invalid; inspect before resume")
    paths = {k: Path(v) for k, v in manifest["reference"]["paths"].items()}
    old_args = SimpleNamespace(output=output, data_dir=args.data_dir, gpu=manifest["same_gpu_uuid"],
        min_free_memory_mib=MIN_FREE_MIB, retry_failed=False)
    audit = ControlAudit(output)
    try:
        audit.emit({"event": "controller_resume", "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "manifest_fingerprint": manifest["fingerprint"], "source_map_sha256": protocol.base.digest(manifest["source_sha256"]),
            "gpu_uuid": manifest["same_gpu_uuid"], "wait_seconds_per_admission": args.wait_seconds,
            "poll_seconds": args.poll_seconds, "minimum_free_mib": MIN_FREE_MIB,
            "maximum_utilization_percent": MAX_UTILIZATION, "automatic_retry_failed_workers": False,
            "complete_before": reviewed["complete_tasks"], "expected_tasks": len(tasks),
            "control_audit": str(audit.path), "scientific_sources_changed": False})
        with waiting_gate(manifest["same_gpu_uuid"], wait_seconds=args.wait_seconds,
                poll_seconds=args.poll_seconds, emit=audit.emit):
            code = original.run_parent(old_args, paths)
        audit.emit({"event": "controller_exit", "code": code})
        return code
    except (Exception, KeyboardInterrupt) as exc:
        audit.emit({"event": "controller_blocked", "exception": type(exc).__name__, "error": str(exc),
                    "completed_tasks_retained": True, "algorithm_health_assessed": False})
        print_current_summary(output)
        return 2
    finally:
        audit.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=protocol.DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--wait-seconds", type=float, default=600., help="maximum wait before each worker; original gate is unchanged")
    parser.add_argument("--poll-seconds", type=float, default=10.)
    parser.add_argument("--summary", action="store_true", help="read-only original summary; no GPU query or controller sidecar")
    args = parser.parse_args(argv)
    if args.summary:
        return print_current_summary(args.output.resolve())
    try:
        return resume(args)
    except (Exception, KeyboardInterrupt) as exc:
        print("PREFIX_RESUME " + json.dumps({"event": "resume_refused", "exception": type(exc).__name__,
            "error": str(exc), "task_execution_state": "see_original_task_summary"}, ensure_ascii=False), flush=True)
        print_current_summary(args.output.resolve())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
