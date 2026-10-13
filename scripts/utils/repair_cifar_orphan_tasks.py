#!/usr/bin/env python3
"""Quarantine v7 pre-initialization remnants without changing a frozen study.

Dry-run by default. No task identities are fabricated, and no training, result,
or checkpoint artifacts are moved. The original runner recreates eligible tasks
from its original final plan after the operator starts it again.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import run_cifar_six_interactive as controller

runner = controller.original
REPO = Path(__file__).resolve().parent
ALLOWED_REMNANT = re.compile(
    r"worker\.log|failure(?:_\d{8}T\d{6}_\d{6})?\.json|"
    r"\.(?:task|failure(?:_\d{8}T\d{6}_\d{6})?)\.json\.\d+\.tmp"
)


def read_json(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"metadata must be a regular file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def study_lock(output):
    path = output / ".runner.lock"
    if path.is_symlink() or not path.is_file():
        raise ValueError("original .runner.lock is missing or symbolic; refusing recovery")
    # Opening read-only does not create or truncate experiment metadata.
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("runner is active; stop it and wait for workers to exit") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def ensure_no_workers(output):
    # Workers do not acquire the parent's lock and can outlive a killed parent.
    listing = subprocess.run(["ps", "-eww", "-o", "pid=,args="], check=True,
                             capture_output=True, text=True).stdout
    for line in listing.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        try:
            argv = shlex.split(fields[1])
        except ValueError:
            if "--worker" in fields[1] and str(output) in fields[1]:
                raise ValueError(f"cannot safely identify possible worker PID {fields[0]}")
            continue
        if "--worker" in argv and "--output" in argv:
            index = argv.index("--output") + 1
            if index < len(argv) and (argv[index] == str(output) or str(output) in fields[1]):
                raise ValueError(f"worker PID {fields[0]} is still active for this output")


def inspect_task(output, task):
    folder = output / "tasks" / task["task_id"]
    row = {"task_id": task["task_id"], "state": "pending", "files": []}
    if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
        raise ValueError("task path is not a regular directory")
    if not folder.exists():
        return row
    identity = folder / "task.json"
    if identity.exists() or identity.is_symlink():
        if read_json(identity) != task:
            raise ValueError("existing task.json differs from the frozen plan")
        row["state"] = "identified_untouched"
        row["snapshot_exists"] = (folder / runner.experiments.COMPLETED_RESULTS_SNAPSHOT).exists()
        row["checkpoint_directory_exists"] = (folder / "checkpoints").exists()
        return row
    paths = sorted(folder.iterdir())
    for path in paths:
        if path.is_symlink() or not path.is_file() or not ALLOWED_REMNANT.fullmatch(path.name):
            raise ValueError(f"unidentified training/checkpoint/result or unknown artifact: {path.name}")
        if path.name == "worker.log":
            with path.open(encoding="utf-8", errors="replace") as handle:
                if any(re.match(r"^(ROUND|COMPLETED|ALREADY_COMPLETED)\s", line) for line in handle):
                    raise ValueError("worker.log contains execution progress without task.json")
        else:
            try:
                value = read_json(path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Partial initialization/error JSON can remain after ENOSPC.
                value = None
            else:
                if path.name.startswith(".task.json."):
                    if value != task:
                        raise ValueError("complete temporary task identity differs from the frozen plan")
                elif not isinstance(value, dict) or value.get("task_id") != task["task_id"]:
                    raise ValueError("failure record does not identify this task")
                elif value.get("execution_context") is not None:
                    raise ValueError("failure record contains training context without task.json")
        row["files"].append({"name": path.name, "bytes": path.stat().st_size,
                             "sha256": file_hash(path)})
    row["state"] = "quarantinable" if any(p.name != "worker.log" for p in paths) else "initializable"
    return row


def audit(output, config):
    """Metadata-only audit. Actual result/health validation remains in the runner."""
    spec = runner.load_spec(config)
    metadata = ["manifest.json", "validation_plan.json", "validation_summary.json", "final_plan.json"]
    manifest = read_json(output / "manifest.json")
    if manifest != runner.build_manifest(spec, manifest["data_contract"], REPO):
        raise ValueError("configuration, manifest or frozen scientific source differs")
    validation_tasks = runner.attach_fingerprints(runner.build_tasks(spec, "validation"), manifest)
    if read_json(output / "validation_plan.json") != {
            "manifest_fingerprint": manifest["fingerprint"], "tasks": validation_tasks}:
        raise ValueError("validation plan differs from the frozen manifest")
    validation = read_json(output / "validation_summary.json")
    decision_path = output / controller.DECISION_FILE
    if decision_path.exists() or decision_path.is_symlink():
        selected = controller.check_decision(read_json(decision_path), manifest, validation)
        metadata.append(controller.DECISION_FILE)
    elif validation.get("status") == "qualified_for_final":
        selected = validation["selected"]
    else:
        raise ValueError("unmet validation requires its existing valid continuation decision")
    plan = read_json(output / "final_plan.json")
    if plan != runner.final_plan(spec, manifest, selected):
        raise ValueError("formal plan differs from the frozen selection/manifest")
    tasks_root = output / "tasks"
    if tasks_root.is_symlink() or not tasks_root.is_dir():
        raise ValueError("tasks must be an existing regular directory")
    rows, blocked = [], []
    for task in plan["tasks"]:
        try:
            rows.append(inspect_task(output, task))
        except (OSError, ValueError, TypeError) as exc:
            blocked.append({"task_id": task["task_id"], "reason": str(exc)})
    counts = dict(Counter(row["state"] for row in rows))
    orphans = [row for row in rows if row["state"] == "quarantinable"]
    return {
        "status": "blocked" if blocked else "ready" if orphans else "no_action",
        "output": str(output), "selected": selected, "counts": counts,
        "identified_snapshot_files": sum(row.get("snapshot_exists", False) for row in rows),
        "snapshot_integrity_checked": False,
        "protected_metadata_sha256": {name: file_hash(output / name) for name in metadata},
        "orphans": orphans, "blocked_tasks": blocked,
    }


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def quarantine(output, config, report):
    if report["blocked_tasks"]:
        raise ValueError("blocked audit: no task directories will be moved")
    if not report["orphans"]:
        return report
    # Refuse changed metadata or artifacts between inspection and mutation.
    if audit(output, config) != report:
        raise ValueError("study changed after audit; no task directories will be moved")
    ensure_no_workers(output)
    root = output / "orphan_recovery"
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError("recovery destination must be a regular directory")
    root.mkdir(exist_ok=True)
    batch = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f") + "_" + uuid.uuid4().hex[:8])
    batch.mkdir()
    sync_directory(output)
    sync_directory(root)
    journal = {**report, "status": "prepared", "quarantine_dir": str(batch),
               "tool_sha256": file_hash(Path(__file__)), "moved_tasks": []}
    # A durable intent records every original file/hash before the first rename.
    # On interruption, moved folders remain here and unmoved folders in tasks/.
    runner.write_json(batch / "audit.json", journal)
    print(f"RECOVERY_BACKUP {batch}", flush=True)
    for row in report["orphans"]:
        source = output / "tasks" / row["task_id"]
        destination = batch / row["task_id"]
        if destination.exists() or destination.is_symlink():
            raise ValueError("quarantine destination unexpectedly exists")
        os.rename(source, destination)
        sync_directory(output / "tasks")
        sync_directory(batch)
        journal["moved_tasks"].append(row["task_id"])
        runner.write_json(batch / "audit.json", journal)
    journal["status"] = "quarantined"
    runner.write_json(batch / "audit.json", journal)
    return journal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=runner.DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply", action="store_true", help="move only audited startup remnants into a preserved backup")
    args = parser.parse_args(argv)
    try:
        spec = runner.load_spec(args.config)
        output = (args.output or REPO / spec["output_dir"]).resolve(strict=True)
        with study_lock(output):
            ensure_no_workers(output)
            report = audit(output, args.config)
            if args.apply and report["status"] != "blocked":
                report = quarantine(output, args.config, report)
        summary = {k: report[k] for k in ("status", "counts", "selected", "identified_snapshot_files", "snapshot_integrity_checked")}
        summary.update(blocked_count=len(report["blocked_tasks"]),
                       quarantinable_count=len(report["orphans"]),
                       moved_count=len(report.get("moved_tasks", [])),
                       quarantine_dir=report.get("quarantine_dir"))
        print("RECOVERY_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
        for row in report["blocked_tasks"][:10]:
            print("RECOVERY_BLOCKED " + json.dumps(row, ensure_ascii=False), flush=True)
        for row in report["orphans"][:3]:
            print("RECOVERY_SAMPLE " + json.dumps(row, ensure_ascii=False), flush=True)
        return 2 if report["status"] == "blocked" else 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"RECOVERY_ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
