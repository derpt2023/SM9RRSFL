#!/usr/bin/env python3
"""Audit a frozen-final-plan conflict; optionally copy validation to a new study.

The source study is never changed. Formal results are not reused. Selection and
health checks come from the unchanged six-method runner, using validation only.
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
import shutil
import sys
import tempfile

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import run_cifar_six_from_scratch as runner


REPO = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO / "configs/cifar10_six_original_v2.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@contextmanager
def source_lock(source):
    # rb preserves the source, even for an audit. A real runner creates this
    # lock before creating the manifest; missing locks are not repaired here.
    with (source / ".runner.lock").open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("source runner is active; stop it and wait for all workers to exit") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def check_destination(source, destination):
    source = Path(source).resolve()
    destination = Path(destination).absolute()
    if destination.is_symlink() or destination.exists():
        raise ValueError(f"destination already exists; no files will be overwritten: {destination}")
    destination = destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("source and destination must be separate, non-nested directories")
    if not destination.parent.is_dir():
        raise ValueError("destination parent directory must already exist")
    return destination


def audit(source, config=DEFAULT_CONFIG):
    """Called under source_lock; no writes and no dataset download/training."""
    source = Path(source).resolve()
    spec = runner.load_spec(config)
    manifest = read_json(source / "manifest.json")
    expected_manifest = runner.build_manifest(spec, manifest["data_contract"], REPO)
    if manifest != expected_manifest:
        raise ValueError("manifest, current configuration, or fingerprinted training source differs")
    tasks = runner.attach_fingerprints(runner.build_tasks(spec, "validation"), manifest)
    expected_validation = {"manifest_fingerprint": manifest["fingerprint"], "tasks": tasks}
    if read_json(source / "validation_plan.json") != expected_validation:
        raise ValueError("validation_plan does not match the immutable manifest")
    old_plan = read_json(source / "final_plan.json")
    old_selected = old_plan.get("selected")
    if not isinstance(old_selected, dict) or set(old_selected) != set(runner.ALL_METHODS):
        raise ValueError("old final_plan must identify one declared candidate for each of six methods")
    expected_final = {"manifest_fingerprint": manifest["fingerprint"], "selected": old_selected,
                      "tasks": runner.attach_fingerprints(runner.build_tasks(spec, "final", old_selected), manifest)}
    if old_plan != expected_final:
        raise ValueError("old final_plan is damaged or differs beyond a valid candidate selection")

    results, statuses = runner.collect_results(source, tasks)
    blockers = []
    for task, status in zip(tasks, statuses):
        folder = source / "tasks" / task["task_id"]
        if status["error"]:
            blockers.append({"task_id": task["task_id"], "reason": status["error"]})
        elif status["status"] == "pending":
            blockers.append({"task_id": task["task_id"], "reason": "validation task is still pending"})
        elif status["status"] == "failed":
            try:
                failure = read_json(folder / "failure.json")
                if (read_json(folder / "task.json") != task or failure.get("task_id") != task["task_id"]
                        or not isinstance(failure.get("exception"), str)
                        or not isinstance(failure.get("message"), str)):
                    raise ValueError("failed task has no matching immutable identity and failure record")
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                blockers.append({"task_id": task["task_id"], "reason": str(exc)})
    summary = runner.select_validation(spec, results, tasks)
    summary["tasks"] = statuses
    if summary["status"] != "qualified_for_final":
        blockers.append({"reason": "no healthy Ours candidate under the unchanged validation policy"})
    new_selected = summary["selected"]
    changes = {method: {"old": old_selected.get(method), "new": new_selected.get(method)}
               for method in runner.ALL_METHODS if old_selected.get(method) != new_selected.get(method)}
    if not changes:
        blockers.append({"reason": "candidate selection has not changed; this recovery is unnecessary"})
    counts = Counter(row["status"] for row in statuses)
    old_folders = [source / "tasks" / task["task_id"] for task in old_plan["tasks"]]
    report = {"status": "ready_to_copy_validation" if not blockers else "blocked",
              "source": str(source), "source_manifest_fingerprint": manifest["fingerprint"],
              "validation_counts": {key: counts[key] for key in ("complete", "failed", "pending")},
              "old_final_artifacts": {"planned_tasks": len(old_folders),
                  "task_directories": sum(folder.is_dir() for folder in old_folders),
                  "completed_snapshot_files": sum((folder / runner.experiments.COMPLETED_RESULTS_SNAPSHOT).is_file()
                                                  for folder in old_folders),
                  "checkpoint_directories": sum((folder / "checkpoints").is_dir() for folder in old_folders),
                  "final_summary_exists": (source / "final_summary.json").is_file(),
                  "final_results_exists": (source / "final_results").is_dir()},
              "validation_status": summary["status"], "ours_health_passed": summary["ours_health_passed"],
              "old_selected": old_selected, "new_selected": new_selected,
              "selection_changes": changes, "blockers": blockers,
              "final_training_results_reused": False, "official_test_used_for_selection": False,
              "selection_policy_changed": False}
    if summary["ours_health_passed"]:
        new_plan = {"manifest_fingerprint": manifest["fingerprint"], "selected": new_selected,
                    "tasks": runner.attach_fingerprints(runner.build_tasks(spec, "final", new_selected), manifest)}
        report["intended_final_plan_digest"] = runner.digest(new_plan)
    return report, summary, tasks


def inventory(source, tasks):
    if (source / "tasks").is_symlink():
        raise ValueError("validation tasks must be local directories, not a symbolic link")
    paths = [source / "manifest.json", source / "validation_plan.json"]
    if (source / "execution_environment.json").exists():
        read_json(source / "execution_environment.json")
        paths.append(source / "execution_environment.json")
    for task in tasks:
        folder = source / "tasks" / task["task_id"]
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError(f"validation task directory missing or symbolic: {folder}")
        for path in sorted(folder.rglob("*")):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise ValueError(f"unsupported validation artifact (must be a regular local file): {path}")
            if path.is_file():
                paths.append(path)
    records = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source metadata must be regular files: {path}")
        records.append({"path": str(path.relative_to(source)), "bytes": path.stat().st_size,
                        "sha256": file_hash(path)})
    return records


def apply_copy(source, destination, report, summary, tasks):
    source = Path(source).resolve()
    if report["blockers"]:
        raise ValueError("audit blocked recovery; inspect blockers before proceeding")
    destination = check_destination(source, destination)
    records = inventory(source, tasks)
    needed = sum(row["bytes"] for row in records)
    free = shutil.disk_usage(destination.parent).free
    reserve = max(64 * 1024 * 1024, needed // 20)
    if free < needed + reserve:
        raise ValueError(f"insufficient disk space: copy needs {needed} bytes plus {reserve} bytes reserve; free={free}")
    # Only this unique temporary directory is cleaned on failure. Source and
    # pre-existing destinations are never deleted, renamed, or overwritten.
    with tempfile.TemporaryDirectory(prefix=".cifar-final-recovery-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        directories = {str(Path("tasks") / task["task_id"]) for task in tasks}
        for task in tasks:
            directories.update(str(path.relative_to(source))
                               for path in (source / "tasks" / task["task_id"]).rglob("*") if path.is_dir())
        for directory in sorted(directories):
            (staging / directory).mkdir(parents=True, exist_ok=True)
        for record in records:
            original, copied = source / record["path"], staging / record["path"]
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, copied)
            with copied.open("rb") as handle:
                os.fsync(handle.fileno())
            if copied.stat().st_size != record["bytes"] or file_hash(copied) != record["sha256"]:
                raise ValueError(f"copied file failed verification: {record['path']}")
            if file_hash(original) != record["sha256"]:
                raise ValueError(f"source changed while copying: {record['path']}")
        # Failure-record paths in the copied summary should point to the copy;
        # original snapshot/log contents remain byte-for-byte unchanged.
        summary = json.loads(json.dumps(runner.json_safe(summary)))
        for row in summary["tasks"]:
            if row.get("failure_record"):
                row["failure_record"] = str(destination / "tasks" / row["task_id"] / "failure.json")
        runner.write_json(staging / "validation_summary.json", summary)
        recovery = {**report, "status": "validation_copied_for_new_final_plan",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(), "destination": str(destination),
                    "reason": "current validation selection differs from the preserved frozen final plan",
                    "source_final_plan_sha256": file_hash(source / "final_plan.json"),
                    "source_validation_plan_sha256": file_hash(source / "validation_plan.json"),
                    "copied_file_count": len(records), "copied_bytes": needed, "copied_files": records,
                    "copied_directories": sorted(directories),
                    "validation_summary_sha256": file_hash(staging / "validation_summary.json"),
                    "excluded": ["all final tasks", "final_plan.json", "final_summary.json", "final_results"],
                    "source_results_preserved": True}
        runner.write_json(staging / "recovery_manifest.json", recovery)
        # mkdir is exclusive (including against an empty destination created
        # during the copy). Publish the experiment manifest last: interruption
        # before then leaves an explicitly unusable partial output.
        destination.mkdir()
        incomplete = destination / "recovery_incomplete.json"
        runner.write_json(incomplete, {"source": str(source), "status": "incomplete_do_not_run"})
        for path in sorted(staging.iterdir(), key=lambda p: p.name == "manifest.json"):
            path.rename(destination / path.name)
        incomplete.unlink()
    return recovery


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO / "outputs/cifar10_six_original_v2")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--destination", type=Path, help="new sibling output directory, which must not exist")
    parser.add_argument("--apply", action="store_true", help="copy verified validation; default is read-only audit")
    args = parser.parse_args(argv)
    if args.apply and args.destination is None:
        parser.error("--apply requires --destination")
    source = args.source.resolve()
    try:
        if args.destination is not None:
            check_destination(source, args.destination)
        with source_lock(source):
            report, summary, tasks = audit(source, args.config)
            print("FINAL_RECOVERY_AUDIT " + json.dumps(runner.json_safe(report), ensure_ascii=False), flush=True)
            if report["blockers"]:
                return 2
            if args.apply:
                recovered = apply_copy(source, args.destination, report, summary, tasks)
                print("FINAL_RECOVERY_READY " + recovered["destination"], flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FINAL_RECOVERY_REFUSED {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
