"""Plan or durably archive old outputs by same-filesystem rename.

Nothing is deleted. Default invocation only prints a plan. Stop experiment
writers first: advisory locks and the process survey cannot identify every
possible external writer. A saved manifest supports --resume after interruption.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys

DEFAULT_KEEP = ("cifar10_six_relative_best_five_day_v7 2", "mnist_v7_target_fair_tuning")


class ArchiveError(RuntimeError):
    pass


def _plain_path(path):
    path = Path(os.path.abspath(path))
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ArchiveError(f"symlink path is not allowed: {item}")
    return path


def inventory(path):
    """SHA256 every regular file, including hidden files; reject all symlinks."""
    root = _plain_path(path)
    rows = []

    def visit(current):
        before = current.lstat()
        relative = str(current.relative_to(root))
        if stat.S_ISLNK(before.st_mode):
            raise ArchiveError(f"symlink is not followed or archived: {current}")
        if stat.S_ISDIR(before.st_mode):
            rows.append({"path": relative, "kind": "directory"})
            for child in sorted(current.iterdir(), key=lambda p: p.name):
                visit(child)
        elif stat.S_ISREG(before.st_mode):
            digest = hashlib.sha256()
            fd = os.open(current, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as handle:
                opened = os.fstat(handle.fileno())
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise ArchiveError(f"file replaced during inventory: {current}")
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                after = os.fstat(handle.fileno())
            keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, k) != getattr(after, k) for k in keys):
                raise ArchiveError(f"file changed during inventory: {current}")
            rows.append({"path": relative, "kind": "file", "size": after.st_size,
                         "sha256": digest.hexdigest()})
        else:
            raise ArchiveError(f"unsupported file type: {current}")

    visit(root)
    return rows


def process_survey():
    try:
        result = subprocess.run(["ps", "-axo", "pid=,command="], text=True,
                                capture_output=True, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": str(exc), "possible_writers": []}
    writers = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or int(parts[0]) == os.getpid():
            continue
        try:
            tokens = shlex.split(parts[1])
        except ValueError:
            tokens = parts[1].split()
        names = [Path(token).name for token in tokens]
        if any(name.startswith(("run_cifar_six", "run_fair_tuning", "run_attack_screen"))
               and name.endswith(".py") for name in names):
            writers.append({"pid": int(parts[0]), "command": parts[1]})
    return {"available": True, "possible_writers": writers,
            "limitation": "Advisory survey only; stop other scripts and external writers."}


@contextmanager
def held_locks(root, entries):
    """Keep known runner locks held across hashing, intent, rename and verify."""
    with ExitStack() as stack:
        for entry in entries:
            directory = root / entry["name"]
            if not directory.exists():
                directory = root / "old" / entry["name"]
            if not directory.is_dir():
                continue
            for name in (".runner.lock", ".environment.lock"):
                lock = directory / name
                if not lock.exists():
                    continue
                _plain_path(lock)
                handle = stack.enter_context(lock.open("rb"))
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ArchiveError(f"runner lock is held: {lock}") from exc
        yield


def build_plan(root, keep=DEFAULT_KEEP, *, ignore_metadata=False):
    root = _plain_path(root)
    if not root.is_dir():
        raise ArchiveError(f"outputs directory missing: {root}")
    keep = list(dict.fromkeys(keep))
    if any(Path(name).name != name or name in ("", ".", "..", "old") for name in keep):
        raise ArchiveError("keep entries must be root basenames other than old")
    for name in keep:
        if not (root / name).is_dir() or (root / name).is_symlink():
            raise ArchiveError(f"required preserved directory missing or symlink: {name}")
    old = _plain_path(root / "old")
    if old.exists() and not old.is_dir():
        raise ArchiveError("outputs/old must be a directory")
    entries = []
    for source in sorted(root.iterdir(), key=lambda p: p.name):
        if source.name in {*keep, "old"} or (ignore_metadata and source.name == ".DS_Store"):
            continue
        if (old / source.name).exists() or (old / source.name).is_symlink():
            raise ArchiveError(f"archive destination collision: {old / source.name}")
        if source.is_symlink():
            raise ArchiveError(f"source symlink is not allowed: {source}")
        device = old.stat().st_dev if old.exists() else root.stat().st_dev
        if source.stat().st_dev != device:
            raise ArchiveError(f"cross-filesystem rename forbidden: {source}")
        entries.append({"name": source.name, "status": "planned"})
    return {"version": 1, "root": str(root), "keep": keep, "status": "planned",
            "created_utc": datetime.now(timezone.utc).isoformat(), "entries": entries,
            "ignored_metadata": [".DS_Store"] if ignore_metadata else [],
            "process_survey": process_survey()}


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_manifest(path, manifest):
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink():
        raise ArchiveError("manifest temporary path must not be a symlink")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def apply_plan(plan, manifest_path, *, allow_unavailable_process_check=False):
    root = _plain_path(plan["root"])
    archive = _plain_path(root / "old")
    manifest_path = _plain_path(manifest_path)
    if manifest_path.parent != archive:
        raise ArchiveError("manifest must be directly inside outputs/old")
    if plan.get("version") != 1:
        raise ArchiveError("unsupported archive manifest version")
    for name in plan["keep"]:
        if Path(name).name != name or name in ("", ".", "..", "old"):
            raise ArchiveError("unsafe preserved name in manifest")
        if not _plain_path(root / name).is_dir():
            raise ArchiveError(f"preserved directory missing: {name}")
    names = [e["name"] for e in plan["entries"]]
    if (len(names) != len(set(names)) or any(Path(n).name != n or n in
            ("", ".", "..", "old", *plan["keep"]) for n in names)):
        raise ArchiveError("unsafe or duplicated archive entry")
    if manifest_path.name in names or manifest_path.name == ".archive.lock":
        raise ArchiveError("manifest name collides with an archived entry or lock")
    survey = process_survey()
    if survey["possible_writers"]:
        raise ArchiveError(f"possible experiment writers are running: {survey['possible_writers']}")
    if not survey["available"] and not allow_unavailable_process_check:
        raise ArchiveError("process survey unavailable; stop writers and explicitly acknowledge "
                           "with --acknowledge-process-check-unavailable")
    plan["apply_process_survey"] = survey
    archive.mkdir(exist_ok=True)
    with _plain_path(archive / ".archive.lock").open("a+b") as archive_lock:
        try:
            fcntl.flock(archive_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArchiveError("another archival operation holds the archive lock") from exc
        with held_locks(root, plan["entries"]):
            # Preflight the entire batch before moving any item.
            for entry in plan["entries"]:
                source, target = root / entry["name"], archive / entry["name"]
                _plain_path(source)
                _plain_path(target)
                if source.exists() and target.exists():
                    raise ArchiveError(f"source and destination both exist: {entry['name']}")
                current = source if source.exists() else target
                if not current.exists():
                    raise ArchiveError(f"source and destination both missing: {entry['name']}")
                if current.stat().st_dev != archive.stat().st_dev:
                    raise ArchiveError("cross-filesystem rename forbidden")
                observed = inventory(current)
                if "inventory" in entry and observed != entry["inventory"]:
                    raise ArchiveError(f"inventory mismatch: {current}")
                if "inventory" not in entry and not source.exists():
                    raise ArchiveError(f"unproven pre-existing destination: {current}")
                entry["inventory"] = observed
            plan["status"] = "in_progress"
            _write_manifest(manifest_path, plan)
            for entry in plan["entries"]:
                source, target = root / entry["name"], archive / entry["name"]
                if source.exists():
                    if inventory(source) != entry["inventory"]:
                        raise ArchiveError(f"source changed after preflight: {source}")
                    if target.exists() or target.is_symlink():
                        raise ArchiveError(f"archive collision: {target}")
                    entry["status"] = "rename_intent"
                    _write_manifest(manifest_path, plan)
                    os.rename(source, target)
                    _sync_directory(root)
                    _sync_directory(archive)
                if inventory(target) != entry["inventory"]:
                    raise ArchiveError(f"post-rename verification failed: {target}")
                entry["status"] = "verified"
                _write_manifest(manifest_path, plan)
            plan["status"] = "completed"
            _write_manifest(manifest_path, plan)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--keep", action="append", help="preserved root basename; repeat")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--ignore-metadata", action="store_true", help="leave root Finder .DS_Store in place")
    parser.add_argument("--resume", type=Path, help="resume/idempotently verify an existing manifest")
    parser.add_argument("--acknowledge-process-check-unavailable", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.resume:
            manifest = _plain_path(args.resume)
            plan = json.loads(manifest.read_text(encoding="utf-8"))
        else:
            plan = build_plan(args.outputs, args.keep or DEFAULT_KEEP, ignore_metadata=args.ignore_metadata)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
            manifest = Path(plan["root"]) / "old" / f"archive_manifest_{stamp}.json"
        if args.apply:
            plan = apply_plan(plan, manifest, allow_unavailable_process_check=
                              args.acknowledge_process_check_unavailable)
        print(json.dumps({"status": plan["status"], "apply": args.apply,
            "manifest": str(manifest) if args.apply or args.resume else None,
            "keep": plan["keep"], "ignored_metadata": plan.get("ignored_metadata", []),
            "entries": [{"name": e["name"], "status": e["status"]}
                for e in plan["entries"]], "process_survey": plan.get("apply_process_survey", plan.get("process_survey")),
            "note": "No deletion. SHA256 manifest stored before rename. Existing writers must be stopped."},
            ensure_ascii=False, indent=2))
        return 0
    except (ArchiveError, OSError, ValueError, KeyError) as exc:
        print(f"ARCHIVE_BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
