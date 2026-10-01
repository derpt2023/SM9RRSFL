"""Read-only display state for the independent schema-8 experiment runners.

This module is deliberately outside every scientific source manifest. It never
loads pickle files, repairs results, creates output directories, or changes a
runner's decisions. Counts describe execution evidence, not algorithm health.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from time import monotonic


def _duration(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "estimating"
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"


def _mapping(devices):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = visible.split(",") if visible is not None else None
    result = {}
    for device in devices:
        match = re.fullmatch(r"cuda:(\d+)", device)
        if not match:
            result[device] = device
            continue
        index = int(match[1])
        result[device] = ("CUDA ordinal " + str(index) + " (unmasked)" if tokens is None else
                          "visible GPU " + tokens[index].strip() if index < len(tokens) else
                          "unavailable in CUDA_VISIBLE_DEVICES")
    return result


def _objects(value):
    return (row for row in value if isinstance(row, dict)) if isinstance(value, list) else ()


class AdaptiveProgress:
    """Observe controller events and bounded JSON metadata without training.

    ``consume`` accepts complete parent records, including a prompt record
    without its trailing newline. ``refresh`` may be called periodically.
    ``waiting_for_choice`` lets a terminal wrapper stop repainting during Y/N.
    A bare snapshot is never sufficient to count a verified completed task.
    """

    def __init__(self, output: Path, spec: dict, now=monotonic):
        self.output, self.spec, self.now = Path(output), spec, now
        self.started = self.rate_started = now()
        self.phase, self.status = "initializing", "loading data / checking experiment identity"
        self.devices, self.mapping = [], {}
        self.tasks, self.lanes = {}, {}
        self.current_ids, self.evaluated_ids = set(), set()
        self.completed, self.failed, self.paused = set(), set(), set()
        self.waiting_for_choice = self.controller_finished = False
        self.total = self.new_rounds = 0
        self.block_id, self.wave_index = None, None
        self._cache, self._plans = {}, {}
        self._view = None
        self._manifest_fingerprint = None
        self._summary = {}
        self._rate_has_started = self._report_received = False
        self._not_started = None
        self.exit_code = None

    def _read(self, path):
        """Cache small JSON by stat; never read model/checkpoint payloads."""
        path = Path(path)
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stat.st_size > 32 * 1024 * 1024:
                return {}
            cached = self._cache.get(path)
            if cached and cached[0] == stamp:
                return cached[1]
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return {}
            self._cache[path] = (stamp, value)
            return value
        except (OSError, UnicodeError, ValueError):
            return {}

    def _stamp(self, path):
        cached = self._cache.get(path)
        return cached[0] if cached else None

    def _load_plan(self, path, phase):
        plan = self._read(path)
        if (not self._manifest_fingerprint or
                plan.get("manifest_fingerprint") != self._manifest_fingerprint):
            return set()
        ids = set()
        for row in _objects(plan.get("tasks")):
            tid = row.get("task_id", "")
            config = row.get("config", {})
            candidate = row.get("candidate", {})
            if not isinstance(config, dict) or not isinstance(candidate, dict):
                continue
            total = config.get("rounds")
            if (not isinstance(tid, str) or not tid or Path(tid).name != tid
                    or tid in (".", "..") or row.get("phase") != phase
                    or not isinstance(row.get("fingerprint"), str)
                    or type(total) is not int or total < 1
                    or not isinstance(candidate.get("candidate_id"), str)):
                continue
            if tid in self.tasks and self.tasks[tid]["identity"] != row:
                continue
            self.tasks.setdefault(tid, {"identity": row, "round": 0, "total": total,
                "state": "queued", "authority": None, "event": None,
                "eta_primed": False, "progress_stamp": None, "healthy": None})
            ids.add(tid)
        self._plans[str(path)] = ids
        return ids

    def _set_view(self, phase, ids, block=None, wave=None):
        view = (phase, block, wave)
        if self._view != view:
            self.new_rounds = 0
            self.rate_started = self.now()
            self._rate_has_started = False
            self._view = view
        self.phase, self.current_ids, self.total = phase, set(ids), len(ids)
        self.block_id, self.wave_index = block, wave

    def _discover(self):
        manifest = self._read(self.output / "manifest.json")
        if manifest.get("schema_version") != 8 or manifest.get("spec") != self.spec:
            if manifest:
                self._set_view("initializing", set())
                if not self.waiting_for_choice and not self._not_started:
                    self.status = "manifest/spec mismatch; awaiting controller identity check"
            return
        self._manifest_fingerprint = manifest.get("fingerprint")
        state = self._read(self.output / "search_state.json")
        if state.get("manifest_fingerprint") != self._manifest_fingerprint:
            state = {}
        current = None
        self.evaluated_ids = set()
        for block in _objects(state.get("blocks")):
            bid = block.get("id", "")
            if not isinstance(bid, str) or not re.fullmatch(r"b\d+", bid):
                continue
            prefix = True
            chosen = None
            for wave in _objects(block.get("waves")):
                index = wave.get("index")
                if type(index) is not int or index < 0:
                    continue
                ids = self._load_plan(self.output / "task_plans" / f"{bid}-wave{index:03d}.json", "validation")
                if prefix and wave.get("status") == "complete":
                    self.evaluated_ids.update(ids)
                    for tid in ids:
                        self.tasks[tid]["ledger_complete"] = True
                else:
                    prefix = False
                if chosen is None or chosen[3]:
                    chosen = (bid, index, ids, wave.get("status") == "complete")
            if chosen is not None:
                current = chosen
        final_ids = self._load_plan(self.output / "final_plan.json", "final")
        if final_ids:
            self._set_view("final", final_ids)
        elif current:
            self._set_view("validation", current[2], current[0], current[1])
        elif state:
            self._set_view("validation", set())
        if not self.waiting_for_choice and not self.controller_finished and not self._not_started:
            if self.phase == "validation":
                self.status = "search " + str(state.get("status", "preparing"))
            elif self.phase == "final":
                self.status = "formal plan observed; execution follows controller; completion is not a health verdict"
        self._summary = self._read(self.output / "final_summary.json") if final_ids else {}

    def _summary_complete(self, task):
        identity = task["identity"]
        for row in _objects(self._summary.get("tasks")):
            if (row.get("task_id") == identity["task_id"]
                    and row.get("fingerprint") == identity["fingerprint"]
                    and row.get("execution_status") == "complete"
                    and row.get("final_round_available") is True
                    and row.get("required_round") == task["total"]):
                return True
        return False

    def _inspect(self, tid, task):
        identity = task["identity"]
        folder = self.output / "tasks" / tid
        if self._read(folder / "task.json") != identity:
            task["state"] = "identity_pending"
            return
        path = folder / "progress.json"
        progress = self._read(path)
        rd = progress.get("last_completed_round")
        if (progress.get("task_id") == tid and progress.get("study_phase") == identity["phase"]
                and progress.get("candidate_id") == identity["candidate"]["candidate_id"]
                and type(rd) is int and 0 <= rd <= task["total"]):
            stamp = self._stamp(path)
            if stamp != task["progress_stamp"]:
                if task.get("event") == "running" and tid in self.current_ids:
                    if task["eta_primed"]:
                        self.new_rounds += max(0, rd - task["round"])
                    task["eta_primed"] = True
                task["progress_stamp"] = stamp
            task["round"] = max(task["round"], rd)
        metric = self._read(folder / "metrics.json")
        accuracy = metric.get("final_accuracy")
        full_metadata = (metric.get("stopped_round") == task["total"]
            and type(accuracy) in (int, float) and math.isfinite(accuracy) and 0 <= accuracy <= 1
            and isinstance(metric.get("reasons"), list)
            and "incomplete_rounds" not in metric["reasons"]
            and not any(str(r).startswith("invalid_") for r in metric["reasons"]))
        snapshot = (folder / ".completed_results.pickle").is_file()
        trusted = (task.get("authority") in ("reuse", "exit0")
                   or task.get("ledger_complete") or self._summary_complete(task))
        if trusted and full_metadata and snapshot:
            task.update(state="complete", round=task["total"], healthy=metric.get("healthy"))
            for device, current in list(self.lanes.items()):
                if current == tid:
                    del self.lanes[device]
            return
        if task.get("event") in ("running", "retry"):
            task["state"] = "running" if task["event"] == "running" else "queued"
            return
        if snapshot:
            task["state"] = "unverified_snapshot"
            return
        if task.get("authority") == "exit0":
            task["state"] = "unverified_snapshot"
            return
        failure = self._read(folder / "failure.json")
        if (failure.get("task_id") == tid and failure.get("task_fingerprint") == identity["fingerprint"]):
            # Infrastructure/budget remnants are retried by this invocation.
            # Only numerical failures are permanently settled by the runner.
            current = task.get("event") in ("failed", "paused") or self.controller_finished
            task["state"] = ("failed" if failure.get("kind") == "algorithm_numerical" else
                             "paused" if current and failure.get("kind") == "budget_or_interrupt" else
                             "failed" if current else "queued")
            return
        if task.get("event") == "paused":
            task["state"] = "paused"
        elif task.get("event") == "failed":
            task["state"] = "failed"
        elif task.get("event") == "stopped":
            task["state"] = "paused" if task["round"] else "queued"
        else:
            task["state"] = "queued"

    def refresh(self):
        self._discover()
        for tid, task in self.tasks.items():
            # Immutable completed wave prefixes need no per-second GPFS scan.
            # START/RETRY events still cause their task to be inspected again.
            if (tid not in self.current_ids and task.get("ledger_complete")
                    and task["state"] in ("complete", "failed")
                    and task.get("event") not in ("running", "retry")):
                continue
            self._inspect(tid, task)
        self.completed = {tid for tid in self.current_ids if self.tasks[tid]["state"] == "complete"}
        self.failed = {tid for tid in self.current_ids if self.tasks[tid]["state"] == "failed"}
        self.paused = {tid for tid in self.current_ids if self.tasks[tid]["state"] == "paused"}
        if self._summary and not self._not_started and (self.controller_finished or self._report_received):
            self.status = ("report=" + str(self._summary.get("report_status", "unknown"))
                           + "; training=" + str(self._summary.get("status", "unknown")))
        if self.controller_finished and self.exit_code not in (None, 0):
            evidence = ("; saved report=" + str(self._summary.get("report_status", "unknown"))
                        + " (may predate this invocation)" if self._summary else "")
            self.status = f"controller exited code={self.exit_code}; execution did not finish successfully" + evidence

    def consume(self, original):
        line = original.strip()
        if not line:
            return False
        if line.startswith("DEVICES "):
            try:
                devices = json.loads(line.split(" ", 1)[1])["selected"]
                if isinstance(devices, list) and all(isinstance(d, str) for d in devices):
                    self.devices = list(dict.fromkeys(devices))
                    self.mapping = _mapping(self.devices)
            except (KeyError, TypeError, ValueError):
                pass
            return True
        if line.startswith("CONTINUATION_PROMPT "):
            self.waiting_for_choice = True
            self.status = "awaiting Y/N; previous approval does not authorize this invocation"
            return True
        self.refresh()
        match = re.fullmatch(r"START (\S+) device=(\S+)", line)
        if match:
            tid, device = match.groups()
            if tid in self.tasks:
                task = self.tasks[tid]
                task.update(event="running", authority=None, state="running", eta_primed=False)
                self.lanes[device] = tid
                self.waiting_for_choice = False
                if not self._rate_has_started:
                    self.rate_started = self.now()
                    self._rate_has_started = True
                if device not in self.devices:
                    self.devices.append(device)
                    self.mapping = _mapping(self.devices)
                self.refresh()
            return True
        match = re.fullmatch(r"REUSE (\S+)", line)
        if match:
            if match[1] in self.tasks:
                self.tasks[match[1]].update(authority="reuse", event="complete")
            self.waiting_for_choice = False
            self.refresh()
            return True
        match = re.fullmatch(r"WORKER_EXIT (\S+) code=(-?\d+) kind=(.*)", line)
        if match:
            tid, code = match[1], int(match[2])
            if tid in self.tasks:
                self.tasks[tid].update(event="complete" if code == 0 else "paused" if code == 75 else "failed",
                                       authority="exit0" if code == 0 else None)
                self.lanes = {d: t for d, t in self.lanes.items() if t != tid}
            self.refresh()
            return True
        if line.startswith("RETRY_SAME_CONFIG "):
            tid = line.split(" ", 1)[1]
            if tid in self.tasks:
                self.tasks[tid].update(event="retry", authority=None, state="queued")
                self.lanes = {d: t for d, t in self.lanes.items() if t != tid}
            return True
        if line.startswith("PROGRESS "):
            return False  # settled is deliberately never interpreted as done.
        if line.startswith("FINAL_NOT_STARTED"):
            self.waiting_for_choice = False
            self.status = self._not_started = line
        elif line.startswith("FINAL_REPORT "):
            self.waiting_for_choice = False
            self._report_received = True
            self.status = "report ready; see training status"
            self.refresh()
        elif line.startswith("RUNNER_EXIT "):
            self.finish()
        return True

    def finish(self, returncode=None):
        self.controller_finished = True
        self.exit_code = returncode
        self.waiting_for_choice = False
        for task in self.tasks.values():
            if task.get("event") == "running":
                task["event"] = "stopped"
        self.lanes.clear()
        self.refresh()
        if returncode in (None, 0) and not self._summary and not self.status.startswith("FINAL_NOT_STARTED"):
            self.status = f"controller exited{'' if returncode is None else ' code=' + str(returncode)}; completion is not a health verdict"

    def eta(self):
        if not self.new_rounds or self.now() <= self.rate_started or not self.lanes:
            return None
        remaining = sum(t["total"] - t["round"] for tid in self.current_ids
                        if (t := self.tasks[tid])["state"] in ("running", "queued", "identity_pending"))
        return max(0, remaining) * (self.now() - self.rate_started) / self.new_rounds

    def lines(self, compact=False):
        current = [self.tasks[tid] for tid in self.current_ids]
        total_rounds = sum(t["total"] for t in current)
        rounds = sum(t["round"] for t in current)
        ratio = rounds / total_rounds if total_rounds else 0.
        bar = "#" * int(ratio * 24) + "-" * (24 - int(ratio * 24))
        queued = sum(t["state"] in ("queued", "identity_pending") for t in current)
        unverified = sum(t["state"] == "unverified_snapshot" for t in current)
        active = sum(t["state"] == "running" for t in current)
        label = "formal" if self.phase == "final" else (
            f"search {self.block_id or 'preparing'} wave={self.wave_index if self.wave_index is not None else '?'}"
            if self.phase == "validation" else "initializing")
        lines = [f"{label} [{bar}] saved_rounds={rounds}/{total_rounds} {ratio:5.1%}",
                 f"done={len(self.completed)}/{self.total} failed={len(self.failed)} paused={len(self.paused)} active={active} queued={queued} unverified={unverified}",
                 f"elapsed={_duration(self.now() - self.started)} ETA~{_duration(self.eta())} (current {'formal plan' if self.phase == 'final' else 'wave'} only)"]
        if self.phase == "validation":
            lines.append(f"cumulative_evaluated={len(self.evaluated_ids)} (fully attempted wave prefixes, including numerical failures); future search size is adaptive")
        lines.append(self.status)
        for device in self.devices:
            tid = self.lanes.get(device)
            if tid and tid in self.tasks:
                task = self.tasks[tid]
                label = tid if not compact else (tid.split("_", 2)[1] if "_" in tid else tid)
                lines.append(f"{device} ({self.mapping.get(device, device)}): round={task['round']}/{task['total']} {label}")
            else:
                lines.append(f"{device} ({self.mapping.get(device, device)}): idle / awaiting controller")
        return lines
