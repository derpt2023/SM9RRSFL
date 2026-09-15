#!/usr/bin/env python3
"""Display progress while the unchanged six-method runner trains/resumes.

All ordinary options are forwarded to run_cifar_six_from_scratch.py. This file
is deliberately outside that runner's source fingerprint: changing the display
must not invalidate existing experiment identities or checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
from time import monotonic

ORPHAN_TERM_AFTER = 2.
ORPHAN_KILL_AFTER = 5.

RESUME = re.compile(r"from_completed_round=(\d+)")
ROUND = re.compile(r"^ROUND (\S+) round=(\d+) accuracy=(\S+) nonfinite=(\d+)$")
START = re.compile(r"^START (\S+) device=(\S+)$")
PHASE = re.compile(r"^(VALIDATION|FINAL)_RUNS (\d+)(?: rounds=(\d+))?")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def physical_mapping(devices, visible=None):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") if visible is None else visible
    tokens = [part.strip() for part in visible.split(",")] if visible is not None else None
    result = {}
    for device in devices:
        match = re.fullmatch(r"cuda:(\d+)", device)
        index = int(match.group(1)) if match else -1
        if tokens is None:
            result[device] = "CUDA ordinal " + str(index) + " (unmasked)"
        elif 0 <= index < len(tokens) and tokens[index] and tokens[index] != "-1":
            result[device] = "visible GPU " + tokens[index]
        else:
            result[device] = "unavailable in CUDA_VISIBLE_DEVICES"
    return result


def duration(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "estimating"
    seconds = round(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"


class Progress:
    def __init__(self, devices, now=monotonic, visible=None):
        self.now, self.devices = now, devices
        self.mapping = physical_mapping(devices, visible)
        self.output = None
        self.phase, self.total, self.default_rounds = "initializing", 0, 100
        self.tasks, self.lanes = {}, {}
        self.completed, self.failed = set(), set()
        self.failure_count = 0
        self.started = self.rate_started = now()
        self.new_rounds = 0
        self.seen_round = set()
        self.status = "loading data / checking experiment identity"
        self.rough_eta = None

    def task(self, task_id):
        return self.tasks.setdefault(task_id, {"round": 0, "total": self.default_rounds,
                                              "accuracy": None, "nonfinite": None})

    def begin_phase(self, phase, total, rounds=100):
        self.phase, self.total, self.default_rounds = phase, total, rounds
        self.tasks, self.lanes = {}, {}
        self.completed, self.failed, self.seen_round = set(), set(), set()
        self.failure_count = self.new_rounds = 0
        self.started = self.rate_started = self.now()
        self.rough_eta = None
        self.status = "running; completion is not a health verdict"
        if self.output is None:
            return
        plan = read_json(self.output / f"{phase}_plan.json")
        for row in plan.get("tasks", []):
            task_id = row["task_id"]
            task = self.task(task_id)
            task["total"] = row.get("config", {}).get("rounds", rounds)
            folder = self.output / "tasks" / task_id
            # Read only small metadata. Never unpickle models in the display.
            if (folder / ".completed_results.pickle").is_file():
                self.completed.add(task_id)
                task["round"] = task["total"]
                continue
            for path in sorted(folder.glob("attempt_*.json"), reverse=True):
                attempt = read_json(path)
                if "last_completed_round" in attempt:
                    task["round"] = min(task["total"], max(0, int(attempt["last_completed_round"])))
                    record = attempt.get("last_round") or {}
                    task["accuracy"] = record.get("accuracy")
                    task["nonfinite"] = record.get("nonfinite_updates")
                    break
        # Historical failure.json describes earlier attempts, not a failure of
        # this invocation; the original runner is about to retry those tasks.

    def finish_task(self, task_id):
        self.completed.add(task_id)
        self.failed.discard(task_id)
        self.task(task_id)["round"] = self.task(task_id)["total"]
        for device, current in list(self.lanes.items()):
            if current == task_id:
                del self.lanes[device]

    def reconcile_phase(self):
        if self.output:
            for task_id in self.tasks:
                folder = self.output / "tasks" / task_id
                if (folder / ".completed_results.pickle").is_file():
                    self.finish_task(task_id)
                elif (folder / "failure.json").is_file():
                    self.failed.add(task_id)
        self.lanes.clear()

    def consume(self, original):
        """Return True for an important line that should remain in the terminal."""
        line = original.rstrip("\r\n")
        worker = None
        if line.startswith("[") and "] " in line:
            worker, line = line[1:].split("] ", 1)
        if line.startswith("SIX_METHOD_OUTPUT "):
            self.output = Path(line.split(" ", 1)[1])
            return True
        match = PHASE.match(line)
        if match:
            self.begin_phase(match[1].lower(), int(match[2]), int(match[3] or 100))
            return True
        match = START.match(line)
        if match:
            task_id, device = match.groups()
            prior = self.lanes.get(device)
            if prior and prior != task_id and prior not in self.completed:
                self.failed.add(prior)
            self.lanes[device] = task_id
            self.task(task_id)
            self.failed.discard(task_id)
            return True
        resume = RESUME.search(line)
        if worker and resume:
            self.task(worker)["round"] = int(resume[1])
            self.seen_round.add(worker)
        match = ROUND.match(line)
        if match:
            task_id, rd, accuracy, nonfinite = match.groups()
            task = self.task(task_id)
            rd = min(task["total"], int(rd))
            # The first observed round may be a restored checkpoint. Establish
            # a per-task baseline rather than counting cached work in the ETA.
            if task_id in self.seen_round:
                self.new_rounds += max(0, rd - task["round"])
            else:
                self.seen_round.add(task_id)
            task.update(round=rd, accuracy=accuracy, nonfinite=int(nonfinite))
            return False
        if line.startswith(("COMPLETED ", "ALREADY_COMPLETED ")):
            self.finish_task(line.split(" ", 1)[1])
            return True
        if line.startswith("PROGRESS "):
            if self.output:
                value = read_json(self.output / f"{self.phase}_progress.json")
                self.failure_count = value.get("worker_failures_this_invocation", self.failure_count)
                self.rough_eta = value.get("rough_remaining_seconds")
            return False
        if line.startswith(("VALIDATION_STATUS ", "FINAL_STATUS ", "FINAL_NOT_STARTED:")):
            self.status = line
            self.reconcile_phase()
        if worker and line.startswith("Traceback (most recent call last):"):
            self.failed.add(worker)
        # Forward every unrecognized line, especially errors and tracebacks.
        return True

    def eta(self):
        elapsed = self.now() - self.rate_started
        if self.new_rounds and elapsed > 0:
            remaining = sum(max(0, task["total"] - task["round"])
                            for name, task in self.tasks.items()
                            if name not in self.failed and name not in self.completed)
            remaining += max(0, self.total - len(self.tasks)) * self.default_rounds
            return remaining * elapsed / self.new_rounds
        return self.rough_eta

    def lines(self, compact=False):
        failures = max(self.failure_count, len(self.failed))
        settled = min(self.total, len(self.completed) + failures)
        denominator = sum(task["total"] for task in self.tasks.values())
        denominator += max(0, self.total - len(self.tasks)) * self.default_rounds
        rounds = sum(task["round"] for task in self.tasks.values())
        ratio = min(1., rounds / denominator) if denominator else 0.
        fill = int(24 * ratio)
        bar = "#" * fill + "-" * (24 - fill)
        lines = [f"{self.phase} [{bar}] rounds={rounds}/{denominator} {ratio:5.1%}",
                 f"done={len(self.completed)}/{self.total} failed={failures} settled={settled}/{self.total}",
                 f"elapsed={duration(self.now()-self.started)} ETA~{duration(self.eta())}"]
        for device in self.devices:
            task_id = self.lanes.get(device)
            if task_id:
                task = self.task(task_id)
                accuracy = task["accuracy"] if task["accuracy"] is not None else "?"
                nf = task["nonfinite"] if task["nonfinite"] is not None else "?"
                detail = f"round={task['round']}/{task['total']} acc={accuracy} nonfinite={nf}"
            else:
                detail = "idle / phase summary"
            lines.append(f"{device} ({self.mapping[device]}): {detail}")
            if task_id:
                if compact:
                    candidate = task_id.split("_")[1] if "_" in task_id else task_id
                    lines[-1] += " " + candidate
                else:
                    lines.append("  " + task_id)
        return lines


class Display:
    def __init__(self, stream=sys.stdout, mode="auto"):
        self.stream = stream
        self.tty = mode == "live" or (mode == "auto" and bool(stream.isatty()))
        self.height = 0

    def clear(self):
        if self.height:
            self.stream.write(f"\x1b[{self.height}F")
            for _ in range(self.height):
                self.stream.write("\x1b[2K\n")
            self.stream.write(f"\x1b[{self.height}F")
            self.height = 0

    def message(self, line):
        self.clear()
        self.stream.write(line.rstrip("\r\n") + "\n")
        self.stream.flush()

    def render(self, progress):
        self.clear()
        lines = progress.lines()
        if self.tty:
            size = shutil.get_terminal_size((160, 24))
            width, height = max(1, size.columns - 1), max(1, size.lines - 1)
            if len(lines) > height:
                lines = progress.lines(compact=True)
            if len(lines) > height:
                lines = lines[:height - 1] + ["... remaining GPU lanes omitted in this short terminal; see raw log"]
            for line in lines:
                self.stream.write(line[:width] + "\n")
            self.height = len(lines)
        else:
            self.stream.write(" | ".join(lines) + "\n")
        self.stream.flush()


def signal_group(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def cleanup_group(process, reader):
    """Bound cleanup even if the display itself fails or a controller dies."""
    if process.poll() is None or reader.is_alive():
        signal_group(process, signal.SIGINT)
    try:
        process.wait(timeout=5.)
    except subprocess.TimeoutExpired:
        signal_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=3.)
        except subprocess.TimeoutExpired:
            signal_group(process, signal.SIGKILL)
            process.wait()
    reader.join(timeout=2.)
    if reader.is_alive():
        signal_group(process, signal.SIGTERM)
        reader.join(timeout=2.)
    if reader.is_alive():
        signal_group(process, signal.SIGKILL)
        reader.join(timeout=2.)


def monitor(command, devices, *, refresh=1., log_interval=15., stream=sys.stdout, mode="auto"):
    """Run the original parent in its own group; forward signals and await it."""
    progress, display = Progress(devices), Display(stream, mode)
    display.message(f"GPU_LANES {len(devices)} " + ", ".join(f"{d} -> {p}" for d, p in progress.mapping.items()))
    display.message("Progress observes unchanged training. done = saved runs, not passed health checks; ETA is approximate.")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1,
                               start_new_session=True)
    messages = queue.Queue()
    def read_output():
        try:
            for line in process.stdout:
                messages.put(line)
        finally:
            messages.put(None)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    raw_log, buffered = None, []
    previous_handlers, signal_count = {}, [0]
    def interrupted(signum, _frame):
        signal_count[0] += 1
        # First Ctrl+C lets the original parent stop scheduling and reap its
        # workers. Additional interruption requests TERM for the same group.
        signal_group(process, signum if signal_count[0] == 1 else signal.SIGTERM)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, interrupted)
    interval = refresh if display.tty else log_interval
    last_display = monotonic() - interval
    eof = False
    exited_at = None
    try:
        while not eof:
            try:
                line = messages.get(timeout=min(.25, interval))
            except queue.Empty:
                line = ""
            if line is None:
                eof = True
            elif line:
                if raw_log:
                    raw_log.write(line)
                    raw_log.flush()
                else:
                    buffered.append(line)
                important = progress.consume(line)
                if raw_log is None and progress.output is not None:
                    # SIX_METHOD_OUTPUT is emitted after immutable manifest
                    # creation. Earlier log creation would make fresh output
                    # nonempty and violate the original runner's safety check.
                    folder = progress.output / "progress_display_logs"
                    try:
                        folder.mkdir(exist_ok=True)
                        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
                        raw_path = folder / f"{stamp}_{os.getpid()}.log"
                        raw_log = raw_path.open("x", encoding="utf-8")
                        raw_log.writelines(buffered)
                        raw_log.flush()
                        buffered.clear()
                        display.message("PROGRESS_RAW_LOG " + str(raw_path))
                    except OSError as exc:
                        display.message(f"Progress log unavailable: {exc}; task worker.log remains unchanged.")
                if important:
                    display.message(line)
            now = monotonic()
            if process.poll() is not None and not eof:
                exited_at = now if exited_at is None else exited_at
                # A dead controller must not leave workers holding the stdout
                # pipe (or GPUs). All descendants inherit its separate group.
                if now - exited_at > ORPHAN_TERM_AFTER:
                    signal_group(process, signal.SIGTERM)
                if now - exited_at > ORPHAN_KILL_AFTER:
                    signal_group(process, signal.SIGKILL)
            if now - last_display >= interval or eof:
                display.render(progress)
                last_display = now
        code = process.wait()
        display.clear()
        display.message(f"RUNNER_EXIT {code}; {progress.status}")
        return 128 + (-code) if code < 0 else code
    finally:
        cleanup_group(process, reader)
        if not reader.is_alive():
            process.stdout.close()
        if raw_log:
            raw_log.close()
        display.clear()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def discover_gpus(repo):
    """Probe in a short-lived process; never initialize CUDA in the display."""
    code = ("import json, sm9rrsfl, torch; "
            "print('GPU_PROBE_JSON '+json.dumps([dict(logical_device='cuda:'+str(i), "
            "name=torch.cuda.get_device_properties(i).name, "
            "compute_capability=[torch.cuda.get_device_properties(i).major, "
            "torch.cuda.get_device_properties(i).minor], "
            "total_memory_bytes=int(torch.cuda.get_device_properties(i).total_memory)) "
            "for i in range(torch.cuda.device_count())]))")
    result = subprocess.run([sys.executable, "-c", code], cwd=repo,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=60, check=False)
    matches = [line.split(" ", 1)[1] for line in result.stdout.splitlines()
               if line.startswith("GPU_PROBE_JSON ")]
    if result.returncode or not matches:
        raise ValueError("CUDA discovery failed: " + (result.stderr or result.stdout).strip())
    found = json.loads(matches[-1])
    if not found:
        raise ValueError("no CUDA GPUs are visible; check CUDA_VISIBLE_DEVICES and the CUDA PyTorch installation")
    return found


def select_devices(found, requested, recorded=None):
    """Respect the original runner's same-device-capability resume contract."""
    by_device = {row["logical_device"]: row for row in found}
    fields = ("name", "compute_capability", "total_memory_bytes")
    def key(row):
        return json.dumps([row.get(field) for field in fields], sort_keys=True)
    if requested != ["auto"]:
        if (len(requested) != len(set(requested))
                or any(d not in by_device for d in requested)):
            raise ValueError("--devices must be auto or distinct available logical CUDA devices")
        identities = {key(by_device[d]) for d in requested}
        if len(identities) != 1 or (recorded and key(recorded) not in identities):
            raise ValueError("requested GPUs have incompatible model/capability/memory; the original runner requires matching hardware")
        return requested, []
    if recorded:
        wanted = key(recorded)
    else:
        # The original runner intentionally rejects mixing numerical hardware
        # identities. Use the largest compatible group, stable on ties.
        groups = {}
        for row in found:
            groups.setdefault(key(row), []).append(row)
        wanted = max(groups, key=lambda k: len(groups[k]))
    chosen = [row["logical_device"] for row in found if key(row) == wanted]
    skipped = [row["logical_device"] for row in found if key(row) != wanted]
    if not chosen:
        raise ValueError("no visible GPU matches the saved experiment GPU model/capability/memory; preserve the existing output")
    return chosen, skipped


def forwarded_devices(arguments, chosen):
    """Replace only --devices; preserve the original options and ordering."""
    result = []
    skipping = False
    for arg in arguments:
        if arg == "--devices" or arg.startswith("--devices="):
            skipping = True
            continue
        if skipping and not arg.startswith("-"):
            continue
        skipping = False
        result.append(arg)
    return [*result, "--devices", *chosen]


def main(argv=None):
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False,
        epilog="Wrapper default: --devices auto selects all compatible visible GPUs. "
               "Explicit --devices cuda:0 cuda:1 selects exactly those cards. "
               "CUDA_VISIBLE_DEVICES is honored; no hidden remapping or CPU fallback.")
    parser.add_argument("--progress-mode", choices=("auto", "live", "log"), default="auto")
    parser.add_argument("--progress-interval", "--progress-refresh", type=float, default=1.)
    parser.add_argument("--progress-log-interval", type=float, default=15.)
    options, forwarded = parser.parse_known_args(argv)
    if not math.isfinite(options.progress_interval) or options.progress_interval < .1:
        parser.error("--progress-interval must be finite and at least 0.1 seconds")
    if not math.isfinite(options.progress_log_interval) or options.progress_log_interval < 1.:
        parser.error("--progress-log-interval must be finite and at least 1 second")
    # Only display metadata is interpreted here; the original runner validates
    # the config and its complete immutable experiment before any training.
    view = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    view.add_argument("--devices", nargs="+", default=["auto"])
    view.add_argument("--config", type=Path, default=repo / "configs/cifar10_six_original_v2.json")
    view.add_argument("--output", type=Path)
    metadata, _ = view.parse_known_args(forwarded)
    if any(arg == "--worker" or arg.startswith("--worker=") for arg in forwarded):
        parser.error("invoke this wrapper as a parent, not an internal --worker")
    runner = str(repo / "run_cifar_six_from_scratch.py")
    if "--help" in forwarded or "-h" in forwarded:
        parser.print_help()
        return subprocess.call([sys.executable, runner, "--help"])
    try:
        found = discover_gpus(repo)
        spec = read_json(metadata.config)
        output = metadata.output or repo / spec.get("output_dir", "outputs/cifar10_six_original_v2")
        recorded = read_json(output / "execution_environment.json").get("actual_compute_device")
        chosen, skipped = select_devices(found, metadata.devices, recorded)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        parser.error(str(exc))
    print("GPU_SELECTION " + json.dumps({"requested": metadata.devices, "selected": chosen,
          "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "detected": found,
          "skipped_incompatible": skipped}, ensure_ascii=False), flush=True)
    if skipped:
        print("GPU_NOTE Skipped incompatible cards to preserve the original runner's hardware identity: "
              + ", ".join(skipped), flush=True)
    command = [sys.executable, "-u", runner, *forwarded_devices(forwarded, chosen)]
    return monitor(command, chosen, refresh=options.progress_interval,
                   log_interval=options.progress_log_interval, mode=options.progress_mode)


if __name__ == "__main__":
    # Use exactly the runner's interpreter selection, without importing its
    # numerical code or changing CUDA / training settings in this process.
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))
    raise SystemExit(main())
