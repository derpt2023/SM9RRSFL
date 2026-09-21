#!/usr/bin/env python3
"""Display progress while the configured six-method runner trains/resumes.

The expanded v4 config selects run_cifar_six_mnist_gate.py; the 630-run v3
config selects run_cifar_six_630.py; existing v2 configs
keep run_cifar_six_from_scratch.py. Ordinary options are forwarded. This file
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
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
from time import monotonic

ORPHAN_TERM_AFTER = 2.
ORPHAN_KILL_AFTER = 5.
RESOURCE_TERM_AFTER = 10.
RESOURCE_KILL_AFTER = 15.

RESUME = re.compile(r"from_completed_round=(\d+)")
ROUND = re.compile(r"^ROUND (\S+) round=(\d+) accuracy=(\S+) nonfinite=(\d+)$")
START = re.compile(r"^START (\S+) device=(cuda:\d+)$")
PHASE = re.compile(r"^(VALIDATION|FINAL)_RUNS (\d+)(?: rounds=(\d+))?")
EVENT_BOUNDARY = re.compile(
    r"(?=\[(?:validation|final)_[^\s\]]+\] "
    r"|START (?:validation|final)_\S+ device=cuda:\d+"
    r"|PROGRESS \d+/\d+ rough_remaining_seconds="
    r"|ALREADY_COMPLETED (?:validation|final)_\S+"
    r"|(?:VALIDATION|FINAL)_(?:RUNS \d+|STATUS \S+)"
    r"|FINAL_NOT_STARTED:)"
)
CUDA_OOM = re.compile(r"(?:RuntimeError|OutOfMemoryError):.*(?:CUDA.*out of memory|out of memory.*CUDA)", re.I)


def output_events(original):
    """Undo adjacent print records whose newline writes interleaved.

    The original controller prints from one thread per GPU. A print's text and
    newline are separate writes, so a pipe line can contain multiple records.
    Keep the original bytes in the raw log; only split the display/parser input.
    Look ahead without consuming: a greedy task token must not swallow the
    next ALREADY_COMPLETED or START marker before the scanner can see it.
    """
    line = original.rstrip("\r\n")
    starts = sorted({0, *(match.start() for match in EVENT_BOUNDARY.finditer(line))})
    for start, end in zip(starts, [*starts[1:], len(line)]):
        event = line[start:end]
        if event.strip():
            yield event


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
        self.tasks, self.lanes, self.task_devices = {}, {}, {}
        self.planned_task_ids = None
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

    def accepts_task(self, task_id):
        if self.planned_task_ids is not None:
            return task_id in self.planned_task_ids
        return re.fullmatch(r"[A-Za-z0-9_.:-]+", task_id) is not None

    def begin_phase(self, phase, total, rounds=100):
        self.phase, self.total, self.default_rounds = phase, total, rounds
        self.tasks, self.lanes, self.task_devices = {}, {}, {}
        self.planned_task_ids = None
        self.completed, self.failed, self.seen_round = set(), set(), set()
        self.failure_count = self.new_rounds = 0
        self.started = self.rate_started = self.now()
        self.rough_eta = None
        self.status = "running; completion is not a health verdict"
        if self.output is None:
            return
        plan = read_json(self.output / f"{phase}_plan.json")
        if plan.get("tasks"):
            self.planned_task_ids = {row["task_id"] for row in plan["tasks"]}
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
        if not self.accepts_task(task_id):
            return
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
        if not line.strip():
            return False
        worker = None
        if line.startswith("[") and "] " in line:
            worker, line = line[1:].split("] ", 1)
        if not line.strip():
            return False
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
            if not self.accepts_task(task_id):
                return True
            prior = self.lanes.get(device)
            if prior and prior != task_id and prior not in self.completed:
                self.failed.add(prior)
            self.lanes[device] = task_id
            self.task_devices[task_id] = device
            self.task(task_id)
            self.failed.discard(task_id)
            return True
        resume = RESUME.search(line)
        if worker and resume and self.accepts_task(worker):
            self.task(worker)["round"] = int(resume[1])
            self.seen_round.add(worker)
        match = ROUND.match(line)
        if match:
            task_id, rd, accuracy, nonfinite = match.groups()
            if not self.accepts_task(task_id):
                return True
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
        if worker and self.accepts_task(worker) and line.startswith("Traceback (most recent call last):"):
            self.failed.add(worker)
            for device, task_id in list(self.lanes.items()):
                if task_id == worker:
                    del self.lanes[device]
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


def monitor(command, devices, *, refresh=1., log_interval=15., stream=sys.stdout, mode="auto",
            oom_devices=None, phase_state=None):
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
    previous_handlers, signal_count, user_signal = {}, [0], [None]
    resource_stop = False
    resource_stop_at, resource_signal_stage = None, 0
    def interrupted(signum, _frame):
        signal_count[0] += 1
        if user_signal[0] is None:
            user_signal[0] = signum
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
                events = list(output_events(line))
                # Recover all concatenated events before drawing or reacting
                # to a failure; otherwise cuda:3PROGRESS becomes a false lane.
                important_events = []
                for event in events:
                    if progress.consume(event):
                        important_events.append(event)
                    if phase_state is not None and progress.phase in ("validation", "final"):
                        phase_state["phase"] = progress.phase
                    if oom_devices is not None and CUDA_OOM.search(event) and event.startswith("["):
                        worker = event[1:].split("] ", 1)[0]
                        device = progress.task_devices.get(worker)
                        if device in devices and not signal_count[0]:
                            oom_devices.add(device)
                            if not resource_stop:
                                resource_stop = True
                                resource_stop_at = monotonic()
                                progress.status = f"paused_after_cuda_oom device={device}; incomplete tasks retain their checkpoints"
                                display.message(f"RESOURCE_OOM {worker} device={device}; stopping this controller to preserve checkpoints and avoid repeated dispatch to an unavailable GPU.")
                                signal_group(process, signal.SIGINT)
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
                for event in important_events:
                    display.message(event)
            now = monotonic()
            if resource_stop_at is not None:
                stopping_seconds = now - resource_stop_at
                if stopping_seconds > RESOURCE_KILL_AFTER and resource_signal_stage < 2:
                    display.message("RESOURCE_STOP_TIMEOUT sending SIGKILL to the stopped scheduling group")
                    signal_group(process, signal.SIGKILL)
                    resource_signal_stage = 2
                elif stopping_seconds > RESOURCE_TERM_AFTER and resource_signal_stage < 1:
                    display.message("RESOURCE_STOP_TIMEOUT sending SIGTERM to the stopped scheduling group")
                    signal_group(process, signal.SIGTERM)
                    resource_signal_stage = 1
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
        if signal_count[0]:
            if oom_devices is not None:
                oom_devices.clear()
            return 128 + user_signal[0]
        if resource_stop:
            return 75
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
    """Enumerate first, then initialize each card in its own short process.

    An occupied card may fail in set_device itself. Its failure must not hide
    the remaining cards or leave a CUDA context in this display process.
    """
    def probe(code, *arguments):
        result = subprocess.run([sys.executable, "-c", code, *arguments], cwd=repo,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=60, check=False)
        matches = [line.split(" ", 1)[1] for line in result.stdout.splitlines()
                   if line.startswith("GPU_PROBE_JSON ")]
        if result.returncode or not matches:
            detail = (result.stderr or result.stdout).strip()[-1200:]
            raise ValueError(f"CUDA probe exited {result.returncode}: {detail or 'no probe result'}")
        return json.loads(matches[-1])

    count = probe("import json, sm9rrsfl, torch; print('GPU_PROBE_JSON '+json.dumps(torch.cuda.device_count()))")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("CUDA enumeration returned an invalid device count")
    if not count:
        raise ValueError("no CUDA GPUs are visible; check CUDA_VISIBLE_DEVICES and the CUDA PyTorch installation")
    code = """import json, sys, sm9rrsfl, torch
index = int(sys.argv[1])
row = {'logical_device': 'cuda:'+str(index), 'initialization_ok': False}
try:
    torch.cuda.set_device(index)
    torch.cuda.init()
    props = torch.cuda.get_device_properties(index)
    row.update(name=props.name, compute_capability=[props.major, props.minor],
               total_memory_bytes=int(props.total_memory))
    free, total = torch.cuda.mem_get_info(index)
    row.update(initialization_ok=True, free_memory_bytes=int(free),
               runtime_total_memory_bytes=int(total))
except Exception as exc:
    row['preflight_error'] = type(exc).__name__+': '+str(exc)[:1200]
print('GPU_PROBE_JSON '+json.dumps(row))
"""
    found = []
    for index in range(count):
        try:
            row = probe(code, str(index))
            if not isinstance(row, dict) or row.get("logical_device") != f"cuda:{index}":
                raise ValueError("invalid per-device CUDA probe result")
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            row = {"logical_device": f"cuda:{index}", "initialization_ok": False,
                   "preflight_error": str(exc)}
        found.append(row)
    return found


def select_devices(found, requested, recorded=None, min_free_memory_mib=4096.):
    """Admit initialized cards with memory headroom and compatible hardware."""
    if not math.isfinite(min_free_memory_mib) or min_free_memory_mib < 0:
        raise ValueError("minimum free GPU memory must be finite and nonnegative")
    by_device = {row["logical_device"]: row for row in found}
    fields = ("name", "compute_capability", "total_memory_bytes")
    def key(row):
        return json.dumps([row.get(field) for field in fields], sort_keys=True)
    def resource_error(row):
        if not row.get("initialization_ok"):
            return "CUDA initialization failed: " + row.get("preflight_error", "no successful initialization recorded")
        if any(row.get(field) is None for field in fields):
            return "GPU hardware identity unavailable"
        free = row.get("free_memory_bytes")
        if not isinstance(free, (int, float)) or not math.isfinite(free) or free < 0:
            return "free GPU memory unavailable"
        if free < min_free_memory_mib * 1024 ** 2:
            return f"insufficient free GPU memory: {free / 1024 ** 2:.0f} MiB < {min_free_memory_mib:g} MiB"
        return None
    errors = {device: error for device, row in by_device.items()
              if (error := resource_error(row)) is not None}
    if requested != ["auto"]:
        if (len(requested) != len(set(requested))
                or any(d not in by_device for d in requested)):
            raise ValueError("--devices must be auto or distinct available logical CUDA devices")
        blocked = [f"{device}: {errors[device]}" for device in requested if device in errors]
        if blocked:
            raise ValueError("requested GPUs failed preflight; no devices were silently removed: " + "; ".join(blocked))
        identities = {key(by_device[d]) for d in requested}
        if len(identities) != 1 or (recorded and key(recorded) not in identities):
            raise ValueError("requested GPUs have incompatible model/capability/memory; the original runner requires matching hardware")
        return requested, []
    eligible = [row for row in found if row["logical_device"] not in errors]
    if not eligible:
        details = "; ".join(f"{device}: {reason}" for device, reason in errors.items())
        raise ValueError("no GPU passed initialization and free-memory preflight" + (": " + details if details else ""))
    if recorded:
        wanted = key(recorded)
    else:
        # The original runner intentionally rejects mixing numerical hardware
        # identities. Use the largest compatible group, stable on ties.
        groups = {}
        for row in eligible:
            groups.setdefault(key(row), []).append(row)
        wanted = max(groups, key=lambda k: len(groups[k]))
    chosen = [row["logical_device"] for row in eligible if key(row) == wanted]
    skipped = [{"logical_device": row["logical_device"],
                "reason": errors.get(row["logical_device"], "incompatible GPU model/capability/memory"),
                "free_memory_bytes": row.get("free_memory_bytes")}
               for row in found if row["logical_device"] not in chosen]
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


def forwarded_phase(arguments, phase):
    """Keep formal resource retries from rerunning candidate validation."""
    result = []
    arguments = iter(arguments)
    for arg in arguments:
        if arg == "--phase":
            next(arguments, None)
        elif not arg.startswith("--phase="):
            result.append(arg)
    return [*result, "--phase", phase]


def render_final_report(output, *, training_exit_code=None):
    """Run CSV/JSON reporting separately from the immutable training identity.

    Imports are lazy so validation, GPU retries and the progress display never
    depend on plotting libraries. A report failure does not become a training
    failure or cause another training attempt.
    """
    output = Path(output).resolve()
    retry = shlex.join([sys.executable, str(Path(__file__).resolve()),
                        "--report-only", "--output", str(output)])
    status = {
        "mode": "report_only" if training_exit_code is None else "post_training",
        "source_output": str(output), "training_exit_code": training_exit_code,
        "training_result_unchanged": True, "retry_command": retry,
    }
    print(f"REPORT_START source={output}", flush=True)
    try:
        from experiment_reporting import check_dependencies, generate_report
        check_dependencies()
        result = generate_report(output)
        status.update(status="completed", report=result)
        print("REPORT_COMPLETED " + json.dumps(result, ensure_ascii=False, default=str), flush=True)
        succeeded = True
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        print(f"REPORT_FAILED {type(exc).__name__}: {exc}", flush=True)
        print("REPORT_NOTE Report generation failed; training results and checkpoints are unchanged. "
              "No training retry was requested.", flush=True)
        print(f"REPORT_RETRY {retry}", flush=True)
        succeeded = False
    status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    # Never make a fresh training output nonempty before its manifest exists.
    # Existing manifests, plans, CSVs and checkpoints remain read-only here.
    if output.is_dir() and (output / "manifest.json").is_file():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output,
                    prefix=".report-status-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(status, handle, ensure_ascii=False, indent=2, default=str)
                handle.write("\n")
            temporary.replace(output / "report_generation_status.json")
        except OSError as exc:
            print(f"REPORT_STATUS_WRITE_FAILED {exc}", flush=True)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"REPORT_STATUS_CLEANUP_FAILED {exc}", flush=True)
    return succeeded


def runner_for_spec(repo, spec):
    """Keep historical entry points stable when a new study is introduced."""
    if spec.get("schema_version") == 7:
        return str(repo / "run_cifar_six_relative_best.py")
    if spec.get("schema_version") == 6:
        return str(repo / "run_cifar_six_relative_asr.py")
    if spec.get("schema_version") == 5:
        return str(repo / "run_cifar_six_final_metrics.py")
    if spec.get("schema_version") == 4:
        return str(repo / "run_cifar_six_mnist_gate.py")
    if spec.get("schema_version") == 3 and "mean_dual_gate" in spec:
        return str(repo / "run_cifar_six_630.py")
    return str(repo / "run_cifar_six_from_scratch.py")


def main(argv=None):
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False,
        epilog="Wrapper default: --devices auto selects all compatible visible GPUs. "
               "Explicit --devices cuda:0 cuda:1 selects exactly those cards. "
               "CUDA_VISIBLE_DEVICES is honored; no hidden remapping or CPU fallback.")
    parser.add_argument("--progress-mode", choices=("auto", "live", "log"), default="auto")
    parser.add_argument("--progress-interval", "--progress-refresh", type=float, default=1.)
    parser.add_argument("--progress-log-interval", type=float, default=15.)
    parser.add_argument("--report-only", action="store_true",
                        help="generate HTML and mean figures from completed CSV/JSON results; no CUDA or training")
    parser.add_argument("--min-free-gpu-memory-mib", type=float, default=4096.,
                        help="require this much free GPU memory after CUDA initialization (default: 4096 MiB)")
    options, forwarded = parser.parse_known_args(argv)
    if not math.isfinite(options.progress_interval) or options.progress_interval < .1:
        parser.error("--progress-interval must be finite and at least 0.1 seconds")
    if not math.isfinite(options.progress_log_interval) or options.progress_log_interval < 1.:
        parser.error("--progress-log-interval must be finite and at least 1 second")
    if not math.isfinite(options.min_free_gpu_memory_mib) or options.min_free_gpu_memory_mib < 0:
        parser.error("--min-free-gpu-memory-mib must be finite and nonnegative")
    # Only display metadata is interpreted here; the original runner validates
    # the config and its complete immutable experiment before any training.
    view = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    view.add_argument("--devices", nargs="+", default=["auto"])
    view.add_argument("--config", type=Path, default=repo / "configs/cifar10_six_original_v2.json")
    view.add_argument("--output", type=Path)
    view.add_argument("--phase", choices=("all", "validation", "final"), default="all")
    metadata, _ = view.parse_known_args(forwarded)
    if any(arg == "--worker" or arg.startswith("--worker=") for arg in forwarded):
        parser.error("invoke this wrapper as a parent, not an internal --worker")
    spec = read_json(metadata.config)
    runner = runner_for_spec(repo, spec)
    if "--help" in forwarded or "-h" in forwarded:
        parser.print_help()
        if options.report_only:
            return 0
        return subprocess.call([sys.executable, runner, "--help"])
    if "--plan-only" in forwarded:
        if options.report_only:
            parser.error("--plan-only and --report-only are separate read-only actions")
        code = subprocess.call([sys.executable, runner, *forwarded])
        if code == 0 and spec.get("runtime_estimate"):
            print("RUNTIME_ESTIMATE " + json.dumps(spec["runtime_estimate"], ensure_ascii=False), flush=True)
        return code
    if options.report_only and metadata.output is None and not spec.get("output_dir"):
        parser.error("--report-only requires --output or a readable config containing output_dir")
    output = (metadata.output or repo / spec.get("output_dir", "outputs/cifar10_six_original_v2")).resolve()
    if options.report_only:
        return 0 if render_final_report(output) else 1
    try:
        found = discover_gpus(repo)
        recorded = read_json(output / "execution_environment.json").get("actual_compute_device")
        chosen, skipped = select_devices(found, metadata.devices, recorded, options.min_free_gpu_memory_mib)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        parser.error(str(exc))
    print("GPU_SELECTION " + json.dumps({"requested": metadata.devices, "selected": chosen,
          "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "detected": found,
          "minimum_free_memory_mib": options.min_free_gpu_memory_mib,
          "excluded": skipped}, ensure_ascii=False), flush=True)
    if skipped:
        print("GPU_NOTE Excluded cards: " + "; ".join(
            f"{row['logical_device']}: {row['reason']}" for row in skipped), flush=True)
    print("GPU_NOTE Preflight checks current availability; it does not reserve memory or prevent later contention. "
          "A CUDA OOM stops the current scheduling group and retries on remaining admitted cards, preserving failure records and checkpoints.",
          flush=True)
    admitted = set(chosen)
    # Freeze the initial compatible group even before execution_environment is
    # written. A resource retry cannot silently migrate to different hardware.
    identity = recorded or next(row for row in found if row["logical_device"] == chosen[0])
    excluded_oom = set()
    run_arguments = list(forwarded)
    while True:
        command = [sys.executable, "-u", runner, *forwarded_devices(run_arguments, chosen)]
        oom_devices = set()
        phase_state = {}
        code = monitor(command, chosen, refresh=options.progress_interval,
                       log_interval=options.progress_log_interval, mode=options.progress_mode,
                       oom_devices=oom_devices, phase_state=phase_state)
        if code != 75 or not oom_devices:
            if code == 0 and metadata.phase != "validation" and phase_state.get("phase") == "final":
                render_final_report(output, training_exit_code=code)
            return code
        if phase_state.get("phase") == "final":
            run_arguments = forwarded_phase(run_arguments, "final")
        newly_excluded = (oom_devices & set(chosen)) - excluded_oom
        if not newly_excluded:
            print("GPU_RESOURCE_ABORT OOM did not identify a new active GPU; automatic retries stopped.", flush=True)
            return 75
        excluded_oom.update(newly_excluded)
        remaining = admitted - excluded_oom
        print("GPU_RESOURCE_RETRY " + json.dumps({"excluded_after_oom": sorted(newly_excluded),
              "excluded_this_invocation": sorted(excluded_oom),
              "remaining_initially_admitted": sorted(remaining),
              "phase_at_oom": phase_state.get("phase", "unknown"),
              "failure_records_and_checkpoints": "preserved"}), flush=True)
        if not remaining:
            print("GPU_RESOURCE_ABORT all initially admitted GPUs encountered CUDA OOM; release resources before resuming.", flush=True)
            return 75
        try:
            found = discover_gpus(repo)
            candidates = [row for row in found if row["logical_device"] in remaining]
            chosen, skipped = select_devices(candidates, ["auto"], identity, options.min_free_gpu_memory_mib)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            print(f"GPU_RESOURCE_ABORT remaining-card preflight failed: {exc}", flush=True)
            return 75
        print("GPU_SELECTION " + json.dumps({"requested": metadata.devices, "selected": chosen,
              "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "detected": found,
              "minimum_free_memory_mib": options.min_free_gpu_memory_mib,
              "excluded": skipped, "excluded_after_oom": sorted(excluded_oom),
              "reason": "resource retry on the same experiment output"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    # Use exactly the runner's interpreter selection, without importing its
    # numerical code or changing CUDA / training settings in this process.
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))
    raise SystemExit(main())
