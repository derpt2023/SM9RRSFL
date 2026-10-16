"""A scoped terminal dashboard for the existing schema-3 fair tuner.

Observe the runner's task commits and round callbacks. No training state,
checkpoint identity, device scheduling, or selection policy is changed.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import math
import os
import shutil
import sys
from threading import Event, RLock, Thread, local
from time import monotonic

from fair_tuning_progress_display import _display_width, _Output, _StdoutProxy
from sm9rrsfl import experiments, fair_tuning
from sm9rrsfl.config_runner import parameters_to_argv
from sm9rrsfl.tuning_resume import runtime_config_key


def duration(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "estimating"
    hours, rest = divmod(round(seconds), 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"


def fit(text, width):
    if _display_width(text) <= width:
        return text
    chars, cells = [], 0
    for char in text:
        size = _display_width(char)
        if cells + size > max(0, width - 1):
            break
        chars.append(char)
        cells += size
    return "".join(chars) + ("…" if width else "")


def bar(done, total, width):
    fraction = min(1., max(0., done / total)) if total else 0.
    filled = int(width * fraction)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def plan(spec):
    """Use the runner's actual expansion, without loading data or touching CUDA."""
    args = experiments.parse_args(parameters_to_argv(spec.shared_parameters))
    bases = experiments.build_experiment_configs(args)
    if spec.auto_ours:
        raise fair_tuning.FairTuningError("This dashboard requires an explicit Ours grid")
    validation = fair_tuning.build_validation_tasks(spec, bases)
    final_count = len(bases) * len(spec.final_seeds) if spec.run_final_evaluation else 0
    return args, validation, {
        "validation_tasks": len(validation),
        "final_tasks": final_count,
        "total_tasks": len(validation) + final_count,
        "validation_rounds": sum(t.config.rounds for t in validation),
        "final_rounds": sum(c.rounds for c in bases) * len(spec.final_seeds)
        if spec.run_final_evaluation else 0,
    }


@dataclass
class Task:
    task: object
    round: int = 0
    committed: bool = False
    measuring: bool = False


class Progress:
    def __init__(self, spec, *, now=monotonic):
        self.args, _, self.work = plan(spec)
        self.now = now
        self.started = now()
        self.lock = RLock()
        self.tasks = {}
        self.lanes = {}
        self.devices = set()
        self.gpu_slots = None
        self.gpu_memory = {}
        self.gpu_query_failures = {}
        self.required_gpu_free_mb = None
        self.dispatch_error = None
        self.phase = "loading_data"
        self.status = "running"
        self.new_rounds = 0
        self.failed_attempts = 0

    @staticmethod
    def key(phase, config):
        return phase, runtime_config_key(config)

    def gpu_plan(self, report):
        with self.lock:
            self.devices = set(report["pool_devices"])
            self.gpu_slots = report["target_jobs"]
            self.gpu_memory = {row["device"]: row for row in report.get("gpu_memory_mb", [])}
            self.gpu_query_failures = report.get("query_failures", {})
            self.required_gpu_free_mb = report.get("required_gpu_free_mb")

    def scheduler_event(self, kind, task, **details):
        fatal = kind == "cuda_worker_failed" or (kind == "cuda_oom" and not details.get("retry", True))
        if not fatal:
            return
        with self.lock:
            self.dispatch_error = {
                "device": str(task.config.device) if task is not None else "CUDA",
                "error": str(details.get("error", kind)).splitlines()[0],
            }
            self.status = "draining_after_worker_error"

    def register(self, tasks):
        with self.lock:
            if tasks:
                if self.phase != tasks[0].phase:
                    self.dispatch_error = None
                self.phase = tasks[0].phase
            for task in tasks:
                self.devices.add(str(task.config.device))
                key = self.key(task.phase, task.config)
                if key in self.tasks:
                    self.tasks[key].task = task
                else:
                    self.tasks[key] = Task(task)

    def start(self, config):
        with self.lock:
            key = self.key(self.phase, config)
            if key not in self.tasks:
                return None
            self.tasks[key].measuring = True
            self.devices.add(str(config.device))
            self.lanes[str(config.device)] = key
            return key

    def resume(self, key, completed_round):
        with self.lock:
            task = self.tasks[key]
            # Only the checkpoint loader supplies this prefix; it adds no ETA samples.
            task.round = min(task.task.config.rounds, max(0, int(completed_round)))

    def advance(self, key, completed_round):
        with self.lock:
            task = self.tasks[key]
            value = min(task.task.config.rounds, max(0, int(completed_round)))
            if value > task.round:
                self.new_rounds += value - task.round
                task.round = value

    def stop(self, key, *, failed=False):
        with self.lock:
            if key is None:
                return
            self.tasks[key].measuring = False
            for device in [d for d, k in self.lanes.items() if k == key]:
                del self.lanes[device]
            if failed:
                self.failed_attempts += 1

    def snapshot(self, executions, status):
        with self.lock:
            for task, result in executions:
                row = self.tasks[self.key(task.phase, task.config)]
                row.committed = True
                row.round = min(task.config.rounds, max(0, result.stopped_round))
            self.status = ("draining_after_worker_error" if self.dispatch_error and status == "running"
                           else f"{self.phase}_{status}")

    def totals(self):
        with self.lock:
            return sum(t.round for t in self.tasks.values()), sum(t.committed for t in self.tasks.values())

    def eta(self):
        if self.dispatch_error:
            return None
        done, _ = self.totals()
        remaining = self.work["validation_rounds"] + self.work["final_rounds"] - done
        if remaining <= 0:
            return 0.
        return (self.now() - self.started) * remaining / self.new_rounds if self.new_rounds else None

    def lines(self, width=120, height=24):
        with self.lock:
            done, committed = self.totals()
            total = self.work["validation_rounds"] + self.work["final_rounds"]
            args = self.args
            phase_tasks = [t for t in self.tasks.values() if t.task.phase == self.phase]
            gpu_status = (f"gpu={len(self.lanes)}/{len(self.devices)} slots={self.gpu_slots} "
                          if self.gpu_slots is not None else "")
            eta_status = "eta=paused" if self.dispatch_error else f"eta~{duration(self.eta())}"
            lines = [
                f"{bar(done, total, max(8, min(48, width - 48)))} rounds={done}/{total} "
                f"{done / total if total else 0:5.1%}",
                f"tasks={committed}/{self.work['total_tasks']} elapsed={duration(self.now() - self.started)} "
                f"{eta_status} phase={self.phase}",
                gpu_status + f"phase_tasks={sum(t.committed for t in phase_tasks)}/{len(phase_tasks)} "
                f"failed_attempts={self.failed_attempts} status={self.status}",
                f"n={args.num_clients} E={args.local_epochs} batch={args.batch_size} lr={args.lr:g} "
                f"decay={args.lr_decay:g} boost={args.attack_boost:g} start={args.attack_start_round} K={args.detector_window}",
            ]
            if self.dispatch_error:
                lines.append(f"DISPATCH STOPPED {self.dispatch_error['device']}: {self.dispatch_error['error']}")
            detailed = height >= len(lines) + 2 * len(self.devices)
            for device in sorted(self.devices):
                key = self.lanes.get(device)
                if key is None:
                    if self.dispatch_error:
                        reason = (f"failed: {self.dispatch_error['error']}" if device == self.dispatch_error["device"]
                                  else "dispatch stopped; waiting for active tasks to finish")
                    elif device in self.gpu_query_failures:
                        error = self.gpu_query_failures[device]
                        reason = "memory query failed: " + str(error.get("error", "unknown error")).splitlines()[0]
                    elif device in self.gpu_memory and self.required_gpu_free_mb is not None:
                        free = self.gpu_memory[device]["free"]
                        reason = (f"waiting memory: free={free:.0f}MiB need~{self.required_gpu_free_mb:.0f}MiB"
                                  if free < self.required_gpu_free_mb else f"ready: free={free:.0f}MiB")
                    else:
                        reason = "idle / waiting for scheduler or memory"
                    lines.append(f"{device} {reason}")
                    continue
                row = self.tasks[key]
                task, config = row.task, row.task.config
                lines.append(
                    f"{device} {bar(row.round, config.rounds, 10)} {row.round}/{config.rounds} "
                    f"{task.candidate_id} {config.partition} ratio={config.malicious_ratio:.0%} seed={config.seed}"
                )
                if detailed:
                    if config.method == "sm9rrs":
                        params = (f"q={config.detector_subspace_dim} distance={config.detector_distance_threshold:g} "
                                  f"history={config.detector_history_threshold:g} reject={config.detector_reject_threshold:g}")
                    elif config.method == "vert":
                        params = (f"window={config.vert_history_window} dim={config.vert_projection_dim} "
                                  f"epochs={config.vert_predict_epochs} lr={config.vert_predict_lr:g} "
                                  f"top_k={config.vert_top_k} ratio_prior={config.vert_use_ratio_prior}")
                    elif config.method == "alignins":
                        params = (f"sparsity={config.alignins_sparsity:g} tda={config.alignins_tda_radius:g} "
                                  f"mpsa={config.alignins_mpsa_radius:g}")
                    else:
                        params = "fixed baseline parameters"
                    lines.append(f"  {params}")
            if len(lines) > height:
                omitted = len(lines) - max(0, height - 1)
                lines = lines[:max(0, height - 1)] + [f"... {omitted} more lines; enlarge terminal for GPU details"]
            return [fit(line, width) for line in lines]


class Output(_Output):
    def __init__(self, stream, progress, mode):
        super().__init__(stream)
        # Never send cursor controls to a redirected log, even for mode=live.
        self.tty = self.tty and mode != "log"
        self.progress = progress
        self.enabled = not progress.args.no_progress
        self.height = 0
        self.last_log = float("-inf")

    def _process_lock(self):
        if self.pid != os.getpid():
            self.height = 0
        return super()._process_lock()

    def _clear(self):
        if self.height:
            self.stream.write(f"\033[{self.height}F")
            for _ in range(self.height):
                self.stream.write("\033[2K\n")
            self.stream.write(f"\033[{self.height}F")
            self.height = 0

    def _show(self):
        if self.tty and self.enabled and not self.partial_log_line:
            self._clear()
            size = shutil.get_terminal_size((120, 24))
            lines = self.progress.lines(max(0, size.columns - 1), max(1, size.lines - 1))
            self.stream.write("\n".join(lines) + "\n")
            self.height = len(lines)
            self.stream.flush()

    def refresh(self, *, final=False):
        with self._process_lock():
            if not self.enabled or self.partial_log_line:
                return
            if self.tty:
                self._show()
            elif final or monotonic() - self.last_log >= 30.:
                self.stream.write(" | ".join(self.progress.lines(width=240, height=100)) + "\n")
                self.stream.flush()
                self.last_log = monotonic()

    def finish(self):
        with self._process_lock():
            if self.partial_log_line:
                self.stream.write("\n")
                self.partial_log_line = False
            self._clear()
            self.stream.flush()


@contextmanager
def dashboard(spec, *, mode="auto", refresh=1.):
    """Scope display hooks to this launcher; always restore them on failure."""
    progress = Progress(spec)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    output = Output(original_stdout, progress, progress.args.progress_mode if mode == "auto" else mode)
    stdout_proxy = _StdoutProxy(output)
    # Ordinary warnings/tracebacks share the same display lock but keep stderr's destination.
    class StderrProxy:
        def write(self, text):
            with output._process_lock():
                output._clear()
                result = original_stderr.write(text)
                original_stderr.flush()
                return result

        def flush(self):
            original_stderr.flush()

        def __getattr__(self, name):
            return getattr(original_stderr, name)

    original_phase = fair_tuning.execute_resumable_tuning_phase
    original_measure = fair_tuning.run_measured_experiment
    original_train = experiments.run_experiment
    original_reporter = fair_tuning.ProgressReporter
    owner_pid = os.getpid()
    worker = local()
    stop = Event()

    @wraps(original_phase)
    def phase(dataset, tasks, args, **kwargs):
        progress.register(tasks)
        snapshot = kwargs.get("on_snapshot")

        def observe(executions, fingerprint, status):
            if snapshot is not None:
                snapshot(executions, fingerprint, status)
            progress.snapshot(executions, status)

        kwargs["on_snapshot"] = observe
        output.refresh()
        return original_phase(dataset, tasks, args, **kwargs)

    @wraps(original_measure)
    def measure(dataset, config, **kwargs):
        if os.getpid() != owner_pid:
            # A forked CPU worker must never acquire a lock owned by a parent
            # refresh thread. Its committed result still updates the parent snapshot.
            return original_measure(dataset, config, **kwargs)
        key = progress.start(config)
        worker.key = key
        # This dashboard observes callbacks directly; disable duplicate JSON rendering.
        kwargs["emit_progress_events"] = False
        try:
            return original_measure(dataset, config, **kwargs)
        except BaseException as error:
            progress.stop(key, failed=True)
            if isinstance(error, Exception):
                from .cifar_fair_diagnostics import record_failure
                try:
                    record_failure(error, config, checkpoint_dir=kwargs.get("checkpoint_dir"),
                                   task_id=kwargs.get("task_id"))
                except Exception as diagnostic_error:
                    # An evidence I/O/query failure must not replace training's
                    # original exception or change its retry classification.
                    print(f"gpu_failure_record_error={diagnostic_error}", file=sys.stderr, flush=True)
            raise
        finally:
            progress.stop(key)
            worker.key = None

    @wraps(original_train)
    def train(dataset, config, **kwargs):
        if os.getpid() != owner_pid:
            return original_train(dataset, config, **kwargs)
        key = getattr(worker, "key", None)
        if key is None:
            return original_train(dataset, config, **kwargs)
        resume = kwargs.get("resume_state")
        progress.resume(key, resume.get("completed_round", 0) if resume else 0)
        callback = kwargs.get("checkpoint_callback")

        def observe(state):
            if callback is not None:
                callback(state)
            progress.advance(key, state["completed_round"])

        kwargs["checkpoint_callback"] = observe
        return original_train(dataset, config, **kwargs)

    class QuietReporter(original_reporter):
        def __init__(self, **kwargs):
            kwargs["mode"] = "log"
            super().__init__(**kwargs)

        def _write(self, message, *, final=False):
            pass

    def refresh_loop():
        while not stop.wait(max(.01, refresh)):
            output.refresh()

    thread = Thread(target=refresh_loop, name="cifar-fair-dashboard", daemon=True)
    sys.stdout, sys.stderr = stdout_proxy, StderrProxy()
    fair_tuning.execute_resumable_tuning_phase = phase
    fair_tuning.run_measured_experiment = measure
    experiments.run_experiment = train
    fair_tuning.ProgressReporter = QuietReporter
    thread.start()
    try:
        output.refresh()
        yield progress
        progress.status = "complete"
    except BaseException:
        progress.status = "interrupted_or_failed"
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        stop.set()
        thread.join(timeout=2.)
        try:
            output.refresh(final=True)
            output.finish()
        except BaseException:
            if not active_error:
                raise
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            fair_tuning.execute_resumable_tuning_phase = original_phase
            fair_tuning.run_measured_experiment = original_measure
            experiments.run_experiment = original_train
            fair_tuning.ProgressReporter = original_reporter
