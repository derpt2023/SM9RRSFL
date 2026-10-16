"""Enhanced terminal display for fair tuning with per-GPU tracking.

This module extends fair_tuning_progress_display.py to show detailed GPU status
including partition type, client count, method, and ratio for each active device.
"""
from contextlib import contextmanager
import json
import os
import re
import shutil
import sys
from threading import Lock, Thread, Event
from time import perf_counter
import unicodedata


def _display_width(text):
    return sum(0 if unicodedata.combining(char) else
               2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
               for char in text)


def _fit_line(message, max_width=None):
    available = max_width or max(0, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)
    if not available or _display_width(message) <= available:
        return message
    budget = available - 1
    kept, width = [], 0
    for char in message:
        cells = _display_width(char)
        if width + cells > budget:
            break
        kept.append(char)
        width += cells
    return "".join(kept) + "…"


def _format_duration(seconds):
    if seconds is None or seconds < 0:
        return "estimating"
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


class GPUProgressDisplay:
    """Track GPU status and display detailed progress information."""

    def __init__(self):
        self.lock = Lock()
        self.gpu_tasks = {}  # device -> {partition, clients, method, ratio, round, total}
        self.completed = 0
        self.total = 0
        self.started = perf_counter()
        self.eta_deadline = None
        self.tuning_phase = "unknown"
        self.refresh_interval = 1.0  # seconds
        self._stop_event = Event()
        self._refresh_thread = None
        self._output_handler = None

    def parse_start_event(self, line):
        """Parse START event: partition=X clients=Y method=Z ratio=W.WW"""
        # Example: running partition=iid clients=100 method=sm9rrs ratio=0.20
        match = re.search(r'partition=(\S+)\s+clients=(\d+)\s+method=(\S+)\s+ratio=([\d.]+)', line)
        if not match:
            return None
        return {
            'partition': match.group(1),
            'clients': match.group(2),
            'method': match.group(3),
            'ratio': match.group(4)
        }

    def parse_cuda_event(self, line):
        """Parse cuda_recovery events to track device assignments."""
        if not line.startswith("cuda_recovery="):
            return None
        try:
            data = json.loads(line[14:])  # Skip "cuda_recovery="
            return data
        except (json.JSONDecodeError, ValueError):
            return None

    def update_gpu_status(self, device, config_info):
        """Update the status of a specific GPU."""
        with self.lock:
            self.gpu_tasks[device] = config_info

    def remove_gpu_task(self, device):
        """Remove a completed task from GPU tracking."""
        with self.lock:
            self.gpu_tasks.pop(device, None)

    def update_progress(self, completed, total, phase="unknown"):
        """Update overall progress."""
        with self.lock:
            if completed > self.completed:
                elapsed = perf_counter() - self.started
                if self.completed > 0:
                    seconds_per_task = elapsed / self.completed
                    remaining = total - completed
                    self.eta_deadline = perf_counter() + seconds_per_task * remaining
            self.completed = completed
            self.total = total
            self.tuning_phase = phase

    def start_refresh_thread(self, output_handler):
        """Start background thread for periodic display refresh."""
        self._output_handler = output_handler
        if self._refresh_thread is None or not self._refresh_thread.is_alive():
            self._stop_event.clear()
            self._refresh_thread = Thread(
                target=self._refresh_loop,
                name="gpu-progress-refresh",
                daemon=True
            )
            self._refresh_thread.start()

    def stop_refresh_thread(self):
        """Stop the background refresh thread."""
        self._stop_event.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=2.0)

    def _refresh_loop(self):
        """Background loop to refresh display periodically."""
        while not self._stop_event.wait(self.refresh_interval):
            if self._output_handler and self._output_handler.tty:
                with self._output_handler._process_lock():
                    if not self._output_handler.partial_log_line:
                        self._output_handler._show()

    def format_display(self):
        """Format the complete display with progress bar and GPU status."""
        with self.lock:
            lines = []

            # Overall configuration progress
            ratio = self.completed / self.total if self.total > 0 else 0
            filled = int(round(24 * ratio))
            bar = "#" * filled + "-" * (24 - filled)
            elapsed = perf_counter() - self.started
            eta_text = _format_duration(self.eta_deadline - perf_counter() if self.eta_deadline else None)

            lines.append(f"[{bar}] {self.completed}/{self.total} {ratio:5.1%} "
                        f"elapsed={_format_duration(elapsed)} eta={eta_text}")

            # Calculate total rounds progress
            total_rounds = 0
            completed_rounds = 0
            if self.gpu_tasks:
                for task in self.gpu_tasks.values():
                    if 'round' in task and 'total_rounds' in task:
                        completed_rounds += task['round']
                        total_rounds += task['total_rounds']

            # Total rounds progress bar
            if total_rounds > 0:
                round_ratio = completed_rounds / total_rounds
                round_filled = int(round(24 * round_ratio))
                round_bar = "#" * round_filled + "-" * (24 - round_filled)
                lines.append(f"[{round_bar}] {completed_rounds}/{total_rounds} rounds {round_ratio:5.1%}")

            lines.append(f"phase={self.tuning_phase} active_gpus={len(self.gpu_tasks)}")

            # Per-GPU status
            if self.gpu_tasks:
                lines.append("─" * min(80, shutil.get_terminal_size(fallback=(80, 24)).columns - 1))
                for device in sorted(self.gpu_tasks.keys()):
                    task = self.gpu_tasks[device]
                    # Show device with actual ID (cuda:0, cuda:1, etc.)
                    device_display = device if ':' in device else f"{device}:?"

                    # Build round progress bar for this GPU
                    if 'round' in task and 'total_rounds' in task:
                        r = task['round']
                        total = task['total_rounds']
                        r_ratio = r / total if total > 0 else 0
                        r_filled = int(round(24 * r_ratio))
                        r_bar = "#" * r_filled + "-" * (24 - r_filled)
                        round_progress = f"[{r_bar}] {r}/{total} {r_ratio:5.1%}"
                    else:
                        round_progress = "waiting..."

                    gpu_line = (f"{device_display}: {task['partition']:<10} "
                               f"n={task['clients']:<3} {task['method']:<8} "
                               f"ratio={task['ratio']:<4} {round_progress}")
                    lines.append(_fit_line(gpu_line))

            return "\n".join(lines)


class _EnhancedOutput:
    """Enhanced output handler with GPU tracking."""

    def __init__(self, stream, gpu_display):
        self.stream = stream
        self.gpu_display = gpu_display
        self.tty = bool(getattr(stream, "isatty", lambda: False)())
        self.lock = Lock()
        self.pid = os.getpid()
        self.line_visible = False
        self.partial_log_line = False
        self.current_device = None
        self.current_config = None

    def _process_lock(self):
        if self.pid != os.getpid():
            self.pid = os.getpid()
            self.lock = Lock()
            self.tty = False
            self.line_visible = False
            self.partial_log_line = False
        return self.lock

    def _clear(self):
        """Clear the previous display lines."""
        if self.line_visible and self.line_visible > 0:
            # Move cursor up and clear each line
            for i in range(self.line_visible):
                if i > 0:
                    self.stream.write("\033[A")  # Move up
                self.stream.write("\r\033[K")  # Clear line
            self.line_visible = 0
            self.stream.flush()

    def _show(self):
        if self.tty and not self.partial_log_line:
            self._clear()
            display = self.gpu_display.format_display()
            lines = display.split("\n")
            self.stream.write(display)
            self.line_visible = len(lines)

    def write(self, text):
        if not isinstance(text, str):
            raise TypeError("write() argument must be str")
        if not text:
            return 0

        with self._process_lock():
            # Parse events to update GPU tracking
            for line in text.splitlines():
                # Try parsing as JSON event first (new format)
                if line.startswith("progress_event="):
                    try:
                        event_data = json.loads(line[15:])  # Skip "progress_event="
                        event_type = event_data.get("type")

                        if event_type == "START":
                            config_info = {
                                'partition': event_data.get('partition', 'unknown'),
                                'clients': str(event_data.get('clients', '?')),
                                'method': event_data.get('method', 'unknown'),
                                'ratio': f"{event_data.get('ratio', 0.0):.2f}",
                                'round': 0,
                                'total_rounds': event_data.get('rounds', 150)  # Default 150
                            }
                            device = event_data.get('device')
                            if device:
                                self.gpu_display.update_gpu_status(device, config_info)

                        elif event_type == "ROUND":
                            device = event_data.get('device')
                            round_num = event_data.get('round')
                            if device and device in self.gpu_display.gpu_tasks and round_num:
                                self.gpu_display.gpu_tasks[device]['round'] = round_num

                        elif event_type == "FINISH":
                            device = event_data.get('device')
                            if device:
                                self.gpu_display.remove_gpu_task(device)
                                self.gpu_display.update_progress(
                                    self.gpu_display.completed + 1,
                                    self.gpu_display.total,
                                    self.gpu_display.tuning_phase
                                )
                    except (json.JSONDecodeError, ValueError, KeyError):
                        pass  # Ignore malformed events

                # Fallback to old format for backward compatibility
                else:
                    # Check for task start (old format)
                    config_info = self.gpu_display.parse_start_event(line)
                    if config_info:
                        self.current_config = config_info

                    # Check for CUDA events (old format)
                    cuda_event = self.gpu_display.parse_cuda_event(line)
                    if cuda_event:
                        event_type = cuda_event.get("event")
                        if event_type == "cuda_task_reassigned":
                            device = cuda_event.get("next_device")
                            if device and self.current_config:
                                self.gpu_display.update_gpu_status(device, self.current_config)

                    # Check for progress updates (old format)
                    if "configurations=" in line and "jobs=" in line:
                        match = re.search(r'configurations=(\d+)', line)
                        if match:
                            total = int(match.group(1))
                            phase_match = re.search(r'tuning_phase=(\S+)', line)
                            phase = phase_match.group(1) if phase_match else "unknown"
                            self.gpu_display.update_progress(self.gpu_display.completed, total, phase)

            self._clear()
            self.stream.write(text)
            self.partial_log_line = not text.endswith("\n")
            if not self.partial_log_line:
                self._show()
            self.stream.flush()
        return len(text)

    def flush(self):
        with self._process_lock():
            self.stream.flush()

    def finish(self):
        with self._process_lock():
            if self.line_visible:
                self.stream.write("\n")
                self.line_visible = 0
            self.stream.flush()


@contextmanager
def gpu_progress_display():
    """Enhanced progress display with per-GPU tracking."""
    from sm9rrsfl import experiments, fair_tuning

    original_stdout = sys.stdout
    original_experiments = experiments.ProgressReporter
    original_tuning = fair_tuning.ProgressReporter
    gpu_display = GPUProgressDisplay()
    output = _EnhancedOutput(original_stdout, gpu_display)

    class _StdoutProxy:
        def write(self, text):
            return output.write(text)

        def flush(self):
            return output.flush()

        def isatty(self):
            return output.tty

        def __getattr__(self, name):
            return getattr(output.stream, name)

    proxy = _StdoutProxy()
    reporters = []

    class GPUProgressReporter(original_experiments):
        def __init__(self, **kwargs):
            stream = kwargs.get("stream")
            if stream in (proxy, original_stdout):
                self._gpu_display = gpu_display
                kwargs["mode"] = "log"  # Disable built-in live mode
            else:
                self._gpu_display = None
            super().__init__(**kwargs)
            reporters.append(self)

        def finish_item(self, label: str) -> None:
            super().finish_item(label)
            if self._gpu_display:
                self._gpu_display.update_progress(self.completed, self.total)

    sys.stdout = proxy
    experiments.ProgressReporter = GPUProgressReporter
    fair_tuning.ProgressReporter = GPUProgressReporter

    # Start background refresh thread
    gpu_display.start_refresh_thread(output)

    try:
        yield
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup_error = None
        try:
            # Stop refresh thread first
            gpu_display.stop_refresh_thread()

            for reporter in reversed(reporters):
                try:
                    reporter.close()
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
            try:
                output.finish()
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
        finally:
            sys.stdout = original_stdout
            experiments.ProgressReporter = original_experiments
            fair_tuning.ProgressReporter = original_tuning
        if cleanup_error is not None and not active_error:
            raise cleanup_error


if __name__ == "__main__":
    """Run as a pipe to display GPU progress."""
    gpu_display = GPUProgressDisplay()
    output = _EnhancedOutput(sys.stdout, gpu_display)

    # Start background refresh thread
    gpu_display.start_refresh_thread(output)

    try:
        for line in sys.stdin:
            output.write(line)
    except KeyboardInterrupt:
        pass
    finally:
        gpu_display.stop_refresh_thread()
        output.finish()

