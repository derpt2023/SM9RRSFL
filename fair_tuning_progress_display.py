"""Scoped terminal rendering for the unchanged MNIST fair-tuning runner.

The launcher opts in; importing this module changes nothing. Training, task
counts, ETA estimation, cache identity and checkpoints remain owned by the
original runner. Redirected output uses readable event logs even when the
configuration requests live rendering.
"""
from contextlib import contextmanager
import os
import re
import shutil
import sys
from threading import RLock
import unicodedata


def _display_width(text):
    return sum(0 if unicodedata.combining(char) else
               2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
               for char in text)


def _fit_line(message):
    # Leave the final column free: writing into it can trigger terminal wrap.
    available = max(0, shutil.get_terminal_size(fallback=(80, 24)).columns - 1)
    if not available:
        return ""
    if available < 72:
        message = re.sub(r"^\[[#-]+\]\s*", "", message)
    if _display_width(message) <= available:
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


class _Output:
    def __init__(self, stream):
        self.stream = stream
        self.tty = bool(getattr(stream, "isatty", lambda: False)())
        self.lock = RLock()
        self.pid = os.getpid()
        self.line_visible = False
        self.partial_log_line = False
        self.message = None

    def _process_lock(self):
        # A CPU process pool can fork while the parent's refresh thread holds
        # the lock. Child stdout remains ordinary text; never inherit that lock
        # or redraw a progress line owned by the parent process.
        if self.pid != os.getpid():
            self.pid = os.getpid()
            self.lock = RLock()
            self.tty = False
            self.line_visible = False
            self.partial_log_line = False
            self.message = None
        return self.lock

    def _clear(self):
        if self.line_visible:
            self.stream.write("\r\033[2K")
            self.line_visible = False

    def _show(self):
        if self.tty and self.message is not None and not self.partial_log_line:
            self._clear()
            self.stream.write("\r" + _fit_line(self.message))
            self.line_visible = True

    def write(self, text):
        if not isinstance(text, str):
            raise TypeError("write() argument must be str")
        if not text:
            return 0
        with self._process_lock():
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

    def render(self, message, *, live, final):
        with self._process_lock():
            if live and self.tty:
                self.message = message
                if final and self.partial_log_line:
                    self.stream.write("\n")
                    self.partial_log_line = False
                self._show()
                if final:
                    self.stream.write("\n")
                    self.line_visible = False
                    self.message = None
            else:
                self._clear()
                self.message = None
                if self.partial_log_line:
                    self.stream.write("\n")
                    self.partial_log_line = False
                self.stream.write(message + "\n")
            self.stream.flush()

    def finish(self):
        with self._process_lock():
            if self.line_visible:
                self.stream.write("\n")
                self.line_visible = False
            self.message = None
            self.stream.flush()


class _StdoutProxy:
    def __init__(self, output):
        self._output = output

    def write(self, text):
        return self._output.write(text)

    def flush(self):
        return self._output.flush()

    def isatty(self):
        with self._output._process_lock():
            return self._output.tty

    def __getattr__(self, name):
        return getattr(self._output.stream, name)


@contextmanager
def progress_display():
    """Coordinate this launcher's Python stdout and progress refreshes.

    Restore the original stream and classes, including on exceptions. The
    scope belongs to one runner invocation, not concurrent protocol launches
    in a shared process. Raw file-descriptor writes are outside this adapter.
    """
    from sm9rrsfl import experiments, fair_tuning

    original_stdout = sys.stdout
    original_experiments = experiments.ProgressReporter
    original_tuning = fair_tuning.ProgressReporter
    output = _Output(original_stdout)
    proxy = _StdoutProxy(output)
    reporters = []

    class TerminalProgressReporter(original_experiments):
        def __init__(self, **kwargs):
            stream = kwargs.get("stream")
            if stream is proxy or stream is original_stdout:
                self._display_output = output
            else:
                self._display_output = _Output(stream if stream is not None else sys.stderr)
            with self._display_output._process_lock():
                if not self._display_output.tty and kwargs.get("mode", "auto") == "live":
                    kwargs["mode"] = "log"
            super().__init__(**kwargs)
            reporters.append(self)

        def _write(self, message, *, final=False):
            self._display_output.render(message, live=self.live, final=final)

    sys.stdout = proxy
    experiments.ProgressReporter = TerminalProgressReporter
    fair_tuning.ProgressReporter = TerminalProgressReporter
    try:
        yield
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup_error = None
        try:
            # Do not hold the shared output lock while joining refresh threads.
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
