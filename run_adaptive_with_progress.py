#!/usr/bin/env python3
"""Live progress for unchanged CIFAR/Fashion schema-8 controllers.

This display-only entry is intentionally outside scientific source fingerprints.
The child owns validation, training, GPU admission, checkpoints and Y/N answers.
In particular stdin is inherited; the observer never supplies a decision.
"""
from __future__ import annotations

import argparse
import codecs
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
from time import monotonic

from adaptive_progress import AdaptiveProgress
from run_cifar_six_with_progress import Display


REPO = Path(__file__).resolve().parent
PROTOCOL_ENTRIES = {
    "cifar-resnet18-gn-tpe-v1": "run_cifar_adaptive.py",
    "fashion-mnist-resnet18-gn-tpe-v1": "run_fashion_adaptive.py",
}


class OutputFramer:
    """Recognize the existing non-newline prompt without waiting for readline."""
    def __init__(self):
        self.buffer = ""
        self.after_prompt = False

    def feed(self, text, *, final=False):
        self.buffer += text
        records = []
        while self.buffer:
            if self.after_prompt:
                # The trailing prompt space can arrive in its own os.read
                # chunk, before a repeated non-newline invalid-input prompt.
                self.buffer = self.buffer.lstrip(" ")
                if not self.buffer:
                    break
                self.after_prompt = False
            if self.buffer.startswith("CONTINUATION_PROMPT "):
                end = self.buffer.find("[Y/N]")
                if end >= 0:
                    end += len("[Y/N]")
                    records.append(("prompt", self.buffer[:end]))
                    self.buffer = self.buffer[end:]
                    self.after_prompt = True
                    continue
            if "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                records.append(("line", line.rstrip("\r")))
                continue
            if final:
                records.append(("line", self.buffer))
                self.buffer = ""
            break
        return records


def _signal_controller(process, signum):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass


def _stop_controller(process):
    # The controller's own cleanup saves/stops its separate worker sessions
    # within 30 seconds. Give it that interval before escalating.
    for signum, timeout in ((signal.SIGINT, 35.), (signal.SIGTERM, 5.), (signal.SIGKILL, 3.)):
        if process.poll() is not None:
            return
        _signal_controller(process, signum)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue


def monitor(command, output, spec, *, mode="auto", refresh=1., log_interval=15.,
            stream=None, input_stream=None):
    """Observe small metadata and byte-stream output; never load model pickles."""
    stream = sys.stdout if stream is None else stream
    output = Path(output)
    progress, display = AdaptiveProgress(output, spec), Display(stream, mode)
    display.message("Progress observes saved metadata; completed tasks are not a health verdict. ETA is approximate.")
    # With None, Popen inherits the original stdin descriptor, including a TTY.
    # Passing a stream is only an explicit embedding/test option, not an answer.
    process = subprocess.Popen(command, stdin=input_stream, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, bufsize=0, start_new_session=True)
    messages = queue.Queue()

    def read_output():
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while True:
                chunk = os.read(process.stdout.fileno(), 8192)
                if not chunk:
                    break
                messages.put(decoder.decode(chunk))
            last = decoder.decode(b"", final=True)
            if last:
                messages.put(last)
        finally:
            messages.put(None)

    reader = threading.Thread(target=read_output, name="adaptive-progress-output", daemon=True)
    reader.start()
    previous_handlers, received_signals = {}, []
    interrupted_at = None

    def interrupted(signum, _frame):
        nonlocal interrupted_at
        received_signals.append(signum)
        if interrupted_at is None:
            interrupted_at = monotonic()
            # The parent controllers catch KeyboardInterrupt to stop their
            # separate worker sessions. SIGTERM would bypass that cleanup.
            # Repeated signals must not interrupt the 30-second worker drain.
            _signal_controller(process, signal.SIGINT)

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, interrupted)
    framer, raw_log = OutputFramer(), None
    buffered, buffered_size = [], 0
    log_checked = False
    interval = refresh if display.tty else log_interval
    last_display, exited_at = monotonic() - interval, None
    waiting, eof = False, False

    def disable_log(error):
        nonlocal raw_log, log_checked
        if raw_log is not None:
            try:
                raw_log.close()
            except OSError:
                pass
        raw_log, log_checked = None, True
        buffered.clear()
        if not waiting:
            display.message(f"PROGRESS_LOG_UNAVAILABLE {error}; training continues")

    def process_records(records):
        nonlocal waiting
        for kind, text in records:
            if kind == "prompt":
                waiting = True
                progress.consume(text)
                display.message(text)
            elif text.strip():
                # Only the original controller consumes input. Any subsequent
                # output means it has processed it (or is asking again).
                waiting = False
                if progress.consume(text):
                    display.message(text)

    try:
        while not eof:
            try:
                chunk = messages.get(timeout=min(.2, interval))
            except queue.Empty:
                chunk = ""
            if chunk is None:
                eof = True
                process_records(framer.feed("", final=True))
            elif chunk:
                if raw_log is not None:
                    try:
                        raw_log.write(chunk)
                        raw_log.flush()
                    except OSError as error:
                        disable_log(error)
                elif not log_checked:
                    # Limit startup buffering; do not make a new output
                    # nonempty before the controller commits its manifest.
                    buffered.append(chunk)
                    buffered_size += len(chunk)
                    while buffered_size > 1024 * 1024 and len(buffered) > 1:
                        buffered_size -= len(buffered.pop(0))
                process_records(framer.feed(chunk))
            now = monotonic()
            if interrupted_at is not None and process.poll() is None:
                if now - interrupted_at > 40.:
                    _signal_controller(process, signal.SIGKILL)
                elif now - interrupted_at > 35.:
                    _signal_controller(process, signal.SIGTERM)
            if now - last_display >= interval or eof:
                progress.refresh()
                if not log_checked:
                    try:
                        manifest_path = output / "manifest.json"
                        if manifest_path.is_file():
                            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                            if manifest.get("spec") == spec:
                                folder = output / "progress_display_logs"
                                folder.mkdir(exist_ok=True)
                                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
                                path = folder / f"adaptive_{stamp}_{os.getpid()}.log"
                                raw_log = path.open("x", encoding="utf-8")
                                raw_log.writelines(buffered)
                                raw_log.flush()
                                buffered.clear()
                                log_checked = True
                                if not waiting:
                                    display.message("PROGRESS_RAW_LOG " + str(path))
                    except (OSError, ValueError) as error:
                        disable_log(error)
                if not waiting:
                    display.render(progress)
                last_display = now
            if process.poll() is not None and not eof:
                exited_at = now if exited_at is None else exited_at
                if now - exited_at > 2.:
                    # Real adaptive workers redirect to their own logs; a
                    # descendant holding stdout must not hang the display.
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    if now - exited_at > 4.:
                        break
        code = process.wait()
        progress.finish(code)
        display.render(progress)
        display.clear()
        display.message(f"RUNNER_EXIT {code}; {progress.status}")
        return 128 + received_signals[0] if received_signals else (128 - code if code < 0 else code)
    finally:
        _stop_controller(process)
        reader.join(timeout=1.)
        if not reader.is_alive():
            process.stdout.close()
        if raw_log is not None:
            try:
                raw_log.close()
            except OSError:
                pass
        display.clear()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=REPO / "configs/cifar10_resnet18_gn_tpe_v8.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--devices", nargs="+", default=["auto"])
    parser.add_argument("--phase", choices=("all", "search", "final"), default="all")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--progress-mode", choices=("auto", "live", "log"), default="auto")
    parser.add_argument("--progress-interval", "--progress-refresh", type=float, default=1.)
    parser.add_argument("--progress-log-interval", type=float, default=15.)
    args = parser.parse_args(argv)
    for value, minimum, label in ((args.progress_interval, .1, "progress interval"),
                                  (args.progress_log_interval, 1., "log interval")):
        if not math.isfinite(value) or value < minimum:
            parser.error(f"{label} must be finite and >= {minimum}")
    try:
        spec = json.loads(args.config.read_text(encoding="utf-8"))
        entry = PROTOCOL_ENTRIES.get(spec.get("protocol"))
        if spec.get("schema_version") != 8 or entry is None:
            raise ValueError("requires a known CIFAR/Fashion schema-8 adaptive protocol")
        output = (args.output or Path(spec["output_dir"])).resolve()
    except (OSError, ValueError, KeyError, AttributeError) as error:
        parser.error(str(error))
    command = [sys.executable, "-u", str(REPO / entry), "--config", str(args.config.resolve()),
               "--devices", *args.devices, "--phase", args.phase]
    for flag, path in (("--output", args.output), ("--data-dir", args.data_dir)):
        if path is not None:
            command += [flag, str(path.resolve())]
    if args.plan_only:
        return subprocess.call([*command, "--plan-only"])
    return monitor(command, output, spec, mode=args.progress_mode, refresh=args.progress_interval,
                   log_interval=args.progress_log_interval)


if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(REPO, launcher_path=Path(__file__))
    raise SystemExit(main())
