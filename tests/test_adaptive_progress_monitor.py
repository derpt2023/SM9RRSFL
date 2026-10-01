"""Exercise the display wrapper with real CPU children, without training."""
import io
import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import run_adaptive_with_progress as runner


class PromptStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.prompt = threading.Event()

    def write(self, value):
        result = super().write(value)
        if "[Y/N]" in value:
            self.prompt.set()
        return result


class MonitorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "output"
        self.spec = {"schema_version": 8, "protocol": "cifar-resnet18-gn-tpe-v1",
                     "output_dir": str(self.output)}

    def observe(self, script, *, stream=None, input_stream=None):
        stream = io.StringIO() if stream is None else stream
        result = runner.monitor([sys.executable, "-u", "-c", script], self.output,
                                self.spec, mode="auto", log_interval=.03,
                                stream=stream, input_stream=input_stream)
        return result, stream.getvalue()

    def test_split_non_newline_and_repeated_prompts_are_immediately_framed(self):
        frame = runner.OutputFramer()
        self.assertEqual(frame.feed("DEVICES {}\nCONTINUATION_PRO"), [("line", "DEVICES {}")])
        self.assertEqual(frame.feed("MPT 是否继续？[Y/"), [])
        self.assertEqual(frame.feed("N] "), [("prompt", "CONTINUATION_PROMPT 是否继续？[Y/N]")])
        self.assertEqual(frame.feed("CONTINUATION_PROMPT 再输入[Y/N] ANSWER=N\n"),
                         [("prompt", "CONTINUATION_PROMPT 再输入[Y/N]"), ("line", "ANSWER=N")])
        self.assertEqual(frame.feed("last", final=True), [("line", "last")])
        frame = runner.OutputFramer()
        self.assertEqual(frame.feed("CONTINUATION_PROMPT first [Y/N]"),
                         [("prompt", "CONTINUATION_PROMPT first [Y/N]")])
        self.assertEqual(frame.feed(" "), [])
        self.assertEqual(frame.feed("CONTINUATION_PROMPT second [Y/N] "),
                         [("prompt", "CONTINUATION_PROMPT second [Y/N]")])

    def test_real_child_receives_answer_only_after_visible_prompt_on_each_invocation(self):
        for protocol in runner.PROTOCOL_ENTRIES:
            self.spec["protocol"] = protocol
            for invocation in range(2):
                with self.subTest(protocol=protocol, invocation=invocation):
                    stream = PromptStream()
                    read_fd, write_fd = os.pipe()
                    seen = []
                    def reply():
                        with os.fdopen(write_fd, "w") as writer:
                            seen.append(stream.prompt.wait(5))
                            if seen[-1]:
                                writer.write("Y\n")
                                writer.flush()
                    writer = threading.Thread(target=reply)
                    writer.start()
                    with os.fdopen(read_fd) as source:
                        result, text = self.observe(
                            "import signal,sys,time; signal.alarm(7); "
                            "sys.stdout.write('CONTINUATION_PROMPT 是否进入主实验？[Y/N] '); "
                            "sys.stdout.flush(); answer=sys.stdin.readline().strip(); "
                            "print('ANSWER='+answer); sys.exit(0 if answer=='Y' else 23)",
                            stream=stream, input_stream=source)
                    writer.join(6)
                    self.assertFalse(writer.is_alive())
                    self.assertEqual(seen, [True])
                    self.assertEqual(result, 0)
                    self.assertIn("ANSWER=Y", text)
                    prompt_to_answer = text.split("[Y/N]", 1)[1].split("ANSWER=Y")[0]
                    self.assertNotIn("saved_rounds", prompt_to_answer)
                    self.assertNotIn("\x1b", text)

    def test_invalid_input_repeats_and_n_or_eof_never_becomes_y(self):
        script = """import signal,sys
signal.alarm(7)
while True:
    print('CONTINUATION_PROMPT 输入[Y/N] ', end='', flush=True)
    answer = sys.stdin.readline()
    if not answer or answer.strip().upper() == 'N':
        print('FINAL_NOT_STARTED declined or EOF', flush=True)
        break
    if answer.strip().upper() == 'Y':
        raise RuntimeError('unexpected automatic Y')
"""
        for answers, prompts in ((b"bad\nN\n", 2), (b"", 1)):
            with self.subTest(answers=answers), tempfile.TemporaryFile() as source:
                source.write(answers)
                source.seek(0)
                code, text = self.observe(script, input_stream=source)
                self.assertEqual(code, 0)
                self.assertEqual(text.count("CONTINUATION_PROMPT"), prompts)
                self.assertIn("RUNNER_EXIT 0; FINAL_NOT_STARTED", text)

    def test_startup_failure_does_not_create_output_or_lose_error_exit(self):
        code, text = self.observe("import sys; print('configuration rejected'); sys.exit(42)")
        self.assertEqual(code, 42)
        self.assertIn("configuration rejected", text)
        self.assertIn("RUNNER_EXIT 42", text)
        self.assertFalse(self.output.exists())

    def test_raw_log_created_only_after_matching_manifest_and_includes_startup(self):
        manifest = {"schema_version": 8, "fingerprint": "test", "spec": self.spec}
        script = ("import json,pathlib; print('EARLY_STARTUP'); "
                  f"p=pathlib.Path({str(self.output)!r}); "
                  "assert not p.exists(); p.mkdir(); "
                  f"(p/'manifest.json').write_text({json.dumps(manifest)!r}); "
                  "print('END_STARTUP')")
        code, text = self.observe(script)
        self.assertEqual(code, 0)
        logs = list((self.output / "progress_display_logs").glob("*.log"))
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0].read_text(), "EARLY_STARTUP\nEND_STARTUP\n")
        self.assertIn("PROGRESS_RAW_LOG", text)

    def test_wrong_manifest_never_gets_display_log(self):
        self.output.mkdir()
        (self.output / "manifest.json").write_text(json.dumps({"spec": {"other": True}}))
        code, _ = self.observe("print('manifest mismatch')")
        self.assertEqual(code, 0)
        self.assertFalse((self.output / "progress_display_logs").exists())

    def test_optional_log_enospc_at_creation_or_append_does_not_stop_controller(self):
        self.output.mkdir()
        (self.output / "manifest.json").write_text(json.dumps(
            {"schema_version": 8, "fingerprint": "test", "spec": self.spec}))
        original_open = Path.open
        for fail_startup in (True, False):
            class FailingLog:
                def __init__(self, actual):
                    self.actual = actual
                def writelines(self, lines):
                    if fail_startup:
                        raise OSError(28, "No space left on device")
                    return self.actual.writelines(lines)
                def write(self, text):
                    raise OSError(28, "No space left on device")
                def flush(self):
                    return self.actual.flush()
                def close(self):
                    return self.actual.close()
            def open_log(path, *args, **kwargs):
                actual = original_open(path, *args, **kwargs)
                return FailingLog(actual) if path.parent.name == "progress_display_logs" else actual
            with self.subTest(fail_startup=fail_startup), mock.patch.object(Path, "open", open_log):
                code, text = self.observe("import time; print('STARTUP'); time.sleep(.1); print('CHILD_DONE')")
            self.assertEqual(code, 0)
            self.assertEqual(text.count("PROGRESS_LOG_UNAVAILABLE"), 1)
            self.assertIn("CHILD_DONE", text)
            self.assertIn("RUNNER_EXIT 0", text)

    def check_signal(self, signum):
        # Run the wrapper in a separate process so the test runner's handlers
        # and foreground terminal cannot be interrupted by this regression.
        child = ("import signal,time; "
                 "signal.signal(signal.SIGINT, lambda *_: (print('CHILD_STOPPED', flush=True), exit(130))); "
                 "print('CHILD_READY', flush=True); time.sleep(20)")
        script = ("import sys; import run_adaptive_with_progress as r; "
                  f"sys.exit(r.monitor({[sys.executable, '-u', '-c', child]!r}, "
                  f"{str(self.output)!r}, {self.spec!r}, mode='log', log_interval=.05))")
        process = subprocess.Popen([sys.executable, "-u", "-c", script],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, start_new_session=True)
        ready = threading.Event()
        lines = []
        def collect():
            for line in process.stdout:
                lines.append(line)
                if "CHILD_READY" in line:
                    ready.set()
        reader = threading.Thread(target=collect)
        reader.start()
        try:
            self.assertTrue(ready.wait(5))
            process.send_signal(signum)
            self.assertEqual(process.wait(timeout=8), 128 + signum)
            reader.join(2)
            self.assertIn("CHILD_STOPPED", "".join(lines))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            reader.join(2)
            process.stdout.close()

    def test_sigint_reaches_controller_and_wrapper_returns_interrupt_status(self):
        self.check_signal(signal.SIGINT)

    def test_sigterm_uses_controller_cooperative_sigint_cleanup(self):
        self.check_signal(signal.SIGTERM)

    def test_real_terminal_inherits_stdin_and_accepts_y_after_prompt(self):
        child = ("import signal,sys; signal.alarm(7); "
                 "print('CONTINUATION_PROMPT 输入[Y/N] ', end='', flush=True); "
                 "answer=sys.stdin.readline().strip(); print('ANSWER='+answer); "
                 "sys.exit(0 if answer=='Y' else 23)")
        script = ("import sys; import run_adaptive_with_progress as r; "
                  f"sys.exit(r.monitor({[sys.executable, '-u', '-c', child]!r}, "
                  f"{str(self.output)!r}, {self.spec!r}, refresh=.03))")
        master, slave = pty.openpty()
        process = subprocess.Popen([sys.executable, "-u", "-c", script],
                                   stdin=slave, stdout=slave, stderr=slave,
                                   start_new_session=True)
        os.close(slave)
        data, answered = b"", False
        deadline = time.monotonic() + 8
        try:
            while time.monotonic() < deadline:
                if select.select([master], [], [], .1)[0]:
                    try:
                        chunk = os.read(master, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    data += chunk
                    if b"[Y/N]" in data and not answered:
                        os.write(master, b"Y\n")
                        answered = True
                elif process.poll() is not None:
                    break
            self.assertTrue(answered)
            self.assertEqual(process.wait(timeout=1), 0)
            self.assertIn(b"ANSWER=Y", data)
            self.assertIn(b"RUNNER_EXIT 0", data)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            os.close(master)

    def test_plan_only_routes_all_arguments_without_monitor_or_gpu_work(self):
        config = self.output.parent / "config.json"
        for protocol, entry in runner.PROTOCOL_ENTRIES.items():
            self.spec["protocol"] = protocol
            config.write_text(json.dumps(self.spec))
            with self.subTest(protocol=protocol), \
                    mock.patch.object(runner.subprocess, "call", return_value=17) as child, \
                    mock.patch.object(runner, "monitor", side_effect=AssertionError("plan-only monitor")):
                code = runner.main(["--config", str(config), "--phase", "final", "--plan-only",
                                    "--devices", "cuda:0", "cuda:2", "--data-dir", "data/custom",
                                    "--output", str(self.output)])
                self.assertEqual(code, 17)
                command = child.call_args.args[0]
                self.assertEqual(Path(command[2]).name, entry)
                self.assertEqual(command[command.index("--phase") + 1], "final")
                self.assertEqual(command[-1], "--plan-only")
                self.assertIn("cuda:2", command)
                self.assertIn(str(Path("data/custom").resolve()), command)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
