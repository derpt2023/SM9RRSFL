"""Real child-process continuation input, terminal rendering and cancellation."""
import io
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import unittest
from unittest import mock

import run_cifar_six_with_progress as progress


PROMPT = "CONTINUATION_PROMPT " + json.dumps({
    "candidate": "sm9rrs-v10-014", "raw_score": .641895570609,
    "message": "验证未达标，是否使用最佳健康候选继续正式实验？",
}, ensure_ascii=False)


class TerminalText(io.StringIO):
    def __init__(self, value):
        super().__init__(value)
        self.read_threads = []

    def isatty(self):
        return True

    def readline(self, *args):
        self.read_threads.append(threading.get_ident())
        return super().readline(*args)


class PromptOutput(io.StringIO):
    def __init__(self):
        super().__init__()
        self.prompt_seen = threading.Event()
        self.prompt_at = None

    def write(self, value):
        if "CONTINUATION_PROMPT candidate=" in value:
            self.prompt_at = time.monotonic()
            self.prompt_seen.set()
        return super().write(value)


def controller_script(prompt=PROMPT):
    # A watchdog also ensures a regression in pipe handling fails promptly.
    return (
        "import signal, sys\n"
        "signal.alarm(5)\n"
        "signal.signal(signal.SIGINT, lambda *_: sys.exit(130))\n"
        "while True:\n"
        f" print({prompt!r}, flush=True)\n"
        " answer = sys.stdin.readline()\n"
        " if not answer or answer.strip().upper() in ('Y', 'N'): break\n"
        " print('INVALID_CHOICE explicit Y or N required', flush=True)\n"
        "print('ANSWER ' + repr(answer), flush=True)\n"
        "if answer.strip().upper() == 'Y':\n"
        " print('FINAL_RUNS 180 parameters_frozen=true', flush=True)\n"
        "else:\n"
        " print('FINAL_NOT_STARTED: user declined or input unavailable', flush=True)\n"
    )


@unittest.skipUnless(os.name == "posix", "the experiment runner uses POSIX process groups")
class InteractiveProgressTests(unittest.TestCase):
    def monitor(self, source, input_stream, output=None, **kwargs):
        output = output if output is not None else io.StringIO()
        result = progress.monitor([sys.executable, "-u", "-c", source], ["cuda:0"],
            stream=output, input_stream=input_stream, refresh=.05, log_interval=.05,
            **kwargs)
        return result, output.getvalue()

    def test_schema7_uses_interactive_controller_and_older_runners_stay_stable(self):
        root = Path("/repository")
        expected = {
            2: "run_cifar_six_from_scratch.py", 3: "run_cifar_six_630.py",
            4: "run_cifar_six_mnist_gate.py", 5: "run_cifar_six_final_metrics.py",
            6: "run_cifar_six_relative_asr.py", 7: "run_cifar_six_interactive.py",
        }
        for schema, name in expected.items():
            with self.subTest(schema=schema):
                spec = {"schema_version": schema}
                if schema == 3:
                    spec["mean_dual_gate"] = {}
                self.assertEqual(Path(progress.runner_for_spec(root, spec)).name, name)

    def test_real_child_receives_explicit_yes_or_no_and_terminal_eof_is_not_yes(self):
        for answer in ("Y\n", "n\n", ""):
            with self.subTest(answer=answer):
                terminal = TerminalText(answer)
                result, output = self.monitor(controller_script(), terminal)
                self.assertEqual(result, 0)
                self.assertEqual("FINAL_RUNS 180" in output, answer.strip().upper() == "Y")
                self.assertIn("ANSWER " + repr(answer), output)
                self.assertEqual(len(terminal.read_threads), 1)

    def test_nonterminal_piped_yes_is_not_silently_accepted(self):
        source = io.StringIO("Y\n")
        result, output = self.monitor(controller_script(), source)
        self.assertEqual(result, 0)
        self.assertIn("CONTINUATION_INPUT_UNAVAILABLE", output)
        self.assertIn("ANSWER ''", output)
        self.assertNotIn("FINAL_RUNS 180", output)
        self.assertEqual(source.tell(), 0)

    def test_invalid_answers_reprompt_with_one_reader_and_no_extra_reads(self):
        source = TerminalText("maybe\n\nY\nN\n")
        result, output = self.monitor(controller_script(), source)
        self.assertEqual(result, 0)
        self.assertEqual(output.count("CONTINUATION_PROMPT candidate="), 3)
        self.assertEqual(output.count("INVALID_CHOICE"), 2)
        self.assertIn("FINAL_RUNS 180", output)
        self.assertEqual(len(source.read_threads), 3)
        self.assertEqual(len(set(source.read_threads)), 1)
        self.assertEqual(source.read(), "N\n")

    def test_another_invocation_asks_again_and_does_not_reuse_previous_yes(self):
        source = TerminalText("Y\nN\n")
        first, output1 = self.monitor(controller_script(), source)
        second, output2 = self.monitor(controller_script(), source)
        self.assertEqual((first, second), (0, 0))
        self.assertIn("FINAL_RUNS 180", output1)
        self.assertNotIn("FINAL_RUNS 180", output2)
        self.assertIn("CONTINUATION_PROMPT candidate=", output2)
        self.assertIn("ANSWER 'N\\n'", output2)
        self.assertEqual(len(source.read_threads), 2)

    def test_worker_prefixed_or_malformed_prompt_cannot_request_consent(self):
        source = TerminalText("Y\n")
        code = f"print('[validation_a] ' + {PROMPT!r}, flush=True)\n"
        result, output = self.monitor(code, source)
        self.assertEqual(result, 0)
        self.assertEqual(source.read_threads, [])
        self.assertIn("[validation_a] CONTINUATION_PROMPT", output)
        result, output = self.monitor(controller_script("CONTINUATION_PROMPT not-json"), source)
        self.assertEqual(result, 0)
        self.assertIn("CONTINUATION_INPUT_REFUSED", output)
        self.assertNotIn("FINAL_RUNS 180", output)
        self.assertEqual(source.read_threads, [])

    def test_real_terminal_delayed_reply_is_not_overwritten_by_live_progress(self):
        import pty
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        terminal = os.fdopen(slave, "r", encoding="utf-8")
        self.addCleanup(terminal.close)
        output = PromptOutput()
        render_times, answer_times = [], []
        original_render = progress.Display.render

        def render(display, state):
            render_times.append(time.monotonic())
            return original_render(display, state)

        def reply():
            if output.prompt_seen.wait(timeout=3):
                time.sleep(.3)
                answer_times.append(time.monotonic())
                os.write(master, b"Y\n")

        thread = threading.Thread(target=reply, daemon=True)
        thread.start()
        with mock.patch.object(progress.Display, "render", render):
            result, text = self.monitor(controller_script(), terminal, output, mode="live")
        thread.join(timeout=1)
        self.assertEqual(result, 0)
        self.assertIn("FINAL_RUNS 180", text)
        self.assertEqual(len(answer_times), 1)
        self.assertFalse(any(output.prompt_at < stamp < answer_times[0] for stamp in render_times))

    def test_ctrl_c_during_terminal_wait_exits_and_releases_input_reader(self):
        import pty
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        terminal = os.fdopen(slave, "r", encoding="utf-8")
        self.addCleanup(terminal.close)
        output = PromptOutput()

        def interrupt():
            if output.prompt_seen.wait(timeout=3):
                os.kill(os.getpid(), signal.SIGINT)

        thread = threading.Thread(target=interrupt, daemon=True)
        thread.start()
        began = time.monotonic()
        result, text = self.monitor(controller_script(), terminal, output)
        thread.join(timeout=1)
        self.assertEqual(result, 130)
        self.assertLess(time.monotonic() - began, 3)
        self.assertNotIn("FINAL_RUNS 180", text)
        self.assertFalse(any(t.name == "cifar-continuation-input" and t.is_alive()
                             for t in threading.enumerate()))


if __name__ == "__main__":
    unittest.main()
