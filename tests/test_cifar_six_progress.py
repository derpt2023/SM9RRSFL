"""Display, CUDA selection and real subprocess lifecycle without GPU training."""
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import run_cifar_six_with_progress as progress


def gpu(index, name="4090D", memory=24):
    return {"logical_device": f"cuda:{index}", "name": name,
            "compute_capability": [8, 9], "total_memory_bytes": memory}


class ProgressTests(unittest.TestCase):
    def test_cuda_mapping_mask_and_auto_group_resume(self):
        self.assertEqual(progress.physical_mapping(["cuda:0", "cuda:1"], "6,7"),
                         {"cuda:0": "visible GPU 6", "cuda:1": "visible GPU 7"})
        self.assertIn("unavailable", progress.physical_mapping(["cuda:1"], "7")["cuda:1"])
        cards = [gpu(0, "A100"), gpu(1), gpu(2)]
        self.assertEqual(progress.select_devices(cards, ["auto"]), (["cuda:1", "cuda:2"], ["cuda:0"]))
        self.assertEqual(progress.select_devices(cards, ["auto"], gpu(5, "A100")),
                         (["cuda:0"], ["cuda:1", "cuda:2"]))
        self.assertEqual(progress.select_devices(cards, ["cuda:2"]), (["cuda:2"], []))
        for devices, recorded in [(["cuda:0", "cuda:1"], None), (["cuda:9"], None),
                                  (["cuda:1", "cuda:1"], None), (["cuda:1"], gpu(0, "A100")),
                                  (["auto"], gpu(0, "absent"))]:
            with self.assertRaises(ValueError):
                progress.select_devices(cards, devices, recorded)

    def test_auto_discovers_cuda_not_nvidia_smi_count(self):
        fake = mock.Mock(returncode=0, stdout='GPU_PROBE_JSON '+json.dumps([gpu(0)]), stderr='')
        with mock.patch.object(progress.subprocess, "run", return_value=fake) as call:
            self.assertEqual(progress.discover_gpus(Path(".")), [gpu(0)])
        command = call.call_args.args[0]
        self.assertEqual(command[:2], [sys.executable, "-c"])
        self.assertIn("torch.cuda.device_count()", command[2])
        fake.stdout = "GPU_PROBE_JSON []"
        with mock.patch.object(progress.subprocess, "run", return_value=fake), self.assertRaises(ValueError):
            progress.discover_gpus(Path("."))

    def test_cli_auto_explicit_forwarding_and_progress_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / "config.json"
            spec.write_text(json.dumps({"output_dir": directory}))
            with mock.patch.object(progress, "discover_gpus", return_value=[gpu(0), gpu(1)]), \
                    mock.patch.object(progress, "monitor", return_value=0) as monitor, \
                    mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(progress.main(["--config", str(spec), "--devices", "auto",
                                               "--phase", "validation", "--progress-mode", "live"]), 0)
                command, chosen = monitor.call_args.args
                self.assertEqual(chosen, ["cuda:0", "cuda:1"])
                self.assertEqual(command[-3:], ["--devices", "cuda:0", "cuda:1"])
                self.assertIn("--phase", command)
                self.assertNotIn("--progress-mode", command)
                self.assertEqual(monitor.call_args.kwargs["mode"], "live")
                progress.main(["--config", str(spec), "--devices=cuda:1"])
                self.assertEqual(monitor.call_args.args[1], ["cuda:1"])

    def test_round_parser_eta_ignores_round_zero_and_resume_prefix(self):
        now = [0.]
        state = progress.Progress(["cuda:0"], now=lambda: now[0], visible="7")
        state.consume("VALIDATION_RUNS 2 rounds=100 devices=cuda:0\n")
        state.consume("START validation_a device=cuda:0\n")
        state.consume("[validation_a] resuming run from_completed_round=40\n")
        now[0] = 10.
        self.assertFalse(state.consume("[validation_a] ROUND validation_a round=41 accuracy=0.55 nonfinite=0\n"))
        self.assertEqual(state.new_rounds, 1)
        self.assertEqual(state.eta(), 1590.)
        state.consume("[validation_a] ROUND validation_a round=42 accuracy=0.56 nonfinite=1\n")
        self.assertEqual(state.new_rounds, 2)
        rendered = "\n".join(state.lines())
        self.assertIn("round=42/100", rendered)
        self.assertIn("visible GPU 7", rendered)
        self.assertIn("nonfinite=1", rendered)
        state.consume("[validation_a] COMPLETED validation_a\n")
        self.assertEqual(state.completed, {"validation_a"})
        self.assertFalse(state.lanes)
        state.consume("START validation_b device=cuda:0\n")
        before = state.new_rounds
        state.consume("[validation_b] ROUND validation_b round=0 accuracy=0.1 nonfinite=0\n")
        self.assertEqual(state.new_rounds, before)

    def test_cached_and_partial_snapshot_load_does_not_accelerate_eta(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            tasks = [{"task_id": name, "config": {"rounds": 100}} for name in ("a", "b")]
            (output / "validation_plan.json").write_text(json.dumps({"tasks": tasks}))
            for name in ("a", "b"):
                (output / "tasks" / name).mkdir(parents=True)
            (output / "tasks/a/.completed_results.pickle").touch()
            (output / "tasks/b/attempt_1.json").write_text(json.dumps({"last_completed_round": 30}))
            (output / "tasks/b/attempt_2.json").write_text(json.dumps({"task_id": "b"}))
            (output / "tasks/b/failure.json").write_text('{}')
            state = progress.Progress(["cuda:0"])
            state.consume("SIX_METHOD_OUTPUT " + str(output))
            state.consume("VALIDATION_RUNS 2 rounds=100 devices=cuda:0")
            self.assertEqual(state.completed, {"a"})
            self.assertEqual(state.tasks["b"]["round"], 30)
            self.assertFalse(state.failed)
            self.assertIsNone(state.eta())
            state.consume("[b] ROUND b round=31 accuracy=0.5 nonfinite=0")
            self.assertEqual(state.new_rounds, 0)
            self.assertIsNone(state.eta())

    def test_failure_settled_is_not_success_and_final_switch_resets(self):
        state = progress.Progress(["cuda:0"])
        state.consume("VALIDATION_RUNS 2 rounds=100 devices=cuda:0")
        state.consume("START a device=cuda:0")
        state.consume("[a] ROUND a round=5 accuracy=0.2 nonfinite=1")
        self.assertTrue(state.consume("[a] Traceback (most recent call last):"))
        self.assertTrue(state.consume("[a] FloatingPointError: invalid model"))
        state.consume("START b device=cuda:0")
        state.consume("[b] COMPLETED b")
        rendered = "\n".join(state.lines())
        self.assertIn("done=1/2 failed=1 settled=2/2", rendered)
        self.assertIn("rounds=105/200", rendered)
        state.consume("FINAL_RUNS 180 parameters_frozen=true")
        self.assertEqual(state.phase, "final")
        self.assertEqual(state.total, 180)
        self.assertFalse(state.failed)
        self.assertFalse(state.completed)

    def test_display_live_and_log_and_durations(self):
        self.assertEqual(progress.duration(3661), "01h01m01s")
        self.assertEqual(progress.duration(None), "estimating")
        state = progress.Progress(["cuda:0"])
        out = io.StringIO()
        display = progress.Display(out, "live")
        display.render(state)
        display.message("failure remains visible")
        self.assertIn("\x1b[", out.getvalue())
        self.assertIn("failure remains visible", out.getvalue())
        out = io.StringIO()
        display = progress.Display(out, "log")
        display.render(state)
        display.message("failure")
        self.assertNotIn("\x1b[", out.getvalue())

    def test_live_progress_fits_short_terminal(self):
        state = progress.Progress([f"cuda:{i}" for i in range(8)])
        for i in range(8):
            state.consume(f"START validation_sm9rrs-v8-001_iid_ratio0_seed{i} device=cuda:{i}")
        out = io.StringIO()
        display = progress.Display(out, "live")
        with mock.patch.object(progress.shutil, "get_terminal_size", return_value=os.terminal_size((80, 10))):
            display.render(state)
        self.assertLessEqual(display.height, 9)
        self.assertTrue(all(len(line) <= 79 for line in out.getvalue().splitlines()))
        self.assertIn("remaining GPU lanes", out.getvalue())

    def test_monitor_preserves_traceback_exitcode_and_full_round_log(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "manifest.json").write_text('{}')
            script = ("import sys; "
                      f"print('SIX_METHOD_OUTPUT {output}'); "
                      "print('VALIDATION_RUNS 1 rounds=100 devices=cuda:0'); "
                      "print('START task device=cuda:0'); "
                      "print('[task] ROUND task round=0 accuracy=0.1 nonfinite=0'); "
                      "print('[task] ROUND task round=1 accuracy=0.2 nonfinite=0'); "
                      "print('[task] Traceback (most recent call last):'); "
                      "print('[task] ValueError: preserved'); sys.exit(7)")
            out = io.StringIO()
            code = progress.monitor([sys.executable, "-u", "-c", script], ["cuda:0"], stream=out)
            self.assertEqual(code, 7)
            self.assertIn("ValueError: preserved", out.getvalue())
            self.assertNotIn("[task] ROUND", out.getvalue())
            self.assertIn("ROUND task round=1", next((output / "progress_display_logs").glob("*.log")).read_text())

    @unittest.skipUnless(os.name == "posix", "process group signals need POSIX")
    def test_dead_controller_does_not_leave_stdout_worker_hanging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready, stopped = root / "ready", root / "stopped"
            worker = root / "worker.py"
            worker.write_text("import signal, time\nfrom pathlib import Path\n"
                              f"def stop(sig, frame):\n Path({str(stopped)!r}).write_text('stopped')\n raise SystemExit(0)\n"
                              "signal.signal(signal.SIGTERM, stop)\n"
                              f"Path({str(ready)!r}).write_text('ready')\n"
                              "while True: time.sleep(.05)\n")
            code = ("import subprocess, sys, time\nfrom pathlib import Path\n"
                    f"subprocess.Popen([sys.executable, {str(worker)!r}])\n"
                    f"while not Path({str(ready)!r}).exists(): time.sleep(.01)\n"
                    "sys.exit(7)\n")
            with mock.patch.object(progress, "ORPHAN_TERM_AFTER", .05), \
                    mock.patch.object(progress, "ORPHAN_KILL_AFTER", .3):
                result = progress.monitor([sys.executable, "-u", "-c", code], ["cuda:0"], stream=io.StringIO())
            self.assertEqual(result, 7)
            self.assertTrue(stopped.exists())

    @unittest.skipUnless(os.name == "posix", "process group signals need POSIX")
    def test_interrupt_reaches_worker_and_waits_for_controller_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready, stopped, exited = root / "ready", root / "stopped", root / "exited"
            worker = root / "worker.py"
            worker.write_text("import signal, time\nfrom pathlib import Path\n"
                              f"def stop(sig, frame):\n Path({str(stopped)!r}).write_text('stopped')\n raise SystemExit(0)\n"
                              "signal.signal(signal.SIGINT, stop)\n"
                              f"Path({str(ready)!r}).write_text('ready')\n"
                              "while True: time.sleep(.05)\n")
            controller = root / "controller.py"
            controller.write_text("import signal, subprocess, sys, time\nfrom pathlib import Path\n"
                                  f"child = subprocess.Popen([sys.executable, {str(worker)!r}])\n"
                                  "def stop(sig, frame):\n child.wait(timeout=5)\n"
                                  f" Path({str(exited)!r}).write_text('exited')\n raise SystemExit(130)\n"
                                  "signal.signal(signal.SIGINT, stop)\n"
                                  "while True: time.sleep(.05)\n")
            def interrupt():
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and not ready.exists():
                    time.sleep(.02)
                if ready.exists():
                    os.kill(os.getpid(), signal.SIGINT)
            thread = threading.Thread(target=interrupt, daemon=True)
            thread.start()
            out = io.StringIO()
            code = progress.monitor([sys.executable, "-u", str(controller)], ["cuda:0"], stream=out)
            thread.join()
            self.assertEqual(code, 130)
            self.assertTrue(stopped.exists())
            self.assertTrue(exited.exists())


if __name__ == "__main__":
    unittest.main()
