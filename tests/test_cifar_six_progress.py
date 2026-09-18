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
            "compute_capability": [8, 9], "total_memory_bytes": memory * 1024 ** 3,
            "initialization_ok": True, "free_memory_bytes": memory * 1024 ** 3}


class ProgressTests(unittest.TestCase):
    def test_cuda_mapping_mask_and_auto_group_resume(self):
        self.assertEqual(progress.physical_mapping(["cuda:0", "cuda:1"], "6,7"),
                         {"cuda:0": "visible GPU 6", "cuda:1": "visible GPU 7"})
        self.assertIn("unavailable", progress.physical_mapping(["cuda:1"], "7")["cuda:1"])
        cards = [gpu(0, "A100"), gpu(1), gpu(2)]
        chosen, skipped = progress.select_devices(cards, ["auto"])
        self.assertEqual(chosen, ["cuda:1", "cuda:2"])
        self.assertEqual([row["logical_device"] for row in skipped], ["cuda:0"])
        self.assertIn("incompatible", skipped[0]["reason"])
        chosen, skipped = progress.select_devices(cards, ["auto"], gpu(5, "A100"))
        self.assertEqual(chosen, ["cuda:0"])
        self.assertEqual([row["logical_device"] for row in skipped], ["cuda:1", "cuda:2"])
        self.assertEqual(progress.select_devices(cards, ["cuda:2"]), (["cuda:2"], []))
        for devices, recorded in [(["cuda:0", "cuda:1"], None), (["cuda:9"], None),
                                  (["cuda:1", "cuda:1"], None), (["cuda:1"], gpu(0, "A100")),
                                  (["auto"], gpu(0, "absent"))]:
            with self.assertRaises(ValueError):
                progress.select_devices(cards, devices, recorded)

    def test_auto_discovers_cuda_not_nvidia_smi_count(self):
        enum = mock.Mock(returncode=0, stdout='GPU_PROBE_JSON 1', stderr='')
        fake = mock.Mock(returncode=0, stdout='GPU_PROBE_JSON '+json.dumps(gpu(0)), stderr='')
        with mock.patch.object(progress.subprocess, "run", side_effect=[enum, fake]) as call:
            self.assertEqual(progress.discover_gpus(Path(".")), [gpu(0)])
        self.assertEqual(call.call_count, 2)
        command = call.call_args_list[0].args[0]
        self.assertEqual(command[:2], [sys.executable, "-c"])
        self.assertIn("torch.cuda.device_count()", command[2])
        self.assertNotIn("set_device", command[2])
        self.assertIn("torch.cuda.set_device(index)", call.call_args.args[0][2])
        self.assertIn("torch.cuda.mem_get_info(index)", call.call_args.args[0][2])
        enum.stdout = "GPU_PROBE_JSON 0"
        with mock.patch.object(progress.subprocess, "run", return_value=enum), self.assertRaises(ValueError):
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

    def test_630_config_routes_to_new_runner_and_keeps_live_all_gpu_display(self):
        repo = Path(progress.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(progress, "discover_gpus", return_value=[gpu(0), gpu(1)]), \
                mock.patch.object(progress, "monitor", return_value=0) as monitor, \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            for config, runner in (("cifar10_six_mnist_gate_v4.json", "run_cifar_six_mnist_gate.py"),
                                   ("cifar10_six_630_mean_v3.json", "run_cifar_six_630.py"),
                                   ("cifar10_six_original_v2.json", "run_cifar_six_from_scratch.py")):
                with self.subTest(config=config):
                    self.assertEqual(progress.main(["--config", str(repo / "configs" / config),
                        "--output", directory, "--devices", "auto", "--progress-mode", "live"]), 0)
                    command, devices = monitor.call_args.args
                    self.assertEqual(Path(command[2]).name, runner)
                    self.assertEqual(devices, ["cuda:0", "cuda:1"])
                    self.assertEqual(monitor.call_args.kwargs["mode"], "live")

    def test_v4_plan_only_bypasses_gpu_discovery_and_training_monitor(self):
        repo = Path(progress.__file__).resolve().parent
        with mock.patch.object(progress, "discover_gpus") as discover, \
                mock.patch.object(progress, "monitor") as monitor, \
                mock.patch.object(progress.subprocess, "call", return_value=0) as run:
            self.assertEqual(progress.main(["--config", str(repo / "configs/cifar10_six_mnist_gate_v4.json"),
                                            "--plan-only", "--devices", "auto"]), 0)
        discover.assert_not_called()
        monitor.assert_not_called()
        command = run.call_args.args[0]
        self.assertEqual(Path(command[1]).name, "run_cifar_six_mnist_gate.py")
        self.assertIn("--plan-only", command)

    def test_plan_only_cannot_mix_with_report_only(self):
        with mock.patch.object(progress, "discover_gpus") as discover, \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as stopped:
                progress.main(["--plan-only", "--report-only"])
            self.assertEqual(stopped.exception.code, 2)
        discover.assert_not_called()

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


class ConcurrentOutputTests(unittest.TestCase):
    def test_adjacent_cached_completions_do_not_inflate_plan_or_lose_start(self):
        a = "validation_sm9rrs-v8-001_dirichlet_ratio0_seed701"
        b = "validation_sm9rrs-v8-001_iid_ratio0_seed701"
        c = "validation_sm9rrs-v8-002_dirichlet_ratio0_seed701"
        d = "validation_sm9rrs-v8-003_dirichlet_ratio0_seed701"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            names = [a, b, c, d] + [f"validation_other{i}" for i in range(80)]
            (output / "validation_plan.json").write_text(json.dumps({
                "tasks": [{"task_id": name, "config": {"rounds": 100}} for name in names]}))
            state = progress.Progress(["cuda:0"])
            state.consume(f"SIX_METHOD_OUTPUT {output}")
            state.consume("VALIDATION_RUNS 84 rounds=100 devices=cuda:0")
            lines = [f"ALREADY_COMPLETED {a}ALREADY_COMPLETED {b}",
                     f"ALREADY_COMPLETED {c}START {d} device=cuda:0"]
            events = [event for line in lines for event in progress.output_events(line)]
            self.assertEqual(len(events), 4)
            for event in events:
                state.consume(event)
            self.assertEqual(state.completed, {a, b, c})
            self.assertEqual(state.lanes, {"cuda:0": d})
            self.assertEqual(len(state.tasks), 84)
            self.assertIn("rounds=300/8400", state.lines()[0])
            # A corrupt or unknown protocol token is visible as a message,
            # never added as an extra scheduled experiment.
            for event in ["COMPLETED validation_unknown", "START validation_unknown device=cuda:0",
                          "ROUND validation_unknown round=50 accuracy=0.5 nonfinite=0",
                          "[validation_unknown] resuming from_completed_round=50",
                          "[validation_unknown] Traceback (most recent call last):"]:
                state.consume(event)
            self.assertEqual(len(state.tasks), 84)
            self.assertEqual(state.completed, {a, b, c})
            self.assertEqual(state.lanes, {"cuda:0": d})
            self.assertNotIn("validation_unknown", state.failed)

    def test_adjacent_rounds_update_every_task_without_console_leak(self):
        state = progress.Progress([f"cuda:{i}" for i in range(8)])
        joined = "".join(f"[validation_task{i}] ROUND validation_task{i} round=25 accuracy=0.5 nonfinite=0"
                         for i in range(8)) + "\n"
        events = list(progress.output_events(joined))
        self.assertEqual(len(events), 8)
        self.assertTrue(all(not state.consume(event) for event in events))
        self.assertEqual({t["round"] for t in state.tasks.values()}, {25})
        self.assertEqual(len(state.tasks), 8)
        self.assertEqual(list(progress.output_events("\n")), [])

    def test_start_progress_concatenation_preserves_cuda_mapping(self):
        state = progress.Progress(["cuda:3"])
        line = "START validation_vert-v8-005_iid_ratio0_seed701 device=cuda:3PROGRESS 0/84 rough_remaining_seconds=unknown"
        for event in progress.output_events(line):
            state.consume(event)
        self.assertEqual(set(state.lanes), {"cuda:3"})
        task = state.lanes["cuda:3"]
        self.assertEqual(state.task_devices[task], "cuda:3")
        state.consume(f"[{task}] Traceback (most recent call last):")
        self.assertFalse(state.lanes)
        self.assertIn(task, state.failed)

    def test_round_adjacent_to_traceback_does_not_hide_failure(self):
        state = progress.Progress(["cuda:0", "cuda:1"])
        line = ("[validation_a] ROUND validation_a round=5 accuracy=0.4 nonfinite=0"
                "[validation_b] RuntimeError: CUDA error: out of memory")
        shown = [event for event in progress.output_events(line) if state.consume(event)]
        self.assertEqual(shown, ["[validation_b] RuntimeError: CUDA error: out of memory"])
        self.assertEqual(state.tasks["validation_a"]["round"], 5)

    def test_real_parallel_print_records_are_all_parsed(self):
        import subprocess
        code = ("from concurrent.futures import ThreadPoolExecutor\nfrom threading import Barrier\n"
                "barrier=Barrier(8)\n"
                "def emit(i):\n"
                " for rd in range(20):\n"
                "  barrier.wait()\n"
                "  print(f'[validation_task{i}] ROUND validation_task{i} round={rd} accuracy=0.5 nonfinite=0', flush=True)\n"
                "with ThreadPoolExecutor(8) as pool: list(pool.map(emit, range(8)))\n")
        raw = subprocess.check_output([sys.executable, "-u", "-c", code], text=True)
        state = progress.Progress([f"cuda:{i}" for i in range(8)])
        events = [event for line in raw.splitlines() for event in progress.output_events(line)]
        self.assertEqual(len(events), 160)
        self.assertTrue(all(not state.consume(event) for event in events))
        self.assertEqual(len(state.tasks), 8)
        self.assertTrue(all(task["round"] == 19 for task in state.tasks.values()))

    @unittest.skipUnless(os.name == "posix", "process group signals need POSIX")
    def test_cuda_oom_stops_controller_before_more_tasks_and_reports_card(self):
        with tempfile.TemporaryDirectory() as directory:
            later = Path(directory) / "should_not_launch"
            code = ("import time\nfrom pathlib import Path\n"
                    "print('START validation_a device=cuda:3', flush=True)\n"
                    "print('[validation_a] Traceback (most recent call last):', flush=True)\n"
                    "print('[validation_a] RuntimeError: CUDA error: out of memory', flush=True)\n"
                    "time.sleep(10)\n"
                    f"Path({str(later)!r}).touch()\n")
            rejected = set()
            out = io.StringIO()
            result = progress.monitor([sys.executable, "-u", "-c", code], ["cuda:3"],
                                      stream=out, oom_devices=rejected)
            self.assertEqual(result, 75)
            self.assertEqual(rejected, {"cuda:3"})
            self.assertIn("RESOURCE_OOM", out.getvalue())
            self.assertIn("RuntimeError: CUDA error: out of memory", out.getvalue())
            self.assertFalse(later.exists())

    @unittest.skipUnless(os.name == "posix", "process group signals need POSIX")
    def test_oom_stop_has_deadline_and_reports_formal_phase(self):
        code = ("import signal,time\n"
                "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "print('FINAL_RUNS 180 parameters_frozen=true', flush=True)\n"
                "print('START final_a device=cuda:1', flush=True)\n"
                "print('[final_a] RuntimeError: CUDA error: out of memory', flush=True)\n"
                "time.sleep(5)\n")
        rejected, phase = set(), {}
        out = io.StringIO()
        with mock.patch.object(progress, "RESOURCE_TERM_AFTER", .05), \
                mock.patch.object(progress, "RESOURCE_KILL_AFTER", .5):
            result = progress.monitor([sys.executable, "-u", "-c", code], ["cuda:1"], stream=out,
                                      oom_devices=rejected, phase_state=phase)
        self.assertEqual(result, 75)
        self.assertEqual(rejected, {"cuda:1"})
        self.assertEqual(phase, {"phase": "final"})
        self.assertIn("sending SIGKILL", out.getvalue())


class ReportHookTests(unittest.TestCase):
    def report_module(self):
        module = mock.Mock()
        module.check_dependencies.return_value = None
        module.generate_report.return_value = {
            "html_path": "paper_figures/index.html", "pdf_path": "paper_figures/figures.pdf",
            "run_count": 180, "health_failed_runs": 35,
        }
        return module

    def prepare_output(self, directory):
        output = Path(directory) / "results"
        output.mkdir()
        (output / "manifest.json").write_text('{"fingerprint":"unchanged"}')
        (output / "final_summary.json").write_text(json.dumps({
            "status": "completed_with_health_failures", "full_execution_completed": True,
        }))
        return output

    def test_report_only_uses_explicit_output_without_gpu_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.prepare_output(directory)
            module = self.report_module()
            with mock.patch.dict(sys.modules, {"experiment_reporting": module}), \
                    mock.patch.object(progress, "discover_gpus", side_effect=AssertionError("no GPU probe")) as probe, \
                    mock.patch.object(progress, "monitor", side_effect=AssertionError("no training")) as monitor, \
                    mock.patch("sys.stdout", new_callable=io.StringIO) as console:
                self.assertEqual(progress.main(["--report-only", "--output", str(output)]), 0)
            probe.assert_not_called()
            monitor.assert_not_called()
            module.check_dependencies.assert_called_once_with()
            module.generate_report.assert_called_once_with(output.resolve())
            status = json.loads((output / "report_generation_status.json").read_text())
            self.assertEqual(status["mode"], "report_only")
            self.assertEqual(status["status"], "completed")
            self.assertIsNone(status["training_exit_code"])
            self.assertIn("REPORT_COMPLETED", console.getvalue())
            self.assertEqual((output / "manifest.json").read_text(), '{"fingerprint":"unchanged"}')

    def test_report_only_resolves_config_output_and_rejects_missing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.prepare_output(directory)
            spec = Path(directory) / "config.json"
            spec.write_text(json.dumps({"output_dir": str(output)}))
            module = self.report_module()
            with mock.patch.dict(sys.modules, {"experiment_reporting": module}), \
                    mock.patch.object(progress, "discover_gpus", side_effect=AssertionError("no GPU probe")), \
                    mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(progress.main(["--report-only", "--config", str(spec)]), 0)
                module.generate_report.assert_called_once_with(output.resolve())
                with mock.patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
                    progress.main(["--report-only", "--config", str(spec.with_name("missing.json"))])
                self.assertEqual(error.exception.code, 2)

    def test_automatic_report_keeps_complete_health_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.prepare_output(directory)
            module = self.report_module()
            def complete(*args, **kwargs):
                kwargs["phase_state"]["phase"] = "final"
                return 0
            with mock.patch.dict(sys.modules, {"experiment_reporting": module}), \
                    mock.patch.object(progress, "discover_gpus", return_value=[gpu(0)]), \
                    mock.patch.object(progress, "monitor", side_effect=complete) as monitor, \
                    mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(progress.main(["--output", str(output)]), 0)
            monitor.assert_called_once()
            module.generate_report.assert_called_once_with(output.resolve())
            status = json.loads((output / "report_generation_status.json").read_text())
            self.assertEqual(status["mode"], "post_training")
            self.assertEqual(status["training_exit_code"], 0)
            self.assertEqual(status["report"]["health_failed_runs"], 35)

    def test_validation_gate_and_nonzero_exits_do_not_generate_reports(self):
        cases = [("validation", "validation", 0), ("all", "validation", 0),
                 ("all", None, 0), ("all", "final", 1), ("all", "final", 75),
                 ("all", "final", 130), ("all", "final", 143)]
        with tempfile.TemporaryDirectory() as directory:
            output = self.prepare_output(directory)  # Deliberately contains old formal metadata.
            module = self.report_module()
            for requested, observed, code in cases:
                with self.subTest(requested=requested, observed=observed, code=code):
                    def finish(*args, **kwargs):
                        if observed:
                            kwargs["phase_state"]["phase"] = observed
                        return code
                    with mock.patch.dict(sys.modules, {"experiment_reporting": module}), \
                            mock.patch.object(progress, "discover_gpus", return_value=[gpu(0)]), \
                            mock.patch.object(progress, "monitor", side_effect=finish), \
                            mock.patch("sys.stdout", new_callable=io.StringIO):
                        self.assertEqual(progress.main(["--output", str(output), "--phase", requested]), code)
            module.check_dependencies.assert_not_called()
            module.generate_report.assert_not_called()
            self.assertFalse((output / "report_generation_status.json").exists())

    def test_failed_automatic_report_preserves_training_success_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.prepare_output(directory)
            module = self.report_module()
            module.generate_report.side_effect = ValueError("incomplete three-seed evidence")
            def complete(*args, **kwargs):
                kwargs["phase_state"]["phase"] = "final"
                return 0
            with mock.patch.dict(sys.modules, {"experiment_reporting": module}), \
                    mock.patch.object(progress, "discover_gpus", return_value=[gpu(0)]), \
                    mock.patch.object(progress, "monitor", side_effect=complete) as monitor, \
                    mock.patch("sys.stdout", new_callable=io.StringIO) as console:
                self.assertEqual(progress.main(["--output", str(output)]), 0)
            monitor.assert_called_once()
            status = json.loads((output / "report_generation_status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["training_exit_code"], 0)
            self.assertTrue(status["training_result_unchanged"])
            self.assertIn("REPORT_FAILED", console.getvalue())
            self.assertIn("--report-only --output", status["retry_command"])
            module.generate_report.side_effect = None
            with mock.patch.dict(sys.modules, {"experiment_reporting": module}), \
                    mock.patch.object(progress, "discover_gpus", side_effect=AssertionError("no GPU probe")), \
                    mock.patch.object(progress, "monitor", side_effect=AssertionError("no training")), \
                    mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(progress.main(["--report-only", "--output", str(output)]), 0)
            self.assertEqual(json.loads((output / "report_generation_status.json").read_text())["status"], "completed")
            self.assertEqual((output / "manifest.json").read_text(), '{"fingerprint":"unchanged"}')

    def test_report_only_missing_dependencies_fails_without_creating_training_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-yet-created"
            module = self.report_module()
            module.check_dependencies.side_effect = ImportError("pip install -r requirements.txt")
            with mock.patch.dict(sys.modules, {"experiment_reporting": module}), \
                    mock.patch.object(progress, "discover_gpus", side_effect=AssertionError("no GPU probe")), \
                    mock.patch("sys.stdout", new_callable=io.StringIO) as console:
                self.assertEqual(progress.main(["--report-only", "--output", str(output)]), 1)
            module.generate_report.assert_not_called()
            self.assertFalse(output.exists())
            self.assertIn("pip install -r requirements.txt", console.getvalue())

    def test_resource_retry_generates_report_only_after_final_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.prepare_output(directory)
            module = self.report_module()
            attempts = []
            def execute(*args, **kwargs):
                attempts.append(args[1])
                kwargs["phase_state"]["phase"] = "final"
                if len(attempts) == 1:
                    kwargs["oom_devices"].add("cuda:0")
                    module.generate_report.assert_not_called()
                    return 75
                module.generate_report.assert_not_called()
                return 0
            with mock.patch.dict(sys.modules, {"experiment_reporting": module}), \
                    mock.patch.object(progress, "discover_gpus", return_value=[gpu(0), gpu(1)]), \
                    mock.patch.object(progress, "monitor", side_effect=execute), \
                    mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(progress.main(["--output", str(output)]), 0)
            self.assertEqual(attempts, [["cuda:0", "cuda:1"], ["cuda:1"]])
            module.generate_report.assert_called_once_with(output.resolve())


if __name__ == "__main__":
    unittest.main()
