"""Parent-only fixed-GPU admission waiting; no GPU or real training is used."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
import signal
from types import SimpleNamespace
import unittest
from unittest import mock

import run_cifar_prefix_probe as runner
import cifar_prefix_probe_report as report
import tests.test_cifar_prefix_probe_protocol as fixtures

protocol, base, GPU = runner.protocol, runner.base, fixtures.GPU


class FakeClock:
    def __init__(self):
        self.now = 0.
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class PrefixResumeTests(fixtures.PrefixFixture):
    def setUp(self):
        super().setUp()
        import resume_cifar_prefix_probe as adapter
        self.adapter = adapter
        self.clock = FakeClock()
        self.inventory = [{"index": 3, "uuid": GPU,
            "name": self.reference["execution_environment"]["actual_compute_device"]["name"],
            "free_mib": 24000., "utilization": 0.}]
        self.capture = io.StringIO()
        self.original_selector = runner.select_gpu

    def argv(self, *extra):
        return ["--output", str(self.output), *extra]

    def complete(self, task):
        base.write_json(self.output / "tasks" / task["task_id"] / "completed.json", self.completion(task))

    def summary_by_completions(self, output):
        n = sum(protocol.load_completed(output, task) is not None for task in self.tasks)
        return {"status": "complete" if n == 6 else "incomplete_or_invalid_evidence",
                "complete_tasks": n, "expected_tasks": 6, "training_started_by_summary": False}

    def context(self, *, inventory=None, summary=True):
        stack = ExitStack()
        stack.enter_context(mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": GPU}))
        self.query = stack.enter_context(mock.patch.object(runner, "gpu_inventory",
            return_value=self.inventory if inventory is None else inventory))
        stack.enter_context(mock.patch.object(runner.time, "monotonic", side_effect=self.clock.monotonic))
        stack.enter_context(mock.patch.object(runner.time, "sleep", side_effect=self.clock.sleep))
        stack.enter_context(redirect_stdout(self.capture))
        if summary:
            stack.enter_context(mock.patch.object(report, "summarize", side_effect=self.summary_by_completions))
        return stack

    def subprocess_factory(self):
        owner = self
        self.started, self.exited, self.active = [], [], set()
        by_id = {task["task_id"]: task for task in self.tasks}

        class Process:
            def __init__(self, command, **kwargs):
                owner.assertFalse(owner.active, "worker overlap")
                owner.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], GPU)
                owner.assertTrue(kwargs["start_new_session"])
                owner.assertNotIn("--retry-failed", command)
                owner.assertNotIn("--checkpoint", command)
                owner.assertEqual(command[2], str(runner.REPO / "run_cifar_prefix_probe.py"))
                task_id = command[command.index("--worker") + 1]
                owner.started.append(task_id)
                owner.active.add(task_id)
                self.task_id, self.returncode, self.pid, self.polls = task_id, None, 900 + len(owner.started), 0

            def poll(self):
                self.polls += 1
                if self.polls == 1:
                    return None
                if self.returncode is None:
                    owner.complete(by_id[self.task_id])
                    owner.active.remove(self.task_id)
                    owner.exited.append(self.task_id)
                    self.returncode = 0
                return self.returncode

        return Process

    def test_gate_preserves_original_inclusive_memory_and_utilization_limits(self):
        adapter = self.adapter
        name = self.inventory[0]["name"]
        for free, utilization, eligible in ((16384., 5., True), (16383.99, 5., False), (16384., 5.01, False)):
            row = {**self.inventory[0], "free_mib": free, "utilization": utilization}
            result = adapter.gate_observation([row], GPU, name, 16384., mask=GPU)
            self.assertEqual(result["eligible"], eligible)
            self.assertEqual((result["free_mib"], result["utilization_percent"]), (free, utilization))
        for requested, minimum in (("auto", 16384.), ("0", 16384.), (GPU, 8192.)):
            with self.assertRaises(ValueError):
                adapter.gate_observation(self.inventory, requested, name, minimum)

    def test_only_resource_pressure_waits_then_original_selector_decides(self):
        busy = [{**self.inventory[0], "free_mib": 16000., "utilization": 12.}]
        events = []
        with self.context(), mock.patch.object(runner, "select_gpu", wraps=self.original_selector) as selector:
            self.query.side_effect = [busy, self.inventory]
            with self.adapter.waiting_gate(GPU, emit=events.append):
                selected = runner.select_gpu(busy, GPU, self.inventory[0]["name"], 16384., mask=GPU)
            self.assertIs(runner.select_gpu, selector)
            selector.assert_called_once_with(self.inventory, GPU, self.inventory[0]["name"], 16384., mask=GPU)
        self.assertEqual(selected["uuid"], GPU)
        self.assertEqual(self.clock.sleeps, [10., 10.])
        self.assertEqual([e["event"] for e in events], ["gpu_wait", "gpu_wait", "gpu_ready"])
        self.assertEqual(events[0]["failed_checks"], ["free_memory_below_16384_mib", "utilization_above_5_percent"])
        self.assertIs(runner.select_gpu, self.original_selector)

    def test_default_wait_stops_at_600_seconds_without_changing_the_gate(self):
        busy = [{**self.inventory[0], "utilization": 6.}]
        events = []
        with self.context(inventory=busy), self.assertRaises(self.adapter.GPUWaitTimeout):
            with self.adapter.waiting_gate(GPU, emit=events.append):
                runner.select_gpu(busy, GPU, self.inventory[0]["name"], 16384.)
        self.assertEqual(self.clock.now, 600.)
        self.assertEqual(self.clock.sleeps, [10.] * 60)
        self.assertEqual(self.query.call_count, 59)
        self.assertEqual(events[-1]["event"], "gpu_wait_timeout")
        self.assertFalse(events[-1]["next_worker_started"])
        self.assertIs(runner.select_gpu, self.original_selector)

    def test_late_ready_query_cannot_start_worker_after_deadline_and_zero_wait_is_single_check(self):
        busy = [{**self.inventory[0], "utilization": 6.}]
        events = []
        def late_ready():
            self.clock.now = 12.
            return self.inventory
        with self.context(), mock.patch.object(runner, "select_gpu", wraps=self.original_selector) as selector, \
                self.assertRaises(self.adapter.GPUWaitTimeout):
            self.query.side_effect = late_ready
            with self.adapter.waiting_gate(GPU, wait_seconds=12., poll_seconds=10., emit=events.append):
                runner.select_gpu(busy, GPU, self.inventory[0]["name"], 16384.)
        selector.assert_not_called()
        self.assertEqual(events[-1]["event"], "gpu_wait_timeout")
        self.assertTrue(events[-1]["eligible"])
        self.assertFalse(events[-1]["next_worker_started"])
        self.assertIs(runner.select_gpu, self.original_selector)
        with self.context():
            with self.adapter.waiting_gate(GPU, wait_seconds=0.):
                self.assertEqual(runner.select_gpu(self.inventory, GPU, self.inventory[0]["name"], 16384.)["uuid"], GPU)
            with self.assertRaises(self.adapter.GPUWaitTimeout):
                with self.adapter.waiting_gate(GPU, wait_seconds=0.):
                    runner.select_gpu(busy, GPU, self.inventory[0]["name"], 16384.)
            self.query.assert_not_called()

    def test_invalid_structure_visibility_or_inventory_is_not_retried(self):
        name = self.inventory[0]["name"]
        bad_inputs = [([], None), (self.inventory * 2, None), ([{**self.inventory[0], "name": "different"}], None),
            ([{**self.inventory[0], "free_mib": float("nan")}], None),
            ([{**self.inventory[0], "utilization": 101.}], None),
            ([{**self.inventory[0], "free_mib": -1.}], None),
            (self.inventory, "0"), (self.inventory, ""), (self.inventory, "GPU-excluded")]
        with self.context():
            for rows, mask in bad_inputs:
                with self.subTest(rows=rows, mask=mask), self.assertRaises(ValueError):
                    with self.adapter.waiting_gate(GPU):
                        runner.select_gpu(rows, GPU, name, 16384., mask=mask)
                self.assertIs(runner.select_gpu, self.original_selector)
        self.assertEqual(self.clock.sleeps, [])
        self.query.assert_not_called()

    def test_poll_inventory_failure_and_interrupt_restore_the_selector(self):
        busy = [{**self.inventory[0], "free_mib": 1.}]
        for problem in (RuntimeError("nvidia-smi failed"), KeyboardInterrupt()):
            with self.subTest(problem=type(problem).__name__), self.context(), self.assertRaises(type(problem)):
                self.query.side_effect = problem
                with self.adapter.waiting_gate(GPU):
                    runner.select_gpu(busy, GPU, self.inventory[0]["name"], 16384.)
            self.assertIs(runner.select_gpu, self.original_selector)
            self.assertFalse(self.adapter._active)

    def test_invalid_wait_arguments_and_nested_context_do_not_leave_patch(self):
        for wait, poll in ((-1., 10.), (601., 10.), (float("inf"), 10.), (600., 0.), (600., 61.), (600., float("nan"))):
            with self.subTest(wait=wait, poll=poll), self.assertRaises(ValueError):
                with self.adapter.waiting_gate(GPU, wait_seconds=wait, poll_seconds=poll):
                    self.fail("invalid configuration entered context")
            self.assertIs(runner.select_gpu, self.original_selector)
        with self.adapter.waiting_gate(GPU):
            selector = runner.select_gpu
            with self.assertRaises(RuntimeError):
                with self.adapter.waiting_gate(GPU):
                    self.fail("nested context entered")
            self.assertIs(runner.select_gpu, selector)
        self.assertIs(runner.select_gpu, self.original_selector)

    def test_existing_two_complete_are_reused_then_only_four_original_workers_start_serially(self):
        for task in self.tasks[:2]:
            self.complete(task)
        preserved = {t["task_id"]: (self.output / "tasks" / t["task_id"] / "completed.json").read_bytes() for t in self.tasks[:2]}
        source_hashes = protocol.source_hashes()
        process = self.subprocess_factory()
        with self.context(), mock.patch.object(runner.subprocess, "Popen", side_effect=process) as spawn, \
                mock.patch.object(protocol, "read_study", wraps=protocol.read_study) as read:
            self.assertEqual(self.adapter.main(self.argv()), 0)
        self.assertEqual(self.started, [t["task_id"] for t in self.tasks[2:]])
        self.assertEqual(self.exited, self.started)
        self.assertEqual((spawn.call_count, self.query.call_count), (4, 4))
        self.assertTrue(all(call.kwargs.get("current_sources") is True for call in read.call_args_list))
        self.assertEqual(source_hashes, protocol.source_hashes())
        self.assertEqual(len(source_hashes), 72)
        self.assertNotIn("resume_cifar_prefix_probe.py", source_hashes)
        self.assertNotIn("README.md", source_hashes)
        for task in self.tasks[:2]:
            self.assertEqual(preserved[task["task_id"]], (self.output / "tasks" / task["task_id"] / "completed.json").read_bytes())
            self.assertIn("REUSE " + task["task_id"], self.capture.getvalue())
        events = [json.loads(line) for p in (self.output / "controller_resumes").glob("*.jsonl") for line in p.read_text().splitlines()]
        self.assertEqual(events[0]["complete_before"], 2)
        self.assertFalse(events[0]["automatic_retry_failed_workers"])
        self.assertEqual(events[-1]["event"], "controller_exit")
        self.assertEqual(events[-1]["code"], 0)
        self.assertIs(runner.select_gpu, self.original_selector)

    def test_timeout_returns_two_and_original_summary_without_starting_next_worker(self):
        for task in self.tasks[:2]:
            self.complete(task)
        busy = [{**self.inventory[0], "free_mib": 16383., "utilization": 0.}]
        with self.context(inventory=busy), mock.patch.object(runner.subprocess, "Popen") as spawn:
            self.assertEqual(self.adapter.main(self.argv("--wait-seconds", "12", "--poll-seconds", "10")), 2)
        self.assertEqual(self.clock.sleeps, [10., 2.])
        spawn.assert_not_called()
        self.assertEqual(sum(protocol.load_completed(self.output, t) is not None for t in self.tasks), 2)
        self.assertIn("gpu_wait_timeout", self.capture.getvalue())
        self.assertIn("CIFAR_PREFIX_PROBE_BEGIN", self.capture.getvalue())
        self.assertIn("CIFAR_PREFIX_PROBE_END", self.capture.getvalue())
        self.assertIs(runner.select_gpu, self.original_selector)

    def test_existing_failure_is_never_automatically_retried(self):
        failed = self.output / "tasks" / self.tasks[0]["task_id"] / "attempts/old/failure.json"
        failed.parent.mkdir(parents=True)
        failed.write_text('{"error":"retained"}\n')
        original = failed.read_bytes()
        with self.context(), mock.patch.object(runner.subprocess, "Popen") as spawn:
            self.assertEqual(self.adapter.main(self.argv()), 2)
        spawn.assert_not_called()
        self.query.assert_not_called()
        self.assertEqual(failed.read_bytes(), original)
        self.assertIn("unsuccessful prefix retained", self.capture.getvalue())
        self.assertIs(runner.select_gpu, self.original_selector)

    def test_nonzero_worker_exit_stops_panel_without_retry(self):
        process = SimpleNamespace(returncode=75, poll=lambda: 75)
        with self.context(), mock.patch.object(runner.subprocess, "Popen", return_value=process) as spawn:
            self.assertEqual(self.adapter.main(self.argv()), 2)
        self.assertEqual(spawn.call_count, 1)
        self.assertFalse((self.output / "tasks" / self.tasks[1]["task_id"] / "worker.log").exists())
        self.assertIs(runner.select_gpu, self.original_selector)

    def test_INT_and_TERM_keep_original_child_cleanup_and_restore_signal_and_patch(self):
        for cause in ("INT", "TERM"):
            with self.subTest(cause=cause):
                previous = signal.getsignal(signal.SIGTERM)
                process = SimpleNamespace(returncode=None, pid=12345)
                process.poll = lambda: process.returncode
                process.wait = mock.Mock(side_effect=lambda timeout=None: process.returncode)
                def terminate(pid, sig):
                    self.assertEqual((pid, sig), (12345, signal.SIGTERM))
                    process.returncode = -15
                def interrupt(_seconds):
                    if cause == "TERM":
                        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                    raise KeyboardInterrupt
                with self.context(), mock.patch.object(runner.subprocess, "Popen", return_value=process) as spawn, \
                        mock.patch.object(runner.os, "killpg", side_effect=terminate) as kill, \
                        mock.patch.object(runner.time, "sleep", side_effect=interrupt):
                    self.assertEqual(self.adapter.main(self.argv()), 2)
                self.assertEqual(spawn.call_count, 1)
                kill.assert_called_once_with(12345, signal.SIGTERM)
                process.wait.assert_called_once_with(timeout=15)
                self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
                self.assertIs(runner.select_gpu, self.original_selector)
        self.assertIn("INTERRUPTED attempt retained", self.capture.getvalue())

    def test_summary_only_is_original_readonly_without_GPU_or_controller_audit(self):
        before = {str(p): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        with self.context(summary=False), self.readonly_guards(), \
                mock.patch.object(runner.subprocess, "Popen", side_effect=AssertionError("spawn")):
            self.query.side_effect = AssertionError("GPU query")
            self.assertEqual(self.adapter.main(self.argv("--summary")), 2)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.output.rglob("*") if p.is_file()})
        self.assertFalse((self.output / "controller_resumes").exists())
        self.assertIn("CIFAR_PREFIX_PROBE_END", self.capture.getvalue())

    def test_manifest_or_preexisting_invalid_evidence_rejected_before_GPU_or_sidecar(self):
        with self.context(), mock.patch.object(protocol, "read_study", side_effect=ValueError("source identity changed")), \
                mock.patch.object(runner.subprocess, "Popen") as spawn:
            self.assertEqual(self.adapter.main(self.argv()), 2)
        self.query.assert_not_called()
        spawn.assert_not_called()
        self.assertFalse((self.output / "controller_resumes").exists())
        with self.context(), mock.patch.object(report, "summarize", return_value={"status": "incomplete_or_invalid_evidence",
                "complete_tasks": 0, "rows": [{"status": "invalid_evidence"}]}):
            self.assertEqual(self.adapter.main(self.argv()), 2)
        self.query.assert_not_called()
        self.assertFalse((self.output / "controller_resumes").exists())


if __name__ == "__main__":
    unittest.main()
