from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
import io
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.experiments.cifar10 import cifar_fair_gpu_pool as policy
from sm9rrsfl import cuda_execution, experiments, fair_tuning
from sm9rrsfl.config_runner import parameters_to_argv
from sm9rrsfl.fl import ExperimentConfig
import run_cifar_v14_with_progress as launcher


ROOT = Path(__file__).resolve().parents[1]
DEVICES = tuple(f"cuda:{i}" for i in range(8))


def tasks(phase="validation", count=16):
    return [fair_tuning.TuningExperimentTask(phase, "sm9rrs-001", "sm9rrs",
            ExperimentConfig(method="sm9rrs", device="auto", compute_backend="torch", seed=i))
            for i in range(count)]


class GpuPoolTest(unittest.TestCase):
    def setUp(self):
        spec = fair_tuning.load_fair_tuning_config(ROOT / "configs/cifar10_v14_fair_grid.json")
        self.args = experiments.parse_args(parameters_to_argv(spec.shared_parameters))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(mock.patch.object(experiments, "available_cuda_devices", return_value=DEVICES))
        self.info = self.stack.enter_context(mock.patch.object(experiments, "cuda_device_memory_info",
            return_value=tuple((d, 20000., 24000.) for d in DEVICES)))
        def probe():
            devices = experiments.available_cuda_devices()
            info = self.info() or ()
            known = {row[0] for row in info}
            failures = {d: {"error_type": "RuntimeError", "error": "CUDA error: out of memory"}
                        for d in devices if d not in known}
            return devices, info, failures
        self.stack.enter_context(mock.patch.object(policy, "probe_gpu_memory", side_effect=probe))
        self.stack.enter_context(mock.patch.object(experiments, "available_cpu_count", return_value=1))
        self.stack.enter_context(mock.patch.object(experiments, "_physical_memory_mb", return_value=128 * 1024.))
        self.stack.enter_context(mock.patch.object(experiments, "_estimated_parallel_worker_memory_mb", return_value=2048.))
        self.stack.enter_context(mock.patch.object(experiments, "_estimated_cuda_worker_memory_mb", return_value=1024.))
        self.release = self.stack.enter_context(mock.patch.object(cuda_execution, "release_cuda_cache"))

    def test_both_phases_use_eight_gpus_despite_one_cpu_slot(self):
        reports = []
        pool = policy.Pool(reports.append)
        for phase in ("validation", "final"):
            source = tasks(phase)
            prepared, jobs, backend, workers = pool.prepare(None, source, self.args, requested_jobs="auto")
            self.assertEqual((jobs, backend, workers), (8, "torch:cuda", 1))
            self.assertEqual({t.config.device for t in prepared}, set(DEVICES))
            self.assertFalse(reports[-1]["cpu_slot_cap_applied"])
            self.assertFalse(reports[-1]["host_memory_limited"])
            for old, new in zip(source, prepared):
                original, actual = asdict(old.config), asdict(new.config)
                actual["device"] = original["device"]
                self.assertEqual(original, actual)

    def test_busy_cards_remain_in_pool_and_no_ready_card_does_not_abort_preparation(self):
        for free_count in (6, 0):
            self.info.return_value = tuple((d, 20000. if i < free_count else 100., 24000.)
                                           for i, d in enumerate(DEVICES))
            reports = []
            pool = policy.Pool(reports.append)
            _, jobs, _, _ = pool.prepare(None, tasks(), self.args)
            self.assertEqual(pool.devices, DEVICES)
            self.assertEqual(jobs, 8)
            self.assertEqual(len(reports[0]["ready_now"]), free_count)
            self.assertEqual(len(reports[0]["waiting_for_memory"]), 8 - free_count)

    def test_host_memory_limit_is_preserved_and_reported(self):
        reports = []
        with mock.patch.object(experiments, "_physical_memory_mb", return_value=8192.):
            _, jobs, _, _ = policy.Pool(reports.append).prepare(None, tasks(), self.args)
        self.assertEqual(jobs, 3)
        self.assertTrue(reports[0]["host_memory_limited"])

    def test_insufficient_total_gpu_memory_is_excluded_but_busy_large_card_is_retained(self):
        self.info.return_value = tuple((d, 100., 512. if i == 7 else 24000.) for i, d in enumerate(DEVICES))
        reports = []
        pool = policy.Pool(reports.append)
        pool.prepare(None, tasks(), self.args)
        self.assertEqual(pool.devices, DEVICES[:7])
        self.assertEqual(reports[0]["excluded_total_memory"], ["cuda:7"])

    def test_no_cuda_or_unknown_memory_is_an_explicit_error(self):
        with mock.patch.object(experiments, "available_cuda_devices", return_value=()):
            with self.assertRaisesRegex(RuntimeError, "requires visible CUDA"):
                policy.check_gpus()
        self.info.return_value = None
        with self.assertRaisesRegex(RuntimeError, "no GPU memory query succeeded.*query_failures"):
            policy.check_gpus()
        self.assertEqual(policy.Pool().ready_devices(1024.), ())

    def test_one_query_oom_keeps_other_cards_and_retries_failed_card_later(self):
        # Reproduce the feedback: cuda:6 unreadable, 2/3 have plenty of memory,
        # all other successful readings are too small for this task.
        self.info.return_value = tuple((d, 22000. if i in (2, 3) else 2200., 24000.)
                                       for i, d in enumerate(DEVICES) if i != 6)
        inventory = policy.check_gpus()
        self.assertEqual(inventory["memory_query_status"], "partial")
        self.assertEqual(set(inventory["query_failures"]), {"cuda:6"})
        reports = []
        with mock.patch.object(experiments, "_estimated_cuda_worker_memory_mb", return_value=3000.):
            pool = policy.Pool(reports.append)
            pool.prepare(None, tasks(), self.args)
        self.assertEqual(pool.devices, DEVICES)
        self.assertEqual(reports[0]["ready_now"], ["cuda:2", "cuda:3"])
        self.assertEqual(reports[0]["waiting_for_memory_query"], ["cuda:6"])
        self.assertEqual(pool.ready_devices(3000.), ("cuda:2", "cuda:3"))
        self.info.return_value += (("cuda:6", 22000., 24000.),)
        self.assertEqual(pool.ready_devices(3000.), ("cuda:2", "cuda:3", "cuda:6"))
        self.assertEqual(pool.query_failures, {})

    def test_runtime_query_failure_never_admits_failed_card_or_blocks_healthy_card(self):
        pool = policy.Pool()
        pool.prepare(None, tasks(), self.args)
        self.info.return_value = tuple(row for row in self.info.return_value if row[0] != "cuda:6")
        ready = pool.ready_devices(1024.)
        self.assertEqual(set(ready), set(DEVICES) - {"cuda:6"})
        self.assertEqual(set(pool.query_failures), {"cuda:6"})

    def test_real_queue_finishes_tasks_on_healthy_cards_while_one_query_keeps_failing(self):
        self.info.return_value = tuple((d, 22000. if i in (2, 3) else 2200., 24000.)
                                       for i, d in enumerate(DEVICES) if i != 6)
        seen, commits = [], []
        def run(item):
            seen.append(item.config.device)
            return item.config.seed

        with mock.patch.object(experiments, "_estimated_cuda_worker_memory_mb", return_value=3000.), \
                policy.gpu_pool():
            prepared, jobs, _, _ = fair_tuning.prepare_tuning_tasks(None, tasks(count=8), self.args)
            fair_tuning.run_cuda_tasks(prepared, devices=DEVICES, jobs=jobs, required_mb=3000.,
                run=run, commit=lambda item, result: commits.append(result), event=lambda *a, **k: None)
        self.assertEqual(len(commits), 8)
        self.assertEqual(set(seen), {"cuda:2", "cuda:3"})

    def test_scoped_generic_resource_reader_also_preserves_partial_results(self):
        self.info.return_value = tuple(row for row in self.info.return_value if row[0] != "cuda:6")
        original = experiments.cuda_device_memory_info
        with policy.gpu_pool():
            rows = experiments.cuda_device_memory_info()
            self.assertEqual({row[0] for row in rows}, set(DEVICES) - {"cuda:6"})
        self.assertIs(experiments.cuda_device_memory_info, original)

    def test_gpu_inventory_is_read_only_and_does_not_start_training(self):
        with mock.patch.object(fair_tuning, "run_fair_tuning") as run, \
                mock.patch.object(fair_tuning, "load_image_dataset") as load:
            launcher.main(["--check-gpus"])
        run.assert_not_called()
        load.assert_not_called()

    def test_resume_queue_keeps_all_cards_even_with_one_remaining_task(self):
        captured = {}

        def queue(items, **kwargs):
            captured.update(kwargs)
            return "original queue result"

        with mock.patch.object(fair_tuning, "run_cuda_tasks", side_effect=queue), policy.gpu_pool() as pool:
            prepared, _, _, _ = fair_tuning.prepare_tuning_tasks(None, tasks(count=1), self.args)
            result = fair_tuning.run_cuda_tasks(prepared, devices=("cuda:0",), jobs=1, run=lambda item: None)
        self.assertEqual(result, "original queue result")
        self.assertEqual(captured["devices"], DEVICES)
        self.assertEqual(captured["jobs"], 1)
        self.assertEqual(captured["poll_seconds"], 1.)
        self.assertEqual(captured["max_idle_seconds"], 600.)

    def test_busy_gpus_join_while_six_other_tasks_are_running(self):
        released, all_running = Event(), Event()
        lock = Lock()
        active, seen, commits = set(), set(), []
        peak = [0]

        def info():
            return tuple((d, 20000. if i < 6 or released.is_set() else 100., 24000.)
                         for i, d in enumerate(DEVICES))

        def run(item):
            device = item.config.device
            with lock:
                self.assertNotIn(device, active)
                active.add(device)
                seen.add(device)
                peak[0] = max(peak[0], len(active))
                if len(active) == 6:
                    released.set()
                if len(active) == 8:
                    all_running.set()
            try:
                self.assertTrue(all_running.wait(4.), "newly free GPUs were never scheduled")
                return item.config.seed
            finally:
                with lock:
                    active.remove(device)

        self.info.side_effect = info
        with policy.gpu_pool():
            prepared, jobs, _, _ = fair_tuning.prepare_tuning_tasks(None, tasks(count=8), self.args)
            fair_tuning.run_cuda_tasks(prepared, devices=DEVICES, jobs=jobs, required_mb=1024.,
                run=run, commit=lambda item, value: commits.append(value), event=lambda *a, **k: None)
        self.assertEqual(peak[0], 8)
        self.assertEqual(seen, set(DEVICES))
        self.assertEqual(len(commits), 8)

    def test_free_card_takes_next_task_while_other_card_is_still_busy(self):
        busy, refilled = Event(), Event()
        records = []
        lock = Lock()
        self.info.return_value = tuple((d, 20000., 24000. if i < 2 else 512.) for i, d in enumerate(DEVICES))

        def run(item):
            with lock:
                records.append((item.config.seed, item.config.device))
            if item.config.seed == 0:
                busy.set()
                self.assertTrue(refilled.wait(4.))
            elif item.config.seed == 1:
                self.assertTrue(busy.wait(4.))
            else:
                refilled.set()
            return item.config.seed

        with policy.gpu_pool():
            prepared, jobs, _, _ = fair_tuning.prepare_tuning_tasks(None, tasks(count=3), self.args)
            fair_tuning.run_cuda_tasks(prepared, devices=DEVICES[:2], jobs=jobs, required_mb=1024.,
                run=run, commit=lambda *a: None, event=lambda *a, **k: None)
        self.assertIn((2, "cuda:1"), records)

    def test_pool_hooks_are_restored_when_training_fails(self):
        original = (fair_tuning.prepare_tuning_tasks, fair_tuning.run_cuda_tasks,
                    cuda_execution.cuda_devices_with_capacity, experiments.cuda_device_memory_info)
        with self.assertRaisesRegex(ValueError, "fixture"):
            with policy.gpu_pool():
                raise ValueError("fixture")
        self.assertEqual(original, (fair_tuning.prepare_tuning_tasks, fair_tuning.run_cuda_tasks,
                         cuda_execution.cuda_devices_with_capacity, experiments.cuda_device_memory_info))

    def test_fatal_cudnn_event_is_forwarded_and_active_task_is_still_committed(self):
        notified = Event()
        original_events, display_events, commits = [], [], []
        def observe(kind, item, **details):
            display_events.append((kind, item, details))
            if kind == "cuda_worker_failed":
                notified.set()

        def run(item):
            if item.config.seed == 0:
                raise RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR")
            self.assertTrue(notified.wait(3.))
            return item.config.seed

        with policy.gpu_pool(on_event=observe):
            prepared, jobs, _, _ = fair_tuning.prepare_tuning_tasks(None, tasks(count=2), self.args)
            with self.assertRaisesRegex(RuntimeError, "CUDNN_STATUS_INTERNAL_ERROR"):
                fair_tuning.run_cuda_tasks(prepared, devices=DEVICES, jobs=jobs, required_mb=1024.,
                    run=run, commit=lambda item, result: commits.append(result),
                    event=lambda kind, item, **details: original_events.append((kind, item, details)))
        self.assertEqual(original_events, display_events)
        self.assertEqual(original_events[-1][0], "cuda_worker_failed")
        self.assertEqual(commits, [1])

    def test_own_unused_cache_cannot_block_the_next_task_on_a_finished_gpu(self):
        # Driver free memory falls below admission after task 1, even though
        # this process can give its unused allocator cache back immediately.
        free = [20000.]
        commits, seen = [], []
        self.info.side_effect = lambda: (("cuda:0", free[0], 24000.),)
        self.release.side_effect = lambda device: free.__setitem__(0, 20000.)

        def run(item):
            seen.append(item.config.seed)
            free[0] = 20.
            return item.config.seed

        def event(kind, *args, **kwargs):
            if kind == "cuda_waiting_for_memory":
                self.fail("own idle cache blocked admission after the previous task finished")

        with mock.patch.object(experiments, "available_cuda_devices", return_value=("cuda:0",)), policy.gpu_pool():
            prepared, jobs, _, _ = fair_tuning.prepare_tuning_tasks(None, tasks(count=2), self.args)
            fair_tuning.run_cuda_tasks(prepared, devices=("cuda:0",), jobs=jobs, required_mb=1024.,
                run=run, commit=lambda item, result: commits.append(result), event=event)
        self.assertEqual(commits, [0, 1])
        self.assertEqual(seen, [0, 1])
        self.assertEqual(self.release.call_args_list, [mock.call("cuda:0"), mock.call("cuda:0")])

    def test_failed_training_frame_does_not_retain_tensors_while_other_gpu_drains(self):
        import weakref
        class TensorSentinel:
            pass
        refs, commits = [], []
        failed = Event()

        def run(item):
            if item.config.seed == 0:
                local_tensor = TensorSentinel()
                refs.append(weakref.ref(local_tensor))
                raise RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR")
            self.assertTrue(failed.wait(3.))
            self.assertIsNone(refs[0](), "fatal Future retained the failed training frame")
            return item.config.seed

        def observe(kind, item, **details):
            if kind == "cuda_worker_failed":
                failed.set()

        with policy.gpu_pool(on_event=observe):
            prepared, jobs, _, _ = fair_tuning.prepare_tuning_tasks(None, tasks(count=2), self.args)
            with self.assertRaisesRegex(RuntimeError, "CUDNN_STATUS_INTERNAL_ERROR"):
                fair_tuning.run_cuda_tasks(prepared, devices=DEVICES, jobs=jobs, required_mb=1024.,
                    run=run, commit=lambda item, value: commits.append(value), event=lambda *a, **kw: None)
        self.assertEqual(commits, [1])

    def test_live_memory_updates_reach_display_callback(self):
        reports = []
        pool = policy.Pool(on_plan=reports.append)
        pool.prepare(None, tasks(), self.args)
        self.info.return_value = tuple((d, 17478. if d == "cuda:2" else 2219., total)
                                       for d, _, total in self.info.return_value if d != "cuda:6")
        pool.ready_devices(3000.)
        self.assertEqual(reports[-1]["target_jobs"], 8)
        self.assertEqual(reports[-1]["pool_devices"], list(DEVICES))
        self.assertEqual(reports[-1]["required_gpu_free_mb"], 3000. / experiments.CUDA_MEMORY_SAFETY_FRACTION)
        self.assertEqual(set(reports[-1]["query_failures"]), {"cuda:6"})
        self.assertEqual(reports[-1]["gpu_memory_mb"][2]["free"], 17478.)


class MemoryProbeTest(unittest.TestCase):
    def test_actual_probe_isolates_oom_and_still_queries_next_logical_device(self):
        def memory(index):
            if index == 6:
                raise RuntimeError("CUDA error: out of memory")
            return 20 * 1024 ** 3, 24 * 1024 ** 3

        reader = mock.Mock(side_effect=memory)
        torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=reader))
        with mock.patch.object(experiments, "available_cuda_devices", return_value=DEVICES), \
                mock.patch.object(policy, "_torch_module", return_value=torch):
            visible, rows, failures = policy.probe_gpu_memory()
        self.assertEqual(visible, DEVICES)
        self.assertEqual({row[0] for row in rows}, set(DEVICES) - {"cuda:6"})
        self.assertEqual(rows[-1], ("cuda:7", 20 * 1024., 24 * 1024.))
        self.assertEqual(reader.call_args_list, [mock.call(i) for i in range(8)])
        self.assertEqual(failures["cuda:6"], {"error_type": "RuntimeError", "error": "CUDA error: out of memory"})

    def test_invalid_or_unknown_readings_are_never_counted_as_free_memory(self):
        reader = mock.Mock(side_effect=[(float("nan"), 100.), (120., 100.), AttributeError("no mem_get_info")])
        torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=reader))
        with mock.patch.object(experiments, "available_cuda_devices", return_value=DEVICES[:3]), \
                mock.patch.object(policy, "_torch_module", return_value=torch):
            _, rows, failures = policy.probe_gpu_memory()
        self.assertEqual(rows, ())
        self.assertEqual(set(failures), set(DEVICES[:3]))


if __name__ == "__main__":
    unittest.main()
