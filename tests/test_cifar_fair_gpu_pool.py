from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict
import io
from pathlib import Path
from threading import Event, Lock
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
        self.stack.enter_context(mock.patch.object(experiments, "available_cpu_count", return_value=1))
        self.stack.enter_context(mock.patch.object(experiments, "_physical_memory_mb", return_value=128 * 1024.))
        self.stack.enter_context(mock.patch.object(experiments, "_estimated_parallel_worker_memory_mb", return_value=2048.))
        self.stack.enter_context(mock.patch.object(experiments, "_estimated_cuda_worker_memory_mb", return_value=1024.))

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
        with self.assertRaisesRegex(RuntimeError, "cannot inspect"):
            policy.check_gpus()
        self.assertEqual(policy.Pool.ready_devices(1024.), ())

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
            result = fair_tuning.run_cuda_tasks(prepared, devices=("cuda:0",), jobs=1)
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
                    cuda_execution.cuda_devices_with_capacity)
        with self.assertRaisesRegex(ValueError, "fixture"):
            with policy.gpu_pool():
                raise ValueError("fixture")
        self.assertEqual(original, (fair_tuning.prepare_tuning_tasks, fair_tuning.run_cuda_tasks,
                         cuda_execution.cuda_devices_with_capacity))


if __name__ == "__main__":
    unittest.main()
