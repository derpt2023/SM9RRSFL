from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import io
from threading import Event, Lock
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.experiments.cifar10 import c7_station_gpu_pool as policy
from sm9rrsfl import cuda_execution, experiments, fair_tuning
from sm9rrsfl.fl import ExperimentConfig


def task(method="fedavg", seed=1):
    config = ExperimentConfig(method=method, num_clients=100, device="auto",
                              compute_backend="torch", seed=seed, malicious_ratio=.8)
    return fair_tuning.TuningExperimentTask("validation", method + "-fixed", method, config)


def dataset():
    return SimpleNamespace(name="cifar10", input_shape=(3, 32, 32), num_classes=10,
        x_train=SimpleNamespace(size=40000 * 3072), y_train=SimpleNamespace(size=40000),
        x_test=SimpleNamespace(size=7500 * 3072), y_test=SimpleNamespace(size=7500),
        x_attack=SimpleNamespace(size=2500 * 3072), y_attack=SimpleNamespace(size=2500))


class MemoryAccountingTest(unittest.TestCase):
    def test_actual_cnn_resident_aux_and_three_adam_states_are_counted_once(self):
        config = task().config
        bill = policy.estimate_task_memory(dataset(), config)
        self.assertEqual(bill["parameter_count"], 1756426)
        expected_dataset = 50000 * (3072 * 4 + 8) / 1024 ** 2
        self.assertAlmostEqual(bill["components_mb"]["resident_train_eval_attack"], expected_dataset)
        vector = 1756426 * 4 / 1024 ** 2
        self.assertAlmostEqual(bill["components_mb"]["c7_adam_moments"], 6 * vector)
        self.assertAlmostEqual(bill["required_gpu_free_mb"], bill["estimated_worker_mb"] / .75)
        clean = policy.estimate_task_memory(dataset(), replace(config, malicious_ratio=0))
        self.assertEqual(clean["components_mb"]["c7_adam_moments"], 0)
        self.assertLess(clean["estimated_worker_mb"], bill["estimated_worker_mb"])

    def test_v4_uses_actual_sketch_path_and_krum_has_larger_temporary_allowance(self):
        vert = policy.estimate_task_memory(dataset(), replace(task("vert").config,
                                            vert_projection_dim=256, vert_top_k=20))
        self.assertEqual(vert["vert_projection_kind"], "signed_sketch")
        self.assertLess(vert["components_mb"]["vert_projection_and_predictor"], 40)
        small = policy.estimate_task_memory(dataset(), replace(task("vert").config,
                                             vert_projection_dim=20))
        self.assertEqual(small["vert_projection_kind"], "dense")
        self.assertGreater(small["components_mb"]["vert_projection_and_predictor"], 134)
        krum = policy.estimate_task_memory(dataset(), task("krum").config)
        self.assertGreater(krum["estimated_worker_mb"], vert["estimated_worker_mb"])
        self.assertEqual(krum["components_mb"]["aggregation_workspace"],
                         4 * krum["components_mb"]["all_client_updates"])


class PoolTest(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(compute_backend="torch", device="auto", jobs="auto",
                                    sm9_workers=1, _sm9_workers_auto=False)
        self.devices = ("cuda:0", "cuda:1")
        self.info = [("cuda:0", 6000., 24000.), ("cuda:1", 6000., 24000.)]
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.probe = self.stack.enter_context(mock.patch.object(policy, "probe_gpu_memory",
            side_effect=lambda: (self.devices, tuple(self.info), {})))
        self.stack.enter_context(mock.patch.object(experiments, "available_cpu_count", return_value=1))
        self.host = self.stack.enter_context(mock.patch.object(experiments, "_physical_memory_mb", return_value=128 * 1024.))
        self.stack.enter_context(mock.patch.object(experiments, "_estimated_parallel_worker_memory_mb", return_value=2048.))
        self.stack.enter_context(mock.patch.object(policy, "estimate_task_memory", side_effect=lambda ds, config: {
            "method": config.method, "estimated_worker_mb": 3000. if config.method == "krum" else 1000.}))
        self.release = self.stack.enter_context(mock.patch.object(cuda_execution, "release_cuda_cache"))

    def prepare(self, items, **kwargs):
        pool = policy.Pool(**kwargs)
        prepared, jobs, backend, workers = pool.prepare(None, items, self.args)
        self.assertEqual((backend, workers), ("torch:cuda", 1))
        return pool, prepared

    def test_cpu_count_is_not_gpu_cap_but_host_75_percent_cap_is_preserved(self):
        pool, _ = self.prepare([task(seed=i) for i in range(4)])
        self.assertEqual(pool.jobs, 2)
        self.host.return_value = 3000.
        pool, _ = self.prepare([task(seed=i) for i in range(4)])
        self.assertEqual(pool.jobs, 1)
        self.host.return_value = 2000.
        with self.assertRaisesRegex(RuntimeError, "host memory"):
            self.prepare([task()])

    def test_read_only_inventory_keeps_unknown_memory_unknown_and_allows_waiting_run(self):
        self.probe.side_effect = lambda: (self.devices, (), {
            d: {"error_type": "RuntimeError", "error": "CUDA error: out of memory"}
            for d in self.devices})
        with mock.patch.object(fair_tuning, "run_fair_tuning") as train:
            report = policy.check_gpus()
        train.assert_not_called()
        self.assertEqual(report["memory_query_status"], "unavailable")
        self.assertEqual(report["gpu_memory_mb"], [])
        self.assertEqual(set(report["query_failures"]), set(self.devices))
        self.assertIn("600 seconds", report["startup_behavior"])
        self.probe.side_effect = lambda: ((), (), {})
        with self.assertRaisesRegex(RuntimeError, "requires visible CUDA"):
            policy.check_gpus()

    def test_partial_inventory_preserves_successful_values(self):
        self.probe.side_effect = lambda: (self.devices, (self.info[0],), {"cuda:1": {"error": "OOM"}})
        report = policy.check_gpus()
        self.assertEqual(report["memory_query_status"], "partial")
        self.assertEqual(report["gpu_memory_mb"], [{"device": "cuda:0", "free": 6000, "total": 24000}])

    def test_total_memory_accepts_small_card_for_light_task_but_needs_a_large_card_for_krum(self):
        self.info = [("cuda:0", 2000., 2500.), ("cuda:1", 5000., 6000.)]
        pool, _ = self.prepare([task("krum"), task()])
        self.assertEqual(pool.devices, self.devices)
        self.info[1] = ("cuda:1", 2000., 2500.)
        with self.assertRaisesRegex(RuntimeError, "some C7 station tasks"):
            self.prepare([task("krum"), task()])

    def test_light_task_bypasses_waiting_krum_on_small_card(self):
        self.info = [("cuda:0", 2000., 24000.), ("cuda:1", 5000., 24000.)]
        pool, prepared = self.prepare([task("krum"), task("fedavg", 2)])
        seen, commits = {}, []
        def run(item):
            seen[item.method] = item.config.device
            return item.method
        pool.queue(prepared, run=run, commit=lambda t, r: commits.append(r), event=None,
                   poll_seconds=.001)
        self.assertEqual(seen, {"fedavg": "cuda:0", "krum": "cuda:1"})
        self.assertCountEqual(commits, ["fedavg", "krum"])
        self.assertEqual(self.probe.call_count, 2)  # preparation + one shared poll

    def test_busy_card_joins_while_other_worker_runs_and_never_overlaps_same_card(self):
        first_running, second_running = Event(), Event()
        lock = Lock()
        active, seen, violations, commits = set(), set(), [], []
        def inventory():
            return self.devices, (("cuda:0", 6000., 24000.),
                ("cuda:1", 6000. if first_running.is_set() else 0., 24000.)), {}
        self.probe.side_effect = inventory
        pool, prepared = self.prepare([task(seed=i) for i in range(4)])
        def run(item):
            with lock:
                if item.config.device in active:
                    violations.append(item.config.device)
                active.add(item.config.device)
                seen.add(item.config.device)
            if item.config.device == "cuda:0":
                first_running.set()
                if not second_running.wait(2):
                    raise AssertionError("released GPU did not join")
            else:
                second_running.set()
            with lock:
                active.remove(item.config.device)
            return item.config.seed
        pool.queue(prepared, run=run, commit=lambda t, r: commits.append(r), event=None, poll_seconds=.001)
        self.assertEqual(seen, set(self.devices))
        self.assertEqual(violations, [])
        self.assertEqual(len(commits), 4)

    def test_display_threshold_tracks_remaining_tasks_after_light_task_finishes(self):
        self.devices = ("cuda:0",)
        self.info = [("cuda:0", 2000., 24000.)]
        reports, completed = [], []
        pool, prepared = self.prepare([task("krum"), task("fedavg", 2)], on_plan=reports.append)
        def commit(item, result):
            completed.append(item.method)
            self.info = [("cuda:0", 6000., 24000.)]
        pool.queue(prepared, run=lambda t: t.method, commit=commit, event=None, poll_seconds=.001)
        self.assertEqual(completed, ["fedavg", "krum"])
        self.assertAlmostEqual(reports[0]["required_gpu_free_mb"], 1000 / .75)
        self.assertAlmostEqual(reports[-1]["required_gpu_free_mb"], 3000 / .75)

    def test_failed_memory_query_is_not_treated_as_free_and_later_recovers(self):
        queries = 0
        def inventory():
            nonlocal queries
            queries += 1
            if queries <= 2:
                return self.devices, (), {d: {"error": "CUDA error: out of memory"} for d in self.devices}
            return self.devices, tuple(self.info), {}
        self.probe.side_effect = inventory
        pool, prepared = self.prepare([task()])
        seen = []
        pool.queue(prepared, run=lambda t: seen.append(t.config.device), commit=lambda *a: None,
                   event=None, poll_seconds=.001, max_idle_seconds=1)
        self.assertEqual(seen, ["cuda:0"])
        self.assertEqual(pool.query_failures, {})
        self.assertGreaterEqual(queries, 3)

    def test_oom_retry_changes_card_once_and_keeps_checkpoint_identity(self):
        pool, prepared = self.prepare([task()])
        seen, commits, events = [], [], []
        identity = prepared[0].config
        def run(item):
            seen.append(item.config.device)
            self.assertEqual(item.checkpoint_config, identity)
            if len(seen) == 1:
                raise RuntimeError("CUDA out of memory")
            return 7
        pool.queue(prepared, run=run, commit=lambda t, r: commits.append(r),
                   event=lambda kind, item, **k: events.append((kind, k)),
                   poll_seconds=.001, cooldown_seconds=10)
        self.assertEqual(seen, ["cuda:0", "cuda:1"])
        self.assertEqual(commits, [7])
        self.assertTrue(next(k["retry"] for event, k in events if event == "cuda_oom"))

    def test_second_oom_is_fatal_without_fabricating_completion(self):
        pool, prepared = self.prepare([task()])
        attempts, commits, retries = [], [], []
        def run(item):
            attempts.append(item.config.device)
            raise RuntimeError("CUDA error: out of memory")
        with self.assertRaisesRegex(cuda_execution.CudaResourcesUnavailable, "retry limit"):
            pool.queue(prepared, run=run, commit=lambda *a: commits.append(a),
                       event=lambda kind, t, **k: retries.append(k["retry"]) if kind == "cuda_oom" else None,
                       poll_seconds=.001, cooldown_seconds=0)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(commits, [])
        self.assertEqual(retries, [True, False])

    def test_non_oom_failure_drains_other_running_task_but_does_not_start_pending(self):
        pool, prepared = self.prepare([task(seed=i) for i in range(3)])
        failure_observed, seen, commits = Event(), [], []
        def run(item):
            seen.append(item.config.seed)
            if item.config.seed == 0:
                raise RuntimeError("CUDNN_STATUS_INTERNAL_ERROR")
            self.assertTrue(failure_observed.wait(2))
            return item.config.seed
        with self.assertRaisesRegex(RuntimeError, "CUDNN_STATUS_INTERNAL_ERROR"):
            pool.queue(prepared, run=run, commit=lambda t, r: commits.append(r),
                       event=lambda kind, t, **k: failure_observed.set() if kind == "cuda_worker_failed" else None,
                       poll_seconds=.001)
        self.assertCountEqual(seen, [0, 1])
        self.assertEqual(commits, [1])

    def test_idle_timeout_is_resumable_and_emits_original_wait_event(self):
        self.info = [(d, 0., 24000.) for d in self.devices]
        pool, prepared = self.prepare([task()])
        events = []
        with self.assertRaisesRegex(cuda_execution.CudaResourcesUnavailable, "checkpoints are retained"):
            pool.queue(prepared, run=lambda t: self.fail("must not run"), commit=lambda *a: None,
                       event=lambda kind, t, **k: events.append(kind), max_idle_seconds=0)
        self.assertEqual(events, ["cuda_waiting_for_memory"])

    def test_context_restores_exact_hooks_on_failure_and_fans_out_events(self):
        originals = fair_tuning.prepare_tuning_tasks, fair_tuning.run_cuda_tasks, experiments.cuda_device_memory_info
        original_events, panel_events = [], []
        with self.assertRaisesRegex(ValueError, "after queue"):
            with policy.gpu_pool(on_event=lambda kind, t, **k: panel_events.append(kind)) as pool:
                prepared, _, _, _ = fair_tuning.prepare_tuning_tasks(None, [task()], self.args)
                self.info = [(d, 0., 24000.) for d in self.devices]
                with self.assertRaises(cuda_execution.CudaResourcesUnavailable):
                    fair_tuning.run_cuda_tasks(prepared, run=lambda t: None, commit=lambda *a: None,
                        event=lambda kind, t, **k: original_events.append(kind), max_idle_seconds=0)
                self.assertIs(experiments.cuda_device_memory_info.__self__, pool)
                raise ValueError("after queue")
        self.assertEqual((fair_tuning.prepare_tuning_tasks, fair_tuning.run_cuda_tasks,
                          experiments.cuda_device_memory_info), originals)
        self.assertEqual(original_events, panel_events)


if __name__ == "__main__":
    unittest.main()
