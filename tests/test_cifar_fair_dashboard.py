import io
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event, enumerate as threads
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.experiments.cifar10 import fair_tuning_dashboard as ui
from sm9rrsfl import experiments, fair_tuning
from sm9rrsfl.datasets import make_synthetic_mnist_like
from sm9rrsfl.fl import ExperimentConfig
import run_cifar_v14_with_progress as launcher


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cifar10_v14_fair_grid.json"


def task(seed=1, device="cuda:0", phase="validation", candidate="sm9rrs-001"):
    config = ExperimentConfig(method="sm9rrs", seed=seed, device=device, rounds=150,
                              detector_subspace_dim=3, detector_distance_threshold=2.5,
                              detector_history_threshold=1.5)
    return fair_tuning.TuningExperimentTask(phase, candidate, "sm9rrs", config)


class Terminal(io.StringIO):
    def isatty(self):
        return True


class DashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = fair_tuning.load_fair_tuning_config(CONFIG)

    def test_plan_matches_actual_validation_and_final_expansion(self):
        args, validation, work = ui.plan(self.spec)
        selected = {method: SimpleNamespace(candidate_id=f"{method}-001", parameters=values[0])
                    for method, values in self.spec.candidates.items()}
        final = fair_tuning.build_final_tasks(self.spec, experiments.build_experiment_configs(args), selected)
        self.assertEqual(len(validation), 1170)
        self.assertEqual(len(final), work["final_tasks"])
        self.assertEqual(work["total_tasks"], 1350)
        self.assertEqual(work["validation_rounds"] + work["final_rounds"], 202500)
        self.assertEqual({t.config.num_clients for t in validation + final}, {100})
        self.assertEqual({t.config.attack_boost for t in validation + final}, {20.})
        self.assertEqual({t.config.attack_source_label for t in validation + final}, {5})
        self.assertEqual({t.config.attack_target_label for t in validation + final}, {7})

    def test_dry_run_never_loads_data_or_runs_training(self):
        output = io.StringIO()
        with redirect_stdout(output), mock.patch.object(fair_tuning, "run_fair_tuning") as run, \
                mock.patch.object(fair_tuning, "load_image_dataset") as load:
            launcher.main(["--config", str(CONFIG), "--dry-run"])
        run.assert_not_called()
        load.assert_not_called()
        self.assertIn('"total_rounds": 202500', output.getvalue())

    def test_cuda_registration_does_not_duplicate_auto_plan(self):
        state = ui.Progress(self.spec)
        self.assertFalse(state.tasks)
        item = task()
        state.register([item])
        moved = replace(item, config=replace(item.config, device="cuda:7", sm9_workers=2))
        state.register([moved])
        self.assertEqual(len(state.tasks), 1)
        key = state.start(moved.config)
        state.advance(key, 12)
        self.assertEqual(state.totals(), (12, 0))
        self.assertEqual(state.lanes["cuda:7"], key)

    def test_resume_and_cache_never_accelerate_eta_or_count_round_zero(self):
        now = [0.]
        state = ui.Progress(self.spec, now=lambda: now[0])
        cached, resumed = task(1), task(2)
        state.register([cached, resumed])
        state.snapshot([(cached, SimpleNamespace(stopped_round=150))], "running")
        key = state.start(resumed.config)
        state.resume(key, 40)
        self.assertIsNone(state.eta())
        now[0] = 10.
        state.advance(key, 41)
        state.advance(key, 41)
        state.advance(key, 0)
        self.assertEqual(state.new_rounds, 1)
        self.assertEqual(state.totals(), (191, 1))
        self.assertEqual(state.eta(), (202500 - 191) * 10)

    def test_failures_do_not_complete_tasks_and_final_keeps_validation_progress(self):
        state = ui.Progress(self.spec)
        first = task()
        state.register([first])
        key = state.start(first.config)
        state.advance(key, 100)
        state.stop(key, failed=True)
        self.assertEqual(state.totals(), (100, 0))
        self.assertEqual(state.failed_attempts, 1)
        self.assertFalse(state.lanes)
        state.snapshot([(first, SimpleNamespace(stopped_round=150))], "complete")
        final = task(401, phase="final")
        state.register([final])
        state.advance(state.start(final.config), 2)
        self.assertEqual(state.totals(), (152, 1))
        self.assertEqual(state.phase, "final")

    def test_all_eight_gpu_lanes_and_parameters_fit_80_columns(self):
        state = ui.Progress(self.spec)
        items = [task(i, f"cuda:{i}") for i in range(8)]
        state.gpu_plan({"pool_devices": [item.config.device for item in items], "target_jobs": 8})
        state.register(items)
        for item in items:
            state.advance(state.start(item.config), 21)
        lines = state.lines(width=79, height=23)
        text = "\n".join(lines)
        self.assertLessEqual(len(lines), 23)
        self.assertTrue(all(ui._display_width(line) <= 79 for line in lines))
        for i in range(8):
            self.assertIn(f"cuda:{i}", text)
        self.assertIn("q=3 distance=2.5 history=1.5", text)
        self.assertIn("gpu=8/8 slots=8", text)
        self.assertIn("rounds=168/202500", text)
        for width, height in ((39, 9), (1, 1), (0, 1)):
            rows = state.lines(width=width, height=height)
            self.assertLessEqual(len(rows), height)
            self.assertTrue(all(ui._display_width(row) <= width for row in rows))

    def test_log_and_redirected_live_modes_never_emit_ansi(self):
        for mode in ("log", "auto", "live"):
            stream = io.StringIO()
            with redirect_stdout(stream), ui.dashboard(self.spec, mode=mode):
                print("log preserved")
            self.assertNotIn("\033", stream.getvalue())
            self.assertIn("rounds=0/202500", stream.getvalue())
            self.assertIn("log preserved", stream.getvalue())

    def test_live_refresh_does_not_split_partial_stdout_or_stderr(self):
        stream, errors = Terminal(), io.StringIO()
        ready = Event()
        with redirect_stdout(stream), redirect_stderr(errors), ui.dashboard(self.spec, refresh=.01):
            print("partial message", end="", flush=True)
            ready.wait(.03)
            print(" finished")
            print("warning preserved", file=__import__("sys").stderr)
        self.assertIn("partial message finished\n", stream.getvalue())
        self.assertIn("warning preserved", errors.getvalue())
        self.assertFalse(any(t.name == "cifar-fair-dashboard" for t in threads()))

    def test_phase_snapshot_observer_retains_original_commit_callback(self):
        item = task()
        seen = []

        def phase(_dataset, tasks, args, **kwargs):
            kwargs["on_snapshot"]([(item, SimpleNamespace(stopped_round=150))], "fingerprint", "complete")
            return "unchanged return"

        with mock.patch.object(fair_tuning, "execute_resumable_tuning_phase", side_effect=phase), \
                redirect_stdout(io.StringIO()), ui.dashboard(self.spec) as state:
            result = fair_tuning.execute_resumable_tuning_phase(None, [item], None,
                on_snapshot=lambda *args: seen.append(args))
            self.assertEqual(result, "unchanged return")
            self.assertEqual(state.totals(), (150, 1))
        self.assertEqual(seen[0][1:], ("fingerprint", "complete"))

    def test_concurrent_workers_keep_rounds_and_callbacks_on_their_own_task(self):
        gate = Barrier(2)
        callback_states = []
        items = [task(1, "cuda:0"), task(2, "cuda:1")]

        def train(_dataset, config, **kwargs):
            gate.wait(timeout=2.)
            state = {"completed_round": config.seed * 3}
            kwargs["checkpoint_callback"](state)
            return state

        def measured(dataset, config, **kwargs):
            self.assertFalse(kwargs["emit_progress_events"])
            return experiments.run_experiment(dataset, config, resume_state=None,
                checkpoint_callback=lambda state: callback_states.append(state))

        with mock.patch.object(experiments, "run_experiment", side_effect=train), \
                mock.patch.object(fair_tuning, "run_measured_experiment", side_effect=measured), \
                redirect_stdout(io.StringIO()), ui.dashboard(self.spec) as state:
            state.register(items)
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda item: fair_tuning.run_measured_experiment(None, item.config), items))
            self.assertEqual(state.totals(), (9, 0))
            self.assertFalse(state.lanes)
            self.assertEqual({id(row) for row in results}, {id(row) for row in callback_states})
            for item in items:
                self.assertEqual(state.tasks[state.key("validation", item.config)].round, item.config.seed * 3)

    def test_real_tiny_cpu_run_matches_without_display(self):
        dataset = make_synthetic_mnist_like(train_samples=40, test_samples=10, seed=7)
        config = ExperimentConfig(method="fedavg", num_clients=2, rounds=2,
            compute_backend="numpy", device="cpu", crypto_mode="simulated", early_stop=False, seed=5)
        baseline = experiments.run_measured_experiment(dataset, config)
        item = fair_tuning.TuningExperimentTask("validation", "fedavg-001", "fedavg", config)
        with redirect_stdout(io.StringIO()), ui.dashboard(self.spec) as state:
            state.register([item])
            observed = fair_tuning.run_measured_experiment(dataset, config)
            self.assertEqual(state.new_rounds, 2)
            self.assertEqual(state.totals(), (2, 0))
        self.assertEqual(observed.records, baseline.records)
        self.assertEqual(observed.final_accuracy, baseline.final_accuracy)
        self.assertEqual(observed.stopped_round, baseline.stopped_round)

    def test_training_exception_restores_hooks_and_preserves_original_error(self):
        class BrokenTerminal(Terminal):
            broken = False

            def write(self, text):
                if self.broken:
                    raise OSError("broken display")
                return super().write(text)

        stream = BrokenTerminal()
        item = task()
        original = (fair_tuning.execute_resumable_tuning_phase, fair_tuning.run_measured_experiment,
                    fair_tuning.ProgressReporter, experiments.run_experiment)
        with redirect_stdout(stream), self.assertRaisesRegex(RuntimeError, "training failure"):
            with ui.dashboard(self.spec) as state:
                state.register([item])
                stream.broken = True
                raise RuntimeError("training failure")
        self.assertEqual(original, (fair_tuning.execute_resumable_tuning_phase,
            fair_tuning.run_measured_experiment, fair_tuning.ProgressReporter, experiments.run_experiment))
        self.assertFalse(any(t.name == "cifar-fair-dashboard" for t in threads()))

    def test_forked_cpu_worker_bypasses_parent_display_state(self):
        original_pid = os.getpid()
        sentinel = object()
        with mock.patch.object(fair_tuning, "run_measured_experiment", return_value=sentinel), \
                redirect_stdout(io.StringIO()), ui.dashboard(self.spec) as state:
            with mock.patch.object(ui.os, "getpid", return_value=original_pid + 1):
                self.assertIs(fair_tuning.run_measured_experiment(None, task().config), sentinel)
            self.assertFalse(state.lanes)

    def test_worker_exception_is_failed_attempt_and_never_a_completion(self):
        with mock.patch.object(fair_tuning, "run_measured_experiment", side_effect=ValueError("failed worker")), \
                redirect_stdout(io.StringIO()), ui.dashboard(self.spec) as state:
            item = task()
            state.register([item])
            with self.assertRaisesRegex(ValueError, "failed worker"):
                fair_tuning.run_measured_experiment(None, item.config)
            self.assertEqual(state.failed_attempts, 1)
            self.assertEqual(state.totals(), (0, 0))


if __name__ == "__main__":
    unittest.main()
