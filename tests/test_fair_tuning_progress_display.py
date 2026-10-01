"""MNIST display-only integration: synthetic results, no dataset or training."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import threading
import unittest
from unittest import mock

import fair_tuning_progress_display as display
from sm9rrsfl import experiments, fair_tuning
from tests.test_cifar_six_pipeline import synthetic_run
from tests.test_tuning_comparison_failures import fixture_dataset


class Terminal(io.StringIO):
    def __init__(self):
        super().__init__()
        self.chunks = []

    def isatty(self):
        return True

    def write(self, value):
        self.chunks.append(value)
        return super().write(value)


def formal_tasks():
    config = experiments.ExperimentConfig(method="fedavg", num_clients=10,
        rounds=3, early_stop=False, crypto_mode="simulated")
    return [fair_tuning.TuningExperimentTask("final", "fedavg-fixed", "fedavg",
        replace(config, seed=seed)) for seed in (241, 242)]


class FairTuningProgressDisplayTests(unittest.TestCase):
    def test_context_restores_stdout_classes_and_joins_abandoned_reporter_on_failure(self):
        stream = Terminal()
        original_experiments = experiments.ProgressReporter
        original_tuning = fair_tuning.ProgressReporter
        with redirect_stdout(stream):
            with self.assertRaisesRegex(RuntimeError, "training failure"):
                with display.progress_display():
                    self.assertIsNot(sys.stdout, stream)
                    self.assertIsNot(fair_tuning.ProgressReporter, original_tuning)
                    reporter = fair_tuning.ProgressReporter(total=2, stream=sys.stdout,
                                                          mode="live", refresh_interval=.01)
                    raise RuntimeError("training failure")
            self.assertIs(sys.stdout, stream)
        self.assertIs(experiments.ProgressReporter, original_experiments)
        self.assertIs(fair_tuning.ProgressReporter, original_tuning)
        self.assertTrue(reporter.closed)
        self.assertFalse(reporter._refresh_thread.is_alive())
        self.assertIn("interrupted", stream.getvalue())
        self.assertNotIn(" complete", stream.getvalue())
        self.assertTrue(stream.getvalue().endswith("\n"))

    def test_live_stage_and_cuda_logs_clear_progress_before_writing(self):
        stream = Terminal()
        with redirect_stdout(stream), display.progress_display():
            reporter = fair_tuning.ProgressReporter(total=2, stream=sys.stdout, mode="live")
            print("tuning_phase=final configurations=2")
            print("cuda_recovery=synthetic retry")
            print("resuming from_completed_round=2")
            reporter.finish_item("finished synthetic task")
            reporter.close()
        raw = stream.getvalue()
        for line in ("tuning_phase=final configurations=2", "cuda_recovery=synthetic retry",
                     "resuming from_completed_round=2"):
            self.assertIn("\r\033[2K" + line + "\n", raw)
        self.assertNotIn("startingtuning_phase", raw)
        self.assertIn("1/2", raw)

    def test_all_live_refreshes_fit_80_and_narrow_column_terminals(self):
        for columns in (1, 20, 40, 80):
            with self.subTest(columns=columns):
                stream = Terminal()
                with mock.patch.object(display.shutil, "get_terminal_size", return_value=os.terminal_size((columns, 24))), \
                        redirect_stdout(stream), display.progress_display():
                    reporter = fair_tuning.ProgressReporter(total=180, completed=73,
                                                          stream=sys.stdout, mode="live")
                    reporter.start_item("模型恢复 " * 40)
                    reporter.finish_item("finished " * 40)
                rendered = [chunk[1:] for chunk in stream.chunks
                            if chunk.startswith("\r") and "\033" not in chunk]
                self.assertTrue(rendered)
                self.assertTrue(all(display._display_width(row) <= max(0, columns - 1) for row in rendered))
                if columns >= 20:
                    self.assertTrue(any("73/180" in row for row in rendered))
                    self.assertTrue(any("74/180" in row for row in rendered))

    def test_non_tty_live_and_auto_are_readable_logs_without_refresh_threads(self):
        for mode in ("live", "auto", "log"):
            with self.subTest(mode=mode):
                stream = io.StringIO()
                with redirect_stdout(stream), display.progress_display():
                    reporter = fair_tuning.ProgressReporter(total=1, stream=sys.stdout, mode=mode)
                    print("tuning_phase=final")
                    reporter.finish_item("finished task")
                    self.assertIsNone(reporter._refresh_thread)
                self.assertNotIn("\r", stream.getvalue())
                self.assertNotIn("\033", stream.getvalue())
                self.assertIn("tuning_phase=final\n", stream.getvalue())
                self.assertIn("1/1 100.0%", stream.getvalue())
                self.assertIn("complete\n", stream.getvalue())

    def test_explicit_log_mode_on_tty_never_uses_cursor_controls(self):
        stream = Terminal()
        with redirect_stdout(stream), display.progress_display():
            reporter = fair_tuning.ProgressReporter(total=1, stream=sys.stdout, mode="log")
            print("phase log")
            reporter.finish_item("finished")
            self.assertIsNone(reporter._refresh_thread)
        self.assertNotIn("\033", stream.getvalue())
        self.assertNotIn("\r", stream.getvalue())

    def test_refresh_thread_does_not_split_partial_print_output(self):
        stream = Terminal()
        ready, refreshed = threading.Event(), threading.Event()
        with redirect_stdout(stream), display.progress_display():
            reporter = fair_tuning.ProgressReporter(total=2, stream=sys.stdout, mode="live", refresh_interval=.01)
            print("CUDA event fragment", end="", flush=True)
            def refresh():
                ready.wait(1.)
                reporter._write(reporter._progress_message())
                refreshed.set()
            worker = threading.Thread(target=refresh)
            worker.start()
            ready.set()
            self.assertTrue(refreshed.wait(1.))
            print(" completed")
            worker.join(1.)
            self.assertFalse(worker.is_alive())
            reporter.close()
        self.assertIn("CUDA event fragment completed\n", stream.getvalue())
        self.assertFalse(reporter._refresh_thread.is_alive())

    def test_live_tty_periodic_refresh_runs_without_completed_tasks(self):
        refreshed = threading.Event()
        class RefreshTerminal(Terminal):
            def write(self, text):
                result = super().write(text)
                if sum(chunk.startswith("\r[") for chunk in self.chunks) >= 2:
                    refreshed.set()
                return result
        stream = RefreshTerminal()
        with redirect_stdout(stream), display.progress_display():
            reporter = fair_tuning.ProgressReporter(total=180, stream=sys.stdout,
                                                  mode="live", refresh_interval=.01)
            self.assertTrue(refreshed.wait(.5))
            self.assertEqual(reporter.completed, 0)
        self.assertFalse(reporter._refresh_thread.is_alive())

    def test_broken_stream_still_closes_all_reporters_and_preserves_training_error(self):
        class BrokenTerminal(Terminal):
            broken = False
            def write(self, text):
                if self.broken:
                    raise OSError("synthetic stdout failure")
                return super().write(text)
        for training_error in (False, True):
            with self.subTest(training_error=training_error):
                stream = BrokenTerminal()
                original = fair_tuning.ProgressReporter
                exception = RuntimeError if training_error else OSError
                message = "original training failure" if training_error else "synthetic stdout failure"
                with redirect_stdout(stream):
                    with self.assertRaisesRegex(exception, message):
                        with display.progress_display():
                            first = fair_tuning.ProgressReporter(total=2, stream=sys.stdout, mode="live")
                            second = fair_tuning.ProgressReporter(total=2, stream=sys.stdout, mode="live")
                            stream.broken = True
                            if training_error:
                                raise RuntimeError("original training failure")
                    self.assertIs(sys.stdout, stream)
                self.assertIs(fair_tuning.ProgressReporter, original)
                for reporter in (first, second):
                    self.assertTrue(reporter.closed)
                    self.assertFalse(reporter._refresh_thread.is_alive())

    def test_many_normal_lines_survive_concurrent_live_refreshes(self):
        stream = Terminal()
        with redirect_stdout(stream), display.progress_display():
            reporter = fair_tuning.ProgressReporter(total=20, stream=sys.stdout, mode="live", refresh_interval=.01)
            def emit():
                for i in range(80):
                    print(f"worker_log_{i:03d}")
            worker = threading.Thread(target=emit)
            worker.start()
            for i in range(20):
                reporter.finish_item(f"finished {i}")
            worker.join(1.)
            self.assertFalse(worker.is_alive())
            reporter.close()
        for i in range(80):
            self.assertIn(f"worker_log_{i:03d}\n", stream.getvalue())
        self.assertIn("20/20 100.0%", stream.getvalue())
        self.assertFalse(reporter._refresh_thread.is_alive())

    def test_real_formal_executor_has_no_stage_line_concatenation(self):
        stream = Terminal()
        with redirect_stdout(stream), display.progress_display(), \
                mock.patch.object(fair_tuning, "run_measured_experiment",
                    side_effect=lambda _dataset, config, **kwargs: synthetic_run(config)):
            completed = fair_tuning.execute_tuning_tasks(fixture_dataset("mnist"), formal_tasks(),
                jobs=1, backend_description="numpy", progress_enabled=True, progress_mode="live")
        self.assertEqual(len(completed), 2)
        self.assertIn("\033[2Ktuning_phase=final configurations=2 jobs=1 executor=serial\n", stream.getvalue())
        self.assertNotIn("startingtuning_phase", stream.getvalue())
        self.assertIn("2/2 100.0%", stream.getvalue())

    def test_failure_and_resume_keep_one_of_two_then_finish_cached_total(self):
        tasks, dataset = formal_tasks(), fixture_dataset("mnist")
        args = experiments.parse_args(["--no-early-stop"])
        stream, snapshots = Terminal(), []
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(output_dir=Path(directory), jobs=1, backend_description="numpy",
                          progress_enabled=True, progress_mode="live",
                          on_snapshot=lambda rows, fp, status: snapshots.append((len(rows), status)))
            with redirect_stdout(stream), display.progress_display(), \
                    mock.patch.object(fair_tuning, "run_measured_experiment",
                        side_effect=[synthetic_run(tasks[0].config), RuntimeError("synthetic failure")]):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    fair_tuning.execute_resumable_tuning_phase(dataset, tasks, args, **kwargs)
            self.assertEqual(snapshots[-1], (1, "interrupted"))
            self.assertIn("1/2", stream.getvalue())
            self.assertNotIn("2/2", stream.getvalue())
            resumed = Terminal()
            with redirect_stdout(resumed), display.progress_display(), \
                    mock.patch.object(fair_tuning, "run_measured_experiment",
                        side_effect=lambda _dataset, config, **kw: synthetic_run(config)) as run:
                completed, _ = fair_tuning.execute_resumable_tuning_phase(dataset, tasks, args, **kwargs)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(len(completed), 2)
            self.assertIn("resumed_completed_configurations=1", resumed.getvalue())
            self.assertIn("1/2  50.0%", resumed.getvalue())
            self.assertIn("2/2 100.0%", resumed.getvalue())
            self.assertEqual(snapshots[-1], (2, "complete"))
            with redirect_stdout(io.StringIO()), display.progress_display(), \
                    mock.patch.object(fair_tuning, "run_measured_experiment") as cached:
                completed, _ = fair_tuning.execute_resumable_tuning_phase(dataset, tasks, args, **kwargs)
            cached.assert_not_called()
            self.assertEqual(len(completed), 2)

    def test_disabled_progress_keeps_existing_plain_output(self):
        stream = Terminal()
        with redirect_stdout(stream), display.progress_display():
            reporter = fair_tuning.ProgressReporter(total=1, enabled=False, stream=sys.stdout, mode="live")
            reporter.start_item("running task")
            reporter.finish_item("finished task")
            self.assertIsNone(reporter._refresh_thread)
        self.assertEqual(stream.getvalue(), "running task\nfinished task\n")

    def test_child_process_resets_inherited_lock_and_does_not_redraw_parent(self):
        stream = Terminal()
        output = display._Output(stream)
        output.render("parent progress", live=True, final=False)
        parent_lock = output.lock
        parent_pid = output.pid
        stream.seek(0)
        stream.truncate()
        with mock.patch.object(display.os, "getpid", return_value=parent_pid + 1):
            output.write("child normal log\n")
        self.assertIsNot(output.lock, parent_lock)
        self.assertFalse(output.tty)
        self.assertEqual(stream.getvalue(), "child normal log\n")

    def test_original_entry_enables_adapter_and_restores_it_after_main(self):
        root = Path(__file__).resolve().parents[1]
        original = fair_tuning.ProgressReporter
        observed = []
        def main(argv):
            observed.append((list(argv), fair_tuning.ProgressReporter is original))
            print("entry reached")
        with mock.patch("run_experiments_from_config._try_project_virtualenv", return_value=False), \
                mock.patch.object(fair_tuning, "main", side_effect=main), \
                mock.patch.object(sys, "argv", ["run_fair_tuning_from_config.py", "--dry-run"]), \
                redirect_stdout(io.StringIO()) as stream:
            runpy.run_path(str(root / "run_fair_tuning_from_config.py"), run_name="__main__")
        self.assertEqual(observed, [(["--dry-run"], False)])
        self.assertIs(fair_tuning.ProgressReporter, original)
        self.assertEqual(stream.getvalue(), "entry reached\n")


if __name__ == "__main__":
    unittest.main()
