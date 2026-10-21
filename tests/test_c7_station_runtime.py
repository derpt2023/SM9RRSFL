"""Small concurrent fixtures for C7 station identity, isolation, and recovery."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from sm9rrsfl import experiments, fair_tuning, fl, torch_backend
from sm9rrsfl.datasets import ImageDataset
from scripts.experiments.cifar10 import attack_stability_attack as attack
from scripts.experiments.cifar10 import c7_station_runtime as runtime


def dataset(count=20):
    return ImageDataset(
        x_train=np.zeros((20, 1, 8, 8), np.float32), y_train=np.arange(20) % 10,
        x_test=np.zeros((count, 1, 8, 8), np.float32), y_test=np.full(count, 5),
        x_attack=np.zeros((5, 1, 8, 8), np.float32), y_attack=np.full(5, 5),
        name="mnist", num_classes=10)


def record(config, round_id):
    return fl.RoundRecord(
        method=config.method, malicious_ratio=config.malicious_ratio,
        round=round_id, accuracy=.5, error=.5, accepted_updates=4,
        rejected_updates=0, blacklisted_clients=0,
        true_positive_revocations=0, false_positive_revocations=0,
        krum_selected_client="", attack_target_success_rate=0., attack_target_confidence=.1)


def state(config, round_id, rows):
    return {"completed_round": round_id, "params": np.zeros(5, np.float32),
            "records": rows, "nonfinite_updates": 0}


def result(config, rows):
    return fl.ExperimentResult(
        config=config, records=rows, final_accuracy=.5, final_error=.5,
        stopped_round=rows[-1].round, malicious_clients=("client-0",),
        blacklisted_clients=(), stage_timings=fl.StageTimings(attack_seconds=1.25))


class StationRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = dataset()
        self.config = fl.ExperimentConfig(
            method="fedavg", rounds=2, num_clients=4, malicious_ratio=.5,
            attack_start_round=1, detector_window=0, early_stop=False,
            compute_backend="torch", device="cuda:0", eval_interval=1,
            attack_source_label=5, attack_target_label=7, attack_target_count=1)
        self.resumes = {}
        self.callback = None
        self.barrier = None
        self.contexts = {}
        self.local_return = (np.ones(5), SimpleNamespace(loss=.1))
        self.observed_original_callbacks = []

        def context(_dataset, _indices, _spec, config):
            value = SimpleNamespace(device=config.device, token=config.seed)
            self.contexts[config.seed] = value
            return value

        def target_metrics(evaluator, vector, group, indices):
            # Different threads must retain their own evaluator/context.
            token = evaluator.torch_context.token
            return ((1., .9) if group == "attack" else (.2, token / 1000.))

        def measured(data, config, **kwargs):
            return experiments.run_experiment(
                data, config, resume_state=self.resumes.get(config.seed),
                checkpoint_callback=self.callback)

        def attack_client(*_args, **kwargs):
            return torch_backend.TorchTrainingContext.alternating_minimization_delta_resident(
                SimpleNamespace(device=kwargs["config"].device), np.zeros(5),
                client_idx=kwargs["client_idx"], target_indices=np.array([0]),
                target_label=7, lr=.01, attack_epochs=1, batch_size=50,
                stealth_steps=1, boost=20., distance_weight=.0001, seed=301)

        def profile_attack(ctx, vector, profile, **kwargs):
            return np.ones(5), SimpleNamespace(loss=.1), {
                "profile_id": "C7", "client_idx": kwargs["client_idx"],
                "status": "complete", "normal_steps": 3, "reference_steps": 3,
                "target_steps": 1, "target_skipped_zero_loss": 0}

        self.patches = [
            mock.patch.object(fair_tuning, "run_measured_experiment", side_effect=measured),
            mock.patch.object(experiments, "run_experiment", side_effect=self.training),
            mock.patch.object(fl, "_maybe_torch_context", side_effect=context),
            mock.patch.object(fl, "_local_train_client_delta", return_value=self.local_return),
            mock.patch.object(fl, "_alternating_minimization_client_delta", side_effect=attack_client),
            mock.patch.object(attack, "run_profile_attack", side_effect=profile_attack),
            mock.patch.object(runtime.StationMetricEvaluator, "_target_metrics", target_metrics),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def training(self, data, config, *, resume_state=None, checkpoint_callback=None):
        fl._maybe_torch_context(data, None, None, config)
        if self.barrier:
            self.barrier.wait(timeout=5)
        rows = list(resume_state["records"]) if resume_state else []
        start = resume_state["completed_round"] + 1 if resume_state else 0
        for round_id in range(start, config.rounds + 1):
            if round_id:
                actual = fl._local_train_client_delta(
                    None, None, None, client_idx=1, round_id=round_id, config=config)
                self.assertIs(actual, self.local_return)
                if config.malicious_ratio:
                    fl._alternating_minimization_client_delta(
                        None, None, None, client_idx=0, round_id=round_id, config=config)
            rows.append(record(config, round_id))
            checkpoint_callback(state(config, round_id, list(rows)))
        return result(config, rows)

    def run_task(self, config=None, phase="validation", data=None):
        config = config or self.config
        return fair_tuning.run_measured_experiment(
            data or self.data, config, task_id=f"{phase}_{config.method}_fixed",
            run_fingerprint="phase-source-sha")

    def path(self, file, config=None, phase="validation"):
        return runtime.task_directory(self.root, phase, config or self.config) / file

    def read(self, file, **kwargs):
        return json.loads(self.path(file, **kwargs).read_text())

    def test_two_cuda_threads_keep_task_client_device_and_metric_context_isolated(self):
        self.barrier = threading.Barrier(2)
        configs = [replace(self.config, seed=301, device="cuda:0"),
                   replace(self.config, seed=402, device="cuda:3", method="vert")]
        with runtime.c7_station_runtime(self.root):
            with ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(self.run_task, config) for config in configs]
                for future in futures:
                    future.result(timeout=10)
        for config in configs:
            rows = self.read("metrics.json", config=config)["rounds"]
            self.assertEqual([row["round"] for row in rows], [0, 1, 2])
            self.assertEqual([row["attack_calls_this_round"] for row in rows], [0, 1, 1])
            self.assertEqual(rows[-1]["source_transfer_confidence"], config.seed / 1000.)
            audits = [json.loads(line) for line in self.path("attack_audit.jsonl", config).read_text().splitlines()]
            self.assertEqual({r["device"] for r in audits}, {config.device})
            self.assertEqual({r["task_key"] for r in audits}, {runtime.task_key(config)})
            self.assertEqual([r["client_idx"] for r in audits], [0, 0])
            self.assertEqual(self.read("status.json", config=config)["status"], "complete")

    def test_clean_tasks_preserve_sgd_result_and_record_explicit_zero_calls(self):
        config = replace(self.config, malicious_ratio=0.)
        with runtime.c7_station_runtime(self.root):
            self.run_task(config)
        self.assertEqual(self.path("attack_audit.jsonl", config).read_text(), "")
        self.assertFalse(self.read("task.json", config=config)["attack_enabled"])
        self.assertEqual([r["attack_calls_this_round"] for r in self.read("metrics.json", config=config)["rounds"]], [0, 0, 0])

    def test_metrics_are_durable_before_original_checkpoint_callback(self):
        def callback(raw):
            committed = self.read("metrics.json")["rounds"]
            self.assertEqual(committed[-1]["round"], raw["completed_round"])
            self.observed_original_callbacks.append(raw["completed_round"])
        self.callback = callback
        with runtime.c7_station_runtime(self.root):
            self.run_task()
        self.assertEqual(self.observed_original_callbacks, [0, 1, 2])

    def test_resume_discards_metrics_beyond_checkpoint_and_preserves_attempt_evidence(self):
        def callback(raw):
            if raw["completed_round"] == 1:
                self.resumes[self.config.seed] = raw
            if raw["completed_round"] == 2:
                raise RuntimeError("CUDA out of memory")
        self.callback = callback
        with runtime.c7_station_runtime(self.root):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                self.run_task()
            before = self.path("first_failure.json").read_bytes()
            self.callback = None
            self.run_task(replace(self.config, device="cuda:4", sm9_workers=8))
        rows = self.read("metrics.json")["rounds"]
        self.assertEqual([r["round"] for r in rows], [0, 1, 2])
        self.assertEqual(rows[-1]["attack_calls_this_round"], 1)
        audits = [json.loads(line) for line in self.path("attack_audit.jsonl").read_text().splitlines()]
        self.assertEqual([row["round"] for row in audits], [1, 2, 2])
        self.assertEqual(len({row["attempt_id"] for row in audits}), 2)
        self.assertEqual(self.path("first_failure.json").read_bytes(), before)
        self.assertEqual(self.read("status.json")["device"], "cuda:4")
        self.assertEqual(self.read("task.json")["config"]["device"], "cuda:0")

    def test_resume_rejects_missing_extended_metric_prefix(self):
        with runtime.c7_station_runtime(self.root):
            self.run_task()
            self.resumes[self.config.seed] = state(self.config, 1, [record(self.config, r) for r in (0, 1)])
            stored = self.read("metrics.json")
            stored["rounds"] = stored["rounds"][:1]
            runtime._write_json(self.path("metrics.json"), stored)
            with self.assertRaisesRegex(ValueError, "missing.*prefix"):
                self.run_task()

    def test_terminal_replay_requires_metrics_and_records_no_new_attack(self):
        with runtime.c7_station_runtime(self.root):
            completed = self.run_task()
        before = self.path("attack_audit.jsonl").read_bytes()
        with mock.patch.object(fair_tuning, "run_measured_experiment", return_value=completed):
            with runtime.c7_station_runtime(self.root):
                self.run_task()
        self.assertEqual(self.path("attack_audit.jsonl").read_bytes(), before)
        self.assertTrue(self.read("result.json")["terminal_checkpoint_replayed"])

    def test_early_result_is_preserved_but_not_committed_as_complete(self):
        def stop_early(data, config, **kwargs):
            return self.training(data, replace(config, rounds=1), **kwargs)
        with mock.patch.object(experiments, "run_experiment", side_effect=stop_early):
            with runtime.c7_station_runtime(self.root):
                with self.assertRaisesRegex(RuntimeError, "round 1/2"):
                    self.run_task()
        self.assertEqual(self.read("result.json")["status"], "incomplete")
        self.assertEqual(self.read("status.json")["status"], "failed")
        self.assertEqual(self.read("status.json")["completed_round"], 1)

    def test_phase_dataset_scope_and_actual_denominators_are_distinct(self):
        with runtime.c7_station_runtime(self.root):
            self.run_task(data=dataset(750))
            self.run_task(phase="final", data=dataset(1000))
        for phase, count, scope in (("validation", 750, "training_validation"), ("final", 1000, "official_test")):
            targets = self.read("targets.json", phase=phase)
            self.assertEqual(targets["evaluation_scope"], scope)
            self.assertEqual(targets["source_transfer_count"], count)
            self.assertEqual(targets["training_target_count"], 1)
            self.assertEqual(targets["legacy_target_count"], 1)
            self.assertFalse(targets["official_test_used_for_selection"])
            self.assertNotIn("legacy_validation_indices", targets)
            identity = self.read("task.json", phase=phase)
            self.assertEqual(identity["data_identity"]["arrays"]["y_test"]["shape"], [count])

    def test_changed_source_fingerprint_or_dataset_cannot_overwrite_task(self):
        with runtime.c7_station_runtime(self.root):
            self.run_task()
            before = self.path("task.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "identity.*changed"):
                fair_tuning.run_measured_experiment(self.data, self.config,
                    task_id="validation_fedavg_fixed", run_fingerprint="different")
            with self.assertRaisesRegex(ValueError, "identity.*changed"):
                self.run_task(data=dataset(21))
        self.assertEqual(before, self.path("task.json").read_bytes())

    def test_first_numerical_attack_failure_has_round_client_and_no_checkpoint_advance(self):
        evidence = {"status": "nonfinite", "first_failure": {"stage": "target.gradients", "block": 1}}
        with mock.patch.object(attack, "run_profile_attack", side_effect=attack.NumericalFailure(evidence)):
            with runtime.c7_station_runtime(self.root):
                with self.assertRaises(attack.NumericalFailure):
                    self.run_task()
        status = self.read("status.json")
        self.assertEqual((status["status"], status["completed_round"]), ("numerical_failure", 0))
        error = self.read("first_failure.json")
        self.assertEqual((error["audit"]["round"], error["audit"]["client_idx"]), (1, 0))
        self.assertEqual(error["audit"]["first_failure"]["stage"], "target.gradients")
        self.assertEqual(self.read("first_numerical_failure.json"), error)

    def test_honest_nonfinite_failure_is_observed_without_replacing_original_function(self):
        with mock.patch.object(fl, "_local_train_client_delta", return_value=(np.array([np.inf]), SimpleNamespace(loss=.1))) as original:
            with runtime.c7_station_runtime(self.root):
                with self.assertRaisesRegex(RuntimeError, "honest_delta"):
                    self.run_task()
            self.assertIs(fl._local_train_client_delta, original)
        self.assertEqual(self.read("first_failure.json")["audit"]["client_idx"], 1)

    def test_hooks_restored_on_failure_and_nested_install_rejected(self):
        originals = (fair_tuning.run_measured_experiment, experiments.run_experiment,
                     fl._maybe_torch_context, fl._alternating_minimization_client_delta,
                     torch_backend.TorchTrainingContext.alternating_minimization_delta_resident)
        with self.assertRaisesRegex(RuntimeError, "outer failed"):
            with runtime.c7_station_runtime(self.root):
                with self.assertRaisesRegex(RuntimeError, "already installed"):
                    with runtime.c7_station_runtime(self.root):
                        pass
                raise RuntimeError("outer failed")
        self.assertEqual(originals, (fair_tuning.run_measured_experiment, experiments.run_experiment,
                     fl._maybe_torch_context, fl._alternating_minimization_client_delta,
                     torch_backend.TorchTrainingContext.alternating_minimization_delta_resident))

    def test_profile_and_phase_fingerprint_are_required_before_scheduling(self):
        with self.assertRaisesRegex(ValueError, "frozen C7"):
            with runtime.c7_station_runtime(self.root, dict(runtime.PROFILE, boost=10.)):
                pass
        with mock.patch.object(fair_tuning, "execute_resumable_tuning_phase", return_value="scheduled") as original:
            with runtime.c7_station_runtime(self.root):
                with self.assertRaisesRegex(ValueError, "fingerprint_context"):
                    fair_tuning.execute_resumable_tuning_phase(None, [], None)
                context = runtime.runtime_fingerprint_context(runtime.PROFILE)
                self.assertEqual(fair_tuning.execute_resumable_tuning_phase(None, [], None,
                    fingerprint_context=context, jobs=8, backend_description="torch:cuda"), "scheduled")
                with self.assertRaisesRegex(ValueError, "CUDA thread"):
                    fair_tuning.execute_resumable_tuning_phase(None, [], None,
                        fingerprint_context=context, jobs=2, backend_description="torch:cpu")
            self.assertEqual(original.call_count, 1)

    def test_task_key_changes_only_for_scientific_parameters(self):
        key = runtime.task_key(self.config)
        self.assertEqual(key, runtime.task_key(replace(self.config, device="cuda:7", sm9_workers=8)))
        self.assertNotEqual(key, runtime.task_key(replace(self.config, lr=.001)))
        self.assertNotEqual(key, runtime.task_key(replace(self.config, seed=999)))

    def test_dashboard_can_nest_on_either_side_without_losing_rounds_or_hooks(self):
        from scripts.experiments.cifar10 import fair_tuning_dashboard as dashboard
        spec = fair_tuning.load_fair_tuning_config(
            Path(__file__).resolve().parents[1] / "configs/cifar10_v14_fair_grid.json")
        for runtime_first in (True, False):
            config = replace(self.config, seed=301 if runtime_first else 302)
            item = fair_tuning.TuningExperimentTask("validation", "fixed", config.method, config)
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(redirect_stderr(io.StringIO()))
                if runtime_first:
                    stack.enter_context(runtime.c7_station_runtime(self.root))
                    progress = stack.enter_context(dashboard.dashboard(spec, mode="log"))
                else:
                    progress = stack.enter_context(dashboard.dashboard(spec, mode="log"))
                    stack.enter_context(runtime.c7_station_runtime(self.root))
                progress.register([item])
                self.run_task(config)
                key = progress.key("validation", config)
                self.assertEqual(progress.tasks[key].round, 2)
            self.assertEqual(self.read("status.json", config=config)["status"], "complete")

    def test_forked_worker_is_rejected_before_training(self):
        with runtime.c7_station_runtime(self.root):
            with mock.patch.object(runtime.os, "getpid", return_value=-1):
                with self.assertRaisesRegex(RuntimeError, "forked"):
                    self.run_task()


class StationRuntimeTorchIntegration(unittest.TestCase):
    def test_real_cpu_training_metrics_and_terminal_checkpoint_replay(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch unavailable")
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        rng = np.random.default_rng(7)
        data = ImageDataset(
            x_train=rng.normal(size=(20, 1, 8, 8)).astype(np.float32),
            y_train=np.arange(20) % 10,
            x_test=rng.normal(size=(10, 1, 8, 8)).astype(np.float32), y_test=np.arange(10),
            x_attack=rng.normal(size=(2, 1, 8, 8)).astype(np.float32), y_attack=np.full(2, 5),
            name="mnist", num_classes=10)
        config = fl.ExperimentConfig(
            method="fedavg", partition="iid", rounds=4, num_clients=4, malicious_ratio=.5,
            attack="alternating_minimization", attack_start_round=4, detector_window=3,
            attack_source_label=5, attack_target_label=7, attack_target_count=1,
            attack_epochs=1, local_epochs=1, batch_size=5, lr=.01, lr_decay=.99,
            early_stop=False, eval_interval=1, compute_backend="torch", device="cpu", seed=301)
        with tempfile.TemporaryDirectory() as output:
            kwargs = dict(checkpoint_dir=Path(output) / "native_checkpoints",
                          run_fingerprint="cpu-integration", retain_success_checkpoint=True,
                          task_id="validation_fedavg_fixed")
            with runtime.c7_station_runtime(output):
                first = fair_tuning.run_measured_experiment(data, config, **kwargs)
                replayed = fair_tuning.run_measured_experiment(data, config, **kwargs)
            self.assertEqual(first, replayed)
            directory = runtime.task_directory(output, "validation", config)
            metrics = json.loads((directory / "metrics.json").read_text())["rounds"]
            self.assertEqual([r["round"] for r in metrics], [0, 1, 2, 3, 4])
            self.assertEqual([r["attack_calls_this_round"] for r in metrics], [0, 0, 0, 0, 2])
            self.assertEqual(metrics[-1]["source_transfer_count"], 1)
            self.assertTrue(json.loads((directory / "result.json").read_text())["terminal_checkpoint_replayed"])
            audits = [json.loads(line) for line in (directory / "attack_audit.jsonl").read_text().splitlines()]
            self.assertEqual({row["optimizer"] for row in audits}, {"adam"})
            self.assertEqual({row["lr"] for row in audits}, {.001})
            self.assertEqual({row["scope"] for row in audits}, {"weights"})


if __name__ == "__main__":
    unittest.main()
