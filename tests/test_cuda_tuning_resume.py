import hashlib
import io
import json
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import pickle
import threading
import time
import unittest
from unittest import mock

from sm9rrsfl import cuda_execution as cuda
from sm9rrsfl import experiments as exp
from sm9rrsfl import fair_tuning as tuning
from sm9rrsfl.tuning_resume import locate_tuning_state, phase_identity, runtime_config_key
from tests.test_fair_tuning import _result
from tests.test_tuning_comparison_failures import fixture_dataset


def task(index=0, device="cuda:2", phase="validation", **fields):
    config = exp.ExperimentConfig(method="fedavg", malicious_ratio=.4, seed=index,
        num_clients=10, rounds=3, attack_start_round=1, compute_backend="torch",
        device=device, crypto_mode="simulated", **fields)
    return tuning.TuningExperimentTask(phase, "fedavg-001", "fedavg", config)


def result(config):
    return _result(config.malicious_ratio, .8, 0, "fedavg", config=config,
                   accepted_updates=[10] * config.rounds)


def legacy_state(root, dataset, tasks, args, context=None, results=()):
    # Frozen legacy layout: runtime CUDA index appears in BOTH configs and
    # candidates and is included in the phase directory's hash.
    manifest = exp.build_run_manifest(args, dataset, [t.config for t in tasks])
    manifest.update(tuning_phase=tasks[0].phase, tuning_context=context or {},
        candidates=[dict(candidate_id=t.candidate_id, method=t.method, config=asdict(t.config)) for t in tasks])
    payload = dict(manifest)
    payload.pop("fingerprint")
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest["fingerprint"] = fingerprint
    folder = root / ".tuning_state" / tasks[0].phase / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    exp.write_run_manifest(folder, manifest)
    if results:
        exp.write_result_files(folder, list(results))
    return folder, manifest


class CudaCacheIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = fixture_dataset("mnist")
        self.args = exp.parse_args(["--no-progress"])

    def test_only_cuda_placement_and_crypto_worker_count_are_ignored(self):
        original = task().config
        moved = replace(original, device="cuda:0", sm9_workers=8)
        self.assertEqual(runtime_config_key(original), runtime_config_key(moved))
        for changes in (dict(device="cpu"), dict(device="mps"), dict(seed=19),
                        dict(attack_boost=3), dict(batch_size=16), dict(rounds=100),
                        dict(crypto_mode="sm9"), dict(detector_window=20),
                        dict(compute_backend="numpy"), dict(checkpoint_interval=2)):
            with self.subTest(changes=changes):
                self.assertNotEqual(runtime_config_key(original), runtime_config_key(replace(original, **changes)))

    def test_old_completed_results_keep_provenance_when_gpu_and_workers_change(self):
        old = [task(1), task(2, device="cuda:7")]
        folder, manifest = legacy_state(self.root, self.data, old, self.args,
            results=[result(t.config) for t in old])
        before = (folder / "run_manifest.json").read_bytes()
        new = [replace(t, config=replace(t.config, device="cuda:0", sm9_workers=8)) for t in old]
        with mock.patch.object(tuning, "run_measured_experiment", side_effect=AssertionError("must reuse")):
            restored, fingerprint = tuning.execute_resumable_tuning_phase(self.data, new, self.args,
                output_dir=self.root, jobs=1, backend_description="torch:cuda",
                progress_enabled=False, progress_mode="log")
        self.assertEqual(fingerprint, manifest["fingerprint"])
        self.assertEqual({t.config.device for t, _ in restored}, {"cuda:0"})
        self.assertEqual({r.config.device for _, r in restored}, {"cuda:2", "cuda:7"})
        self.assertEqual((folder / "run_manifest.json").read_bytes(), before)

    def test_partial_round_checkpoint_survives_reassignment_and_terminal_replay(self):
        old = task()
        folder, manifest = legacy_state(self.root, self.data, [old], self.args)
        checkpoint = exp._checkpoint_path(folder / ".checkpoints", old.config)
        exp._write_round_checkpoint(checkpoint, old.config, manifest["fingerprint"],
            {"completed_round": 2, "sentinel": "old-GPU-state"}, runtime_seconds=5., peak_memory_mb=1.)
        new = replace(old, config=replace(old.config, device="cuda:0", sm9_workers=8))
        received = []

        def train(dataset, config, *, resume_state, checkpoint_callback):
            received.append((config, resume_state))
            checkpoint_callback({"completed_round": 3})
            return result(config)

        with mock.patch.object(exp, "run_experiment", side_effect=train), \
             mock.patch.object(tuning, "run_cuda_tasks", side_effect=lambda tasks, **kw:
                               [kw["commit"](t, kw["run"](t)) for t in tasks]):
            restored, fingerprint = tuning.execute_resumable_tuning_phase(self.data, [new], self.args,
                output_dir=self.root, jobs=1, backend_description="torch:cuda",
                progress_enabled=False, progress_mode="log")
        self.assertEqual(received[0][0].device, "cuda:0")
        self.assertEqual(received[0][1], {"completed_round": 2, "sentinel": "old-GPU-state"})
        self.assertEqual(fingerprint, manifest["fingerprint"])
        self.assertFalse(checkpoint.exists())
        self.assertGreaterEqual(restored[0][1].runtime_seconds, 5.)
        # A worker finished before its parent could commit: terminal state must
        # replay its historical result without executing any training code.
        exp._write_round_checkpoint(checkpoint, old.config, fingerprint,
            {"completed_round": 3, "terminal_result": result(old.config)},
            runtime_seconds=5., peak_memory_mb=1.)
        with mock.patch.object(exp, "run_experiment", side_effect=AssertionError("terminal replay")):
            replayed = exp.run_measured_experiment(self.data, new.config,
                checkpoint_dir=folder / ".checkpoints", run_fingerprint=fingerprint,
                checkpoint_identity_config=old.config)
        self.assertEqual(replayed.config.device, old.config.device)
        self.assertFalse(checkpoint.exists())

    def test_final_parent_validation_alias_is_checked_without_requiring_same_gpu_hash(self):
        old_validation = [task()]
        _, first = legacy_state(self.root, self.data, old_validation, self.args)
        _, second = legacy_state(self.root, self.data, [task(device="cuda:0")], self.args)
        old_final = [replace(old_validation[0], phase="final")]
        _, final1 = legacy_state(self.root, self.data, old_final, self.args,
            context={"selected_validation_fingerprint": first["fingerprint"]})
        _, final2 = legacy_state(self.root, self.data, [replace(t, config=replace(t.config, device="cuda:0"))
            for t in old_final], self.args, context={"selected_validation_fingerprint": second["fingerprint"]})
        root = self.root / ".tuning_state"
        self.assertEqual(phase_identity(final1, root), phase_identity(final2, root))
        _, different = legacy_state(self.root, self.data, old_validation, self.args, context={"split_seed": 999})
        altered = json.loads(json.dumps(final2))
        altered["tuning_context"]["selected_validation_fingerprint"] = different["fingerprint"]
        self.assertNotEqual(phase_identity(final1, root), phase_identity(altered, root))

    def test_changed_data_split_attack_or_candidate_matrix_cannot_use_cache(self):
        old = task()
        _, manifest = legacy_state(self.root, self.data, [old], self.args, results=[result(old.config)])
        for changes in ("data", "split", "attack", "candidates"):
            with self.subTest(changes=changes):
                changed = json.loads(json.dumps(manifest))
                if changes == "data":
                    changed["dataset"]["train_content_digest"] = "different"
                elif changes == "split":
                    changed["tuning_context"]["split_seed"] = 19
                elif changes == "attack":
                    changed["configs"][0]["attack_boost"] = 999
                    changed["candidates"][0]["config"]["attack_boost"] = 999
                else:
                    changed["candidates"][0]["candidate_id"] = "different-candidate"
                changed["fingerprint"] = "f" * 64
                _, _, recovered = locate_tuning_state(self.root, changed)
                self.assertEqual(recovered, [])

    def test_richer_legacy_cache_wins_over_empty_current_gpu_cache(self):
        old = [task(1), task(2)]
        folder, _ = legacy_state(self.root, self.data, old, self.args, results=[result(t.config) for t in old])
        _, current = legacy_state(self.root, self.data,
            [replace(t, config=replace(t.config, device="cuda:0")) for t in old], self.args)
        found, _, recovered = locate_tuning_state(self.root, current)
        self.assertEqual(found, folder)
        self.assertEqual(len(recovered), 2)

    def test_restored_vert_and_tad_rebind_device_without_resetting_history(self):
        from sm9rrsfl.datasets import make_synthetic_mnist_like
        from sm9rrsfl.vert import VERTDefense
        from sm9rrsfl.ding13_detector import Ding13TrajectoryDetector
        data = make_synthetic_mnist_like(train_samples=30, test_samples=10, seed=7)
        for method, field, cls in (("vert", "vert_defense", VERTDefense),
                                   ("ding13", "ding13_detector", Ding13TrajectoryDetector)):
            with self.subTest(method=method):
                config = exp.ExperimentConfig(method=method, num_clients=3, rounds=3,
                    malicious_ratio=0., attack="none", early_stop=False, compute_backend="numpy",
                    device="cpu", crypto_mode="simulated", vert_predict_epochs=1)
                saved = []
                def save(state):
                    if state["completed_round"] == 1:
                        saved.append(pickle.loads(pickle.dumps(state)))
                        raise RuntimeError("interrupt fixture")
                with self.assertRaisesRegex(RuntimeError, "interrupt fixture"):
                    exp.run_experiment(data, config, checkpoint_callback=save)
                state = saved[0]
                defense = state[field]
                defense.device = "cuda:7"
                if method == "vert":
                    before = [x.copy() for x in defense._global_history]
                else:
                    before = {k: x.copy() for k, x in defense.previous_singulars.items()}
                original = cls.evaluate_round
                devices = []
                def evaluate(current, *args, **kwargs):
                    devices.append(current.device)
                    if len(devices) == 1:
                        import numpy as np
                        if method == "vert":
                            for old, actual in zip(before, current._global_history):
                                np.testing.assert_array_equal(old, actual)
                        else:
                            for key, value in before.items():
                                np.testing.assert_array_equal(value, current.previous_singulars[key])
                    return original(current, *args, **kwargs)
                with mock.patch.object(cls, "evaluate_round", new=evaluate):
                    resumed = exp.run_experiment(data, config, resume_state=state)
                self.assertEqual(set(devices), {"cpu"})
                self.assertEqual(resumed.stopped_round, 3)


class CudaSchedulerTest(unittest.TestCase):
    def run_queue(self, tasks, run, **options):
        commits, events = [], []
        cuda.run_cuda_tasks(tasks, devices=("cuda:0", "cuda:1"), jobs=2, required_mb=100., run=run,
            commit=lambda t, r: commits.append((t, r)),
            event=lambda kind, task, **details: events.append((kind, task, details)),
            poll_seconds=.001, max_idle_seconds=.1, **options)
        return commits, events

    def test_oom_moves_to_other_card_and_preserves_checkpoint_identity(self):
        failed = threading.Event()
        active = set()
        lock = threading.Lock()
        seen = []

        def run(t):
            with lock:
                self.assertNotIn(t.config.device, active)
                active.add(t.config.device)
                seen.append(t)
            try:
                if t.config.device == "cuda:0":
                    failed.set()
                    raise RuntimeError("CUDA out of memory. fixture")
                time.sleep(.01)
                return t.config.seed
            finally:
                with lock:
                    active.remove(t.config.device)

        with mock.patch.object(cuda, "cuda_devices_with_capacity", side_effect=lambda _:
                               ("cuda:1",) if failed.is_set() else ("cuda:0", "cuda:1")), \
             mock.patch.object(cuda, "release_cuda_cache"):
            commits, events = self.run_queue([task(0, "cuda:0"), task(1, "cuda:1")], run)
        self.assertEqual(len(commits), 2)
        recovered = next(t for t, _ in commits if t.config.seed == 0)
        self.assertEqual(recovered.config.device, "cuda:1")
        self.assertEqual(recovered.checkpoint_config.device, "cuda:0")
        self.assertEqual(recovered.config.batch_size, 32)
        self.assertEqual(len(seen), 3)
        self.assertIn("cuda_oom", [e[0] for e in events])

    def test_fatal_oom_drains_and_commits_other_inflight_workers(self):
        committed = []
        def run(t):
            if t.config.seed == 0:
                raise RuntimeError("CUDA out of memory")
            time.sleep(.02)
            return t.config.seed
        with mock.patch.object(cuda, "cuda_devices_with_capacity", return_value=("cuda:0", "cuda:1")), \
             mock.patch.object(cuda, "release_cuda_cache"), \
             self.assertRaises(cuda.CudaResourcesUnavailable):
            cuda.run_cuda_tasks([task(0, "cuda:0"), task(1, "cuda:1")],
                devices=("cuda:0", "cuda:1"), jobs=2, required_mb=100., run=run,
                commit=lambda t, r: committed.append(r), event=lambda *a, **kw: None,
                max_retries=0, poll_seconds=.001)
        self.assertEqual(committed, [1])

    def test_non_oom_error_is_not_retried_or_hidden(self):
        run = mock.Mock(side_effect=ValueError("invalid detector configuration"))
        with mock.patch.object(cuda, "cuda_devices_with_capacity", return_value=("cuda:0",)), \
             self.assertRaisesRegex(ValueError, "invalid detector"):
            self.run_queue([task(0, "cuda:0")], run)
        self.assertEqual(run.call_count, 1)

    def test_retry_is_bounded(self):
        run = mock.Mock(side_effect=RuntimeError("CUDA out of memory"))
        with mock.patch.object(cuda, "cuda_devices_with_capacity", return_value=("cuda:0",)), \
             mock.patch.object(cuda, "release_cuda_cache"), \
             self.assertRaisesRegex(cuda.CudaResourcesUnavailable, "retry limit"):
            self.run_queue([task(0, "cuda:0")], run, cooldown_seconds=0.)
        self.assertEqual(run.call_count, 2)

    def test_unavailable_or_unallowed_devices_do_not_start_training(self):
        for ready in ((), ("cuda:7",)):
            with self.subTest(ready=ready), \
                 mock.patch.object(cuda, "cuda_devices_with_capacity", return_value=ready):
                run = mock.Mock()
                with self.assertRaises(cuda.CudaResourcesUnavailable):
                    cuda.run_cuda_tasks([task(0, "cuda:0")], devices=("cuda:0",), jobs=1,
                        required_mb=100., run=run, commit=lambda *a: None,
                        event=lambda *a, **kw: None, max_idle_seconds=0.)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
