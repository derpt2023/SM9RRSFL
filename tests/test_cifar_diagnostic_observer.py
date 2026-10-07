"""Small real CPU runs: observation is passive and survives durable resume."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import run_cifar_diagnostic as runner
from cifar_diagnostic_observer import observe
from sm9rrsfl import model, torch_backend as backend
from sm9rrsfl.datasets import ImageDataset

base = runner.base


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)
        rng = np.random.default_rng(71)
        self.dataset = ImageDataset(rng.normal(size=(4, 3, 32, 32)).astype(np.float32),
            np.array([0, 1, 0, 1]), rng.normal(size=(4, 3, 32, 32)).astype(np.float32),
            np.array([0, 1, 0, 1]), name="cifar10",
            x_attack=rng.normal(size=(2, 3, 32, 32)).astype(np.float32), y_attack=np.array([0, 1]))
        self.config = base.fl.ExperimentConfig(method="fedavg", malicious_ratio=0.,
            num_clients=2, rounds=2, lr=.005, local_epochs=1, batch_size=2,
            compute_backend="torch", device="cpu", crypto_mode="simulated", early_stop=False,
            checkpoint_interval=1, eval_interval=1, attack="alternating_minimization",
            attack_source_label=0, attack_target_label=1, attack_target_count=1,
            detector_window=3, attack_start_round=4, seed=71)
        self.config.validate()
        self.task = {"fingerprint": "small-cpu-observer-test", "task_id": "cpu",
                     "phase": "validation", "candidate": {"candidate_id": "test"},
                     "config": asdict(self.config)}

    def run_actual(self, callback, resume_state=None):
        with redirect_stdout(io.StringIO()):
            return base.experiments.run_experiment(self.dataset, self.config,
                checkpoint_callback=callback, resume_state=resume_state)

    def test_loss_capture_leaves_both_models_and_rng_identical(self):
        for name, size in (("v7_cnn", 1756426), ("resnet18_gn2", 11173962)):
            with self.subTest(model=name), runner.model_runtime(name):
                spec = model.model_spec_for_dataset(self.dataset)
                self.assertEqual(spec.parameter_size, size)
                plain, observed = {}, {}
                torch.manual_seed(313)
                initial_rng = torch.get_rng_state().clone()
                self.run_actual(lambda state: plain.update(params=state["params"].copy()))
                plain_rng = torch.get_rng_state().clone()
                torch.set_rng_state(initial_rng)
                with observe(self.task, self.folder):
                    self.run_actual(lambda state: observed.update(params=state["params"].copy(),
                        diagnostics=deepcopy(state["clean_diagnostic_observations"])))
                np.testing.assert_array_equal(plain["params"], observed["params"])
                self.assertTrue(torch.equal(plain_rng, torch.get_rng_state()))
                rows = observed["diagnostics"]["rounds"]
                self.assertEqual([r["round"] for r in rows], [0, 1, 2])
                self.assertIsNone(rows[0]["local_train_loss"])
                self.assertEqual(rows[2]["local_train_samples"], 4)
                self.assertEqual(rows[2]["calibration_samples"], 4)
                params = backend._torch_params_from_tensor(torch, torch.from_numpy(observed["params"]),
                    spec, requires_grad=False, clone=False)
                with torch.no_grad():
                    expected = torch.nn.functional.cross_entropy(backend._torch_forward(torch, params,
                        torch.from_numpy(self.dataset.x_test), spec), torch.from_numpy(self.dataset.y_test)).item()
                self.assertAlmostEqual(rows[-1]["calibration_loss"], expected, places=6)
        self.assertFalse(runner.resnet.runtime_installed())

    def test_actual_checkpoint_resume_has_identical_parameters_and_loss_rows(self):
        states, final = {}, {}
        with observe(self.task, self.folder):
            self.run_actual(lambda s: final.update(params=s["params"].copy(),
                            rows=deepcopy(s["clean_diagnostic_observations"]["rounds"])))
        checkpoint = self.folder / "round.pickle"

        def stop(state):
            base.experiments._write_round_checkpoint(checkpoint, self.config, self.task["fingerprint"],
                                                    state, runtime_seconds=1., peak_memory_mb=1.)
            if state["completed_round"] == 1:
                raise runner.runtime.BudgetPause("test interruption")

        with observe(self.task, self.folder):
            with self.assertRaises(runner.runtime.BudgetPause):
                self.run_actual(stop)
        state, _, _ = base.experiments._load_round_checkpoint(checkpoint, self.config, self.task["fingerprint"])
        self.assertEqual(state["completed_round"], 1)
        self.assertEqual(len(state["clean_diagnostic_observations"]["rounds"]), 2)
        with observe(self.task, self.folder):
            self.run_actual(lambda s: states.update(params=s["params"].copy()), resume_state=state)
        np.testing.assert_array_equal(final["params"], states["params"])
        rows = json.loads((self.folder / "observations.json").read_text())["rounds"]
        for before, after in zip(final["rows"], rows):
            self.assertEqual(before["local_train_loss"], after["local_train_loss"])
            self.assertEqual(before["calibration_loss"], after["calibration_loss"])

    def test_patches_restore_after_failure_and_bad_resume_rejected(self):
        local, accuracy, forward, run = (base.fl._local_train_client_delta, base.fl._evaluate_accuracy,
                                        backend._torch_forward, base.experiments.run_experiment)
        with observe(self.task, self.folder):
            with self.assertRaisesRegex(ValueError, "missing"):
                self.run_actual(lambda _: None, resume_state={"completed_round": 1})
        self.assertIs(base.fl._local_train_client_delta, local)
        self.assertIs(base.fl._evaluate_accuracy, accuracy)
        self.assertIs(backend._torch_forward, forward)
        self.assertIs(base.experiments.run_experiment, run)

    def test_round_monitor_interrupt_occurs_after_loss_checkpoint(self):
        import signal
        saved = {}

        def callback(state):
            saved.update(deepcopy(state))
            if state["completed_round"] == 1:
                signal.raise_signal(signal.SIGTERM)

        with runner.runtime.round_monitor(self.task, self.folder), observe(self.task, self.folder):
            with self.assertRaises(runner.runtime.BudgetPause):
                self.run_actual(callback)
        self.assertEqual(saved["completed_round"], 1)
        self.assertEqual(saved["clean_diagnostic_observations"]["rounds"][-1]["round"], 1)


if __name__ == "__main__":
    unittest.main()
