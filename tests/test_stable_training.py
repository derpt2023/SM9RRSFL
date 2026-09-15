from dataclasses import replace
import json
import unittest

import numpy as np
import torch

from sm9rrsfl.datasets import ImageDataset, make_synthetic_mnist_like
from sm9rrsfl.model import init_params, model_spec_for_dataset
from sm9rrsfl import torch_backend as backend
from sm9rrsfl.stable_training import (
    NumericalTrainingError, StableTrainingPolicy, stable_client_training,
)


class FiniteStepTest(unittest.TestCase):
    def step(self, policy, parameter, objective, lr, extra_checks=()):
        def update(gradients, step):
            parameter.add_(gradients[0], alpha=-step)
        return policy.step(torch, [parameter], objective, update, learning_rate=lr,
                           phase="honest", client_idx=7, seed=1409, batch=2,
                           extra_checks=extra_checks)

    def test_finite_parameters_with_overflowing_next_loss_trigger_backtracking(self):
        parameter = torch.tensor([1e18], requires_grad=True)
        events = []
        policy = StableTrainingPolicy(event_callback=events.append)
        loss = self.step(policy, parameter, lambda: parameter.square().sum(), 10.)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(parameter.square()).all())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "numerical_training_backtrack")
        self.assertEqual(events[0]["original_lr"], 10.)
        self.assertEqual(events[0]["actual_lr"], 5.)
        self.assertEqual(events[0]["retries"], 1)
        self.assertIsNone(parameter.grad)
        counts = policy.summary()["totals"]
        self.assertEqual(counts["backtracked_steps"], 1)
        self.assertEqual(counts["accepted_steps"], 1)
        self.assertEqual(counts["actual_lr_min"], 5.)
        json.dumps(events, allow_nan=False)

    def test_exhaustion_restores_parameters_and_never_substitutes_zero(self):
        parameter = torch.tensor([1e18], requires_grad=True)
        before = parameter.detach().clone()
        policy = StableTrainingPolicy(max_backtracks=1)
        with self.assertRaisesRegex(NumericalTrainingError, "client=7.*seed=1409.*batch=2.*exhausted"):
            self.step(policy, parameter, lambda: parameter.square().sum(), 1000.)
        self.assertTrue(torch.equal(parameter.detach(), before))
        self.assertIsNone(parameter.grad)
        counts = policy.summary()["totals"]
        self.assertEqual(counts["failures"], 1)
        self.assertEqual(counts["retries"], 1)
        self.assertEqual(counts["backtracked_steps"], 1)
        self.assertEqual(counts["accepted_steps"], 0)

    def test_nonfinite_input_loss_and_gradient_fail_without_backtracking(self):
        for value, objective, reason in (
            (1e20, lambda p: p.square().sum(), "input_loss"),
            (0., lambda p: p.sqrt().sum(), "input_gradient"),
            (float("inf"), lambda p: p.sum(), "input_parameters"),
        ):
            with self.subTest(reason=reason):
                parameter = torch.tensor([value], requires_grad=True)
                events = []
                policy = StableTrainingPolicy(event_callback=events.append)
                with self.assertRaisesRegex(NumericalTrainingError, reason):
                    self.step(policy, parameter, lambda: objective(parameter), .1)
                self.assertEqual(policy.summary()["totals"]["retries"], 0)
                self.assertIsNone(parameter.grad)
                self.assertEqual(events[0]["event"], "numerical_training_failure")
                json.dumps(events, allow_nan=False)

    def test_finite_candidate_loss_but_nonfinite_gradient_triggers_retry(self):
        parameter = torch.tensor([1.], requires_grad=True)
        policy = StableTrainingPolicy()
        self.step(policy, parameter, lambda: parameter.sqrt().sum(), 2.)
        # Step 2 produces p=0: sqrt is finite, its derivative is infinite.
        # Retrying from p=1 at step 1 must produce p=.5.
        self.assertTrue(torch.equal(parameter.detach(), torch.tensor([.5])))
        self.assertEqual(policy.summary()["totals"]["retries"], 1)

    def test_extra_objective_prevents_finite_primary_but_broken_secondary(self):
        parameter = torch.tensor([1.], requires_grad=True)
        policy = StableTrainingPolicy()
        self.step(policy, parameter, lambda: parameter.sum(), 1.,
                  extra_checks=((lambda: parameter.sqrt().sum(), None),))
        self.assertTrue(torch.equal(parameter.detach(), torch.tensor([.5])))
        self.assertEqual(policy.summary()["totals"]["retries"], 1)
        self.assertIsNone(parameter.grad)

    def test_finite_operand_subtraction_overflow_is_an_explicit_client_failure(self):
        maximum = torch.finfo(torch.float32).max
        local = torch.tensor([maximum], dtype=torch.float32)
        global_vector = torch.tensor([-maximum], dtype=torch.float32)
        self.assertTrue(torch.isfinite(local).all() and torch.isfinite(global_vector).all())
        delta = local.sub(global_vector)
        events = []
        policy = StableTrainingPolicy(event_callback=events.append)
        for phase in ("honest", "benign_reference", "attack_output"):
            with self.subTest(phase=phase):
                with self.assertRaisesRegex(NumericalTrainingError, "nonfinite_client_delta"):
                    policy.validate_client_delta(torch, delta, phase=phase, client_idx=3,
                        seed=215, batch=4, learning_rate=.05)
                self.assertTrue(torch.isinf(delta).all())
                self.assertEqual(policy.summary()["by_phase"][phase]["failures"], 1)
        self.assertEqual(events[0]["delta"]["finite_elements"], 0)
        json.dumps(events, allow_nan=False)
        valid_delta = torch.tensor([1., -2.])
        policy.validate_client_delta(torch, valid_delta, phase="honest", client_idx=3,
            seed=215, batch=4, learning_rate=.05)
        self.assertTrue(torch.equal(valid_delta, torch.tensor([1., -2.])))
        self.assertEqual(len(events), 3)


class StableClientTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        data = make_synthetic_mnist_like(train_samples=8, test_samples=4, seed=807)
        cls.data = replace(data, x_attack=data.x_train[:2].copy(),
                           y_attack=data.y_train[:2].copy())
        cls.spec = model_spec_for_dataset(cls.data)
        cls.params = init_params(seed=190, spec=cls.spec)
        cls.indices = [np.arange(8)]

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def make_context(self):
        return backend.TorchTrainingContext(self.data, self.indices, spec=self.spec, device="cpu")

    def test_no_backtracks_preserve_exact_honest_delta_and_loss(self):
        kwargs = dict(client_idx=0, lr=.01, epochs=2, batch_size=3, seed=515)
        original = self.make_context().local_train_delta_resident(self.params, **kwargs)
        events = []
        with stable_client_training(event_callback=events.append) as policy:
            protected = self.make_context().local_train_delta_resident(self.params, **kwargs)
        self.assertTrue(torch.equal(original[0], protected[0]))
        self.assertEqual(original[1], protected[1])
        self.assertEqual(events, [])
        self.assertEqual(policy.summary()["by_phase"]["honest"]["total_steps"], 6)
        self.assertEqual(policy.summary()["totals"]["backtracked_steps"], 0)

    def test_no_backtracks_preserve_exact_attack_delta_and_loss_all_phases(self):
        kwargs = dict(client_idx=0, target_indices=np.array([0, 1]), target_label=7,
                      lr=.01, attack_epochs=1, batch_size=3, stealth_steps=2,
                      boost=5., distance_weight=.0001, seed=515)
        original = self.make_context().alternating_minimization_delta_resident(self.params, **kwargs)
        with stable_client_training() as policy:
            protected = self.make_context().alternating_minimization_delta_resident(self.params, **kwargs)
        self.assertTrue(torch.equal(original[0], protected[0]))
        self.assertEqual(original[1], protected[1])
        phases = policy.summary()["by_phase"]
        self.assertEqual(phases["benign_reference"]["total_steps"], 3)
        self.assertEqual(phases["attack_stealth"]["total_steps"], 3)
        self.assertEqual(phases["attack_target"]["total_steps"], 2)
        self.assertEqual(phases["attack_target"]["original_lr_min"], .05)
        self.assertEqual(policy.summary()["totals"]["backtracked_steps"], 0)

    def test_context_restores_original_class_after_failure_and_rejects_overlap(self):
        original = backend.TorchTrainingContext
        existing = self.make_context()
        with self.assertRaisesRegex(ValueError, "deliberate"):
            with stable_client_training():
                self.assertIsNot(backend.TorchTrainingContext, original)
                self.assertIs(type(existing), original)
                with self.assertRaisesRegex(RuntimeError, "one context"):
                    with stable_client_training():
                        pass
                raise ValueError("deliberate")
        self.assertIs(backend.TorchTrainingContext, original)
        with stable_client_training():
            pass
        self.assertIs(backend.TorchTrainingContext, original)

    def test_full_cifar_model_preserves_both_ordinary_paths_without_backtracks(self):
        rng = np.random.default_rng(715)
        features = rng.normal(0, .2, size=(4, 3, 32, 32)).astype(np.float32)
        labels = np.arange(4, dtype=np.int64)
        data = ImageDataset(features, labels, features[:1], labels[:1], name="cifar10",
                            x_attack=features[:2], y_attack=labels[:2])
        spec = model_spec_for_dataset(data)
        params = init_params(seed=181, spec=spec)
        indices = [np.arange(4)]
        for attack in (False, True):
            with self.subTest(attack=attack):
                kwargs = dict(client_idx=0, lr=.05, batch_size=2, seed=455)
                if attack:
                    kwargs.update(target_indices=np.array([0, 1]), target_label=7,
                                  attack_epochs=1, stealth_steps=1, boost=5., distance_weight=.0001)
                    name = "alternating_minimization_delta_resident"
                else:
                    kwargs["epochs"] = 1
                    name = "local_train_delta_resident"
                context = backend.TorchTrainingContext(data, indices, spec=spec, device="cpu")
                original = getattr(context, name)(params, **kwargs)
                with stable_client_training() as policy:
                    context = backend.TorchTrainingContext(data, indices, spec=spec, device="cpu")
                    protected = getattr(context, name)(params, **kwargs)
                self.assertTrue(torch.equal(original[0], protected[0]))
                self.assertEqual(original[1], protected[1])
                self.assertEqual(policy.summary()["totals"]["backtracked_steps"], 0)
if __name__ == "__main__":
    unittest.main()
