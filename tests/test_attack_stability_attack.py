"""Behavioral checks for the independent attack adapter."""

import inspect
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from sm9rrsfl import torch_backend as backend
from scripts.experiments.cifar10 import attack_stability_attack as attack


@unittest.skipIf(torch is None, "PyTorch unavailable")
class AttackStabilityAttackTest(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.patches = [
            patch.object(backend, "_torch_params_from_tensor", self.parameters),
            patch.object(backend, "_torch_flat_vector_from_params",
                         lambda _torch, params: torch.cat([p.reshape(-1) for p in params])),
            patch.object(backend, "_torch_forward",
                         lambda _torch, params, x, _spec: x @ params[0] + params[1]),
        ]
        for item in self.patches:
            item.start()
        self.ctx = SimpleNamespace(
            torch=torch, device=torch.device("cpu"),
            spec=SimpleNamespace(num_classes=2, architecture="toy"),
            client_indices=[torch.arange(4)],
            x_train=torch.tensor([[.3, -.2], [.8, .1], [-.4, .5], [.2, .7]]),
            y_train=torch.tensor([0, 1, 0, 1]),
            x_attack=torch.tensor([[.6, -.3]]),
            _ensure_global_vector=lambda vector: torch.as_tensor(vector),
        )
        self.ctx.local_train_delta_resident = (
            backend.TorchTrainingContext.local_train_delta_resident.__get__(self.ctx)
        )
        self.vector = torch.tensor([.15, -.2, .25, .4, .05, -.02])
        self.kwargs = dict(
            client_idx=0, target_indices=np.array([0]), target_label=1,
            lr=.01, attack_epochs=2, batch_size=2, stealth_steps=1,
            boost=2., distance_weight=1e-4, seed=301,
        )
        self.profile = dict(
            id="legacy", optimizer="sgd", lr=None, boost=2.,
            stealth_steps=1, scope="all",
        )

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        torch.set_num_threads(self.old_threads)

    @staticmethod
    def parameters(_torch, vector, _spec, *, requires_grad, clone):
        parts = [vector[:4].reshape(2, 2), vector[4:]]
        return [p.detach().clone().requires_grad_(requires_grad)
                if clone else p.detach().requires_grad_(requires_grad) for p in parts]

    def run_attack(self, **profile_changes):
        return attack.run_profile_attack(
            self.ctx, self.vector, dict(self.profile, **profile_changes), **self.kwargs
        )

    def test_sgd_inherited_rate_matches_production_bitwise(self):
        original_delta, original_stats = (
            backend.TorchTrainingContext.alternating_minimization_delta_resident(
                self.ctx, self.vector, **self.kwargs
            )
        )
        delta, stats, audit = self.run_attack()
        self.assertTrue(torch.equal(delta, original_delta))
        self.assertEqual(stats, original_stats)
        self.assertEqual((audit["reference_steps"], audit["normal_steps"],
                          audit["target_steps"]), (4, 4, 4))
        self.assertEqual(audit["status"], "complete")
        json.dumps(audit, allow_nan=False)

    def test_sgd_multiple_stealth_batches_match_production(self):
        self.kwargs["stealth_steps"] = 3
        self.profile["stealth_steps"] = 3
        original = backend.TorchTrainingContext.alternating_minimization_delta_resident(
            self.ctx, self.vector, **self.kwargs
        )
        delta, stats, audit = self.run_attack()
        self.assertTrue(torch.equal(delta, original[0]))
        self.assertEqual(stats, original[1])
        self.assertEqual(audit["target_steps"], 2)

    def test_sgd_zero_float32_loss_retains_nonzero_gradient_like_production(self):
        self.ctx.x_train.zero_()
        self.ctx.x_attack.zero_()
        self.ctx.y_train.zero_()
        self.vector[4:] = torch.tensor([20., 0.])
        self.kwargs.update(
            target_label=0, attack_epochs=1, batch_size=4, distance_weight=0.
        )
        logits = torch.tensor([[20., 0.]], requires_grad=True)
        rounded_loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0]))
        self.assertEqual(rounded_loss.item(), 0.)
        rounded_loss.backward()
        self.assertGreater(logits.grad[0, 1].item(), 0.)
        expected, expected_stats = (
            backend.TorchTrainingContext.alternating_minimization_delta_resident(
                self.ctx, self.vector, **self.kwargs
            )
        )
        delta, stats, audit = self.run_attack()
        self.assertTrue(torch.equal(delta, expected))
        self.assertEqual(stats, expected_stats)
        self.assertLess(delta[-1].item(), -4e-11)
        self.assertEqual(audit["target_steps"], 1)
        self.assertEqual(audit["target_skipped_zero_loss"], 0)

    def test_adam_target_boosts_displacement_not_gradient(self):
        self.kwargs.update(attack_epochs=1, batch_size=4, distance_weight=0.)
        rate = .003
        params = self.parameters(torch, self.vector, self.ctx.spec,
                                 requires_grad=True, clone=True)
        stealth = torch.optim.Adam(params, lr=rate, foreach=False)
        target = torch.optim.Adam(params, lr=rate, foreach=False)
        loss = torch.nn.functional.cross_entropy(
            self.ctx.x_train @ params[0] + params[1], self.ctx.y_train
        )
        loss.backward()
        stealth.step()
        stealth.zero_grad(set_to_none=True)
        old = [p.detach().clone() for p in params]
        torch.nn.functional.cross_entropy(
            self.ctx.x_attack @ params[0] + params[1], torch.tensor([1])
        ).backward()
        target.step()
        with torch.no_grad():
            for param, before in zip(params, old):
                param.copy_(before + 7. * (param - before))
        expected = torch.cat([p.detach().reshape(-1) for p in params]) - self.vector
        actual, _, audit = self.run_attack(optimizer="adam", lr=rate, boost=7.)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(audit["target_steps"], 1)

    def test_weights_scope_keeps_bias_at_unboosted_target_step(self):
        self.kwargs.update(attack_epochs=1, batch_size=4, distance_weight=0.)
        no_boost, _, _ = self.run_attack(boost=1.)
        weights_boost, _, _ = self.run_attack(boost=7., scope="weights")
        all_boost, _, _ = self.run_attack(boost=7.)
        self.assertTrue(torch.equal(weights_boost[4:], no_boost[4:]))
        self.assertTrue(torch.equal(weights_boost[:4], all_boost[:4]))
        self.assertFalse(torch.equal(weights_boost[4:], all_boost[4:]))

    def test_weights_scope_applies_no_bias_distance_penalty(self):
        self.kwargs.update(attack_epochs=1, batch_size=4, distance_weight=0.)
        no_distance, _, _ = self.run_attack(boost=1., scope="weights")
        self.kwargs["distance_weight"] = 5.
        distance, _, _ = self.run_attack(boost=1., scope="weights")
        # Disable target gradient to isolate the stealth regularizer.
        real_ce = torch.nn.functional.cross_entropy
        counter = 0

        def zero_target(logits, labels):
            nonlocal counter
            counter += 1
            loss = real_ce(logits, labels)
            return loss * 0 if counter == 3 else loss

        with patch.object(torch.nn.functional, "cross_entropy", side_effect=zero_target):
            distance, _, _ = self.run_attack(boost=1., scope="weights")
        counter = 0
        self.kwargs["distance_weight"] = 0.
        with patch.object(torch.nn.functional, "cross_entropy", side_effect=zero_target):
            no_distance, _, _ = self.run_attack(boost=1., scope="weights")
        self.assertTrue(torch.equal(distance[4:], no_distance[4:]))
        self.assertFalse(torch.equal(distance[:4], no_distance[:4]))

    def test_adam_states_are_isolated_between_clients_and_calls(self):
        first, _, _ = self.run_attack(optimizer="adam", lr=.001)
        self.run_attack(optimizer="adam", lr=.004, boost=10.)
        again, _, _ = self.run_attack(optimizer="adam", lr=.001)
        self.assertTrue(torch.equal(first, again))
        self.assertTrue(torch.equal(
            self.vector, torch.tensor([.15, -.2, .25, .4, .05, -.02])
        ))

    def test_numerical_failure_preserves_first_boundary_and_json(self):
        self.ctx.x_attack[0, 1] = float("nan")
        with self.assertRaises(attack.NumericalFailure) as caught:
            self.run_attack()
        audit = caught.exception.audit
        self.assertEqual(audit["first_failure"]["stage"], "target.input")
        self.assertEqual(audit["first_failure"]["nan_count"], 1)
        self.assertEqual(audit["reference_steps"], 0)
        json.dumps(audit, allow_nan=False)

    def test_forward_failure_catches_logits_before_loss(self):
        self.ctx.x_train.fill_(float("inf"))
        with self.assertRaises(attack.NumericalFailure) as caught:
            self.run_attack()
        self.assertEqual(caught.exception.audit["first_failure"]["stage"],
                         "reference.input")
        with patch.object(backend, "_torch_forward",
                          lambda _torch, params, x, _spec:
                          (x @ params[0] + params[1]) * float("inf")):
            self.ctx.x_train.fill_(.1)
            with self.assertRaises(attack.NumericalFailure) as caught:
                self.run_attack()
        self.assertEqual(caught.exception.audit["first_failure"]["stage"],
                         "reference.logits")

    def test_gradient_failure_names_parameter(self):
        original_forward = backend._torch_forward

        def broken(_torch, params, x, spec):
            params[0].register_hook(lambda gradient: gradient * float("nan"))
            return original_forward(_torch, params, x, spec)

        with patch.object(backend, "_torch_forward", side_effect=broken):
            with self.assertRaises(attack.NumericalFailure) as caught:
                self.run_attack()
        failure = caught.exception.audit["first_failure"]
        self.assertEqual(failure["stage"], "reference.gradients")
        self.assertEqual(failure["param"], "param_0")
        self.assertEqual(failure["block"], 1)

    def test_scope_same_signature_callbacks_and_restoration(self):
        original = backend.TorchTrainingContext.alternating_minimization_delta_resident
        reports = []
        with attack.attack_scope(self.profile, reports.append):
            wrapped = backend.TorchTrainingContext.alternating_minimization_delta_resident
            self.assertEqual(list(inspect.signature(original).parameters),
                             list(inspect.signature(wrapped).parameters))
            wrapped(self.ctx, self.vector, **self.kwargs)
            self.assertEqual(reports[-1]["status"], "complete")
            self.ctx.x_attack.fill_(float("nan"))
            with self.assertRaises(attack.NumericalFailure):
                wrapped(self.ctx, self.vector, **self.kwargs)
        self.assertIs(backend.TorchTrainingContext.alternating_minimization_delta_resident,
                      original)
        self.assertEqual(reports[-1]["status"], "nonfinite")
        self.assertEqual(len(reports), 2)

    def test_scope_restores_after_escaping_failure_and_nested_scope(self):
        original = backend.TorchTrainingContext.alternating_minimization_delta_resident
        self.ctx.x_attack.fill_(float("inf"))
        with self.assertRaises(attack.NumericalFailure):
            with attack.attack_scope(self.profile):
                outer = backend.TorchTrainingContext.alternating_minimization_delta_resident
                with attack.attack_scope(dict(self.profile, id="inner")):
                    pass
                self.assertIs(backend.TorchTrainingContext.alternating_minimization_delta_resident,
                              outer)
                outer(self.ctx, self.vector, **self.kwargs)
        self.assertIs(backend.TorchTrainingContext.alternating_minimization_delta_resident,
                      original)

    def test_scope_reports_runtime_errors_and_restores(self):
        original = backend.TorchTrainingContext.alternating_minimization_delta_resident
        reports = []
        with self.assertRaisesRegex(RuntimeError, "simulated backend failure"):
            with attack.attack_scope(self.profile, reports.append):
                with patch.object(backend, "_torch_forward",
                                  side_effect=RuntimeError("simulated backend failure")):
                    backend.TorchTrainingContext.alternating_minimization_delta_resident(
                        self.ctx, self.vector, **self.kwargs
                    )
        self.assertIs(backend.TorchTrainingContext.alternating_minimization_delta_resident,
                      original)
        self.assertEqual(reports[0]["status"], "error")
        self.assertEqual(reports[0]["error_type"], "RuntimeError")
        json.dumps(reports[0], allow_nan=False)

    def test_zero_target_loss_does_not_advance_adam_target_optimizer(self):
        self.kwargs.update(attack_epochs=1, batch_size=4)
        real_ce = torch.nn.functional.cross_entropy
        count = 0

        def zero_target(logits, labels):
            nonlocal count
            count += 1
            result = real_ce(logits, labels)
            return result * 0 if count == 3 else result

        with patch.object(torch.nn.functional, "cross_entropy", side_effect=zero_target):
            _, _, audit = self.run_attack(optimizer="adam", lr=.001)
        self.assertEqual(audit["target_steps"], 0)
        self.assertEqual(audit["target_skipped_zero_loss"], 1)
        self.assertEqual(audit["target_delta_l2_total"], 0.)

    def test_expired_deadline_stops_before_training(self):
        with self.assertRaises(TimeoutError) as caught:
            attack.run_profile_attack(
                self.ctx, self.vector, self.profile, deadline_epoch=0, **self.kwargs
            )
        self.assertEqual(caught.exception.audit["status"], "budget_expired")
        self.assertEqual(caught.exception.audit["reference_steps"], 0)
        json.dumps(caught.exception.audit, allow_nan=False)

    def test_zero_bias_does_not_inflate_weight_growth_metric(self):
        self.vector[4:] = 0.
        _, _, audit = self.run_attack()
        self.assertGreater(audit["max_parameter_growth_ratio"], 1000.)
        self.assertLess(audit["max_weight_growth_ratio"], 2.)


@unittest.skipIf(torch is None, "PyTorch unavailable")
class RealCnnParityTest(unittest.TestCase):
    def test_real_cifar_cnn_retains_sgd_result(self):
        from sm9rrsfl.datasets import ImageDataset
        from sm9rrsfl.model import ModelSpec, init_params

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            spec = ModelSpec(
                architecture="cifar10", input_shape=(3, 8, 8), num_classes=3,
                cifar_conv_filters=(2, 3), cifar_hidden_dims=(5, 4),
            )
            rng = np.random.default_rng(9)
            x = rng.normal(size=(7, 3, 8, 8)).astype(np.float32)
            y = np.array([0, 1, 2, 0, 1, 2, 1])
            dataset = ImageDataset(
                x[:5], y[:5], x[5:6], y[5:6], num_classes=3,
                x_attack=x[6:], y_attack=y[6:],
            )
            ctx = backend.TorchTrainingContext(
                dataset, [np.arange(5)], spec=spec, device="cpu"
            )
            vector = init_params(seed=8, spec=spec)
            kwargs = dict(
                client_idx=0, target_indices=np.array([0]), target_label=2,
                lr=.001, attack_epochs=2, batch_size=2, stealth_steps=2,
                boost=2., distance_weight=.01, seed=301,
            )
            profile = dict(
                id="parity", optimizer="sgd", lr=None,
                boost=2., stealth_steps=2, scope="all",
            )
            old_delta, old_stats = ctx.alternating_minimization_delta_resident(
                vector, **kwargs
            )
            delta, stats, audit = attack.run_profile_attack(ctx, vector, profile, **kwargs)
            self.assertTrue(torch.equal(delta, old_delta))
            self.assertEqual(stats, old_stats)
            self.assertEqual(audit["normal_steps"], 6)
            self.assertEqual(audit["target_steps"], 3)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
