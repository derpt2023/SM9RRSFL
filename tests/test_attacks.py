import unittest
from unittest import mock

import numpy as np

from sm9rrsfl.attacks import (
    is_alternating_minimization_attack,
    poison_update,
)
from sm9rrsfl.model import (
    DEFAULT_SPEC,
    _loss_and_gradient,
    alternating_minimization_delta,
    init_params,
    targeted_metrics,
)


class AttackTest(unittest.TestCase):
    def test_alternating_names_select_model_aware_attack(self):
        self.assertTrue(is_alternating_minimization_attack("alternating"))
        self.assertTrue(
            is_alternating_minimization_attack("alternating_minimization")
        )
        self.assertFalse(is_alternating_minimization_attack("gaussian"))

    def test_alternating_attack_cannot_be_used_as_posthoc_vector_noise(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires model/data-aware local training",
        ):
            poison_update(
                np.ones(16, dtype=np.float32),
                attack="alternating_minimization",
                scale=5.0,
                seed=3,
            )

    def test_alternating_minimization_reduces_target_loss(self):
        rng = np.random.default_rng(7)
        params = init_params(seed=11, spec=DEFAULT_SPEC)
        local_x = rng.normal(size=(8, 1, 28, 28)).astype(np.float32)
        local_y = np.arange(8, dtype=np.int64) % DEFAULT_SPEC.num_classes
        auxiliary_x = rng.normal(size=(1, 1, 28, 28)).astype(np.float32)
        target_labels = np.array([7], dtype=np.int64)

        before, _ = _loss_and_gradient(
            params,
            auxiliary_x,
            target_labels,
            DEFAULT_SPEC,
        )
        delta, stats = alternating_minimization_delta(
            params,
            local_x,
            local_y,
            auxiliary_x,
            target_labels,
            lr=0.01,
            attack_epochs=1,
            batch_size=4,
            stealth_steps=2,
            boost=20.0,
            distance_weight=1e-4,
            seed=13,
            spec=DEFAULT_SPEC,
        )
        after, _ = _loss_and_gradient(
            params + delta,
            auxiliary_x,
            target_labels,
            DEFAULT_SPEC,
        )
        _, confidence_before = targeted_metrics(
            params,
            auxiliary_x,
            target_labels,
            spec=DEFAULT_SPEC,
        )
        _, confidence_after = targeted_metrics(
            params + delta,
            auxiliary_x,
            target_labels,
            spec=DEFAULT_SPEC,
        )

        self.assertLess(after, before)
        self.assertGreater(confidence_after, confidence_before)
        self.assertEqual(stats.samples, len(local_y))
        self.assertTrue(np.isfinite(delta).all())
        self.assertGreater(np.count_nonzero(delta), len(delta) // 8)

    def test_alternating_schedule_separates_stealth_distance_and_boost(self):
        params = np.zeros(DEFAULT_SPEC.parameter_size, dtype=np.float32)
        local_x = np.zeros((2, 1, 28, 28), dtype=np.float32)
        local_y = np.zeros(2, dtype=np.int64)
        auxiliary_x = np.ones((1, 1, 28, 28), dtype=np.float32)
        target_labels = np.array([7], dtype=np.int64)
        benign_reference_delta = np.full_like(params, 0.5)
        call_order: list[str] = []

        def fake_loss_and_gradient(vector, x, labels, spec):
            del vector, x, spec
            is_target = int(np.asarray(labels).reshape(-1)[0]) == 7
            call_order.append("target" if is_target else "stealth")
            gradient_value = 3.0 if is_target else 1.0
            return 0.0, np.full_like(params, gradient_value)

        with (
            mock.patch(
                "sm9rrsfl.model.local_train_delta",
                return_value=(
                    benign_reference_delta,
                    mock.Mock(loss=0.0, samples=2),
                ),
            ),
            mock.patch(
                "sm9rrsfl.model._loss_and_gradient",
                side_effect=fake_loss_and_gradient,
            ),
        ):
            delta, _ = alternating_minimization_delta(
                params,
                local_x,
                local_y,
                auxiliary_x,
                target_labels,
                lr=0.1,
                attack_epochs=1,
                batch_size=1,
                stealth_steps=1,
                boost=4.0,
                distance_weight=2.0,
                seed=31,
                spec=DEFAULT_SPEC,
            )

        # Starting at zero with a benign reference of 0.5, the first stealth
        # gradient is 1 + 2 * (0 - 0.5) = 0.  Each target step contributes
        # -0.1 * 4 * 3 = -1.2.  The second stealth step pulls the model back by
        # +0.24 before the final boosted target step, yielding -2.16.
        self.assertEqual(
            call_order,
            ["stealth", "target", "stealth", "target"],
        )
        np.testing.assert_allclose(delta, -2.16, rtol=1e-6, atol=1e-6)

    def test_alternating_minimization_is_reproducible(self):
        rng = np.random.default_rng(21)
        params = init_params(seed=22, spec=DEFAULT_SPEC)
        local_x = rng.normal(size=(6, 1, 28, 28)).astype(np.float32)
        local_y = np.arange(6, dtype=np.int64)
        auxiliary_x = local_x[:1]
        target_labels = np.array([7], dtype=np.int64)
        kwargs = dict(
            lr=0.005,
            attack_epochs=1,
            batch_size=3,
            stealth_steps=2,
            boost=10.0,
            distance_weight=1e-4,
            seed=23,
            spec=DEFAULT_SPEC,
        )
        first, _ = alternating_minimization_delta(
            params,
            local_x,
            local_y,
            auxiliary_x,
            target_labels,
            **kwargs,
        )
        second, _ = alternating_minimization_delta(
            params,
            local_x,
            local_y,
            auxiliary_x,
            target_labels,
            **kwargs,
        )
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
