"""Explicit finite-backtracking client optimizer for a new experiment protocol.

This module is opt-in and never changes production defaults.  It checks
numerical finiteness, not whether a step improves its objective.  All local
optimizer phases use the same rule, including the attacker's reference and
target steps.  The extra forwards/backwards have a material runtime cost.
Checks cover the current minibatch and the attack's current cross-objective,
not every future minibatch. A new minibatch with an already nonfinite input
loss/gradient fails explicitly; this policy is not a guarantee of convergence.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from threading import Lock
from typing import Any

import numpy as np

from .model import TrainStats
from . import torch_backend as backend


IMPLEMENTATION_VARIANT = "shared-client-finite-backtracking-v1"
_PATCH_LOCK = Lock()


class NumericalTrainingError(RuntimeError):
    """A client optimizer cannot produce a finite step under its declared rule."""

    def __init__(self, event):
        self.event = event
        super().__init__(
            f"numerical training failure phase={event['phase']} "
            f"client={event['client_idx']} seed={event['seed']} "
            f"batch={event['batch']} reason={event['reason']}"
        )


def _finite(torch, tensors):
    # Reduce on-device first; the list contains one scalar per model tensor.
    # A single host synchronization replaces one synchronization per layer.
    checks = [torch.isfinite(value).all() for value in tensors if value is not None]
    return not checks or bool(torch.stack(checks).all().detach().cpu().item())


def _tensor_statistics(torch, tensors):
    total = finite = 0
    maximum = 0.0
    for tensor in tensors:
        if tensor is None:
            continue
        value = tensor.detach()
        mask = torch.isfinite(value)
        count = int(mask.sum().cpu().item())
        total += value.numel()
        finite += count
        if count:
            maximum = max(maximum, float(value[mask].abs().max().cpu().item()))
    return {"elements": total, "finite_elements": finite, "finite_abs_max": maximum}


def _new_counters():
    return {"total_steps": 0, "accepted_steps": 0, "backtracked_steps": 0,
            "retries": 0, "failures": 0, "original_lr_min": None,
            "original_lr_max": None, "actual_lr_min": None, "actual_lr_max": None}


class StableTrainingPolicy:
    """Policy/counters yielded by :func:`stable_client_training`."""

    def __init__(self, max_backtracks=12, event_callback=None):
        if isinstance(max_backtracks, bool) or not isinstance(max_backtracks, int) or max_backtracks < 0:
            raise ValueError("max_backtracks must be a non-negative integer")
        if event_callback is not None and not callable(event_callback):
            raise TypeError("event_callback must be callable")
        self.max_backtracks = max_backtracks
        self.event_callback = event_callback
        self.by_phase = {}

    def summary(self):
        phases = {phase: dict(counts) for phase, counts in self.by_phase.items()}
        totals = _new_counters()
        for counts in phases.values():
            for name in ("total_steps", "accepted_steps", "backtracked_steps", "retries", "failures"):
                totals[name] += counts[name]
            for name in ("original_lr_min", "actual_lr_min"):
                if counts[name] is not None:
                    totals[name] = counts[name] if totals[name] is None else min(totals[name], counts[name])
            for name in ("original_lr_max", "actual_lr_max"):
                if counts[name] is not None:
                    totals[name] = counts[name] if totals[name] is None else max(totals[name], counts[name])
        return {"implementation_variant": IMPLEMENTATION_VARIANT,
                "max_backtracks": self.max_backtracks, "totals": totals, "by_phase": phases}

    @staticmethod
    def _observe_lr(counts, prefix, value):
        for suffix, operation in (("min", min), ("max", max)):
            key = f"{prefix}_lr_{suffix}"
            counts[key] = value if counts[key] is None else operation(counts[key], value)

    def _emit(self, event):
        if self.event_callback is not None:
            self.event_callback(event)

    def validate_client_delta(self, torch, delta, *, phase, client_idx, seed, batch, learning_rate):
        """Reject overflow in local-minus-global even if both operands are finite."""

        if _finite(torch, (delta,)):
            return
        counts = self.by_phase.setdefault(phase, _new_counters())
        counts["failures"] += 1
        event = {"implementation_variant": IMPLEMENTATION_VARIANT,
                 "event": "numerical_training_failure", "reason": "nonfinite_client_delta",
                 "phase": phase, "client_idx": int(client_idx), "seed": int(seed),
                 "batch": int(batch), "original_lr": float(learning_rate),
                 "actual_lr": None, "retries": 0,
                 "delta": _tensor_statistics(torch, (delta,))}
        self._emit(event)
        raise NumericalTrainingError(event)

    def step(self, torch, params, objective, apply_update, *, learning_rate,
             phase, client_idx, seed, batch, extra_checks=(), gradient_transform=None):
        """Try the unchanged arithmetic, then halve its scalar step if needed.

        ``objective`` returns the original scalar loss. ``apply_update`` applies
        the original phase's arithmetic to cached gradients. Extra checks are
        ``(objective, optional_gradient_transform)`` pairs; they do not add an
        optimization step. Validation uses autograd.grad and cannot accumulate
        into parameter .grad fields.
        """

        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        counts = self.by_phase.setdefault(phase, _new_counters())
        counts["total_steps"] += 1
        self._observe_lr(counts, "original", learning_rate)
        context = {"implementation_variant": IMPLEMENTATION_VARIANT, "phase": phase,
                   "client_idx": int(client_idx), "seed": int(seed), "batch": int(batch),
                   "original_lr": float(learning_rate)}
        loss = None
        gradients = ()

        def clear_gradients():
            for param in params:
                param.grad = None

        def fail(reason, retries=0, actual_lr=None):
            counts["failures"] += 1
            event = {**context, "event": "numerical_training_failure", "reason": reason,
                     "retries": retries, "actual_lr": actual_lr,
                     "parameters": _tensor_statistics(torch, params),
                     "gradients": _tensor_statistics(torch, gradients),
                     "loss": float(loss.detach().cpu().item())
                     if loss is not None and _finite(torch, (loss,)) else None}
            clear_gradients()
            self._emit(event)
            raise NumericalTrainingError(event)

        if not _finite(torch, params):
            fail("nonfinite_input_parameters")
        clear_gradients()
        loss = objective()
        if not _finite(torch, (loss,)):
            fail("nonfinite_input_loss")
        loss.backward()
        gradients = tuple(param.grad.detach() if param.grad is not None else None for param in params)
        if not _finite(torch, gradients):
            fail("nonfinite_input_gradient")
        if gradient_transform is not None and not _finite(torch, gradient_transform(gradients)):
            fail("nonfinite_input_effective_gradient")
        before = tuple(param.detach().clone() for param in params)
        clear_gradients()

        def restore_parameters():
            with torch.no_grad():
                for param, previous in zip(params, before):
                    param.copy_(previous)
            clear_gradients()

        def valid_objective(check, transform):
            # autograd.grad leaves .grad untouched; explicitly restore it even
            # if a custom objective raises, rather than retaining probe state.
            saved_gradients = [param.grad for param in params]
            try:
                checked_loss = check()
                if not _finite(torch, (checked_loss,)):
                    return False
                checked_gradients = torch.autograd.grad(checked_loss, params, allow_unused=True)
                if not _finite(torch, checked_gradients):
                    return False
                return transform is None or _finite(torch, transform(checked_gradients))
            finally:
                for param, previous in zip(params, saved_gradients):
                    param.grad = previous

        failure_reason = "nonfinite_proposed_parameters"
        try:
            for retries in range(self.max_backtracks + 1):
                actual_lr = math.ldexp(float(learning_rate), -retries)
                if retries:
                    counts["retries"] += 1
                    if retries == 1:
                        counts["backtracked_steps"] += 1
                    restore_parameters()
                with torch.no_grad():
                    apply_update(gradients, actual_lr)
                accepted = _finite(torch, params)
                if accepted:
                    failure_reason = "nonfinite_post_step_loss_or_gradient"
                    accepted = valid_objective(objective, gradient_transform)
                else:
                    failure_reason = "nonfinite_proposed_parameters"
                if accepted:
                    for check, transform in extra_checks:
                        if not valid_objective(check, transform):
                            failure_reason = "nonfinite_post_step_cross_objective"
                            accepted = False
                            break
                if accepted:
                    counts["accepted_steps"] += 1
                    self._observe_lr(counts, "actual", actual_lr)
                    clear_gradients()
                    if retries:
                        self._emit({**context, "event": "numerical_training_backtrack",
                                    "retries": retries, "actual_lr": actual_lr,
                                    "parameters": _tensor_statistics(torch, params),
                                    "gradients": _tensor_statistics(torch, gradients)})
                    return loss.detach()
            # Preserve evidence about the last rejected proposal in the event;
            # restore the caller's parameters even if its callback raises.
            try:
                fail("backtracking_exhausted:" + failure_reason,
                     retries=self.max_backtracks, actual_lr=actual_lr)
            finally:
                restore_parameters()
        except BaseException:
            restore_parameters()
            raise


def _objective(context, params, features, labels):
    def loss():
        return context.torch.nn.functional.cross_entropy(
            backend._torch_forward(context.torch, params, features, context.spec), labels)
    return loss


def _local_train(context, global_vector, *, client_idx, lr, epochs, batch_size, seed,
                 phase="honest"):
    torch = context.torch
    indices = context.client_indices[client_idx]
    samples = int(indices.numel())
    global_tensor = context._ensure_global_vector(global_vector)
    if samples == 0:
        return torch.zeros_like(global_tensor), TrainStats(loss=0.0, samples=0)
    params = list(backend._torch_params_from_tensor(torch, global_tensor, context.spec,
                                                  requires_grad=True, clone=True))
    rng = np.random.default_rng(seed)
    loss_sum = None
    loss_batches = 0
    for _ in range(epochs):
        order = rng.permutation(samples)
        for start in range(0, samples, batch_size):
            local_index = torch.as_tensor(order[start:start + batch_size], dtype=torch.long,
                                          device=context.device)
            batch_index = indices.index_select(0, local_index)
            objective = _objective(context, params, context.x_train.index_select(0, batch_index),
                                   context.y_train.index_select(0, batch_index))

            def update(gradients, step):
                for param, gradient in zip(params, gradients):
                    if gradient is not None:
                        # Exactly the ordinary resident path's multiply/subtract.
                        param -= step * gradient

            loss = context._stable_policy.step(torch, params, objective, update,
                learning_rate=lr, phase=phase, client_idx=client_idx, seed=seed,
                batch=loss_batches)
            loss_sum = loss if loss_sum is None else loss_sum + loss
            loss_batches += 1
    delta = backend._torch_flat_vector_from_params(torch, params).sub(global_tensor).detach()
    context._stable_policy.validate_client_delta(torch, delta, phase=phase,
        client_idx=client_idx, seed=seed, batch=loss_batches, learning_rate=lr)
    mean_loss = float((loss_sum / loss_batches).detach().cpu().item()) if loss_batches else 0.0
    return delta, TrainStats(loss=mean_loss, samples=samples)


def _attack_train(context, global_vector, *, client_idx, target_indices, target_label,
                  lr, attack_epochs, batch_size, stealth_steps, boost, distance_weight, seed):
    torch = context.torch
    indices = context.client_indices[client_idx]
    samples = int(indices.numel())
    global_tensor = context._ensure_global_vector(global_vector)
    if samples == 0:
        return torch.zeros_like(global_tensor), TrainStats(loss=0.0, samples=0)
    targets = np.asarray(target_indices, dtype=np.int64).reshape(-1)
    if targets.size == 0:
        raise ValueError("alternating minimization requires auxiliary target samples")
    if attack_epochs < 1 or batch_size < 1 or stealth_steps < 1:
        raise ValueError("attack_epochs, batch_size and stealth_steps must be at least 1")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not np.isfinite(boost) or boost <= 0:
        raise ValueError("boost must be finite and positive")
    if not np.isfinite(distance_weight) or distance_weight < 0:
        raise ValueError("distance_weight must be finite and non-negative")
    if target_label < 0 or target_label >= context.spec.num_classes:
        raise ValueError("target_label is outside the model class range")
    if context.x_attack.shape[0] == 0:
        raise ValueError("alternating minimization requires a training-derived attack split")
    benign_delta, _ = _local_train(context, global_vector, client_idx=client_idx, lr=lr,
        epochs=attack_epochs, batch_size=batch_size, seed=seed + 1_000_003,
        phase="benign_reference")
    reference_vector = global_tensor.add(benign_delta)
    reference = tuple(backend._torch_params_from_tensor(torch, reference_vector, context.spec,
                                                       requires_grad=False, clone=False))
    params = list(backend._torch_params_from_tensor(torch, global_tensor, context.spec,
                                                  requires_grad=True, clone=True))
    target_index = torch.as_tensor(targets, dtype=torch.long, device=context.device)
    target_features = context.x_attack.index_select(0, target_index)
    target_labels = torch.full((len(targets),), int(target_label), dtype=torch.long,
                               device=context.device)
    target_objective = _objective(context, params, target_features, target_labels)
    rng = np.random.default_rng(seed)
    benign_batches = []
    for _ in range(attack_epochs):
        order = rng.permutation(samples)
        benign_batches.extend(order[start:start + batch_size]
                              for start in range(0, len(order), batch_size))

    def stealth_gradients(gradients):
        return tuple(gradient.add(param.sub(ref), alpha=float(distance_weight))
                     if gradient is not None else None
                     for param, ref, gradient in zip(params, reference, gradients))

    def stealth_update(gradients, step):
        for param, gradient in zip(params, stealth_gradients(gradients)):
            if gradient is not None:
                param.add_(gradient, alpha=-float(step))

    def target_update(gradients, step):
        for param, gradient in zip(params, gradients):
            if gradient is not None:
                param.add_(gradient, alpha=-float(step))

    loss_sum = None
    loss_batches = 0
    for block_start in range(0, len(benign_batches), stealth_steps):
        for batch_idx in benign_batches[block_start:block_start + stealth_steps]:
            local_index = torch.as_tensor(batch_idx, dtype=torch.long, device=context.device)
            batch_index = indices.index_select(0, local_index)
            stealth_objective = _objective(context, params,
                context.x_train.index_select(0, batch_index),
                context.y_train.index_select(0, batch_index))
            loss = context._stable_policy.step(torch, params, stealth_objective, stealth_update,
                learning_rate=lr, phase="attack_stealth", client_idx=client_idx, seed=seed,
                batch=loss_batches, gradient_transform=stealth_gradients,
                extra_checks=((target_objective, None),))
            loss_sum = loss if loss_sum is None else loss_sum + loss
            loss_batches += 1
        context._stable_policy.step(torch, params, target_objective, target_update,
            learning_rate=float(lr * boost), phase="attack_target", client_idx=client_idx,
            seed=seed, batch=block_start // stealth_steps,
            extra_checks=((stealth_objective, stealth_gradients),))
    delta = backend._torch_flat_vector_from_params(torch, params).sub(global_tensor).detach()
    context._stable_policy.validate_client_delta(torch, delta, phase="attack_output",
        client_idx=client_idx, seed=seed, batch=loss_batches, learning_rate=float(lr * boost))
    mean_loss = float((loss_sum / loss_batches).detach().cpu().item()) if loss_batches else 0.0
    return delta, TrainStats(loss=mean_loss, samples=samples)


@contextmanager
def stable_client_training(max_backtracks=12, event_callback=None):
    """Install the explicit protocol for contexts created inside a fresh worker.

    Yields a :class:`StableTrainingPolicy`; call ``policy.summary()`` to persist
    counters. This process-wide class replacement must surround the entire
    experiment. Overlapping/nested installations are rejected; use separate
    spawned processes for concurrent experiments. Existing contexts are not
    changed. The original class is restored on success and on every exception.
    """

    policy = StableTrainingPolicy(max_backtracks, event_callback)
    if not _PATCH_LOCK.acquire(blocking=False):
        raise RuntimeError("stable_client_training requires one context per fresh worker process")
    original = backend.TorchTrainingContext

    class StableTorchTrainingContext(original):
        _stable_policy = policy

        def local_train_delta_resident(self, global_vector, **kwargs):
            return _local_train(self, global_vector, **kwargs)

        def alternating_minimization_delta_resident(self, global_vector, **kwargs):
            return _attack_train(self, global_vector, **kwargs)

    backend.TorchTrainingContext = StableTorchTrainingContext
    try:
        yield policy
    finally:
        backend.TorchTrainingContext = original
        _PATCH_LOCK.release()


__all__ = ["IMPLEMENTATION_VARIANT", "NumericalTrainingError", "StableTrainingPolicy",
           "stable_client_training"]
