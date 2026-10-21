"""Process-local, audited attack variants for the independent stability study.

The production backend is never edited.  ``attack_scope`` temporarily replaces
its attack entry point and restores that exact callable, including on failure.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import math
import time

import numpy as np

from sm9rrsfl.model import TrainStats
from sm9rrsfl import torch_backend as backend


class NumericalFailure(RuntimeError):
    """The first observed nonfinite boundary, with JSON-safe public evidence."""

    def __init__(self, audit):
        self.audit = deepcopy(audit)
        failure = self.audit["first_failure"]
        super().__init__(
            f"nonfinite attack computation: {failure['stage']} "
            f"block={failure['block']} param={failure.get('param')}"
        )


def _profile_values(profile, inherited_lr):
    optimizer = profile["optimizer"]
    scope = profile["scope"]
    lr = inherited_lr if profile.get("lr") is None else float(profile["lr"])
    boost = float(profile["boost"])
    steps = profile["stealth_steps"]
    if optimizer not in {"sgd", "adam"} or scope not in {"all", "weights"}:
        raise ValueError("unsupported attack optimizer or parameter scope")
    if not all(math.isfinite(v) and v > 0 for v in (lr, boost)):
        raise ValueError("attack learning rate and boost must be finite and positive")
    if isinstance(steps, bool) or int(steps) != steps or steps < 1:
        raise ValueError("stealth_steps must be a positive integer")
    return optimizer, scope, lr, boost, int(steps)


def run_profile_attack(
    ctx,
    global_vector,
    profile,
    *,
    client_idx,
    target_indices,
    target_label,
    lr,
    attack_epochs,
    batch_size,
    stealth_steps,
    boost,
    distance_weight,
    seed,
    check=True,
    deadline_epoch=None,
):
    """Return resident ``(delta, TrainStats, audit)`` for one client.

    Adam has three fresh, independent optimizer states: benign reference,
    stealth, and target.  Target boosting applies to the *Adam displacement*,
    not to its gradient.  SGD retains the production arithmetic ordering.
    """
    optimizer, scope, lr, boost, stealth_steps = _profile_values(profile, lr)
    if attack_epochs < 1 or batch_size < 1:
        raise ValueError("attack_epochs and batch_size must be positive")
    if not np.isfinite(distance_weight) or distance_weight < 0:
        raise ValueError("distance_weight must be finite and non-negative")
    if not 0 <= target_label < ctx.spec.num_classes:
        raise ValueError("target_label is outside the model class range")
    torch = ctx.torch
    indices = ctx.client_indices[client_idx]
    samples = int(indices.numel())
    audit = {
        "profile_id": str(profile["id"]),
        "client_idx": int(client_idx),
        "source_client": f"client-{client_idx}",
        "optimizer": optimizer,
        "scope": scope,
        "lr": float(lr),
        "boost": boost,
        "stealth_steps": stealth_steps,
        "checks_enabled": bool(check),
        "reference_steps": 0,
        "normal_steps": 0,
        "target_steps": 0,
        "target_skipped_zero_loss": 0,
        "target_success": None,
        "target_confidence": None,
        "target_delta_l2_total": 0.0,
        "max_parameter_growth_ratio": 1.0,
        "max_weight_growth_ratio": 1.0,
        "parameter_growth_denominator_floor": 1e-12,
        "status": "running",
        "first_failure": None,
    }

    def budget(stage, block=0):
        if deadline_epoch is not None and time.time() >= deadline_epoch:
            audit["status"] = "budget_expired"
            audit["budget_boundary"] = {"stage": stage, "block": int(block)}
            error = TimeoutError("attack stability study deadline reached")
            error.audit = deepcopy(audit)
            raise error

    def finite(tensors, stage, block=0, names=None):
        if not check:
            return
        values = list(tensors)
        if not values:
            return
        flags = torch.stack([torch.isfinite(t).all() for t in values])
        if bool(flags.all().item()):
            return
        position = int((~flags).nonzero()[0].item())
        value = values[position].detach()
        audit["status"] = "nonfinite"
        audit["first_failure"] = {
            "stage": stage,
            "block": int(block),
            "param": names[position] if names else None,
            "shape": list(value.shape),
            "nan_count": int(torch.isnan(value).sum().item()),
            "inf_count": int(torch.isinf(value).sum().item()),
        }
        raise NumericalFailure(audit)

    def norms(values):
        # Normal operation transfers only small scalars. Large finite tensors
        # get a float64 CPU reduction solely for audit, never for optimization.
        values = list(values)
        measured = torch.stack(
            [torch.linalg.vector_norm(v.detach()) for v in values]
        ).detach().cpu().tolist()
        for i, number in enumerate(measured):
            if not math.isfinite(number):
                array = values[i].detach().cpu().numpy().astype(np.float64)
                maximum = float(np.max(np.abs(array))) if array.size else 0.0
                measured[i] = (
                    maximum * float(np.sqrt(np.sum((array / maximum) ** 2)))
                    if math.isfinite(maximum) and maximum > 0 else maximum
                )
        return measured

    global_tensor = ctx._ensure_global_vector(global_vector)
    finite([global_tensor], "global")
    if not samples:
        audit["status"] = "complete"
        return torch.zeros_like(global_tensor), TrainStats(loss=0.0, samples=0), audit
    target_array = np.asarray(target_indices, dtype=np.int64).reshape(-1)
    if not target_array.size or not ctx.x_attack.shape[0]:
        raise ValueError("attack requires training-derived auxiliary target samples")
    target_index = torch.as_tensor(target_array, dtype=torch.long, device=ctx.device)
    target_features = ctx.x_attack.index_select(0, target_index)
    target_labels = torch.full(
        (len(target_array),), int(target_label), dtype=torch.long, device=ctx.device
    )
    finite([target_features], "target.input")

    def new_params():
        return list(backend._torch_params_from_tensor(
            torch, global_tensor, ctx.spec, requires_grad=True, clone=True
        ))

    params = new_params()
    names = (
        ["conv1_w", "conv1_b", "conv2_w", "conv2_b", "dense1_w",
         "dense1_b", "dense2_w", "dense2_b", "logits_w", "logits_b"]
        if ctx.spec.architecture == "cifar10"
        else ["conv_w", "conv_b", "dense_w", "dense_b"]
    )
    if len(names) != len(params):
        names = [f"param_{i}" for i in range(len(params))]
    initial_norms = norms(params)

    def growth(values):
        ratios = [
            now / max(before, 1e-12)
            for now, before in zip(norms(values), initial_norms)
        ]
        maximum = max(ratios, default=1.0)
        if math.isfinite(maximum):
            audit["max_parameter_growth_ratio"] = max(
                audit["max_parameter_growth_ratio"], maximum
            )
        weight_maximum = max(
            (ratio for p, ratio in zip(values, ratios) if p.ndim > 1), default=1.0
        )
        if math.isfinite(weight_maximum):
            audit["max_weight_growth_ratio"] = max(
                audit["max_weight_growth_ratio"], weight_maximum
            )

    def batches(rng_seed):
        rng = np.random.default_rng(rng_seed)
        result = []
        for _ in range(attack_epochs):
            order = rng.permutation(samples)
            result.extend(order[start:start + batch_size]
                          for start in range(0, samples, batch_size))
        return result

    def batch_data(batch):
        local = torch.as_tensor(batch, dtype=torch.long, device=ctx.device)
        selected = indices.index_select(0, local)
        return (ctx.x_train.index_select(0, selected),
                ctx.y_train.index_select(0, selected))

    def adam(values):
        return torch.optim.Adam(
            values, lr=lr, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0, amsgrad=False, foreach=False,
        )

    def backward(values, features, labels, phase, block):
        finite([features], f"{phase}.input", block)
        logits = backend._torch_forward(torch, values, features, ctx.spec)
        finite([logits], f"{phase}.logits", block)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        finite([loss], f"{phase}.loss", block)
        loss.backward()
        finite([p.grad for p in values if p.grad is not None],
               f"{phase}.gradients", block,
               [name for p, name in zip(values, names) if p.grad is not None])
        return loss

    # This reproduces local_train_delta_resident's SGD batches and arithmetic,
    # while exposing its first nonfinite boundary rather than just its delta.
    reference = new_params()
    reference_optimizer = adam(reference) if optimizer == "adam" else None
    for step, batch in enumerate(batches(seed + 1_000_003), 1):
        budget("reference", step)
        features, labels = batch_data(batch)
        backward(reference, features, labels, "reference", step)
        with torch.no_grad():
            if reference_optimizer is None:
                for param in reference:
                    if param.grad is not None:
                        param -= lr * param.grad
            else:
                reference_optimizer.step()
            for param in reference:
                param.grad = None
        audit["reference_steps"] += 1
        finite(reference, "reference.parameters", step, names)
        growth(reference)
    # Production forms global + (trained - global), so retain the two
    # floating-point operations rather than directly using trained parameters.
    reference_vector = global_tensor.add(
        backend._torch_flat_vector_from_params(torch, reference).sub(global_tensor)
    )
    finite([reference_vector], "reference.vector")
    reference = list(backend._torch_params_from_tensor(
        torch, reference_vector, ctx.spec, requires_grad=False, clone=False
    ))
    stealth_optimizer = adam(params) if optimizer == "adam" else None
    target_optimizer = adam(params) if optimizer == "adam" else None
    benign_batches = batches(seed)
    loss_sum = None
    loss_batches = 0
    for block_start in range(0, len(benign_batches), stealth_steps):
        block = block_start // stealth_steps + 1
        for batch in benign_batches[block_start:block_start + stealth_steps]:
            budget("stealth", block)
            features, labels = batch_data(batch)
            loss = backward(params, features, labels, "stealth", block)
            with torch.no_grad():
                gradients = []
                gradient_names = []
                for param, ref in zip(params, reference):
                    if param.grad is not None:
                        regularize = scope == "all" or param.ndim > 1
                        if regularize:
                            gradient = param.grad.add(
                                param.sub(ref), alpha=float(distance_weight)
                            )
                        else:
                            gradient = param.grad
                        gradients.append(gradient)
                        gradient_names.append(names[len(gradients) - 1])
                    else:
                        gradients.append(None)
                        gradient_names.append(names[len(gradients) - 1])
                finite([g for g in gradients if g is not None],
                       "stealth.regularized_gradients", block,
                       [n for g, n in zip(gradients, gradient_names) if g is not None])
                for param, gradient in zip(params, gradients):
                    if gradient is not None:
                        if stealth_optimizer is None:
                            param.add_(gradient, alpha=-float(lr))
                        else:
                            param.grad = gradient
                if stealth_optimizer is not None:
                    stealth_optimizer.step()
                for param in params:
                    param.grad = None
            audit["normal_steps"] += 1
            finite(params, "stealth.parameters", block, names)
            growth(params)
            detached = loss.detach()
            loss_sum = detached if loss_sum is None else loss_sum + detached
            finite([loss_sum], "stealth.loss_accumulator", block)
            loss_batches += 1

        budget("target", block)
        target_logits = backend._torch_forward(torch, params, target_features, ctx.spec)
        finite([target_logits], "target.logits", block)
        target_loss = torch.nn.functional.cross_entropy(target_logits, target_labels)
        finite([target_loss], "target.loss", block)
        # Float32 cross entropy can round to zero while non-target softmax
        # gradients remain nonzero. Preserve the production SGD backward;
        # only the independent Adam author-style variant skips a zero loss.
        if optimizer == "adam" and float(target_loss.detach().item()) == 0.0:
            audit["target_skipped_zero_loss"] += 1
            continue
        target_loss.backward()
        finite([p.grad for p in params if p.grad is not None],
               "target.gradients", block,
               [n for p, n in zip(params, names) if p.grad is not None])
        before = [p.detach().clone() for p in params]
        with torch.no_grad():
            if target_optimizer is None:
                for param in params:
                    if param.grad is not None:
                        factor = boost if scope == "all" or param.ndim > 1 else 1.0
                        param.add_(param.grad, alpha=-float(lr * factor))
            else:
                target_optimizer.step()
                finite(params, "target.optimizer_parameters", block, names)
                for param, old in zip(params, before):
                    if scope == "all" or param.ndim > 1:
                        displacement = param.sub(old)
                        param.copy_(old.add(displacement, alpha=float(boost)))
            for param in params:
                param.grad = None
        audit["target_steps"] += 1
        finite(params, "target.parameters", block, names)
        step_norm = math.hypot(*norms(p.detach().sub(old)
                                     for p, old in zip(params, before)))
        if math.isfinite(step_norm):
            audit["target_delta_l2_total"] += step_norm
        growth(params)

    delta = backend._torch_flat_vector_from_params(torch, params).sub(global_tensor).detach()
    finite([delta], "delta")
    with torch.no_grad():
        logits = backend._torch_forward(torch, params, target_features, ctx.spec)
        finite([logits], "evaluation.logits")
        probabilities = torch.softmax(logits, dim=1)
        finite([probabilities], "evaluation.probabilities")
        audit["target_success"] = float(
            (logits.argmax(dim=1) == target_labels).float().mean().item()
        )
        audit["target_confidence"] = float(
            probabilities[:, int(target_label)].mean().item()
        )
    mean_loss = 0.0
    if loss_sum is not None and loss_batches:
        mean_loss = float((loss_sum / loss_batches).detach().cpu().item())
        if check and not math.isfinite(mean_loss):
            finite([loss_sum], "loss_accumulator")
    if not math.isfinite(audit["target_confidence"]):
        audit["target_confidence"] = None
        audit["status"] = "unchecked_nonfinite"
    else:
        audit["status"] = "complete"
    return delta, TrainStats(loss=mean_loss, samples=samples), audit


@contextmanager
def attack_scope(profile, on_audit=None, check=True):
    """Install one profile in this process, restoring any enclosing adapter."""
    original = backend.TorchTrainingContext.alternating_minimization_delta_resident
    frozen_profile = deepcopy(profile)

    def attack(
        self, global_vector, *, client_idx, target_indices, target_label, lr,
        attack_epochs, batch_size, stealth_steps, boost, distance_weight, seed,
    ):
        try:
            delta, stats, audit = run_profile_attack(
                self, global_vector, frozen_profile, client_idx=client_idx,
                target_indices=target_indices, target_label=target_label, lr=lr,
                attack_epochs=attack_epochs, batch_size=batch_size,
                stealth_steps=stealth_steps, boost=boost,
                distance_weight=distance_weight, seed=seed, check=check,
            )
        except BaseException as exc:
            if on_audit is not None:
                failure_audit = getattr(exc, "audit", None)
                if failure_audit is None:
                    failure_audit = {
                        "profile_id": str(frozen_profile["id"]),
                        "client_idx": int(client_idx),
                        "source_client": f"client-{client_idx}",
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "first_failure": None,
                    }
                on_audit(deepcopy(failure_audit))
            raise
        if on_audit is not None:
            on_audit(deepcopy(audit))
        return delta, stats

    backend.TorchTrainingContext.alternating_minimization_delta_resident = attack
    try:
        yield
    finally:
        backend.TorchTrainingContext.alternating_minimization_delta_resident = original
