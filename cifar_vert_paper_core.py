"""VERT paper/code core adapted only to the common experiment model dimension.

Reference: mylab426/VERT cc71a9d, defenses.vert_fedavg, utils.utils and
models.ProjectHead. Dense frozen Linear(d,256), raw-coordinate A/B,
three Linear layers with two ReLUs, fresh projector/predictor/A/B each round,
sequential participant fitting and immediate cosine scoring. No ratio prior.
Private initialization RNG preserves the common local-training RNG stream.
History transitions are batched (mathematically equivalent, not bitwise claimed).
"""
from contextlib import contextmanager
from functools import partial
import json
from pathlib import Path
import random
import time

import numpy as np

from sm9rrsfl.vert import VERTDefense, VERTResult
from cifar_vert_kmeans_selection import PARAMETERS, select


def networks(dimension, projected, seed, device):
    import torch
    from torch import nn
    # Author constructors initialize on CPU, then move to CUDA. Isolate that RNG
    # so changing the defense never changes the honest training randomness.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)  # CPU only
        projector = nn.Linear(dimension, projected)
        predictor = nn.Sequential(nn.Linear(projected, projected), nn.ReLU(),
                                  nn.Linear(projected, projected), nn.ReLU(),
                                  nn.Linear(projected, projected))
    projector = projector.to(device).requires_grad_(False)
    predictor = predictor.to(device)
    a = torch.ones(dimension, device=device, requires_grad=True)
    b = torch.ones(dimension, device=device, requires_grad=True)
    return projector, predictor, a, b


def fit_client(projector, predictor, a, b, optimizers, local, global_, target,
               epochs, *, batched=True):
    """Sum of transition L2 norms, with separate author Adam optimizers."""
    import torch
    with torch.no_grad():
        projected_target = projector(target)
    losses = []
    for _ in range(epochs):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        if batched:
            prediction = predictor(projector(a * local + b * global_))
            loss = torch.linalg.vector_norm(prediction - projected_target, dim=1).sum()
        else:
            # Executable small-tensor oracle following the author's Python loop.
            loss = sum(torch.norm(predictor(projector(a * l + b * g)) - projector(t))
                       for l, g, t in zip(local, global_, target))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite VERT predictor loss')
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


class PaperVERT(VERTDefense):
    def __init__(self, *args, audit_path=None, task_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.top_k or self.malicious_ratio_prior is not None:
            raise ValueError('Paper-core experiment requires prior-free K-means')
        self.audit_path = str(audit_path) if audit_path else None
        self.task_id = task_id
        # Super's compatibility projection is never used by this independent core.
        self._dense_projection = None
        self._projection_bucket = None
        self._projection_sign = None

    def _history_feature(self, update):
        value = np.asarray(update, dtype=np.float32)
        if value.shape != (self.parameter_size,) or not np.isfinite(value).all():
            raise ValueError('Finite raw-coordinate VERT update required')
        return value.copy()

    def finalize_round(self, updates, aggregate, result):
        # Rejected histories share an immutable aggregate array. No value changes;
        # avoids 100 redundant full-model copies and enormous checkpoint files.
        global_ = self._history_feature(aggregate)
        selected = set(result.selected_clients)
        current = {c: self._history_feature(u) if c in selected else global_
                   for c, u in updates.items()}
        self._global_history.append(global_)
        self._client_history.append(current)
        self._global_history = self._global_history[-self.history_window:]
        self._client_history = self._client_history[-self.history_window:]

    def evaluate_round(self, updates, *, round_id):
        import torch
        started = time.monotonic()
        if not updates:
            raise ValueError('Empty participant set')
        order = random.Random(self.seed + 101111 + round_id * 1000003).sample(list(updates), len(updates))
        details = dict(mode='bootstrap', round=round_id, task_id=self.task_id,
                       parameters=PARAMETERS, ratio_prior_used=False,
                       core='paper_dense_raw_v1', client_order=order,
                       participant_order_policy='private_seeded_random_per_round_like_author_sampling')
        if len(self._global_history) < 2:
            scores = {c: 1.0 for c in updates}
            selected = sorted(updates)
            rejected = []
        else:
            device = torch.device(self.device)
            projector, predictor, a, b = networks(self.parameter_size, self.projection_dim,
                self.seed + 97003 + round_id * 1000003, device)
            optimizers = [torch.optim.Adam(predictor.parameters(), lr=self.learning_rate),
                          torch.optim.Adam([a], lr=self.learning_rate),
                          torch.optim.Adam([b], lr=self.learning_rate)]
            global_ = torch.as_tensor(np.stack(self._global_history), device=device)
            scores, losses = {}, {}
            for client in order:
                local = torch.as_tensor(np.stack([
                    h.get(client, g) for h, g in zip(self._client_history, self._global_history)
                ]), device=device)
                values = fit_client(projector, predictor, a, b, optimizers,
                    local[:-1], global_[:-1], local[1:], self.predict_epochs)
                losses[client] = dict(first=values[0], last=values[-1])
                with torch.no_grad():
                    predicted = predictor(projector(a * local[-1] + b * global_[-1]))
                    actual = projector(torch.as_tensor(updates[client], device=device))
                    scores[client] = float(torch.nn.functional.cosine_similarity(
                        predicted, actual, dim=0, eps=1e-8).cpu())
                del local
            if not np.isfinite(list(scores.values())).all():
                raise FloatingPointError('Nonfinite VERT similarity')
            ascending = sorted(scores.values())
            chosen, labels, mode = select(ascending)
            ranked = sorted(scores, key=lambda c: (-scores[c], c))
            selected, rejected = ranked[:len(chosen)], ranked[len(chosen):]
            details.update(mode=mode, scores_ascending=ascending, labels=labels,
                           predictor_losses=losses, coefficient_dimension=self.parameter_size,
                           projection_shape=[self.projection_dim, self.parameter_size],
                           fresh_projector=True, fresh_predictor=True,
                           initialization_seed=self.seed + 97003 + round_id * 1000003)
            del projector, predictor, a, b, optimizers, global_
        result = VERTResult(tuple(selected), tuple(rejected),
                            {c: 1 / len(selected) for c in selected}, scores)
        details.update(scores=scores, selected=selected, rejected=rejected,
                       weights=result.weights, selected_count=len(selected),
                       elapsed_seconds=time.monotonic() - started)
        if self.audit_path:
            path = Path(self.audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as stream:
                stream.write(json.dumps(details, sort_keys=True, allow_nan=False) + '\n')
        return result


@contextmanager
def paper_scope(path, task_id):
    from sm9rrsfl import fl
    original = fl.VERTDefense
    fl.VERTDefense = partial(PaperVERT, audit_path=path, task_id=task_id)
    try:
        yield
    finally:
        fl.VERTDefense = original


@contextmanager
def bounded_checkpoints(rounds):
    """Keep every metric, but write full raw-history checkpoints only at anchors/end.

Applies equally to all study arms. There is no automatic resume of partial tasks.
This affects persistence cost only; all original callbacks/evaluations still run.
"""
    from scripts.experiments.cifar10 import attack_stability_worker as worker
    original = worker._write_pickle
    def write(path, value):
        if Path(path).name == 'checkpoint.pickle' and value['completed_round'] not in (0, 24, rounds):
            return
        return original(path, value)
    worker._write_pickle = write
    try:
        yield
    finally:
        worker._write_pickle = original


@contextmanager
def actual_weights_scope(path, task_id):
    """Observe actual linear aggregation, after decisions; no labels enter VERT."""
    from sm9rrsfl import fl
    original = fl._aggregation_weight_diagnostics
    rid = 0
    def observe(counts, malicious, weights):
        nonlocal rid
        result = original(counts, malicious, weights)
        rid += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(json.dumps(dict(task_id=task_id, round=rid, weights=weights,
                honest_weight_loss=result[0], malicious_weight_mass=result[1]),
                sort_keys=True, allow_nan=False)+'\n')
        return result
    fl._aggregation_weight_diagnostics = observe
    try:
        yield
    finally:
        fl._aggregation_weight_diagnostics = original
