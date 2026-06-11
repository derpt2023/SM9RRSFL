"""Model poisoning attacks for federated update experiments."""

from __future__ import annotations

import numpy as np


_ALTERNATING_TRIGGER_SHARDS = 8
_ALTERNATING_TRIGGER_SEED = 104729


def poison_update(
    update: np.ndarray,
    *,
    attack: str = "alternating",
    scale: float = 5.0,
    seed: int = 0,
) -> np.ndarray:
    """Return a malicious version of a client update."""

    if attack == "none":
        return update.astype(np.float32, copy=True)
    if attack == "sign_flip":
        return (-scale * update).astype(np.float32)
    if attack == "gaussian":
        rng = np.random.default_rng(seed)
        std = float(np.std(update))
        if std <= 1e-12:
            std = float(np.linalg.norm(update) / max(1, update.size) ** 0.5)
        std = max(std, 1e-3)
        return rng.normal(0.0, scale * std, size=update.shape).astype(np.float32)
    if attack == "alternating":
        return _alternating_trigger_poison(update, scale=scale, seed=seed)
    raise ValueError("attack must be one of: none, sign_flip, gaussian, alternating")


def _alternating_trigger_poison(
    update: np.ndarray,
    *,
    scale: float,
    seed: int,
) -> np.ndarray:
    """Inject one small shard of a decomposed trigger into the update."""

    poisoned = update.astype(np.float32, copy=True)
    flat = poisoned.reshape(-1)
    if flat.size == 0:
        return poisoned

    shard_count = min(_ALTERNATING_TRIGGER_SHARDS, flat.size)
    shard_index = seed % shard_count
    start = flat.size * shard_index // shard_count
    end = flat.size * (shard_index + 1) // shard_count

    rms = float(np.linalg.norm(flat) / max(1, flat.size) ** 0.5)
    magnitude = max(rms, 1e-6) * (float(scale) / 100.0)
    rng = np.random.default_rng(_ALTERNATING_TRIGGER_SEED + shard_index)
    trigger = rng.choice((-1.0, 1.0), size=end - start).astype(np.float32)
    flat[start:end] += trigger * magnitude
    return poisoned
