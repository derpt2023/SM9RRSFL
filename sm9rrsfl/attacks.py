"""Model poisoning attacks for federated update experiments."""

from __future__ import annotations

import numpy as np


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
        sign = -1.0 if seed % 2 == 0 else 1.0
        poisoned = update.astype(np.float32, copy=True)
        half = poisoned.size // 2
        poisoned[:half] *= sign * scale
        poisoned[half:] *= -sign * scale
        return poisoned
    raise ValueError("attack must be one of: none, sign_flip, gaussian, alternating")
