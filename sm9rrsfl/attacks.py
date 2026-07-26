"""Model poisoning attacks for federated update experiments."""

from __future__ import annotations

import numpy as np


ALTERNATING_MINIMIZATION_ATTACKS = frozenset(
    {"alternating", "alternating_minimization"}
)


def is_alternating_minimization_attack(attack: str) -> bool:
    """Return whether ``attack`` needs the in-training Bhagoji attack path."""

    return str(attack).strip().lower() in ALTERNATING_MINIMIZATION_ATTACKS


def poison_update(
    update: np.ndarray,
    *,
    attack: str = "none",
    scale: float = 5.0,
    seed: int = 0,
) -> np.ndarray:
    """Apply attacks that can be expressed as a post-training update transform.

    Bhagoji's alternating-minimization attack is intentionally rejected here:
    it needs the model, the client's benign data and target-labelled auxiliary
    samples, so treating it as a vector perturbation would silently implement a
    different attack.
    """

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
    if is_alternating_minimization_attack(attack):
        raise ValueError(
            "alternating minimization requires model/data-aware local training; "
            "use the federated experiment attack path instead of poison_update()"
        )
    raise ValueError(
        "attack must be one of: none, sign_flip, gaussian, "
        "alternating_minimization (or legacy alias alternating)"
    )
