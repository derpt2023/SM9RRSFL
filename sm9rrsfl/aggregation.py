"""Federated aggregation rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KrumResult:
    update: np.ndarray
    selected_index: int
    scores: np.ndarray
    neighbor_count: int


def fedavg(
    updates: list[np.ndarray] | np.ndarray,
    sample_counts: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    stacked = _stack_updates(updates)
    if sample_counts is not None:
        return weighted_fedavg(stacked, sample_counts)
    return np.mean(stacked, axis=0).astype(np.float32)


def weighted_fedavg(
    updates: list[np.ndarray] | np.ndarray,
    weights: list[float] | np.ndarray,
    sample_counts: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    stacked = _stack_updates(updates)
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (stacked.shape[0],):
        raise ValueError("weights must have shape [num_updates]")
    if sample_counts is not None:
        sample_array = np.asarray(sample_counts, dtype=np.float64)
        if sample_array.shape != (stacked.shape[0],):
            raise ValueError("sample_counts must have shape [num_updates]")
        weight_array = weight_array * np.maximum(sample_array, 0.0)
    total = float(np.sum(weight_array))
    if total <= 0.0:
        return fedavg(stacked)
    normalized = weight_array / total
    return (normalized.astype(np.float32) @ stacked).astype(np.float32)


def krum(updates: list[np.ndarray] | np.ndarray, byzantine_count: int) -> KrumResult:
    """Select one update using the Krum rule."""

    stacked = _stack_updates(updates)
    n = stacked.shape[0]
    if n < 3:
        raise ValueError("Krum requires at least 3 updates")
    if byzantine_count < 0:
        raise ValueError("byzantine_count must be non-negative")
    neighbor_count = n - byzantine_count - 2
    if neighbor_count < 1:
        raise ValueError(
            "Krum requires n - f - 2 >= 1; reduce malicious ratio or increase clients"
        )

    distances = _pairwise_squared_distances(stacked)
    scores = np.empty(n, dtype=np.float64)
    for idx in range(n):
        nearest = np.sort(np.delete(distances[idx], idx))[:neighbor_count]
        scores[idx] = float(np.sum(nearest))
    selected = int(np.argmin(scores))
    return KrumResult(
        update=stacked[selected].astype(np.float32),
        selected_index=selected,
        scores=scores,
        neighbor_count=neighbor_count,
    )


def _stack_updates(updates: list[np.ndarray] | np.ndarray) -> np.ndarray:
    stacked = np.asarray(updates, dtype=np.float32)
    if stacked.ndim != 2:
        raise ValueError("updates must have shape [num_updates, num_parameters]")
    if stacked.shape[0] == 0:
        raise ValueError("at least one update is required")
    return stacked


def _pairwise_squared_distances(updates: np.ndarray) -> np.ndarray:
    norms = np.sum(updates.astype(np.float64) ** 2, axis=1, keepdims=True)
    distances = norms + norms.T - 2.0 * (updates.astype(np.float64) @ updates.astype(np.float64).T)
    return np.maximum(distances, 0.0)
