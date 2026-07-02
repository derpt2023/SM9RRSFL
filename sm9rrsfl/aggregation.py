"""Federated aggregation rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# 大模型下先堆叠全部更新会瞬间复制数百 MB；超过该阈值后改用流式加权。
_STREAMING_AVERAGE_THRESHOLD_BYTES = 256 * 1024 * 1024


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
    if sample_counts is not None:
        return weighted_fedavg(updates, sample_counts)
    stacked = _stack_updates(updates)
    return np.mean(stacked, axis=0).astype(np.float32)


def weighted_fedavg(
    updates: list[np.ndarray] | np.ndarray,
    weights: list[float] | np.ndarray,
    sample_counts: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    if isinstance(updates, np.ndarray):
        stacked = _stack_updates(updates)
        update_count = stacked.shape[0]
        parameter_count = stacked.shape[1]
    else:
        if not updates:
            raise ValueError("at least one update is required")
        update_count = len(updates)
        parameter_count = int(np.asarray(updates[0]).size)
        stacked = None
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (update_count,):
        raise ValueError("weights must have shape [num_updates]")
    if sample_counts is not None:
        sample_array = np.asarray(sample_counts, dtype=np.float64)
        if sample_array.shape != (update_count,):
            raise ValueError("sample_counts must have shape [num_updates]")
        weight_array = weight_array * np.maximum(sample_array, 0.0)
    total = float(np.sum(weight_array))
    if total <= 0.0:
        weight_array = np.ones(update_count, dtype=np.float64)
        total = float(update_count)
    normalized = weight_array / total
    estimated_bytes = update_count * parameter_count * np.dtype(np.float32).itemsize
    if stacked is None and estimated_bytes >= _STREAMING_AVERAGE_THRESHOLD_BYTES:
        return _streaming_weighted_average(updates, normalized)
    if stacked is None:
        stacked = _stack_updates(updates)
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
    # 对角线设为无穷后可整批 partition，只选最近邻而无需逐行完整排序。
    np.fill_diagonal(distances, np.inf)
    nearest = np.partition(distances, neighbor_count - 1, axis=1)[:, :neighbor_count]
    scores = np.sum(nearest, axis=1, dtype=np.float64)
    selected = int(np.argmin(scores))
    return KrumResult(
        update=stacked[selected].astype(np.float32),
        selected_index=selected,
        scores=scores,
        neighbor_count=neighbor_count,
    )


def torch_weighted_fedavg(
    updates: list[np.ndarray] | np.ndarray,
    weights: list[float] | np.ndarray,
    sample_counts: list[int] | np.ndarray | None = None,
    *,
    device: str = "auto",
) -> np.ndarray:
    """Compute weighted FedAvg with torch when the experiment already uses it."""

    from .torch_backend import torch_weighted_average

    return torch_weighted_average(updates, weights, sample_counts=sample_counts, device=device)


def torch_krum(
    updates: list[np.ndarray] | np.ndarray,
    byzantine_count: int,
    *,
    device: str = "auto",
) -> KrumResult:
    """Select a Krum update with torch pairwise distances."""

    from .torch_backend import torch_krum_select

    selected, scores, neighbor_count = torch_krum_select(
        updates,
        byzantine_count=byzantine_count,
        device=device,
    )
    stacked = _stack_updates(updates)
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
    # 模型更新本来就是 float32。若为 Krum 再复制成 float64，100 个 CIFAR-10
    # 客户端会额外占用约 1.4 GB；直接使用 BLAS float32 Gram 矩阵即可保持
    # 相同的平方欧氏距离规则，最终分数仍在上层以 float64 求和。
    updates32 = updates.astype(np.float32, copy=False)
    norms = np.einsum("ij,ij->i", updates32, updates32)[:, None]
    distances = norms + norms.T - 2.0 * (updates32 @ updates32.T)
    return np.maximum(distances, 0.0)


def _streaming_weighted_average(
    updates: list[np.ndarray],
    normalized_weights: np.ndarray,
) -> np.ndarray:
    """以两个参数向量的固定内存完成大模型 FedAvg。"""

    result = np.zeros_like(np.asarray(updates[0], dtype=np.float32))
    scratch = np.empty_like(result)
    for update, weight in zip(updates, normalized_weights):
        np.multiply(update, np.float32(weight), out=scratch, casting="unsafe")
        np.add(result, scratch, out=result)
    return result.astype(np.float32, copy=False)
