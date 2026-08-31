"""Direction Alignment Inspection (AlignIns) aggregation defense.

This module implements Algorithm 1 from Xu, Zhang, and Hu, CVPR 2025.
The detector consumes only the current global parameter vector and submitted
client updates.  Experiment-only malicious-client labels are deliberately not
part of its interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class AlignInsResult:
    """Per-round AlignIns decisions and exact aggregation coefficients.

    ``aggregation_coefficients`` already include both median-norm clipping and
    the paper's ``1 / |S|`` equal-client factor.  They generally do *not* sum
    to one and must therefore never be normalized again by an aggregation
    helper.
    """

    selected_clients: tuple[str, ...]
    rejected_clients: tuple[str, ...]
    tda_scores: dict[str, float]
    mpsa_scores: dict[str, float]
    tda_mz_scores: dict[str, float]
    mpsa_mz_scores: dict[str, float]
    update_norms: dict[str, float]
    clip_norm: float
    clip_factors: dict[str, float]
    aggregation_coefficients: dict[str, float]


class AlignInsDefense:
    """Stateless implementation of the AlignIns filtering rule.

    Parameters use project-level names so configuration, CLI, manifests, and
    tuning grids can share one spelling without compatibility aliases.
    """

    def __init__(
        self,
        *,
        alignins_sparsity: float = 0.3,
        alignins_tda_radius: float = 1.0,
        alignins_mpsa_radius: float = 1.0,
    ) -> None:
        if (
            not np.isfinite(alignins_sparsity)
            or not 0.0 < alignins_sparsity <= 1.0
        ):
            raise ValueError("alignins_sparsity must be in (0, 1]")
        for name, value in {
            "alignins_tda_radius": alignins_tda_radius,
            "alignins_mpsa_radius": alignins_mpsa_radius,
        }.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        self.alignins_sparsity = float(alignins_sparsity)
        self.alignins_tda_radius = float(alignins_tda_radius)
        self.alignins_mpsa_radius = float(alignins_mpsa_radius)

    def evaluate_round(
        self,
        global_params: Any,
        updates_by_client: Mapping[str, Any],
        *,
        round_id: int | None = None,
    ) -> AlignInsResult:
        """Inspect one round without consulting attack labels or ratio priors."""

        del round_id  # AlignIns has no longitudinal or round-specific state.
        client_ids, updates = _validated_updates(updates_by_client)
        torch = _torch_module_for(updates[0])
        if torch is None:
            tda_values, mpsa_values, update_norms = _numpy_alignment_scores(
                global_params,
                updates,
                sparsity=self.alignins_sparsity,
            )
        else:
            tda_values, mpsa_values, update_norms = _torch_alignment_scores(
                torch,
                global_params,
                updates,
                sparsity=self.alignins_sparsity,
            )

        tda_mz_values = _absolute_median_z_scores(tda_values)
        mpsa_mz_values = _absolute_median_z_scores(mpsa_values)
        selected_mask = (
            (tda_mz_values < self.alignins_tda_radius)
            & (mpsa_mz_values < self.alignins_mpsa_radius)
        )
        selected_clients = tuple(
            client_id
            for client_id, selected in zip(client_ids, selected_mask)
            if bool(selected)
        )
        rejected_clients = tuple(
            client_id
            for client_id, selected in zip(client_ids, selected_mask)
            if not bool(selected)
        )

        selected_norms = update_norms[selected_mask]
        clip_norm = (
            float(np.median(selected_norms)) if selected_norms.size else 0.0
        )
        clip_factors: dict[str, float] = {}
        aggregation_coefficients: dict[str, float] = {}
        selected_count = len(selected_clients)
        for client_id, selected, update_norm in zip(
            client_ids,
            selected_mask,
            update_norms,
        ):
            if not bool(selected):
                factor = 0.0
            elif float(update_norm) == 0.0:
                # A zero update is unchanged by clipping and contributes zero
                # regardless of its nominal coefficient.
                factor = 1.0
            else:
                factor = min(1.0, clip_norm / float(update_norm))
            clip_factors[client_id] = float(factor)
            aggregation_coefficients[client_id] = (
                float(factor) / selected_count if selected_count else 0.0
            )

        return AlignInsResult(
            selected_clients=selected_clients,
            rejected_clients=rejected_clients,
            tda_scores=_values_by_client(client_ids, tda_values),
            mpsa_scores=_values_by_client(client_ids, mpsa_values),
            tda_mz_scores=_values_by_client(client_ids, tda_mz_values),
            mpsa_mz_scores=_values_by_client(client_ids, mpsa_mz_values),
            update_norms=_values_by_client(client_ids, update_norms),
            clip_norm=clip_norm,
            clip_factors=clip_factors,
            aggregation_coefficients=aggregation_coefficients,
        )


def aggregate_with_coefficients(
    updates_by_client: Mapping[str, Any],
    aggregation_coefficients: Mapping[str, float],
) -> Any:
    """Apply AlignIns coefficients without re-normalizing them.

    The implementation is streaming for both NumPy and torch tensors, so it
    does not create an additional ``num_clients x num_parameters`` stack.
    """

    client_ids, updates = _validated_updates(updates_by_client)
    if set(aggregation_coefficients) != set(client_ids):
        raise ValueError(
            "aggregation_coefficients must contain every submitted client exactly once"
        )
    coefficients = []
    for client_id in client_ids:
        coefficient = float(aggregation_coefficients[client_id])
        if not np.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError(
                "aggregation coefficients must be finite and non-negative"
            )
        coefficients.append(coefficient)

    torch = _torch_module_for(updates[0])
    if torch is not None:
        with torch.no_grad():
            aggregate = torch.zeros_like(updates[0])
            for update, coefficient in zip(updates, coefficients):
                aggregate.add_(update, alpha=coefficient)
        return aggregate

    aggregate = np.zeros_like(updates[0], dtype=np.float32)
    scratch = np.empty_like(aggregate)
    for update, coefficient in zip(updates, coefficients):
        np.multiply(
            update,
            np.float32(coefficient),
            out=scratch,
            casting="unsafe",
        )
        np.add(aggregate, scratch, out=aggregate)
    return aggregate


def _validated_updates(
    updates_by_client: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    if not isinstance(updates_by_client, Mapping) or not updates_by_client:
        raise ValueError("updates_by_client must contain at least one update")
    client_ids = tuple(updates_by_client)
    if any(not isinstance(client_id, str) or not client_id for client_id in client_ids):
        raise ValueError("client identifiers must be non-empty strings")
    updates = tuple(updates_by_client[client_id] for client_id in client_ids)
    torch = _torch_module_for(updates[0])

    if torch is not None:
        first = updates[0]
        if first.ndim != 1 or first.numel() == 0:
            raise ValueError("updates must be non-empty one-dimensional vectors")
        if not torch.is_floating_point(first):
            raise TypeError("torch updates must use a floating-point dtype")
        expected_shape = tuple(first.shape)
        expected_device = first.device
        expected_dtype = first.dtype
        for update in updates:
            if not torch.is_tensor(update):
                raise TypeError("all updates must use the same array backend")
            if tuple(update.shape) != expected_shape:
                raise ValueError("all updates must have the same shape")
            if update.device != expected_device:
                raise ValueError("all torch updates must be on the same device")
            if update.dtype != expected_dtype:
                raise ValueError("all torch updates must use the same dtype")
            if not bool(torch.isfinite(update).all().detach().cpu().item()):
                raise ValueError("updates must contain only finite values")
        return client_ids, updates

    normalized = []
    expected_shape: tuple[int, ...] | None = None
    for update in updates:
        if _torch_module_for(update) is not None:
            raise TypeError("all updates must use the same array backend")
        array = np.asarray(update, dtype=np.float32)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("updates must be non-empty one-dimensional vectors")
        if expected_shape is None:
            expected_shape = array.shape
        elif array.shape != expected_shape:
            raise ValueError("all updates must have the same shape")
        if not np.all(np.isfinite(array)):
            raise ValueError("updates must contain only finite values")
        normalized.append(array)
    return client_ids, tuple(normalized)


def _numpy_alignment_scores(
    global_params: Any,
    updates: tuple[np.ndarray, ...],
    *,
    sparsity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = updates[0]
    global_vector = np.asarray(global_params, dtype=np.float32)
    if global_vector.shape != first.shape:
        raise ValueError("global_params must have the same shape as each update")
    if not np.all(np.isfinite(global_vector)):
        raise ValueError("global_params must contain only finite values")

    principal_votes = np.zeros_like(first, dtype=np.float32)
    for update in updates:
        np.add(principal_votes, np.sign(update), out=principal_votes)
    principal_sign = np.sign(principal_votes)
    top_count = _top_count(first.size, sparsity)
    global_norm = float(np.linalg.norm(global_vector))

    tda_values = []
    mpsa_values = []
    update_norms = []
    for update in updates:
        update_norm = float(np.linalg.norm(update))
        update_norms.append(update_norm)
        if update_norm == 0.0 or global_norm == 0.0:
            tda = 0.0
        else:
            tda = float(np.dot(update, global_vector)) / (
                update_norm * global_norm
            )
            tda = min(1.0, max(-1.0, tda))
        top_indices = np.argpartition(
            np.abs(update),
            update.size - top_count,
        )[-top_count:]
        mpsa = float(
            np.mean(np.sign(update[top_indices]) == principal_sign[top_indices])
        )
        tda_values.append(tda)
        mpsa_values.append(mpsa)

    return (
        np.asarray(tda_values, dtype=np.float64),
        np.asarray(mpsa_values, dtype=np.float64),
        np.asarray(update_norms, dtype=np.float64),
    )


def _torch_alignment_scores(
    torch,
    global_params: Any,
    updates: tuple[Any, ...],
    *,
    sparsity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = updates[0]
    global_vector = torch.as_tensor(
        global_params,
        dtype=first.dtype,
        device=first.device,
    )
    if tuple(global_vector.shape) != tuple(first.shape):
        raise ValueError("global_params must have the same shape as each update")
    if not bool(torch.isfinite(global_vector).all().detach().cpu().item()):
        raise ValueError("global_params must contain only finite values")

    with torch.no_grad():
        principal_votes = torch.zeros_like(first)
        for update in updates:
            principal_votes.add_(torch.sign(update))
        principal_sign = torch.sign(principal_votes)
        top_count = _top_count(first.numel(), sparsity)
        global_norm = torch.linalg.vector_norm(global_vector)
        global_norm_value = float(global_norm.detach().cpu().item())

        tda_values = []
        mpsa_values = []
        update_norms = []
        for update in updates:
            update_norm = torch.linalg.vector_norm(update)
            update_norm_value = float(update_norm.detach().cpu().item())
            update_norms.append(update_norm_value)
            if update_norm_value == 0.0 or global_norm_value == 0.0:
                tda = 0.0
            else:
                tda_tensor = torch.dot(update, global_vector).div(
                    update_norm * global_norm
                )
                tda = float(
                    torch.clamp(tda_tensor, -1.0, 1.0).detach().cpu().item()
                )
            top_indices = torch.topk(
                torch.abs(update),
                k=top_count,
                largest=True,
                sorted=False,
            ).indices
            mpsa = float(
                (
                    torch.sign(update.index_select(0, top_indices))
                    == principal_sign.index_select(0, top_indices)
                )
                .to(dtype=torch.float32)
                .mean()
                .detach()
                .cpu()
                .item()
            )
            tda_values.append(tda)
            mpsa_values.append(mpsa)

    return (
        np.asarray(tda_values, dtype=np.float64),
        np.asarray(mpsa_values, dtype=np.float64),
        np.asarray(update_norms, dtype=np.float64),
    )


def _absolute_median_z_scores(values: np.ndarray) -> np.ndarray:
    """Return ``abs((x - median(x)) / population_std(x))`` safely."""

    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    standard_deviation = float(np.std(values, ddof=0))
    scale = max(1.0, float(np.max(np.abs(values))))
    if standard_deviation <= np.finfo(np.float64).eps * scale:
        return np.zeros_like(values)
    return np.abs((values - median) / standard_deviation)


def _top_count(parameter_count: int, sparsity: float) -> int:
    return max(1, min(parameter_count, int(parameter_count * sparsity)))


def _values_by_client(
    client_ids: tuple[str, ...],
    values: np.ndarray,
) -> dict[str, float]:
    return {
        client_id: float(value)
        for client_id, value in zip(client_ids, values)
    }


def _torch_module_for(value: Any):
    module = type(value).__module__
    if not (module == "torch" or module.startswith("torch.")):
        return None
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - inconsistent environment
        raise RuntimeError("torch tensor supplied but torch is unavailable") from exc
    return torch if torch.is_tensor(value) else None


__all__ = [
    "AlignInsDefense",
    "AlignInsResult",
    "aggregate_with_coefficients",
]
