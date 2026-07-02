"""Dynamic client weights for suspected poisoning clients."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeightUpdateResult:
    weights: dict[str, float]
    suspicious_clients: set[str]
    newly_removed: set[str]
    true_positive_removed: int
    false_positive_removed: int


class SuspicionWeightManager:
    """维护疑似客户端的惩罚、恢复和连续异常移除状态。"""

    def __init__(
        self,
        client_ids: list[str],
        *,
        penalty_factor: float = 0.5,
        recovery_factor: float = 2.0,
        remove_after: int = 3,
    ) -> None:
        if not client_ids:
            raise ValueError("client_ids must not be empty")
        if not 0.0 < penalty_factor < 1.0:
            raise ValueError("penalty_factor must be in (0, 1)")
        if recovery_factor <= 1.0:
            raise ValueError("recovery_factor must be greater than 1")
        if remove_after < 0:
            raise ValueError("remove_after must be non-negative")
        self.client_ids = tuple(client_ids)
        self.base_weight = 1.0 / len(client_ids)
        self.penalty_factor = penalty_factor
        self.recovery_factor = recovery_factor
        self.remove_after = remove_after
        self.weights = {identity: self.base_weight for identity in client_ids}
        self.consecutive_suspicions = {identity: 0 for identity in client_ids}
        self.removed: set[str] = set()

    def update(
        self,
        active_ids: list[str],
        suspicious_ids: set[str],
        malicious_ids: set[str],
    ) -> WeightUpdateResult:
        newly_removed: set[str] = set()
        active = [identity for identity in active_ids if identity not in self.removed]

        for identity in active:
            # 单次异常只降权；只有连续异常达到阈值后才永久移出聚合集合。
            if identity in suspicious_ids:
                self.weights[identity] *= self.penalty_factor
                self.consecutive_suspicions[identity] += 1
            else:
                if self.weights[identity] < self.base_weight:
                    self.weights[identity] = min(
                        self.base_weight,
                        self.weights[identity] * self.recovery_factor,
                    )
                self.consecutive_suspicions[identity] = 0

            if self.remove_after and self.consecutive_suspicions[identity] >= self.remove_after:
                self.weights[identity] = 0.0
                newly_removed.add(identity)
                self.removed.add(identity)

        self._renormalize()
        true_positive = sum(1 for identity in newly_removed if identity in malicious_ids)
        false_positive = len(newly_removed) - true_positive
        return WeightUpdateResult(
            weights=dict(self.weights),
            suspicious_clients=set(suspicious_ids),
            newly_removed=newly_removed,
            true_positive_removed=true_positive,
            false_positive_removed=false_positive,
        )

    def _renormalize(self) -> None:
        active = [identity for identity in self.client_ids if identity not in self.removed]
        total = sum(self.weights[identity] for identity in active)
        if total <= 0.0:
            reset = 1.0 / max(1, len(active))
            for identity in active:
                self.weights[identity] = reset
        else:
            for identity in active:
                self.weights[identity] /= total
        for identity in self.removed:
            self.weights[identity] = 0.0
