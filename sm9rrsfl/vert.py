"""VERT: vertical historical-gradient prediction for robust FL aggregation.

This is a repository-native implementation of Wang et al.'s VERT baseline.
The original implementation keeps a fixed random projector, trains a shared
three-layer predictor plus two integration coefficients with Adam, scores
clients by cosine similarity, and aggregates the top-k updates by normalized
similarity.  MNIST uses the paper's fixed dense linear projector.  When that
matrix would exceed 256 MiB, the same fixed-linear-projector role is implemented
with a sparse signed feature hash so CIFAR experiments remain memory bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class VERTResult:
    """One round of VERT selection."""

    selected_clients: tuple[str, ...]
    rejected_clients: tuple[str, ...]
    weights: dict[str, float]
    scores: dict[str, float]


class VERTDefense:
    """Predict projected client updates from their vertical history."""

    def __init__(
        self,
        client_ids: list[str],
        *,
        parameter_size: int,
        malicious_ratio: float,
        history_window: int = 10,
        projection_dim: int = 128,
        predict_epochs: int = 5,
        learning_rate: float = 1e-3,
        top_k: int = 0,
        seed: int = 0,
        eps: float = 1e-12,
    ) -> None:
        if not client_ids:
            raise ValueError("VERT requires at least one client")
        if parameter_size < 1:
            raise ValueError("parameter_size must be at least 1")
        if not 0.0 <= malicious_ratio < 1.0:
            raise ValueError("malicious_ratio must be in [0, 1)")
        if history_window < 2:
            raise ValueError("VERT history_window must be at least 2")
        if projection_dim < 2:
            raise ValueError("VERT projection_dim must be at least 2")
        if predict_epochs < 1:
            raise ValueError("VERT predict_epochs must be at least 1")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("VERT learning_rate must be finite and positive")
        if top_k < 0:
            raise ValueError("VERT top_k must be non-negative")

        self.client_ids = tuple(client_ids)
        self.parameter_size = int(parameter_size)
        self.malicious_ratio = float(malicious_ratio)
        self.history_window = int(history_window)
        self.projection_dim = min(int(projection_dim), self.parameter_size)
        self.predict_epochs = int(predict_epochs)
        self.learning_rate = float(learning_rate)
        self.top_k = int(top_k)
        self.seed = int(seed)
        self.eps = float(eps)

        rng = np.random.default_rng(self.seed + 71_003)
        dense_elements = self.parameter_size * self.projection_dim
        if dense_elements * np.dtype(np.float32).itemsize <= 256 * 1024 * 1024:
            projector_bound = 1.0 / math.sqrt(self.parameter_size)
            self._dense_projection = rng.uniform(
                -projector_bound,
                projector_bound,
                size=(self.projection_dim, self.parameter_size),
            ).astype(np.float32)
            self._projection_bias = rng.uniform(
                -projector_bound,
                projector_bound,
                size=self.projection_dim,
            ).astype(np.float32)
            self._projection_bucket = None
            self._projection_sign = None
            self._projection_scale = np.float32(1.0)
        else:
            self._dense_projection = None
            self._projection_bias = np.zeros(
                self.projection_dim,
                dtype=np.float32,
            )
            self._projection_bucket = rng.integers(
                0,
                self.projection_dim,
                size=self.parameter_size,
                dtype=np.int32,
            )
            self._projection_sign = rng.choice(
                np.asarray([-1.0, 1.0], dtype=np.float32),
                size=self.parameter_size,
            )
            self._projection_scale = np.float32(
                math.sqrt(self.projection_dim / self.parameter_size)
            )

        dimension = self.projection_dim
        coefficient_dimension = (
            self.parameter_size
            if self._dense_projection is not None
            else self.projection_dim
        )
        bound = 1.0 / math.sqrt(dimension)
        self._parameters = {
            "w1": rng.uniform(-bound, bound, size=(dimension, dimension)).astype(
                np.float32
            ),
            "b1": rng.uniform(-bound, bound, size=dimension).astype(np.float32),
            "w2": rng.uniform(-bound, bound, size=(dimension, dimension)).astype(
                np.float32
            ),
            "b2": rng.uniform(-bound, bound, size=dimension).astype(np.float32),
            "w3": rng.uniform(-bound, bound, size=(dimension, dimension)).astype(
                np.float32
            ),
            "b3": rng.uniform(-bound, bound, size=dimension).astype(np.float32),
            # The official code initializes both element-wise coefficients at
            # zero and optimizes them together with the predictor.
            "a": np.zeros(coefficient_dimension, dtype=np.float32),
            "b": np.zeros(coefficient_dimension, dtype=np.float32),
        }
        self._client_history: list[dict[str, np.ndarray]] = []
        self._global_history: list[np.ndarray] = []

    def evaluate_round(
        self,
        update_by_client: dict[str, np.ndarray],
        *,
        round_id: int,
    ) -> VERTResult:
        """Score current updates and return the paper's top-k selection."""

        del round_id
        if not update_by_client:
            return VERTResult(tuple(), tuple(), {}, {})
        features = {
            client_id: self._history_feature(update)
            for client_id, update in update_by_client.items()
        }

        # VERT uses two benign history rounds before predictor-based selection.
        if len(self._global_history) < 2:
            selected = tuple(sorted(features))
            uniform = 1.0 / len(selected)
            return VERTResult(
                selected_clients=selected,
                rejected_clients=tuple(),
                weights={client_id: uniform for client_id in selected},
                scores={client_id: 1.0 for client_id in selected},
            )

        self._train_predictor(tuple(sorted(features)))
        scores: dict[str, float] = {}
        last_global = self._global_history[-1]
        last_clients = self._client_history[-1]
        for client_id, current_feature in features.items():
            last_local = last_clients.get(client_id, last_global)
            integrated = (
                self._parameters["a"] * last_local
                + self._parameters["b"] * last_global
            )
            predicted, _cache = self._predict(
                self._project_history_feature(integrated)
            )
            scores[client_id] = self._cosine(
                predicted,
                self._project_history_feature(current_feature),
            )

        top_k = self._effective_top_k(len(scores))
        ranked = sorted(scores, key=lambda item: (-scores[item], item))
        selected = tuple(ranked[:top_k])
        rejected = tuple(ranked[top_k:])
        positive = np.asarray(
            [max(scores[client_id], 0.0) for client_id in selected],
            dtype=np.float64,
        )
        total = float(np.sum(positive))
        if total <= self.eps:
            positive.fill(1.0 / len(selected))
        else:
            positive /= total
        weights = {
            client_id: float(weight)
            for client_id, weight in zip(selected, positive)
        }
        return VERTResult(selected, rejected, weights, scores)

    def finalize_round(
        self,
        update_by_client: dict[str, np.ndarray],
        aggregate: np.ndarray,
        result: VERTResult,
    ) -> None:
        """Store the sanitized vertical history after aggregation."""

        global_feature = self._history_feature(aggregate)
        selected = set(result.selected_clients)
        current_history = {}
        for client_id, update in update_by_client.items():
            current_history[client_id] = (
                self._history_feature(update)
                if client_id in selected
                else global_feature.copy()
            )
        if len(self._global_history) >= self.history_window:
            self._global_history.pop(0)
            self._client_history.pop(0)
        self._global_history.append(global_feature)
        self._client_history.append(current_history)

    def _effective_top_k(self, active_count: int) -> int:
        if self.top_k:
            return min(self.top_k, active_count)
        if self.malicious_ratio <= 0.0:
            return active_count
        # The VERT experiments use k=15 for 80 selected clients at 80%
        # poisoning and k=8 for 90 selected clients at 90% poisoning: one
        # fewer than the expected number of honest selected clients.
        expected_honest = int(math.ceil((1.0 - self.malicious_ratio) * active_count))
        return min(active_count, max(1, expected_honest - 1))

    def _history_feature(self, update: np.ndarray) -> np.ndarray:
        vector = np.asarray(update, dtype=np.float32).reshape(-1)
        if vector.size != self.parameter_size:
            raise ValueError("VERT update size does not match parameter_size")
        if not np.isfinite(vector).all():
            raise ValueError("VERT update contains NaN or infinity")
        if self._dense_projection is not None:
            # Exact MNIST path: coefficient matrices and history remain in the
            # original update dimension, as in the paper and official code.
            return vector.copy()
        return self._linear_project(vector)

    def _linear_project(self, vector: np.ndarray) -> np.ndarray:
        if self._dense_projection is not None:
            return (
                self._dense_projection @ vector + self._projection_bias
            ).astype(np.float32)
        assert self._projection_bucket is not None
        assert self._projection_sign is not None
        projected = np.bincount(
            self._projection_bucket,
            weights=vector * self._projection_sign,
            minlength=self.projection_dim,
        ).astype(np.float32)
        projected *= self._projection_scale
        projected += self._projection_bias
        return projected

    def _project_history_feature(self, feature: np.ndarray) -> np.ndarray:
        if self._dense_projection is not None:
            return self._softmax(self._linear_project(feature))
        return self._softmax(feature)

    def _train_predictor(self, active_clients: tuple[str, ...]) -> None:
        transition_count = len(self._global_history) - 1
        if transition_count < 1:
            return
        first_transition = max(0, transition_count - self.history_window + 1)
        moment1 = {
            name: np.zeros_like(value) for name, value in self._parameters.items()
        }
        moment2 = {
            name: np.zeros_like(value) for name, value in self._parameters.items()
        }
        step = 0
        for client_id in active_clients:
            for _epoch in range(self.predict_epochs):
                gradients = {
                    name: np.zeros_like(value)
                    for name, value in self._parameters.items()
                }
                samples = 0
                for index in range(first_transition, transition_count):
                    local = self._client_history[index].get(
                        client_id,
                        self._global_history[index],
                    )
                    global_feature = self._global_history[index]
                    target = self._client_history[index + 1].get(
                        client_id,
                        self._global_history[index + 1],
                    )
                    self._accumulate_training_gradient(
                        local,
                        global_feature,
                        self._project_history_feature(target),
                        gradients,
                    )
                    samples += 1
                if not samples:
                    continue
                for gradient in gradients.values():
                    gradient /= samples
                    np.clip(gradient, -10.0, 10.0, out=gradient)
                step += 1
                self._adam_step(gradients, moment1, moment2, step)

    def _accumulate_training_gradient(
        self,
        local: np.ndarray,
        global_feature: np.ndarray,
        target: np.ndarray,
        gradients: dict[str, np.ndarray],
    ) -> None:
        integrated = (
            self._parameters["a"] * local
            + self._parameters["b"] * global_feature
        )
        predictor_input = self._project_history_feature(integrated)
        predicted, cache = self._predict(predictor_input)
        difference = predicted - target
        norm = max(float(np.linalg.norm(difference)), self.eps)
        output_gradient = difference / norm
        input_gradient, parameter_gradients = self._predict_backward(
            output_gradient,
            cache,
        )
        for name, value in parameter_gradients.items():
            gradients[name] += value
        projected_gradient = self._softmax_backward(
            predictor_input,
            input_gradient,
        )
        integration_gradient = (
            self._dense_projection.T @ projected_gradient
            if self._dense_projection is not None
            else projected_gradient
        )
        gradients["a"] += integration_gradient * local
        gradients["b"] += integration_gradient * global_feature

    def _predict(
        self,
        value: np.ndarray,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        w1, b1 = self._parameters["w1"], self._parameters["b1"]
        w2, b2 = self._parameters["w2"], self._parameters["b2"]
        w3, b3 = self._parameters["w3"], self._parameters["b3"]
        pre1 = w1 @ value + b1
        hidden1 = np.maximum(pre1, 0.0)
        pre2 = w2 @ hidden1 + b2
        hidden2 = np.maximum(pre2, 0.0)
        output = self._softmax(w3 @ hidden2 + b3)
        return output, (value, pre1, hidden1, pre2, hidden2, output)

    def _predict_backward(
        self,
        output_gradient: np.ndarray,
        cache: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        value, pre1, hidden1, pre2, hidden2, output = cache
        logits_gradient = self._softmax_backward(output, output_gradient)
        gradients = {
            "w3": np.outer(logits_gradient, hidden2).astype(np.float32),
            "b3": logits_gradient.astype(np.float32),
        }
        hidden2_gradient = self._parameters["w3"].T @ logits_gradient
        pre2_gradient = hidden2_gradient * (pre2 > 0.0)
        gradients["w2"] = np.outer(pre2_gradient, hidden1).astype(np.float32)
        gradients["b2"] = pre2_gradient.astype(np.float32)
        hidden1_gradient = self._parameters["w2"].T @ pre2_gradient
        pre1_gradient = hidden1_gradient * (pre1 > 0.0)
        gradients["w1"] = np.outer(pre1_gradient, value).astype(np.float32)
        gradients["b1"] = pre1_gradient.astype(np.float32)
        input_gradient = self._parameters["w1"].T @ pre1_gradient
        return input_gradient.astype(np.float32), gradients

    def _adam_step(
        self,
        gradients: dict[str, np.ndarray],
        moment1: dict[str, np.ndarray],
        moment2: dict[str, np.ndarray],
        step: int,
    ) -> None:
        beta1, beta2 = 0.9, 0.999
        for name, parameter in self._parameters.items():
            gradient = gradients[name]
            moment1[name] *= beta1
            moment1[name] += (1.0 - beta1) * gradient
            moment2[name] *= beta2
            moment2[name] += (1.0 - beta2) * np.square(gradient)
            corrected1 = moment1[name] / (1.0 - beta1**step)
            corrected2 = moment2[name] / (1.0 - beta2**step)
            parameter -= self.learning_rate * corrected1 / (
                np.sqrt(corrected2) + 1e-8
            )

    @staticmethod
    def _softmax(value: np.ndarray) -> np.ndarray:
        shifted = np.asarray(value, dtype=np.float32) - float(np.max(value))
        exponential = np.exp(shifted).astype(np.float32)
        return exponential / max(float(np.sum(exponential)), 1e-12)

    @staticmethod
    def _softmax_backward(output: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        return output * (gradient - float(np.dot(gradient, output)))

    def _cosine(self, left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= self.eps:
            return 0.0
        return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


__all__ = ["VERTDefense", "VERTResult"]
