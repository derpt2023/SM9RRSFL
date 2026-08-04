"""VERT: vertical historical-gradient prediction for robust FL aggregation.

This is a repository-native implementation of Wang et al.'s VERT baseline.
The paper uses a fixed random projector, retrains a shared three-layer
predictor plus two integration coefficients in every global round, scores users
by cosine similarity, and applies FedAvg uniformly to the selected updates.
Only the final selection policy varies here: a positive ``top_k`` fixes the
paper's Top-k value, ``malicious_ratio_prior`` enables the legacy automatic
known-ratio rule, and the default clusters scores with K-means (K=2) and keeps
the higher-similarity cluster without reading the experiment's malicious ratio.
MNIST uses the paper's fixed dense linear projector.  When that matrix would
exceed 256 MiB, the same fixed-linear-projector role is implemented with a
sparse signed feature hash so CIFAR experiments remain memory bounded.
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
        history_window: int = 10,
        projection_dim: int = 128,
        predict_epochs: int = 5,
        learning_rate: float = 1e-2,
        top_k: int = 0,
        malicious_ratio_prior: float | None = None,
        seed: int = 0,
        eps: float = 1e-12,
    ) -> None:
        if not client_ids:
            raise ValueError("VERT requires at least one client")
        if parameter_size < 1:
            raise ValueError("parameter_size must be at least 1")
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
        if malicious_ratio_prior is not None and not (
            0.0 <= malicious_ratio_prior < 1.0
        ):
            raise ValueError("VERT malicious_ratio_prior must be in [0, 1)")
        if top_k > 0 and malicious_ratio_prior is not None:
            raise ValueError(
                "VERT fixed top_k and malicious_ratio_prior are mutually exclusive"
            )

        self.client_ids = tuple(client_ids)
        self.parameter_size = int(parameter_size)
        self.history_window = int(history_window)
        self.projection_dim = min(int(projection_dim), self.parameter_size)
        self.predict_epochs = int(predict_epochs)
        self.learning_rate = float(learning_rate)
        self.top_k = int(top_k)
        self.malicious_ratio_prior = (
            None
            if malicious_ratio_prior is None
            else float(malicious_ratio_prior)
        )
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

        self._client_history: list[dict[str, np.ndarray]] = []
        self._global_history: list[np.ndarray] = []

    def evaluate_round(
        self,
        update_by_client: dict[str, np.ndarray],
        *,
        round_id: int,
    ) -> VERTResult:
        """Score current updates, then apply the configured selection policy."""

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

        parameters = self._initialize_trainable_parameters(round_id=round_id)
        moment1 = {
            name: np.zeros_like(value) for name, value in parameters.items()
        }
        moment2 = {
            name: np.zeros_like(value) for name, value in parameters.items()
        }
        optimizer_step = 0
        scores: dict[str, float] = {}
        last_global = self._global_history[-1]
        last_clients = self._client_history[-1]
        for client_id in sorted(features):
            current_feature = features[client_id]
            optimizer_step = self._train_predictor(
                client_id,
                parameters,
                moment1,
                moment2,
                optimizer_step,
            )
            last_local = last_clients.get(client_id, last_global)
            integrated = (
                parameters["a"] * last_local
                + parameters["b"] * last_global
            )
            predicted, _cache = self._predict(
                self._project_history_feature(integrated),
                parameters,
            )
            scores[client_id] = self._cosine(
                predicted,
                self._project_history_feature(current_feature),
            )

        ranked = sorted(scores, key=lambda item: (-scores[item], item))
        top_k = self._effective_top_k(
            [scores[client_id] for client_id in ranked]
        )
        selected = tuple(ranked[:top_k])
        rejected = tuple(ranked[top_k:])
        uniform = 1.0 / len(selected)
        weights = {client_id: uniform for client_id in selected}
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

    def _effective_top_k(self, ranked_scores: list[float]) -> int:
        """Choose a fixed, ratio-prior, or predictor-only selection size."""

        active_count = len(ranked_scores)
        if active_count == 0:
            return 0
        if self.top_k:
            return min(self.top_k, active_count)
        if self.malicious_ratio_prior is not None:
            if self.malicious_ratio_prior <= 0.0:
                return active_count
            expected_honest = int(
                math.ceil(
                    (1.0 - self.malicious_ratio_prior) * active_count
                )
            )
            return min(active_count, max(1, expected_honest - 1))
        return self._high_similarity_cluster_size(ranked_scores)

    def _high_similarity_cluster_size(self, ranked_scores: list[float]) -> int:
        """Run deterministic one-dimensional K-means and keep the high cluster.

        Section VI-C3 of the VERT paper removes the Top-k poisoning-rate prior
        by clustering cosine similarities with K=2.  Initializing the two
        centers at the observed extrema makes the otherwise random clustering
        reproducible.  Equal or non-finite scores are treated as an
        uninformative round and all active clients are retained.
        """

        active_count = len(ranked_scores)
        if active_count < 2:
            return active_count

        values = np.asarray(ranked_scores, dtype=np.float64)
        if not np.isfinite(values).all():
            return active_count
        low_center = float(np.min(values))
        high_center = float(np.max(values))
        if high_center - low_center <= self.eps:
            return active_count

        high_cluster = np.zeros(active_count, dtype=bool)
        for _iteration in range(100):
            updated_cluster = (
                np.abs(values - high_center) <= np.abs(values - low_center)
            )
            if updated_cluster.all() or not updated_cluster.any():
                return active_count
            updated_high = float(np.mean(values[updated_cluster]))
            updated_low = float(np.mean(values[~updated_cluster]))
            high_cluster = updated_cluster
            if (
                abs(updated_high - high_center) <= self.eps
                and abs(updated_low - low_center) <= self.eps
            ):
                break
            high_center = updated_high
            low_center = updated_low

        return int(np.count_nonzero(high_cluster))

    def _initialize_trainable_parameters(
        self,
        *,
        round_id: int,
    ) -> dict[str, np.ndarray]:
        """Create the shared predictor and coefficients for one global round."""

        rng = np.random.default_rng(
            self.seed
            + 97_003
            + int(round_id) * 1_000_003
        )
        dimension = self.projection_dim
        coefficient_dimension = (
            self.parameter_size
            if self._dense_projection is not None
            else self.projection_dim
        )
        predictor_bound = 1.0 / math.sqrt(dimension)
        return {
            "w1": rng.uniform(
                -predictor_bound,
                predictor_bound,
                size=(dimension, dimension),
            ).astype(np.float32),
            "b1": rng.uniform(
                -predictor_bound,
                predictor_bound,
                size=dimension,
            ).astype(np.float32),
            "w2": rng.uniform(
                -predictor_bound,
                predictor_bound,
                size=(dimension, dimension),
            ).astype(np.float32),
            "b2": rng.uniform(
                -predictor_bound,
                predictor_bound,
                size=dimension,
            ).astype(np.float32),
            "w3": rng.uniform(
                -predictor_bound,
                predictor_bound,
                size=(dimension, dimension),
            ).astype(np.float32),
            "b3": rng.uniform(
                -predictor_bound,
                predictor_bound,
                size=dimension,
            ).astype(np.float32),
            # The released VERT implementation initializes both element-wise
            # coefficient vectors from its saved all-ones templates each round.
            "a": np.ones(coefficient_dimension, dtype=np.float32),
            "b": np.ones(coefficient_dimension, dtype=np.float32),
        }

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
            return self._linear_project(feature)
        return np.asarray(feature, dtype=np.float32)

    def _train_predictor(
        self,
        client_id: str,
        parameters: dict[str, np.ndarray],
        moment1: dict[str, np.ndarray],
        moment2: dict[str, np.ndarray],
        step: int,
    ) -> int:
        transition_count = len(self._global_history) - 1
        if transition_count < 1:
            return step
        first_transition = max(0, transition_count - self.history_window + 1)
        for _epoch in range(self.predict_epochs):
            gradients = {
                name: np.zeros_like(value) for name, value in parameters.items()
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
                    parameters,
                    gradients,
                )
                samples += 1
            if not samples:
                continue
            step += 1
            self._adam_step(parameters, gradients, moment1, moment2, step)
        return step

    def _accumulate_training_gradient(
        self,
        local: np.ndarray,
        global_feature: np.ndarray,
        target: np.ndarray,
        parameters: dict[str, np.ndarray],
        gradients: dict[str, np.ndarray],
    ) -> None:
        integrated = (
            parameters["a"] * local
            + parameters["b"] * global_feature
        )
        predictor_input = self._project_history_feature(integrated)
        predicted, cache = self._predict(predictor_input, parameters)
        difference = predicted - target
        norm = max(float(np.linalg.norm(difference)), self.eps)
        output_gradient = difference / norm
        input_gradient, parameter_gradients = self._predict_backward(
            output_gradient,
            cache,
            parameters,
        )
        for name, value in parameter_gradients.items():
            gradients[name] += value
        integration_gradient = (
            self._dense_projection.T @ input_gradient
            if self._dense_projection is not None
            else input_gradient
        )
        gradients["a"] += integration_gradient * local
        gradients["b"] += integration_gradient * global_feature

    def _predict(
        self,
        value: np.ndarray,
        parameters: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        w1, b1 = parameters["w1"], parameters["b1"]
        w2, b2 = parameters["w2"], parameters["b2"]
        w3, b3 = parameters["w3"], parameters["b3"]
        pre1 = w1 @ value + b1
        hidden1 = np.maximum(pre1, 0.0)
        pre2 = w2 @ hidden1 + b2
        hidden2 = np.maximum(pre2, 0.0)
        pre3 = w3 @ hidden2 + b3
        return pre3, (value, pre1, hidden1, pre2, hidden2)

    def _predict_backward(
        self,
        output_gradient: np.ndarray,
        cache: tuple[np.ndarray, ...],
        parameters: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        value, pre1, hidden1, pre2, hidden2 = cache
        pre3_gradient = output_gradient
        gradients = {
            "w3": np.outer(pre3_gradient, hidden2).astype(np.float32),
            "b3": pre3_gradient.astype(np.float32),
        }
        hidden2_gradient = parameters["w3"].T @ pre3_gradient
        pre2_gradient = hidden2_gradient * (pre2 > 0.0)
        gradients["w2"] = np.outer(pre2_gradient, hidden1).astype(np.float32)
        gradients["b2"] = pre2_gradient.astype(np.float32)
        hidden1_gradient = parameters["w2"].T @ pre2_gradient
        pre1_gradient = hidden1_gradient * (pre1 > 0.0)
        gradients["w1"] = np.outer(pre1_gradient, value).astype(np.float32)
        gradients["b1"] = pre1_gradient.astype(np.float32)
        input_gradient = parameters["w1"].T @ pre1_gradient
        return input_gradient.astype(np.float32), gradients

    def _adam_step(
        self,
        parameters: dict[str, np.ndarray],
        gradients: dict[str, np.ndarray],
        moment1: dict[str, np.ndarray],
        moment2: dict[str, np.ndarray],
        step: int,
    ) -> None:
        beta1, beta2 = 0.9, 0.999
        for name, parameter in parameters.items():
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

    def _cosine(self, left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= self.eps:
            return 0.0
        return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


__all__ = ["VERTDefense", "VERTResult"]
