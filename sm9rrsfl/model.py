"""Small NumPy softmax model used by the federated MNIST experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


INPUT_DIM = 28 * 28
NUM_CLASSES = 10


@dataclass(frozen=True)
class TrainStats:
    loss: float
    samples: int


def parameter_size(input_dim: int = INPUT_DIM, num_classes: int = NUM_CLASSES) -> int:
    return input_dim * num_classes + num_classes


def init_params(
    *,
    seed: int = 0,
    input_dim: int = INPUT_DIM,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(input_dim, num_classes)).astype(np.float32)
    bias = np.zeros(num_classes, dtype=np.float32)
    return params_to_vector(weights, bias)


def params_to_vector(weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.concatenate([weights.reshape(-1), bias.reshape(-1)]).astype(np.float32)


def vector_to_params(
    vector: np.ndarray,
    *,
    input_dim: int = INPUT_DIM,
    num_classes: int = NUM_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    expected = parameter_size(input_dim, num_classes)
    if vector.shape[0] != expected:
        raise ValueError(f"expected parameter vector of length {expected}, got {vector.shape[0]}")
    split = input_dim * num_classes
    weights = vector[:split].reshape(input_dim, num_classes)
    bias = vector[split:]
    return weights, bias


def predict(vector: np.ndarray, x: np.ndarray) -> np.ndarray:
    weights, bias = vector_to_params(vector, input_dim=x.shape[1], num_classes=NUM_CLASSES)
    logits = x @ weights + bias
    return np.argmax(logits, axis=1)


def accuracy(vector: np.ndarray, x: np.ndarray, y: np.ndarray, *, batch_size: int = 2048) -> float:
    if len(y) == 0:
        return 0.0
    correct = 0
    for start in range(0, len(y), batch_size):
        end = start + batch_size
        pred = predict(vector, x[start:end])
        correct += int(np.sum(pred == y[start:end]))
    return correct / len(y)


def local_train_delta(
    global_vector: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.3,
    epochs: int = 1,
    batch_size: int = 32,
    seed: int = 0,
) -> tuple[np.ndarray, TrainStats]:
    """Train a local copy for a few epochs and return its model delta."""

    if len(y) == 0:
        return np.zeros_like(global_vector), TrainStats(loss=0.0, samples=0)

    rng = np.random.default_rng(seed)
    local = global_vector.astype(np.float32, copy=True)
    weights, bias = vector_to_params(local, input_dim=x.shape[1], num_classes=NUM_CLASSES)
    losses: list[float] = []

    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            xb = x[batch_idx]
            yb = y[batch_idx]
            probs = _softmax(xb @ weights + bias)
            losses.append(_cross_entropy(probs, yb))
            probs[np.arange(len(yb)), yb] -= 1.0
            probs /= len(yb)
            grad_w = xb.T @ probs
            grad_b = np.sum(probs, axis=0)
            weights -= lr * grad_w.astype(np.float32)
            bias -= lr * grad_b.astype(np.float32)

    updated = params_to_vector(weights, bias)
    delta = (updated - global_vector).astype(np.float32)
    return delta, TrainStats(loss=float(np.mean(losses)), samples=len(y))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))
