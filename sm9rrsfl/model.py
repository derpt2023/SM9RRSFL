"""Small NumPy CNN model used by the federated image experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MNIST_INPUT_SHAPE = (1, 28, 28)
INPUT_DIM = 28 * 28
NUM_CLASSES = 10


@dataclass(frozen=True)
class TrainStats:
    loss: float
    samples: int


@dataclass(frozen=True)
class ModelSpec:
    input_shape: tuple[int, int, int] = MNIST_INPUT_SHAPE
    num_classes: int = NUM_CLASSES
    conv_filters: int = 8
    kernel_size: int = 3
    conv_stride: int = 2
    pool_size: int = 2

    @property
    def padding(self) -> int:
        return self.kernel_size // 2

    @property
    def conv_output_shape(self) -> tuple[int, int, int]:
        _, height, width = self.input_shape
        out_h = (height + 2 * self.padding - self.kernel_size) // self.conv_stride + 1
        out_w = (width + 2 * self.padding - self.kernel_size) // self.conv_stride + 1
        return self.conv_filters, out_h, out_w

    @property
    def pooled_shape(self) -> tuple[int, int, int]:
        filters, height, width = self.conv_output_shape
        return filters, height // self.pool_size, width // self.pool_size

    @property
    def feature_dim(self) -> int:
        filters, height, width = self.pooled_shape
        return filters * height * width

    @property
    def conv_weight_size(self) -> int:
        channels, _, _ = self.input_shape
        return self.conv_filters * channels * self.kernel_size * self.kernel_size

    @property
    def conv_bias_offset(self) -> int:
        return self.conv_weight_size

    @property
    def dense_weight_offset(self) -> int:
        return self.conv_bias_offset + self.conv_filters

    @property
    def dense_weight_size(self) -> int:
        return self.feature_dim * self.num_classes

    @property
    def dense_bias_offset(self) -> int:
        return self.dense_weight_offset + self.dense_weight_size

    @property
    def parameter_size(self) -> int:
        return self.dense_bias_offset + self.num_classes

    @property
    def svd_matrix_offset(self) -> int:
        return self.dense_weight_offset

    @property
    def svd_matrix_shape(self) -> tuple[int, int]:
        return self.feature_dim, self.num_classes


DEFAULT_SPEC = ModelSpec()


@dataclass
class _ForwardCache:
    x: np.ndarray
    conv: np.ndarray
    pooled: np.ndarray
    flat: np.ndarray
    probs: np.ndarray


def model_spec_for_dataset(dataset: object) -> ModelSpec:
    input_shape = getattr(dataset, "input_shape", MNIST_INPUT_SHAPE)
    num_classes = getattr(dataset, "num_classes", NUM_CLASSES)
    return ModelSpec(input_shape=tuple(input_shape), num_classes=int(num_classes))


def parameter_size(
    input_dim: int | None = None,
    num_classes: int = NUM_CLASSES,
    *,
    spec: ModelSpec | None = None,
) -> int:
    if spec is not None:
        return spec.parameter_size
    if input_dim is not None:
        return int(input_dim) * int(num_classes) + int(num_classes)
    return DEFAULT_SPEC.parameter_size


def init_params(
    *,
    seed: int = 0,
    spec: ModelSpec | None = None,
    input_shape: tuple[int, int, int] = MNIST_INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    model_spec = spec or ModelSpec(input_shape=input_shape, num_classes=num_classes)
    rng = np.random.default_rng(seed)
    channels, _, _ = model_spec.input_shape
    kernel = model_spec.kernel_size
    conv_scale = np.sqrt(2.0 / (channels * kernel * kernel))
    dense_scale = np.sqrt(2.0 / model_spec.feature_dim)

    conv_w = rng.normal(
        0.0,
        conv_scale,
        size=(model_spec.conv_filters, channels, kernel, kernel),
    ).astype(np.float32)
    conv_b = np.zeros(model_spec.conv_filters, dtype=np.float32)
    dense_w = rng.normal(
        0.0,
        dense_scale,
        size=(model_spec.feature_dim, model_spec.num_classes),
    ).astype(np.float32)
    dense_b = np.zeros(model_spec.num_classes, dtype=np.float32)
    return params_to_vector(conv_w, conv_b, dense_w, dense_b)


def params_to_vector(*arrays: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(array, dtype=np.float32).reshape(-1) for array in arrays])


def vector_to_params(
    vector: np.ndarray,
    *,
    spec: ModelSpec | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model_spec = spec or DEFAULT_SPEC
    if vector.shape[0] != model_spec.parameter_size:
        raise ValueError(
            f"expected parameter vector of length {model_spec.parameter_size}, "
            f"got {vector.shape[0]}"
        )

    channels, _, _ = model_spec.input_shape
    kernel = model_spec.kernel_size
    conv_end = model_spec.conv_bias_offset
    conv_b_end = model_spec.dense_weight_offset
    dense_end = model_spec.dense_bias_offset
    conv_w = vector[:conv_end].reshape(model_spec.conv_filters, channels, kernel, kernel)
    conv_b = vector[conv_end:conv_b_end]
    dense_w = vector[conv_b_end:dense_end].reshape(model_spec.feature_dim, model_spec.num_classes)
    dense_b = vector[dense_end:]
    return conv_w, conv_b, dense_w, dense_b


def predict(
    vector: np.ndarray,
    x: np.ndarray,
    *,
    spec: ModelSpec | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    model_spec = spec or DEFAULT_SPEC
    preds: list[np.ndarray] = []
    conv_w, conv_b, dense_w, dense_b = vector_to_params(vector, spec=model_spec)
    for start in range(0, len(x), batch_size):
        logits, _ = _forward_from_params(
            conv_w,
            conv_b,
            dense_w,
            dense_b,
            x[start : start + batch_size],
            model_spec,
            cache=False,
        )
        preds.append(np.argmax(logits, axis=1).astype(np.int64))
    return np.concatenate(preds, axis=0) if preds else np.empty(0, dtype=np.int64)


def accuracy(
    vector: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 512,
    spec: ModelSpec | None = None,
) -> float:
    if len(y) == 0:
        return 0.0
    correct = 0
    for start in range(0, len(y), batch_size):
        end = start + batch_size
        pred = predict(vector, x[start:end], spec=spec, batch_size=batch_size)
        correct += int(np.sum(pred == y[start:end]))
    return correct / len(y)


def local_train_delta(
    global_vector: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.05,
    epochs: int = 1,
    batch_size: int = 32,
    seed: int = 0,
    spec: ModelSpec | None = None,
) -> tuple[np.ndarray, TrainStats]:
    """Train a local CNN copy for a few epochs and return its model delta."""

    if len(y) == 0:
        return np.zeros_like(global_vector), TrainStats(loss=0.0, samples=0)

    model_spec = spec or DEFAULT_SPEC
    rng = np.random.default_rng(seed)
    local = global_vector.astype(np.float32, copy=True)
    conv_w, conv_b, dense_w, dense_b = vector_to_params(local, spec=model_spec)
    labels = np.asarray(y, dtype=np.int64)
    losses: list[float] = []

    for _ in range(epochs):
        order = rng.permutation(len(labels))
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            xb = x[batch_idx]
            yb = labels[batch_idx]
            logits, cache = _forward_from_params(
                conv_w,
                conv_b,
                dense_w,
                dense_b,
                xb,
                model_spec,
                cache=True,
            )
            probs = cache.probs
            losses.append(_cross_entropy(probs, yb))
            grads = _backward(cache, yb, conv_w, dense_w, model_spec)
            grad_conv_w, grad_conv_b, grad_dense_w, grad_dense_b = grads
            conv_w -= lr * grad_conv_w.astype(np.float32)
            conv_b -= lr * grad_conv_b.astype(np.float32)
            dense_w -= lr * grad_dense_w.astype(np.float32)
            dense_b -= lr * grad_dense_b.astype(np.float32)

    updated = params_to_vector(conv_w, conv_b, dense_w, dense_b)
    delta = (updated - global_vector).astype(np.float32)
    return delta, TrainStats(loss=float(np.mean(losses)), samples=len(labels))


def _forward_from_params(
    conv_w: np.ndarray,
    conv_b: np.ndarray,
    dense_w: np.ndarray,
    dense_b: np.ndarray,
    x: np.ndarray,
    spec: ModelSpec,
    *,
    cache: bool,
) -> tuple[np.ndarray, _ForwardCache | None]:
    x_nchw = _as_nchw(x, spec)
    conv = _conv2d_forward(
        x_nchw,
        conv_w,
        conv_b,
        stride=spec.conv_stride,
        padding=spec.padding,
    )
    relu = np.maximum(conv, 0.0).astype(np.float32)
    pooled = _avg_pool2d_forward(relu, spec.pool_size)
    flat = pooled.reshape(pooled.shape[0], -1)
    logits = flat @ dense_w + dense_b
    if not cache:
        return logits.astype(np.float32), None
    probs = _softmax(logits)
    return logits.astype(np.float32), _ForwardCache(
        x=x_nchw,
        conv=conv,
        pooled=pooled,
        flat=flat,
        probs=probs,
    )


def _backward(
    cache: _ForwardCache,
    labels: np.ndarray,
    conv_w: np.ndarray,
    dense_w: np.ndarray,
    spec: ModelSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_size = len(labels)
    grad_logits = cache.probs.copy()
    grad_logits[np.arange(batch_size), labels] -= 1.0
    grad_logits /= batch_size

    grad_dense_w = cache.flat.T @ grad_logits
    grad_dense_b = np.sum(grad_logits, axis=0)
    grad_flat = grad_logits @ dense_w.T
    grad_pool = grad_flat.reshape(cache.pooled.shape)
    grad_relu = _avg_pool2d_backward(grad_pool, cache.conv.shape, spec.pool_size)
    grad_conv = grad_relu * (cache.conv > 0.0)
    _, grad_conv_w, grad_conv_b = _conv2d_backward(
        cache.x,
        conv_w,
        grad_conv,
        stride=spec.conv_stride,
        padding=spec.padding,
    )
    return (
        grad_conv_w.astype(np.float32),
        grad_conv_b.astype(np.float32),
        grad_dense_w.astype(np.float32),
        grad_dense_b.astype(np.float32),
    )


def _as_nchw(x: np.ndarray, spec: ModelSpec) -> np.ndarray:
    data = np.asarray(x, dtype=np.float32)
    if data.ndim == 4:
        return data
    if data.ndim == 3 and spec.input_shape[0] == 1:
        return data[:, None, :, :]
    if data.ndim == 2 and data.shape[1] == int(np.prod(spec.input_shape)):
        return data.reshape(data.shape[0], *spec.input_shape)
    raise ValueError(
        "expected image batch with shape [samples, channels, height, width] "
        f"compatible with {spec.input_shape}, got {data.shape}"
    )


def _conv2d_forward(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    stride: int,
    padding: int,
) -> np.ndarray:
    n_samples, _, height, width = x.shape
    filters, _, kernel, _ = weights.shape
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    out_h = (height + 2 * padding - kernel) // stride + 1
    out_w = (width + 2 * padding - kernel) // stride + 1
    out = np.empty((n_samples, filters, out_h, out_w), dtype=np.float32)

    for out_row in range(out_h):
        row = out_row * stride
        for out_col in range(out_w):
            col = out_col * stride
            patch = x_padded[:, :, row : row + kernel, col : col + kernel]
            out[:, :, out_row, out_col] = np.tensordot(
                patch,
                weights,
                axes=([1, 2, 3], [1, 2, 3]),
            )
    out += bias[None, :, None, None]
    return out


def _conv2d_backward(
    x: np.ndarray,
    weights: np.ndarray,
    grad_out: np.ndarray,
    *,
    stride: int,
    padding: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, height, width = x.shape
    _, _, kernel, _ = weights.shape
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    grad_x_padded = np.zeros_like(x_padded, dtype=np.float32)
    grad_w = np.zeros_like(weights, dtype=np.float32)
    grad_b = np.sum(grad_out, axis=(0, 2, 3)).astype(np.float32)

    out_h, out_w = grad_out.shape[2:]
    for out_row in range(out_h):
        row = out_row * stride
        for out_col in range(out_w):
            col = out_col * stride
            patch = x_padded[:, :, row : row + kernel, col : col + kernel]
            grad_here = grad_out[:, :, out_row, out_col]
            grad_w += np.tensordot(grad_here, patch, axes=([0], [0]))
            grad_x_padded[:, :, row : row + kernel, col : col + kernel] += np.tensordot(
                grad_here,
                weights,
                axes=([1], [0]),
            )

    if padding == 0:
        grad_x = grad_x_padded
    else:
        grad_x = grad_x_padded[:, :, padding : padding + height, padding : padding + width]
    return grad_x.astype(np.float32), grad_w.astype(np.float32), grad_b


def _avg_pool2d_forward(x: np.ndarray, pool_size: int) -> np.ndarray:
    n_samples, channels, height, width = x.shape
    out_h = height // pool_size
    out_w = width // pool_size
    cropped = x[:, :, : out_h * pool_size, : out_w * pool_size]
    pooled = cropped.reshape(n_samples, channels, out_h, pool_size, out_w, pool_size)
    return pooled.mean(axis=(3, 5)).astype(np.float32)


def _avg_pool2d_backward(
    grad_pool: np.ndarray,
    input_shape: tuple[int, int, int, int],
    pool_size: int,
) -> np.ndarray:
    grad = np.repeat(np.repeat(grad_pool, pool_size, axis=2), pool_size, axis=3)
    grad = grad / float(pool_size * pool_size)
    full = np.zeros(input_shape, dtype=np.float32)
    full[:, :, : grad.shape[2], : grad.shape[3]] = grad.astype(np.float32)
    return full


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32)


def _cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))
