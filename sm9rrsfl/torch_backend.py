"""Optional PyTorch compute backend for CNN training and evaluation."""

from __future__ import annotations

from functools import lru_cache
import numpy as np

from .model import (
    DEFAULT_SPEC,
    ModelSpec,
    TrainStats,
    _as_nchw,
    _vector_to_cifar_params,
    params_to_vector,
    vector_to_params,
)


SUPPORTED_BACKENDS = {"numpy", "auto", "torch"}
SUPPORTED_DEVICES = {"auto", "cpu", "cuda", "mps"}


@lru_cache(maxsize=1)
def _torch_module():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for --compute-backend torch. "
            "Install torch first, or use --compute-backend numpy."
        ) from exc
    return torch


def should_use_torch(compute_backend: str, device: str) -> bool:
    backend = _normalize_backend(compute_backend)
    if backend == "numpy":
        return False
    if backend == "torch":
        torch = _torch_module()
        _resolve_device(torch, device)
        return True

    try:
        torch = _torch_module()
    except RuntimeError:
        return False
    if _normalize_device(device) != "auto":
        _resolve_device(torch, device)
        return True
    return _best_gpu_device(torch) is not None


def describe_backend(compute_backend: str, device: str) -> str:
    if not should_use_torch(compute_backend, device):
        return "numpy"
    torch = _torch_module()
    return f"torch:{_resolve_device(torch, device)}"


def torch_accuracy(
    vector: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 512,
    spec: ModelSpec | None = None,
    device: str = "auto",
) -> float:
    labels = np.asarray(y, dtype=np.int64)
    if len(labels) == 0:
        return 0.0

    torch = _torch_module()
    torch_device = _resolve_device(torch, device)
    model_spec = spec or DEFAULT_SPEC
    params = _torch_params_from_vector(torch, vector, model_spec, torch_device, requires_grad=False)
    correct = 0
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            end = start + batch_size
            xb = _torch_data(torch, x[start:end], model_spec, torch_device)
            yb = torch.as_tensor(labels[start:end], dtype=torch.long, device=torch_device)
            logits = _torch_forward(torch, params, xb, model_spec)
            correct += int((torch.argmax(logits, dim=1) == yb).sum().detach().cpu().item())
    return correct / len(labels)


def torch_local_train_delta(
    global_vector: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.05,
    epochs: int = 1,
    batch_size: int = 32,
    seed: int = 0,
    spec: ModelSpec | None = None,
    device: str = "auto",
) -> tuple[np.ndarray, TrainStats]:
    labels_np = np.asarray(y, dtype=np.int64)
    if len(labels_np) == 0:
        return np.zeros_like(global_vector), TrainStats(loss=0.0, samples=0)

    torch = _torch_module()
    torch_device = _resolve_device(torch, device)
    model_spec = spec or DEFAULT_SPEC
    params = list(_torch_params_from_vector(torch, global_vector, model_spec, torch_device, requires_grad=True))
    features = _torch_data(torch, x, model_spec, torch_device)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=torch_device)
    rng = np.random.default_rng(seed)

    loss_sum = None
    loss_batches = 0
    for _ in range(epochs):
        order = rng.permutation(len(labels_np))
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            index = torch.as_tensor(batch_idx, dtype=torch.long, device=torch_device)
            logits = _torch_forward(torch, params, features.index_select(0, index), model_spec)
            loss = torch.nn.functional.cross_entropy(logits, labels.index_select(0, index))
            loss.backward()

            with torch.no_grad():
                for param in params:
                    if param.grad is not None:
                        param -= lr * param.grad
                        param.grad = None
            detached = loss.detach()
            loss_sum = detached if loss_sum is None else loss_sum + detached
            loss_batches += 1

    updated = _torch_vector_from_params(params)
    delta = (updated - global_vector).astype(np.float32)
    mean_loss = 0.0
    if loss_sum is not None and loss_batches:
        mean_loss = float((loss_sum / loss_batches).detach().cpu().item())
    return delta, TrainStats(loss=mean_loss, samples=len(labels_np))


def _normalize_backend(compute_backend: str) -> str:
    backend = (compute_backend or "numpy").strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError("compute_backend must be one of: numpy, auto, torch")
    return backend


def _normalize_device(device: str) -> str:
    normalized = (device or "auto").strip().lower()
    if normalized not in SUPPORTED_DEVICES:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return normalized


def _best_gpu_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return None


def _resolve_device(torch, device: str):
    requested = _normalize_device(device)
    if requested == "auto":
        return torch.device(_best_gpu_device(torch) or "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for this PyTorch installation.")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS is not available for this PyTorch installation.")
    return torch.device(requested)


def _torch_data(torch, x: np.ndarray, spec: ModelSpec, device) -> object:
    array = np.ascontiguousarray(_as_nchw(x, spec), dtype=np.float32)
    return torch.from_numpy(array).to(device=device, dtype=torch.float32)


def _torch_params_from_vector(
    torch,
    vector: np.ndarray,
    spec: ModelSpec,
    device,
    *,
    requires_grad: bool,
) -> tuple[object, ...]:
    local = np.asarray(vector, dtype=np.float32)
    if spec.architecture == "cifar10":
        params = _vector_to_cifar_params(local, spec=spec)
        arrays = (
            params.conv1_w,
            params.conv1_b,
            params.conv2_w,
            params.conv2_b,
            params.dense1_w,
            params.dense1_b,
            params.dense2_w,
            params.dense2_b,
            params.logits_w,
            params.logits_b,
        )
    else:
        arrays = vector_to_params(local, spec=spec)
    tensors = []
    for array in arrays:
        tensor = torch.tensor(np.ascontiguousarray(array), dtype=torch.float32, device=device)
        tensor.requires_grad_(requires_grad)
        tensors.append(tensor)
    return tuple(tensors)


def _torch_vector_from_params(params: list[object] | tuple[object, ...]) -> np.ndarray:
    arrays = [param.detach().cpu().numpy().astype(np.float32, copy=False) for param in params]
    return params_to_vector(*arrays)


def _torch_forward(torch, params: list[object] | tuple[object, ...], x, spec: ModelSpec):
    if spec.architecture == "cifar10":
        (
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            dense1_w,
            dense1_b,
            dense2_w,
            dense2_b,
            logits_w,
            logits_b,
        ) = params
        conv1 = torch.nn.functional.conv2d(x, conv1_w, conv1_b, stride=1, padding=spec.padding)
        pooled1 = torch.nn.functional.avg_pool2d(torch.relu(conv1), kernel_size=spec.pool_size)
        conv2 = torch.nn.functional.conv2d(pooled1, conv2_w, conv2_b, stride=1, padding=spec.padding)
        pooled2 = torch.nn.functional.avg_pool2d(torch.relu(conv2), kernel_size=spec.pool_size)
        flat = torch.flatten(pooled2, start_dim=1)
        hidden1 = torch.relu(flat.matmul(dense1_w) + dense1_b)
        hidden2 = torch.relu(hidden1.matmul(dense2_w) + dense2_b)
        return hidden2.matmul(logits_w) + logits_b

    conv_w, conv_b, dense_w, dense_b = params
    conv = torch.nn.functional.conv2d(
        x,
        conv_w,
        conv_b,
        stride=spec.conv_stride,
        padding=spec.padding,
    )
    pooled = torch.nn.functional.avg_pool2d(torch.relu(conv), kernel_size=spec.pool_size)
    flat = torch.flatten(pooled, start_dim=1)
    return flat.matmul(dense_w) + dense_b
