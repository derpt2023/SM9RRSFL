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


class TorchTrainingContext:
    """Keep dataset tensors and client indices resident on the selected device."""

    def __init__(
        self,
        dataset,
        client_indices: list[np.ndarray],
        *,
        spec: ModelSpec | None = None,
        device: str = "auto",
    ) -> None:
        torch = _torch_module()
        self.torch = torch
        self.device = _resolve_device(torch, device)
        self.spec = spec or DEFAULT_SPEC
        self.x_train = _torch_data(torch, dataset.x_train, self.spec, self.device)
        self.y_train = torch.as_tensor(np.asarray(dataset.y_train, dtype=np.int64), dtype=torch.long, device=self.device)
        self.x_test = _torch_data(torch, dataset.x_test, self.spec, self.device)
        self.y_test = torch.as_tensor(np.asarray(dataset.y_test, dtype=np.int64), dtype=torch.long, device=self.device)
        self.client_indices = [
            torch.as_tensor(np.asarray(indices, dtype=np.int64), dtype=torch.long, device=self.device)
            for indices in client_indices
        ]

    def accuracy(self, vector: np.ndarray, *, batch_size: int = 512) -> float:
        if int(self.y_test.numel()) == 0:
            return 0.0
        params = _torch_params_from_vector(self.torch, vector, self.spec, self.device, requires_grad=False)
        correct = 0
        with self.torch.no_grad():
            for start in range(0, int(self.y_test.numel()), batch_size):
                end = start + batch_size
                logits = _torch_forward(self.torch, params, self.x_test[start:end], self.spec)
                correct += int((self.torch.argmax(logits, dim=1) == self.y_test[start:end]).sum().detach().cpu().item())
        return correct / int(self.y_test.numel())

    def local_train_delta(
        self,
        global_vector: np.ndarray,
        *,
        client_idx: int,
        lr: float,
        epochs: int,
        batch_size: int,
        seed: int,
    ) -> tuple[np.ndarray, TrainStats]:
        indices = self.client_indices[client_idx]
        samples = int(indices.numel())
        if samples == 0:
            return np.zeros_like(global_vector), TrainStats(loss=0.0, samples=0)

        params = list(_torch_params_from_vector(self.torch, global_vector, self.spec, self.device, requires_grad=True))
        features = self.x_train.index_select(0, indices)
        labels = self.y_train.index_select(0, indices)
        rng = np.random.default_rng(seed)

        loss_sum = None
        loss_batches = 0
        for _ in range(epochs):
            order = rng.permutation(samples)
            for start in range(0, samples, batch_size):
                batch_idx = order[start : start + batch_size]
                index = self.torch.as_tensor(batch_idx, dtype=self.torch.long, device=self.device)
                logits = _torch_forward(self.torch, params, features.index_select(0, index), self.spec)
                loss = self.torch.nn.functional.cross_entropy(logits, labels.index_select(0, index))
                loss.backward()

                with self.torch.no_grad():
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
        return delta, TrainStats(loss=mean_loss, samples=samples)


def torch_weighted_average(
    updates: list[np.ndarray] | np.ndarray,
    weights: list[float] | np.ndarray,
    sample_counts: list[int] | np.ndarray | None = None,
    *,
    device: str = "auto",
) -> np.ndarray:
    torch = _torch_module()
    torch_device = _resolve_device(torch, device)
    stacked_np = np.asarray(updates, dtype=np.float32)
    if stacked_np.ndim != 2 or stacked_np.shape[0] == 0:
        raise ValueError("updates must have shape [num_updates, num_parameters]")
    weight_np = np.asarray(weights, dtype=np.float64)
    if weight_np.shape != (stacked_np.shape[0],):
        raise ValueError("weights must have shape [num_updates]")
    if sample_counts is not None:
        sample_np = np.asarray(sample_counts, dtype=np.float64)
        if sample_np.shape != (stacked_np.shape[0],):
            raise ValueError("sample_counts must have shape [num_updates]")
        weight_np = weight_np * np.maximum(sample_np, 0.0)
    total = float(np.sum(weight_np))
    if total <= 0.0:
        weight_np = np.full(stacked_np.shape[0], 1.0 / stacked_np.shape[0], dtype=np.float64)
    else:
        weight_np = weight_np / total
    stacked = torch.as_tensor(stacked_np, dtype=torch.float32, device=torch_device)
    weights_t = torch.as_tensor(weight_np.astype(np.float32), dtype=torch.float32, device=torch_device)
    return weights_t.matmul(stacked).detach().cpu().numpy().astype(np.float32)


def torch_krum_select(
    updates: list[np.ndarray] | np.ndarray,
    *,
    byzantine_count: int,
    device: str = "auto",
) -> tuple[int, np.ndarray, int]:
    if byzantine_count < 0:
        raise ValueError("byzantine_count must be non-negative")
    torch = _torch_module()
    torch_device = _resolve_device(torch, device)
    stacked_np = np.asarray(updates, dtype=np.float32)
    if stacked_np.ndim != 2 or stacked_np.shape[0] == 0:
        raise ValueError("updates must have shape [num_updates, num_parameters]")
    n = stacked_np.shape[0]
    if n < 3:
        raise ValueError("Krum requires at least 3 updates")
    neighbor_count = n - byzantine_count - 2
    if neighbor_count < 1:
        raise ValueError("Krum requires n - f - 2 >= 1; reduce malicious ratio or increase clients")

    stacked = torch.as_tensor(stacked_np, dtype=torch.float32, device=torch_device)
    distances = torch.cdist(stacked, stacked, p=2).square()
    scores = []
    for idx in range(n):
        nearest = torch.topk(distances[idx], k=neighbor_count + 1, largest=False).values[1:]
        scores.append(torch.sum(nearest))
    score_tensor = torch.stack(scores)
    selected = int(torch.argmin(score_tensor).detach().cpu().item())
    return selected, score_tensor.detach().cpu().numpy().astype(np.float64), neighbor_count


def torch_top_singular_feature(matrix: np.ndarray, *, device: str = "auto") -> tuple[float, np.ndarray]:
    torch = _torch_module()
    torch_device = _resolve_device(torch, device)
    matrix_np = np.asarray(matrix, dtype=np.float32)
    if getattr(torch_device, "type", None) == "mps":
        u, singular_values, _ = np.linalg.svd(matrix_np, full_matrices=False)
        return float(singular_values[0]), u[:, 0].astype(np.float32)
    tensor = torch.as_tensor(matrix_np, dtype=torch.float32, device=torch_device)
    u, singular_values, _ = torch.linalg.svd(tensor, full_matrices=False)
    sigma = float(singular_values[0].detach().cpu().item())
    u0 = u[:, 0].detach().cpu().numpy().astype(np.float32)
    return sigma, u0


def torch_singular_values_from_gram(matrix: np.ndarray, *, device: str = "auto") -> np.ndarray:
    torch = _torch_module()
    torch_device = _resolve_device(torch, device)
    matrix_np = np.asarray(matrix, dtype=np.float32)
    if getattr(torch_device, "type", None) == "mps":
        gram = matrix_np.T @ matrix_np
        return np.linalg.svd(gram, compute_uv=False).astype(np.float32)
    tensor = torch.as_tensor(matrix_np, dtype=torch.float32, device=torch_device)
    gram = tensor.T.matmul(tensor)
    return torch.linalg.svdvals(gram).detach().cpu().numpy().astype(np.float32)


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
