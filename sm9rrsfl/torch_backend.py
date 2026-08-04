"""Optional PyTorch compute backend for CNN training and evaluation."""

from __future__ import annotations

from functools import lru_cache
from threading import Lock
import weakref
import numpy as np

from .model import (
    DEFAULT_SPEC,
    ModelSpec,
    TrainStats,
    _as_nchw,
)


SUPPORTED_BACKENDS = {"numpy", "auto", "torch"}
SUPPORTED_DEVICES = {"auto", "cpu", "cuda", "mps"}
_STREAMING_TORCH_AVERAGE_THRESHOLD_BYTES = 256 * 1024 * 1024
_DATASET_TENSOR_CACHE: dict[tuple[int, str, ModelSpec], tuple[object, ...]] = {}
_DATASET_TENSOR_CACHE_LOCK = Lock()


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


def torch_targeted_metrics(
    vector: np.ndarray,
    x: np.ndarray,
    target_labels: np.ndarray,
    *,
    batch_size: int = 512,
    spec: ModelSpec | None = None,
    device: str = "auto",
) -> tuple[float, float]:
    """Evaluate target-label success and confidence on a PyTorch device."""

    labels_np = np.asarray(target_labels, dtype=np.int64).reshape(-1)
    if len(labels_np) == 0:
        return 0.0, 0.0
    if len(x) != len(labels_np):
        raise ValueError("x and target_labels must contain the same number of samples")
    torch = _torch_module()
    torch_device = _resolve_device(torch, device)
    model_spec = spec or DEFAULT_SPEC
    params = _torch_params_from_vector(
        torch,
        vector,
        model_spec,
        torch_device,
        requires_grad=False,
    )
    successes = 0
    confidence_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(labels_np), batch_size):
            end = start + batch_size
            xb = _torch_data(torch, x[start:end], model_spec, torch_device)
            labels = torch.as_tensor(
                labels_np[start:end],
                dtype=torch.long,
                device=torch_device,
            )
            probs = torch.softmax(
                _torch_forward(torch, params, xb, model_spec),
                dim=1,
            )
            successes += int(
                (torch.argmax(probs, dim=1) == labels).sum().detach().cpu().item()
            )
            confidence_sum += float(
                probs.gather(1, labels.reshape(-1, 1))
                .sum()
                .detach()
                .cpu()
                .item()
            )
    return successes / len(labels_np), confidence_sum / len(labels_np)


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


def _resident_dataset_tensors(torch, dataset, spec: ModelSpec, device) -> tuple[object, ...]:
    """跨配置复用同一数据集的设备张量，避免反复上传完整 CIFAR-10。"""

    key = (id(dataset), str(device), spec)
    with _DATASET_TENSOR_CACHE_LOCK:
        cached = _DATASET_TENSOR_CACHE.get(key)
        if cached is not None and cached[0]() is dataset:
            return cached[1:]

        x_train = _torch_data(torch, dataset.x_train, spec, device)
        y_train = torch.as_tensor(
            np.asarray(dataset.y_train, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        x_test = _torch_data(torch, dataset.x_test, spec, device)
        y_test = torch.as_tensor(
            np.asarray(dataset.y_test, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )

        def clear_cache(_reference, cache_key=key):
            with _DATASET_TENSOR_CACHE_LOCK:
                _DATASET_TENSOR_CACHE.pop(cache_key, None)

        _DATASET_TENSOR_CACHE[key] = (
            weakref.ref(dataset, clear_cache),
            x_train,
            y_train,
            x_test,
            y_test,
        )
        return x_train, y_train, x_test, y_test


class TorchTrainingContext:
    """Keep dataset tensors and client indices resident on the selected device.

    中文说明：同一轮的全局参数只从 CPU 传到设备一次。每个客户端都在设备端
    克隆这份参数进行本地训练，最后仅把必须交给密码/检测层的更新传回 CPU。
    """

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
        (
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test,
        ) = _resident_dataset_tensors(torch, dataset, self.spec, self.device)
        self.client_indices = [
            torch.as_tensor(np.asarray(indices, dtype=np.int64), dtype=torch.long, device=self.device)
            for indices in client_indices
        ]
        self._global_vector = None
        self._global_vector_source: np.ndarray | None = None

    def _ensure_global_vector(self, vector):
        """缓存当前轮全局参数，避免每个客户端重复执行主机到设备传输。"""

        if self._global_vector_source is not vector:
            if self.torch.is_tensor(vector):
                self._global_vector = vector.detach().to(
                    device=self.device,
                    dtype=self.torch.float32,
                )
            else:
                array = np.ascontiguousarray(vector, dtype=np.float32)
                self._global_vector = self.torch.from_numpy(array).to(
                    device=self.device,
                    dtype=self.torch.float32,
                )
            # 保留源数组引用，既能判断同一轮，也可防止 Python 复用旧对象 id。
            self._global_vector_source = vector
        return self._global_vector

    def accuracy(self, vector: np.ndarray, *, batch_size: int = 512) -> float:
        if int(self.y_test.numel()) == 0:
            return 0.0
        global_vector = self._ensure_global_vector(vector)
        params = _torch_params_from_tensor(
            self.torch,
            global_vector,
            self.spec,
            requires_grad=False,
            clone=False,
        )
        correct = 0
        with self.torch.no_grad():
            for start in range(0, int(self.y_test.numel()), batch_size):
                end = start + batch_size
                logits = _torch_forward(self.torch, params, self.x_test[start:end], self.spec)
                correct += int((self.torch.argmax(logits, dim=1) == self.y_test[start:end]).sum().detach().cpu().item())
        return correct / int(self.y_test.numel())

    def targeted_metrics(
        self,
        vector,
        *,
        target_indices: np.ndarray,
        target_label: int,
    ) -> tuple[float, float]:
        """Evaluate the configured auxiliary targets without copying the model."""

        target_index_array = np.asarray(target_indices, dtype=np.int64).reshape(-1)
        if target_index_array.size == 0:
            return 0.0, 0.0
        index = self.torch.as_tensor(
            target_index_array,
            dtype=self.torch.long,
            device=self.device,
        )
        global_vector = self._ensure_global_vector(vector)
        params = _torch_params_from_tensor(
            self.torch,
            global_vector,
            self.spec,
            requires_grad=False,
            clone=False,
        )
        with self.torch.no_grad():
            probs = self.torch.softmax(
                _torch_forward(
                    self.torch,
                    params,
                    self.x_test.index_select(0, index),
                    self.spec,
                ),
                dim=1,
            )
            labels = self.torch.full(
                (len(target_index_array),),
                int(target_label),
                dtype=self.torch.long,
                device=self.device,
            )
            success = float(
                (self.torch.argmax(probs, dim=1) == labels)
                .float()
                .mean()
                .detach()
                .cpu()
                .item()
            )
            confidence = float(
                probs.gather(1, labels.reshape(-1, 1))
                .mean()
                .detach()
                .cpu()
                .item()
            )
        return success, confidence

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

        global_tensor = self._ensure_global_vector(global_vector)
        params = list(
            _torch_params_from_tensor(
                self.torch,
                global_tensor,
                self.spec,
                requires_grad=True,
                clone=True,
            )
        )
        rng = np.random.default_rng(seed)

        loss_sum = None
        loss_batches = 0
        for _ in range(epochs):
            order = rng.permutation(samples)
            for start in range(0, samples, batch_size):
                batch_idx = order[start : start + batch_size]
                local_index = self.torch.as_tensor(
                    batch_idx,
                    dtype=self.torch.long,
                    device=self.device,
                )
                # 直接组合客户端索引和批索引，避免每轮复制完整客户端数据子集。
                batch_index = indices.index_select(0, local_index)
                features = self.x_train.index_select(0, batch_index)
                labels = self.y_train.index_select(0, batch_index)
                logits = _torch_forward(self.torch, params, features, self.spec)
                loss = self.torch.nn.functional.cross_entropy(logits, labels)
                loss.backward()

                with self.torch.no_grad():
                    for param in params:
                        if param.grad is not None:
                            param -= lr * param.grad
                            param.grad = None
                detached = loss.detach()
                loss_sum = detached if loss_sum is None else loss_sum + detached
                loss_batches += 1

        updated_tensor = _torch_flat_vector_from_params(self.torch, params)
        delta = (
            updated_tensor.sub(global_tensor)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        mean_loss = 0.0
        if loss_sum is not None and loss_batches:
            mean_loss = float((loss_sum / loss_batches).detach().cpu().item())
        return delta, TrainStats(loss=mean_loss, samples=samples)

    def local_train_delta_resident(
        self,
        global_vector,
        *,
        client_idx: int,
        lr: float,
        epochs: int,
        batch_size: int,
        seed: int,
    ):
        """在设备端返回更新，供完整实验避免每客户端往返复制整个模型。"""

        indices = self.client_indices[client_idx]
        samples = int(indices.numel())
        global_tensor = self._ensure_global_vector(global_vector)
        if samples == 0:
            return self.torch.zeros_like(global_tensor), TrainStats(loss=0.0, samples=0)

        params = list(
            _torch_params_from_tensor(
                self.torch,
                global_tensor,
                self.spec,
                requires_grad=True,
                clone=True,
            )
        )
        rng = np.random.default_rng(seed)
        loss_sum = None
        loss_batches = 0
        for _ in range(epochs):
            order = rng.permutation(samples)
            for start in range(0, samples, batch_size):
                local_index = self.torch.as_tensor(
                    order[start : start + batch_size],
                    dtype=self.torch.long,
                    device=self.device,
                )
                batch_index = indices.index_select(0, local_index)
                logits = _torch_forward(
                    self.torch,
                    params,
                    self.x_train.index_select(0, batch_index),
                    self.spec,
                )
                loss = self.torch.nn.functional.cross_entropy(
                    logits,
                    self.y_train.index_select(0, batch_index),
                )
                loss.backward()
                with self.torch.no_grad():
                    for param in params:
                        if param.grad is not None:
                            param -= lr * param.grad
                            param.grad = None
                detached = loss.detach()
                loss_sum = detached if loss_sum is None else loss_sum + detached
                loss_batches += 1

        delta = _torch_flat_vector_from_params(self.torch, params).sub(global_tensor).detach()
        mean_loss = 0.0
        if loss_sum is not None and loss_batches:
            mean_loss = float((loss_sum / loss_batches).detach().cpu().item())
        return delta, TrainStats(loss=mean_loss, samples=samples)

    def alternating_minimization_delta_resident(
        self,
        global_vector,
        *,
        client_idx: int,
        target_indices: np.ndarray,
        target_label: int,
        lr: float,
        attack_epochs: int,
        batch_size: int,
        stealth_steps: int,
        boost: float,
        distance_weight: float,
        seed: int,
    ):
        """Run the Bhagoji alternating attack while keeping tensors resident."""

        indices = self.client_indices[client_idx]
        samples = int(indices.numel())
        global_tensor = self._ensure_global_vector(global_vector)
        if samples == 0:
            return self.torch.zeros_like(global_tensor), TrainStats(loss=0.0, samples=0)
        target_index_array = np.asarray(target_indices, dtype=np.int64).reshape(-1)
        if target_index_array.size == 0:
            raise ValueError("alternating minimization requires auxiliary target samples")
        if attack_epochs < 1 or batch_size < 1 or stealth_steps < 1:
            raise ValueError(
                "attack_epochs, batch_size and stealth_steps must be at least 1"
            )
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be finite and positive")
        if not np.isfinite(boost) or boost <= 0.0:
            raise ValueError("boost must be finite and positive")
        if not np.isfinite(distance_weight) or distance_weight < 0.0:
            raise ValueError("distance_weight must be finite and non-negative")
        if target_label < 0 or target_label >= self.spec.num_classes:
            raise ValueError("target_label is outside the model class range")

        benign_delta, _ = self.local_train_delta_resident(
            global_vector,
            client_idx=client_idx,
            lr=lr,
            epochs=attack_epochs,
            batch_size=batch_size,
            seed=seed + 1_000_003,
        )
        benign_reference_vector = global_tensor.add(benign_delta)
        reference_params = tuple(
            _torch_params_from_tensor(
                self.torch,
                benign_reference_vector,
                self.spec,
                requires_grad=False,
                clone=False,
            )
        )
        params = list(
            _torch_params_from_tensor(
                self.torch,
                global_tensor,
                self.spec,
                requires_grad=True,
                clone=True,
            )
        )

        target_index = self.torch.as_tensor(
            target_index_array,
            dtype=self.torch.long,
            device=self.device,
        )
        target_features = self.x_test.index_select(0, target_index)
        target_labels = self.torch.full(
            (len(target_index_array),),
            int(target_label),
            dtype=self.torch.long,
            device=self.device,
        )

        rng = np.random.default_rng(seed)
        benign_batches: list[np.ndarray] = []
        for _ in range(attack_epochs):
            order = rng.permutation(samples)
            benign_batches.extend(
                order[start : start + batch_size]
                for start in range(0, len(order), batch_size)
            )

        loss_sum = None
        loss_batches = 0
        for block_start in range(0, len(benign_batches), stealth_steps):
            for batch_idx in benign_batches[
                block_start : block_start + stealth_steps
            ]:
                local_index = self.torch.as_tensor(
                    batch_idx,
                    dtype=self.torch.long,
                    device=self.device,
                )
                batch_index = indices.index_select(0, local_index)
                logits = _torch_forward(
                    self.torch,
                    params,
                    self.x_train.index_select(0, batch_index),
                    self.spec,
                )
                loss = self.torch.nn.functional.cross_entropy(
                    logits,
                    self.y_train.index_select(0, batch_index),
                )
                loss.backward()
                with self.torch.no_grad():
                    for param, reference in zip(params, reference_params):
                        if param.grad is not None:
                            gradient = param.grad.add(
                                param.sub(reference),
                                alpha=float(distance_weight),
                            )
                            param.add_(gradient, alpha=-float(lr))
                            param.grad = None
                detached = loss.detach()
                loss_sum = detached if loss_sum is None else loss_sum + detached
                loss_batches += 1

            target_logits = _torch_forward(
                self.torch,
                params,
                target_features,
                self.spec,
            )
            adversarial_loss = self.torch.nn.functional.cross_entropy(
                target_logits,
                target_labels,
            )
            adversarial_loss.backward()
            with self.torch.no_grad():
                for param in params:
                    if param.grad is not None:
                        param.add_(param.grad, alpha=-float(lr * boost))
                        param.grad = None

        delta = _torch_flat_vector_from_params(self.torch, params).sub(global_tensor).detach()
        mean_loss = 0.0
        if loss_sum is not None and loss_batches:
            mean_loss = float((loss_sum / loss_batches).detach().cpu().item())
        return delta, TrainStats(loss=mean_loss, samples=samples)

    def to_numpy(self, tensor) -> np.ndarray:
        """仅在密码摘要或 CPU 检测确实需要时执行设备到主机传输。"""

        if not self.torch.is_tensor(tensor):
            return np.asarray(tensor, dtype=np.float32)
        return tensor.detach().cpu().numpy().astype(np.float32, copy=False)

    def poison_update(self, update, *, attack: str, scale: float, seed: int):
        """在设备上执行与 NumPy 版本等价的投毒变换。"""

        if attack == "none":
            return update.clone()
        if attack == "sign_flip":
            return update.mul(-float(scale))
        if attack == "gaussian":
            std = float(self.torch.std(update, correction=0).detach().cpu().item())
            if std <= 1e-12:
                std = float(self.torch.linalg.vector_norm(update).detach().cpu().item()) / max(
                    1,
                    int(update.numel()),
                ) ** 0.5
            std = max(std, 1e-3)
            if self.device.type == "mps":
                # MPS 当前不提供独立 Generator；在 CPU 生成后一次性上传。
                rng = np.random.default_rng(seed)
                noise = rng.normal(
                    0.0,
                    float(scale) * std,
                    size=tuple(update.shape),
                ).astype(np.float32)
                return self.torch.from_numpy(noise).to(device=self.device)
            generator = self.torch.Generator(device=self.device.type)
            generator.manual_seed(int(seed))
            return self.torch.randn(
                update.shape,
                dtype=update.dtype,
                device=update.device,
                generator=generator,
            ).mul(float(scale) * std)
        from .attacks import is_alternating_minimization_attack

        if is_alternating_minimization_attack(attack):
            raise ValueError(
                "alternating minimization requires model/data-aware local training; "
                "use alternating_minimization_delta_resident()"
            )
        raise ValueError(
            "attack must be one of: none, sign_flip, gaussian, "
            "alternating_minimization (or legacy alias alternating)"
        )

    def weighted_average(self, updates, weights, sample_counts=None):
        """流式设备端加权聚合，不构造 N×P 的额外堆叠张量。"""

        if not updates:
            raise ValueError("at least one update is required")
        weight_array = np.asarray(weights, dtype=np.float64)
        if sample_counts is not None:
            weight_array *= np.maximum(np.asarray(sample_counts, dtype=np.float64), 0.0)
        total = float(np.sum(weight_array))
        if total <= 0.0:
            weight_array.fill(1.0 / len(updates))
        else:
            weight_array /= total
        result = self.torch.zeros_like(updates[0])
        for update, weight in zip(updates, weight_array):
            result.add_(update, alpha=float(weight))
        return result

    def krum_select(self, updates, *, byzantine_count: int):
        """直接在驻留设备的更新上计算 Krum 距离。"""

        count = len(updates)
        if count < 3:
            raise ValueError("Krum requires at least 3 updates")
        neighbor_count = count - byzantine_count - 2
        if neighbor_count < 1:
            raise ValueError("Krum requires n - f - 2 >= 1; reduce malicious ratio or increase clients")
        stacked = self.torch.stack(updates)
        distances = self.torch.cdist(stacked, stacked, p=2).square()
        nearest = self.torch.topk(
            distances,
            k=neighbor_count + 1,
            dim=1,
            largest=False,
        ).values[:, 1:]
        scores = nearest.sum(dim=1)
        selected = int(self.torch.argmin(scores).detach().cpu().item())
        return selected, updates[selected], scores.detach().cpu().numpy().astype(np.float64)

    def add_update(self, params, update):
        """让新一轮全局参数继续驻留设备，避免聚合结果回传再上传。"""

        return self._ensure_global_vector(params).add(update).detach()

    def zeros_like(self, update):
        return self.torch.zeros_like(update)


def torch_weighted_average(
    updates: list[np.ndarray] | np.ndarray,
    weights: list[float] | np.ndarray,
    sample_counts: list[int] | np.ndarray | None = None,
    *,
    device: str = "auto",
) -> np.ndarray:
    torch = _torch_module()
    torch_device = _resolve_device(torch, device)
    if isinstance(updates, np.ndarray):
        stacked_np = np.asarray(updates, dtype=np.float32)
        if stacked_np.ndim != 2 or stacked_np.shape[0] == 0:
            raise ValueError("updates must have shape [num_updates, num_parameters]")
        update_count, parameter_count = stacked_np.shape
    else:
        if not updates:
            raise ValueError("updates must have shape [num_updates, num_parameters]")
        update_count = len(updates)
        parameter_count = int(np.asarray(updates[0]).size)
        stacked_np = None
    weight_np = np.asarray(weights, dtype=np.float64)
    if weight_np.shape != (update_count,):
        raise ValueError("weights must have shape [num_updates]")
    if sample_counts is not None:
        sample_np = np.asarray(sample_counts, dtype=np.float64)
        if sample_np.shape != (update_count,):
            raise ValueError("sample_counts must have shape [num_updates]")
        weight_np = weight_np * np.maximum(sample_np, 0.0)
    total = float(np.sum(weight_np))
    if total <= 0.0:
        weight_np = np.full(update_count, 1.0 / update_count, dtype=np.float64)
    else:
        weight_np = weight_np / total
    estimated_bytes = update_count * parameter_count * np.dtype(np.float32).itemsize
    if stacked_np is None and estimated_bytes >= _STREAMING_TORCH_AVERAGE_THRESHOLD_BYTES:
        # 大模型逐条上传并累加，避免同时保留 CPU/GPU 两份完整更新矩阵。
        result = torch.zeros(parameter_count, dtype=torch.float32, device=torch_device)
        for update, weight in zip(updates, weight_np):
            update_tensor = torch.as_tensor(
                np.asarray(update, dtype=np.float32),
                dtype=torch.float32,
                device=torch_device,
            )
            result.add_(update_tensor, alpha=float(weight))
        return result.detach().cpu().numpy().astype(np.float32, copy=False)
    if stacked_np is None:
        stacked_np = np.asarray(updates, dtype=np.float32)
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
        return _numpy_top_singular_feature(matrix_np)
    tensor = torch.as_tensor(matrix_np, dtype=torch.float32, device=torch_device)
    try:
        u, singular_values, _ = torch.linalg.svd(tensor, full_matrices=False)
    except (NotImplementedError, RuntimeError):
        return _numpy_top_singular_feature(matrix_np)
    sigma = float(singular_values[0].detach().cpu().item())
    u0 = u[:, 0].detach().cpu().numpy().astype(np.float32)
    return sigma, u0


def torch_singular_values_from_gram(matrix, *, device: str = "auto") -> np.ndarray:
    torch = _torch_module()
    torch_device = _resolve_device(torch, device)
    if getattr(torch_device, "type", None) == "mps":
        matrix_np = (
            matrix.detach().cpu().numpy().astype(np.float32, copy=False)
            if torch.is_tensor(matrix)
            else np.asarray(matrix, dtype=np.float32)
        )
        return _numpy_singular_values_from_gram(matrix_np)
    if torch.is_tensor(matrix):
        tensor = matrix.detach().to(device=torch_device, dtype=torch.float32)
        matrix_np = None
    else:
        matrix_np = np.asarray(matrix, dtype=np.float32)
        tensor = torch.as_tensor(matrix_np, dtype=torch.float32, device=torch_device)
    gram = tensor.T.matmul(tensor)
    try:
        return torch.linalg.svdvals(gram).detach().cpu().numpy().astype(np.float32)
    except (NotImplementedError, RuntimeError):
        if matrix_np is None:
            matrix_np = tensor.cpu().numpy().astype(np.float32, copy=False)
        return _numpy_singular_values_from_gram(matrix_np)


def _numpy_top_singular_feature(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    scaled, scale = _scaled_finite_matrix(matrix)
    rows = scaled.shape[0]
    if scale == 0.0:
        direction = np.zeros(rows, dtype=np.float32)
        if rows:
            direction[0] = 1.0
        return 0.0, direction
    try:
        u, singular_values, _ = np.linalg.svd(scaled, full_matrices=False)
        sigma = float(singular_values[0]) * scale
        direction = u[:, 0]
    except np.linalg.LinAlgError:
        # LAPACK 偶发不收敛时使用确定性的幂迭代求第一左奇异向量。
        sigma_scaled, direction = _power_top_singular_feature(scaled)
        sigma = sigma_scaled * scale
    sigma = min(sigma, float(np.finfo(np.float32).max))
    return float(sigma), np.asarray(direction, dtype=np.float32)


def _numpy_singular_values_from_gram(matrix: np.ndarray) -> np.ndarray:
    scaled, scale = _scaled_finite_matrix(matrix)
    columns = scaled.shape[1]
    if scale == 0.0:
        return np.zeros(columns, dtype=np.float32)
    try:
        # svd(A)^2 与 svd(A.T@A) 数学等价，但避免先构造 Gram 导致条件数
        # 平方和 float32 溢出；统一用 float64 缩放矩阵计算。
        singular = np.linalg.svd(scaled, compute_uv=False)
        values = np.zeros(columns, dtype=np.float64)
        values[: len(singular)] = np.square(singular)
    except np.linalg.LinAlgError:
        gram = scaled.T @ scaled
        try:
            values = np.linalg.eigvalsh(gram)[::-1]
        except np.linalg.LinAlgError:
            values = np.sort(_jacobi_eigenvalues_symmetric(gram))[::-1]
        values = np.maximum(values, 0.0)
    values *= scale * scale
    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=float(np.finfo(np.float32).max),
        neginf=0.0,
    )
    values = np.clip(values, 0.0, float(np.finfo(np.float32).max))
    return values.astype(np.float32)


def numpy_top_singular_feature(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """公开稳定 CPU 实现，供未启用 Torch 的检测器复用。"""

    return _numpy_top_singular_feature(matrix)


def numpy_singular_values_from_gram(matrix: np.ndarray) -> np.ndarray:
    """公开稳定 CPU 实现，返回与 Gram 矩阵 SVD 相同的奇异值。"""

    return _numpy_singular_values_from_gram(matrix)


def _scaled_finite_matrix(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("SVD input must be a two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError("SVD input contains NaN or infinity")
    if array.size == 0:
        return array, 0.0
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return array, 0.0
    return array / scale, scale


def _power_top_singular_feature(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        return 0.0, np.zeros(rows, dtype=np.float64)
    column_norms = np.einsum("ij,ij->j", matrix, matrix)
    vector = np.zeros(columns, dtype=np.float64)
    vector[int(np.argmax(column_norms))] = 1.0
    direction = np.zeros(rows, dtype=np.float64)
    for _ in range(128):
        left = matrix @ vector
        sigma = float(np.sqrt(np.dot(left, left)))
        if sigma <= np.finfo(np.float64).eps:
            direction.fill(0.0)
            direction[0] = 1.0
            return 0.0, direction
        next_direction = left / sigma
        right = matrix.T @ next_direction
        norm = float(np.sqrt(np.dot(right, right)))
        if norm <= np.finfo(np.float64).eps:
            return sigma, next_direction
        next_vector = right / norm
        if np.max(np.abs(next_vector - vector)) <= 1e-12:
            vector = next_vector
            direction = next_direction
            break
        vector = next_vector
        direction = next_direction
    left = matrix @ vector
    sigma = float(np.sqrt(np.dot(left, left)))
    if sigma > np.finfo(np.float64).eps:
        direction = left / sigma
    return sigma, direction


def _jacobi_eigenvalues_symmetric(matrix: np.ndarray) -> np.ndarray:
    """小型对称矩阵的纯 NumPy Jacobi 兜底，避免再次依赖 LAPACK。"""

    values = np.asarray(matrix, dtype=np.float64).copy()
    size = values.shape[0]
    if size <= 1:
        return np.diag(values).copy()
    for _ in range(max(32, size * size * 16)):
        upper = np.abs(np.triu(values, k=1))
        flat_index = int(np.argmax(upper))
        maximum = float(upper.flat[flat_index])
        tolerance = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(values))))
        if maximum <= tolerance:
            break
        p, q = np.unravel_index(flat_index, upper.shape)
        apq = values[p, q]
        tau = (values[q, q] - values[p, p]) / (2.0 * apq)
        sign = 1.0 if tau >= 0.0 else -1.0
        tangent = sign / (abs(tau) + np.sqrt(1.0 + tau * tau))
        cosine = 1.0 / np.sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine
        app = values[p, p]
        aqq = values[q, q]
        for k in range(size):
            if k == p or k == q:
                continue
            akp = values[k, p]
            akq = values[k, q]
            values[k, p] = values[p, k] = cosine * akp - sine * akq
            values[k, q] = values[q, k] = sine * akp + cosine * akq
        values[p, p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
        values[q, q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
        values[p, q] = values[q, p] = 0.0
    return np.diag(values).copy()


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
    local = np.ascontiguousarray(vector, dtype=np.float32)
    flat = torch.from_numpy(local).to(device=device, dtype=torch.float32)
    return _torch_params_from_tensor(
        torch,
        flat,
        spec,
        requires_grad=requires_grad,
        clone=True,
    )


def _torch_params_from_tensor(
    torch,
    flat_vector,
    spec: ModelSpec,
    *,
    requires_grad: bool,
    clone: bool,
) -> tuple[object, ...]:
    """把设备端扁平参数映射为模型张量；训练客户端使用独立 clone。"""

    tensors = []
    offset = 0
    for shape in _parameter_shapes(spec):
        size = int(np.prod(shape))
        tensor = flat_vector.narrow(0, offset, size).reshape(shape)
        tensor = tensor.clone().detach() if clone else tensor.detach()
        tensor.requires_grad_(requires_grad)
        tensors.append(tensor)
        offset += size
    if offset != int(flat_vector.numel()):
        raise ValueError("parameter vector size does not match model specification")
    return tuple(tensors)


def _torch_vector_from_params(params: list[object] | tuple[object, ...]) -> np.ndarray:
    torch = _torch_module()
    return (
        _torch_flat_vector_from_params(torch, params)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def _torch_flat_vector_from_params(torch, params: list[object] | tuple[object, ...]):
    return torch.cat([param.detach().reshape(-1) for param in params], dim=0)


def _parameter_shapes(spec: ModelSpec) -> tuple[tuple[int, ...], ...]:
    channels, _, _ = spec.input_shape
    kernel = spec.kernel_size
    if spec.architecture == "cifar10":
        filters1, filters2 = spec.cifar_conv_filters
        hidden1, hidden2 = spec.cifar_hidden_dims
        return (
            (filters1, channels, kernel, kernel),
            (filters1,),
            (filters2, filters1, kernel, kernel),
            (filters2,),
            (spec.feature_dim, hidden1),
            (hidden1,),
            (hidden1, hidden2),
            (hidden2,),
            (hidden2, spec.num_classes),
            (spec.num_classes,),
        )
    return (
        (spec.conv_filters, channels, kernel, kernel),
        (spec.conv_filters,),
        (spec.feature_dim, spec.num_classes),
        (spec.num_classes,),
    )


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
