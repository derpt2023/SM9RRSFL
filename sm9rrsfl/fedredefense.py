"""FedREDefense model-update reconstruction-error baseline.

The implementation follows the official ICML 2024 code path: each client owns
persistent synthetic images, soft labels, and a trainable synthetic learning
rate.  Starting from the current global model, differentiable synthetic SGD is
unrolled and optimized to reconstruct the submitted client model.  Clients
whose normalized reconstruction error is above the paper's threshold are
filtered and remain unavailable in later rounds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import ModelSpec


@dataclass(frozen=True)
class FedREDefenseResult:
    accepted_clients: tuple[str, ...]
    rejected_clients: tuple[str, ...]
    reconstruction_errors: dict[str, float]


@dataclass
class _SyntheticState:
    images: np.ndarray
    labels: np.ndarray
    teacher_lr: float


class FedREDefense:
    """Filter clients using normalized model-update reconstruction error."""

    def __init__(
        self,
        client_ids: list[str],
        *,
        model_spec: ModelSpec,
        threshold: float = 0.6,
        initial_iterations: int = 800,
        max_iterations: int = 2000,
        synthetic_steps: int = 5,
        images_per_class: int = 1,
        image_lr: float = 0.5,
        label_lr: float = 0.2,
        teacher_lr: float = 0.1,
        teacher_lr_lr: float = 5e-6,
        device: str = "auto",
        seed: int = 0,
        eps: float = 1e-9,
    ) -> None:
        if not client_ids:
            raise ValueError("FedREDefense requires at least one client")
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("FedREDefense threshold must be finite and positive")
        if initial_iterations < 1 or max_iterations < 1:
            raise ValueError("FedREDefense iterations must be at least 1")
        if synthetic_steps < 1 or images_per_class < 1:
            raise ValueError(
                "FedREDefense synthetic_steps and images_per_class must be at least 1"
            )
        for name, value in {
            "image_lr": image_lr,
            "label_lr": label_lr,
            "teacher_lr": teacher_lr,
            "teacher_lr_lr": teacher_lr_lr,
        }.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"FedREDefense {name} must be finite and positive")

        try:
            from .torch_backend import _resolve_device, _torch_module
        except ImportError as exc:  # pragma: no cover - package integrity guard
            raise RuntimeError("FedREDefense requires the PyTorch backend") from exc
        try:
            torch = _torch_module()
        except RuntimeError as exc:
            raise RuntimeError(
                "FedREDefense requires PyTorch for differentiable reconstruction"
            ) from exc

        # Higher-order convolution gradients are substantially more reliable on
        # CPU than MPS.  CUDA remains the preferred official-code execution path.
        requested_device = device
        if (device or "auto").strip().lower() == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = _resolve_device(torch, requested_device)
        self.client_ids = tuple(client_ids)
        self.model_spec = model_spec
        self.threshold = float(threshold)
        self.initial_iterations = int(initial_iterations)
        self.max_iterations = int(max_iterations)
        self.synthetic_steps = int(synthetic_steps)
        self.images_per_class = int(images_per_class)
        self.image_lr = float(image_lr)
        self.label_lr = float(label_lr)
        self.initial_teacher_lr = float(teacher_lr)
        self.teacher_lr_lr = float(teacher_lr_lr)
        self.seed = int(seed)
        self.eps = float(eps)
        self._states: dict[str, _SyntheticState] = {}
        self._client_index = {
            client_id: index for index, client_id in enumerate(self.client_ids)
        }

    def __getstate__(self) -> dict[str, object]:
        """Exclude the imported module so round checkpoints remain picklable."""

        state = dict(self.__dict__)
        state["_checkpoint_device"] = str(self.device)
        state.pop("torch", None)
        state.pop("device", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        from .torch_backend import _resolve_device, _torch_module

        checkpoint_device = str(state.pop("_checkpoint_device", "cpu"))
        self.__dict__.update(state)
        self.torch = _torch_module()
        try:
            self.device = _resolve_device(self.torch, checkpoint_device)
        except RuntimeError:
            self.device = _resolve_device(self.torch, "cpu")

    def evaluate_round(
        self,
        global_params: np.ndarray,
        update_by_client: dict[str, np.ndarray],
        *,
        round_id: int,
    ) -> FedREDefenseResult:
        if not update_by_client:
            return FedREDefenseResult(tuple(), tuple(), {})
        global_vector = np.asarray(global_params, dtype=np.float32).reshape(-1)
        if global_vector.size != self.model_spec.parameter_size:
            raise ValueError(
                "FedREDefense global parameter size does not match model specification"
            )
        errors = {}
        for client_id in sorted(update_by_client):
            update = np.asarray(update_by_client[client_id], dtype=np.float32).reshape(
                -1
            )
            if update.size != global_vector.size:
                raise ValueError(
                    "FedREDefense update size does not match global parameters"
                )
            errors[client_id] = self._reconstruction_error(
                client_id,
                global_vector,
                update,
                round_id=round_id,
            )
        accepted = tuple(
            client_id
            for client_id in sorted(errors)
            if np.isfinite(errors[client_id])
            and errors[client_id] <= self.threshold
        )
        rejected = tuple(
            client_id for client_id in sorted(errors) if client_id not in accepted
        )
        return FedREDefenseResult(accepted, rejected, errors)

    def _reconstruction_error(
        self,
        client_id: str,
        global_vector: np.ndarray,
        update: np.ndarray,
        *,
        round_id: int,
    ) -> float:
        update_norm = float(np.dot(update.astype(np.float64), update.astype(np.float64)))
        if update_norm <= self.eps:
            return 0.0
        first_observation = client_id not in self._states
        state = self._states.get(client_id)
        if state is None:
            state = self._initial_state(client_id)

        torch = self.torch
        images = torch.tensor(
            state.images,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        labels = torch.tensor(
            state.labels,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        synthetic_lr = torch.tensor(
            float(state.teacher_lr),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        optimizer = torch.optim.SGD(
            [
                {"params": [images], "lr": self.image_lr},
                {"params": [labels], "lr": self.label_lr},
                {"params": [synthetic_lr], "lr": self.teacher_lr_lr},
            ],
            momentum=0.5,
        )
        start = torch.as_tensor(
            np.ascontiguousarray(global_vector),
            dtype=torch.float32,
            device=self.device,
        )
        target = start + torch.as_tensor(
            np.ascontiguousarray(update),
            dtype=torch.float32,
            device=self.device,
        )
        denominator = torch.nn.functional.mse_loss(
            start,
            target,
            reduction="sum",
        ).detach() + self.eps
        iterations = (
            self.initial_iterations if first_observation else self.max_iterations
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self.seed
            + self._client_index.get(client_id, 0) * 100_003
            + int(round_id) * 10_007
        )
        error = torch.tensor(float("inf"), device=self.device)
        for _iteration in range(iterations):
            student = start.detach().clone().requires_grad_(True)
            chunks: list[object] = []
            for _step in range(self.synthetic_steps):
                if not chunks:
                    permutation = torch.randperm(
                        int(images.shape[0]),
                        generator=generator,
                    ).to(device=self.device)
                    chunks = list(torch.split(permutation, int(images.shape[0])))
                indices = chunks.pop()
                logits = self._functional_forward(
                    student,
                    images.index_select(0, indices),
                )
                soft_labels = torch.softmax(labels.index_select(0, indices), dim=1)
                loss = torch.mean(
                    -soft_labels * torch.log_softmax(logits, dim=1)
                )
                gradient = torch.autograd.grad(
                    loss,
                    student,
                    create_graph=True,
                )[0]
                student = student - synthetic_lr * gradient

            error = (
                torch.nn.functional.mse_loss(
                    student,
                    target,
                    reduction="sum",
                )
                + self.eps
            ) / denominator
            if float(error.detach().cpu().item()) < self.threshold:
                break
            optimizer.zero_grad()
            error.backward()
            optimizer.step()

        self._states[client_id] = _SyntheticState(
            images=images.detach().cpu().numpy().astype(np.float32, copy=True),
            labels=labels.detach().cpu().numpy().astype(np.float32, copy=True),
            teacher_lr=float(synthetic_lr.detach().cpu().item()),
        )
        return float(error.detach().cpu().item())

    def _initial_state(self, client_id: str) -> _SyntheticState:
        index = self._client_index.get(client_id, 0)
        rng = np.random.default_rng(self.seed + 19_999 + index * 1_009)
        sample_count = self.model_spec.num_classes * self.images_per_class
        images = rng.normal(
            0.0,
            1.0,
            size=(sample_count, *self.model_spec.input_shape),
        ).astype(np.float32)
        # The official configuration sets label_init=0, so one-hot labels are
        # multiplied by zero and optimized as unconstrained soft-label logits.
        labels = np.zeros(
            (sample_count, self.model_spec.num_classes),
            dtype=np.float32,
        )
        return _SyntheticState(images, labels, self.initial_teacher_lr)

    def _functional_forward(self, flat_params, features):
        from .torch_backend import _torch_forward

        parameters = []
        offset = 0
        for shape in self._parameter_shapes():
            size = int(np.prod(shape))
            parameters.append(flat_params.narrow(0, offset, size).reshape(shape))
            offset += size
        if offset != int(flat_params.numel()):
            raise ValueError(
                "FedREDefense flat parameter size does not match model specification"
            )
        return _torch_forward(
            self.torch,
            tuple(parameters),
            features,
            self.model_spec,
        )

    def _parameter_shapes(self) -> tuple[tuple[int, ...], ...]:
        spec = self.model_spec
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


__all__ = ["FedREDefense", "FedREDefenseResult"]
