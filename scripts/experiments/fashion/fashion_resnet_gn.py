"""Opt-in Fashion-MNIST ResNet-18/GN adapter for a separate experiment protocol.

Importing this module changes nothing. Only a dedicated new worker may call
``install_runtime()``. The original source files, explicit legacy model specs,
and processes which do not install this adapter retain their old behavior.
Adapted independently from the frozen cifar_resnet_gn v1 architecture; only
the input stem changes from three channels to one. Native 28x28 images are
not resized or repeated. The adapter is process-wide: do not mix protocols
concurrently in one worker.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache, wraps
from math import prod

import numpy as np


ARCHITECTURE = "fashion_resnet18_gn2_v1"


@dataclass(frozen=True)
class FashionResNet18GNSpec:
    input_shape: tuple[int, int, int] = (1, 28, 28)
    num_classes: int = 10
    architecture: str = ARCHITECTURE
    groups: int = 2
    norm_epsilon: float = 1e-5

    def __post_init__(self):
        if (self.input_shape != (1, 28, 28) or self.num_classes != 10
                or self.architecture != ARCHITECTURE or self.groups != 2
                or self.norm_epsilon != 1e-5):
            raise ValueError("ResNet18-GN v1 fixes Fashion-MNIST28, ten classes, GN2 and epsilon=1e-5")

    @property
    def feature_dim(self):
        return 512

    @property
    def parameter_size(self):
        return sum(prod(shape) for _, shape in parameter_layout())

    @property
    def svd_matrix_offset(self):
        return self.parameter_size - self.feature_dim * self.num_classes - self.num_classes

    @property
    def svd_matrix_shape(self):
        # Both Ours and Ding13 expect columns to represent classes.
        return self.feature_dim, self.num_classes


SPEC = FashionResNet18GNSpec()


@lru_cache(maxsize=1)
def parameter_layout():
    """Stable flat layout; all GN affine parameters participate in FL."""
    entries = []

    def conv_norm(name, incoming, outgoing, kernel):
        entries.extend(((name + ".conv", (outgoing, incoming, kernel, kernel)),
                        (name + ".gn_weight", (outgoing,)),
                        (name + ".gn_bias", (outgoing,))))

    conv_norm("stem", 1, 64, 3)
    incoming = 64
    for stage, outgoing in enumerate((64, 128, 256, 512)):
        for block in range(2):
            name = f"stage{stage + 1}.block{block + 1}"
            conv_norm(name + ".first", incoming, outgoing, 3)
            conv_norm(name + ".second", outgoing, outgoing, 3)
            if incoming != outgoing:
                conv_norm(name + ".shortcut", incoming, outgoing, 1)
            incoming = outgoing
    entries.extend((("classifier.weight", (512, 10)), ("classifier.bias", (10,))))
    return tuple(entries)


def parameter_shapes(spec=SPEC):
    _check_spec(spec)
    return tuple(shape for _, shape in parameter_layout())


def protocol_descriptor():
    """Include this descriptor AND this source hash in the new manifest."""
    return {
        "architecture": ARCHITECTURE, "dataset": "fashion_mnist", "input_shape": [1, 28, 28],
        "input_preprocessing": "float32_div255_no_resize_no_channel_repeat_no_augmentation",
        "stage_channels": [64, 128, 256, 512], "blocks_per_stage": [2, 2, 2, 2],
        "stem": "3x3_stride1_no_maxpool", "normalization": "GroupNorm",
        "groups": 2, "norm_epsilon": 1e-5, "running_buffers": False,
        "classifier_layout": "features_by_classes_then_bias",
        "initialization": "numpy_default_rng_conv_kaiming_fan_out_gn_one_zero_head_normal_0.01_v1",
        "parameter_count": SPEC.parameter_size, "compute_backend": "torch",
    }


def _is_resnet(spec):
    return getattr(spec, "architecture", None) == ARCHITECTURE


def _check_spec(spec):
    if not isinstance(spec, FashionResNet18GNSpec) or spec != SPEC:
        raise ValueError("expected the frozen ResNet18-GN v1 model specification")


def init_params(*, seed=0, spec=SPEC):
    """Deterministic float32 initialization without changing global RNG state."""
    _check_spec(spec)
    rng = np.random.default_rng(seed)
    vector = np.empty(spec.parameter_size, dtype=np.float32)
    offset = 0
    for name, shape in parameter_layout():
        size = prod(shape)
        destination = vector[offset:offset + size].reshape(shape)
        if name.endswith(".gn_weight"):
            destination.fill(1.)
        elif name.endswith(".gn_bias") or name == "classifier.bias":
            destination.fill(0.)
        else:
            scale = np.sqrt(2. / (shape[0] * shape[2] * shape[3])) if len(shape) == 4 else .01
            destination[:] = rng.standard_normal(shape, dtype=np.float32) * np.float32(scale)
        offset += size
    return vector


def forward(torch, params, x, spec=SPEC):
    """Functional forward preserves autograd through the original flat trainer."""
    _check_spec(spec)
    if x.ndim != 4 or tuple(x.shape[1:]) != spec.input_shape:
        raise ValueError("ResNet18-GN expects NCHW images of shape [N,1,28,28]")
    shapes = parameter_shapes(spec)
    if len(params) != len(shapes) or any(tuple(p.shape) != shape for p, shape in zip(params, shapes)):
        raise ValueError("ResNet18-GN parameters do not match the frozen layout")
    functional = torch.nn.functional
    cursor = 0

    def conv_norm(value, stride=1, padding=1):
        nonlocal cursor
        weight, gain, bias = params[cursor:cursor + 3]
        cursor += 3
        value = functional.conv2d(value, weight, stride=stride, padding=padding)
        return functional.group_norm(value, spec.groups, gain, bias, spec.norm_epsilon)

    value = torch.relu(conv_norm(x))
    for stage in range(4):
        for block in range(2):
            downsample = stage > 0 and block == 0
            residual = value
            value = torch.relu(conv_norm(value, stride=2 if downsample else 1))
            value = conv_norm(value)
            if downsample:
                residual = conv_norm(residual, stride=2, padding=0)
            value = torch.relu(value + residual)
    value = functional.adaptive_avg_pool2d(value, 1).flatten(1)
    return value.matmul(params[cursor]) + params[cursor + 1]


_PATCHES = []


def runtime_installed():
    return bool(_PATCHES)


def install_runtime():
    """Install only in the new worker before building models/running any task.

    Idempotent in a dedicated worker. Use ``runtime()`` for scoped tests. This
    does not change optimizers, attacks, aggregation or detector decisions.
    """
    if runtime_installed():
        return
    from sm9rrsfl import fl, model, torch_backend

    def patch(module, name, replacement):
        _PATCHES.append((module, name, getattr(module, name), replacement))
        setattr(module, name, replacement)

    old_spec = model.model_spec_for_dataset

    def select_spec(dataset):
        if str(getattr(dataset, "name", "")).lower() == "fashion_mnist":
            if tuple(getattr(dataset, "input_shape", ())) != SPEC.input_shape or getattr(dataset, "num_classes", 10) != 10:
                raise ValueError("new Fashion-MNIST protocol requires 1x28x28 images and ten classes")
            return SPEC
        return old_spec(dataset)

    old_init = model.init_params

    @wraps(old_init)
    def initialize(**kwargs):
        if _is_resnet(kwargs.get("spec")):
            return init_params(seed=kwargs.get("seed", 0), spec=kwargs["spec"])
        return old_init(**kwargs)

    old_shapes, old_forward = torch_backend._parameter_shapes, torch_backend._torch_forward

    def shapes(spec):
        return parameter_shapes(spec) if _is_resnet(spec) else old_shapes(spec)

    def dispatch_forward(torch, params, x, spec):
        return forward(torch, params, x, spec) if _is_resnet(spec) else old_forward(torch, params, x, spec)

    old_context = fl._maybe_torch_context

    def context(dataset, indices, spec, config):
        if _is_resnet(spec) and config.compute_backend != "torch":
            raise ValueError("ResNet18-GN requires explicit compute_backend='torch'; no NumPy fallback")
        return old_context(dataset, indices, spec, config)

    def guard(function, *, torch_capable):
        @wraps(function)
        def checked(*args, **kwargs):
            if _is_resnet(kwargs.get("spec")):
                _check_spec(kwargs["spec"])
                if not torch_capable or kwargs.get("compute_backend", "numpy") != "torch":
                    raise ValueError("ResNet18-GN has no NumPy path; use explicit torch/resident APIs")
            return function(*args, **kwargs)
        return checked

    patch(model, "model_spec_for_dataset", select_spec)
    patch(fl, "model_spec_for_dataset", select_spec)
    patch(model, "init_params", initialize)
    patch(fl, "init_params", initialize)
    patch(torch_backend, "_parameter_shapes", shapes)
    patch(torch_backend, "_torch_forward", dispatch_forward)
    patch(fl, "_maybe_torch_context", context)
    for name in ("accuracy", "targeted_metrics", "local_train_delta", "predict",
                 "vector_to_params", "alternating_minimization_delta"):
        original = getattr(model, name)
        replacement = guard(original, torch_capable=name in {"accuracy", "targeted_metrics", "local_train_delta"})
        patch(model, name, replacement)
        if hasattr(fl, name):
            patch(fl, name, replacement)


def uninstall_runtime():
    """Restore exactly the functions replaced here; adapters must unwind LIFO."""
    for module, name, _, replacement in _PATCHES:
        if getattr(module, name) is not replacement:
            raise RuntimeError(f"another runtime adapter changed {module.__name__}.{name}; uninstall it first")
    for module, name, original, _ in reversed(_PATCHES):
        setattr(module, name, original)
    _PATCHES.clear()


@contextmanager
def runtime():
    already_installed = runtime_installed()
    install_runtime()
    try:
        yield
    finally:
        if not already_installed:
            uninstall_runtime()
