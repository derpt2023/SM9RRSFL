"""Read-only numerical runtime and hardware provenance.

This module never changes environment variables or Torch runtime flags.
Importing it does not load Torch; taking a snapshot may load optional numerical
libraries, but never initializes CUDA or changes the requested device.  The
existing package initialization policy is independent of this metadata module.
"""

from __future__ import annotations

import csv
import importlib
import os
import platform
import shutil
import subprocess
from typing import Any


_RECORDED_ENVIRONMENT = (
    "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
    "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
)


def _load_torch():
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise RuntimeError("PyTorch metadata is unavailable") from exc


def _json_value(value):
    """Keep metadata JSON-compatible across NumPy/backend versions."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return str(value)


def _optional_call(function, default=None):
    if function is None:
        return default
    try:
        return function()
    except (RuntimeError, OSError, AttributeError, TypeError, ValueError):
        return default


def _nvidia_information() -> dict[str, Any]:
    """Query installed driver tooling without creating a CUDA context."""

    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "gpus": [], "reason": "nvidia-smi unavailable"}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=index,uuid,name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "gpus": [], "reason": type(exc).__name__}
    if result.returncode != 0:
        return {"available": False, "gpus": [],
                "reason": f"nvidia-smi exit status {result.returncode}"}
    gpus = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 5:
            continue
        index, uuid, name, driver, memory = (value.strip() for value in row)
        gpus.append({"physical_index": index, "uuid": uuid, "name": name,
                     "driver_version": driver, "memory_total_mib": memory})
    return {"available": bool(gpus), "gpus": gpus,
            "scope": "physical devices reported by nvidia-smi; not CUDA visibility order"}


def _native_sm9_information() -> dict[str, Any]:
    try:
        from . import sm9_backend

        native = getattr(sm9_backend, "_native", None)
        return {
            "available": sm9_backend.available(),
            "backend": sm9_backend.backend_name(),
            "bridge_abi_version": getattr(native, "ABI_VERSION", None),
            "module_path": getattr(native, "__file__", None),
            # The extension does not expose its linked GmSSL release number.
            "gmssl_version": None,
        }
    except (ImportError, OSError, RuntimeError) as exc:
        return {"available": False, "reason": type(exc).__name__}


def _numpy_information() -> dict[str, Any]:
    try:
        np = importlib.import_module("numpy")
    except (ImportError, OSError):
        return {"available": False}
    np_config = getattr(np, "__config__", None)
    config = getattr(np_config, "CONFIG", None)
    if config is None:
        get_info = getattr(np_config, "get_info", None)
        config = ({name: _optional_call(lambda: get_info(name)) for name in
                   ("blas_opt_info", "blas_ilp64_opt_info", "lapack_opt_info")}
                  if get_info is not None else None)
    return {"available": True, "version": str(np.__version__),
            "build_configuration": _json_value(config)}


def numerical_environment(device=None, *, torch_module=None) -> dict[str, Any]:
    """Snapshot numerical libraries, flags and hardware without initializing CUDA.

    ``device`` records the requested placement, without changing it.  Logical
    CUDA device properties are included only if CUDA is already initialized;
    physical GPU/driver information is obtained independently via nvidia-smi
    when installed.  Missing optional information is represented by null or an
    explicit unavailable result, so CPU-only machines remain supported.
    """

    metadata: dict[str, Any] = {
        "python": {"version": platform.python_version(),
                   "implementation": platform.python_implementation()},
        "platform": platform.platform(),
        "requested_device": None if device is None else str(device),
        "environment": {name: os.environ.get(name) for name in _RECORDED_ENVIRONMENT},
        "numpy": _numpy_information(),
        "native_sm9": _native_sm9_information(),
        "nvidia": _nvidia_information(),
    }
    try:
        torch = _load_torch() if torch_module is None else torch_module
    except RuntimeError:
        metadata["torch"] = {"available": False}
        return metadata
    cuda = torch.cuda
    cuda_initialized = bool(cuda.is_initialized())
    cudnn = torch.backends.cudnn
    matmul = torch.backends.cuda.matmul
    torch_config = getattr(torch, "__config__", None)
    metadata["torch"] = {
        "available": True,
        "version": str(torch.__version__),
        "cuda_version": getattr(torch.version, "cuda", None),
        "hip_version": getattr(torch.version, "hip", None),
        "cudnn_version": _optional_call(getattr(cudnn, "version", None)),
        "cuda_initialized": cuda_initialized,
        "deterministic_algorithms": _optional_call(
            getattr(torch, "are_deterministic_algorithms_enabled", None)),
        "deterministic_warn_only": _optional_call(
            getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)),
        "cudnn_deterministic": bool(cudnn.deterministic),
        "cudnn_benchmark": bool(cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(matmul.allow_tf32),
        "cudnn_allow_tf32": bool(cudnn.allow_tf32),
        "float32_matmul_precision": _optional_call(
            getattr(torch, "get_float32_matmul_precision", None)),
        "num_threads": _optional_call(getattr(torch, "get_num_threads", None)),
        "num_interop_threads": _optional_call(getattr(torch, "get_num_interop_threads", None)),
        "build_configuration": _optional_call(getattr(torch_config, "show", None)),
        "logical_cuda_devices": [],
        "logical_cuda_devices_status": "not queried before CUDA initialization",
    }
    if cuda_initialized:
        try:
            properties = []
            for index in range(cuda.device_count()):
                prop = cuda.get_device_properties(index)
                properties.append({"logical_index": index, "name": prop.name,
                                   "total_memory_bytes": int(prop.total_memory),
                                   "compute_capability": [int(prop.major), int(prop.minor)],
                                   "uuid": str(prop.uuid) if hasattr(prop, "uuid") else None})
            metadata["torch"]["logical_cuda_devices"] = properties
            metadata["torch"]["logical_cuda_devices_status"] = "queried"
        except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
            metadata["torch"]["logical_cuda_devices_status"] = type(exc).__name__
    return _json_value(metadata)


__all__ = ["numerical_environment"]
