"""Failure evidence for v14; never rerun training or alter CUDA flags."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
from threading import Lock, current_thread
import traceback

from sm9rrsfl.torch_backend import _torch_module


ROOT = Path(__file__).resolve().parents[3]
SOURCES = (
    "run_cifar_v14_with_progress.py",
    "scripts/experiments/cifar10/cifar_fair_gpu_pool.py",
    "scripts/experiments/cifar10/fair_tuning_dashboard.py",
    "scripts/experiments/cifar10/cifar_fair_diagnostics.py",
    "sm9rrsfl/cuda_execution.py", "sm9rrsfl/experiments.py",
    "sm9rrsfl/fair_tuning.py", "sm9rrsfl/fl.py", "sm9rrsfl/torch_backend.py",
    "configs/cifar10_v14_fair_grid.json",
)
_WRITE_LOCK = Lock()


@lru_cache(maxsize=1)
def source_identity():
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        head = "unavailable"
    return {"git_head": head, "source_sha256": {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES
    }}


def _read(call):
    try:
        return call()
    except Exception as exc:
        return {"error_type": type(exc).__name__, "error": str(exc)}


def runtime_info():
    torch = _torch_module()
    return {
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cudnn": _read(torch.backends.cudnn.version),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unmasked"),
    }


def _tensor_info(value):
    # Metadata only: do not synchronize, copy tensor contents, or keep tensors.
    return {"shape": list(value.shape), "device": str(value.device), "dtype": str(value.dtype)}


def failure_context(error):
    context = {}
    frames = []
    tb = error.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        module, name = frame.f_globals.get("__name__"), frame.f_code.co_name
        values = frame.f_locals
        frames.append({"module": module, "function": name, "line": tb.tb_lineno})
        if module == "sm9rrsfl.fl" and name == "run_experiment":
            for key in ("round_id", "client_idx", "identity", "attack_active"):
                if key in values and type(values[key]) in (int, float, bool, str):
                    context[key] = values[key]
        if module == "sm9rrsfl.torch_backend" and name in (
            "local_train_delta_resident", "local_train_delta", "alternating_minimization_delta_resident",
        ):
            context["training_function"] = name
            for key in ("client_idx", "seed", "lr", "epochs", "attack_epochs", "batch_size", "samples", "start", "loss_batches"):
                if key in values and type(values[key]) in (int, float, bool, str):
                    context[key] = values[key]
            for key in ("global_tensor", "logits", "loss", "local_index"):
                if key in values:
                    context[key] = _read(lambda: _tensor_info(values[key]))
            if "params" in values:
                context["parameters"] = _read(lambda: [_tensor_info(p) for p in values["params"]])
        tb = tb.tb_next
    context["frames"] = frames
    return context


def record_failure(error, config, *, checkpoint_dir=None, task_id=None):
    """Save each attempt separately from the phase's overwriteable last_failure."""
    torch = _torch_module()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "thread": current_thread().name, "task_id": task_id,
        "config": asdict(config), "error_type": type(error).__name__, "error": str(error),
        "context": failure_context(error),
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        "runtime": runtime_info(), "code_at_start": source_identity(),
    }
    if str(config.device).startswith("cuda"):
        report["cuda_memory"] = {
            "device": config.device,
            "current_thread_device": _read(torch.cuda.current_device),
            "free_total_bytes": _read(lambda: list(torch.cuda.mem_get_info(config.device))),
            "allocated_bytes": _read(lambda: torch.cuda.memory_allocated(config.device)),
            "reserved_bytes": _read(lambda: torch.cuda.memory_reserved(config.device)),
            "allocator_stats": _read(lambda: {k: v for k, v in torch.cuda.memory_stats(config.device).items()
                if k in ("num_alloc_retries", "num_ooms", "inactive_split_bytes.all.current")}),
        }
    path = Path(checkpoint_dir).parent / "gpu_failures.jsonl" if checkpoint_dir else None
    if path is not None:
        with _WRITE_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(report, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
    print("gpu_worker_failure=" + json.dumps({
        "timestamp_utc": report["timestamp_utc"], "device": config.device,
        "task_id": task_id, "error": str(error), "context": report["context"],
        "cuda_memory": report.get("cuda_memory"), "evidence_path": str(path) if path else None,
    }, sort_keys=True), flush=True)
    return report


def clear_failure_frames(error):
    # Preserve traceback line information while releasing failed GPU tensors.
    # A fatal Future otherwise pins its entire training state during draining.
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        error = error.__cause__ or error.__context__


def diagnose(output_dir):
    """Read JSON evidence only; never unpickle checkpoints or start CUDA work."""
    root = Path(output_dir)
    phases = []
    state_root = root / ".tuning_state"
    for folder in sorted(state_root.glob("*/*")):
        if not folder.is_dir():
            continue
        row = {"state_dir": str(folder)}
        failure = folder / "last_failure.json"
        if failure.exists():
            row["last_failure"] = _read(lambda: json.loads(failure.read_text(encoding="utf-8")))
        for name, limit in (("cuda_recovery.jsonl", 8), ("gpu_failures.jsonl", 4)):
            path = folder / name
            if path.exists():
                # Keep bounded memory even for a long-running experiment log.
                from collections import deque
                def tail():
                    with path.open(encoding="utf-8") as handle:
                        return [json.loads(line) for line in deque(handle, maxlen=limit) if line.strip()]
                row[name] = _read(tail)
        phases.append(row)
    result = {**source_identity(), "output_dir": str(root.resolve()), "output_exists": root.exists(),
              "states": phases, "note": "Historical failure records alone do not prove a new run failed."}
    print("CIFAR_V14_DIAGNOSIS_BEGIN", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("CIFAR_V14_DIAGNOSIS_END", flush=True)
    return result
