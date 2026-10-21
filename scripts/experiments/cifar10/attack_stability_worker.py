"""Isolated, resumable workers for the versioned attack-stability screen.

The original experiment loop and its scientific state are reused unchanged.
This module must run in a dedicated process: its observation hooks are scoped
to that process and restored even on numerical failure or a budget stop.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import tempfile
import time
import traceback

import numpy as np

from sm9rrsfl import fl, model
from sm9rrsfl.datasets import load_image_dataset, stratified_training_three_way_split
from sm9rrsfl.ours_calibration import split_metadata


PROTOCOL = "cifar_attack_stability_task_v1"


class BudgetStop(Exception):
    """Interrupt unfinished work without invalidating a complete checkpoint."""


class WorkerNumericalFailure(RuntimeError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__(str(audit.get("stage", "nonfinite worker state")))


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _atomic(path, writer):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", mode="wb", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, value):
    data = json.dumps(_json_safe(value), sort_keys=True, indent=2,
                      allow_nan=False).encode()
    _atomic(path, lambda handle: handle.write(data))


def _write_pickle(path, value):
    _atomic(path, lambda handle: pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL))


def _finite(value):
    if type(value).__module__.startswith("torch"):
        return bool(value.detach().isfinite().all().cpu().item())
    return bool(np.isfinite(np.asarray(value)).all())


def _check_finite(value, stage, **context):
    if not _finite(value):
        raise WorkerNumericalFailure({"stage": stage, **context})


def _check_scalar_fields(row, stage, **context):
    for name, value in row.items():
        if isinstance(value, (float, np.floating)) and not math.isfinite(value):
            raise WorkerNumericalFailure(
                {"stage": stage, "field": name, "value": str(value), **context})


def task_identity(request):
    """Separate scientific identity from device placement and wall-clock budget."""
    config = asdict(fl.ExperimentConfig(**request["config"]))
    config.pop("device")
    identity = {
        "protocol": PROTOCOL,
        "root_fingerprint": request["fingerprint"],
        "config": config,
        "profile": request.get("profile"),
        "data_config": request["data_config"],
    }
    return {**identity, "fingerprint": _digest(identity)}


def _load_dataset(data_config):
    raw = load_image_dataset(
        data_config["dataset"], data_config["data_dir"], download=False,
        train_limit=data_config["train_samples"],
        test_limit=data_config["test_samples"], seed=data_config["seed"])
    fraction = data_config["validation_fraction"]
    split = stratified_training_three_way_split(
        raw, seed=data_config["split_seed"], train_fraction=0.95 - fraction,
        calibration_fraction=fraction, attack_fraction=0.05)
    # Only this training-derived holdout is exposed to training/evaluation.
    return split.calibration_dataset, split_metadata(split, data_config["split_seed"])


def _array_digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    view = memoryview(array).cast("B")
    for start in range(0, len(view), 8 * 1024 * 1024):
        digest.update(view[start:start + 8 * 1024 * 1024])
    return digest.hexdigest()


def _dataset_identity(dataset, contract):
    return {
        "contract": contract,
        "arrays": {name: _array_digest(getattr(dataset, name))
                   for name in ("x_train", "y_train", "x_test", "y_test",
                                "x_attack", "y_attack")},
    }


class RoundMetricEvaluator:
    """Keep the legacy one-image metric separate from fit and transfer metrics."""

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.spec = model.model_spec_for_dataset(dataset)
        # Clean controls use exactly the same target selection as attacked runs.
        self.aux_indices = fl._select_attack_target_indices(dataset, config, required=True)
        self.legacy_indices = fl._select_evaluation_target_indices(dataset, config)
        self.source_indices = np.flatnonzero(
            np.asarray(dataset.y_test) == config.attack_source_label)
        if not len(self.legacy_indices) or not len(self.source_indices):
            raise ValueError("Training holdout must contain the evaluation source label")
        self.torch_context = None
        self._tensor_indices = {}

    def target_manifest(self):
        return {
            "official_test_used_for_selection": False,
            "training_target_indices": self.aux_indices.tolist(),
            "legacy_validation_indices": self.legacy_indices.tolist(),
            "transfer_validation_indices": self.source_indices.tolist(),
            "training_target_count": len(self.aux_indices),
            "legacy_target_count": len(self.legacy_indices),
            "source_transfer_count": len(self.source_indices),
            "training_target_is_optimizer_input": True,
            "transfer_target_is_optimizer_input": False,
        }

    def _target_metrics(self, vector, group, indices):
        context = self.torch_context
        if context is None:
            features = self.dataset.x_attack if group == "attack" else self.dataset.x_test
            labels = np.full(len(indices), self.config.attack_target_label, dtype=np.int64)
            metrics = model.targeted_metrics(
                vector, features[indices], labels, spec=self.spec,
                compute_backend=self.config.compute_backend, device=self.config.device)
            _check_finite(metrics, "evaluation_metrics", metric_group=group)
            return metrics
        from sm9rrsfl.torch_backend import _torch_forward, _torch_params_from_tensor
        torch = context.torch
        key = (group, tuple(indices))
        if key not in self._tensor_indices:
            self._tensor_indices[key] = torch.as_tensor(
                indices, dtype=torch.long, device=context.device)
        x = context.x_attack if group == "attack" else context.x_test
        x = x.index_select(0, self._tensor_indices[key])
        params = _torch_params_from_tensor(
            torch, context._ensure_global_vector(vector), self.spec,
            requires_grad=False, clone=False)
        count, successes, confidence_sum = len(indices), 0, 0.0
        with torch.no_grad():
            for start in range(0, count, 512):
                logits = _torch_forward(torch, params, x[start:start + 512], self.spec)
                _check_finite(logits, "evaluation_logits", metric_group=group)
                probabilities = torch.softmax(logits, dim=1)
                _check_finite(probabilities, "evaluation_probabilities", metric_group=group)
                successes += int((probabilities.argmax(1) == self.config.attack_target_label)
                                 .sum().cpu().item())
                confidence_sum += float(probabilities[:, self.config.attack_target_label]
                                        .sum().cpu().item())
        return successes / count, confidence_sum / count

    def evaluate(self, state):
        round_id = int(state["completed_round"])
        records = state["records"]
        if not records or records[-1].round != round_id:
            raise ValueError("The stability protocol requires an evaluation every round")
        record = asdict(records[-1])
        _check_finite(state["params"], "round_parameters", round=round_id)
        _check_scalar_fields(record, "round_record", round=round_id)
        if int(state.get("nonfinite_updates", 0)):
            raise WorkerNumericalFailure(
                {"stage": "nonfinite_update_counter", "round": round_id,
                 "nonfinite_updates": int(state["nonfinite_updates"])})
        train_success, train_confidence = self._target_metrics(
            state["params"], "attack", self.aux_indices)
        transfer_success, transfer_confidence = self._target_metrics(
            state["params"], "validation", self.source_indices)
        row = {
            **record,
            "legacy_asr": record["attack_target_success_rate"],
            "legacy_confidence": record["attack_target_confidence"],
            "legacy_target_count": len(self.legacy_indices),
            "training_target_success": train_success,
            "training_target_confidence": train_confidence,
            "training_target_count": len(self.aux_indices),
            "source_transfer_asr": transfer_success,
            "source_transfer_confidence": transfer_confidence,
            "source_transfer_count": len(self.source_indices),
            "cumulative_nonfinite_updates": int(state.get("nonfinite_updates", 0)),
        }
        _check_scalar_fields(row, "extended_round_metrics", round=round_id)
        return row


@contextmanager
def _observation_scope(evaluator, check_budget, audit_context):
    original_context = fl._maybe_torch_context
    original_local = fl._local_train_client_delta
    original_attack = fl._alternating_minimization_client_delta

    def context(*args, **kwargs):
        value = original_context(*args, **kwargs)
        evaluator.torch_context = value
        return value

    def wrap(original, kind):
        def call(*args, **kwargs):
            check_budget()
            details = {"round": int(kwargs["round_id"]),
                       "client_idx": int(kwargs["client_idx"]),
                       "client_id": f"client-{kwargs['client_idx']}"}
            audit_context.clear()
            audit_context.update(details)
            try:
                delta, stats = original(*args, **kwargs)
                _check_finite(delta, kind + "_delta", **details)
                _check_finite(stats.loss, kind + "_loss", **details)
                return delta, stats
            except BaseException as error:
                if hasattr(error, "audit"):
                    error.audit = {**details, **error.audit}
                raise
            finally:
                audit_context.clear()
        return call

    fl._maybe_torch_context = context
    fl._local_train_client_delta = wrap(original_local, "local_train")
    fl._alternating_minimization_client_delta = wrap(original_attack, "attack")
    try:
        yield
    finally:
        fl._maybe_torch_context = original_context
        fl._local_train_client_delta = original_local
        fl._alternating_minimization_client_delta = original_attack


def execute_task(request):
    """Execute one task in a dedicated child; return JSON-compatible status.

    ``output_dir/task_id`` is immutable scientific identity. A partial task can
    resume with a new deadline or device, but never different data, source
    fingerprint, attack profile, or experiment parameters. Numerical failures
    are terminal in that directory, preventing silent retries/selection erasure.
    """
    task_id = str(request["task_id"])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task_id):
        raise ValueError("task_id must be a single safe directory name")
    directory = Path(request["output_dir"]).resolve() / task_id
    directory.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _execute_locked(request, task_id, directory)


def _execute_locked(request, task_id, directory):
    started = time.monotonic()
    identity = task_identity(request)
    fingerprint = identity["fingerprint"]
    identity_file = directory / "task.json"
    if identity_file.exists():
        if json.loads(identity_file.read_text()) != identity:
            raise ValueError("Existing task identity differs; use a new output/task_id")
    else:
        if any(p.name != "worker.lock" for p in directory.iterdir()):
            raise ValueError("Refusing unidentified task artifacts")
        _write_json(identity_file, identity)

    config = replace(fl.ExperimentConfig(**request["config"]),
                     device=request.get("device", request["config"].get("device", "auto")))
    if config.eval_interval != 1 or config.early_stop:
        raise ValueError("Stability tasks require eval_interval=1 and early_stop=False")
    if config.malicious_ratio > 0 and request.get("profile") is None:
        raise ValueError("Attacked tasks require an explicit versioned attack profile")
    paths = {
        "task_dir": str(directory), "checkpoint": str(directory / "checkpoint.pickle"),
        "metrics_path": str(directory / "metrics.json"),
        "result_json": str(directory / "result.json"),
        "result_pickle": str(directory / "result.pickle"),
    }
    status_file = directory / "status.json"
    old_status = json.loads(status_file.read_text()) if status_file.exists() else {}
    if old_status.get("status") in {"complete", "numerical_failure", "failed"}:
        return old_status
    elapsed_before = float(old_status.get("elapsed_seconds", 0.))
    if not math.isfinite(elapsed_before) or elapsed_before < 0:
        raise ValueError("Stored elapsed time must be finite and non-negative")
    completed = -1
    metrics = []
    checkpoint = None
    state = {
        "task_id": task_id, "fingerprint": fingerprint, "status": "running",
        "requested_rounds": config.rounds, "completed_round": completed,
        "started_utc": _utc(), "pid": os.getpid(), "device": config.device, **paths,
        "elapsed_before_seconds": elapsed_before,
        "elapsed_seconds": elapsed_before,
        "timing_has_unclean_resume": (
            old_status.get("timing_has_unclean_resume", False)
            or old_status.get("status") == "running"),
    }
    _write_json(status_file, state)

    def progress(event, **fields):
        print(json.dumps({"type": "attack_stability_progress", "event": event,
                          "task_id": task_id, "timestamp_utc": _utc(), **fields},
                         sort_keys=True, allow_nan=False), flush=True)

    def check_budget():
        if time.time() >= float(request.get("deadline_epoch", float("inf"))):
            raise BudgetStop("Task paused at the last complete checkpoint: deadline reached")

    audit_context = {}

    def on_audit(audit):
        with (directory / "audit.jsonl").open("a") as handle:
            handle.write(json.dumps(_json_safe({"task_id": task_id, **audit_context,
                                               **audit}), sort_keys=True,
                                    allow_nan=False) + "\n")
            handle.flush()

    try:
        check_budget()
        if config.compute_backend in ("torch", "auto"):
            import torch
            torch.set_num_threads(2)
        dataset, contract = _load_dataset(request["data_config"])
        actual_data = _dataset_identity(dataset, contract)
        data_path = directory / "dataset_identity.json"
        if data_path.exists() and json.loads(data_path.read_text()) != actual_data:
            raise ValueError("Dataset content changed for the same task identity")
        _write_json(data_path, actual_data)
        evaluator = RoundMetricEvaluator(dataset, config)
        _write_json(directory / "targets.json", evaluator.target_manifest())
        if Path(paths["checkpoint"]).exists():
            with Path(paths["checkpoint"]).open("rb") as handle:
                checkpoint = pickle.load(handle)
            if checkpoint.get("_attack_stability_fingerprint") != fingerprint:
                raise ValueError("Checkpoint identity does not match task")
            completed = int(checkpoint["completed_round"])
            _check_finite(checkpoint["params"], "resume_parameters", round=completed)
            if Path(paths["metrics_path"]).exists():
                stored_metrics = json.loads(Path(paths["metrics_path"]).read_text())
                if stored_metrics.get("fingerprint") != fingerprint:
                    raise ValueError("Extended metric identity does not match task")
                metrics = stored_metrics["rounds"]
                metrics = [r for r in metrics if r["round"] <= completed]
            expected = list(range(completed + 1))
            if [row["round"] for row in metrics] != expected:
                raise ValueError("Checkpoint is missing its complete extended metric prefix")
        progress("start", completed_round=completed, requested_rounds=config.rounds)

        def checkpoint_callback(raw_state):
            nonlocal completed, checkpoint, metrics
            round_id = int(raw_state["completed_round"])
            row = evaluator.evaluate(raw_state)
            metrics = [r for r in metrics if r["round"] < round_id] + [row]
            # Metrics precede the state; a crash before state replacement leaves
            # a harmless extra row, discarded on recovery above.
            _write_json(paths["metrics_path"], {
                "fingerprint": fingerprint, "official_test_used_for_selection": False,
                "rounds": metrics})
            checkpoint = dict(raw_state, _attack_stability_fingerprint=fingerprint)
            # Publish required public anchors before committing the resumable
            # state. A crash cannot advance the checkpoint past a missing anchor.
            if round_id in (0, 24):
                anchor = directory / f"anchor_r{round_id}.npz"
                if not anchor.exists():
                    _atomic(anchor, lambda handle: np.savez(
                        handle, params=np.asarray(raw_state["params"], dtype=np.float32)))
            _write_pickle(paths["checkpoint"], checkpoint)
            completed = round_id
            state.update(completed_round=completed,
                         elapsed_seconds=elapsed_before + time.monotonic() - started)
            _write_json(status_file, state)
            progress("round", completed_round=completed, requested_rounds=config.rounds,
                     accuracy=row["accuracy"],
                     training_target_success=row["training_target_success"],
                     source_transfer_asr=row["source_transfer_asr"])
            if completed < config.rounds:
                check_budget()

        with ExitStack() as stack:
            if request.get("profile") is not None:
                from scripts.experiments.cifar10.attack_stability_attack import attack_scope
                stack.enter_context(attack_scope(request["profile"], on_audit=on_audit))
            stack.enter_context(_observation_scope(evaluator, check_budget, audit_context))
            result = fl.run_experiment(dataset, config, resume_state=checkpoint,
                                       checkpoint_callback=checkpoint_callback)
        if result.stopped_round != config.rounds:
            raise RuntimeError("Task ended before its requested horizon")
        _check_scalar_fields(result.summary_dict(), "result")
        result = replace(result, runtime_seconds=elapsed_before + time.monotonic() - started)
        _write_pickle(paths["result_pickle"], result)
        state.update(status="complete", completed_round=result.stopped_round,
                     summary=result.summary_dict(), final_metrics=metrics[-1],
                     nonfinite_updates=result.nonfinite_updates)
        _write_json(paths["result_json"], {
            **state, "profile": request.get("profile"),
            "metric_scope": evaluator.target_manifest(),
            "qualification": "short-screen; no long-horizon or clean-health certification",
        })
    except (BudgetStop, KeyboardInterrupt) as error:
        state.update(status="partial", error_type=type(error).__name__, error=str(error))
    except Exception as error:
        numerical = hasattr(error, "audit")
        state.update(status="numerical_failure" if numerical else "failed",
                     error_type=type(error).__name__, error=str(error))
        failure = {"timestamp_utc": _utc(), "task_id": task_id,
                   "last_completed_round": completed,
                   "error_type": type(error).__name__, "error": str(error),
                   "audit": getattr(error, "audit", None),
                   "traceback": traceback.format_exc()}
        first_failure = directory / "first_failure.json"
        if not first_failure.exists():
            _write_json(first_failure, failure)
        state["first_failure_path"] = str(first_failure)
        if numerical:
            on_audit({"event": "first_numerical_failure", **failure})
    finally:
        state.update(completed_round=completed, ended_utc=_utc(),
                     attempt_elapsed_seconds=time.monotonic() - started,
                     elapsed_seconds=elapsed_before + time.monotonic() - started)
        _write_json(status_file, state)
        progress("task_status", status=state["status"], completed_round=completed,
                 requested_rounds=config.rounds)
    return _json_safe(state)
