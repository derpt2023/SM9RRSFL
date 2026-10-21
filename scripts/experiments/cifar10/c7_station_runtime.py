"""Thread-isolated C7 observation/attack adapter for the independent station run.

Install once around both tuning phases, outside or inside the dashboard.  No
per-task monkey patching, global Torch device/RNG changes, or training-source
edits are needed.  The original checkpoint writer remains authoritative.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time
import traceback
import uuid

import numpy as np

from sm9rrsfl import experiments, fair_tuning, fl
from .attack_stability_attack import attack_scope
from .attack_stability_worker import (
    RoundMetricEvaluator, _array_digest, _check_finite, _check_scalar_fields,
    _digest, _json_safe, _utc, _write_json,
)


PROFILE = {"id": "C7", "optimizer": "adam", "lr": .001, "boost": 20.,
           "stealth_steps": 10, "scope": "weights"}
PROTOCOL = "cifar_c7_station_runtime_v1"
_INSTALL_LOCK = threading.Lock()


def semantic_config(config):
    values = dict(config) if isinstance(config, dict) else asdict(config)
    values.pop("device", None)
    values.pop("sm9_workers", None)
    return values


def task_key(config):
    return _digest(semantic_config(config))


def task_directory(output_dir, phase, config):
    if phase not in ("validation", "final"):
        raise ValueError("C7 station phase must be validation or final")
    return Path(output_dir) / "c7_runtime" / phase / task_key(config)


def runtime_fingerprint_context(profile):
    if profile != PROFILE:
        raise ValueError("This runtime requires the frozen C7 attack profile")
    return {"c7_runtime_protocol": PROTOCOL, "c7_attack_profile": deepcopy(profile)}


def _append(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(value), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class StationMetricEvaluator(RoundMetricEvaluator):
    """Use the supplied phase dataset, labelling official-test metrics honestly."""

    def __init__(self, dataset, config, phase):
        super().__init__(dataset, config)
        self.phase = phase
        self.evaluation_scope = ("official_test" if phase == "final"
                                 else "training_validation")

    def target_manifest(self):
        return {
            "stage": self.phase, "evaluation_scope": self.evaluation_scope,
            "evaluation_count": len(self.dataset.y_test),
            "official_test_used_for_selection": False,
            "training_target_indices": self.aux_indices.tolist(),
            "legacy_evaluation_indices": self.legacy_indices.tolist(),
            "source_transfer_indices": self.source_indices.tolist(),
            "training_target_count": len(self.aux_indices),
            "legacy_target_count": len(self.legacy_indices),
            "source_transfer_count": len(self.source_indices),
            "training_target_is_optimizer_input": True,
            "transfer_target_is_optimizer_input": False,
        }


class _Task:
    def __init__(self, directory, dataset, config, phase, task_id, run_fingerprint,
                 profile, data_identity):
        self.directory = directory
        self.config = config
        self.evaluator = StationMetricEvaluator(dataset, config, phase)
        self.attempt_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.details = {}
        self.calls = {}
        self.metrics = []
        self.completed = -1
        self.train_called = False
        identity_core = {
            "protocol": PROTOCOL, "stage": phase, "task_id": task_id,
            "task_key": task_key(config), "semantic_config": semantic_config(config),
            "run_fingerprint": run_fingerprint, "profile": deepcopy(profile),
            "attack_enabled": config.malicious_ratio > 0,
            "evaluation_scope": self.evaluator.evaluation_scope,
            "data_identity": data_identity,
        }
        self.fingerprint = _digest(identity_core)
        self.identity = dict(identity_core, fingerprint=self.fingerprint, config=asdict(config))
        directory.mkdir(parents=True, exist_ok=True)
        identity_path = directory / "task.json"
        if identity_path.exists():
            previous = _read(identity_path)
            if previous.get("fingerprint") != self.fingerprint:
                raise ValueError("C7 task identity/profile/data/fingerprint changed; preserve old output")
        else:
            _write_json(identity_path, self.identity)
        for filename in ("identity.json", "targets.json"):
            expected = self.identity if filename == "identity.json" else self.evaluator.target_manifest()
            path = directory / filename
            if path.exists():
                actual = _read(path)
                valid = (actual.get("fingerprint") == self.fingerprint if filename == "identity.json"
                         else actual == expected)
                if not valid:
                    raise ValueError(f"C7 {filename} differs from this task")
            else:
                _write_json(path, expected)
        # An explicitly empty audit file is observable zero calls, not a missing file.
        (directory / "attack_audit.jsonl").touch(exist_ok=True)
        prior = _read(directory / "status.json") if (directory / "status.json").exists() else {}
        self.elapsed_before = float(prior.get("elapsed_seconds", 0.))
        if (directory / "metrics.json").exists():
            stored = _read(directory / "metrics.json")
            if stored.get("fingerprint") != self.fingerprint:
                raise ValueError("C7 extended metric identity differs")
            self.metrics = stored["rounds"]
        self.status("running")
        self.event("start")

    def common(self):
        return {"task_id": self.identity["task_id"], "task_key": self.identity["task_key"],
                "stage": self.identity["stage"], "fingerprint": self.fingerprint,
                "run_fingerprint": self.identity["run_fingerprint"],
                "attempt_id": self.attempt_id, "device": str(self.config.device),
                "timestamp_utc": _utc()}

    def event(self, kind, **values):
        _append(self.directory / "attempts.jsonl", dict(self.common(), event=kind, **values))

    def status(self, status, **values):
        _write_json(self.directory / "status.json", {
            **self.common(), "status": status, "completed_round": self.completed,
            "requested_rounds": self.config.rounds, "config": asdict(self.config),
            "profile": self.identity["profile"],
            "elapsed_seconds": self.elapsed_before + time.monotonic() - self.started,
            **values,
        })

    def prefix(self, completed):
        prefix = [row for row in self.metrics if row["round"] <= completed]
        if [row["round"] for row in prefix] != list(range(completed + 1)):
            raise ValueError("C7 checkpoint is missing its complete extended metric prefix")
        return prefix

    def begin_training(self, resume):
        self.train_called = True
        self.completed = int(resume["completed_round"]) if resume else -1
        self.metrics = self.prefix(self.completed) if resume else []
        self.event("resume" if resume else "fresh", completed_round=self.completed)

    def audit(self, audit):
        if not self.details:
            raise RuntimeError("C7 attack audit arrived outside a task/client context")
        row = {**audit, **self.common(), **self.details}
        _append(self.directory / "attack_audit.jsonl", row)
        round_id = row["round"]
        self.calls[round_id] = self.calls.get(round_id, 0) + 1

    def observe(self, state, callback):
        round_id = int(state["completed_round"])
        # Crypto finalization may publish the same completed model twice.
        if not self.metrics or self.metrics[-1]["round"] != round_id:
            row = self.evaluator.evaluate(state)
            row["attack_calls_this_round"] = self.calls.get(round_id, 0)
            self.metrics = [r for r in self.metrics if r["round"] < round_id] + [row]
            _write_json(self.directory / "metrics.json", {
                "fingerprint": self.fingerprint, "stage": self.identity["stage"],
                "evaluation_scope": self.evaluator.evaluation_scope,
                "official_test_used_for_selection": False, "rounds": self.metrics,
            })
        # Metric commit precedes any durable training checkpoint. Extra rows from
        # an interrupted attempt are discarded when its checkpoint is loaded.
        if callback is not None:
            callback(state)
        self.completed = round_id
        self.status("running", nonfinite_updates=int(state.get("nonfinite_updates", 0)))

    def finish(self, result):
        completed = int(result.stopped_round)
        self.metrics = self.prefix(completed)
        self.completed = completed
        summary = result.summary_dict()
        _check_scalar_fields(summary, "station_result")
        if result.nonfinite_updates:
            _check_finite(np.nan, "station_result_nonfinite_updates")
        status = "complete" if completed == self.config.rounds else "incomplete"
        payload = {
            **self.common(), "status": status, "completed_round": completed,
            "requested_rounds": self.config.rounds, "config": asdict(self.config),
            "profile": self.identity["profile"], "summary": summary,
            "malicious_clients": list(result.malicious_clients),
            "blacklisted_clients": list(result.blacklisted_clients),
            "attack_seconds": result.stage_timings.attack_seconds,
            "nonfinite_updates": result.nonfinite_updates,
            "final_metrics": self.metrics[-1],
            "metric_scope": self.evaluator.target_manifest(),
            "elapsed_seconds": self.elapsed_before + time.monotonic() - self.started,
            "training_runtime_seconds": result.runtime_seconds,
            "terminal_checkpoint_replayed": not self.train_called,
        }
        _write_json(self.directory / "result.json", payload)
        self.status(status, nonfinite_updates=result.nonfinite_updates,
                    training_runtime_seconds=result.runtime_seconds,
                    final_metrics=self.metrics[-1],
                    terminal_checkpoint_replayed=not self.train_called)
        self.event(status, completed_round=completed)
        if status != "complete":
            raise RuntimeError(
                f"C7 task ended at round {completed}/{self.config.rounds}; "
                "incomplete result preserved, phase completion refused")

    def fail(self, error):
        audit = getattr(error, "audit", None)
        status = "numerical_failure" if audit is not None else "failed"
        evidence = {**self.common(), "status": status, "completed_round": self.completed,
                    "error_type": type(error).__name__, "error": str(error),
                    "traceback": traceback.format_exc(), "audit": audit,
                    "client_context": getattr(error, "_c7_client_context", self.details)}
        first = self.directory / "first_failure.json"
        if not first.exists():
            _write_json(first, evidence)
        numerical = self.directory / "first_numerical_failure.json"
        if audit is not None and not numerical.exists():
            _write_json(numerical, evidence)
        self.event("failure", evidence=evidence)
        self.status(status, error_type=type(error).__name__, error=str(error))


@contextmanager
def c7_station_runtime(output_dir, profile=PROFILE):
    """Observe threaded tuning phases and install the same C7 for every method."""
    required_context = runtime_fingerprint_context(profile)
    profile = deepcopy(profile)
    if not _INSTALL_LOCK.acquire(blocking=False):
        raise RuntimeError("C7 station runtime is already installed in this process")
    owner = os.getpid()
    worker = threading.local()
    registry_lock = threading.Lock()
    active = set()
    data_cache = {}
    originals = {
        "phase": fair_tuning.execute_resumable_tuning_phase,
        "measure": fair_tuning.run_measured_experiment,
        "train": experiments.run_experiment,
        "context": fl._maybe_torch_context,
        "local": fl._local_train_client_delta,
        "attack": fl._alternating_minimization_client_delta,
    }

    def task():
        return getattr(worker, "task", None)

    def data_identity(dataset, phase, config):
        key = (id(dataset), phase, config.attack_source_label)
        with registry_lock:
            if key not in data_cache:
                arrays = {}
                for name in ("x_train", "y_train", "x_test", "y_test", "x_attack", "y_attack"):
                    value = np.asarray(getattr(dataset, name))
                    arrays[name] = {"sha256": _array_digest(value),
                                    "shape": list(value.shape), "dtype": value.dtype.str}
                data_cache[key] = (dataset, {
                    "arrays": arrays, "evaluation_count": len(dataset.y_test),
                    "source_transfer_count": int(np.count_nonzero(
                        np.asarray(dataset.y_test) == config.attack_source_label)),
                    "attack_auxiliary_count": len(dataset.y_attack),
                })
            return data_cache[key][1]

    @wraps(originals["phase"])
    def phase(dataset, tasks, args, **kwargs):
        given = kwargs.get("fingerprint_context") or {}
        if any(given.get(k) != v for k, v in required_context.items()):
            raise ValueError("C7 profile/runtime protocol must be in the phase fingerprint_context")
        if kwargs.get("jobs", 1) > 1 and kwargs.get("backend_description") != "torch:cuda":
            raise ValueError("C7 station parallelism requires the shared-process CUDA thread scheduler")
        return originals["phase"](dataset, tasks, args, **kwargs)

    @wraps(originals["measure"])
    def measure(dataset, config, **kwargs):
        if os.getpid() != owner:
            raise RuntimeError("C7 station runtime cannot execute in a forked process")
        if task() is not None:
            raise RuntimeError("C7 station tasks cannot nest on the same thread")
        if config.compute_backend != "torch" or config.eval_interval != 1:
            raise ValueError("C7 station requires Torch and an evaluation every round")
        name = kwargs.get("task_id", "")
        phase_name = name.split("_", 1)[0]
        directory = task_directory(output_dir, phase_name, config)
        fingerprint = kwargs.get("run_fingerprint")
        if not fingerprint:
            raise ValueError("C7 station requires resumable phase fingerprints")
        with registry_lock:
            if directory in active:
                raise RuntimeError("The same C7 task is already active on another device")
            active.add(directory)
        current = None
        try:
            current = _Task(directory, dataset, config, phase_name, name, fingerprint,
                            profile, data_identity(dataset, phase_name, config))
            worker.task = current
            result = originals["measure"](dataset, config, **kwargs)
            current.finish(result)
            return result
        except BaseException as error:
            if current is not None:
                try:
                    current.fail(error)
                except Exception as audit_error:
                    # Evidence I/O cannot replace the original CUDA/numerical failure.
                    if hasattr(error, "add_note"):
                        error.add_note(f"C7 failure evidence write also failed: {audit_error}")
            raise
        finally:
            worker.task = None
            with registry_lock:
                active.discard(directory)

    @wraps(originals["train"])
    def train(dataset, config, **kwargs):
        current = task()
        if current is None:
            return originals["train"](dataset, config, **kwargs)
        current.begin_training(kwargs.get("resume_state"))
        callback = kwargs.get("checkpoint_callback")
        kwargs["checkpoint_callback"] = lambda state: current.observe(state, callback)
        return originals["train"](dataset, config, **kwargs)

    @wraps(originals["context"])
    def context(*args, **kwargs):
        value = originals["context"](*args, **kwargs)
        current = task()
        if current is not None:
            current.evaluator.torch_context = value
        return value

    def client_call(original, kind):
        @wraps(original)
        def call(*args, **kwargs):
            current = task()
            if current is None:
                if kind == "attack":
                    raise RuntimeError("C7 attack requires a managed station task")
                return original(*args, **kwargs)
            details = {"round": int(kwargs["round_id"]),
                       "client_idx": int(kwargs["client_idx"]),
                       "client_id": f"client-{kwargs['client_idx']}"}
            current.details = details
            try:
                value = original(*args, **kwargs)
                delta, stats = value
                _check_finite(delta, kind + "_delta", **details)
                _check_finite(stats.loss, kind + "_loss", **details)
                return value
            except BaseException as error:
                error._c7_client_context = details
                if hasattr(error, "audit"):
                    error.audit = {**details, **error.audit}
                raise
            finally:
                current.details = {}
        return call

    def audit(row):
        current = task()
        if current is None:
            raise RuntimeError("C7 audit has no active thread-local task")
        current.audit(row)

    try:
        with attack_scope(profile, on_audit=audit):
            fair_tuning.execute_resumable_tuning_phase = phase
            fair_tuning.run_measured_experiment = measure
            experiments.run_experiment = train
            fl._maybe_torch_context = context
            fl._local_train_client_delta = client_call(originals["local"], "honest")
            fl._alternating_minimization_client_delta = client_call(originals["attack"], "attack")
            try:
                yield
            finally:
                fair_tuning.execute_resumable_tuning_phase = originals["phase"]
                fair_tuning.run_measured_experiment = originals["measure"]
                experiments.run_experiment = originals["train"]
                fl._maybe_torch_context = originals["context"]
                fl._local_train_client_delta = originals["local"]
                fl._alternating_minimization_client_delta = originals["attack"]
    finally:
        _INSTALL_LOCK.release()
