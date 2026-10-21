"""Deterministic client probes and a bounded native SM9 thread benchmark.

Only public diagnostics are written.  The benchmark's shared initial crypto
state stays in memory; no private checkpoint is exported by this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np

from sm9rrsfl import fl
from sm9rrsfl.datasets import (
    ImageDataset, load_image_dataset, partition_clients,
    stratified_training_three_way_split,
)
from sm9rrsfl.model import model_spec_for_dataset
from sm9rrsfl.torch_backend import TorchTrainingContext


SELECTION_RULE = (
    "Use only the fixed 80%-malicious set. Include smallest and largest clients, "
    "the median-size client in each nonempty source-present/source-absent group, "
    "then evenly spaced sample-count rank quantiles, filling rank gaps greedily. "
    "Break ties by client index; execute in decreasing sample count."
)
_DATA_CACHE = {}
_CONTEXT_CACHE = {}
_INITIALIZED = False


def _safe(value):
    """Keep strict JSON without hiding observed NaN/Inf as zero."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), sort_keys=True, indent=2,
                                    allow_nan=False) + "\n")
    os.replace(temporary, path)


def select_representatives(labels, partitions, malicious_indices, count=16,
                           source_label=5):
    """Choose coverage using sample metadata only, never an attack outcome."""
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError("representative count must be a positive integer")
    labels = np.asarray(labels)
    indices = sorted(set(int(i) for i in malicious_indices))
    if not indices or any(i < 0 or i >= len(partitions) for i in indices):
        raise ValueError("malicious client indices must be nonempty and in range")
    ordered = sorted(indices, key=lambda i: (len(partitions[i]), i))
    wanted = min(int(count), len(ordered))
    selected = []

    def add(index):
        if len(selected) < wanted and index not in selected:
            selected.append(index)

    add(ordered[0])
    add(ordered[-1])
    present = {i: bool(np.any(labels[np.asarray(partitions[i], dtype=np.int64)]
                             == source_label)) for i in ordered}
    for has_source in (False, True):
        group = [i for i in ordered if present[i] == has_source]
        if group:
            add(group[(len(group) - 1) // 2])
    for rank in np.rint(np.linspace(0, len(ordered) - 1, wanted)).astype(int):
        add(ordered[int(rank)])
    ranks = {i: rank for rank, i in enumerate(ordered)}
    while len(selected) < wanted:
        remaining = [i for i in ordered if i not in selected]
        add(max(remaining, key=lambda i: (
            min(abs(ranks[i] - ranks[j]) for j in selected), -ranks[i])))
    return sorted(selected, key=lambda i: (-len(partitions[i]), i))


def initialize_probe_worker(data_config=None):
    """Pool initializer: at most two Torch CPU threads per probe worker."""
    global _INITIALIZED
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # An already initialized process cannot reset its interop pool.
        pass
    _INITIALIZED = True
    if data_config is not None:
        _load_data(data_config)


def _load_data(data_config):
    key = json.dumps(data_config, sort_keys=True)
    if key not in _DATA_CACHE:
        source = load_image_dataset(
            data_config.get("dataset", "cifar10"), data_config.get("data_dir"),
            download=bool(data_config.get("download", False)),
            train_limit=data_config.get("train_samples", data_config.get("train_limit")),
            test_limit=data_config.get("test_samples", data_config.get("test_limit")),
            seed=int(data_config.get("seed", 42)),
        )
        validation_fraction = float(data_config.get("validation_fraction", .15))
        split = stratified_training_three_way_split(
            source, seed=int(data_config.get("split_seed", 20261013)),
            train_fraction=.95 - validation_fraction,
            calibration_fraction=validation_fraction, attack_fraction=.05,
        )
        _DATA_CACHE[key] = split.calibration_dataset
    return key, _DATA_CACHE[key]


def _context(data_config, config, device):
    data_key, dataset = _load_data(data_config)
    key = (data_key, config.partition, config.num_clients,
           config.dirichlet_alpha, config.seed, device)
    if key not in _CONTEXT_CACHE:
        partitions = partition_clients(
            dataset.y_train, config.num_clients, strategy=config.partition,
            dirichlet_alpha=config.dirichlet_alpha, seed=config.seed,
        )
        _CONTEXT_CACHE[key] = (
            TorchTrainingContext(dataset, partitions,
                                 spec=model_spec_for_dataset(dataset), device=device),
            partitions,
        )
    context, partitions = _CONTEXT_CACHE[key]
    return dataset, context, partitions


def _load_anchor(path, expected_sha256=None):
    content = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("anchor changed after immutable probe identity was checked")
    with np.load(io.BytesIO(content), allow_pickle=False) as arrays:
        keys = [key for key in ("params", "vector", "global_vector") if key in arrays]
        if len(keys) != 1:
            raise ValueError("anchor must contain one params/vector/global_vector array")
        vector = np.ascontiguousarray(arrays[keys[0]], dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("anchor must be a finite one-dimensional model vector")
    return vector


def probe_rankable(result):
    """Partial coverage and nonfinite evidence never qualify a candidate."""
    rows = result.get("actual_results", [])
    return (
        result.get("status") == "complete"
        and result.get("all_finite") is True
        and len(rows) == result.get("requested_clients")
        and bool(rows)
        and all(row.get("status") == "complete" and row.get("all_finite") is True
                for row in rows)
    )


def run_probe(request):
    """Run one profile/partition/anchor with identical fresh client starts.

    ``request`` contains profile, config (ExperimentConfig fields), data_config,
    anchor_path, clients, device, outdir, task_id, fingerprint (root protocol),
    and deadline_epoch. Scientific identity is immutable; only the deadline may
    change on a same-directory resume. Terminal results are reused.
    """
    import fcntl
    device = request.get("device", "cpu")
    if device not in {"cpu", "mps"}:
        raise ValueError("local stability probes require cpu or mps")
    task_id = str(request.get("task_id", request.get("id", "probe")))
    if not task_id or Path(task_id).name != task_id or task_id in {".", ".."}:
        raise ValueError("task_id must be a single safe path component")
    out = Path(request["outdir"]).resolve() / task_id
    clients = [int(i) for i in request["clients"]]
    if not clients or len(set(clients)) != len(clients):
        raise ValueError("probe clients must be nonempty and unique")
    if not isinstance(request.get("fingerprint"), str) or not request["fingerprint"]:
        raise ValueError("probe requires the root protocol fingerprint")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "probe.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        anchor_path = Path(request["anchor_path"]).resolve()
        identity = {
            "schema": "attack_stability_probe_identity_v1",
            "root_fingerprint": request["fingerprint"],
            "config": asdict(fl.ExperimentConfig(**request["config"])),
            "profile": request["profile"], "data_config": request["data_config"],
            "clients": clients, "device": device,
            "anchor_id": request.get("anchor_id", anchor_path.stem),
            "anchor_path": str(anchor_path),
            "anchor_file_sha256": hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
        }
        identity["fingerprint"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        identity_path = out / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("existing probe identity differs; use a new task_id/output")
        else:
            if any(path.name != "probe.lock" for path in out.iterdir()):
                raise ValueError("probe output has evidence without immutable identity")
            _write_json(identity_path, identity)
        if not (out / "request.json").exists():
            _write_json(out / "request.json", request)
        result_path = out / "result.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous.get("fingerprint") != identity["fingerprint"]:
                raise ValueError("stored probe result fingerprint differs")
            if previous.get("status") in {"complete", "numerical_failure", "failed"}:
                return previous
        return _run_probe_locked(request, out, identity)


def _run_probe_locked(request, out, identity):
    from .attack_stability_attack import NumericalFailure, run_profile_attack

    started = time.time()
    config = fl.ExperimentConfig(**request["config"])
    device = request.get("device", "cpu")
    task_id = str(request.get("task_id", request.get("id", "probe")))
    clients = [int(i) for i in request["clients"]]
    deadline = float(request.get("deadline_epoch", math.inf))
    previous_result_path = out / "result.json"
    previous_runtime = (float(json.loads(previous_result_path.read_text()).get(
        "runtime_seconds", 0.)) if previous_result_path.exists() else 0.)
    result = {
        "task_id": task_id, "profile_id": request["profile"]["id"],
        "fingerprint": identity["fingerprint"],
        "root_fingerprint": identity["root_fingerprint"],
        "anchor_file_sha256": identity["anchor_file_sha256"],
        "profile": request["profile"], "partition": config.partition,
        "anchor_id": request.get("anchor_id", Path(request["anchor_path"]).stem),
        "anchor_path": str(request["anchor_path"]), "device": device,
        "seed": config.seed, "round": 25, "requested_clients": len(clients),
        "clients": clients, "status": "running", "all_finite": False,
        "actual_results": [], "started_epoch": started,
        "selection_rule": SELECTION_RULE,
    }
    previous_rows = {}
    for path in sorted(out.glob("client_*.json")):
        row = json.loads(path.read_text())
        if row.get("fingerprint") != identity["fingerprint"]:
            raise ValueError("stored client probe fingerprint differs")
        index = row.get("client_idx")
        if index not in clients or path.name != f"client_{index:03d}.json":
            raise ValueError("stored client probe index differs")
        if row.get("status") not in {"complete", "partial", "numerical_failure", "failed"}:
            raise ValueError("stored client probe has an unsupported status")
        previous_rows[index] = row
    result["actual_results"] = [previous_rows[i] for i in clients if i in previous_rows
                               and previous_rows[i]["status"] != "partial"]

    def finish(status):
        rows = result["actual_results"]
        complete = [r for r in rows if r["status"] == "complete"]
        finite = (len(complete) == len(clients)
                  and all(r["all_finite"] for r in complete))
        result.update(status=status, all_finite=finite,
                      runtime_seconds=previous_runtime + time.time() - started,
                      attempt_runtime_seconds=time.time() - started,
                      completed_clients=len(complete),
                      mean_target_success=(float(np.mean(
                          [r["target_success"] for r in complete])) if complete else None),
                      mean_target_confidence=(float(np.mean(
                          [r["target_confidence"] for r in complete])) if complete else None))
        worst = max(complete, key=lambda r: (r["parameter_growth_ratio"],
                                             -r["client_idx"]), default=None)
        result.update(worst_client_idx=worst["client_idx"] if worst else None,
                      worst_growth_ratio=worst["parameter_growth_ratio"] if worst else None)
        result["rankable"] = probe_rankable(result)
        _write_json(out / "result.json", result)
        return result

    # A process can die after the durable failure row but before the summary.
    # Recover that evidence before considering any client retry.
    for row in result["actual_results"]:
        if row["status"] in {"numerical_failure", "failed"}:
            return finish(row["status"])
    if (out / "setup_failure.json").exists():
        setup_failure = json.loads((out / "setup_failure.json").read_text())
        if setup_failure.get("fingerprint") != identity["fingerprint"]:
            raise ValueError("stored setup failure fingerprint differs")
        result["failure"] = setup_failure
        return finish("failed")
    if len(result["actual_results"]) == len(clients):
        return finish("complete")
    if time.time() >= deadline:
        return finish("partial")
    try:
        if not _INITIALIZED:
            initialize_probe_worker()
        dataset, ctx, partitions = _context(request["data_config"], config, device)
    except RuntimeError as exc:
        result["failure"] = {
            "status": "failed", "stage": "setup", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
            "fingerprint": identity["fingerprint"],
        }
        _write_json(out / "setup_failure.json", result["failure"])
        return finish("failed")
    if any(i < 0 or i >= len(partitions) for i in clients):
        raise ValueError("probe client index is outside the partition")
    anchor = _load_anchor(request["anchor_path"], identity["anchor_file_sha256"])
    if anchor.size != ctx.spec.parameter_size:
        raise ValueError("anchor parameter count does not match the model")
    result["anchor_sha256"] = hashlib.sha256(anchor.tobytes()).hexdigest()
    target_indices = fl._select_attack_target_indices(dataset, config)
    result["target_auxiliary_indices"] = target_indices.tolist()
    result["target_count"] = int(len(target_indices))
    lr = config.lr * config.lr_decay ** 24
    for index in clients:
        if index in previous_rows and previous_rows[index]["status"] == "complete":
            continue
        if time.time() >= deadline:
            return finish("partial")
        if index in previous_rows:
            old_path = out / f"client_{index:03d}.json"
            _write_json(out / "partial_attempts" /
                        f"client_{index:03d}_{old_path.stat().st_mtime_ns}.json",
                        previous_rows[index])
        client_started = time.time()
        row = {
            "client_idx": index, "anchor_id": result["anchor_id"],
            "fingerprint": identity["fingerprint"],
            "samples": len(partitions[index]),
            "source_label_samples": int(np.count_nonzero(
                dataset.y_train[partitions[index]] == config.attack_source_label)),
            "local_seed": config.seed + 25 * 1009 + index,
            "round_lr": lr, "status": "running", "all_finite": False,
        }
        try:
            delta, stats, audit = run_profile_attack(
                ctx, anchor, request["profile"], client_idx=index,
                target_indices=target_indices, target_label=config.attack_target_label,
                lr=lr, attack_epochs=config.attack_epochs, batch_size=config.batch_size,
                stealth_steps=config.attack_stealth_steps, boost=config.attack_boost,
                distance_weight=config.attack_distance_weight, seed=row["local_seed"],
                deadline_epoch=deadline,
            )
            finite = bool(ctx.torch.isfinite(delta).all().item())
            values = (stats.loss, audit["target_success"], audit["target_confidence"],
                      audit["max_parameter_growth_ratio"])
            finite = finite and all(v is not None and math.isfinite(float(v)) for v in values)
            row.update(
                status="complete" if finite else "numerical_failure",
                all_finite=finite, loss=stats.loss, audit=audit,
                target_success=audit["target_success"],
                target_confidence=audit["target_confidence"],
                parameter_growth_ratio=audit.get(
                    "max_weight_growth_ratio", audit["max_parameter_growth_ratio"]),
                growth_measure="max_weight_layer_l2_ratio",
                normal_steps=audit["normal_steps"], target_steps=audit["target_steps"],
                target_delta_l2_total=audit["target_delta_l2_total"],
            )
        except NumericalFailure as exc:
            row.update(status="numerical_failure", audit=exc.audit, error=str(exc))
        except TimeoutError as exc:
            row.update(status="partial", error=str(exc),
                       audit=getattr(exc, "audit", None))
        except RuntimeError as exc:
            row.update(status="failed", error_type=type(exc).__name__,
                       error=str(exc), traceback=traceback.format_exc())
        row["runtime_seconds"] = time.time() - client_started
        result["actual_results"].append(row)
        result["actual_results"].sort(key=lambda r: clients.index(r["client_idx"]))
        _write_json(out / f"client_{index:03d}.json", row)
        if row["status"] != "complete":
            return finish(row["status"])
        # Live public evidence is usable even if a later worker is interrupted.
        finish("running")
    return finish("complete")


def benchmark_signature(result, model_hashes):
    """Exclude timing/config differences, retain all scientific result fields."""
    payload = {
        "round_model_hashes": list(model_hashes),
        "records": [asdict(row) for row in result.records],
        "diagnostics": [asdict(row) for row in result.diagnostics],
        "final_accuracy": result.final_accuracy, "final_error": result.final_error,
        "stopped_round": result.stopped_round,
        "malicious_clients": list(result.malicious_clients),
        "blacklisted_clients": list(result.blacklisted_clients),
        "nonfinite_updates": result.nonfinite_updates,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def sm9_threads_benchmark(output_dir, deadline_epoch):
    """Compare 1/4/8 native SM9 workers on a small real CPU CIFAR-shaped run.

    Eight clients and four rounds cover the three-round clean history and one
    detector round. All arms reuse the same round-zero crypto state in memory,
    removing random task-tag changes from this concurrency-only comparison.
    """
    import torch
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    started = time.time()
    stop_at = min(float(deadline_epoch), started + 120)
    rng = np.random.default_rng(20261020)
    dataset = ImageDataset(
        rng.normal(0, .4, (16, 3, 32, 32)).astype(np.float32),
        np.arange(16, dtype=np.int64) % 4,
        rng.normal(0, .4, (4, 3, 32, 32)).astype(np.float32),
        np.arange(4, dtype=np.int64), name="cifar10", num_classes=10,
    )
    config = fl.ExperimentConfig(
        method="sm9rrs", malicious_ratio=0., num_clients=8, rounds=4,
        local_epochs=1, batch_size=2, lr=.001, compute_backend="torch",
        device="cpu", partition="iid", attack="none", attack_start_round=4,
        detector_window=3, crypto_mode="sm9", checkpoint_interval=1,
        early_stop=False, seed=301,
    )
    initial = None
    rows = []
    report = {
        "status": "running", "selected_workers": 1, "all_equivalent": False,
        "fixture": {"clients": 8, "rounds": 4, "train_samples": 16,
                    "input_shape": [3, 32, 32], "crypto_mode": "sm9",
                    "detector_window": 3, "attack": "none"},
        "interpretation": (
            "Local CPU fixture equivalence only, not a guarantee for all full runs. "
            "All arms start at round zero with the same private crypto state held "
            "only in memory. An unranked single-worker warmup precedes all three "
            "measured arms, which each resume round zero. No keys or private "
            "checkpoint are written."
        ),
        "arms": rows,
    }
    try:
        for role, workers in (("warmup", 1), ("measured", 1),
                              ("measured", 4), ("measured", 8)):
            if time.time() >= stop_at:
                break
            hashes = {}
            arm_start = time.time()

            def checkpoint(state):
                nonlocal initial
                if initial is None and state["completed_round"] == 0:
                    initial = deepcopy(state)
                hashes[int(state["completed_round"])] = hashlib.sha256(
                    np.ascontiguousarray(state["params"]).tobytes()).hexdigest()
                if time.time() >= stop_at:
                    raise TimeoutError("SM9 benchmark budget reached at round boundary")

            try:
                result = fl.run_experiment(
                    dataset, replace(config, sm9_workers=workers),
                    resume_state=deepcopy(initial) if initial is not None else None,
                    checkpoint_callback=checkpoint,
                )
                # Resuming at round zero does not call its checkpoint callback.
                hashes.setdefault(0, hashlib.sha256(
                    np.ascontiguousarray(initial["params"]).tobytes()).hexdigest())
                arm = {
                    "workers": workers, "status": "complete",
                    "maximum_packet_workers": min(workers, config.num_clients),
                    "accepted_clients_by_round": {
                        str(record.round): record.accepted_updates
                        for record in result.records if record.round > 0},
                    "runtime_seconds": time.time() - arm_start,
                    "signature": benchmark_signature(result, sorted(hashes.items())),
                    "round_model_sha256": sorted(hashes.items()),
                    "nonfinite_updates": result.nonfinite_updates,
                    "diagnostic_records": len(result.diagnostics),
                }
            except Exception as exc:
                arm = {"workers": workers, "status": "failed",
                       "runtime_seconds": time.time() - arm_start,
                       "error_type": type(exc).__name__, "error": str(exc)}
            if role == "warmup":
                report["warmup"] = arm
            else:
                rows.append(arm)
            if arm["status"] != "complete":
                break
        complete = [row for row in rows if row["status"] == "complete"]
        same = (len(complete) == 3 and
                len({row["signature"] for row in complete}
                    | {report.get("warmup", {}).get("signature")}) == 1 and
                all(row["nonfinite_updates"] == 0 for row in complete))
        report.update(status="complete" if len(complete) == 3 else "partial",
                      all_equivalent=same,
                      selected_workers=min(complete, key=lambda r: r["runtime_seconds"]
                                           )["workers"] if same else 1)
    finally:
        torch.set_num_threads(previous_threads)
        report["runtime_seconds"] = time.time() - started
        _write_json(Path(output_dir) / "sm9_threads_benchmark.json", report)
    return report
