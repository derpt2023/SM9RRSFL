"""Training-only, versioned normal-state policy calibration.

Scheme B only builds the bounded candidate space here; its shared tuner does
ALL training and selection. Standalone auto uses the very same scorer,
resumable executor and normal-state candidates, without historical outputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np

from .datasets import stratified_training_three_way_split
from .ours_policy import OursParameters, bounded_candidates

CALIBRATION_SCHEMA_VERSION = 4
CALIBRATION_ALGORITHM_VERSION = "ours-normal-states-v2-aggressive-revocation"


class OursCalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class OursCalibrationArtifact:
    schema_version: int
    algorithm_version: str
    status: str
    calibration_fingerprint: str
    artifact_fingerprint: str
    dataset_digest: str
    created_at_utc: str
    calibration_runtime_seconds: float
    split: dict
    calibration_seeds: tuple
    covered_scenarios: dict
    constraints: dict
    objective: dict
    objective_learning: dict
    selected_parameters: OursParameters | None
    candidate_results: tuple

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        if payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION or payload.get("algorithm_version") != CALIBRATION_ALGORITHM_VERSION:
            raise ValueError("obsolete Ours calibration artifact")
        if payload["artifact_fingerprint"] != _artifact_digest(payload):
            raise ValueError("calibration artifact checksum mismatch")
        parameters = payload["selected_parameters"]
        if parameters is not None:
            parameters = OursParameters(**parameters)
            parameters.validate()
        payload["selected_parameters"] = parameters
        payload["calibration_seeds"] = tuple(payload["calibration_seeds"])
        payload["candidate_results"] = tuple(payload["candidate_results"])
        return cls(**payload)


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode())
    view = memoryview(array).cast("B")
    for start in range(0, len(view), 8 * 1024 * 1024):
        digest.update(view[start:start + 8 * 1024 * 1024])
    return digest.hexdigest()


def _artifact_digest(payload):
    return _digest({k: v for k, v in payload.items() if k != "artifact_fingerprint"})


def split_metadata(split, seed):
    total = len(split.train_indices) + len(split.calibration_indices) + len(split.attack_indices)
    return {
        "source": "original training split only",
        "official_test_used_for_selection": False,
        "shared_training_arrays_across_phases": True,
        "split_seed": seed,
        **{name + "_samples": len(indices) for name, indices in (
            ("train", split.train_indices), ("calibration", split.calibration_indices),
            ("attack_auxiliary", split.attack_indices))},
        **{name + "_indices_digest": _array_digest(indices) for name, indices in (
            ("train", split.train_indices), ("calibration", split.calibration_indices),
            ("attack", split.attack_indices))},
        "train_fraction_actual": len(split.train_indices) / total,
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def validate_targeted_split(split, args):
    """Fail BEFORE a long search when its mandatory ASR cannot be measured."""
    if args.attack != "alternating_minimization":
        raise OursCalibrationError("ASR-based calibration requires alternating_minimization")
    for name, labels in (("attack auxiliary", split.main_dataset.y_attack),
                         ("validation", split.calibration_dataset.y_test),
                         ("official evaluation", split.main_dataset.y_test)):
        available = int(np.count_nonzero(labels == args.attack_source_label))
        if available < args.attack_target_count:
            raise OursCalibrationError(
                f"{name} has {available} source-label samples, fewer than attack_target_count={args.attack_target_count}")


def resolve_or_run_ours_calibration(dataset, args, output_dir, run_fn=None, *,
                                   split=None, split_seed=None):
    from .experiments import build_experiment_configs
    started = perf_counter()
    if args.detector_window < 3 or args.rounds <= args.detector_window:
        raise OursCalibrationError("normal-state calibration requires rounds > K >= 3")
    attack_start = args.attack_start_round or args.detector_window + 2
    if not args.detector_window < attack_start <= args.rounds or args.attack != "alternating_minimization":
        raise OursCalibrationError("calibration needs an attack after the clean K rounds")
    deferred = getattr(args, "ours_calibration_selection_mode", "") == "defer_to_unified_tuner"
    if split_seed is None:
        if split is not None:
            raise OursCalibrationError("an externally supplied split needs its actual split_seed")
        split_seed = args.seed
    split = split or stratified_training_three_way_split(dataset, seed=split_seed)
    validate_targeted_split(split, args)
    metadata = split_metadata(split, split_seed)
    main = split.main_dataset
    validation = split.calibration_dataset
    candidates = bounded_candidates(args.calibration_candidate_budget)
    ratios = tuple(args.calibration_ratios)
    if 0.0 not in ratios or not any(r > 0 for r in ratios):
        raise OursCalibrationError("calibration needs clean and attacked ratios")
    seeds = () if deferred else tuple((args.seed + 104729 * (i + 1)) % (2**31 - 1) for i in range(3))
    bases = build_experiment_configs(args)
    unique_bases = {}
    for base in bases:
        if base.method == "sm9rrs":
            normalized = replace(base, malicious_ratio=0.0)
            unique_bases.setdefault(normalized, normalized)
    bases = list(unique_bases.values())
    if not bases:
        raise OursCalibrationError("Ours auto requires method sm9rrs")
    constraints = {
        "min_round_completion_rate": args.calibration_min_round_completion_rate,
        "max_nonfinite_updates": args.calibration_max_nonfinite_updates,
        "max_clean_accuracy_drop": None if deferred else 0.05,
    }
    dataset_digest = _digest({"name": dataset.name, "x": _array_digest(dataset.x_train),
                              "y": _array_digest(dataset.y_train)})
    protocol = {
        "algorithm": CALIBRATION_ALGORITHM_VERSION, "schema": CALIBRATION_SCHEMA_VERSION,
        "dataset": dataset_digest, "split": metadata, "candidates": candidates,
        "base_configs": [asdict(c) for c in bases], "ratios": ratios,
        "seeds": seeds, "deferred": deferred, "constraints": constraints,
    }
    fingerprint = _digest(protocol)
    output = Path(output_dir)
    artifact_path = output / ".ours_calibration" / fingerprint / "ours_parameters.json"
    if artifact_path.exists() and args.resume:
        try:
            artifact = OursCalibrationArtifact.from_dict(json.loads(artifact_path.read_text()))
            if artifact.calibration_fingerprint != fingerprint:
                raise ValueError("calibration protocol mismatch")
        except (ValueError, TypeError, KeyError) as exc:
            print(f"ours_calibration_cache_rejected={exc}", flush=True)
        else:
            _write_json(output / "ours_calibration.json", artifact.to_dict())
            print(f"ours_calibration_cache_hit={fingerprint[:12]}", flush=True)
            return main, artifact
    objective, learning = {}, {"status": "pending_unified_validation"}
    selected = None
    rows = tuple({"parameters": p} for p in candidates)
    if not deferred:
        from .fair_tuning import (
            TuningExperimentTask, prepare_tuning_tasks, execute_resumable_tuning_phase,
            _learn_unified_objective_weights, _matched_fedavg_clean_accuracy, score_trial,
            OBJECTIVE_DEFAULTS, _validation_rows, _write_csv,
        )
        tasks = [
            TuningExperimentTask("ours_calibration", f"sm9rrs-{i:03d}", "sm9rrs",
                                 replace(base, seed=seed, malicious_ratio=ratio, **parameters))
            for i, parameters in enumerate(candidates, 1) for base in bases
            for seed in seeds for ratio in ratios
        ] + [
            TuningExperimentTask("ours_calibration", "fedavg-001", "fedavg",
                                 replace(base, method="fedavg", seed=seed))
            for base in bases for seed in seeds
        ]
        if run_fn is None:
            tasks, jobs, backend, _ = prepare_tuning_tasks(validation, tasks, args)
            pairs, _ = execute_resumable_tuning_phase(
                validation, tasks, args, output_dir=output, jobs=jobs,
                backend_description=backend, progress_enabled=not args.no_progress,
                progress_mode=args.progress_mode, fingerprint_context=protocol)
        else:
            pairs = [(task, run_fn(validation, task.config)) for task in tasks]
        results = {}
        for task, result in pairs:
            results.setdefault(task.candidate_id, []).append(result)
        spec = SimpleNamespace(calibration_ratios=ratios, candidates={"sm9rrs": candidates},
                               **constraints)
        reference = _matched_fedavg_clean_accuracy(results)
        _write_csv(output / "ours_validation_results.csv", _validation_rows(pairs))
        feasible = [
            score_trial("sm9rrs", f"sm9rrs-{i:03d}", p, results[f"sm9rrs-{i:03d}"],
                        objective=OBJECTIVE_DEFAULTS, clean_accuracy_reference=reference, **constraints).row()
            for i, p in enumerate(candidates, 1)
        ]
        _write_csv(output / "ours_candidate_feasibility.csv", feasible)
        # A failed weight search still leaves all feasibility reasons and raw
        # per-seed evidence. No hidden threshold inflation or stale fallback.
        objective, learning = _learn_unified_objective_weights(spec, results)
        trials = [
            score_trial("sm9rrs", f"sm9rrs-{i:03d}", p, results[f"sm9rrs-{i:03d}"],
                        objective=objective, clean_accuracy_reference=reference, **constraints)
            for i, p in enumerate(candidates, 1)
        ]
        rows = tuple({
            key: (None if isinstance(value, float) and not np.isfinite(value) else value)
            for key, value in {**trial.row(), "parameters": trial.parameters}.items()
        } for trial in trials)
        _write_json(output / "ours_selection_report.json", {"candidates": rows, "objective": objective})
        valid = [t for t in trials if t.valid]
        if not valid:
            raise OursCalibrationError("no valid Ours candidate; inspect ours_selection_report.json")
        winner = max(valid, key=lambda t: (t.score, -t.worst_attack_success_rate, t.candidate_id))
        selected = OursParameters(**winner.parameters)
    artifact = OursCalibrationArtifact(
        CALIBRATION_SCHEMA_VERSION, CALIBRATION_ALGORITHM_VERSION,
        "candidate_space_only" if deferred else "frozen", fingerprint, "", dataset_digest,
        datetime.now(timezone.utc).isoformat(), perf_counter() - started, metadata, seeds,
        {"calibration_ratios": ratios, "formal_ratios": tuple(args.ratios),
         "partitions": sorted({c.partition for c in bases}),
         "client_counts": sorted({c.num_clients for c in bases})},
        constraints, objective, learning, selected, rows)
    artifact = replace(artifact, artifact_fingerprint=_artifact_digest(artifact.to_dict()))
    _write_json(artifact_path, artifact.to_dict())
    _write_json(output / "ours_calibration.json", artifact.to_dict())
    print(f"ours_calibration={artifact.status} candidates={len(candidates)} "
          f"train={len(main.y_train)} validation={len(validation.y_test)} "
          f"attack_aux={len(main.y_attack)}", flush=True)
    return main, artifact


def apply_ours_parameters(args, artifact):
    if artifact.status != "frozen" or artifact.selected_parameters is None:
        raise OursCalibrationError("candidate-space artifact is not a frozen selection")
    for name, value in asdict(artifact.selected_parameters).items():
        setattr(args, name, value)
    return args


def calibration_metadata(artifact):
    return {k: v for k, v in artifact.to_dict().items() if k != "candidate_results"}
