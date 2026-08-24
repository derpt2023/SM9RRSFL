"""Automatic, training-only calibration for the SM9-RRS detector.

The public experiment configuration supplies the shared learning/attack
protocol and the detector window ``K``.  This module derives all remaining
Ours-specific parameters once, validates them in closed loop, persists an
auditable artifact, and returns one frozen parameter set for every main-run
scenario.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np

from .datasets import (
    ImageDataset,
    TrainingThreeWaySplit,
    stratified_training_three_way_split,
)
from .fl import ExperimentConfig, ExperimentResult, malicious_client_count


CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_ALGORITHM_VERSION = "ours-offline-v2"
_HUGE_THRESHOLD = 1.0e300
_SCORE_EPS = 1.0e-8
_CLEAN_ALPHA = 0.01
_MIN_CLEAN_ACCEPTANCE = 0.95
_MAX_CLEAN_SUSPICIOUS = 0.05
_MIN_CLEAN_ESS_RATIO = 0.90
_SAFETY_MARGIN = 1.05
_MAX_CLEAN_ENVELOPE_REFINEMENTS = 4
_REFINABLE_INCOMPLETE_RULE = (
    "early_stop disabled; every configured client is blacklisted and counted "
    "as a clean false-positive revocation; detector evidence present; no "
    "non-finite updates"
)
_OBJECTIVE = {
    "clean_accuracy_weight": 0.25,
    "robust_accuracy_weight": 0.50,
    "attack_success_weight": 0.20,
    "clean_false_positive_weight": 0.05,
}


class OursCalibrationError(RuntimeError):
    """No safe frozen Ours parameter set could be produced."""


@dataclass(frozen=True)
class OursParameters:
    q: int
    g0: float
    theta_adj: float
    theta_anc: float
    beta: float
    kappa: float
    h: float
    C_tol: int
    C_max: int
    penalty_factor: float
    recovery_factor: float
    decision_rule: str = "any"


@dataclass(frozen=True)
class OursCalibrationArtifact:
    schema_version: int
    algorithm_version: str
    status: str
    protocol_fingerprint: str
    calibration_fingerprint: str
    artifact_fingerprint: str
    dataset_digest: str
    created_at_utc: str
    calibration_runtime_seconds: float
    split: dict[str, Any]
    calibration_seeds: tuple[int, ...]
    covered_scenarios: dict[str, Any]
    constraints: dict[str, Any]
    objective: dict[str, float]
    selected_parameters: OursParameters
    selected_candidate: dict[str, Any]
    candidate_results: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["calibration_seeds"] = list(self.calibration_seeds)
        payload["candidate_results"] = list(self.candidate_results)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OursCalibrationArtifact":
        if not isinstance(payload, dict):
            raise ValueError("calibration artifact must be a JSON object")
        expected = {item.name for item in fields(cls)}
        if set(payload) != expected:
            raise ValueError("calibration artifact fields do not match the schema")
        parameters = payload.get("selected_parameters")
        if not isinstance(parameters, dict):
            raise ValueError("selected_parameters must be a JSON object")
        return cls(
            schema_version=int(payload["schema_version"]),
            algorithm_version=str(payload["algorithm_version"]),
            status=str(payload["status"]),
            protocol_fingerprint=str(payload["protocol_fingerprint"]),
            calibration_fingerprint=str(payload["calibration_fingerprint"]),
            artifact_fingerprint=str(payload["artifact_fingerprint"]),
            dataset_digest=str(payload["dataset_digest"]),
            created_at_utc=str(payload["created_at_utc"]),
            calibration_runtime_seconds=float(payload["calibration_runtime_seconds"]),
            split=dict(payload["split"]),
            calibration_seeds=tuple(int(item) for item in payload["calibration_seeds"]),
            covered_scenarios=dict(payload["covered_scenarios"]),
            constraints=dict(payload["constraints"]),
            objective={str(key): float(value) for key, value in payload["objective"].items()},
            selected_parameters=OursParameters(**parameters),
            selected_candidate=dict(payload["selected_candidate"]),
            candidate_results=tuple(dict(item) for item in payload["candidate_results"]),
        )


def resolve_or_run_ours_calibration(
    dataset: ImageDataset,
    args,
    output_dir: str | Path,
    run_fn: Callable[[ImageDataset, ExperimentConfig], ExperimentResult] | None = None,
) -> tuple[ImageDataset, OursCalibrationArtifact]:
    """Resolve a matching artifact or run calibration before the main study.

    The returned dataset contains only the 90% federated-training subset plus
    a disjoint training-derived attack auxiliary split.  Its ``x_test`` and
    ``y_test`` remain the untouched official evaluation split.
    """

    calibration_started = perf_counter()
    detector_window = int(_arg(args, "detector_window", 3))
    if detector_window < 2:
        raise OursCalibrationError("K must be at least 2 for automatic calibration")
    rounds = int(_arg(args, "rounds", 0))
    if rounds <= detector_window:
        raise OursCalibrationError(
            "automatic calibration requires rounds > K so the detector has scored observations"
        )
    requested_ratios = _ratios(args)
    if not any(value > 0.0 for value in requested_ratios):
        raise OursCalibrationError(
            "automatic calibration requires at least one attacked malicious ratio"
        )
    if str(_arg(args, "attack", "none")) == "none":
        raise OursCalibrationError(
            "automatic calibration requires an enabled attack for closed-loop validation"
        )
    requested_attack_start = int(_arg(args, "attack_start_round", 0))
    effective_attack_start = requested_attack_start or detector_window + 2
    if effective_attack_start < detector_window + 2:
        raise OursCalibrationError(
            "automatic calibration requires attack_start_round >= K + 2"
        )
    if effective_attack_start > rounds:
        raise OursCalibrationError(
            "automatic calibration requires the effective attack start round "
            "to be within the configured training rounds"
        )
    main_seed = int(_arg(args, "seed", 0))
    split_seed = _derived_seed(main_seed, "ours-calibration-split")
    shadow_seed = _derived_seed(main_seed, "ours-calibration-shadow")
    validation_seed = _derived_seed(main_seed, "ours-calibration-validation")
    if validation_seed == main_seed:
        validation_seed = (validation_seed + 1) % (2**31 - 1)
    if shadow_seed in {main_seed, validation_seed}:
        shadow_seed = (shadow_seed + 1) % (2**31 - 1)

    split = stratified_training_three_way_split(dataset, seed=split_seed)
    dataset_digest = _training_dataset_digest(dataset)
    split_payload = _split_payload(split, split_seed)
    protocol_payload = _protocol_payload(
        dataset,
        args,
        split_seed=split_seed,
        shadow_seed=shadow_seed,
        validation_seed=validation_seed,
    )
    protocol_fingerprint = _json_fingerprint(protocol_payload)
    calibration_fingerprint = _json_fingerprint(
        {
            "protocol_fingerprint": protocol_fingerprint,
            "dataset_digest": dataset_digest,
            "split_indices": {
                key: split_payload[key]
                for key in (
                    "train_indices_digest",
                    "calibration_indices_digest",
                    "attack_indices_digest",
                )
            },
        }
    )
    output_path = Path(output_dir).expanduser().resolve()
    artifact_dir = output_path / ".ours_calibration" / calibration_fingerprint
    artifact_path = artifact_dir / "ours_parameters.json"
    cached = _load_matching_artifact(
        artifact_path,
        protocol_fingerprint=protocol_fingerprint,
        calibration_fingerprint=calibration_fingerprint,
        dataset_digest=dataset_digest,
    )
    if cached is not None:
        print(
            "ours_calibration_cache_hit="
            f"{cached.calibration_fingerprint[:12]} artifact="
            f"{cached.artifact_fingerprint[:12]}",
            flush=True,
        )
        _write_json_atomic(output_path / "ours_calibration.json", cached.to_dict())
        return split.main_dataset, cached

    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir: Path | None = None
    completed_calibration_configs: set[ExperimentConfig] = set()
    finalize_config_checkpoint = None
    if run_fn is None:
        # Reuse the main runner's durable per-round checkpoint format.  A
        # terminal calibration checkpoint is intentionally retained until the
        # immutable artifact exists, so an interrupted multi-hour calibration
        # can replay completed candidates instead of starting over.
        from .experiments import (
            finalize_config_checkpoint as finalize_checkpoint,
            run_measured_experiment,
        )

        finalize_config_checkpoint = finalize_checkpoint
        checkpoint_dir = (
            artifact_dir / ".checkpoints"
            if bool(_arg(args, "resume", True))
            else None
        )

        def run_fn(calibration_dataset, config):
            print(
                "ours_calibration_run="
                f"partition={config.partition} clients={config.num_clients} "
                f"ratio={config.malicious_ratio:.2f} "
                f"q={config.detector_subspace_dim} "
                f"C_tol={config.suspicion_remove_after} "
                f"shadow={not config.detector_enforce}",
                flush=True,
            )
            result = run_measured_experiment(
                calibration_dataset,
                config,
                checkpoint_dir=checkpoint_dir,
                run_fingerprint=calibration_fingerprint,
                retain_success_checkpoint=checkpoint_dir is not None,
            )
            completed_calibration_configs.add(config)
            return result

    partitions = _partitions(args)
    client_counts = _client_counts(args)
    ratios = requested_ratios
    beta = float(math.exp(math.log(0.5) / detector_window))
    penalty = float(min(0.1, 1.0 / max(client_counts)))
    recovery = float(penalty ** -0.5)
    q_values = range(1, min(3, int(dataset.num_classes) - 1) + 1)
    print(
        "ours_calibration_start="
        f"{calibration_fingerprint[:12]} train={len(split.train_indices)} "
        f"validation={len(split.calibration_indices)} "
        f"attack_aux={len(split.attack_indices)} "
        f"partitions={list(partitions)} clients={list(client_counts)} "
        f"ratios={list(ratios)}",
        flush=True,
    )
    derived_by_q: dict[int, dict[str, Any]] = {}
    for q in q_values:
        first_probe = _run_clean_shadow_probe(
            split.calibration_dataset,
            args,
            run_fn,
            partitions=partitions,
            client_counts=client_counts,
            seed=shadow_seed,
            q=q,
            g0=1.0,
            beta=beta,
            kappa=1.0,
            penalty=penalty,
            recovery=recovery,
        )
        gap_values = _diagnostic_values(
            first_probe,
            "spectral_gap",
            detector_window,
        )
        if not gap_values:
            raise OursCalibrationError(
                f"q={q} clean shadow probe produced no scored spectral gaps"
            )
        g0 = _positive_quantile(gap_values, 0.25)
        second_probe = _run_clean_shadow_probe(
            split.calibration_dataset,
            args,
            run_fn,
            partitions=partitions,
            client_counts=client_counts,
            seed=shadow_seed,
            q=q,
            g0=g0,
            beta=beta,
            kappa=1.0,
            penalty=penalty,
            recovery=recovery,
        )
        anchor_values = _diagnostic_values(
            second_probe,
            "anchor_score",
            detector_window,
        )
        if not anchor_values:
            raise OursCalibrationError(
                f"q={q} clean shadow probe produced no scored anchor values"
            )
        kappa = _positive_quantile(anchor_values, 0.90)
        adjacent_maxima = _per_tag_maxima(
            second_probe,
            "adjacent_score",
            detector_window,
        )
        anchor_maxima = _per_tag_maxima(
            second_probe,
            "anchor_score",
            detector_window,
        )
        drift_maxima = _replay_drift_maxima(
            second_probe,
            detector_window=detector_window,
            beta=beta,
            kappa=kappa,
        )
        theta_adj, adjacent_rule = _upper_block_threshold(
            adjacent_maxima,
            tail_probability=_CLEAN_ALPHA / 3.0,
        )
        theta_anc, anchor_rule = _upper_block_threshold(
            anchor_maxima,
            tail_probability=_CLEAN_ALPHA / 3.0,
        )
        h, drift_rule = _upper_block_threshold(
            drift_maxima,
            tail_probability=_CLEAN_ALPHA / 3.0,
        )
        derived_by_q[q] = {
            "g0": g0,
            "beta": beta,
            "kappa": kappa,
            "theta_adj": theta_adj,
            "theta_anc": theta_anc,
            "h": h,
            "threshold_rules": {
                "theta_adj": adjacent_rule,
                "theta_anc": anchor_rule,
                "h": drift_rule,
            },
            "shadow_blocks": len(adjacent_maxima),
        }

    c_tol_values = sorted(
        {
            max(1, min(3, detector_window)),
            max(1, min(5, detector_window)),
        }
    )
    candidate_results: list[dict[str, Any]] = []
    candidate_parameters: dict[str, OursParameters] = {}
    attacked_ratios = tuple(value for value in ratios if value > 0.0)
    for q, derived in derived_by_q.items():
        for c_tol in c_tol_values:
            parameters = OursParameters(
                q=q,
                g0=float(derived["g0"]),
                theta_adj=float(derived["theta_adj"]),
                theta_anc=float(derived["theta_anc"]),
                beta=float(derived["beta"]),
                kappa=float(derived["kappa"]),
                h=float(derived["h"]),
                C_tol=int(c_tol),
                C_max=int(c_tol),
                penalty_factor=penalty,
                recovery_factor=recovery,
            )
            candidate_id = f"q{q}-ctol{c_tol}"
            clean_executions: list[ExperimentResult] = []
            clean_summary: dict[str, Any] = {}
            clean_safety_trace: list[dict[str, Any]] = []
            for refinement in range(_MAX_CLEAN_ENVELOPE_REFINEMENTS + 1):
                clean_executions = _run_closed_loop_candidate(
                    split.calibration_dataset,
                    args,
                    run_fn,
                    parameters=parameters,
                    partitions=partitions,
                    client_counts=client_counts,
                    ratios=(0.0,),
                    seed=validation_seed,
                )
                clean_summary = _score_closed_loop_candidate(
                    candidate_id,
                    clean_executions,
                )
                clean_safety_trace.append(
                    _clean_safety_trace_entry(
                        refinement,
                        parameters,
                        clean_summary,
                    )
                )
                print(
                    "ours_calibration_clean_gate="
                    f"candidate={candidate_id} refinement={refinement} "
                    f"valid={bool(clean_summary['valid'])} "
                    "suspicious="
                    f"{clean_summary['worst_clean_round_suspicious_rate']:.6f} "
                    f"ess={clean_summary['worst_clean_round_ess_ratio']:.6f} "
                    "reasons="
                    f"{','.join(clean_summary['invalid_reasons']) or 'none'}",
                    flush=True,
                )
                if clean_summary["valid"]:
                    break
                if not _clean_gate_can_be_refined(
                    clean_summary,
                    clean_executions,
                ):
                    break
                expanded = _expand_clean_safety_envelope(
                    parameters,
                    clean_executions,
                    detector_window=detector_window,
                )
                if expanded == parameters:
                    break
                parameters = expanded

            if clean_summary.get("valid", False):
                attacked_executions = _run_closed_loop_candidate(
                    split.calibration_dataset,
                    args,
                    run_fn,
                    parameters=parameters,
                    partitions=partitions,
                    client_counts=client_counts,
                    ratios=attacked_ratios,
                    seed=validation_seed,
                )
                summary = _score_closed_loop_candidate(
                    candidate_id,
                    [*clean_executions, *attacked_executions],
                )
            else:
                # Preserve the final failed clean trial in the auditable
                # artifact.  Attack scenarios are intentionally not run for a
                # parameter set that cannot satisfy the clean safety gates.
                summary = clean_summary
            summary["parameters"] = asdict(parameters)
            summary["threshold_rules"] = {
                **dict(derived["threshold_rules"]),
                "closed_loop_clean_envelope": {
                    "rule": (
                        "familywise max(current, 1.05 * maximum observed "
                        "clean closed-loop score + 1e-8)"
                    ),
                    "max_refinements": _MAX_CLEAN_ENVELOPE_REFINEMENTS,
                    "refinements_used": max(0, len(clean_safety_trace) - 1),
                    "refinable_incomplete_rule": _REFINABLE_INCOMPLETE_RULE,
                },
            }
            summary["shadow_blocks"] = int(derived["shadow_blocks"])
            summary["clean_safety_trace"] = clean_safety_trace
            candidate_results.append(summary)
            candidate_parameters[candidate_id] = parameters

    feasible = [item for item in candidate_results if item["valid"]]
    if not feasible:
        concise = "; ".join(
            f"{item['candidate_id']}={','.join(item['invalid_reasons'])}"
            for item in candidate_results
        )
        raise OursCalibrationError(
            "automatic Ours calibration failed closed: no candidate satisfied "
            f"the clean/full-round hard constraints ({concise})"
        )
    selected_summary = max(
        feasible,
        key=lambda item: (
            float(item["score"]),
            float(item["malicious_revocation_rate"]),
            -float(item["first_detection_delay"]),
            -int(item["parameters"]["q"]),
            int(item["parameters"]["C_tol"]),
        ),
    )
    selected = candidate_parameters[str(selected_summary["candidate_id"])]
    constraints = {
        "require_full_rounds": True,
        "require_finite_updates": True,
        "max_clean_false_positive_rate": _CLEAN_ALPHA,
        "min_clean_round_acceptance_rate": _MIN_CLEAN_ACCEPTANCE,
        "max_clean_round_suspicious_rate": _MAX_CLEAN_SUSPICIOUS,
        "min_clean_round_ess_ratio": _MIN_CLEAN_ESS_RATIO,
        "aggregation": "worst partition/client-count/seed/round",
    }
    covered = {
        "partitions": list(partitions),
        "client_counts": list(client_counts),
        "ratios": list(ratios),
    }
    artifact_without_fingerprint = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "algorithm_version": CALIBRATION_ALGORITHM_VERSION,
        "status": "complete",
        "protocol_fingerprint": protocol_fingerprint,
        "calibration_fingerprint": calibration_fingerprint,
        "dataset_digest": dataset_digest,
        "split": split_payload,
        "calibration_seeds": [shadow_seed, validation_seed],
        "covered_scenarios": covered,
        "constraints": constraints,
        "objective": dict(_OBJECTIVE),
        "selected_parameters": asdict(selected),
        "selected_candidate": dict(selected_summary),
        "candidate_results": candidate_results,
    }
    artifact_fingerprint = _json_fingerprint(artifact_without_fingerprint)
    artifact = OursCalibrationArtifact(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        algorithm_version=CALIBRATION_ALGORITHM_VERSION,
        status="complete",
        protocol_fingerprint=protocol_fingerprint,
        calibration_fingerprint=calibration_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        dataset_digest=dataset_digest,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        calibration_runtime_seconds=max(0.0, perf_counter() - calibration_started),
        split=split_payload,
        calibration_seeds=(shadow_seed, validation_seed),
        covered_scenarios=covered,
        constraints=constraints,
        objective=dict(_OBJECTIVE),
        selected_parameters=selected,
        selected_candidate=dict(selected_summary),
        candidate_results=tuple(candidate_results),
    )
    _write_json_atomic(artifact_path, artifact.to_dict())
    _write_json_atomic(output_path / "ours_calibration.json", artifact.to_dict())
    if checkpoint_dir is not None and finalize_config_checkpoint is not None:
        for config in completed_calibration_configs:
            finalize_config_checkpoint(
                checkpoint_dir,
                config,
                calibration_fingerprint,
            )
    print(
        "ours_calibration_complete="
        f"{calibration_fingerprint[:12]} artifact={artifact_fingerprint[:12]} "
        f"candidate={selected_summary['candidate_id']} "
        f"seconds={artifact.calibration_runtime_seconds:.3f}",
        flush=True,
    )
    return split.main_dataset, artifact


def apply_ours_parameters(args, artifact: OursCalibrationArtifact):
    """Mutate an argparse namespace with the artifact's frozen parameters."""

    parameters = artifact.selected_parameters
    mapping = {
        "detector_subspace_dim": parameters.q,
        "detector_gap_threshold": parameters.g0,
        "detector_adjacent_threshold": parameters.theta_adj,
        "detector_anchor_threshold": parameters.theta_anc,
        "detector_drift_memory": parameters.beta,
        "detector_drift_allowance": parameters.kappa,
        "detector_drift_threshold": parameters.h,
        "suspicion_remove_after": parameters.C_tol,
        "suspicion_count_max": parameters.C_max,
        "suspicion_penalty_factor": parameters.penalty_factor,
        "suspicion_recovery_factor": parameters.recovery_factor,
        "detector_decision_rule": parameters.decision_rule,
        # Retain a positive compatibility value even though the two explicit
        # thresholds are authoritative in v3.
        "z_threshold": parameters.theta_adj,
    }
    for name, value in mapping.items():
        setattr(args, name, value)
    return args


def calibration_metadata(artifact: OursCalibrationArtifact) -> dict[str, Any]:
    """Return portable audit metadata suitable for the main run manifest."""

    return {
        "schema_version": artifact.schema_version,
        "algorithm_version": artifact.algorithm_version,
        "status": artifact.status,
        "protocol_fingerprint": artifact.protocol_fingerprint,
        "calibration_fingerprint": artifact.calibration_fingerprint,
        "artifact_fingerprint": artifact.artifact_fingerprint,
        "dataset_digest": artifact.dataset_digest,
        "calibration_seeds": list(artifact.calibration_seeds),
        "split": dict(artifact.split),
        "constraints": dict(artifact.constraints),
        "selected_parameters": asdict(artifact.selected_parameters),
    }


def _run_clean_shadow_probe(
    dataset: ImageDataset,
    args,
    run_fn,
    *,
    partitions: tuple[str, ...],
    client_counts: tuple[int, ...],
    seed: int,
    q: int,
    g0: float,
    beta: float,
    kappa: float,
    penalty: float,
    recovery: float,
) -> list[ExperimentResult]:
    results = []
    for partition in partitions:
        for client_count in client_counts:
            config = _experiment_config(
                args,
                partition=partition,
                num_clients=client_count,
                ratio=0.0,
                seed=seed,
                parameters=OursParameters(
                    q=q,
                    g0=g0,
                    theta_adj=_HUGE_THRESHOLD,
                    theta_anc=_HUGE_THRESHOLD,
                    beta=beta,
                    kappa=kappa,
                    h=_HUGE_THRESHOLD,
                    C_tol=max(1, min(5, int(_arg(args, "detector_window", 3)))),
                    C_max=max(1, min(5, int(_arg(args, "detector_window", 3)))),
                    penalty_factor=penalty,
                    recovery_factor=recovery,
                ),
                enforce=False,
                clean_probe=True,
            )
            results.append(run_fn(dataset, config))
    return results


def _run_closed_loop_candidate(
    dataset: ImageDataset,
    args,
    run_fn,
    *,
    parameters: OursParameters,
    partitions: tuple[str, ...],
    client_counts: tuple[int, ...],
    ratios: tuple[float, ...],
    seed: int,
) -> list[ExperimentResult]:
    results = []
    for partition in partitions:
        for client_count in client_counts:
            for ratio in ratios:
                config = _experiment_config(
                    args,
                    partition=partition,
                    num_clients=client_count,
                    ratio=ratio,
                    seed=seed,
                    parameters=parameters,
                    enforce=True,
                    clean_probe=False,
                )
                results.append(run_fn(dataset, config))
    return results


def _clean_gate_can_be_refined(
    summary: dict[str, Any],
    results: Iterable[ExperimentResult],
) -> bool:
    """Return whether higher evidence thresholds can repair this clean trial.

    Calibration configs always disable accuracy-based early stopping.  In the
    production runner, an incomplete clean SM9-RRS result can therefore still
    be detector-caused: revoking every benign signer finalizes the task and
    exhausts the round loop.  ``ExperimentResult`` has no explicit stop-reason
    field, so require the strongest equivalent evidence available before
    treating ``incomplete_rounds`` as refinable: all clients are blacklisted,
    all are recorded as false-positive revocations, detector evidence exists,
    and no non-finite update occurred.  Any other premature stop remains a
    structural failure.
    """

    refinable = {
        "clean_false_positive_rate",
        "clean_acceptance_rate",
        "clean_suspicious_rate",
        "clean_ess_ratio",
    }
    reasons = set(summary.get("invalid_reasons", ()))
    if not reasons:
        return False
    if reasons <= refinable:
        return True
    if "incomplete_rounds" not in reasons:
        return False
    remaining_reasons = reasons - {"incomplete_rounds"}
    if not remaining_reasons or not remaining_reasons <= refinable:
        return False
    return _clean_task_exhausted_by_false_revocation(results)


def _clean_task_exhausted_by_false_revocation(
    results: Iterable[ExperimentResult],
) -> bool:
    incomplete = [
        result
        for result in results
        if int(result.stopped_round) < int(result.config.rounds)
    ]
    if not incomplete:
        return False
    for result in incomplete:
        config = result.config
        if (
            config.method != "sm9rrs"
            or bool(config.early_stop)
            or abs(float(config.malicious_ratio)) >= 1e-12
            or int(getattr(result, "nonfinite_updates", 0)) != 0
        ):
            return False
        client_count = int(config.num_clients)
        final_record = result.records[-1] if result.records else None
        if final_record is None or int(final_record.round) != int(result.stopped_round):
            return False
        false_revocations = int(
            getattr(final_record, "false_positive_revocations", 0)
        )
        blacklisted = set(getattr(result, "blacklisted_clients", ()))
        expected_clients = {f"client-{index}" for index in range(client_count)}
        if client_count < 1 or blacklisted != expected_clients:
            return False
        if false_revocations != client_count:
            return False
        if int(getattr(final_record, "blacklisted_clients", 0)) != client_count:
            return False
        if any(
            int(getattr(record, "nonfinite_updates", 0)) != 0
            for record in result.records
        ):
            return False
        diagnostics = list(getattr(result, "diagnostics", ()))
        if not any(
            bool(getattr(diagnostic, "suspicious", False))
            or bool(getattr(diagnostic, "count_increment", False))
            or bool(getattr(diagnostic, "revoked", False))
            for diagnostic in diagnostics
        ):
            return False
    return True


def _expand_clean_safety_envelope(
    parameters: OursParameters,
    results: Iterable[ExperimentResult],
    *,
    detector_window: int,
) -> OursParameters:
    """Raise each OR threshold just above its observed clean score envelope.

    Enforced detector feedback can make a fixed one-shot multiplier brittle:
    once a benign update is flagged it is withheld from the trusted history,
    which changes later scores.  Replaying the clean calibration controls and
    monotonically enveloping all three score families reaches a zero-flag
    trajectory when one exists, while the unchanged hard gates still decide
    feasibility.  The official test split is never an input to this routine.
    """

    observed = tuple(results)
    adjacent = _diagnostic_values(observed, "adjacent_score", detector_window)
    anchor = _diagnostic_values(observed, "anchor_score", detector_window)
    drift = _diagnostic_values(observed, "cumulative_drift", detector_window)
    if not adjacent or not anchor or not drift:
        return parameters
    return replace(
        parameters,
        theta_adj=_envelope_threshold(parameters.theta_adj, max(adjacent)),
        theta_anc=_envelope_threshold(parameters.theta_anc, max(anchor)),
        h=_envelope_threshold(parameters.h, max(drift)),
    )


def _envelope_threshold(current: float, observed_maximum: float) -> float:
    proposed = float(observed_maximum) * _SAFETY_MARGIN + _SCORE_EPS
    if not math.isfinite(proposed):
        return float(current)
    return max(float(current), proposed)


def _clean_safety_trace_entry(
    refinement: int,
    parameters: OursParameters,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Keep the safety search auditable without retaining full executions."""

    return {
        "refinement": int(refinement),
        "theta_adj": float(parameters.theta_adj),
        "theta_anc": float(parameters.theta_anc),
        "h": float(parameters.h),
        "valid": bool(summary.get("valid", False)),
        "invalid_reasons": list(summary.get("invalid_reasons", ())),
        "worst_clean_false_positive_rate": float(
            summary.get("worst_clean_false_positive_rate", 1.0)
        ),
        "worst_clean_round_acceptance_rate": float(
            summary.get("worst_clean_round_acceptance_rate", 0.0)
        ),
        "worst_clean_round_suspicious_rate": float(
            summary.get("worst_clean_round_suspicious_rate", 1.0)
        ),
        "worst_clean_round_ess_ratio": float(
            summary.get("worst_clean_round_ess_ratio", 0.0)
        ),
    }


def _experiment_config(
    args,
    *,
    partition: str,
    num_clients: int,
    ratio: float,
    seed: int,
    parameters: OursParameters,
    enforce: bool,
    clean_probe: bool,
) -> ExperimentConfig:
    default = ExperimentConfig()
    sm9_workers = _arg(args, "sm9_workers", 1)
    if isinstance(sm9_workers, str):
        sm9_workers = 1
    values = {
        "method": "sm9rrs",
        "malicious_ratio": float(ratio),
        "num_clients": int(num_clients),
        "rounds": int(_arg(args, "rounds", default.rounds)),
        "target_error": float(_arg(args, "target_error", default.target_error)),
        "local_epochs": int(_arg(args, "local_epochs", default.local_epochs)),
        "batch_size": int(_arg(args, "batch_size", default.batch_size)),
        "lr": float(_arg(args, "lr", default.lr)),
        "lr_decay": float(_arg(args, "lr_decay", default.lr_decay)),
        "compute_backend": str(_arg(args, "compute_backend", default.compute_backend)),
        "device": str(_arg(args, "device", default.device)),
        "partition": str(partition),
        "dirichlet_alpha": float(
            _arg(args, "dirichlet_alpha", default.dirichlet_alpha)
        ),
        "attack": "none" if clean_probe else str(_arg(args, "attack", default.attack)),
        "attack_scale": float(_arg(args, "attack_scale", default.attack_scale)),
        "attack_boost": float(_arg(args, "attack_boost", default.attack_boost)),
        "attack_epochs": int(_arg(args, "attack_epochs", default.attack_epochs)),
        "attack_stealth_steps": int(
            _arg(args, "attack_stealth_steps", default.attack_stealth_steps)
        ),
        "attack_distance_weight": float(
            _arg(args, "attack_distance_weight", default.attack_distance_weight)
        ),
        "attack_source_label": int(
            _arg(args, "attack_source_label", default.attack_source_label)
        ),
        "attack_target_label": int(
            _arg(args, "attack_target_label", default.attack_target_label)
        ),
        "attack_target_count": int(
            _arg(args, "attack_target_count", default.attack_target_count)
        ),
        "attack_start_round": int(
            _arg(args, "attack_start_round", default.attack_start_round)
        ),
        "detector_window": int(_arg(args, "detector_window", default.detector_window)),
        "z_threshold": float(parameters.theta_adj),
        "detector_subspace_dim": int(parameters.q),
        "detector_gap_threshold": float(parameters.g0),
        "detector_adjacent_threshold": float(parameters.theta_adj),
        "detector_anchor_threshold": float(parameters.theta_anc),
        "detector_drift_memory": float(parameters.beta),
        "detector_drift_allowance": float(parameters.kappa),
        "detector_drift_threshold": float(parameters.h),
        "detector_decision_rule": "any",
        "crypto_mode": "simulated",
        "dkg_threshold": int(_arg(args, "dkg_threshold", default.dkg_threshold)),
        "dkg_nodes": int(_arg(args, "dkg_nodes", default.dkg_nodes)),
        "early_stop": False,
        "eval_interval": 1,
        "checkpoint_interval": 0,
        "sm9_workers": int(sm9_workers),
        "suspicion_penalty_factor": float(parameters.penalty_factor),
        "suspicion_recovery_factor": float(parameters.recovery_factor),
        "suspicion_remove_after": int(parameters.C_tol),
        "suspicion_count_max": int(parameters.C_max),
        "seed": int(seed),
    }
    if "detector_enforce" in {item.name for item in fields(ExperimentConfig)}:
        values["detector_enforce"] = bool(enforce)
    return replace(default, **values)


def _score_closed_loop_candidate(
    candidate_id: str,
    results: list[ExperimentResult],
) -> dict[str, Any]:
    clean = [item for item in results if abs(item.config.malicious_ratio) < 1e-12]
    attacked = [item for item in results if item.config.malicious_ratio > 0.0]
    invalid: list[str] = []
    if not clean:
        invalid.append("missing_clean_control")
    if any(int(item.stopped_round) != int(item.config.rounds) for item in results):
        invalid.append("incomplete_rounds")
    if any(int(getattr(item, "nonfinite_updates", 0)) != 0 for item in results):
        invalid.append("nonfinite_updates")

    clean_fp_rates: list[float] = []
    clean_acceptance_rates: list[float] = []
    clean_suspicious_rates: list[float] = []
    clean_ess_ratios: list[float] = []
    for result in clean:
        final_record = result.records[-1] if result.records else None
        fp = int(getattr(final_record, "false_positive_revocations", 0))
        clean_fp_rates.append(fp / max(1, result.config.num_clients))
        round_records = [record for record in result.records if int(record.round) > 0]
        if not round_records:
            clean_acceptance_rates.append(0.0)
        else:
            clean_acceptance_rates.extend(
                float(record.accepted_updates) / max(1, result.config.num_clients)
                for record in round_records
            )
        by_round: dict[int, int] = {}
        diagnostics = list(getattr(result, "diagnostics", ()))
        for diagnostic in diagnostics:
            if bool(getattr(diagnostic, "suspicious", False)):
                round_id = int(getattr(diagnostic, "round", 0))
                by_round[round_id] = by_round.get(round_id, 0) + 1
        observed_rounds = {
            int(getattr(diagnostic, "round", 0)) for diagnostic in diagnostics
        }
        if not observed_rounds:
            clean_suspicious_rates.append(1.0)
        else:
            clean_suspicious_rates.extend(
                by_round.get(round_id, 0) / max(1, result.config.num_clients)
                for round_id in observed_rounds
                if round_id > 0
            )
        weights_by_round: dict[int, list[float]] = {}
        for diagnostic in diagnostics:
            round_id = int(getattr(diagnostic, "round", 0))
            weight = float(getattr(diagnostic, "aggregation_weight", float("nan")))
            if round_id > 0 and math.isfinite(weight) and weight >= 0.0:
                weights_by_round.setdefault(round_id, []).append(weight)
        if not weights_by_round:
            clean_ess_ratios.append(0.0)
        else:
            for weights in weights_by_round.values():
                total = sum(weights)
                square_sum = sum(weight * weight for weight in weights)
                ess = (total * total / square_sum) if square_sum > 0.0 else 0.0
                clean_ess_ratios.append(
                    ess / max(1, result.config.num_clients)
                )

    worst_fp = max(clean_fp_rates, default=1.0)
    worst_acceptance = min(clean_acceptance_rates, default=0.0)
    worst_suspicious = max(clean_suspicious_rates, default=1.0)
    worst_ess_ratio = min(clean_ess_ratios, default=0.0)
    if worst_fp > _CLEAN_ALPHA + 1e-12:
        invalid.append("clean_false_positive_rate")
    if worst_acceptance + 1e-12 < _MIN_CLEAN_ACCEPTANCE:
        invalid.append("clean_acceptance_rate")
    if worst_suspicious > _MAX_CLEAN_SUSPICIOUS + 1e-12:
        invalid.append("clean_suspicious_rate")
    if worst_ess_ratio + 1e-12 < _MIN_CLEAN_ESS_RATIO:
        invalid.append("clean_ess_ratio")

    clean_accuracy = fmean(float(item.final_accuracy) for item in clean) if clean else 0.0
    robust_accuracy = (
        fmean(float(item.final_accuracy) for item in attacked)
        if attacked
        else clean_accuracy
    )
    attack_rates = [
        float(item.records[-1].attack_target_success_rate)
        for item in attacked
        if item.records
        and item.records[-1].attack_target_success_rate is not None
    ]
    attack_success = fmean(attack_rates) if attack_rates else 0.0
    score = (
        _OBJECTIVE["clean_accuracy_weight"] * clean_accuracy
        + _OBJECTIVE["robust_accuracy_weight"] * robust_accuracy
        - _OBJECTIVE["attack_success_weight"] * attack_success
        - _OBJECTIVE["clean_false_positive_weight"] * worst_fp
    )
    malicious_rates: list[float] = []
    detection_delays: list[float] = []
    for item in attacked:
        final_record = item.records[-1] if item.records else None
        malicious_count = malicious_client_count(
            item.config.num_clients,
            item.config.malicious_ratio,
        )
        malicious_rates.append(
            int(getattr(final_record, "true_positive_revocations", 0))
            / max(1, malicious_count)
        )
        attack_start = item.config.attack_start_round or item.config.detector_window + 2
        flagged_rounds = [
            int(getattr(diagnostic, "round", 0))
            for diagnostic in getattr(item, "diagnostics", ())
            if bool(getattr(diagnostic, "is_malicious", False))
            and bool(getattr(diagnostic, "suspicious", False))
            and int(getattr(diagnostic, "round", 0)) >= attack_start
        ]
        detection_delays.append(
            float(min(flagged_rounds) - attack_start)
            if flagged_rounds
            else float(item.config.rounds + 1)
        )
    return {
        "candidate_id": candidate_id,
        "valid": not invalid,
        "invalid_reasons": invalid,
        # JSON deliberately rejects NaN/Infinity so an artifact can be hashed
        # and audited identically across runtimes.  Invalid candidates carry a
        # null score and are excluded before ranking.
        "score": float(score) if not invalid else None,
        "clean_accuracy": float(clean_accuracy),
        "robust_accuracy": float(robust_accuracy),
        "attack_success_rate": float(attack_success),
        "worst_clean_false_positive_rate": float(worst_fp),
        "worst_clean_round_acceptance_rate": float(worst_acceptance),
        "worst_clean_round_suspicious_rate": float(worst_suspicious),
        "worst_clean_round_ess_ratio": float(worst_ess_ratio),
        "malicious_revocation_rate": fmean(malicious_rates) if malicious_rates else 0.0,
        "first_detection_delay": fmean(detection_delays) if detection_delays else 0.0,
        "result_count": len(results),
    }


def _diagnostic_values(
    results: Iterable[ExperimentResult],
    field_name: str,
    detector_window: int,
) -> list[float]:
    values = []
    for result in results:
        for diagnostic in getattr(result, "diagnostics", ()):
            if int(getattr(diagnostic, "round", 0)) <= detector_window:
                continue
            value = float(getattr(diagnostic, field_name, float("nan")))
            if math.isfinite(value) and value >= 0.0:
                values.append(value)
    return values


def _per_tag_maxima(
    results: Iterable[ExperimentResult],
    field_name: str,
    detector_window: int,
) -> list[float]:
    maxima: list[float] = []
    for result_index, result in enumerate(results):
        grouped: dict[tuple[int, str], list[float]] = {}
        for diagnostic in getattr(result, "diagnostics", ()):
            if int(getattr(diagnostic, "round", 0)) <= detector_window:
                continue
            tag = str(
                getattr(diagnostic, "task_tag", "")
                or getattr(diagnostic, "client_id", "")
            )
            value = float(getattr(diagnostic, field_name, float("nan")))
            if tag and math.isfinite(value) and value >= 0.0:
                grouped.setdefault((result_index, tag), []).append(value)
        maxima.extend(max(values) for values in grouped.values() if values)
    if not maxima:
        raise OursCalibrationError(f"clean shadow produced no {field_name} blocks")
    return maxima


def _replay_drift_maxima(
    results: Iterable[ExperimentResult],
    *,
    detector_window: int,
    beta: float,
    kappa: float,
) -> list[float]:
    maxima: list[float] = []
    for result in results:
        grouped: dict[str, list[tuple[int, float]]] = {}
        for diagnostic in getattr(result, "diagnostics", ()):
            round_id = int(getattr(diagnostic, "round", 0))
            if round_id <= detector_window:
                continue
            tag = str(
                getattr(diagnostic, "task_tag", "")
                or getattr(diagnostic, "client_id", "")
            )
            score = float(getattr(diagnostic, "anchor_score", float("nan")))
            if tag and math.isfinite(score) and score >= 0.0:
                grouped.setdefault(tag, []).append((round_id, score))
        for observations in grouped.values():
            drift = 0.0
            maximum = 0.0
            for _round_id, score in sorted(observations):
                drift = max(0.0, beta * drift + score - kappa)
                maximum = max(maximum, drift)
            maxima.append(maximum)
    if not maxima:
        raise OursCalibrationError("clean shadow produced no cumulative-drift blocks")
    return maxima


def _upper_block_threshold(
    values: Iterable[float],
    *,
    tail_probability: float,
) -> tuple[float, str]:
    finite = sorted(float(value) for value in values if math.isfinite(value) and value >= 0.0)
    if not finite:
        raise OursCalibrationError("cannot calibrate a threshold from an empty block set")
    rank = int(math.ceil((len(finite) + 1) * (1.0 - tail_probability)))
    if rank <= len(finite):
        base = finite[max(0, rank - 1)]
        rule = f"upper_order_statistic_rank_{rank}_of_{len(finite)}"
    else:
        base = finite[-1]
        rule = f"finite_sample_max_fallback_n_{len(finite)}"
    return max(_SCORE_EPS, base * _SAFETY_MARGIN + _SCORE_EPS), rule


def _positive_quantile(values: Iterable[float], probability: float) -> float:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(value) and value >= 0.0],
        dtype=np.float64,
    )
    if finite.size == 0:
        raise OursCalibrationError("cannot derive a positive parameter from no values")
    return max(_SCORE_EPS, float(np.quantile(finite, probability)))


def _load_matching_artifact(
    path: Path,
    *,
    protocol_fingerprint: str,
    calibration_fingerprint: str,
    dataset_digest: str,
) -> OursCalibrationArtifact | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        artifact = OursCalibrationArtifact.from_dict(payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if (
        artifact.schema_version != CALIBRATION_SCHEMA_VERSION
        or artifact.algorithm_version != CALIBRATION_ALGORITHM_VERSION
        or artifact.status != "complete"
        or artifact.protocol_fingerprint != protocol_fingerprint
        or artifact.calibration_fingerprint != calibration_fingerprint
        or artifact.dataset_digest != dataset_digest
        or artifact.artifact_fingerprint != _artifact_fingerprint(artifact)
    ):
        return None
    return artifact


def _artifact_fingerprint(artifact: OursCalibrationArtifact) -> str:
    payload = artifact.to_dict()
    payload.pop("artifact_fingerprint", None)
    payload.pop("created_at_utc", None)
    payload.pop("calibration_runtime_seconds", None)
    return _json_fingerprint(payload)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _training_dataset_digest(dataset: ImageDataset) -> str:
    digest = hashlib.sha256()
    digest.update(str(dataset.name).encode("utf-8"))
    digest.update(str(tuple(dataset.input_shape or ())).encode("ascii"))
    digest.update(str(int(dataset.num_classes)).encode("ascii"))
    _update_array_digest(digest, dataset.x_train)
    _update_array_digest(digest, dataset.y_train)
    return digest.hexdigest()


def _update_array_digest(digest, value: np.ndarray) -> None:
    array = np.asarray(value)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    contiguous = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
    block_size = 8 * 1024 * 1024
    for start in range(0, contiguous.size, block_size):
        digest.update(memoryview(contiguous[start : start + block_size]))


def _split_payload(split: TrainingThreeWaySplit, split_seed: int) -> dict[str, Any]:
    return {
        "source": "original training split only",
        "official_test_used_for_selection": False,
        "train_fraction": 0.90,
        "calibration_fraction": 0.05,
        "attack_auxiliary_fraction": 0.05,
        "split_seed": int(split_seed),
        "train_samples": int(len(split.train_indices)),
        "calibration_samples": int(len(split.calibration_indices)),
        "attack_auxiliary_samples": int(len(split.attack_indices)),
        "train_indices_digest": _index_digest(split.train_indices),
        "calibration_indices_digest": _index_digest(split.calibration_indices),
        "attack_indices_digest": _index_digest(split.attack_indices),
    }


def _index_digest(indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    _update_array_digest(digest, np.asarray(indices, dtype=np.int64))
    return digest.hexdigest()


def _protocol_payload(
    dataset: ImageDataset,
    args,
    *,
    split_seed: int,
    shadow_seed: int,
    validation_seed: int,
) -> dict[str, Any]:
    names = (
        "rounds",
        "target_error",
        "local_epochs",
        "batch_size",
        "lr",
        "lr_decay",
        "dirichlet_alpha",
        "attack",
        "attack_scale",
        "attack_boost",
        "attack_epochs",
        "attack_stealth_steps",
        "attack_distance_weight",
        "attack_source_label",
        "attack_target_label",
        "attack_target_count",
        "attack_start_round",
        "detector_window",
        "seed",
    )
    shared = {name: _jsonable(_arg(args, name, None)) for name in names}
    from .model import describe_compute_backend

    resolved_backend = describe_compute_backend(
        str(_arg(args, "compute_backend", "numpy")),
        str(_arg(args, "device", "auto")),
    )
    if resolved_backend.startswith("torch:cuda:"):
        resolved_backend = "torch:cuda"
    shared["resolved_compute_backend"] = resolved_backend
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "algorithm_version": CALIBRATION_ALGORITHM_VERSION,
        "dataset": {
            "name": dataset.name,
            "input_shape": list(dataset.input_shape or ()),
            "num_classes": int(dataset.num_classes),
            "training_samples": int(len(dataset.y_train)),
        },
        "shared_parameters": shared,
        "partitions": list(_partitions(args)),
        "client_counts": list(_client_counts(args)),
        "ratios": list(_ratios(args)),
        "split": {
            "fractions": [0.90, 0.05, 0.05],
            "split_seed": int(split_seed),
        },
        "shadow_seed": int(shadow_seed),
        "validation_seed": int(validation_seed),
        "q_candidates": f"1..min(3,{int(dataset.num_classes)}-1)",
        "C_tol_candidates": "unique(min(3,K),min(5,K))",
        "clean_alpha": _CLEAN_ALPHA,
        "hard_constraints": {
            "max_clean_false_positive_rate": _CLEAN_ALPHA,
            "min_clean_round_acceptance_rate": _MIN_CLEAN_ACCEPTANCE,
            "max_clean_round_suspicious_rate": _MAX_CLEAN_SUSPICIOUS,
            "min_clean_round_ess_ratio": _MIN_CLEAN_ESS_RATIO,
            "require_full_rounds": True,
            "require_finite_updates": True,
        },
        "or_risk_allocation": "alpha/3 per evidence family",
        "safety_margin": _SAFETY_MARGIN,
        "closed_loop_clean_envelope": {
            "rule": (
                "familywise max(current, 1.05 * maximum observed clean "
                "closed-loop score + 1e-8)"
            ),
            "max_refinements": _MAX_CLEAN_ENVELOPE_REFINEMENTS,
            "attack_evaluation": "only after all clean hard constraints pass",
            "refinable_incomplete_rule": _REFINABLE_INCOMPLETE_RULE,
        },
        "beta_rule": "exp(log(0.5)/K)",
        "g0_rule": "clean scored spectral-gap q0.25",
        "kappa_rule": "clean scored anchor-score q0.90",
        "penalty_rule": "min(0.1,1/max(client_counts))",
        "recovery_rule": "penalty**(-0.5)",
        "objective": dict(_OBJECTIVE),
    }


def _derived_seed(seed: int, domain: str) -> int:
    payload = f"{CALIBRATION_ALGORITHM_VERSION}:{domain}:{int(seed)}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)
    return int(value or 1)


def _partitions(args) -> tuple[str, ...]:
    values = _arg(args, "partitions", None) or [_arg(args, "partition", "iid")]
    return tuple(dict.fromkeys(str(value) for value in values))


def _client_counts(args) -> tuple[int, ...]:
    values = _arg(args, "client_counts", None) or [_arg(args, "num_clients", 20)]
    result = tuple(dict.fromkeys(int(value) for value in values))
    if not result or any(value < 1 for value in result):
        raise OursCalibrationError("client counts must be positive")
    return result


def _ratios(args) -> tuple[float, ...]:
    values = _arg(args, "ratios", (0.0,))
    result = tuple(dict.fromkeys(float(value) for value in values))
    if not any(abs(value) < 1e-12 for value in result):
        result = (0.0, *result)
    return result


def _arg(args, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _json_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "CALIBRATION_ALGORITHM_VERSION",
    "CALIBRATION_SCHEMA_VERSION",
    "OursCalibrationArtifact",
    "OursCalibrationError",
    "OursParameters",
    "apply_ours_parameters",
    "calibration_metadata",
    "resolve_or_run_ours_calibration",
]
