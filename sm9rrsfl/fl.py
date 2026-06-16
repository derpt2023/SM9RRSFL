"""Federated learning experiment loop for SM9-RRS-FL, Krum, and Ding13."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import numpy as np

from .aggregation import fedavg, krum, torch_krum, torch_weighted_fedavg, weighted_fedavg
from .attacks import poison_update
from .crypto import SM9RRSContext, RRSPacket, digest_update
from .ding13_detector import Ding13TrajectoryDetector
from .datasets import ImageDataset, partition_clients
from .model import accuracy, init_params, local_train_delta, model_spec_for_dataset
from .svd_detector import LongitudinalSVDDetector
from .weighting import SuspicionWeightManager


@dataclass(frozen=True)
class ExperimentConfig:
    method: str = "sm9rrs"
    malicious_ratio: float = 0.1
    num_clients: int = 20
    rounds: int = 30
    target_error: float = 0.12
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.05
    lr_decay: float = 1.0
    compute_backend: str = "numpy"
    device: str = "auto"
    partition: str = "iid"
    dirichlet_alpha: float = 0.5
    attack: str = "alternating"
    attack_scale: float = 5.0
    attack_start_round: int = 0
    detector_window: int = 3
    z_threshold: float = 3.0
    ring_size: int = 5
    crypto_mode: str = "sm9"
    accumulator_mode: str = "dynamic"
    strict_ring_verify: bool = False
    early_stop: bool = True
    eval_interval: int = 1
    sm9_workers: int = 1
    suspicion_penalty_factor: float = 0.5
    suspicion_recovery_factor: float = 2.0
    suspicion_remove_after: int = 3
    seed: int = 0


@dataclass(frozen=True)
class RoundRecord:
    method: str
    malicious_ratio: float
    round: int
    accuracy: float
    error: float
    accepted_updates: int
    rejected_updates: int
    blacklisted_clients: int
    true_positive_revocations: int
    false_positive_revocations: int
    krum_selected_client: str


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    records: list[RoundRecord]
    final_accuracy: float
    final_error: float
    stopped_round: int
    malicious_clients: tuple[str, ...]
    blacklisted_clients: tuple[str, ...]
    runtime_seconds: float = 0.0
    peak_memory_mb: float = 0.0

    def summary_dict(self) -> dict[str, object]:
        return {
            **asdict(self.config),
            "final_accuracy": self.final_accuracy,
            "final_error": self.final_error,
            "stopped_round": self.stopped_round,
            "malicious_clients": ",".join(self.malicious_clients),
            "blacklisted_clients": ",".join(self.blacklisted_clients),
            "runtime_seconds": self.runtime_seconds,
            "peak_memory_mb": self.peak_memory_mb,
        }


@dataclass(frozen=True)
class _ClientUpdateCandidate:
    identity: str
    delta: np.ndarray
    samples: int


@dataclass(frozen=True)
class _VerifiedSM9Candidate:
    identity: str
    delta: np.ndarray
    samples: int
    packet: RRSPacket
    verified: bool


@dataclass(frozen=True)
class _SM9ProcessingResult:
    updates: list[np.ndarray]
    samples: list[int]
    clients: list[str]
    suspicious_clients: set[str]
    rejected: int


def run_experiment(dataset: ImageDataset, config: ExperimentConfig) -> ExperimentResult:
    """Run one federated experiment configuration."""

    if config.method not in {"sm9rrs", "krum", "ding13", "fedavg"}:
        raise ValueError("method must be one of: sm9rrs, krum, ding13, fedavg")
    if not 0.0 <= config.malicious_ratio < 1.0:
        raise ValueError("malicious_ratio must be in [0, 1)")
    if config.eval_interval < 1:
        raise ValueError("eval_interval must be at least 1")
    if config.sm9_workers < 1:
        raise ValueError("sm9_workers must be at least 1")

    client_ids = [f"client-{idx}" for idx in range(config.num_clients)]
    client_indices = partition_clients(
        dataset.y_train,
        config.num_clients,
        strategy=config.partition,
        dirichlet_alpha=config.dirichlet_alpha,
        seed=config.seed,
    )
    malicious_clients = _choose_malicious(client_ids, config.malicious_ratio, config.seed)
    malicious_set = set(malicious_clients)
    detector_window, z_threshold, suspicion_remove_after = _effective_detector_settings(
        dataset,
        config,
    )
    attack_start = config.attack_start_round or (detector_window + 2)

    model_spec = model_spec_for_dataset(dataset)
    torch_context = _maybe_torch_context(dataset, client_indices, model_spec, config)
    params = init_params(seed=config.seed, spec=model_spec)
    records = [
        _make_record(
            config,
            round_id=0,
            acc=_evaluate_accuracy(params, dataset, model_spec, config, torch_context),
            accepted=0,
            rejected=0,
            blacklisted=0,
            tp=0,
            fp=0,
            krum_selected="",
        )
    ]

    crypto = None
    detector = None
    ding13_detector = None
    sm9_weight_manager = None
    if config.method == "sm9rrs":
        crypto = SM9RRSContext(
            client_ids,
            ring_size=config.ring_size,
            crypto_mode=config.crypto_mode,
            accumulator_mode=config.accumulator_mode,
            strict_ring_verify=config.strict_ring_verify,
            seed=config.seed,
        )
        detector = LongitudinalSVDDetector(
            window_size=detector_window,
            z_threshold=z_threshold,
            matrix_offset=model_spec.svd_matrix_offset,
            matrix_shape=model_spec.svd_matrix_shape,
            compute_backend=config.compute_backend,
            device=config.device,
        )
        sm9_weight_manager = SuspicionWeightManager(
            client_ids,
            penalty_factor=config.suspicion_penalty_factor,
            recovery_factor=config.suspicion_recovery_factor,
            remove_after=suspicion_remove_after,
        )
    elif config.method == "ding13":
        ding13_detector = Ding13TrajectoryDetector(
            client_ids,
            contamination=config.malicious_ratio,
            seed=config.seed,
            matrix_offset=model_spec.svd_matrix_offset,
            matrix_shape=model_spec.svd_matrix_shape,
            compute_backend=config.compute_backend,
            device=config.device,
        )

    blacklisted: set[str] = set()
    true_positive_revocations = 0
    false_positive_revocations = 0

    for round_id in range(1, config.rounds + 1):
        updates: list[np.ndarray] = []
        update_samples: list[int] = []
        update_clients: list[str] = []
        sm9_candidates: list[_ClientUpdateCandidate] = []
        suspicious_clients: set[str] = set()
        rejected = 0

        for client_idx, identity in enumerate(client_ids):
            if identity in blacklisted:
                continue
            indices = client_indices[client_idx]
            delta, stats = _local_train_client_delta(
                params,
                dataset,
                indices,
                client_idx=client_idx,
                round_id=round_id,
                model_spec=model_spec,
                config=config,
                torch_context=torch_context,
            )
            attack_active = identity in malicious_set and round_id >= attack_start
            if attack_active:
                delta = poison_update(
                    delta,
                    attack=config.attack,
                    scale=config.attack_scale,
                    seed=config.seed + round_id * 4099 + client_idx,
                )

            if config.method == "sm9rrs":
                sm9_candidates.append(
                    _ClientUpdateCandidate(
                        identity=identity,
                        delta=delta,
                        samples=stats.samples,
                    )
                )
                continue

            updates.append(delta)
            update_samples.append(stats.samples)
            update_clients.append(identity)

        if config.method == "sm9rrs" and sm9_candidates:
            assert crypto is not None and detector is not None
            sm9_result = _process_sm9_candidates(
                sm9_candidates,
                crypto=crypto,
                detector=detector,
                malicious_clients=malicious_clients,
                round_id=round_id,
                task_id=dataset.name,
                workers=config.sm9_workers,
            )
            updates.extend(sm9_result.updates)
            update_samples.extend(sm9_result.samples)
            update_clients.extend(sm9_result.clients)
            suspicious_clients.update(sm9_result.suspicious_clients)
            rejected += sm9_result.rejected

        krum_selected = ""
        record_accepted = len(updates)
        record_rejected = rejected
        if updates:
            if config.method == "krum":
                result = _krum(updates, byzantine_count=len(malicious_clients), config=config)
                aggregate = result.update
                krum_selected = update_clients[result.selected_index]
            elif config.method == "sm9rrs":
                assert sm9_weight_manager is not None
                weight_result = sm9_weight_manager.update(
                    update_clients,
                    suspicious_clients,
                    malicious_set,
                )
                blacklisted.update(weight_result.newly_removed)
                true_positive_revocations += weight_result.true_positive_removed
                false_positive_revocations += weight_result.false_positive_removed
                weights = [weight_result.weights[identity] for identity in update_clients]
                effective_total = sum(weight * samples for weight, samples in zip(weights, update_samples))
                if effective_total <= 0.0:
                    aggregate = np.zeros_like(updates[0])
                else:
                    aggregate = _weighted_fedavg(
                        updates,
                        weights,
                        sample_counts=update_samples,
                        config=config,
                    )
                record_accepted = sum(
                    1 for weight, samples in zip(weights, update_samples) if weight > 0.0 and samples > 0
                )
                record_rejected = len(suspicious_clients)
            elif config.method == "ding13":
                assert ding13_detector is not None
                update_by_client = dict(zip(update_clients, updates))
                ding13_result = ding13_detector.evaluate_round(
                    update_by_client,
                    malicious_set,
                    round_id=round_id,
                )
                blacklisted.update(ding13_result.newly_removed)
                true_positive_revocations += ding13_result.true_positive_removed
                false_positive_revocations += ding13_result.false_positive_removed
                weights = [ding13_result.weights[identity] for identity in update_clients]
                aggregate = _weighted_fedavg(
                    updates,
                    weights,
                    sample_counts=update_samples,
                    config=config,
                )
                record_accepted = sum(
                    1 for weight, samples in zip(weights, update_samples) if weight > 0.0 and samples > 0
                )
                record_rejected = len(ding13_result.outliers)
            else:
                aggregate = _fedavg(updates, update_samples, config=config)
            params = (params + aggregate).astype(np.float32)

        should_evaluate = round_id == config.rounds or round_id % config.eval_interval == 0
        if should_evaluate:
            acc = _evaluate_accuracy(params, dataset, model_spec, config, torch_context)
            records.append(
                _make_record(
                    config,
                    round_id=round_id,
                    acc=acc,
                    accepted=record_accepted,
                    rejected=record_rejected,
                    blacklisted=len(blacklisted),
                    tp=true_positive_revocations,
                    fp=false_positive_revocations,
                    krum_selected=krum_selected,
                )
            )
            can_stop = not malicious_clients or round_id >= attack_start
            if config.early_stop and can_stop and 1.0 - acc <= config.target_error:
                break

    final = records[-1]
    return ExperimentResult(
        config=config,
        records=records,
        final_accuracy=final.accuracy,
        final_error=final.error,
        stopped_round=final.round,
        malicious_clients=tuple(malicious_clients),
        blacklisted_clients=tuple(sorted(blacklisted)),
    )


def _effective_detector_settings(
    dataset: ImageDataset,
    config: ExperimentConfig,
) -> tuple[int, float, int]:
    if dataset.name != "cifar10":
        return config.detector_window, config.z_threshold, config.suspicion_remove_after
    return (
        max(config.detector_window, 8),
        max(config.z_threshold, 5.0),
        max(config.suspicion_remove_after, 5),
    )


def _maybe_torch_context(
    dataset: ImageDataset,
    client_indices: list[np.ndarray],
    model_spec,
    config: ExperimentConfig,
):
    try:
        from .torch_backend import should_use_torch, TorchTrainingContext
    except RuntimeError:
        return None
    if not should_use_torch(config.compute_backend, config.device):
        return None
    return TorchTrainingContext(
        dataset,
        client_indices,
        spec=model_spec,
        device=config.device,
    )


def _process_sm9_candidates(
    candidates: list[_ClientUpdateCandidate],
    *,
    crypto: SM9RRSContext,
    detector: LongitudinalSVDDetector,
    malicious_clients: tuple[str, ...],
    round_id: int,
    task_id: str,
    workers: int,
) -> _SM9ProcessingResult:
    if workers > 1 and crypto.accumulator_mode == "dynamic":
        with ThreadPoolExecutor(max_workers=min(workers, len(candidates))) as executor:
            verified_candidates = list(
                executor.map(
                    lambda candidate: _create_verified_sm9_candidate(
                        candidate,
                        crypto=crypto,
                        round_id=round_id,
                        task_id=task_id,
                    ),
                    candidates,
                )
            )
    else:
        verified_candidates = [
            _create_verified_sm9_candidate(
                candidate,
                crypto=crypto,
                round_id=round_id,
                task_id=task_id,
            )
            for candidate in candidates
        ]

    updates: list[np.ndarray] = []
    samples: list[int] = []
    clients: list[str] = []
    suspicious_clients: set[str] = set()
    rejected = 0
    has_malicious_clients = bool(malicious_clients)
    for candidate in verified_candidates:
        if not candidate.verified:
            rejected += 1
            continue
        if has_malicious_clients:
            decision = detector.evaluate(candidate.packet.link_tag, candidate.delta)
            if not decision.accepted:
                suspicious_clients.add(candidate.identity)
        updates.append(candidate.delta)
        samples.append(candidate.samples)
        clients.append(candidate.identity)
    return _SM9ProcessingResult(
        updates=updates,
        samples=samples,
        clients=clients,
        suspicious_clients=suspicious_clients,
        rejected=rejected,
    )


def _create_verified_sm9_candidate(
    candidate: _ClientUpdateCandidate,
    *,
    crypto: SM9RRSContext,
    round_id: int,
    task_id: str,
) -> _VerifiedSM9Candidate:
    update_digest = digest_update(candidate.delta)
    packet = crypto.create_packet(
        candidate.identity,
        candidate.delta,
        round_id=round_id,
        task_id=task_id,
        update_digest=update_digest,
    )
    verified = crypto.verify_packet(packet, candidate.delta, update_digest=update_digest)
    return _VerifiedSM9Candidate(
        identity=candidate.identity,
        delta=candidate.delta,
        samples=candidate.samples,
        packet=packet,
        verified=verified,
    )


def _evaluate_accuracy(params, dataset, model_spec, config: ExperimentConfig, torch_context) -> float:
    if torch_context is not None:
        return torch_context.accuracy(params)
    return accuracy(
        params,
        dataset.x_test,
        dataset.y_test,
        spec=model_spec,
        compute_backend=config.compute_backend,
        device=config.device,
    )


def _local_train_client_delta(
    params: np.ndarray,
    dataset: ImageDataset,
    indices: np.ndarray,
    *,
    client_idx: int,
    round_id: int,
    model_spec,
    config: ExperimentConfig,
    torch_context,
):
    seed = config.seed + round_id * 1009 + client_idx
    lr = config.lr * (config.lr_decay ** (round_id - 1))
    if torch_context is not None:
        return torch_context.local_train_delta(
            params,
            client_idx=client_idx,
            lr=lr,
            epochs=config.local_epochs,
            batch_size=config.batch_size,
            seed=seed,
        )
    return local_train_delta(
        params,
        dataset.x_train[indices],
        dataset.y_train[indices],
        lr=lr,
        epochs=config.local_epochs,
        batch_size=config.batch_size,
        seed=seed,
        spec=model_spec,
        compute_backend=config.compute_backend,
        device=config.device,
    )


def _fedavg(
    updates: list[np.ndarray],
    sample_counts: list[int],
    *,
    config: ExperimentConfig,
) -> np.ndarray:
    if _can_use_torch_ops(config):
        return torch_weighted_fedavg(updates, sample_counts, device=config.device)
    return fedavg(updates, sample_counts)


def _weighted_fedavg(
    updates: list[np.ndarray],
    weights: list[float],
    *,
    sample_counts: list[int],
    config: ExperimentConfig,
) -> np.ndarray:
    if _can_use_torch_ops(config):
        return torch_weighted_fedavg(updates, weights, sample_counts=sample_counts, device=config.device)
    return weighted_fedavg(updates, weights, sample_counts=sample_counts)


def _krum(updates: list[np.ndarray], *, byzantine_count: int, config: ExperimentConfig):
    if _can_use_torch_ops(config):
        return torch_krum(updates, byzantine_count=byzantine_count, device=config.device)
    return krum(updates, byzantine_count=byzantine_count)


def _can_use_torch_ops(config: ExperimentConfig) -> bool:
    try:
        from .torch_backend import should_use_torch
    except RuntimeError:
        return False
    return should_use_torch(config.compute_backend, config.device)


def _choose_malicious(client_ids: list[str], ratio: float, seed: int) -> tuple[str, ...]:
    count = int(round(len(client_ids) * ratio))
    count = min(max(count, 0), len(client_ids) - 1)
    rng = np.random.default_rng(seed + 17)
    if count == 0:
        return tuple()
    selected = rng.choice(client_ids, size=count, replace=False)
    return tuple(sorted(str(item) for item in selected))


def _make_record(
    config: ExperimentConfig,
    *,
    round_id: int,
    acc: float,
    accepted: int,
    rejected: int,
    blacklisted: int,
    tp: int,
    fp: int,
    krum_selected: str,
) -> RoundRecord:
    return RoundRecord(
        method=config.method,
        malicious_ratio=config.malicious_ratio,
        round=round_id,
        accuracy=float(acc),
        error=float(1.0 - acc),
        accepted_updates=accepted,
        rejected_updates=rejected,
        blacklisted_clients=blacklisted,
        true_positive_revocations=tp,
        false_positive_revocations=fp,
        krum_selected_client=krum_selected,
    )
