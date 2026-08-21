"""Federated learning loop for six poisoning-defense comparison methods."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Callable
import numpy as np

from .aggregation import KrumResult, fedavg, krum, torch_krum, torch_weighted_fedavg, weighted_fedavg
from .attacks import is_alternating_minimization_attack, poison_update
from .crypto import (
    ASVerifier,
    AuditorService,
    ClientSigner,
    RRSPacket,
    SM9RRSContext,
)
from .ding13_detector import Ding13TrajectoryDetector
from .datasets import ImageDataset, partition_clients
from .fedredefense import FedREDefense
from .model import (
    accuracy,
    alternating_minimization_delta,
    init_params,
    local_train_delta,
    model_spec_for_dataset,
    targeted_metrics,
)
from .svd_detector import DetectionResult, LongitudinalSVDDetector
from .vert import VERTDefense
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
    attack: str = "alternating_minimization"
    attack_scale: float = 5.0
    attack_boost: float = 10.0
    attack_epochs: int = 10
    attack_stealth_steps: int = 10
    attack_distance_weight: float = 1e-4
    attack_source_label: int = 5
    attack_target_label: int = 7
    attack_target_count: int = 1
    attack_start_round: int = 0
    detector_window: int = 3
    z_threshold: float = 3.0
    detector_subspace_dim: int = 2
    detector_gap_threshold: float = 0.1
    detector_adjacent_threshold: float | None = None
    detector_anchor_threshold: float | None = None
    detector_drift_memory: float = 0.9
    detector_drift_allowance: float = 1.0
    detector_drift_threshold: float = 5.0
    detector_decision_rule: str = "any"
    crypto_mode: str = "sm9"
    dkg_threshold: int = 2
    dkg_nodes: int = 3
    early_stop: bool = True
    eval_interval: int = 1
    checkpoint_interval: int = 0
    sm9_workers: int = 1
    suspicion_penalty_factor: float = 0.5
    suspicion_recovery_factor: float = 2.0
    suspicion_remove_after: int = 3
    suspicion_count_max: int = 0
    vert_history_window: int = 10
    vert_projection_dim: int = 128
    vert_predict_epochs: int = 5
    vert_predict_lr: float = 1e-2
    vert_top_k: int = 0
    vert_use_ratio_prior: bool = False
    fedre_threshold: float = 0.6
    fedre_initial_iterations: int = 800
    fedre_max_iterations: int = 2000
    fedre_synthetic_steps: int = 5
    fedre_images_per_class: int = 1
    fedre_image_lr: float = 0.5
    fedre_label_lr: float = 0.2
    fedre_teacher_lr: float = 0.1
    fedre_teacher_lr_lr: float = 5e-6
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
    attack_target_success_rate: float | None = None
    attack_target_confidence: float | None = None
    nonfinite_updates: int = 0
    attack_active: bool = False


@dataclass(frozen=True)
class ClientDiagnosticRecord:
    """Per-client SM9-RRS detector and weighting state for one round."""

    round: int
    client_id: str
    task_tag: str
    is_malicious: bool
    decision_reason: str
    z_sigma: float
    z_direction: float
    sigma_delta: float
    cosine_similarity: float
    sigma_exceeded: bool
    direction_exceeded: bool
    suspicious: bool
    count_increment: bool
    weight_before: float
    weight_after_penalty_recovery: float
    aggregation_weight: float
    count_before: int
    count_after: int
    trace_requested: bool
    trace_pending: bool
    revoked: bool
    spectrum_adjacent_distance: float
    subspace_adjacent_distance: float
    spectrum_anchor_distance: float
    subspace_anchor_distance: float
    z_spectrum_adjacent: float
    z_subspace_adjacent: float
    z_spectrum_anchor: float
    z_subspace_anchor: float
    adjacent_score: float
    anchor_score: float
    cumulative_drift: float
    spectral_gap: float
    direction_reliability: float
    adjacent_exceeded: bool
    anchor_exceeded: bool
    drift_exceeded: bool
    trusted_history_size: int
    attack_active: bool


@dataclass(frozen=True)
class StageTimings:
    """单个配置各阶段的累计耗时，便于直接定位性能瓶颈。"""

    training_seconds: float = 0.0
    attack_seconds: float = 0.0
    hash_seconds: float = 0.0
    packet_build_seconds: float = 0.0
    sign_seconds: float = 0.0
    verify_seconds: float = 0.0
    detection_seconds: float = 0.0
    aggregation_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    # The per-operation fields above are sums across clients and may exceed
    # wall time when sm9_workers > 1.  These four fields instead measure
    # non-overlapping wall-clock spans and are safe to subtract from the
    # end-to-end runtime for the protocol-free comparison.
    crypto_setup_wall_seconds: float = 0.0
    crypto_packet_wall_seconds: float = 0.0
    crypto_audit_wall_seconds: float = 0.0
    crypto_finalize_wall_seconds: float = 0.0

    @property
    def crypto_wall_seconds(self) -> float:
        return sum(
            (
                self.crypto_setup_wall_seconds,
                self.crypto_packet_wall_seconds,
                self.crypto_audit_wall_seconds,
                self.crypto_finalize_wall_seconds,
            )
        )

    def summary_dict(self) -> dict[str, float]:
        return {
            **asdict(self),
            "crypto_wall_seconds": self.crypto_wall_seconds,
        }


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    records: list[RoundRecord]
    final_accuracy: float
    final_error: float
    stopped_round: int
    malicious_clients: tuple[str, ...]
    blacklisted_clients: tuple[str, ...]
    diagnostics: list[ClientDiagnosticRecord] = field(default_factory=list)
    runtime_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    stage_timings: StageTimings = field(default_factory=StageTimings)
    nonfinite_updates: int = 0
    checkpoint_io_seconds: float = 0.0

    @property
    def runtime_without_crypto_seconds(self) -> float:
        """End-to-end wall time with measured SM9/RRS protocol spans removed."""

        return max(0.0, self.runtime_seconds - self.stage_timings.crypto_wall_seconds)

    def summary_dict(self) -> dict[str, object]:
        final_record = self.records[-1] if self.records else None
        return {
            **asdict(self.config),
            "final_accuracy": self.final_accuracy,
            "final_error": self.final_error,
            "final_attack_target_success_rate": (
                final_record.attack_target_success_rate
                if final_record is not None
                else None
            ),
            "final_attack_target_confidence": (
                final_record.attack_target_confidence
                if final_record is not None
                else None
            ),
            "stopped_round": self.stopped_round,
            "effective_attack_start_round": (
                self.config.attack_start_round or self.config.detector_window + 2
            ),
            "malicious_clients": ",".join(self.malicious_clients),
            "blacklisted_clients": ",".join(self.blacklisted_clients),
            "runtime_seconds": self.runtime_seconds,
            "runtime_without_crypto_seconds": self.runtime_without_crypto_seconds,
            "checkpoint_io_seconds": self.checkpoint_io_seconds,
            "nonfinite_updates": self.nonfinite_updates,
            "peak_memory_mb": self.peak_memory_mb,
            **self.stage_timings.summary_dict(),
        }


@dataclass(frozen=True)
class _ClientUpdateCandidate:
    identity: str
    delta: Any
    cpu_delta: np.ndarray
    samples: int


@dataclass(frozen=True)
class _VerifiedSM9Candidate:
    # AS-facing state contains only the opaque, verified packet and update.
    # Experiment ground truth remains in malicious_set outside this object.
    delta: Any
    cpu_delta: np.ndarray
    samples: int
    packet: RRSPacket
    verified: bool
    hash_seconds: float
    packet_build_seconds: float
    sign_seconds: float
    verify_seconds: float


@dataclass(frozen=True)
class _SM9ProcessingResult:
    updates: list[Any]
    samples: list[int]
    tags: list[str]
    suspicious_tags: set[str]
    count_increment_tags: set[str]
    candidates_by_tag: dict[str, _VerifiedSM9Candidate]
    decisions_by_tag: dict[str, DetectionResult]
    client_ids_by_tag: dict[str, str]
    rejected: int
    hash_seconds: float
    packet_build_seconds: float
    sign_seconds: float
    verify_seconds: float
    detection_seconds: float


def run_experiment(
    dataset: ImageDataset,
    config: ExperimentConfig,
    *,
    resume_state: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ExperimentResult:
    """运行一个完整联邦学习配置。

    主流程按“客户端本地优化/投毒 -> 密码验证/检测 -> 聚合 -> 评估”推进。
    交替最小化攻击发生在恶意客户端的优化过程中，其余攻击仍是训练后的
    更新变换；StageTimings 对这些阶段分别计时。
    """

    supported_methods = {
        "sm9rrs",
        "vert",
        "fedredefense",
        "krum",
        "ding13",
        "fedavg",
    }
    if config.method not in supported_methods:
        raise ValueError(
            "method must be one of: sm9rrs, vert, fedredefense, "
            "krum, ding13, fedavg"
        )
    if not 0.0 <= config.malicious_ratio < 1.0:
        raise ValueError("malicious_ratio must be in [0, 1)")
    if config.eval_interval < 1:
        raise ValueError("eval_interval must be at least 1")
    if config.sm9_workers < 1:
        raise ValueError("sm9_workers must be at least 1")
    if not 1 <= config.dkg_threshold <= config.dkg_nodes:
        raise ValueError("dkg_threshold must satisfy 1 <= threshold <= dkg_nodes")
    if config.num_clients < 1:
        raise ValueError("num_clients must be at least 1")
    if config.rounds < 1:
        raise ValueError("rounds must be at least 1")
    if config.local_epochs < 1:
        raise ValueError("local_epochs must be at least 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.lr <= 0.0 or config.lr_decay <= 0.0:
        raise ValueError("lr and lr_decay must be positive")
    if config.attack not in {
        "none",
        "sign_flip",
        "gaussian",
        "alternating",
        "alternating_minimization",
    }:
        raise ValueError(
            "attack must be one of: none, sign_flip, gaussian, "
            "alternating_minimization (or legacy alias alternating)"
        )
    if not np.isfinite(config.attack_scale) or config.attack_scale <= 0.0:
        raise ValueError("attack_scale must be finite and positive")
    if not np.isfinite(config.attack_boost) or config.attack_boost <= 0.0:
        raise ValueError("attack_boost must be finite and positive")
    if config.attack_epochs < 1:
        raise ValueError("attack_epochs must be at least 1")
    if config.attack_stealth_steps < 1:
        raise ValueError("attack_stealth_steps must be at least 1")
    if (
        not np.isfinite(config.attack_distance_weight)
        or config.attack_distance_weight < 0.0
    ):
        raise ValueError("attack_distance_weight must be finite and non-negative")
    if config.attack_target_count < 1:
        raise ValueError("attack_target_count must be at least 1")
    if not 0 <= config.attack_source_label < dataset.num_classes:
        raise ValueError("attack_source_label is outside the dataset class range")
    if not 0 <= config.attack_target_label < dataset.num_classes:
        raise ValueError("attack_target_label is outside the dataset class range")
    if (
        is_alternating_minimization_attack(config.attack)
        and config.attack_source_label == config.attack_target_label
    ):
        raise ValueError(
            "alternating minimization requires different source and target labels"
        )
    if config.attack_start_round < 0:
        raise ValueError("attack_start_round must be non-negative")
    if config.detector_window < 2:
        raise ValueError("detector_window must be at least 2")
    if not np.isfinite(config.z_threshold) or config.z_threshold <= 0.0:
        raise ValueError("z_threshold must be finite and positive")
    if config.detector_subspace_dim < 1:
        raise ValueError("detector_subspace_dim must be at least 1")
    if config.detector_subspace_dim >= dataset.num_classes:
        raise ValueError(
            "detector_subspace_dim must leave room for the q+1 singular value"
        )
    if (
        not np.isfinite(config.detector_gap_threshold)
        or config.detector_gap_threshold <= 0.0
    ):
        raise ValueError("detector_gap_threshold must be finite and positive")
    for name, value in {
        "detector_adjacent_threshold": config.detector_adjacent_threshold,
        "detector_anchor_threshold": config.detector_anchor_threshold,
    }.items():
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if (
        not np.isfinite(config.detector_drift_memory)
        or not 0.0 < config.detector_drift_memory <= 1.0
    ):
        raise ValueError("detector_drift_memory must be in (0, 1]")
    for name, value in {
        "detector_drift_allowance": config.detector_drift_allowance,
        "detector_drift_threshold": config.detector_drift_threshold,
    }.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if config.detector_decision_rule != "any":
        raise ValueError(
            "detector_decision_rule must be 'any' (the v3 OR formula)"
        )
    if config.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative")
    if not 0.0 < config.suspicion_penalty_factor < 1.0:
        raise ValueError("suspicion_penalty_factor must be in (0, 1)")
    if config.suspicion_recovery_factor <= 1.0:
        raise ValueError("suspicion_recovery_factor must be greater than 1")
    if config.suspicion_remove_after < 1:
        raise ValueError("suspicion_remove_after must be at least 1")
    if (
        config.suspicion_count_max != 0
        and config.suspicion_count_max < config.suspicion_remove_after
    ):
        raise ValueError("suspicion_count_max must be 0 or at least C_tol")
    if config.vert_history_window < 2:
        raise ValueError("vert_history_window must be at least 2")
    if config.vert_projection_dim < 2:
        raise ValueError("vert_projection_dim must be at least 2")
    if config.vert_predict_epochs < 1:
        raise ValueError("vert_predict_epochs must be at least 1")
    if not np.isfinite(config.vert_predict_lr) or config.vert_predict_lr <= 0.0:
        raise ValueError("vert_predict_lr must be finite and positive")
    if config.vert_top_k < 0:
        raise ValueError("vert_top_k must be non-negative")
    if config.vert_use_ratio_prior and config.vert_top_k > 0:
        raise ValueError(
            "vert_use_ratio_prior cannot be combined with a positive vert_top_k"
        )
    if not np.isfinite(config.fedre_threshold) or config.fedre_threshold <= 0.0:
        raise ValueError("fedre_threshold must be finite and positive")
    if config.fedre_initial_iterations < 1 or config.fedre_max_iterations < 1:
        raise ValueError("FedREDefense iteration counts must be at least 1")
    if config.fedre_synthetic_steps < 1 or config.fedre_images_per_class < 1:
        raise ValueError(
            "FedREDefense synthetic steps and images per class must be at least 1"
        )
    for name, value in {
        "fedre_image_lr": config.fedre_image_lr,
        "fedre_label_lr": config.fedre_label_lr,
        "fedre_teacher_lr": config.fedre_teacher_lr,
        "fedre_teacher_lr_lr": config.fedre_teacher_lr_lr,
    }.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if config.partition == "dirichlet" and config.dirichlet_alpha <= 0.0:
        raise ValueError("dirichlet_alpha must be positive")
    invalid_reason = experiment_config_error(config)
    if invalid_reason is not None:
        # 在加载/划分数据和执行本地训练前失败，避免无效 Krum 参数跑数小时后
        # 才在第一轮聚合阶段报错。
        raise ValueError(invalid_reason)

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
    adjacent_threshold = (
        z_threshold
        if config.detector_adjacent_threshold is None
        else config.detector_adjacent_threshold
    )
    anchor_threshold = (
        z_threshold
        if config.detector_anchor_threshold is None
        else config.detector_anchor_threshold
    )
    suspicion_count_max = config.suspicion_count_max or suspicion_remove_after
    attack_start = config.attack_start_round or (detector_window + 2)

    model_spec = model_spec_for_dataset(dataset)
    attack_target_indices = (
        _select_attack_target_indices(
            dataset,
            config,
            required=bool(malicious_set),
        )
        if is_alternating_minimization_attack(config.attack)
        else np.empty(0, dtype=np.int64)
    )
    torch_context = _maybe_torch_context(dataset, client_indices, model_spec, config)
    params = init_params(seed=config.seed, spec=model_spec)
    training_seconds = 0.0
    attack_seconds = 0.0
    hash_seconds = 0.0
    packet_build_seconds = 0.0
    sign_seconds = 0.0
    verify_seconds = 0.0
    detection_seconds = 0.0
    aggregation_seconds = 0.0
    evaluation_seconds = 0.0
    crypto_setup_wall_seconds = 0.0
    crypto_packet_wall_seconds = 0.0
    crypto_audit_wall_seconds = 0.0
    crypto_finalize_wall_seconds = 0.0
    nonfinite_updates = 0
    start_round = 1
    if resume_state is None:
        evaluation_started = perf_counter()
        initial_accuracy = _evaluate_accuracy(params, dataset, model_spec, config, torch_context)
        (
            initial_target_success,
            initial_target_confidence,
        ) = _evaluate_attack_target_metrics(
            params,
            dataset,
            attack_target_indices,
            model_spec,
            config,
            torch_context,
        )
        evaluation_seconds = perf_counter() - evaluation_started
        records = [
            _make_record(
                config,
                round_id=0,
                acc=initial_accuracy,
                accepted=0,
                rejected=0,
                blacklisted=0,
                tp=0,
                fp=0,
                krum_selected="",
                attack_target_success=initial_target_success,
                attack_target_confidence=initial_target_confidence,
            )
        ]
        diagnostics: list[ClientDiagnosticRecord] = []
    else:
        # 新方案的 Tag_pi depends on dS_pi and h_t.  The D-KGC shares and task
        # salt are therefore restored together with the longitudinal detector.
        params = np.asarray(resume_state["params"], dtype=np.float32)
        records = list(resume_state["records"])
        diagnostics = list(resume_state.get("diagnostics", ()))
        start_round = int(resume_state["completed_round"]) + 1
        timings = resume_state.get("timings", {})
        training_seconds = float(timings.get("training_seconds", 0.0))
        attack_seconds = float(timings.get("attack_seconds", 0.0))
        hash_seconds = float(timings.get("hash_seconds", 0.0))
        packet_build_seconds = float(timings.get("packet_build_seconds", 0.0))
        sign_seconds = float(timings.get("sign_seconds", 0.0))
        verify_seconds = float(timings.get("verify_seconds", 0.0))
        detection_seconds = float(timings.get("detection_seconds", 0.0))
        aggregation_seconds = float(timings.get("aggregation_seconds", 0.0))
        evaluation_seconds = float(timings.get("evaluation_seconds", 0.0))
        crypto_setup_wall_seconds = float(
            timings.get("crypto_setup_wall_seconds", 0.0)
        )
        crypto_packet_wall_seconds = float(
            timings.get("crypto_packet_wall_seconds", 0.0)
        )
        crypto_audit_wall_seconds = float(
            timings.get("crypto_audit_wall_seconds", 0.0)
        )
        crypto_finalize_wall_seconds = float(
            timings.get("crypto_finalize_wall_seconds", 0.0)
        )
        nonfinite_updates = int(resume_state.get("nonfinite_updates", 0))

    blacklisted: set[str] = set(resume_state.get("blacklisted", ())) if resume_state else set()
    true_positive_revocations = (
        int(resume_state.get("true_positive_revocations", 0)) if resume_state else 0
    )
    false_positive_revocations = (
        int(resume_state.get("false_positive_revocations", 0)) if resume_state else 0
    )

    crypto = None
    client_signers: dict[str, ClientSigner] = {}
    as_verifier: ASVerifier | None = None
    auditor: AuditorService | None = None
    detector = None
    ding13_detector = None
    vert_defense = None
    fedre_defense = None
    sm9_weight_manager = None
    if config.method == "sm9rrs":
        crypto_setup_started = perf_counter()
        crypto_state = resume_state.get("crypto_state") if resume_state else None
        if resume_state is not None and crypto_state is None:
            raise ValueError("SM9-RRS checkpoint is missing D-KGC/task state")
        crypto = SM9RRSContext(
            client_ids,
            crypto_mode=config.crypto_mode,
            dkg_threshold=config.dkg_threshold,
            dkg_nodes=config.dkg_nodes,
            seed=config.seed,
            state=crypto_state,
        )
        crypto.register_task(
            dataset.name,
            [identity for identity in client_ids if identity not in blacklisted],
        )
        client_signers = {
            identity: crypto.client_signer(identity) for identity in client_ids
        }
        as_verifier = crypto.as_verifier(
            expected_update_shape=(model_spec.parameter_size,),
        )
        auditor = crypto.auditor_service()
        crypto_setup_wall_seconds += perf_counter() - crypto_setup_started
        detector = LongitudinalSVDDetector(
            window_size=detector_window,
            z_threshold=z_threshold,
            subspace_dim=config.detector_subspace_dim,
            gap_threshold=config.detector_gap_threshold,
            adjacent_threshold=adjacent_threshold,
            anchor_threshold=anchor_threshold,
            drift_memory=config.detector_drift_memory,
            drift_allowance=config.detector_drift_allowance,
            drift_threshold=config.detector_drift_threshold,
            decision_rule=config.detector_decision_rule,
            num_classes=model_spec.num_classes,
            expected_update_size=model_spec.parameter_size,
            compute_backend=config.compute_backend,
            device=config.device,
        )
        sm9_weight_manager = SuspicionWeightManager(
            participant_count=len(client_ids),
            penalty_factor=config.suspicion_penalty_factor,
            recovery_factor=config.suspicion_recovery_factor,
            remove_after=suspicion_remove_after,
            max_count=suspicion_count_max,
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
    elif config.method == "vert":
        vert_defense = VERTDefense(
            client_ids,
            parameter_size=model_spec.parameter_size,
            history_window=config.vert_history_window,
            projection_dim=config.vert_projection_dim,
            predict_epochs=config.vert_predict_epochs,
            learning_rate=config.vert_predict_lr,
            top_k=config.vert_top_k,
            malicious_ratio_prior=(
                config.malicious_ratio
                if config.vert_use_ratio_prior
                else None
            ),
            compute_backend=config.compute_backend,
            device=config.device,
            seed=config.seed,
        )
    elif config.method == "fedredefense":
        fedre_defense = FedREDefense(
            client_ids,
            model_spec=model_spec,
            threshold=config.fedre_threshold,
            initial_iterations=config.fedre_initial_iterations,
            max_iterations=config.fedre_max_iterations,
            synthetic_steps=config.fedre_synthetic_steps,
            images_per_class=config.fedre_images_per_class,
            image_lr=config.fedre_image_lr,
            label_lr=config.fedre_label_lr,
            teacher_lr=config.fedre_teacher_lr,
            teacher_lr_lr=config.fedre_teacher_lr_lr,
            device=config.device,
            seed=config.seed,
        )

    if resume_state is not None:
        if config.method == "sm9rrs":
            detector = resume_state["detector"]
            sm9_weight_manager = resume_state["weight_manager"]
        elif config.method == "ding13":
            ding13_detector = resume_state["ding13_detector"]
        elif config.method == "vert":
            vert_defense = resume_state["vert_defense"]
        elif config.method == "fedredefense":
            fedre_defense = resume_state["fedre_defense"]

    task_exhausted = False
    if config.method == "sm9rrs":
        assert (
            crypto is not None
            and as_verifier is not None
            and auditor is not None
            and detector is not None
            and sm9_weight_manager is not None
        )
        # A failed audit is checkpointed with its complete immutable evidence.
        # Resolve it before accepting any later-round update so a restart cannot
        # strand a digest-only pending entry or silently bypass C_tol.
        for evidence in as_verifier.pending_audit_evidence(dataset.name):
            tag = as_verifier.tag_key(evidence.packet)
            if tag not in sm9_weight_manager.pending_trace:
                raise ValueError(
                    "checkpoint pending audit is absent from the weight-manager state"
                )
            try:
                audit_started = perf_counter()
                trace_result = _trace_and_archive(
                    as_verifier,
                    auditor,
                    evidence,
                )
                crypto_audit_wall_seconds += perf_counter() - audit_started
            except (RuntimeError, TypeError, ValueError) as exc:
                crypto_audit_wall_seconds += perf_counter() - audit_started
                raise RuntimeError(
                    "a restored SM9-RRS audit is still unresolved; task remains active"
                ) from exc
            sm9_weight_manager.confirm_revocation(tag)
            detector.forget(tag)
            blacklisted.add(trace_result.identity)
            if trace_result.identity in malicious_set:
                true_positive_revocations += 1
            else:
                false_positive_revocations += 1

        if as_verifier.pending_audit_digests(dataset.name):
            raise RuntimeError("restored audit queue was not closed after successful retry")
        if blacklisted:
            remaining_ring = [
                identity for identity in client_ids if identity not in blacklisted
            ]
            if remaining_ring:
                audit_started = perf_counter()
                crypto.update_task_ring(dataset.name, remaining_ring)
                crypto_audit_wall_seconds += perf_counter() - audit_started
            else:
                finalize_started = perf_counter()
                crypto.finalize_task(dataset.name)
                crypto_finalize_wall_seconds += perf_counter() - finalize_started
                task_exhausted = True

    if checkpoint_callback is not None and resume_state is None:
        # 第 1 轮开始前也建立一致状态。若首轮训练、密码运算或聚合即报错，
        # 修复后可从 round 0 恢复，不必重复初始化和初始评估。
        checkpoint_callback(
            _build_checkpoint_state(
                completed_round=0,
                params=params,
                records=records,
                diagnostics=diagnostics,
                blacklisted=blacklisted,
                true_positive_revocations=true_positive_revocations,
                false_positive_revocations=false_positive_revocations,
                detector=detector,
                weight_manager=sm9_weight_manager,
                ding13_detector=ding13_detector,
                vert_defense=vert_defense,
                fedre_defense=fedre_defense,
                crypto=crypto,
                training_seconds=training_seconds,
                attack_seconds=attack_seconds,
                hash_seconds=hash_seconds,
                packet_build_seconds=packet_build_seconds,
                sign_seconds=sign_seconds,
                verify_seconds=verify_seconds,
                detection_seconds=detection_seconds,
                aggregation_seconds=aggregation_seconds,
                evaluation_seconds=evaluation_seconds,
                crypto_setup_wall_seconds=crypto_setup_wall_seconds,
                crypto_packet_wall_seconds=crypto_packet_wall_seconds,
                crypto_audit_wall_seconds=crypto_audit_wall_seconds,
                crypto_finalize_wall_seconds=crypto_finalize_wall_seconds,
                nonfinite_updates=nonfinite_updates,
            )
        )

    for round_id in (
        () if task_exhausted else range(start_round, config.rounds + 1)
    ):
        # 每轮重新收集仍处于活跃状态的客户端更新；黑名单客户端不再参与训练。
        unresolved_audit_error: BaseException | None = None
        updates: list[Any] = []
        update_samples: list[int] = []
        update_clients: list[str] = []
        sm9_candidates: list[_ClientUpdateCandidate] = []
        suspicious_tags: set[str] = set()
        count_increment_tags: set[str] = set()
        sm9_candidates_by_tag: dict[str, _VerifiedSM9Candidate] = {}
        rejected = 0
        round_nonfinite_updates = 0

        for client_idx, identity in enumerate(client_ids):
            if identity in blacklisted:
                continue
            indices = client_indices[client_idx]
            attack_active = identity in malicious_set and round_id >= attack_start
            if attack_active and is_alternating_minimization_attack(config.attack):
                attack_started = perf_counter()
                delta, stats = _alternating_minimization_client_delta(
                    params,
                    dataset,
                    indices,
                    attack_target_indices=attack_target_indices,
                    client_idx=client_idx,
                    round_id=round_id,
                    model_spec=model_spec,
                    config=config,
                    torch_context=torch_context,
                )
                attack_seconds += perf_counter() - attack_started
            else:
                training_started = perf_counter()
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
                training_seconds += perf_counter() - training_started

            if attack_active and not is_alternating_minimization_attack(config.attack):
                attack_started = perf_counter()
                delta = _poison_client_update(
                    delta,
                    attack=config.attack,
                    scale=config.attack_scale,
                    seed=config.seed + round_id * 4099 + client_idx,
                    torch_context=torch_context,
                )
                attack_seconds += perf_counter() - attack_started

            # 任意 NaN/Inf 更新都不能进入摘要、检测或聚合。否则即使 SVD
            # 做了兜底，FedAvg/Krum 仍可能把全局模型永久污染为非有限值。
            if not _update_is_finite(delta):
                rejected += 1
                round_nonfinite_updates += 1
                nonfinite_updates += 1
                continue

            if config.method == "sm9rrs":
                # SM3 必须摘要服务器实际收到的字节，因此密码流程只在这里做一次
                # 必需的 D2H；设备端副本继续用于后面的聚合。
                cpu_delta = (
                    torch_context.to_numpy(delta)
                    if torch_context is not None
                    else np.asarray(delta, dtype=np.float32)
                )
                sm9_candidates.append(
                    _ClientUpdateCandidate(
                        identity=identity,
                        delta=delta,
                        cpu_delta=cpu_delta,
                        samples=stats.samples,
                    )
                )
                continue

            updates.append(delta)
            update_samples.append(stats.samples)
            update_clients.append(identity)

        if config.method == "sm9rrs" and sm9_candidates:
            # 密码封包可并行，轨迹检测必须按客户端稳定顺序更新历史状态。
            assert detector is not None and as_verifier is not None
            packet_started = perf_counter()
            sm9_result = _process_sm9_candidates(
                sm9_candidates,
                client_signers=client_signers,
                as_verifier=as_verifier,
                detector=detector,
                round_id=round_id,
                task_id=dataset.name,
                workers=config.sm9_workers,
            )
            crypto_packet_wall_seconds += max(
                0.0,
                perf_counter() - packet_started - sm9_result.detection_seconds,
            )
            updates.extend(sm9_result.updates)
            update_samples.extend(sm9_result.samples)
            update_clients.extend(sm9_result.tags)
            suspicious_tags.update(sm9_result.suspicious_tags)
            count_increment_tags.update(sm9_result.count_increment_tags)
            sm9_candidates_by_tag.update(sm9_result.candidates_by_tag)
            rejected += sm9_result.rejected
            hash_seconds += sm9_result.hash_seconds
            packet_build_seconds += sm9_result.packet_build_seconds
            sign_seconds += sm9_result.sign_seconds
            verify_seconds += sm9_result.verify_seconds
            detection_seconds += sm9_result.detection_seconds

        krum_selected = ""
        record_accepted = len(updates)
        record_rejected = rejected
        if updates:
            # 六种方法共享训练结果，只在服务端检测和聚合规则上分支。
            aggregation_started = perf_counter()
            detection_inside_aggregation = 0.0
            crypto_inside_aggregation = 0.0
            aggregate = None
            if config.method == "krum":
                active_neighbor_count = len(updates) - len(malicious_clients) - 2
                if active_neighbor_count < 1:
                    # 若多个非有限更新被拒绝后 Krum 条件临时失效，本轮保持
                    # 全局模型不变，而不是再次让整个长任务崩溃。
                    record_rejected += len(updates)
                else:
                    result = _krum(
                        updates,
                        byzantine_count=len(malicious_clients),
                        config=config,
                        torch_context=torch_context,
                    )
                    aggregate = result.update
                    krum_selected = update_clients[result.selected_index]
            elif config.method == "sm9rrs":
                assert (
                    sm9_weight_manager is not None
                    and crypto is not None
                    and as_verifier is not None
                    and auditor is not None
                )
                weight_before_by_tag = {
                    tag: sm9_weight_manager.weights.get(
                        tag,
                        1.0 / len(update_clients),
                    )
                    for tag in update_clients
                }
                count_before_by_tag = {
                    tag: sm9_weight_manager.consecutive_suspicions.get(tag, 0)
                    for tag in update_clients
                }
                weight_result = sm9_weight_manager.update(
                    update_clients,
                    suspicious_tags,
                    count_increment_tags,
                )
                accepted_trace_identities: set[str] = set()
                for tag in weight_result.trace_requested_tags:
                    candidate = sm9_candidates_by_tag.get(tag)
                    if candidate is None:
                        raise RuntimeError(
                            "C_tol trace request has no matching verified evidence"
                        )
                    try:
                        audit_started = perf_counter()
                        evidence = as_verifier.build_trace_evidence(
                            candidate.packet,
                            candidate.cpu_delta,
                        )
                        trace_result = _trace_and_archive(
                            as_verifier,
                            auditor,
                            evidence,
                        )
                        audit_elapsed = perf_counter() - audit_started
                        crypto_audit_wall_seconds += audit_elapsed
                        crypto_inside_aggregation += audit_elapsed
                    except (RuntimeError, TypeError, ValueError) as exc:
                        audit_elapsed = perf_counter() - audit_started
                        crypto_audit_wall_seconds += audit_elapsed
                        crypto_inside_aggregation += audit_elapsed
                        # Word 4.3.3 requires the C_tol trigger update to be
                        # rejected immediately.  Preserve its exact evidence
                        # for retry and keep its round weight at zero; no
                        # permanent identity removal occurs without Eq. (7).
                        unresolved_audit_error = exc
                        continue
                    sm9_weight_manager.confirm_revocation(tag)
                    detector.forget(tag)
                    accepted_trace_identities.add(trace_result.identity)
                    blacklisted.add(trace_result.identity)
                    if trace_result.identity in malicious_set:
                        true_positive_revocations += 1
                    else:
                        false_positive_revocations += 1

                if accepted_trace_identities:
                    remaining_ring = [
                        identity for identity in client_ids if identity not in blacklisted
                    ]
                    if remaining_ring:
                        audit_started = perf_counter()
                        crypto.update_task_ring(dataset.name, remaining_ring)
                        audit_elapsed = perf_counter() - audit_started
                        crypto_audit_wall_seconds += audit_elapsed
                        crypto_inside_aggregation += audit_elapsed
                    else:
                        # There is no valid non-empty ring to install.  Close
                        # the task immediately so the revoked signer cannot use
                        # stale client/AS material from the previous RID.
                        finalize_started = perf_counter()
                        crypto.finalize_task(dataset.name)
                        finalize_elapsed = perf_counter() - finalize_started
                        crypto_finalize_wall_seconds += finalize_elapsed
                        crypto_inside_aggregation += finalize_elapsed
                        task_exhausted = True

                for tag in update_clients:
                    decision = sm9_result.decisions_by_tag[tag]
                    identity = sm9_result.client_ids_by_tag[tag]
                    z_sigma = float(getattr(decision, "z_sigma", 0.0))
                    z_direction = float(
                        getattr(decision, "z_direction", 0.0)
                    )
                    z_spectrum_adjacent = float(
                        getattr(decision, "z_spectrum_adjacent", 0.0)
                    )
                    z_subspace_adjacent = float(
                        getattr(decision, "z_subspace_adjacent", 0.0)
                    )
                    z_spectrum_anchor = float(
                        getattr(decision, "z_spectrum_anchor", 0.0)
                    )
                    z_subspace_anchor = float(
                        getattr(decision, "z_subspace_anchor", 0.0)
                    )
                    direction_reliability = float(
                        getattr(decision, "direction_reliability", 0.0)
                    )
                    diagnostics.append(
                        ClientDiagnosticRecord(
                            round=round_id,
                            client_id=identity,
                            task_tag=tag,
                            is_malicious=identity in malicious_set,
                            decision_reason=str(
                                getattr(decision, "reason", "unknown")
                            ),
                            z_sigma=z_sigma,
                            z_direction=z_direction,
                            sigma_delta=float(
                                getattr(decision, "sigma_delta", 0.0)
                            ),
                            cosine_similarity=float(
                                getattr(decision, "cosine_similarity", 0.0)
                            ),
                            sigma_exceeded=(
                                z_spectrum_adjacent > adjacent_threshold
                                or z_spectrum_anchor > anchor_threshold
                            ),
                            direction_exceeded=(
                                direction_reliability * z_subspace_adjacent
                                > adjacent_threshold
                                or direction_reliability * z_subspace_anchor
                                > anchor_threshold
                            ),
                            suspicious=tag in suspicious_tags,
                            count_increment=tag in count_increment_tags,
                            weight_before=float(weight_before_by_tag[tag]),
                            weight_after_penalty_recovery=float(
                                weight_result.pre_normalization_weights[tag]
                            ),
                            aggregation_weight=float(
                                sm9_weight_manager.weights[tag]
                            ),
                            count_before=int(count_before_by_tag[tag]),
                            count_after=int(
                                sm9_weight_manager.consecutive_suspicions[tag]
                            ),
                            trace_requested=(
                                tag in weight_result.trace_requested_tags
                            ),
                            trace_pending=tag in sm9_weight_manager.pending_trace,
                            revoked=tag in sm9_weight_manager.revoked,
                            spectrum_adjacent_distance=float(
                                getattr(
                                    decision,
                                    "spectrum_adjacent_distance",
                                    0.0,
                                )
                            ),
                            subspace_adjacent_distance=float(
                                getattr(
                                    decision,
                                    "subspace_adjacent_distance",
                                    0.0,
                                )
                            ),
                            spectrum_anchor_distance=float(
                                getattr(
                                    decision,
                                    "spectrum_anchor_distance",
                                    0.0,
                                )
                            ),
                            subspace_anchor_distance=float(
                                getattr(
                                    decision,
                                    "subspace_anchor_distance",
                                    0.0,
                                )
                            ),
                            z_spectrum_adjacent=z_spectrum_adjacent,
                            z_subspace_adjacent=z_subspace_adjacent,
                            z_spectrum_anchor=z_spectrum_anchor,
                            z_subspace_anchor=z_subspace_anchor,
                            adjacent_score=float(
                                getattr(decision, "adjacent_score", 0.0)
                            ),
                            anchor_score=float(
                                getattr(decision, "anchor_score", 0.0)
                            ),
                            cumulative_drift=float(
                                getattr(decision, "cumulative_drift", 0.0)
                            ),
                            spectral_gap=float(
                                getattr(decision, "spectral_gap", 0.0)
                            ),
                            direction_reliability=direction_reliability,
                            adjacent_exceeded=bool(
                                getattr(decision, "adjacent_exceeded", False)
                            ),
                            anchor_exceeded=bool(
                                getattr(decision, "anchor_exceeded", False)
                            ),
                            drift_exceeded=bool(
                                getattr(decision, "drift_exceeded", False)
                            ),
                            trusted_history_size=int(
                                getattr(decision, "trusted_history_size", 0)
                            ),
                            attack_active=(
                                identity in malicious_set
                                and round_id >= attack_start
                                and config.attack != "none"
                            ),
                        )
                    )

                # The manager contains the final zero/non-zero decision for
                # every C_tol trigger and any successfully revoked tag.
                weights = [sm9_weight_manager.weights[tag] for tag in update_clients]
                # Section 4.3.3 already normalizes w_pi over A^(r).  Applying
                # sample counts again would implement w_pi*n_pi rather than
                # the aggregation equation stated in the Word scheme.
                effective_total = sum(weights)
                if effective_total <= 0.0:
                    aggregate = (
                        torch_context.zeros_like(updates[0])
                        if torch_context is not None
                        else np.zeros_like(updates[0])
                    )
                else:
                    aggregate = _weighted_fedavg(
                        updates,
                        weights,
                        sample_counts=None,
                        config=config,
                        torch_context=torch_context,
                    )
                record_accepted = sum(
                    1 for weight, samples in zip(weights, update_samples) if weight > 0.0 and samples > 0
                )
                # Single anomalies are downweighted but still aggregated.  Only
                # C_tol-trigger packets have zero weight and are rejected.
                record_rejected = rejected + sum(weight <= 0.0 for weight in weights)
            elif config.method == "ding13":
                assert ding13_detector is not None
                update_by_client = dict(zip(update_clients, updates))
                detection_started = perf_counter()
                ding13_result = ding13_detector.evaluate_round(
                    update_by_client,
                    malicious_set,
                    round_id=round_id,
                )
                detection_elapsed = perf_counter() - detection_started
                detection_seconds += detection_elapsed
                detection_inside_aggregation += detection_elapsed
                blacklisted.update(ding13_result.newly_removed)
                true_positive_revocations += ding13_result.true_positive_removed
                false_positive_revocations += ding13_result.false_positive_removed
                weights = [ding13_result.weights[identity] for identity in update_clients]
                aggregate = _weighted_fedavg(
                    updates,
                    weights,
                    sample_counts=update_samples,
                    config=config,
                    torch_context=torch_context,
                )
                record_accepted = sum(
                    1 for weight, samples in zip(weights, update_samples) if weight > 0.0 and samples > 0
                )
                record_rejected = rejected + len(ding13_result.outliers)
            elif config.method == "vert":
                assert vert_defense is not None
                cpu_update_by_client = _cpu_updates_by_client(
                    update_clients,
                    updates,
                    torch_context,
                )
                detection_started = perf_counter()
                vert_result = vert_defense.evaluate_round(
                    cpu_update_by_client,
                    round_id=round_id,
                )
                detection_elapsed = perf_counter() - detection_started
                detection_seconds += detection_elapsed
                detection_inside_aggregation += detection_elapsed
                selected_set = set(vert_result.selected_clients)
                selected_indices = [
                    index
                    for index, identity in enumerate(update_clients)
                    if identity in selected_set
                ]
                selected_updates = [updates[index] for index in selected_indices]
                selected_weights = [
                    vert_result.weights[update_clients[index]]
                    for index in selected_indices
                ]
                aggregate = _weighted_fedavg(
                    selected_updates,
                    selected_weights,
                    sample_counts=None,
                    config=config,
                    torch_context=torch_context,
                )
                detection_started = perf_counter()
                vert_defense.finalize_round(
                    cpu_update_by_client,
                    (
                        torch_context.to_numpy(aggregate)
                        if torch_context is not None
                        else np.asarray(aggregate, dtype=np.float32)
                    ),
                    vert_result,
                )
                detection_elapsed = perf_counter() - detection_started
                detection_seconds += detection_elapsed
                detection_inside_aggregation += detection_elapsed
                record_accepted = len(selected_indices)
                record_rejected = rejected + len(vert_result.rejected_clients)
            elif config.method == "fedredefense":
                assert fedre_defense is not None
                cpu_update_by_client = _cpu_updates_by_client(
                    update_clients,
                    updates,
                    torch_context,
                )
                detection_started = perf_counter()
                fedre_result = fedre_defense.evaluate_round(
                    _params_for_checkpoint(params),
                    cpu_update_by_client,
                    round_id=round_id,
                )
                detection_elapsed = perf_counter() - detection_started
                detection_seconds += detection_elapsed
                detection_inside_aggregation += detection_elapsed
                newly_removed = set(fedre_result.rejected_clients) - blacklisted
                blacklisted.update(newly_removed)
                true_positive_revocations += len(newly_removed & malicious_set)
                false_positive_revocations += len(newly_removed - malicious_set)
                accepted_set = set(fedre_result.accepted_clients)
                accepted_indices = [
                    index
                    for index, identity in enumerate(update_clients)
                    if identity in accepted_set
                ]
                if accepted_indices:
                    # The official FedREDefense path applies equal-client
                    # FedAvg after filtering rather than reusing sample counts.
                    aggregate = _fedavg(
                        [updates[index] for index in accepted_indices],
                        None,
                        config=config,
                        torch_context=torch_context,
                    )
                record_accepted = len(accepted_indices)
                record_rejected = rejected + len(fedre_result.rejected_clients)
            else:
                aggregate = _fedavg(
                    updates,
                    update_samples,
                    config=config,
                    torch_context=torch_context,
                )
            if aggregate is not None:
                params = (
                    torch_context.add_update(params, aggregate)
                    if torch_context is not None
                    else (params + aggregate).astype(np.float32)
                )
            aggregation_seconds += (
                perf_counter()
                - aggregation_started
                - detection_inside_aggregation
                - crypto_inside_aggregation
            )

        should_evaluate = round_id == config.rounds or round_id % config.eval_interval == 0
        if should_evaluate:
            evaluation_started = perf_counter()
            acc = _evaluate_accuracy(params, dataset, model_spec, config, torch_context)
            (
                target_success,
                target_confidence,
            ) = _evaluate_attack_target_metrics(
                params,
                dataset,
                attack_target_indices,
                model_spec,
                config,
                torch_context,
            )
            evaluation_seconds += perf_counter() - evaluation_started
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
                    attack_target_success=target_success,
                    attack_target_confidence=target_confidence,
                    nonfinite_updates=round_nonfinite_updates,
                )
            )
            can_stop = not malicious_clients or round_id >= attack_start
            should_stop = config.early_stop and can_stop and 1.0 - acc <= config.target_error
        else:
            should_stop = False

        if checkpoint_callback is not None:
            checkpoint_callback(
                _build_checkpoint_state(
                    completed_round=round_id,
                    params=params,
                    records=records,
                    diagnostics=diagnostics,
                    blacklisted=blacklisted,
                    true_positive_revocations=true_positive_revocations,
                    false_positive_revocations=false_positive_revocations,
                    detector=detector,
                    weight_manager=sm9_weight_manager,
                    ding13_detector=ding13_detector,
                    vert_defense=vert_defense,
                    fedre_defense=fedre_defense,
                    crypto=crypto,
                    training_seconds=training_seconds,
                    attack_seconds=attack_seconds,
                    hash_seconds=hash_seconds,
                    packet_build_seconds=packet_build_seconds,
                    sign_seconds=sign_seconds,
                    verify_seconds=verify_seconds,
                    detection_seconds=detection_seconds,
                    aggregation_seconds=aggregation_seconds,
                    evaluation_seconds=evaluation_seconds,
                    crypto_setup_wall_seconds=crypto_setup_wall_seconds,
                    crypto_packet_wall_seconds=crypto_packet_wall_seconds,
                    crypto_audit_wall_seconds=crypto_audit_wall_seconds,
                    crypto_finalize_wall_seconds=crypto_finalize_wall_seconds,
                    nonfinite_updates=nonfinite_updates,
                )
            )
        if config.method == "sm9rrs" and unresolved_audit_error is not None:
            raise RuntimeError(
                "SM9-RRS trace remains pending; retry from the durable checkpoint"
            ) from unresolved_audit_error
        if task_exhausted:
            break
        if should_stop:
            break

    # Every trace result accepted above was explicitly archived through the AS
    # capability. finalize_task checks its own pending-evidence registry before
    # destroying h_t, kappa_t and cached task-linking material.
    if crypto is not None:
        if not crypto.is_task_finalized(dataset.name):
            finalize_started = perf_counter()
            crypto.finalize_task(dataset.name)
            crypto_finalize_wall_seconds += perf_counter() - finalize_started
        if checkpoint_callback is not None:
            checkpoint_callback(
                _build_checkpoint_state(
                    completed_round=records[-1].round,
                    params=params,
                    records=records,
                    diagnostics=diagnostics,
                    blacklisted=blacklisted,
                    true_positive_revocations=true_positive_revocations,
                    false_positive_revocations=false_positive_revocations,
                    detector=detector,
                    weight_manager=sm9_weight_manager,
                    ding13_detector=ding13_detector,
                    vert_defense=vert_defense,
                    fedre_defense=fedre_defense,
                    crypto=crypto,
                    training_seconds=training_seconds,
                    attack_seconds=attack_seconds,
                    hash_seconds=hash_seconds,
                    packet_build_seconds=packet_build_seconds,
                    sign_seconds=sign_seconds,
                    verify_seconds=verify_seconds,
                    detection_seconds=detection_seconds,
                    aggregation_seconds=aggregation_seconds,
                    evaluation_seconds=evaluation_seconds,
                    crypto_setup_wall_seconds=crypto_setup_wall_seconds,
                    crypto_packet_wall_seconds=crypto_packet_wall_seconds,
                    crypto_audit_wall_seconds=crypto_audit_wall_seconds,
                    crypto_finalize_wall_seconds=crypto_finalize_wall_seconds,
                    nonfinite_updates=nonfinite_updates,
                )
            )

    final = records[-1]
    return ExperimentResult(
        config=config,
        records=records,
        diagnostics=diagnostics,
        final_accuracy=final.accuracy,
        final_error=final.error,
        stopped_round=final.round,
        malicious_clients=tuple(malicious_clients),
        blacklisted_clients=tuple(sorted(blacklisted)),
        stage_timings=StageTimings(
            training_seconds=training_seconds,
            attack_seconds=attack_seconds,
            hash_seconds=hash_seconds,
            packet_build_seconds=packet_build_seconds,
            sign_seconds=sign_seconds,
            verify_seconds=verify_seconds,
            detection_seconds=detection_seconds,
            aggregation_seconds=aggregation_seconds,
            evaluation_seconds=evaluation_seconds,
            crypto_setup_wall_seconds=crypto_setup_wall_seconds,
            crypto_packet_wall_seconds=crypto_packet_wall_seconds,
            crypto_audit_wall_seconds=crypto_audit_wall_seconds,
            crypto_finalize_wall_seconds=crypto_finalize_wall_seconds,
        ),
        nonfinite_updates=nonfinite_updates,
    )


def _trace_and_archive(
    as_verifier: ASVerifier,
    auditor: AuditorService,
    evidence,
):
    """Complete one pending audit or leave its durable entry untouched."""

    trace_result = auditor.trace(evidence)
    if not as_verifier.verify_trace_result(evidence, trace_result):
        raise ValueError("AS rejected the returned threshold trace result")
    if not as_verifier.archive_trace_result(evidence, trace_result):
        raise RuntimeError("AS could not archive the verified trace result")
    return trace_result


def _effective_detector_settings(
    dataset: ImageDataset,
    config: ExperimentConfig,
) -> tuple[int, float, int]:
    del dataset
    # Word 4.3.3 uses the configured K, theta=3, and C_tol without a
    # dataset-specific hidden override.
    return config.detector_window, config.z_threshold, config.suspicion_remove_after


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
    client_signers: dict[str, ClientSigner],
    as_verifier: ASVerifier,
    detector: LongitudinalSVDDetector,
    round_id: int,
    task_id: str,
    workers: int,
) -> _SM9ProcessingResult:
    if workers > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(candidates))) as executor:
            verified_candidates = list(
                executor.map(
                    lambda candidate: _create_verified_sm9_candidate(
                        candidate,
                        signer=client_signers[candidate.identity],
                        as_verifier=as_verifier,
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
                signer=client_signers[candidate.identity],
                as_verifier=as_verifier,
                round_id=round_id,
                task_id=task_id,
            )
            for candidate in candidates
        ]

    updates: list[np.ndarray] = []
    samples: list[int] = []
    tags: list[str] = []
    suspicious_tags: set[str] = set()
    count_increment_tags: set[str] = set()
    candidates_by_tag: dict[str, _VerifiedSM9Candidate] = {}
    decisions_by_tag: dict[str, DetectionResult] = {}
    client_ids_by_tag: dict[str, str] = {}
    rejected = 0
    detection_seconds = 0.0
    for source_candidate, candidate in zip(candidates, verified_candidates):
        if not candidate.verified:
            rejected += 1
            continue
        tag = as_verifier.tag_key(candidate.packet)
        # One authenticated Tag may submit at most once in a round.  The FL
        # driver normally produces one packet per client; this guard preserves
        # the server-side protocol invariant under adversarial inputs.
        if not tag or tag in candidates_by_tag:
            rejected += 1
            continue
        detection_started = perf_counter()
        try:
            decision = detector.evaluate(
                tag,
                candidate.cpu_delta,
                round_id=round_id,
            )
        except (TypeError, ValueError):
            # A correctly signed but schema-invalid model vector is not a
            # valid G_pi^(r); reject it without mutating the tag trajectory.
            detection_seconds += perf_counter() - detection_started
            rejected += 1
            continue
        detection_seconds += perf_counter() - detection_started
        if not decision.accepted:
            suspicious_tags.add(tag)
            if decision.count_increment:
                count_increment_tags.add(tag)
        updates.append(candidate.delta)
        samples.append(candidate.samples)
        tags.append(tag)
        candidates_by_tag[tag] = candidate
        decisions_by_tag[tag] = decision
        client_ids_by_tag[tag] = source_candidate.identity
    return _SM9ProcessingResult(
        updates=updates,
        samples=samples,
        tags=tags,
        suspicious_tags=suspicious_tags,
        count_increment_tags=count_increment_tags,
        candidates_by_tag=candidates_by_tag,
        decisions_by_tag=decisions_by_tag,
        client_ids_by_tag=client_ids_by_tag,
        rejected=rejected,
        # 多线程时这些值是所有客户端操作耗时之和，用来判断热点而非相加还原墙钟时间。
        hash_seconds=sum(candidate.hash_seconds for candidate in verified_candidates),
        packet_build_seconds=sum(
            candidate.packet_build_seconds for candidate in verified_candidates
        ),
        sign_seconds=sum(candidate.sign_seconds for candidate in verified_candidates),
        verify_seconds=sum(candidate.verify_seconds for candidate in verified_candidates),
        detection_seconds=detection_seconds,
    )


def _create_verified_sm9_candidate(
    candidate: _ClientUpdateCandidate,
    *,
    signer: ClientSigner,
    as_verifier: ASVerifier,
    round_id: int,
    task_id: str,
) -> _VerifiedSM9Candidate:
    # 摘要算法由密码上下文决定：真实模式使用 SM3，仿真模式走快速 SHA-256。
    hash_started = perf_counter()
    update_digest = signer.digest_update(candidate.cpu_delta)
    hash_seconds = perf_counter() - hash_started

    packet_started = perf_counter()
    unsigned = signer.build_unsigned_packet(
        candidate.cpu_delta,
        round_id=round_id,
        task_id=task_id,
        update_digest=update_digest,
    )
    packet_build_seconds = perf_counter() - packet_started

    sign_started = perf_counter()
    packet = signer.sign_packet(unsigned)
    sign_seconds = perf_counter() - sign_started

    verify_started = perf_counter()
    verified = as_verifier.verify_packet(
        packet,
        candidate.cpu_delta,
        expected_task_id=task_id,
        expected_round_id=round_id,
    )
    verify_seconds = perf_counter() - verify_started
    return _VerifiedSM9Candidate(
        delta=candidate.delta,
        cpu_delta=candidate.cpu_delta,
        samples=candidate.samples,
        packet=packet,
        verified=verified,
        hash_seconds=hash_seconds,
        packet_build_seconds=packet_build_seconds,
        sign_seconds=sign_seconds,
        verify_seconds=verify_seconds,
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


def _evaluate_attack_target_metrics(
    params,
    dataset: ImageDataset,
    attack_target_indices: np.ndarray,
    model_spec,
    config: ExperimentConfig,
    torch_context,
) -> tuple[float | None, float | None]:
    """Evaluate the targeted objective separately from clean test accuracy."""

    if len(attack_target_indices) == 0:
        return None, None
    if torch_context is not None:
        return torch_context.targeted_metrics(
            params,
            target_indices=attack_target_indices,
            target_label=config.attack_target_label,
        )
    labels = np.full(
        len(attack_target_indices),
        config.attack_target_label,
        dtype=np.int64,
    )
    return targeted_metrics(
        params,
        dataset.x_test[attack_target_indices],
        labels,
        spec=model_spec,
        compute_backend=config.compute_backend,
        device=config.device,
    )


def _params_for_checkpoint(params) -> np.ndarray:
    """把可能驻留 GPU 的全局参数转换为可移植的 float32 检查点。"""

    if type(params).__module__.startswith("torch"):
        return params.detach().cpu().numpy().astype(np.float32, copy=True)
    return np.asarray(params, dtype=np.float32).copy()


def _cpu_updates_by_client(
    client_ids: list[str],
    updates: list[Any],
    torch_context,
) -> dict[str, np.ndarray]:
    """Materialize one CPU copy only for defenses that inspect full updates."""

    if len(client_ids) != len(updates):
        raise ValueError("client_ids and updates must have the same length")
    return {
        client_id: (
            torch_context.to_numpy(update).copy()
            if torch_context is not None
            else np.asarray(update, dtype=np.float32).copy()
        )
        for client_id, update in zip(client_ids, updates)
    }


def _build_checkpoint_state(
    *,
    completed_round: int,
    params,
    records: list[RoundRecord],
    diagnostics: list[ClientDiagnosticRecord],
    blacklisted: set[str],
    true_positive_revocations: int,
    false_positive_revocations: int,
    detector,
    weight_manager,
    ding13_detector,
    vert_defense,
    fedre_defense,
    crypto: SM9RRSContext | None,
    training_seconds: float,
    attack_seconds: float,
    hash_seconds: float,
    packet_build_seconds: float,
    sign_seconds: float,
    verify_seconds: float,
    detection_seconds: float,
    aggregation_seconds: float,
    evaluation_seconds: float,
    crypto_setup_wall_seconds: float,
    crypto_packet_wall_seconds: float,
    crypto_audit_wall_seconds: float,
    crypto_finalize_wall_seconds: float,
    nonfinite_updates: int,
) -> dict[str, Any]:
    """只在轮次边界构造一致快照，绝不保存完成一半的聚合状态。"""

    return {
        "completed_round": completed_round,
        "params": _params_for_checkpoint(params),
        "records": list(records),
        "diagnostics": list(diagnostics),
        "blacklisted": tuple(sorted(blacklisted)),
        "true_positive_revocations": true_positive_revocations,
        "false_positive_revocations": false_positive_revocations,
        "nonfinite_updates": nonfinite_updates,
        "detector": detector,
        "weight_manager": weight_manager,
        "ding13_detector": ding13_detector,
        "vert_defense": vert_defense,
        "fedre_defense": fedre_defense,
        "crypto_state": crypto.export_state() if crypto is not None else None,
        "timings": {
            "training_seconds": training_seconds,
            "attack_seconds": attack_seconds,
            "hash_seconds": hash_seconds,
            "packet_build_seconds": packet_build_seconds,
            "sign_seconds": sign_seconds,
            "verify_seconds": verify_seconds,
            "detection_seconds": detection_seconds,
            "aggregation_seconds": aggregation_seconds,
            "evaluation_seconds": evaluation_seconds,
            "crypto_setup_wall_seconds": crypto_setup_wall_seconds,
            "crypto_packet_wall_seconds": crypto_packet_wall_seconds,
            "crypto_audit_wall_seconds": crypto_audit_wall_seconds,
            "crypto_finalize_wall_seconds": crypto_finalize_wall_seconds,
        },
    }


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
        return torch_context.local_train_delta_resident(
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


def _alternating_minimization_client_delta(
    params: np.ndarray,
    dataset: ImageDataset,
    indices: np.ndarray,
    *,
    attack_target_indices: np.ndarray,
    client_idx: int,
    round_id: int,
    model_spec,
    config: ExperimentConfig,
    torch_context,
):
    """Optimize the target and stealth objectives inside malicious training."""

    if len(attack_target_indices) == 0:
        raise ValueError(
            "alternating minimization has no selected auxiliary target samples"
        )
    seed = config.seed + round_id * 1009 + client_idx
    lr = config.lr * (config.lr_decay ** (round_id - 1))
    if torch_context is not None:
        return torch_context.alternating_minimization_delta_resident(
            params,
            client_idx=client_idx,
            target_indices=attack_target_indices,
            target_label=config.attack_target_label,
            lr=lr,
            attack_epochs=config.attack_epochs,
            batch_size=config.batch_size,
            stealth_steps=config.attack_stealth_steps,
            boost=config.attack_boost,
            distance_weight=config.attack_distance_weight,
            seed=seed,
        )

    target_labels = np.full(
        len(attack_target_indices),
        config.attack_target_label,
        dtype=np.int64,
    )
    return alternating_minimization_delta(
        params,
        dataset.x_train[indices],
        dataset.y_train[indices],
        dataset.x_test[attack_target_indices],
        target_labels,
        lr=lr,
        attack_epochs=config.attack_epochs,
        batch_size=config.batch_size,
        stealth_steps=config.attack_stealth_steps,
        boost=config.attack_boost,
        distance_weight=config.attack_distance_weight,
        seed=seed,
        spec=model_spec,
    )


def _select_attack_target_indices(
    dataset: ImageDataset,
    config: ExperimentConfig,
    *,
    required: bool = True,
) -> np.ndarray:
    """Select deterministic held-out auxiliary samples for attack/evaluation.

    A no-malicious control run records the same targeted metrics whenever its
    limited test split contains enough source-class samples.  An active attack
    must have the requested auxiliary set and therefore fails explicitly.
    """

    labels = np.asarray(dataset.y_test, dtype=np.int64)
    candidates = np.flatnonzero(labels == config.attack_source_label)
    if len(candidates) < config.attack_target_count:
        if not required:
            return np.empty(0, dtype=np.int64)
        raise ValueError(
            "not enough held-out source-label samples for alternating minimization: "
            f"label={config.attack_source_label}, requested={config.attack_target_count}, "
            f"available={len(candidates)}"
        )
    rng = np.random.default_rng(config.seed + 271_828)
    chosen = rng.choice(
        candidates,
        size=config.attack_target_count,
        replace=False,
    )
    return np.sort(np.asarray(chosen, dtype=np.int64))


def _fedavg(
    updates: list[Any],
    sample_counts: list[int] | None,
    *,
    config: ExperimentConfig,
    torch_context=None,
):
    weights = (
        np.ones(len(updates), dtype=np.float64)
        if sample_counts is None
        else sample_counts
    )
    if torch_context is not None:
        return torch_context.weighted_average(updates, weights)
    if _can_use_torch_ops(config):
        return torch_weighted_fedavg(updates, weights, device=config.device)
    return fedavg(updates, sample_counts)


def _weighted_fedavg(
    updates: list[Any],
    weights: list[float],
    *,
    sample_counts: list[int] | None,
    config: ExperimentConfig,
    torch_context=None,
):
    if torch_context is not None:
        return torch_context.weighted_average(updates, weights, sample_counts)
    if _can_use_torch_ops(config):
        return torch_weighted_fedavg(updates, weights, sample_counts=sample_counts, device=config.device)
    return weighted_fedavg(updates, weights, sample_counts=sample_counts)


def _krum(
    updates: list[Any],
    *,
    byzantine_count: int,
    config: ExperimentConfig,
    torch_context=None,
):
    if torch_context is not None:
        selected, update, scores = torch_context.krum_select(
            updates,
            byzantine_count=byzantine_count,
        )
        return KrumResult(
            update=update,
            selected_index=selected,
            scores=scores,
            neighbor_count=len(updates) - byzantine_count - 2,
        )
    if _can_use_torch_ops(config):
        return torch_krum(updates, byzantine_count=byzantine_count, device=config.device)
    return krum(updates, byzantine_count=byzantine_count)


def _poison_client_update(update, *, attack: str, scale: float, seed: int, torch_context):
    if torch_context is not None:
        return torch_context.poison_update(
            update,
            attack=attack,
            scale=scale,
            seed=seed,
        )
    return poison_update(update, attack=attack, scale=scale, seed=seed)


def _update_is_finite(update) -> bool:
    """同时支持 NumPy 和驻留设备的 Torch 更新，只回传一个布尔标量。"""

    if type(update).__module__.startswith("torch"):
        return bool(update.isfinite().all().detach().cpu().item())
    return bool(np.isfinite(np.asarray(update)).all())


def _can_use_torch_ops(config: ExperimentConfig) -> bool:
    try:
        from .torch_backend import should_use_torch
    except RuntimeError:
        return False
    return should_use_torch(config.compute_backend, config.device)


def _choose_malicious(client_ids: list[str], ratio: float, seed: int) -> tuple[str, ...]:
    count = malicious_client_count(len(client_ids), ratio)
    rng = np.random.default_rng(seed + 17)
    if count == 0:
        return tuple()
    selected = rng.choice(client_ids, size=count, replace=False)
    return tuple(sorted(str(item) for item in selected))


def malicious_client_count(num_clients: int, ratio: float) -> int:
    """按实验原有取整规则计算恶意客户端数。"""

    count = int(round(num_clients * ratio))
    return min(max(count, 0), num_clients - 1)


def experiment_config_error(config: ExperimentConfig) -> str | None:
    """返回配置在算法定义上不可执行的原因；可执行时返回 ``None``。"""

    if config.method != "krum":
        return None
    malicious_count = malicious_client_count(
        config.num_clients,
        config.malicious_ratio,
    )
    neighbor_count = config.num_clients - malicious_count - 2
    if config.num_clients < 3:
        return (
            "Krum is undefined: at least 3 clients are required "
            f"(n={config.num_clients})"
        )
    if neighbor_count < 1:
        return (
            "Krum is undefined for this configuration: "
            f"n={config.num_clients}, f={malicious_count}, "
            f"n-f-2={neighbor_count}; require n-f-2 >= 1"
        )
    return None


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
    attack_target_success: float | None = None,
    attack_target_confidence: float | None = None,
    nonfinite_updates: int = 0,
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
        attack_target_success_rate=(
            float(attack_target_success)
            if attack_target_success is not None
            else None
        ),
        attack_target_confidence=(
            float(attack_target_confidence)
            if attack_target_confidence is not None
            else None
        ),
        nonfinite_updates=nonfinite_updates,
        attack_active=(
            config.malicious_ratio > 0.0
            and config.attack != "none"
            and round_id >= (config.attack_start_round or config.detector_window + 2)
        ),
    )
