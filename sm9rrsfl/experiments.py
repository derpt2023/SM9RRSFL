"""Command line runner for image poisoning experiments."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, fields, is_dataclass, replace
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import resource
import shutil
import sys
from threading import Event, Lock, Thread
from time import perf_counter
import traceback

import numpy as np

from .datasets import load_image_dataset
from .crypto import rrs_backend_name, sm3_backend_name
from .fl import (
    ClientDiagnosticRecord,
    ExperimentConfig,
    ExperimentResult,
    RoundRecord,
    StageTimings,
    experiment_config_error,
    malicious_client_count,
    run_experiment,
)
from .model import describe_compute_backend
from .visualization import generate_visualizations


DEFAULT_RATIOS = (0.00, 0.10, 0.20, 0.40, 0.45, 0.60, 0.80)
DEFAULT_OUTPUT_ROOT = Path("outputs")
PROGRESS_BAR_WIDTH = 28
_WORKER_DATASET = None
_WORKER_CHECKPOINT_DIR = None
_WORKER_RUN_FINGERPRINT = None
_CHECKPOINT_WRITE_LOCK = Lock()
CHECKPOINT_SCHEMA_VERSION = 11
COMPLETED_RESULTS_SNAPSHOT = ".completed_results.pickle"
CUDA_MEMORY_SAFETY_FRACTION = 0.75
DATASET_TRAINING_PRESETS = {
    "mnist": {
        "rounds": 30,
        "local_epochs": 1,
        "batch_size": 32,
        "lr": 0.05,
        "lr_decay": 1.0,
    },
    "synthetic": {
        "rounds": 30,
        "local_epochs": 1,
        "batch_size": 32,
        "lr": 0.05,
        "lr_decay": 1.0,
    },
    "cifar10": {
        "rounds": 300,
        "local_epochs": 5,
        "batch_size": 50,
        "lr": 0.05,
        "lr_decay": 0.99,
    },
}


def main(argv: list[str] | None = None) -> None:
    """Run experiments from explicit CLI arguments or the process command line."""

    args = parse_args(argv)
    output_dir = resolve_output_dir(args)
    if args.visualize_only:
        results = read_results(output_dir / "summary.csv", output_dir / "rounds.csv")
        visualization_path = generate_visualizations(results, output_dir)
        print(f"wrote {visualization_path}")
        return

    backend_description = describe_compute_backend(
        args.compute_backend,
        args.device,
    )
    print(f"compute_backend={backend_description}", flush=True)
    print(
        "execution_request="
        f"jobs={args.jobs} sm9_workers={'auto' if args._sm9_workers_auto else args.sm9_workers} "
        f"device={args.device}",
        flush=True,
    )
    if "sm9rrs" in args.methods:
        print(f"sm3_backend={sm3_backend_name()}", flush=True)
        if args.crypto_mode == "sm9":
            print(f"rrs_backend={rrs_backend_name()}", flush=True)

    dataset = load_image_dataset(
        args.dataset,
        args.data_dir,
        download=args.download,
        train_limit=args.train_samples,
        test_limit=args.test_samples,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    configs = build_experiment_configs(args)
    cuda_worker_mb = _estimated_cuda_worker_memory_mb(dataset, configs)
    usable_cuda_devices = (
        cuda_devices_with_capacity(cuda_worker_mb)
        if backend_description == "torch:cuda"
        and args.device.strip().lower() == "auto"
        else available_cuda_devices()
    )
    if (
        backend_description == "torch:cuda"
        and args.device.strip().lower() == "auto"
        and not usable_cuda_devices
    ):
        raise RuntimeError(_cuda_capacity_error(cuda_worker_mb))
    planned_jobs = resolve_parallel_jobs(args.jobs, dataset, configs, args)
    if args._sm9_workers_auto:
        effective_sm9_workers = resolve_sm9_workers(
            "auto",
            max(config.num_clients for config in configs),
            parallel_jobs=planned_jobs,
        )
        args.sm9_workers = effective_sm9_workers
        configs = [replace(config, sm9_workers=effective_sm9_workers) for config in configs]
    configs = assign_auto_cuda_devices(
        configs,
        backend_description,
        args.device,
        cuda_devices=usable_cuda_devices,
    )
    skipped_configs = [
        (config, reason)
        for config in configs
        if (reason := experiment_config_error(config)) is not None
    ]
    runnable_configs = [
        config for config in configs if experiment_config_error(config) is None
    ]
    manifest = build_run_manifest(args, dataset, configs)
    previous_manifest = read_run_manifest(output_dir)
    exact_manifest_match = bool(
        args.resume
        and previous_manifest is not None
        and previous_manifest.get("fingerprint") == manifest["fingerprint"]
    )
    runtime_assignment_match = bool(
        args.resume
        and previous_manifest is not None
        and _manifests_differ_only_by_indexed_cuda_device(
            previous_manifest,
            manifest,
        )
    )
    current_manifest_matches = exact_manifest_match or runtime_assignment_match
    results: list[ExperimentResult] = []
    archived_resume_dir = None
    if current_manifest_matches:
        if runtime_assignment_match and not exact_manifest_match:
            print(
                "resume_cuda_assignment_changed=true; "
                "reusing completed results across equivalent CUDA device indices",
                flush=True,
            )
        results = load_completed_results_snapshot(output_dir) or []
        if not results:
            try:
                results = read_results(output_dir / "summary.csv", output_dir / "rounds.csv")
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                results = []
    elif args.resume:
        archived = load_archived_results(
            output_dir,
            str(manifest["fingerprint"]),
            configs,
        )
        if archived is not None:
            results, archived_resume_dir = archived
            print(
                "resume_manifest_mismatch=true; "
                f"recovering_archived_run={archived_resume_dir}",
                flush=True,
            )
        elif previous_manifest is not None:
            print("resume_manifest_mismatch=true; starting a fresh run", flush=True)
    if not current_manifest_matches:
        archive_stale_results(output_dir, previous_manifest)
    write_run_manifest(output_dir, manifest)
    write_skipped_configs(output_dir, skipped_configs)
    if archived_resume_dir is not None and results:
        # 复制生成新的权威快照和 CSV；原归档目录继续保留作为只读备份。
        write_result_files(output_dir, results)
    completed_configs = {_resume_config_key(result.config) for result in results}
    pending_configs = [
        config
        for config in runnable_configs
        if _resume_config_key(config) not in completed_configs
    ]
    if results:
        print(f"resumed_completed_configs={len(results)}", flush=True)
    if skipped_configs:
        print(f"skipped_invalid_configs={len(skipped_configs)}", flush=True)
        for config, reason in skipped_configs:
            print(f"skipped {_format_config_key(config)} reason={reason}", flush=True)
    checkpoint_dir = output_dir / ".checkpoints" if args.resume else None
    # 若上次恰好在“结果快照已落盘、检查点尚未删除”的极短窗口中退出，
    # 已完成结果是权威状态，启动时清理对应的冗余终态检查点。
    for result in results:
        finalize_config_checkpoint(
            checkpoint_dir,
            result.config,
            manifest["fingerprint"],
        )
    confirm_matching_checkpoints(
        checkpoint_dir,
        pending_configs,
        str(manifest["fingerprint"]),
    )
    jobs = min(planned_jobs, max(1, len(pending_configs)))
    print_resource_plan(
        backend_description,
        dataset,
        configs,
        jobs=jobs,
        sm9_workers=args.sm9_workers,
        requested_jobs=args.jobs,
        cuda_devices=available_cuda_devices() if backend_description == "torch:cuda" else (),
    )
    print(f"experiment_jobs={jobs}", flush=True)
    executor_kind = parallel_executor_kind(backend_description, jobs)
    print(f"experiment_executor={executor_kind}", flush=True)
    progress = ProgressReporter(
        total=len(runnable_configs),
        completed=sum(
            _resume_config_key(result.config)
            in {_resume_config_key(config) for config in runnable_configs}
            for result in results
        ),
        enabled=not args.no_progress,
        # The configuration launcher preserves stdout from the user's terminal.
        # Use that same stream as direct CLI invocation, so both entry points
        # render one in-place progress line when a TTY is available.
        stream=sys.stdout,
        mode=args.progress_mode,
    )
    if jobs == 1:
        for config in pending_configs:
            progress.start_config(config)
            results.append(
                run_measured_experiment(
                    dataset,
                    config,
                    checkpoint_dir=checkpoint_dir,
                    run_fingerprint=manifest["fingerprint"],
                    retain_success_checkpoint=True,
                )
            )
            write_result_files(output_dir, results)
            finalize_config_checkpoint(
                checkpoint_dir,
                config,
                manifest["fingerprint"],
            )
            progress.finish_config(config)
    elif pending_configs:
        progress.start_parallel(jobs, len(pending_configs))
        if executor_kind == "thread":
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(
                        _run_config_in_thread,
                        (
                            dataset,
                            config,
                            checkpoint_dir,
                            manifest["fingerprint"],
                        ),
                    ): config
                    for config in pending_configs
                }
                _consume_parallel_futures(
                    futures,
                    results=results,
                    output_dir=output_dir,
                    checkpoint_dir=checkpoint_dir,
                    run_fingerprint=manifest["fingerprint"],
                    progress=progress,
                )
        else:
            try:
                with ProcessPoolExecutor(
                    max_workers=jobs,
                    initializer=_init_worker_dataset,
                    initargs=(dataset, checkpoint_dir, manifest["fingerprint"]),
                ) as executor:
                    futures = {
                        executor.submit(_run_config_in_worker, config): config
                        for config in pending_configs
                    }
                    _consume_parallel_futures(
                        futures,
                        results=results,
                        output_dir=output_dir,
                        checkpoint_dir=checkpoint_dir,
                        run_fingerprint=manifest["fingerprint"],
                        progress=progress,
                    )
            except PermissionError as exc:
                print(
                    f"process_pool_unavailable={exc}; falling back to thread pool",
                    flush=True,
                )
                finished_configs = {result.config for result in results}
                remaining_configs = [
                    config
                    for config in pending_configs
                    if config not in finished_configs
                ]
                with ThreadPoolExecutor(max_workers=jobs) as executor:
                    futures = {
                        executor.submit(
                            _run_config_in_thread,
                            (
                                dataset,
                                config,
                                checkpoint_dir,
                                manifest["fingerprint"],
                            ),
                        ): config
                        for config in remaining_configs
                    }
                    _consume_parallel_futures(
                        futures,
                        results=results,
                        output_dir=output_dir,
                        checkpoint_dir=checkpoint_dir,
                        run_fingerprint=manifest["fingerprint"],
                        progress=progress,
                    )
    progress.close()

    summary_path = output_dir / "summary.csv"
    rounds_path = output_dir / "rounds.csv"
    diagnostics_path = output_dir / "sm9rrs_diagnostics.csv"
    json_path = output_dir / "summary.json"
    write_result_files(output_dir, results)
    visualization_path = None
    if not args.no_visualizations:
        visualization_path = generate_visualizations(results, output_dir)

    elapsed = perf_counter() - started
    print_summary(results)
    print(f"wrote {summary_path}")
    print(f"wrote {rounds_path}")
    print(f"wrote {diagnostics_path}")
    print(f"wrote {json_path}")
    if visualization_path is not None:
        print(f"wrote {visualization_path}")
    print(f"elapsed_seconds={elapsed:.2f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "synthetic"], default="mnist")
    parser.add_argument(
        "--cifar10-clean-baseline",
        action="store_true",
        help=(
            "Run a full-data clean CIFAR-10 FedAvg baseline: dataset=cifar10, "
            "methods=fedavg, ratios=0, attack=none, partition=iid."
        ),
    )
    parser.add_argument(
        "--data-dir",
        help="Dataset directory. Defaults to data/mnist or data/cifar10 based on --dataset.",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--output-dir",
        help=(
            "Override the default output directory. Without this, SM9 runs write "
            "to outputs/<dataset> and simulated runs write to outputs/<dataset>_simulated."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "sm9rrs",
            "vert",
            "fedredefense",
            "krum",
            "ding13",
            "fedavg",
        ],
        default=[
            "sm9rrs",
            "vert",
            "fedredefense",
            "krum",
            "ding13",
            "fedavg",
        ],
    )
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--client-counts", "--num-clients-list", nargs="+", type=int)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--target-error", type=float, default=0.12)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Limit training samples. Omit to use the full real dataset.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Limit test samples. Omit to use the full real dataset.",
    )
    parser.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    parser.add_argument("--partitions", nargs="+", choices=["iid", "dirichlet"])
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=1.0,
        help="Per-round learning-rate multiplier. 1.0 keeps a constant learning rate.",
    )
    parser.add_argument(
        "--compute-backend",
        choices=["numpy", "auto", "torch"],
        default="auto",
        help=(
            "Local CNN compute backend. auto uses CUDA/MPS when available and otherwise "
            "falls back to NumPy; numpy and torch force a specific implementation."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="PyTorch device for --compute-backend torch/auto: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--jobs",
        default="auto",
        help=(
            "Number of experiment configurations to run in parallel. Use an integer "
            "or 'auto' to estimate a memory-safe limit (default: auto)."
        ),
    )
    parser.add_argument(
        "--attack",
        choices=[
            "none",
            "sign_flip",
            "gaussian",
            "alternating_minimization",
            "alternating",
        ],
        default="alternating_minimization",
        help=(
            "Model-poisoning attack. 'alternating' is a compatibility alias for "
            "the Bhagoji alternating-minimization implementation."
        ),
    )
    parser.add_argument(
        "--attack-scale",
        type=float,
        default=5.0,
        help=(
            "Post-training attack scale: sign-flip multiplier or Gaussian noise "
            "standard-deviation multiplier. It is not used by alternating minimization."
        ),
    )
    parser.add_argument(
        "--attack-boost",
        type=float,
        default=10.0,
        help="Bhagoji explicit boosting factor lambda for each adversarial step.",
    )
    parser.add_argument(
        "--attack-epochs",
        type=int,
        default=10,
        help="Number of local benign/stealth epochs used by a malicious client.",
    )
    parser.add_argument(
        "--attack-stealth-steps",
        type=int,
        default=10,
        help="Number of benign-distance steps per adversarial step (Bhagoji ls).",
    )
    parser.add_argument(
        "--attack-distance-weight",
        type=float,
        default=1e-4,
        help="Distance-constraint coefficient rho in the stealth objective.",
    )
    parser.add_argument(
        "--attack-source-label",
        type=int,
        default=5,
        help="True class of held-out auxiliary samples used by the targeted attack.",
    )
    parser.add_argument(
        "--attack-target-label",
        type=int,
        default=7,
        help="Adversarial target class for auxiliary samples.",
    )
    parser.add_argument(
        "--attack-target-count",
        type=int,
        default=1,
        help="Number of deterministic held-out auxiliary targets (Bhagoji r).",
    )
    parser.add_argument(
        "--attack-start-round",
        type=int,
        default=0,
        help="0 means K + 2, leaving K benign SVD baseline observations.",
    )
    parser.add_argument(
        "--K",
        "--k",
        "--detector-window",
        dest="detector_window",
        type=int,
        default=3,
        help=(
            "Paper parameter K: number of initial observations used to build "
            "each task-tag SVD baseline. Scoring starts at observation K + 1."
        ),
    )
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=3.0,
        help=(
            "Compatibility threshold used for both theta_adj and theta_anc "
            "when their explicit options are omitted."
        ),
    )
    parser.add_argument("--detector-subspace-dim", type=int, default=2)
    parser.add_argument("--detector-gap-threshold", type=float, default=0.1)
    parser.add_argument("--detector-adjacent-threshold", type=float)
    parser.add_argument("--detector-anchor-threshold", type=float)
    parser.add_argument("--detector-drift-memory", type=float, default=0.9)
    parser.add_argument("--detector-drift-allowance", type=float, default=1.0)
    parser.add_argument("--detector-drift-threshold", type=float, default=5.0)
    parser.add_argument(
        "--detector-decision-rule",
        choices=["any"],
        default="any",
        help="Fixed to any, implementing the v3 logical-OR formula.",
    )
    parser.add_argument("--crypto-mode", choices=["sm9", "simulated"], default="sm9")
    parser.add_argument("--dkg-threshold", type=int, default=2)
    parser.add_argument("--dkg-nodes", type=int, default=3)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluate test accuracy every N rounds; the final round is always evaluated.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help=(
            "Durable round-checkpoint interval. 0 selects a detector-state-size-aware "
            "interval; positive values force an exact interval."
        ),
    )
    parser.add_argument(
        "--sm9-workers",
        default="auto",
        help=(
            "SM9-RRS per-round packet/signature worker count. Use an integer or 'auto'. "
            "Only independent packet creation and verification are parallelized."
        ),
    )
    parser.add_argument("--suspicion-penalty-factor", type=float, default=0.5)
    parser.add_argument("--suspicion-recovery-factor", type=float, default=2.0)
    parser.add_argument(
        "--C_tol",
        "--c-tol",
        "--suspicion-remove-after",
        dest="suspicion_remove_after",
        type=int,
        default=3,
        help=(
            "Paper parameter C_tol: composite anomaly-evidence count required "
            "before threshold trace and revocation are requested."
        ),
    )
    parser.add_argument(
        "--C_max",
        "--c-max",
        "--suspicion-count-max",
        dest="suspicion_count_max",
        type=int,
        default=0,
        help="Paper C_max. 0 resolves to C_tol and avoids a redundant tuning axis.",
    )
    parser.add_argument(
        "--vert-history-window",
        type=int,
        default=10,
        help="VERT historical projected-gradient window H (paper default: 10).",
    )
    parser.add_argument(
        "--vert-projection-dim",
        type=int,
        default=128,
        help="VERT low-dimensional projector output size for MNIST.",
    )
    parser.add_argument(
        "--vert-predict-epochs",
        type=int,
        default=5,
        help="VERT predictor/coefficient Adam epochs per client and round.",
    )
    parser.add_argument(
        "--vert-predict-lr",
        type=float,
        default=1e-2,
        help="VERT predictor/coefficient Adam learning rate (paper default: 0.01).",
    )
    parser.add_argument(
        "--vert-top-k",
        type=int,
        default=0,
        help=(
            "VERT aggregation client count. 0 runs K-means with K=2 on "
            "predictor similarity scores and keeps the higher-score cluster "
            "without using malicious ratio."
        ),
    )
    parser.add_argument(
        "--vert-use-ratio-prior",
        action="store_true",
        help=(
            "Restore the legacy VERT known-ratio rule and derive k from each "
            "configuration's malicious ratio and active client count."
        ),
    )
    parser.add_argument(
        "--fedre-threshold",
        type=float,
        default=0.6,
        help="FedREDefense normalized reconstruction-error threshold.",
    )
    parser.add_argument(
        "--fedre-initial-iterations",
        type=int,
        default=800,
        help="FedREDefense synthesis iterations for a client's first observation.",
    )
    parser.add_argument(
        "--fedre-max-iterations",
        type=int,
        default=2000,
        help="FedREDefense synthesis iterations after the first observation.",
    )
    parser.add_argument(
        "--fedre-synthetic-steps",
        type=int,
        default=5,
        help="Differentiable synthetic SGD steps per reconstruction iteration.",
    )
    parser.add_argument(
        "--fedre-images-per-class",
        type=int,
        default=1,
        help="FedREDefense persistent synthetic images per class.",
    )
    parser.add_argument("--fedre-image-lr", type=float, default=0.5)
    parser.add_argument("--fedre-label-lr", type=float, default=0.2)
    parser.add_argument("--fedre-teacher-lr", type=float, default=0.1)
    parser.add_argument("--fedre-teacher-lr-lr", type=float, default=5e-6)
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress and ETA output.")
    parser.add_argument(
        "--progress-mode",
        choices=["auto", "live", "log"],
        default="auto",
        help=(
            "Progress rendering mode: auto uses a live line on a TTY, live forces "
            "in-place refresh for IDE consoles, and log writes only state changes."
        ),
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore completed configurations and round checkpoints in the output directory.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument("--visualize-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    raw_args = sys.argv[1:] if argv is None else argv
    args = apply_presets(parser.parse_args(argv), raw_args)
    if args.attack == "alternating":
        args.attack = "alternating_minimization"
    if (
        args.attack == "alternating_minimization"
        and _has_any_option(raw_args, "--attack-scale")
    ):
        parser.error(
            "--attack-scale does not control alternating minimization; "
            "use --attack-boost for lambda"
        )
    if _has_any_option(raw_args, "--z-threshold") and _has_any_option(
        raw_args,
        "--detector-adjacent-threshold",
        "--detector-anchor-threshold",
    ):
        parser.error(
            "--z-threshold cannot be combined with explicit adjacent/anchor thresholds"
        )
    max_clients = max(args.client_counts or [args.num_clients])
    args._sm9_workers_auto = (
        isinstance(args.sm9_workers, str)
        and args.sm9_workers.strip().lower() == "auto"
    )
    try:
        args.sm9_workers = resolve_sm9_workers(args.sm9_workers, max_clients)
    except ValueError as exc:
        parser.error(str(exc))
    _validate_cli_args(parser, args)
    return args


def _validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """在加载数据或启动长任务前统一拒绝明显不可执行的数值组合。"""

    client_counts = args.client_counts or [args.num_clients]
    checks = (
        (all(0.0 <= ratio < 1.0 for ratio in args.ratios), "--ratios must be in [0, 1)"),
        (all(count >= 1 for count in client_counts), "client counts must be at least 1"),
        (args.rounds >= 1, "--rounds must be at least 1"),
        (args.local_epochs >= 1, "--local-epochs must be at least 1"),
        (args.batch_size >= 1, "--batch-size must be at least 1"),
        (args.lr > 0.0, "--lr must be positive"),
        (args.lr_decay > 0.0, "--lr-decay must be positive"),
        (0.0 <= args.target_error <= 1.0, "--target-error must be in [0, 1]"),
        (args.dirichlet_alpha > 0.0, "--dirichlet-alpha must be positive"),
        (
            math.isfinite(args.attack_scale) and args.attack_scale > 0.0,
            "--attack-scale must be finite and positive",
        ),
        (
            math.isfinite(args.attack_boost) and args.attack_boost > 0.0,
            "--attack-boost must be finite and positive",
        ),
        (args.attack_epochs >= 1, "--attack-epochs must be at least 1"),
        (
            args.attack_stealth_steps >= 1,
            "--attack-stealth-steps must be at least 1",
        ),
        (
            math.isfinite(args.attack_distance_weight)
            and args.attack_distance_weight >= 0.0,
            "--attack-distance-weight must be finite and non-negative",
        ),
        (
            0 <= args.attack_source_label < 10,
            "--attack-source-label must be in [0, 10)",
        ),
        (
            0 <= args.attack_target_label < 10,
            "--attack-target-label must be in [0, 10)",
        ),
        (
            args.attack != "alternating_minimization"
            or args.attack_source_label != args.attack_target_label,
            "alternating minimization requires different source and target labels",
        ),
        (
            args.attack_target_count >= 1,
            "--attack-target-count must be at least 1",
        ),
        (
            args.attack_start_round >= 0,
            "--attack-start-round must be non-negative",
        ),
        (args.detector_window >= 2, "--K must be at least 2"),
        (
            math.isfinite(args.z_threshold) and args.z_threshold > 0.0,
            "--z-threshold must be finite and positive",
        ),
        (
            1 <= args.detector_subspace_dim < 10,
            "--detector-subspace-dim must be in [1, 9] so lambda_(q+1) exists",
        ),
        (
            math.isfinite(args.detector_gap_threshold)
            and args.detector_gap_threshold > 0.0,
            "--detector-gap-threshold must be finite and positive",
        ),
        (
            args.detector_adjacent_threshold is None
            or (
                math.isfinite(args.detector_adjacent_threshold)
                and args.detector_adjacent_threshold > 0.0
            ),
            "--detector-adjacent-threshold must be finite and positive",
        ),
        (
            args.detector_anchor_threshold is None
            or (
                math.isfinite(args.detector_anchor_threshold)
                and args.detector_anchor_threshold > 0.0
            ),
            "--detector-anchor-threshold must be finite and positive",
        ),
        (
            math.isfinite(args.detector_drift_memory)
            and 0.0 < args.detector_drift_memory <= 1.0,
            "--detector-drift-memory must be in (0, 1]",
        ),
        (
            math.isfinite(args.detector_drift_allowance)
            and args.detector_drift_allowance > 0.0,
            "--detector-drift-allowance must be finite and positive",
        ),
        (
            math.isfinite(args.detector_drift_threshold)
            and args.detector_drift_threshold > 0.0,
            "--detector-drift-threshold must be finite and positive",
        ),
        (args.suspicion_remove_after >= 1, "--C_tol must be at least 1"),
        (
            args.suspicion_count_max == 0
            or args.suspicion_count_max >= args.suspicion_remove_after,
            "--C_max must be 0 or at least --C_tol",
        ),
        (
            math.isfinite(args.suspicion_penalty_factor)
            and 0.0 < args.suspicion_penalty_factor < 1.0,
            "--suspicion-penalty-factor must be in (0, 1)",
        ),
        (
            math.isfinite(args.suspicion_recovery_factor)
            and args.suspicion_recovery_factor > 1.0,
            "--suspicion-recovery-factor must be greater than 1",
        ),
        (
            args.vert_history_window >= 2,
            "--vert-history-window must be at least 2",
        ),
        (
            args.vert_projection_dim >= 2,
            "--vert-projection-dim must be at least 2",
        ),
        (
            args.vert_predict_epochs >= 1,
            "--vert-predict-epochs must be at least 1",
        ),
        (
            math.isfinite(args.vert_predict_lr) and args.vert_predict_lr > 0.0,
            "--vert-predict-lr must be finite and positive",
        ),
        (args.vert_top_k >= 0, "--vert-top-k must be non-negative"),
        (
            not args.vert_use_ratio_prior or args.vert_top_k == 0,
            "--vert-use-ratio-prior cannot be combined with a positive "
            "--vert-top-k",
        ),
        (
            math.isfinite(args.fedre_threshold) and args.fedre_threshold > 0.0,
            "--fedre-threshold must be finite and positive",
        ),
        (
            args.fedre_initial_iterations >= 1,
            "--fedre-initial-iterations must be at least 1",
        ),
        (
            args.fedre_max_iterations >= 1,
            "--fedre-max-iterations must be at least 1",
        ),
        (
            args.fedre_synthetic_steps >= 1,
            "--fedre-synthetic-steps must be at least 1",
        ),
        (
            args.fedre_images_per_class >= 1,
            "--fedre-images-per-class must be at least 1",
        ),
        (
            math.isfinite(args.fedre_image_lr) and args.fedre_image_lr > 0.0,
            "--fedre-image-lr must be finite and positive",
        ),
        (
            math.isfinite(args.fedre_label_lr) and args.fedre_label_lr > 0.0,
            "--fedre-label-lr must be finite and positive",
        ),
        (
            math.isfinite(args.fedre_teacher_lr)
            and args.fedre_teacher_lr > 0.0,
            "--fedre-teacher-lr must be finite and positive",
        ),
        (
            math.isfinite(args.fedre_teacher_lr_lr)
            and args.fedre_teacher_lr_lr > 0.0,
            "--fedre-teacher-lr-lr must be finite and positive",
        ),
        (
            1 <= args.dkg_threshold <= args.dkg_nodes,
            "--dkg-threshold must satisfy 1 <= threshold <= --dkg-nodes",
        ),
        (args.eval_interval >= 1, "--eval-interval must be at least 1"),
        (
            args.checkpoint_interval >= 0,
            "--checkpoint-interval must be non-negative",
        ),
        (
            args.train_samples is None or args.train_samples >= 1,
            "--train-samples must be at least 1",
        ),
        (
            args.test_samples is None or args.test_samples >= 1,
            "--test-samples must be at least 1",
        ),
    )
    for valid, message in checks:
        if not valid:
            parser.error(message)


def build_experiment_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    configs: list[ExperimentConfig] = []
    partitions = args.partitions or [args.partition]
    client_counts = args.client_counts or [args.num_clients]
    for partition in partitions:
        for num_clients in client_counts:
            for ratio in args.ratios:
                for method in args.methods:
                    configs.append(
                        ExperimentConfig(
                            method=method,
                            malicious_ratio=ratio,
                            num_clients=num_clients,
                            rounds=args.rounds,
                            target_error=args.target_error,
                            local_epochs=args.local_epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            lr_decay=args.lr_decay,
                            compute_backend=args.compute_backend,
                            device=args.device,
                            partition=partition,
                            dirichlet_alpha=args.dirichlet_alpha,
                            attack=args.attack,
                            attack_scale=args.attack_scale,
                            attack_boost=args.attack_boost,
                            attack_epochs=args.attack_epochs,
                            attack_stealth_steps=args.attack_stealth_steps,
                            attack_distance_weight=args.attack_distance_weight,
                            attack_source_label=args.attack_source_label,
                            attack_target_label=args.attack_target_label,
                            attack_target_count=args.attack_target_count,
                            attack_start_round=args.attack_start_round,
                            detector_window=args.detector_window,
                            z_threshold=args.z_threshold,
                            detector_subspace_dim=args.detector_subspace_dim,
                            detector_gap_threshold=args.detector_gap_threshold,
                            detector_adjacent_threshold=(
                                args.z_threshold
                                if args.detector_adjacent_threshold is None
                                else args.detector_adjacent_threshold
                            ),
                            detector_anchor_threshold=(
                                args.z_threshold
                                if args.detector_anchor_threshold is None
                                else args.detector_anchor_threshold
                            ),
                            detector_drift_memory=args.detector_drift_memory,
                            detector_drift_allowance=args.detector_drift_allowance,
                            detector_drift_threshold=args.detector_drift_threshold,
                            detector_decision_rule=args.detector_decision_rule,
                            crypto_mode=args.crypto_mode,
                            dkg_threshold=args.dkg_threshold,
                            dkg_nodes=args.dkg_nodes,
                            early_stop=not args.no_early_stop,
                            eval_interval=args.eval_interval,
                            checkpoint_interval=args.checkpoint_interval,
                            sm9_workers=args.sm9_workers,
                            suspicion_penalty_factor=args.suspicion_penalty_factor,
                            suspicion_recovery_factor=args.suspicion_recovery_factor,
                            suspicion_remove_after=args.suspicion_remove_after,
                            suspicion_count_max=(
                                args.suspicion_remove_after
                                if args.suspicion_count_max == 0
                                else args.suspicion_count_max
                            ),
                            vert_history_window=args.vert_history_window,
                            vert_projection_dim=args.vert_projection_dim,
                            vert_predict_epochs=args.vert_predict_epochs,
                            vert_predict_lr=args.vert_predict_lr,
                            vert_top_k=args.vert_top_k,
                            vert_use_ratio_prior=args.vert_use_ratio_prior,
                            fedre_threshold=args.fedre_threshold,
                            fedre_initial_iterations=args.fedre_initial_iterations,
                            fedre_max_iterations=args.fedre_max_iterations,
                            fedre_synthetic_steps=args.fedre_synthetic_steps,
                            fedre_images_per_class=args.fedre_images_per_class,
                            fedre_image_lr=args.fedre_image_lr,
                            fedre_label_lr=args.fedre_label_lr,
                            fedre_teacher_lr=args.fedre_teacher_lr,
                            fedre_teacher_lr_lr=args.fedre_teacher_lr_lr,
                            seed=args.seed,
                        )
                    )
    return configs


def apply_presets(args: argparse.Namespace, raw_args: list[str]) -> argparse.Namespace:
    _apply_dataset_training_preset(args, raw_args)
    if not args.cifar10_clean_baseline:
        return args

    args.dataset = "cifar10"
    args.methods = ["fedavg"]
    args.ratios = [0.0]
    args.attack = "none"
    args.partition = "iid"
    args.partitions = ["iid"]
    _apply_dataset_training_preset(args, raw_args)
    if not _has_any_option(raw_args, "--train-samples"):
        args.train_samples = None
    if not _has_any_option(raw_args, "--test-samples"):
        args.test_samples = None
    if not _has_any_option(raw_args, "--num-clients", "--client-counts", "--num-clients-list"):
        args.num_clients = 20
        args.client_counts = [20]
    return args


def resolve_sm9_workers(
    value: str | int,
    num_clients: int,
    *,
    parallel_jobs: int = 1,
) -> int:
    """Resolve per-experiment SM9 threads without oversubscribing auto mode.

    A configuration can issue independent native SM9 operations in parallel, but
    an experiment grid can also run several configurations at once.  In auto
    mode the CPU budget is divided between those configurations; explicit user
    values remain authoritative.
    """

    if isinstance(value, int):
        workers = value
    else:
        text = value.strip().lower()
        if text == "auto":
            cpu_share = max(1, available_cpu_count() // max(1, parallel_jobs))
            # 原生配对运算会释放 GIL；超过每个并发配置的 CPU 预算只会增加
            # 线程调度和每轮临时内存，而不会提高真实吞吐。
            workers = min(8, cpu_share, max(1, num_clients))
        else:
            workers = int(text)
    if workers < 1:
        raise ValueError("--sm9-workers must be at least 1")
    return workers


def resolve_parallel_jobs(
    value: str | int,
    dataset,
    configs: list[ExperimentConfig],
    args: argparse.Namespace,
) -> int:
    if not configs:
        return 1
    if isinstance(value, int):
        requested = value
    else:
        text = value.strip().lower()
        if text != "auto":
            requested = int(text)
        else:
            requested = _auto_parallel_jobs(dataset, configs, args)
    if requested < 1:
        raise ValueError("--jobs must be at least 1")
    return min(requested, len(configs))


def _auto_parallel_jobs(dataset, configs: list[ExperimentConfig], args: argparse.Namespace) -> int:
    total_mb = _physical_memory_mb()
    per_worker_mb = _estimated_parallel_worker_memory_mb(dataset, configs)
    memory_limited = max(1, int((total_mb * 0.75) // per_worker_mb))
    cpu_limited = available_cpu_count()
    jobs = min(memory_limited, cpu_limited, len(configs))
    backend = describe_compute_backend(args.compute_backend, args.device)
    if backend == "torch:cuda":
        cuda_worker_mb = _estimated_cuda_worker_memory_mb(dataset, configs)
        cuda_limited = _cuda_memory_parallel_limit(cuda_worker_mb)
        if cuda_limited is not None:
            if cuda_limited < 1:
                raise RuntimeError(_cuda_capacity_error(cuda_worker_mb))
            jobs = min(jobs, cuda_limited)
    elif backend == "torch:mps":
        # Apple exposes one shared MPS device backed by unified memory. Running
        # several complete experiments in threads makes their resident models,
        # client updates and autograd buffers contend for the same device and
        # can be substantially slower than serial execution. Explicit --jobs
        # remains authoritative; only auto mode is conservatively capped.
        jobs = min(jobs, 1)
    return max(1, jobs)


def _estimated_parallel_worker_memory_mb(dataset, configs: list[ExperimentConfig]) -> float:
    """Estimate peak host memory needed by one experiment config."""

    if not configs:
        return 1024.0
    max_clients = max(config.num_clients for config in configs)
    max_params = _parameter_size_for_dataset(dataset)
    dataset_mb = _dataset_memory_mb(dataset)
    update_mb = max_clients * max_params * 4 / (1024 * 1024)
    detector_state_mb = _detector_state_memory_mb(dataset, configs)
    # Besides the resident image tensors, a configuration retains client model
    # updates and temporary autograd/aggregation buffers.  Keep the existing
    # conservative host estimate and apply a separate 25% CUDA safety margin
    # when deciding whether a physical device can accept the configuration.
    return max(
        1024.0,
        dataset_mb * 0.35 + update_mb * 2.5 + detector_state_mb * 1.1 + 768.0,
    )


def _estimated_cuda_worker_memory_mb(dataset, configs: list[ExperimentConfig]) -> float:
    """Estimate accelerator memory without charging CPU-only detector history."""

    if not configs:
        return 1024.0
    max_clients = max(config.num_clients for config in configs)
    max_params = _parameter_size_for_dataset(dataset)
    dataset_mb = _dataset_memory_mb(dataset)
    update_mb = max_clients * max_params * 4 / (1024 * 1024)
    return max(1024.0, dataset_mb * 0.35 + update_mb * 2.5 + 768.0)


def _detector_state_memory_mb(dataset, configs: list[ExperimentConfig]) -> float:
    """Conservative host footprint of trusted/last-observed float32 thin bases."""

    sm9_configs = [config for config in configs if config.method == "sm9rrs"]
    if not sm9_configs:
        return 0.0
    max_params = _parameter_size_for_dataset(dataset)
    classes = max(1, int(dataset.num_classes))
    rows = math.ceil(max_params / classes)
    return max(
        config.num_clients
        * (config.detector_window + 1)
        * rows
        * config.detector_subspace_dim
        * 4
        / (1024 * 1024)
        for config in sm9_configs
    )


def _effective_checkpoint_interval(dataset, config: ExperimentConfig) -> int:
    """Resolve 0=auto while bounding write amplification for large v3 state."""

    if config.checkpoint_interval > 0 or config.method != "sm9rrs":
        return max(1, config.checkpoint_interval)
    state_mb = _detector_state_memory_mb(dataset, [config])
    if state_mb > 2048.0:
        return 25
    if state_mb > 1024.0:
        return 10
    if state_mb > 256.0:
        return 5
    return 1


def available_cpu_count() -> int:
    """Return the CPU capacity available to this process, including cgroup limits."""

    count = max(1, os.cpu_count() or 1)
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            count = min(count, max(1, len(affinity(0))))
        except OSError:
            pass
    quota_count = _cgroup_cpu_quota_count()
    if quota_count is not None:
        count = min(count, quota_count)
    return max(1, count)


def _cgroup_cpu_quota_count() -> int | None:
    """Read common Linux cgroup CPU quotas when the program runs in a container."""

    try:
        text = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").strip()
        quota, period = text.split()[:2]
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError, IndexError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
        if quota > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


def available_cuda_devices() -> tuple[str, ...]:
    """List CUDA devices visible to this process without making CUDA mandatory."""

    try:
        from .torch_backend import _torch_module

        torch = _torch_module()
        if not torch.cuda.is_available():
            return ()
        return tuple(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    except (RuntimeError, ImportError):
        return ()


def cuda_device_memory_info() -> tuple[tuple[str, float, float], ...] | None:
    """Return ``(device, free_mb, total_mb)`` for every visible CUDA device.

    ``None`` means the installed PyTorch/CUDA runtime cannot inspect memory;
    an empty tuple means CUDA is available but exposes no devices.
    """

    try:
        from .torch_backend import _torch_module

        torch = _torch_module()
        if not torch.cuda.is_available():
            return ()
        info = []
        for index in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            info.append(
                (
                    f"cuda:{index}",
                    float(free_bytes) / (1024 * 1024),
                    float(total_bytes) / (1024 * 1024),
                )
            )
        return tuple(info)
    except (AttributeError, RuntimeError, ImportError):
        return None


def cuda_devices_with_capacity(per_worker_mb: float) -> tuple[str, ...]:
    """Return visible devices with enough current free memory for one job."""

    visible = available_cuda_devices()
    if not visible:
        return ()
    info = cuda_device_memory_info()
    if info is None:
        # Older PyTorch builds may not expose mem_get_info.  Preserve the prior
        # behavior rather than disabling CUDA solely because inspection failed.
        return visible
    by_device = {device: free_mb for device, free_mb, _total_mb in info}
    return tuple(
        device
        for device in visible
        if by_device.get(device, 0.0) * CUDA_MEMORY_SAFETY_FRACTION
        >= per_worker_mb
    )


def _cuda_capacity_error(per_worker_mb: float) -> str:
    info = cuda_device_memory_info()
    if info:
        free_text = ",".join(
            f"{device}:{free_mb:.0f}/{total_mb:.0f}MiB-free"
            for device, free_mb, total_mb in info
        )
    else:
        free_text = "unavailable"
    required_free_mb = per_worker_mb / CUDA_MEMORY_SAFETY_FRACTION
    return (
        "no visible CUDA device has enough free memory for one experiment; "
        f"estimated_required_free_mb={required_free_mb:.0f} "
        f"cuda_memory={free_text}. Stop unrelated GPU processes or restrict "
        "CUDA_VISIBLE_DEVICES to idle GPUs, then rerun with resume enabled."
    )


def _cuda_memory_parallel_limit(per_worker_mb: float) -> int | None:
    """Estimate a safe grid width from currently free CUDA memory.

    The estimate deliberately follows the host-memory calculation's update
    retention model.  Returning ``None`` means the runtime cannot inspect CUDA
    memory, so CPU and host-memory limits remain the safe fallback.
    """

    info = cuda_device_memory_info()
    if info is None:
        return None
    # Auto mode deliberately runs at most one independent experiment per GPU.
    # This avoids silently counting a nearly full GPU as one slot and avoids
    # cross-job contention that would also bias paper-facing runtime results.
    return sum(
        free_mb * CUDA_MEMORY_SAFETY_FRACTION >= per_worker_mb
        for _device, free_mb, _total_mb in info
    )


def assign_auto_cuda_devices(
    configs: list[ExperimentConfig],
    backend_description: str,
    requested_device: str,
    *,
    cuda_devices: tuple[str, ...] | None = None,
) -> list[ExperimentConfig]:
    """Spread auto-selected CUDA configurations over usable GPUs.

    Explicit ``--device cuda`` remains pinned to PyTorch's default device.
    With one GPU the assignments intentionally share that device, which lets
    independent small CNN jobs keep it busy while CPU work from another job is
    in flight.
    """

    if backend_description != "torch:cuda" or requested_device.strip().lower() != "auto":
        return configs
    devices = available_cuda_devices() if cuda_devices is None else cuda_devices
    if not devices:
        return configs
    return [
        replace(config, device=devices[index % len(devices)])
        for index, config in enumerate(configs)
    ]


def print_resource_plan(
    backend_description: str,
    dataset,
    configs: list[ExperimentConfig],
    *,
    jobs: int,
    sm9_workers: int,
    requested_jobs: str | int,
    cuda_devices: tuple[str, ...],
) -> None:
    """Emit one compact, machine-readable explanation of effective auto choices."""

    assigned_devices = sorted({config.device for config in configs})
    if assigned_devices == ["auto"] and backend_description.startswith("torch:"):
        assigned_devices = [backend_description.split(":", 1)[1]]
    fields = [
        f"backend={backend_description}",
        f"cpu_slots={available_cpu_count()}",
        f"host_memory_mb={_physical_memory_mb():.0f}",
        f"dataset_memory_mb={_dataset_memory_mb(dataset):.0f}",
        f"requested_jobs={requested_jobs}",
        f"jobs={jobs}",
        f"sm9_workers_per_experiment={sm9_workers}",
        f"estimated_host_worker_mb={_estimated_parallel_worker_memory_mb(dataset, configs):.0f}",
        "checkpoint_intervals="
        + ",".join(
            str(value)
            for value in sorted(
                {_effective_checkpoint_interval(dataset, config) for config in configs}
            )
        ),
        "assigned_devices=" + ",".join(assigned_devices),
    ]
    if cuda_devices:
        fields.append("visible_cuda_devices=" + ",".join(cuda_devices))
        per_worker_mb = _estimated_cuda_worker_memory_mb(dataset, configs)
        usable_devices = cuda_devices_with_capacity(per_worker_mb)
        fields.append("usable_cuda_devices=" + ",".join(usable_devices))
        fields.append(f"estimated_cuda_worker_mb={per_worker_mb:.0f}")
        info = cuda_device_memory_info()
        if info:
            fields.append(
                "cuda_free_mb="
                + ",".join(f"{device}={free_mb:.0f}" for device, free_mb, _ in info)
            )
    print("resource_plan=" + " ".join(fields), flush=True)


def parallel_executor_kind(backend_description: str, jobs: int) -> str:
    """Choose process isolation for CPU and threads for one accelerator device."""

    if jobs <= 1:
        return "serial"
    if backend_description in {"torch:cuda", "torch:mps"}:
        return "thread"
    return "process"


def _physical_memory_mb() -> float:
    physical_mb: float | None = None
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            physical_mb = status.ullTotalPhys / (1024 * 1024)
    if physical_mb is None and hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            physical_mb = float(pages * page_size) / (1024 * 1024)
        except (ValueError, OSError, AttributeError):
            pass
    if physical_mb is None:
        physical_mb = 16 * 1024.0
    cgroup_limit_mb = _cgroup_memory_limit_mb()
    return (
        physical_mb
        if cgroup_limit_mb is None
        else min(physical_mb, cgroup_limit_mb)
    )


def _cgroup_memory_limit_mb(
    paths: tuple[Path, ...] | None = None,
) -> float | None:
    """Return a finite Linux cgroup v2/v1 memory limit when one is active."""

    candidates = paths or (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    limits: list[int] = []
    for path in candidates:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw or raw.lower() == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        # cgroup v1 uses a near-2^63 sentinel for an unlimited hierarchy.
        if 0 < value < (1 << 60):
            limits.append(value)
    if not limits:
        return None
    return min(limits) / (1024 * 1024)


def _dataset_memory_mb(dataset) -> float:
    arrays = (dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test)
    return sum(float(getattr(array, "nbytes", 0)) for array in arrays) / (1024 * 1024)


def _parameter_size_for_dataset(dataset) -> int:
    from .model import model_spec_for_dataset

    return model_spec_for_dataset(dataset).parameter_size


def _init_worker_dataset(dataset, checkpoint_dir=None, run_fingerprint=None) -> None:
    global _WORKER_DATASET, _WORKER_CHECKPOINT_DIR, _WORKER_RUN_FINGERPRINT
    _WORKER_DATASET = dataset
    _WORKER_CHECKPOINT_DIR = checkpoint_dir
    _WORKER_RUN_FINGERPRINT = run_fingerprint


def _run_config_in_worker(config: ExperimentConfig) -> ExperimentResult:
    if _WORKER_DATASET is None:
        raise RuntimeError("worker dataset was not initialized")
    return run_measured_experiment(
        _WORKER_DATASET,
        config,
        checkpoint_dir=_WORKER_CHECKPOINT_DIR,
        run_fingerprint=_WORKER_RUN_FINGERPRINT,
        retain_success_checkpoint=True,
    )


def _run_config_in_thread(task) -> ExperimentResult:
    dataset, config, checkpoint_dir, run_fingerprint = task
    return run_measured_experiment(
        dataset,
        config,
        checkpoint_dir=checkpoint_dir,
        run_fingerprint=run_fingerprint,
        retain_success_checkpoint=True,
    )


def _consume_parallel_futures(
    futures,
    *,
    results: list[ExperimentResult],
    output_dir: Path,
    checkpoint_dir: Path | None,
    run_fingerprint: str,
    progress,
) -> None:
    """Commit completed configurations serially in the parent process."""

    for future in as_completed(futures):
        config = futures[future]
        result = future.result()
        results.append(result)
        write_result_files(output_dir, results)
        finalize_config_checkpoint(
            checkpoint_dir,
            config,
            run_fingerprint,
        )
        progress.finish_config(config)


class ProgressReporter:
    def __init__(
        self,
        *,
        total: int,
        completed: int = 0,
        enabled: bool = True,
        stream=None,
        refresh_interval: float = 1.0,
        mode: str = "auto",
    ) -> None:
        self.total = max(0, total)
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.completed = min(max(0, completed), self.total)
        self.started = perf_counter()
        self.completed_this_session = 0
        self.eta_deadline: float | None = None
        self.current = "starting"
        self.refresh_interval = max(0.01, float(refresh_interval))
        self.last_message_length = 0
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.mode = mode
        if mode not in {"auto", "live", "log"}:
            raise ValueError("progress mode must be one of: auto, live, log")
        self.live = mode == "live" or (mode == "auto" and self.is_tty)
        self.closed = False
        self._lock = Lock()
        self._stop_event = Event()
        self._refresh_thread: Thread | None = None
        if self.enabled and self.total:
            self._write(self._progress_message())
            if self.live:
                self._refresh_thread = Thread(
                    target=self._refresh_loop,
                    name="experiment-progress-refresh",
                    daemon=True,
                )
                self._refresh_thread.start()

    def start_config(self, config: ExperimentConfig) -> None:
        if not self.enabled:
            print(_format_running_config(config))
            return
        with self._lock:
            self.current = f"running {_format_config_key(config)}"
            self._write(self._progress_message())

    def start_parallel(self, workers: int, pending: int) -> None:
        if not self.enabled:
            print(f"running {pending} configurations with {workers} workers")
            return
        with self._lock:
            self.current = (
                f"running {pending} configurations with {workers} workers"
            )
            self._write(self._progress_message())

    def finish_config(self, config: ExperimentConfig) -> None:
        if not self.enabled:
            self.completed += 1
            print(f"finished {_format_config_key(config)}")
            return
        with self._lock:
            self.completed += 1
            self.completed_this_session += 1
            now = perf_counter()
            remaining = max(0, self.total - self.completed)
            elapsed = max(0.0, now - self.started)
            seconds_per_config = elapsed / self.completed_this_session
            self.eta_deadline = now + seconds_per_config * remaining
            self.current = f"finished {_format_config_key(config)}"
            self._write(self._progress_message(now=now))

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
        self._stop_event.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join(timeout=max(1.0, self.refresh_interval * 2.0))
        if self.enabled and self.total:
            with self._lock:
                self.current = "complete"
                self._write(self._progress_message(), final=True)

    def _refresh_loop(self) -> None:
        while not self._stop_event.wait(self.refresh_interval):
            with self._lock:
                if self.closed:
                    return
                self._write(self._progress_message())

    def _progress_message(self, *, now: float | None = None) -> str:
        current_time = perf_counter() if now is None else now
        elapsed = current_time - self.started
        ratio = 1.0 if self.total == 0 else min(1.0, self.completed / self.total)
        filled = int(round(PROGRESS_BAR_WIDTH * ratio))
        bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
        percent = ratio * 100.0
        if self.completed >= self.total:
            eta_text = "0s"
        elif self.eta_deadline is not None:
            eta_text = _format_duration(self.eta_deadline - current_time)
        else:
            eta_text = "estimating"
        return (
            f"[{bar}] {self.completed}/{self.total} {percent:5.1f}% "
            f"elapsed={_format_duration(elapsed)} eta={eta_text} {self.current}"
        )

    def _write(self, message: str, *, final: bool = False) -> None:
        if self.live:
            padding = " " * max(0, self.last_message_length - len(message))
            self.stream.write(f"\r{message}{padding}")
            if final:
                self.stream.write("\n")
            self.stream.flush()
            self.last_message_length = len(message)
        else:
            self.stream.write(f"{message}\n")
            self.stream.flush()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _format_running_config(config: ExperimentConfig) -> str:
    return f"running {_format_config_key(config)}"


def _format_config_key(config: ExperimentConfig) -> str:
    return (
        f"partition={config.partition} clients={config.num_clients} "
        f"method={config.method} ratio={config.malicious_ratio:.2f}"
    )


def _apply_dataset_training_preset(args: argparse.Namespace, raw_args: list[str]) -> None:
    preset = DATASET_TRAINING_PRESETS.get(args.dataset)
    if preset is None:
        return
    option_names = {
        "rounds": ("--rounds",),
        "local_epochs": ("--local-epochs",),
        "batch_size": ("--batch-size",),
        "lr": ("--lr",),
        "lr_decay": ("--lr-decay",),
    }
    for field, names in option_names.items():
        if not _has_any_option(raw_args, *names):
            setattr(args, field, preset[field])


def _has_any_option(raw_args: list[str], *names: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in raw_args for name in names)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Return the user supplied output directory or the mode-specific default."""

    if args.output_dir:
        return Path(args.output_dir)
    if getattr(args, "cifar10_clean_baseline", False):
        return DEFAULT_OUTPUT_ROOT / "cifar10_clean_baseline"
    return default_output_dir(args.dataset, args.crypto_mode)


def default_output_dir(dataset: str, crypto_mode: str) -> Path:
    """Choose a default output directory that separates simulated runs."""

    directory_name = dataset
    if crypto_mode == "simulated":
        directory_name = f"{directory_name}_simulated"
    return DEFAULT_OUTPUT_ROOT / directory_name


def build_run_manifest(
    args: argparse.Namespace,
    dataset,
    configs: list[ExperimentConfig],
) -> dict[str, object]:
    """记录会影响实验结果的数据与配置，防止错误复用旧检查点。"""

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "dataset": {
            "name": dataset.name,
            "train_samples": int(len(dataset.y_train)),
            "test_samples": int(len(dataset.y_test)),
            "input_shape": list(dataset.input_shape),
            "num_classes": int(dataset.num_classes),
            "data_dir": str(Path(args.data_dir).expanduser().resolve()) if args.data_dir else None,
            "seed": int(args.seed),
        },
        "configs": [asdict(config) for config in configs],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _normalized_cuda_device(value: object) -> object:
    """Treat indexed CUDA placement as runtime scheduling, not an algorithm change."""

    if isinstance(value, str) and value.startswith("cuda:"):
        suffix = value.split(":", 1)[1]
        if suffix.isdigit():
            return "cuda:auto-index"
    return value


def _normalized_manifest_payload(manifest: dict[str, object]) -> dict[str, object]:
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "fingerprint"
    }
    configs = payload.get("configs")
    if isinstance(configs, list):
        normalized_configs = []
        for config in configs:
            if not isinstance(config, dict):
                normalized_configs.append(config)
                continue
            normalized = dict(config)
            normalized["device"] = _normalized_cuda_device(normalized.get("device"))
            normalized_configs.append(normalized)
        payload["configs"] = normalized_configs
    return payload


def _manifests_differ_only_by_indexed_cuda_device(
    previous: dict[str, object],
    current: dict[str, object],
) -> bool:
    """Allow completed configs to survive changes in auto-selected GPU index."""

    if previous.get("fingerprint") == current.get("fingerprint"):
        return False
    return _normalized_manifest_payload(previous) == _normalized_manifest_payload(current)


def _resume_config_key(config: ExperimentConfig) -> str:
    payload = asdict(config)
    payload["device"] = _normalized_cuda_device(payload.get("device"))
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def read_run_manifest(output_dir: Path) -> dict[str, object] | None:
    path = output_dir / "run_manifest.json"
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_run_manifest(output_dir: Path, manifest: dict[str, object]) -> None:
    temporary = output_dir / ".run_manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    temporary.replace(output_dir / "run_manifest.json")


def write_skipped_configs(
    output_dir: Path,
    skipped_configs: list[tuple[ExperimentConfig, str]],
) -> None:
    """记录算法上无定义的配置，缺失结果点因此可以被审计而非静默消失。"""

    rows = []
    for config, reason in skipped_configs:
        malicious_count = malicious_client_count(
            config.num_clients,
            config.malicious_ratio,
        )
        rows.append(
            {
                "partition": config.partition,
                "num_clients": config.num_clients,
                "method": config.method,
                "malicious_ratio": config.malicious_ratio,
                "malicious_clients": malicious_count,
                "neighbor_count": config.num_clients - malicious_count - 2,
                "reason": reason,
            }
        )
    temporary = output_dir / ".skipped_configs.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    temporary.replace(output_dir / "skipped_configs.json")


def load_archived_results(
    output_dir: Path,
    run_fingerprint: str,
    configs: list[ExperimentConfig],
) -> tuple[list[ExperimentResult], Path] | None:
    """按清单指纹找回曾因切换参数而归档的同一长任务结果。"""

    stale_root = output_dir / ".stale"
    if not stale_root.exists():
        return None
    prefix = run_fingerprint[:12]
    candidates = [
        path
        for path in stale_root.iterdir()
        if path.is_dir() and (path.name == prefix or path.name.startswith(f"{prefix}-"))
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    allowed_configs = set(configs)
    for candidate in candidates:
        results = load_completed_results_snapshot(candidate) or []
        if not results:
            try:
                results = read_results(
                    candidate / "summary.csv",
                    candidate / "rounds.csv",
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError):
                continue
        # 目录名来自指纹，但仍再次核对每个有效配置，防止误恢复手工放入的文件。
        if results and all(result.config in allowed_configs for result in results):
            return results, candidate
    return None


def archive_stale_results(
    output_dir: Path,
    previous_manifest: dict[str, object] | None,
) -> None:
    """新运行参数不一致时保留旧结果，但不让它们冒充当前进度。"""

    existing = [
        path
        for path in (
            output_dir / "summary.csv",
            output_dir / "rounds.csv",
            output_dir / "sm9rrs_diagnostics.csv",
            output_dir / "summary.json",
            output_dir / "skipped_configs.json",
            output_dir / "last_failure.json",
            output_dir / COMPLETED_RESULTS_SNAPSHOT,
        )
        if path.exists()
    ]
    if not existing:
        return
    fingerprint = str((previous_manifest or {}).get("fingerprint") or "legacy")[:12]
    base = output_dir / ".stale" / fingerprint
    destination = base
    suffix = 1
    while destination.exists():
        destination = base.with_name(f"{base.name}-{suffix}")
        suffix += 1
    destination.mkdir(parents=True)
    for path in existing:
        path.replace(destination / path.name)


def _checkpoint_path(checkpoint_dir: Path, config: ExperimentConfig) -> Path:
    canonical = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return checkpoint_dir / f"{key}.pickle"


def _load_round_checkpoint(
    path: Path,
    config: ExperimentConfig,
    run_fingerprint: str | None,
) -> tuple[dict[str, object] | None, float, float]:
    if not path.exists() or run_fingerprint is None:
        return None, 0.0, 0.0
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except (
        OSError,
        EOFError,
        pickle.PickleError,
        AttributeError,
        ValueError,
        ImportError,
        TypeError,
    ):
        return None, 0.0, 0.0
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or payload.get("run_fingerprint") != run_fingerprint
        or payload.get("config") != asdict(config)
        or not isinstance(payload.get("state"), dict)
    ):
        return None, 0.0, 0.0
    return (
        payload["state"],
        float(payload.get("runtime_seconds", 0.0)),
        float(payload.get("peak_memory_mb", 0.0)),
    )


def confirm_matching_checkpoints(
    checkpoint_dir: Path | None,
    pending_configs: list[ExperimentConfig],
    run_fingerprint: str,
    *,
    input_func=None,
    interactive: bool | None = None,
) -> dict[str, int]:
    """对完全匹配的断点逐一询问续跑或从头开始。"""

    counts = {"found": 0, "resumed": 0, "restarted": 0}
    if checkpoint_dir is None:
        return counts
    checkpoint_dir = Path(checkpoint_dir)
    if interactive is None:
        interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())
    reader = input if input_func is None else input_func

    for config in pending_configs:
        checkpoint_path = _checkpoint_path(checkpoint_dir, config)
        state, _, _ = _load_round_checkpoint(
            checkpoint_path,
            config,
            run_fingerprint,
        )
        if state is None:
            continue
        counts["found"] += 1
        completed_round = int(state.get("completed_round", 0))
        terminal = isinstance(state.get("terminal_result"), ExperimentResult)
        print("发现存在相同配置的断点：", flush=True)
        print(f"  {_format_config_key(config)}", flush=True)
        print(
            f"  已完成轮次={completed_round}/{config.rounds}"
            + ("（配置已完成，等待写入总结果）" if terminal else ""),
            flush=True,
        )

        if not interactive:
            print("  当前为非交互式终端，默认选择 Y，沿断点继续运行。", flush=True)
            counts["resumed"] += 1
            continue

        while True:
            try:
                answer = str(reader("发现存在相同配置的断点，是否沿着断点运行（Y/N）？ "))
            except EOFError:
                answer = "Y"
                print("未读取到输入，默认选择 Y。", flush=True)
            normalized = answer.strip().lower()
            if normalized in {"y", "yes"}:
                counts["resumed"] += 1
                print("已选择 Y：将从该断点继续运行。", flush=True)
                break
            if normalized in {"n", "no"}:
                destination = _archive_rejected_checkpoint(
                    checkpoint_path,
                    completed_round=completed_round,
                )
                counts["restarted"] += 1
                print(
                    "已选择 N：本次将从第 0 轮开始；原断点已备份至 "
                    f"{destination}",
                    flush=True,
                )
                break
            print("请输入 Y 或 N。", flush=True)
    return counts


def _archive_rejected_checkpoint(
    checkpoint_path: Path,
    *,
    completed_round: int,
) -> Path:
    """用户选择从头运行时备份旧断点，而不是不可逆删除。"""

    destination_dir = checkpoint_path.parent / "discarded"
    destination_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = destination_dir / (
        f"{checkpoint_path.stem}.round-{completed_round}.{timestamp}.pickle"
    )
    destination = base
    suffix = 1
    while destination.exists():
        destination = base.with_name(f"{base.stem}-{suffix}{base.suffix}")
        suffix += 1
    checkpoint_path.replace(destination)
    return destination


def _write_round_checkpoint(
    path: Path,
    config: ExperimentConfig,
    run_fingerprint: str,
    state: dict[str, object],
    *,
    runtime_seconds: float,
    peak_memory_mb: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A stable per-target temporary name is safe because one configuration has
    # one writer.  It also lets a restart truncate/reuse a multi-GiB temporary
    # file left by an interrupted pickle instead of leaking one file per PID.
    temporary = path.with_name(f".{path.name}.tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "config": asdict(config),
        "runtime_seconds": runtime_seconds,
        "peak_memory_mb": peak_memory_mb,
        "state": state,
    }
    with _CHECKPOINT_WRITE_LOCK:
        reclaimable_bytes = (
            temporary.stat().st_size if temporary.exists() else 0
        )
        free_bytes = shutil.disk_usage(path.parent).free + reclaimable_bytes
        required_bytes = _estimated_checkpoint_write_bytes(state)
        if free_bytes < required_bytes:
            raise OSError(
                errno.ENOSPC,
                "insufficient free space for atomic checkpoint write: "
                f"required~{required_bytes / (1024**3):.2f} GiB, "
                f"free={free_bytes / (1024**3):.2f} GiB",
                str(path),
            )
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        _durable_replace(temporary, path)


def _estimated_checkpoint_write_bytes(state: dict[str, object]) -> int:
    """Conservative temporary-file requirement before an atomic replacement."""

    estimate = _unique_numpy_array_bytes(state) + 256 * 1024**2
    # Pickle metadata, diagnostics and crypto scalar objects retain a fixed plus
    # proportional margin.  Traversing the full object graph also counts model
    # updates held in pending audit evidence, not only detector U_q factors.
    return max(estimate, int(estimate * 1.1))


def _unique_numpy_array_bytes(root: object) -> int:
    """Count unique ndarray storage reachable from project checkpoint state.

    Only built-in containers and objects implemented by this package are
    traversed. Walking arbitrary third-party ``__dict__`` mappings can escape
    into complete module graphs; for PyTorch it also touches deprecated lazy
    aliases such as ``torch.distributed.reduce_op`` and emits warnings.
    """

    total = 0
    seen: set[int] = set()
    stack = [root]
    scalar_types = {str, bytes, bytearray, int, float, bool, complex, type(None)}
    container_types = {list, tuple, set, frozenset, deque}
    while stack:
        value = stack.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        value_type = type(value)
        value_module = getattr(value_type, "__module__", "")
        if value_type is np.ndarray or (
            value_module.startswith("numpy") and isinstance(value, np.ndarray)
        ):
            total += int(value.nbytes)
            continue
        if value_type in scalar_types:
            continue
        if value_type is dict:
            stack.extend(value.values())
            continue
        if value_type in container_types:
            stack.extend(value)
            continue
        if not value_module.startswith("sm9rrsfl."):
            continue
        if is_dataclass(value) and not isinstance(value, type):
            stack.extend(getattr(value, item.name) for item in fields(value))
            continue
        try:
            attributes = vars(value)
        except TypeError:
            continue
        if type(attributes) is dict:
            stack.extend(attributes.values())
    return total


def run_measured_experiment(
    dataset,
    config: ExperimentConfig,
    *,
    checkpoint_dir: Path | None = None,
    run_fingerprint: str | None = None,
    retain_success_checkpoint: bool = False,
) -> ExperimentResult:
    checkpoint_path = (
        _checkpoint_path(Path(checkpoint_dir), config) if checkpoint_dir is not None else None
    )
    if checkpoint_path is not None:
        resume_state, previous_runtime, previous_peak = _load_round_checkpoint(
            checkpoint_path,
            config,
            run_fingerprint,
        )
    else:
        resume_state, previous_runtime, previous_peak = None, 0.0, 0.0
    if resume_state is not None and isinstance(
        resume_state.get("terminal_result"),
        ExperimentResult,
    ):
        # 子任务已经完成，只是父进程尚未来得及写入总结果快照。无需重算，
        # 直接重放终态结果完成两阶段提交。
        terminal_result = resume_state["terminal_result"]
        if not retain_success_checkpoint:
            finalize_config_checkpoint(
                checkpoint_dir,
                config,
                run_fingerprint,
            )
        return terminal_result
    latest_completed_round = int(resume_state.get("completed_round", 0)) if resume_state else 0
    latest_state = resume_state
    last_checkpointed_round = latest_completed_round if resume_state is not None else -1
    checkpoint_io_seconds = 0.0
    checkpoint_interval = _effective_checkpoint_interval(dataset, config)
    if resume_state is not None:
        print(
            f"resuming {_format_config_key(config)} "
            f"from_completed_round={latest_completed_round}",
            flush=True,
        )
    started = perf_counter()
    callback = None
    if checkpoint_path is not None and run_fingerprint is not None:
        def callback(state) -> None:
            nonlocal latest_completed_round, latest_state
            nonlocal last_checkpointed_round, checkpoint_io_seconds
            completed_round = int(state["completed_round"])
            latest_state = state
            if completed_round == last_checkpointed_round:
                return
            if completed_round != 0 and completed_round % checkpoint_interval != 0:
                return
            checkpoint_started = perf_counter()
            _write_round_checkpoint(
                checkpoint_path,
                config,
                run_fingerprint,
                state,
                runtime_seconds=(
                    previous_runtime
                    + perf_counter()
                    - started
                    - checkpoint_io_seconds
                ),
                peak_memory_mb=max(previous_peak, _peak_rss_mb()),
            )
            checkpoint_io_seconds += perf_counter() - checkpoint_started
            latest_completed_round = completed_round
            last_checkpointed_round = completed_round

    try:
        result = run_experiment(
            dataset,
            config,
            resume_state=resume_state,
            checkpoint_callback=callback,
        )
    except BaseException as exc:
        if checkpoint_dir is not None and run_fingerprint is not None:
            try:
                _write_failure_record(
                    Path(checkpoint_dir).parent / "last_failure.json",
                    config,
                    run_fingerprint,
                    exc,
                    last_completed_round=latest_completed_round,
                    checkpoint_path=checkpoint_path,
                )
            except OSError as record_error:
                print(f"failed_to_write_error_record={record_error}", file=sys.stderr)
        raise
    runtime = previous_runtime + perf_counter() - started - checkpoint_io_seconds
    final_result = replace(
        result,
        runtime_seconds=runtime,
        peak_memory_mb=max(previous_peak, _peak_rss_mb()),
        checkpoint_io_seconds=checkpoint_io_seconds,
    )
    if (
        retain_success_checkpoint
        and checkpoint_path is not None
        and run_fingerprint is not None
        and latest_state is not None
    ):
        # The loader replays terminal_result directly.  Copying the multi-GiB
        # detector history into the short-lived two-phase terminal checkpoint
        # would add a redundant third full-state write on large CIFAR runs.
        terminal_state = {
            "completed_round": int(latest_state.get("completed_round", 0)),
            "terminal_result": final_result,
        }
        _write_round_checkpoint(
            checkpoint_path,
            config,
            run_fingerprint,
            terminal_state,
            runtime_seconds=runtime,
            peak_memory_mb=final_result.peak_memory_mb,
        )
    else:
        finalize_config_checkpoint(
            checkpoint_dir,
            config,
            run_fingerprint,
        )
    return final_result


def finalize_config_checkpoint(
    checkpoint_dir: Path | None,
    config: ExperimentConfig,
    run_fingerprint: str | None,
) -> None:
    """父进程确认总结果已落盘后，再提交并删除单配置终态检查点。"""

    if checkpoint_dir is None:
        return
    checkpoint_path = _checkpoint_path(Path(checkpoint_dir), config)
    try:
        checkpoint_path.unlink()
    except FileNotFoundError:
        pass
    if run_fingerprint is not None:
        _mark_failure_resolved(
            Path(checkpoint_dir).parent / "last_failure.json",
            config,
            run_fingerprint,
        )


def _durable_replace(temporary: Path, target: Path) -> None:
    """原子替换并尽力同步目录，降低断电后丢失重命名结果的概率。"""

    temporary.replace(target)
    try:
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


def _write_failure_record(
    path: Path,
    config: ExperimentConfig,
    run_fingerprint: str,
    error: BaseException,
    *,
    last_completed_round: int,
    checkpoint_path: Path | None,
) -> None:
    """保存未预料异常及最后一致轮次，便于修复代码后原命令续跑。"""

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_fingerprint": run_fingerprint,
        "config": asdict(config),
        "last_completed_round": last_completed_round,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "resolved": False,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    _durable_replace(temporary, path)


def _mark_failure_resolved(
    path: Path,
    config: ExperimentConfig,
    run_fingerprint: str,
) -> None:
    """同一失败配置完成后保留诊断记录，并明确标注已经解决。"""

    if not path.exists():
        return
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return
    if (
        payload.get("run_fingerprint") != run_fingerprint
        or payload.get("config") != asdict(config)
        or payload.get("resolved") is True
    ):
        return
    payload["resolved"] = True
    payload["resolved_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.resolved.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        _durable_replace(temporary, path)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass


def _peak_rss_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    return peak / 1024


def load_completed_results_snapshot(output_dir: Path) -> list[ExperimentResult] | None:
    """读取已完成配置的事务式快照；不兼容时由调用方回退到 CSV。"""

    path = output_dir / COMPLETED_RESULTS_SNAPSHOT
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except (
        OSError,
        EOFError,
        pickle.PickleError,
        AttributeError,
        ValueError,
        ImportError,
        TypeError,
    ):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or not isinstance(payload.get("results"), list)
        or not all(isinstance(result, ExperimentResult) for result in payload["results"])
    ):
        return None
    return payload["results"]


def _write_completed_results_snapshot(
    output_dir: Path,
    results: list[ExperimentResult],
) -> None:
    """先持久化权威快照，再派生 CSV/JSON，防止多文件替换到一半断电。"""

    target = output_dir / COMPLETED_RESULTS_SNAPSHOT
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "results": list(results),
    }
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    _durable_replace(temporary, target)


def write_result_files(output_dir: Path, results: list[ExperimentResult]) -> None:
    """以原子替换方式保存当前已完成配置，长实验中断后仍可保留进度。"""

    if not results:
        return
    _write_completed_results_snapshot(output_dir, results)
    targets = {
        "summary": output_dir / "summary.csv",
        "rounds": output_dir / "rounds.csv",
        "diagnostics": output_dir / "sm9rrs_diagnostics.csv",
        "json": output_dir / "summary.json",
    }
    temporary = {
        name: path.with_name(f".{path.name}.tmp") for name, path in targets.items()
    }
    write_summary(temporary["summary"], results)
    write_rounds(temporary["rounds"], results)
    write_diagnostics(temporary["diagnostics"], results)
    with temporary["json"].open("w", encoding="utf-8") as handle:
        json.dump([result.summary_dict() for result in results], handle, indent=2)
    for name, target in targets.items():
        _durable_replace(temporary[name], target)


def write_summary(path: Path, results: list[ExperimentResult]) -> None:
    rows = [result.summary_dict() for result in results]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_rounds(path: Path, results: list[ExperimentResult]) -> None:
    rows = []
    for result in results:
        for record in result.records:
            row = {
                "partition": result.config.partition,
                "dirichlet_alpha": result.config.dirichlet_alpha,
                "num_clients": result.config.num_clients,
                "seed": result.config.seed,
                **asdict(record),
            }
            rows.append(row)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_diagnostics(path: Path, results: list[ExperimentResult]) -> None:
    """Write per-client SM9-RRS detector/weight state for every active round."""

    prefix_fields = [
        "partition",
        "dirichlet_alpha",
        "num_clients",
        "method",
        "malicious_ratio",
    ]
    diagnostic_fields = [item.name for item in fields(ClientDiagnosticRecord)]
    rows = []
    for result in results:
        for record in getattr(result, "diagnostics", ()):
            rows.append(
                {
                    "partition": result.config.partition,
                    "dirichlet_alpha": result.config.dirichlet_alpha,
                    "num_clients": result.config.num_clients,
                    "method": result.config.method,
                    "malicious_ratio": result.config.malicious_ratio,
                    **asdict(record),
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=prefix_fields + diagnostic_fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def read_results(summary_path: Path, rounds_path: Path) -> list[ExperimentResult]:
    if not summary_path.exists() or not rounds_path.exists():
        raise FileNotFoundError("summary.csv and rounds.csv must exist for --visualize-only")

    with rounds_path.open(newline="", encoding="utf-8") as handle:
        round_rows = list(csv.DictReader(handle))
    rounds_by_key = {}
    for row in round_rows:
        partition = row.get("partition", "iid")
        alpha = float(row.get("dirichlet_alpha") or 0.0)
        num_clients = int(row["num_clients"]) if row.get("num_clients") else None
        seed = int(row["seed"]) if row.get("seed") else None
        ratio = float(row["malicious_ratio"])
        key = (partition, alpha, num_clients, row["method"], ratio, seed)
        rounds_by_key.setdefault(key, []).append(
            RoundRecord(
                method=row["method"],
                malicious_ratio=ratio,
                round=int(row["round"]),
                accuracy=float(row["accuracy"]),
                error=float(row["error"]),
                accepted_updates=int(row["accepted_updates"]),
                rejected_updates=int(row["rejected_updates"]),
                blacklisted_clients=int(row["blacklisted_clients"]),
                true_positive_revocations=int(row["true_positive_revocations"]),
                false_positive_revocations=int(row["false_positive_revocations"]),
                krum_selected_client=row["krum_selected_client"],
                attack_target_success_rate=_parse_optional_float(
                    row.get("attack_target_success_rate")
                ),
                attack_target_confidence=_parse_optional_float(
                    row.get("attack_target_confidence")
                ),
                nonfinite_updates=int(row.get("nonfinite_updates") or 0),
                attack_active=_parse_bool(row.get("attack_active", "False")),
            )
        )

    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))

    results = []
    for row in summary_rows:
        ratio = float(row["malicious_ratio"])
        partition = row["partition"]
        alpha = float(row["dirichlet_alpha"])
        num_clients = int(row["num_clients"])
        config = ExperimentConfig(
            method=row["method"],
            malicious_ratio=ratio,
            num_clients=num_clients,
            rounds=int(row["rounds"]),
            target_error=float(row["target_error"]),
            local_epochs=int(row["local_epochs"]),
            batch_size=int(row["batch_size"]),
            lr=float(row["lr"]),
            lr_decay=float(row.get("lr_decay") or 1.0),
            compute_backend=row.get("compute_backend") or "numpy",
            device=row.get("device") or "auto",
            partition=partition,
            dirichlet_alpha=alpha,
            attack=row["attack"],
            attack_scale=float(row["attack_scale"]),
            attack_boost=float(row.get("attack_boost") or 10.0),
            attack_epochs=int(row.get("attack_epochs") or 10),
            attack_stealth_steps=int(row.get("attack_stealth_steps") or 10),
            attack_distance_weight=float(
                row.get("attack_distance_weight") or 1e-4
            ),
            attack_source_label=int(row.get("attack_source_label") or 5),
            attack_target_label=int(row.get("attack_target_label") or 7),
            attack_target_count=int(row.get("attack_target_count") or 1),
            attack_start_round=int(row["attack_start_round"]),
            detector_window=int(row["detector_window"]),
            z_threshold=float(row["z_threshold"]),
            checkpoint_interval=int(row.get("checkpoint_interval") or 0),
            detector_subspace_dim=int(row.get("detector_subspace_dim") or 2),
            detector_gap_threshold=float(
                row.get("detector_gap_threshold") or 0.1
            ),
            detector_adjacent_threshold=float(
                row.get("detector_adjacent_threshold")
                or row.get("z_threshold")
                or 3.0
            ),
            detector_anchor_threshold=float(
                row.get("detector_anchor_threshold")
                or row.get("z_threshold")
                or 3.0
            ),
            detector_drift_memory=float(
                row.get("detector_drift_memory") or 0.9
            ),
            detector_drift_allowance=float(
                row.get("detector_drift_allowance") or 1.0
            ),
            detector_drift_threshold=float(
                row.get("detector_drift_threshold") or 5.0
            ),
            detector_decision_rule=(
                row.get("detector_decision_rule") or "any"
            ),
            crypto_mode=row["crypto_mode"],
            dkg_threshold=int(row.get("dkg_threshold") or 2),
            dkg_nodes=int(row.get("dkg_nodes") or 3),
            early_stop=_parse_bool(row.get("early_stop", "True")),
            eval_interval=int(row.get("eval_interval") or 1),
            sm9_workers=int(row.get("sm9_workers") or 1),
            suspicion_penalty_factor=float(row.get("suspicion_penalty_factor") or 0.5),
            suspicion_recovery_factor=float(row.get("suspicion_recovery_factor") or 2.0),
            suspicion_remove_after=int(row.get("suspicion_remove_after") or 3),
            suspicion_count_max=int(
                row.get("suspicion_count_max")
                or row.get("suspicion_remove_after")
                or 3
            ),
            vert_history_window=int(row.get("vert_history_window") or 10),
            vert_projection_dim=int(row.get("vert_projection_dim") or 128),
            vert_predict_epochs=int(row.get("vert_predict_epochs") or 5),
            vert_predict_lr=float(row.get("vert_predict_lr") or 1e-2),
            vert_top_k=int(row.get("vert_top_k") or 0),
            vert_use_ratio_prior=_parse_bool(
                row.get("vert_use_ratio_prior", "False")
            ),
            fedre_threshold=float(row.get("fedre_threshold") or 0.6),
            fedre_initial_iterations=int(
                row.get("fedre_initial_iterations") or 800
            ),
            fedre_max_iterations=int(row.get("fedre_max_iterations") or 2000),
            fedre_synthetic_steps=int(row.get("fedre_synthetic_steps") or 5),
            fedre_images_per_class=int(
                row.get("fedre_images_per_class") or 1
            ),
            fedre_image_lr=float(row.get("fedre_image_lr") or 0.5),
            fedre_label_lr=float(row.get("fedre_label_lr") or 0.2),
            fedre_teacher_lr=float(row.get("fedre_teacher_lr") or 0.1),
            fedre_teacher_lr_lr=float(
                row.get("fedre_teacher_lr_lr") or 5e-6
            ),
            seed=int(row["seed"]),
        )
        seed = int(row["seed"])
        record_key = (partition, alpha, num_clients, row["method"], ratio, seed)
        legacy_seed_key = (partition, alpha, num_clients, row["method"], ratio, None)
        legacy_record_key = (partition, alpha, None, row["method"], ratio, None)
        legacy_iid_key = ("iid", 0.0, None, row["method"], ratio, None)
        records = rounds_by_key.get(record_key)
        if records is None:
            records = rounds_by_key.get(legacy_seed_key)
        if records is None:
            records = rounds_by_key.get(legacy_record_key)
        if records is None and partition == "iid":
            records = rounds_by_key.get(legacy_iid_key)
        results.append(
            ExperimentResult(
                config=config,
                records=records or [],
                final_accuracy=float(row["final_accuracy"]),
                final_error=float(row["final_error"]),
                stopped_round=int(row["stopped_round"]),
                malicious_clients=tuple(filter(None, row["malicious_clients"].split(","))),
                blacklisted_clients=tuple(filter(None, row["blacklisted_clients"].split(","))),
                runtime_seconds=float(row.get("runtime_seconds") or 0.0),
                checkpoint_io_seconds=float(
                    row.get("checkpoint_io_seconds") or 0.0
                ),
                peak_memory_mb=float(row.get("peak_memory_mb") or 0.0),
                nonfinite_updates=int(row.get("nonfinite_updates") or 0),
                stage_timings=StageTimings(
                    training_seconds=float(row.get("training_seconds") or 0.0),
                    attack_seconds=float(row.get("attack_seconds") or 0.0),
                    hash_seconds=float(row.get("hash_seconds") or 0.0),
                    packet_build_seconds=float(row.get("packet_build_seconds") or 0.0),
                    sign_seconds=float(row.get("sign_seconds") or 0.0),
                    verify_seconds=float(row.get("verify_seconds") or 0.0),
                    detection_seconds=float(row.get("detection_seconds") or 0.0),
                    aggregation_seconds=float(row.get("aggregation_seconds") or 0.0),
                    evaluation_seconds=float(row.get("evaluation_seconds") or 0.0),
                    crypto_setup_wall_seconds=float(
                        row.get("crypto_setup_wall_seconds") or 0.0
                    ),
                    crypto_packet_wall_seconds=float(
                        row.get("crypto_packet_wall_seconds") or 0.0
                    ),
                    crypto_audit_wall_seconds=float(
                        row.get("crypto_audit_wall_seconds") or 0.0
                    ),
                    crypto_finalize_wall_seconds=float(
                        row.get("crypto_finalize_wall_seconds") or 0.0
                    ),
                ),
            )
        )
    return results


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    return float(value)


def print_summary(results: list[ExperimentResult]) -> None:
    print(
        "partition clients method ratio final_acc final_error target_asr target_conf "
        "stopped blacklisted runtime_s no_crypto_s crypto_s train_s hash_s sign_s "
        "verify_s eval_s peak_mem_mb"
    )
    for result in results:
        timings = result.stage_timings
        final_record = result.records[-1] if result.records else None
        target_asr = (
            final_record.attack_target_success_rate
            if final_record is not None
            else None
        )
        target_confidence = (
            final_record.attack_target_confidence
            if final_record is not None
            else None
        )
        target_asr_text = "-" if target_asr is None else f"{target_asr:0.3f}"
        target_confidence_text = (
            "-" if target_confidence is None else f"{target_confidence:0.3f}"
        )
        print(
            f"{result.config.partition:9s} "
            f"{result.config.num_clients:7d} "
            f"{result.config.method:13s} "
            f"{result.config.malicious_ratio:0.2f} "
            f"{result.final_accuracy:0.4f} "
            f"{result.final_error:0.4f} "
            f"{target_asr_text:10s} "
            f"{target_confidence_text:11s} "
            f"{result.stopped_round:3d} "
            f"{len(result.blacklisted_clients):3d} "
            f"{result.runtime_seconds:8.2f} "
            f"{result.runtime_without_crypto_seconds:11.2f} "
            f"{timings.crypto_wall_seconds:8.2f} "
            f"{timings.training_seconds:8.2f} "
            f"{timings.hash_seconds:7.2f} "
            f"{timings.sign_seconds:7.2f} "
            f"{timings.verify_seconds:8.2f} "
            f"{timings.evaluation_seconds:7.2f} "
            f"{result.peak_memory_mb:10.2f}"
        )


if __name__ == "__main__":
    main()
