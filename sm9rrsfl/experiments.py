"""Command line runner for image poisoning experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, replace
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sys
from time import perf_counter
import traceback

from .datasets import load_image_dataset
from .crypto import rrs_backend_name, sm3_backend_name
from .fl import (
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
CHECKPOINT_SCHEMA_VERSION = 1
COMPLETED_RESULTS_SNAPSHOT = ".completed_results.pickle"
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


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args)
    if args.visualize_only:
        results = read_results(output_dir / "summary.csv", output_dir / "rounds.csv")
        visualization_path = generate_visualizations(results, output_dir)
        print(f"wrote {visualization_path}")
        return

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
    print(f"compute_backend={describe_compute_backend(args.compute_backend, args.device)}", flush=True)

    configs = build_experiment_configs(args)
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
    current_manifest_matches = bool(
        args.resume
        and previous_manifest is not None
        and previous_manifest.get("fingerprint") == manifest["fingerprint"]
    )
    results: list[ExperimentResult] = []
    archived_resume_dir = None
    if current_manifest_matches:
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
    completed_configs = {result.config for result in results}
    pending_configs = [
        config for config in runnable_configs if config not in completed_configs
    ]
    if results:
        print(f"resumed_completed_configs={len(results)}", flush=True)
    if skipped_configs:
        print(f"skipped_invalid_configs={len(skipped_configs)}", flush=True)
        for config, reason in skipped_configs:
            print(f"skipped {_format_config_key(config)} reason={reason}", flush=True)
    if any(config.method == "sm9rrs" for config in configs):
        print(f"sm3_backend={sm3_backend_name()}", flush=True)
        if args.crypto_mode == "sm9" and args.accumulator_mode == "dynamic":
            print(f"rrs_backend={rrs_backend_name()}", flush=True)
    jobs = resolve_parallel_jobs(args.jobs, dataset, pending_configs, args)
    print(f"experiment_jobs={jobs}", flush=True)
    progress = ProgressReporter(
        total=len(runnable_configs),
        completed=sum(result.config in runnable_configs for result in results),
        enabled=not args.no_progress,
    )
    checkpoint_dir = output_dir / ".checkpoints" if args.resume else None
    # 若上次恰好在“结果快照已落盘、检查点尚未删除”的极短窗口中退出，
    # 已完成结果是权威状态，启动时清理对应的冗余终态检查点。
    for result in results:
        finalize_config_checkpoint(
            checkpoint_dir,
            result.config,
            manifest["fingerprint"],
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
                for future in as_completed(futures):
                    config = futures[future]
                    result = future.result()
                    results.append(result)
                    write_result_files(output_dir, results)
                    finalize_config_checkpoint(
                        checkpoint_dir,
                        config,
                        manifest["fingerprint"],
                    )
                    progress.finish_config(config)
        except PermissionError as exc:
            print(f"process_pool_unavailable={exc}; falling back to thread pool", flush=True)
            finished_configs = {result.config for result in results}
            remaining_configs = [
                config for config in pending_configs if config not in finished_configs
            ]
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(
                        _run_config_in_thread,
                        (dataset, config, checkpoint_dir, manifest["fingerprint"]),
                    ): config
                    for config in remaining_configs
                }
                for future in as_completed(futures):
                    config = futures[future]
                    result = future.result()
                    results.append(result)
                    write_result_files(output_dir, results)
                    finalize_config_checkpoint(
                        checkpoint_dir,
                        config,
                        manifest["fingerprint"],
                    )
                    progress.finish_config(config)
    progress.close()

    summary_path = output_dir / "summary.csv"
    rounds_path = output_dir / "rounds.csv"
    json_path = output_dir / "summary.json"
    write_result_files(output_dir, results)
    visualization_path = None
    if not args.no_visualizations:
        visualization_path = generate_visualizations(results, output_dir)

    elapsed = perf_counter() - started
    print_summary(results)
    print(f"wrote {summary_path}")
    print(f"wrote {rounds_path}")
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
        choices=["sm9rrs", "krum", "ding13", "fedavg"],
        default=["sm9rrs", "krum", "ding13", "fedavg"],
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
        default="1",
        help=(
            "Number of experiment configurations to run in parallel. Use an integer "
            "or 'auto' to estimate a memory-safe limit."
        ),
    )
    parser.add_argument("--attack", choices=["none", "sign_flip", "gaussian", "alternating"], default="alternating")
    parser.add_argument("--attack-scale", type=float, default=5.0)
    parser.add_argument(
        "--attack-start-round",
        type=int,
        default=0,
        help="0 means detector_window + 2, leaving a benign SVD baseline window.",
    )
    parser.add_argument("--detector-window", type=int, default=3)
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--ring-size", type=int, default=5)
    parser.add_argument("--crypto-mode", choices=["sm9", "simulated"], default="sm9")
    parser.add_argument("--accumulator-mode", choices=["dynamic", "none"], default="dynamic")
    parser.add_argument("--strict-ring-verify", action="store_true")
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluate test accuracy every N rounds; the final round is always evaluated.",
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
    parser.add_argument("--suspicion-remove-after", type=int, default=3)
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress and ETA output.")
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
    max_clients = max(args.client_counts or [args.num_clients])
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
        (args.detector_window >= 1, "--detector-window must be at least 1"),
        (args.ring_size >= 1, "--ring-size must be at least 1"),
        (args.eval_interval >= 1, "--eval-interval must be at least 1"),
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
                            attack_start_round=args.attack_start_round,
                            detector_window=args.detector_window,
                            z_threshold=args.z_threshold,
                            ring_size=args.ring_size,
                            crypto_mode=args.crypto_mode,
                            accumulator_mode=args.accumulator_mode,
                            strict_ring_verify=args.strict_ring_verify,
                            early_stop=not args.no_early_stop,
                            eval_interval=args.eval_interval,
                            sm9_workers=args.sm9_workers,
                            suspicion_penalty_factor=args.suspicion_penalty_factor,
                            suspicion_recovery_factor=args.suspicion_recovery_factor,
                            suspicion_remove_after=args.suspicion_remove_after,
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


def resolve_sm9_workers(value: str | int, num_clients: int) -> int:
    if isinstance(value, int):
        workers = value
    else:
        text = value.strip().lower()
        if text == "auto":
            # 原生配对运算会释放 GIL；8 个 worker 通常已能吃满桌面 CPU，继续
            # 增大会增加线程调度和每轮临时内存而几乎没有收益。
            workers = min(8, max(1, os.cpu_count() or 1), max(1, num_clients))
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
    resolved = min(requested, len(configs))
    backend = describe_compute_backend(args.compute_backend, args.device)
    if backend in {"torch:cuda", "torch:mps"}:
        # 当前 CLI 没有“进程 -> GPU”映射；同一设备始终只运行一个配置。
        return 1
    return resolved


def _auto_parallel_jobs(dataset, configs: list[ExperimentConfig], args: argparse.Namespace) -> int:
    total_mb = _physical_memory_mb()
    max_clients = max(config.num_clients for config in configs)
    max_params = max(_parameter_size_for_dataset(dataset) for _ in configs)
    dataset_mb = _dataset_memory_mb(dataset)
    update_mb = max_clients * max_params * 4 / (1024 * 1024)
    per_worker_mb = max(1024.0, dataset_mb * 0.35 + update_mb * 2.5 + 768.0)
    memory_limited = max(1, int((total_mb * 0.75) // per_worker_mb))
    cpu_limited = max(1, os.cpu_count() or 1)
    jobs = min(memory_limited, cpu_limited, len(configs))
    backend = describe_compute_backend(args.compute_backend, args.device)
    if backend in {"torch:cuda", "torch:mps"}:
        # 单 GPU 上并发配置既会争抢显存，也可能触发 CUDA fork 初始化错误。
        jobs = 1
    elif backend.startswith("torch:"):
        jobs = min(jobs, 2)
    return max(1, jobs)


def _physical_memory_mb() -> float:
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
            return status.ullTotalPhys / (1024 * 1024)
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return float(pages * page_size) / (1024 * 1024)
        except (ValueError, OSError, AttributeError):
            pass
    return 16 * 1024.0


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


class ProgressReporter:
    def __init__(
        self,
        *,
        total: int,
        completed: int = 0,
        enabled: bool = True,
        stream=None,
    ) -> None:
        self.total = max(0, total)
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.completed = min(max(0, completed), self.total)
        self.started = perf_counter()
        self.last_message_length = 0
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.closed = False
        if self.enabled and self.total:
            self._write(self._progress_message(current="starting"))

    def start_config(self, config: ExperimentConfig) -> None:
        if not self.enabled:
            print(_format_running_config(config))
            return
        self._write(self._progress_message(current=f"running {_format_config_key(config)}"))

    def finish_config(self, config: ExperimentConfig) -> None:
        self.completed += 1
        if not self.enabled:
            print(f"finished {_format_config_key(config)}")
            return
        self._write(self._progress_message(current=f"finished {_format_config_key(config)}"))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.enabled and self.total:
            self._write(self._progress_message(current="complete"), final=True)

    def _progress_message(self, *, current: str) -> str:
        elapsed = perf_counter() - self.started
        ratio = 1.0 if self.total == 0 else min(1.0, self.completed / self.total)
        filled = int(round(PROGRESS_BAR_WIDTH * ratio))
        bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
        percent = ratio * 100.0
        remaining = self.total - self.completed
        if self.completed:
            eta = elapsed / self.completed * remaining
            eta_text = _format_duration(eta)
        else:
            eta_text = "estimating"
        return (
            f"[{bar}] {self.completed}/{self.total} {percent:5.1f}% "
            f"elapsed={_format_duration(elapsed)} eta={eta_text} {current}"
        )

    def _write(self, message: str, *, final: bool = False) -> None:
        if self.is_tty:
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "config": asdict(config),
        "runtime_seconds": runtime_seconds,
        "peak_memory_mb": peak_memory_mb,
        "state": state,
    }
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    _durable_replace(temporary, path)


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
            _write_round_checkpoint(
                checkpoint_path,
                config,
                run_fingerprint,
                state,
                runtime_seconds=previous_runtime + perf_counter() - started,
                peak_memory_mb=max(previous_peak, _peak_rss_mb()),
            )
            latest_completed_round = int(state["completed_round"])
            latest_state = state

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
    runtime = previous_runtime + perf_counter() - started
    final_result = replace(
        result,
        runtime_seconds=runtime,
        peak_memory_mb=max(previous_peak, _peak_rss_mb()),
    )
    if (
        retain_success_checkpoint
        and checkpoint_path is not None
        and run_fingerprint is not None
        and latest_state is not None
    ):
        terminal_state = dict(latest_state)
        terminal_state["terminal_result"] = final_result
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
        "json": output_dir / "summary.json",
    }
    temporary = {
        name: path.with_name(f".{path.name}.tmp") for name, path in targets.items()
    }
    write_summary(temporary["summary"], results)
    write_rounds(temporary["rounds"], results)
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
                **asdict(record),
            }
            rows.append(row)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
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
        ratio = float(row["malicious_ratio"])
        key = (partition, alpha, num_clients, row["method"], ratio)
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
            attack_start_round=int(row["attack_start_round"]),
            detector_window=int(row["detector_window"]),
            z_threshold=float(row["z_threshold"]),
            ring_size=int(row["ring_size"]),
            crypto_mode=row["crypto_mode"],
            accumulator_mode=row.get("accumulator_mode") or "dynamic",
            strict_ring_verify=_parse_bool(row["strict_ring_verify"]),
            early_stop=_parse_bool(row.get("early_stop", "True")),
            eval_interval=int(row.get("eval_interval") or 1),
            sm9_workers=int(row.get("sm9_workers") or 1),
            suspicion_penalty_factor=float(row.get("suspicion_penalty_factor") or 0.5),
            suspicion_recovery_factor=float(row.get("suspicion_recovery_factor") or 2.0),
            suspicion_remove_after=int(row.get("suspicion_remove_after") or 3),
            seed=int(row["seed"]),
        )
        record_key = (partition, alpha, num_clients, row["method"], ratio)
        legacy_record_key = (partition, alpha, None, row["method"], ratio)
        legacy_iid_key = ("iid", 0.0, None, row["method"], ratio)
        records = rounds_by_key.get(record_key)
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
                peak_memory_mb=float(row.get("peak_memory_mb") or 0.0),
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
                ),
            )
        )
    return results


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def print_summary(results: list[ExperimentResult]) -> None:
    print(
        "partition clients method ratio final_acc final_error stopped blacklisted "
        "runtime_s train_s hash_s sign_s verify_s eval_s peak_mem_mb"
    )
    for result in results:
        timings = result.stage_timings
        print(
            f"{result.config.partition:9s} "
            f"{result.config.num_clients:7d} "
            f"{result.config.method:7s} "
            f"{result.config.malicious_ratio:0.2f} "
            f"{result.final_accuracy:0.4f} "
            f"{result.final_error:0.4f} "
            f"{result.stopped_round:3d} "
            f"{len(result.blacklisted_clients):3d} "
            f"{result.runtime_seconds:8.2f} "
            f"{timings.training_seconds:8.2f} "
            f"{timings.hash_seconds:7.2f} "
            f"{timings.sign_seconds:7.2f} "
            f"{timings.verify_seconds:8.2f} "
            f"{timings.evaluation_seconds:7.2f} "
            f"{result.peak_memory_mb:10.2f}"
        )


if __name__ == "__main__":
    main()
