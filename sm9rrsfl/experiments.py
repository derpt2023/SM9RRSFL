"""Command line runner for image poisoning experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
from dataclasses import asdict, replace
import ctypes
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

from .datasets import load_image_dataset
from .fl import ExperimentConfig, ExperimentResult, RoundRecord, run_experiment
from .model import describe_compute_backend
from .visualization import generate_visualizations


DEFAULT_RATIOS = (0.00, 0.10, 0.20, 0.40, 0.45, 0.60, 0.80)
DEFAULT_OUTPUT_ROOT = Path("outputs")
PROGRESS_BAR_WIDTH = 28
_WORKER_DATASET = None
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
    results: list[ExperimentResult] = []
    started = perf_counter()
    print(f"compute_backend={describe_compute_backend(args.compute_backend, args.device)}", flush=True)

    configs = build_experiment_configs(args)
    jobs = resolve_parallel_jobs(args.jobs, dataset, configs, args)
    print(f"experiment_jobs={jobs}", flush=True)
    progress = ProgressReporter(total=len(configs), enabled=not args.no_progress)
    if jobs == 1:
        for config in configs:
            progress.start_config(config)
            results.append(run_measured_experiment(dataset, config))
            progress.finish_config(config)
    else:
        try:
            with ProcessPoolExecutor(
                max_workers=jobs,
                initializer=_init_worker_dataset,
                initargs=(dataset,),
            ) as executor:
                result_iter = executor.map(_run_config_in_worker, configs)
                for config, result in zip(configs, result_iter):
                    results.append(result)
                    progress.finish_config(config)
        except PermissionError as exc:
            print(f"process_pool_unavailable={exc}; falling back to thread pool", flush=True)
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                tasks = [(dataset, config) for config in configs]
                for config, result in zip(configs, executor.map(_run_config_in_thread, tasks)):
                    results.append(result)
                    progress.finish_config(config)
    progress.close()

    summary_path = output_dir / "summary.csv"
    rounds_path = output_dir / "rounds.csv"
    json_path = output_dir / "summary.json"
    write_summary(summary_path, results)
    write_rounds(rounds_path, results)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump([result.summary_dict() for result in results], handle, indent=2)
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
        default="numpy",
        help=(
            "Local CNN compute backend. numpy preserves the original implementation; "
            "torch enables optional PyTorch CPU/GPU execution; auto uses torch only "
            "when the requested device is available."
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
        default="1",
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
    parser.add_argument("--visualize-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    raw_args = sys.argv[1:] if argv is None else argv
    args = apply_presets(parser.parse_args(argv), raw_args)
    max_clients = max(args.client_counts or [args.num_clients])
    try:
        args.sm9_workers = resolve_sm9_workers(args.sm9_workers, max_clients)
    except ValueError as exc:
        parser.error(str(exc))
    if args.eval_interval < 1:
        parser.error("--eval-interval must be at least 1")
    return args


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
            workers = min(max(1, os.cpu_count() or 1), max(1, num_clients))
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
    max_clients = max(config.num_clients for config in configs)
    max_params = max(_parameter_size_for_dataset(dataset) for _ in configs)
    dataset_mb = _dataset_memory_mb(dataset)
    update_mb = max_clients * max_params * 4 / (1024 * 1024)
    per_worker_mb = max(1024.0, dataset_mb * 0.35 + update_mb * 2.5 + 768.0)
    memory_limited = max(1, int((total_mb * 0.75) // per_worker_mb))
    cpu_limited = max(1, os.cpu_count() or 1)
    jobs = min(memory_limited, cpu_limited, len(configs))
    if args.compute_backend != "numpy":
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


def _init_worker_dataset(dataset) -> None:
    global _WORKER_DATASET
    _WORKER_DATASET = dataset


def _run_config_in_worker(config: ExperimentConfig) -> ExperimentResult:
    if _WORKER_DATASET is None:
        raise RuntimeError("worker dataset was not initialized")
    return run_measured_experiment(_WORKER_DATASET, config)


def _run_config_in_thread(task) -> ExperimentResult:
    dataset, config = task
    return run_measured_experiment(dataset, config)


class ProgressReporter:
    def __init__(self, *, total: int, enabled: bool = True, stream=None) -> None:
        self.total = max(0, total)
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.completed = 0
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


def run_measured_experiment(dataset, config: ExperimentConfig) -> ExperimentResult:
    started = perf_counter()
    result = run_experiment(dataset, config)
    runtime = perf_counter() - started
    return replace(
        result,
        runtime_seconds=runtime,
        peak_memory_mb=_peak_rss_mb(),
    )


def _peak_rss_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak / (1024 * 1024)
    return peak / 1024


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
            )
        )
    return results


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def print_summary(results: list[ExperimentResult]) -> None:
    print("partition clients method ratio final_acc final_error stopped blacklisted runtime_s peak_mem_mb")
    for result in results:
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
            f"{result.peak_memory_mb:10.2f}"
        )


if __name__ == "__main__":
    main()
