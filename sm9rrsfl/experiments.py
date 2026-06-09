"""Command line runner for image poisoning experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
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
    print(f"compute_backend={describe_compute_backend(args.compute_backend, args.device)}")

    partitions = args.partitions or [args.partition]
    client_counts = args.client_counts or [args.num_clients]
    for partition in partitions:
        for num_clients in client_counts:
            for ratio in args.ratios:
                for method in args.methods:
                    config = ExperimentConfig(
                        method=method,
                        malicious_ratio=ratio,
                        num_clients=num_clients,
                        rounds=args.rounds,
                        target_error=args.target_error,
                        local_epochs=args.local_epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
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
                        suspicion_penalty_factor=args.suspicion_penalty_factor,
                        suspicion_recovery_factor=args.suspicion_recovery_factor,
                        suspicion_remove_after=args.suspicion_remove_after,
                        seed=args.seed,
                    )
                    print(
                        "running "
                        f"partition={partition} clients={num_clients} "
                        f"method={method} ratio={ratio:.2f}"
                    )
                    results.append(run_measured_experiment(dataset, config))

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
    parser.add_argument("--suspicion-penalty-factor", type=float, default=0.5)
    parser.add_argument("--suspicion-recovery-factor", type=float, default=2.0)
    parser.add_argument("--suspicion-remove-after", type=int, default=3)
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--visualize-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    raw_args = sys.argv[1:] if argv is None else argv
    return apply_presets(parser.parse_args(argv), raw_args)


def apply_presets(args: argparse.Namespace, raw_args: list[str]) -> argparse.Namespace:
    if not args.cifar10_clean_baseline:
        return args

    args.dataset = "cifar10"
    args.methods = ["fedavg"]
    args.ratios = [0.0]
    args.attack = "none"
    args.partition = "iid"
    args.partitions = ["iid"]
    if not _has_any_option(raw_args, "--train-samples"):
        args.train_samples = None
    if not _has_any_option(raw_args, "--test-samples"):
        args.test_samples = None
    if not _has_any_option(raw_args, "--rounds"):
        args.rounds = 100
    if not _has_any_option(raw_args, "--local-epochs"):
        args.local_epochs = 2
    if not _has_any_option(raw_args, "--batch-size"):
        args.batch_size = 64
    if not _has_any_option(raw_args, "--lr"):
        args.lr = 0.01
    if not _has_any_option(raw_args, "--num-clients", "--client-counts", "--num-clients-list"):
        args.num_clients = 20
        args.client_counts = [20]
    return args


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
