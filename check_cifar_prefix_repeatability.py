#!/usr/bin/env python3
"""Compare two-round Ours prefixes in fresh processes, without changing training.

A1/A2 use the production measured wrapper, its checkpoint writer, and one worker
thread. B uses the diagnostic wrapper. All three receive the same observational
hashing hooks. This does not recreate the original multi-GPU scheduling history.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import sm9rrsfl  # Initialize the same thread defaults before NumPy/Torch.
import numpy as np
from sm9rrsfl import experiments, fl
import diagnose_cifar_nonfinite as probe


METRICS = ("accuracy", "attack_target_success_rate", "accepted_updates",
           "rejected_updates", "nonfinite_updates", "malicious_weight_mass")


def array_identity(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}


def run_case(dataset, config, output, runner, reference, fingerprint, *, metadata=None):
    """Observe inputs/outputs; never alter model tensors or optimizer settings."""
    if config.method != "sm9rrs" or config.rounds < 2:
        raise ValueError("this short diagnostic requires an Ours config of at least two rounds")
    output.mkdir(parents=True, exist_ok=False)
    if metadata is not None:
        probe.write_json(output / "environment.json", metadata)
    trace = {"runner": runner, "status": "running", "config": asdict(config),
             "models": [], "clients": [], "historical_metric_mismatches": []}
    training_originals = {name: getattr(fl, name) for name in (
        "_local_train_client_delta", "_alternating_minimization_client_delta")}
    base_run = fl.run_experiment
    measured_base_run = experiments.run_experiment

    def recorded_client(name):
        original = training_originals[name]

        def call(*args, **kwargs):
            input_identity = array_identity(args[0])
            index_identity = array_identity(args[2])
            delta, stats = original(*args, **kwargs)
            trace["clients"].append({
                "round": kwargs["round_id"], "client_idx": kwargs["client_idx"],
                "entry_point": name, "input": input_identity,
                "indices": index_identity, "delta": array_identity(delta),
            })
            return delta, stats
        return call

    def observed_run(*args, **kwargs):
        upstream = kwargs.get("checkpoint_callback")

        def callback(state):
            rd = int(state["completed_round"])
            record = state["records"][-1]
            metrics = {key: getattr(record, key) for key in METRICS}
            trace["models"].append({"round": rd, "params": array_identity(state["params"]),
                                    "metrics": metrics})
            np.save(output / f"params_round_{rd:03d}.npy", state["params"])
            expected = reference.get(rd, {})
            changes = {key: {"original": expected[key], "current": metrics[key]}
                       for key in METRICS if expected.get(key) not in (None, "")
                       and (metrics[key] is None or not np.isfinite(float(metrics[key]))
                            or abs(float(expected[key]) - float(metrics[key])) > 1e-7)}
            if changes:
                trace["historical_metric_mismatches"].append({"round": rd, "values": changes})
            probe.write_json(output / "numeric_trace.json", trace)
            print(f"PREFIX runner={runner} round={rd} accuracy={record.accuracy:.6f}", flush=True)
            # The measured control writes its real checkpoint before stopping.
            if upstream is not None:
                upstream(state)
            if rd >= 2:
                raise probe.ProbeStop("intentional two-round diagnostic limit")
        return base_run(*args, **{**kwargs, "checkpoint_callback": callback})

    try:
        for name in training_originals:
            setattr(fl, name, recorded_client(name))
        fl.run_experiment = observed_run
        experiments.run_experiment = observed_run
        if runner == "probe":
            result = probe.run_probe(dataset, config, output / "probe", 2, reference)
            if result["status"] != "no_nonfinite_client_update_within_limit":
                raise RuntimeError("probe stopped unexpectedly: " + result["status"])
        elif runner == "measured":
            with ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    executor.submit(
                        experiments.run_measured_experiment, dataset, config,
                        checkpoint_dir=output / "checkpoints", run_fingerprint=fingerprint,
                    ).result()
                except probe.ProbeStop:
                    pass
        else:
            raise ValueError("unknown runner")
        if [r["round"] for r in trace["models"]] != [0, 1, 2]:
            raise RuntimeError("the prefix did not complete rounds 0, 1, 2")
        trace["status"] = "prefix_complete"
    except BaseException as exc:
        trace.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        fl.run_experiment = base_run
        experiments.run_experiment = measured_base_run
        for name, original in training_originals.items():
            setattr(fl, name, original)
        probe.write_json(output / "numeric_trace.json", trace)
    return trace


def compare_cases(left, right):
    if left["status"] != "prefix_complete" or right["status"] != "prefix_complete":
        raise ValueError("cannot compare incomplete prefixes")
    if left["config"] != right["config"]:
        raise ValueError("prefix configurations differ")
    if any([r["round"] for r in trace["models"]] != [0, 1, 2] for trace in (left, right)):
        raise ValueError("missing or duplicate model rounds")
    differences = []
    for a, b in zip(left["models"], right["models"]):
        if a["round"] != b["round"]:
            raise ValueError("round identities differ")
        for field in ("params", "metrics"):
            if a[field] != b[field]:
                differences.append({"round": a["round"], "field": field})
    key = lambda r: (r["round"], r["client_idx"], r["entry_point"])
    a_clients = {key(row): row for row in left["clients"]}
    b_clients = {key(row): row for row in right["clients"]}
    if len(a_clients) != len(left["clients"]) or len(b_clients) != len(right["clients"]):
        raise ValueError("duplicate client identities in numeric trace")
    first = None
    for identity in sorted(a_clients.keys() | b_clients.keys()):
        a, b = a_clients.get(identity), b_clients.get(identity)
        fields = (["client_presence"] if a is None or b is None else
                  [f for f in ("input", "indices", "delta") if a[f] != b[f]])
        if fields:
            first = {"round": identity[0], "client_idx": identity[1],
                     "entry_point": identity[2], "fields": fields,
                     "same_input_and_indices": bool(a is not None and b is not None
                         and a["input"] == b["input"] and a["indices"] == b["indices"])}
            break
    return {"equal": not differences and first is None,
            "model_or_metric_differences": differences, "first_client_difference": first}


def environment(device):
    import torch
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("this server diagnostic requires the recorded CUDA backend")
    repo = Path(__file__).resolve().parent
    return {
        "python": sys.version, "executable": sys.executable, "numpy": np.__version__,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(device),
        "device": device, "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NVIDIA_TF32_OVERRIDE")},
        "source_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((repo / "sm9rrsfl").glob("*.py"))
            + [Path(__file__).resolve(), Path(probe.__file__).resolve()]},
    }


def main():
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, default=repo / "outputs/cifar10_v7_target_fair_tuning")
    parser.add_argument("--candidate", default="sm9rrs-002")
    parser.add_argument("--partition", default="dirichlet")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=403)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", choices=("measured", "probe"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = (args.output_dir or repo / "outputs" /
              ("cifar10_prefix_repeatability_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    source = args.source_output.resolve()
    if output == source or source in output.parents or output.exists():
        parser.error("choose a new output directory outside the original tuning output")
    if args.worker:
        config, manifest, reference = probe.select_recorded_run(
            source, args.candidate, args.partition, args.ratio, args.seed)
        if config.method != "sm9rrs" or not config.device.startswith("cuda"):
            parser.error("only the recorded Ours CUDA configuration is supported")
        config = replace(config, device=args.device)
        metadata = environment(args.device)
        print("VERIFYING_RECORDED_DATA", flush=True)
        dataset = probe.load_verified_data(manifest, args.data_dir)
        print("DATA_DIGESTS_MATCH candidate=" + args.candidate, flush=True)
        run_case(dataset, config, output, args.worker, reference, manifest["fingerprint"],
                 metadata=metadata)
        return

    output.mkdir(parents=True, exist_ok=False)
    print("REPEATABILITY_OUTPUT " + str(output), flush=True)
    traces, environments = {}, {}
    for label, runner in (("A1", "measured"), ("B", "probe"), ("A2", "measured")):
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", runner,
                   "--source-output", str(source), "--candidate", args.candidate,
                   "--partition", args.partition, "--ratio", str(args.ratio), "--seed", str(args.seed),
                   "--device", args.device, "--output-dir", str(output / label)]
        if args.data_dir:
            command.extend(("--data-dir", str(args.data_dir.resolve())))
        print(f"START {label} runner={runner} until_round=2", flush=True)
        with (output / f"{label}.log").open("w") as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, cwd=repo) as process:
                try:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(f"[{label}] {line}", end="", flush=True)
                except BaseException:
                    process.terminate()
                    raise
                returncode = process.wait()
        if returncode:
            probe.write_json(output / "comparison.json", {"status": "worker_failed", "case": label})
            raise SystemExit(returncode)
        traces[label] = json.loads((output / label / "numeric_trace.json").read_text())
        environments[label] = json.loads((output / label / "environment.json").read_text())

    report = {"status": "compared", "environment_equal": environments["A1"] == environments["B"] == environments["A2"],
              "A1_vs_A2": compare_cases(traces["A1"], traces["A2"]),
              "A1_vs_B": compare_cases(traces["A1"], traces["B"]),
              "B_vs_A2": compare_cases(traces["B"], traces["A2"]),
              "historical_mismatches": {name: trace["historical_metric_mismatches"] for name, trace in traces.items()},
              "note": "Current short prefixes only; original multi-GPU execution and the round-25 NaN are not reproduced by this check."}
    for left, right in (("A1", "A2"), ("A1", "B"), ("B", "A2")):
        comparison = report[f"{left}_vs_{right}"]
        for difference in comparison["model_or_metric_differences"]:
            if difference["field"] != "params":
                continue
            filename = f"params_round_{difference['round']:03d}.npy"
            a = np.load(output / left / filename, allow_pickle=False).astype(np.float64)
            b = np.load(output / right / filename, allow_pickle=False).astype(np.float64)
            difference["max_abs_difference"] = float(np.max(np.abs(a - b)))
            difference["relative_l2_difference"] = float(
                np.linalg.norm(a - b) / max(float(np.linalg.norm(a)), np.finfo(np.float64).tiny))
    probe.write_json(output / "comparison.json", report)
    print("REPEATABILITY_RESULT " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
