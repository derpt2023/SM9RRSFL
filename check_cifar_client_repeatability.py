#!/usr/bin/env python3
"""Replay one recorded Ours client; strict settings belong only to child probes.

No training, attack, defense, or original result is changed. Each case is a fresh
process. This checks present numerical repeatability, not the old round-25 NaN.
"""
from __future__ import annotations

import argparse
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

import sm9rrsfl  # Set package thread defaults before NumPy or Torch.
import numpy as np
from sm9rrsfl import fl
import diagnose_cifar_nonfinite as probe
from check_cifar_prefix_repeatability import array_identity, environment


NOTE = ("Independent numerical diagnosis only. Matching current replays does not "
        "reproduce the original multi-GPU execution or establish that its NaN is fixed.")
FLAG_KEYS = ("deterministic", "cudnn_benchmark", "cudnn_deterministic",
             "cudnn_tf32", "matmul_tf32", "torch_threads", "torch_interop_threads")


def validate_baseline_environment(current, recorded, *, strict=False):
    """Require the recorded current-prefix environment, except device mapping.

    A strict child's workspace setting is an explicit experimental difference.
    Matching a device name or mapping is not proof of the old physical GPU.
    """
    differences = {}
    for key in ("python", "executable", "numpy", "torch", "cuda", "cudnn", "gpu", *FLAG_KEYS):
        if key not in recorded or current.get(key) != recorded[key]:
            differences[key] = {"recorded": recorded.get(key), "current": current.get(key)}
    for key, value in recorded.get("environment", {}).items():
        if key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
            continue
        if strict and key == "CUBLAS_WORKSPACE_CONFIG":
            continue
        if current.get("environment", {}).get(key) != value:
            differences["environment." + key] = {
                "recorded": value, "current": current.get("environment", {}).get(key)}
    if not recorded.get("source_sha256"):
        differences["source_sha256"] = "recorded source hashes are missing"
    for key, value in recorded.get("source_sha256", {}).items():
        if current.get("source_sha256", {}).get(key) != value:
            differences["source." + key] = {
                "recorded": value, "current": current.get("source_sha256", {}).get(key)}
    if differences:
        raise ValueError("current environment differs from prefix A1: " + json.dumps(differences))
    return {"recorded_device": recorded.get("device"), "current_device": current.get("device"),
            "recorded_mapping": {k: recorded.get("environment", {}).get(k)
                                 for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")},
            "current_mapping": {k: current.get("environment", {}).get(k)
                                for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")},
            "note": "Mappings are recorded, not treated as proof of historical physical GPU identity."}


def strict_child_environment(parent_environment):
    """Set this before spawning the process, hence before any CUDA call."""
    result = dict(parent_environment)
    result["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return result


def apply_strict_settings(torch_module, baseline_flags):
    """Only call inside the strict child; do not alter either TF32 setting."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("strict child must start with CUBLAS_WORKSPACE_CONFIG=:4096:8")
    before = (torch_module.backends.cudnn.allow_tf32,
              torch_module.backends.cuda.matmul.allow_tf32)
    expected = (baseline_flags["cudnn_tf32"], baseline_flags["matmul_tf32"])
    if before != expected:
        raise ValueError("TF32 settings already differ from the recorded baseline")
    torch_module.use_deterministic_algorithms(True, warn_only=False)
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False
    if (torch_module.backends.cudnn.allow_tf32,
            torch_module.backends.cuda.matmul.allow_tf32) != before:
        raise RuntimeError("strict setup unexpectedly changed TF32 settings")


def _numpy_copy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def run_client_replays(dataset, config, params, indices, client_idx, output, *,
                       warmup=False, repeats=3, expected_delta_sha256=None,
                       environment_metadata=None):
    """CPU-testable entry; output must be new, indices describe the target client.

    Warmup uses round-zero evaluation and clients preceding client_idx. Their GPU
    deltas and CPU copies stay alive through the target replays, as in Ours' loop.
    Every local call starts from the same global parameters; no update is applied.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "config": asdict(config), "round": 1,
              "client_idx": client_idx, "warmup": bool(warmup), "repeats": repeats,
              "local_seed": config.seed + 1009 + client_idx, "runs": [], "note": NOTE}
    try:
        if environment_metadata is not None:
            probe.write_json(output / "environment.json", environment_metadata)
        if config.method != "sm9rrs" or not 0 <= client_idx < config.num_clients:
            raise ValueError("an existing Ours client is required")
        if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        start = config.attack_start_round or config.detector_window + 2
        if start <= 1 and config.malicious_ratio > 0 and config.attack != "none":
            raise ValueError("round one must precede the attack for this local-training probe")
        params = np.asarray(params).copy()
        if params.dtype != np.float32 or not np.isfinite(params).all():
            raise ValueError("recorded model must contain finite float32 parameters")
        partitions = fl.partition_clients(dataset.y_train, config.num_clients,
                                         strategy=config.partition,
                                         dirichlet_alpha=config.dirichlet_alpha, seed=config.seed)
        indices = np.asarray(indices)
        if array_identity(partitions[client_idx]) != array_identity(indices):
            raise ValueError("target indices do not match the original partition")
        spec = fl.model_spec_for_dataset(dataset)
        context = fl._maybe_torch_context(dataset, partitions, spec, config)
        initial_identity = array_identity(params)
        report.update(input=initial_identity, indices=array_identity(indices))

        def local(index):
            return fl._local_train_client_delta(
                params, dataset, partitions[index], client_idx=index, round_id=1,
                model_spec=spec, config=config, torch_context=context)

        retained = []
        if warmup:
            report["warmup_accuracy"] = fl._evaluate_accuracy(params, dataset, spec, config, context)
            if fl.is_alternating_minimization_attack(config.attack):
                targets = fl._select_evaluation_target_indices(dataset, config)
                report["warmup_target_metrics"] = fl._evaluate_attack_target_metrics(
                    params, dataset, targets, spec, config, context)
            for preceding in range(client_idx):
                delta, _ = local(preceding)
                if not fl._update_is_finite(delta):
                    raise ValueError(f"warmup client {preceding} produced a nonfinite update")
                retained.append((delta, _numpy_copy(delta)))
            report["retained_warmup_clients"] = len(retained)

        for repetition in range(1, repeats + 1):
            actual_input = context._ensure_global_vector(params) if context is not None else params
            if array_identity(actual_input) != initial_identity:
                raise ValueError("global model changed before a replay")
            delta, stats = local(client_idx)
            delta_np = _numpy_copy(delta)
            filename = output / f"delta_{repetition:02d}.npy"
            np.save(filename, delta_np, allow_pickle=False)
            identity = array_identity(delta_np)
            report["runs"].append({
                "repeat": repetition, "delta": identity, "delta_path": str(filename.resolve()),
                "finite": bool(np.isfinite(delta_np).all()), "samples": int(stats.samples),
                "matches_A1_delta": (identity["sha256"] == expected_delta_sha256
                                     if expected_delta_sha256 is not None else None),
            })
            probe.write_json(output / "report.json", report)
            print(f"CLIENT_REPEAT warmup={int(warmup)} repeat={repetition} "
                  f"sha256={identity['sha256']} finite={report['runs'][-1]['finite']}", flush=True)
            if not report["runs"][-1]["finite"]:
                raise ValueError("target replay produced a nonfinite update; captured and stopped")
            # Do not accumulate target GPU updates between repeats.
            del delta, delta_np, actual_input
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        probe.write_json(output / "report.json", report)
    return report


def summarize_group(cases):
    runs = [run for case in cases for run in case["replays"]["runs"]]
    if not runs or any(case["status"] != "complete" for case in cases):
        raise ValueError("cannot summarize an incomplete replay group")
    first = np.load(runs[0]["delta_path"], allow_pickle=False).astype(np.float64)
    reference_norm = float(np.linalg.norm(first))
    max_abs = max_l2 = max_relative_l2 = 0.0
    for run in runs[1:]:
        current = np.load(run["delta_path"], allow_pickle=False).astype(np.float64)
        if current.shape != first.shape or not np.isfinite(current).all():
            raise ValueError("saved replay updates differ in shape or contain nonfinite values")
        difference = current - first
        distance = float(np.linalg.norm(difference))
        max_abs = max(max_abs, float(np.max(np.abs(difference))))
        max_l2 = max(max_l2, distance)
        max_relative_l2 = max(max_relative_l2,
                              distance / max(reference_norm, np.finfo(np.float64).tiny))
    return {"cases": len(cases), "replays": len(runs),
            "all_equal": all(run["delta"] == runs[0]["delta"] for run in runs),
            "max_abs_difference_from_first": max_abs, "max_l2_difference_from_first": max_l2,
            "max_relative_l2_difference_from_first": max_relative_l2,
            "matches_A1_delta_count": sum(run["matches_A1_delta"] is True for run in runs),
            "first_delta_sha256": runs[0]["delta"]["sha256"]}


def _read_prefix(prefix, config, client_idx):
    first = prefix / "A1"
    trace = json.loads((first / "numeric_trace.json").read_text())
    recorded_environment = json.loads((first / "environment.json").read_text())
    if trace.get("status") != "prefix_complete" or trace.get("runner") != "measured":
        raise ValueError("prefix A1 must be a completed measured prefix")
    recorded_config = dict(trace["config"])
    requested = asdict(config)
    recorded_config.pop("device", None)
    requested.pop("device", None)
    if recorded_config != requested:
        raise ValueError("prefix A1 and recorded candidate configurations differ")
    models = [r for r in trace["models"] if r["round"] == 0]
    clients = [r for r in trace["clients"] if r["round"] == 1 and r["client_idx"] == client_idx
               and r["entry_point"] == "_local_train_client_delta"]
    if len(models) != 1 or len(clients) != 1:
        raise ValueError("A1 must contain exactly one initial model and target client record")
    params = np.load(first / "params_round_000.npy", allow_pickle=False)
    if array_identity(params) != models[0]["params"] or array_identity(params) != clients[0]["input"]:
        raise ValueError("saved A1 initial parameters do not match its recorded target input")
    return params, clients[0], recorded_environment


def _worker(args, output, source, prefix):
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "mode": args.worker, "warmup": args.warmup, "note": NOTE}
    try:
        config, manifest, _ = probe.select_recorded_run(
            source, args.candidate, args.partition, args.ratio, args.seed)
        if not config.device.startswith("cuda") or not args.device.startswith("cuda"):
            raise ValueError("server replay must retain the recorded CUDA backend")
        config = replace(config, device=args.device)
        params, target, recorded_environment = _read_prefix(prefix, config, args.client)
        # Strict workspace is already set in the environment passed to Popen.
        before = environment(args.device)
        before["source_sha256"][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        probe.write_json(output / "environment_before.json", before)
        mapping = validate_baseline_environment(before, recorded_environment, strict=args.worker == "strict")
        result["baseline_environment_check"] = {"matched": True, "mapping": mapping}
        if args.worker == "strict":
            import torch
            apply_strict_settings(torch, recorded_environment)
        after = environment(args.device)
        after["source_sha256"][Path(__file__).name] = before["source_sha256"][Path(__file__).name]
        probe.write_json(output / "environment.json", after)
        print("VERIFYING_RECORDED_DATA", flush=True)
        dataset = probe.load_verified_data(manifest, args.data_dir)
        partitions = fl.partition_clients(dataset.y_train, config.num_clients,
                                         strategy=config.partition,
                                         dirichlet_alpha=config.dirichlet_alpha, seed=config.seed)
        if not 0 <= args.client < len(partitions) or array_identity(partitions[args.client]) != target["indices"]:
            raise ValueError("reconstructed client indices do not match prefix A1")
        print("DATA_AND_CLIENT_HASHES_MATCH", flush=True)
        result["replays"] = run_client_replays(
            dataset, config, params, partitions[args.client], args.client, output / "replays",
            warmup=args.warmup, repeats=args.repeats,
            expected_delta_sha256=target["delta"]["sha256"], environment_metadata=after)
        result.update(status="complete", source_fingerprint=manifest["fingerprint"])
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        if args.worker == "strict" and any(word in str(exc).lower() for word in ("deterministic", "cublas")):
            result["strict_failure_note"] = "Strict execution failed; no warn-only or CPU fallback was attempted."
        raise
    finally:
        probe.write_json(output / "case_result.json", result)


class CaseFailure(RuntimeError):
    pass


def _spawn_case(args, output, label, mode, warmup):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", mode,
               "--prefix-output", str(args.prefix_output.resolve()),
               "--source-output", str(args.source_output.resolve()), "--candidate", args.candidate,
               "--partition", args.partition, "--ratio", str(args.ratio), "--seed", str(args.seed),
               "--client", str(args.client), "--device", args.device, "--repeats", str(args.repeats),
               "--output-dir", str(output / label)]
    if warmup:
        command.append("--warmup")
    if args.data_dir:
        command.extend(("--data-dir", str(args.data_dir.resolve())))
    child_environment = strict_child_environment(os.environ) if mode == "strict" else dict(os.environ)
    print(f"START {label} mode={mode} warmup={int(warmup)} client={args.client}", flush=True)
    with (output / f"{label}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, cwd=Path(__file__).resolve().parent, env=child_environment)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"[{label}] {line}", end="", flush=True)
            returncode = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
    if returncode:
        raise CaseFailure(f"{label} failed with exit code {returncode}; see {output / label / 'case_result.json'}")
    result = json.loads((output / label / "case_result.json").read_text())
    if result["status"] != "complete":
        raise CaseFailure(label + " did not complete")
    return result


def main():
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-output", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, default=repo / "outputs/cifar10_v7_target_fair_tuning")
    parser.add_argument("--candidate", default="sm9rrs-002")
    parser.add_argument("--partition", default="dirichlet")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=403)
    parser.add_argument("--client", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", choices=("baseline", "strict"), help=argparse.SUPPRESS)
    parser.add_argument("--warmup", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or args.client < 0:
        parser.error("repeats must be 1..10 and client must be nonnegative")
    source, prefix = args.source_output.resolve(), args.prefix_output.resolve()
    output = (args.output_dir or repo / "outputs" /
              ("cifar10_client_repeatability_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    if output.exists() or any(output == old or old in output.parents for old in (source, prefix)):
        parser.error("choose a new output directory outside both source and prefix outputs")
    if args.worker:
        _worker(args, output, source, prefix)
        return
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "client_idx": args.client, "round": 1,
              "groups": {}, "strict": None, "note": NOTE}
    print("CLIENT_REPEATABILITY_OUTPUT " + str(output), flush=True)
    try:
        direct = [_spawn_case(args, output, name, "baseline", False) for name in ("D1", "D2")]
        report["groups"]["baseline_direct"] = summarize_group(direct)
        strict_warmup = False
        need_strict = not report["groups"]["baseline_direct"]["all_equal"]
        trigger = "baseline_direct_not_repeatable" if need_strict else None
        if not need_strict:
            warm = [_spawn_case(args, output, name, "baseline", True) for name in ("W1", "W2")]
            report["groups"]["baseline_warmup"] = summarize_group(warm)
            combined = summarize_group(direct + warm)
            report["groups"]["baseline_combined"] = combined
            need_strict = not combined["all_equal"]
            strict_warmup = True
            if need_strict:
                trigger = ("baseline_warmup_not_repeatable"
                           if not report["groups"]["baseline_warmup"]["all_equal"]
                           else "baseline_changes_with_warmup")
        if need_strict:
            report["strict_trigger"] = trigger
            probe.write_json(output / "comparison.json", report)
            strict = [_spawn_case(args, output, name, "strict", strict_warmup) for name in ("S1", "S2")]
            report["strict"] = {"warmup": strict_warmup, **summarize_group(strict)}
        report["status"] = "compared"
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        probe.write_json(output / "comparison.json", report)
        print("CLIENT_REPEATABILITY_RESULT " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
