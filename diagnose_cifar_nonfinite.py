#!/usr/bin/env python3
"""Bounded single-run diagnosis; never writes to the original tuning output.

Uses the recorded candidate and verifies all three data digests. The original
100-round config is retained; a callback stops the diagnostic at --until-round.
Only an already failing client is replayed with Python-level tensor inspection.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import inspect
import json
import linecache
import math
from pathlib import Path
import pickle
import sys

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import numpy as np

from sm9rrsfl import fl, torch_backend
from sm9rrsfl.datasets import load_image_dataset, stratified_training_three_way_split
from sm9rrsfl.experiments import _array_content_digest


class ProbeStop(Exception):
    pass


def write_json(path, value):
    # Represent bad diagnostic numbers explicitly; never alter training tensors.
    def serializable(item):
        if isinstance(item, dict):
            return {k: serializable(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [serializable(v) for v in item]
        if isinstance(item, float) and not math.isfinite(item):
            return str(item)
        return item
    path.write_text(json.dumps(serializable(value), ensure_ascii=False,
                               indent=2, allow_nan=False) + "\n")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def tensor_stats(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    finite = np.isfinite(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "finite": bool(finite.all()), "nan": int(np.isnan(array).sum()),
            "inf": int(np.isinf(array).sum()),
            "max_abs_finite": float(np.abs(array[finite]).max()) if finite.any() else None}


def replay_with_trace(function, call_args, call_kwargs):
    """Observe the first nonfinite tensor in ONE replay, without changing SGD.

    The reported previous source line is an observation boundary, not a CUDA
    kernel traceback. Loss/gradient/parameter checks also cover no_grad updates.
    """
    methods = (torch_backend.TorchTrainingContext.local_train_delta_resident,
               torch_backend.TorchTrainingContext.alternating_minimization_delta_resident)
    codes = {method.__code__ for method in methods}
    source, offset = inspect.getsourcelines(methods[1])
    target_line = offset + next(i for i, line in enumerate(source) if "target_logits =" in line)
    loop_line = offset + next(i for i, line in enumerate(source) if "for block_start in" in line)
    previous_lines, seen = {}, {}
    found = None

    def tracer(frame, event, arg):
        nonlocal found
        if frame.f_code not in codes:
            return None
        if event not in ("line", "return"):
            return tracer
        local = frame.f_locals
        torch = local["self"].torch
        named = {key: local[key] for key in (
            "global_tensor", "benign_delta", "benign_reference_vector", "logits",
            "target_logits", "loss", "adversarial_loss", "gradient", "delta"
        ) if key in local and torch.is_tensor(local[key])}
        for index, param in enumerate(local.get("params", ())):
            named[f"params[{index}]"] = param
            if param.grad is not None:
                named[f"params[{index}].grad"] = param.grad
        for name, value in named.items():
            key = (id(frame), name)
            old = seen.get(key)
            version = value._version
            if old is not None and old[0] is value and old[1] == version:
                continue
            seen[key] = (value, version)
            if not bool(torch.isfinite(value.detach()).all().item()):
                previous = previous_lines.get(id(frame), frame.f_lineno)
                if frame.f_code is methods[0].__code__:
                    caller = frame.f_back.f_code if frame.f_back else None
                    phase = "benign_reference" if caller is methods[1].__code__ else "local_training"
                else:
                    phase = ("attack_target" if previous >= target_line else
                             "attack_stealth" if previous >= loop_line else "attack_setup")
                found = {"phase": phase, "function": frame.f_code.co_name,
                         "tensor": name, "tensor_stats": tensor_stats(value),
                         "previous_line": previous,
                         "previous_source": linecache.getline(frame.f_code.co_filename, previous).strip(),
                         "observed_at_line": frame.f_lineno,
                         "block_start": local.get("block_start"),
                         "local_batch_start": local.get("start")}
                raise ProbeStop("first nonfinite tensor observed during client replay")
        previous_lines[id(frame)] = frame.f_lineno
        return tracer

    previous_trace = sys.gettrace()
    try:
        sys.settrace(tracer)
        delta, _ = function(*call_args, **call_kwargs)
        return {"status": "replay_returned", "delta": tensor_stats(delta)}
    except ProbeStop:
        return {"status": "first_nonfinite_observed", **found}
    except Exception as exc:
        return {"status": "replay_exception", "type": type(exc).__name__, "message": str(exc)}
    finally:
        sys.settrace(previous_trace)


def select_recorded_run(root, candidate, partition, ratio, seed):
    phase = json.loads((root / "tuning_progress.json").read_text())["phases"]["validation"]
    state = root / ".tuning_state" / "validation" / phase["fingerprint"]
    manifest = json.loads((state / "run_manifest.json").read_text())
    if phase["status"] != "complete" or manifest["fingerprint"] != phase["fingerprint"]:
        raise ValueError("source validation phase is incomplete or mismatched")
    matches = [entry for entry in manifest["candidates"]
               if entry["candidate_id"] == candidate
               and entry["config"]["partition"] == partition
               and abs(entry["config"]["malicious_ratio"] - ratio) < 1e-12
               and entry["config"]["seed"] == seed]
    if len(matches) != 1:
        raise ValueError("candidate/scenario must identify exactly one recorded experiment")
    config = fl.ExperimentConfig(**matches[0]["config"])
    validation, summaries = read_csv(root / "validation_results.csv"), read_csv(state / "summary.csv")
    if len(validation) != len(summaries) or len(validation) != phase["total"]:
        raise ValueError("source result counts differ")
    indexes = []
    for index, (v, s) in enumerate(zip(validation, summaries)):
        if any(v.get(k) != value for k, value in s.items()):
            raise ValueError("source summary/validation order differs")
        if (v["candidate_id"] == candidate and v["partition"] == partition
                and abs(float(v["malicious_ratio"]) - ratio) < 1e-12 and int(v["seed"]) == seed):
            indexes.append(index)
    if len(indexes) != 1:
        raise ValueError("source CSV does not uniquely identify selected candidate")
    selected = indexes[0]
    # CSV contains a runtime device, which may differ from its planned manifest.
    for key, expected in asdict(config).items():
        if key not in ("device", "sm9_workers") and str(expected) != summaries[selected][key]:
            raise ValueError(f"selected CSV and manifest disagree on {key}")
    blocks = []
    for row in read_csv(state / "rounds.csv"):
        if int(row["round"]) == 0:
            blocks.append([])
        if not blocks:
            raise ValueError("missing initial round")
        blocks[-1].append(row)
    if len(blocks) != len(summaries):
        raise ValueError("round block count mismatch")
    block = blocks[selected]
    s = summaries[selected]
    keys = ("method", "partition", "dirichlet_alpha", "num_clients", "malicious_ratio", "seed")
    if (config.eval_interval != 1
            or [int(r["round"]) for r in block] != list(range(int(s["stopped_round"]) + 1))
            or any(r[k] != s[k] for r in block for k in keys)
            or sum(int(r["nonfinite_updates"]) for r in block) != int(s["nonfinite_updates"])):
        raise ValueError("selected round block is inconsistent")
    return config, manifest, {int(row["round"]): row for row in block}


def load_verified_data(manifest, data_dir=None):
    identity, context = manifest["dataset"], manifest["tuning_context"]
    if identity["name"] != "cifar10":
        raise ValueError("this probe requires the full CIFAR-10 protocol")
    original = load_image_dataset("cifar10", data_dir or identity["data_dir"], download=False,
                                  train_limit=50000, test_limit=10000, seed=identity["seed"])
    fraction = context["validation_fraction"]
    dataset = stratified_training_three_way_split(
        original, seed=context["split_seed"], train_fraction=0.95 - fraction,
        calibration_fraction=fraction, attack_fraction=0.05).calibration_dataset
    for key, arrays in (
        ("train_content_digest", (dataset.x_train, dataset.y_train)),
        ("test_content_digest", (dataset.x_test, dataset.y_test)),
        ("attack_auxiliary_content_digest", (dataset.x_attack, dataset.y_attack)),
    ):
        if _array_content_digest(*arrays) != identity[key]:
            raise ValueError(f"{key} differs: refusing to train on a different split")
    return dataset


def run_probe(dataset, config, output, until_round, reference, environment=None):
    output.mkdir(parents=True, exist_ok=False)
    if environment is not None:
        write_json(output / "environment.json", environment)
    malicious = set(fl._choose_malicious([f"client-{i}" for i in range(config.num_clients)],
                                        config.malicious_ratio, config.seed))
    report = {"status": "running", "config": asdict(config), "until_round": until_round,
              "prefix_metric_mismatches": [], "note": "Diagnostic only; never a replacement tuning result."}
    originals = {name: getattr(fl, name) for name in
                 ("_local_train_client_delta", "_alternating_minimization_client_delta")}

    def callback(state):
        round_id = int(state["completed_round"])
        latest = state["records"][-1]
        write_json(output / "probe_rounds.json", [asdict(r) for r in state["records"]])
        expected = reference.get(round_id)
        if expected is not None:
            fields = ("accuracy", "attack_target_success_rate", "accepted_updates", "rejected_updates",
                      "nonfinite_updates", "malicious_weight_mass")
            different = [key for key in fields if expected.get(key) not in ("", None)
                         and (getattr(latest, key) is None
                              or not math.isfinite(float(getattr(latest, key)))
                              or not math.isfinite(float(expected[key]))
                              or abs(float(expected[key]) - float(getattr(latest, key))) > 1e-7)]
            if different:
                report["prefix_metric_mismatches"].append({"round": round_id, "fields": different})
        if round_id == until_round - 1:
            with (output / f"checkpoint_round_{round_id:03d}.pickle").open("wb") as handle:
                pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        report["last_completed_round"] = round_id
        write_json(output / "report.json", report)
        print(f"probe_round={round_id} accuracy={latest.accuracy:.6f} "
              f"nonfinite={state['nonfinite_updates']}", flush=True)
        if not np.isfinite(state["params"]).all():
            report.update(status="nonfinite_global_params_at_round_boundary",
                          global_params=tensor_stats(state["params"]))
            np.save(output / "nonfinite_global_params.npy", state["params"])
            raise ProbeStop()
        if round_id >= until_round:
            report["status"] = "no_nonfinite_client_update_within_limit"
            raise ProbeStop()

    def wrapped(name):
        original = originals[name]

        def call(*args, **kwargs):
            delta, stats = original(*args, **kwargs)
            if fl._update_is_finite(delta):
                return delta, stats
            context = kwargs["torch_context"]
            params = context.to_numpy(args[0]) if context is not None else np.asarray(args[0])
            identity = f"client-{kwargs['client_idx']}"
            fault = {"round": kwargs["round_id"], "client_idx": kwargs["client_idx"],
                     "client_id": identity, "is_malicious": identity in malicious,
                     "entry_point": name, "client_samples": len(args[2]),
                     "input_params": tensor_stats(params), "delta": tensor_stats(delta)}
            np.savez(output / "fault_client_input.npz", params=params, client_indices=args[2],
                     attack_target_indices=kwargs.get("attack_target_indices", np.array([], dtype=np.int64)))
            report.update(status="nonfinite_client_update_captured", fault=fault)
            write_json(output / "report.json", report)
            print("FAULT " + json.dumps(fault, ensure_ascii=False), flush=True)
            report["client_replay"] = replay_with_trace(original, args, kwargs)
            print("REPLAY " + json.dumps(report["client_replay"], ensure_ascii=False), flush=True)
            raise ProbeStop()

        return call

    try:
        for name in originals:
            setattr(fl, name, wrapped(name))
        fl.run_experiment(dataset, config, checkpoint_callback=callback)
        report["status"] = "experiment_returned_before_diagnostic_limit"
    except ProbeStop:
        pass
    except BaseException as exc:
        report.update(status="probe_interrupted", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        for name, original in originals.items():
            setattr(fl, name, original)
        write_json(output / "report.json", report)
    return report


def main():
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, default=repo / "outputs/cifar10_v7_target_fair_tuning")
    parser.add_argument("--candidate", default="vert-001")
    parser.add_argument("--partition", default="dirichlet")
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--until-round", type=int, default=25)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Verify candidate and data; do not train.")
    args = parser.parse_args()
    config, manifest, reference = select_recorded_run(
        args.source_output, args.candidate, args.partition, args.ratio, args.seed)
    if not 1 <= args.until_round <= config.rounds:
        parser.error("until-round must be within the original training rounds")
    if not str(config.device).startswith("cuda") or not args.device.startswith("cuda"):
        parser.error("server reproduction must retain the CUDA backend")
    config = replace(config, device=args.device)
    output = (args.output_dir or repo / "outputs" /
              ("cifar10_nonfinite_probe_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    if output == args.source_output.resolve() or args.source_output.resolve() in output.parents:
        parser.error("diagnostic output must be outside the original tuning output")
    if output.exists():
        parser.error("diagnostic output already exists; choose a new directory")
    print("VERIFYING_RECORDED_DATA", flush=True)
    dataset = load_verified_data(manifest, args.data_dir)
    print("DATA_DIGESTS_MATCH candidate=" + args.candidate, flush=True)
    if args.dry_run:
        print("DRY_RUN_OK", json.dumps(asdict(config)), flush=True)
        return
    import torch
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; no CPU fallback for this reproduction")
    metadata = {"source_fingerprint": manifest["fingerprint"], "candidate_id": args.candidate,
                "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(args.device),
                "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
                "source_sha256": {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (repo / "sm9rrsfl/fl.py", repo / "sm9rrsfl/torch_backend.py",
                                            repo / "sm9rrsfl/vert.py", Path(__file__).resolve())}}
    print("PROBE_OUTPUT " + str(output), flush=True)
    report = run_probe(dataset, config, output, args.until_round, reference, environment=metadata)
    print("PROBE_STATUS " + report["status"], flush=True)
    print("PREFIX_METRIC_MISMATCHES " + str(len(report["prefix_metric_mismatches"])), flush=True)


if __name__ == "__main__":
    main()
