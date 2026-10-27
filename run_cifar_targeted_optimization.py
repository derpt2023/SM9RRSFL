#!/usr/bin/env python3
"""Versioned local development batches, with unchanged upstream training code.

Each JSON is immutable within its output directory. New hypotheses require a
new JSON/protocol/output. Attack selection sees FedAvg only; all results remain.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import fcntl
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from run_cifar_attack_stability import load_protocol, digest, read_json, write_json, utc

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / "configs/cifar10_target_lr_v1.json"
TERMINAL = {"complete", "failed", "numerical_failure"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan(path):
    path = Path(path).resolve()
    p = read_json(path)
    if p["schema"] != "cifar_targeted_development_v1":
        raise ValueError("Unsupported protocol")
    _, bases, data = load_protocol()
    station = read_json(ROOT / "configs/cifar10_c7_station_150r_v1.json")
    specs = []
    for item in p["tasks"]:
        name, method = item["id"], item["method"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Unsafe task id")
        base = bases.get(method, replace(bases["fedavg"], method=method))
        changes = {**station["fixed_parameters"][method], **p["common"],
                   **item.get("parameters", {}), "method": method, "device": "mps"}
        profile = p["profiles"].get(item.get("profile"))
        if profile:
            changes.update(attack_boost=profile["boost"],
                           attack_stealth_steps=profile["stealth_steps"],
                           attack_target_count=profile["target_count"])
            profile = {k: v for k, v in profile.items() if k != "target_count"}
        config = replace(base, **changes)
        if config.num_clients != 100 or config.rounds < 30:
            raise ValueError("This development series preserves 100 clients, >=30 rounds")
        if config.malicious_ratio and not profile:
            raise ValueError("Every attacked task needs an explicit attack profile")
        specs.append({"task_id": name, "config": asdict(config), "profile": profile,
                      "probe": item.get("probe", False)})
    if len(set(s["task_id"] for s in specs)) != len(specs):
        raise ValueError("Duplicate tasks")
    sources = set((ROOT / "sm9rrsfl").rglob("*.py"))
    sources.update((ROOT / "scripts/experiments/cifar10").glob("*.py"))
    sources.update([Path(__file__).resolve(), path, ROOT / "run_cifar_attack_stability.py",
        ROOT / "configs/cifar10_attack_stability_v1.json",
        ROOT / "configs/cifar10_v14_fair_grid.json",
        ROOT / "configs/cifar10_c7_station_150r_v1.json"])
    references = {f: sha(ROOT / f) for f in p.get("references", [])}
    identity = {"protocol": p, "specs": specs, "data_config": data,
                "sources": {str(f.relative_to(ROOT)): sha(f) for f in sorted(sources)},
                "references": references, "official_test_used_for_selection": False,
                "maximum_rounds": sum(s["config"]["rounds"] for s in specs)}
    return p, specs, data, {**identity, "fingerprint": digest(identity)}


def verify(manifest):
    for field in ("sources", "references"):
        for path, expected in manifest[field].items():
            if sha(ROOT / path) != expected:
                raise ValueError(f"Immutable {field} changed: {path}")


def classification_metrics(labels, predictions, source=5, classes=10):
    import numpy as np
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != predictions.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("Invalid classification observations")
    if np.any((labels < 0) | (labels >= classes) | (predictions < 0) | (predictions >= classes)):
        raise ValueError("Class index out of range")
    source_mask = labels == source
    if not np.any(source_mask) or not np.any(~source_mask):
        raise ValueError("Both source and non-source samples are required")
    matrix = np.bincount(labels * classes + predictions, minlength=classes**2).reshape(classes, classes)
    return {"non_source_accuracy": float((predictions[~source_mask] == labels[~source_mask]).mean()),
            "non_source_count": int((~source_mask).sum()),
            "source_accuracy": float((predictions[source_mask] == labels[source_mask]).mean()),
            "prediction_histogram": matrix.sum(0).tolist(),
            "confusion_true_rows_predicted_columns": matrix.tolist()}


@contextmanager
def metric_scope():
    import numpy as np
    from sm9rrsfl import fl
    from scripts.experiments.cifar10 import attack_stability_worker as worker
    original = worker.RoundMetricEvaluator

    class Evaluator(original):
        def __init__(self, dataset, config):
            super().__init__(dataset, config)
            self.paired = {n: fl._select_attack_target_indices(dataset,
                replace(config, attack_target_count=n), required=True) for n in (1, 128)}

        def target_manifest(self):
            return {**super().target_manifest(), "paired_auxiliary_targets": {
                str(n): x.tolist() for n, x in self.paired.items()},
                "non_source_count": int((self.dataset.y_test != self.config.attack_source_label).sum())}

        def evaluate(self, state):
            row = super().evaluate(state)
            for n, indices in self.paired.items():
                hit, confidence = self._target_metrics(state["params"], "attack", indices)
                row[f"aux{n}_success"], row[f"aux{n}_confidence"] = hit, confidence
            from sm9rrsfl.torch_backend import _torch_forward, _torch_params_from_tensor
            ctx = self.torch_context
            if ctx is None:
                raise ValueError("Classification observation requires torch context")
            torch = ctx.torch
            params = _torch_params_from_tensor(torch, ctx._ensure_global_vector(state["params"]),
                self.spec, requires_grad=False, clone=False)
            predictions = []
            with torch.no_grad():
                for start in range(0, len(self.dataset.y_test), 512):
                    logits = _torch_forward(torch, params, ctx.x_test[start:start + 512], self.spec)
                    worker._check_finite(logits, "full_classification_logits", round=row["round"])
                    predictions.append(logits.argmax(1).cpu().numpy())
            row.update(classification_metrics(self.dataset.y_test, np.concatenate(predictions),
                                             self.config.attack_source_label))
            matrix = np.asarray(row["confusion_true_rows_predicted_columns"])
            if abs(float(matrix.trace() / matrix.sum()) - row["accuracy"]) > 1e-7:
                raise ValueError("Full classification observations disagree with original accuracy")
            return row

    worker.RoundMetricEvaluator = Evaluator
    try:
        yield
    finally:
        worker.RoundMetricEvaluator = original


def worker(request):
    from scripts.experiments.cifar10.c7_alignment_attack import alignment_scope
    with alignment_scope(), metric_scope():
        if request["kind"] == "probe":
            from scripts.experiments.cifar10.attack_stability_probes import run_probe
            return run_probe(request)
        from scripts.experiments.cifar10.attack_stability_worker import execute_task
        return execute_task(request)


def mean(rows, key, window):
    chosen = [r for r in rows if window[0] <= r["round"] <= window[1]]
    if [r["round"] for r in chosen] != list(range(window[0], window[1] + 1)):
        raise ValueError("Incomplete evaluation window")
    return sum(float(r[key]) for r in chosen) / len(chosen)


def attack_evidence(rows, clean, policy, count):
    window = policy["window"]
    evidence = {
        "auxiliary_asr": mean(rows, "training_target_success", window),
        "auxiliary_confidence_gain": mean(rows, "training_target_confidence", window)
            - mean(clean, f"aux{count}_confidence", window),
        "clean_auxiliary_confidence": mean(clean, f"aux{count}_confidence", window),
        "transfer_asr": mean(rows, "source_transfer_asr", window),
        "transfer_gain": mean(rows, "source_transfer_asr", window) - mean(clean, "source_transfer_asr", window),
        "non_source_accuracy_drop": mean(clean, "non_source_accuracy", window) - mean(rows, "non_source_accuracy", window),
        "accuracy": mean(rows, "accuracy", window)}
    checks = {"auxiliary_effect": evidence["auxiliary_asr"] >= policy["min_auxiliary_asr"],
        "confidence_gain": evidence["auxiliary_confidence_gain"] >= policy["min_confidence_gain"],
        "clean_not_saturated": evidence["clean_auxiliary_confidence"] < policy["max_clean_confidence"],
        "non_source_utility": evidence["non_source_accuracy_drop"] <= policy["max_non_source_accuracy_drop"],
        "independent_transfer": evidence["transfer_gain"] >= policy["min_transfer_gain"]}
    return {**evidence, "checks": checks, "eligible": all(checks.values())}


def summarize(out, p, specs):
    result = {"protocol": p["protocol"], "tasks": {}, "selected_attack": None,
              "defense_outcomes_used_for_attack_selection": False}
    for spec in specs:
        ident = spec["task_id"]
        status = read_json(out / "tasks" / ident / "status.json", {})
        item = {"status": status.get("status", "not_run"),
                "completed_round": status.get("completed_round", 0),
                "nonfinite_updates": status.get("nonfinite_updates"),
                "elapsed_seconds": status.get("elapsed_seconds")}
        if status.get("status") == "complete":
            rows = read_json(out / "tasks" / ident / "metrics.json")["rounds"]
            tail = [spec["config"]["rounds"] - 9, spec["config"]["rounds"]]
            item["tail"] = {key: mean(rows, key, tail) for key in (
                "accuracy", "training_target_success", "source_transfer_asr", "non_source_accuracy")}
            selection = p.get("attack_selection")
            if selection and ident in selection["candidates"]:
                if spec["config"]["method"] != "fedavg":
                    raise ValueError("Only FedAvg can select attack")
                clean = read_json(out / "tasks" / selection["clean_task"] / "metrics.json")["rounds"]
                item["attack_evidence"] = attack_evidence(rows, clean, selection,
                    spec["config"]["attack_target_count"])
                item["attack_evidence"]["eligible"] &= status.get("nonfinite_updates") == 0
        result["tasks"][ident] = item
    eligible = [(name, item["attack_evidence"]) for name, item in result["tasks"].items()
                if item.get("attack_evidence", {}).get("eligible")]
    if eligible:
        result["selected_attack"] = max(eligible, key=lambda t: (
            t[1]["transfer_asr"], t[1]["auxiliary_asr"], t[1]["auxiliary_confidence_gain"]))[0]
    write_json(out / "summary.json", result)
    return result


class Controller:
    def __init__(self, p, specs, data, manifest):
        self.p, self.specs, self.data, self.manifest = p, specs, data, manifest
        self.out = ROOT / p["output_dir"]
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out / "controller.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = read_json(self.out / "manifest.json")
        if old is not None and old != manifest:
            raise ValueError("Protocol/source changed; use an independent output")
        if old is None and any(f.name != "controller.lock" for f in self.out.iterdir()):
            raise ValueError("Unidentified output")
        write_json(self.out / "manifest.json", manifest)
        self.start = time.time()

    def request(self, spec):
        return {**spec, "kind": "train", "device": "mps", "data_config": self.data,
                "output_dir": str(self.out / "tasks"), "fingerprint": self.manifest["fingerprint"]}

    def execute(self, request):
        verify(self.manifest)
        name = request["task_id"]
        probe = request["kind"] == "probe"
        statepath = self.out / ("probes" if probe else "tasks") / name / ("result.json" if probe else "status.json")
        old = read_json(statepath, {})
        if old.get("status") in TERMINAL:
            return old
        reqpath = self.out / "requests" / f"{name}.json"
        write_json(reqpath, request)
        logpath = self.out / "logs" / f"{name}.log"
        logpath.parent.mkdir(exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        with logpath.open("a") as log:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                "--worker", str(reqpath)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                last = 0
                while child.poll() is None:
                    if time.time() - last >= 20:
                        last = time.time()
                        state = read_json(statepath, {})
                        current = max(0, state.get("completed_clients" if probe else "completed_round", 0))
                        total = len(request["clients"]) if probe else request["config"]["rounds"]
                        statuses = [read_json(f) for f in (self.out / "tasks").glob("*/status.json")]
                        rounds = sum(max(0, s.get("completed_round", 0)) for s in statuses)
                        elapsed = time.time() - self.start
                        duration = state.get("elapsed_seconds", state.get("runtime_seconds", 0))
                        eta = duration / current * (total - current) if current else None
                        progress = {"status": "running", "current": name, "device": request["device"],
                            "completed_rounds": rounds, "maximum_rounds": self.manifest["maximum_rounds"],
                            "current_progress": current, "current_total": total,
                            "elapsed_seconds_this_invocation": elapsed, "current_eta_seconds": eta,
                            "updated_utc": utc(), "pid": os.getpid()}
                        write_json(self.out / "controller_status.json", progress)
                        print(f"{rounds}/{self.manifest['maximum_rounds']} rounds | {name} "
                              f"{request['device']} {current}/{total} | elapsed={elapsed/60:.1f}m "
                              f"task_ETA={None if eta is None else round(eta/60, 1)}m", flush=True)
                    time.sleep(1)
            except BaseException:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=15)
                raise
        result = read_json(statepath, {})
        if result.get("status") not in TERMINAL:
            raise RuntimeError(f"Worker not terminal; checkpoint retained. Inspect {logpath}")
        print(f"FINISH {name}: {result['status']}", flush=True)
        return result

    def probe(self, spec):
        from sm9rrsfl import fl
        from sm9rrsfl.datasets import partition_clients
        from scripts.experiments.cifar10.attack_stability_worker import _load_dataset
        from scripts.experiments.cifar10.attack_stability_probes import select_representatives
        dataset, _ = _load_dataset(self.data)
        config = fl.ExperimentConfig(**spec["config"])
        if config.attack_start_round != 25:
            raise ValueError("Existing local probes use round 25; different timing needs new probes")
        partitions = partition_clients(dataset.y_train, config.num_clients,
            strategy=config.partition, dirichlet_alpha=config.dirichlet_alpha, seed=config.seed)
        malicious = sorted(int(x.split("-")[1]) for x in fl._choose_malicious(
            [f"client-{i}" for i in range(config.num_clients)], config.malicious_ratio, config.seed))
        clients = select_representatives(dataset.y_train, partitions, malicious, count=16)
        write_json(self.out / "representatives" / f"{spec['task_id']}.json", {
            "rule": "Data-only quantiles, extremes and source-label presence; fixed before outcomes",
            "malicious_clients": malicious, "clients": [{"index": i, "samples": len(partitions[i]),
            "source_samples": int((dataset.y_train[partitions[i]] == 5).sum())} for i in clients]})
        del dataset, partitions
        clean = self.p["probe_clean_task"]
        for device, indices in (("mps", clients), ("cpu", clients[:2])):
            for anchor in (0, 24):
                req = {**self.request(spec), "kind": "probe", "device": device,
                    "task_id": f"{spec['task_id']}_{device}_r{anchor}", "clients": indices,
                    "outdir": str(self.out / "probes"), "anchor_id": f"{clean}_r{anchor}",
                    "anchor_path": str(self.out / "tasks" / clean / f"anchor_r{anchor}.npz")}
                result = self.execute(req)
                if not result.get("all_finite"):
                    return False
        return True

    def run(self):
        for spec in self.specs:
            if spec["probe"] and not self.probe(spec):
                write_json(self.out / "screen_rejections" / (spec["task_id"] + ".json"),
                    {"reason": "nonfinite_or_incomplete_probe", "task": spec["task_id"]})
                continue
            status = self.execute(self.request(spec))
            if spec["task_id"] == self.p.get("probe_clean_task") and status["status"] != "complete":
                raise RuntimeError("Required clean reference failed")
            summarize(self.out, self.p, self.specs)
        verify(self.manifest)
        result = summarize(self.out, self.p, self.specs)
        write_json(self.out / "controller_status.json", {"status": "batch_complete",
            "updated_utc": utc(), "elapsed_seconds_this_invocation": time.time()-self.start,
            "selected_attack": result["selected_attack"]})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(read_json(args.worker))
        return
    p, specs, data, manifest = plan(args.config)
    import json
    if args.dry_run:
        print(json.dumps({"protocol": p["protocol"], "tasks": [s["task_id"] for s in specs],
                          "maximum_rounds": manifest["maximum_rounds"], "fingerprint": manifest["fingerprint"]}))
        return
    if args.summary:
        print(json.dumps(summarize(ROOT / p["output_dir"], p, specs), indent=2))
        return
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("Local MPS device unavailable")
    print(json.dumps(Controller(p, specs, data, manifest).run(), indent=2))


if __name__ == "__main__":
    main()
