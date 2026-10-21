#!/usr/bin/env python3
"""Supplement the completed C7 replication with three fixed comparison methods."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import json

from run_cifar_attack_stability import read_json, write_json, digest, utc
from run_cifar_c7_replication import PROFILE, load_replication, sha, seconds_text
from scripts.experiments.cifar10.attack_stability_worker import execute_task, task_identity

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/cifar10_c7_six_methods_v1.json"
METHODS = ("alignins", "krum", "ding13")
TERMINAL = {"complete", "failed", "numerical_failure"}


def load_six_methods(path=DEFAULT_CONFIG):
    protocol = read_json(path)
    expected = {
        "protocol": "cifar_c7_six_methods_v1", "partition": "dirichlet",
        "num_clients": 100, "rounds": 30, "seed": 401, "ratios": [.6, .8],
        "sm9_workers": 8, "profile": PROFILE, "task_count": 6,
        "candidate_ids": {method: f"{method}-001" for method in METHODS},
        "target_selection": "original_per_experiment_seed",
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Configuration differs from the fixed six-method supplement")
    previous, _, data = load_replication(ROOT / protocol["source_protocol"])
    out = (ROOT / protocol["output_dir"]).resolve()
    reference = (ROOT / protocol["reference_dir"]).resolve()
    if reference != (ROOT / previous["output_dir"]).resolve():
        raise ValueError("Reference directory differs from the declared replication")
    for ancestor in (reference, (ROOT / previous["reference_dir"]).resolve()):
        if (not out.is_relative_to(ROOT / "outputs") or out == ancestor
                or out.is_relative_to(ancestor) or ancestor.is_relative_to(out)):
            raise ValueError("Supplement outputs must be separate from all references")
    from sm9rrsfl import experiments, fair_tuning
    from sm9rrsfl.config_runner import parameters_to_argv
    attack_protocol = read_json(ROOT / previous["source_protocol"])
    spec = fair_tuning.load_fair_tuning_config(ROOT / attack_protocol["source_config"])
    args = experiments.parse_args(parameters_to_argv(spec.shared_parameters))
    base = experiments.build_experiment_configs(args)[0]
    bases = {
        method: replace(base, **spec.candidates[method][0], method=method,
                        rounds=30, num_clients=100, seed=401, device="mps",
                        partition="dirichlet", sm9_workers=8)
        for method in METHODS
    }
    return protocol, bases, data


def task_specs(protocol, bases):
    return [
        {"task_id": f"C7_{method}_dir{int(ratio * 100)}_s401",
         "config": asdict(replace(bases[method], malicious_ratio=ratio)),
         "profile": dict(protocol["profile"])}
        for ratio in (.8, .6) for method in METHODS
    ]


def verify_frozen(manifest):
    for field in ("source_files", "reference_files"):
        for path, expected in manifest[field].items():
            if sha(ROOT / path) != expected:
                raise ValueError(f"Immutable {field} changed: {path}")


def reference_files(protocol):
    """No training or immutable snapshot until all eight previous tasks finish."""
    directory = ROOT / protocol["reference_dir"]
    previous = read_json(directory / "manifest.json")
    if not previous or previous.get("fingerprint") != protocol["reference_fingerprint"]:
        raise ValueError("Unexpected replication reference")
    if digest({k: v for k, v in previous.items() if k != "fingerprint"}) != previous["fingerprint"]:
        raise ValueError("Replication manifest fingerprint is corrupt")
    verify_frozen(previous)
    controller = read_json(directory / "controller_status.json", {})
    if controller.get("status") != "complete":
        raise ValueError("Replication controller must finish completely before supplementary training")
    from scripts.experiments.cifar10.cifar_c7_replication_report import build_report
    report = build_report(directory, ROOT / previous["protocol"]["reference_dir"])
    if not (report["all_complete"] and report["all_numerically_finite"]
            and report["reference_integrity_valid"] and report["all_comparisons_available"]):
        raise ValueError("All eight replication tasks and their comparisons must be valid and finite")
    specs = previous["task_specs"]
    identities = sorted((directory / "tasks").glob("*/task.json"))
    if len(specs) != 8 or {p.parent.name for p in identities} != {s["task_id"] for s in specs}:
        raise ValueError("Replication task coverage is not the exact eight-task protocol")
    references = dict(previous["reference_files"])
    paths = [directory / name for name in ("manifest.json", "controller_status.json", "report.json")]
    for spec in specs:
        task = directory / "tasks" / spec["task_id"]
        expected = task_identity({**spec, "data_config": previous["data_config"],
                                  "fingerprint": previous["fingerprint"]})
        if read_json(task / "task.json") != expected:
            raise ValueError(f"Replication task differs from its frozen specification: {task.name}")
        paths.extend(task / name for name in ("task.json", "status.json", "metrics.json",
                                              "targets.json", "dataset_identity.json", "result.json"))
        if (task / "audit.jsonl").exists():
            paths.append(task / "audit.jsonl")
    references.update({str(path.relative_to(ROOT)): sha(path) for path in paths})
    return previous, references


def build_manifest(protocol, specs, data, config_path=DEFAULT_CONFIG):
    previous, references = reference_files(protocol)
    sources = dict(previous["source_files"])
    for path in (Path(__file__).resolve(), Path(config_path).resolve(),
                 ROOT / "scripts/experiments/cifar10/cifar_c7_six_report.py"):
        sources[str(path.relative_to(ROOT))] = sha(path)
    identity = {
        "protocol": protocol, "source_files": sources, "reference_files": references,
        "task_specs": specs, "data_config": data, "backend": "torch:mps",
        "official_test_used_for_selection": False,
        "comparison": "Six fixed first-grid candidates; no tuned-SOTA or long-horizon qualification",
    }
    return {**identity, "fingerprint": digest(identity)}


class SixMethodController:
    def __init__(self, protocol, specs, data, manifest, mode):
        self.p, self.specs, self.data, self.manifest = protocol, specs, data, manifest
        self.out = (ROOT / protocol["output_dir"]).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        import fcntl
        self.lock = (self.out / "controller.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = read_json(self.out / "manifest.json")
        if old is not None and old != manifest:
            raise ValueError("Supplement manifest mismatch; preserve this directory")
        if old is None and any(p.name != "controller.lock" for p in self.out.iterdir()):
            raise ValueError("Refusing unidentified preexisting outputs")
        write_json(self.out / "manifest.json", manifest)
        self.state = read_json(self.out / "controller_status.json", {})
        self.elapsed_before = self.state.get("elapsed_seconds", 0.)
        if self.state.get("status") == "running":
            self.elapsed_before += max(0., time.time() - self.state["updated_epoch"])
        self.start = time.time()
        self.state.update(status="running", pid=os.getpid(),
                          started_utc=self.state.get("started_utc", utc()),
                          planned_tasks=len(specs),
                          planned_rounds=sum(s["config"]["rounds"] for s in specs),
                          explicit_new_time_limit_seconds=None, current=None)
        self.child = None
        self.live = mode == "live" or (mode == "auto" and sys.stdout.isatty())
        self.last_display = 0.
        self.save()

    def save(self):
        self.state.update(updated_utc=utc(), updated_epoch=time.time(),
                          elapsed_seconds=self.elapsed_before + time.time() - self.start)
        write_json(self.out / "controller_status.json", self.state)

    def display(self, force=False):
        if not force and time.time() - self.last_display < 20:
            return
        self.last_display = time.time()
        completed = rounds = 0
        elapsed_tasks = 0.
        current_line = ""
        unfinished_rounds = 0
        for spec in self.specs:
            status = read_json(self.out / "tasks" / spec["task_id"] / "status.json", {})
            done = max(0, status.get("completed_round", 0))
            rounds += done
            completed += status.get("status") == "complete"
            elapsed_tasks += status.get("elapsed_seconds", 0.)
            if status.get("status") not in TERMINAL:
                unfinished_rounds += spec["config"]["rounds"] - done
            if spec["task_id"] == self.state.get("current"):
                config = spec["config"]
                current_line = (
                    f"MPS {config['method']} seed=401 ratio={config['malicious_ratio']} "
                    f"round={done}/30 | C7 Adam lr=.001 boost=20 normal_steps<=10 weights")
        remaining = elapsed_tasks / rounds * unfinished_rounds if rounds else None
        total = self.state["planned_rounds"]
        fill = min(36, int(36 * rounds / total))
        self.state.update(completed_tasks=completed, completed_rounds=rounds,
                          estimated_remaining_seconds=remaining)
        message = (
            f"[{'#' * fill}{'-' * (36-fill)}] rounds {rounds}/{total} | "
            f"tasks {completed}/{len(self.specs)} | elapsed "
            f"{seconds_text(self.elapsed_before + time.time() - self.start)} | "
            f"ETA ~{seconds_text(remaining)} (observed rate, methods differ) | "
            f"{current_line or self.state['status']}")
        print(("\r\033[2K" if self.live else "") + message,
              end="" if self.live else "\n", flush=True)
        self.save()

    def request(self, spec):
        return {**spec, "kind": "fl", "output_dir": str(self.out / "tasks"),
                "data_config": self.data, "device": "mps",
                "fingerprint": self.manifest["fingerprint"],
                "return_path": str(self.out / "returns" / f"{spec['task_id']}.json")}

    def run_task(self, spec):
        verify_frozen(self.manifest)
        request = self.request(spec)
        directory = self.out / "tasks" / spec["task_id"]
        status = read_json(directory / "status.json")
        if status:
            if read_json(directory / "task.json") != task_identity(request):
                raise ValueError("Task identity mismatch")
            if status["status"] in TERMINAL:
                return status
        self.state["current"] = spec["task_id"]
        self.display(True)
        path = self.out / "requests" / f"{spec['task_id']}.json"
        write_json(path, request)
        returns = Path(request["return_path"])
        returns.unlink(missing_ok=True)
        log_path = self.out / "logs" / f"{spec['task_id']}.log"
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("a") as log:
            self.child = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()),
                 "--worker-request", str(path)], cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                     "OPENBLAS_NUM_THREADS": "2"})
            try:
                while self.child.poll() is None:
                    self.display()
                    time.sleep(1)
                code = self.child.returncode
            except BaseException:
                if self.child.poll() is None:
                    self.child.send_signal(signal.SIGINT)
                try:
                    self.child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.child.terminate()
                    self.child.wait(timeout=10)
                raise
            finally:
                self.child = None
        if not returns.exists():
            raise RuntimeError(f"Worker {spec['task_id']} exited {code} without return; see {log_path}")
        return read_json(returns)

    def snapshot(self):
        from scripts.experiments.cifar10.cifar_c7_six_report import build_report
        report = build_report(self.out, ROOT / self.p["reference_dir"])
        report.update(fingerprint=self.manifest["fingerprint"], updated_utc=utc(),
                      elapsed_seconds=self.elapsed_before + time.time() - self.start)
        write_json(self.out / "report.json", report)
        return report

    def run(self):
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError("This local supplement requires Apple MPS")
        report = self.snapshot()
        if not report["reference_integrity_valid"]:
            raise ValueError("Reference integrity failed; no new training started")
        for spec in self.specs:
            result = self.run_task(spec)
            self.snapshot()
            if result["status"] == "partial":
                self.state["status"] = "interrupted"
                return
        verify_frozen(self.manifest)
        report = self.snapshot()
        accepted = (report["all_complete"] and report["all_numerically_finite"]
                    and report["reference_integrity_valid"]
                    and report["all_six_method_comparisons_available"])
        # A completed but unhealthy method remains a scientific result, never a silent retry.
        self.state["status"] = "complete" if accepted else "finished_with_failures"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--display-mode", choices=["auto", "live", "log"], default="auto")
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.worker_request:
        request = read_json(options.worker_request)
        write_json(request["return_path"], execute_task(request))
        return 0
    protocol, bases, data = load_six_methods(options.config)
    specs = task_specs(protocol, bases)
    if options.summary:
        out = ROOT / protocol["output_dir"]
        print(json.dumps({"controller": read_json(out / "controller_status.json"),
                          "report": read_json(out / "report.json")}, indent=2))
        return 0
    if options.dry_run:
        # Planning is safe while the required reference run is still in flight.
        state = read_json(ROOT / protocol["reference_dir"] / "controller_status.json", {})
        ready = state.get("status") == "complete"
        manifest = build_manifest(protocol, specs, data, options.config) if ready else None
        print(json.dumps({
            "protocol": protocol, "tasks": specs, "data": data,
            "planned_tasks": len(specs), "planned_rounds": 180,
            "reference_ready": ready, "reference_status": state.get("status", "not_started"),
            "fingerprint": manifest["fingerprint"] if manifest else None,
            "source_files": len(manifest["source_files"]) if manifest else None,
            "reference_files": len(manifest["reference_files"]) if manifest else None,
            "note": "No output or immutable manifest is written during dry-run.",
        }, indent=2))
        return 0
    manifest = build_manifest(protocol, specs, data, options.config)
    controller = SixMethodController(protocol, specs, data, manifest, options.display_mode)
    old_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        controller.run()
    except KeyboardInterrupt:
        controller.state.update(status="interrupted", reason="complete checkpoints retained")
    except Exception as error:
        controller.state.update(status="failed", error_type=type(error).__name__,
                                error=str(error), traceback=traceback.format_exc())
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        controller.display(True)
        if controller.live:
            print()
        controller.save()
    return 0 if controller.state["status"] == "complete" else 2


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
