#!/usr/bin/env python3
"""Clean-only baseline calibration followed by an untouched-seed C7 panel."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from run_cifar_attack_stability import digest, read_json, utc, write_json
from run_cifar_c7_replication import PROFILE, load_replication, seconds_text, sha
from run_cifar_c7_six_methods import load_six_methods, verify_frozen
from scripts.experiments.cifar10.attack_stability_worker import execute_task, task_identity

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/cifar10_c7_calibrated_60r_v1.json"
REPORT_SOURCE = ROOT / "scripts/experiments/cifar10/c7_calibrated_report.py"
METHODS = ("fedavg", "sm9rrs", "vert", "alignins", "krum", "ding13")
TERMINAL = {"complete", "failed", "numerical_failure"}
SELECTION_RULE = {
    "max_final_clean_accuracy_drop": .05,
    "max_last10_clean_accuracy_drop": .05,
    "max_clean_false_revocation_rate": .1,
    "require_every_training_round_nonempty_aggregation": True,
    "require_complete_and_finite": True,
    "ranking": ["last10_accuracy_desc", "final_accuracy_desc", "candidate_id_asc"],
    "no_eligible_candidate": "retain_best_complete_finite_as_unqualified_stress_comparison",
    "official_test_used": False, "attack_outcomes_used": False,
}


def load_plan(path=DEFAULT_CONFIG):
    p = read_json(path)
    expected = {
        "protocol": "cifar_c7_calibrated_60r_v1", "calibration_seed": 301,
        "calibration_rounds": 30, "final_seed": 402, "final_rounds": 60,
        "num_clients": 100, "partition": "dirichlet", "ratios": [.6, .8],
        "sm9_workers": 8, "profile": PROFILE,
        "final_clean_methods": list(METHODS), "final_attack_methods": list(METHODS),
        "target_selection": "original_per_experiment_seed",
        "selection": SELECTION_RULE,
    }
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("Unexpected independent calibration/60-round protocol")
    output = (ROOT / p["output_dir"]).resolve()
    for path in (p["reference_dir"], "outputs/cifar10_c7_replication_v1",
                 "outputs/cifar10_attack_stability_v1"):
        reference = (ROOT / path).resolve()
        if (not output.is_relative_to(ROOT / "outputs") or output == reference
                or output.is_relative_to(reference) or reference.is_relative_to(output)):
            raise ValueError("Output must be independent of old reference directories")
    _, base3, data = load_replication()
    _, extra3, data2 = load_six_methods(ROOT / p["source_protocol"])
    if data != data2:
        raise ValueError("Source protocols disagree on data")
    bases = {**base3, **extra3}
    for method in METHODS:
        bases[method] = replace(bases[method], partition=p["partition"],
                                num_clients=100, sm9_workers=8, device="mps")
    candidates = p["calibration_candidates"]
    if set(candidates) != {"vert", "alignins"}:
        raise ValueError("Only VERT and AlignIns are calibrated")
    if any(len(rows) != 4 for rows in candidates.values()):
        raise ValueError("Each calibrated method must have four candidates")
    ids = [row["id"] for rows in candidates.values() for row in rows]
    if len(set(ids)) != 8:
        raise ValueError("Candidate identifiers must be unique")
    for method, rows in candidates.items():
        for row in rows:
            allowed = {k for k in asdict(bases[method]) if k.startswith(method + "_")}
            if not set(row["overrides"]).issubset(allowed):
                raise ValueError("Calibration may only change its own defense parameters")
            if method == "vert" and (row["overrides"]["vert_use_ratio_prior"]
                                      or row["overrides"]["vert_top_k"] not in (10, 20)):
                raise ValueError("VERT calibration must use fixed Top-k10/20 without ratio prior")
    return p, bases, data


def calibration_specs(p, bases):
    return [
        {"task_id": f"clean_{method}_{row['id']}_s301", "stage": "calibration",
         "candidate_id": row["id"],
         "config": asdict(replace(bases[method], **row["overrides"],
                                 seed=301, rounds=30, malicious_ratio=0.)),
         "profile": None}
        for method in ("vert", "alignins")
        for row in p["calibration_candidates"][method]
    ]


def final_specs(p, bases, selection):
    chosen = dict(bases)
    for method in ("vert", "alignins"):
        chosen[method] = replace(chosen[method], **selection["selected"][method]["overrides"])
    rows = []
    # Finish matched clean before attacks; no selection is made using seed402.
    for ratio in (0., .8, .6):
        for method in METHODS:
            candidate = (selection["selected"][method]["candidate_id"]
                         if method in selection["selected"] else f"{method}-001")
            rows.append({
                "task_id": (f"clean_{method}_s402" if ratio == 0. else
                            f"C7_{method}_dir{int(ratio * 100)}_s402"),
                "stage": "final", "candidate_id": candidate,
                "config": asdict(replace(chosen[method], seed=402, rounds=60,
                                        malicious_ratio=ratio)),
                "profile": dict(PROFILE) if ratio else None,
            })
    return rows


def frozen_foundation(p, config_path):
    directory = ROOT / p["reference_dir"]
    previous = read_json(directory / "manifest.json")
    if (previous["fingerprint"] != p["reference_fingerprint"]
            or digest({k: v for k, v in previous.items() if k != "fingerprint"})
            != previous["fingerprint"]):
        raise ValueError("Old six-method manifest mismatch")
    verify_frozen(previous)
    if read_json(directory / "controller_status.json")["status"] != "complete":
        raise ValueError("Old six-method experiment is incomplete")
    from scripts.experiments.cifar10.cifar_c7_six_report import build_report
    old_report = build_report(directory, ROOT / previous["protocol"]["reference_dir"])
    if not (old_report["all_complete"] and old_report["all_numerically_finite"]
            and old_report["reference_integrity_valid"]
            and old_report["all_six_method_comparisons_available"]):
        raise ValueError("Old six-method scientific evidence is invalid")
    sources = dict(previous["source_files"])
    for file in (Path(__file__).resolve(), Path(config_path).resolve(), REPORT_SOURCE):
        sources[str(file.relative_to(ROOT))] = sha(file)
    references = dict(previous["reference_files"])
    files = [directory / name for name in
             ("manifest.json", "controller_status.json", "report.json")]
    for task in sorted((directory / "tasks").iterdir()):
        if task.is_dir():
            files.extend(task / name for name in
                         ("task.json", "status.json", "metrics.json", "targets.json",
                          "dataset_identity.json", "result.json", "audit.jsonl"))
    files.extend(ROOT / path for path in p["diagnostic_references"])
    # Include the paired seed301 FedAvg baseline explicitly.
    files.extend(ROOT / p["clean_reference"] / name for name in
                 ("task.json", "status.json", "metrics.json", "targets.json",
                  "dataset_identity.json", "result.json"))
    for file in files:
        references[str(file.relative_to(ROOT))] = sha(file)
    return sources, references


def stage_manifest(p, specs, data, stage, foundation, extra_references=None):
    sources, references = foundation
    identity = {
        "protocol": p, "stage": stage, "task_specs": specs, "data_config": data,
        "source_files": sources,
        "reference_files": {**references, **(extra_references or {})},
        "backend": "torch:mps", "official_test_used_for_selection": False,
        "selection_uses_attack_outcomes": False,
        "qualification": "local 60-round study; not 150-round or tuned-SOTA certification",
    }
    return {**identity, "fingerprint": digest(identity)}


def clean_measure(directory, spec, fingerprint):
    """Selection reads clean accuracy/health only, never an attack metric."""
    if spec["config"]["malicious_ratio"] != 0. or spec.get("profile") is not None:
        raise ValueError("Only clean tasks may inform calibration selection")
    from scripts.experiments.cifar10.c7_calibrated_report import task_summary
    summary = task_summary(directory, spec, fingerprint)
    if not summary["valid_complete"] or not summary["numerically_finite"]:
        return {"valid": False, "errors": summary.get("integrity_errors", [])}
    raw = read_json(Path(directory) / "tasks" / spec["task_id"] / "metrics.json")
    metrics = raw["rounds"]
    training = [r for r in metrics if int(r["round"]) > 0]
    tail = training[-10:]
    def maximum(key):
        return max(float(r[key]) for r in training)
    return {
        "valid": True, "final_accuracy": float(training[-1]["accuracy"]),
        "last10_accuracy": sum(float(r["accuracy"]) for r in tail) / 10,
        "zero_aggregation_rounds": [int(r["round"]) for r in training
                                    if int(r["accepted_updates"]) == 0],
        "max_false_positives": maximum("false_positive_revocations"),
        "max_blacklist_size": maximum("blacklisted_clients"),
        "max_nonfinite_updates": maximum("cumulative_nonfinite_updates"),
    }


def select_candidates(p, records, reference):
    """Deterministic prespecified clean-only selection; failures remain visible."""
    if p["selection"] != SELECTION_RULE:
        raise ValueError("Selection rule differs from the implemented frozen policy")
    selected, evidence = {}, {}
    for method, candidates in p["calibration_candidates"].items():
        rows = []
        for candidate in candidates:
            measure = records[candidate["id"]]
            reasons = []
            if not measure["valid"]:
                reasons.append("incomplete_or_nonfinite_or_invalid")
            else:
                if reference["final_accuracy"] - measure["final_accuracy"] > .05 + 1e-12:
                    reasons.append("final_clean_accuracy_drop")
                if reference["last10_accuracy"] - measure["last10_accuracy"] > .05 + 1e-12:
                    reasons.append("last10_clean_accuracy_drop")
                if measure["zero_aggregation_rounds"]:
                    reasons.append("empty_aggregation")
                if measure["max_false_positives"] / 100 > .1:
                    reasons.append("clean_false_revocation_rate")
                if measure["max_blacklist_size"] >= 100:
                    reasons.append("all_clients_revoked")
                if measure["max_nonfinite_updates"]:
                    reasons.append("nonfinite_updates")
            rows.append({**candidate, **measure, "eligible": not reasons, "failure_reasons": reasons})
        viable = [row for row in rows if row["eligible"]]
        fallback = not viable
        if fallback:
            viable = [row for row in rows if row["valid"]]
        if not viable:
            raise ValueError(f"No complete finite clean candidate for {method}; final stage blocked")
        best = sorted(viable, key=lambda r: (-r["last10_accuracy"], -r["final_accuracy"], r["id"]))[0]
        selected[method] = {
            "candidate_id": best["id"], "overrides": best["overrides"],
            "clean_qualified": not fallback, "failure_reasons": best["failure_reasons"],
        }
        evidence[method] = rows
    return {
        "protocol": p["protocol"], "calibration_seed": 301, "final_seed": 402,
        "selection_rule": p["selection"], "reference_clean": reference,
        "selected": selected, "candidate_evidence": evidence,
        "attack_outcomes_used": False, "final_seed_results_used": False,
    }


class StageController:
    def __init__(self, directory, manifest, mode):
        self.out, self.manifest = Path(directory), manifest
        self.specs, self.data = manifest["task_specs"], manifest["data_config"]
        self.out.mkdir(parents=True, exist_ok=True)
        old = read_json(self.out / "manifest.json")
        if old is not None and old != manifest:
            raise ValueError("Frozen stage identity changed; preserve outputs")
        if old is None:
            if (self.out / "tasks").exists():
                raise ValueError("Task files exist without a stage identity")
            write_json(self.out / "manifest.json", manifest)
        self.state = read_json(self.out / "controller_status.json", {})
        self.elapsed_before = float(self.state.get("elapsed_seconds", 0.))
        self.start = time.time()
        self.child = None
        self.last_display = 0.
        self.live = mode == "live" or (mode == "auto" and sys.stdout.isatty())
        self.state.update(status="running", pid=os.getpid(), current=None,
                          planned_tasks=len(self.specs),
                          planned_rounds=sum(s["config"]["rounds"] for s in self.specs),
                          started_utc=self.state.get("started_utc", utc()))

    def save(self):
        self.state.update(updated_utc=utc(), updated_epoch=time.time(),
                          elapsed_seconds=self.elapsed_before + time.time() - self.start)
        write_json(self.out / "controller_status.json", self.state)

    def display(self, force=False):
        if not force and time.time() - self.last_display < 30:
            return
        self.last_display = time.time()
        rounds = complete = 0
        remaining = 0.
        current = ""
        base_rates = {"fedavg": 11, "sm9rrs": 22, "vert": 30,
                      "alignins": 12, "krum": 11, "ding13": 12}
        for spec in self.specs:
            status = read_json(self.out / "tasks" / spec["task_id"] / "status.json", {})
            done = max(0, int(status.get("completed_round", 0)))
            rounds += done
            complete += status.get("status") == "complete"
            c = spec["config"]
            if status.get("status") not in TERMINAL:
                used = status.get("elapsed_seconds", 0)
                rate = used / done if done > 2 else base_rates[c["method"]]
                remaining += rate * (c["rounds"] - done)
            if spec["task_id"] == self.state.get("current"):
                current = (f"MPS {c['method']} {spec['candidate_id']} seed={c['seed']} "
                           f"ratio={c['malicious_ratio']} round={done}/{c['rounds']}")
        total = self.state["planned_rounds"]
        fill = min(36, int(36 * rounds / total))
        self.state.update(completed_tasks=complete, completed_rounds=rounds,
                          estimated_remaining_seconds=remaining)
        message = (f"{self.manifest['stage']} [{'#' * fill}{'-' * (36-fill)}] "
                   f"rounds {rounds}/{total} tasks {complete}/{len(self.specs)} "
                   f"elapsed {seconds_text(self.elapsed_before + time.time()-self.start)} "
                   f"ETA ~{seconds_text(remaining)} | {current or self.state['status']}")
        print(("\r\033[2K" if self.live else "") + message,
              end="" if self.live else "\n", flush=True)
        self.save()

    def snapshot(self):
        from scripts.experiments.cifar10.c7_calibrated_report import build_report
        report = build_report(self.out)
        write_json(self.out / "report.json", report)
        return report

    def run_task(self, spec):
        verify_frozen(self.manifest)
        request = {
            **spec, "kind": "fl", "data_config": self.data, "device": "mps",
            "output_dir": str(self.out / "tasks"),
            "fingerprint": self.manifest["fingerprint"],
            "return_path": str(self.out / "returns" / f"{spec['task_id']}.json"),
        }
        task = self.out / "tasks" / spec["task_id"]
        old = read_json(task / "status.json")
        if old:
            if read_json(task / "task.json") != task_identity(request):
                raise ValueError("Existing task identity mismatch")
            if old["status"] in TERMINAL:
                return old
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
            raise RuntimeError(f"Worker {spec['task_id']} exit {code} without return; see {log_path}")
        return read_json(returns)

    def run(self):
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError("This local protocol requires MPS")
        try:
            for spec in self.specs:
                result = self.run_task(spec)
                self.snapshot()
                if result["status"] == "partial":
                    self.state["status"] = "interrupted"
                    return
            verify_frozen(self.manifest)
            report = self.snapshot()
            if stage_ready(self.out, self.manifest, report):
                self.state["status"] = ("complete" if report["all_complete"]
                                        and report["all_numerically_finite"]
                                        else "complete_with_rejected_candidates")
            else:
                self.state["status"] = "finished_with_failures"
        except KeyboardInterrupt:
            self.state.update(status="interrupted", reason="complete checkpoints retained")
        except Exception as error:
            self.state.update(status="failed", error=str(error),
                              error_type=type(error).__name__, traceback=traceback.format_exc())
        finally:
            self.display(True)
            if self.live:
                print()


def stage_ready(directory, manifest, report):
    if not report["reference_integrity_valid"]:
        return False
    if manifest["stage"] != "calibration":
        return report["all_complete"] and report["all_numerically_finite"]
    # Explicit candidate failures are selection evidence, not a reason to retry.
    for spec in manifest["task_specs"]:
        status = read_json(Path(directory) / "tasks" / spec["task_id"] / "status.json", {})
        if status.get("status") not in TERMINAL:
            return False
        if report["tasks"][spec["task_id"]].get("integrity_errors"):
            return False
        if status["status"] == "complete" and not report["tasks"][spec["task_id"]]["valid_complete"]:
            return False
    return True


def run_stage(directory, manifest, mode):
    state = read_json(Path(directory) / "controller_status.json", {})
    if state.get("status") in ("complete", "complete_with_rejected_candidates"):
        if read_json(Path(directory) / "manifest.json") != manifest:
            raise ValueError("Completed stage identity mismatch")
        verify_frozen(manifest)
        from scripts.experiments.cifar10.c7_calibrated_report import build_report
        report = build_report(directory)
        if not stage_ready(directory, manifest, report):
            raise ValueError("Completed stage no longer has valid complete evidence")
        return True
    controller = StageController(directory, manifest, mode)
    controller.run()
    return controller.state["status"] in ("complete", "complete_with_rejected_candidates")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--display-mode", choices=("auto", "live", "log"), default="auto")
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_request:
        request = read_json(args.worker_request)
        write_json(request["return_path"], execute_task(request))
        return 0
    p, bases, data = load_plan(args.config)
    out = ROOT / p["output_dir"]
    if args.summary:
        print(json.dumps({"selection": read_json(out / "selection.json"),
                          **{stage: {"state": read_json(out / stage / "controller_status.json"),
                                     "report": read_json(out / stage / "report.json")}
                             for stage in ("calibration", "final")}}, indent=2))
        return 0
    foundation = frozen_foundation(p, args.config)
    cal_specs = calibration_specs(p, bases)
    calibration = stage_manifest(p, cal_specs, data, "calibration", foundation)
    if args.dry_run:
        print(json.dumps({"protocol": p, "calibration_specs": cal_specs,
                          "sources": len(foundation[0]), "references": len(foundation[1]),
                          "calibration_fingerprint": calibration["fingerprint"],
                          "planned_tasks": 26, "planned_rounds": 1320,
                          "final_seed_used_only_after_selection": True}, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    with (out / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not run_stage(out / "calibration", calibration, args.display_mode):
            return 2
        records = {spec["candidate_id"]: clean_measure(
            out / "calibration", spec, calibration["fingerprint"]) for spec in cal_specs}
        # The reference was already validated and frozen by the original screen.
        raw = read_json(ROOT / p["clean_reference"] / "metrics.json")["rounds"]
        reference = {"task": p["clean_reference"], "seed": 301,
                     "final_accuracy": raw[-1]["accuracy"],
                     "last10_accuracy": sum(row["accuracy"] for row in raw[-10:]) / 10}
        selection = select_candidates(p, records, reference)
        old = read_json(out / "selection.json")
        if old is not None and old != selection:
            raise ValueError("Frozen selection mismatch; never reselect using final outcomes")
        if old is None:
            write_json(out / "selection.json", selection)
        print("FROZEN_CLEAN_SELECTION " + json.dumps(selection["selected"], sort_keys=True), flush=True)
        frozen = {str((out / "selection.json").relative_to(ROOT)): sha(out / "selection.json")}
        for file in sorted((out / "calibration").rglob("*.json")):
            # Volatile worker requests/returns/logs are not scientific references.
            if file.parent.name in ("requests", "returns"):
                continue
            frozen[str(file.relative_to(ROOT))] = sha(file)
        final = stage_manifest(p, final_specs(p, bases, selection), data, "final", foundation, frozen)
        return 0 if run_stage(out / "final", final, args.display_mode) else 2


if __name__ == "__main__":
    os.chdir(ROOT)
    previous_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        raise SystemExit(main())
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
