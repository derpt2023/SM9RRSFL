#!/usr/bin/env python3
"""Run fixed v14 candidates on the attacked training-holdout scenarios."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from sm9rrsfl import experiments, fair_tuning
from sm9rrsfl.config_runner import parameters_to_argv
from sm9rrsfl.datasets import load_image_dataset, stratified_training_three_way_split
from sm9rrsfl.ours_calibration import split_metadata, validate_targeted_split
from sm9rrsfl.performance_target import evaluate_target, scenario_key
from scripts.experiments.cifar10.fair_tuning_dashboard import dashboard, plan

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/cifar10_v14_attack_screen.json"


def load_screen(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    if payload["protocol"] != "cifar_v14_fixed_attack_screen_v1":
        raise ValueError("Unsupported screen protocol")
    source = ROOT / payload["source_config"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != payload["source_config_sha256"]:
        raise ValueError("Source v14 configuration changed; review the screen configuration")
    original = fair_tuning.load_fair_tuning_config(source)
    shared = dict(original.shared_parameters)
    shared.update(rounds=payload["rounds"], num_clients=payload["num_clients"],
                  ratios=payload["ratios"], output_dir=payload["output_dir"])
    if shared["ratios"] != [0.6, 0.8]:
        raise ValueError("This screen is restricted to ratios 0.6 and 0.8")
    if set(payload["candidate_ids"]) != {"sm9rrs", "vert", "fedavg"}:
        raise ValueError("The screen requires Ours, VERT and FedAvg")
    args = experiments.parse_args(parameters_to_argv(shared))
    selected = {}
    for method, candidate_id in payload["candidate_ids"].items():
        ids = {f"{method}-{i:03d}": p for i, p in enumerate(original.candidates[method], 1)}
        if candidate_id not in ids:
            raise ValueError(f"Unknown source candidate: {candidate_id}")
        selected[method] = ids[candidate_id]
    # This specification is only for display/task construction, not grid selection.
    screen = replace(
        original, shared_parameters=shared,
        candidates={m: (selected[m],) if m in selected else () for m in original.candidates},
        validation_seeds=(payload["seed"],), run_final_evaluation=False,
        formal_ratios=tuple(payload["ratios"]), calibration_ratios=tuple(payload["ratios"]),
    )
    _, tasks, work = plan(screen)
    tasks = [replace(t, candidate_id=payload["candidate_ids"][t.method]) for t in tasks]
    # Finish the hardest paired scenario first, so feedback need not wait for all 12 runs.
    order = {"sm9rrs": 0, "vert": 1, "fedavg": 2}
    tasks.sort(key=lambda t: (t.config.partition != "dirichlet", -t.config.malicious_ratio,
                             order[t.method]))
    return payload, screen, args, tasks, work


def screen_report(tasks, executions, policy):
    results = [r for _, r in executions]
    expected = {scenario_key(t) for t in tasks}
    expected_runs = {(t.method, scenario_key(t)) for t in tasks}
    actual_runs = [(r.config.method, scenario_key(r)) for r in results]
    complete = (len(actual_runs) == len(expected_runs) and set(actual_runs) == expected_runs)
    audit = evaluate_target(
        [r for r in results if r.config.method == "sm9rrs"],
        [r for r in results if r.config.method == "vert"], policy,
        expected_scenarios=expected,
    )
    health = [
        {"method": r.config.method, "scenario": list(scenario_key(r)),
         "reasons": list(fair_tuning._training_health_reasons(r))}
        for r in results if fair_tuning._training_health_reasons(r)
    ]
    execution_bad = [
        {"method": r.config.method, "scenario": list(scenario_key(r))}
        for r in results if r.stopped_round != r.config.rounds or r.nonfinite_updates
        or [x.round for x in r.records if x.round > 0] != list(range(1, r.config.rounds + 1))
    ]
    attacked_pass = (
        complete and not health and not execution_bad
        and audit["failures"] == ["missing_clean_or_attacked_scenarios"]
        and len(audit["scenarios"]) == len(expected)
        and not any(r["failures"] for r in audit["scenarios"])
    )
    return {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed_tasks": len(results), "planned_tasks": len(tasks),
        "run_dimensions": {
            name: sorted({getattr(t.config, name) for t in tasks})
            for name in ("num_clients", "rounds", "seed")
        },
        "numerical_audit": [
            {"method": r.config.method, "scenario": list(scenario_key(r)),
             "nonfinite_updates": r.nonfinite_updates,
             "first_nonfinite_round": next(
                 (x.round for x in r.records if x.nonfinite_updates), None),
             "max_nonfinite_updates_per_round": max(
                 (x.nonfinite_updates for x in r.records), default=0)}
            for r in results
        ],
        "attack_screen_status": ("passed" if attacked_pass else "unmet") if complete else "pending",
        "main_experiment_target_status": "not_evaluated",
        "interpretation": (
            "Fixed source candidates; training holdout only; no tuning or official-test selection. "
            "No clean-control qualification. A reduced-horizon single-seed MPS screen "
            "does not certify or disprove the 150-round/100-client CUDA grid."
        ),
        "health_failures": health, "execution_failures": execution_bad,
        "original_target_audit": audit,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args(argv)
    payload, spec, args, tasks, work = load_screen(options.config)
    print("screen_plan=" + json.dumps({"work": work, "protocol": payload,
          "configurations": [{"candidate_id": t.candidate_id, "config": vars(t.config)}
                             for t in tasks]}, sort_keys=True), flush=True)
    if options.dry_run:
        return
    from sm9rrsfl.model import describe_compute_backend
    backend = describe_compute_backend(args.compute_backend, args.device)
    if backend != "torch:mps":
        raise RuntimeError(f"This local screen requires Apple MPS; resolved {backend}")
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"status": "running", "pid": os.getpid(), "backend": backend,
             "started_utc": datetime.now(timezone.utc).isoformat()}
    fair_tuning._write_json(out / "controller_status.json", state)
    try:
        dataset = load_image_dataset(args.dataset, args.data_dir, download=args.download,
                                     train_limit=args.train_samples, test_limit=args.test_samples,
                                     seed=args.seed)
        split = stratified_training_three_way_split(
            dataset, seed=spec.split_seed, train_fraction=0.95 - spec.validation_fraction,
            calibration_fraction=spec.validation_fraction, attack_fraction=0.05)
        validate_targeted_split(split, args)
        contract = split_metadata(split, spec.split_seed)
        fair_tuning._write_json(out / "screen_protocol.json", {**payload, "data_contract": contract})
        dataset = split.calibration_dataset
        tasks, jobs, backend, _ = fair_tuning.prepare_tuning_tasks(dataset, tasks, args)

        def snapshot(executions, fingerprint, status):
            results = [r for _, r in executions]
            if results:
                experiments.write_result_files(out, results)
                fair_tuning._write_csv(out / "scenario_audit.csv",
                                      fair_tuning._validation_rows(executions))
            report = screen_report(tasks, executions, spec.performance_target)
            report.update(execution_status=status, fingerprint=fingerprint)
            fair_tuning._write_json(out / "screen_report.json", report)
            print("screen_snapshot=" + json.dumps(report, sort_keys=True), flush=True)

        with dashboard(spec, mode="log"):
            fair_tuning.execute_resumable_tuning_phase(
                dataset, tasks, args, output_dir=out, jobs=jobs, backend_description=backend,
                progress_enabled=True, progress_mode="log",
                fingerprint_context={"screen": payload, "data_contract": contract},
                on_snapshot=snapshot)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="interrupted_or_failed", error_type=type(error).__name__,
                     error=str(error))
        raise
    finally:
        state["ended_utc"] = datetime.now(timezone.utc).isoformat()
        fair_tuning._write_json(out / "controller_status.json", state)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
