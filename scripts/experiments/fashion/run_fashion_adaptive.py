#!/usr/bin/env python3
"""Independent Fashion-MNIST two-seed search -> single-seed six-method study.

The controller is adapted from the frozen CIFAR v8 controller. Keeping a
separate entry preserves its byte-level source identity for existing studies."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

if __name__ == "__main__":
    from run_experiments_from_config import _try_project_virtualenv
    _try_project_virtualenv(Path(__file__).resolve().parent, launcher_path=Path(__file__))

import run_cifar_six_from_scratch as base
import cifar_adaptive_gate as gate
import fashion_adaptive_runtime as runtime
from fashion_dataset import load_split
from cifar_adaptive_search import (AdaptiveTPESampler, FIXED_METHODS, METHODS,
                                  compute_relative_objective, space_contract)

REPO = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO / "configs/fashion_mnist_resnet18_gn_tpe_v8.json"


def load_spec(path):
    spec = runtime.read_json(path)
    if spec.get("schema_version") != 8 or spec.get("protocol") != "fashion-mnist-resnet18-gn-tpe-v1":
        raise ValueError("this entry requires the independent schema-8 TPE protocol")
    if spec.get("model") != {"architecture": "fashion_resnet18_gn", "group_norm_groups": 2}:
        raise ValueError("all six methods must use grayscale Fashion ResNet-18 with GN2")
    shared = spec["shared_parameters"]
    config = base.fl.ExperimentConfig(**shared)
    config.validate()
    if set(shared) != set(asdict(config)):
        raise ValueError("every common ExperimentConfig field must be declared")
    if (config.rounds != 150 or config.num_clients != 100 or config.early_stop
            or config.compute_backend != "torch" or config.crypto_mode != "sm9"
            or config.checkpoint_interval != 1 or config.eval_interval != 1
            or config.attack != "alternating_minimization" or config.lr_decay != .99
            or config.attack_stealth_steps != 1 or config.attack_distance_weight != .0001
            or config.attack_target_count != 200 or config.attack_source_label != 5
            or config.attack_target_label != 7 or config.dirichlet_alpha != .5
            or config.attack_start_round != config.detector_window + 2):
        raise ValueError("new shared protocol must retain the declared 150-round execution and K+2 attack")
    for key, value in space_contract("public")["default_coordinates"].items():
        if shared[key] != value:
            raise ValueError("shared template must match the first declared public point")
    if spec.get("search_spaces") != {m: space_contract(m) for m in ("public", *METHODS)}:
        raise ValueError("search-space contract differs from the versioned TPE implementation")
    data = spec["dataset"]
    if (data["name"] != "fashion_mnist" or data["train_samples"] != 60000
            or data["test_samples"] != 10000 or data["validation_fraction"] != .05):
        raise ValueError("full Fashion-MNIST with the fixed 54k/3000/3000 split is required")
    for phase, count, ratios in (("validation", 2, (0., .1, .3, .5, .7)),
                                 ("final", 1, (0., .2, .4, .6, .8))):
        seeds = spec[phase]["seeds"]
        if len(seeds) != count or len(set(seeds)) != count or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
            raise ValueError(f"{phase} requires {count} distinct nonnegative 32-bit seeds")
        scenarios = spec[phase]["scenarios"]
        if len(scenarios) != 10 or {(s["partition"], s["malicious_ratio"]) for s in scenarios} != {(p, r) for p in ("iid", "dirichlet") for r in ratios}:
            raise ValueError("the ten declared clean/attack scenarios must be retained")
    if set(spec["validation"]["seeds"]) & set(spec["final"]["seeds"]):
        raise ValueError("formal and search seeds must be disjoint")
    if spec["objective"] != gate.OBJECTIVE or spec["performance_target"] != gate.PERFORMANCE_TARGET:
        raise ValueError("retain original final-round Score and inclusive two-point gaps")
    if spec["gates"] != {"max_clean_accuracy_drop": .03, "max_nonfinite_updates": 0,
                         "min_round_completion_rate": 1.}:
        raise ValueError("original health and clean utility constraints are mandatory")
    if spec["promotion"] != {"require_mean_dual_best": False, "manual_healthy_override": True,
                             "ask_on_every_resume": True}:
        raise ValueError("retain optional mean preference and per-invocation healthy override")
    search = spec["search"]
    for name in ("max_public_trials", "defense_trials_per_method", "startup_trials", "max_gpus", "seed"):
        if type(search[name]) is not int or search[name] < 1:
            raise ValueError("search integer settings must be positive")
    if search["startup_trials"] < 2 or search["defense_trials_per_method"] <= search["startup_trials"]:
        raise ValueError("defense budget must include adaptive proposals after startup")
    if not math.isfinite(search["budget_hours"]) or not 0 < search["budget_hours"] <= 48:
        raise ValueError("the declared search budget must be within 48 hours")
    return spec


def new_state(spec, manifest):
    sampler = AdaptiveTPESampler("public", spec["search"]["seed"],
                                 startup_trials=spec["search"]["startup_trials"])
    return {"schema_version": 1, "manifest_fingerprint": manifest["fingerprint"],
            "public_sampler": sampler.state_dict(), "blocks": [],
            "elapsed_search_seconds": 0., "status": "searching"}


def block_spec(master, block):
    spec = deepcopy(master)
    spec["shared_parameters"].update(block["proposal"]["parameters"])
    spec["shared_parameters"]["attack_start_round"] = spec["shared_parameters"]["detector_window"] + 2
    spec["candidates"] = {method: [] for method in METHODS}
    for wave in block["waves"]:
        for item in wave["candidates"]:
            spec["candidates"][item["method"]].append(deepcopy(item["candidate"]))
    spec["fallback_candidates"] = {m: cs[0]["candidate_id"] for m, cs in spec["candidates"].items() if m != "sm9rrs" and cs}
    return spec


def add_block(state, spec):
    sampler = AdaptiveTPESampler.from_state(state["public_sampler"])
    proposal = sampler.ask()
    index = len(state["blocks"])
    context = base.digest(proposal["parameters"])
    block = {"id": f"b{index:03d}", "proposal": proposal, "context_id": context,
             "waves": [], "status": "running", "method_samplers": {
                 m: AdaptiveTPESampler(m, spec["search"]["seed"] + index * 100 + i + 1,
                     context_id=context, startup_trials=spec["search"]["startup_trials"]).state_dict()
                 for i, m in enumerate(METHODS)}}
    state["public_sampler"] = sampler.state_dict()
    state["blocks"].append(block)
    return block


def add_wave(block):
    index = len(block["waves"])
    wave = {"index": index, "status": "pending", "candidates": []}
    for method in METHODS:
        if index and method in FIXED_METHODS:
            continue
        sampler = AdaptiveTPESampler.from_state(block["method_samplers"][method])
        proposal = sampler.ask()
        candidate = {"candidate_id": f"{method}-{block['id']}-d{index:03d}",
                     "variant": "original", "parameters": proposal["parameters"]}
        wave["candidates"].append({"method": method, "candidate": candidate, "proposal": proposal})
        block["method_samplers"][method] = sampler.state_dict()
    block["waves"].append(wave)
    return wave


def rows_for_objective(runs):
    return [{"scenario": list(base.scenario_key(r)), "accuracy": r.final_accuracy,
             "asr": r.records[-1].attack_target_success_rate,
             "attacked": r.config.malicious_ratio > 0,
             "complete": gate.run_audit(r)["scorable"], "healthy": gate.run_audit(r)["healthy"]}
            for r in runs if gate.run_audit(r)["scorable"]]


def observe_wave(block, wave, report):
    candidates = report["final_metric_gate"]["candidate_rows"]
    for item in wave["candidates"]:
        cid, method = item["candidate"]["candidate_id"], item["method"]
        row = candidates[cid]
        sampler = AdaptiveTPESampler.from_state(block["method_samplers"][method])
        # Numerical failures have no fabricated score. Complete health failures
        # are measured with a penalty and remain ineligible for Ours promotion.
        loss = ((0. if row["eligible"] else 2.) - row["raw_score"]) if row["scorable"] else None
        sampler.tell(item["proposal"]["trial_id"], loss,
                     status="complete" if loss is not None else "failed",
                     metrics={"candidate_id": cid, "healthy": row["eligible"],
                              "raw_score": row["raw_score"], "reasons": row["invalid_reasons"]})
        block["method_samplers"][method] = sampler.state_dict()
    wave["status"] = "complete"


def observe_block(state, block, report, results):
    rows = report["final_metric_gate"]["candidate_rows"]
    possible = [cid for cid, row in rows.items() if row["method"] == "sm9rrs" and row["scorable"]]
    objectives = {}
    baselines = {m: rows_for_objective(results.get(report["selected"][m], [])) for m in METHODS if m != "sm9rrs"}
    for cid in possible:
        ours = rows_for_objective(results[cid])
        for row in ours:
            row["healthy"] = row["healthy"] and rows[cid]["eligible"]
        objectives[cid] = compute_relative_objective(ours, baselines,
                                                    original_score=rows[cid]["raw_score"])
        objectives[cid]["candidate_health_reasons"] = rows[cid]["invalid_reasons"]
    finite = [r["loss"] for r in objectives.values() if r["loss"] is not None]
    loss = min(finite) if finite else None
    sampler = AdaptiveTPESampler.from_state(state["public_sampler"])
    sampler.tell(block["proposal"]["trial_id"], loss,
                 status="complete" if loss is not None else "failed",
                 metrics={"block": block["id"], "status": report["status"],
                          "objectives": objectives})
    state["public_sampler"] = sampler.state_dict()
    block.update(status="complete", report=report, objectives=objectives)


def audit_block(output, spec, manifest, block):
    current = block_spec(spec, block)
    tasks = runtime.attach_tasks(current, "validation", manifest)
    results, statuses = runtime.collect(output, tasks)
    if not runtime.evidence_resolved(statuses):
        raise ValueError("completed search evidence is missing or operationally unresolved")
    return gate.select_validation(current, results, tasks), results, tasks


def selection_view(block):
    """The frozen, fully attempted prefix; pending proposals remain in the ledger."""
    if block["status"] == "complete":
        return block
    snapshot = block.get("budget_selection_snapshot")
    if snapshot is None:
        return None
    view = deepcopy(block)
    view["waves"] = view["waves"][:snapshot["completed_waves"]]
    view["report"] = snapshot["report"]
    return view


def prepare_selection(output, spec, manifest, state):
    """Re-audit raw evidence before freezing or reusing a formal selection."""
    for block in state["blocks"]:
        if block["status"] != "complete" and state["status"] == "budget_exhausted":
            prefix = []
            for wave in block["waves"]:
                if wave["status"] != "complete":
                    break
                prefix.append(wave)
            if prefix and "budget_selection_snapshot" not in block:
                view = deepcopy(block)
                view["waves"] = prefix
                report, _, tasks = audit_block(output, spec, manifest, view)
                block["budget_selection_snapshot"] = {
                    "completed_waves": len(prefix), "report": report,
                    "task_fingerprints": {t["task_id"]: t["fingerprint"] for t in tasks},
                    "excluded_candidates": [i["candidate"]["candidate_id"]
                        for w in block["waves"][len(prefix):] for i in w["candidates"]],
                    "reason": "search_walltime_budget_exhausted",
                    "cross_public_condition_search_budgets_may_differ": True}
        view = selection_view(block)
        if view is not None:
            report, _, tasks = audit_block(output, spec, manifest, view)
            if report != view["report"]:
                raise ValueError("saved selection report no longer matches raw validation evidence")
            snapshot = block.get("budget_selection_snapshot")
            if snapshot and snapshot["task_fingerprints"] != {t["task_id"]: t["fingerprint"] for t in tasks}:
                raise ValueError("budget prefix task identity changed")


def run_search(args, spec, manifest, state):
    output = args.output
    state_path = output / "search_state.json"
    before = state["elapsed_search_seconds"]
    start = time.monotonic()
    remaining = max(0., spec["search"]["budget_hours"] * 3600 - before)
    deadline = time.time() + remaining
    last_save = [0.]

    def save(force=False):
        if force or time.monotonic() - last_save[0] >= 10:
            state["elapsed_search_seconds"] = before + time.monotonic() - start
            base.write_json(state_path, state)
            last_save[0] = time.monotonic()

    try:
        while state["status"] == "searching":
            # A crash can fall between the durable last-wave update and the
            # public observation. Finish that transaction without new training.
            for block in state["blocks"]:
                if (block["status"] != "complete" and len(block["waves"]) == spec["search"]["defense_trials_per_method"]
                        and all(w["status"] == "complete" for w in block["waves"])):
                    report, results, _ = audit_block(output, spec, manifest, block)
                    observe_block(state, block, report, results)
                    save(True)
            if time.time() >= deadline:
                state["status"] = "budget_exhausted"
                break
            unfinished = [b for b in state["blocks"] if b["status"] != "complete"]
            if unfinished:
                block = unfinished[-1]
            elif len(state["blocks"]) >= spec["search"]["max_public_trials"]:
                state["status"] = "trial_limit_reached"
                break
            else:
                block = add_block(state, spec)
                save(True)  # Proposal and sampler RNG history precede execution.
            print("PUBLIC_TRIAL " + json.dumps({"block": block["id"],
                  "strategy": block["proposal"]["strategy"], "parameters": block["proposal"]["parameters"]}), flush=True)
            while len([w for w in block["waves"] if w["status"] == "complete"]) < spec["search"]["defense_trials_per_method"]:
                pending = [w for w in block["waves"] if w["status"] != "complete"]
                wave = pending[-1] if pending else add_wave(block)
                save(True)
                current = block_spec(spec, block)
                all_tasks = runtime.attach_tasks(current, "validation", manifest)
                wave_ids = {i["candidate"]["candidate_id"] for i in wave["candidates"]}
                tasks = [t for t in all_tasks if t["candidate"]["candidate_id"] in wave_ids]
                plan = runtime.save_plan(output, f"{block['id']}-wave{wave['index']:03d}", tasks, manifest)
                statuses = runtime.execute(args, tasks, plan, deadline=deadline, heartbeat=save)
                if not runtime.evidence_resolved(statuses):
                    state["status"] = "budget_exhausted" if time.time() >= deadline else "execution_blocked"
                    state["blockers"] = [r for r in statuses if r["status"] != "complete"]
                    print("SEARCH_PENDING evidence incomplete; checkpoints retained", flush=True)
                    save(True)
                    return
                results, _ = runtime.collect(output, all_tasks)
                report = gate.select_validation(current, results, all_tasks)
                observe_wave(block, wave, report)
                base.write_json(output / f"{block['id']}_validation_summary.json", report)
                save(True)
                if wave["index"] + 1 == spec["search"]["defense_trials_per_method"]:
                    observe_block(state, block, report, results)
                    save(True)
                    print(f"PUBLIC_TRIAL_COMPLETE {block['id']} status={report['status']}", flush=True)
                del results
                if time.time() >= deadline:
                    state["status"] = "budget_exhausted"
                    break
    finally:
        save(True)


def choose_block(state):
    """Only fully attempted waves with matched method budgets enter selection."""
    qualified, healthy = [], []
    for original in state["blocks"]:
        block = selection_view(original)
        if block is None:
            continue
        report = block["report"]
        rows = report["final_metric_gate"]["candidate_rows"]
        trials = {r["candidate_id"]: r for r in report["trials"]}
        for cid, row in rows.items():
            if row["method"] != "sm9rrs" or not row["eligible"]:
                continue
            target = report["ours_candidate_targets"][cid]
            tie = (row["raw_score"], -trials[cid]["worst_attack_success_rate"], cid, block["id"])
            healthy.append((tie, block, cid))
            if target["promotion_qualified"]:
                qualified.append(((target.get("mean_dual_best_passed", False), *tie), block, cid))
    pool = qualified or healthy
    if not pool:
        return None
    _, block, cid = max(pool, key=lambda item: item[0])
    return {"block_id": block["id"], "ours_candidate": cid,
            "completed_waves": len(block["waves"]),
            "target_qualified": bool(qualified),
            "selected": {**block["report"]["selected"], "sm9rrs": cid},
            "raw_score": block["report"]["final_metric_gate"]["candidate_rows"][cid]["raw_score"]}


def ask_override(output, choice, manifest):
    folder = output / "continuation_responses"
    folder.mkdir(exist_ok=True)
    while True:
        print(f"CONTINUATION_PROMPT 所有Ours均未达到双指标≤2个百分点。是否采用最佳健康候选 {choice['ours_candidate']} (Score={choice['raw_score']:.6f}) 进入主实验？[Y/N] ", end="", flush=True)
        raw = sys.stdin.readline()
        answer = raw.strip().upper()
        if answer in ("Y", "N") or raw == "":
            approved = answer == "Y"
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
            base.immutable_json(folder / (stamp + ".json"), {
                "approved": approved, "response": answer if raw else "EOF",
                "manifest_fingerprint": manifest["fingerprint"], "choice": choice,
                "previous_Y_is_not_current_approval": True})
            return approved


def run_final(args, spec, manifest, state, choice):
    output = args.output
    frozen_path = output / "continuation_decision.json"
    decision = runtime.read_json(frozen_path) if frozen_path.exists() else None
    basis = {"manifest_fingerprint": manifest["fingerprint"], "choice": choice,
             "validation_target_passed": choice["target_qualified"],
             "source_search_digest": base.digest(state)}
    if decision is not None and decision != basis:
        raise ValueError("frozen selection no longer matches its search evidence")
    if not choice["target_qualified"] and not ask_override(output, choice, manifest):
        print("FINAL_NOT_STARTED 用户未确认；搜索证据与检查点保留。", flush=True)
        return 0
    base.immutable_json(frozen_path, basis)
    block = selection_view(next(b for b in state["blocks"] if b["id"] == choice["block_id"]))
    current = block_spec(spec, block)
    tasks = runtime.attach_tasks(current, "final", manifest, choice["selected"])
    if len(tasks) != 60:
        raise ValueError("one formal seed requires six methods x ten scenarios = 60 tasks")
    plan = runtime.save_plan(output, "formal", tasks, manifest)
    base.immutable_json(output / "final_plan.json", {
        "manifest_fingerprint": manifest["fingerprint"], "selected": choice["selected"],
        "public_parameters": block["proposal"]["parameters"], "tasks": tasks})
    runtime.execute(args, tasks, plan)
    results, statuses = runtime.collect(output, tasks)
    from fashion_adaptive_reporting import write_final_report
    write_final_report(output, current, choice["selected"], results, tasks, statuses,
                       block["report"], basis)
    print("FINAL_REPORT " + str(output / "final_results" / "visualizations.html"), flush=True)
    return 0 if runtime.evidence_resolved(statuses) else 2


def run_parent(args, spec):
    args.output = (args.output or Path(spec["output_dir"])).resolve()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with base.file_lock(output / ".runner.lock", nonblocking=True):
        path = output / "manifest.json"
        if path.exists():
            manifest = runtime.read_json(path)
            if manifest != runtime.build_manifest(spec, manifest["data_contract"]):
                raise ValueError("config/model/source identity changed; do not overwrite this study")
        else:
            if any(p.name != ".runner.lock" for p in output.iterdir()):
                raise ValueError("nonempty output has no immutable manifest")
            data, contract = load_split(spec, args.data_dir)
            del data
            manifest = runtime.build_manifest(spec, contract)
            base.immutable_json(path, manifest)
        state_path = output / "search_state.json"
        state = runtime.read_json(state_path) if state_path.exists() else new_state(spec, manifest)
        if state["manifest_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("search state belongs to a different study")
        # Restoring the samplers validates their space, proposals and evidence.
        AdaptiveTPESampler.from_state(state["public_sampler"])
        for block in state["blocks"]:
            for value in block["method_samplers"].values():
                AdaptiveTPESampler.from_state(value)
        if (output / "continuation_decision.json").exists():
            if args.phase == "search":
                raise ValueError("formal selection is frozen; search cannot restart")
        elif args.phase != "final":
            if state["status"] == "execution_blocked":
                state["status"] = "searching"
                state.pop("blockers", None)
            if state["status"] == "searching":
                run_search(args, spec, manifest, state)
        if state["status"] == "execution_blocked":
            return 2
        if state["status"] == "searching":
            raise ValueError("search is not complete; resume --phase search/all first")
        prepare_selection(output, spec, manifest, state)
        base.write_json(state_path, state)
        choice = choose_block(state)
        summary = {"status": state["status"], "elapsed_search_seconds": state["elapsed_search_seconds"],
                   "completed_public_blocks": sum(b["status"] == "complete" for b in state["blocks"]),
                   "actual_search_budget": [{"block_id": b["id"], "status": b["status"],
                        "fully_attempted_waves": sum(w["status"] == "complete" for w in b["waves"]),
                        "public_strategy": b["proposal"]["strategy"],
                        "defense_tpe_proposals": {m: sum(t["strategy"] == "tpe" for t in s["trials"])
                            for m, s in b["method_samplers"].items()},
                        "budget_prefix_selected_for_comparison": "budget_selection_snapshot" in b}
                        for b in state["blocks"]],
                   "public_tpe_proposals": sum(t["strategy"] == "tpe" for t in state["public_sampler"]["trials"]),
                   "cross_public_condition_search_budgets_may_differ": True,
                   "choice": choice, "formal_results_used_for_selection": False,
                   "search_budget_is_not_a_global_optimality_guarantee": True}
        base.write_json(output / "search_summary.json", summary)
        if choice is None:
            print("FINAL_NOT_STARTED 无完整公平波次中的健康Ours候选；结果及检查点保留。", flush=True)
            return 2
        block = selection_view(next(b for b in state["blocks"] if b["id"] == choice["block_id"]))
        base.write_json(output / "best_parameters.json", {**summary,
            "public_parameters": block["proposal"]["parameters"],
            "public_proposal_strategy": block["proposal"]["strategy"],
            "public_fit_audit": block["proposal"]["fit_audit"],
            "attack_start_round": block["proposal"]["parameters"]["detector_window"] + 2,
            "selected_parameters": {m: next(c["parameters"] for c in block_spec(spec, block)["candidates"][m]
                                            if c["candidate_id"] == cid) for m, cid in choice["selected"].items()}})
        if args.phase == "search":
            return 0
        return run_final(args, spec, manifest, state, choice)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--devices", nargs="+", default=["auto"])
    parser.add_argument("--phase", choices=("all", "search", "final"), default="all")
    parser.add_argument("--plan-only", action="store_true", help="Validate and show search contract; no data/GPU/training")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stop-at", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return runtime.worker(args)
    spec = load_spec(args.config)
    if args.plan_only:
        print(json.dumps({"dataset": spec["dataset"], "model": spec["model"],
            "input_shape": [1, 28, 28],
            "split_counts": {"train": 54000, "validation": 3000, "attack_auxiliary": 3000, "official_test": 10000},
            "attack_classes": {"source_label": 5, "source_name": "Sandal", "target_label": 7, "target_name": "Sneaker"},
            "validation": spec["validation"],
            "final": spec["final"], "search": spec["search"], "rounds": 150,
            "attack_start": "selected K + 2", "search_spaces": spec["search_spaces"],
            "maximum_validation_tasks": spec["search"]["max_public_trials"] *
                (3 * spec["search"]["defense_trials_per_method"] + 3) * 20,
            "formal_tasks": 60, "execution_started": False}, indent=2, ensure_ascii=False))
        return 0
    from run_cifar_six_with_progress import discover_gpus, select_devices
    found = discover_gpus(REPO)
    args.devices, skipped = select_devices(found, args.devices, min_free_memory_mib=8192.)
    args.devices = args.devices[:spec["search"]["max_gpus"]]
    print("DEVICES " + json.dumps({"selected": args.devices, "skipped": skipped}), flush=True)
    print("SEARCH_CONTRACT 双种子TPE搜索；预算内最佳已评估配置，不保证全局最优或达标。正式实验另计时。", flush=True)
    return run_parent(args, spec)


if __name__ == "__main__":
    raise SystemExit(main())
