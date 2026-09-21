#!/usr/bin/env python3
"""Read historical validation CSV/JSON and audit final-round selection separately.

No unpickling, training, formal planning, or writes to the source study. This is
an explicitly post-hoc analysis, not a prospective validation qualification.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
import csv
from dataclasses import fields
import hashlib
import io
import json
from pathlib import Path
from statistics import fmean

import run_cifar_six_final_metrics as runner
from cifar_final_metric_gate import SELECTION_METRICS, select_validation


def read_run(task, summary, rows):
    """Reconstruct metric-only evidence; never invent missing final metrics."""
    config = runner.fl.ExperimentConfig(**task["config"])
    saved_config = {f.name: summary[f.name] for f in fields(runner.fl.ExperimentConfig)}
    if runner.semantic_config(config) != runner.semantic_config(saved_config):
        raise ValueError("saved summary configuration differs from task identity")
    integers = {"round", "accepted_updates", "rejected_updates", "blacklisted_clients",
                "true_positive_revocations", "false_positive_revocations", "nonfinite_updates"}
    texts = {"method", "krum_selected_client"}
    records = []
    for row in rows:
        values = {}
        for field in fields(runner.fl.RoundRecord):
            value = row[field.name]
            if field.name in texts:
                values[field.name] = value
            elif field.name == "attack_active":
                if value not in ("True", "False"):
                    raise ValueError("invalid boolean in round evidence")
                values[field.name] = value == "True"
            else:
                values[field.name] = None if value == "" else int(value) if field.name in integers else float(value)
        records.append(runner.fl.RoundRecord(**values))
    if [r.round for r in records] != list(range(config.rounds + 1)):
        raise ValueError("complete CSV lacks the declared final round")
    if (records[-1].accuracy != summary["final_accuracy"] or
            records[-1].attack_target_success_rate != summary["final_attack_target_success_rate"]):
        raise ValueError("summary and final CSV metrics differ")
    return runner.fl.ExperimentResult(config, records, summary["final_accuracy"], summary["final_error"],
        summary["stopped_round"], tuple(filter(None, summary["malicious_clients"].split(","))),
        tuple(filter(None, summary["blacklisted_clients"].split(","))),
        runtime_seconds=summary["runtime_seconds"], nonfinite_updates=summary["nonfinite_updates"])


def reanalyze(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("analysis output must be separate from the source study")
    if output.exists() and any(output.iterdir()):
        raise ValueError("analysis output must be new or empty; keep earlier audits")
    hashes = {}

    def read(path):
        content = path.read_bytes()
        hashes[str(path.relative_to(source))] = hashlib.sha256(content).hexdigest()
        return content.decode("utf-8")

    def load(path):
        return json.loads(read(path))

    manifest = load(source / "manifest.json")
    frozen = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if runner.digest(frozen) != manifest["fingerprint"] or manifest["schema_version"] != 4:
        raise ValueError("requires an intact historical v4 manifest")
    plan = load(source / "validation_plan.json")
    if (plan["manifest_fingerprint"] != manifest["fingerprint"] or
            plan["tasks"] != runner.attach_fingerprints(runner.build_tasks(manifest["spec"], "validation"), manifest)):
        raise ValueError("historical task plan differs from immutable manifest")
    original = load(source / "validation_summary.json")
    statuses = original["tasks"]
    by_id = {r["task_id"]: r for r in statuses}
    if len(by_id) != len(statuses) or set(by_id) != {t["task_id"] for t in plan["tasks"]}:
        raise ValueError("historical status matrix is incomplete or duplicated")
    blockers = runner.validation_blockers(source, plan["tasks"], statuses)
    if blockers:
        raise ValueError("unresolved execution blockers: " + str(blockers))
    results, run_rows = defaultdict(list), []
    for task in plan["tasks"]:
        folder = source / "tasks" / task["task_id"]
        if load(folder / "task.json") != task:
            raise ValueError("task identity mismatch: " + task["task_id"])
        status = by_id[task["task_id"]]
        if status["status"] == "failed":
            load(folder / "failure.json")
            continue
        if status["status"] != "complete":
            raise ValueError("unresolved task: " + task["task_id"])
        summaries = load(folder / "summary.json")
        if len(summaries) != 1:
            raise ValueError("each task must contain one run")
        run = read_run(task, summaries[0], list(csv.DictReader(io.StringIO(read(folder / "rounds.csv")))))
        if runner.metrics(run) != status["metrics"]:
            raise ValueError("reconstructed health/metrics differ: " + task["task_id"])
        cid = task["candidate"]["candidate_id"]
        results[cid].append(run)
        run_rows.append({"task_id": task["task_id"], "method": task["method"], "candidate": cid,
            "partition": run.config.partition, "ratio": run.config.malicious_ratio, "seed": run.config.seed,
            "healthy": status["metrics"]["healthy"], "health_reasons": ";".join(status["metrics"]["reasons"]),
            "final_accuracy": run.records[-1].accuracy, "final_asr": run.records[-1].attack_target_success_rate,
            "final_target_confidence": run.records[-1].attack_target_confidence})
    spec = deepcopy(manifest["spec"])
    spec.update(schema_version=5, selection_metrics=dict(SELECTION_METRICS))
    report = select_validation(spec, results, plan["tasks"])
    # Ours014 is descriptive only if no candidate qualifies for formal training.
    selected = {**report["selected"], "sm9rrs": report["best_healthy_score_candidate"]}
    candidates = report["final_metric_gate"]["candidate_rows"]
    clean, overall, scenarios = [], [], []
    for method, cid in selected.items():
        group = [r for r in run_rows if r["candidate"] == cid]
        attacked = [r for r in group if r["ratio"] > 0]
        candidate = candidates[cid]
        overall.append({"method": method, "candidate": cid, "healthy": candidate["eligible"],
            "complete_tasks": len(group), "scorable": candidate["scorable"], "raw_score": candidate["raw_score"],
            "mean_final_accuracy": candidate["mean_accuracy"], "mean_final_asr": candidate["mean_asr"],
            "observed_attacked_tasks": len(attacked),
            "observed_partial_final_accuracy": fmean(r["final_accuracy"] for r in group) if group else None,
            "observed_partial_final_asr": fmean(r["final_asr"] for r in attacked) if attacked else None})
        for partition in ("all", "iid", "dirichlet"):
            subset = [r for r in group if r["ratio"] == 0 and (partition == "all" or r["partition"] == partition)]
            clean.append({"method": method, "candidate": cid, "partition": partition, "count": len(subset),
                "healthy_runs": sum(r["healthy"] for r in subset),
                "accuracy": fmean(r["final_accuracy"] for r in subset) if subset else None,
                "asr": fmean(r["final_asr"] for r in subset) if subset else None})
        for partition in ("iid", "dirichlet"):
            for ratio in sorted({s["malicious_ratio"] for s in spec["validation"]["scenarios"]}):
                subset = [r for r in group if r["partition"] == partition and r["ratio"] == ratio]
                scenarios.append({"method": method, "candidate": cid, "partition": partition, "ratio": ratio,
                    "count": len(subset), "healthy_runs": sum(r["healthy"] for r in subset),
                    "accuracy": fmean(r["final_accuracy"] for r in subset) if subset else None,
                    "asr": fmean(r["final_asr"] for r in subset) if subset else None})
    matched = defaultdict(dict)
    for row in run_rows:
        if row["candidate"] == selected[row["method"]] and row["ratio"] > 0:
            matched[row["partition"], row["ratio"], row["seed"]][row["method"]] = row
    taskwise_wins = []
    for row in run_rows:
        peers = matched.get((row["partition"], row["ratio"], row["seed"]), {})
        if row["candidate"] not in report["final_metric_gate"]["eligible_ours_candidates"] or len(peers) != 6:
            continue
        best = min(r["final_asr"] for method, r in peers.items() if method != "sm9rrs")
        if row["final_asr"] <= best + 1e-12:
            taskwise_wins.append({**row, "lowest_baseline_final_asr": best,
                "strictly_lower": row["final_asr"] < best - 1e-12})
    if any(hashlib.sha256((source / name).read_bytes()).hexdigest() != value for name, value in hashes.items()):
        raise ValueError("historical evidence changed while reading")
    source_checks = {name: hashlib.sha256((runner.REPO / name).read_bytes()).hexdigest() == value
                     for name, value in manifest["source_sha256"].items()}
    report.update(post_hoc_reanalysis=True, source_study=str(source), source_manifest_fingerprint=manifest["fingerprint"],
        original_selected=original["selected"], selected_for_description=selected,
        input_sha256=hashes, original_source_hash_checks=source_checks,
        analysis_source_sha256=runner.source_hashes(runner.REPO),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        complete_six_way_attacked_tasks=sum(len(peers) == 6 for peers in matched.values()),
        healthy_ours_taskwise_final_asr_minima=taskwise_wins,
        final_plan_created=False, training_started=False, historical_evidence_unchanged=True)
    output.mkdir(parents=True, exist_ok=True)
    runner.write_json(output / "validation_final_metric_reanalysis.json", report)
    for name, rows in (("per_run_final.csv", run_rows), ("selected_final_comparison.csv", overall),
                       ("clean_final_comparison.csv", clean), ("scenario_final_comparison.csv", scenarios)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = reanalyze(args.source, args.output)
    print(json.dumps({key: report[key] for key in ("status", "ours_health_passed", "selected",
        "best_healthy_score_candidate", "final_plan_created", "historical_evidence_unchanged")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
