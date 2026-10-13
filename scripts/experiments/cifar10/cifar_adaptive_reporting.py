"""Independent final reporting for the CIFAR ResNet18-GN search protocol.

Reporting never selects parameters or changes task health. Only complete,
finite final-round runs contribute observations; finite health failures stay.
Missing tasks remain explicit and disable their method's overall aggregate.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import html
import json
import math
from pathlib import Path
from statistics import fmean, stdev

from run_cifar_six_from_scratch import ALL_METHODS, json_safe, metrics, semantic_config, write_json
from cifar_resnet_gn import protocol_descriptor


LABELS = dict(zip(ALL_METHODS, ("Ours", "VERT", "AlignIns", "Krum", "TAD", "FedAvg")))
COLORS = dict(zip(ALL_METHODS, ("#8DBAD9", "#E99092", "#8ADDE4", "#DCDD8F", "#FEBD85", "#C4A9A2")))
STYLES = dict(zip(ALL_METHODS, ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (8, 3)))))


class ReportIntegrityError(ValueError):
    pass


def _canonical(value):
    return json.dumps(json_safe(value), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _canonical(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items() if key in fields})
    temporary.replace(path)


def _valid_probability(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1


def _result_key(candidate_id, config):
    return candidate_id, _canonical(semantic_config(config))


def _scenario(config):
    return config["partition"], float(config["malicious_ratio"]), int(config["seed"])


def _check_inputs(spec, selected, tasks, statuses, results):
    if set(selected) != set(ALL_METHODS):
        raise ReportIntegrityError("final selection must contain all six methods")
    final = spec["final"]
    rounds = spec["shared_parameters"]["rounds"]
    if (not final["seeds"] or len(set(final["seeds"])) != len(final["seeds"])
            or not final["scenarios"] or rounds < 1 or len(set(selected.values())) != len(ALL_METHODS)):
        raise ReportIntegrityError("invalid final seeds, scenarios or candidate identities")
    expected = {(m, scenario["partition"], float(scenario["malicious_ratio"]), seed)
                for m in ALL_METHODS for scenario in final["scenarios"] for seed in final["seeds"]}
    actual, task_ids, by_key = set(), set(), {}
    for task in tasks:
        task_id, method, config = task["task_id"], task["method"], task["config"]
        if (not task_id or Path(task_id).name != task_id or task_id in (".", "..")
                or task_id in task_ids or not task.get("fingerprint") or task.get("phase") != "final"):
            raise ReportIntegrityError("invalid, duplicate or unfingerprinted formal task")
        cid = task["candidate"]["candidate_id"]
        key = (method, *_scenario(config))
        if (method not in selected or cid != selected[method] or config["method"] != method
                or config["rounds"] != rounds or key in actual):
            raise ReportIntegrityError("formal task differs from frozen selection/protocol")
        actual.add(key)
        task_ids.add(task_id)
        by_key[_result_key(cid, config)] = task
    if actual != expected:
        raise ReportIntegrityError("formal task matrix differs from spec.final")
    status_by_id = {}
    for status in statuses:
        task_id = status["task_id"]
        if task_id not in task_ids or task_id in status_by_id or status["status"] not in ("pending", "failed", "complete"):
            raise ReportIntegrityError("unknown, duplicate or invalid task status")
        status_by_id[task_id] = status
    if set(status_by_id) != task_ids:
        raise ReportIntegrityError("missing formal task status")
    found = {}
    for cid, runs in results.items():
        for run in runs:
            key = _result_key(cid, run.config)
            if key not in by_key or key in found:
                raise ReportIntegrityError("unplanned, mismatched or duplicate result")
            found[key] = run
    return status_by_id, found


def _inspect_result(run, rounds):
    if run.stopped_round != rounds:
        return False, "incomplete_rounds", None
    if [row.round for row in run.records] != list(range(rounds + 1)):
        raise ReportIntegrityError("finished result has missing/duplicate round records")
    for row in run.records:
        if row.method != run.config.method or row.malicious_ratio != run.config.malicious_ratio:
            raise ReportIntegrityError("round record identity differs from task")
        for name in ("accuracy", "honest_weight_loss", "malicious_weight_mass"):
            if not _valid_probability(getattr(row, name)):
                raise ReportIntegrityError(f"finished result contains invalid {name}")
        for name in ("attack_target_success_rate", "attack_target_confidence"):
            value = getattr(row, name)
            if value is not None and not _valid_probability(value):
                raise ReportIntegrityError(f"finished result contains invalid {name}")
    final = run.records[-1]
    if (not _valid_probability(run.final_accuracy)
            or not math.isclose(run.final_accuracy, final.accuracy, rel_tol=0, abs_tol=1e-9)):
        raise ReportIntegrityError("summary final accuracy disagrees with final round")
    if run.config.malicious_ratio > 0 and final.attack_target_success_rate is None:
        raise ReportIntegrityError("finished attacked task has no final ASR")
    for name in ("runtime_seconds", "runtime_without_crypto_seconds", "peak_memory_mb"):
        value = getattr(run, name)
        if not math.isfinite(value) or value < 0:
            raise ReportIntegrityError(f"invalid {name}")
    health = metrics(run)
    if any(reason.startswith("invalid_") for reason in health["reasons"]):
        raise ReportIntegrityError("finished result has invalid scientific metrics")
    return True, None, health


def _plots(destination, tasks, available, *, rounds, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    figures = []
    for partition in ("iid", "dirichlet"):
        ratios = sorted({t["config"]["malicious_ratio"] for t in tasks if t["config"]["partition"] == partition})
        if not ratios:
            continue
        for metric, label in (("accuracy", "Accuracy"), ("attack_target_success_rate", "ASR")):
            ncols = min(3, len(ratios))
            nrows = math.ceil(len(ratios) / ncols)
            figure, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.7 * nrows), squeeze=False)
            for axis, ratio in zip(axes.flat, ratios):
                for method in ALL_METHODS:
                    matched = [available[t["task_id"]] for t in tasks
                               if t["method"] == method and t["config"]["partition"] == partition
                               and t["config"]["malicious_ratio"] == ratio and t["task_id"] in available]
                    data = np.array([[np.nan if getattr(r, metric) is None else getattr(r, metric)
                                      for r in run.records] for run in matched], dtype=float)
                    legend = f"{LABELS[method]} (n={len(matched)}/{len(seeds)})"
                    if not matched:
                        axis.plot([], [], color=COLORS[method], label=legend + " missing")
                        continue
                    mean = np.array([fmean(v for v in data[:, i] if np.isfinite(v))
                                     if np.isfinite(data[:, i]).any() else np.nan
                                     for i in range(rounds + 1)])
                    axis.plot(range(rounds + 1), mean * 100, color=COLORS[method], label=legend,
                              linestyle=STYLES[method], linewidth=1.8)
                    if len(matched) > 1:
                        sd = np.array([stdev(data[:, i]) if np.isfinite(data[:, i]).all() else np.nan
                                       for i in range(rounds + 1)])
                        axis.fill_between(range(rounds + 1), (mean - sd) * 100, (mean + sd) * 100,
                                          color=COLORS[method], alpha=.12)
                axis.set(title=f"{partition.upper()} | malicious={ratio:.0%}" +
                         (" | background ASR" if ratio == 0 and label == "ASR" else ""),
                         xlabel="Round", ylabel=label + " (%)", ylim=(0, 100))
                axis.grid(alpha=.2)
                axis.legend(fontsize=7)
            for axis in list(axes.flat)[len(ratios):]:
                axis.set_visible(False)
            complete = all(t["task_id"] in available for t in tasks if t["config"]["partition"] == partition)
            figure.suptitle("CIFAR-10 ResNet18-GN | " + label + " | " +
                            ("single formal seed; SD unavailable" if len(seeds) == 1 else "mean; sample SD where n>1") +
                            (" | INCOMPLETE" if not complete else ""), fontsize=11)
            figure.tight_layout(rect=(0, 0, 1, .95))
            stem = f"{partition}_{label.lower()}"
            try:
                for extension in ("png", "svg"):
                    figure.savefig(destination / f"{stem}.{extension}", dpi=150)
            finally:
                plt.close(figure)
            figures.append(stem)
    return figures


def write_final_report(output, spec, selected, results_by_candidate, tasks, statuses,
                       validation_summary, continuation_decision=None):
    """Write repeatable report artifacts from the runner's verified results.

    Return final_summary; never promote a candidate, execute training or alter
    per-task sources. A master spec or selected public block is accepted;
    actual trained parameters are always preserved from each frozen task.
    """
    output = Path(output)
    rounds = int(spec["shared_parameters"]["rounds"])
    status_by_id, found = _check_inputs(spec, selected, tasks, statuses, results_by_candidate)
    available, per_task, merged_rounds, source_files, health_failures = {}, [], [], {}, []
    for task in tasks:
        tid, config, cid = task["task_id"], task["config"], task["candidate"]["candidate_id"]
        status = status_by_id[tid]
        for field, wanted in (("method", task["method"]), ("candidate_id", cid)):
            if field in status and status[field] != wanted:
                raise ReportIntegrityError("task status identity mismatch")
        folder = output / "tasks" / tid
        identity = folder / "task.json"
        if identity.exists() and json.loads(identity.read_text()) != task:
            raise ReportIntegrityError("task.json identity differs from final plan")
        for filename in ("task.json", "metrics.json", "rounds.csv", "failure.json"):
            path = folder / filename
            if path.is_file():
                source_files[str(path.relative_to(output))] = _file_digest(path)
        run = found.get(_result_key(cid, config))
        complete, reason, health = False, status.get("error") or status["status"], None
        failure = None
        failure_path = folder / "failure.json"
        if failure_path.exists():
            failure = json.loads(failure_path.read_text())
            if failure.get("task_id") != tid:
                raise ReportIntegrityError("failure.json identity differs from formal task")
            if status["status"] == "failed":
                reason = ": ".join(str(failure.get(name, "")) for name in ("exception", "message")).strip(": ") or reason
        if run is None and status["status"] == "complete":
            raise ReportIntegrityError("status claims complete but result is missing")
        if run is not None:
            if status["status"] == "pending":
                raise ReportIntegrityError("pending task has a completed result")
            complete, reason, health = _inspect_result(run, rounds)
        if complete:
            if status["status"] != "complete":
                raise ReportIntegrityError("complete result conflicts with failed task status")
            available[tid] = run
            final = run.records[-1]
            for record in run.records:
                merged_rounds.append({"task_id": tid, "candidate_id": cid,
                    "partition": config["partition"], "seed": config["seed"], **asdict(record)})
        reasons = health["reasons"] if health else [reason]
        if not complete or not health["healthy"]:
            health_failures.append({"task_id": tid, "completed": complete, "reasons": reasons})
        per_task.append({"task_id": tid, "candidate_id": cid, "method": task["method"],
            "partition": config["partition"], "malicious_ratio": config["malicious_ratio"],
            "seed": config["seed"], "fingerprint": task["fingerprint"],
            "execution_status": status["status"], "final_round_available": complete,
            "healthy": health["healthy"] if health else None, "health_reasons": reasons,
            "last_completed_round": run.stopped_round if run else (
                (failure.get("execution_context", {}) or {}).get("last_completed_round") if failure else None),
            "required_round": rounds,
            "final_accuracy": final.accuracy if complete else None,
            "final_asr": final.attack_target_success_rate if complete else None,
            "runtime_seconds": run.runtime_seconds if complete else None,
            "runtime_without_crypto_seconds": run.runtime_without_crypto_seconds if complete else None,
            "peak_memory_mb": run.peak_memory_mb if complete else None,
            "failure_evidence": failure, "failure_is_historical": bool(failure and complete),
            "parameters": config})

    aggregate, methods = [], {}
    for method in ALL_METHODS:
        method_rows = [row for row in per_task if row["method"] == method]
        observed = [row for row in method_rows if row["final_round_available"]]
        full = len(observed) == len(method_rows)
        healthy_count = sum(row["healthy"] is True for row in observed)
        attacked = [row for row in observed if row["malicious_ratio"] > 0]
        clean = [row for row in observed if row["malicious_ratio"] == 0]
        methods[method] = {"selected_candidate": selected[method], "completed_runs": len(observed),
            "expected_runs": len(method_rows), "healthy_runs": healthy_count,
            "status": "complete_healthy" if full and healthy_count == len(observed) else "complete_unhealthy" if full else "incomplete",
            "overall": {"final_accuracy": fmean(row["final_accuracy"] for row in observed),
                        "attack_final_asr": fmean(row["final_asr"] for row in attacked) if attacked else None,
                        "clean_accuracy": fmean(row["final_accuracy"] for row in clean) if clean else None}
                        if full else None}
        for partition, ratio in sorted({(row["partition"], row["malicious_ratio"]) for row in method_rows}):
            expected_rows = [row for row in method_rows if (row["partition"], row["malicious_ratio"]) == (partition, ratio)]
            valid = [row for row in expected_rows if row["final_round_available"]]
            accuracy = [row["final_accuracy"] for row in valid]
            asr = [row["final_asr"] for row in valid if row["final_asr"] is not None]
            aggregate.append({"method": method, "candidate_id": selected[method], "partition": partition,
                "malicious_ratio": ratio, "n": len(valid), "expected_n": len(expected_rows),
                "healthy_n": sum(row["healthy"] is True for row in valid),
                "status": "complete" if len(valid) == len(expected_rows) else "incomplete",
                "observed_seeds": [row["seed"] for row in valid],
                "missing_seeds": [row["seed"] for row in expected_rows if not row["final_round_available"]],
                "final_accuracy_mean": fmean(accuracy) if accuracy else None,
                "final_accuracy_sd": stdev(accuracy) if len(accuracy) > 1 else None,
                "final_asr_mean": fmean(asr) if asr else None,
                "final_asr_sd": stdev(asr) if len(asr) > 1 else None,
                "asr_role": "attacked" if ratio > 0 else "background_diagnostic"})

    # The new gate's public adapter is filled by its module, keeping this
    # report independent from old-schema promotion and scoring code.
    from cifar_adaptive_gate import describe_final_targets
    target_audit = describe_final_targets(spec, selected, results_by_candidate, tasks)
    target_audit["role"] = "descriptive_only_no_parameter_reselection_or_health_override"
    full = len(available) == len(tasks)
    all_attempted = all(row["status"] != "pending" for row in statuses)
    search_path = output / "search_summary.json"
    search_summary = json.loads(search_path.read_text(encoding="utf-8")) if search_path.is_file() else None
    summary = {"schema_version": 1, "model": protocol_descriptor(), "selected": selected,
        "experiment_type": "ours_favorable_conditions_selected_on_two_development_seeds",
        "search_summary": search_summary,
        "required_final_round": rounds, "formal_seeds": spec["final"]["seeds"],
        "status": "completed" if full and not health_failures else "completed_with_health_failures" if full else
                  "completed_with_task_failures" if all_attempted else "incomplete",
        "report_status": "completed" if full else "completed_partial",
        "full_execution_completed": full, "all_scheduled_tasks_attempted": all_attempted,
        "completed_tasks": len(available), "expected_tasks": len(tasks),
        "all_methods_healthy": full and not health_failures, "methods": methods,
        "health_failures": health_failures, "tasks": per_task, "ours_final_metric_gate": target_audit,
        "parameters_reselected": False, "official_test_used_for_selection": False,
        "missing_policy": "no_zero_imputation_no_last_round_substitution_no_incomplete_method_overall",
        "finite_health_failures_included": True,
        "sd_policy": "sample_SD_only_if_n_greater_than_one; one_seed_has_no_SD",
        "continuation_decision": continuation_decision,
        "validation_status": validation_summary.get("status")}
    destination = output / "final_results"
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "aggregate.csv", aggregate, list(aggregate[0]))
    _write_csv(destination / "per-task.csv", per_task, list(per_task[0]))
    round_fields = list(merged_rounds[0]) if merged_rounds else ["task_id", "candidate_id", "partition", "seed", "round", "accuracy", "attack_target_success_rate"]
    _write_csv(destination / "rounds.csv", merged_rounds, round_fields)
    write_json(destination / "selection_provenance.json", {"spec": spec, "selected": selected,
               "validation_summary": validation_summary, "continuation_decision": continuation_decision,
               "search_summary": search_summary})
    figures = _plots(destination, tasks, available, rounds=rounds, seeds=spec["final"]["seeds"])
    source_modules = {Path(__file__).name: _file_digest(Path(__file__))}
    for name in ("cifar_adaptive_gate.py", "cifar_resnet_gn.py"):
        path = Path(__file__).with_name(name)
        source_modules[name] = _file_digest(path)
    for name in ("manifest.json", "final_plan.json", "validation_summary.json", "continuation_decision.json",
                 "search_summary.json", "search_state.json", "best_parameters.json"):
        path = output / name
        if path.is_file():
            source_files[name] = _file_digest(path)
    audit = {"source_sha256": source_files, "report_source_sha256": source_modules,
        "inputs_sha256": {"spec": _digest(spec), "tasks": _digest(tasks), "statuses": _digest(statuses),
                           "selected": _digest(selected), "validation_summary": _digest(validation_summary),
                           "rounds": _digest(merged_rounds)},
        "complete_matrix_verified": full, "planned_matrix_verified": True,
        "completed_tasks": len(available), "expected_tasks": len(tasks),
        "missing_tasks": [row for row in per_task if not row["final_round_available"]],
        "statistical_policy": {key: summary[key] for key in ("missing_policy", "finite_health_failures_included", "sd_policy")}}
    write_json(destination / "data_audit.json", audit)
    _write_html(destination, summary, aggregate, validation_summary, figures)
    for relative, checksum in source_files.items():
        if _file_digest(output / relative) != checksum:
            raise ReportIntegrityError("source changed while generating final report")
    write_json(output / "final_summary.json", summary)
    return summary


def _write_html(destination, summary, aggregate, validation_summary, figures):
    summary, validation_summary = json_safe(summary), json_safe(validation_summary)
    def cell(value, *, percent=False):
        if value is None:
            return "—"
        return html.escape(f"{value:.2%}" if percent else str(value))
    rows = []
    for row in aggregate:
        rows.append("<tr>" + "".join("<td>" + value + "</td>" for value in (
            cell(LABELS[row["method"]]), cell(row["partition"]), cell(row["malicious_ratio"], percent=True),
            cell(f'{row["n"]}/{row["expected_n"]}'), cell(row["healthy_n"]),
            cell(row["final_accuracy_mean"], percent=True), cell(row["final_asr_mean"], percent=True),
            cell(row["final_accuracy_sd"], percent=True), cell(row["final_asr_sd"], percent=True))) + "</tr>")
    images = "".join(f'<figure><a href="{name}.svg"><img src="{name}.png" alt="{name}"></a></figure>' for name in figures)
    configs = [{"task_id": row["task_id"], "parameters": row["parameters"]} for row in summary["tasks"]]
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CIFAR ResNet18-GN 正式结果</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#192536}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:7px;text-align:right}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:14px}}img{{max-width:100%}}.notice{{background:#fff0c7;padding:16px}}</style>
<h1>CIFAR-10 · ResNet-18 + GroupNorm · 第{summary['required_final_round']}轮</h1>
<p class="notice">正式任务完整数 {summary['completed_tasks']}/{summary['expected_tasks']}；训练状态：{cell(summary['status'])}；报告状态：{cell(summary['report_status'])}。
报告生成不代表训练全部健康或Ours达标。完整且数值有效的健康失败结果保留；未完成任务留空，不以旧轮顶替最终轮。缺失方法不计算总体指标。单seed没有标准差（SD不可用），不构成跨seed稳定性证据。</p>
<p>实际formal seeds：{cell(summary['formal_seeds'])}。Clean场景ASR仅为背景误分类诊断。逐场景n为可用完整seed数，缺失时仅描述现有观察，存在幸存样本偏差。</p>
<p>实验条件在两个开发seed上以Ours相对优势为目标搜索，六法共用选定公共条件；本结果属于该条件选择下的比较。
每组公共条件内可调方法候选预算匹配，预算截止可能使不同公共条件的实际搜索次数不同。实际完整波次和TPE次数见下方搜索记录；不能把随机启动或未完成搜索称为已学得全局最优。</p>
<p><a href="aggregate.csv">场景统计CSV</a> · <a href="per-task.csv">逐任务CSV</a> · <a href="rounds.csv">逐轮CSV</a> · <a href="selection_provenance.json">参数与选择来源</a> · <a href="data_audit.json">来源审计</a> · <a href="../final_summary.json">最终汇总JSON</a></p>
<table><thead><tr><th>方法</th><th>划分</th><th>恶意比例</th><th>n/计划n</th><th>健康n</th><th>最终Acc</th><th>最终ASR</th><th>Acc SD</th><th>ASR SD</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{images}<h2>方法总体与完整性</h2><pre>{html.escape(json.dumps(summary['methods'],ensure_ascii=False,indent=2))}</pre>
<h2>最终双指标≤2个百分点：仅描述性审计</h2><pre>{html.escape(json.dumps(summary['ours_final_metric_gate'],ensure_ascii=False,indent=2))}</pre>
<details><summary>健康失败和未完成任务</summary><pre>{html.escape(json.dumps(summary['health_failures'],ensure_ascii=False,indent=2))}</pre></details>
<details><summary>验证选择来源与手动继续决定</summary><pre>{html.escape(json.dumps({'validation':validation_summary,'continuation':summary['continuation_decision']},ensure_ascii=False,indent=2))}</pre></details>
<details><summary>实际搜索预算与学习次数</summary><pre>{html.escape(json.dumps(summary['search_summary'],ensure_ascii=False,indent=2))}</pre></details>
<details><summary>正式参数（未用正式结果重选）</summary><pre>{html.escape(json.dumps(configs,ensure_ascii=False,indent=2))}</pre></details></html>'''
    (destination / "visualizations.html").write_text(content, encoding="utf-8")
