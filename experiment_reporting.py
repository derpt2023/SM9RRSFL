#!/usr/bin/env python3
"""Rebuild offline six-method reports from completed CSV/JSON, without training.

Kept outside sm9rrsfl so reporting changes do not invalidate training manifests.
Raw observations, source manifests and checkpoints are never rewritten.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

METHODS = ("sm9rrs", "vert", "alignins", "krum", "ding13", "fedavg")
LABELS = dict(zip(METHODS, ("Ours", "VERT", "AlignIns", "Krum", "TAD", "FedAvg")))
METRICS = ("final_accuracy", "final_asr", "attack_accuracy", "attack_asr",
           "runtime_seconds", "runtime_without_crypto_seconds", "crypto_wall_seconds", "peak_rss_mib")


class ReportIncompleteError(ValueError):
    """The available observations cannot support a complete report."""


def check_dependencies():
    for module in ("numpy", "matplotlib"):
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise RuntimeError("报告依赖缺失；请运行 python -m pip install -r requirements.txt，"
                               "再使用 --report-only 补报，无须重新训练。") from exc


def _require(condition, message):
    if not condition:
        raise ReportIncompleteError(message)


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReportIncompleteError(f"Cannot read report input: {path}: {exc}") from exc


def _csv(path):
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ReportIncompleteError(f"Cannot read report input: {path}: {exc}") from exc


def _number(value, context, *, probability=False):
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ReportIncompleteError(f"Missing/non-numeric observation: {context}") from exc
    _require(math.isfinite(result), f"Non-finite observation: {context}")
    if probability:
        _require(0 <= result <= 1, f"Out-of-range probability: {context}: {result}")
    return result


def _group(row):
    return (row["partition"], float(row["dirichlet_alpha"]), int(row["num_clients"]))


def _key(row):
    return (*_group(row), row["method"], float(row["malicious_ratio"]), int(row["seed"]))


def _index(rows, key, context):
    result = {}
    for row in rows:
        identity = key(row)
        _require(identity not in result, f"Duplicate {context}: {identity}")
        result[identity] = row
    return result


def _same(actual, expected, context):
    if isinstance(expected, bool):
        equal = str(actual).lower() == str(expected).lower()
    elif isinstance(expected, (int, float)):
        equal = math.isclose(_number(actual, context), expected, rel_tol=1e-10, abs_tol=1e-12)
    else:
        equal = str(actual) == str(expected)
    _require(equal, f"Observation/plan mismatch: {context}: {actual!r} != {expected!r}")


def _mean_sd(values):
    values = list(values)
    _require(bool(values), "Cannot average an empty observation group")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def load_completed_study(output):
    """Validate the declared final matrix, every round and merged observations."""
    output = Path(output).resolve()
    manifest = _json(output / "manifest.json")
    plan = _json(output / "final_plan.json")
    final = _json(output / "final_summary.json")
    _require(final.get("full_execution_completed") is True,
             "Formal execution is incomplete; no complete mean report was generated")
    _require(plan.get("manifest_fingerprint") == manifest.get("fingerprint"),
             "Final plan belongs to a different manifest")
    spec = manifest["spec"]
    tasks = plan["tasks"]
    _require(bool(tasks), "Final plan contains no tasks")
    planned = _index(tasks, lambda t: _key(t["config"]), "planned run")
    by_id = _index(tasks, lambda t: t["task_id"], "planned task id")
    summaries = _index(_csv(output / "final_results/summary.csv"), _key, "summary row")
    statuses = _index(final["tasks"], lambda t: t["task_id"], "task status")
    _require(set(statuses) == set(by_id), "Final task status matrix differs from plan")
    _require(set(summaries) == set(planned), "Final summary CSV has missing or extra runs")
    seeds = sorted(int(s) for s in spec["final"]["seeds"])
    _require(len(set(seeds)) == len(seeds), "Duplicate final seeds")
    methods = [m for m in METHODS if any(t["method"] == m for t in tasks)]
    _require(len(methods) == len({t["method"] for t in tasks}), "Unknown method in final plan")
    _require(set(methods) == set(spec["candidates"]), "Final plan is missing a declared method")
    shared = spec["shared_parameters"]
    rounds = int(shared["rounds"])
    attack_start = int(shared["attack_start_round"])
    _require(1 <= attack_start <= rounds, "Report requires a declared attack window within the run")
    declared = set()
    for scenario in spec["final"]["scenarios"]:
        config = dict(shared, **scenario)
        for method, seed in itertools.product(methods, seeds):
            declared.add(_key(dict(config, method=method, seed=seed)))
    _require(set(planned) == declared, "Final plan does not cover the manifest's full seed/scenario matrix")
    expected_count = manifest.get("final_run_count", len(declared))
    _require(expected_count == len(planned), "Manifest final run count differs from final plan")
    per_round = _index(_csv(output / "final_results/rounds.csv"),
                       lambda r: (*_key(r), int(r["round"])), "round observation")
    expected_rounds = {(*key, r) for key in planned for r in range(rounds + 1)}
    _require(set(per_round) == expected_rounds,
             "Round CSV is incomplete or contains unexpected rows (expected rounds 0 through final)")
    group_keys = sorted({_group(t["config"]) for t in tasks}, key=lambda g: (g[0] != "iid", g))
    group_slugs = {g: f"{g[0]}_alpha{g[1]:g}_clients{g[2]}".replace(".", "p") for g in group_keys}
    groups = [{"slug": group_slugs[g], "label": ("IID" if g[0] == "iid" else f"Dirichlet (alpha={g[1]:g})")
               + f" | {g[2]} clients", "partition": g[0], "dirichlet_alpha": g[1], "num_clients": g[2],
               "ratios": sorted({k[4] for k in planned if k[:3] == g})} for g in group_keys]
    runs, curves = [], []
    for key, task in planned.items():
        config, row, state = task["config"], summaries[key], statuses[task["task_id"]]
        context = task["task_id"]
        _require(state["status"] == "complete" and state.get("error") is None
                 and state.get("failure_record") is None, f"Task not completed: {context}")
        _require(_json(output / "tasks" / context / "task.json") == task,
                 f"Task identity differs from final plan: {context}")
        metrics = _json(output / "tasks" / context / "metrics.json")
        _require(metrics == state["metrics"], f"Task health metrics differ from final summary: {context}")
        for field, expected in config.items():
            if field not in ("device", "sm9_workers"):
                _same(row.get(field), expected, f"{context}.{field}")
        for field in ("rounds", "attack_start_round", "attack_source_label", "attack_target_label", "attack_target_count"):
            _same(config[field], shared[field], f"shared {field}")
        _same(row["stopped_round"], rounds, f"{context}.stopped_round")
        _same(metrics["stopped_round"], rounds, f"{context}.health.stopped_round")
        _same(row["effective_attack_start_round"], attack_start, f"{context}.effective_attack_start_round")
        healthy, reasons = metrics.get("healthy"), metrics.get("reasons")
        _require(isinstance(healthy, bool) and isinstance(reasons, list) and healthy == (len(reasons) == 0),
                 f"Inconsistent health verdict: {context}")
        raw = []
        for r in range(rounds + 1):
            obs = per_round[(*key, r)]
            accuracy = _number(obs["accuracy"], f"{context}.round{r}.accuracy", probability=True)
            asr = _number(obs["attack_target_success_rate"], f"{context}.round{r}.asr", probability=True)
            _same(obs["attack_active"], key[4] > 0 and r >= attack_start, f"{context}.round{r}.attack_active")
            raw.append({"group_slug": group_slugs[key[:3]], "method": key[3], "malicious_ratio": key[4],
                        "seed": key[5], "round": r, "accuracy": accuracy, "asr": asr})
        _same(row["final_accuracy"], raw[-1]["accuracy"], f"{context}.final_accuracy")
        _same(row["final_attack_target_success_rate"], raw[-1]["asr"], f"{context}.final_asr")
        _same(metrics["final_accuracy"], raw[-1]["accuracy"], f"{context}.health.final_accuracy")
        attack_accuracy = statistics.mean(o["accuracy"] for o in raw[attack_start:]) if key[4] > 0 else None
        attack_asr = statistics.mean(o["asr"] for o in raw[attack_start:]) if key[4] > 0 else None
        if key[4] > 0:
            _same(metrics["attack_mean_accuracy"], attack_accuracy, f"{context}.attack_mean_accuracy")
            _same(metrics["attack_mean_asr"], attack_asr, f"{context}.attack_mean_asr")
        runtime = _number(row["runtime_seconds"], f"{context}.runtime_seconds")
        no_crypto = _number(row["runtime_without_crypto_seconds"], f"{context}.runtime_without_crypto_seconds")
        crypto = _number(row["crypto_wall_seconds"], f"{context}.crypto_wall_seconds")
        rss = _number(row["peak_memory_mb"], f"{context}.peak_memory_mb")
        _require(min(runtime, no_crypto, crypto, rss) >= 0, f"Negative timing/memory observation: {context}")
        _same(no_crypto, max(0, runtime - crypto), f"{context}.runtime_subtraction")
        _same(row["nonfinite_updates"], metrics["nonfinite_updates"], f"{context}.nonfinite_updates")
        runs.append({"task_id": context, "candidate_id": task["candidate"]["candidate_id"],
                     "group_slug": group_slugs[key[:3]], "method": key[3], "malicious_ratio": key[4], "seed": key[5],
                     "healthy": healthy, "failure_reasons": reasons, "nonfinite_updates": int(row["nonfinite_updates"]),
                     "final_accuracy": raw[-1]["accuracy"], "final_asr": raw[-1]["asr"],
                     "attack_accuracy": attack_accuracy, "attack_asr": attack_asr,
                     "runtime_seconds": runtime, "runtime_without_crypto_seconds": no_crypto,
                     "crypto_wall_seconds": crypto, "peak_rss_mib": rss})
        curves.extend(raw)
    actual_failures = {r["task_id"]: r["failure_reasons"] for r in runs if not r["healthy"]}
    reported_failures = {r["task_id"]: r["reasons"] for r in final["health_failures"]}
    _require(actual_failures == reported_failures, "Health failure list differs from individual task metrics")
    for method in methods:
        observed = [r for r in runs if r["method"] == method]
        info = final["methods"][method]
        _require(info["expected_runs"] == info["completed_runs"] == len(observed)
                 and info["healthy_runs"] == sum(r["healthy"] for r in observed), f"Method counts inconsistent: {method}")
    return {"dataset_label": {"cifar10": "CIFAR-10", "mnist": "MNIST"}.get(spec["dataset"]["name"], spec["dataset"]["name"]),
            "seeds": seeds, "rounds": rounds, "attack_start": attack_start,
            "target_source": shared["attack_source_label"], "target_label": shared["attack_target_label"],
            "target_count": shared["attack_target_count"], "methods": methods, "groups": groups,
            "runs": runs, "raw_curves": curves, "health_failed_runs": len(actual_failures),
            "manifest": manifest, "final_summary": final, "output": str(output)}


def aggregate_study(study, seeds=None):
    """Aggregate corresponding seeds, using sample SD; n=1 has no SD."""
    result = dict(study)
    result["seeds"] = list(study["seeds"] if seeds is None else seeds)
    _require(bool(result["seeds"]) and set(result["seeds"]) <= set(study["seeds"]), "Invalid report seeds")
    result["runs"] = [r for r in study["runs"] if r["seed"] in result["seeds"]]
    result["health_failed_runs"] = sum(not r["healthy"] for r in result["runs"])
    buckets = defaultdict(list)
    for row in result["runs"]:
        buckets[(row["group_slug"], row["method"], row["malicious_ratio"])].append(row)
    scenarios = []
    for (group, method, ratio), rows in buckets.items():
        _require(sorted(r["seed"] for r in rows) == sorted(result["seeds"]), "Incomplete scenario seed group")
        record = {"group_slug": group, "method": method, "malicious_ratio": ratio, "n": len(rows),
                  "failed_runs": sum(not r["healthy"] for r in rows),
                  "failure_reasons": sorted({reason for r in rows for reason in r["failure_reasons"]})}
        for metric in METRICS:
            values = [r[metric] for r in rows]
            if all(v is None for v in values):
                mean = sd = None
            else:
                _require(all(v is not None for v in values), f"Partially missing metric: {metric}")
                mean, sd = _mean_sd(values)
            record[metric + "_mean"], record[metric + "_sd"] = mean, sd
        scenarios.append(record)
    buckets.clear()
    for row in study["raw_curves"]:
        if row["seed"] in result["seeds"]:
            buckets[(row["group_slug"], row["method"], row["malicious_ratio"], row["round"])].append(row)
    curves = []
    for (group, method, ratio, round_), rows in buckets.items():
        _require(len(rows) == len(result["seeds"]), "Incomplete curve seed group")
        record = {"group_slug": group, "method": method, "malicious_ratio": ratio, "round": round_, "n": len(rows)}
        for metric in ("accuracy", "asr"):
            record[metric + "_mean"], record[metric + "_sd"] = _mean_sd(r[metric] for r in rows)
        curves.append(record)
    result.update(scenarios=scenarios, curves=curves)
    return result


def _write_csv(path, rows, empty_fields=()):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else list(empty_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def _table(headers, rows):
    return "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def _format(mean, sd=None, scale=1):
    if mean is None:
        return "—"
    return f"{mean * scale:.2f}" + (f" ± {sd * scale:.2f}" if sd is not None else "")


def _validation_gate_html(data):
    final_only = "validation_final_metric_gate" in data["final_summary"]
    gate = data["final_summary"].get("validation_final_metric_gate" if final_only else "validation_mnist_target_gate")
    if not gate:
        return ""
    policy = data["manifest"]["spec"]["performance_target"]
    relative_asr = data["manifest"]["spec"].get("asr_target")
    target = data["final_summary"].get("validation_ours_target", {})
    relative_accuracy = data["manifest"]["spec"].get("accuracy_target")
    if relative_accuracy:
        return ("<section><h2>验证推进条件：最终 Accuracy / ASR 距六方案最优值不超过2个百分点</h2>"
            "<p>逐 seed、逐场景比较第100轮：最高 Accuracy 减去 Ours Accuracy ≤ 2 个百分点；"
            "受攻击场景同时要求 Ours ASR 减去最低 ASR ≤ 2 个百分点。恰好相等也通过。"
            "两项最优值可来自不同方法，无攻击 ASR 仅作诊断；过程均值、末尾窗口和峰值不决定性能达标。"
            "六法使用各自固定入选候选；完整且数值有效的健康失败任务也参与。缺失仅排除对应任务参照，"
            "并标注比较不完整，不能声称完整六方法目标已通过。</p>"
            + _table(["项目", "验证记录"], [
                ["推进状态", gate.get("status", "—")],
                ["实际推进分支", gate.get("selected_pass_route", "—")],
                ["Accuracy 相对最高值通过", target.get("accuracy_maximum_target_passed", False)],
                ["ASR 相对最低值通过", target.get("asr_minimum_target_passed", False)],
                ["完整逐场景目标通过", target.get("full_target_passed", False)],
                ["均值双优作为硬门槛", gate.get("require_mean_dual_best", False)],
                ["比较范围", gate.get("comparison_scope", "—")]])
            + "<p>这是验证阶段推进条件，不保证正式测试排名。Ours须通过完整健康检查；"
            "其它方案保留原健康最高Score、完整失败最高raw Score、固定fallback顺序。</p></section>")
    rows = [["推进状态", gate.get("status", "—")],
            ["实际推进分支", gate.get("selected_pass_route", gate.get("pass_route", "—"))],
            ["完整逐场景目标通过", target.get("full_target_passed", False)],
            ["相对 VERT 目标", target.get("relative_target_status", "unassessed")],
            ["VERT 参照健康通过", gate.get("reference_health_qualified", "—")],
            ["均值双优作为硬门槛", gate.get("require_mean_dual_best", False)],
            ["验证均值双优", target.get("mean_dual_best_passed", "—")],
            ["比较范围", gate.get("comparison_scope", "—")]]
    if relative_asr:
        description = ("<section><h2>验证推进条件：最终 ASR 接近六方案最低值</h2>"
            "<p>逐 seed、逐受攻击场景比较第100轮：Ours ASR 减去该任务可用六方案最低 ASR，"
            f"须严格小于 {relative_asr['max_gap'] * 100:g} 个百分点；恰好相等不通过。"
            "不设5%的绝对上限。各基线固定使用验证选定候选，数值有效且完整的健康失败任务也参与；"
            "只在对应任务排除缺失基线，不阻塞推进，但比较不完整时不声称通过完整六方案目标。"
            "过程均值、末尾窗口和峰值仅作诊断。")
    else:
        description = ("<section><h2>验证推进条件：最终轮逐场景目标</h2>"
            f"<p>逐 seed、逐场景仅检查第 {data['manifest']['spec']['selection_metrics']['round']} 轮 Accuracy 与 ASR："
            f"受攻击场景最终 ASR ≤ {policy['max_asr'] * 100:g}%；"
            "过程均值、末尾窗口和峰值仅作诊断，不决定性能达标。"
            if final_only else "<section><h2>验证推进条件：MNIST 逐场景目标</h2>"
            f"<p>逐 seed、逐场景检查攻击窗口、末尾 {policy['tail_rounds']} 轮和最终轮次："
            f"ASR ≤ {policy['max_asr'] * 100:g}%，峰值 ≤ {policy['max_peak_asr'] * 100:g}%；"
                )
    return (description + f"有可评分 VERT 时，准确率落后 ≤ {policy['accuracy_gap'] * 100:g} 个百分点，"
            f"ASR 高出 ≤ {policy['asr_gap'] * 100:g} 个百分点。</p>"
            + _table(["项目", "验证记录"], rows)
            + "<p>这些是验证阶段的推进记录，不是正式测试最优性的保证。失败但可评分的 VERT 仍参与相对比较；"
            "无可评分 VERT 时，相对要求保留为未评估，不能宣称完整相对目标已通过。"
            "其它方案没有健康候选时，保留其最高完整 raw Score 候选及失败标记。</p></section>")


def _html(data, figures, *, mean_page):
    final_only = data["manifest"]["spec"].get("selection_metrics", {}).get("asr") == "final_round"
    asr_label = "最终 ASR 均值 ± seed SD (%)" if final_only else "攻击窗口 ASR 均值 ± seed SD (%)"
    asr_note = "每个任务仅取最终轮 ASR。过程统计仅作诊断。" if final_only else "先在单任务攻击窗口内求均值。"
    title = f"{data['dataset_label']} · " + ("三种子均值报告" if len(data['seeds']) == 3 else "实验报告")
    seed_text = ", ".join(map(str, data["seeds"]))
    prefix = "mean_plots/" if mean_page else "plots/"
    navigation = ("<a href='mean_plots/mean_figures.pdf'>下载均值图 PDF 合集</a> · " + " · ".join(
        f"<a href='seed_{s}/visualizations.html'>Seed {s}</a>" for s in data["seeds"])) if mean_page else "<a href='../visualizations.html'>返回均值汇总</a>"
    counts, overall = [], []
    for method in data["methods"]:
        rows = [r for r in data["runs"] if r["method"] == method]
        info = data["final_summary"]["methods"][method]
        counts.append([LABELS[method], info["selected_candidate"], len(rows), sum(r["healthy"] for r in rows),
                       sum(not r["healthy"] for r in rows), sum(r["nonfinite_updates"] for r in rows),
                       info.get("validation_selection_status", "—")])
        seed_acc, seed_asr = [], []
        for seed in data["seeds"]:
            selected = [r for r in rows if r["seed"] == seed]
            seed_acc.append(statistics.mean(r["final_accuracy"] for r in selected))
            key = "final_asr" if final_only else "attack_asr"
            attacked = [r[key] for r in selected if r["malicious_ratio"] > 0 and r[key] is not None]
            if attacked:
                seed_asr.append(statistics.mean(attacked))
        overall.append([LABELS[method], _format(*_mean_sd(seed_acc), scale=100),
                        _format(*_mean_sd(seed_asr), scale=100) if seed_asr else "—"])
    failures = [[r["task_id"], LABELS[r["method"]], r["seed"], f"{100*r['malicious_ratio']:g}%",
                 ", ".join(r["failure_reasons"]), r["nonfinite_updates"]] for r in data["runs"] if not r["healthy"]]
    cards = "".join(f"<section class='figure' id='{escape(f['name'])}'><h2>{escape(f['title'])}</h2>"
                    f"<p><a href='{prefix}{escape(str(f['svg']))}'>SVG 矢量图</a> · "
                    f"<a href='{prefix}{escape(str(f['png']))}'>PNG 预览</a></p>"
                    f"<img loading='lazy' src='{prefix}{escape(str(f['svg']))}' alt='{escape(f['title'])}'></section>" for f in figures)
    scenario_rows = [[s["group_slug"], LABELS[s["method"]], f"{s['malicious_ratio']*100:g}%", s["n"], s["failed_runs"],
                      _format(s["final_accuracy_mean"], s["final_accuracy_sd"], 100),
                      _format(s["final_asr_mean"], s["final_asr_sd"], 100),
                      _format(s["attack_asr_mean"], s["attack_asr_sd"], 100)] for s in data["scenarios"]]
    downloads = ("<p>可复核数据：" + " · ".join(f"<a href='mean_plots/{f}'>{f}</a>" for f in
                 ("scenario_mean_sd.csv", "curve_mean_sd.csv", "per_run.csv", "health_failures.csv", "data_audit.json")) +
                 " · <a href='summary.csv'>原始 summary.csv</a> · <a href='rounds.csv'>原始 rounds.csv</a></p>") if mean_page else ""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title><style>body{{margin:auto;padding:32px;max-width:1440px;background:#f5f7fa;color:#202936;font:16px/1.65 system-ui,sans-serif}}h1,h2{{line-height:1.3}}a{{color:#175c9f}}section,.panel{{background:white;border:1px solid #dce2eb;border-radius:10px;padding:24px;margin:24px 0}}.notice{{border-left:5px solid #b57416;background:#fff7e6;padding:18px}}img{{display:block;width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #ddd;white-space:nowrap}}th{{background:#eaf0f6}}.scroll{{overflow-x:auto}}code{{background:#eee;padding:2px 4px}}.muted{{color:#596577}}</style></head><body>
<h1>{escape(title)}</h1><p>{navigation}</p>
<p>正式 seed：{seed_text}；本页 {len(data['runs'])} 组完整任务；通信轮次 0–{data['rounds']}；攻击窗口 {data['attack_start']}–{data['rounds']}（含端点）。目标 {data['target_source']} → {data['target_label']}，{data['target_count']} 个评价目标样本。</p>
<div class="notice"><strong>执行完成 ≠ 全部健康通过。</strong>本页保留全部已记录的完整运行，其中 {data['health_failed_runs']} 组未通过健康检查。图表中的有限观测值仍纳入均值；未筛掉失败 seed，未插补缺失值。结果反映当前实现与配置，不能据此断言原论文算法普遍失效。</div>
<section><h2>完成状态与选参来源</h2>{_table(['方法','固定候选','完整运行','健康通过','健康失败','非有限更新数','验证选参状态'], counts)}
<p>best_scored_unqualified 表示验证候选未通过健康门槛，但按预设规则从完整可评分候选中选出最高 raw Score；不代表健康通过。正式结果不参与重选参数。原始 ours_target 中的 incomplete 属于过滤不健康参照后的比较状态，不等于任务未运行完。</p></section>
{_validation_gate_html(data)}
<section><h2>总体指标</h2>{_table(['方法','最终准确率均值 ± seed SD (%)',asr_label], overall)}
<p>先对每个 seed 的各场景等权平均，再对 seed 求均值与样本标准差。准确率包含干净和受攻击场景；ASR 仅包含恶意比例大于 0 的场景，{asr_note}高 ASR 表示攻击更成功；仅有相对优势不代表绝对攻击成功率足够低。</p></section>
<section><h2>如何理解阴影、误差棒与散点</h2><p>曲线为同一轮次跨 seed 的均值；阴影和带端帽的误差棒均为均值 ± 1 个样本标准差（ddof=1），表示不同随机种子之间的离散程度，并非 95% 置信区间。柱高为均值；每个灰色散点为一个 seed 的原始测量，轻微横向错开只为避免重叠。单 seed 页面不画标准差。</p>
<p>准确率和 ASR 图的可视范围限定在 0–100%；落在范围外的误差带会被裁切，CSV 中的真实标准差不裁切。0% 条件的目标误分类率属于无攻击背景，不能解释为发生了攻击。竖虚线标记攻击开始轮次。未进行曲线平滑。</p>
<p>运行耗时来自本次实际并行执行日志。扣除密码耗时图使用 runtime_seconds − crypto_wall_seconds，是事后统计扣除，并非关闭密码模块后重跑。检查点 I/O 单独记录，已不计入 runtime_seconds。设备、并行作业和运行顺序影响时间，不能把它当严格控制环境的速度基准。RSS 是进程峰值常驻内存（MiB），不是 GPU 显存或算法独占内存。</p>
<p>论文插图：LaTeX 优先 PDF；支持 SVG 的 Word 版本可插入 SVG；不建议截图。SVG 与 PDF 均为矢量输出，PNG 供预览。PDF 合集可按页提取或用 LaTeX 的 page 参数选择。图注应说明 seed、n、样本 SD、攻击窗口和健康失败保留策略。</p>{downloads}</section>
{cards}<section><h2>场景统计明细</h2>{_table(['分区','方法','恶意比例','n','健康失败','最终准确率 (%)','最终目标误分类率 (%)','攻击窗口 ASR (%)'], scenario_rows)}</section>
<section><h2>健康失败明细</h2>{_table(['任务','方法','seed','恶意比例','原因','非有限更新数'], failures) if failures else '<p>本页所有任务通过已记录的健康检查。</p>'}
<p>nonfinite_updates：执行中出现 NaN/Inf 更新；不意味着每个最终指标都是 NaN。all_honest_revoked：全部诚实客户端已被撤销。一次任务可同时命中多个原因。健康失败任务即使耗时较短，也不能被解释为成功完成相同有效训练后的性能优势。</p></section>
<p class="muted">本页根据完成后的 CSV/JSON 重建；HTML、SVG、PNG 和 PDF 可离线查看。生成时间：{datetime.now(timezone.utc).isoformat()}。</p></body></html>"""


def generate_report(output):
    """Validate first, render in staging, then publish derived report files."""
    check_dependencies()
    source_names = ["manifest.json", "final_plan.json", "final_summary.json", "final_results/summary.csv", "final_results/rounds.csv"]
    def source_hashes():
        return {name: hashlib.sha256((Path(output) / name).read_bytes()).hexdigest() for name in source_names}
    original_hashes = source_hashes()
    study = load_completed_study(output)
    data = aggregate_study(study)
    from experiment_report_plots import render_figures
    destination = Path(output).resolve() / "final_results"
    with tempfile.TemporaryDirectory(prefix=".report-", dir=destination) as temporary:
        staging = Path(temporary)
        mean_dir = staging / "mean_plots"
        mean_dir.mkdir()
        figures = render_figures(data, mean_dir)
        _write_csv(mean_dir / "scenario_mean_sd.csv", data["scenarios"])
        _write_csv(mean_dir / "curve_mean_sd.csv", data["curves"])
        _write_csv(mean_dir / "per_run.csv", data["runs"])
        _write_csv(mean_dir / "health_failures.csv", [r for r in data["runs"] if not r["healthy"]], data["runs"][0].keys())
        audit = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": data["dataset_label"],
                 "seeds": data["seeds"], "run_count": len(data["runs"]), "round_row_count": len(data["raw_curves"]),
                 "health_failed_runs": data["health_failed_runs"], "complete_matrix_verified": True,
                 "all_health_failed_observations_retained": True, "standard_deviation_ddof": 1,
                 "singleton_sd": None, "attack_window_inclusive": [data["attack_start"], data["rounds"]],
                 "source_sha256": original_hashes,
                 "figures": figures}
        (mean_dir / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for seed in data["seeds"]:
            individual = aggregate_study(study, [seed])
            seed_dir = staging / f"seed_{seed}"
            (seed_dir / "plots").mkdir(parents=True)
            seed_figures = render_figures(individual, seed_dir / "plots", include_pdf=False, png_dpi=120)
            (seed_dir / "visualizations.html").write_text(_html(individual, seed_figures, mean_page=False), encoding="utf-8")
        (staging / "visualizations.html").write_text(_html(data, figures, mean_page=True), encoding="utf-8")
        (staging / "visualized.html").write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=visualizations.html"><a href="visualizations.html">Open report</a>', encoding="utf-8")
        _require(source_hashes() == original_hashes, "Source results changed during rendering; rerun reporting after training finishes")
        # Publish the main index last; a rendering failure leaves any old report untouched.
        files = sorted((p for p in staging.rglob("*") if p.is_file()), key=lambda p: p == staging / "visualizations.html")
        for path in files:
            target = destination / path.relative_to(staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)
    return {"html_path": str(destination / "visualizations.html"),
            "pdf_path": str(destination / "mean_plots/mean_figures.pdf"),
            "run_count": len(data["runs"]), "health_failed_runs": data["health_failed_runs"], "figure_count": len(figures)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Completed study directory containing manifest.json")
    args = parser.parse_args(argv)
    try:
        result = generate_report(args.output)
    except (ReportIncompleteError, RuntimeError, OSError, KeyError, ValueError) as exc:
        print(f"REPORT_FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
