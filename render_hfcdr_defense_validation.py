#!/usr/bin/env python3
"""Unfiltered bounded-candidate validation report; no training or selection."""
from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
from statistics import mean, stdev

os.environ.setdefault('MPLCONFIGDIR', '/tmp/hfcdr-validation-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import hfcdr_validation_protocol as p
from hfcdr_validation_analysis import report
from hfcdr_health_context import context

COLORS = {'hfcdr': '#4D91BD', 'alignins': '#DC7477', 'vert': '#42A6A1',
          'tad': '#A98EC8', 'krum': '#E3AD63'}


def scalar(values, expected_n):
    """A missing replicate is never plotted as a complete candidate mean."""
    return dict(n=len(values), expected_n=expected_n, raw=values,
        mean=mean(values) if len(values) == expected_n else None,
        sd=stdev(values) if len(values) == expected_n and len(values) > 1 else None)


def plot_data(manifest, assessment, dataset, partition, metric):
    policy = manifest['policy']; seeds = policy['seeds']
    candidates = [(m, c['id']) for m, cs in policy['defense_candidates'].items() for c in cs]
    conditions = [('clean', 0.)] + [(a['name'], r) for r in policy['ratios'][1:] for a in policy['attacks']]
    result = []
    for condition, ratio in conditions:
        cells = []
        for method, candidate in candidates:
            tasks = [t for t in manifest['tasks'] if t['config']['dataset'] == dataset
                and t['config']['partition'] == partition and t['condition'] == condition
                and t['config']['malicious_ratio'] == ratio and t['config']['method'] == method
                and t['candidate_id'] == candidate]
            if len(tasks) != len(seeds) or {t['config']['seed'] for t in tasks} != set(seeds):
                raise ValueError('Unequal candidate replicate allocation')
            summaries = [assessment['summaries'][t['task_id']] for t in tasks
                         if t['task_id'] in assessment['summaries']]
            raw = [s['timings']['runtime_seconds'] if metric == 'runtime'
                   else 100 * s['tail10'][metric] for s in summaries]
            cells.append(dict(method=method, candidate=candidate,
                tasks=[s['task_id'] for s in summaries], healthy_n=sum(s['healthy'] for s in summaries),
                stats=scalar(raw, len(seeds))))
        result.append(dict(condition=condition, malicious_ratio=ratio, cells=cells))
    return dict(dataset=dataset, partition=partition, metric=metric, seeds=seeds, panels=result)


def draw(payload, folder):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    metric = payload['metric']
    ylabel = {'accuracy': 'Tail-10 validation accuracy (%)', 'asr': 'Tail-10 validation ASR (%)',
              'runtime': 'Runtime (s; shared GPU)'}[metric]
    for ax, panel in zip(axes.flat, payload['panels']):
        labels = []
        for i, cell in enumerate(panel['cells']):
            stats = cell['stats']; labels.append(cell['method'].upper() + '\n' + cell['candidate'])
            if stats['mean'] is None:
                ax.text(i, 0, f"n={stats['n']}", rotation=90, va='bottom', ha='center', fontsize=7)
                continue
            unhealthy = cell['healthy_n'] != stats['expected_n']
            ax.bar(i, stats['mean'], yerr=stats['sd'], capsize=2, width=.74,
                color=COLORS[cell['method']], edgecolor='#ad2433' if unhealthy else 'white',
                linewidth=1.4 if unhealthy else .5, hatch='//' if unhealthy else None)
        ax.set_xticks(range(len(labels)), labels, rotation=65, ha='right', fontsize=7)
        ax.set_title('Shared clean baseline' if panel['condition'] == 'clean' else
                     f"{panel['condition']} | {panel['malicious_ratio']:.0%} malicious")
        ax.set_ylim(bottom=0)
        if metric != 'runtime': ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True); ax.set_ylabel(ylabel)
    axes[1, 3].axis('off')
    axes[1, 3].legend(handles=[Patch(facecolor=c, label=m.upper()) for m, c in COLORS.items()],
                      loc='upper left', frameon=False)
    axes[1, 3].text(.02, .52, 'All candidate outcomes retained.\nMean requires both development seeds.\nError bars: sample SD (n-1).\nHatching: at least one raw health flag.\nSee report for warmup/attack context.\nMissing runs are not imputed.\nFixed methods are never retuned.\nFedAvg references: Stage A paired runs.',
                    transform=axes[1, 3].transAxes, fontsize=10, va='top', linespacing=1.7)
    fig.suptitle(f"{payload['dataset'].upper()} | {payload['partition']} | {ylabel}", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .95))
    name = f"{payload['dataset']}_{payload['partition']}_{metric}"
    fig.savefig(folder/(name+'.svg'), bbox_inches='tight',
                metadata={'Description': json.dumps(payload, sort_keys=True, allow_nan=False)})
    fig.savefig(folder/(name+'.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
    p.write(folder/(name+'.plot.json'), payload)
    return name


def render(study, destination, selection=None):
    study = p.safe_output(study); p.check(study)
    destination = Path(destination).resolve()
    if destination.exists() or study not in destination.parents:
        raise ValueError('Fresh report directory inside this study required')
    manifest = p.read(study/'manifest.json')
    if manifest['policy']['phase'] != 'defense_search': raise ValueError('Defense-search report only')
    assessment = report(study, manifest)
    if selection is not None and selection['defense_fingerprint'] != manifest['fingerprint']:
        raise ValueError('Selection belongs to another study')
    destination.mkdir(); folder = destination/'plots'; folder.mkdir(); figures = []
    for dataset in manifest['policy']['datasets']:
        for partition in manifest['policy']['partitions']:
            for metric in ('accuracy', 'asr', 'runtime'):
                figures.append(draw(plot_data(manifest, assessment, dataset, partition, metric), folder))
    p.write(destination/'report.json', assessment)
    p.write(destination/'selection.json', selection)
    health_context = [context(p.read(study/'tasks'/tid/'result.json'), summary)
                      for tid,summary in assessment['summaries'].items() if not summary['healthy']]
    p.write(destination/'health_context.json',health_context)
    rows = []
    for decision in (selection or {}).get('decisions', []):
        for candidate in decision['candidates']:
            values = [decision['dataset'], decision['method'], candidate['id'],
                'fixed' if decision['fixed_no_search'] else 'bounded search', candidate['full_runs'],
                f"{candidate['score']:.6f}", candidate['feasible'],
                decision['selected'] == candidate['id'], len(candidate['reasons'])]
            rows.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in values)+'</tr>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>防御验证完整结果</title>
<style>body{font-family:system-ui,sans-serif;max-width:1550px;margin:30px auto;padding:0 24px;background:#f5f7fa;color:#25374a}p{line-height:1.7}.card{background:white;border:1px solid #dce3ec;border-radius:8px;padding:18px;margin:20px 0}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child{text-align:left}a{color:#24658b}</style><h1>无密码学补充攻击 · 防御验证</h1>'''
    page += f'<div class="card"><p><b>已完成 {assessment["completed"]}/{assessment["expected_tasks"]}；允许独立验证：{(selection or {}).get("confirmation_allowed", False)}。</b></p>'
    page += '<p>HFCDR、AlignIns各最多4组有来源的候选；VERT、TAD、Krum固定，跳过搜索。每候选覆盖相同56个场景；未使用的预算不转移。FedAvg引用攻击筛选阶段的56个配对结果。所有结果来自训练集留出的验证集，未读取官方测试数据。</p>'
    page += '<p>每方法每数据集只选一套参数，跨分布、攻击和恶意比例共用。效用为攻击场景等权平均的0.5×Acc+0.5×(1−ASR)。可调方法须满足健康门槛，且各干净配对相对FedAvg的Acc下降≤3个百分点、背景ASR增加≤5个百分点。并列优先原基线。固定方法的不合格标记保留，不借机更换参数。</p>'
    page += '<p>TAD使用用户确认的PriorFreeTAD解释版本；不宣称原论文歧义已消除。Krum在60%/80%超过其理论保证范围；本阶段只验证0%/40%/80%，20%/60%留待后续完整矩阵。此页不构成正式测试结论。旧46项官方测试探索结果不参与选参。</p>'
    page += '<p>原健康指标按实验开始前指定的客户端身份统计。Krum在攻击启动前连续选择预定恶意身份时，也可能触发权重标记；这不证明攻击已造成失效。另表区分纯暖身、跨启动边界和纯攻击阶段的五轮窗口，原健康标记及选择规则均不更改。误剔除的分母是预先指定的诚实客户端数。</p>'
    page += '<p><a href="report.json">全部逐任务指标</a> · <a href="selection.json">全部候选分数、选择及未达标原因</a> · <a href="health_context.json">健康标记的发生阶段和解释</a> · <a href="visualization_audit.json">绘图凭据</a></p></div>'
    page += '<div class="card"><table><tr>'+''.join('<th>'+x+'</th>' for x in
        ('数据', '方法', '候选', '策略', '训练数', '效用', '满足门槛', '选定', '问题数'))+'</tr>'+''.join(rows)+'</table></div>'
    for name in figures:
        page += f'<div class="card"><h2>{escape(name)}</h2><a href="plots/{name}.svg"><img loading="lazy" src="plots/{name}.png" alt="{name}"></a><p><a href="plots/{name}.svg">SVG</a> · <a href="plots/{name}.plot.json">绘图原值</a></p></div>'
    (destination/'report.html').write_text(page+'</html>', encoding='utf-8')
    audit = dict(fingerprint=manifest['fingerprint'], plots=len(figures), complete_tasks=assessment['completed'],
        expected_seed_count=len(manifest['policy']['seeds']), official_test_used=False,
        all_outcomes_retained=True, plot_hashes={f.name:p.sha(f) for f in folder.glob('*.plot.json')},
        reader_sources={n:p.sha(p.ROOT/n) for n in ('render_hfcdr_defense_validation.py','hfcdr_health_context.py')})
    p.write(destination/'visualization_audit.json', audit)
    return audit
