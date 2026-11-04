"""Render a verified corrected CIFAR timing matrix with the MNIST plot template.

Read-only JSON inputs, all 180 observations retained, no training or pickle loads.
The destination must be new and separate from the accepted performance package.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
from html import escape
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

from render_cifar_best import checked_timing, dashboard, png_exports, with_metadata
from sm9rrsfl import visualization as style

SEEDS = (2026102801, 2026102802, 2026102803)
METHODS = ('sm9rrs', 'vert', 'alignins', 'ding13', 'krum', 'fedavg')
PARTITIONS = {'iid': ('iid', 'IID'), 'dirichlet': ('dirichlet_alpha_0_5', 'Non-IID（α=0.5）')}
RATIOS = (0., .2, .4, .6, .8)
METRICS = ('runtime_seconds', 'runtime_without_crypto_seconds', 'crypto_wall_seconds')
EXPECTED_FINGERPRINT = '7146c985051ac1a96120843c28816dd29c758f96cef05fdad1560db5ac14a3a1'
NOTE = ('MNIST 同一 run_measured_experiment 计时入口；数据加载在外，扣除 checkpoint I/O。'
        '扣除密码学时间仅减四段互不重叠的协议墙钟时间，未关闭密码学重跑。'
        '每项 150 轮；共享服务器实测，非独占硬件基准。')
LIMITATIONS = ('CIFAR 使用 CNN、150 轮、batch 50、攻击第 25 轮开始、10 个 stealth steps、SM9 workers=8；'
               '原 MNIST 为 50 轮、batch 32、第 12 轮开始、1 个 stealth step、SM9 workers=1。'
               '模型、数据、攻击及实际参与客户端数量不同，计时口径一致不代表工作量一致。'
               '方法按原配置进行隔离和撤销；耗时较短不能单独证明算法更优。'
               '30 个场景组内六法同 GPU 串行，方法顺序轮换，各位置出现 5 次；'
               '同组仍在不同时间测量，共享 CPU/GPU 负载是限制。'
               '三 seed 样本标准差不等于专门的硬件重复测速误差，不能据此推断显著性。')
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def finite(v):
    if isinstance(v, float): return math.isfinite(v)
    if isinstance(v, dict): return all(finite(x) for x in v.values())
    if isinstance(v, list): return all(finite(x) for x in v)
    return True


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows: groups[(row['partition'], row['malicious_ratio'], row['method'])].append(row)
    result = []
    for (partition, ratio, method), group in sorted(groups.items()):
        if {r['seed'] for r in group} != set(SEEDS) or len(group) != 3:
            raise ValueError('Every aggregate requires exactly three distinct seeds')
        item = dict(partition=partition, malicious_ratio=ratio, method=method, n=3)
        for metric in METRICS:
            values = [r[metric] for r in group]
            item[metric] = dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values))
        result.append(item)
    return result


def validate_rows(rows):
    expected = {(s, p, r, m) for s in SEEDS for p in PARTITIONS for r in RATIOS for m in METHODS}
    keys = [(r['seed'], r['partition'], r['malicious_ratio'], r['method']) for r in rows]
    if len(keys) != 180 or set(keys) != expected or len({r['task_id'] for r in rows}) != 180:
        raise ValueError('Exactly 180 distinct fixed matrix tasks required')
    for row in rows:
        vals = [row[k] for k in METRICS] + [row['checkpoint_io_seconds']]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in vals):
            raise ValueError('Invalid timing value')
        if row['runtime_seconds'] <= 0 or not math.isclose(
                row['runtime_seconds'] - row['crypto_wall_seconds'], row['runtime_without_crypto_seconds'], abs_tol=1e-6):
            raise ValueError('Disjoint crypto wall subtraction mismatch')


def load(source, expected_sha):
    p = source / 'analysis/timing_results.json'
    if digest(p) != expected_sha: raise ValueError('Timing report SHA mismatch')
    report, manifest = read(p), read(source / 'manifest.json')
    if report['source_fingerprint'] != EXPECTED_FINGERPRINT or manifest['fingerprint'] != EXPECTED_FINGERPRINT:
        raise ValueError('Wrong corrected timing study')
    if report['counts'] != {'complete': 180} or not report['execution_audit']['passed'] or not report['algorithm_binding_audit']['passed']:
        raise ValueError('Incomplete or unaudited study')
    if report['history']['corrected_fresh_attempts'] != 180 or report['history']['reused_results'] != 0 or report['history']['cumulative_actual_attempts'] != 276:
        raise ValueError('History provenance mismatch')
    rows = report['rows']; validate_rows(rows)
    calculated = aggregate(rows)
    by_key = {(r['partition'], r['malicious_ratio'], r['method']): r for r in report['aggregate']}
    if len(by_key) != 60: raise ValueError('Incomplete aggregate matrix')
    for r in calculated:
        old = by_key[(r['partition'], r['malicious_ratio'], r['method'])]
        if old['n'] != 3 or any(not math.isclose(r[m][k], old[m][k], rel_tol=0, abs_tol=1e-9) for m in METRICS for k in ('mean', 'sample_sd')):
            raise ValueError('Aggregate values differ from raw rows')
    for row in rows:
        folder = source / 'tasks' / row['task_id']; status = read(folder / 'status.json')
        for name, sha in status['artifacts'].items():
            if digest(folder / name) != sha: raise ValueError('Task artifact SHA mismatch: ' + row['task_id'])
        rounds, result, timing = (read(folder / f) for f in ('rounds.json', 'result.json', 'timing.json'))
        clock = checked_timing(timing)
        if status['completed_round'] != 150 or status['nonfinite_updates'] != 0 or [r['round'] for r in rounds] != list(range(151)) or not finite(rounds) or not finite(result):
            raise ValueError('Invalid 150-round trajectory')
        for metric in METRICS:
            if row[metric] != result[metric]: raise ValueError('Raw time differs from report')
        if clock['runtime_with_crypto_seconds'] != row['runtime_seconds'] or clock['runtime_without_crypto_seconds'] != row['runtime_without_crypto_seconds']:
            raise ValueError('Timing ledger differs')
        if row['health_reasons'] != status['health_reasons']: raise ValueError('Health marker lost')
    return report, report['aggregate']


def chart(rows, partition, metric, y_max, seed=None):
    group = [r for r in rows if r['partition'] == partition and (seed is None or r['seed'] == seed)]
    values, evidence = {}, []
    for ratio in RATIOS:
        values[f'{ratio * 100:.0f}%'] = {}
        for method in METHODS:
            items = [r for r in group if r['malicious_ratio'] == ratio and r['method'] == method]
            if len(items) != (3 if seed is None else 1): raise ValueError('Plot input matrix incomplete')
            v = statistics.mean(r[metric] for r in items)
            label = style.METHOD_LABELS[method]; values[f'{ratio * 100:.0f}%'][label] = v
            evidence.append(dict(method=method, label=label, ratio=ratio, value=v, n=len(items),
                sample_sd=statistics.stdev(r[metric] for r in items) if len(items)>1 else None,
                tasks=[r['task_id'] for r in items], raw_values=[r[metric] for r in items]))
    without = metric == 'runtime_without_crypto_seconds'
    subtitle = '扣除密码学耗时' if without else '总耗时（含密码学）'
    title = PARTITIONS[partition][1] + '：' + subtitle + (' · 三 seed 均值' if seed is None else f' · seed {seed}')
    svg = style._grouped_bar_chart(title=title, values=values, x_label='恶意节点比例', y_label='运行时间（秒）', y_max=y_max)
    root = ET.fromstring(svg)
    for el in root.findall('{*}text'):
        if el.get('font-size') == '10':
            el.text = f'{float(el.text):.1f}'
            el.set('transform', f'rotate(-60 {el.get("x")} {el.get("y")})'); el.set('text-anchor', 'start')
    bars = [el for el in root.findall('{*}rect') if el.get('height') not in ('100%', '14')]
    if len(bars) != 30: raise ValueError('MNIST bar template changed')
    for el, item in zip(bars, evidence):
        el.set('data-value', repr(item['value'])); el.set('data-method', item['method']); el.set('data-ratio', str(item['ratio']))
        ET.SubElement(el, '{'+NS+'}title').text = f'{item["label"]}: {item["value"]:.6f} s; n={item["n"]}; SD={item["sample_sd"]}'
    ET.SubElement(root, '{'+NS+'}desc').text = NOTE
    return with_metadata(ET.tostring(root, encoding='unicode'), dict(metric=metric, partition=partition, seed=seed,
                definition=NOTE, unit='seconds', bars=evidence, y_max=y_max))


def table(rows, columns):
    def cell(v):
        if isinstance(v, float): return f'{v:.6f}'
        if isinstance(v, list): return '; '.join(str(s) for s in v) or '—'
        return str(v)
    return '<div style="overflow:auto"><table><thead><tr>'+''.join('<th>'+escape(c)+'</th>' for c in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(cell(row.get(c,'')))+'</td>' for c in columns)+'</tr>' for row in rows)+'</tbody></table></div>'


def render(source, output, expected_sha, browser=None):
    report, aggregates = load(source, expected_sha)
    output.mkdir(parents=True, exist_ok=False)
    rows = report['rows']; write(output/'summary.json', rows); write(output/'aggregate.json', aggregates)
    import csv
    def csv_file(name, data):
        with (output/name).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader()
            writer.writerows({k: (';'.join(v) if isinstance(v,list) else v) for k,v in row.items()} for row in data)
    csv_file('summary.csv',rows)
    flat=[]
    for r in aggregates:
        item={k:r[k] for k in ('partition','malicious_ratio','method','n')}
        for m in METRICS:
            for stat in ('mean','sample_sd'):item[m+'_'+stat]=r[m][stat]
        flat.append(item)
    csv_file('aggregate.csv',flat)
    all_plots=[]
    for seed in SEEDS+(None,):
        folder=output/(f'seed_{seed}' if seed else 'seed_mean'); (folder/'plots').mkdir(parents=True)
        plots=[]
        for partition,(slug,label) in PARTITIONS.items():
            # Paired total/excluding panels have identical axes; mean is not a new task.
            ymax=max(r['runtime_seconds'] for r in rows if r['partition']==partition and (seed is None or r['seed']==seed))*1.30
            for metric,suffix,caption in [('runtime_seconds','runtime_overhead','总耗时'),('runtime_without_crypto_seconds','runtime_without_crypto','扣除密码学耗时')]:
                p=folder/'plots'/f'{slug}_{suffix}.svg';p.write_text(chart(rows,partition,metric,ymax,seed),encoding='utf-8')
                plots.append((label+' · '+caption,p.relative_to(folder).as_posix()))
                all_plots.append((f'{seed or "三 seed 均值"} · '+label+' · '+caption,p.relative_to(output).as_posix()))
        intro='<p>'+escape(NOTE)+'</p><p>均值和样本标准差见完整报告；所有健康失败原值照录。</p><p><a href="../report.html">完整原值及审计</a></p>'
        (folder/'visualizations.html').write_text(dashboard('CIFAR 同口径计时 · '+str(seed or '三 seed 均值'),plots,intro),encoding='utf-8')
    nav='<p>'+' · '.join(f'<a href="seed_{s}/visualizations.html">seed {s}</a>' for s in SEEDS)+' · <a href="seed_mean/visualizations.html">三 seed 均值</a></p>'
    intro='<p>'+escape(NOTE)+'</p>'+nav+'<p><a href="report.html">完整报告</a> · <a href="summary.csv">180项原值</a> · <a href="aggregate.csv">60组均值/样本SD/n</a></p>'
    (output/'visualizations.html').write_text(dashboard('CIFAR 同口径计时',all_plots[-4:],intro),encoding='utf-8')
    health=report['health_failures']
    if len(health) != 3 or any(h['method'] != 'ding13' or h['partition'] != 'dirichlet' or h['malicious_ratio'] != 0 or h['health_reasons'] != ['clean_false_revocation_rate'] for h in health):
        raise ValueError('Health findings changed: review the report narrative')
    text='<h2>完成与口径</h2><p>180/180 fresh，各150轮，0复用/失败/自动重试。180实际算法调用审计通过；健康检查177通过、3项TAD Non-IID clean标记失败，所有原值均保留。</p><p>'+escape(NOTE)+'</p>'
    text+='<h2>限制</h2><p>'+escape(LIMITATIONS)+'</p><p>本次辅助ASR仅用于诊断；cifar-best官方Acc/ASR与原bundle保持原样。原官方test与种子此前已见，验证数据曾参与开发；本次固定配置不据计时或test回流调参。</p>'
    text+='<h2>历史计算成本</h2><p>此前96次实际尝试含1次GPU1 OOM及17项错误TAD实现，均封存。新180项全部fresh，总计276次尝试；旧78项其他法完成数据也未复用。本次从唯一启动至结束持续运行，SSH监测断连没有触发重启。</p>'
    text+='<h2>健康失败（不删样本）</h2><p>三项均与原accepted同场景健康标记一致；151轮共同字段Acc、FP/TP、诚实权重损失及恶意权重质量逐值一致，非旧Ding13别名错误。核对证据见 provenance/health_comparison.json。</p>'+table(health,['task_id','health_reasons','runtime_seconds','runtime_without_crypto_seconds'])
    text+='<h2>全部60组统计：各组n=3，样本SD使用ddof=1</h2>'+table(flat,list(flat[0]))
    text+='<h2>全部180项原值（秒）</h2>'+table(rows,['task_id','seed','partition','malicious_ratio','method',*METRICS,'checkpoint_io_seconds','health_reasons'])
    text+='<h2>审计与来源</h2><p>来源报告SHA256：<code>'+expected_sha+'</code>。</p><p>下载及最终独立审计见 provenance/；原始结果的本地路径记录在 render_provenance.json。成功临时checkpoint已由公共入口清理，这不是完整断点备份。</p>'
    page=dashboard('CIFAR 六法同MNIST计时复测 · 完整报告',all_plots[-4:],intro)
    page=page.replace('</main>',text+'</main>').replace('</style>','table{border-collapse:collapse;font-size:12px}th,td{padding:7px;border:1px solid #ddd;white-space:nowrap}th{background:#e2e8f0}</style>')
    (output/'report.html').write_text(page,encoding='utf-8')
    write(output/'render_provenance.json',dict(source=str(source.resolve()),report_sha256=expected_sha,
        source_fingerprint=report['source_fingerprint'],timing_definition=report['timing'],history=report['history'],
        health_failures=health,plots=[p for _,p in all_plots],all_rows_retained=180,aggregate_n=3,sd_ddof=1,
        performance_artifacts_modified=False,checkpoint_backup=False))
    import os
    relative_source = Path(os.path.relpath(source, output)).as_posix()
    raw_link = '<p><a href="'+escape(relative_source+'/analysis/timing_results.json')+'">服务器原始完成报告（JSON）</a> · <a href="provenance/final_verification.json">独立核验记录</a></p>'
    rp = output/'report.html'; rp.write_text(rp.read_text(encoding='utf-8').replace('</main>', raw_link+'</main>'), encoding='utf-8')
    if browser:png_exports(output,browser)
    return output/'visualizations.html'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--png-browser',type=Path)
    a=p.parse_args();print(render(a.source,a.output,a.expected_sha256,a.png_browser))

if __name__=='__main__':main()
