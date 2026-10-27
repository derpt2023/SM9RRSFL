"""Render an audited remote H84 export without modifying scientific evidence."""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import tempfile

import assess_h84_krum_reference as amended
import run_cifar_h84_full_validation as full

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / full.OUTPUT / 'export/final_audited_bundle.json'
LABELS = {'iid': 'IID', 'dirichlet': 'Non-IID (Dirichlet alpha=0.5)'}
STYLES = dict(FedAvg=('#777777', 's'), H84=('#b2182b', 'o'),
              TAD=('#16866d', 'D'), VERT=('#2166ac', '^'),
              AlignIns=('#9467bd', 'v'), Krum=('#d17911', 'P'))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def stored_bytes(content):
    # The frozen writers use this exact serialization, including a final newline.
    return (json.dumps(content, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def validate(bundle):
    if bundle['schema'] != 'h84-complete-audited-export-v1':
        raise ValueError('Unknown export schema')
    files = bundle['files']
    for name in ('manifest.json', 'reference_lock.json', 'analysis/full_validation.json',
                 'analysis/krum_reference_exception_assessment.json'):
        if digest(stored_bytes(files[name]['content'])) != files[name]['sha256']:
            raise ValueError('Exported evidence digest differs: ' + name)
    report = files['analysis/full_validation.json']['content']
    latest = files['analysis/krum_reference_exception_assessment.json']['content']
    protocol = bundle['amendment']
    if report['fingerprint'] != protocol['frozen_matrix_fingerprint']:
        raise ValueError('Matrix fingerprint differs')
    if latest['provenance']['original_report_sha256'] != files['analysis/full_validation.json']['sha256']:
        raise ValueError('Amended report is not derived from this strict report')
    computed = amended.apply_amendment(report, protocol)
    if computed != {k: v for k, v in latest.items() if k != 'provenance'}:
        raise ValueError('Latest user decision cannot be reproduced')
    names = {t['id'] for t in full.all_tasks()}
    if set(bundle['trajectories']) != names or set(bundle['trajectory_sources']) != names:
        raise ValueError('All 60 trajectories required; missing cells cannot be filled')
    for name, rows in bundle['trajectories'].items():
        if [r['round'] for r in rows] != list(range(151)):
            raise ValueError('Incomplete trajectory: ' + name)
        source = bundle['trajectory_sources'][name]
        if report['input_sha256'].get(source['path']) != source['sha256']:
            raise ValueError('Trajectory source was not audited: ' + name)
        summary = report['cells'][name]['summary']
        for key in ('accuracy', 'source_transfer_asr'):
            if any(not math.isfinite(r[key]) or not 0 <= r[key] <= 1 for r in rows):
                raise ValueError('Invalid metric: ' + name)
            if not math.isclose(sum(r[key] for r in rows[-10:]) / 10,
                                summary['tail'][key], abs_tol=1e-12, rel_tol=0):
                raise ValueError('Curve and audited tail disagree: ' + name)
        if rows[-1]['false_positive_revocations'] != summary['health']['final_false_positive_revocations']:
            raise ValueError('Curve and audited health disagree: ' + name)
    v = bundle['validation']
    if (not v['all_inputs_unchanged'] or not v['full_assessment_reproduced']
            or v['training_started'] or v['official_test_used']):
        raise ValueError('Remote export validation is incomplete')
    return report, latest


def save_figure(fig, output, name, plt):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output / (name + '.' + ext), dpi=170, bbox_inches='tight')
    plt.close(fig)


def figures(bundle, report, latest, output):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'h84_matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    x = [100 * r for r in full.RATIOS.values()]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3), sharex=True, layout='constrained')
    for col, part in enumerate(full.PARTITIONS):
        for row, metric in enumerate(('accuracy', 'asr')):
            ax = axes[row, col]
            for arm, (color, marker) in STYLES.items():
                y = [100 * latest['scenarios'][part + '_' + label]['raw_metrics'][arm][metric]
                     for label in full.RATIOS]
                eligible = latest['reference_eligibility'][part][arm]['valid']
                ax.plot(x, y, color=color, marker=marker, ls='-' if eligible else '--',
                        lw=2.8 if arm == 'H84' else 1.5, label='Ours (H84)' if arm == 'H84' else arm,
                        zorder=5 if arm == 'H84' else 2)
            ax.set(xticks=x, xlim=(-3, 83), ylim=(0, 65 if row == 0 else 100),
                   ylabel='Accuracy (%)' if row == 0 else 'Independent ASR (%)')
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(LABELS[part])
            else:
                ax.set_xlabel('Malicious clients (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [Line2D([], [], color='#555555', ls='--')]
    labels += ['Excluded from reference set (raw results retained)']
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    fig.suptitle('CIFAR-10 / fixed H84 / all six methods / mean rounds 141-150\n'
                 'Single observed development seed; IID Krum retained by user exception', fontsize=13)
    save_figure(fig, output, 'raw_six_methods', plt)

    names = [d + '_' + label for d in full.PARTITIONS for label in full.RATIOS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), sharey=True, layout='constrained')
    for j, metric in enumerate(('accuracy', 'asr')):
        ax = axes[j]
        for i, name in enumerate(names):
            s = latest['scenarios'][name]; clean = name.endswith('_clean')
            limit = (1 if clean else 2) if metric == 'accuracy' else (2 if clean else 5)
            gap = s[metric + '_gap_pp']; rank = s[metric + '_rank']
            ok = gap <= limit + 1e-9 and (clean or rank is not None and rank <= 2)
            ax.barh(i, gap, color='#247d72' if ok else '#b2182b', height=.6)
            ax.plot(limit, i, marker='|', color='#333333', ms=17, mew=2)
            ax.text(gap + .055, i, f'{gap:.3f} pp; rank {rank}', va='center', fontsize=9)
        ax.set_xlim(0, 3.15 if j == 0 else 6.3)
        ax.set_title(('Accuracy' if j == 0 else 'ASR') + ' gap to best eligible reference')
        ax.set_xlabel('Percentage points; | marks the allowed gap')
        ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(range(10), [n.replace('dirichlet', 'Non-IID').replace('iid', 'IID')
                                  .replace('_clean', ' 0%').replace('_r', ' ') + ('' if n.endswith('clean') else '%')
                                  for n in names])
    axes[0].invert_yaxis()
    pass_count = sum(s['passed'] for s in latest['scenarios'].values())
    fig.suptitle(f'Latest user criteria: {pass_count}/10 scenarios pass\n'
                 'Clean: no rank requirement, Acc/ASR gaps <=1/2 pp. Attacks: both top 2, gaps <=2/5 pp.\n'
                 'Red includes a rank failure even when the gap is within its limit.', fontsize=12)
    save_figure(fig, output, 'h84_selection', plt)

    colors = ['#777777', '#2166ac', '#16866d', '#d17911', '#b2182b']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True, layout='constrained')
    for col, part in enumerate(full.PARTITIONS):
        for row, metric in enumerate(('accuracy', 'source_transfer_asr')):
            ax = axes[row, col]
            for (label, ratio), color in zip(full.RATIOS.items(), colors):
                rows = bundle['trajectories'][f'H84_{part}_{label}']
                ax.plot([r['round'] for r in rows], [100 * r[metric] for r in rows],
                        color=color, lw=1.5, ls='--' if ratio == 0 else '-', label=f'{100 * ratio:.0f}%')
            ax.axvline(25, color='#333333', ls=':', lw=1)
            ax.set(xlim=(0, 150), ylim=(0, 65 if row == 0 else 100),
                   ylabel='Accuracy (%)' if row == 0 else 'Independent ASR (%)')
            ax.grid(alpha=.2)
            if row == 0: ax.set_title(LABELS[part])
            else: ax.set_xlabel('Training round; attack begins at round 25')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=5,
               title='Malicious client ratio', frameon=False)
    fig.suptitle('Ours (H84): complete trajectories, including early attack transients\n'
                 'Single development seed; no smoothing, no uncertainty estimate', fontsize=13)
    save_figure(fig, output, 'h84_trajectories', plt)


def table(headers, rows):
    return '<table><thead><tr>' + ''.join('<th>' + html.escape(str(x)) + '</th>' for x in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>'
         for x in row) + '</tr>' for row in rows) + '</tbody></table>'


def make_html(bundle, report, latest, output, source_sha):
    selected, raw, eligibility, process = [], [], [], []
    for part in full.PARTITIONS:
        for arm in full.METHODS:
            h = latest['reference_eligibility'][part][arm]
            eligibility.append([LABELS[part], arm, '参评（IID clean效用例外）' if h.get('waived_reasons')
                                else '参评' if h['valid'] else '排除',
                                f"{100 * h['clean_accuracy_drop']:.3f}", ', '.join(h['reasons']) or '无'])
        for label, ratio in full.RATIOS.items():
            s = latest['scenarios'][part + '_' + label]; h = report['cells'][f'H84_{part}_{label}']['summary']
            selected.append([LABELS[part], f'{ratio:.0%}', f"{100*s['raw_metrics']['H84']['accuracy']:.3f}",
                s['accuracy_rank'], f"{s['accuracy_gap_pp']:.3f}", f"{100*s['raw_metrics']['H84']['asr']:.3f}",
                s['asr_rank'], f"{s['asr_gap_pp']:.3f}", f"{100*h['health']['false_revocation_rate']:.2f}%",
                '通过' if s['passed'] else '未通过'])
            for arm in full.METHODS:
                cell = report['cells'][f'{arm}_{part}_{label}']['summary']; health = cell['health']
                raw.append([LABELS[part], f'{ratio:.0%}', arm, f"{100*s['raw_metrics'][arm]['accuracy']:.3f}",
                    s['raw_accuracy_ranks'][arm], f"{100*s['raw_metrics'][arm]['asr']:.3f}", s['raw_asr_ranks'][arm],
                    f"{100*health['false_revocation_rate']:.2f}%", '是' if latest['reference_eligibility'][part][arm]['valid'] else '否',
                    ', '.join(health['v14_failure_reasons'] + health['additional_review_reasons']) or '单元健康通过'])
            if ratio:
                rows = bundle['trajectories'][f'H84_{part}_{label}']
                first = next((r['round'] for r in rows[25:] if r['source_transfer_asr'] < .2), None)
                process.append([LABELS[part], f'{ratio:.0%}', f"{100*rows[25]['source_transfer_asr']:.3f}",
                    f"{100*h['peak_attack_asr']:.3f}", f"{100*h['attack_period']['source_transfer_asr']:.3f}",
                    f"{100*h['attack_period']['accuracy']:.3f}", first])
    gates = [[n, f"{100*g['accuracy_drop']:.3f}", f"{100*g['asr_rise']:.3f}", '通过' if g['passed'] else '未通过']
             for n, g in report['attack_gates'].items()]
    counts = sum(c['attack']['actual_calls'] for c in report['cells'].values())
    passed = sum(s['passed'] for s in latest['scenarios'].values())
    images = {n: (output / (n + '.svg')).read_text() for n in ('raw_six_methods', 'h84_selection', 'h84_trajectories')}
    images = {n: svg[svg.index('<svg'): ] for n, svg in images.items()}
    strict_passes = sum(s['passed'] for s in report['assessment']['scenarios'].values())
    title = 'CIFAR-10 H84 完整 CUDA 开发验证'
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{title}</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#18303c;background:#f3f6f8;margin:0}}main{{max-width:1280px;margin:32px auto;padding:32px;background:white}}h1{{font-size:30px}}h2{{margin-top:40px}}.lead{{padding:18px;background:#fff0ed;border-left:5px solid #b2182b}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}}td,th{{border-bottom:1px solid #ddd;padding:7px;text-align:right}}th{{background:#eef4f6}}td:first-child,th:first-child{{text-align:left}}svg{{width:100%;height:auto}}code{{overflow-wrap:anywhere}}.note{{color:#526570}}@media print{{main{{margin:0;padding:0}}body{{background:white}}table{{font-size:10px}}}}</style>
<main><h1>{title}</h1><p class="lead"><b>本轮未达标：{passed}/10场景通过，Ours原健康10/10通过。</b>
未通过：IID20% ASR第三（10.04%，距Krum3.107pp、距TAD0.600pp）；Non-IID clean Accuracy距健康最优1.735pp，超过1pp上限。
因此不进入正式实验；这是固定H84候选的有边界失败，不证明整个参数空间无解。</p>
<p>60/60单元完整150轮；54新增+6同CUDA复用。比较末10轮均值，ASR使用独立750张源类验证图。
IID和Dirichlet(alpha=0.5)，五比例、六方法。此前12+6项开发加本54项，共72个独立训练任务、10800轮，失败均保留。
seed2026102201已经用于开发，不是未用seed或正式实验；没有使用官方测试集选参。共享GPU耗时不作算法速度证据。</p>
<h2>最新用户标准与逐场景结论</h2><p>同分布clean效用和五比例原健康均通过后才参评；仅Krum IID clean效用采用用户已知结果后的例外。
clean不限制排名，Accuracy/ASR差距≤1/2pp；攻击场景双前二、差距≤2/5pp。所有差距单位是百分点，数值越小越好。</p>
{table(['分布','恶意比例','Acc %','Acc排名','Acc差pp','ASR %','ASR排名','ASR差pp','误撤销率','结论'], selected)}
{images['h84_selection']}
<h2>参照资格与原严格判定</h2><p>IID六法全部参评，其中Krum原clean效用仍失败，依用户例外保留。
Non-IID参评FedAvg/H84/VERT/AlignIns；TAD clean误撤销21%被排除，Krum clean Acc比FedAvg低27.409pp被排除。
这些排除按分布内五比例作用，不是逐场景择优。</p>
{table(['分布','方法','当前参照资格','clean Acc损失pp','未豁免失败原因'], eligibility)}
<p>原严格判定（无Krum例外）{strict_passes}/10场景通过，仅Non-IID clean失败；当前独立判定{passed}/10通过。
原报告、原始六法名次和所有失败完整保留，未覆盖冻结协议。Krum的IID例外依据是选择集中和局部过拟合实证，
并非已隔离全部15.407pp差距的因果份额，也未回放历史CUDA距离分数。</p>
<h2>六法原始结果</h2><p>虚线仅标明未进入当前参照集；其原始数值始终展示。</p>{images['raw_six_methods']}
{table(['分布','恶意比例','方法','Acc %','原Acc排名','ASR %','原ASR排名','误撤销率','当前参评','单元健康'], raw)}
<h2>全程表现与早期代价</h2>{images['h84_trajectories']}
{table(['分布','恶意比例','r25 ASR %','攻击期峰ASR %','攻击期均ASR %','攻击期均Acc %','首次ASR低20%的轮次'], process)}
<p>H84 IID60首次ASR低20%在r27，W85是r26；全攻击期ASR较W85高0.213757pp。
IID80峰ASR21.466667%，W85为23.866667%；这些早期代价与收益不被尾10均值掩盖。
Non-IID80诚实误撤销2/20=10%，恰好达到上限，不能称有充足健康余量。
Non-IID80还有中期退化：r29–57的ASR持续不低于20%，r53达到30.4%，全攻击期均值13.308%，尾10才降到7.267%。
该场景r26首次出现恶意历史准入；这是一条观察到的机制路径，未通过新的反事实实验隔离因果份额。
表中“首次ASR低20%”不等于从此持续恢复，不能用r25的一次低值掩盖此后升高。</p>
<h2>攻击有效性和执行审计</h2>{table(['FedAvg场景','Acc下降pp','ASR上升pp','攻击有效性'], gates)}
<p>8/8攻击门槛通过（Acc下降至少10pp、ASR上升至少30pp）；{counts:,}次实际攻击调用的SGD/目标视图与在线客户端覆盖已核验。
82个冻结来源、162份父引用、638份原始输入通过SHA检查，十个Ours单元机制覆盖通过。
48组攻击前配对中40组要求一致者全部通过；8组Krum前缀不相同，原因是原基线使用配置的Byzantine数量，依预声明单独披露。
TAD十项共1500轮实际无先验决策通过：决策接口不接收真实恶意比例/身份或验证成绩；内部谱状态未逐位回放，论文歧义解释仍按冻结配置披露。</p>
<details><summary>TAD冻结复现解释与边界（完整保留）</summary><pre style="white-space:pre-wrap">{html.escape(json.dumps(report['tad_reproduction'], ensure_ascii=False, indent=2))}</pre></details>
<p class="note">训练提交：<code>{bundle['frozen_training_commit']}</code>；独立判定时HEAD：<code>{bundle['remote_head']}</code>。
导出UTC：{bundle['observed_utc']}；本地导出SHA256：<code>{source_sha}</code>。
结果由原始严格报告与独立用户修订报告重新计算后渲染，原证据不写回。</p></main></html>'''
    (output / 'report.html').write_text(text)


def render(source, output, expected_sha=None):
    raw = source.read_bytes()
    if expected_sha is not None and digest(raw) != expected_sha:
        raise ValueError('Export file changed since its verified transfer')
    bundle = json.loads(raw); report, latest = validate(bundle)
    output.mkdir(parents=True, exist_ok=True)
    figures(bundle, report, latest, output)
    make_html(bundle, report, latest, output, digest(raw))
    if source.read_bytes() != raw:
        raise ValueError('Evidence changed during rendering')
    provenance = dict(bundle_sha256=digest(raw), renderer_sha256=digest(Path(__file__).read_bytes()),
        original_report_sha256=bundle['files']['analysis/full_validation.json']['sha256'],
        latest_report_sha256=bundle['files']['analysis/krum_reference_exception_assessment.json']['sha256'],
        scenario_passes=sum(s['passed'] for s in latest['scenarios'].values()),
        all_raw_results_retained=True, source_unchanged=True, formal_plan_condition_met=latest['formal_plan_condition_met'],
        scientific_plots='PNG/SVG/PDF; no error bars from a single observed development seed')
    full.base.write(output / 'figure_provenance.json', provenance)
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=ROOT / full.OUTPUT / 'report')
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    print(json.dumps(render(args.bundle, args.output, args.expected_sha256), indent=2))
