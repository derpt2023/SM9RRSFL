"""Validate an immutable formal export and render all seeds without selecting results."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics as stats
import tempfile

import cifar_h84_formal_report as formal_report
import run_cifar_h84_full_validation as full

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'outputs/cifar_h84_formal150_v1/export/formal_audited_bundle.json'
SEEDS = (2026102801, 2026102802, 2026102803)
COLORS = dict(FedAvg='#737373', H84='#b2182b', TAD='#16866d',
              VERT='#2166ac', AlignIns='#9467bd', Krum='#d17911')
LABELS = dict(iid='IID', dirichlet='Non-IID (Dirichlet alpha=0.5)')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def close(a, b):
    if not math.isclose(a, b, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f'Inconsistent numeric evidence: {a} != {b}')


def validate_groups(cells, saved):
    computed = formal_report.grouped_statistics(cells, SEEDS)
    # Python statistics.stdev differs by a few ULPs across the deployed runtimes.
    # Permit rounding only in derived SD; samples, means, counts and flags stay exact.
    adjusted = deepcopy(saved)
    if set(computed) != set(adjusted):
        raise ValueError('Seed groups are missing')
    for name, group in computed.items():
        for metric, sample in group['samples'].items():
            left = sample['sample_sd']; right = adjusted[name]['samples'][metric]['sample_sd']
            if left is None or right is None:
                if left != right: raise ValueError('Sample SD missing')
            elif not math.isclose(left, right, rel_tol=2e-15, abs_tol=1e-15):
                raise ValueError('Sample SD differs beyond floating-point rounding')
            adjusted[name]['samples'][metric]['sample_sd'] = left
    if computed != adjusted:
        raise ValueError('Seed statistics differ; health failures must remain in raw means')


def validate(bundle):
    if bundle['schema'] != 'h84-formal-audited-export-v1':
        raise ValueError('Unexpected export schema')
    files = {}
    for name, entry in bundle['files'].items():
        if sha(entry['raw'].encode()) != entry['sha256']:
            raise ValueError('Embedded evidence digest differs: ' + name)
        files[name] = json.loads(entry['raw'])
    r = files['analysis/formal_results.json']
    recovery = files['analysis/recovery_final_audit.json']
    manifest = files['manifest.json']
    if (r['status'] != 'complete_audited_formal' or r['official_test_used_for_selection']
            or not r['no_retuning_or_selection'] or r['fingerprint'] != manifest['fingerprint']):
        raise ValueError('Not a complete, fixed formal evaluation')
    names = {t['id'] + '_s' + str(s) for t in full.all_tasks() for s in SEEDS}
    for values in (r['cells'], r['audited_cells'], bundle['trajectories'], bundle['trajectory_sources']):
        if set(values) != names:
            raise ValueError('All 180 cells required; no exclusion or zero filling')
    v = bundle['validation']
    if (not v['all_inputs_unchanged'] or not v['source_commit_matches']
            or v['source_count'] != len(manifest['sources']) or v['source_count'] != 89
            or v['audited_input_count'] != len(r['input_sha256'])
            or v['training_started_by_export'] or v['official_test_re_evaluated_by_export']):
        raise ValueError('Incomplete remote provenance verification')
    if (recovery['formal_report_sha256'] != bundle['files']['analysis/formal_results.json']['sha256']
            or recovery['manifest_fingerprint'] != r['fingerprint']
            or not recovery['same_task_identity_and_initial_model']
            or not recovery['original_failed_attempt_retained']
            or recovery['scientific_cells'] != 180 or recovery['execution_attempts'] != 181
            or recovery['extra_completed_training_rounds'] != 0):
        raise ValueError('Reviewed failure/recovery evidence differs')
    for name, c in r['cells'].items():
        rows = bundle['trajectories'][name]
        src = bundle['trajectory_sources'][name]
        audit = r['audited_cells'][name]
        if (not c['complete_finite'] or c['status'] != 'complete' or c['completed_round'] != 150
                or [x['round'] for x in rows] != list(range(151)) or audit['summary'] != c
                or r['input_sha256'].get(src['path']) != src['sha256']):
            raise ValueError('Trajectory identity or completeness differs: ' + name)
        for key in ('accuracy', 'source_transfer_asr', 'honest_weight_loss', 'malicious_weight_mass'):
            if any(not math.isfinite(x[key]) or not -1e-9 <= x[key] <= 1+1e-9 for x in rows):
                raise ValueError('Invalid trajectory metric')
            close(stats.mean(x[key] for x in rows[-10:]), c['tail'][key])
            close(rows[-1][key], c['final'][key])
        for key in ('accuracy', 'source_transfer_asr'):
            close(stats.mean(x[key] for x in rows[25:]), c['attack_period'][key])
        close(max(x['source_transfer_asr'] for x in rows[25:]), c['peak_attack_asr'])
        if rows[-1]['false_positive_revocations'] != c['final_fp']:
            raise ValueError('False revocation count differs')
        a = audit['attack']
        if not (a['complete_coverage'] and a['target_views_and_optimizers_verified']
                and a['actual_calls'] == a['expected_calls']):
            raise ValueError('Attack execution audit incomplete')
        if c['arm'] == 'H84' and not audit['mechanism']['actual_rule_count_coverage_and_mass_verified']:
            raise ValueError('Ours mechanism audit incomplete')
        if c['arm'] == 'TAD' and (not audit['tad_audit']['actual_no_prior_decisions']
                or audit['tad_audit']['rounds_verified'] != 150):
            raise ValueError('TAD no-prior audit incomplete')
        t = c['timing']
        close(t['runtime_with_crypto_seconds'] - t['crypto_wall_seconds'], t['runtime_without_crypto_seconds'])
        close(sum(t['stages'][k] for k in ('crypto_setup_wall_seconds', 'crypto_packet_wall_seconds',
              'crypto_audit_wall_seconds', 'crypto_finalize_wall_seconds')), t['crypto_wall_seconds'])
        if not t['available_and_consistent'] or min(t['runtime_without_crypto_seconds'], t['crypto_wall_seconds']) < 0:
            raise ValueError('Invalid timing')
    validate_groups(r['cells'], r['groups'])
    for seed in SEEDS:
        saved = r['per_seed'][str(seed)]; suffix = '_s' + str(seed)
        gates = {}
        for task in full.all_tasks():
            if not task['ratio']:
                continue
            name = task['id'] + suffix
            clean = f"{task['arm']}_{task['partition']}_clean" + suffix
            fields = ('accuracy', 'source_transfer_asr', 'non_source_accuracy', 'aux128_confidence')
            prefix = all(bundle['trajectories'][name][i][k] == bundle['trajectories'][clean][i][k]
                         for i in range(25) for k in fields)
            if prefix != saved['pairing'][task['id']]['classification_prefix_equal']:
                raise ValueError('Classification prefix audit differs')
            if task['arm'] == 'FedAvg':
                a, c = r['cells'][name]['tail'], r['cells'][clean]['tail']
                drop, rise = c['accuracy'] - a['accuracy'], a['source_transfer_asr'] - c['source_transfer_asr']
                gates[task['id']] = dict(accuracy_drop=drop, asr_rise=rise,
                                        passed=drop >= .1-1e-12 and rise >= .3-1e-12)
        if gates != saved['fedavg_attack_gates']:
            raise ValueError('FedAvg attack gate differs')
        selected = {t['id']: r['audited_cells'][t['id'] + suffix] for t in full.all_tasks()}
        strict = full.assess(selected, gates, saved['pairing'])
        current = formal_report.with_krum_exception(strict)
        if strict != saved['strict'] or current != saved['with_previous_krum_exception']:
            raise ValueError('Per-seed strict/current assessment cannot be reproduced')
    return r, recovery


def save(fig, out, name, plt):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(out / (name + '.' + ext), dpi=170, bbox_inches='tight')
    plt.close(fig)


def figures(bundle, r, out):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'h84_formal_matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    x = [100 * ratio for ratio in full.RATIOS.values()]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for col, d in enumerate(full.PARTITIONS):
        for row, metric in enumerate(('accuracy', 'asr')):
            ax = axes[row, col]
            for arm in full.METHODS:
                vals = [r['groups'][f'{arm}_{d}_{label}']['samples'][metric] for label in full.RATIOS]
                ax.errorbar(x, [v['mean']*100 for v in vals], yerr=[v['sample_sd']*100 for v in vals],
                            color=COLORS[arm], marker='o', capsize=3, lw=2.5 if arm=='H84' else 1.4,
                            label='Ours (H84)' if arm=='H84' else arm)
            ax.set(title=LABELS[d], xticks=x, xlabel='Malicious clients (%)',
                   ylabel='Accuracy (%)' if row==0 else 'ASR (%)', ylim=(0, 65 if row==0 else 102))
            ax.grid(alpha=.18)
    axes[0, 0].legend(ncol=2, fontsize=9)
    fig.suptitle('CIFAR-10 formal test | tail 10 rounds | mean ± sample SD, n=3\nAll six methods retained, including health failures')
    save(fig, out, 'all_methods', plt)
    for d in full.PARTITIONS:
        fig, axes = plt.subplots(2, 5, figsize=(19, 7), sharex=True, sharey='row', layout='constrained')
        for col, label in enumerate(full.RATIOS):
            for row, metric in enumerate(('accuracy', 'source_transfer_asr')):
                ax = axes[row, col]
                for arm in full.METHODS:
                    a = np.array([[v[metric]*100 for v in bundle['trajectories'][f'{arm}_{d}_{label}_s{s}']] for s in SEEDS])
                    mean, sd = a.mean(0), a.std(0, ddof=1)
                    ax.plot(range(151), mean, color=COLORS[arm], lw=2 if arm=='H84' else 1.1,
                            label='Ours (H84)' if arm=='H84' else arm)
                    ax.fill_between(range(151), mean-sd, mean+sd, color=COLORS[arm], alpha=.08)
                if label != 'clean': ax.axvline(25, color='black', ls=':', lw=.8)
                ax.set(title=f'{int(full.RATIOS[label]*100)}% malicious', xlim=(0,150),
                       ylim=(0,65 if row==0 else 105), xlabel='Round' if row else '')
                if col==0: ax.set_ylabel('Accuracy (%)' if row==0 else 'ASR (%)')
                ax.grid(alpha=.15)
        axes[0,0].legend(ncol=2, fontsize=8)
        fig.suptitle(f'{LABELS[d]} | all 151 observations | mean ± sample SD, n=3\nNo smoothing; all methods and failed-health seeds retained; dotted line = attack starts')
        save(fig, out, 'trajectories_'+d, plt)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax,d in zip(axes,full.PARTITIONS):
        for arm in ('H84','TAD'):
            for idx,s in enumerate(SEEDS):
                vals=[r['cells'][f'{arm}_{d}_{label}_s{s}']['health']['false_revocation_rate']*100 for label in full.RATIOS]
                ax.plot(x,vals,color=COLORS[arm],ls=('-', '--', ':')[idx],marker='o',label=f'{arm} seed {str(s)[-2:]}')
        ax.axhline(10,color='black',ls='-.',lw=1,label='10% limit')
        ax.set(title=LABELS[d],xticks=x,xlabel='Malicious clients (%)',ylabel='Final honest false revocations (%)',ylim=(-.5,18))
        ax.grid(alpha=.18)
    axes[1].legend(fontsize=8,ncol=2)
    fig.suptitle('False revocations | denominator = honest clients in each scenario')
    save(fig,out,'false_revocations',plt)
    fig,ax=plt.subplots(figsize=(12,3.7),layout='constrained')
    names=[d+'_'+label for d in full.PARTITIONS for label in full.RATIOS]
    a=np.array([[r['per_seed'][str(s)]['with_previous_krum_exception']['scenarios'][n]['passed'] for n in names] for s in SEEDS])
    ax.imshow(a, cmap=matplotlib.colors.ListedColormap(['#f3c0b7','#d0e8dc']),vmin=0,vmax=1,aspect='auto')
    for i in range(3):
        for j in range(10): ax.text(j,i,'PASS' if a[i,j] else 'FAIL',ha='center',va='center',fontsize=9)
    ax.set(xticks=range(10),xticklabels=[n.replace('dirichlet','Non-IID').replace('clean','0%').replace('_r',' ').replace('_',' ') for n in names],
           yticks=range(3),yticklabels=[str(s) for s in SEEDS])
    ax.tick_params(axis='x',labelrotation=30)
    ax.set_title('Frozen descriptive criteria with prior Krum IID clean exception | 8/10, 8/10, 9/10\nPassing cell health does not imply passing these comparative criteria')
    save(fig,out,'criteria',plt)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    without=[];crypto=[]
    for arm in full.METHODS:
        cs=[c for c in r['cells'].values() if c['arm']==arm]
        without.append(sum(c['timing']['runtime_without_crypto_seconds'] for c in cs)/3600)
        crypto.append(sum(c['timing']['crypto_wall_seconds'] for c in cs)/3600)
    ax.bar(list(full.METHODS),without,color='#537a95',label='Worker wall time minus measured crypto spans')
    ax.bar(list(full.METHODS),crypto,bottom=without,color='#c4825e',label='Four non-overlapping crypto wall spans')
    for i,(a,b) in enumerate(zip(without,crypto)):ax.text(i,a+b+.12,f'{a+b:.2f} h',ha='center')
    ax.set(ylabel='Sum of worker wall hours (30 tasks per method)',ylim=(0,max(a+b for a,b in zip(without,crypto))*1.22))
    ax.legend(fontsize=8)
    ax.set_title('Measured execution cost on shared GPUs/CPUs\nNot an isolated algorithm speed comparison; subtraction is not a crypto-disabled rerun')
    save(fig,out,'timing',plt)


def table(headers, rows):
    esc=lambda x:html.escape(str(x))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def document(bundle, r, recovery, out, digest):
    f=lambda v: f'{v:.3f}'
    pm=lambda s,scale=100:f"{s['mean']*scale:.3f} ± {s['sample_sd']*scale:.3f}"
    title='固定H84：CIFAR-10三种子正式实验'
    body=[f'<h1>{title}</h1><p>180/180项 × 150轮全部完成；官方10000图Accuracy、source5的1000图→target7 ASR。'
          '固定H84与V2攻击，无测试集调参。表中比例/差距单位为%/百分点；均值±样本SD以3个seed为样本，健康失败未从原始均值剔除。</p>',
          '<p><strong>结论：H84的30项单元健康通过，但既定比较标准未全面通过。</strong>含此前Krum IID clean效用例外，三个seed通过8/10、8/10、9/10；'
          '原严格口径为8/10、9/10、10/10。seed之间健康参照不同，不为跨seed平均值创造新的参照资格。</p>',
          '<p>开发阶段用户接受H84是已知结果后的采纳决定；原开发8/10与严格9/10仍保留。这里的正式评估不再次选择候选。'
          'Krum例外只豁免IID clean精度效用失败，不表示该效用门槛原本通过。clean不要求名次，但精度/ASR差距须≤1/2pp；攻击场景双前二且差距≤2/5pp。</p>',
          '<h2>三种子原始汇总</h2><img src="all_methods.png" alt="六法尾十轮均值和样本标准差">']
    rows=[]
    for d in full.PARTITIONS:
        for label in full.RATIOS:
            for arm in full.METHODS:
                g=r['groups'][f'{arm}_{d}_{label}'];s=g['samples']
                count=sum(r['cells'][f'{arm}_{d}_{label}_s{seed}']['health']['complete_observed_health_pass'] for seed in SEEDS)
                rows.append([d,label,arm,g['actual_n'],pm(s['accuracy']),pm(s['asr']),pm(s['false_revocation_rate']),f'{count}/3'])
    body.append(table(['分布','恶意比例','方法','n','尾10 Acc %','尾10 ASR %','最终误撤销率 %','单元健康'],rows))
    body+=['<h2>逐seed、逐场景判定</h2><img src="criteria.png" alt="三种子十场景通过失败矩阵">']
    rows=[];raw=[];health=[]
    for seed in SEEDS:
        saved=r['per_seed'][str(seed)];latest=saved['with_previous_krum_exception']
        for name,s in latest['scenarios'].items():
            rows.append([seed,name,','.join(s['eligible_methods']),s['accuracy_rank'],s['asr_rank'],f(s['accuracy_gap_pp']),f(s['asr_gap_pp']),s['passed'],saved['strict']['scenarios'][name]['passed']])
            for arm in full.METHODS:
                m=s['raw_metrics'][arm]
                raw.append([seed,name,arm,f(m['accuracy']*100),f(m['asr']*100),s['raw_accuracy_ranks'][arm],s['raw_asr_ranks'][arm],arm in s['eligible_methods']])
        for d,methods in latest['method_health'].items():
            for arm,h in methods.items():
                health.append([seed,d,arm,h['valid'],f(h['clean_accuracy_drop']*100),'; '.join(h['reasons']) or '—',h.get('waived_clean_utility_only',False)])
    body.append(table(['seed','场景','健康参照','H84 Acc名次','H84 ASR名次','Acc差距pp','ASR差距pp','最新通过','原严格通过'],rows))
    body+=['<h2>参照资格与完整排除原因</h2><p>资格作用于同seed、同分布的clean效用与五个比例；原clean效用精度损失门槛为5pp。'
           'TAD有13项单元健康失败，全部保留；VERT和Non-IID Krum的clean效用失败另列，不能与执行异常混为一谈。</p>',table(['seed','分布','方法','参照资格','clean Acc降幅pp','失败原因','仅IID Krum例外'],health),
           '<h2>六法完整原始排名</h2>',table(['seed','场景','方法','Acc %','ASR %','原始Acc名次','原始ASR名次','本场景参照资格'],raw),
           '<h2>全程曲线与误撤销</h2><p>完整0..150轮，无平滑或裁剪。浅色区间为seed间样本SD，不是置信区间。clean的ASR是未攻击背景转移率。</p>']
    for name in ('trajectories_iid','trajectories_dirichlet','false_revocations'):
        body.append(f'<img src="{name}.png" alt="{name}">')
    body+=['<h2>最终轮、全攻击期、峰值与成本：180项原值</h2><p>全攻击期为25..150轮；clean对应相同时间窗口。'
           '峰值不跨seed先平均，表中逐项保留。总墙钟包含加载/训练/攻击/评估/观测/审计/I/O；'
           '不含密码学=同次总墙钟减setup/packet/audit/finalize四段，并非关闭密码学重跑，不重复扣除并行签名/验签累计值。'
           '基线未执行Ours密码协议，crypto为0。checkpoint与结构化写盘为另记实测值，checkpoint是后者子集，不能相加重复扣除。</p>',
           '<img src="timing.png" alt="含与扣除密码学的累计运行时间">']
    rows=[]
    for name,c in sorted(r['cells'].items()):
        t=c['timing'];mp=t['measured_persistence']
        rows.append([name,f(c['final']['accuracy']*100),f(c['final']['source_transfer_asr']*100),
            f(c['attack_period']['accuracy']*100),f(c['attack_period']['source_transfer_asr']*100),f(c['peak_attack_asr']*100),
            c['final_fp'],c['health']['complete_observed_health_pass'],f(t['runtime_with_crypto_seconds']),f(t['runtime_without_crypto_seconds']),
            f(t['crypto_wall_seconds']),f(mp['checkpoint_write_seconds']),f(mp['structured_persistence_seconds'])])
    body.append(table(['任务','r150 Acc','r150 ASR','r25–150 Acc','r25–150 ASR','峰ASR','FP人数','单元健康','总秒','扣crypto秒','crypto秒','checkpoint秒','结构化写盘秒'],rows))
    calls=sum(c['attack']['actual_calls'] for c in r['audited_cells'].values())
    required=[p for x in r['per_seed'].values() for p in x['pairing'].values() if p['required']]
    krum=[p for x in r['per_seed'].values() for p in x['pairing'].values() if not p['required']]
    elapsed=(datetime.fromisoformat(bundle['controller']['updated_utc'])-datetime.fromisoformat(bundle['launch']['started_utc'])).total_seconds()
    body+=['<h2>审计与边界</h2>',f'<p>89项冻结来源匹配科学提交；1764份报告输入SHA再次核验。实际攻击调用{calls:,}次覆盖完整，训练辅助目标128图与SGD/增强视图已核。'
           f'要求配对的{len(required)}组均通过；Krum额外24组单独披露，其中{sum(not p["passed"] for p in krum)}组前缀/模型不一致，源于配置中使用Byzantine数，不能称Krum无先验。'
           '24个FedAvg攻击门槛全部通过。H84逐客户端计数、历史准入、聚合质量已核。TAD4500轮实际无先验决策已核，未声称其内部谱状态逐位回放。</p>',
           '<p>TAD沿用冻结实现选择与论文歧义披露，完整说明在原formal_results.json的tad_reproduction字段。'
           '三个seed只刻画本次固定协议下的变化，不代表所有seed或攻击。高比例/Non-IID压力场景保持原配置。</p>',
           f'<p>批次启动至训练控制器收尾{elapsed/3600:.3f}小时，包含资源等待；各任务累计墙钟不能当作并行批次耗时或独占速度比较。'
           f'180科学单元、181执行尝试，初始VERT OOM永久封存，失败成本{recovery["failed_attempt_elapsed_seconds"]:.6f}秒、额外完成训练轮数0；同配置/seed/初始化核验通过。</p>',
           '<h2>可核验文件</h2><p><a href="formal_results.json">原正式报告JSON</a> · <a href="recovery_final_audit.json">原恢复审计JSON</a> · '
           '<a href="render_manifest.json">本地绘图核验记录</a></p>',
           f'<p>导出SHA256：<code>{digest}</code><br>正式报告SHA256：<code>{bundle["files"]["analysis/formal_results.json"]["sha256"]}</code></p>']
    css='body{font:15px/1.6 system-ui;margin:36px auto;max-width:1400px;padding:0 24px;color:#202b32;background:#f9fafb}h1,h2{color:#17364a}h2{margin-top:36px}img{width:100%;background:white;margin:15px 0}.scroll{overflow:auto}table{border-collapse:collapse;background:white;font-size:12px;width:100%}th,td{padding:6px 8px;border:1px solid #dce2e6;text-align:right;white-space:nowrap}th{background:#e9eff3}td:first-child,th:first-child{text-align:left}code{overflow-wrap:anywhere}a{color:#2166ac}'
    (out/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+title+'</title><style>'+css+'</style>'+''.join(body)+'</html>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();raw=args.source.read_bytes();digest=sha(raw)
    if digest != args.expected_sha256: raise ValueError('Pinned export SHA differs')
    bundle=json.loads(raw);r,recovery=validate(bundle)
    out=args.output or args.source.parent.parent/'report'
    out.mkdir(parents=True,exist_ok=False)
    figures(bundle,r,out);document(bundle,r,recovery,out,digest)
    for name in ('formal_results.json','recovery_final_audit.json'):
        (out/name).write_text(bundle['files']['analysis/'+name]['raw'])
    if sha(args.source.read_bytes()) != digest: raise ValueError('Export changed while rendering')
    record=dict(source_sha256=digest,all_180_cells_validated=True,all_60_seed_groups_recomputed=True,
        all_three_strict_and_amended_assessments_recomputed=True,source_unchanged=True,
        renderer_sha256=sha(Path(__file__).read_bytes()),
        outputs={p.name:sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'render_manifest.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(report=str(out/'report.html'),source_sha256=digest,cells=180,seeds=list(SEEDS))))


if __name__=='__main__':
    main()
