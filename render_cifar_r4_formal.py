"""Render user-requested fixed R4 official test against the immutable cifar-paper five methods."""
import argparse
from copy import deepcopy
import html
import json
import math
import os
from pathlib import Path
import statistics as stats
import tempfile

import render_cifar_h84_formal as original_render
import run_cifar_h84_full_validation as full
import run_cifar_r4_formal as base
import cifar_r4_formal_report as reporting

ROOT=base.ROOT
SEEDS=base.SEEDS
COLORS=dict(H84='#8DBAD9',VERT='#E99092',AlignIns='#8ADDE4',Krum='#DCDD8F',TAD='#FEBD85',FedAvg='#C4A9A2')
STYLES=dict(H84='-',VERT='--',AlignIns=':',Krum='-.',TAD=(0,(5,2,1,2)),FedAvg=(0,(2,2)))
LABELS=dict(H84='Ours (R4)',VERT='VERT (paper)',AlignIns='AlignIns',Krum='Krum',TAD='TAD',FedAvg='FedAvg')
sha=original_render.sha
close=original_render.close


def validate(bundle,baseline_raw):
    if sha(baseline_raw)!=base.BASELINE_SHA:raise ValueError('Exact accepted cifar-paper export required')
    parent=json.loads(baseline_raw);old=parent['report'];r=bundle['report'];m=bundle['manifest']
    if (bundle['schema']!='user-requested-r4-official-export-v1'
            or r['status']!='complete_audited_user_requested_r4_formal'
            or bundle['accepted_baseline_sha256']!=base.BASELINE_SHA
            or not bundle['source_verified'] or not bundle['all_inputs_unchanged']
            or r['fingerprint']!=m['fingerprint'] or len(m['sources'])!=126
            or m['protocol']['parameters']!=base.PARAMETERS or m['protocol']['authorization']!=base.AUTHORIZATION
            or r['acceptance']!=base.AUTHORIZATION or r['prior_validation_selected'] is not None
            or r['validation_selection_passed'] or r['official_test_used_for_selection']
            or not r['no_formal_retuning_or_selection']):
        raise ValueError('Fixed user-directed R4 provenance required')
    raw=(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if sha(raw)!=bundle['combined_report_sha256']:raise ValueError('Combined report digest differs')
    names=set(old['cells']);ours={n for n in names if old['cells'][n]['arm']=='H84'}
    if (len(names)!=180 or len(ours)!=30 or set(r['cells'])!=names or set(r['audited_cells'])!=names
            or set(bundle['trajectories'])!=names or set(r['replacement_lineage'])!=ours):
        raise ValueError('Exactly180 cells and30 fresh R4 cells required')
    for name,c in r['cells'].items():
        a=r['audited_cells'][name];rows=bundle['trajectories'][name]
        if name not in ours:
            if c!=old['cells'][name] or a!=old['audited_cells'][name] or rows!=parent['trajectories'][name]:
                raise ValueError('Other five values, health, audit or curves changed')
        else:
            lineage=r['replacement_lineage'][name]
            if (c['config']!=dict(old['cells'][name]['config'],**base.PARAMETERS)
                    or not lineage['same_initial_model_data_targets_malicious_ids']
                    or lineage['only_changed_parameter']!=base.PARAMETERS
                    or not a['actual_weights_verified']
                    or not a['mechanism']['actual_rule_count_coverage_and_mass_verified']):
                raise ValueError('R4 actual execution lineage differs')
            expected=reporting.classification_equal(rows,parent['trajectories'][name],20)
            if expected!=r['r4_original_h84_warmup_pairing'][c['source_task']]:raise ValueError('Warmup audit differs')
        if (c['status']!='complete' or not c['complete_finite'] or c['completed_round']!=150
                or a['summary']!=c or [x['round'] for x in rows]!=list(range(151))):
            raise ValueError('Incomplete evidence cannot be plotted as complete')
        for key in ('accuracy','source_transfer_asr','honest_weight_loss','malicious_weight_mass'):
            if any(not math.isfinite(x[key]) or not -1e-9<=x[key]<=1+1e-9 for x in rows):raise ValueError('Invalid curve')
            close(stats.mean(x[key] for x in rows[-10:]),c['tail'][key]);close(rows[-1][key],c['final'][key])
        for key in ('accuracy','source_transfer_asr'):
            close(stats.mean(x[key] for x in rows[25:]),c['attack_period'][key])
        if not a['attack']['complete_coverage'] or not a['attack']['target_views_and_optimizers_verified']:
            raise ValueError('Attack audit incomplete')
    original_render.validate_groups(r['cells'],r['groups'])
    pairs={str(s):{n:v for n,v in r['per_seed'][str(s)]['pairing'].items() if n.startswith('H84_')} for s in SEEDS}
    for seed,checks in pairs.items():
        for name,pair in checks.items():
            d=name.split('_')[1];clean=f'H84_{d}_clean_s{seed}'
            if reporting.classification_equal(bundle['trajectories'][name+'_s'+seed],bundle['trajectories'][clean],24)!=pair['classification_prefix_equal']:
                raise ValueError('Paired prefix audit differs')
    computed=reporting.combine(old,{n:r['cells'][n] for n in ours},{n:r['audited_cells'][n] for n in ours},pairs,r['r4_original_h84_warmup_pairing'],SEEDS)
    for key in ('per_seed','paired_ours_changes','integrity_checks_passed'):
        if computed[key]!=r[key]:raise ValueError('Comparison cannot be reproduced: '+key)
    for part,summary in computed['goal_summary'].items():
        for key,value in summary.items():close(value,r['goal_summary'][part][key])
    return r


def figures(bundle,r,out):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'r4_formal_plot'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    saved=[]
    def save(fig,name):
        for ext in ('png','svg'):fig.savefig(out/(name+'.'+ext),dpi=170,bbox_inches='tight')
        plt.close(fig);saved.append(name)
    x=[100*v for v in full.RATIOS.values()]
    for seed in SEEDS:
        suffix='_s'+str(seed)
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for col,d in enumerate(full.PARTITIONS):
            for row,key in enumerate(('accuracy','source_transfer_asr')):
                ax=axes[row,col]
                for arm in full.METHODS:
                    vals=[100*r['cells'][f'{arm}_{d}_{label}'+suffix]['tail'][key] for label in full.RATIOS]
                    ax.plot(x,vals,color=COLORS[arm],ls=STYLES[arm],marker='o',lw=2.5 if arm=='H84' else 1.7,label=LABELS[arm])
                ax.set(title=original_render.LABELS[d],xticks=x,xlabel='Malicious clients (%)',ylabel='Accuracy (%)' if row==0 else 'ASR (%)',ylim=(0,70 if row==0 else 102));ax.grid(alpha=.2)
        axes[0,0].legend(ncol=2,fontsize=8)
        fig.suptitle(f'CIFAR-10 | seed {seed} | tail 10 rounds\nUser-requested fixed Ours R4; other five methods reused')
        save(fig,'tail10'+suffix)
        for d in full.PARTITIONS:
            fig,axes=plt.subplots(2,5,figsize=(19,7),sharex=True,sharey='row',layout='constrained')
            for col,label in enumerate(full.RATIOS):
                for row,key in enumerate(('accuracy','source_transfer_asr')):
                    ax=axes[row,col]
                    for arm in full.METHODS:
                        vals=[100*v[key] for v in bundle['trajectories'][f'{arm}_{d}_{label}'+suffix]]
                        ax.plot(range(151),vals,color=COLORS[arm],ls=STYLES[arm],lw=2 if arm=='H84' else 1.4,label=LABELS[arm])
                    if label!='clean':ax.axvline(25,color='#64748B',ls=':',lw=.8)
                    ax.set(title=f'{100*full.RATIOS[label]:.0f}% malicious',xlim=(0,150),ylim=(0,70 if row==0 else 102),xlabel='Round' if row else '')
                    if col==0:ax.set_ylabel('Accuracy (%)' if row==0 else 'ASR (%)')
                    ax.grid(alpha=.15)
            axes[0,0].legend(ncol=2,fontsize=8)
            fig.suptitle(f'{original_render.LABELS[d]} | seed {seed} | full 0–150 rounds\nFixed Ours R4; no smoothing; dotted vertical line = attack starts')
            save(fig,'trajectories_'+d+suffix)
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,key,title in zip(axes,('runtime_with_crypto_seconds','runtime_without_crypto_seconds'),('Total worker wall time','Worker wall time minus crypto spans')):
            vals=[sum(c['timing'][key] for c in r['cells'].values() if c['arm']==arm and c['config']['seed']==seed)/60 for arm in full.METHODS]
            ax.bar([LABELS[a] for a in full.METHODS],vals,color=[COLORS[a] for a in full.METHODS],edgecolor='#64748B',linewidth=.5)
            for i,v in enumerate(vals):ax.text(i,v+max(vals)*.025,f'{v:.1f}',ha='center',fontsize=9)
            ax.set(title=title,ylabel='Sum of 10 task wall minutes',ylim=(0,max(vals)*1.2));ax.tick_params(axis='x',labelrotation=25);ax.grid(axis='y',alpha=.15)
        fig.suptitle(f'Seed {seed} | measured cost on shared GPUs / CPUs\nSubtraction is not a crypto-disabled rerun; not an isolated speed benchmark')
        save(fig,'timing'+suffix)
    return saved


def document(r,out,digest):
    tab=original_render.table
    f=lambda x:f'{x:.4f}'
    body=['<h1>CIFAR-10：固定 R4 三 seed 官方 test 补做</h1>',
        '<p>按用户2026-11-01明确指定，新做Ours R4共30项（3 seed × 2分布 × 5比例），每项150轮；其他五法150项及曲线逐项复用cifar-paper。</p>',
        '<p>R4只将原公式恢复系数1.25改为4，裁剪参数仍为1。原验证IID增益0.4794pp未通过冻结0.5pp门槛，原selected=null和首批95/96配对失败均保留。本次属于已见原test、seed及验证结果后的用户指定补做，不宣称原验证筛选通过；新test不回流调参或择优删场景。</p>',
        '<p>尾10=141–150轮，全攻击期=25–150轮；clean ASR为背景转移率。官方test共10000图，ASR使用source5的1000图，不能与验证7500/750图混算。训练40000、aux2500、CNN和V2攻击保持cifar-paper口径；并非论文完整实验安排逐位复现。</p>',
        f'<p>当前完整配对/暖身审计通过：{r["integrity_checks_passed"]}。所有健康失败及失配原值保留，详见JSON；失败不触发自动重试。</p>']
    rows=[]
    for part,x in r['goal_summary'].items():
        rows.append([part,f(100*x['r4_mean_accuracy']),f(100*x['r4_seed_mean_accuracy_sample_sd']),x['n_seeds'],
            f(x['gap_to_best_method_mean_pp']),f(x['mean_paired_gap_pp']),f(100*x['r4_attack_tail_asr']),
            f(x['mean_accuracy_change_pp']),f(x['worst_accuracy_change_pp']),f(x['worst_full_attack_asr_change_pp']),
            f"{x['within_1pp']}/{x['scenarios']}",f"{x['within_1_5pp']}/{x['scenarios']}"])
    body+=[tab(['分布','均Acc%','seed均Acc样本SD%','n','距最高方法均值pp','逐场景距第一均pp','攻击ASR%','Acc均变化pp','最坏Acc变化pp','最坏全期ASR变化pp','≤1pp','≤1.5pp'],rows)]
    for seed in SEEDS:
        body.append(f'<h2>seed {seed}：独立六法比较</h2>')
        for name in ('tail10','trajectories_iid','trajectories_dirichlet','timing'):
            path=f'{name}_s{seed}'
            body.append(f'<figure><img src="{path}.png" alt="{path}"><figcaption><a href="{path}.svg">SVG</a></figcaption></figure>')
        rows=[]
        for name,x in r['paired_ours_changes'].items():
            if x['seed']!=seed:continue
            rows.append([x['partition'],int(x['malicious_ratio']*100),f(x['r4_tail']['accuracy']*100),
                f(x['r4_tail']['source_transfer_asr']*100),f(x['iid_or_non_iid_raw_accuracy_gap_pp']),
                ','.join(LABELS[a] for a in x['raw_best_accuracy_methods']),
                *[f(x['delta_pp'][period][key]) for period in ('tail','attack_period') for key in ('accuracy','source_transfer_asr')]])
        body.append(tab(['分布','恶意%','R4 Acc%','R4 ASR%','距六法第一pp','第一方法','尾Acc变化pp','尾ASR变化pp','全期Acc变化pp','全期ASR变化pp'],rows))
    rows=[]
    for part in full.PARTITIONS:
        for label in full.RATIOS:
            for arm in full.METHODS:
                group=r['groups'][f'{arm}_{part}_{label}'];samples=group['samples']
                rows.append([part,label,LABELS[arm],group['actual_n'],
                    *[f"{100*samples[k]['mean']:.4f} ± {100*samples[k]['sample_sd']:.4f}" for k in ('accuracy','asr')]])
    body+=['<h2>跨seed尾10均值 ± 样本SD（全部原值、不按健康删样本）</h2>',tab(['分布','比例','方法','n','Acc%','ASR%'],rows)]
    ranks=[];health=[];criteria=[]
    for seed in SEEDS:
        saved=r['per_seed'][str(seed)];current=saved['with_previous_krum_exception'];strict=saved['strict']
        for name,s in current['scenarios'].items():
            criteria.append([seed,name,s['passed'],strict['scenarios'][name]['passed'],f(s['accuracy_gap_pp']),f(s['asr_gap_pp'])])
            for arm in full.METHODS:
                x=s['raw_metrics'][arm];ranks.append([seed,name,LABELS[arm],f(x['accuracy']*100),f(x['asr']*100),s['raw_accuracy_ranks'][arm],s['raw_asr_ranks'][arm],arm in s['eligible_methods']])
        for part,methods in current['method_health'].items():
            for arm,h in methods.items():health.append([seed,part,LABELS[arm],h['valid'],f(h['clean_accuracy_drop']*100),'; '.join(h['reasons']) or '—',h.get('waived_clean_utility_only',False)])
    body+=['<h2>完整180原值、原排名与健康资格</h2>',tab(['seed','场景','方法','Acc%','ASR%','Acc排名','ASR排名','参照资格'],ranks),
        tab(['seed','分布','方法','资格','clean Acc降幅pp','全部失败原因','仅Krum效用例外'],health),
        '<h2>原描述性标准：严格与既有Krum例外并列</h2>',tab(['seed','场景','既有例外通过','严格通过','Acc差pp','ASR差pp'],criteria),
        '<p>原标准按健康参照计算；上方用户目标差距按六法全部原值计算。Krum例外仅限IID clean效用，不改变原始排名；不以任一视图重选R4。TAD内部谱状态不声称逐位回放。</p>',
        '<p>共享GPU实测总用时及减去crypto区间的用时不等同独占速度测试；减法不是关闭密码学重新运行。历史全部验证失败、初始化失败及成本保留于accepted_baseline_metadata/prior_first_panel及原归档。</p>',
        '<p><a href="combined_results.json">完整指标、全期及末轮原值、健康、审计JSON</a> · <a href="render_manifest.json">绘图来源及SHA记录</a></p>',f'<p>新导出SHA256：<code>{digest}</code></p>']
    css='body{font:15px/1.6 system-ui;max-width:1500px;margin:32px auto;padding:0 20px;color:#374151;background:#fafafa}img{width:100%}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px;width:100%;background:white}td,th{padding:7px;border:1px solid #ddd;white-space:nowrap;text-align:right}th{background:#eee}code{overflow-wrap:anywhere}'
    (out/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>CIFAR-10 Ours R4</title><style>'+css+'</style>'+''.join(body)+'</html>')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/base.OUTPUT/'export/combined_bundle.json')
    p.add_argument('--baseline',type=Path,default=ROOT/'outputs/cifar-paper/export/combined_bundle.json')
    p.add_argument('--expected-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=a.source.read_bytes();digest=sha(raw)
    if digest!=a.expected_sha256:raise ValueError('Pinned R4 export SHA differs')
    baseline=a.baseline.read_bytes();bundle=json.loads(raw);r=validate(bundle,baseline)
    a.output.mkdir(parents=True,exist_ok=False)
    names=figures(bundle,r,a.output);document(r,a.output,digest)
    (a.output/'combined_results.json').write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+'\n')
    if sha(a.source.read_bytes())!=digest or sha(a.baseline.read_bytes())!=base.BASELINE_SHA:raise ValueError('Input changed during rendering')
    record=dict(source_sha256=digest,baseline_sha256=base.BASELINE_SHA,renderer_sha256=sha(Path(__file__).read_bytes()),
        all180_cells_validated=True,all30_ours_replaced=True,other150_values_and_curves_unchanged=True,
        old_top20_plotted=False,figures=names,mnist_palette=COLORS,visual_inspection_pending=True,
        outputs={f.name:sha(f.read_bytes()) for f in sorted(a.output.iterdir())})
    (a.output/'render_manifest.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(report=str(a.output/'report.html'),figures=len(names),source_sha256=digest)))


if __name__=='__main__':main()
