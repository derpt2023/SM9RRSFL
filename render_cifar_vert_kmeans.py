"""Render the corrected six-method comparison, with separate MNIST-palette seed figures."""
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
import cifar_h84_formal_report as formal_report
import run_cifar_h84_full_validation as full

ROOT=Path(__file__).resolve().parent
SEEDS=original_render.SEEDS
COLORS=dict(H84='#8DBAD9',VERT='#E99092',AlignIns='#8ADDE4',Krum='#DCDD8F',TAD='#FEBD85',FedAvg='#C4A9A2')
STYLES=dict(H84='-',VERT='--',AlignIns=':',Krum='-.',TAD=(0,(5,2,1,2)),FedAvg=(0,(2,2)))
LABELS=dict(H84='Ours (H84)',VERT='VERT (K-means)',AlignIns='AlignIns',Krum='Krum',TAD='TAD',FedAvg='FedAvg')
sha=original_render.sha
close=original_render.close
DEFAULT=ROOT/'outputs/cifar_h84_formal_vert_kmeans150_v1/export/combined_bundle.json'
PARENT=original_render.DEFAULT


def validate(bundle, parent_raw):
    parent=json.loads(parent_raw);old,_=original_render.validate(parent)
    r=bundle['report'];m=bundle['manifest'];protocol=m['protocol']
    if (bundle['schema']!='h84-vert-kmeans-corrected-export-v1'
            or r['status']!='complete_audited_vert_kmeans_replacement'
            or not r['posthoc_user_requested_protocol_correction'] or not r['no_retuning_or_selection']
            or r['old_vert_in_current_tables'] or not bundle['source_verified'] or not bundle['all_inputs_unchanged']
            or r['fingerprint']!=m['fingerprint'] or len(m['sources'])!=95
            or sha(parent_raw)!=bundle['original_export_sha256']
            or protocol['original_export_sha256']!=bundle['original_export_sha256']
            or bundle['original_report_sha256']!=parent['files']['analysis/formal_results.json']['sha256']):
        raise ValueError('Corrected export identity or provenance differs')
    encoded=(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if sha(encoded)!=bundle['combined_report_sha256']:raise ValueError('Combined report digest differs')
    names=set(old['cells']);vert={k for k,c in old['cells'].items() if c['arm']=='VERT'}
    if (set(r['cells'])!=names or set(r['audited_cells'])!=names or set(bundle['trajectories'])!=names
            or set(r['replacement_lineage'])!=vert or len(vert)!=30):
        raise ValueError('All180 cells and30 new VERT cells required')
    for name,c in r['cells'].items():
        rows=bundle['trajectories'][name];a=r['audited_cells'][name]
        if c['arm']!='VERT':
            if (c!=old['cells'][name] or a!=old['audited_cells'][name] or rows!=parent['trajectories'][name]):
                raise ValueError('Other five methods changed')
        else:
            config=deepcopy(old['cells'][name]['config']);config['vert_top_k']=0
            lineage=r['replacement_lineage'][name];selection=a['selection']
            source=protocol['output_dir']+'/tasks/'+name+'/metrics.json'
            if (c['config']!=config or source not in r['input_sha256']
                    or lineage['config_changed_fields']!=['vert_top_k']
                    or not lineage['same_seed_initial_model_data_targets_malicious_ids']
                    or not lineage['unchanged_bootstrap_rounds_match']
                    or not selection['actual_scores_labels_and_selection_replayed']
                    or selection['rounds_verified']!=150 or selection['kmeans_rounds']!=148
                    or len(selection['selected_counts'])!=150):
                raise ValueError('Replacement VERT lineage or actual selection incomplete')
        if (c['status']!='complete' or not c['complete_finite'] or c['completed_round']!=150
                or a['summary']!=c or [x['round'] for x in rows]!=list(range(151))):
            raise ValueError('Incomplete evidence cannot be plotted as complete')
        for key in ('accuracy','source_transfer_asr','honest_weight_loss','malicious_weight_mass'):
            if any(not math.isfinite(x[key]) or not -1e-9<=x[key]<=1+1e-9 for x in rows):
                raise ValueError('Invalid curve value')
            close(stats.mean(x[key] for x in rows[-10:]),c['tail'][key]);close(rows[-1][key],c['final'][key])
        for key in ('accuracy','source_transfer_asr'):
            close(stats.mean(x[key] for x in rows[25:]),c['attack_period'][key])
        close(max(x['source_transfer_asr'] for x in rows[25:]),c['peak_attack_asr'])
        if rows[-1]['false_positive_revocations']!=c['final_fp']:raise ValueError('FP differs')
        attack=a['attack']
        if not (attack['complete_coverage'] and attack['target_views_and_optimizers_verified']
                and attack['actual_calls']==attack['expected_calls']):raise ValueError('Attack audit incomplete')
        t=c['timing'];close(t['runtime_with_crypto_seconds']-t['crypto_wall_seconds'],t['runtime_without_crypto_seconds'])
        close(sum(t['stages'][k] for k in ('crypto_setup_wall_seconds','crypto_packet_wall_seconds',
            'crypto_audit_wall_seconds','crypto_finalize_wall_seconds')),t['crypto_wall_seconds'])
        if not t['available_and_consistent']:raise ValueError('Timing unavailable')
    original_render.validate_groups(r['cells'],r['groups'])
    fields=('accuracy','source_transfer_asr','non_source_accuracy','aux128_confidence')
    for seed in SEEDS:
        saved=r['per_seed'][str(seed)];suffix='_s'+str(seed)
        for task in full.all_tasks():
            if not task['ratio']:continue
            name=task['id']+suffix;clean=f"{task['arm']}_{task['partition']}_clean"+suffix
            prefix=all(bundle['trajectories'][name][i][k]==bundle['trajectories'][clean][i][k] for i in range(25) for k in fields)
            if prefix!=saved['pairing'][task['id']]['classification_prefix_equal']:raise ValueError('Prefix differs')
        if saved['fedavg_attack_gates']!=old['per_seed'][str(seed)]['fedavg_attack_gates']:raise ValueError('FedAvg gates changed')
        selected={t['id']:r['audited_cells'][t['id']+suffix] for t in full.all_tasks()}
        strict=full.assess(selected,saved['fedavg_attack_gates'],saved['pairing'])
        current=formal_report.with_krum_exception(strict)
        if strict!=saved['strict'] or current!=saved['with_previous_krum_exception']:
            raise ValueError('Comparative rankings or criteria cannot be reproduced')
    return r


def figures(bundle,r,out):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'vert_kmeans_plot'))
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
        fig.suptitle(f'CIFAR-10 | seed {seed} | tail 10 rounds\nVERT K-means supplement; all six methods retained')
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
            fig.suptitle(f'{original_render.LABELS[d]} | seed {seed} | full 0–150 rounds\nVERT K-means; no smoothing; dotted vertical line = attack starts')
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


def document(r,out,digest,recovery):
    esc=lambda x:html.escape(str(x))
    tab=original_render.table
    f=lambda x:f'{x:.3f}'
    body=['<h1>CIFAR-10 六方法：VERT K-means 补充实验</h1>',
          '<p>三个原seed，各含两种分布、五种恶意比例；新VERT为30项×150轮，其余五法150项原值保留。下列新图与当前对比只使用K-means版VERT。</p>',
          '<p>这是查看先前测试结果后，按用户要求修正筛选方式的补充实验；未搜索超参数、替换seed或根据成绩挑选版本。仍保留本项目预测器/投影适配，不能称作者完整模型的逐位复现。正式论文VI-A主比较统一Top-κ，VI-C3讨论未知投毒比例时K-means（K=2），本次采用后者。</p>',
          '<p>配色来自MNIST实验；各seed独立出图。所有方法及健康失败保留，clean ASR为背景转移率。尾10=141..150；全攻击期=25..150，clean也使用同窗口。</p>']
    for seed in SEEDS:
        current=r['per_seed'][str(seed)]['with_previous_krum_exception'];strict=r['per_seed'][str(seed)]['strict']
        passed=sum(s['passed'] for s in current['scenarios'].values());sp=sum(s['passed'] for s in strict['scenarios'].values())
        body.append(f'<h2>seed {seed}</h2><p>H84当前描述性标准 {passed}/10；原严格口径 {sp}/10。Krum例外仅限IID clean精度效用，不等于原效用通过。</p>')
        for name in ('tail10','trajectories_iid','trajectories_dirichlet','timing'):
            path=f'{name}_s{seed}'
            body.append(f'<figure><img src="{path}.png" alt="{esc(path)}"><figcaption><a href="{path}.svg">下载SVG</a></figcaption></figure>')
        rows=[]
        for name,s in current['scenarios'].items():
            rows.append([name,','.join(s['eligible_methods']),s['accuracy_rank'],s['asr_rank'],f(s['accuracy_gap_pp']),f(s['asr_gap_pp']),s['passed'],strict['scenarios'][name]['passed']])
        body.append(tab(['场景','健康参照','H84 Acc名次','ASR名次','Acc差距pp','ASR差距pp','当前通过','严格通过'],rows))
    body.append('<h2>跨seed均值±样本SD（n=3，健康失败不删）</h2>')
    rows=[]
    pm=lambda vals,scale=1:f'{stats.mean(vals)*scale:.3f} ± {stats.stdev(vals)*scale:.3f}'
    for d in full.PARTITIONS:
        for label in full.RATIOS:
            for arm in full.METHODS:
                cs=[r['cells'][f'{arm}_{d}_{label}_s{s}'] for s in SEEDS]
                vals=[pm([c[phase][key] for c in cs],100) for phase in ('tail','final','attack_period') for key in ('accuracy','source_transfer_asr')]
                rows.append([d,label,LABELS[arm],3,*vals,pm([c['peak_attack_asr'] for c in cs],100),pm([c['timing']['runtime_with_crypto_seconds'] for c in cs]),pm([c['timing']['runtime_without_crypto_seconds'] for c in cs])])
    body.append(tab(['分布','比例','方法','n','尾10Acc%','尾10ASR%','最终Acc%','最终ASR%','全攻击期Acc%','全攻击期ASR%','逐seed峰ASR%','总墙钟秒','扣crypto秒'],rows))
    body.append('<h2>六法完整原始排名与健康资格</h2>');ranks=[];health=[]
    for seed in SEEDS:
        current=r['per_seed'][str(seed)]['with_previous_krum_exception']
        for name,s in current['scenarios'].items():
            for arm in full.METHODS:
                x=s['raw_metrics'][arm];ranks.append([seed,name,LABELS[arm],f(x['accuracy']*100),f(x['asr']*100),s['raw_accuracy_ranks'][arm],s['raw_asr_ranks'][arm],arm in s['eligible_methods']])
        for d,methods in current['method_health'].items():
            for arm,h in methods.items():health.append([seed,d,LABELS[arm],h['valid'],f(h['clean_accuracy_drop']*100),'; '.join(h['reasons']) or '—',h.get('waived_clean_utility_only',False)])
    body.append(tab(['seed','场景','方法','Acc%','ASR%','Acc排名','ASR排名','参照资格'],ranks))
    body.append(tab(['seed','分布','方法','资格','clean Acc降幅pp','全部原因','仅Krum例外'],health))
    rows=[]
    for name,c in sorted(r['cells'].items()):
        t=c['timing'];mp=t['measured_persistence']
        rows.append([name,f(c['tail']['accuracy']*100),f(c['tail']['source_transfer_asr']*100),c['health']['complete_observed_health_pass'],c['final_fp'],f(t['runtime_with_crypto_seconds']),f(t['runtime_without_crypto_seconds']),f(t['crypto_wall_seconds']),f(mp['checkpoint_write_seconds']),f(mp['structured_persistence_seconds'])])
    body+=['<h2>180项当前结果与实测开销</h2>',tab(['任务','尾10Acc%','尾10ASR%','单元健康','FP人数','总秒','扣crypto秒','crypto秒','checkpoint秒','结构化写盘秒'],rows),
           '<p>总墙钟含加载、训练、评估、观测、审计及I/O。扣crypto为减去四个互不重叠协议墙钟区间，不是关闭密码学重跑；不扣并行签名/验签累计值。checkpoint是结构化写盘的子集，不能相加重复扣除。共享资源用时不能视为独占算法速度对比。</p>',
           '<p>健康资格按同seed同分布的clean效用和五比例整体核定，H84也须健康。clean距健康最优Acc/ASR≤1/2pp，无名次要求；攻击双前二且差≤2/5pp。只作描述，不据此重新选参。TAD内部谱状态不声称逐位回放；Krum原Byzantine计数和前缀例外保持披露。</p>',
           f'<p>补跑执行尝试 {recovery["supplement_execution_attempts"]} 次，30科学单元；初始化失败额外墙钟 {recovery["failed_attempt_elapsed_seconds"]:.6f} 秒，未完成任何训练轮，失败原件已归档。此前正式实验另有一次已封存初始化失败5.421214秒。未将失败成本混入成功任务时间或隐藏。</p>',
           '<p><a href="combined_results.json">当前合并报告JSON</a> · <a href="recovery_final_audit.json">补跑恢复审计</a> · <a href="render_manifest.json">绘图核验记录</a></p>',f'<p>当前导出SHA256：<code>{digest}</code></p>']
    css='body{font:15px/1.6 system-ui;margin:32px auto;max-width:1450px;padding:0 20px;color:#374151;background:#fafafa}h2{margin-top:40px}img{width:100%;background:white}figure{margin:24px 0}.scroll{overflow:auto}table{border-collapse:collapse;background:white;font-size:12px;width:100%}td,th{padding:7px;border:1px solid #ddd;white-space:nowrap;text-align:right}th{background:#eee}code{overflow-wrap:anywhere}'
    (out/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>CIFAR-10 VERT K-means</title><style>'+css+'</style>'+''.join(body)+'</html>')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=DEFAULT)
    p.add_argument('--parent',type=Path,default=PARENT);p.add_argument('--expected-sha256',required=True)
    p.add_argument('--recovery',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();raw=a.source.read_bytes();digest=sha(raw)
    if digest!=a.expected_sha256:raise ValueError('Pinned new export SHA differs')
    bundle=json.loads(raw);r=validate(bundle,a.parent.read_bytes())
    recovery_path=a.recovery or a.source.parent.parent/'analysis/recovery_final_audit.json'
    recovery=json.loads(recovery_path.read_bytes())
    if (recovery['combined_report_sha256']!=bundle['combined_report_sha256']
            or not recovery['original_failed_attempt_retained'] or recovery['supplement_execution_attempts']!=31
            or not recovery['same_task_identity_data_targets'] or recovery['extra_completed_training_rounds']!=0):
        raise ValueError('Supplement failure/recovery evidence required')
    out=a.output or a.source.parent.parent/'report';out.mkdir(parents=True,exist_ok=False)
    names=figures(bundle,r,out);document(r,out,digest,recovery)
    (out/'combined_results.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    (out/'recovery_final_audit.json').write_bytes(recovery_path.read_bytes())
    if sha(a.source.read_bytes())!=digest:raise ValueError('Export changed during rendering')
    record=dict(source_sha256=digest,parent_sha256=sha(a.parent.read_bytes()),renderer_sha256=sha(Path(__file__).read_bytes()),
        all180_cells_validated=True,all30_vert_replaced=True,other150_values_unchanged=True,
        per_seed_assessments_recomputed=True,old_top20_plotted=False,mnist_palette=COLORS,
        figures=names,outputs={f.name:sha(f.read_bytes()) for f in sorted(out.iterdir())})
    (out/'render_manifest.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(report=str(out/'report.html'),figures=len(names),source_sha256=digest)))

if __name__=='__main__':main()
