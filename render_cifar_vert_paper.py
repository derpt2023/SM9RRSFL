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
import run_cifar_vert_paper_formal as selected_run
from collections import Counter
import cifar_h84_formal_report as formal_report
import run_cifar_h84_full_validation as full

ROOT=Path(__file__).resolve().parent
SEEDS=original_render.SEEDS
COLORS=dict(H84='#8DBAD9',VERT='#E99092',AlignIns='#8ADDE4',Krum='#DCDD8F',TAD='#FEBD85',FedAvg='#C4A9A2')
STYLES=dict(H84='-',VERT='--',AlignIns=':',Krum='-.',TAD=(0,(5,2,1,2)),FedAvg=(0,(2,2)))
LABELS=dict(H84='Ours (H84)',VERT='VERT (paper)',AlignIns='AlignIns',Krum='Krum',TAD='TAD',FedAvg='FedAvg')
sha=original_render.sha
close=original_render.close
DEFAULT=ROOT/'outputs/cifar_h84_formal_vert_paper150_v1/export/combined_bundle.json'
PARENT=original_render.DEFAULT


def validate(bundle, parent_raw):
    parent=json.loads(parent_raw);old,_=original_render.validate(parent)
    r=bundle['report'];m=bundle['manifest'];protocol=m['protocol']
    if (bundle['schema']!='h84-vert-paper-selected-export-v1'
            or r['status']!='complete_audited_vert_paper_replacement'
            or not r['posthoc_user_requested_protocol_correction'] or not r['no_formal_retuning_or_selection']
            or r['old_vert_in_current_tables'] or not bundle['source_verified'] or not bundle['all_inputs_unchanged']
            or r['fingerprint']!=m['fingerprint'] or len(m['sources'])!=105
            or sha(parent_raw)!=bundle['original_export_sha256']
            or protocol['original_export_sha256']!=bundle['original_export_sha256']
            or bundle['original_report_sha256']!=parent['files']['analysis/formal_results.json']['sha256']):
        raise ValueError('Paper-core export identity or provenance differs')
    if (protocol['parameters']!=selected_run.PARAMETERS or protocol['retune_on_formal_test'] or protocol['rank_target'] is not None
            or protocol['validation_report_sha256']!=selected_run.VALIDATION_SHA
            or r['selected_validation_report_sha256']!=selected_run.VALIDATION_SHA
            or m['validation_report_sha256']!=selected_run.VALIDATION_SHA
            or m['official_test_used_for_selection'] or not m['prior_test_results_seen']
            or not m['formal_results_cannot_change_selected_configuration']
            or len(protocol['validation_frozen_sources'])!=100
            or any(m['sources'].get(k)!=h for k,h in protocol['validation_frozen_sources'].items())):
        raise ValueError('Fixed validation selection and frozen100-source lineage required')
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
            config=deepcopy(old['cells'][name]['config']);config.update(selected_run.PARAMETERS)
            lineage=r['replacement_lineage'][name];selection=a['selection']
            source=protocol['output_dir']+'/tasks/'+name+'/metrics.json'
            if (c['config']!=config or source not in r['input_sha256']
                    or lineage['config_changed_fields']!=['vert_top_k','vert_predict_lr']
                    or not lineage['paper_core_adapter'] or lineage['selector_only_adapter']
                    or lineage['validation_report_sha256']!=selected_run.VALIDATION_SHA
                    or not a['actual_weights_verified']
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
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'vert_paper_plot'))
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
        fig.suptitle(f'CIFAR-10 | seed {seed} | tail 10 rounds\nValidation-selected VERT paper core; all six methods retained')
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
            fig.suptitle(f'{original_render.LABELS[d]} | seed {seed} | full 0–150 rounds\nVERT paper core; no smoothing; dotted vertical line = attack starts')
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


def document(r,out,digest,execution):
    esc=lambda x:html.escape(str(x))
    tab=original_render.table
    f=lambda x:f'{x:.3f}'
    body=['<h1>CIFAR-10 六方法：VERT 原文核心三seed正式补跑</h1>',
          '<p>三个原seed，各含两种分布、五种恶意比例；新VERT为30项×150轮，其余五法150项原值保留。下列新图与当前对比全部使用验证后固定的原文核心VERT。</p>',
          '<p>这是查看先前官方test后按用户要求进行的后续开发：原文核心在独立seed2026102901、训练衍生7500验证图上通过预注册整体提升条件后固定。没有再按正式test成绩改参、替换seed、择优场景或控制名次。采用dense256、raw A/B、每轮新投影器和预测器、history10、5epoch、Adam.01；公共CNN、本地训练与V2定向攻击保持一致，因此不是作者完整基准的逐位复现。论文VI-A主比较Top-κ，VI-C3未知比例采用K-means(K=2)，本次沿用后者。验证改善集中IID，Dirichlet高攻击比例仍弱，正式全部场景保留。</p>',
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
           f'<p>本批执行尝试 {execution["execution_attempts"]} 次，对应30科学单元，逐任务只有一次启动及成功退出，未发现本批失败或恢复。原H84初始化失败5.421214秒与先前K-means补跑初始化失败2.178092秒仍独立归档，不计入本批成功任务耗时。</p>',
           '<p><a href="combined_results.json">当前合并报告JSON</a> · <a href="execution_audit.json">本批执行审计</a> · <a href="selected_validation_results.json">选参验证原件</a> · <a href="render_manifest.json">绘图核验记录</a></p>',f'<p>当前导出SHA256：<code>{digest}</code></p>']
    css='body{font:15px/1.6 system-ui;margin:32px auto;max-width:1450px;padding:0 20px;color:#374151;background:#fafafa}h2{margin-top:40px}img{width:100%;background:white}figure{margin:24px 0}.scroll{overflow:auto}table{border-collapse:collapse;background:white;font-size:12px;width:100%}td,th{padding:7px;border:1px solid #ddd;white-space:nowrap;text-align:right}th{background:#eee}code{overflow-wrap:anywhere}'
    (out/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>CIFAR-10 VERT paper core</title><style>'+css+'</style>'+''.join(body)+'</html>')



def validation_evidence(raw):
    if sha(raw)!=selected_run.VALIDATION_SHA:raise ValueError('Exact selected validation report required')
    report=json.loads(raw)
    if (report['status']!='complete_audited_validation' or report['evaluation_scope']!='training_validation_7500'
            or report['official_test_used_for_selection'] or len(report['execution_audits'])!=30
            or report['decision']!=selected_run.validation.assess(report) or not report['decision']['passed']):
        raise ValueError('Predeclared validation rule must pass')
    return report


def audit_execution_events(events, expected):
    starts=[e for e in events if e['event']=='worker_started']
    ends=[e for e in events if e['event']=='worker_exited']
    counts=Counter({name:1 for name in expected})
    if Counter(e['task'] for e in starts)!=counts or Counter(e['task'] for e in ends)!=counts:
        raise ValueError('Exactly one recorded attempt per formal cell required; review any recovery separately')
    if any(e['event']=='dispatch_blocked' for e in events):raise ValueError('Dispatch failure retained for review')
    if any(e['returncode']!=0 or e['status']!='complete' for e in ends):raise ValueError('Worker failure retained for review')
    by_task={e['task']:e for e in starts}
    for e in ends:
        start=by_task[e['task']];gpu=start['admission']
        if (e['child_pid']!=start['child_pid'] or e['gpu_uuid']!=gpu['uuid'] or gpu['uuid'] in selected_run.QUARANTINE
                or gpu['free_mb']<8192):raise ValueError('Attempt identity or GPU admission differs')
    return dict(execution_attempts=len(starts),scientific_units=len(expected),failed_attempts=0,
                one_attempt_per_task=True,all_workers_exited_successfully=True)


def collect_execution(source, bundle):
    """Read Station controller/events only after final export; never starts or resumes training."""
    out=source.parent.parent;controller=json.loads((out/'controller_status.json').read_text())
    if (controller['status']!='panel_terminal' or controller['active'] or controller['pending'] or controller['failure']):
        raise ValueError('Terminal successful controller required')
    proc=Path('/proc')/str(controller['pid'])/'cmdline'
    if not Path('/proc').is_dir():raise ValueError('Collect execution evidence on AI Station')
    if proc.exists() and b'run_cifar_vert_paper_formal.py' in proc.read_bytes():
        raise ValueError('Main process still running; wait for the existing audit to exit')
    if (out/'recovery').exists():raise ValueError('Recovery exists; requires separately reviewed execution audit')
    paths=sorted((out/'scheduler_vert_paper_formal_v1').glob('*/events.jsonl'))
    events=[json.loads(line) for path in paths for line in path.read_text().splitlines()]
    names={x['task_id'] for x in bundle['manifest']['specs']}
    if len(names)!=30:raise ValueError('Exactly30 scientific units required')
    result=audit_execution_events(events,names)
    inputs={str(p.relative_to(out)):sha(p.read_bytes()) for p in [out/'controller_status.json',*paths]}
    result.update(status='complete_execution_audit',source_sha256=sha(source.read_bytes()),
        combined_report_sha256=bundle['combined_report_sha256'],fingerprint=bundle['manifest']['fingerprint'],input_sha256=inputs)
    target=out/'analysis/execution_audit.json'
    encoded=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode()
    if target.exists() and target.read_bytes()!=encoded:raise ValueError('Existing audit differs; preserve it')
    if not target.exists():target.write_bytes(encoded)
    return result


def validate_execution(execution,bundle,digest):
    if (execution['status']!='complete_execution_audit' or execution['source_sha256']!=digest
            or execution['combined_report_sha256']!=bundle['combined_report_sha256']
            or execution['fingerprint']!=bundle['manifest']['fingerprint']
            or execution['execution_attempts']!=30 or execution['scientific_units']!=30 or execution['failed_attempts']!=0
            or not execution['one_attempt_per_task'] or not execution['all_workers_exited_successfully']
            or not execution['input_sha256']):
        raise ValueError('Complete current-batch execution audit required; never assume no failed attempts')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=DEFAULT)
    p.add_argument('--parent',type=Path,default=PARENT);p.add_argument('--expected-sha256',required=True)
    p.add_argument('--validation',type=Path,default=ROOT/'outputs/cifar_vert_paper_validation150_v1/analysis/paper_validation_results.json')
    p.add_argument('--execution-audit',type=Path);p.add_argument('--audit-execution',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();raw=a.source.read_bytes();digest=sha(raw)
    if digest!=a.expected_sha256:raise ValueError('Pinned new export SHA differs')
    bundle=json.loads(raw);validation_raw=a.validation.read_bytes();validation_evidence(validation_raw)
    r=validate(bundle,a.parent.read_bytes())
    if a.audit_execution:
        print(json.dumps(collect_execution(a.source,bundle)));return
    ep=a.execution_audit or a.source.parent.parent/'analysis/execution_audit.json'
    eraw=ep.read_bytes();execution=json.loads(eraw);validate_execution(execution,bundle,digest)
    out=a.output or a.source.parent.parent/'report';out.mkdir(parents=True,exist_ok=False)
    names=figures(bundle,r,out);document(r,out,digest,execution)
    (out/'combined_results.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    (out/'execution_audit.json').write_bytes(eraw);(out/'selected_validation_results.json').write_bytes(validation_raw)
    if sha(a.source.read_bytes())!=digest:raise ValueError('Export changed during rendering')
    record=dict(source_sha256=digest,parent_sha256=sha(a.parent.read_bytes()),renderer_sha256=sha(Path(__file__).read_bytes()),
        selected_validation_sha256=sha(validation_raw),execution_audit_sha256=sha(eraw),
        all180_cells_validated=True,all30_vert_replaced=True,other150_values_unchanged=True,
        per_seed_assessments_recomputed=True,old_top20_plotted=False,mnist_palette=COLORS,
        figures=names,outputs={f.name:sha(f.read_bytes()) for f in sorted(out.iterdir())})
    (out/'render_manifest.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(report=str(out/'report.html'),figures=len(names),source_sha256=digest)))

if __name__=='__main__':main()
