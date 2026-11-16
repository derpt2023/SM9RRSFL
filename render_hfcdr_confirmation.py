#!/usr/bin/env python3
"""Read-only scientific data to standalone PNG/SVG figures and HTML dashboard."""
from __future__ import annotations

import argparse
from collections import Counter
from html import escape
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/hfcdr-study-matplotlib')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import hfcdr_validation_protocol as p
from hfcdr_validation_analysis import report
import hfcdr_confirmation as confirmation
from hfcdr_health_context import context

METHODS=('hfcdr','vert','alignins','tad','krum','fedavg')
LABELS=dict(zip(METHODS,('HFCDR-FL','VERT','AlignIns','TAD','Krum','FedAvg')))
COLORS=dict(zip(METHODS,('#4D91BD','#DC7477','#42B6C1','#E69A4A','#A8AA35','#A27B72')))
STYLES=dict(zip(METHODS,('-','--',':','-.',(0,(6,3)),(0,(8,2,2,2)))))
ATTACK_LABELS={'label_flip':'Label flipping (5 to 7)','scale2':'Scaled poisoned updates (lambda=2)',
               'scale5':'Scaled poisoned updates (lambda=5)'}
RATIOS=(0.,.2,.4,.6,.8)


def title(dataset,part,condition,seed):
    return '{} | {} | {} | {}'.format('CIFAR-10' if dataset=='cifar10' else 'MNIST',
        'IID' if part=='iid' else 'Non-IID (alpha=0.5)',ATTACK_LABELS[condition],
        '3-seed mean +/- 1 SD' if seed=='mean' else 'Seed '+str(seed))


def curve_statistics(results,metric,expected_n):
    """No unequal-horizon or incomplete-replicate mean curves."""
    if len(results)!=expected_n:
        return None
    grids=[[r['round'] for r in result['records']] for result in results]
    if any(g!=grids[0] for g in grids) or any(g!=list(range(x['config']['rounds']+1)) for g,x in zip(grids,results)):
        raise ValueError('Incomplete or unequal curve horizons')
    y=np.asarray([[r[metric] for r in x['records']] for x in results],dtype=float)*100
    return dict(rounds=grids[0],mean=y.mean(axis=0).tolist(),
                sd=y.std(axis=0,ddof=1).tolist() if len(y)>1 else None,n=len(y))


def lookup(by_key,dataset,part,condition,ratio,method,seeds):
    condition='clean' if ratio==0 else condition
    return [by_key[(dataset,part,condition,ratio,method,seed)] for seed in seeds
            if (dataset,part,condition,ratio,method,seed) in by_key]


def save(fig,folder,name,evidence):
    fig.savefig(folder/(name+'.svg'),bbox_inches='tight',metadata={'Description':json.dumps(evidence,sort_keys=True,allow_nan=False)})
    fig.savefig(folder/(name+'.png'),dpi=160,bbox_inches='tight')
    plt.close(fig)
    p.write(folder/(name+'.plot.json'),evidence)


def curves(by_key,folder,dataset,part,condition,seed,metric,seeds):
    expected=3 if seed=='mean' else 1
    selection=seeds if seed=='mean' else [seed]
    if not any(len(lookup(by_key,dataset,part,condition,r,m,selection))==expected for r in RATIOS for m in METHODS):
        return None
    fig,axes=plt.subplots(2,3,figsize=(14.6,7.5),sharey=True)
    evidence=[];any_data=False
    for ax,ratio in zip(axes.flat,RATIOS):
        for method in METHODS:
            results=lookup(by_key,dataset,part,condition,ratio,method,selection)
            stats=curve_statistics(results,metric,expected)
            evidence.append(dict(ratio=ratio,method=method,tasks=[r['task_id'] for r in results],stats=stats))
            if stats is None:continue
            any_data=True;x=np.asarray(stats['rounds']);y=np.asarray(stats['mean'])
            ax.plot(x,y,color=COLORS[method],linestyle=STYLES[method],linewidth=1.7,label=LABELS[method])
            if stats['sd'] is not None:
                sd=np.asarray(stats['sd']);ax.fill_between(x,np.maximum(0,y-sd),np.minimum(100,y+sd),color=COLORS[method],alpha=.12)
        ax.set_title('0% (background transfer)' if ratio==0 and metric=='asr' else f'{ratio:.0%} malicious clients',fontsize=11)
        ax.set_xlim(0,150 if dataset=='cifar10' else 50);ax.set_ylim(0,100)
        ax.set_xticks([0,30,60,90,120,150] if dataset=='cifar10' else [0,10,20,30,40,50])
        ax.set_xlabel('Communication round');ax.grid(alpha=.22)
        if ratio:
            ax.axvline(25 if dataset=='cifar10' else 12,color='#666666',linestyle=':',alpha=.45,linewidth=1)
    axes[0,0].set_ylabel('ASR (%)' if metric=='asr' else 'Validation accuracy (%)')
    axes[1,0].set_ylabel('ASR (%)' if metric=='asr' else 'Validation accuracy (%)')
    axes[1,2].axis('off')
    handles=[Line2D([0],[0],color=COLORS[m],linestyle=STYLES[m],lw=2,label=LABELS[m]) for m in METHODS]
    axes[1,2].legend(handles=handles,loc='upper left',frameon=False,fontsize=11)
    axes[1,2].text(.02,.2,'Training validation; every round retained.\n0% reuses the identical clean run.\nMissing/failed replicates are not imputed.\nSingle independent validation seed (n=1).',transform=axes[1,2].transAxes,fontsize=9,color='#555555',va='top')
    fig.suptitle(title(dataset,part,condition,seed),fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96))
    name=f'{dataset}_{part}_{condition}_{seed}_{metric}'
    if any_data:
        save(fig,folder,name,dict(metric=metric,seed=seed,condition=condition,series=evidence))
        return name
    plt.close(fig)
    return None


def resources(by_key,folder,dataset,part,condition,seed,seeds):
    expected=3 if seed=='mean' else 1;selection=seeds if seed=='mean' else [seed]
    if not any(len(lookup(by_key,dataset,part,condition,r,m,selection))==expected for r in RATIOS for m in METHODS):
        return None
    fig,axes=plt.subplots(1,2,figsize=(14.6,4.8));evidence=[];any_data=False
    width=.12;x=np.arange(5)
    for ax,metric,ylabel in zip(axes,('runtime_seconds','cuda_peak_allocated_mib'),
                              ('Runtime (s; shared GPU)','Peak allocated CUDA memory (MiB)')):
        for i,method in enumerate(METHODS):
            means=[];sds=[]
            for ratio in RATIOS:
                results=lookup(by_key,dataset,part,condition,ratio,method,selection)
                raw=[r['timings'][metric] if metric=='runtime_seconds' else r['memory'][metric] for r in results]
                valid=len(raw)==expected and all(v is not None and np.isfinite(v) for v in raw)
                means.append(float(np.mean(raw)) if valid else np.nan)
                sds.append(float(np.std(raw,ddof=1)) if valid and expected>1 else 0.)
                any_data |= valid
                evidence.append(dict(metric=metric,method=method,ratio=ratio,values=raw,
                    tasks=[r['task_id'] for r in results],plotted=valid))
            ax.bar(x+(i-2.5)*width,means,width,color=COLORS[method],label=LABELS[method],
                   yerr=sds if expected>1 else None,capsize=2,error_kw=dict(linewidth=.7))
        ax.set_xticks(x,[f'{r:.0%}' for r in RATIOS]);ax.set_xlabel('Malicious client ratio');ax.set_ylabel(ylabel)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_ylim(bottom=0)
    axes[0].legend(ncol=3,fontsize=8,frameon=False)
    fig.suptitle(title(dataset,part,condition,seed),fontsize=12)
    fig.text(.5,.01,'Runtime excludes data/context setup and artifact I/O; includes synchronized training, defense and evaluation. Shared load recorded.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.94))
    name=f'{dataset}_{part}_{condition}_{seed}_resources'
    if any_data:
        save(fig,folder,name,dict(seed=seed,condition=condition,bars=evidence))
        return name
    plt.close(fig);return None


def render(study,destination,decision=None):
    study=p.safe_output(study);p.check(study);destination=Path(destination).resolve()
    manifest=p.read(study/'manifest.json');policy=manifest['policy']
    if policy['phase']!='confirmation' or policy['seeds']!=[2026111703]:
        raise ValueError('Single independent validation seed required')
    if destination.exists() or study not in destination.parents:
        raise ValueError('Fresh report directory inside confirmation output required')
    summary=report(study,manifest);calculated=confirmation.compare(manifest,summary)
    if decision is not None and decision!=calculated:raise ValueError('Decision differs from raw results')
    decision=calculated;by_key={};health=[]
    for task in manifest['tasks']:
        tid=task['task_id']
        if tid not in summary['summaries']:continue
        r=p.read(study/'tasks'/tid/'result.json');c=task['config']
        by_key[(c['dataset'],c['partition'],task['condition'],c['malicious_ratio'],c['method'],c['seed'])]=r
        if not summary['summaries'][tid]['healthy']:health.append(context(r,summary['summaries'][tid]))
    destination.mkdir();folder=destination/'plots';folder.mkdir();figures=[]
    for dataset in policy['datasets']:
        for part in policy['partitions']:
            for condition in ATTACK_LABELS:
                seed=policy['seeds'][0]
                for metric in ('accuracy','asr'):
                    name=curves(by_key,folder,dataset,part,condition,seed,metric,policy['seeds'])
                    if name:figures.append(name)
                name=resources(by_key,folder,dataset,part,condition,seed,policy['seeds'])
                if name:figures.append(name)
    for name,value in (('report',summary),('confirmation',decision),('health_context',health)):
        p.write(destination/(name+'.json'),value)
    rows=[]
    for cell in decision['comparisons']:
        values=[cell['dataset'],cell['partition'],cell['condition'],f"{cell['malicious_ratio']:.0%}",
            'reference' if cell['reference'] else 'descriptive',cell['status'],
            ', '.join(k for k,v in cell.get('checks',{}).items() if not v)]
        rows.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in values)+'</tr>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>独立验证结果</title>
<style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:30px auto;padding:0 24px;background:#f5f7fa;color:#25374a}p{line-height:1.7}.card{background:white;border:1px solid #dce3ec;border-radius:8px;padding:18px;margin:20px 0}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child{text-align:left}a{color:#24658b}</style><h1>无密码学补充攻击 · 独立验证</h1>'''
    page+=f'<div class="card"><p><b>完成 {summary["completed"]}/312；允许正式测试：{decision["formal_allowed"]}。</b></p>'
    page+='<p>独立验证种子2026111703，仅使用训练集留出的验证数据。六法均使用已选定参数，不再调整；每方法每数据集一套，跨攻击、分布及恶意比例共用。此处n=1，不报告三种子标准差，也不是正式测试结论。</p>'
    page+='<p>沿用预定末10轮标准：HFCDR相对自身干净Acc下降≤3个百分点、距最佳对照Acc≤1.5个百分点、配对ASR增量≤5个百分点；FedAvg攻击增量≥15个百分点时要求相对抑制≥80%。40%/80%参考场景须重新确认FedAvg攻击有效性。20%/60%的弱攻击不作为防御优越性证据，仍保留Acc/ASR和健康要求。干净HFCDR还须相对FedAvg保持Acc/背景ASR。完整门槛及所有失败见JSON。</p>'
    page+='<p>TAD为固定且有披露的PriorFreeTAD解释版本；Krum高恶意比例超出理论保证。原健康标记按预定身份计算，暖身期的身份权重标记不能解释为攻击导致失效；诊断上下文另列，不更改原规则。任何验证不通过均不扩大搜索或回调参数，旧46项官方测试探索暴露继续披露。</p>'
    page+='<p><a href="report.json">全部逐任务指标</a> · <a href="confirmation.json">干净及48个攻击场景判定</a> · <a href="health_context.json">健康诊断上下文</a> · <a href="visualization_audit.json">绘图凭据</a></p></div>'
    page+='<div class="card"><table><tr>'+''.join('<th>'+s+'</th>' for s in ('数据','分布','攻击','恶意比例','用途','状态','未通过项'))+'</tr>'+''.join(rows)+'</table></div>'
    for name in figures:
        page+=f'<div class="card"><h2>{escape(name)}</h2><a href="plots/{name}.svg"><img loading="lazy" src="plots/{name}.png" alt="{name}"></a><p><a href="plots/{name}.svg">SVG</a> · <a href="plots/{name}.plot.json">绘图原值</a></p></div>'
    (destination/'report.html').write_text(page+'</html>',encoding='utf-8')
    evidence=list(folder.glob('*.plot.json'))
    audit=dict(fingerprint=manifest['fingerprint'],plots=len(evidence),complete_tasks=summary['completed'],
        statistical_replicates=1,official_test_used=False,all_outcomes_retained=True,
        plot_hashes={f.name:p.sha(f) for f in evidence},
        reader_sources={n:p.sha(p.ROOT/n) for n in ('render_hfcdr_confirmation.py','hfcdr_health_context.py')})
    p.write(destination/'visualization_audit.json',audit)
    return audit

