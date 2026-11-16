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

import hfcdr_formal_protocol as p
from hfcdr_study_report import report
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
    axes[0,0].set_ylabel('ASR (%)' if metric=='asr' else 'Test accuracy (%)')
    axes[1,0].set_ylabel('ASR (%)' if metric=='asr' else 'Test accuracy (%)')
    axes[1,2].axis('off')
    handles=[Line2D([0],[0],color=COLORS[m],linestyle=STYLES[m],lw=2,label=LABELS[m]) for m in METHODS]
    axes[1,2].legend(handles=handles,loc='upper left',frameon=False,fontsize=11)
    axes[1,2].text(.02,.2,'Official test; every round retained.\n0% reuses the identical clean run.\nMissing/failed replicates are not imputed.\nMean curves require all 3 complete runs.',transform=axes[1,2].transAxes,fontsize=9,color='#555555',va='top')
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


def html_dashboard(output,summary,figures,manifest):
    cards=[]
    for meta in figures:
        attrs=' '.join(f'data-{k}="{escape(str(meta[k]))}"' for k in ('dataset','partition','condition','seed'))
        imgs=''.join(f'<figure><a href="plots/{name}.svg"><img loading="lazy" src="plots/{name}.png" alt="{escape(name)}"></a><figcaption><a href="plots/{name}.svg">SVG</a> · <a href="plots/{name}.png">PNG</a> · <a href="plots/{name}.plot.json">原始绘图数据</a></figcaption></figure>' for name in meta['plots'])
        cards.append(f'<section class="card" {attrs}><h2>{escape(title(meta["dataset"],meta["partition"],meta["condition"],meta["seed"]))}</h2>{imgs}</section>')
    controls=[]
    for field,label,values in [('dataset','数据集',manifest['policy']['datasets']),('partition','分布',manifest['policy']['partitions']),
                              ('condition','攻击',['label_flip','scale2','scale5']),('seed','重复',['mean',*manifest['policy']['seeds']])]:
        choices=''.join(f'<option value="{v}">{v}</option>' for v in values)
        controls.append(f'<label>{label} <select id="{field}"><option value="all">全部</option>{choices}</select></label>')
    comparisons=Counter(c['status'] for c in summary['comparisons'])
    failed=''.join(f'<tr><td>{escape(f["task_id"])}</td><td>{escape(str(f["status"]))}</td></tr>' for f in summary['failed_or_incomplete'])
    table=[]
    def fmt(stat,percent=False):
        if stat['mean'] is None:return '—'
        f=100 if percent else 1
        return f'{stat["mean"]*f:.3f}' + (f' ± {stat["sd"]*f:.3f}' if stat['sd'] is not None else '')
    for row in summary['aggregate']:
        values=[row[k] for k in ('dataset','partition','attack','scale','malicious_ratio','method')]
        values += [f'{row["n"]}/3',fmt(row['tail10_accuracy'],True),fmt(row['tail10_asr'],True),fmt(row['runtime_seconds']),f'{row["healthy_n"]}/{row["n"]}']
        table.append('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in values)+'</tr>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>HFCDR-FL 补充攻击实验</title>
<style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:30px auto;padding:0 24px;color:#243447;background:#f5f7fa}h1{font-size:30px}p{line-height:1.65}.box,.card{background:white;border:1px solid #dce2e9;border-radius:10px;padding:22px;margin:18px 0}.controls{position:sticky;top:0;background:#eef3f8;padding:16px;z-index:2;display:flex;gap:24px;flex-wrap:wrap}select{padding:6px}figure{margin:10px 0 28px}img{width:100%;height:auto}figcaption{font-size:13px}a{color:#24658b}table{border-collapse:collapse;font-size:12px;width:100%}td,th{border-bottom:1px solid #e1e5eb;padding:8px;text-align:right}td:first-child{text-align:left}thead{background:#eaf0f5}details{margin:20px 0}h2{font-size:18px}.overflow{overflow:auto}.muted{color:#64748b}</style>
<h1>HFCDR-FL · 无密码学补充攻击实验</h1>'''
    page+=f'<div class="box"><p><b>已完整记录 {summary["completed"]} / 936 项。</b> 未完成/失败 {len(summary["failed_or_incomplete"])} 项，尚无结果 {len(summary["missing"])} 项。完整性状态：{summary["complete"]}。</p>'
    page+='<p>六种方法，CIFAR-10 / MNIST，IID / Dirichlet α=0.5，0/20/40/60/80% 恶意客户端，3 个固定种子。缩放是在标签翻转训练所得增量上乘 λ=2 或 5。0% 是同一无攻击基线的重复引用，ASR 为背景转移率。</p>'
    page+='<p>准确率使用官方测试集全部 10,000 张；ASR 使用全部真实类别 5（CIFAR-10 1,000 张，MNIST 892 张）被预测为 7 的比例。主指标为最后 10 轮平均；图中保留全部逐轮数值。阴影和误差条为三个独立种子的样本标准差。</p>'
    page+='<p>参数已通过训练验证集筛选及独立验证种子确认，每方法每数据集仅用一套，跨攻击、比例、分布共用。正式测试不回流选参。历史46项官方测试探索结果未参与本轮选择或均值；新种子不消除该历史暴露。</p>'
    page+='<p>运行时间为共享 GPU 实测同步墙钟时间，包含本地训练、防御聚合和评估，排除数据/上下文准备与文件写入；原始日志保留并发负载。新增实验未执行密码协议。VERT固定原文history10/epochs5/lr0.01与作者投影维数，TAD为用户确认的固定PriorFreeTAD解释版本；原文歧义仍披露，不能声称是唯一作者复现，也不能与历史MNIST旧配置混合。</p>'
    page+=f'<p>预设门槛检查：{escape(str(dict(comparisons)))}。所有不利结果、健康异常和失败均保留。Krum 的 60/80% 属于理论保证之外的压力测试。</p>'
    page+='<p>原健康标记按预先指定的客户端身份计算；攻击前的身份权重标记不等于攻击造成失效。<a href="health_context.json">另表列出暖身期、跨边界及攻击期的诊断上下文</a>，不更改原健康判定或选择规则。</p>'
    page+='<p><a href="report.json">完整报告 JSON</a> · <a href="summary.json">逐任务汇总</a> · <a href="aggregate.json">三种子统计</a> · <a href="comparisons.json">逐场景门槛检查</a> · <a href="visualization_audit.json">绘图核验</a></p></div>'
    page+='<div class="controls">'+''.join(controls)+'</div>'+''.join(cards)
    page+='<details><summary>完整汇总表（最后10轮均值 ± 种子间SD）</summary><div class="overflow"><table><thead><tr>'+''.join(f'<th>{v}</th>' for v in ['数据','分布','攻击','λ','恶意比例','方法','n','Acc %','ASR %','运行秒数','健康数'])+'</tr></thead><tbody>'+''.join(table)+'</tbody></table></div></details>'
    page+='<details open><summary>未完成或失败任务</summary><div class="overflow"><table>'+failed+'</table></div></details>'
    default_seed='mean' if any(x['seed']=='mean' for x in figures) else str(manifest['policy']['seeds'][0])
    page+='''<script>const fields=['dataset','partition','condition','seed'];function filter(){document.querySelectorAll('.card').forEach(card=>{card.hidden=!fields.every(k=>document.getElementById(k).value==='all'||card.dataset[k]===document.getElementById(k).value)})}fields.forEach(k=>document.getElementById(k).addEventListener('change',filter));document.getElementById('seed').value=DEFAULT_SEED;filter();</script></html>'''.replace('DEFAULT_SEED',json.dumps(default_seed))
    (output/'report.html').write_text(page,encoding='utf-8')


def render(study,destination):
    study=p.safe_output(study);destination=Path(destination).resolve()
    p.check(study)
    if destination.exists() or destination==study or study not in destination.parents:
        raise ValueError('Use a fresh report directory inside the new study output')
    manifest=p.read(study/'manifest.json');summary=report(study,manifest)
    destination.mkdir();folder=destination/'plots';folder.mkdir()
    by_key={}
    for t in manifest['tasks']:
        if t['task_id'] not in summary['summaries']:continue
        r=p.read(study/'tasks'/t['task_id']/'result.json');c=t['config']
        by_key[(c['dataset'],c['partition'],t['condition'],c['malicious_ratio'],c['method'],c['seed'])]=r
    p.write(destination/'report.json',summary)
    p.write(destination/'summary.json',list(summary['summaries'].values()))
    p.write(destination/'aggregate.json',summary['aggregate']);p.write(destination/'comparisons.json',summary['comparisons'])
    p.write(destination/'health_context.json',[context(p.read(study/'tasks'/tid/'result.json'),s)
        for tid,s in summary['summaries'].items() if not s['healthy']])
    figures=[]
    for dataset in manifest['policy']['datasets']:
        for part in manifest['policy']['partitions']:
            for condition in ATTACK_LABELS:
                for seed in [*manifest['policy']['seeds'],'mean']:
                    names=[curves(by_key,folder,dataset,part,condition,seed,metric,manifest['policy']['seeds']) for metric in ('accuracy','asr')]
                    names.append(resources(by_key,folder,dataset,part,condition,seed,manifest['policy']['seeds']))
                    names=[n for n in names if n]
                    if names:figures.append(dict(dataset=dataset,partition=part,condition=condition,seed=seed,plots=names))
    evidence=list(folder.glob('*.plot.json'))
    audit=dict(fingerprint=manifest['fingerprint'],plots=len(evidence),panels_per_curve=5,
        complete_tasks=summary['completed'],all_outcomes_retained=True,mean_requires_n=3,
        raw_plot_data={f.name:p.sha(f) for f in evidence},
        warning='Figure numeric data are embedded in SVG Description and sidecar JSON; shared clean references are not independent replicates.')
    p.write(destination/'visualization_audit.json',audit)
    html_dashboard(destination,summary,figures,manifest)
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=p.OUTPUT)
    parser.add_argument('--destination',type=Path,required=True)
    a=parser.parse_args();print(json.dumps(render(a.study,a.destination),indent=2))


if __name__=='__main__':main()
