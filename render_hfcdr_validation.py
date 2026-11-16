#!/usr/bin/env python3
"""Read-only FedAvg validation figures; all seeds, failures and gates retained."""
from __future__ import annotations
import argparse
from html import escape
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/hfcdr-validation-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hfcdr_validation_protocol as p
from hfcdr_validation_analysis import report
from render_hfcdr_attack_study import curve_statistics

CONDITIONS=('label_flip','scale2','scale5')
LABELS={'label_flip':'Label flip (5 to 7)','scale2':'Scaled label flip (lambda=2)',
        'scale5':'Scaled label flip (lambda=5)','clean':'Clean (shared)'}
COLORS={'label_flip':'#4D91BD','scale2':'#DC7477','scale5':'#42A6A1','clean':'#616B75'}


def series(by_key,dataset,partition,condition,ratio,seeds):
    condition='clean' if ratio==0 else condition
    return [by_key[(dataset,partition,condition,ratio,s)] for s in seeds
            if (dataset,partition,condition,ratio,s) in by_key]


def draw(by_key,folder,dataset,partition,seeds,replicate,metric):
    evidence=[];fig,axes=plt.subplots(2,3,figsize=(14.6,7.5))
    any_data=False
    for ax,ratio in zip(axes.flat,(0.,.2,.4,.6,.8)):
        for j,condition in enumerate(('clean',) if ratio==0 else CONDITIONS):
            results=series(by_key,dataset,partition,condition,ratio,seeds)
            if metric in ('accuracy','asr'):
                stats=curve_statistics(results,metric,len(seeds))
                if stats:
                    any_data=True;x=stats['rounds'];y=np.asarray(stats['mean'])
                    ax.plot(x,y,color=COLORS[condition],label=LABELS[condition],lw=1.8)
                    if stats['sd'] is not None:
                        sd=np.asarray(stats['sd']);ax.fill_between(x,np.clip(y-sd,0,100),np.clip(y+sd,0,100),color=COLORS[condition],alpha=.14)
                evidence.append(dict(ratio=ratio,condition=condition,tasks=[r['task_id'] for r in results],stats=stats))
            else:
                values=[r['timings']['runtime_seconds'] for r in results]
                valid=len(values)==len(seeds)
                if valid:
                    any_data=True
                    ax.bar(j,np.mean(values),color=COLORS[condition],width=.6,
                        yerr=np.std(values,ddof=1) if len(values)>1 else None,capsize=3)
                evidence.append(dict(ratio=ratio,condition=condition,tasks=[r['task_id'] for r in results],values=values,plotted=valid))
        ax.set_title('0% (shared clean)' if ratio==0 else f'{ratio:.0%} malicious clients')
        ax.grid(alpha=.2);ax.set_axisbelow(True)
        if metric in ('accuracy','asr'):
            ax.set_xlim(0,150 if dataset=='cifar10' else 50);ax.set_ylim(0,100)
            ax.set_xlabel('Communication round')
            if ratio:ax.axvline(25 if dataset=='cifar10' else 12,color='#777',ls=':',lw=1)
        else:
            ax.set_ylim(bottom=0);ax.set_xticks([0] if ratio==0 else [0,1,2],['Clean'] if ratio==0 else ['LF','Scale2','Scale5'])
    ylabel={'accuracy':'Validation accuracy (%)','asr':'Validation source 5 to 7 ASR (%)','runtime':'Runtime (s; shared GPU)'}[metric]
    axes[0,0].set_ylabel(ylabel);axes[1,0].set_ylabel(ylabel)
    axes[1,2].axis('off')
    from matplotlib.lines import Line2D
    axes[1,2].legend(handles=[Line2D([0],[0],color=COLORS[c],lw=2,label=LABELS[c]) for c in ('clean',*CONDITIONS)],loc='upper left',frameon=False)
    axes[1,2].text(.02,.38,'FedAvg screening; training holdout only.\nAll rounds retained; no smoothing.\n0% ASR is background transfer.\nMean requires all 2 independent seeds.\nShading/error bars: sample SD (n-1).\nMissing runs are not imputed.',transform=axes[1,2].transAxes,va='top',fontsize=9)
    title=f"{dataset.upper()} | {'IID' if partition=='iid' else 'Non-IID (alpha=0.5)'} | {replicate} | {ylabel}"
    fig.suptitle(title,fontsize=13);fig.tight_layout(rect=(0,0,1,.95))
    name=f'{dataset}_{partition}_{replicate}_{metric}'
    if not any_data:plt.close(fig);return None
    payload=dict(dataset=dataset,partition=partition,replicate=replicate,seeds=seeds,metric=metric,series=evidence)
    fig.savefig(folder/(name+'.svg'),bbox_inches='tight',metadata={'Description':json.dumps(payload,sort_keys=True,allow_nan=False)})
    fig.savefig(folder/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
    p.write(folder/(name+'.plot.json'),payload)
    return name


def render(study,destination):
    study=p.safe_output(study);destination=Path(destination).resolve();p.check(study)
    if destination.exists() or study not in destination.parents:
        raise ValueError('Fresh report directory inside validation output required')
    manifest=p.read(study/'manifest.json')
    if manifest['policy']['phase']!='attack_screen':raise ValueError('This renderer is for FedAvg attack screening')
    summary=report(study,manifest);by_key={}
    for t in manifest['tasks']:
        if t['task_id'] not in summary['summaries']:continue
        r=p.read(study/'tasks'/t['task_id']/'result.json');c=t['config']
        by_key[(c['dataset'],c['partition'],t['condition'],c['malicious_ratio'],c['seed'])]=r
    destination.mkdir();folder=destination/'plots';folder.mkdir();figures=[]
    for dataset in manifest['policy']['datasets']:
        for partition in manifest['policy']['partitions']:
            options=[('mean',manifest['policy']['seeds'])]+[(str(s),[s]) for s in manifest['policy']['seeds']]
            for replicate,seeds in options:
                for metric in ('accuracy','asr','runtime'):
                    name=draw(by_key,folder,dataset,partition,seeds,replicate,metric)
                    if name:figures.append(dict(name=name,replicate=replicate))
    p.write(destination/'report.json',summary)
    rows=[]
    def fmt(value):return '—' if value is None else f'{100*value:.2f}'
    for c in summary['cells']:
        values=[c['dataset'],c['partition'],c['condition'],f"{100*c['malicious_ratio']:.0f}%",
            '主门槛' if c['reference'] else '描述性',f"{c['n']}/{c['expected_n']}",
            fmt(c['mean_excess_asr']),fmt(c['sd_excess_asr']),c['status']]
        rows.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in values)+'</tr>')
    links=[]
    for f in figures:
        name=f['name'];tag='div' if f['replicate']=='mean' else 'details'
        caption='<summary>'+escape(name)+'</summary>' if tag=='details' else '<h2>'+escape(name)+'</h2>'
        links.append(f'<{tag} class="card">{caption}<a href="plots/{name}.svg"><img loading="lazy" src="plots/{name}.png" alt="{name}"></a><p><a href="plots/{name}.svg">SVG</a> · <a href="plots/{name}.png">PNG</a> · <a href="plots/{name}.plot.json">绘图原值</a></p></{tag}>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>FedAvg 验证集攻击筛选</title>
<style>body{font-family:system-ui,sans-serif;margin:30px auto;padding:0 24px;max-width:1450px;background:#f5f7fa;color:#25374a}p{line-height:1.7}.card{background:white;border:1px solid #dce3ec;border-radius:8px;padding:18px;margin:20px 0}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child{text-align:left}a{color:#24658b}h2{font-size:18px}</style>
<h1>FedAvg · 无密码学补充攻击验证</h1>'''
    page+=f'<div class="card"><p><b>完整结果 {summary["completed"]}/{summary["expected_tasks"]} 项；全部攻击门槛通过：{summary["all_attack_settings_pass"]}；允许进入下一阶段：{summary["next_stage_allowed"]}。</b></p>'
    page+='<p>两种开发种子；仅使用训练集留出的验证数据。CIFAR-10：7500张、源类750张；MNIST：2500张、源类225张。标签5→7；缩放为标签翻转所得完整更新乘λ=2或5。0%共用同一干净训练，ASR为背景转移率。</p>'
    page+='<p>预先固定40%和80%为主门槛：每种数据/分布中，末10轮的配对ASR增量，双种子均值≥15个百分点且每种子≥10个百分点。20%/60%与失败结果全部保留。此页只判断攻击是否有效，不代表任何防御性能结论。</p>'
    page+='<p>旧批次已有46项官方测试探索结果；不将其用于新筛选，也不混入新正式均值。运行时间包含同步训练、防御/聚合及验证，排除数据准备与文件写入；服务器共享负载已留档。</p>'
    page+='<p><a href="report.json">完整逐任务指标及门槛JSON</a> · <a href="visualization_audit.json">绘图凭据</a></p></div>'
    page+='<div class="card"><table><tr>'+''.join('<th>'+v+'</th>' for v in ['数据','分布','攻击','恶意比例','用途','种子数','ΔASR pp','SD pp','状态'])+'</tr>'+''.join(rows)+'</table></div>'
    page+=''.join(links)+'</html>'
    (destination/'report.html').write_text(page,encoding='utf-8')
    audit=dict(fingerprint=manifest['fingerprint'],plots=len(figures),complete_tasks=summary['completed'],
        expected_seed_count=2,official_test_used=False,all_outcomes_retained=True,
        plot_hashes={f.name:p.sha(f) for f in folder.glob('*.plot.json')})
    p.write(destination/'visualization_audit.json',audit)
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=p.OUTPUT)
    parser.add_argument('--destination',type=Path,required=True)
    a=parser.parse_args();print(json.dumps(render(a.study,a.destination),indent=2))
