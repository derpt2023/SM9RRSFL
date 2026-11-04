"""Pure transformations for combining accepted performance with measured timing."""
import copy
import hashlib
import json

REPORT_SHA = '074ca251f670509e60f2625a1b73fd4a7084b82d98713295ebf80fad09468077'
NOTE = ('耗时采用2026-11-03完整180项复测，与MNIST使用同一run_measured_experiment：'
        '数据加载在计时外，扣除checkpoint I/O；扣除密码学仅减四段互不重叠的协议墙钟时间。'
        'Acc/ASR来自原正式测试，耗时来自独立复测；共享负载与模型、轮数等工作量差异仍需区分。')
TAG_FIELDS = {'timing_source_task', 'timing_run', 'timing_definition', 'timing_health_reasons', 'timing_health_pass'}


def canonical(task):
    return 'H84_' + task[3:] if task.startswith('R4_') else task


def projection(rows):
    return [{k:v for k,v in row.items() if not k.endswith('_seconds') and k not in TAG_FIELDS} for row in rows]


def projection_sha(rows):
    raw=json.dumps(projection(rows),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def validate_payload(payload, cells):
    if payload['report_sha256'] != REPORT_SHA or set(payload['cells']) != set(cells) or len(cells)!=180:
        raise ValueError('Timing source or matrix mismatch')
    for name,item in payload['cells'].items():
        old=cells[name]['config'];new=item['config']
        for key in ('method','seed','partition','malicious_ratio','rounds'):
            if old[key]!=new[key]:raise ValueError('Timing/performance scenario mismatch: '+name+'/'+key)
        if canonical(item['source_task'])!=name:raise ValueError('Wrong source task')
        if old['method']=='sm9rrs' and (new['suspicion_recovery_factor']!=4 or new['detector_clip_factor']!=1):
            raise ValueError('Expected fixed R4')


def overlay(report,payload):
    validate_payload(payload,report['cells']);out=copy.deepcopy(report)
    for name,item in payload['cells'].items():
        out['cells'][name]['timing']=copy.deepcopy(item['timing'])
        out['cells'][name]['timing_source_task']=item['source_task']
        if name in out.get('audited_cells',{}):
            out['audited_cells'][name]['summary']['timing']=copy.deepcopy(item['timing'])
            out['audited_cells'][name]['summary']['timing_source_task']=item['source_task']
    for name,group in out.get('groups',{}).items():
        for metric,sourcekey in [('runtime_with_crypto_seconds','runtime_seconds'),('runtime_without_crypto_seconds','runtime_without_crypto_seconds'),('crypto_wall_seconds','crypto_wall_seconds')]:
            items=[payload['cells'][f'{name}_s{s}'] for s in (2026102801,2026102802,2026102803)]
            values=[i['summary'][sourcekey] for i in items]
            a=payload['aggregate'][name][sourcekey]
            group['samples'][metric]=dict(values=values,mean=a['mean'],sample_sd=a['sample_sd'])
    out['current_timing_note']=NOTE;out['timing_note']=NOTE
    out['timing_provenance']=dict(report_sha256=REPORT_SHA,definition='mnist_common_runner_v1',fresh_tasks=180,
        performance_source='unchanged accepted official test',timing_source='independent corrected rerun',health_failures=payload['health_failures'])
    return out


def merge_summary(rows,payload):
    out=copy.deepcopy(rows)
    for row in out:
        item=payload['cells'][row['task_id']];s=item['summary']
        for key in list(row):
            if key.endswith('_seconds'):
                row[key]=s['checkpoint_io_seconds'] if key=='measured_checkpoint_write_seconds' else s[key]
        row.update(timing_source_task=item['source_task'],timing_run='cifar_mnist_timing_corrected150_v1',
            timing_definition='mnist_common_runner_v1',timing_health_reasons=';'.join(item['health_reasons']),
            timing_health_pass=not item['health_reasons'])
    if projection(out)!=projection(rows):raise ValueError('Performance summary changed')
    return out


def merge_aggregate(rows,payload):
    out=copy.deepcopy(rows)
    for row in out:
        arm={'sm9rrs':'H84','vert':'VERT','alignins':'AlignIns','ding13':'TAD','krum':'Krum','fedavg':'FedAvg'}[row['method']]
        ratio=row['malicious_ratio'];tag='clean' if ratio==0 else f'r{int(round(ratio*100))}'
        a=payload['aggregate'][f'{arm}_{row["partition"]}_{tag}']
        for target,source in [('runtime_end_to_end','runtime_seconds'),('runtime_without_crypto','runtime_without_crypto_seconds'),('crypto_wall','crypto_wall_seconds')]:
            row[target+'_mean']=a[source]['mean'];row[target+'_sample_sd']=a[source]['sample_sd']
        row['timing_n']=a['n'];row['timing_definition']='mnist_common_runner_v1'
    return out


def section(payload):
    from html import escape
    from render_cifar_timing import table
    rows=[]
    for name,item in sorted(payload['cells'].items()):
        s=item['summary'];c=item['config']
        rows.append(dict(task_id=name,timing_source_task=item['source_task'],seed=c['seed'],partition=c['partition'],malicious_ratio=c['malicious_ratio'],method=c['method'],runtime_seconds=s['runtime_seconds'],runtime_without_crypto_seconds=s['runtime_without_crypto_seconds'],crypto_wall_seconds=s['crypto_wall_seconds'],health_reasons=item['health_reasons']))
    aggregates=[]
    for key,a in sorted(payload['aggregate'].items()):
        row=dict(scenario=key,n=a['n'])
        for metric in ('runtime_seconds','runtime_without_crypto_seconds','crypto_wall_seconds'):
            for stat in ('mean','sample_sd'):row[metric+'_'+stat]=a[metric][stat]
        aggregates.append(row)
    return ('<section id="timing-rerun"><h2>正式耗时：MNIST同口径180项复测</h2><p>'+escape(NOTE)+
            '</p><p>180项均150轮，0执行失败/重试；3项TAD Non-IID clean原有健康警告保留。此前17项错误TAD及OOM不纳入本次耗时。'
            '原始官方Acc/ASR、健康资格和排名均保持不变。</p><p><a href="runtime_visualizations.html">全部新耗时图</a> · '
            '<a href="../timing_mnist_boundary/report.html">复测审计与完整来源</a></p>'+
            ''.join('<figure><img src="seed_mean/plots/'+slug+'_'+kind+'.svg"><figcaption>三seed均值 · '+label+'</figcaption></figure>' for slug in ('iid','dirichlet_alpha_0_5') for kind,label in [('runtime_overhead','总耗时'),('runtime_without_crypto','扣除密码学耗时')])+
            '<details><summary>60组耗时均值、样本标准差与n</summary>'+table(aggregates,list(aggregates[0]))+'</details>'+
            '<details><summary>全部180项耗时原值（秒）</summary>'+table(rows,list(rows[0]))+'</details></section>')
