"""Promote corrected timing into CIFAR-BEST, archiving the previous main report.

Default is plan-only. --apply replaces only timing fields, figures, and provenance.
Accepted performance source bundles, trajectories, accuracy and ASR remain intact.
"""
from __future__ import annotations
import argparse,copy,csv,hashlib,io,json,os,re,shutil
from pathlib import Path
import cifar_best_timing_integration as core

ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rawjson(x):return (json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def performance(x):
    if isinstance(x,dict):return {k:performance(v) for k,v in x.items() if k!='timing' and not k.startswith(('timing_','current_timing','runtime_')) and not k.endswith('_seconds')}
    if isinstance(x,list):return [performance(v) for v in x]
    return x

def make_payload(source):
    from render_cifar_timing import load
    report,_=load(source,core.REPORT_SHA)
    payload=dict(schema='cifar-best-main-timing-v1',report_sha256=core.REPORT_SHA,source_fingerprint=report['source_fingerprint'],
        cells={},aggregate={},health_failures=report['health_failures'],definition=report['timing'],history=report['history'])
    for row in report['rows']:
        folder=source/'tasks'/row['task_id'];task=read(folder/'task.json');s=read(folder/'result.json');clock=read(folder/'timing.json')
        payload['cells'][core.canonical(row['task_id'])]=dict(source_task=row['task_id'],config=task['config'],summary={k:v for k,v in s.items() if k.endswith('_seconds')},timing=clock,health_reasons=row['health_reasons'],artifacts=read(folder/'status.json')['artifacts'])
    arms={'sm9rrs':'H84','vert':'VERT','alignins':'AlignIns','ding13':'TAD','krum':'Krum','fedavg':'FedAvg'}
    for row in report['aggregate']:
        q=row['malicious_ratio'];tag='clean' if q==0 else f'r{int(round(q*100))}'
        payload['aggregate'][f'{arms[row["method"]]}_{row["partition"]}_{tag}']=row
    return payload

def csv_update(path,rows,keyfields,changed):
    old=list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))));fields=list(old[0]);updates={tuple(str(r[k]) for k in keyfields):r for r in rows}
    def key(row):
        return tuple(str(float(row[k])) if k=='malicious_ratio' else str(row[k]) for k in keyfields)
    updates={key(r):r for r in rows}
    for r in old:
        new=updates[key(r)]
        for k in changed(new):
            if k not in fields:fields.append(k)
            r[k]=new[k]
    out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=fields);w.writeheader();w.writerows(old)
    return out.getvalue().encode('utf-8-sig')

def plan(package,source,archive):
    if (package/'provenance/main_timing_integration.json').exists():raise ValueError('Main timing already integrated; use render_cifar_best.py to redraw')
    if archive.exists():raise ValueError('Archive must be new')
    from render_cifar_best import TIMING_NOTE, checked_timing, load_inputs, SOURCE_SHA
    load_inputs(package,SOURCE_SHA)
    p=package/'final_evaluation';old=read(p/'combined_results.json');summary=read(p/'summary.json');agg=read(p/'aggregate.json')
    payload=make_payload(source);new=core.overlay(old,payload)
    for x in payload['cells'].values():checked_timing(x['timing'])
    merged=core.merge_summary(summary,payload);mergedagg=core.merge_aggregate(agg,payload)
    assert performance(old)==performance(new)
    assert [{k:v for k,v in x.items() if not k.startswith(('runtime_','crypto_wall','timing_'))} for x in agg]==[{k:v for k,v in x.items() if not k.startswith(('runtime_','crypto_wall','timing_'))} for x in mergedagg]
    files={}
    def add(name,data):files[name]=data.encode() if isinstance(data,str) else data
    add('final_evaluation/summary.json',rawjson(merged));add('final_evaluation/aggregate.json',rawjson(mergedagg));add('final_evaluation/combined_results.json',rawjson(new))
    add('final_evaluation/summary.csv',csv_update(p/'summary.csv',merged,['task_id'],lambda r:[k for k in r if k.endswith('_seconds') or k in core.TAG_FIELDS]))
    add('final_evaluation/aggregate.csv',csv_update(p/'aggregate.csv',mergedagg,['method','partition','malicious_ratio'],lambda r:[k for k in r if k.startswith(('runtime_','crypto_wall','timing_'))]))
    # Copy the already checked timing figures exactly; preserve accuracy/ASR images.
    for path in sorted((package/'timing_mnist_boundary').glob('seed_*/plots/*')):
        if path.suffix in ('.png','.svg'):add('final_evaluation/'+path.relative_to(package/'timing_mnist_boundary').as_posix(),path.read_bytes())
    for path in sorted(p.glob('**/*.html')):
        text=path.read_text();text=text.replace(TIMING_NOTE,core.NOTE)
        if 'asr_with_clean' not in path.parts:
            text=text.replace('端到端时间开销（含密码协议）','总时间开销（含密码协议，同MNIST计时）')
            if path.name in ('visualizations.html','runtime_visualizations.html','report.html'):
                prefix='../' if path.parent!=p else ''
                intro='<p class="timing-source">Acc/ASR保留原正式测试；耗时已更新为MNIST同口径180项复测。<a href="'+prefix+'report.html#timing-rerun">计时来源和完整原值</a></p>'
                text=re.sub(r'(<h1>.*?</h1>)',lambda m:m[0]+intro,text,count=1)
            if path==p/'report.html':
                text=text.replace('按用户2026-11-01明确指定，新做Ours R4共30项','准确率与ASR按用户2026-11-01明确指定，新做Ours R4共30项',1)
                text=text.replace('<h2>seed ',core.section(payload)+'<h2>seed ',1)
                text=text.replace('共享GPU实测总用时及减去crypto区间的用时不等同独占速度测试；减法不是关闭密码学重新运行。','当前耗时来自独立180项复测，数据加载在外并扣除checkpoint I/O；共享负载仍不等同独占速度测试，减法不是关闭密码学重新运行。')
            if text!=path.read_text():add(path.relative_to(package).as_posix(),text)
    oldreadme=(package/'README.txt').read_text();oldreadme=oldreadme.replace('四攻击比例官方 ASR','官方 ASR（主图保留原样，含0%补充图见ASR-with-clean.html）')
    oldreadme=oldreadme.replace('仍包含数据加载、评估、观测、断点 I/O，非关闭密码学重跑或独占速度测试。','当前正式耗时与MNIST计时边界一致：数据加载在外，扣除checkpoint I/O，包含标准评估；非关闭密码学重跑或独占速度测试。')
    add('README.txt',oldreadme+'\n2026-11-04 主报告耗时已整合：summary/aggregate/combined_results及所有主报告耗时图均采用新180项复测，Acc/ASR及轨迹不改。旧主报告、旧耗时表和图归档于 '+os.path.relpath(archive,package)+'/。\nexport、analysis与raw保持原始性能取证内容（含当时的历史计时），不作为当前计时来源；当前计时来源为provenance/current_timing.json，主报告清单为provenance/current_file_manifest.json，原file_manifest.json仅作历史清单。\n')
    add('TIMING-MNIST.html', '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=final_evaluation/report.html#timing-rerun"><a href="final_evaluation/report.html#timing-rerun">CIFAR 主报告：MNIST同口径耗时</a>\n')
    payloadbytes=rawjson(payload);add('provenance/current_timing.json',payloadbytes)
    metadata=dict(schema='cifar-best-timing-integration-v1',timing_report_sha256=core.REPORT_SHA,timing_payload_sha256=hashlib.sha256(payloadbytes).hexdigest(),
        performance_source_bundle_sha256=sha(package/'export/combined_bundle.json'),performance_summary_projection_sha256=core.projection_sha(summary),
        original_summary_sha256=sha(p/'summary.json'),archive_relative_path=os.path.relpath(archive,package),performance_metrics_unchanged=True,
        timing_tasks=180,timing_figures=16,accuracy_asr_figures_modified=False,source_separation='official performance preserved; corrected timing only')
    add('provenance/main_timing_integration.json',rawjson(metadata))
    return files,metadata

def apply(package,source,archive):
    files,metadata=plan(package,source,archive)
    before={p.relative_to(package).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(package.rglob('*')) if p.is_file()}
    archive.mkdir(parents=True)
    shutil.copytree(package/'final_evaluation',archive/'final_evaluation')
    shutil.copy2(package/'README.txt',archive/'README.txt');shutil.copy2(package/'TIMING-MNIST.html',archive/'TIMING-MNIST.html');(archive/'provenance').mkdir()
    for name in ('file_manifest.json','current_file_manifest.json'):
        if (package/'provenance'/name).exists():shutil.copy2(package/'provenance'/name,archive/'provenance'/name)
    archived={p.relative_to(archive).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(archive.rglob('*')) if p.is_file()}
    for name,v in archived.items():assert v==before[name]
    (archive/'archive_manifest.json').write_bytes(rawjson(dict(kind='previous_main_report_and_worker_timing',files=archived)))
    for name,data in files.items():
        path=package/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    # Keep the old render receipt in the archive; write a receipt for the current main report.
    for name in ('render_manifest.json','visual_and_data_verification.json'):
        path=package/'final_evaluation'/name
        data=dict(**metadata,
            verification=dict(summary_non_timing_values_unchanged=True,aggregate_non_timing_values_unchanged=True,
                report_performance_values_unchanged=True,rounds_and_performance_plot_bytes_unchanged=True),
            original_receipt_archive=os.path.relpath(archive/'final_evaluation'/name,path.parent))
        data['schema']='cifar-best-integrated-timing-receipt-v1';path.write_bytes(rawjson(data));files['final_evaluation/'+name]=path.read_bytes()
    unchanged=[name for name in before if name not in files]
    assert all(sha(package/name)==before[name]['sha256'] for name in unchanged)
    after={p.relative_to(package).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(package.rglob('*')) if p.is_file() and p!=package/'provenance/current_file_manifest.json'}
    (package/'provenance/current_file_manifest.json').write_bytes(rawjson(dict(schema='cifar-best-current-package-v1',timing_report_sha256=core.REPORT_SHA,files=after)))
    verification=dict(archive=str(archive),archived_files=len(archived),changed_or_new_files=sorted(files),unchanged_original_files=len(unchanged),all_unchanged_sha_verified=True,**metadata)
    (archive/'integration_verification.json').write_bytes(rawjson(verification))
    return verification

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,default=ROOT/'outputs/cifar-best');p.add_argument('--source',type=Path,default=ROOT/'outputs/old/cifar_timing_corrected_complete_20261104/raw');p.add_argument('--archive',type=Path,default=ROOT/'outputs/old/cifar_best_worker_timing_20261104');p.add_argument('--apply',action='store_true');a=p.parse_args()
    if a.apply:r=apply(a.package,a.source,a.archive)
    else:
        files,meta=plan(a.package,a.source,a.archive);r=dict(plan_only=True,files=sorted(files),**meta)
    print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
