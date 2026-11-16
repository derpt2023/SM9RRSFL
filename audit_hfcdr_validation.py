#!/usr/bin/env python3
"""Read-only live/final integrity audit. Does not launch, resume or retry training."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gzip
import json
import math
from pathlib import Path

import numpy as np
import hfcdr_validation_protocol as p
from hfcdr_pilot_data import array_digest
from hfcdr_validation_analysis import summarize

IMPLEMENTATIONS={'hfcdr':'hfcdr_pilot_runtime.DirectHFCDR','vert':'cifar_vert_paper_core.PaperVERT',
    'alignins':'sm9rrsfl.alignins.AlignInsDefense','tad':'cifar_tad_no_prior.PriorFreeTAD',
    'krum':'sm9rrsfl.aggregation.krum','fedavg':'sm9rrsfl.aggregation.fedavg'}


def inspect_task(folder,task,manifest,*,full=False):
    status=p.read(folder/'status.json');result=p.read(folder/'result.json');c=task['config']
    if status['status']!='complete' or result['task_id']!=task['task_id'] or result['config']!=c:
        raise ValueError('Task/result identity differs')
    if status['manifest_fingerprint']!=manifest['fingerprint'] or status['result_sha256']!=p.sha(folder/'result.json'):
        raise ValueError('Frozen result fingerprint differs')
    if p.read(folder/'task.json')!=task:
        raise ValueError('Actual task configuration differs')
    if result['binding']!=p.read(folder/'binding.json') or result['binding']['implementation']!=IMPLEMENTATIONS[c['method']]:
        raise ValueError('Wrong actual defense binding')
    if result['binding']['cryptography']!='absent':
        raise ValueError('Unexpected cryptography')
    if c['method']=='krum' and result['binding']['krum_count_bound']!=round(c['malicious_ratio']*100):
        raise ValueError('Krum count bound differs')
    data=p.read(folder/'dataset_identity.json')
    if (result.get('official_test_used') is not False or result.get('evaluation_scope')!='training_validation_all_source5'
            or data.get('official_test_files_opened') is not False):
        raise ValueError('Official test data cannot enter validation evidence')
    if data!=p.read(folder.parents[1]/'dataset_identity.json')[c['dataset']]:
        raise ValueError('Dataset identity differs')
    pairing=p.read(folder/'pairing.json')
    if pairing!=p.read(folder.parents[1]/'pairing.json')[p.pairing_key(task)]:
        raise ValueError('Actual client/attacker assignment differs')
    if p.read(folder/'summary.json')!=summarize(result):
        raise ValueError('Summary differs from raw results')
    rows=[json.loads(line) for line in (folder/'rounds.jsonl').read_text().splitlines()]
    if rows!=result['records'] or [r['round'] for r in rows]!=list(range(c['rounds']+1)):
        raise ValueError('Incomplete or inconsistent per-round trace')
    for row in rows:
        cm=np.asarray(row['confusion'],dtype=np.int64)
        if cm.shape!=(10,10) or cm.sum()!=data['accuracy_samples'] or cm[5].sum()!=data['source_samples']:
            raise ValueError('Evaluation sample counts differ')
        if (row['accuracy']!=float(cm.trace()/cm.sum()) or row['asr']!=float(cm[5,7]/cm[5].sum())
                or row['source_accuracy']!=float(cm[5,5]/cm[5].sum())):
            raise ValueError('Acc/ASR differ from recorded confusion matrix')
    timing=result['timings']
    for key in ('local_training_seconds','defense_aggregation_seconds','evaluation_seconds'):
        if not math.isclose(timing[key],sum(r['round_timing'][key] for r in rows),rel_tol=1e-12,abs_tol=1e-8):
            raise ValueError('Stage timing ledger differs')
    if timing['runtime_seconds']+1e-6<sum(timing[k] for k in ('local_training_seconds','defense_aggregation_seconds','evaluation_seconds')):
        raise ValueError('Runtime shorter than disjoint synchronized stages')
    if any(v is None or v<=0 or not math.isfinite(v) for v in result['memory'].values()):
        raise ValueError('Memory measurement missing or invalid')
    environment=p.read(folder/'environment.json')
    if environment['numerics']!=manifest['numerics'] or environment['process_environment']['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']!='python':
        raise ValueError('Actual numerical/process policy differs')
    if full:
        if status['final_model_sha256']!=p.sha(folder/'final_model.npz'):
            raise ValueError('Saved final model archive changed')
        with np.load(folder/'final_model.npz',allow_pickle=False) as archive:
            if array_digest(archive['params'])!=result['model_sha256'] or not np.isfinite(archive['params']).all():
                raise ValueError('Saved final model parameters differ')
        removed=set();malicious=set(pairing['malicious_clients']);seen=[]
        with gzip.open(folder/'audits.jsonl.gz','rt') as stream:
            for line in stream:
                row=json.loads(line);round_id=row['round'];seen.append(round_id)
                if row['binding']!=result['binding']:
                    raise ValueError('Per-round actual binding differs')
                exposures=row['exposures'];coefs=row['coefficients']
                if set(exposures)!=set(pairing['clients'])-removed or set(coefs)!=set(exposures):
                    raise ValueError('Removed client was trained or a participant was dropped')
                for client,x in exposures.items():
                    poison=c['attack']!='none' and client in malicious and round_id>=c['attack_start_round']
                    expected=dict(poisoned=poison,labels_flipped=pairing['clients'][client]['source5_samples'] if poison else 0,
                                  scale=c['scale'] if poison else 1.)
                    if x!=expected:
                        raise ValueError('Actual attack exposure differs')
                if any(not math.isfinite(v) or v<0 for v in coefs.values()):
                    raise ValueError('Nonfinite/negative actual aggregation coefficient')
                # The TAD audit exposes its post-round removed set, HFCDR its newly removed IDs.
                if c['method']=='hfcdr':
                    removed.update(row['defense']['newly_removed'])
                elif c['method']=='tad':
                    # Different TAD audit versions use newly_removed; inspect actual key.
                    removed.update(row['defense'].get('newly_removed',[]))
        if seen!=list(range(1,c['rounds']+1)):
            raise ValueError('Incomplete mechanism audit')
    return dict(task_id=task['task_id'],healthy=summarize(result)['healthy'],
        health_reasons=summarize(result)['health_reasons'],result_sha256=status['result_sha256'],
        final_model_sha256=status['final_model_sha256'],initial_model_sha256=p.read(folder/'initial_model.json')['sha256'],
        runtime_seconds=timing['runtime_seconds'],full_model_and_mechanism_checked=full)


def inspect_scheduler(output,manifest):
    path=output/'scheduler_events.jsonl'
    if not path.exists():return dict(started=0,exited=0,peak_workers=0)
    text=path.read_text()
    # An active append may expose an unfinished last line; it is not yet an event.
    rows=[json.loads(x) for x in text.splitlines(keepends=True) if x.endswith('\n') and x.strip()]
    active={};starts={};exits={};bindings={};peak=0;per_gpu=defaultdict(int);last={}
    min_free=float('inf');min_delay=float('inf')
    groups={g['group_id']:g for g in manifest['groups']};positions=Counter()
    for e in rows:
        if e['kind']=='start':
            tid=e['task_id'];gpu=e['gpu'];gid=e['group_id']
            if (tid in starts or gid not in groups or positions[gid]>=len(groups[gid]['tasks'])
                    or groups[gid]['tasks'][positions[gid]]!=tid):
                raise ValueError('Duplicate launch or wrong within-group method order')
            if any(x['group_id']==gid for x in active.values()):
                raise ValueError('Concurrent methods within same group')
            if gid in bindings and bindings[gid]!=gpu:
                raise ValueError('Scenario group moved GPUs')
            bindings[gid]=gpu;positions[gid]+=1
            if e['admission_elapsed_seconds']<5 or min(e['admission_before']['free_mib'],e['admission_after']['free_mib'])<8192:
                raise ValueError('GPU admitted without both free-memory checks')
            at=datetime.fromisoformat(e['at']).timestamp()
            if gpu in last and at-last[gpu]<60:
                raise ValueError('Same GPU launch spacing below 60 seconds')
            last[gpu]=at;starts[tid]=e;active[tid]=e
            per_gpu[gpu]+=1;peak=max(peak,len(active))
            if len(active)>8 or per_gpu[gpu]>2:
                raise ValueError('Concurrency bound exceeded')
            min_free=min(min_free,e['admission_before']['free_mib'],e['admission_after']['free_mib'])
            min_delay=min(min_delay,e['admission_elapsed_seconds'])
        elif e['kind']=='exit':
            tid=e['task_id']
            if tid not in active or tid in exits or starts[tid]['pid']!=e['pid']:
                raise ValueError('Exit without a unique matching launch')
            exits[tid]=e;per_gpu[e['gpu']]-=1;active.pop(tid)
    return dict(started=len(starts),exited=len(exits),peak_workers=peak,active=list(active),
        minimum_admission_free_mib=None if not starts else min_free,
        minimum_admission_delay_seconds=None if not starts else min_delay,
        no_duplicate_attempts=True,same_gpu_and_rotated_method_order=True)


def audit(study,*,full=False,previous=None):
    out=p.safe_output(study);check=p.check(out);manifest=p.read(out/'manifest.json')
    validated={};failures=[];prior=previous.get('validated',{}) if previous else {}
    if previous and previous.get('fingerprint')!=manifest['fingerprint']:
        raise ValueError('Previous audit belongs to another source manifest')
    for task in manifest['tasks']:
        folder=out/'tasks'/task['task_id'];status_path=folder/'status.json'
        if not status_path.exists():continue
        status=p.read(status_path)
        if status['status']=='complete':
            cached=prior.get(task['task_id'])
            if cached and cached['result_sha256']==p.sha(folder/'result.json') and (not full or cached['full_model_and_mechanism_checked']):
                validated[task['task_id']]=cached
            else:validated[task['task_id']]=inspect_task(folder,task,manifest,full=full)
        elif status['status']=='failed':failures.append(dict(task_id=task['task_id'],status=status))
    starts=defaultdict(set)
    for task in manifest['tasks']:
        if task['task_id'] in validated:
            c=task['config'];starts[c['dataset'],c['seed']].add(validated[task['task_id']]['initial_model_sha256'])
    if any(len(values)!=1 for values in starts.values()):
        raise ValueError('Initial model differs across paired methods/partitions/attacks')
    controller=p.read(out/'controller.json') if (out/'controller.json').exists() else None
    return dict(at=datetime.now(timezone.utc).isoformat(),fingerprint=manifest['fingerprint'],
        counts=check['counts'],validated=validated,execution_or_scientific_failures=failures,
        health_failures=[v['task_id'] for v in validated.values() if not v['healthy']],
        scheduler=inspect_scheduler(out,manifest),controller=controller,
        full_matrix_executed=len(validated)+len(failures)==len(manifest['tasks']),
        all_results_complete=len(validated)==len(manifest['tasks']),paired_initial_models=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=p.OUTPUT)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--save',type=Path)
    a=parser.parse_args();result=audit(a.study,full=a.full,previous=p.read(a.previous) if a.previous else None)
    if a.save:p.write(a.save,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('validated','controller')},indent=2))


if __name__=='__main__':main()
