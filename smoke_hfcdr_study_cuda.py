#!/usr/bin/env python3
"""Explicit tiny real-SGD plumbing checks; never part of the 936 study runs."""
from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')


def main():
    import numpy as np
    import hfcdr_study_protocol as p
    from hfcdr_study_runtime import configure_cuda,run_loop
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=p.ROOT/'data')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite smoke evidence')
    args.output.mkdir(parents=True)
    manifest=p.plan();environment=configure_cuda(manifest);rows=[]
    for dataset in p.policy()['datasets']:
        ds,_=p.load_dataset(dataset,args.data_root)
        train=np.concatenate([np.flatnonzero(ds.y_train==i)[:24] for i in range(10)])
        test=np.concatenate([np.flatnonzero(ds.y_train==i)[24:26] for i in range(10)])
        tiny=replace(ds,x_train=ds.x_train[train],y_train=ds.y_train[train],
                     x_test=ds.x_train[test],y_test=ds.y_train[test])
        del ds
        starts=[]
        for method in p.policy()['methods']:
            original=next(t for t in manifest['tasks'] if t['config']['dataset']==dataset
                and t['config']['method']==method and t['condition']=='scale5'
                and t['config']['malicious_ratio']==.4 and t['config']['partition']=='iid')
            task=dict(original,config=dict(original['config'],num_clients=6,rounds=4,attack_start_round=2))
            logs=[]
            result=run_loop(tiny,task,device='cuda:0',emit=lambda k,v:logs.append((k,v)))
            assert [r['round'] for r in result['records']]==[0,1,2,3,4]
            starts.append(next(v for k,v in logs if k=='initial_model')['sha256'])
            audits=[v for k,v in logs if k=='audit']
            assert not any(x['poisoned'] for x in audits[0]['exposures'].values())
            assert sum(x['poisoned'] for x in audits[1]['exposures'].values())==2
            final=next(v for k,v in logs if k=='final_model')
            assert np.isfinite(final).all()
            rows.append(dict(dataset=dataset,method=method,passed=True,rounds=4,
                training_samples=240,evaluation_samples=20,binding=result['binding'],
                runtime_seconds=result['timings']['runtime_seconds'],memory=result['memory']))
            p.write(args.output/'progress.json',dict(completed=len(rows),total=12,last=rows[-1]))
            print(dataset,method,'PASS',flush=True)
            del result,logs,final
        assert len(set(starts))==1
        del tiny
    p.write(args.output/'report.json',dict(scope='non_scientific_plumbing_only',
        evaluation_source='disjoint original training subset; official test unused',
        real_sgd=True,training_rounds=48,checks=rows,environment=environment,
        fingerprint=manifest['fingerprint'],passed=True,official_study_tasks_started=0))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
