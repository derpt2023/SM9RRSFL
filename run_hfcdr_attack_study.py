#!/usr/bin/env python3
"""Default read-only plan; explicit --prepare, --batch, --run-task, --report."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name,'1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import hfcdr_study_protocol as p
from hfcdr_study_scheduler import now


def run_task(output, data_root, task_id):
    out=p.safe_output(output)
    p.check(out)
    manifest=p.read(out/'manifest.json')
    task=next((t for t in manifest['tasks'] if t['task_id']==task_id),None)
    if task is None:
        raise ValueError('Unknown task')
    folder=out/'tasks'/task_id
    if folder.resolve()!=folder:
        raise ValueError('Task path cannot contain symlinks')
    folder.parent.mkdir(exist_ok=True)
    folder.mkdir()  # Atomic one-attempt guard, before any GPU allocation.
    identity=dict(manifest_fingerprint=manifest['fingerprint'],pid=os.getpid(),task_id=task_id)
    p.write(folder/'task.json',task)
    p.write(folder/'status.json',dict(identity,status='loading',at=now()))
    started=perf_counter()
    try:
        from hfcdr_pilot_data import participant_record
        from hfcdr_study_runtime import configure_cuda,run_loop,assert_no_cryptography
        environment=configure_cuda(manifest)
        p.write(folder/'environment.json',environment)
        data_start=perf_counter()
        dataset,data_identity=p.load_dataset(task['config']['dataset'],data_root)
        if data_identity!=p.read(out/'dataset_identity.json')[task['config']['dataset']]:
            raise ValueError('Dataset changed after preparation')
        pairing=p.read(out/'pairing.json')[p.pairing_key(task)]
        if participant_record(dataset.y_train,task['config'])!=pairing:
            raise ValueError('Client partition or attack identity mismatch')
        data_seconds=perf_counter()-data_start
        p.write(folder/'dataset_identity.json',data_identity)
        p.write(folder/'pairing.json',pairing)
        p.write(folder/'status.json',dict(identity,status='running',at=now()))
        with (folder/'rounds.jsonl').open('x') as rounds, gzip.open(folder/'audits.jsonl.gz','xt') as audits:
            def emit(kind,value):
                if kind=='final_model':
                    import numpy as np
                    np.savez_compressed(folder/'final_model.npz',params=value)
                elif kind in ('binding','initial_model'):
                    p.write(folder/(kind+'.json'),value)
                else:
                    stream=rounds if kind=='round' else audits
                    stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
                    stream.flush()
                    if kind=='round':
                        p.write(folder/'progress.json',dict(round=value['round'],at=now(),
                            accuracy=value['accuracy'],asr=value['asr'],
                            elapsed_wall_seconds=perf_counter()-started))
            result=run_loop(dataset,task,device='cuda:0',emit=emit)
        assert_no_cryptography()
        if p.plan()!=manifest:
            raise ValueError('Frozen scientific sources changed during training')
        result['timings'].update(data_loading_seconds=data_seconds,
                                 worker_wall_seconds=perf_counter()-started)
        result['memory']['host_peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(2**20 if sys.platform=='darwin' else 1024)
        from hfcdr_study_report import summarize
        p.write(folder/'result.json',result)
        summary=summarize(result)
        p.write(folder/'summary.json',summary)
        p.write(folder/'status.json',dict(identity,status='complete',at=now(),healthy=summary['healthy'],
            result_sha256=p.sha(folder/'result.json'),final_model_sha256=p.sha(folder/'final_model.npz')))
    except BaseException as error:
        scientific=isinstance(error,FloatingPointError) or str(error).startswith('Scientific collapse:')
        p.write(folder/'status.json',dict(identity,status='failed',at=now(),error=repr(error),
            failure_type='scientific' if scientific else 'execution',automatic_retry=False,
            elapsed_wall_seconds=perf_counter()-started))
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--plan',action='store_true')
    actions.add_argument('--prepare',action='store_true')
    actions.add_argument('--check',action='store_true')
    actions.add_argument('--ready',action='store_true',help='Check data/source/CUDA; no training')
    actions.add_argument('--batch',action='store_true',help='Run the frozen 936 tasks once')
    actions.add_argument('--run-task',metavar='TASK_ID')
    actions.add_argument('--report',action='store_true')
    parser.add_argument('--output',type=Path,default=p.OUTPUT)
    parser.add_argument('--data-root',type=Path,default=p.ROOT/'data')
    args=parser.parse_args(argv)
    if args.prepare:
        result=p.prepare(args.output,args.data_root)
    elif args.check:
        result=p.check(args.output)
    elif args.ready:
        result=p.check(args.output)
        from hfcdr_study_runtime import configure_cuda
        result['environment']=configure_cuda(p.read(args.output/'manifest.json'))
        result['data']={}
        for name in p.policy()['datasets']:
            ds,identity=p.load_dataset(name,args.data_root)
            if identity!=p.read(args.output/'dataset_identity.json')[name]:
                raise ValueError('Prepared data identity differs')
            result['data'][name]=dict(accuracy_samples=identity['accuracy_samples'],source_samples=identity['source_samples'])
            del ds
    elif args.run_task:
        run_task(args.output,args.data_root,args.run_task);return 0
    elif args.batch:
        from hfcdr_study_scheduler import batch
        batch(args.output,args.data_root);return 0
    elif args.report:
        p.check(args.output)
        from hfcdr_study_report import report
        result=report(p.safe_output(args.output),p.read(args.output/'manifest.json'))
        p.write(args.output/'report.json',result)
        result={k:v for k,v in result.items() if k not in ('summaries','aggregate','comparisons','missing','failed_or_incomplete')}
    else:
        result=p.plan()
        result=dict(result['audit'],fingerprint=result['fingerprint'],training_started=False)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__=='__main__':
    sys.exit(main())
