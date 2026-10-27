#!/usr/bin/env python3
"""Independent CUDA development, fixed V2 attack, immutable tasks, idle-GPU queue.

This panel is diagnostic development. It cannot certify two distributions,
five ratios and six methods until that separate complete matrix is audited.
"""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT/'configs/cifar10_cp1_cuda_diagnosis150_v1.json'
TERMINAL = {'complete','failed','numerical_failure'}
OWN = ('run_cifar_cp1_cuda_development.py','diagnose_cifar_cp1_iid.py',
       'cifar_optimized_station_runtime.py','cifar_alternating_target_views.py',
       'cifar_tad_no_prior.py','tests/test_cp1_cuda_development.py')


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name('.'+path.name+'.tmp')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def specs_for(p):
    from sm9rrsfl.fl import ExperimentConfig
    from sm9rrsfl.ours_policy import OursParameters, OURS_PARAMETER_NAMES
    frozen=read(ROOT/p['parent_config'])
    if sha(ROOT/p['parent_config'])!=p['parent_config_sha256']:
        raise ValueError('Frozen MPS protocol changed')
    for method,base in p['base_configs'].items():
        if base != frozen['cells']['iid_clean_'+method]['config']:
            raise ValueError('Public training/base method parameters changed')
    if p['profile'] != frozen['cells']['iid_r60_sm9rrs']['profile']:
        raise ValueError('Attack profile changed')
    expected_policy=dict(clean_accuracy_gap=.01,clean_asr_gap=.02,clean_rank_required=False,
        attack_accuracy_gap=.02,attack_asr_gap=.05,attack_max_rank=2,
        max_clean_accuracy_drop=.05,max_honest_false_revocation_rate=.1,
        reference_scope='healthy_method_across_all_five_ratios_within_partition',
        attack_min_accuracy_drop=.10,attack_min_asr_rise=.30)
    if p['selection_policy']!=expected_policy:
        raise ValueError('Latest user selection criteria changed')
    if p['numerics'] != dict(deterministic_algorithms=True,cudnn_deterministic=True,
            cudnn_benchmark=False,cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False,
            cublas_workspace_config=':4096:8',torch_threads=2):
        raise ValueError('Explicit CUDA numerical policy required')
    specs=[]
    for t in p['tasks']:
        import re
        if not re.fullmatch(r'[A-Za-z0-9_]+',t['id']):raise ValueError('Unsafe task id')
        arm=p['arms'][t['arm']];method=arm['method'];changes=arm['parameters']
        if changes and (method!='sm9rrs' or not set(changes)<=set(OURS_PARAMETER_NAMES)):
            raise ValueError('Only Ours policy parameters can be optimized')
        if t['partition'] not in ('iid','dirichlet') or t['ratio'] not in (0.,.2,.4,.6,.8):
            raise ValueError('Unexpected validation scenario')
        cfg=dict(p['base_configs'][method],**changes,partition=t['partition'],
            malicious_ratio=t['ratio'],device='cuda:0')
        if cfg['rounds']!=150 or cfg['seed']!=2026102201:
            raise ValueError('Observed-seed 150-round development only')
        obj=ExperimentConfig(**cfg)
        if method=='sm9rrs':OursParameters.from_object(obj).validate()
        specs.append(dict(task_id=t['id'],arm=t['arm'],config=cfg,profile=deepcopy(p['profile'])))
    if len({s['task_id'] for s in specs})!=len(specs):raise ValueError('Duplicate task')
    return specs


def plan(config):
    p=read(config);specs=specs_for(p)
    for name,h in p['frozen_source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen scientific source changed: '+name)
    names=set(p['frozen_source_sha256'])|set(OWN)|{str(Path(config).resolve().relative_to(ROOT))}
    source={n:sha(ROOT/n) for n in sorted(names)}
    data=dict(p['data_config'],data_dir=str(ROOT/p['data_config']['data_dir']))
    identity=dict(protocol=p,specs=specs,data_config=data,sources=source,
        official_test_used_for_selection=False,maximum_rounds=150*len(specs))
    return p,specs,{**identity,'fingerprint':digest(identity)}


def verify(manifest):
    for name,h in manifest['sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('Active manifest source changed: '+name)


def gpu_snapshot():
    r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20,check=True)
    records=[]
    for line in r.stdout.strip().splitlines():
        idx,uid,name,total,free,util=(v.strip() for v in line.split(','))
        records.append(dict(index=int(idx),uuid=uid,name=name,total_mb=int(total),free_mb=int(free),util=int(util)))
    if not records:raise ValueError('No visible NVIDIA devices')
    return records


def qualified_gpus(rows, policy, pinned=None):
    return [r for r in rows if (pinned is None or r['uuid']==pinned)
        and r['free_mb']>=policy['min_free_mb'] and r['util']<=policy['max_utilization']
        and r['name']=='NVIDIA GeForce RTX 4090 D']


def guard_task_directory(folder):
    if not folder.exists():return 'new'
    state=read(folder/'status.json')
    if state and state.get('status') in TERMINAL:return state['status']
    raise ValueError('Partial or unidentified task retained; no automatic retry: '+str(folder))


@contextmanager
def observation_scope(expected_data):
    from scripts.experiments.cifar10 import attack_stability_worker as worker
    from run_cifar_targeted_optimization import metric_scope
    from cifar_optimized_station_runtime import aggregation_observation
    loader=worker._load_dataset
    def checked_load(config):
        data,contract=loader(config)
        if worker._dataset_identity(data,contract)!=expected_data:
            raise ValueError('CUDA dataset differs from the audited MPS dataset')
        return data,contract
    with metric_scope():
        evaluator=worker.RoundMetricEvaluator
        class Evaluator(evaluator):
            def evaluate(self,state):
                row=super().evaluate(state)
                row['blacklisted_client_ids']=sorted(state['blacklisted'])
                row['aggregation_coefficients']=aggregation_observation(state,self.config.method)
                return row
        worker.RoundMetricEvaluator=Evaluator;worker._load_dataset=checked_load
        try:yield
        finally:worker.RoundMetricEvaluator=evaluator;worker._load_dataset=loader


def run_worker(p,spec,m,out):
    verify(m)
    name=spec['task_id'];folder=out/'tasks'/name
    if guard_task_directory(folder)!='new':raise ValueError('Worker only accepts a fresh task')
    import torch
    n=p['numerics']
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=n['cublas_workspace_config']:
        raise ValueError('Set CUBLAS policy before launching worker')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Exactly one controller-selected CUDA device required')
    torch.set_num_threads(n['torch_threads'])
    torch.use_deterministic_algorithms(n['deterministic_algorithms'])
    torch.backends.cudnn.deterministic=n['cudnn_deterministic']
    torch.backends.cudnn.benchmark=n['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32=n['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32=n['cudnn_allow_tf32']
    env=dict(python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],numerics=n,
        native_binary_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'sm9rrsfl').glob('_native_*.so')})
    write(out/'environments'/f'{name}.json',env)
    from cifar_alternating_target_views import target_view_scope
    from cifar_tad_no_prior import prior_free_tad_scope
    from scripts.experiments.cifar10.attack_stability_worker import execute_task
    req=dict(spec,kind='train',device='cuda:0',data_config=m['data_config'],
        fingerprint=m['fingerprint'],output_dir=str(out/'tasks'))
    write(out/'requests'/f'{name}.json',req)
    with target_view_scope(),observation_scope(p['expected_dataset_identity']),(
            prior_free_tad_scope() if spec['config']['method']=='ding13' else nullcontext()):
        state=execute_task(req)
    verify(m)
    return 0 if state['status'] in TERMINAL else 2


def summarize(p,specs,m,out):
    from scripts.experiments.cifar10.attack_stability_worker import task_identity
    from scripts.experiments.cifar10.c7_station_report import _health
    verify(m);cells={};inputs={};trajectories={}
    def load(path):
        blob=Path(path).read_bytes();inputs[str(Path(path).relative_to(ROOT))]=hashlib.sha256(blob).hexdigest()
        return json.loads(blob)
    for s in specs:
        name=s['task_id'];folder=out/'tasks'/name;state=read(folder/'status.json',{})
        cell=dict(status=state.get('status','not_run'),arm=s['arm'],config=s['config'],
            completed_round=state.get('completed_round'),complete_finite=False)
        cells[name]=cell
        if state.get('status') not in TERMINAL:continue
        state=load(folder/'status.json');identity=task_identity(dict(s,fingerprint=m['fingerprint'],data_config=m['data_config']))
        if load(folder/'task.json')!=identity or state['fingerprint']!=identity['fingerprint']:
            raise ValueError('Task identity differs: '+name)
        if state['status']!='complete':
            cell['failure']=dict(error_type=state.get('error_type'),error=state.get('error'));continue
        metrics=load(folder/'metrics.json');rows=metrics['rounds']
        if (metrics['fingerprint']!=identity['fingerprint'] or state['completed_round']!=150
                or [r['round'] for r in rows]!=list(range(151)) or state['final_metrics']!=rows[-1]):
            raise ValueError('Invalid complete trajectory')
        if load(folder/'dataset_identity.json')!=p['expected_dataset_identity']:
            raise ValueError('Data identity differs')
        keys=('accuracy','source_transfer_asr','honest_weight_loss','malicious_weight_mass')
        if any(not math.isfinite(r[k]) or not -1e-9<=r[k]<=1+1e-9 for r in rows for k in keys):
            raise ValueError('Invalid metrics')
        finite=state['nonfinite_updates']==0 and all(r['cumulative_nonfinite_updates']==0 for r in rows)
        cell.update(complete_finite=finite,tail={k:sum(r[k] for r in rows[-10:])/10 for k in keys},
            health=_health(dict(valid_complete=True,numerically_finite=finite),rows,s['config']),
            attack_period={k:sum(r[k] for r in rows[25:])/126 for k in ('accuracy','source_transfer_asr')},
            peak_attack_asr=max(r['source_transfer_asr'] for r in rows[25:]),
            final_fp=rows[-1]['false_positive_revocations'],final_tp=rows[-1]['true_positive_revocations'])
        trajectories[name]=rows
    pairing={};gates={}
    for s in specs:
        if not s['config']['malicious_ratio']:continue
        name=s['task_id'];clean=next((v for v in specs if v['arm']==s['arm'] and
            v['config']['partition']==s['config']['partition'] and not v['config']['malicious_ratio']),None)
        if name not in trajectories or clean is None or clean['task_id'] not in trajectories:continue
        c=clean['task_id'];fields=('accuracy','source_transfer_asr','non_source_accuracy','aux128_confidence')
        prefix=all(trajectories[name][r][k]==trajectories[c][r][k] for r in range(25) for k in fields)
        import numpy as np
        anchors=[]
        for task in (name,c):
            f=out/'tasks'/task/'anchor_r24.npz';inputs[str(f.relative_to(ROOT))]=sha(f)
            with np.load(f) as z:anchors.append(z['params'].copy())
        pairing[name]=dict(classification_prefix_equal=prefix,round24_model_equal=bool(np.array_equal(*anchors)))
        if s['config']['method']=='fedavg':
            drop=cells[c]['tail']['accuracy']-cells[name]['tail']['accuracy']
            rise=cells[name]['tail']['source_transfer_asr']-cells[c]['tail']['source_transfer_asr']
            gates[name]=dict(accuracy_drop=drop,asr_rise=rise,passed=drop>=.10 and rise>=.30)
    for name,h in inputs.items():
        if sha(ROOT/name)!=h:raise ValueError('Inputs changed during summary')
    report=dict(fingerprint=m['fingerprint'],cells=cells,attack_pairing=pairing,
        fedavg_attack_gates=gates,input_sha256=inputs,main_qualified=False,
        status='complete_panel' if all(c['status'] in TERMINAL for c in cells.values()) else 'incomplete',
        limitations='Independent deterministic CUDA development. No MPS scores reused for ranking. '
            'No final health eligibility or user qualification without all five ratios and six methods in each distribution.')
    write(out/'analysis/development_summary.json',report)
    return report


def batch(config,p,specs,m,out):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a') as control:
        fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=read(out/'manifest.json')
        if old is not None and old!=m:raise ValueError('Immutable CUDA manifest differs')
        if old is None and any(f.name!='controller.lock' for f in out.iterdir()):
            raise ValueError('Unidentified output retained')
        write(out/'manifest.json',m)
        previous=read(out/'controller_status.json',{})
        pinned=previous.get('pinned_gpu_uuid');child=None;gpu_lock=None
        def interrupted(signum,frame):raise KeyboardInterrupt('Controller stop requested')
        old_signal=signal.signal(signal.SIGTERM,interrupted)
        try:
            for s in specs:
                verify(m);name=s['task_id'];folder=out/'tasks'/name
                state=guard_task_directory(folder)
                if state in TERMINAL:continue
                counts={};start=time.monotonic()
                while True:
                    rows=gpu_snapshot();eligible=qualified_gpus(rows,p['gpu_admission'],pinned)
                    ids={r['uuid'] for r in eligible}
                    counts={uid:counts.get(uid,0)+1 for uid in ids}
                    ready=[r for r in eligible if counts[r['uuid']]>=p['gpu_admission']['consecutive_samples']]
                    selected=None
                    for r in sorted(ready,key=lambda r:(-r['free_mb'],r['index'])):
                        lockdir=ROOT/'outputs/.codex_gpu_locks';lockdir.mkdir(exist_ok=True)
                        handle=(lockdir/(r['uuid']+'.lock')).open('a')
                        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:handle.close();continue
                        gpu_lock=handle;selected=r;break
                    if selected is not None:break
                    write(out/'controller_status.json',dict(status='waiting_for_idle_gpu',task=name,
                        pid=os.getpid(),pinned_gpu_uuid=pinned,gpu_snapshot=rows,
                        idle_samples=counts,wait_seconds=time.monotonic()-start,updated_utc=utc()))
                    if time.monotonic()-start>p['gpu_admission']['maximum_wait_seconds']:
                        raise TimeoutError('No qualified idle GPU; infrastructure wait, not scientific failure')
                    time.sleep(p['gpu_admission']['sample_interval_seconds'])
                pinned=selected['uuid']
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=pinned,
                    CUBLAS_WORKSPACE_CONFIG=p['numerics']['cublas_workspace_config'])
                log=out/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
                with log.open('a') as stream:
                    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                        '--config',str(Path(config).resolve()),'--task',name],cwd=ROOT,env=env,
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    write(out/'controller_status.json',dict(status='training',task=name,pid=os.getpid(),
                        child_pid=child.pid,pinned_gpu_uuid=pinned,admission=selected,updated_utc=utc()))
                    code=child.wait();child=None
                gpu_lock.close();gpu_lock=None
                state=read(folder/'status.json',{})
                if code!=0 or state.get('status') not in TERMINAL:
                    raise RuntimeError(f'{name}: exit={code}; state={state.get("status")}')
                summarize(p,specs,m,out)
                if state['status']=='failed':
                    raise RuntimeError('Worker infrastructure/unclassified failure retained: '+name)
            summarize(p,specs,m,out)
            write(out/'controller_status.json',dict(status='panel_terminal',pinned_gpu_uuid=pinned,updated_utc=utc()))
        except BaseException as e:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGINT)
                try:child.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
            write(out/'controller_status.json',dict(status='interrupted_or_blocked',task=locals().get('name'),
                error_type=type(e).__name__,error=str(e),pinned_gpu_uuid=pinned,updated_utc=utc()))
            raise
        finally:
            if gpu_lock is not None:gpu_lock.close()
            signal.signal(signal.SIGTERM,old_signal)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT)
    group=parser.add_mutually_exclusive_group();group.add_argument('--batch',action='store_true')
    group.add_argument('--task');group.add_argument('--summary',action='store_true')
    args=parser.parse_args();p,specs,m=plan(args.config);out=ROOT/p['output_dir']
    if args.task:
        if read(out/'manifest.json')!=m:raise ValueError('Worker needs controller manifest')
        s=next(s for s in specs if s['task_id']==args.task)
        raise SystemExit(run_worker(p,s,m,out))
    elif args.batch:batch(args.config,p,specs,m,out)
    elif args.summary:print(json.dumps(summarize(p,specs,m,out),indent=2))
    else:print(json.dumps(dict(fingerprint=m['fingerprint'],tasks=[s['task_id'] for s in specs],
        maximum_rounds=m['maximum_rounds'],training_started=False),indent=2))


if __name__=='__main__':main()
