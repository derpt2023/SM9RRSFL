"""User-requested fixed R4 official-test supplement; prior failed selection is retained."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path

import run_cifar_h84_formal as formal
import run_cifar_h84_full_validation as full
import run_cifar_ours_reserve as history
from run_cifar_cp1_cuda_development import ROOT, TERMINAL, read, write, sha, digest, utc, verify, qualified_gpus, guard_task_directory

DEFAULT=ROOT/'configs/cifar10_ours_r4_formal150_v1.json'
OWN=('run_cifar_r4_formal.py','cifar_r4_formal_scheduler.py','cifar_r4_formal_report.py',
     'render_cifar_r4_formal.py','tests/test_r4_formal.py')
SEEDS=(2026102801,2026102802,2026102803)
PARAMETERS={'suspicion_recovery_factor':4.0}
QUARANTINE=history.QUARANTINE
PROCESS_ENV=history.PROCESS_ENV
BASELINE='outputs/cifar_h84_formal_vert_paper150_v1/export/combined_bundle.json'
BASELINE_SHA=history.BASELINE_SHA
RESERVE='outputs/cifar_ours_reserve150_v1/analysis/reserve_results.json'
RESERVE_SHA='1ca906b845ea6e05d0d9fc02273d8eb89726d381f7ea234b6053ccb7b85ad1f5'
ORIGINAL='outputs/cifar_h84_formal150_v1'
OUTPUT='outputs/cifar_ours_r4_formal150_v1'
AUTHORIZATION=dict(date='2026-11-01',user_requested_candidate='R4',
    user_instruction='用R4配置补做ours的cifar-paper相同3seed试验，在ai station上补做，每两小时检查一次进度，做完后将新试验结果同其他5法的对比图发给我。',
    validation_selection_passed=False,validation_selected=None,
    interpretation='Explicit user-directed follow-up after the bounded validation failed; not a reversal of its frozen decision.',
    prior_official_test_and_seeds_seen=True,official_test_retuning=False)


def gpu_snapshot():
    return [g for g in formal.gpu_snapshot() if g['uuid'] not in QUARANTINE]


def plan(config=DEFAULT):
    a=read(config)
    expected=dict(protocol='cifar10_ours_r4_formal150_v1',candidate='R4',parameters=PARAMETERS,
        output_dir=OUTPUT,seeds=list(SEEDS),rounds=150,evaluation_scope='official_test',
        original_formal_config='configs/cifar10_h84_formal150_v1.json',
        accepted_bundle=BASELINE,accepted_bundle_sha256=BASELINE_SHA,
        reserve_report=RESERVE,reserve_report_sha256=RESERVE_SHA,authorization=AUTHORIZATION,
        process_environment=PROCESS_ENV,quarantine_gpu_uuids=list(QUARANTINE),
        automatic_retry=False,retune_on_test=False,other_five_retrained=False,
        checkpoint_rounds=[0,24,150],maximum_training_rounds=4500)
    if set(a)!=set(expected)|{'frozen_source_sha256','pinned_reference_sha256'} or any(a[k]!=v for k,v in expected.items()):
        raise ValueError('Only the explicitly requested fixed R4 official-test protocol is allowed')
    frozen=a['frozen_source_sha256']
    if len(frozen)!=120 or frozen!=read(ROOT/history.DEFAULT)['frozen_source_sha256']|{
            n:sha(ROOT/n) for n in (*history.OWN,str(history.DEFAULT.relative_to(ROOT)))}:
        raise ValueError('Original 120 scientific sources must remain frozen')
    for n,h in frozen.items():
        if sha(ROOT/n)!=h:raise ValueError('Frozen source changed: '+n)
    p,old_specs,old=formal.plan(ROOT/a['original_formal_config'])
    if p['seeds']!=list(SEEDS):raise ValueError('Original three seeds differ')
    specs=[]
    for prior in old_specs:
        if prior['arm']!='H84':continue
        s=deepcopy(prior);s.update(arm='R4',task_id=prior['task_id'].replace('H84_','R4_',1))
        s['config'].update(PARAMETERS)
        if s['config']['detector_clip_factor']!=1.0:raise ValueError('Original clipping must remain one')
        specs.append(s)
    if len(specs)!=30:raise ValueError('Exactly30 fresh Ours tasks required')
    refs=a['pinned_reference_sha256']
    if refs.get(BASELINE)!=BASELINE_SHA or refs.get(RESERVE)!=RESERVE_SHA or set(refs)!={
            ORIGINAL+'/manifest.json',ORIGINAL+'/reference_lock.json',ORIGINAL+'/analysis/formal_results.json',
            history.PARENT.replace('cifar_ours_paper_refinement150_v1','cifar_ours_reserve150_v1')+'/manifest.json',
            'outputs/cifar_ours_reserve150_v1/reference_lock.json',RESERVE,BASELINE}:
        raise ValueError('Pinned original and unsuccessful validation evidence required')
    sources=dict(frozen)
    for n in (*OWN,str(Path(config).resolve().relative_to(ROOT))):sources[n]=sha(ROOT/n)
    p=deepcopy(p);p.update(output_dir=OUTPUT,amendment=a,process_environment=PROCESS_ENV,
        acceptance=AUTHORIZATION,candidate='R4')
    identity=dict(protocol=a,specs=specs,sources=sources,data_config=old['data_config'],
        original_formal_fingerprint=old['fingerprint'],maximum_rounds=4500,
        evaluation_scope='official_test',official_test_used_for_selection=False,
        prior_test_results_seen=True,prior_validation_selected=None,
        user_requested_after_validation_failure=True,other_five_source_sha256=BASELINE_SHA)
    return p,specs,dict(identity,fingerprint=digest(identity))


def validate_prior(r):
    if (r['status']!='complete_audited_reserve_validation' or not r['execution_audited']
            or r['decision']['selected'] is not None or r['decision']['candidates']['R4']['eligible']
            or r['decision']['candidates']['R4']['failures']!=['mean_IID_accuracy_gain_below_0.5pp']
            or len(r['cells'])!=120 or len(r['attack_pairing'])!=96
            or not all(v['classification_prefix_equal'] and v['round24_model_equal'] for v in r['attack_pairing'].values())
            or len(r['h84_warmup_pairing'])!=90 or not all(r['h84_warmup_pairing'].values())
            or r['prior_panel']['original_pairing_gate_passed'] is not False):
        raise ValueError('Preserve exact failed selection and completed reserve audit')


def references(p,m):
    full.verify_files(p['amendment']['pinned_reference_sha256'])
    r=read(ROOT/RESERVE);validate_prior(r)
    original=read(ROOT/ORIGINAL/'manifest.json')
    if original['fingerprint']!=m['original_formal_fingerprint']:raise ValueError('Original formal identity differs')
    old_lock=read(ROOT/ORIGINAL/'reference_lock.json')
    lock=formal.references(p,m)
    for key in ('expected_environment','dataset_identity'):
        if lock[key]!=old_lock[key]:raise ValueError('Official data or runtime differs from cifar-paper')
    lock['references'].update(p['amendment']['pinned_reference_sha256'])
    bundle=read(ROOT/BASELINE)
    if len(bundle['report']['cells'])!=180 or len(bundle['trajectories'])!=180:raise ValueError('Accepted180 required')
    for s in m['specs']:
        old=bundle['report']['cells'][s['task_id'].replace('R4_','H84_',1)]
        cfg=dict(old['config'],**PARAMETERS)
        if cfg!=s['config']:raise ValueError('Only recovery1.25 to4 may differ from accepted Ours')
    lock['acceptance']=AUTHORIZATION
    from run_cifar_vert_paper_validation import runtime_environment
    runtime=runtime_environment()
    prior_runtime=read(ROOT/'outputs/cifar_ours_reserve150_v1/reference_lock.json')['runtime_environment']
    if runtime!=prior_runtime:raise ValueError('Validated software/process environment differs')
    lock.update(runtime_environment=runtime,prior_validation_selected=None,
        prior_validation_report_sha256=RESERVE_SHA,accepted_bundle_sha256=BASELINE_SHA)
    return lock


@contextmanager
def r4_metadata_scope():
    # The frozen formal observer labels H84. Override only its result annotation,
    # after the observer constructs it; training/evaluation operations are unchanged.
    from scripts.experiments.cifar10 import attack_stability_worker as w
    original=w._write_json
    def labelled(path,value):
        if Path(path).name=='result.json':
            value=deepcopy(value)
            value['qualification']='User-requested fixed R4 official test; prior validation selection failed and remains unchanged'
            value['user_authorization']=AUTHORIZATION
        return original(path,value)
    w._write_json=labelled
    try:yield
    finally:w._write_json=original


def summarize(p,specs,m,out):
    from cifar_h84_formal_report import summarize as summary
    return summary(p,specs,m,out)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=DEFAULT)
    group=parser.add_mutually_exclusive_group()
    for flag in ('ready','batch','summary','finish'):group.add_argument('--'+flag,action='store_true')
    group.add_argument('--task');args=parser.parse_args()
    os.environ.update(PROCESS_ENV)
    p,specs,m=plan(args.config);out=ROOT/p['output_dir']
    if args.ready or args.batch or args.task:history.first.legacy.require_station()
    if args.task:
        verify(m)
        if read(out/'manifest.json')!=m:raise ValueError('Frozen R4 identity required')
        from run_cifar_vert_paper_validation import runtime_environment
        if runtime_environment()!=read(out/'reference_lock.json')['runtime_environment']:raise ValueError('Worker process environment changed')
        from cifar_vert_paper_core import bounded_checkpoints,actual_weights_scope
        spec=next(s for s in specs if s['task_id']==args.task);folder=out/'tasks'/args.task
        with r4_metadata_scope(),bounded_checkpoints(150),actual_weights_scope(folder/'actual_weights.jsonl',args.task):
            code=formal.run_worker(p,spec,m,out)
        raise SystemExit(code)
    if args.ready or args.batch:
        verify(m);lock=references(p,m);full.check_runtime(lock['expected_environment'])
        if args.ready:
            print(json.dumps(dict(ready=True,tasks=30,sources=len(m['sources']),fingerprint=m['fingerprint'],prior_selected=None,user_requested_R4=True)));return
        from run_cifar_w85_history_admission import prepare
        prepare(out,m,lock)
        from cifar_r4_formal_scheduler import run,plan_policy
        run(args.config,p,specs,m,out,plan_policy(m))
        if read(out/'controller_status.json')['status']!='panel_terminal':return
    if args.summary:print(json.dumps(summarize(p,specs,m,out),indent=2));return
    if args.batch or args.finish:
        if args.finish:history.require_finished_controller(out)
        verify(m)
        if read(out/'manifest.json')!=m:raise ValueError('Frozen R4 identity required')
        from cifar_r4_formal_scheduler import audit_packing
        audit_packing(out)
        from cifar_r4_formal_report import finish
        print(json.dumps(finish(p,specs,m,out),indent=2));return
    print(json.dumps(dict(tasks=30,rounds=4500,sources=len(m['sources']),fingerprint=m['fingerprint'],training_started=False,user_requested_R4=True,prior_selected=None)))


if __name__=='__main__':main()
