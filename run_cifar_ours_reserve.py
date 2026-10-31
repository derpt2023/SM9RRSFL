"""Independent, bounded reserve validation; never retries the failed first panel."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import run_cifar_ours_only as first
from run_cifar_cp1_cuda_development import ROOT, TERMINAL, read, write, sha, digest, utc, qualified_gpus, guard_task_directory

DEFAULT=ROOT/'configs/cifar10_ours_reserve150_v1.json'
OWN=('run_cifar_ours_reserve.py','cifar_ours_reserve_scheduler.py','cifar_ours_reserve_report.py','tests/test_ours_reserve.py')
SEEDS=first.SEEDS
CANDIDATES=deepcopy(first.legacy.RESERVE)
ARMS=('H84',*CANDIDATES)
RULE=deepcopy(first.RULE)
QUARANTINE=first.QUARANTINE
PROCESS_ENV=first.PROCESS_ENV
BASELINE_SHA=first.BASELINE_SHA
PARENT='outputs/cifar_ours_paper_refinement150_v1'
REPORT=PARENT+'/analysis/ours_only_results.json'
REPORT_SHA='616b9e0b7ee215c90d768322498440179edbf32d4c4ea3cdc6447ad205914b0d'
PARENT_FP='803ebb48aa10b3f9fd4fd8a349424fa21959f39b0f0d0b448aa8883cc261cd09'
SCOPE_FP='8a563ad705c130152b72345f3eab73677124af5b1c82d7d2bfcdb83d1c8e7564'
FAILED_PAIR='R2C125_iid_r40_s2026102803'
DESIGN=dict(new_training_tasks=90,reused_h84_tasks=30,comparison_cells=120,required_prefix_pairs=96,
    maximum_additional_rounds=13500,additional_integrity_gate='Each new task r0-20 classification equals same-seed same-partition H84 clean; no numerical-setting change.',automatic_retry=False,automatic_search_expansion=False,
    original_panel_requalified=False,old_pairing_gate_unchanged=True,
    reserve_scope='Three previously declared reserve arms plus immutable H84 references; first-panel candidates remain ineligible.',
    prior_failure='One first-panel R2C125 prefix mismatch of unknown internal cause is retained; no retry, replacement or tolerance relaxation.',
    selection_gate='All 120 reserve-comparison cells finite and healthy; all 96 reserve-comparison prefixes and execution audits must pass.',
    stopping_rule='No further candidates after these reserves; no automatic official-test launch.')


def gpu_snapshot():return first.gpu_snapshot()


def check_accepted_snapshot():
    if sha(ROOT/'outputs/cifar_h84_formal_vert_paper150_v1/export/combined_bundle.json')!=BASELINE_SHA:
        raise ValueError('Accepted snapshot changed')


def verify(m):
    first.verify(m)
    if 'parent_report_sha256' in m and sha(ROOT/REPORT)!=m['parent_report_sha256']:
        raise ValueError('First-panel report changed')


def plan(config=DEFAULT):
    p=read(config)
    expected=dict(protocol='cifar10_ours_reserve150_v1',output_dir='outputs/cifar_ours_reserve150_v1',
        rounds=150,seeds=list(SEEDS),candidates=CANDIDATES,selection_rule=RULE,design=DESIGN,
        evaluation_scope='training_validation_7500',parent_output_dir=PARENT,parent_report_sha256=REPORT_SHA,
        accepted_baseline_sha256=BASELINE_SHA,process_environment=PROCESS_ENV,
        quarantine_gpu_uuids=list(QUARANTINE),max_gpus=6,
        gpu_admission=dict(min_free_mb=8192,max_utilization=100,consecutive_samples=2,sample_interval_seconds=5,maximum_wait_seconds=86400))
    if set(p)!=set(expected)|{'frozen_source_sha256','parent_metadata_sha256'} or any(p.get(k)!=v for k,v in expected.items()):raise ValueError('Bounded reserve protocol differs')
    old,old_specs,parent,scope=first.plan()
    frozen=dict(parent['sources'],**scope['sources'])
    if p['frozen_source_sha256']!=frozen or len(frozen)!=115:raise ValueError('First-panel 115 source map changed')
    if set(p['parent_metadata_sha256'])!={REPORT,PARENT+'/manifest.json',PARENT+'/reference_lock.json',PARENT+'/ours_only_scope_v1/manifest.json'} or p['parent_metadata_sha256'][REPORT]!=REPORT_SHA:
        raise ValueError('Pinned first-panel metadata missing')
    refs=[deepcopy(s) for s in old_specs if s['arm']=='H84'];specs=[]
    for b in refs:
        for arm,changes in CANDIDATES.items():
            s=deepcopy(b);s['arm']=arm;s['task_id']=s['task_id'].replace('H84_',arm+'_',1);s['config'].update(changes);specs.append(s)
    if len(specs)!=90 or len(refs)!=30:raise ValueError('Reserve coverage differs')
    sources=dict(frozen)
    for n in set(OWN)|{str(Path(config).resolve().relative_to(ROOT))}:sources[n]=sha(ROOT/n)
    runtime=dict(p,numerics=old['numerics'],expected_dataset_identity=old['expected_dataset_identity'])
    identity=dict(protocol=p,specs=specs,sources=sources,data_config=parent['data_config'],
        reused_h84_specs=refs,parent_report_sha256=REPORT_SHA,parent_worker_fingerprint=PARENT_FP,
        parent_scope_fingerprint=SCOPE_FP,prior_failed_pair=FAILED_PAIR,design=DESIGN,
        maximum_rounds=13500,official_test_used_for_selection=False,prior_test_results_seen=True,
        same_previously_reported_seeds_reused=True,evaluation_scope='training_validation_7500')
    return runtime,specs,dict(identity,fingerprint=digest(identity))


def validate_parent_report(r,refs):
    if (r['fingerprint']!=PARENT_FP or r['scope_fingerprint']!=SCOPE_FP or
        r['status']!='complete_audited_validation' or r['execution_audited'] is not True or
        r['evaluation_scope']!='training_validation_7500' or r['official_test_used_for_selection'] is not False or
        r['decision']['selected'] is not None or len(r['cells'])!=120 or len(r['attack_pairing'])!=96 or
        any(not c['complete_finite'] for c in r['cells'].values())):
        raise ValueError('Audited, unsuccessful first panel required')
    failed={n:v for n,v in r['attack_pairing'].items() if not (v['classification_prefix_equal'] and v['round24_model_equal'])}
    if failed!={FAILED_PAIR:dict(classification_prefix_equal=False,round24_model_equal=False)}:
        raise ValueError('Known first-panel prefix failure was removed or changed')
    names={s['task_id'] for s in refs}
    if names!={n for n,c in r['cells'].items() if c['arm']=='H84'} or len(names)!=30:
        raise ValueError('All thirty immutable H84 references required')
    for s in refs:
        n=s['task_id'];c=r['cells'][n]
        if c['config']!=s['config'] or not c['health']['complete_observed_health_pass'] or n not in r['execution_audits']:
            raise ValueError('Unusable H84 reference')
        if s['config']['malicious_ratio'] and not all(r['attack_pairing'][n][k] for k in ('classification_prefix_equal','round24_model_equal')):
            raise ValueError('H84 prefix failed')
    return r


def parent_evidence(p,m,full=True):
    # No summary/finish call is made against the old output: every byte stays frozen.
    for n,h in p['parent_metadata_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Parent metadata changed: '+n)
    r=validate_parent_report(read(ROOT/REPORT),m['reused_h84_specs'])
    old=read(ROOT/PARENT/'manifest.json');scope=read(ROOT/PARENT/'ours_only_scope_v1/manifest.json')
    if old['fingerprint']!=PARENT_FP or scope['fingerprint']!=SCOPE_FP or dict(old['sources'],**scope['sources'])!=p['frozen_source_sha256']:
        raise ValueError('Parent sources or identity differ')
    hashes=dict(r['input_sha256'],**p['parent_metadata_sha256'])
    if full:
        for n,h in hashes.items():
            if sha(ROOT/n)!=h:raise ValueError('First-panel evidence changed: '+n)
    return r,hashes


def summarize(p,specs,m,out):
    previous,_=parent_evidence(p,m,full=False)
    # Frozen single-seed reader is explicitly restricted per seed by legacy.summarize.
    r=first.legacy.summarize(p,specs,m,out)
    names={s['task_id'] for s in m['reused_h84_specs']}
    r['cells'].update({n:deepcopy(c) for n,c in previous['cells'].items() if n in names})
    r['attack_pairing'].update({n:deepcopy(v) for n,v in previous['attack_pairing'].items() if n in names})
    r.update(design=DESIGN,expected_new_tasks=90,reused_h84_tasks=30,
        prior_panel=dict(report_sha256=REPORT_SHA,selected=None,decision=previous['decision'],
            failed_pairs={FAILED_PAIR:previous['attack_pairing'][FAILED_PAIR]},original_pairing_gate_passed=False,
            execution_summary=deepcopy(previous['execution_summary'])),
        final_other_five_source_sha256=BASELINE_SHA,official_test_used_for_selection=False)
    write(out/'analysis/reserve_progress.json',r)
    return r


def controller_alive(state):
    pid=state.get('pid')
    if not isinstance(pid,int) or isinstance(pid,bool) or pid<=0:return False
    try:os.kill(pid,0)
    except ProcessLookupError:return False
    except PermissionError:return True
    return True


def require_finished_controller(out):
    state=read(out/'controller_status.json',{})
    if state.get('status')!='panel_terminal' or state.get('active') or state.get('pending') or state.get('failure'):
        raise ValueError('All workers must be terminal before derived-report repair')
    if controller_alive(state):raise ValueError('Controller still alive; it may be auditing, do not finish twice')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT);parser.add_argument('--max-gpus',type=int,default=6)
    g=parser.add_mutually_exclusive_group()
    for f in ('ready','batch','summary','finish'):g.add_argument('--'+f,action='store_true')
    g.add_argument('--task');args=parser.parse_args()
    if not 1<=args.max_gpus<=6:raise ValueError('At most six GPUs')
    os.environ.update(PROCESS_ENV)
    p,specs,m=plan(args.config);out=ROOT/p['output_dir']
    if args.task or args.ready or args.batch:first.legacy.require_station()
    if args.task:
        verify(m)
        if read(out/'manifest.json')!=m:raise ValueError('Frozen reserve manifest required')
        lock=read(out/'reference_lock.json')
        from run_cifar_vert_kmeans_supplement import check_selector_environment
        if first.legacy.validation.runtime_environment()!=lock['runtime_environment'] or check_selector_environment()!=lock['selector_environment']:
            raise ValueError('Worker software/process environment differs')
        s=next(s for s in specs if s['task_id']==args.task);folder=out/'tasks'/args.task
        from cifar_vert_paper_core import bounded_checkpoints,actual_weights_scope
        with bounded_checkpoints(150),actual_weights_scope(folder/'actual_weights.jsonl',args.task):
            code=first.legacy.base.run_worker(p,s,m,out)
        raise SystemExit(code)
    if args.ready or args.batch:
        verify(m);previous,hashes=parent_evidence(p,m)
        lock=first.legacy.validation.ready(p,m);lock['accepted_baseline_sha256']=BASELINE_SHA
        if lock!=read(ROOT/PARENT/'reference_lock.json'):raise ValueError('Parent data/software lock differs')
        check_accepted_snapshot()
        state=read(ROOT/PARENT/'controller_status.json',{})
        if state.get('status')!='panel_terminal' or state.get('active') or state.get('pending') or state.get('failure') or controller_alive(state):
            raise ValueError('First controller must be finished and exited')
        if args.ready:
            print(__import__('json').dumps(dict(ready=True,new_tasks=90,reused_h84=30,source_count=len(m['sources']),fingerprint=m['fingerprint'],prior_failure_retained=FAILED_PAIR)));return
        from run_cifar_w85_history_admission import prepare
        prepare(out,m,lock)
        from cifar_ours_reserve_scheduler import run
        run(args.config,p,specs,m,out,args.max_gpus)
        if read(out/'controller_status.json')['status']!='panel_terminal':return
    if args.summary:print(__import__('json').dumps(summarize(p,specs,m,out),indent=2));return
    if args.batch or args.finish:
        if args.finish:require_finished_controller(out)
        verify(m)
        if read(out/'manifest.json')!=m:raise ValueError('Frozen reserve manifest required')
        from cifar_ours_reserve_report import finish
        print(__import__('json').dumps(finish(p,specs,m,out),indent=2));return
    print(__import__('json').dumps(dict(new_tasks=90,reused_h84=30,maximum_new_rounds=13500,fingerprint=m['fingerprint'],source_count=len(m['sources']),prior_failure_retained=FAILED_PAIR,training_started=False)))


if __name__=='__main__':main()
