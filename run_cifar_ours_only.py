"""User-directed Ours-only execution amendment; frozen worker identities are reused."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import run_cifar_ours_paper_refinement as legacy
from run_cifar_cp1_cuda_development import ROOT, TERMINAL, read, write, sha, digest, utc, verify, qualified_gpus, guard_task_directory

DEFAULT=ROOT/'configs/cifar10_ours_only150_v1.json'
OWN=('run_cifar_ours_only.py','cifar_ours_only_scheduler.py','cifar_ours_only_report.py','tests/test_ours_only.py')
SEEDS=legacy.SEEDS
CANDIDATES=legacy.CANDIDATES
ARMS=('H84',*CANDIDATES)
QUARANTINE=legacy.QUARANTINE
RULE={k:v for k,v in legacy.RULE.items() if k not in ('clean_acc_drop_vs_fedavg_max','iid_mean_gap_to_vert_goal','iid_worst_gap_to_vert_goal')}
RULE.update(comparison_reference='paired H84 on validation only; other five methods reused only in final official-test comparison',
            scope_change_reason='User explicitly requested Ours-only execution before any candidate completed the full matrix')
PROCESS_ENV=legacy.PROCESS_ENV
BASELINE_SHA=legacy.BASELINE_SHA
SCOPE_DIR='ours_only_scope_v1'


def gpu_snapshot():return legacy.gpu_snapshot()


def plan(config=DEFAULT):
    a=read(config)
    expected=dict(protocol='cifar10_ours_only150_v1',parent_config='configs/cifar10_ours_paper_refinement150_v1.json',
        permitted_arms=list(ARMS),selection_rule=RULE,worker_identity_unchanged=True,
        other_methods_retrained=False,accepted_baseline_sha256=BASELINE_SHA,scope_dir=SCOPE_DIR)
    if any(a.get(k)!=v for k,v in expected.items()):raise ValueError('Ours-only scope amendment differs')
    if sha(ROOT/a['parent_config'])!=a['parent_config_sha256']:raise ValueError('Frozen parent configuration changed')
    p,all_specs,m=legacy.plan(ROOT/a['parent_config'])
    specs=[s for s in all_specs if s['arm'] in ARMS]
    if len(specs)!=120 or any(s['config']['method']!='sm9rrs' for s in specs):raise ValueError('Exactly 120 Ours-only identities required')
    sources={n:sha(ROOT/n) for n in set(OWN)|{str(Path(config).resolve().relative_to(ROOT))}}
    identity=dict(amendment=a,parent_fingerprint=m['fingerprint'],sources=sources,
        parent_source_count=len(m['sources']),tasks=[s['task_id'] for s in specs],
        omitted_from_future_dispatch=[s['task_id'] for s in all_specs if s['arm'] not in ARMS],
        maximum_ours_rounds=18000,scope_selected_from_user_instruction_not_results=True,
        evaluation_scope='training_validation_7500',official_test_used_for_selection=False,
        final_comparison='fixed selected Ours official-test30 + accepted other-five official-test150')
    scope=dict(identity,fingerprint=digest(identity))
    p=deepcopy(p);p['scope_manifest']=scope;p['scope_sources']=sources
    p['legacy_worker_config']=str((ROOT/a['parent_config']).resolve())
    return p,specs,m,scope


def verify_scope(scope,m):
    verify(m)
    if scope['parent_fingerprint']!=m['fingerprint']:raise ValueError('Scope parent identity differs')
    for n,h in scope['sources'].items():
        if sha(ROOT/n)!=h:raise ValueError('Active amendment source changed: '+n)


def prepare_scope(out,m,scope):
    """Waiter invokes this only after the old controller drains. Never migrates tasks."""
    import fcntl
    if read(out/'manifest.json')!=m:raise ValueError('Original immutable worker manifest required')
    with (out/'controller.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(out/'controller_status.json',{})
        if state.get('active') or state.get('failure'):raise ValueError('Existing active workers/failure must be reviewed')
        pid=state.get('pid')
        if pid and pid!=os.getpid() and Path('/proc',str(pid)).exists():raise ValueError('Existing controller is still alive')
        if state.get('status') not in ('stopped_after_drain','panel_terminal'):raise ValueError('Cleanly drained controller required')
        folder=out/SCOPE_DIR;old=read(folder/'manifest.json')
        if old is not None:
            if old!=scope:raise ValueError('Existing Ours-only scope differs')
            verify_scope(scope,m);return
        known={s['task_id']:s for s in m['specs']};states={}
        for path in (out/'tasks').glob('*'):
            if not path.is_dir():continue
            if path.name not in known:raise ValueError('Unknown prior task retained')
            status=guard_task_directory(path)
            if status!='complete':raise ValueError('No partial/failed/numerical task is retried')
            states[path.name]=dict(status=status,task_identity_sha256=sha(path/'task.json'))
        verify_scope(scope,m)
        write(folder/'prior_controller_status.json',state)
        write(folder/'transition.json',dict(utc=utc(),reason='User explicitly requests Ours-only tasks and reuse of accepted other methods',
            initial_task_states=states,retained_ours=[n for n in states if n in scope['tasks']],
            excluded_completed=[n for n in states if n in scope['omitted_from_future_dispatch']],
            official_test_not_used_for_scope_change=True,workers_terminated=False,original_manifest_unchanged=True))
        write(folder/'manifest.json',scope)


def summarize(p,specs,m,out):
    report=legacy.summarize(p,specs,m,out)
    report.update(scope_fingerprint=p['scope_manifest']['fingerprint'],expected_ours_cells=120,
        original_manifest_planned_cells=180,excluded_methods_are_not_missing_ours_results=True,
        final_other_five_source_sha256=BASELINE_SHA)
    write(out/'analysis/ours_only_progress.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT);parser.add_argument('--max-gpus',type=int,default=6)
    g=parser.add_mutually_exclusive_group()
    for f in ('ready','batch','summary','finish'):g.add_argument('--'+f,action='store_true')
    args=parser.parse_args()
    if not 1<=args.max_gpus<=6:raise ValueError('At most six non-quarantined GPUs')
    os.environ.update(PROCESS_ENV)
    p,specs,m,scope=plan(args.config);out=ROOT/p['output_dir'];verify_scope(scope,m)
    if args.ready or args.batch:
        legacy.require_station()
        lock=legacy.validation.ready(p,m);lock['accepted_baseline_sha256']=BASELINE_SHA
        if read(out/'reference_lock.json')!=lock:raise ValueError('Original validation data/software lock differs')
        if sha(ROOT/'outputs/cifar_h84_formal_vert_paper150_v1/export/combined_bundle.json')!=BASELINE_SHA:raise ValueError('Accepted baseline changed')
        if args.ready:
            print(json.dumps(dict(ready=True,tasks=120,worker_source_count=len(m['sources']),amendment_sources=len(scope['sources']),fingerprint=scope['fingerprint'])));return
        prepare_scope(out,m,scope)
        from cifar_ours_only_scheduler import run
        run(args.config,p,specs,m,out,args.max_gpus)
        if read(out/'controller_status.json')['status']!='panel_terminal':return
    if args.summary:print(json.dumps(summarize(p,specs,m,out),indent=2));return
    if args.batch or args.finish:
        if read(out/SCOPE_DIR/'manifest.json')!=scope:raise ValueError('Frozen Ours-only scope required')
        from cifar_ours_only_report import finish
        print(json.dumps(finish(p,specs,m,out),indent=2));return
    print(json.dumps(dict(ours_tasks=120,maximum_ours_rounds=18000,scope_fingerprint=scope['fingerprint'],parent_fingerprint=m['fingerprint'],training_started=False)))


if __name__=='__main__':main()
