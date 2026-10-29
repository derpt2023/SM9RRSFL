"""Freeze validation-qualified paper-core VERT for the original three formal seeds."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import json

import run_cifar_vert_kmeans_supplement as previous
import run_cifar_vert_paper_validation as validation
from run_cifar_vert_kmeans_supplement import ROOT, TERMINAL, read, write, sha, digest, utc, verify, qualified_gpus, guard_task_directory

formal = previous.formal
full = previous.full
DEFAULT = ROOT/'configs/cifar10_vert_paper_formal150_v1.json'
OWN = ('run_cifar_vert_paper_formal.py','cifar_vert_paper_formal_scheduler.py',
       'cifar_vert_paper_formal_report.py','tests/test_vert_paper_formal.py')
QUARANTINE = validation.QUARANTINE
PARAMETERS = dict(vert_top_k=0,vert_use_ratio_prior=False,vert_projection_dim=256,
                  vert_history_window=10,vert_predict_epochs=5,vert_predict_lr=.01)
VALIDATION_SHA = '23e23a137f4622b867e4841958c0b961425ce10fc739dbf7d3cacb4b1271417a'


def gpu_snapshot():
    return [g for g in formal.gpu_snapshot() if g['uuid'] not in QUARANTINE]


def selected_validation(amendment, verify_inputs=False):
    path=ROOT/amendment['validation_report']
    if amendment['validation_report_sha256']!=VALIDATION_SHA or sha(path)!=VALIDATION_SHA:
        raise ValueError('Exact completed validation evidence required')
    report=read(path)
    if (report['status']!='complete_audited_validation' or
            report['evaluation_scope']!='training_validation_7500' or
            report['official_test_used_for_selection'] or len(report['execution_audits'])!=30 or
            report['decision']!=validation.assess(report) or not report['decision']['passed']):
        raise ValueError('Prespecified validation selection did not pass')
    if verify_inputs:full.verify_files(report['input_sha256'])
    return report


def plan(config=DEFAULT):
    a=read(config)
    if (a['protocol']!='cifar10_vert_paper_formal150_v1' or a['parameters']!=PARAMETERS or
        a['output_dir']!='outputs/cifar_h84_formal_vert_paper150_v1' or
        a['seeds']!=[2026102801,2026102802,2026102803] or a['rounds']!=150 or
        a['process_environment']!=validation.PROCESS_ENV or a['quarantine_gpu_uuids']!=list(QUARANTINE) or
        a['retune_on_formal_test'] or a['rank_target'] is not None or
        a['parent_config']!='configs/cifar10_vert_kmeans_supplement150_v1.json'):
        raise ValueError('Only the fixed validation-selected three-seed formal protocol is allowed')
    if sha(ROOT/a['parent_config'])!=a['parent_config_sha256']:
        raise ValueError('Original paired protocol changed')
    selected_validation(a)
    for name,h in a['validation_frozen_sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen validation source changed: '+name)
    if len(a['validation_frozen_sources'])!=100:raise ValueError('All100 validation sources required')
    p,specs,old=previous.plan(ROOT/a['parent_config'])
    specs=deepcopy(specs)
    for s in specs:s['config'].update(PARAMETERS)
    p.update(output_dir=a['output_dir'],amendment=a,process_environment=a['process_environment'])
    sources=dict(a['validation_frozen_sources'])
    for name in set(OWN)|{str(Path(config).resolve().relative_to(ROOT))}:sources[name]=sha(ROOT/name)
    identity=dict(protocol=a,specs=specs,data_config=old['data_config'],sources=sources,
        parent_fingerprint=old['parent_fingerprint'],maximum_rounds=4500,
        validation_report_sha256=VALIDATION_SHA,official_test_used_for_selection=False,
        evaluation_scope='official_test',prior_test_results_seen=True,
        formal_results_cannot_change_selected_configuration=True)
    return p,specs,dict(identity,fingerprint=digest(identity))


def references(p,m):
    evidence=selected_validation(p['amendment'],verify_inputs=True)
    lock=previous.references(p,m)
    vpath=ROOT/p['amendment']['validation_report'];vm=vpath.parents[1]/'manifest.json'
    if read(vm)['sources']!=p['amendment']['validation_frozen_sources']:
        raise ValueError('Original validation manifest source map differs')
    if read(vm)['fingerprint']!=evidence['fingerprint']:raise ValueError('Validation identity differs')
    lock['references'].update({str(vpath.relative_to(ROOT)):sha(vpath),str(vm.relative_to(ROOT)):sha(vm)})
    lock['paper_runtime_environment']=validation.runtime_environment()
    lock['selected_validation_report_sha256']=VALIDATION_SHA
    return lock


def summarize(p,specs,m,out):
    return previous.summarize(p,specs,m,out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--max-gpus',type=int,default=6)
    group=parser.add_mutually_exclusive_group()
    for f in ('ready','batch','summary','finish'):group.add_argument('--'+f,action='store_true')
    group.add_argument('--task');args=parser.parse_args()
    if not 1<=args.max_gpus<=6:raise ValueError('At most six non-quarantined GPUs')
    os.environ.update(validation.PROCESS_ENV)
    p,specs,m=plan(args.config);out=ROOT/p['output_dir']
    if args.task:
        if read(out/'manifest.json')!=m:raise ValueError('Frozen formal identity required')
        lock=read(out/'reference_lock.json')
        if (previous.check_selector_environment()!=lock['selector_environment'] or
            validation.runtime_environment()!=lock['paper_runtime_environment']):
            raise ValueError('Frozen selector/software/process environment differs')
        from cifar_vert_paper_core import paper_scope,bounded_checkpoints,actual_weights_scope
        spec=next(s for s in specs if s['task_id']==args.task);folder=out/'tasks'/args.task
        with paper_scope(folder/'vert_selection.jsonl',args.task),bounded_checkpoints(150),actual_weights_scope(folder/'actual_weights.jsonl',args.task):
            code=formal.run_worker(p,spec,m,out)
        raise SystemExit(code)
    if args.summary:print(json.dumps(summarize(p,specs,m,out),indent=2));return
    if args.ready or args.batch:
        lock=references(p,m);full.check_runtime(lock['expected_environment'])
        if args.ready:
            print(json.dumps(dict(ready=True,tasks=30,source_count=len(m['sources']),fingerprint=m['fingerprint'],validation_sha256=VALIDATION_SHA)));return
        from run_cifar_w85_history_admission import prepare
        prepare(out,m,lock)
        from cifar_vert_paper_formal_scheduler import run,shared_parameters
        run(args.config,shared_parameters(p),specs,m,out,args.max_gpus)
        if read(out/'controller_status.json')['status']!='panel_terminal':return
    if args.batch or args.finish:
        if read(out/'manifest.json')!=m:raise ValueError('Existing formal identity required')
        from cifar_vert_paper_formal_report import finish
        print(json.dumps(finish(p,specs,m,out),indent=2));return
    print(json.dumps(dict(tasks=30,rounds=4500,seeds=p['seeds'],source_count=len(m['sources']),fingerprint=m['fingerprint'],training_started=False)))


if __name__=='__main__':main()
