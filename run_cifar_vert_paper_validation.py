"""Bounded paper-first VERT validation. Never select on official test or rank."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import sys

import run_cifar_cp1_cuda_development as base
from run_cifar_cp1_cuda_development import ROOT, TERMINAL, read, write, sha, digest, utc, verify, qualified_gpus, guard_task_directory

DEFAULT = ROOT/'configs/cifar10_vert_paper_validation150_v1.json'
OWN = ('run_cifar_vert_paper_validation.py', 'cifar_vert_paper_core.py',
       'cifar_vert_paper_scheduler.py', 'tests/test_vert_paper_validation.py')
QUARANTINE = ('GPU-430587b5-b622-2d06-0bb1-0a2e4e4b4c7c',
              'GPU-5d9f949a-f126-67b3-5c96-af785e33aa5c')
ARMS = {'FedAvg': ('fedavg', None), 'CurrentVERT': ('vert', .001), 'PaperVERT': ('vert', .01)}
RULE = dict(clean_accuracy_drop_limit=.05, mean_attack_accuracy_vs_fedavg_min=0.,
            mean_attack_asr_improvement_vs_fedavg_min=.05,
            mean_attack_asr_improvement_vs_current_min=.05,
            mean_attack_accuracy_vs_current_min=-.01,
            all_cells_complete_finite=True, all_pairs_equal=True,
            all_fedavg_attack_gates=True, ranking_target=None,
            scope='validation_only; clear improvement is a descriptive selection rule, not significance')


def gpu_snapshot():
    return [g for g in base.gpu_snapshot() if g['uuid'] not in QUARANTINE]


def plan(config=DEFAULT):
    from run_cifar_vert_kmeans_supplement import plan as old_plan
    p = read(config)
    parent = read(ROOT/p['parent_config'])
    if sha(ROOT/p['parent_config']) != p['parent_config_sha256']:
        raise ValueError('Frozen common experiment parameters changed')
    if p['selection_rule'] != RULE or p['quarantine_gpu_uuids'] != list(QUARANTINE):
        raise ValueError('Predeclared selection/resource rules changed')
    probe = p['phase'] == 'resource_probe'
    if p['phase'] not in ('resource_probe', 'paper_first_validation'):
        raise ValueError('No implicit additional search or formal evaluation')
    rounds, seed = (12, 2026102990) if probe else (150, 2026102901)
    if p['rounds'] != rounds or p['seed'] != seed or p['evaluation_scope'] != 'training_validation_7500':
        raise ValueError('Validation-only seed/horizon/scope differs')
    expected_out = 'outputs/cifar_vert_paper_' + ('resource_probe12_v1' if probe else 'validation150_v1')
    if p['output_dir'] != expected_out:
        raise ValueError('Independent declared output required')
    specs = []
    cases = [('iid', 0.)] if probe else [(part, ratio) for part in ('iid','dirichlet') for ratio in (0.,.2,.4,.6,.8)]
    for part, ratio in cases:
        for arm in (('PaperVERT',) if probe else ARMS):
            method, lr = ARMS[arm]
            cfg = dict(parent['base_configs'][method], seed=seed, rounds=rounds,
                       partition=part, malicious_ratio=ratio, device='cuda:0')
            if method == 'vert':
                cfg.update(vert_top_k=0, vert_use_ratio_prior=False,
                           vert_projection_dim=256, vert_history_window=10,
                           vert_predict_epochs=5, vert_predict_lr=lr)
            tag = 'clean' if ratio == 0 else 'r'+str(round(ratio*100))
            specs.append(dict(task_id=f'{arm}_{part}_{tag}_s{seed}', arm=arm,
                              config=cfg, profile=deepcopy(parent['profile'])))
    _, _, old = old_plan()
    sources = dict(old['sources'])
    names = set(OWN) | {str(Path(config).resolve().relative_to(ROOT)), p['parent_config']}
    sources.update({n:sha(ROOT/n) for n in names})
    runtime = dict(p, numerics=parent['numerics'], expected_dataset_identity=parent['expected_dataset_identity'])
    identity = dict(protocol=p, specs=specs, sources=sources,
        data_config=dict(parent['data_config'], data_dir=str(ROOT/parent['data_config']['data_dir'])),
        maximum_rounds=rounds*len(specs), official_test_used_for_selection=False,
        prior_test_results_seen=True, formal_result_replacement=False)
    return runtime, specs, dict(identity, fingerprint=digest(identity))


def summarize(p, specs, m, out):
    if p['phase'] != 'resource_probe':
        return base.summarize(p, specs, m, out)
    verify(m)
    report = dict(fingerprint=m['fingerprint'], resource_probe_only=True,
                  cells={s['task_id']:read(out/'tasks'/s['task_id']/'status.json', {}) for s in specs})
    write(out/'analysis/resource_progress.json', report)
    return report


def assess(report):
    """No rank, test metric, seed replacement or per-scenario parameter selection."""
    cells = report['cells']
    if len(cells) != 30 or any(not c['complete_finite'] for c in cells.values()):
        return dict(passed=False, reason='All30 complete finite validation cells required; failures retained')
    def cell(arm, part, ratio):
        return next(c for c in cells.values() if c['arm']==arm and
                    c['config']['partition']==part and c['config']['malicious_ratio']==ratio)
    def avg(arm, field):
        return sum(cell(arm,p,r)['tail'][field] for p in ('iid','dirichlet') for r in (.2,.4,.6,.8))/8
    acc, asr = 'accuracy', 'source_transfer_asr'
    values = dict(clean_drops={p:cell('FedAvg',p,0)['tail'][acc]-cell('PaperVERT',p,0)['tail'][acc]
                              for p in ('iid','dirichlet')},
        accuracy_vs_fedavg=avg('PaperVERT',acc)-avg('FedAvg',acc),
        asr_improvement_vs_fedavg=avg('FedAvg',asr)-avg('PaperVERT',asr),
        asr_improvement_vs_current=avg('CurrentVERT',asr)-avg('PaperVERT',asr),
        accuracy_vs_current=avg('PaperVERT',acc)-avg('CurrentVERT',acc))
    checks = dict(clean_utility=all(v<=RULE['clean_accuracy_drop_limit'] for v in values['clean_drops'].values()),
        accuracy_vs_fedavg=values['accuracy_vs_fedavg']>=RULE['mean_attack_accuracy_vs_fedavg_min'],
        asr_vs_fedavg=values['asr_improvement_vs_fedavg']>=RULE['mean_attack_asr_improvement_vs_fedavg_min'],
        asr_vs_current=values['asr_improvement_vs_current']>=RULE['mean_attack_asr_improvement_vs_current_min'],
        accuracy_vs_current=values['accuracy_vs_current']>=RULE['mean_attack_accuracy_vs_current_min'],
        pairing=len(report['attack_pairing'])==24 and all(v['classification_prefix_equal'] and
            v['round24_model_equal'] for v in report['attack_pairing'].values()),
        fedavg_gates=len(report['fedavg_attack_gates'])==8 and all(v['passed'] for v in report['fedavg_attack_gates'].values()))
    return dict(passed=all(checks.values()), checks=checks, values=values, rule=RULE,
                next_step='Freeze paper parameters for three-seed formal evaluation' if all(checks.values())
                else 'Read diagnostics before designing a separate bounded validation search; no automatic expansion')


def finish(p, specs, m, out):
    report = summarize(p, specs, m, out)
    if p['phase']=='resource_probe':
        state = report['cells'][specs[0]['task_id']]
        metrics = read(out/'tasks'/specs[0]['task_id']/'metrics.json', {})
        result = dict(status=state.get('status'), resource_probe_only=True,
                      completed_round=state.get('completed_round'),
                      expected_rounds=12, performance_selection_permitted=False,
                      all_rounds_present=[r['round'] for r in metrics.get('rounds',[])]==list(range(13)))
        write(out/'analysis/resource_result.json', result)
        return result
    if any(c['status'] not in TERMINAL for c in report['cells'].values()):
        raise ValueError('Unfinished workers; do not finish or restart')
    from audit_cifar_cuda_panel import audit_calls
    from cifar_vert_kmeans_selection import audit as selector_audit
    audits, input_hashes, labels, initial = {}, dict(report['input_sha256']), {}, {}
    for s in specs:
        name=s['task_id']; folder=out/'tasks'/name
        if not report['cells'][name]['complete_finite']:
            audits[name]=dict(complete=False);continue
        def load(path, binary=False, lines=False):
            input_hashes[str(path.relative_to(ROOT))]=sha(path)
            raw=path.read_bytes()
            return pickle.loads(raw) if binary else ([json.loads(l) for l in raw.splitlines()] if lines else json.loads(raw))
        result=load(folder/'result.pickle',binary=True)
        if asdict(result.config)!=s['config']: raise ValueError('Result config differs')
        rows=load(folder/'metrics.json')['rounds']; targets=load(folder/'targets.json')
        records=load(folder/'audit.jsonl',lines=True) if (folder/'audit.jsonl').exists() else []
        checks=dict(attack=audit_calls(records,s,targets,result.malicious_clients,rows))
        key=(s['config']['partition'],s['config']['malicious_ratio'])
        ids=sorted(result.malicious_clients)
        if key in labels and ids!=labels[key]:raise ValueError('Unpaired malicious identities')
        labels[key]=ids
        import numpy as np
        with np.load(folder/'anchor_r0.npz') as z: model=z['params'].copy()
        part=s['config']['partition']
        if part in initial and not np.array_equal(model,initial[part]):raise ValueError('Initial models differ')
        initial[part]=model
        if s['config']['method']=='vert':
            decisions=load(folder/'vert_selection.jsonl',lines=True)
            checks['selection']=selector_audit(decisions,name)
            actual_rows=load(folder/'actual_weights.jsonl',lines=True)
            if [r['round'] for r in actual_rows]!=list(range(1,151)):
                raise ValueError('Missing actual aggregation observations')
            for decision,row,observed in zip(decisions,rows[1:],actual_rows):
                actual=observed['weights']
                if any(abs(actual.get(c,0)-decision['weights'].get(c,0))>1e-12 for c in actual.keys()|decision['weights'].keys()):
                    raise ValueError('Logged selection differs from actual aggregation')
                if any(abs(observed[k]-row[k])>1e-12 for k in ('honest_weight_loss','malicious_weight_mass')):
                    raise ValueError('Weight diagnostic differs')
        audits[name]=checks
    for path,h in input_hashes.items():
        if sha(ROOT/path)!=h:raise ValueError('Input changed during audit')
    final=dict(report, execution_audits=audits, input_sha256=input_hashes,
               decision=assess(report), evaluation_scope='training_validation_7500',
               prior_test_results_seen=True, official_test_used_for_selection=False,
               status='complete_audited_validation', updated_utc=utc())
    write(out/'analysis/paper_validation_results.json',final)
    return dict(status=final['status'],decision=final['decision'])


def ready(p,m):
    from scripts.experiments.cifar10 import attack_stability_worker as worker
    from run_cifar_vert_kmeans_supplement import check_selector_environment
    data,contract=worker._load_dataset(m['data_config'])
    if worker._dataset_identity(data,contract)!=p['expected_dataset_identity']:
        raise ValueError('Validation data changed')
    if len(data.y_test)!=7500 or sum(data.y_test==5)!=750:
        raise ValueError('Official test must not be used')
    return dict(dataset_identity=p['expected_dataset_identity'], selector_environment=check_selector_environment(),
                runtime_environment=runtime_environment())


def runtime_environment():
    import torch
    import numpy
    return dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), numpy=numpy.__version__,
                gpu=torch.cuda.get_device_name(0),
                native_binary_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'sm9rrsfl').glob('_native_*.so')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--max-gpus',type=int,default=6)
    group=parser.add_mutually_exclusive_group()
    for flag in ('batch','ready','summary','finish'):group.add_argument('--'+flag,action='store_true')
    group.add_argument('--task')
    args=parser.parse_args();p,specs,m=plan(args.config);out=ROOT/p['output_dir']
    if args.task:
        if read(out/'manifest.json')!=m:raise ValueError('Frozen study manifest required')
        from run_cifar_vert_kmeans_supplement import check_selector_environment
        lock=read(out/'reference_lock.json')
        if (runtime_environment()!=lock['runtime_environment'] or
                check_selector_environment()!=lock['selector_environment']):
            raise ValueError('Locked runtime or selector environment differs')
        s=next(s for s in specs if s['task_id']==args.task)
        from cifar_vert_paper_core import paper_scope, bounded_checkpoints, actual_weights_scope
        from cifar_vert_kmeans_selection import selection_scope
        path=out/'tasks'/args.task/'vert_selection.jsonl'
        scope=paper_scope(path,args.task) if s['arm']=='PaperVERT' else (
            selection_scope(path,args.task) if s['arm']=='CurrentVERT' else nullcontext())
        with scope,bounded_checkpoints(p['rounds']),actual_weights_scope(path.with_name('actual_weights.jsonl'),args.task):
            code=base.run_worker(p,s,m,out)
        import torch
        write(out/'tasks'/args.task/'resource_usage.json',dict(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            checkpoint_rounds=sorted({r for r in (0,24,p['rounds']) if r<=p['rounds']})))
        raise SystemExit(code)
    if args.summary:
        print(json.dumps(summarize(p,specs,m,out),indent=2));return
    if args.ready or args.batch:
        lock=ready(p,m)
        if args.ready:print(json.dumps(dict(ready=True,tasks=len(specs),**lock)));return
        if p['phase']!='resource_probe':
            probe=read(ROOT/'outputs/cifar_vert_paper_resource_probe12_v1/analysis/resource_result.json',{})
            if probe.get('status')!='complete' or not probe.get('all_rounds_present'):
                raise ValueError('Complete resource probe must precede the panel')
        from run_cifar_w85_history_admission import prepare
        prepare(out,m,lock)
        from cifar_vert_paper_scheduler import run
        run(args.config,p,specs,m,out,args.max_gpus)
        if read(out/'controller_status.json')['status']!='panel_terminal':return
    if args.finish or args.batch:
        if read(out/'manifest.json')!=m:raise ValueError('Existing manifest required')
        print(json.dumps(finish(p,specs,m,out),indent=2));return
    print(json.dumps(dict(tasks=len(specs),maximum_rounds=m['maximum_rounds'],
                         fingerprint=m['fingerprint'],phase=p['phase'],training_started=False)))


if __name__=='__main__':main()
