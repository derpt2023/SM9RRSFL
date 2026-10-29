"""AI Station only: bounded parameter refinement of the unchanged Ours paper method."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import run_cifar_cp1_cuda_development as base
import run_cifar_vert_paper_validation as validation
from run_cifar_cp1_cuda_development import ROOT, TERMINAL, read, write, sha, digest, utc, verify, qualified_gpus, guard_task_directory

DEFAULT = ROOT/'configs/cifar10_ours_paper_refinement150_v1.json'
OWN = ('run_cifar_ours_paper_refinement.py','cifar_ours_paper_refinement_report.py',
       'cifar_ours_paper_refinement_scheduler.py','tests/test_ours_paper_refinement.py')
SEEDS = (2026102801,2026102802,2026102803)
CANDIDATES = {'R2': {'suspicion_recovery_factor':2.0},
              'C125': {'detector_clip_factor':1.25},
              'R2C125': {'suspicion_recovery_factor':2.0,'detector_clip_factor':1.25}}
# Additional hypotheses, only in a separately frozen successor panel if needed.
RESERVE = {'R4': {'suspicion_recovery_factor':4.0},
           'C150': {'detector_clip_factor':1.5},
           'R2C150': {'suspicion_recovery_factor':2.0,'detector_clip_factor':1.5}}
ARMS = ('H84','FedAvg','VERT',*CANDIDATES)
QUARANTINE = validation.QUARANTINE
PROCESS_ENV = validation.PROCESS_ENV
RULE = dict(non_iid_tail_acc_drop_max=.01,non_iid_tail_asr_rise_max=.02,
            non_iid_attack_period_acc_drop_max=.01,non_iid_attack_period_asr_rise_max=.02,
            iid_tail_asr_rise_max=.02,iid_clean_acc_drop_max=.005,
            clean_acc_drop_vs_fedavg_max=.05,iid_mean_acc_gain_min=.005,
            iid_mean_gap_to_vert_goal=.01,iid_worst_gap_to_vert_goal=.02,
            all_seeds_and_scenarios=True,complete_observed_health_required=True,
            ranking_target=None,selection_metric='maximum IID tail10 mean accuracy subject to all guardrails',
            tie_break='minimum worst Non-IID tail ASR rise, then candidate name',
            interpretation='Empirical paired validation guardrails, not a statistical guarantee or significance test')
BASELINE_SHA = '2fd0d9cdad96b9cfacb6ed69da0b482dce4bfab537c118e03b9111810714f1a1'
PAPER = dict(fifth_sha256='7902b5ab2817f826ed635eaaaefd9461211d8775ee4def3a1de2816aecd17318',
             sixth_sha256='002fbb514b2eb414ade2bdd4ce27d39c0c979e97344937143e6494487c100ace',
             section='4.3.3 steps 3 and 4; identical core in fifth and sixth manuscripts',
             free_parameters=['r_omega > 1: recovery factor','c_n: norm clipping multiplier'],
             immutable_mechanisms=['suspicious update isolated in same round','frozen clean reference and norm bound',
                'bounded trusted-history admission and drift','recovery eligibility and confirmation',
                'reliability <= 1','absolute coefficient cap','no renormalization after clipping/capping',
                'valid threshold certificate before permanent revocation','SM9 protocol unchanged'],
             protocol_difference='Accepted 150-round V2 attack, split and seeds retained by user request; not the draft 100-round experimental schedule')


def gpu_snapshot():
    return [g for g in base.gpu_snapshot() if g['uuid'] not in QUARANTINE]


def require_station():
    if sys.platform!='linux' or ROOT!=Path('/3251901002/SM9RRSFL'):
        raise RuntimeError('Training and CUDA readiness checks are restricted to the authorized AI Station')


def plan(config=DEFAULT):
    p=read(config)
    required=dict(protocol='cifar10_ours_paper_refinement150_v1',rounds=150,seeds=list(SEEDS),
        evaluation_scope='training_validation_7500',output_dir='outputs/cifar_ours_paper_refinement150_v1',
        candidates=CANDIDATES,reserve_candidates=RESERVE,selection_rule=RULE,paper_contract=PAPER,
        accepted_baseline_sha256=BASELINE_SHA,process_environment=PROCESS_ENV,
        quarantine_gpu_uuids=list(QUARANTINE),automatic_retry=False,maximum_candidate_count=6,
        automatic_search_expansion=False, max_gpus=6, checkpoint_rounds=[0,24,150],
        gpu_admission=dict(min_free_mb=8192,max_utilization=100,consecutive_samples=2,
            sample_interval_seconds=5,maximum_wait_seconds=86400))
    if any(p.get(k)!=v for k,v in required.items()):raise ValueError('Frozen refinement protocol differs')
    from run_cifar_vert_paper_formal import plan as paper_plan
    from run_cifar_h84_formal import plan as original_plan
    _,vs,old=paper_plan();_,hs,_=original_plan()
    parent=read(ROOT/'configs/cifar10_h84_full_validation150_v1.json')
    by_key={(s['config']['seed'],s['config']['partition'],s['config']['malicious_ratio'],s['arm']):s
            for s in hs if s['arm'] in ('H84','FedAvg')}
    by_key.update({(s['config']['seed'],s['config']['partition'],s['config']['malicious_ratio'],'VERT'):s for s in vs})
    specs=[]
    # Interleave paired controls/candidates; no score-dependent ordering or pruning.
    for seed in SEEDS:
        for part in ('iid','dirichlet'):
            for ratio in (0.,.2,.4,.6,.8):
                for arm in ARMS:
                    s=deepcopy(by_key[(seed,part,ratio,arm if arm in ('H84','FedAvg','VERT') else 'H84')])
                    s['arm']=arm;s['task_id']=s['task_id'].replace(s['task_id'].split('_')[0],arm,1)
                    s['config'].update(CANDIDATES.get(arm,{}));specs.append(s)
    sources=dict(old['sources'])
    for name in set(OWN)|{str(Path(config).resolve().relative_to(ROOT))}:sources[name]=sha(ROOT/name)
    runtime=dict(p,numerics=parent['numerics'],expected_dataset_identity=parent['expected_dataset_identity'])
    identity=dict(protocol=p,specs=specs,sources=sources,data_config=old['data_config'],
        maximum_rounds=27000,official_test_used_for_selection=False,prior_test_results_seen=True,
        same_previously_reported_seeds_reused=True,evaluation_scope='training_validation_7500',
        original_algorithm_source_modified=False,accepted_baseline_replacement=False)
    return runtime,specs,dict(identity,fingerprint=digest(identity))


def summarize(p,specs,m,out):
    # The original single-seed reader pairs only by arm/distribution. Restrict each
    # call to one seed before merging, preventing any cross-seed prefix comparison.
    reports=[base.summarize(p,[s for s in specs if s['config']['seed']==seed],m,out) for seed in SEEDS]
    report=dict(fingerprint=m['fingerprint'],evaluation_scope='training_validation_7500',
        official_test_used_for_selection=False,prior_test_results_seen=True,
        status='complete_panel' if all(r['status']=='complete_panel' for r in reports) else 'incomplete')
    for key in ('cells','attack_pairing','fedavg_attack_gates','input_sha256'):
        report[key]={k:v for r in reports for k,v in r[key].items()}
    write(out/'analysis/development_summary.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT);parser.add_argument('--max-gpus',type=int,default=6)
    group=parser.add_mutually_exclusive_group()
    for f in ('ready','batch','summary','finish'):group.add_argument('--'+f,action='store_true')
    group.add_argument('--task');args=parser.parse_args()
    if not 1<=args.max_gpus<=6:raise ValueError('At most six non-quarantined GPUs')
    os.environ.update(PROCESS_ENV)
    p,specs,m=plan(args.config);out=ROOT/p['output_dir']
    if args.task or args.ready or args.batch:require_station()
    if args.task:
        if read(out/'manifest.json')!=m:raise ValueError('Existing frozen manifest required')
        lock=read(out/'reference_lock.json')
        from run_cifar_vert_kmeans_supplement import check_selector_environment
        if validation.runtime_environment()!=lock['runtime_environment'] or check_selector_environment()!=lock['selector_environment']:
            raise ValueError('Software/process environment differs')
        s=next(s for s in specs if s['task_id']==args.task);folder=out/'tasks'/args.task
        from cifar_vert_paper_core import paper_scope,bounded_checkpoints,actual_weights_scope
        with (paper_scope(folder/'vert_selection.jsonl',args.task) if s['arm']=='VERT' else nullcontext()),bounded_checkpoints(150),actual_weights_scope(folder/'actual_weights.jsonl',args.task):
            code=base.run_worker(p,s,m,out)
        raise SystemExit(code)
    if args.summary:print(json.dumps(summarize(p,specs,m,out),indent=2));return
    if args.ready or args.batch:
        verify(m);lock=validation.ready(p,m)
        if sha(ROOT/'outputs/cifar_h84_formal_vert_paper150_v1/export/combined_bundle.json')!=BASELINE_SHA:
            raise ValueError('Accepted six-method baseline changed')
        lock['accepted_baseline_sha256']=BASELINE_SHA
        if args.ready:
            print(json.dumps(dict(ready=True,tasks=len(specs),source_count=len(m['sources']),fingerprint=m['fingerprint'])));return
        from run_cifar_w85_history_admission import prepare
        prepare(out,m,lock)
        from cifar_ours_paper_refinement_scheduler import run
        run(args.config,p,specs,m,out,args.max_gpus)
        if read(out/'controller_status.json')['status']!='panel_terminal':return
    if args.batch or args.finish:
        if read(out/'manifest.json')!=m:raise ValueError('Frozen manifest required')
        from cifar_ours_paper_refinement_report import finish
        print(json.dumps(finish(p,specs,m,out),indent=2));return
    print(json.dumps(dict(tasks=len(specs),maximum_rounds=m['maximum_rounds'],fingerprint=m['fingerprint'],source_count=len(m['sources']),training_started=False)))


if __name__=='__main__':main()
