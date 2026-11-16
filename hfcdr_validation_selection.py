"""Bounded, method-independent validation selection; never reads test results."""
from __future__ import annotations
from statistics import mean
from pathlib import Path
import hfcdr_validation_protocol as p
from hfcdr_validation_analysis import report


SPEC=p.ROOT/'configs/hfcdr_validation_defense_v1.json'
OUTPUT=p.ROOT/'outputs/hfcdr_validation_defense_v1'


def verify_selection_sources(spec):
    sources=spec.get('selection_sources')
    if not sources:raise ValueError('Selection procedure must be frozen before candidate training')
    for name,expected in sources.items():
        path=(p.ROOT/name).resolve()
        if p.ROOT not in path.parents or p.sha(path)!=expected:
            raise ValueError('Frozen selection source differs: '+name)


def attack_gate(attack_output,spec,*,full_audit=True):
    """A successful process exit alone is never an attack-effectiveness gate."""
    attack_output=p.safe_output(attack_output);p.check(attack_output)
    manifest=p.read(attack_output/'manifest.json')
    if manifest['fingerprint']!=spec['entry_gate']['attack_manifest_fingerprint']:
        raise ValueError('Wrong upstream attack study')
    controller=p.read(attack_output/'controller.json')
    if controller.get('status')!='complete' or controller.get('active'):
        raise ValueError('Attack screening must finish all tasks before defense search')
    assessment=report(attack_output,manifest)
    if not assessment['complete'] or not assessment['next_stage_allowed']:
        raise ValueError('FedAvg attack-effectiveness gate failed; defense/formal training held')
    if full_audit:
        from audit_hfcdr_validation import audit
        evidence=audit(attack_output,full=True)
        if not evidence['all_results_complete'] or evidence['health_failures']:
            raise ValueError('Attack screening integrity/health gate failed')
    return dict(attack_fingerprint=manifest['fingerprint'],report_sha256=p.digest(assessment),
        attack_decisions=assessment['attack_decisions'],all_pass=True)


def select(attack_output,defense_output):
    defense_output=p.safe_output(defense_output);p.check(defense_output)
    manifest=p.read(defense_output/'manifest.json');spec=manifest['policy']
    verify_selection_sources(spec);upstream=attack_gate(attack_output,spec)
    if p.read(defense_output/'entry_gate.json')!=upstream:
        raise ValueError('Upstream evidence changed after defense launch')
    if p.read(defense_output/'dataset_identity.json')!=p.read(Path(attack_output)/'dataset_identity.json'):
        raise ValueError('Defense and FedAvg validation data differ')
    ap=p.read(Path(attack_output)/'pairing.json')
    if any(ap[k]!=v for k,v in p.read(defense_output/'pairing.json').items()):
        raise ValueError('Defense and FedAvg client/attacker identities differ')
    from audit_hfcdr_validation import audit
    evidence=audit(defense_output,full=True)
    result=report(defense_output,manifest)
    if not result['complete'] or not evidence['all_results_complete']:
        raise ValueError('All candidate runs are required; no dropping failed trials')
    am=p.read(Path(attack_output)/'manifest.json');ar=report(Path(attack_output),am)
    avg_clean={(s['config']['dataset'],s['config']['partition'],s['config']['seed']):s
        for s in ar['summaries'].values() if s['config']['attack']=='none'}
    decisions=[];rules=spec['selection'];summaries=result['summaries']
    for method,candidates in spec['defense_candidates'].items():
        fixed=len(candidates)==1
        for dataset in spec['datasets']:
            candidate_rows=[]
            for index,candidate in enumerate(candidates):
                ts=[t for t in manifest['tasks'] if t['config']['method']==method and
                    t['config']['dataset']==dataset and t['candidate_id']==candidate['id']]
                ss=[summaries[t['task_id']] for t in ts];reasons=[]
                expected=2*len(spec['seeds'])*(1+(len(spec['ratios'])-1)*len(spec['attacks']))
                if len(ss)!=expected:raise ValueError('Candidate received an unequal scenario budget')
                for s in ss:
                    if not s['healthy']:reasons.append(dict(task_id=s['task_id'],reason='health',details=s['health_reasons']))
                    c=s['config']
                    if c['attack']=='none':
                        reference=avg_clean[(dataset,c['partition'],c['seed'])]['tail10']
                        if reference['accuracy']-s['tail10']['accuracy']>rules['clean_max_accuracy_drop_vs_paired_fedavg']:
                            reasons.append(dict(task_id=s['task_id'],reason='clean_accuracy_drop'))
                        if s['tail10']['asr']-reference['asr']>rules['clean_max_asr_increase_vs_paired_fedavg']:
                            reasons.append(dict(task_id=s['task_id'],reason='clean_background_asr'))
                attacked=[s for s in ss if s['config']['attack']!='none']
                score=mean(.5*s['tail10']['accuracy']+.5*(1-s['tail10']['asr']) for s in attacked)
                candidate_rows.append(dict(id=candidate['id'],index=index,full_runs=len(ss),attacked_runs=len(attacked),
                    score=score,feasible=not reasons,reasons=reasons))
            eligible=[c for c in candidate_rows if c['feasible']]
            choice=candidates[0]['id'] if fixed else max(eligible,key=lambda c:(c['score'],-c['index']))['id'] if eligible else None
            decisions.append(dict(method=method,dataset=dataset,fixed_no_search=fixed,selected=choice,
                candidates=candidate_rows,fixed_failure_retained=fixed and not candidate_rows[0]['feasible']))
    return dict(schema='hfcdr_defense_selection_v1',attack_fingerprint=upstream['attack_fingerprint'],
        defense_fingerprint=manifest['fingerprint'],decisions=decisions,
        all_tunable_methods_have_feasible_choice=all(d['selected'] is not None for d in decisions),
        confirmation_allowed=all(d['selected'] is not None for d in decisions),
        official_test_used=False,unused_budget_transferred=False,automatic_search_expansion=False,
        scope='One frozen configuration per method/dataset across all attack settings and partitions')
