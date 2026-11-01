"""Audit fresh fixed R4, then replace only Ours in the accepted six-method snapshot."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
import pickle
from statistics import mean, stdev
import numpy as np

import run_cifar_r4_formal as base
import cifar_h84_formal_report as report
import run_cifar_h84_full_validation as full

FIELDS=('accuracy','source_transfer_asr','non_source_accuracy','aux128_confidence')


def classification_equal(rows,other,last):
    if [r['round'] for r in rows[:last+1]]!=list(range(last+1)) or [r['round'] for r in other[:last+1]]!=list(range(last+1)):
        raise ValueError('Complete classification prefix required')
    return all(rows[r][k]==other[r][k] for r in range(last+1) for k in FIELDS)


def audit_weights(actual,rows,diagnostics,name):
    if len(rows)!=151 or [r['round'] for r in actual]!=list(range(1,151)):
        raise ValueError('All150 actual weights required')
    grouped={rid:{} for rid in range(1,151)}
    for d in diagnostics:grouped[d.round][d.client_id]=d.aggregation_weight
    for a,r in zip(actual,rows[1:]):
        if a['task_id']!=name or a['round']!=r['round']:raise ValueError('Weight task/round identity differs')
        for key in ('honest_weight_loss','malicious_weight_mass'):
            if not math.isfinite(a[key]) or abs(a[key]-r[key])>1e-9:raise ValueError('Actual mass differs')
        wanted=grouped[r['round']];given=a['weights']
        if any(not math.isfinite(v) or v<0 for v in given.values()):raise ValueError('Invalid actual coefficient')
        if any(abs(wanted.get(k,0)-given.get(k,0))>1e-12 for k in wanted.keys()|given.keys()):
            raise ValueError('Actual aggregation differs from Ours diagnostic')
    return True


def execution(out,specs):
    names={s['task_id'] for s in specs};starts={};ends={};hashes={}
    if len(names)!=30 or any(not n.startswith('R4_') for n in names):raise ValueError('Exactly30 R4 units required')
    state=base.read(out/'controller_status.json',{})
    if state.get('status')!='panel_terminal' or state.get('active') or state.get('pending') or state.get('failure'):
        raise ValueError('Controller must be terminal without error')
    for p in (out/'scheduler_r4_formal_v1').glob('*/events.jsonl'):
        hashes[str(p.relative_to(base.ROOT))]=base.sha(p)
        for e in map(json.loads,p.read_text().splitlines()):
            if e['event']=='worker_started':starts.setdefault(e['task'],[]).append(e)
            elif e['event']=='worker_exited':ends.setdefault(e['task'],[]).append(e)
            elif e['event']=='dispatch_blocked':raise ValueError('Dispatch failure retained')
    if set(starts)!=names or set(ends)!=names:raise ValueError('Exactly30 R4 tasks, no other method, required')
    for name in names:
        if len(starts[name])!=1 or len(ends[name])!=1:raise ValueError('Non-single execution retained')
        s,e=starts[name][0],ends[name][0]
        if (e['returncode']!=0 or e['status']!='complete' or e['child_pid']!=s['child_pid']
                or s['admission']['uuid']!=e['gpu_uuid'] or e['gpu_uuid'] in base.QUARANTINE
                or s['admission']['free_mb']<8192):raise ValueError('Execution/admission identity differs')
    return dict(new_tasks=30,execution_attempts=30,failures=0,automatic_retries=0,
        other_five_training_tasks=0,input_sha256=hashes)


def combine(original,replacements,audits,pairing,warmup,seeds):
    expected={n for n,c in original['cells'].items() if c['arm']=='H84'}
    if set(replacements)!=expected or set(audits)!=expected or len(expected)!=30:
        raise ValueError('All30 new Ours cells required, no old/new fallback')
    if any(not c['complete_finite'] or c['status']!='complete' for c in replacements.values()):
        raise ValueError('Incomplete R4 cannot become a complete comparison')
    required_pairs={t['id'] for t in full.all_tasks() if t['arm']=='H84' and t['ratio']}
    if (tuple(seeds)!=base.SEEDS or set(warmup)!={n.replace('H84_','R4_',1) for n in expected}
            or set(pairing)!={str(s) for s in seeds}
            or any(set(checks)!=required_pairs for checks in pairing.values())):
        raise ValueError('All30 warmup and24 paired audits required')
    for name,c in replacements.items():
        if (c['config']!=dict(original['cells'][name]['config'],**base.PARAMETERS)
                or c.get('candidate')!='R4' or c.get('source_task')!=name.replace('H84_','R4_',1)
                or audits[name]['summary']!=c):
            raise ValueError('Fixed R4 configuration and source identity required')
    result=deepcopy(original)
    result['accepted_baseline_metadata']={k:deepcopy(original.get(k)) for k in ('acceptance','selection_note','selected_validation_report_sha256','replacement_lineage')}
    result['cells'].update(replacements);result['audited_cells'].update(audits)
    result['groups']=report.grouped_statistics(result['cells'],seeds)
    changes={}
    for seed in seeds:
        suffix='_s'+str(seed);old=original['per_seed'][str(seed)]
        pairs=deepcopy(old['pairing']);pairs.update(pairing[str(seed)])
        selected={t['id']:result['audited_cells'][t['id']+suffix] for t in full.all_tasks()}
        strict=full.assess(selected,old['fedavg_attack_gates'],pairs)
        exception=report.with_krum_exception(strict)
        result['per_seed'][str(seed)]=dict(strict=strict,with_previous_krum_exception=exception,
            pairing=pairs,fedavg_attack_gates=deepcopy(old['fedavg_attack_gates']))
        for name in sorted(n for n in expected if n.endswith(suffix)):
            new=result['cells'][name];prior=original['cells'][name];cfg=new['config']
            peers=[c for c in result['cells'].values() if all(c['config'][k]==cfg[k] for k in ('seed','partition','malicious_ratio'))]
            changes[name]=dict(seed=seed,partition=cfg['partition'],malicious_ratio=cfg['malicious_ratio'],
                r4_tail=deepcopy(new['tail']),original_h84_tail=deepcopy(prior['tail']),
                delta_pp={w:{k:100*(new[w][k]-prior[w][k]) for k in ('accuracy','source_transfer_asr')} for w in ('tail','attack_period')},
                iid_or_non_iid_raw_accuracy_gap_pp=100*(max(c['tail']['accuracy'] for c in peers)-new['tail']['accuracy']),
                raw_best_accuracy_methods=[c['arm'] for c in peers if c['tail']['accuracy']==max(x['tail']['accuracy'] for x in peers)])
    for name,c in original['cells'].items():
        if c['arm']!='H84' and (result['cells'][name]!=c or result['audited_cells'][name]!=original['audited_cells'][name]):
            raise ValueError('Other five results/audits must remain identical')
    summary={}
    for part in ('iid','dirichlet'):
        rs=[v for v in changes.values() if v['partition']==part]
        means={arm:mean(c['tail']['accuracy'] for c in result['cells'].values() if c['arm']==arm and c['config']['partition']==part) for arm in full.METHODS}
        seed_accuracy=[mean(v['r4_tail']['accuracy'] for v in rs if v['seed']==s) for s in seeds]
        summary[part]=dict(r4_mean_accuracy=means['H84'],r4_seed_mean_accuracy_sample_sd=stdev(seed_accuracy),n_seeds=len(seeds),
            gap_to_best_method_mean_pp=100*(max(means.values())-means['H84']),
            mean_paired_gap_pp=mean(v['iid_or_non_iid_raw_accuracy_gap_pp'] for v in rs),
            within_1pp=sum(v['iid_or_non_iid_raw_accuracy_gap_pp']<=1+1e-12 for v in rs),
            within_1_5pp=sum(v['iid_or_non_iid_raw_accuracy_gap_pp']<=1.5+1e-12 for v in rs),scenarios=len(rs),
            r4_attack_tail_asr=mean(v['r4_tail']['source_transfer_asr'] for v in rs if v['malicious_ratio']),
            mean_accuracy_change_pp=mean(v['delta_pp']['tail']['accuracy'] for v in rs),
            worst_accuracy_change_pp=min(v['delta_pp']['tail']['accuracy'] for v in rs),
            worst_tail_asr_change_pp=max(v['delta_pp']['tail']['source_transfer_asr'] for v in rs),
            worst_full_attack_asr_change_pp=max(v['delta_pp']['attack_period']['source_transfer_asr'] for v in rs if v['malicious_ratio']))
    integrity=all(warmup.values()) and all(x['passed'] for checks in pairing.values() for x in checks.values())
    result.update(status='complete_audited_user_requested_r4_formal',acceptance=base.AUTHORIZATION,
        user_requested_fixed_candidate='R4',prior_validation_selected=None,validation_selection_passed=False,
        no_retuning_or_selection=True,no_formal_retuning_or_selection=True,
        official_test_used_for_selection=False,prior_test_results_seen=True,integrity_checks_passed=integrity,
        r4_original_h84_warmup_pairing=warmup,paired_ours_changes=changes,goal_summary=summary,
        selection_note='User explicitly requested fixed R4 official-test follow-up after its validation gain0.4794pp failed the frozen0.5pp gate. Original failed decisions are retained. No official-test tuning, selection or pruning.',
        canonical_ours_key='H84 is the legacy comparison slot only; all30 Ours cells in this export are R4 and identify their source tasks.')
    return result


def finish(p,specs,m,out):
    base.verify(m);full.verify_files(p['amendment']['pinned_reference_sha256'])
    prior=base.read(base.ROOT/base.RESERVE);base.validate_prior(prior)
    progress=base.summarize(p,specs,m,out)
    if not all(c['status']=='complete' and c['complete_finite'] for c in progress['cells'].values()):
        base.write(out/'analysis/r4_results.json',dict(progress,status='incomplete_or_failed_evidence_retained'))
        return dict(status='incomplete_or_failed_evidence_retained',complete=progress['complete'],total=30)
    executed=execution(out,specs);lock=base.read(out/'reference_lock.json')
    original_bundle=base.read(base.ROOT/base.BASELINE);original=original_bundle['report']
    inputs=dict(progress['input_sha256'],**executed['input_sha256'],**lock['references'])
    def load(path,binary=False,lines=False):
        inputs[str(path.relative_to(base.ROOT))]=base.sha(path);raw=path.read_bytes()
        return pickle.loads(raw) if binary else ([json.loads(x) for x in raw.splitlines()] if lines else json.loads(raw))
    def model(path):
        inputs[str(path.relative_to(base.ROOT))]=base.sha(path)
        with np.load(path) as z:return z['params'].copy()
    replacements={};audits={};rows_by_name={};anchors={};warmup={};lineage={}
    from diagnose_cifar_cp1_iid import cell_audit
    for spec in specs:
        name=spec['task_id'];canonical=name.replace('R4_','H84_',1)
        folder=out/'tasks'/name;old=base.ROOT/base.ORIGINAL/'tasks'/canonical
        env=load(out/'environments'/(name+'.json'))
        if full.environment_key(env)!=lock['expected_environment'] or env['gpu_uuid'] in base.QUARANTINE:raise ValueError('Formal environment differs')
        obj=load(folder/'result.pickle',True);old_obj=load(old/'result.pickle',True)
        if asdict(obj.config)!=spec['config'] or obj.stopped_round!=150:raise ValueError('R4 config/horizon differs')
        if dict(asdict(old_obj.config),**base.PARAMETERS)!=spec['config'] or sorted(obj.malicious_clients)!=sorted(old_obj.malicious_clients):raise ValueError('Original config/identities differ')
        if not np.array_equal(model(folder/'anchor_r0.npz'),model(old/'anchor_r0.npz')):raise ValueError('Original initial model differs')
        if load(folder/'dataset_identity.json')!=load(old/'dataset_identity.json'):raise ValueError('Original data identity differs')
        targets=load(folder/'targets.json')
        if targets!=load(old/'targets.json'):raise ValueError('Original attack/test targets differ')
        rows=load(folder/'metrics.json')['rounds'];old_rows=load(old/'metrics.json')['rounds']
        warmup[name]=classification_equal(rows,old_rows,20)
        rows_by_name[canonical]=rows;anchors[canonical]=model(folder/'anchor_r24.npz')
        calls=load(folder/'audit.jsonl',lines=True) if (folder/'audit.jsonl').exists() else []
        actual=load(folder/'actual_weights.jsonl',lines=True)
        weights_ok=audit_weights(actual,rows,obj.diagnostics,name)
        c=deepcopy(progress['cells'][name]);c.update(arm='H84',candidate='R4',source_task=name)
        replacements[canonical]=c
        audits[canonical]=dict(summary=c,attack=report.audit_calls(calls,spec,targets,sorted(obj.malicious_clients),rows),
            mechanism=cell_audit(obj,rows),actual_weights_verified=weights_ok,original_h84_warmup_equal=warmup[name])
        lineage[canonical]=dict(source_task=name,original_task=canonical,only_changed_parameter=base.PARAMETERS,
            same_initial_model_data_targets_malicious_ids=True,original_h84_warmup_equal=warmup[name],
            user_requested_after_failed_validation=True,prior_report_sha256=base.RESERVE_SHA)
    pairing={}
    for seed in p['seeds']:
        checks={};suffix='_s'+str(seed)
        for task in full.all_tasks():
            if task['arm']!='H84' or not task['ratio']:continue
            name=task['id']+suffix;clean=f"H84_{task['partition']}_clean"+suffix
            same=classification_equal(rows_by_name[name],rows_by_name[clean],24)
            anchor=bool(np.array_equal(anchors[name],anchors[clean]))
            checks[task['id']]=dict(classification_prefix_equal=same,round24_model_equal=anchor,passed=same and anchor,required=True)
        pairing[str(seed)]=checks
    result=combine(original,replacements,audits,pairing,warmup,p['seeds'])
    result.update(fingerprint=m['fingerprint'],observed_utc=base.utc(),input_sha256=inputs,
        replacement_lineage=lineage,prior_reserve_decision=prior['decision'],prior_first_panel=prior['prior_panel'],
        accepted_baseline_sha256=base.BASELINE_SHA,execution_summary=executed)
    trajectories=deepcopy(original_bundle['trajectories'])
    fields=('round',*FIELDS,'false_positive_revocations','true_positive_revocations','honest_weight_loss','malicious_weight_mass')
    trajectories.update({n:[{k:r[k] for k in fields} for r in rows] for n,rows in rows_by_name.items()})
    for n,c in original['cells'].items():
        if c['arm']!='H84' and trajectories[n]!=original_bundle['trajectories'][n]:raise ValueError('Other five curves changed')
    full.verify_files(inputs);base.verify(m)
    base.write(out/'analysis/execution_audit.json',executed)
    base.write(out/'analysis/r4_results.json',dict(status=result['status'],cells=progress['cells'],audits=audits,pairing=pairing,warmup=warmup,integrity_checks_passed=result['integrity_checks_passed']))
    target=out/'analysis/combined_results.json';base.write(target,result)
    base.write(out/'export/combined_bundle.json',dict(schema='user-requested-r4-official-export-v1',manifest=m,
        report=result,trajectories=trajectories,combined_report_sha256=base.sha(target),accepted_baseline_sha256=base.BASELINE_SHA,
        source_verified=True,all_inputs_unchanged=True,exported_utc=base.utc()))
    return dict(status=result['status'],new_ours_tasks=30,reused_other_five=150,
        integrity_checks_passed=result['integrity_checks_passed'],goal_summary=result['goal_summary'])
