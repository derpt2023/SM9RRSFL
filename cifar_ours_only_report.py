"""Complete paired validation and transparent candidate decision; no official test input."""
from dataclasses import asdict
import json
import math
import pickle
from statistics import mean
import run_cifar_ours_only as run


def assess(report):
    cells=report['cells'];rule=run.RULE
    if (report.get('evaluation_scope')!='training_validation_7500' or report.get('official_test_used_for_selection') is not False):
        raise ValueError('Selection is permitted only on training-derived validation')
    if len(cells)!=120 or any(not c['complete_finite'] for c in cells.values()):
        return dict(selected=None,reason='All120 complete finite cells required; failures and missingness retained',candidates={})
    keys={(c['arm'],c['config']['seed'],c['config']['partition'],c['config']['malicious_ratio']):c for c in cells.values()}
    expected={(a,s,p,r) for a in run.ARMS for s in run.SEEDS for p in ('iid','dirichlet') for r in (0.,.2,.4,.6,.8)}
    if set(keys)!=expected or len(keys)!=120:raise ValueError('Duplicate or missing paired scientific identity')
    pairs=report['attack_pairing']
    execution_ok=(report.get('execution_audited') is True and len(pairs)==96 and
        all(v['classification_prefix_equal'] and v['round24_model_equal'] for v in pairs.values()))
    decisions={}
    for arm in run.CANDIDATES:
        failures=[];iid_gain=[];noniid_asr_rises=[];details={}
        for seed in run.SEEDS:
            for part in ('iid','dirichlet'):
                for ratio in (0.,.2,.4,.6,.8):
                    c=keys[arm,seed,part,ratio];b=keys['H84',seed,part,ratio]
                    da=c['tail']['accuracy']-b['tail']['accuracy'];ds=c['tail']['source_transfer_asr']-b['tail']['source_transfer_asr']
                    checks=dict(healthy=c['health']['complete_observed_health_pass'],baseline_healthy=b['health']['complete_observed_health_pass'])
                    if part=='dirichlet':
                        checks.update(non_iid_accuracy=da>=-rule['non_iid_tail_acc_drop_max']-1e-12,
                                      non_iid_asr=ds<=rule['non_iid_tail_asr_rise_max']+1e-12)
                        noniid_asr_rises.append(ds)
                        if ratio:
                            checks.update(non_iid_full_attack_accuracy=c['attack_period']['accuracy']>=b['attack_period']['accuracy']-rule['non_iid_attack_period_acc_drop_max']-1e-12,
                                non_iid_full_attack_asr=c['attack_period']['source_transfer_asr']<=b['attack_period']['source_transfer_asr']+rule['non_iid_attack_period_asr_rise_max']+1e-12)
                    else:
                        checks['iid_asr']=ds<=rule['iid_tail_asr_rise_max']+1e-12
                        if ratio==0:checks['iid_clean_accuracy']=da>=-rule['iid_clean_acc_drop_max']-1e-12
                        iid_gain.append(da)
                    name=f'{seed}_{part}_{ratio}'
                    details[name]=dict(accuracy_change=da,asr_change=ds,checks=checks)
                    failures += [name+':'+k for k,ok in checks.items() if not ok]
        gain=mean(iid_gain)
        if gain<rule['iid_mean_acc_gain_min']-1e-12:failures.append('mean_IID_accuracy_gain_below_0.5pp')
        if not execution_ok:failures.append('complete_execution_or_pairing_missing')
        iid_accuracy=mean(keys[arm,s,'iid',r]['tail']['accuracy'] for s in run.SEEDS for r in (0.,.2,.4,.6,.8))
        decisions[arm]=dict(eligible=not failures,failures=failures,paired_details=details,
            iid_mean_accuracy=iid_accuracy,iid_mean_accuracy_gain=gain,
            worst_non_iid_asr_rise=max(noniid_asr_rises))
    eligible=[(a,d) for a,d in decisions.items() if d['eligible']]
    selected=min(eligible,key=lambda x:(-x[1]['iid_mean_accuracy'],x[1]['worst_non_iid_asr_rise'],x[0]))[0] if eligible else None
    return dict(selected=selected,candidates=decisions,rule=rule,
        proximity_goal_met=None,proximity_goal_status='Requires fixed-candidate official-test comparison to accepted five methods',
        next_step='Freeze selected parameters before a separate same-seed official-test panel' if selected else
            'Inspect complete validation mechanism evidence; at most the three predeclared reserve candidates in a separate manifest; no automatic retry',
        formal_test_used_for_selection=False,accepted_snapshot_unchanged=True)


def execution_audit(out,specs,m):
    state=run.read(out/'controller_status.json',{})
    if state.get('status')!='panel_terminal' or state.get('active') or state.get('pending') or state.get('failure'):
        raise ValueError('Controller has not completed without failure')
    scope=run.read(out/run.SCOPE_DIR/'manifest.json');transition=run.read(out/run.SCOPE_DIR/'transition.json')
    names={s['task_id'] for s in specs}
    if set(scope['tasks'])!=names or scope['parent_fingerprint']!=m['fingerprint']:
        raise ValueError('Execution scope differs from worker identity')
    excluded=set(transition['excluded_completed'])
    if not excluded.issubset(set(scope['omitted_from_future_dispatch'])):
        raise ValueError('Unidentified historical controls')
    started={};ended={};files={}
    for basename in ('manifest.json','transition.json','prior_controller_status.json'):
        f=out/run.SCOPE_DIR/basename;files[str(f.relative_to(run.ROOT))]=run.sha(f)
    for dirname in ('scheduler_ours_paper_refinement_v1','scheduler_ours_only_v1'):
        for f in sorted((out/dirname).glob('*/events.jsonl')):
            files[str(f.relative_to(run.ROOT))]=run.sha(f)
            for row in map(json.loads,f.read_text().splitlines()):
                if row['event']=='worker_started':
                    if dirname=='scheduler_ours_only_v1' and row['task'] not in names:
                        raise ValueError('Ours-only controller dispatched a non-Ours task')
                    started.setdefault(row['task'],[]).append(row)
                elif row['event']=='worker_exited':ended.setdefault(row['task'],[]).append(row)
                elif row['event']=='dispatch_blocked':raise ValueError('Dispatch failure retained; independent review required')
    if set(started)!=names|excluded or set(ended)!=names|excluded:
        raise ValueError('Execution event coverage differs')
    for name in names|excluded:
        if len(started[name])!=1 or len(ended[name])!=1:raise ValueError('Non-single execution attempt requires review')
        s,e=started[name][0],ended[name][0];g=s['admission']
        if (e['returncode']!=0 or e['status']!='complete' or s['child_pid']!=e['child_pid'] or
            g['uuid']!=e['gpu_uuid'] or g['uuid'] in run.QUARANTINE or g['free_mb']<8192):
            raise ValueError('Execution/admission mismatch')
    return dict(tasks=len(names),ours_attempts=len(names),ours_failures=0,automatic_retries=0,
        historical_control_attempts=len(excluded),historical_controls=sorted(excluded),
        historical_controls_used_for_selection=False,source_fingerprint=m['fingerprint'],
        scope_fingerprint=scope['fingerprint'],input_sha256=files)


def finish(p,specs,m,out):
    report=run.summarize(p,specs,m,out)
    if any(c['status'] not in run.TERMINAL for c in report['cells'].values()):raise ValueError('Active or missing tasks; do not finish/restart')
    if any(not c['complete_finite'] for c in report['cells'].values()):
        report['decision']=assess(report);report['status']='incomplete_or_failed_validation_retained'
        run.write(out/'analysis/ours_only_results.json',report);return dict(status=report['status'],decision=report['decision'])
    execution=execution_audit(out,specs,m)
    from audit_cifar_cuda_panel import audit_calls
    from diagnose_cifar_cp1_iid import cell_audit
    import numpy as np
    hashes=dict(report['input_sha256'],**execution['input_sha256']);labels={};initial={};audits={}
    lock=run.read(out/'reference_lock.json')
    for s in specs:
        name=s['task_id'];folder=out/'tasks'/name
        def load(path,binary=False,lines=False):
            hashes[str(path.relative_to(run.ROOT))]=run.sha(path);raw=path.read_bytes()
            return pickle.loads(raw) if binary else ([json.loads(x) for x in raw.splitlines()] if lines else json.loads(raw))
        result=load(folder/'result.pickle',binary=True)
        if asdict(result.config)!=s['config']:raise ValueError('Result configuration differs')
        env=load(out/'environments'/f'{name}.json')
        if any(env[k]!=lock['runtime_environment'][k] for k in ('python','torch','cuda','cudnn','gpu','native_binary_sha256')) or env['numerics']!=p['numerics'] or env['gpu_uuid'] in run.QUARANTINE:
            raise ValueError('Task software/numerics/GPU differs')
        rows=load(folder/'metrics.json')['rounds'];targets=load(folder/'targets.json')
        records=load(folder/'audit.jsonl',lines=True) if (folder/'audit.jsonl').exists() else []
        checks=dict(attack=audit_calls(records,s,targets,result.malicious_clients,rows))
        key=tuple(s['config'][k] for k in ('seed','partition','malicious_ratio'));ids=sorted(result.malicious_clients)
        if key in labels and labels[key]!=ids:raise ValueError('Unpaired malicious identities')
        labels[key]=ids
        f=folder/'anchor_r0.npz';hashes[str(f.relative_to(run.ROOT))]=run.sha(f)
        with np.load(f) as z:model=z['params'].copy()
        key=key[:2]
        if key in initial and not np.array_equal(initial[key],model):raise ValueError('Unpaired same-seed initial model')
        initial[key]=model
        actual=load(folder/'actual_weights.jsonl',lines=True)
        if [x['round'] for x in actual]!=list(range(1,151)):raise ValueError('Missing actual aggregation decisions')
        for observed,row in zip(actual,rows[1:]):
            if observed['task_id']!=name or any(abs(observed[k]-row[k])>1e-9 for k in ('honest_weight_loss','malicious_weight_mass')):
                raise ValueError('Actual aggregation diagnostic differs')
            if any(not math.isfinite(v) or v<0 for v in observed['weights'].values()):raise ValueError('Invalid actual coefficient')
        if s['config']['method']=='sm9rrs':
            checks['ours_mechanism']=cell_audit(result,rows)
            grouped={rid:{} for rid in range(1,151)}
            for d in result.diagnostics:grouped[d.round][d.client_id]=d.aggregation_weight
            for observed in actual:
                wanted=grouped[observed['round']];given=observed['weights']
                if any(abs(wanted.get(c,0)-given.get(c,0))>1e-12 for c in wanted.keys()|given.keys()):raise ValueError('Ours diagnostic differs from actual aggregation')
        audits[name]=checks
    run.verify_scope(p['scope_manifest'],m)
    for f,h in hashes.items():
        if run.sha(run.ROOT/f)!=h:raise ValueError('Evidence changed during audit')
    report.update(execution_audited=True,execution_audits=audits,input_sha256=hashes,
        execution_summary=execution,status='complete_audited_validation',updated_utc=run.utc())
    report['decision']=assess(report)
    run.write(out/'analysis/ours_only_execution_audit.json',execution)
    run.write(out/'analysis/ours_only_results.json',report)
    return dict(status=report['status'],selected=report['decision']['selected'],proximity_goal_met=report['decision']['proximity_goal_met'],tasks=120)
