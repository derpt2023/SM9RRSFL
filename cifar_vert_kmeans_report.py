"""Audit30 replacement cells and assemble180 cells without old VERT plot data."""
from copy import deepcopy
from dataclasses import asdict
import json
import pickle

import numpy as np

import run_cifar_vert_kmeans_supplement as base
import cifar_h84_formal_report as formal_report
import run_cifar_h84_full_validation as full
from cifar_vert_kmeans_selection import audit as audit_selection


def combine(original, replacements, audits, pair_checks, seeds):
    expected={k for k,v in original['cells'].items() if v['arm']=='VERT'}
    if set(replacements)!=expected or set(audits)!=expected:
        raise ValueError('All30 replacement VERT cells required; no mixed old/new fallback')
    if any(c['status']!='complete' or not c['complete_finite'] for c in replacements.values()):
        raise ValueError('Incomplete replacement cells cannot replace original VERT')
    result=deepcopy(original)
    result['cells'].update(replacements);result['audited_cells'].update(audits)
    result['groups']=formal_report.grouped_statistics(result['cells'],seeds)
    for seed in seeds:
        suffix='_s'+str(seed);previous=original['per_seed'][str(seed)]
        pairing=deepcopy(previous['pairing']);pairing.update(pair_checks[str(seed)])
        selected={t['id']:result['audited_cells'][t['id']+suffix] for t in full.all_tasks()}
        strict=full.assess(selected,previous['fedavg_attack_gates'],pairing)
        current=formal_report.with_krum_exception(strict)
        result['per_seed'][str(seed)]=dict(strict=strict,with_previous_krum_exception=current,
            pairing=pairing,fedavg_attack_gates=deepcopy(previous['fedavg_attack_gates']))
    for name,c in original['cells'].items():
        if c['arm']!='VERT' and result['cells'][name]!=c:
            raise ValueError('Other five methods must remain byte-value identical')
    result.update(status='complete_audited_vert_kmeans_replacement',
        posthoc_user_requested_protocol_correction=True,
        no_retuning_or_selection=True,old_vert_in_current_tables=False,
        selection_note='VERT selection corrected after prior test results were observed; same seeds and all other conditions. No search or outcome-based choice between attempts.')
    return result


def finish(p,specs,manifest,out):
    report=base.summarize(p,specs,manifest,out)
    if not all(c['status']=='complete' and c['complete_finite'] for c in report['cells'].values()):
        base.write(out/'analysis/supplement_results.json',dict(status='incomplete_evidence_retained',**report))
        return dict(status='incomplete_evidence_retained',complete=report['complete'],total=30)
    parent=base.ROOT/p['parent_output_dir']
    original=base.read(parent/'analysis/formal_results.json')
    full.verify_files(original['input_sha256'])
    lock=base.read(out/'reference_lock.json');full.verify_files(lock['references'])
    inputs=dict(report['input_sha256']);audits={};trajectories={};anchors={};lineage={}
    def load(path,binary=False):
        inputs[str(path.relative_to(base.ROOT))]=base.sha(path)
        return pickle.loads(path.read_bytes()) if binary else base.read(path)
    def model(path):
        inputs[str(path.relative_to(base.ROOT))]=base.sha(path)
        with np.load(path) as z:return z['params'].copy()
    fields=('accuracy','source_transfer_asr','non_source_accuracy','aux128_confidence')
    for spec in specs:
        name=spec['task_id'];folder=out/'tasks'/name;old=parent/'tasks'/name
        env=load(out/'environments'/(name+'.json'))
        if full.environment_key(env)!=lock['expected_environment']:raise ValueError('Environment differs')
        obj=load(folder/'result.pickle',True);old_obj=load(old/'result.pickle',True)
        if asdict(obj.config)!=spec['config'] or obj.stopped_round!=150:
            raise ValueError('Result config/horizon differs')
        expected=asdict(old_obj.config);expected['vert_top_k']=0
        if expected!=spec['config'] or obj.malicious_clients!=old_obj.malicious_clients:
            raise ValueError('Only selection may differ from original paired task')
        rows=load(folder/'metrics.json')['rounds'];old_rows=load(old/'metrics.json')['rounds']
        trajectories[name]=rows;anchors[name]=model(folder/'anchor_r24.npz')
        if not np.array_equal(model(folder/'anchor_r0.npz'),model(old/'anchor_r0.npz')):
            raise ValueError('Paired original initial model differs')
        if any(rows[r][k]!=old_rows[r][k] for r in range(3) for k in fields):
            raise ValueError('Two unchanged bootstrap rounds differ from original')
        targets=load(folder/'targets.json')
        if targets!=load(old/'targets.json'):raise ValueError('Training/evaluation target identity differs')
        audit_path=folder/'audit.jsonl'
        calls=[]
        if audit_path.exists():
            inputs[str(audit_path.relative_to(base.ROOT))]=base.sha(audit_path)
            calls=[json.loads(x) for x in audit_path.read_text().splitlines()]
        sp=folder/'vert_selection.jsonl';inputs[str(sp.relative_to(base.ROOT))]=base.sha(sp)
        selection=audit_selection([json.loads(x) for x in sp.read_text().splitlines()],name)
        audits[name]=dict(summary=report['cells'][name],
            attack=formal_report.audit_calls(calls,spec,targets,sorted(obj.malicious_clients),rows),
            selection=selection)
        lineage[name]=dict(original_task=name,config_changed_fields=['vert_top_k'],
            same_seed_initial_model_data_targets_malicious_ids=True,
            unchanged_bootstrap_rounds_match=True,selector_only_adapter=True)
    pair_checks={}
    for seed in p['seeds']:
        checks={};suffix='_s'+str(seed)
        for task in full.all_tasks():
            if task['arm']!='VERT' or not task['ratio']:continue
            name=task['id']+suffix;clean=f"VERT_{task['partition']}_clean"+suffix
            classification=all(trajectories[name][r][k]==trajectories[clean][r][k]
                               for r in range(25) for k in fields)
            model_equal=bool(np.array_equal(anchors[name],anchors[clean]))
            checks[task['id']]=dict(classification_prefix_equal=classification,
                round24_model_equal=model_equal,passed=classification and model_equal,required=True)
        pair_checks[str(seed)]=checks
    result=combine(original,report['cells'],audits,pair_checks,p['seeds'])
    original_bundle_path=parent/'export/formal_audited_bundle.json'
    if base.sha(original_bundle_path)!=p['amendment']['original_export_sha256']:
        raise ValueError('Original exported trajectories differ')
    old_bundle=base.read(original_bundle_path)
    merged={k:v for k,v in old_bundle['trajectories'].items() if not k.startswith('VERT_')}
    keys=('round','accuracy','source_transfer_asr','non_source_accuracy','aux128_confidence',
          'false_positive_revocations','true_positive_revocations','honest_weight_loss','malicious_weight_mass')
    merged.update({k:[{n:r[n] for n in keys} for r in rows] for k,rows in trajectories.items()})
    if len(merged)!=180:raise ValueError('Exactly180 current trajectories required')
    full.verify_files(inputs);full.verify_files(original['input_sha256']);base.verify(manifest)
    result.update(fingerprint=manifest['fingerprint'],observed_utc=base.utc(),
        input_sha256=dict(original['input_sha256'],**inputs),
        original_report_sha256=p['amendment']['original_report_sha256'],replacement_lineage=lineage)
    base.write(out/'analysis/supplement_results.json',dict(status='complete_audited_supplement',
        fingerprint=manifest['fingerprint'],cells=report['cells'],audited_cells=audits,
        pairing=pair_checks,input_sha256=inputs,lineage=lineage))
    rp=out/'analysis/combined_results.json';base.write(rp,result)
    export=dict(schema='h84-vert-kmeans-corrected-export-v1',report=result,trajectories=merged,
        combined_report_sha256=base.sha(rp),original_report_sha256=p['amendment']['original_report_sha256'],
        original_export_sha256=p['amendment']['original_export_sha256'],
        manifest=manifest,source_verified=True,all_inputs_unchanged=True,exported_utc=base.utc())
    base.write(out/'export/combined_bundle.json',export)
    return dict(status=result['status'],new_cells=30,current_comparison_cells=180,
                export=str(out/'export/combined_bundle.json'))
