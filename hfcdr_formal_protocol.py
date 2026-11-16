"""Qualified, frozen-parameter 936-run official test contract."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import hfcdr_validation_protocol as v
import hfcdr_validation_selection as selection
import hfcdr_confirmation as confirmation
from hfcdr_pilot_protocol import ROOT, read, write, sha, digest, baseline, verify_frozen_sources

CONFIG = ROOT/'configs/hfcdr_formal_v1.json'
OUTPUT = ROOT/'outputs/hfcdr_formal_v1'
OWN = ('hfcdr_formal_protocol.py', 'hfcdr_formal_scheduler.py', 'run_hfcdr_formal.py',
       'audit_hfcdr_formal.py', 'render_hfcdr_formal.py', 'tests/test_hfcdr_formal.py',
       'HFCDR_FORMAL.txt', 'hfcdr_health_context.py', 'hfcdr_study_runtime.py', 'hfcdr_study_report.py',
       'hfcdr_study_protocol.py', 'configs/hfcdr_attack_study_v1.json',
       'hfcdr_pilot_report.py', 'hfcdr_confirmation.py', 'run_hfcdr_confirmation.py',
       'configs/hfcdr_confirmation_rules_v1.json', 'configs/hfcdr_validation_confirmation_v1.json',
       'hfcdr_validation_selection.py', 'configs/hfcdr_validation_defense_v1.json',
       'outputs/hfcdr_validation_confirmation_v1/manifest.json',
       'outputs/hfcdr_validation_confirmation_v1/confirmation.json',
       'outputs/hfcdr_validation_defense_v1/selection.json')


def make_policy(manifest, decision):
    """Copy exactly the confirmed parameters; there is no new candidate search."""
    policy = deepcopy(manifest['policy'])
    if (policy['phase'] != 'confirmation' or not decision['complete'] or not decision['formal_allowed']
            or decision['fingerprint'] != manifest['fingerprint'] or decision['completed'] != 312):
        raise ValueError('Complete qualified confirmation required')
    if set(policy['defense_candidates']) != set(confirmation.METHODS):
        raise ValueError('All six confirmed methods required')
    policy['defense_candidates'] = {m:policy['defense_candidates'][m] for m in confirmation.METHODS}
    if any(len(cs) != 1 for cs in policy['defense_candidates'].values()):
        raise ValueError('One frozen candidate per method required')
    policy.update(schema='hfcdr_formal_v1', phase='formal', seeds=policy['formal_seeds'],
        confirmation_seeds=policy['seeds'], methods=list(confirmation.METHODS),
        evaluation='official_test_all_source5', official_test_used=True,
        official_test_used_for_selection=False,
        entry_gate=dict(confirmation_fingerprint=manifest['fingerprint'],
            confirmation_decision_sha256=digest(decision), defense_gate=deepcopy(policy['entry_gate']),
            requires_passed_confirmation=True),
        budget_accounting=dict(fresh_runs=936, candidates_per_method=1, additional_search=False))
    policy['formal_ordering'] = dict(rule='Rotate independently within each dataset/partition; complementary offsets balance all six positions per dataset',
        offsets=dict(cifar10=dict(iid=0,dirichlet=3),mnist=dict(iid=1,dirichlet=4)))
    return policy


def gate():
    out = confirmation.OUTPUT
    expected = confirmation.upstream(v.OUTPUT, selection.OUTPUT)
    manifest = read(out/'manifest.json')
    if manifest['policy'] != expected:
        raise ValueError('Confirmation did not use the frozen defense selection')
    decision = confirmation.assess_output(out)
    if read(out/'confirmation.json') != decision or not decision['formal_allowed']:
        raise ValueError('Confirmation failed or its frozen decision changed; official test held')
    return manifest, decision


def policy():
    value = read(CONFIG)
    if (value['schema'] != 'hfcdr_formal_v1' or value['phase'] != 'formal'
            or value['seeds'] != [2026111801, 2026111802, 2026111803]
            or value['datasets'] != ['cifar10','mnist'] or value['partitions'] != ['iid','dirichlet']
            or value['ratios'] != [0.,.2,.4,.6,.8] or value['cryptography'] != 'absent'
            or value['official_test_used'] is not True or value['official_test_used_for_selection'] is not False
            or value['evaluation'] != 'official_test_all_source5' or value['automatic_retry'] is not False):
        raise ValueError('Complete no-crypto fixed formal contract required')
    reference = make_policy(read(confirmation.OUTPUT/'manifest.json'), read(confirmation.OUTPUT/'confirmation.json'))
    if value != reference: raise ValueError('Formal parameters differ from qualified confirmation')
    return value


def task_specs(spec):
    base_tasks,base_groups = v.task_specs(spec)
    lookup = {t['task_id']:t for t in base_tasks};counts = Counter();tasks,groups = [],[]
    methods = spec['methods']
    for group in base_groups:
        members = {lookup[tid]['config']['method']:lookup[tid] for tid in group['tasks']}
        if set(members) != set(methods) or len(group['tasks']) != 6:
            raise ValueError('Exactly six paired methods required')
        representative = next(iter(members.values()))['config']
        key = representative['dataset'],representative['partition']
        offset = (counts[key]+spec['formal_ordering']['offsets'][key[0]][key[1]]) % 6
        counts[key] += 1
        order = methods[offset:]+methods[:offset]
        tasks.extend(members[m] for m in order)
        groups.append(dict(group_id=group['group_id'],tasks=[members[m]['task_id'] for m in order]))
    return tasks,groups


def plan():
    spec = policy(); tasks, groups = task_specs(spec); b = baseline()
    if len(tasks) != 936 or len(groups) != 156:
        raise ValueError('Full 936-run/156-scenario matrix required')
    audit = dict(passed=True,tasks=936,groups=156,seeds=spec['seeds'],
        by_method=dict(Counter(t['config']['method'] for t in tasks)),
        by_dataset=dict(Counter(t['config']['dataset'] for t in tasks)),
        by_partition=dict(Counter(t['config']['partition'] for t in tasks)),
        by_condition=dict(Counter(t['condition'] for t in tasks)),
        by_ratio=dict(Counter(str(t['config']['malicious_ratio']) for t in tasks)),
        official_test_used_for_selection=False,cryptography='absent',automatic_retry=False)
    names = sorted(set((*OWN,*v.OWN,*v.BASELINES.values(),str(CONFIG.relative_to(ROOT)))))
    payload = dict(schema=spec['schema'],policy=spec,tasks=tasks,groups=groups,audit=audit,
        numerics=b['numerics'],original_environment={k:x for k,x in b['original_environment'].items() if k!='native_binary_sha256'},
        criteria=read(ROOT/'configs/hfcdr_attack_pilot_v1.json')['criteria'],
        frozen_sources=verify_frozen_sources(),new_sources={n:sha(ROOT/n) for n in names})
    return dict(payload,fingerprint=digest(payload))


def build():
    manifest,decision = gate()
    if CONFIG.exists(): raise FileExistsError('Existing formal specification retained')
    write(CONFIG,make_policy(manifest,decision))
    result = plan()
    return dict(result['audit'],fingerprint=result['fingerprint'],training_started=False)


def safe_output(path):
    raw = Path(path).expanduser(); out = raw.resolve()
    if raw.is_symlink() or out.parent != (ROOT/'outputs').resolve() or not out.name.startswith('hfcdr_formal_'):
        raise ValueError('Use an independent outputs/hfcdr_formal_* directory')
    return out


def load_dataset(dataset,data_root):
    # This existing loader verifies the original training split and official arrays.
    # It is called only after the confirmation gate, never during candidate selection.
    from hfcdr_study_protocol import load_dataset as load_official
    ds,identity = load_official(dataset,data_root)
    validation = read(confirmation.OUTPUT/'dataset_identity.json')[dataset]
    if identity['contract'] != validation['contract'] or any(identity['arrays'][k] != validation['arrays'][k]
            for k in ('x_train','y_train')):
        raise ValueError('Formal training arrays differ from validation')
    if identity['evaluation_scope'] != 'official_test_all_source5' or identity['accuracy_samples'] != 10000:
        raise ValueError('Full official evaluation set required')
    return ds,identity


pairing_key = v.pairing_key


def prepare(output,data_root):
    upstream_manifest,decision = gate()
    if policy() != make_policy(upstream_manifest,decision): raise ValueError('Upstream gate changed')
    out = safe_output(output)
    if out.exists(): raise FileExistsError('Fresh formal output required')
    manifest = plan(); identities,pairing = {},{}
    from hfcdr_pilot_data import participant_record
    for dataset in manifest['policy']['datasets']:
        ds,identities[dataset] = load_dataset(dataset,data_root)
        for task in manifest['tasks']:
            if task['config']['dataset'] == dataset and pairing_key(task) not in pairing:
                pairing[pairing_key(task)] = participant_record(ds.y_train,task['config'])
        del ds
    if len(pairing) != 60: raise ValueError('60 shared client/attacker assignments required')
    out.mkdir(parents=True)
    for name,value in (('manifest',manifest),('dataset_identity',identities),('pairing',pairing),('entry_gate',manifest['policy']['entry_gate'])):
        write(out/(name+'.json'),value)
    write(out/'provenance/original_results_before.json',read(confirmation.OUTPUT/'provenance/original_results_before.json'))
    audit = dict(manifest['audit'],data_verified=True,pairings=60,training_started=False)
    write(out/'configuration_audit.json',audit)
    return dict(audit,fingerprint=manifest['fingerprint'])


def check(output):
    out = safe_output(output); manifest = read(out/'manifest.json')
    if manifest != plan(): raise ValueError('Frozen formal source/configuration changed')
    counts = Counter()
    for task in manifest['tasks']:
        folder = out/'tasks'/task['task_id']; file = folder/'status.json'
        counts[read(file)['status'] if file.exists() else 'incomplete' if folder.exists() else 'pending'] += 1
    return dict(fingerprint=manifest['fingerprint'],counts=dict(counts),tasks=len(manifest['tasks']))
