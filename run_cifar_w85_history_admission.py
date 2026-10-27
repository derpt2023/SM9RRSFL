#!/usr/bin/env python3
"""W85 single-variable history admission panel; frozen attack and worker reused."""
import argparse
from copy import deepcopy
import fcntl
from pathlib import Path

import run_cifar_cp1_cuda_development as base
import run_cifar_cp1_shared_gpu as shared
from audit_cifar_cuda_panel import audit

DEFAULT = base.ROOT / 'configs/cifar10_w85_history_admission150_v1.json'
HISTORY = {'H80': .80, 'H84': .84}
OWN = ('run_cifar_w85_history_admission.py', 'audit_cifar_cuda_panel.py',
       'tests/test_w85_history_admission.py', 'run_cifar_cp1_shared_gpu.py',
       'tests/test_cp1_shared_gpu.py', 'run_cifar_cp1_dynamic_gpu.py',
       'tests/test_cp1_dynamic_gpu.py')
DESIGN = dict(variable='detector_history_threshold', reference_arm='W85',
              candidates=HISTORY, ratios=[0., .6, .8], partition='iid',
              additional_training_rounds=900, reuse_previous_completed_controls=True,
              clean_fp_must_strictly_decrease=True,
              clean_accuracy_loss_limit=.01, clean_asr_increase_limit=.02,
              attack_accuracy_loss_limit=.02, attack_asr_increase_limit=.05,
              selection_order=['clean_false_revocations', 'worst_attack_asr_increase',
                               'negative_attack_mean_accuracy', 'arm_id'],
              limitation='Development progression only; final health-scoped ranks require two distributions, five ratios, six methods.')


def validate_protocol(p, parent):
    if p.get('history_admission_design') != DESIGN:
        raise ValueError('History-only design or development criteria changed')
    for key in ('base_configs', 'profile', 'data_config', 'expected_dataset_identity',
                'selection_policy', 'numerics', 'parent_config', 'parent_config_sha256'):
        if p[key] != parent[key]:
            raise ValueError('Frozen public setting changed: ' + key)
    reference = parent['arms']['W85']
    expected_arms = {arm: dict(method=reference['method'], parameters=dict(
        reference['parameters'], detector_history_threshold=threshold)) for arm, threshold in HISTORY.items()}
    if p['arms'] != expected_arms:
        raise ValueError('Only W85 history threshold may change')
    tasks = [dict(id=f'{arm}_iid_{suffix}', arm=arm, partition='iid', ratio=ratio)
             for arm in HISTORY for suffix, ratio in [('clean', 0.), ('r60', .6), ('r80', .8)]]
    if p['tasks'] != tasks or p['output_dir'] != 'outputs/cifar_w85_history_admission150_v1':
        raise ValueError('History panel task coverage/output differs')
    base.specs_for(p)


def plan(config):
    p = base.read(config)
    parent, _, parent_manifest = base.plan(base.DEFAULT)
    validate_protocol(p, parent)
    if p['parent_cuda_config_sha256'] != shared.file_sha(base.DEFAULT):
        raise ValueError('Parent CUDA protocol changed')
    for name, digest in parent_manifest['sources'].items():
        if p['frozen_source_sha256'].get(name) != digest:
            raise ValueError('Parent scientific source omitted or changed: ' + name)
    for name in OWN:
        if p['frozen_source_sha256'].get(name) != shared.file_sha(base.ROOT / name):
            raise ValueError('History execution source omitted or changed: ' + name)
    return base.plan(config)


def reference_evidence():
    p, specs, manifest = base.plan(base.DEFAULT)
    out = base.ROOT / p['output_dir']
    if base.read(out / 'manifest.json') != manifest:
        raise ValueError('Parent manifest differs')
    path = out / 'analysis/attack_execution_audit.json'
    evidence = base.read(path)
    if (not evidence or evidence['fingerprint'] != manifest['fingerprint']
            or evidence['auditor_sha256'] != shared.file_sha(base.ROOT / 'audit_cifar_cuda_panel.py')
            or set(evidence['cells']) != {s['task_id'] for s in specs}):
        raise ValueError('Complete parent actual-execution audit required')
    if not evidence['fedavg_attack_gates'] or not all(v['passed'] for v in evidence['fedavg_attack_gates'].values()):
        raise ValueError('Parent FedAvg attack ineffective')
    refs = dict(evidence['input_sha256'])
    refs[str(path.relative_to(base.ROOT))] = shared.file_sha(path)
    for name, digest in refs.items():
        if shared.file_sha(base.ROOT / name) != digest:
            raise ValueError('Frozen reference differs: ' + name)
    for suffix in ('clean', 'r60', 'r80'):
        cell = evidence['cells']['W85_iid_' + suffix]['summary']
        if not cell['health']['complete_observed_health_pass']:
            raise ValueError('W85 reference health failed')
    return dict(parent_fingerprint=manifest['fingerprint'], references=refs,
                reused_arms=['W85', 'FedAvg'], no_reference_training_repeated=True), evidence


def assess(cells, reference, execution_audited):
    results = {}
    ref = {suffix: reference['cells']['W85_iid_' + suffix]['summary'] for suffix in ('clean', 'r60', 'r80')}
    fedavg_clean = reference['cells']['FedAvg_iid_clean']['summary']['tail']['accuracy']
    for arm in HISTORY:
        selected = {suffix: cells[arm + '_iid_' + suffix] for suffix in ref}
        if any(v['status'] != 'complete' for v in selected.values()):
            results[arm] = dict(complete=False, development_pass=False, reason='Incomplete or failed task retained')
            continue
        differences = {suffix: dict(
            accuracy_change_pp=100 * (cell['tail']['accuracy'] - ref[suffix]['tail']['accuracy']),
            asr_change_pp=100 * (cell['tail']['source_transfer_asr'] - ref[suffix]['tail']['source_transfer_asr']),
            fp_change=cell['final_fp'] - ref[suffix]['final_fp']) for suffix, cell in selected.items()}
        health = all(v['health']['complete_observed_health_pass'] for v in selected.values())
        utility = fedavg_clean - selected['clean']['tail']['accuracy'] <= .05 + 1e-12
        lower_fp = selected['clean']['final_fp'] < ref['clean']['final_fp']
        limits = all(
            d['accuracy_change_pp'] >= (-1 if suffix == 'clean' else -2) - 1e-9
            and d['asr_change_pp'] <= (2 if suffix == 'clean' else 5) + 1e-9
            for suffix, d in differences.items())
        passed = health and utility and lower_fp and limits and execution_audited
        results[arm] = dict(complete=True, development_pass=passed, health=health,
                           clean_utility=utility, clean_fp_strictly_lower=lower_fp,
                           preservation_limits=limits, execution_audited=execution_audited,
                           differences_vs_w85=differences,
                           selection_key=[selected['clean']['final_fp'],
                               max(differences[k]['asr_change_pp'] for k in ('r60', 'r80')),
                               -sum(selected[k]['tail']['accuracy'] for k in ('r60', 'r80')) / 2, arm])
    eligible = [arm for arm, v in results.items() if v['development_pass']]
    chosen = min(eligible, key=lambda arm: results[arm]['selection_key']) if eligible else None
    return dict(candidates=results, selected_for_full_validation=chosen, main_qualified=False,
                status='development_candidate_for_full_validation' if chosen else
                       'bounded_history_panel_failed' if all(v['complete'] for v in results.values()) and execution_audited
                       else 'pending_execution_or_audit',
                final_selection_policy_unchanged=True,
                limitations=DESIGN['limitation'])


def prepare(out, manifest, refs):
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = base.read(out / 'manifest.json')
        if old is None:
            if any(p.name != 'controller.lock' for p in out.iterdir()):
                raise ValueError('Unidentified history output retained')
            base.write(out / 'manifest.json', manifest)
            base.write(out / 'reference_lock.json', refs)
        elif old != manifest or base.read(out / 'reference_lock.json') != refs:
            raise ValueError('Frozen history identity or references differ')


def finish(config, p, specs, manifest, out, reference):
    complete = all(base.read(out / 'tasks' / s['task_id'] / 'status.json', {}).get('status') == 'complete' for s in specs)
    if complete:
        audit(config)
    report = base.summarize(p, specs, manifest, out)
    refs, current = reference_evidence()
    if base.read(out / 'reference_lock.json') != refs:
        raise ValueError('Parent reference lock differs')
    decision = assess(report['cells'], current, execution_audited=complete)
    base.write(out / 'analysis/history_decision.json', decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--batch', action='store_true')
    group.add_argument('--finish', action='store_true')
    args = parser.parse_args()
    p, specs, manifest = plan(args.config)
    if not (args.batch or args.finish):
        print(base.json.dumps(dict(fingerprint=manifest['fingerprint'], tasks=[s['task_id'] for s in specs],
                                   maximum_rounds=900, history_thresholds=HISTORY, training_started=False), indent=2))
        return
    refs, reference = reference_evidence()
    out = base.ROOT / p['output_dir']
    if args.batch:
        prepare(out, manifest, refs)
        shared.run(args.config, shared.shared_parameters(p), specs, manifest, out, 8)
        # A user-requested graceful stop is not permission to label a partial
        # panel finished or to start the next phase.
        if base.read(out / 'controller_status.json')['status'] != 'panel_terminal':
            return
    elif base.read(out / 'manifest.json') != manifest or base.read(out / 'reference_lock.json') != refs:
        raise ValueError('History manifest/reference lock required')
    print(base.json.dumps(finish(args.config, p, specs, manifest, out, reference), indent=2))


if __name__ == '__main__':
    main()
