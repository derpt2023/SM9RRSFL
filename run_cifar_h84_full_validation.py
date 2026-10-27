#!/usr/bin/env python3
"""Frozen H84 CUDA matrix: 54 fresh tasks + six audited references, no test set."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import pickle
import sys

import numpy as np

import run_cifar_cp1_cuda_development as base
import run_cifar_cp1_shared_gpu as shared
import run_cifar_w85_history_admission as history
from audit_cifar_cuda_panel import audit_calls
from cifar_tad_no_prior import PriorFreeTAD, PAPER_CHOICES

DEFAULT = base.ROOT / 'configs/cifar10_h84_full_validation150_v1.json'
OUTPUT = 'outputs/cifar_h84_full_validation150_v1'
METHODS = dict(FedAvg='fedavg', H84='sm9rrs', TAD='ding13', VERT='vert',
               AlignIns='alignins', Krum='krum')
RATIOS = dict(clean=0., r20=.2, r40=.4, r60=.6, r80=.8)
PARTITIONS = ('iid', 'dirichlet')
OWN = ('run_cifar_h84_full_validation.py', 'tests/test_h84_full_validation.py')
DESIGN = dict(candidate='H84', distributions=list(PARTITIONS), dirichlet_alpha=.5,
              ratios=list(RATIOS.values()), methods=METHODS, total_cells=60,
              fresh_tasks=54, reused_tasks=6, additional_training_rounds=8100,
              official_test_used_for_selection=False, result_pruning=False,
              reference_scope='all_five_ratios_within_each_distribution',
              required_pre_attack_pairing='all_methods_except_Krum',
              krum_disclosure='Unchanged baseline uses the configured Byzantine count; clean/attack prefixes are reported but are not required to agree for Krum.',
              limitation='Previously observed development seed; not formal or unused-seed confirmation.')


def all_tasks():
    return [dict(id=f'{arm}_{d}_{label}', arm=arm, partition=d, ratio=r)
            for d in PARTITIONS for label, r in RATIOS.items() for arm in METHODS]


def reuse_map():
    return {f'{arm}_iid_{label}': dict(
        output='outputs/cifar_cp1_cuda_diagnosis150_v1' if arm == 'FedAvg'
               else 'outputs/cifar_w85_history_admission150_v1',
        task=f'{arm}_iid_{label}')
        for arm in ('FedAvg', 'H84') for label in ('clean', 'r60', 'r80')}


def validate(p, parent):
    for key in ('base_configs', 'profile', 'data_config', 'expected_dataset_identity',
                'selection_policy', 'numerics', 'parent_config', 'parent_config_sha256'):
        if p[key] != parent[key]:
            raise ValueError('Frozen public setting changed: ' + key)
    arms = {a: dict(method=m, parameters=deepcopy(parent['arms']['H84']['parameters'])
                   if a == 'H84' else {}) for a, m in METHODS.items()}
    if p['arms'] != arms or p['validation_design'] != DESIGN or p['reuse'] != reuse_map():
        raise ValueError('Candidate, baseline, matrix design or references changed')
    expected = [t for t in all_tasks() if t['id'] not in reuse_map()]
    if p['tasks'] != expected or p['output_dir'] != OUTPUT:
        raise ValueError('Exactly 54 declared fresh tasks required')
    if p['tad_reproduction'] != PAPER_CHOICES:
        raise ValueError('No-prior TAD interpretation changed')
    base.specs_for(p)


def plan(config):
    p = base.read(config)
    parent, _, pm = history.plan(history.DEFAULT)
    validate(p, parent)
    if p['history_parent_sha256'] != shared.file_sha(history.DEFAULT):
        raise ValueError('History protocol changed')
    for name, digest in pm['sources'].items():
        if p['frozen_source_sha256'].get(name) != digest:
            raise ValueError('History source not frozen: ' + name)
    for name in OWN:
        if p['frozen_source_sha256'].get(name) != shared.file_sha(base.ROOT / name):
            raise ValueError('Validation source not frozen: ' + name)
    return base.plan(config)


def verify_files(refs):
    for name, digest in refs.items():
        if shared.file_sha(base.ROOT / name) != digest:
            raise ValueError('Evidence changed: ' + name)


def environment_key(env):
    return {k: env[k] for k in ('python', 'torch', 'cuda', 'cudnn', 'gpu',
                                'numerics', 'native_binary_sha256')}


def references(p):
    old_lock, old = history.reference_evidence()
    hp, hs, hm = history.plan(history.DEFAULT)
    out = base.ROOT / hp['output_dir']
    if base.read(out / 'manifest.json') != hm or base.read(out / 'reference_lock.json') != old_lock:
        raise ValueError('History identity/reference lock differs')
    ap = out / 'analysis/attack_execution_audit.json'
    evidence = base.read(ap)
    if (evidence['fingerprint'] != hm['fingerprint']
            or evidence['auditor_sha256'] != shared.file_sha(base.ROOT / 'audit_cifar_cuda_panel.py')
            or set(evidence['cells']) != {s['task_id'] for s in hs}):
        raise ValueError('Full six-task history execution audit required')
    decision_path = out / 'analysis/history_decision.json'
    decision = base.read(decision_path)
    computed = history.assess({n: c['summary'] for n, c in evidence['cells'].items()}, old, True)
    if decision != computed or decision['selected_for_full_validation'] != 'H84':
        raise ValueError('H84 selection is not supported by the frozen selection rule')
    refs = dict(old_lock['references'], **evidence['input_sha256'])
    for path in (ap, decision_path, out / 'reference_lock.json'):
        refs[str(path.relative_to(base.ROOT))] = shared.file_sha(path)
    full_specs = {s['task_id']: s for s in base.specs_for(dict(p, tasks=all_tasks()))}
    reused = {}
    envs = []
    for name, loc in p['reuse'].items():
        owner = base.ROOT / loc['output']
        om = base.read(owner / 'manifest.json')
        original = next(s for s in om['specs'] if s['task_id'] == loc['task'])
        if original != full_specs[name]:
            raise ValueError('Reused configuration/profile identity differs: ' + name)
        record = (old if name.startswith('FedAvg_') else evidence)['cells'][name]
        if record['summary']['status'] != 'complete' or not record['summary']['complete_finite']:
            raise ValueError('Unfinished reference cannot be reused')
        reused[name] = record
        envs.append(environment_key(base.read(owner / 'environments' / (name + '.json'))))
    if any(e != envs[0] for e in envs):
        raise ValueError('Reused software/numerics/native environments differ')
    verify_files(refs)
    return dict(references=refs, reuse=p['reuse'], expected_environment=envs[0],
                parent_decision=decision, previously_observed_seed=True), reused


def check_runtime(expected):
    import torch
    current = dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda,
                   cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
                   numerics=expected['numerics'], native_binary_sha256={
                       str(f.relative_to(base.ROOT)): shared.file_sha(f)
                       for f in (base.ROOT / 'sm9rrsfl').glob('_native_*.so')})
    if current != expected:
        raise ValueError('Current runtime differs from reusable CUDA results')


def tad_audit(rows):
    if set(inspect.signature(PriorFreeTAD.__init__).parameters) != {
            'self', 'client_ids', 'seed', 'matrix_offset', 'matrix_shape', 'compute_backend', 'device'}:
        raise ValueError('TAD constructor API changed')
    if set(inspect.signature(PriorFreeTAD.evaluate_round).parameters) != {
            'self', 'updates', 'global_params', 'round_id'}:
        raise ValueError('TAD decision API changed')
    previous = set()
    for row in rows[1:]:
        d = row.get('tad_no_prior')
        if not d or d['version'] != 'TAD-no-prior-v1' or d['round'] != row['round']:
            raise ValueError('Missing actual no-prior TAD decision')
        if set(d['outliers']) != {i for i, score in d['scores'].items() if score > .5}:
            raise ValueError('TAD score rule differs')
        newly = {i for i, n in d['observations'].items()
                 if n >= 3 and 2 * d['abnormal_counts'][i] > n} - previous
        if newly != set(d['newly_removed']) or set(d['removed']) != previous | newly:
            raise ValueError('TAD cumulative removal rule differs')
        previous = set(d['removed'])
        if previous != set(row['blacklisted_client_ids']):
            raise ValueError('TAD blacklist differs')
        if any(v < 0 for v in d['weights'].values()) or any(d['weights'][i] for i in previous):
            raise ValueError('Invalid TAD weights')
        if len(previous) < len(d['weights']) and abs(sum(d['weights'].values()) - 1) > 1e-9:
            raise ValueError('TAD weight mass differs')
    return dict(rounds_verified=len(rows) - 1, actual_no_prior_decisions=True,
                internal_spectral_state_replayed=False)


def rank_comparison(metrics, eligible, ours_healthy, clean):
    # Preserve all six methods before health filtering; ineligible Ours gets no rank.
    result = dict(raw_metrics=metrics, eligible_methods=list(eligible))
    for key, reverse in (('accuracy', True), ('asr', False)):
        ordered = sorted(metrics, key=lambda a: (-metrics[a][key] if reverse else metrics[a][key], a))
        result['raw_' + key + '_order'] = ordered
        result['raw_' + key + '_ranks'] = {a: 1 + sum(
            (m[key] > metrics[a][key] + 1e-12 if reverse else m[key] < metrics[a][key] - 1e-12)
            for m in metrics.values()) for a in metrics}
        result['healthy_' + key + '_order'] = [a for a in ordered if a in eligible]
    if not eligible:
        return dict(result, passed=False, reason='no_healthy_reference', accuracy_rank=None, asr_rank=None)
    ar = 1 + sum(metrics[a]['accuracy'] > metrics['H84']['accuracy'] + 1e-12 for a in eligible)
    sr = 1 + sum(metrics[a]['asr'] < metrics['H84']['asr'] - 1e-12 for a in eligible)
    ag = 100 * (max(metrics[a]['accuracy'] for a in eligible) - metrics['H84']['accuracy'])
    sg = 100 * (metrics['H84']['asr'] - min(metrics[a]['asr'] for a in eligible))
    numeric = ag <= (1 if clean else 2) + 1e-9 and sg <= (2 if clean else 5) + 1e-9
    numeric &= clean or max(ar, sr) <= 2
    return dict(result, accuracy_rank=ar if ours_healthy else None, asr_rank=sr if ours_healthy else None,
                diagnostic_accuracy_rank=ar, diagnostic_asr_rank=sr,
                accuracy_gap_pp=ag, asr_gap_pp=sg, metrics_pass=bool(numeric),
                ours_distribution_healthy=ours_healthy, passed=bool(numeric and ours_healthy))


def assess(cells, gates, pairing):
    expected = {t['id'] for t in all_tasks()}
    if set(cells) != expected or any(c['summary']['status'] != 'complete'
                                   or not c['summary']['complete_finite'] for c in cells.values()):
        raise ValueError('All 60 complete finite cells required; missing methods cannot be excluded')
    health, scenarios = {}, {}
    for d in PARTITIONS:
        health[d] = {}
        fed = cells[f'FedAvg_{d}_clean']['summary']['tail']['accuracy']
        for a in METHODS:
            drop = fed - cells[f'{a}_{d}_clean']['summary']['tail']['accuracy']
            reasons = ['clean_accuracy_drop'] if drop > .05 + 1e-12 else []
            for label in RATIOS:
                h = cells[f'{a}_{d}_{label}']['summary']['health']
                reasons += [label + ':' + r for r in h['v14_failure_reasons'] + h['additional_review_reasons']]
                if not h['complete_observed_health_pass'] and not (h['v14_failure_reasons'] or h['additional_review_reasons']):
                    reasons.append(label + ':health_failure')
            health[d][a] = dict(valid=not reasons, reasons=reasons, clean_accuracy_drop=drop)
        eligible = [a for a in METHODS if health[d][a]['valid']]
        for label in RATIOS:
            metrics = {a: dict(accuracy=cells[f'{a}_{d}_{label}']['summary']['tail']['accuracy'],
                               asr=cells[f'{a}_{d}_{label}']['summary']['tail']['source_transfer_asr']) for a in METHODS}
            scenarios[f'{d}_{label}'] = rank_comparison(metrics, eligible, health[d]['H84']['valid'], label == 'clean')
    prerequisites = len(gates) == 8 and all(v['passed'] for v in gates.values())
    prerequisites &= len(pairing) == 48 and all(v['passed'] for v in pairing.values() if v['required'])
    passed = bool(prerequisites and all(v['passed'] for v in scenarios.values()))
    return dict(method_health=health, scenarios=scenarios, development_criteria_passed=passed,
                formal_plan_condition_met=passed, main_qualified=False, unused_seed_confirmation_passed=False,
                prerequisites_pass=bool(prerequisites), limitation=DESIGN['limitation'])


def finish(config, p, specs, manifest, out):
    from diagnose_cifar_cp1_iid import cell_audit
    lock, reused = references(p)
    if base.read(out / 'reference_lock.json') != lock:
        raise ValueError('Locked references changed')
    report = base.summarize(p, specs, manifest, out)
    if any(c['status'] != 'complete' or not c['complete_finite'] for c in report['cells'].values()):
        result = dict(status='incomplete_or_failed_evidence_retained', main_qualified=False,
                      cells=report['cells'], no_missing_method_excluded=True)
        base.write(out / 'analysis/full_validation.json', result)
        return result
    inputs = dict(lock['references'], **report['input_sha256'])
    def load(path, binary=False):
        inputs[str(path.relative_to(base.ROOT))] = shared.file_sha(path)
        return pickle.loads(path.read_bytes()) if binary else base.read(path)
    cells, trajectories, anchors, malicious = {}, {}, {}, {}
    for spec in base.specs_for(dict(p, tasks=all_tasks())):
        name = spec['task_id']
        loc = p['reuse'].get(name, dict(output=p['output_dir'], task=name))
        owner = base.ROOT / loc['output']; folder = owner / 'tasks' / loc['task']
        env = load(owner / 'environments' / (loc['task'] + '.json'))
        if environment_key(env) != lock['expected_environment']:
            raise ValueError('Cross-platform/software evidence: ' + name)
        rows = load(folder / 'metrics.json')['rounds']; trajectories[name] = rows
        f = folder / 'anchor_r24.npz'; inputs[str(f.relative_to(base.ROOT))] = shared.file_sha(f)
        with np.load(f) as z:
            anchors[name] = z['params'].copy()
        result = load(folder / 'result.pickle', True)
        if asdict(result.config) != spec['config'] or result.stopped_round != 150:
            raise ValueError('Result identity/horizon differs')
        key = (spec['config']['partition'], spec['config']['malicious_ratio'])
        ids = sorted(result.malicious_clients)
        if key in malicious and malicious[key] != ids:
            raise ValueError('Malicious identities differ across methods')
        malicious[key] = ids
        if name in reused:
            cells[name] = deepcopy(reused[name])
        else:
            targets = load(folder / 'targets.json')
            f = folder / 'audit.jsonl'
            if f.exists():
                inputs[str(f.relative_to(base.ROOT))] = shared.file_sha(f)
                records = [json.loads(line) for line in f.read_bytes().splitlines()]
            else:
                records = []
            cells[name] = dict(summary=report['cells'][name],
                               attack=audit_calls(records, spec, targets, ids, rows))
            if spec['config']['method'] == 'sm9rrs':
                cells[name]['mechanism'] = cell_audit(result, rows)
            if spec['config']['method'] == 'ding13':
                cells[name]['tad_audit'] = tad_audit(rows)
        cells[name]['source'] = loc
    pairing, gates = {}, {}
    fields = ('accuracy', 'source_transfer_asr', 'non_source_accuracy', 'aux128_confidence')
    for t in all_tasks():
        if not t['ratio']:
            continue
        n = t['id']; c = f"{t['arm']}_{t['partition']}_clean"
        checks = dict(classification_prefix_equal=all(trajectories[n][r][k] == trajectories[c][r][k]
                      for r in range(25) for k in fields), round24_model_equal=bool(np.array_equal(anchors[n], anchors[c])))
        if t['arm'] == 'TAD':
            checks['detector_prefix_equal'] = all(trajectories[n][r].get('tad_no_prior') ==
                                                  trajectories[c][r].get('tad_no_prior') for r in range(25))
        pairing[n] = dict(checks, passed=all(checks.values()), required=t['arm'] != 'Krum')
        if t['arm'] == 'FedAvg':
            a, b = cells[c]['summary']['tail'], cells[n]['summary']['tail']
            drop, rise = a['accuracy'] - b['accuracy'], b['source_transfer_asr'] - a['source_transfer_asr']
            gates[n] = dict(accuracy_drop=drop, asr_rise=rise, passed=drop >= .1-1e-12 and rise >= .3-1e-12)
    assessment = assess(cells, gates, pairing)
    verify_files(inputs); base.verify(manifest)
    result = dict(status='complete_audited_development', fingerprint=manifest['fingerprint'],
                  observed_utc=base.utc(), cells=cells, attack_gates=gates, pre_attack_pairing=pairing,
                  assessment=assessment, input_sha256=inputs, tad_reproduction=PAPER_CHOICES,
                  official_test_used_for_selection=False, fresh_tasks=54, reused_tasks=6)
    base.write(out / 'analysis/full_validation.json', result)
    return dict(status=result['status'], assessment=assessment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--batch', action='store_true')
    group.add_argument('--finish', action='store_true')
    group.add_argument('--ready', action='store_true')
    args = parser.parse_args()
    p, specs, manifest = plan(args.config)
    if not (args.batch or args.finish or args.ready):
        print(json.dumps(dict(fingerprint=manifest['fingerprint'], fresh_tasks=len(specs),
                              reused_tasks=6, matrix_cells=60, additional_rounds=8100, training_started=False)))
        return
    lock, _ = references(p)
    out = base.ROOT / p['output_dir']
    if args.ready or args.batch:
        check_runtime(lock['expected_environment'])
    if args.ready:
        print(json.dumps(dict(ready=True, verified_references=len(lock['references']), fresh_tasks=54, reused_tasks=6)))
        return
    if args.batch:
        history.prepare(out, manifest, lock)
        shared.run(args.config, shared.shared_parameters(p), specs, manifest, out, 8)
        if base.read(out / 'controller_status.json')['status'] != 'panel_terminal':
            return
    elif base.read(out / 'manifest.json') != manifest:
        raise ValueError('Validation manifest required')
    print(json.dumps(finish(args.config, p, specs, manifest, out), indent=2))


if __name__ == '__main__':
    main()
