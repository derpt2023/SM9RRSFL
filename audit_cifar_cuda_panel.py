#!/usr/bin/env python3
"""Audit completed fixed-attack CUDA panels without training or changing results."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import pickle

import run_cifar_cp1_cuda_development as base
from run_cifar_cp1_shared_gpu import file_sha


def audit_calls(records, spec, targets, malicious, rows):
    from cifar_optimized_station_runtime import validate_target_view_record
    cfg, profile = spec['config'], spec['profile']
    indices = targets['training_target_indices']
    if (len(indices) != 128 or len(set(indices)) != 128
            or targets['source_transfer_count'] != 750
            or targets['official_test_used_for_selection'] is not False
            or targets['training_target_is_optimizer_input'] is not True
            or targets['transfer_target_is_optimizer_input'] is not False):
        raise ValueError('Unexpected optimizer/evaluation target provenance')
    expected = {(rid, int(cid.split('-')[1]))
                for rid in range(cfg['attack_start_round'], 151)
                for cid in malicious if cid not in rows[rid - 1]['blacklisted_client_ids']}
    seen = set()
    for row in records:
        pair = row['round'], row['client_idx']
        if pair in seen or pair not in expected:
            raise ValueError('Duplicate or unexpected actual attack call')
        seen.add(pair)
        if (row['task_id'] != spec['task_id'] or row['client_id'] != 'client-' + str(pair[1])
                or row['source_client'] != row['client_id'] or row['status'] != 'complete'
                or row['checks_enabled'] is not True or row['first_failure'] is not None):
            raise ValueError('Attack task/client identity or execution differs')
        for key in ('optimizer', 'normal_optimizer', 'lr', 'boost', 'stealth_steps', 'scope'):
            if row[key] != profile[key]:
                raise ValueError('Actual attack profile differs: ' + key)
        if row['profile_id'] != profile['id'] or not math.isclose(
                row['normal_lr'], cfg['lr'] * cfg['lr_decay'] ** (pair[0] - 1), rel_tol=1e-12):
            raise ValueError('Profile id or scheduled ordinary SGD differs')
        if row['normal_steps'] != row['reference_steps'] or row['normal_steps'] <= 0:
            raise ValueError('Ordinary/reference steps differ')
        validate_target_view_record(row, profile, cfg, indices)
    if seen != expected:
        raise ValueError('Missing committed attack calls')
    return dict(actual_calls=len(seen), expected_calls=len(expected),
                complete_coverage=True, target_views_and_optimizers_verified=True)


def audit(config):
    from diagnose_cifar_cp1_iid import cell_audit
    p, specs, manifest = base.plan(config)
    out = base.ROOT / p['output_dir']
    if base.read(out / 'manifest.json') != manifest:
        raise ValueError('Frozen manifest differs')
    if any(base.read(out / 'tasks' / s['task_id'] / 'status.json', {}).get('status') != 'complete'
           for s in specs):
        raise ValueError('Full completed panel required for execution audit')
    report = base.summarize(p, specs, manifest, out)
    inputs = dict(report['input_sha256'])
    inputs[str((out / 'manifest.json').relative_to(base.ROOT))] = file_sha(out / 'manifest.json')
    cells, labels = {}, {}

    def read(path, binary=False):
        raw = path.read_bytes()
        inputs[str(path.relative_to(base.ROOT))] = hashlib.sha256(raw).hexdigest()
        return pickle.loads(raw) if binary else json.loads(raw)

    for spec in specs:
        name = spec['task_id']
        folder = out / 'tasks' / name
        result = read(folder / 'result.pickle', True)
        if asdict(result.config) != spec['config'] or result.stopped_round != 150:
            raise ValueError('Result config or horizon differs')
        rows = read(folder / 'metrics.json')['rounds']
        targets = read(folder / 'targets.json')
        audit_path = folder / 'audit.jsonl'
        if audit_path.exists():
            raw = audit_path.read_bytes()
            inputs[str(audit_path.relative_to(base.ROOT))] = hashlib.sha256(raw).hexdigest()
            records = [json.loads(line) for line in raw.splitlines()]
        else:
            records = []  # audit_calls still requires exact expected coverage
        attack = audit_calls(records, spec, targets, result.malicious_clients, rows)
        key = spec['config']['partition'], spec['config']['malicious_ratio']
        ids = sorted(result.malicious_clients)
        if key in labels and labels[key] != ids:
            raise ValueError('Malicious identity set differs across paired arms')
        labels[key] = ids
        cell = dict(attack=attack, summary=report['cells'][name])
        if spec['config']['method'] == 'sm9rrs':
            cell['mechanism'] = cell_audit(result, rows)
        cells[name] = cell
        env = out / 'environments' / (name + '.json')
        read(env)
    if any(not all(v.values()) for v in report['attack_pairing'].values()):
        raise ValueError('Pre-attack clean pairing differs')
    for name, digest in inputs.items():
        if file_sha(base.ROOT / name) != digest:
            raise ValueError('Audit input changed: ' + name)
    base.verify(manifest)
    evidence = dict(observed_utc=base.utc(), fingerprint=manifest['fingerprint'],
                    auditor_sha256=file_sha(__file__), sources_verified=len(manifest['sources']),
                    input_sha256=inputs, cells=cells, attack_pairing=report['attack_pairing'],
                    fedavg_attack_gates=report['fedavg_attack_gates'],
                    main_qualified=False,
                    limitation='Observed-seed development only; not full two-distribution six-method qualification.')
    base.write(out / 'analysis/attack_execution_audit.json', evidence)
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=base.Path, default=base.DEFAULT)
    args = parser.parse_args()
    evidence = audit(args.config)
    print(json.dumps(dict(cells=len(evidence['cells']), sources=evidence['sources_verified'],
                          calls={n: c['attack']['actual_calls'] for n, c in evidence['cells'].items()},
                          fedavg_attack_gates=evidence['fedavg_attack_gates']), indent=2))
