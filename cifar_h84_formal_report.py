"""Read-only input audit and independent reporting of fixed H84 formal experiments."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
import pickle
import statistics

import numpy as np

import run_cifar_h84_formal as formal
import run_cifar_h84_full_validation as full
from watch_cifar_experiment import timing


def audit_calls(records, spec, targets, malicious, rows):
    from cifar_optimized_station_runtime import validate_target_view_record
    cfg, profile = spec['config'], spec['profile']
    indices = targets['training_target_indices']
    if (len(indices) != 128 or len(set(indices)) != 128
            or targets['source_transfer_count'] != 1000
            or targets.get('evaluation_scope') != 'official_test'
            or targets.get('evaluation_count') != 10000
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


def summarize(p, specs, manifest, out):
    from scripts.experiments.cifar10.attack_stability_worker import task_identity
    from scripts.experiments.cifar10.c7_station_report import _health
    formal.verify(manifest)
    if formal.read(out / 'manifest.json') != manifest:
        raise ValueError('Formal manifest changed or missing')
    lock = formal.read(out / 'reference_lock.json')
    full.verify_files(lock['references'])
    cells, inputs = {}, {}
    def load(path):
        inputs[str(path.relative_to(formal.ROOT))] = formal.sha(path)
        return formal.read(path)
    for spec in specs:
        name = spec['task_id']; folder = out / 'tasks' / name
        state = formal.read(folder / 'status.json', {})
        cell = dict(status=state.get('status', 'not_run'), config=spec['config'],
                    arm=spec['arm'], complete_finite=False,
                    completed_round=state.get('completed_round'))
        cells[name] = cell
        if cell['status'] not in formal.TERMINAL: continue
        state = load(folder / 'status.json')
        expected = task_identity(dict(spec, fingerprint=manifest['fingerprint'], data_config=manifest['data_config']))
        if load(folder / 'task.json') != expected or state['fingerprint'] != expected['fingerprint']:
            raise ValueError('Formal task identity differs: ' + name)
        if cell['status'] != 'complete':
            cell['failure'] = dict(error_type=state.get('error_type'), error=state.get('error'))
            continue
        metrics = load(folder / 'metrics.json'); rows = metrics['rounds']
        if (metrics['fingerprint'] != expected['fingerprint'] or state['completed_round'] != 150
                or [r['round'] for r in rows] != list(range(151)) or state['final_metrics'] != rows[-1]):
            raise ValueError('Incomplete formal trajectory: ' + name)
        if load(folder / 'dataset_identity.json') != lock['dataset_identity']:
            raise ValueError('Formal dataset identity differs')
        keys = ('accuracy', 'source_transfer_asr', 'honest_weight_loss', 'malicious_weight_mass')
        if any(not math.isfinite(r[k]) or not -1e-9 <= r[k] <= 1+1e-9 for r in rows for k in keys):
            raise ValueError('Nonfinite or invalid metric')
        finite = state['nonfinite_updates'] == 0 and all(r['cumulative_nonfinite_updates'] == 0 for r in rows)
        time_record = load(folder / 'timing.json')
        computed = timing(state['summary'])
        if not computed['available_and_consistent'] or any(time_record[k] != computed[k]
                for k in ('runtime_with_crypto_seconds', 'runtime_without_crypto_seconds', 'crypto_wall_seconds', 'stages')):
            raise ValueError('Formal timing sidecar does not match original result')
        cell.update(complete_finite=finite,
            tail={k: statistics.mean(r[k] for r in rows[-10:]) for k in keys},
            final={k: rows[-1][k] for k in keys},
            health=_health(dict(valid_complete=True, numerically_finite=finite), rows, spec['config']),
            attack_period={k: statistics.mean(r[k] for r in rows[25:]) for k in keys[:2]},
            peak_attack_asr=max(r['source_transfer_asr'] for r in rows[25:]),
            final_fp=rows[-1]['false_positive_revocations'], final_tp=rows[-1]['true_positive_revocations'],
            timing=time_record)
    result = dict(fingerprint=manifest['fingerprint'], cells=cells, input_sha256=inputs,
                  complete=sum(c['status'] == 'complete' for c in cells.values()), total=len(specs),
                  official_test_used_for_selection=False, results_are_provisional_until_audit=True)
    formal.write(out / 'analysis/progress_summary.json', result)
    return result


def grouped_statistics(cells, seeds):
    """No rounds-as-replicates, health filtering, zero filling or incomplete group ranking."""
    groups = {}
    for task in full.all_tasks():
        records = [cells.get(task['id'] + '_s' + str(s), {}) for s in seeds]
        valid = [c for c in records if c.get('status') == 'complete' and c.get('complete_finite')]
        group = dict(expected_n=len(seeds), actual_n=len(valid), all_seeds_complete=len(valid) == len(seeds),
                     health_failures_retained=True, samples={})
        extractors = dict(accuracy=lambda c:c['tail']['accuracy'], asr=lambda c:c['tail']['source_transfer_asr'],
            false_revocation_rate=lambda c:c['final_fp']/(100*(1-c['config']['malicious_ratio'])),
            runtime_with_crypto_seconds=lambda c:c['timing']['runtime_with_crypto_seconds'],
            runtime_without_crypto_seconds=lambda c:c['timing']['runtime_without_crypto_seconds'],
            crypto_wall_seconds=lambda c:c['timing']['crypto_wall_seconds'])
        for metric, get in extractors.items():
            values = [get(c) for c in valid]
            group['samples'][metric] = dict(values=values, mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values) > 1 else None)
        groups[task['id']] = group
    return groups


def with_krum_exception(assessment):
    """Keep the old strict view and independently apply the previously accepted narrow exception."""
    current = deepcopy(assessment)
    entry = current['method_health']['iid']['Krum']
    entry['original_reasons'] = list(entry['reasons'])
    entry['reasons'] = [r for r in entry['reasons'] if r != 'clean_accuracy_drop']
    entry['valid'] = not entry['reasons']
    entry['waived_clean_utility_only'] = 'clean_accuracy_drop' in entry['original_reasons']
    for d in full.PARTITIONS:
        eligible = [a for a in full.METHODS if current['method_health'][d][a]['valid']]
        for label in full.RATIOS:
            name = d + '_' + label
            current['scenarios'][name] = full.rank_comparison(assessment['scenarios'][name]['raw_metrics'],
                eligible, current['method_health'][d]['H84']['valid'], label == 'clean')
    for view in (assessment, current):
        for key in ('formal_plan_condition_met', 'main_qualified', 'unused_seed_confirmation_passed', 'limitation'):
            view.pop(key, None)
        view['descriptive_frozen_criteria_passed'] = bool(view['prerequisites_pass'] and all(
            c['passed'] for c in view['scenarios'].values()))
        view.pop('development_criteria_passed', None)
    return current


def finish(p, specs, manifest, out):
    from diagnose_cifar_cp1_iid import cell_audit
    report = summarize(p, specs, manifest, out)
    result = dict(status='incomplete_or_failed_evidence_retained', acceptance=p['acceptance'],
        cells=report['cells'], groups=grouped_statistics(report['cells'], p['seeds']),
        official_test_used_for_selection=False, no_retuning_or_selection=True,
        timing_note='Shared-GPU measured worker wall times, not isolated method speed estimates.')
    if not all(c['status'] == 'complete' and c['complete_finite'] for c in report['cells'].values()):
        formal.write(out / 'analysis/formal_results.json', result)
        return dict(status=result['status'], complete=report['complete'], total=report['total'])
    inputs = dict(report['input_sha256']); lock = formal.read(out / 'reference_lock.json')
    cells, trajectories, anchors, labels = {}, {}, {}, {}
    def load(path, binary=False):
        inputs[str(path.relative_to(formal.ROOT))] = formal.sha(path)
        return pickle.loads(path.read_bytes()) if binary else formal.read(path)
    for spec in specs:
        name = spec['task_id']; folder = out / 'tasks' / name
        env = load(out / 'environments' / (name + '.json'))
        if full.environment_key(env) != lock['expected_environment']:
            raise ValueError('Cross-platform/software result: ' + name)
        rows = load(folder / 'metrics.json')['rounds']; trajectories[name] = rows
        model_path = folder / 'anchor_r24.npz'; inputs[str(model_path.relative_to(formal.ROOT))] = formal.sha(model_path)
        with np.load(model_path) as z: anchors[name] = z['params'].copy()
        obj = load(folder / 'result.pickle', True)
        if asdict(obj.config) != spec['config'] or obj.stopped_round != 150:
            raise ValueError('Result configuration or horizon differs')
        key = tuple(spec['config'][k] for k in ('seed', 'partition', 'malicious_ratio'))
        malicious = sorted(obj.malicious_clients)
        if key in labels and labels[key] != malicious: raise ValueError('Paired malicious IDs differ')
        labels[key] = malicious
        targets = load(folder / 'targets.json'); audit = folder / 'audit.jsonl'
        if audit.exists():
            inputs[str(audit.relative_to(formal.ROOT))] = formal.sha(audit)
            calls = [json.loads(line) for line in audit.read_bytes().splitlines()]
        else: calls = []
        cell = dict(summary=report['cells'][name], attack=audit_calls(calls, spec, targets, malicious, rows))
        if spec['arm'] == 'H84': cell['mechanism'] = cell_audit(obj, rows)
        if spec['arm'] == 'TAD': cell['tad_audit'] = full.tad_audit(rows)
        cells[name] = cell
    seed_assessments = {}
    fields = ('accuracy', 'source_transfer_asr', 'non_source_accuracy', 'aux128_confidence')
    for seed in p['seeds']:
        suffix = '_s' + str(seed); pairing, gates = {}, {}
        for task in full.all_tasks():
            if not task['ratio']: continue
            name = task['id'] + suffix
            clean = f"{task['arm']}_{task['partition']}_clean" + suffix
            checks = dict(classification_prefix_equal=all(trajectories[name][r][k] == trajectories[clean][r][k]
                for r in range(25) for k in fields), round24_model_equal=bool(np.array_equal(anchors[name], anchors[clean])))
            if task['arm'] == 'TAD':
                checks['detector_prefix_equal'] = all(trajectories[name][r].get('tad_no_prior') ==
                    trajectories[clean][r].get('tad_no_prior') for r in range(25))
            pairing[task['id']] = dict(checks, passed=all(checks.values()), required=task['arm'] != 'Krum')
            if task['arm'] == 'FedAvg':
                c, a = cells[clean]['summary']['tail'], cells[name]['summary']['tail']
                drop, rise = c['accuracy'] - a['accuracy'], a['source_transfer_asr'] - c['source_transfer_asr']
                gates[task['id']] = dict(accuracy_drop=drop, asr_rise=rise, passed=drop >= .1-1e-12 and rise >= .3-1e-12)
        selected = {task['id']:cells[task['id'] + suffix] for task in full.all_tasks()}
        strict = full.assess(selected, gates, pairing)
        current = with_krum_exception(strict)
        seed_assessments[str(seed)] = dict(strict=strict, with_previous_krum_exception=current,
                                          pairing=pairing, fedavg_attack_gates=gates)
    full.verify_files(inputs); formal.verify(manifest)
    result.update(status='complete_audited_formal', fingerprint=manifest['fingerprint'],
        observed_utc=formal.utc(), audited_cells=cells, per_seed=seed_assessments,
        input_sha256=inputs, tad_reproduction=full.PAPER_CHOICES)
    formal.write(out / 'analysis/formal_results.json', result)
    return dict(status=result['status'], complete=len(cells), seeds=p['seeds'],
                official_test_used_for_selection=False)
