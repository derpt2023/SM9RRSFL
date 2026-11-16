"""Predeclared validation gates; incomplete and unfavorable cells stay visible."""
from __future__ import annotations

import math
from statistics import mean

from hfcdr_pilot_protocol import read, CONFIG, sha


def summarize(result):
    c, rows = result['config'], result['records']
    reasons = []
    complete = [r['round'] for r in rows] == list(range(c['rounds'] + 1))
    if not complete or result.get('stopped_round') != c['rounds']:
        reasons.append('incomplete_horizon')
    if result.get('cryptography') != 'absent' or result.get('official_test_used') is not False:
        reasons.append('wrong_experiment_scope')
    if result.get('nonfinite_updates') != 0:
        reasons.append('nonfinite_updates')
    if any(not math.isfinite(r[k]) or not 0 <= r[k] <= 1 for r in rows
           for k in ('accuracy', 'asr', 'source_accuracy')):
        reasons.append('invalid_metrics')
    actual = [r for r in rows if r['round'] > 0]
    criteria = read(CONFIG)['criteria']
    high_fp = [r['round'] for r in actual if
        r['false_positive_removals'] > criteria['max_honest_false_removal_rate'] * r['honest_clients']]
    empty = [r['round'] for r in actual if r['accepted_updates'] == 0]
    starved, streak = [], 0
    for r in actual:
        streak = streak + 1 if r['honest_weight_loss'] >= criteria['starvation_honest_weight_loss'] else 0
        if streak >= criteria['starvation_consecutive_rounds']:
            starved.append(r['round'])
    if high_fp:
        reasons.append('honest_false_removal_above_10pct')
    if empty:
        reasons.append('empty_aggregation')
    if starved:
        reasons.append('persistent_honest_weight_starvation')
    windows = {}
    for name, bounds in (('tail10', criteria['primary_window']), ('attack_period', criteria['secondary_window'])):
        selected = [r for r in rows if bounds[0] <= r['round'] <= bounds[1]]
        windows[name] = ({k: mean(r[k] for r in selected)
                         for k in ('accuracy', 'asr', 'source_accuracy')}
                        if [r['round'] for r in selected] == list(range(bounds[0], bounds[1] + 1)) else None)
    return dict(task_id=result['task_id'], complete=complete, healthy=not reasons,
                health_reasons=reasons, high_false_removal_rounds=high_fp,
                empty_rounds=empty, starvation_rounds=starved, **windows,
                final={k: rows[-1][k] for k in ('accuracy', 'asr', 'source_accuracy')} if rows else None)


def assess(ours, clean, fedavg, fedavg_clean, peers, criteria):
    """All inputs are paired tail-window summaries, never selected best rounds."""
    excess_fedavg = max(0., fedavg['asr'] - fedavg_clean['asr'])
    excess_ours = max(0., ours['asr'] - clean['asr'])
    reduction = 1 - excess_ours / excess_fedavg if excess_fedavg > 0 else None
    values = dict(fedavg_excess_asr=excess_fedavg, ours_excess_asr=excess_ours,
        ours_accuracy_drop=clean['accuracy'] - ours['accuracy'],
        ours_accuracy_gap_to_best_peer=max(p['accuracy'] for p in peers) - ours['accuracy'],
        excess_asr_reduction=reduction)
    # Tiny tolerance only handles decimal boundary roundoff, not scientific slack.
    eps = 1e-12
    checks = dict(attack_effective=excess_fedavg + eps >= criteria['fedavg_min_excess_asr'],
        accuracy_retained=values['ours_accuracy_drop'] <= criteria['ours_max_accuracy_drop'] + eps,
        accuracy_competitive=values['ours_accuracy_gap_to_best_peer'] <= criteria['ours_max_accuracy_gap_to_best_peer'] + eps,
        asr_retained=excess_ours <= criteria['ours_max_excess_asr'] + eps,
        attack_reduced=reduction is not None and reduction + eps >= criteria['min_excess_asr_reduction'])
    return dict(values=values, checks=checks, performance_pass=all(checks.values()))


def report(output, manifest):
    summaries, missing, failures = {}, [], []
    for task in manifest['tasks']:
        folder = output / 'tasks' / task['task_id']
        status_path = folder / 'status.json'
        if not status_path.exists():
            missing.append(task['task_id'])
            continue
        status = read(status_path)
        if status['status'] != 'complete':
            failures.append(dict(task_id=task['task_id'], status=status))
            continue
        result = read(folder / 'result.json')
        if (result['task_id'] != task['task_id'] or result['config'] != task['config']
                or status.get('manifest_fingerprint') != manifest['fingerprint']
                or sha(folder / 'result.json') != status['result_sha256']):
            raise ValueError('Result identity/hash differs from prepared task')
        summaries[task['task_id']] = summarize(result)
    comparisons = []
    by_key = {(t['config']['partition'], t['condition'], t['config']['method']): t['task_id']
              for t in manifest['tasks']}
    for part in manifest['policy']['partitions']:
        for condition in ('label_flip', 'scale2', 'scale5'):
            required = [(part, condition, m) for m in manifest['policy']['methods']]
            required += [(part, 'clean', m) for m in ('hfcdr', 'fedavg')]
            ready = all(by_key[k] in summaries and summaries[by_key[k]]['complete']
                        and summaries[by_key[k]]['tail10'] is not None for k in required)
            if not ready:
                comparisons.append(dict(partition=part, condition=condition, status='incomplete'))
                continue
            def get(cond, method):
                return summaries[by_key[(part, cond, method)]]
            outcome = assess(get(condition, 'hfcdr')['tail10'], get('clean', 'hfcdr')['tail10'],
                get(condition, 'fedavg')['tail10'], get('clean', 'fedavg')['tail10'],
                [get(condition, m)['tail10'] for m in manifest['policy']['methods'] if m != 'hfcdr'],
                manifest['policy']['criteria'])
            healthy = get(condition, 'hfcdr')['healthy'] and get('clean', 'hfcdr')['healthy']
            passed = outcome['performance_pass'] and healthy
            comparisons.append(dict(partition=part, condition=condition, **outcome,
                ours_health_pass=healthy, status='preliminary_pass' if passed else
                'attack_ineffective' if not outcome['checks']['attack_effective'] else 'not_passed'))
    return dict(schema='hfcdr_pilot_report_v1', complete=len(summaries) == 48 and not missing and not failures
                and all(s['complete'] for s in summaries.values()),
                summaries=summaries, missing=missing, failed_or_incomplete=failures,
                comparisons=comparisons, qualification='First development seed only; confirmation and formal evaluation are not run automatically.',
                statistical_replicates=1, official_test_used=False, all_outcomes_retained=True)
