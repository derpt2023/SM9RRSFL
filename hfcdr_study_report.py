"""Unfiltered study summaries, paired attack gates and replicate statistics."""
from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, stdev

import hfcdr_study_protocol as p
from hfcdr_pilot_report import assess


def summarize(result):
    c, rows = result['config'], result['records']
    reasons = []
    complete = ([r['round'] for r in rows] == list(range(c['rounds']+1))
                and result.get('stopped_round') == c['rounds'])
    if not complete:
        reasons.append('incomplete_horizon')
    if (result.get('cryptography') != 'absent' or result.get('official_test_used') is not True
            or result.get('evaluation_scope') != 'official_test_all_source5'
            or result.get('official_test_used_for_selection') is not False):
        reasons.append('wrong_experiment_scope')
    if result.get('nonfinite_updates') != 0:
        reasons.append('nonfinite_updates')
    if any(not math.isfinite(r[k]) or not 0 <= r[k] <= 1 for r in rows
           for k in ('accuracy', 'asr', 'source_accuracy')):
        reasons.append('invalid_metrics')
    actual = [r for r in rows if r['round'] > 0]
    high_fp = [r['round'] for r in actual if r['false_positive_removals'] > .1*r['honest_clients']]
    empty = [r['round'] for r in actual if r['accepted_updates'] == 0]
    starved, streak = [], 0
    for r in actual:
        low_support = r['honest_weight_loss'] >= .99
        # Krum intentionally selects exactly one update. With 100 honest clients,
        # the nominal per-client loss is .99 even when that update has full mass.
        # Preserve the raw diagnostic, but do not call this healthy clean design
        # starvation. A selected malicious-only update still has zero honest mass.
        if c['method'] == 'krum' and r.get('malicious_weight_mass', 1.) < 1.:
            low_support = False
        streak = streak+1 if low_support else 0
        if streak >= 5:
            starved.append(r['round'])
    if high_fp: reasons.append('honest_false_removal_above_10pct')
    if empty: reasons.append('empty_aggregation')
    if starved: reasons.append('persistent_honest_weight_starvation')
    windows = {}
    for name, first in [('tail10', c['rounds']-9), ('attack_period', c['attack_start_round'])]:
        selected = [r for r in rows if first <= r['round'] <= c['rounds']]
        windows[name] = ({k: mean(r[k] for r in selected) for k in ('accuracy', 'asr', 'source_accuracy')}
                        if [r['round'] for r in selected] == list(range(first,c['rounds']+1)) else None)
    for key, value in result.get('timings', {}).items():
        if key.endswith('_seconds') and (not math.isfinite(value) or value < 0):
            reasons.append('invalid_timing')
    return dict(task_id=result['task_id'], config=c, complete=complete, healthy=not reasons,
        health_reasons=reasons, high_false_removal_rounds=high_fp, empty_rounds=empty,
        starvation_rounds=starved, **windows,
        final={k: rows[-1][k] for k in ('accuracy','asr','source_accuracy')} if rows else None,
        timings=result.get('timings', {}), memory=result.get('memory', {}))


def report(output, manifest):
    summaries, missing, failures = {}, [], []
    for task in manifest['tasks']:
        folder = output/'tasks'/task['task_id']
        if not (folder/'status.json').exists():
            missing.append(task['task_id']); continue
        status = p.read(folder/'status.json')
        if status['status'] != 'complete':
            failures.append(dict(task_id=task['task_id'], status=status)); continue
        result = p.read(folder/'result.json')
        if (result['task_id'] != task['task_id'] or result['config'] != task['config']
                or status.get('manifest_fingerprint') != manifest['fingerprint']
                or p.sha(folder/'result.json') != status['result_sha256']):
            raise ValueError('Completed task provenance mismatch: '+task['task_id'])
        summaries[task['task_id']] = summarize(result)
    by_key = {(t['config']['dataset'],t['config']['partition'],t['config']['seed'],t['condition'],
               t['config']['malicious_ratio'],t['config']['method']): t['task_id'] for t in manifest['tasks']}
    comparisons = []
    for dataset in manifest['policy']['datasets']:
        for part in manifest['policy']['partitions']:
            for seed in manifest['policy']['seeds']:
                for a in manifest['policy']['attacks']:
                    for ratio in manifest['policy']['ratios'][1:]:
                        def get(condition, method):
                            key = (dataset,part,seed,condition,0. if condition=='clean' else ratio,method)
                            return summaries.get(by_key[key])
                        attacked = [get(a['name'],m) for m in manifest['policy']['methods']]
                        required = attacked + [get('clean','hfcdr'),get('clean','fedavg')]
                        cell = dict(dataset=dataset,partition=part,seed=seed,condition=a['name'],malicious_ratio=ratio)
                        if any(x is None or not x['complete'] or x['tail10'] is None for x in required):
                            comparisons.append(dict(cell,status='incomplete')); continue
                        ours = get(a['name'],'hfcdr'); clean = get('clean','hfcdr')
                        outcome = assess(ours['tail10'], clean['tail10'],get(a['name'],'fedavg')['tail10'],
                            get('clean','fedavg')['tail10'],
                            [s['tail10'] for s in attacked if s['config']['method']!='hfcdr'],manifest['criteria'])
                        healthy = ours['healthy'] and clean['healthy']
                        comparisons.append(dict(cell,**outcome,ours_health_pass=healthy,
                            status='pass' if outcome['performance_pass'] and healthy else
                            'attack_ineffective' if not outcome['checks']['attack_effective'] else 'not_passed'))
    groups = defaultdict(list)
    for summary in summaries.values():
        c=summary['config']
        key=(c['dataset'],c['partition'],c['attack'],c['scale'],c['malicious_ratio'],c['method'])
        groups[key].append(summary)
    aggregate=[]
    for key, values in sorted(groups.items()):
        row=dict(zip(('dataset','partition','attack','scale','malicious_ratio','method'),key))
        row.update(n=len(values), seeds=sorted(v['config']['seed'] for v in values),
                   healthy_n=sum(v['healthy'] for v in values), expected_n=3, all_replicates_available=len(values)==3)
        for window in ('tail10','attack_period','final'):
            for metric in ('accuracy','asr','source_accuracy'):
                raw=[v[window][metric] for v in values if v[window] is not None]
                row[f'{window}_{metric}']=dict(mean=mean(raw) if raw else None,
                    sd=stdev(raw) if len(raw)>1 else None,n=len(raw))
        for metric in ('runtime_seconds','local_training_seconds','defense_aggregation_seconds','evaluation_seconds'):
            raw=[v['timings'][metric] for v in values]
            row[metric]=dict(mean=mean(raw),sd=stdev(raw) if len(raw)>1 else None,n=len(raw))
        aggregate.append(row)
    return dict(schema='hfcdr_attack_study_report_v1',fingerprint=manifest['fingerprint'],
        expected_tasks=936,completed=len(summaries),missing=missing,failed_or_incomplete=failures,
        complete=len(summaries)==936 and not missing and not failures,
        summaries=summaries,aggregate=aggregate,comparisons=comparisons,statistical_replicates=3,
        all_outcomes_retained=True,official_test_used_for_selection=False,
        timing_caveat=manifest['policy']['scheduler']['timing_caveat'],
        qualification='Descriptive fixed-protocol test results, not retuning feedback. All ratios, seeds and failures retained. SD uses n-1; n is shown explicitly.')
