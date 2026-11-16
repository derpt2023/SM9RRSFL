"""Unfiltered study summaries, paired attack gates and replicate statistics."""
from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, stdev

import hfcdr_validation_protocol as p


def summarize(result):
    c, rows = result['config'], result['records']
    reasons = []
    complete = ([r['round'] for r in rows] == list(range(c['rounds']+1))
                and result.get('stopped_round') == c['rounds'])
    if not complete:
        reasons.append('incomplete_horizon')
    if (result.get('cryptography') != 'absent' or result.get('official_test_used') is not False
            or result.get('evaluation_scope') != 'training_validation_all_source5'
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
                        if selected and [r['round'] for r in selected] == list(range(first,c['rounds']+1)) else None)
    for key, value in result.get('timings', {}).items():
        if key.endswith('_seconds') and (not math.isfinite(value) or value < 0):
            reasons.append('invalid_timing')
    return dict(task_id=result['task_id'], config=c, complete=complete, healthy=not reasons,
        health_reasons=reasons, high_false_removal_rounds=high_fp, empty_rounds=empty,
        starvation_rounds=starved, **windows,
        final={k: rows[-1][k] for k in ('accuracy','asr','source_accuracy')} if rows else None,
        timings=result.get('timings', {}), memory=result.get('memory', {}))


def attack_gates(summaries, manifest):
    """Paired within seed, then mean across seeds; never select a best round."""
    policy = manifest['policy']; rules = policy['screening']
    lookup = {(s['config']['dataset'],s['config']['partition'],s['config']['seed'],
        s['config']['attack'],s['config']['scale'],s['config']['malicious_ratio']):s
        for s in summaries.values()}
    cells, settings = [], []
    for attack in policy['attacks']:
        for dataset in policy['datasets']:
            for partition in policy['partitions']:
                for ratio in policy['ratios'][1:]:
                    pairs=[]
                    for seed in policy['seeds']:
                        a=lookup.get((dataset,partition,seed,attack['attack'],attack['scale'],ratio))
                        clean=lookup.get((dataset,partition,seed,'none',1.,0.))
                        if a is None or clean is None or not a['complete'] or not clean['complete']:
                            continue
                        pairs.append(dict(seed=seed,attack_task=a['task_id'],clean_task=clean['task_id'],
                            attacked_accuracy=a['tail10']['accuracy'],clean_accuracy=clean['tail10']['accuracy'],
                            attacked_asr=a['tail10']['asr'],clean_asr=clean['tail10']['asr'],
                            excess_asr=a['tail10']['asr']-clean['tail10']['asr'],
                            accuracy_drop=clean['tail10']['accuracy']-a['tail10']['accuracy'],
                            healthy=a['healthy'] and clean['healthy']))
                    complete=len(pairs)==len(policy['seeds'])
                    excess=[v['excess_asr'] for v in pairs]
                    passed=(complete and all(v['healthy'] for v in pairs)
                        and mean(excess)>=rules['mean_min_excess_asr']
                        and min(excess)>=rules['each_seed_min_excess_asr'])
                    cells.append(dict(condition=attack['name'],dataset=dataset,partition=partition,
                        malicious_ratio=ratio,reference=ratio in rules['reference_ratios'],pairs=pairs,
                        expected_n=len(policy['seeds']),n=len(pairs),complete=complete,
                        mean_excess_asr=mean(excess) if excess else None,
                        sd_excess_asr=stdev(excess) if len(excess)>1 else None,
                        pass_gate=passed,status='pass' if passed else 'failed' if complete else 'incomplete'))
        relevant=[c for c in cells if c['condition']==attack['name'] and c['reference']]
        settings.append(dict(condition=attack['name'],pass_gate=all(c['pass_gate'] for c in relevant),
            complete=all(c['complete'] for c in relevant),
            failed_cells=[{k:c[k] for k in ('dataset','partition','malicious_ratio','status','mean_excess_asr')}
                          for c in relevant if not c['pass_gate']]))
    return dict(cells=cells,attack_decisions=settings,
        all_attack_settings_pass=all(s['pass_gate'] for s in settings),
        next_stage_allowed=all(s['pass_gate'] for s in settings),
        scope='FedAvg attack effectiveness on training-derived validation only; not a defense ranking')


def report(output,manifest):
    summaries,missing,failures={},[],[]
    for task in manifest['tasks']:
        folder=output/'tasks'/task['task_id']
        if not (folder/'status.json').exists():
            missing.append(task['task_id']);continue
        status=p.read(folder/'status.json')
        if status['status']!='complete':
            failures.append(dict(task_id=task['task_id'],status=status));continue
        result=p.read(folder/'result.json')
        if (result['task_id']!=task['task_id'] or result['config']!=task['config']
                or status.get('manifest_fingerprint')!=manifest['fingerprint']
                or p.sha(folder/'result.json')!=status['result_sha256']):
            raise ValueError('Completed task provenance mismatch: '+task['task_id'])
        summaries[task['task_id']]=summarize(result)
    complete=len(summaries)==len(manifest['tasks']) and not missing and not failures
    gates=attack_gates(summaries,manifest) if manifest['policy']['phase']=='attack_screen' else {}
    if not complete:gates['next_stage_allowed']=False
    return dict(schema='hfcdr_validation_report_v1',fingerprint=manifest['fingerprint'],
        expected_tasks=len(manifest['tasks']),completed=len(summaries),complete=complete,
        missing=missing,failed_or_incomplete=failures,summaries=summaries,**gates,
        official_test_used=False,all_outcomes_retained=True,
        timing_caveat=manifest['policy']['scheduler']['timing_caveat'])

