"""Explain frozen health flags by attack phase without changing any decision."""
from __future__ import annotations


def context(result, summary):
    config = result['config']; onset = config['attack_start_round']
    attacked = config['attack'] != 'none' and config['malicious_ratio'] > 0
    groups = dict(warmup_only=[],crosses_onset=[],attack_phase_only=[],no_attack=[])
    for end in summary['starvation_rounds']:
        first = max(1,end-4)
        key = ('no_attack' if not attacked else 'warmup_only' if end < onset
               else 'crosses_onset' if first < onset else 'attack_phase_only')
        groups[key].append(dict(first_round=first,last_round=end))
    actual = [r for r in result['records'] if r['round'] > 0]
    false_removal = summary['high_false_removal_rounds']
    max_fp = max((r['false_positive_removals'] for r in actual),default=0)
    honest = actual[0]['honest_clients'] if actual else None
    return dict(task_id=result['task_id'],method=config['method'],attack_start_round=onset if attacked else None,
        raw_healthy=summary['healthy'],raw_health_reasons=summary['health_reasons'],
        starvation_windows=groups,
        starvation_exclusively_before_attack=bool(groups['warmup_only']) and not any(groups[k] for k in ('crosses_onset','attack_phase_only','no_attack')),
        high_false_removal_rounds_before_attack=[r for r in false_removal if attacked and r<onset],
        high_false_removal_rounds_in_attack_phase=[r for r in false_removal if attacked and r>=onset],
        high_false_removal_rounds_without_attack=false_removal if not attacked else [],
        max_declared_honest_false_removals=max_fp,declared_honest_clients=honest,
        max_declared_honest_false_removal_rate=max_fp/honest if honest else None,
        interpretation='Client identities are declared before training. Warmup-only identity-weight flags are not evidence of an attack-induced failure. Crossing windows contain both warmup and attack rounds.',
        changes_frozen_health_or_selection=False)
