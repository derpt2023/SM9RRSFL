"""Optional target audit for multi-profile development screens.

Kept outside the training package: adding CIFAR screening cannot change the
running MNIST tuner, model, checkpoints, or detector policy.
"""
from statistics import fmean
from pathlib import Path
import json

from sm9rrsfl.fl import ExperimentConfig, ExperimentResult, RoundRecord, _choose_malicious
from sm9rrsfl.performance_target import PerformanceTarget, evaluate_target


def restore_run(payload):
    c = ExperimentConfig(**payload['config'])
    s = payload['summary']
    records = [RoundRecord(**r) for r in payload['rounds']]
    return ExperimentResult(c, records, s['final_accuracy'], 1 - s['final_accuracy'],
        s['stopped_round'], _choose_malicious([f'client-{i}' for i in range(c.num_clients)],
        c.malicious_ratio, c.seed), (), nonfinite_updates=max((r.nonfinite_updates for r in records), default=0))


def screen_performance_target(directory, spec, phase, selected, *, scores=None, choose=False):
    """Audit every profile against a frozen VERT; only screen phase may select."""
    policy = PerformanceTarget.parse(spec.get('performance_target'))
    if policy is None:
        return dict(selected), {'status': 'disabled'}
    if choose and phase != 'screen':
        raise ValueError('confirmation data must never reselect parameters')
    if not {'sm9rrs', 'vert'} <= set(selected):
        return dict(selected), {'status': 'unmet', 'reason': 'no_healthy_paired_selection'}
    loaded = [(p, json.loads(p.read_text())) for p in sorted(Path(directory).glob(phase + '-*.json'))]
    seeds = spec['screen_seeds'] if phase == 'screen' else spec['confirmation_seeds']
    expected = {(p, spec['training'].get('dirichlet_alpha', .5), spec['training']['num_clients'], r, s)
                for p in spec['partitions'] for r in [0., *spec['ratios']] for s in seeds}
    names = list(spec['candidates']['sm9rrs']) if choose else [selected['sm9rrs']]

    def health_failures(method, name):
        subset = [v['summary'] for _, v in loaded if v['summary']['method'] == method
                  and v['summary']['candidate'] == name]
        failures = [] if subset else ['missing_runs']
        for row in subset:
            failures.extend(row['health_reasons'])
            if row['profile'] == 'clean':
                references = [v['summary'] for _, v in loaded if v['summary']['method'] == 'fedavg'
                              and v['summary']['profile'] == 'clean'
                              and v['summary']['partition'] == row['partition']
                              and v['summary']['seed'] == row['seed']]
                if len(references) != 1:
                    failures.append('missing_or_duplicate_clean_reference')
                elif references[0]['health_reasons']:
                    failures.append('unhealthy_clean_reference')
                elif references[0]['final_accuracy'] - row['final_accuracy'] > .03:
                    failures.append('clean_accuracy_drop')
        if phase == 'confirmation' and any(v['summary']['method'] == method
                and v['summary']['candidate'] != name for _, v in loaded):
            failures.append('unfrozen_candidate')
        if scores is not None and not scores[method + '/' + name]['valid']:
            failures.append('independent_selection_health_gate')
        return sorted(set(failures))

    candidates, ranked = {}, []
    for name in names:
        all_ours = [v for _, v in loaded if v['summary']['method'] == 'sm9rrs' and v['summary']['candidate'] == name]
        health = health_failures('sm9rrs', name)
        health_valid = not health
        profiles = {}
        for profile in spec['attacks']:
            def subset(method, candidate):
                return [restore_run(v) for _, v in loaded if v['summary']['method'] == method
                        and v['summary']['candidate'] == candidate
                        and v['summary']['profile'] in ('clean', profile)]
            profiles[profile] = evaluate_target(subset('sm9rrs', name), subset('vert', selected['vert']),
                                                policy, expected_scenarios=expected)
        all_pass = health_valid and bool(profiles) and all(a['status'] == 'passed' for a in profiles.values())
        candidates[name] = {'status': 'passed' if all_pass else 'unmet', 'health_valid': health_valid,
                            'health_failures': health, 'profiles': profiles}
        if health_valid and profiles and all(a['structurally_complete'] for a in profiles.values()):
            attacked = [v['summary'] for v in all_ours if v['summary']['profile'] != 'clean']
            rank = (all_pass, -max(a['worst_normalized_excess'] for a in profiles.values()),
                    -fmean(a['mean_normalized_excess'] for a in profiles.values()),
                    -fmean(r['postattack_asr'] for r in attacked),
                    fmean(r['postattack_accuracy'] for r in attacked), name)
            ranked.append((rank, name))
    chosen = max(ranked)[1] if choose and ranked else selected['sm9rrs']
    status = candidates[chosen]['status']
    vert_health = health_failures('vert', selected['vert'])
    if vert_health:
        status = 'unmet'
    return {**selected, 'sm9rrs': chosen}, {
        'status': status, 'phase': phase, 'official_test_used': False,
        'parameters_reselected': bool(choose), 'policy': vars(policy),
        'independent_selection': selected, 'selected': {**selected, 'sm9rrs': chosen},
        'vert_health_failures': vert_health, 'candidates': candidates,
    }
