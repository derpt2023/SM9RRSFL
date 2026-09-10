#!/usr/bin/env python3
"""Resumable, training-holdout Ours/VERT screening; never selects on test data.

All attack profiles and failures are retained. One defense policy per method
is selected across ALL screen profiles, then frozen on confirmation seeds.
This exploratory search is separate from the six-method formal evaluation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from statistics import fmean
import time

from sm9rrsfl.datasets import load_image_dataset, stratified_training_three_way_split
from sm9rrsfl.fl import ExperimentConfig, run_experiment
from sm9rrsfl.ours_calibration import CALIBRATION_ALGORITHM_VERSION, split_metadata
from sm9rrsfl.ours_policy import OursParameters
from sm9rrsfl.fair_tuning import _training_health_reasons


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def load_screen_data(spec):
    dataset = spec.get('dataset', 'mnist')
    if dataset not in ('mnist', 'cifar10'):
        raise ValueError('screen dataset must be mnist or cifar10')
    if dataset == 'cifar10':
        required = {'rounds', 'local_epochs', 'batch_size', 'lr', 'lr_decay',
                    'attack_epochs', 'attack_start_round', 'detector_window'}
        missing = required - set(spec['training'])
        if missing:
            raise ValueError('CIFAR screen requires explicit training settings: ' + ', '.join(sorted(missing)))
        for method, space in spec.get('candidates', {}).items():
            for parameters in space.values():
                config = ExperimentConfig(**{**spec['training'], **parameters, 'method': method})
                OursParameters.from_object(config).validate()
    return load_image_dataset(dataset, spec.get('data_dir', 'data/' + dataset),
        download=spec.get('download', False), train_limit=spec['train_samples'],
        test_limit=100, seed=spec['split_seed'])


def _initialize_worker(spec):
    global _WORKER_DATA
    import torch
    torch.set_num_threads(spec.get('cpu_threads', 1))
    original = load_screen_data(spec)
    _WORKER_DATA = split_screen_data(original, spec).calibration_dataset


def split_screen_data(data, spec):
    fraction = float(spec.get('validation_fraction', .05))
    if not 0 < fraction < .95:
        raise ValueError('validation_fraction must be in (0, .95)')
    return stratified_training_three_way_split(data, seed=spec['split_seed'],
        train_fraction=.95 - fraction, calibration_fraction=fraction, attack_fraction=.05)


def _execute(task, validation=None):
    path, fields, parameters = task
    path = Path(path)
    if path.exists():
        cached = json.loads(path.read_text())
        if cached['config'] != parameters or any(cached['summary'].get(k) != v for k, v in fields.items()):
            raise ValueError(f'cached experiment configuration mismatch: {path}')
        return cached['summary']
    started = time.monotonic()
    result = run_experiment(_WORKER_DATA if validation is None else validation,
                            ExperimentConfig(**parameters))
    row = dict(**fields, **summarize(result))
    row['runtime_seconds'] = time.monotonic() - started
    write_json(path, dict(config=parameters, summary=row,
                          rounds=[asdict(r) for r in result.records],
                          diagnostics=[asdict(d) for d in result.diagnostics]))
    return row


def summarize(result):
    start = result.config.attack_start_round or result.config.detector_window + 2
    attack = [r for r in result.records if r.round >= start]
    tail = attack[-10:]
    honest = result.config.num_clients - len(result.malicious_clients)
    health = list(_training_health_reasons(result))
    if result.stopped_round != result.config.rounds:
        health.append('round_completion_rate')
    if result.nonfinite_updates:
        health.append('nonfinite_updates')
    return {
        'final_accuracy': result.final_accuracy,
        'final_asr': result.records[-1].attack_target_success_rate,
        'postattack_accuracy': fmean(r.accuracy for r in attack) if attack else 0.,
        'postattack_asr': fmean(r.attack_target_success_rate for r in attack) if attack else 1.,
        'tail_accuracy': fmean(r.accuracy for r in tail) if tail else None,
        'tail_asr': fmean(r.attack_target_success_rate for r in tail) if tail else None,
        'false_revocation_rate': result.records[-1].false_positive_revocations / max(honest, 1),
        'honest_weight_loss': result.records[-1].honest_weight_loss,
        'postwarmup_history_admissions': sum(d.history_admitted for d in result.diagnostics
                                            if d.round > result.config.detector_window),
        'stopped_round': result.stopped_round,
        'health_reasons': health,
        'runtime_seconds': result.runtime_seconds,
    }


def select_candidates(rows, candidates):
    selected, scores = {}, {}
    for method, space in candidates.items():
        if method == 'fedavg':
            continue
        scored = []
        for name in space:
            subset = [r for r in rows if r['method'] == method and r['candidate'] == name]
            clean = [r for r in subset if r['profile'] == 'clean']
            attacked = [r for r in subset if r['profile'] != 'clean']
            drops = []
            for r in clean:
                baseline = next(b for b in rows if b['method'] == 'fedavg'
                                and b['partition'] == r['partition'] and b['seed'] == r['seed'])
                drops.append(baseline['final_accuracy'] - r['final_accuracy'])
            valid = bool(clean and attacked) and max(drops) <= .03 and not any(r['health_reasons'] for r in subset)
            # Explicit balanced utility, identical for both methods. Attack
            # strength is not optimized separately for one defense.
            utility = .5 * fmean(r['postattack_accuracy'] for r in attacked) + .5 * (1 - fmean(r['postattack_asr'] for r in attacked))
            scores[method + '/' + name] = dict(valid=valid, utility=utility, worst_clean_drop=max(drops))
            if valid:
                scored.append((utility, name))
        if scored:
            selected[method] = max(scored)[1]
    return selected, scores


def confirmation_report(rows, selected, spec=None):
    """Distinguish numerical wins from a policy that passed confirmation.

    A clean failure on a confirmation seed cannot be hidden by reporting
    only that seed's attacked scenarios. No reselection uses confirmation.
    """
    confirmed = [r for r in rows if r['phase'] == 'confirmation']
    def scenario(r):
        return r['partition'], r['seed'], r['profile'], r['ratio']
    expected = ({(p, s, a, r) for p in spec['partitions'] for s in spec['confirmation_seeds']
                 for a, r in [('clean', 0.)] + [(a, r) for a in spec['attacks'] for r in spec['ratios']]}
                if spec is not None else {scenario(r) for r in confirmed if r['method'] in selected})
    audit = {}
    for method in selected:
        subset = [r for r in confirmed if r['method'] == method]
        failures = []
        missing = expected - {scenario(r) for r in subset}
        if missing:
            failures.append(dict(reasons=['incomplete_confirmation'], missing_scenarios=sorted(missing)))
        if len(subset) != len({scenario(r) for r in subset}):
            failures.append(dict(reasons=['duplicate_confirmation']))
        if any(r['candidate'] != selected[method] for r in subset):
            failures.append(dict(reasons=['unfrozen_candidate']))
        for r in subset:
            reasons = list(r['health_reasons'])
            if r['profile'] == 'clean':
                baseline = next((b for b in confirmed if b['method'] == 'fedavg'
                    and b['profile'] == 'clean' and b['partition'] == r['partition']
                    and b['seed'] == r['seed']), None)
                if baseline is None:
                    reasons.append('missing_clean_reference')
                elif baseline['final_accuracy'] - r['final_accuracy'] > .03:
                    reasons.append('clean_accuracy_drop')
            if reasons:
                failures.append({k: r[k] for k in ('partition', 'seed', 'profile', 'ratio')}
                                | {'reasons': reasons})
        audit[method] = dict(status='not_run' if not subset else 'failed' if failures else 'passed',
                             candidate=selected[method], failures=failures)
    comparisons = []
    scenarios = sorted({(r['partition'], r['profile'], r['ratio']) for r in confirmed
                        if r['profile'] != 'clean'})
    for part, profile, ratio in scenarios:
        groups = {m: [r for r in confirmed if r['method'] == m and
            r['partition'] == part and r['profile'] == profile and r['ratio'] == ratio]
            for m in ('sm9rrs', 'vert')}
        a, v = groups['sm9rrs'], groups['vert']
        if not a or not v or {r['seed'] for r in a} != {r['seed'] for r in v}:
            continue
        da = fmean(r['postattack_accuracy'] for r in a) - fmean(r['postattack_accuracy'] for r in v)
        ds = fmean(r['postattack_asr'] for r in a) - fmean(r['postattack_asr'] for r in v)
        numeric_win = da > 0 and ds < 0
        comparisons.append(dict(partition=part, profile=profile, ratio=ratio,
            seeds=sorted(r['seed'] for r in a), accuracy_delta=da, asr_delta=ds,
            strictly_better_both=numeric_win,
            confirmation_passed=all(audit[m]['status'] == 'passed' for m in groups),
            validated_better_both=numeric_win and all(audit[m]['status'] == 'passed' for m in groups),
            ours_health_failures=[r['seed'] for r in a if r['health_reasons']]))
    return audit, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--config')
    inputs.add_argument('--report-dir', help='Rebuild confirmation audit from existing runs, without training')
    parser.add_argument('--clean-only', action='store_true',
                        help='Run only clean FedAvg on the same training holdout before an expensive screen')
    args = parser.parse_args()
    if args.clean_only and args.report_dir:
        parser.error('--clean-only requires --config')
    if args.report_dir:
        out = Path(args.report_dir)
        selected = json.loads((out / 'selection.json').read_text())['selected']
        spec = json.loads((out / 'manifest.json').read_text())['spec']
        rows = [json.loads(p.read_text())['summary'] for pattern in ('screen-*.json', 'confirmation-*.json')
                for p in sorted(out.glob(pattern))]
        audit, comparisons = confirmation_report(rows, selected, spec)
        write_json(out / 'confirmation_audit.json', audit)
        write_json(out / 'comparison.json', comparisons)
        if spec.get('performance_target'):
            from screen_performance import screen_performance_target
            _, target = screen_performance_target(out, spec, 'confirmation', selected)
            write_json(out / 'performance_target_confirmation.json', target)
        print('Reports rebuilt without training:', out)
        return
    spec = json.loads(Path(args.config).read_text())
    if spec.get('performance_target') is not None:
        from sm9rrsfl.performance_target import PerformanceTarget
        PerformanceTarget.parse(spec['performance_target'])
    if args.clean_only:
        spec = {**spec, 'attacks': {}, 'ratios': [], 'confirmation_seeds': [],
                'candidates': {'fedavg': {'reference': {}}}, 'clean_preflight_only': True}
    out = Path(spec['output_dir']); out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(spec.get('cpu_threads', 1))
    data = load_screen_data(spec)
    split = split_screen_data(data, spec)
    validation = split.calibration_dataset
    manifest = {'algorithm': CALIBRATION_ALGORITHM_VERSION, 'spec': spec,
                'split': split_metadata(split, spec['split_seed']),
                'default_ours': asdict(OursParameters()),
                'official_test_used': False, 'exploratory_only': True,
                'selection': 'same 0.5 accuracy + 0.5 (1-ASR); clean drop <= .03; health gates',
                'runner_digest': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'source_digest': hashlib.sha256(b''.join(p.read_bytes() for p in
                    sorted(Path('sm9rrsfl').glob('*.py')))).hexdigest()}
    if spec.get('performance_target'):
        manifest['target_runner_digest'] = hashlib.sha256(Path('screen_performance.py').read_bytes()).hexdigest()
        manifest['selection'] += '; freeze VERT, then select Ours against the declared target across all profiles'
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    out = out / key; out.mkdir(exist_ok=True)
    write_json(out / 'manifest.json', manifest)
    candidates = spec['candidates']
    rows = []
    pending = []

    def run(phase, method, name, profile, ratio, part, seed):
        identity = f'{phase}-{method}-{name}-{profile}-{ratio}-{part}-{seed}'
        path = out / (identity + '.json')
        config = ExperimentConfig(**{**spec['training'], **candidates[method][name],
            **spec['attacks'].get(profile, {}), 'method': method,
            'malicious_ratio': ratio, 'partition': part, 'seed': seed})
        pending.append((str(path), dict(phase=phase, method=method, candidate=name,
                        profile=profile, ratio=ratio, partition=part, seed=seed), asdict(config)))

    def record(row):
        rows.append(row)
        print('DONE', row['phase'], row['method'], row['candidate'], row['profile'],
              row['ratio'], row['partition'], row['seed'], 'acc/asr',
              round(row['final_accuracy'], 4), round(row['final_asr'], 4),
              'FP', row['false_revocation_rate'], flush=True)
        with (out / 'summary.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    def flush():
        jobs = spec.get('jobs', 1)
        print('PHASE tasks', len(pending), 'jobs', jobs, flush=True)
        if jobs == 1:
            for task in pending:
                record(_execute(task, validation))
        else:
            with ProcessPoolExecutor(max_workers=jobs, initializer=_initialize_worker,
                                     initargs=(spec,)) as pool:
                for future in as_completed([pool.submit(_execute, task) for task in pending]):
                    record(future.result())
        pending.clear()

    for part in spec['partitions']:
        for seed in spec['screen_seeds']:
            for method, space in candidates.items():
                for name in space:
                    run('screen', method, name, 'clean', 0., part, seed)
                    if method != 'fedavg':
                        for profile in spec['attacks']:
                            for ratio in spec['ratios']:
                                run('screen', method, name, profile, ratio, part, seed)
    flush()
    if args.clean_only:
        write_json(out / 'clean_preflight.json', {'official_test_used': False, 'runs': rows})
        print('Clean training-holdout preflight:', out, flush=True)
        return
    selected, scores = select_candidates(rows, candidates)
    if spec.get('performance_target'):
        from screen_performance import screen_performance_target
        selected, target = screen_performance_target(out, spec, 'screen', selected, scores=scores, choose=True)
        write_json(out / 'performance_target_screen.json', target)
    write_json(out / 'selection.json', dict(selected=selected, scores=scores,
               parameters={m: candidates[m][n] for m, n in selected.items()}))
    if len(selected) != 2:
        print('No feasible paired selection; all results retained at', out, flush=True)
        return
    for part in spec['partitions']:
        for seed in spec['confirmation_seeds']:
            for method, name in {**selected, 'fedavg': 'reference'}.items():
                run('confirmation', method, name, 'clean', 0., part, seed)
                if method != 'fedavg':
                    for profile in spec['attacks']:
                        for ratio in spec['ratios']:
                            run('confirmation', method, name, profile, ratio, part, seed)
    flush()
    audit, comparisons = confirmation_report(rows, selected, spec)
    write_json(out / 'confirmation_audit.json', audit)
    write_json(out / 'comparison.json', comparisons)
    if spec.get('performance_target'):
        from screen_performance import screen_performance_target
        _, target = screen_performance_target(out, spec, 'confirmation', selected)
        write_json(out / 'performance_target_confirmation.json', target)
    print('Results:', out, flush=True)


if __name__ == '__main__':
    main()
