#!/usr/bin/env python3
"""Independent formal CIFAR-10 evaluation of the user-accepted, fixed H84.

No search, test-driven selection, outcome pruning or automatic partial retry.
The frozen development sources are used read-only in isolated worker processes.
"""
import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

import run_cifar_cp1_cuda_development as development
from run_cifar_cp1_cuda_development import (ROOT, TERMINAL, read, write, sha, digest,
    utc, verify, gpu_snapshot, qualified_gpus, guard_task_directory)
import run_cifar_h84_full_validation as full
from watch_cifar_experiment import timing

DEFAULT = ROOT / 'configs/cifar10_h84_formal150_v1.json'
OWN = ('run_cifar_h84_formal.py', 'cifar_h84_formal_scheduler.py',
       'cifar_h84_formal_report.py', 'watch_cifar_experiment.py',
       'tests/test_h84_formal.py', 'tests/test_cifar_run_observer.py')


def specs_for(p):
    parent = read(ROOT / p['development_config'])
    if sha(ROOT / p['development_config']) != p['development_config_sha256']:
        raise ValueError('Accepted development configuration changed')
    # Revalidate the original scientific settings, without using test outcomes.
    originals = development.specs_for(dict(parent, tasks=full.all_tasks()))
    seeds = p['seeds']
    if (not seeds or len(seeds) != len(set(seeds))
            or any(type(s) is not int or not 0 < s < 2**32 or s == 2026102201 for s in seeds)):
        raise ValueError('Distinct new positive seeds required; development seed forbidden')
    if p['evaluation'] != dict(scope='official_test', count=10000, source_count=1000,
            source_label=5, target_label=7, selection=False, refit_on_validation=False):
        raise ValueError('Official-test evaluation contract changed')
    if not p['acceptance']['post_result_user_acceptance'] or p['acceptance']['original_passed_scenarios'] != 8:
        raise ValueError('Preserve the original 8/10 result and subsequent human acceptance')
    if (p['result_pruning'] or p['retune_on_test'] or p['reuse_development_cells']
            or p['rounds'] != 150 or p['candidate'] != 'H84'):
        raise ValueError('Only fixed H84, fresh 150-round final evaluation is supported')
    if Path(p['output_dir']).is_absolute() or '..' in Path(p['output_dir']).parts:
        raise ValueError('Output must be a dedicated repository-relative directory')
    if not p['output_dir'].startswith('outputs/cifar_h84_formal'):
        raise ValueError('Do not overwrite a development directory')
    specs = []
    for seed in seeds:
        for old in originals:
            s = deepcopy(old)
            s['task_id'] += '_s' + str(seed)
            s['config']['seed'] = seed
            specs.append(s)
    return specs


def plan(config=DEFAULT):
    p = read(config); specs = specs_for(p)
    parent = read(ROOT / p['development_config'])
    source = dict(parent['frozen_source_sha256'])
    for name, expected in source.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen development source changed: ' + name)
    names = set(OWN) | {p['development_config'], str(Path(config).resolve().relative_to(ROOT))}
    source.update({n: sha(ROOT / n) for n in names})
    data = dict(parent['data_config'], data_dir=str(ROOT / parent['data_config']['data_dir']))
    runtime = dict(p, numerics=parent['numerics'], gpu_admission=p['gpu_admission'],
                   selection_policy=parent['selection_policy'])
    identity = dict(protocol=p, specs=specs, data_config=data, sources=source,
        official_test_used_for_selection=False, evaluation_scope='official_test',
        maximum_rounds=len(specs) * 150)
    return runtime, specs, dict(identity, fingerprint=digest(identity))


def official_data(data_config, expected_validation):
    from scripts.experiments.cifar10 import attack_stability_worker as w
    raw = w.load_image_dataset(data_config['dataset'], data_config['data_dir'], download=False,
        train_limit=data_config['train_samples'], test_limit=data_config['test_samples'],
        seed=data_config['seed'])
    split = w.stratified_training_three_way_split(raw, seed=data_config['split_seed'],
        train_fraction=.95 - data_config['validation_fraction'],
        calibration_fraction=data_config['validation_fraction'], attack_fraction=.05)
    contract = w.split_metadata(split, data_config['split_seed'])
    if w._dataset_identity(split.calibration_dataset, contract) != expected_validation:
        raise ValueError('Training/validation/auxiliary arrays changed from development')
    data = split.main_dataset
    if len(data.y_train) != 40000 or len(data.y_test) != 10000 or len(data.y_attack) != 2500:
        raise ValueError('Formal 40000/10000/2500 split required; no validation refit')
    if int((data.y_test == 5).sum()) != 1000:
        raise ValueError('Official source-class denominator must be 1000')
    return data, contract


def references(p, manifest):
    """Read immutable development evidence before any official-test training."""
    parent = read(ROOT / p['development_config'])
    refs = dict(p['development_evidence'])
    for name, expected in refs.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Accepted development evidence changed: ' + name)
    report = read(ROOT / p['development_report'])
    if report['status'] != 'complete_audited_development' or len(report['cells']) != 60:
        raise ValueError('Complete accepted development report required')
    environment = full.environment_key(read(ROOT / parent['output_dir'] /
        'environments/TAD_iid_clean.json'))
    data, contract = official_data(manifest['data_config'], parent['expected_dataset_identity'])
    from scripts.experiments.cifar10 import attack_stability_worker as w
    return dict(references=refs, expected_environment=environment,
        dataset_identity=w._dataset_identity(data, contract), acceptance=p['acceptance'],
        official_test_used_for_selection=False)


@contextmanager
def formal_observation(p, manifest, lock, folder):
    """Only evaluation and instrumentation change; training and attack are frozen."""
    from scripts.experiments.cifar10 import attack_stability_worker as w
    from run_cifar_targeted_optimization import metric_scope
    from cifar_optimized_station_runtime import aggregation_observation
    parent = read(ROOT / p['development_config'])
    loader, atomic, writer, pickler = w._load_dataset, w._atomic, w._write_json, w._write_pickle
    measured = dict(structured_persistence_seconds=0., checkpoint_write_seconds=0.)
    started = time.monotonic()

    def load(config):
        data, contract = official_data(config, parent['expected_dataset_identity'])
        if w._dataset_identity(data, contract) != lock['dataset_identity']:
            raise ValueError('Formal data content changed after preflight')
        return data, contract

    def atomic_measured(path, callback):
        before = time.monotonic()
        try:
            return atomic(path, callback)
        finally:
            elapsed = time.monotonic() - before
            measured['structured_persistence_seconds'] += elapsed
            if Path(path).name == 'checkpoint.pickle':
                measured['checkpoint_write_seconds'] += elapsed

    def pickle_measured(path, value):
        if Path(path).name == 'result.pickle':
            # Capture at the exact original result boundary, before result-file I/O.
            measured['at_result_boundary'] = dict(measured)
        return pickler(path, value)

    def json_labelled(path, value):
        value = deepcopy(value) if Path(path).name == 'result.json' else value
        if Path(path).name == 'result.json':
            value['qualification'] = 'fixed accepted H84 formal evaluation; audit and health reported separately'
        return writer(path, value)

    with metric_scope():
        evaluator = w.RoundMetricEvaluator
        class Evaluator(evaluator):
            def target_manifest(self):
                d = super().target_manifest()
                d['legacy_evaluation_indices'] = d.pop('legacy_validation_indices')
                d['source_transfer_indices'] = d.pop('transfer_validation_indices')
                return dict(d, evaluation_scope='official_test', evaluation_count=10000,
                            official_test_used_for_selection=False)

            def evaluate(self, state):
                row = super().evaluate(state)
                row['blacklisted_client_ids'] = sorted(state['blacklisted'])
                row['aggregation_coefficients'] = aggregation_observation(state, self.config.method)
                row['cumulative_stage_timings'] = dict(state['timings'])
                row['worker_elapsed_seconds_at_evaluation'] = time.monotonic() - started
                row['checkpoint_write_seconds_before_round_save'] = measured['checkpoint_write_seconds']
                return row
        w.RoundMetricEvaluator = Evaluator; w._load_dataset = load
        w._atomic = atomic_measured; w._write_pickle = pickle_measured; w._write_json = json_labelled
        try:
            yield measured
        finally:
            w.RoundMetricEvaluator = evaluator; w._load_dataset = loader
            w._atomic = atomic; w._write_pickle = pickler; w._write_json = writer


def run_worker(p, spec, manifest, out):
    verify(manifest)
    name = spec['task_id']; folder = out / 'tasks' / name
    if guard_task_directory(folder) != 'new':
        raise ValueError('Formal worker only accepts fresh tasks')
    lock = read(out / 'reference_lock.json')
    full.verify_files(lock['references'])
    import torch
    n = p['numerics']
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != n['cublas_workspace_config']:
        raise ValueError('CUBLAS policy must be set before worker import')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Exactly one selected CUDA device required')
    full.check_runtime(lock['expected_environment'])
    torch.set_num_threads(n['torch_threads']); torch.use_deterministic_algorithms(n['deterministic_algorithms'])
    torch.backends.cudnn.deterministic = n['cudnn_deterministic']
    torch.backends.cudnn.benchmark = n['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = n['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = n['cudnn_allow_tf32']
    env = dict(lock['expected_environment'], gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
    write(out / 'environments' / (name + '.json'), env)
    req = dict(spec, kind='train', device='cuda:0', data_config=manifest['data_config'],
               fingerprint=manifest['fingerprint'], output_dir=str(out / 'tasks'))
    write(out / 'requests' / (name + '.json'), req)
    from cifar_alternating_target_views import target_view_scope
    from cifar_tad_no_prior import prior_free_tad_scope
    from scripts.experiments.cifar10 import attack_stability_worker as w
    with target_view_scope(), formal_observation(p, manifest, lock, folder) as measured, (
            prior_free_tad_scope() if spec['config']['method'] == 'ding13' else nullcontext()):
        state = w.execute_task(req)
    if state['status'] == 'complete':
        record = timing(state['summary'])
        if not record['available_and_consistent']:
            raise ValueError('Missing or inconsistent final crypto-wall measurements')
        record.update(measured_persistence=measured['at_result_boundary'],
            definition='Worker elapsed from task setup through final in-memory result, including data loading, '
                       'training, attack, evaluation, diagnostics, audit and checkpoint I/O. Excluding-crypto '
                       'subtracts only four non-overlapping crypto wall spans. Not a crypto-disabled rerun.',
            shared_gpu=True, isolated_speed_comparison=False,
            checkpoint_io_field_note='Use measured_persistence.checkpoint_write_seconds; inherited summary checkpoint_io_seconds is not populated by this worker.')
        write(folder / 'timing.json', record)
    verify(manifest)
    return 0 if state['status'] in TERMINAL else 2


def summarize(p, specs, manifest, out):
    from cifar_h84_formal_report import summarize as report
    return report(p, specs, manifest, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--max-gpus', type=int, default=8)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--batch', action='store_true'); group.add_argument('--ready', action='store_true')
    group.add_argument('--task'); group.add_argument('--summary', action='store_true')
    group.add_argument('--finish', action='store_true')
    args = parser.parse_args(); p, specs, manifest = plan(args.config)
    out = ROOT / p['output_dir']
    if args.task:
        if read(out / 'manifest.json') != manifest:
            raise ValueError('Frozen formal manifest required')
        raise SystemExit(run_worker(p, next(s for s in specs if s['task_id'] == args.task), manifest, out))
    if not any((args.batch, args.ready, args.summary, args.finish)):
        print(json.dumps(dict(fingerprint=manifest['fingerprint'], tasks=len(specs),
            seeds=p['seeds'], maximum_rounds=manifest['maximum_rounds'], training_started=False)))
        return
    if args.summary:
        print(json.dumps(summarize(p, specs, manifest, out), indent=2)); return
    if args.ready or args.batch:
        lock = references(p, manifest)
        full.check_runtime(lock['expected_environment'])
        if args.ready:
            print(json.dumps(dict(ready=True, tasks=len(specs), data=lock['dataset_identity']))); return
        from run_cifar_w85_history_admission import prepare
        prepare(out, manifest, lock)
        from cifar_h84_formal_scheduler import run
        run(args.config, p, specs, manifest, out, args.max_gpus)
        if read(out / 'controller_status.json')['status'] != 'panel_terminal': return
    elif read(out / 'manifest.json') != manifest:
        raise ValueError('Existing formal manifest required')
    from cifar_h84_formal_report import finish
    print(json.dumps(finish(p, specs, manifest, out), indent=2))


if __name__ == '__main__':
    main()
