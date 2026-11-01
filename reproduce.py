#!/usr/bin/env python3
"""Re-run the two fixed final matrices, without replaying parameter searches."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CONFIGS = ROOT / 'configs/reproduction'
NAMES = ('cifar-best', 'mnist_v7_target_fair_tuning')
QUARANTINED = {'GPU-430587b5-b622-2d06-0bb1-0a2e4e4b4c7c',
               'GPU-5d9f949a-f126-67b3-5c96-af785e33aa5c'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specs(protocol, methods):
    unknown = set(methods) - set(protocol['methods'])
    if unknown:
        raise ValueError('Unknown method(s): ' + ', '.join(sorted(unknown)))
    result = []
    for seed, part, ratio, method in itertools.product(
            protocol['seeds'], protocol['partitions'], protocol['ratios'], methods):
        base = protocol['methods'][method]
        c = dict(base['config'], seed=seed, partition=part, malicious_ratio=ratio)
        label = base.get('arm', method)
        tag = 'clean' if ratio == 0 else 'r' + str(round(ratio * 100))
        result.append(dict(task_id=f'{label}_{part}_{tag}_s{seed}', config=c,
                           profile=base.get('profile')))
    if len(methods) != len(set(methods)):
        raise ValueError('Duplicate methods would repeat an experiment')
    return result


def verify_sources(protocol):
    frozen = read(CONFIGS / 'cifar-best.json')['scientific_sources']
    for name, digest in frozen.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen scientific source changed: ' + name)
    files = read(CONFIGS / 'source_files.json')
    return {name: sha(ROOT / name) for name in files}


def load_data(protocol, data_dir, download=False):
    from sm9rrsfl.datasets import load_image_dataset, stratified_training_three_way_split
    from sm9rrsfl.ours_calibration import split_metadata
    d = protocol['data_config']
    raw = load_image_dataset(d['dataset'], data_dir, download=download,
                             train_limit=d['train_samples'], test_limit=d['test_samples'],
                             seed=d['seed'])
    split = stratified_training_three_way_split(raw, seed=d['split_seed'],
            train_fraction=.95 - d['validation_fraction'],
            calibration_fraction=d['validation_fraction'], attack_fraction=.05)
    contract = split_metadata(split, d['split_seed'])
    expected = protocol.get('training_data_contract',
                            protocol.get('official_dataset_identity', {}).get('contract'))
    if contract != expected:
        raise ValueError('Training/validation/attack split differs from accepted experiment')
    if protocol['dataset'] == 'cifar10':
        from scripts.experiments.cifar10.attack_stability_worker import _dataset_identity
        if _dataset_identity(split.main_dataset, contract) != protocol['official_dataset_identity']:
            raise ValueError('CIFAR official dataset arrays differ')
        if _dataset_identity(split.calibration_dataset, contract) != protocol['expected_dataset_identity']:
            raise ValueError('CIFAR validation dataset arrays differ')
    return split.main_dataset, contract


def runtime(protocol, allow_difference):
    import numpy
    import torch
    import sklearn
    import scipy
    from sklearn.cluster import KMeans  # also detect incompatible NumPy/sklearn binaries
    from sm9rrsfl import _native_sm3, _native_sm9  # fail early if the real bridges are absent
    if not torch.cuda.is_available():
        raise ValueError('Final experiment reproduction requires CUDA')
    current = dict(python=sys.version, numpy=str(numpy.__version__), sklearn=str(sklearn.__version__),
        scipy=str(scipy.__version__),
        torch=str(torch.__version__), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
        native_binary_sha256={str(p.relative_to(ROOT)): sha(p)
                              for p in (ROOT / 'sm9rrsfl').glob('_native_*.so')})
    original = protocol.get('original_environment', {})
    differences = {k: dict(original=v, current=current.get(k)) for k, v in original.items()
                   if k != 'numerics' and current.get(k) != v}
    if differences and not allow_difference:
        raise ValueError('Environment differs from original CIFAR run: ' +
            ', '.join(differences) + '. Use --allow-environment-difference for a recorded portability run.')
    return dict(current=current, original_differences=differences,
                historical_environment_known=bool(original))


def gpu_rows():
    text = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    return [dict(index=parts[0], uuid=parts[1], free=int(parts[2]))
            for line in text.strip().splitlines()
            for parts in [[s.strip() for s in line.split(',')]]]


def safe_output(path):
    output = Path(path).expanduser().resolve()
    for protected in (ROOT, ROOT / 'outputs', ROOT / 'outputs/old',
                      ROOT / 'outputs/cifar-best', ROOT / 'outputs/mnist_v7_target_fair_tuning'):
        if output == protected or (protected.name in NAMES and protected in output.parents):
            raise ValueError('Use a separate fresh reproduction directory')
    return output


def make_manifest(protocol, selected, data_dir):
    payload = dict(protocol=protocol, specs=selected, data_dir=str(data_dir),
                   sources=verify_sources(protocol), kind='fixed_parameters_reproduction_v1')
    payload['fingerprint'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def run_task(args, protocol, task, out):
    manifest = read(out / 'manifest.json')
    if manifest != make_manifest(protocol, manifest['specs'], Path(manifest['data_dir'])):
        raise ValueError('Reproduction manifest/source/data path changed')
    folder = out / 'tasks' / task['task_id']
    if folder.exists():
        raise ValueError('Existing task is retained; automatic retries/resume are disabled')
    env = runtime(protocol, args.allow_environment_difference)
    write(out / 'environments' / (task['task_id'] + '.json'), env)
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError('Worker requires exactly one visible GPU')
    torch.set_num_threads(2)
    if protocol['dataset'] == 'cifar10':
        n = protocol['numerics']
        torch.use_deterministic_algorithms(n['deterministic_algorithms'])
        torch.backends.cudnn.deterministic = n['cudnn_deterministic']
        torch.backends.cudnn.benchmark = n['cudnn_benchmark']
        torch.backends.cuda.matmul.allow_tf32 = n['cuda_matmul_allow_tf32']
        torch.backends.cudnn.allow_tf32 = n['cudnn_allow_tf32']
        from cifar_alternating_target_views import target_view_scope
        from cifar_tad_no_prior import prior_free_tad_scope
        from cifar_vert_paper_core import paper_scope, bounded_checkpoints, actual_weights_scope
        from run_cifar_h84_formal import formal_observation
        from scripts.experiments.cifar10 import attack_stability_worker as w
        from watch_cifar_experiment import timing
        p = dict(development_config=str(CONFIGS / 'cifar-best.json'))
        lock = dict(dataset_identity=protocol['official_dataset_identity'])
        request = dict(task, kind='train', device='cuda:0',
            data_config=dict(protocol['data_config'], data_dir=manifest['data_dir']),
            fingerprint=manifest['fingerprint'], output_dir=str(out / 'tasks'))
        write(out / 'requests' / (task['task_id'] + '.json'), request)
        with ExitStack() as stack:
            stack.enter_context(bounded_checkpoints(protocol['rounds']))
            stack.enter_context(actual_weights_scope(folder / 'actual_weights.jsonl', task['task_id']))
            if task['config']['method'] == 'vert':
                stack.enter_context(paper_scope(folder / 'vert_selection.jsonl', task['task_id']))
            stack.enter_context(target_view_scope())
            measured = stack.enter_context(formal_observation(p, manifest, lock, folder))
            if task['config']['method'] == 'ding13':
                stack.enter_context(prior_free_tad_scope())
            status = w.execute_task(request)
        if status['status'] != 'complete':
            raise RuntimeError('Task failed; evidence retained, no automatic retry')
        result = read(folder / 'result.json')
        result['qualification'] = 'Fresh fixed-parameter reproduction; not the historical accepted audit'
        write(folder / 'result.json', result)
        timing_record = timing(status['summary'])
        if not timing_record['available_and_consistent']:
            raise ValueError('Inconsistent crypto/wall timing; task evidence retained')
        write(folder / 'timing.json', dict(timing_record, measured_persistence=measured))
    else:
        from sm9rrsfl.fl import ExperimentConfig
        from sm9rrsfl.experiments import run_measured_experiment
        dataset, contract = load_data(protocol, Path(manifest['data_dir']))
        folder.mkdir(parents=True)
        write(folder / 'task.json', task)
        write(folder / 'dataset_identity.json', contract)
        write(folder / 'status.json', dict(status='running', pid=os.getpid()))
        try:
            result = run_measured_experiment(dataset, ExperimentConfig(**task['config']),
                checkpoint_dir=folder / 'checkpoints', run_fingerprint=manifest['fingerprint'],
                task_id=task['task_id'])
            with (folder / 'result.pickle').open('wb') as stream:
                pickle.dump(result, stream)
            write(folder / 'result.json', result.summary_dict())
            if result.stopped_round != protocol['rounds']:
                raise ValueError('Incomplete horizon')
            write(folder / 'status.json', dict(status='complete', completed_round=result.stopped_round,
                                              nonfinite_updates=result.nonfinite_updates))
        except BaseException as error:
            write(folder / 'status.json', dict(status='failed', error=str(error)))
            raise
    if verify_sources(protocol) != manifest['sources']:
        raise ValueError('Source changed during task')


def official_result(result, rows):
    """A separate reporting view; never mutate raw auxiliary-target records."""
    by_round = {row['round']: row for row in rows}
    if len(rows) != len(result.records) or set(by_round) != {record.round for record in result.records}:
        raise ValueError('Official metrics do not cover the complete result trajectory')
    records = [replace(record, accuracy=by_round[record.round]['accuracy'],
        error=by_round[record.round]['error'],
        attack_target_success_rate=by_round[record.round]['source_transfer_asr'],
        attack_target_confidence=by_round[record.round]['source_transfer_confidence'])
        for record in result.records]
    return replace(result, records=records, final_accuracy=records[-1].accuracy,
                   final_error=records[-1].error)


def summarize(out):
    from sm9rrsfl.experiments import write_result_files
    from sm9rrsfl.fair_tuning import _write_final_aggregate, _training_health_reasons
    from sm9rrsfl.visualization import generate_visualizations
    m = read(out / 'manifest.json')
    results, status, official, official_results = [], [], [], []
    for task in m['specs']:
        folder = out / 'tasks' / task['task_id']
        s = read(folder / 'status.json') if (folder / 'status.json').exists() else dict(status='pending')
        entry = dict(task_id=task['task_id'], status=s['status'])
        if s['status'] == 'complete':
            with (folder / 'result.pickle').open('rb') as stream:
                result = pickle.load(stream)  # own local experiment artifacts only
            entry['health_failures'] = _training_health_reasons(result)
            results.append(result)
            if (folder / 'metrics.json').exists():
                rows = read(folder / 'metrics.json')['rounds']
                official.extend(dict(task_id=task['task_id'], **row) for row in rows)
                official_results.append(official_result(result, rows))
        status.append(entry)
    target = out / 'final_evaluation'
    write(target / 'task_status.json', status)
    if results:
        write_result_files(target, results)
        _write_final_aggregate(target / 'aggregate.csv', results)
        # CIFAR uses its full-source official metrics, never auxiliary target ASR.
        if official:
            write(target / 'official_rounds.json', official)
            target = target / 'official_test'
            write(target / 'metric_scope.json', dict(accuracy='official CIFAR test, n=10000',
                attack_target_success_rate='source_transfer_asr: all source-5 official test images, n=1000',
                attack_target_confidence='source_transfer_confidence on the same 1000 images',
                original_auxiliary_records='../rounds.csv', historical_audit_inherited=False))
            results_for_plot = official_results
            write_result_files(target, official_results)
            _write_final_aggregate(target / 'aggregate.csv', official_results)
        else:
            results_for_plot = results
        for seed in sorted({r.config.seed for r in results_for_plot}):
            generate_visualizations([r for r in results_for_plot if r.config.seed == seed], target / f'seed_{seed}')
    complete = len(results) == len(status)
    print(json.dumps(dict(complete=len(results), total=len(status),
                         all_complete=complete, output=str(target))))
    return complete


def batch(args, protocol, selected, out, data_dir):
    import fcntl
    m = make_manifest(protocol, selected, data_dir)
    if out.exists():
        if not args.continue_pending or not (out / 'manifest.json').exists() or read(out / 'manifest.json') != m:
            raise ValueError('Output exists/differs; use a fresh directory or --continue-pending with identical settings')
    else:
        out.mkdir(parents=True)
        write(out / 'manifest.json', m)
    with (out / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_ids = args.gpus.split(',')
        rows = gpu_rows()
        devices = [next(r for r in rows if g in (r['index'], r['uuid'])) for g in gpu_ids]
        if not 1 <= len(devices) <= 6 or len({r['uuid'] for r in devices}) != len(devices):
            raise ValueError('Select 1–6 distinct GPUs')
        if any(r['uuid'] in QUARANTINED for r in devices):
            raise ValueError('Selected GPU is quarantined')
        pending = [s for s in selected if not (out / 'tasks' / s['task_id']).exists()]
        active, stopped = {}, []
        def stop(signum, frame):
            stopped.append(signum)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while pending or active:
            for uid, (process, stream, gpu_lock, task) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close(); gpu_lock.close(); del active[uid]
                if code:
                    stopped.append(code)
                    write(out / 'controller_failure.json', dict(task=task, exit_code=code))
            if not stopped:
                for device in devices:
                    uid = device['uuid']
                    if not pending or uid in active:
                        continue
                    gpu_lock = Path('/tmp', 'sm9-reproduction-' + uid + '.lock').open('a')
                    try:
                        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        gpu_lock.close(); continue
                    first = next(r for r in gpu_rows() if r['uuid'] == uid)
                    if first['free'] < 8192:
                        gpu_lock.close(); continue
                    time.sleep(5)
                    second = next(r for r in gpu_rows() if r['uuid'] == uid)
                    if stopped or second['free'] < 8192:
                        gpu_lock.close(); continue
                    task = pending.pop(0)
                    log = out / 'logs' / (task['task_id'] + '.log')
                    log.parent.mkdir(exist_ok=True)
                    stream = log.open('x')
                    command = [sys.executable, str(ROOT / 'reproduce.py'), args.experiment,
                               '--output', str(out), '--internal-task', task['task_id']]
                    if args.allow_environment_difference:
                        command.append('--allow-environment-difference')
                    process = subprocess.Popen(command, cwd=ROOT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=uid), stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    active[uid] = (process, stream, gpu_lock, task['task_id'])
                    write(out / 'admission' / (task['task_id'] + '.json'),
                          dict(first=first, second=second, pid=process.pid))
            write(out / 'controller_status.json', dict(pid=os.getpid(),
                active={u: v[3] for u, v in active.items()}, pending=len(pending), draining=bool(stopped)))
            if stopped and not active:
                break
            if pending or active:
                time.sleep(5)
        complete = summarize(out)
        if stopped or not complete:
            raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=NAMES)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--run', action='store_true')
    action.add_argument('--download', action='store_true')
    action.add_argument('--check-data', action='store_true')
    action.add_argument('--ready', action='store_true')
    action.add_argument('--summarize', action='store_true')
    action.add_argument('--internal-task', help=argparse.SUPPRESS)
    parser.add_argument('--methods', default='all', help='comma-separated sm9rrs,vert,alignins,krum,ding13,fedavg')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpus', default='0', help='physical NVIDIA indices or UUIDs; one worker per GPU')
    parser.add_argument('--continue-pending', action='store_true', help='only start tasks with no directory; never retry')
    parser.add_argument('--allow-environment-difference', action='store_true')
    args = parser.parse_args(argv)
    protocol = read(CONFIGS / (args.experiment + '.json'))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    if protocol['dataset'] == 'cifar10':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = protocol['numerics']['cublas_workspace_config']
    methods = sorted(protocol['methods']) if args.methods == 'all' else args.methods.split(',')
    selected = specs(protocol, methods)
    data_dir = (args.data_dir or ROOT / protocol['data_config']['data_dir']).expanduser().resolve()
    out = safe_output(args.output or ROOT / 'outputs/old/reproductions' / args.experiment)
    if args.internal_task:
        m = read(out / 'manifest.json')
        task = next(t for t in m['specs'] if t['task_id'] == args.internal_task)
        run_task(args, protocol, task, out)
    elif args.summarize:
        summarize(out)
    elif args.download or args.check_data or args.ready or args.run:
        verify_sources(protocol)
        _, contract = load_data(protocol, data_dir, download=args.download)
        print(json.dumps(dict(data_verified=True, contract=contract)), flush=True)
        if args.ready or args.run:
            print(json.dumps(runtime(protocol, args.allow_environment_difference)), flush=True)
        if args.run:
            batch(args, protocol, selected, out, data_dir)
    else:
        verify_sources(protocol)
        print(json.dumps(dict(experiment=args.experiment, tasks=len(selected),
            training_rounds=len(selected) * protocol['rounds'], methods=methods,
            seeds=protocol['seeds'], output=str(out), training_started=False), indent=2))


if __name__ == '__main__':
    main()
