#!/usr/bin/env python3
"""Fresh CIFAR timing matrix using the exact MNIST measured-run boundary.

Performance artifacts remain separate from the accepted cifar-best package.
No retries, resume, parameter selection, or mutation of historical results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import signal
import statistics
import subprocess
import sys
import time

import reproduce as base

ROOT = base.ROOT
METHODS = ('sm9rrs', 'vert', 'alignins', 'krum', 'ding13', 'fedavg')
DEFAULT_OUTPUT = ROOT / 'outputs/old/cifar_mnist_timing150_v1'
EXTRA_SOURCES = ('run_cifar_timing.py', 'tests/test_cifar_mnist_timing.py',
                 'configs/reproduction/source_files.json')
DEFINITION = dict(
    version='mnist_common_runner_v1',
    function='sm9rrsfl.experiments.run_measured_experiment',
    total='perf_counter span around run_experiment minus measured checkpoint callback I/O',
    without_crypto='max(0, runtime_seconds - sum(four non-overlapping crypto wall spans))',
    excluded='data loading, imports, task setup, result serialization, extra CIFAR forensic observers',
    included='model/defense/crypto initialization, training, attack, defense, standard evaluation, checkpoint state construction',
    performance_policy='Timing-only fresh runs; auxiliary-target ASR is diagnostic, never replaces accepted official ASR',
    limitations='Shared CPU/GPU host; one study task per GPU is not exclusive GPU ownership. Fixed CIFAR and MNIST workloads differ.')


def now():
    return datetime.now(timezone.utc).isoformat()


def protocol():
    return base.read(base.CONFIGS / 'cifar-best.json')


def blocks(p):
    tasks = base.specs(p, METHODS)
    result = []
    for i in range(0, len(tasks), 6):
        group = tasks[i:i + 6]
        offset = (i // 6) % 6
        result.append(dict(block_id=f'block_{i // 6:02d}',
                           tasks=group[offset:] + group[:offset]))
    return result


def source_map(p):
    result = base.verify_sources(p)
    result.update({name: base.sha(ROOT / name) for name in EXTRA_SOURCES})
    return result


def manifest(data_dir):
    p = protocol()
    payload = dict(kind='cifar_mnist_timing150_v1', protocol=p, blocks=blocks(p),
        sources=source_map(p), data_dir=str(Path(data_dir).resolve()), timing=DEFINITION,
        scheduling=dict(max_gpus=6, tasks_per_gpu=1, same_gpu_per_six_method_block=True,
            method_order='cyclic rotation across 30 blocks; each method five times at each position',
            min_free_mib=8192, admission_samples=2, admission_interval_seconds=5,
            quarantined=sorted(base.QUARANTINED), retries=0))
    payload['fingerprint'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def verify_manifest(m):
    if m != manifest(m['data_dir']):
        raise ValueError('Frozen timing manifest/source/config changed')


def prepare_environment(p):
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = p['numerics']['cublas_workspace_config']


def numerical_settings(p):
    import torch
    n = p['numerics']
    torch.set_num_threads(n['torch_threads'])
    torch.use_deterministic_algorithms(n['deterministic_algorithms'])
    torch.backends.cudnn.deterministic = n['cudnn_deterministic']
    torch.backends.cudnn.benchmark = n['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = n['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = n['cudnn_allow_tf32']


@contextmanager
def scientific_scopes(task):
    from cifar_alternating_target_views import target_view_scope
    from cifar_tad_no_prior import prior_free_tad_scope
    from cifar_vert_paper_core import paper_scope
    from scripts.experiments.cifar10.attack_stability_attack import attack_scope
    with ExitStack() as stack:
        stack.enter_context(target_view_scope())
        stack.enter_context(attack_scope(task['profile']))
        if task['config']['method'] == 'vert':
            stack.enter_context(paper_scope(None, task['task_id']))
        if task['config']['method'] == 'ding13':
            stack.enter_context(prior_free_tad_scope())
        yield


@contextmanager
def checkpoint_progress(folder, experiments):
    """Progress I/O is inside the common runner's measured checkpoint exclusion."""
    original = experiments._write_round_checkpoint
    def write(path, config, fingerprint, state, **kwargs):
        original(path, config, fingerprint, state, **kwargs)
        base.write(folder / 'progress.json', dict(completed_round=int(state['completed_round']),
                                                updated_at=now(), pid=os.getpid()))
    experiments._write_round_checkpoint = write
    try:
        yield
    finally:
        experiments._write_round_checkpoint = original


def measured(dataset, task, folder, fingerprint):
    from sm9rrsfl import experiments
    from sm9rrsfl.fl import ExperimentConfig
    with scientific_scopes(task), checkpoint_progress(folder, experiments):
        return experiments.run_measured_experiment(dataset, ExperimentConfig(**task['config']),
            checkpoint_dir=folder / 'checkpoints', run_fingerprint=fingerprint,
            task_id=task['task_id'], emit_progress_events=False)


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


def timing_record(result):
    from watch_cifar_experiment import timing
    summary = result.summary_dict()
    record = timing(summary)
    record['definition'] = DEFINITION
    record['checkpoint_io_seconds'] = result.checkpoint_io_seconds
    if not finite(summary) or not record['available_and_consistent']:
        raise ValueError('Non-finite or inconsistent measured timing')
    if result.checkpoint_io_seconds < 0 or result.runtime_seconds <= 0:
        raise ValueError('Invalid measured clock span')
    return record


def worker(out, task_id):
    m = base.read(out / 'manifest.json')
    verify_manifest(m)
    task = next(t for b in m['blocks'] for t in b['tasks'] if t['task_id'] == task_id)
    folder = out / 'tasks' / task_id
    folder.mkdir(parents=True, exist_ok=False)  # a partial task must never be retried
    base.write(folder / 'task.json', task)
    base.write(folder / 'status.json', dict(status='loading', pid=os.getpid(), started_at=now()))
    try:
        p = m['protocol']
        prepare_environment(p)
        environment = base.runtime(p, False)
        import torch
        if torch.cuda.device_count() != 1:
            raise ValueError('Exactly one visible GPU is required')
        numerical_settings(p)
        environment.update(visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), numerics=p['numerics'])
        base.write(folder / 'environment.json', environment)
        dataset, contract = base.load_data(p, Path(m['data_dir']))
        base.write(folder / 'dataset_identity.json', dict(contract=contract,
                   official_dataset_identity=p['official_dataset_identity']))
        base.write(folder / 'status.json', dict(status='training', pid=os.getpid(), started_at=now()))
        result = measured(dataset, task, folder, m['fingerprint'])
        # All serialization and health reporting happen after the common timer stopped.
        with (folder / 'result.pickle').open('wb') as stream:
            pickle.dump(result, stream)
        base.write(folder / 'result.json', result.summary_dict())
        base.write(folder / 'rounds.json', [asdict(r) for r in result.records])
        base.write(folder / 'timing.json', timing_record(result))
        if result.stopped_round != p['rounds'] or result.nonfinite_updates != 0:
            raise ValueError('Incomplete or non-finite training; evidence retained')
        if [r.round for r in result.records] != list(range(p['rounds'] + 1)) or not finite(
                [asdict(r) for r in result.records]):
            raise ValueError('Incomplete/non-finite round trajectory')
        from sm9rrsfl.fair_tuning import _training_health_reasons
        health = _training_health_reasons(result)
        verify_manifest(m)
        hashes = {name: base.sha(folder / name) for name in
                  ('task.json', 'environment.json', 'dataset_identity.json', 'result.pickle',
                   'result.json', 'rounds.json', 'timing.json')}
        base.write(folder / 'status.json', dict(status='complete', pid=os.getpid(),
            completed_round=result.stopped_round, nonfinite_updates=result.nonfinite_updates,
            health_reasons=health, artifacts=hashes, completed_at=now()))
    except BaseException as exc:
        base.write(folder / 'status.json', dict(status='failed', pid=os.getpid(), error=repr(exc), at=now()))
        raise


def gpu_snapshot():
    rows = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    gpus = [dict(index=x[0], uuid=x[1], free=int(x[2]), utilization=int(x[3]))
            for line in rows.strip().splitlines() for x in [[s.strip() for s in line.split(',')]]]
    apps = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True)
    return dict(at=now(), gpus=gpus, compute_apps=apps.strip().splitlines(), load_average=os.getloadavg())


def event(out, kind, **fields):
    with (out / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(at=now(), event=kind, **fields), sort_keys=True) + '\n')


def acquire_gpu(uuid):
    # Shared with the fresh reproduction entrypoint; held for a whole six-method block.
    stream = open(f'/tmp/sm9-reproduction-{uuid}.lock', 'a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def admitted(uuid, first, second):
    if uuid in base.QUARANTINED:
        return False
    for snapshot in (first, second):
        rows = [g for g in snapshot['gpus'] if g['uuid'] == uuid]
        if len(rows) != 1 or rows[0]['free'] < 8192:
            return False
    return True


def collect(out):
    m = base.read(out / 'manifest.json')
    rows, counts = [], Counter()
    for b in m['blocks']:
        for t in b['tasks']:
            folder = out / 'tasks' / t['task_id']
            s = base.read(folder / 'status.json') if (folder / 'status.json').exists() else {'status': 'pending'}
            counts[s['status']] += 1
            if s['status'] == 'complete':
                for name, digest in s['artifacts'].items():
                    if base.sha(folder / name) != digest:
                        raise ValueError('Completed artifact changed: ' + t['task_id'] + '/' + name)
                r = base.read(folder / 'result.json')
                rows.append(dict(task_id=t['task_id'], block_id=b['block_id'],
                    method=t['config']['method'], seed=t['config']['seed'], partition=t['config']['partition'],
                    malicious_ratio=t['config']['malicious_ratio'],
                    runtime_seconds=r['runtime_seconds'],
                    runtime_without_crypto_seconds=r['runtime_without_crypto_seconds'],
                    crypto_wall_seconds=r['crypto_wall_seconds'], checkpoint_io_seconds=r['checkpoint_io_seconds'],
                    health_reasons=s['health_reasons']))
    return m, rows, dict(counts)


def final_report(out):
    m, rows, counts = collect(out)
    verify_manifest(m)
    events = [json.loads(line) for line in (out / 'events.jsonl').read_text().splitlines()]
    starts = [e for e in events if e['event'] == 'start']
    exits = [e for e in events if e['event'] == 'exit']
    ids = [t['task_id'] for b in m['blocks'] for t in b['tasks']]
    per_block = defaultdict(set)
    active, max_active, failures = {}, 0, []
    for e in events:
        if e['event'] == 'start':
            if e['uuid'] in active or e['uuid'] in base.QUARANTINED or not admitted(
                    e['uuid'], e['admission'][0], e['admission'][1]):
                failures.append('GPU admission/overlap: ' + e['task_id'])
            stamps = [datetime.fromisoformat(s['at']) for s in e['admission']]
            if (stamps[1] - stamps[0]).total_seconds() < 5:
                failures.append('Admission samples less than five seconds apart: ' + e['task_id'])
            active[e['uuid']] = e['task_id']
            max_active = max(max_active, len(active))
            per_block[e['block_id']].add(e['uuid'])
        elif e['event'] == 'exit':
            if active.pop(e['uuid'], None) != e['task_id'] or e['returncode'] != 0:
                failures.append('Unpaired or failed exit: ' + e['task_id'])
    if sorted(e['task_id'] for e in starts) != sorted(ids) or sorted(e['task_id'] for e in exits) != sorted(ids):
        failures.append('Not exactly 180 starts and exits')
    if active or max_active > 6 or any(len(v) != 1 for v in per_block.values()):
        failures.append('Concurrency/same-GPU block mismatch')
    for block in m['blocks']:
        observed = [e['task_id'] for e in starts if e['block_id'] == block['block_id']]
        if observed != [t['task_id'] for t in block['tasks']]:
            failures.append('Incomplete/unbalanced method order: ' + block['block_id'])
    if counts != {'complete': 180}:
        failures.append('Incomplete matrix: ' + repr(counts))
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['method'], r['partition'], r['malicious_ratio'])].append(r)
    aggregates = []
    for key, group in sorted(grouped.items()):
        a = dict(method=key[0], partition=key[1], malicious_ratio=key[2], n=len(group))
        for metric in ('runtime_seconds', 'runtime_without_crypto_seconds', 'crypto_wall_seconds'):
            values = [r[metric] for r in group]
            a[metric] = dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values) > 1 else None)
        aggregates.append(a)
    report = dict(at=now(), counts=counts, timing=DEFINITION, source_fingerprint=m['fingerprint'],
        rows=rows, aggregate=aggregates, execution_audit=dict(passed=not failures, failures=failures,
            starts=len(starts), exits=len(exits), max_active=max_active, block_gpu={k:sorted(v) for k,v in per_block.items()}),
        health_failures=[r for r in rows if r['health_reasons']],
        accepted_performance_results_modified=False)
    base.write(out / 'analysis/timing_results.json', report)
    return report


def batch(out, data_dir, gpu_indices, max_gpus):
    if not 1 <= max_gpus <= 6:
        raise ValueError('At most six GPUs')
    m = manifest(data_dir)
    inventory = gpu_snapshot()['gpus']
    if gpu_indices is not None and (not gpu_indices or
            gpu_indices - {g['index'] for g in inventory}):
        raise ValueError('Requested GPU indices are not available')
    if not any(g['uuid'] not in base.QUARANTINED and
               (gpu_indices is None or g['index'] in gpu_indices) for g in inventory):
        raise ValueError('No non-quarantined GPU selected')
    out.mkdir(parents=True, exist_ok=False)
    base.write(out / 'manifest.json', m)
    control = open(out / 'controller.lock', 'a+')
    fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending, active = list(m['blocks']), {}
    stopping, failure, last_telemetry = False, None, 0.0
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        event(out, 'stop_dispatch', signal=signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    event(out, 'controller_start', pid=os.getpid())
    try:
        while pending or active:
            snap = gpu_snapshot()
            base.write(out / 'gpu_status.json', snap)
            if time.monotonic() - last_telemetry >= 60:
                with (out / 'telemetry.jsonl').open('a') as stream:
                    stream.write(json.dumps(snap, sort_keys=True) + '\n')
                last_telemetry = time.monotonic()
            for uuid, item in list(active.items()):
                process = item.get('process')
                if process is not None and process.poll() is not None:
                    code = process.returncode
                    task_id = item['block']['tasks'][item['position']]['task_id']
                    event(out, 'exit', task_id=task_id, block_id=item['block']['block_id'],
                          uuid=uuid, pid=process.pid, returncode=code, telemetry=snap)
                    item['log'].close()
                    state_file = out / 'tasks' / task_id / 'status.json'
                    state = base.read(state_file) if state_file.exists() else {}
                    if code != 0 or state.get('status') != 'complete':
                        failure = f'{task_id}: exit={code}, status={state.get("status")}'
                        stopping = True
                    item['process'] = None
                    item['position'] += 1
                if item.get('process') is None and (item['position'] == 6 or stopping):
                    item['lock'].close()
                    del active[uuid]
            if not stopping:
                for gpu in snap['gpus']:
                    if not pending or len(active) >= max_gpus:
                        break
                    if gpu['uuid'] in active or gpu['uuid'] in base.QUARANTINED or gpu['free'] < 8192:
                        continue
                    if gpu_indices is not None and gpu['index'] not in gpu_indices:
                        continue
                    lock = acquire_gpu(gpu['uuid'])
                    if lock:
                        active[gpu['uuid']] = dict(block=pending.pop(0), position=0,
                                                  lock=lock, process=None, index=gpu['index'])
                for uuid, item in active.items():
                    if item.get('process') is not None or stopping:
                        continue
                    first = gpu_snapshot()
                    if not admitted(uuid, first, first):
                        continue
                    time.sleep(5)
                    second = gpu_snapshot()
                    if not admitted(uuid, first, second) or stopping:
                        continue
                    verify_manifest(m)
                    task = item['block']['tasks'][item['position']]
                    if (out / 'tasks' / task['task_id']).exists():
                        raise ValueError('Refusing repeated task ' + task['task_id'])
                    log_path = out / 'logs' / (task['task_id'] + '.log')
                    log_path.parent.mkdir(exist_ok=True)
                    log = open(log_path, 'x')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
                    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                        '--output', str(out), '--task', task['task_id']], cwd=ROOT, env=env,
                        stdout=log, stderr=subprocess.STDOUT)
                    item.update(process=child, log=log)
                    event(out, 'start', task_id=task['task_id'], block_id=item['block']['block_id'],
                          uuid=uuid, pid=child.pid, order=item['position'], admission=[first, second])
            counts = Counter()
            for task in (t for b in m['blocks'] for t in b['tasks']):
                path = out / 'tasks' / task['task_id'] / 'status.json'
                counts[base.read(path)['status'] if path.exists() else 'pending'] += 1
            base.write(out / 'controller_status.json', dict(at=now(), pid=os.getpid(),
                status='draining' if stopping else 'training', failure=failure, counts=dict(counts),
                pending_blocks=len(pending), active=[dict(uuid=u, block_id=i['block']['block_id'],
                task_id=i['block']['tasks'][i['position']]['task_id'], pid=i['process'].pid if i.get('process') else None)
                for u,i in active.items()]))
            if stopping and not active:
                break
            if not pending and not active:
                break
            time.sleep(10)
        report = final_report(out)
        base.write(out / 'controller_status.json', dict(at=now(), pid=os.getpid(),
            status='complete' if report['execution_audit']['passed'] else 'incomplete',
            counts=report['counts'], failure=failure, active=[], pending_blocks=len(pending)))
    except BaseException as exc:
        event(out, 'controller_error', error=repr(exc))
        base.write(out / 'controller_status.json', dict(at=now(), pid=os.getpid(),
            status='error_draining', failure=repr(exc)))
        # Never kill a worker or restart it. Existing children finish naturally.
        for item in active.values():
            if item.get('process') is not None:
                item['process'].wait()
                item['log'].close()
        raise
    finally:
        for item in active.values():
            item['lock'].close()
        control.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--batch', action='store_true')
    mode.add_argument('--ready', action='store_true')
    mode.add_argument('--summary', action='store_true')
    mode.add_argument('--task', help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--gpus', default='auto')
    parser.add_argument('--max-gpus', type=int, default=6)
    args = parser.parse_args()
    out = base.safe_output(args.output)
    p = protocol()
    prepare_environment(p)
    if args.task:
        worker(out, args.task)
    elif args.summary:
        m, rows, counts = collect(out)
        verify_manifest(m)
        print(json.dumps(dict(counts=counts, completed=len(rows), fingerprint=m['fingerprint']), indent=2))
    elif args.ready or args.batch:
        environment = base.runtime(p, False)
        dataset, contract = base.load_data(p, args.data_dir)
        del dataset  # the controller must not retain a second full CIFAR dataset
        m = manifest(args.data_dir)
        if args.ready:
            print(json.dumps(dict(ready=True, environment=environment, contract=contract,
                                 sources=len(m['sources']), fingerprint=m['fingerprint']), indent=2))
        else:
            gpus = None if args.gpus == 'auto' else set(args.gpus.split(','))
            report = batch(out, args.data_dir, gpus, args.max_gpus)
            if not report['execution_audit']['passed']:
                raise SystemExit(1)
    else:
        m = manifest(args.data_dir)
        print(json.dumps(dict(tasks=180, rounds_per_task=150, total_rounds=27000,
            blocks=len(m['blocks']), methods=METHODS, output=str(out), timing=DEFINITION,
            scheduling=m['scheduling'], fingerprint=m['fingerprint']), indent=2))


if __name__ == '__main__':
    main()
