#!/usr/bin/env python3
"""Scheduling-only adapter for the frozen CP1 CUDA panel (one task per GPU).

Reuses the original manifest, task identity and worker. Never resumes partial
tasks. SIGINT/SIGTERM stop new dispatch and drain already running experiments.
"""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_cifar_cp1_cuda_development as base

SOURCES = ('run_cifar_cp1_dynamic_gpu.py', 'tests/test_cp1_dynamic_gpu.py')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def retained_files(out, specs):
    hashes = {}
    for spec in specs:
        folder = out / 'tasks' / spec['task_id']
        if base.guard_task_directory(folder) in base.TERMINAL:
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    hashes[str(path.relative_to(out))] = file_sha(path)
    return hashes


def verify_retained(out, hashes):
    for name, expected in hashes.items():
        if file_sha(out / name) != expected:
            raise ValueError('Previously completed evidence changed: ' + name)


def ready_gpus(rows, policy, counts, occupied):
    """A busy or absent sample resets a GPU's consecutive idle count."""
    eligible = [r for r in base.qualified_gpus(rows, policy)
                if r['uuid'] not in occupied]
    counts = {r['uuid']: counts.get(r['uuid'], 0) + 1 for r in eligible}
    ready = [r for r in eligible if counts[r['uuid']] >= policy['consecutive_samples']]
    return counts, sorted(ready, key=lambda r: (-r['free_mb'], r['index']))


def take_gpu_lock(uuid):
    folder = base.ROOT / 'outputs/.codex_gpu_locks'
    folder.mkdir(parents=True, exist_ok=True)
    handle = (folder / (uuid + '.lock')).open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def run(config, p, specs, manifest, out, max_gpus=8):
    if not 1 <= max_gpus <= 8:
        raise ValueError('Maximum GPU concurrency must be between 1 and 8')
    # This adapter must attach to an identified panel, not invent a new manifest.
    if base.read(out / 'manifest.json') != manifest:
        raise ValueError('Existing frozen manifest required; no identity rewrite')
    with (out / 'controller.lock').open('a') as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.verify(manifest)
        known = {s['task_id'] for s in specs}
        unknown = {f.name for f in (out / 'tasks').glob('*') if f.is_dir()} - known
        if unknown:
            raise ValueError('Unidentified task directories retained: ' + repr(sorted(unknown)))
        previous = retained_files(out, specs)  # rejects every partial before any dispatch
        pending = [s for s in specs if base.guard_task_directory(out / 'tasks' / s['task_id']) == 'new']
        if any(base.read(out / 'tasks' / s['task_id'] / 'status.json', {}).get('status') == 'failed'
               for s in specs):
            raise ValueError('Existing infrastructure/unclassified failure retained; inspect before continuing')
        base.summarize(p, specs, manifest, out)  # validates existing identity, data and rounds
        source_hashes = {name: file_sha(base.ROOT / name) for name in SOURCES}
        policy = dict(schema=1, scheduling_only=True, manifest_fingerprint=manifest['fingerprint'],
                      max_gpus=max_gpus, tasks_per_gpu=1, pinned_gpu_uuid=None,
                      gpu_admission=p['gpu_admission'], adapter_sha256=source_hashes,
                      worker='run_cifar_cp1_cuda_development.py',
                      authorization='2026-10-27 user: dynamically use idle GPUs, one independent experiment per GPU, maximum eight; preserve results, attack and selection criteria.')
        audit = out / 'scheduler_dynamic_v1'
        audit.mkdir(exist_ok=True)
        old = base.read(audit / 'policy.json')
        if old is not None and old != policy:
            raise ValueError('Scheduling amendment differs; preserve audit and use a new adapter version')
        if old is None:
            base.write(audit / 'policy.json', policy)
        run_dir = audit / (time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '_' + str(os.getpid()))
        run_dir.mkdir()
        base.write(run_dir / 'retained_files.json', previous)
        active, counts, rows = {}, {}, []
        stop = False
        failure = None
        last_progress = time.monotonic()

        def event(kind, **details):
            import json
            with (run_dir / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(event=kind, utc=base.utc(), **details), sort_keys=True) + '\n')

        def interrupted(signum, frame):
            nonlocal stop
            stop = True  # never interrupt a worker to change scheduling

        def verify_sources():
            base.verify(manifest)
            for name, expected in source_hashes.items():
                if file_sha(base.ROOT / name) != expected:
                    raise ValueError('Active scheduling source changed: ' + name)

        def status(state):
            base.write(out / 'controller_status.json', dict(
                status=state, scheduler='dynamic_v1', pid=os.getpid(), max_gpus=max_gpus,
                pinned_gpu_uuid=None, updated_utc=base.utc(),
                active={name: dict(child_pid=a['child'].pid, gpu_uuid=a['gpu']['uuid'],
                                   gpu_index=a['gpu']['index'], started_utc=a['started'])
                        for name, a in active.items()},
                pending=[s['task_id'] for s in pending], gpu_snapshot=rows, idle_samples=counts,
                wait_seconds=time.monotonic() - last_progress, failure=failure,
                audit_run=str(run_dir.relative_to(out))))

        handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        event('controller_started', retained_tasks=len(specs) - len(pending), pending=len(pending))
        try:
            while pending or active:
                # Reap without blocking other GPUs. Failure stops new dispatch,
                # while all already running workers are allowed to finish.
                for name, a in list(active.items()):
                    code = a['child'].poll()
                    if code is None:
                        continue
                    state = base.read(out / 'tasks' / name / 'status.json', {})
                    event('worker_exited', task=name, returncode=code, status=state.get('status'),
                          gpu_uuid=a['gpu']['uuid'], child_pid=a['child'].pid)
                    a['lock'].close()
                    del active[name]
                    counts.pop(a['gpu']['uuid'], None)
                    last_progress = time.monotonic()
                    if code != 0 or state.get('status') not in base.TERMINAL or state.get('status') == 'failed':
                        failure = failure or f'{name}: exit={code}, status={state.get("status")}; evidence retained'
                        stop = True
                if stop:
                    if not active:
                        break
                    status('draining_after_failure' if failure else 'draining')
                    time.sleep(p['gpu_admission']['sample_interval_seconds'])
                    continue
                if not pending:
                    if active:
                        status('training')
                        time.sleep(p['gpu_admission']['sample_interval_seconds'])
                    continue
                try:
                    verify_sources()
                    rows = base.gpu_snapshot()
                    counts, ready = ready_gpus(rows, p['gpu_admission'], counts,
                                               {a['gpu']['uuid'] for a in active.values()})
                    for gpu in ready:
                        if stop or not pending or len(active) >= max_gpus:
                            break
                        handle = take_gpu_lock(gpu['uuid'])
                        if handle is None:
                            counts.pop(gpu['uuid'], None)
                            continue
                        try:
                            # Recheck under the cooperative UUID lock before launching.
                            fresh = base.qualified_gpus(base.gpu_snapshot(), p['gpu_admission'], gpu['uuid'])
                            if len(fresh) != 1:
                                counts.pop(gpu['uuid'], None)
                                continue
                            spec = pending[0]
                            name = spec['task_id']
                            verify_sources()
                            if base.guard_task_directory(out / 'tasks' / name) != 'new':
                                raise ValueError('Queued task appeared before dispatch: ' + name)
                            log = out / 'logs' / (name + '.log')
                            log.parent.mkdir(exist_ok=True)
                            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'],
                                       CUBLAS_WORKSPACE_CONFIG=p['numerics']['cublas_workspace_config'])
                            with log.open('a') as stream:
                                child = subprocess.Popen([sys.executable, '-u', str(base.ROOT / policy['worker']),
                                                          '--config', str(Path(config).resolve()), '--task', name],
                                                         cwd=base.ROOT, env=env, stdin=subprocess.DEVNULL,
                                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                            active[name] = dict(child=child, lock=handle, gpu=fresh[0], started=base.utc())
                            handle = None  # held until the specific worker exits
                            pending.pop(0)
                            counts.pop(gpu['uuid'], None)
                            last_progress = time.monotonic()
                            event('worker_started', task=name, child_pid=child.pid, admission=fresh[0])
                        finally:
                            if handle is not None:
                                handle.close()
                    if not active and time.monotonic() - last_progress > p['gpu_admission']['maximum_wait_seconds']:
                        raise TimeoutError('No qualified idle GPU; resource wait, not algorithm failure')
                except Exception as error:
                    failure = f'{type(error).__name__}: {error}'
                    stop = True
                    event('dispatch_blocked', error=failure)
                status('draining_after_failure' if failure else ('training' if active else 'waiting_for_idle_gpu'))
                if pending or active:
                    time.sleep(p['gpu_admission']['sample_interval_seconds'])
            verify_sources()
            verify_retained(out, previous)
            base.summarize(p, specs, manifest, out)
            state = 'blocked' if failure else ('stopped_after_drain' if stop else 'panel_terminal')
            status(state)
            event('controller_finished', status=state, error=failure, retained_files_unchanged=True)
            if failure:
                raise RuntimeError(failure)
        except BaseException as error:
            # Unexpected controller errors must not orphan workers or release
            # their UUID locks while they are still training. Drain, then fail.
            failure = failure or f'{type(error).__name__}: {error}'
            for name, a in list(active.items()):
                a['child'].wait()
                a['lock'].close()
                del active[name]
            status('blocked')
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=base.DEFAULT)
    parser.add_argument('--max-gpus', type=int, default=8)
    parser.add_argument('--batch', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_gpus <= 8:
        parser.error('--max-gpus must be between 1 and 8')
    p, specs, manifest = base.plan(args.config)
    out = base.ROOT / p['output_dir']
    if args.batch:
        run(args.config, p, specs, manifest, out, args.max_gpus)
    else:
        import json
        print(json.dumps(dict(manifest_fingerprint=manifest['fingerprint'], max_gpus=args.max_gpus,
                              tasks_per_gpu=1, pinned_gpu_uuid=None, training_started=False), indent=2))


if __name__ == '__main__':
    main()
