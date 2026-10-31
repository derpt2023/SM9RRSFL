"""Scheduling-only continuation of the frozen 90-task Ours reserve panel.

Completed work is retained. The original controller must drain before this
controller takes its lock. Workers, manifests, seeds and selection rules remain
unchanged. Multiple independent workers may share a non-quarantined GPU.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_cifar_ours_reserve as base
import cifar_ours_reserve_scheduler as old

DIRECTORY = 'resource_packing_v1'
SOURCES = ('run_cifar_ours_resource_packing.py', 'tests/test_ours_resource_packing.py')
POLICY = dict(max_tasks=6, max_tasks_per_gpu=3, min_free_mb=8192, max_utilization=100,
              consecutive_samples=2, sample_interval_seconds=5,
              same_gpu_launch_interval_seconds=60, maximum_wait_seconds=86400)


def command(pid):
    try:
        return Path('/proc', str(pid), 'cmdline').read_bytes().split(b'\0')[:-1]
    except FileNotFoundError:
        return []


def validate_old_controller(state, expected_pid):
    args = command(expected_pid)
    if (state.get('pid') != expected_pid or expected_pid <= 1 or
            state.get('failure') or state.get('status') not in
            ('training', 'waiting_for_idle_gpu', 'draining')):
        raise ValueError('Expected healthy original controller required')
    if (len(args) < 3 or Path(os.fsdecode(args[2])).name != 'run_cifar_ours_reserve.py'
            or b'--batch' not in args or b'--task' in args):
        raise ValueError('Original controller process identity differs')
    return args


def plan_policy(m):
    return dict(schema=1, scheduling_only=True, original_manifest_fingerprint=m['fingerprint'],
                original_sources_unchanged=True, worker='run_cifar_ours_reserve.py',
                config=str(base.DEFAULT), parameters=POLICY,
                quarantine_gpu_uuids=list(base.QUARANTINE),
                adapter_sha256={n:old.file_sha(base.ROOT/n) for n in SOURCES},
                authorization='2026-11-01 user requested fuller server utilization; independent workers may share GPUs with unchanged admission and science.',
                worker_environment_unchanged=True, automatic_retry=False,
                events='scheduler_ours_reserve_v1/*_packed/events.jsonl',
                checkpoint_preservation='size/inode/mtime only; other retained task files SHA256; checkpoints are never loaded or rewritten')


def verify(m, policy):
    base.verify(m)
    for name, expected in policy['adapter_sha256'].items():
        if old.file_sha(base.ROOT/name) != expected:
            raise ValueError('Active packing adapter changed: '+name)


def retain(out, specs):
    evidence = {}
    known = {s['task_id'] for s in specs}
    if any(f.is_dir() and f.name not in known for f in (out/'tasks').glob('*')):
        raise ValueError('Unidentified prior task directory')
    for s in specs:
        folder = out/'tasks'/s['task_id']
        state = base.guard_task_directory(folder)
        if state == 'new':
            continue
        if state != 'complete':
            raise ValueError('Partial or failed task retained; no automatic retry')
        for p in sorted(folder.rglob('*')):
            if not p.is_file():
                continue
            st = p.stat()
            value = dict(size=st.st_size, inode=st.st_ino, mtime_ns=st.st_mtime_ns)
            if p.name != 'checkpoint.pickle':
                value['sha256'] = old.file_sha(p)
            evidence[str(p.relative_to(out))] = value
    return evidence


def verify_retained(out, evidence):
    for rel, expected in evidence.items():
        p = out/rel
        st = p.stat()
        if (st.st_size, st.st_ino, st.st_mtime_ns) != (expected['size'], expected['inode'], expected['mtime_ns']):
            raise ValueError('Retained task file changed: '+rel)
        if 'sha256' in expected and old.file_sha(p) != expected['sha256']:
            raise ValueError('Retained task content changed: '+rel)


def ready_gpus(rows, active, last_launch, counts, now):
    occupied = Counter(a['gpu']['uuid'] for a in active.values())
    eligible = [r for r in base.qualified_gpus(rows, POLICY)
                if r['uuid'] not in base.QUARANTINE
                and occupied[r['uuid']] < POLICY['max_tasks_per_gpu']
                and now-last_launch.get(r['uuid'], -float('inf')) >= POLICY['same_gpu_launch_interval_seconds']]
    counts = {r['uuid']:counts.get(r['uuid'], 0)+1 for r in eligible}
    ready = [r for r in eligible if counts[r['uuid']] >= POLICY['consecutive_samples']]
    return counts, sorted(ready, key=lambda r:(-r['free_mb'], occupied[r['uuid']], r['index']))


def check_drained(out):
    state = base.read(out/'controller_status.json', {})
    if (state.get('status') not in ('stopped_after_drain', 'panel_terminal')
            or state.get('active') or state.get('failure') or command(state.get('pid'))):
        raise ValueError('Original controller must have exited after a clean drain')
    return state


def run(config, p, specs, m, out, policy):
    with (out/'controller.lock').open('a') as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = check_drained(out)
        if base.read(out/'manifest.json') != m:
            raise ValueError('Frozen worker manifest differs')
        verify(m, policy)
        evidence = retain(out, specs)
        pending = [s for s in specs if base.guard_task_directory(out/'tasks'/s['task_id']) == 'new']
        base.summarize(p, specs, m, out)
        folder = out/DIRECTORY
        folder.mkdir(exist_ok=True)
        existing = base.read(folder/'policy.json')
        if existing is not None and existing != policy:
            raise ValueError('Packing amendment differs; no overwrite')
        base.write(folder/'policy.json', policy)
        audit = out/'scheduler_ours_reserve_v1'/(time.strftime('%Y%m%dT%H%M%S', time.gmtime())+'_'+str(os.getpid())+'_packed')
        audit.mkdir(parents=True)
        base.write(audit/'policy.json', policy)
        base.write(audit/'previous_controller.json', prior)
        base.write(audit/'retained_files.json', evidence)
        active, locks, counts, last_launch, rows = {}, {}, {}, {}, []
        stop, failure = False, None
        last_progress = time.monotonic()

        def event(kind, **details):
            with (audit/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(event=kind, utc=base.utc(), monotonic_seconds=time.monotonic(), scheduler=DIRECTORY,
                                             policy_sha256=base.sha(audit/'policy.json'), **details), sort_keys=True)+'\n')

        def status(value):
            base.write(out/'controller_status.json', dict(status=value, scheduler=DIRECTORY,
                pid=os.getpid(), max_gpus=6, max_tasks=POLICY['max_tasks'],
                max_tasks_per_gpu=POLICY['max_tasks_per_gpu'], updated_utc=base.utc(),
                active={n:dict(child_pid=a['child'].pid, gpu_uuid=a['gpu']['uuid'],
                               gpu_index=a['gpu']['index'], started_utc=a['started']) for n,a in active.items()},
                pending=[s['task_id'] for s in pending], gpu_snapshot=rows, idle_samples=counts,
                wait_seconds=time.monotonic()-last_progress, failure=failure,
                audit_run=str(audit.relative_to(out))))

        def interrupted(signum, frame):
            nonlocal stop
            stop = True

        handlers = {sig:signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        event('controller_started', pending=len(pending), retained_tasks=len(specs)-len(pending))
        try:
            while pending or active:
                for name,a in list(active.items()):
                    code = a['child'].poll()
                    if code is None:
                        continue
                    state = base.read(out/'tasks'/name/'status.json', {})
                    event('worker_exited', task=name, returncode=code, status=state.get('status'),
                          gpu_uuid=a['gpu']['uuid'], child_pid=a['child'].pid)
                    del active[name]
                    uuid = a['gpu']['uuid']
                    if not any(x['gpu']['uuid']==uuid for x in active.values()):
                        locks.pop(uuid).close()
                    last_progress = time.monotonic()
                    if code != 0 or state.get('status') != 'complete':
                        failure = failure or f'{name}: exit={code}, status={state.get("status")}; retained without retry'
                        stop = True
                if stop and not active:
                    break
                if not stop and pending and len(active) < POLICY['max_tasks']:
                    try:
                        verify(m, policy)
                        rows = base.gpu_snapshot()
                        counts, ready = ready_gpus(rows, active, last_launch, counts, time.monotonic())
                        for gpu in ready:
                            if stop or not pending or len(active) >= POLICY['max_tasks']:
                                break
                            uuid = gpu['uuid']
                            acquired = uuid not in locks
                            if acquired:
                                handle = old.take_gpu_lock(uuid)
                                if handle is None:
                                    counts.pop(uuid, None)
                                    continue
                                locks[uuid] = handle
                            try:
                                fresh = [r for r in base.qualified_gpus(base.gpu_snapshot(), POLICY, uuid)
                                         if r['uuid'] not in base.QUARANTINE]
                                if len(fresh) != 1:
                                    counts.pop(uuid, None)
                                    continue
                                spec = pending[0]
                                name = spec['task_id']
                                verify(m, policy)
                                if base.guard_task_directory(out/'tasks'/name) != 'new':
                                    raise ValueError('Queued task appeared before dispatch: '+name)
                                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid,
                                           CUBLAS_WORKSPACE_CONFIG=p['numerics']['cublas_workspace_config'])
                                env.update(p['process_environment'])
                                log = out/'logs'/(name+'.log')
                                log.parent.mkdir(exist_ok=True)
                                with log.open('a') as stream:
                                    child = subprocess.Popen([sys.executable, '-u', str(base.ROOT/policy['worker']),
                                        '--config', str(Path(config).resolve()), '--task', name], cwd=base.ROOT, env=env,
                                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                                active[name] = dict(child=child, gpu=fresh[0], started=base.utc())
                                pending.pop(0)
                                last_launch[uuid] = last_progress = time.monotonic()
                                counts.pop(uuid, None)
                                event('worker_started', task=name, child_pid=child.pid, admission=fresh[0],
                                      launch_monotonic_seconds=last_launch[uuid],
                                      active_on_gpu=sum(x['gpu']['uuid']==uuid for x in active.values()), active_total=len(active))
                            finally:
                                if acquired and not any(x['gpu']['uuid']==uuid for x in active.values()):
                                    locks.pop(uuid).close()
                        if not active and time.monotonic()-last_progress > POLICY['maximum_wait_seconds']:
                            raise TimeoutError('No admissible GPU; resource wait, not an algorithm failure')
                    except Exception as error:
                        failure = f'{type(error).__name__}: {error}'
                        stop = True
                        event('dispatch_blocked', error=failure)
                status('draining_after_failure' if failure else ('draining' if stop else ('training' if active else 'waiting_for_idle_gpu')))
                if pending or active:
                    time.sleep(POLICY['sample_interval_seconds'])
            verify(m, policy)
            verify_retained(out, evidence)
            base.summarize(p, specs, m, out)
            final = 'blocked' if failure else ('stopped_after_drain' if stop else 'panel_terminal')
            status(final)
            event('controller_finished', status=final, error=failure, retained_files_unchanged=True)
            if failure:
                raise RuntimeError(failure)
        except BaseException as error:
            failure = failure or f'{type(error).__name__}: {error}'
            for name,a in list(active.items()):
                code = a['child'].wait()
                state = base.read(out/'tasks'/name/'status.json', {})
                event('worker_exited', task=name, child_pid=a['child'].pid, gpu_uuid=a['gpu']['uuid'],
                      returncode=code, status=state.get('status'), exceptional_drain=True)
                del active[name]
            status('blocked')
            raise
        finally:
            for handle in locks.values():
                handle.close()
            for sig,handler in handlers.items():
                signal.signal(sig, handler)


def wait_for_drain(out, expected_pid, policy, timeout=3600):
    state = base.read(out/'controller_status.json', {})
    args = validate_old_controller(state, expected_pid)
    folder = out/DIRECTORY
    folder.mkdir(exist_ok=True)
    if (folder/'handoff.json').exists():
        raise ValueError('Handoff already requested; do not repeat')
    base.write(folder/'handoff.json', dict(utc=base.utc(), old_controller_pid=expected_pid,
        old_command=[os.fsdecode(x) for x in args], waiter_pid=os.getpid(),
        old_state=state, policy=policy, signal='SIGTERM to controller only; workers drain naturally'))
    # Revalidate the precise process immediately before signaling it.
    validate_old_controller(base.read(out/'controller_status.json', {}), expected_pid)
    os.kill(expected_pid, signal.SIGTERM)
    deadline = time.monotonic()+timeout
    while command(expected_pid):
        if time.monotonic() >= deadline:
            raise TimeoutError('Original controller has not drained; no workers were killed')
        time.sleep(5)
    check_drained(out)


def audit_packing(out):
    audits = {}
    for folder in sorted((out/'scheduler_ours_reserve_v1').glob('*_packed')):
        policy = base.read(folder/'policy.json')
        settings = policy['parameters']
        active, last, peak, per_gpu_peak = {}, {}, 0, Counter()
        events = folder/'events.jsonl'
        rows = [json.loads(line) for line in events.read_text().splitlines()]
        for row in rows:
            if row['policy_sha256'] != base.sha(folder/'policy.json'):
                raise ValueError('Packing policy audit changed')
            name = row.get('task')
            if row['event'] == 'worker_started':
                uuid = row['admission']['uuid']
                if (name in active or uuid in policy['quarantine_gpu_uuids']
                        or row['admission']['free_mb'] < settings['min_free_mb']
                        or row['launch_monotonic_seconds']-last.get(uuid, -float('inf')) < settings['same_gpu_launch_interval_seconds']):
                    raise ValueError('Packing launch audit mismatch')
                active[name] = uuid
                occupancy = Counter(active.values())
                if (len(active) > settings['max_tasks'] or max(occupancy.values()) > settings['max_tasks_per_gpu']
                        or row['active_total'] != len(active) or row['active_on_gpu'] != occupancy[uuid]):
                    raise ValueError('Packing concurrency audit mismatch')
                peak = max(peak, len(active))
                for gpu, count in occupancy.items():
                    per_gpu_peak[gpu] = max(per_gpu_peak[gpu], count)
                last[uuid] = row['launch_monotonic_seconds']
            elif row['event'] == 'worker_exited':
                if name not in active or active.pop(name) != row['gpu_uuid'] or row['returncode'] != 0 or row['status'] != 'complete':
                    raise ValueError('Packing exit audit mismatch')
            elif row['event'] == 'dispatch_blocked':
                raise ValueError('Packing failure retained for review')
        if active or not rows or rows[-1]['event'] != 'controller_finished' or rows[-1]['status'] != 'panel_terminal':
            raise ValueError('Packing controller not complete')
        audits[str(folder.relative_to(base.ROOT))] = dict(peak_tasks=peak, peak_per_gpu=dict(per_gpu_peak),
            event_sha256=base.sha(events), policy_sha256=base.sha(folder/'policy.json'))
    if not audits:
        raise ValueError('No packing execution evidence')
    base.write(out/DIRECTORY/'execution_audit.json', dict(scheduling_only=True, runs=audits,
        original_worker_manifest_unchanged=True, selection_rule_unchanged=True))
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', type=int, metavar='EXPECTED_CONTROLLER_PID')
    parser.add_argument('--continue', dest='resume', action='store_true')
    args = parser.parse_args()
    if args.handoff and args.resume:
        parser.error('Choose handoff or continue')
    os.environ.update(base.PROCESS_ENV)
    p,specs,m = base.plan()
    policy = plan_policy(m)
    if not args.handoff and not args.resume:
        print(json.dumps(policy, indent=2))
        return
    base.first.legacy.require_station()
    out = base.ROOT/p['output_dir']
    if base.read(out/'manifest.json') != m:
        raise ValueError('Existing frozen manifest required')
    verify(m, policy)
    base.check_accepted_snapshot()
    with (out/'resource_packing_handoff.lock').open('a') as handoff_lock:
        fcntl.flock(handoff_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.handoff:
            wait_for_drain(out, args.handoff, policy)
        run(base.DEFAULT, p, specs, m, out, policy)
        if base.read(out/'controller_status.json')['status'] == 'panel_terminal':
            audit_packing(out)
            from cifar_ours_reserve_report import finish
            print(json.dumps(finish(p, specs, m, out), indent=2))


if __name__ == '__main__':
    main()
