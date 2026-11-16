"""Bounded GPU admission and one-attempt scheduling for the full matrix."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import hfcdr_validation_protocol as p


def now():
    return datetime.now(timezone.utc).isoformat()


def gpu_snapshot():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in text.strip().splitlines():
        idx, uuid, name, total, free, util = [v.strip() for v in line.split(',')]
        rows.append(dict(index=int(idx), uuid=uuid, name=name, total_mib=int(total),
                         free_mib=int(free), utilization_pct=int(util)))
    return rows


def eligible_gpu(rows, active, last_start, config, at):
    counts = Counter(a['gpu'] for a in active)
    candidates = [r for r in rows if r['free_mib'] >= config['minimum_free_mib']
                  and counts[r['index']] < config['max_workers_per_gpu']
                  and at-last_start.get(r['index'], -1e20) >= config['same_gpu_start_spacing_seconds']]
    return min(candidates, key=lambda r: (counts[r['index']], -r['free_mib'], r['index'])) if candidates else None


def task_outcome(folder, returncode):
    status = p.read(folder/'status.json') if (folder/'status.json').exists() else {}
    if returncode == 0 and status.get('status') == 'complete':
        return 'complete', status
    if status.get('status') == 'failed' and status.get('failure_type') == 'scientific':
        return 'scientific_failure', status
    return 'execution_failure', status


def batch(output, data_root):
    out = p.safe_output(output)
    p.check(out)
    manifest = p.read(out/'manifest.json'); settings = manifest['policy']['scheduler']
    # The directory/marker are exclusive; a second controller never resumes or retries.
    if (out/'controller.json').exists() or (out/'tasks').exists():
        raise FileExistsError('Batch has already been started; no automatic resume or retry')
    with (out/'controller.lock').open('a') as master:
        fcntl.flock(master, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = list(manifest['groups']); slots = []; last_start = {}; completed = 0
        failed = []; stop = {'reason': None}; events = out/'scheduler_events.jsonl'
        def event(kind, **kw):
            with events.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(at=now(), kind=kind, **kw), sort_keys=True, allow_nan=False)+'\n')
        def stop_dispatch(signum, frame):
            stop['reason'] = 'signal_' + str(signum)
            event('drain_requested', signal=signum)
        signal.signal(signal.SIGTERM, stop_dispatch)
        signal.signal(signal.SIGINT, stop_dispatch)
        def status():
            p.write(out/'controller.json', dict(pid=os.getpid(), at=now(),
                fingerprint=manifest['fingerprint'], status='draining' if stop['reason'] else 'training',
                completed=completed, failed=failed, pending_groups=len(pending),
                active=[dict(group_id=s['group']['group_id'], gpu=s['gpu'],
                    task_id=s.get('task_id'), pid=s['process'].pid if s.get('process') else None,
                    remaining_tasks=len(s['remaining'])) for s in slots],
                stop_reason=stop['reason']))
        event('controller_start', pid=os.getpid(), fingerprint=manifest['fingerprint'], settings=settings)
        status()
        try:
            while pending or slots:
                for slot in list(slots):
                    process = slot.get('process')
                    if process is None or process.poll() is None:
                        continue
                    slot['log'].close()
                    code = process.returncode; tid = slot['task_id']
                    outcome, result = task_outcome(out/'tasks'/tid, code)
                    event('exit', task_id=tid, group_id=slot['group']['group_id'], gpu=slot['gpu'],
                          pid=process.pid, returncode=code, outcome=outcome)
                    if outcome == 'complete':
                        completed += 1
                    else:
                        failed.append(dict(task_id=tid, outcome=outcome, status=result))
                        if outcome == 'execution_failure':
                            stop['reason'] = 'execution_failure:' + tid
                    slot['process'] = None
                    if not slot['remaining'] or stop['reason']:
                        slots.remove(slot)
                if stop['reason']:
                    slots[:] = [s for s in slots if s.get('process')]
                    status()
                    if not slots:
                        break
                    time.sleep(3)
                    continue
                rows = gpu_snapshot()
                event('telemetry', gpus=rows, active_workers=sum(bool(s.get('process')) for s in slots))
                if len(slots) < settings['max_workers'] and pending:
                    choice = eligible_gpu(rows, slots, last_start, settings, time.monotonic())
                    if choice:
                        group = pending.pop(0)
                        slots.append(dict(group=group, gpu=choice['index'], uuid=choice['uuid'],
                                          remaining=list(group['tasks']), process=None))
                for slot in slots:
                    if slot.get('process') or not slot['remaining']:
                        continue
                    gpu = slot['gpu']
                    if time.monotonic()-last_start.get(gpu, -1e20) < settings['same_gpu_start_spacing_seconds']:
                        continue
                    before = next(r for r in gpu_snapshot() if r['index'] == gpu)
                    if before['free_mib'] < settings['minimum_free_mib']:
                        continue
                    # The controller lock serializes both admission probes and all launches.
                    probe_start = time.monotonic()
                    time.sleep(settings['admission_delay_seconds'])
                    after = next(r for r in gpu_snapshot() if r['index'] == gpu)
                    if stop['reason'] or after['uuid'] != slot['uuid'] or after['free_mib'] < settings['minimum_free_mib']:
                        continue
                    tid = slot['remaining'].pop(0)
                    if (out/'tasks'/tid).exists():
                        raise FileExistsError('Existing task would be launched twice: '+tid)
                    log_dir = out/'worker_logs'; log_dir.mkdir(exist_ok=True)
                    log = (log_dir/(tid+'.log')).open('x')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot['uuid'], OMP_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    command = [sys.executable, '-u', str(p.ROOT/'run_hfcdr_validation.py'),
                               '--run-task', tid, '--output', str(out), '--data-root', str(data_root)]
                    worker = subprocess.Popen(command, env=env, cwd=p.ROOT, stdout=log, stderr=subprocess.STDOUT)
                    slot.update(process=worker, task_id=tid, log=log)
                    last_start[gpu] = time.monotonic()
                    event('start', task_id=tid, group_id=slot['group']['group_id'], gpu=gpu,
                          uuid=slot['uuid'], pid=worker.pid, admission_before=before, admission_after=after,
                          admission_elapsed_seconds=time.monotonic()-probe_start,
                          active_workers=sum(bool(s.get('process')) for s in slots))
                    status()
                status()
                time.sleep(5)
        finally:
            # Never terminate workers on a controller exception. Retain exact evidence.
            live = [s['task_id'] for s in slots if s.get('process') and s['process'].poll() is None]
            p.write(out/'controller.json', dict(pid=os.getpid(), at=now(), fingerprint=manifest['fingerprint'],
                status='complete' if completed+len(failed) == len(manifest['tasks']) and not live else 'incomplete',
                completed=completed, failed=failed, pending_groups=len(pending), active=live,
                stop_reason=stop['reason'], automatic_retry=False))
            event('controller_terminal', completed=completed, failed=len(failed), active=live,
                  stop_reason=stop['reason'])
