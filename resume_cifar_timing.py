#!/usr/bin/env python3
"""User-authorized one-off continuation after the GPU1 OOM on 2026-11-02.

Wait for the original controller to drain; reuse its four completed tasks and
run the 175 unstarted tasks plus one explicitly authorized fresh TAD attempt.
The original 191 sources and all original artifacts remain immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import run_cifar_timing as t

ROOT = t.ROOT
OLD = ROOT / 'outputs/old/cifar_mnist_timing150_v1'
OUTPUT = ROOT / 'outputs/old/cifar_mnist_timing150_followup_v1'
ALLOWED_INDICES = {'2', '4', '5', '7'}
ALLOWED_UUIDS = {
    'GPU-f3b82a8d-b1fb-1320-896e-50e1d846ffa7',
    'GPU-f5517558-d812-fe0c-8a93-654da0b7af75',
    'GPU-f6948b25-3ecf-1ca1-33b1-bd2618fbccb7',
    'GPU-6c2e2803-c8f8-7f4d-daa0-625be5dd185e'}
FAILED_TASK = 'TAD_iid_r80_s2026102801'
OWN_SOURCES = ('resume_cifar_timing.py', 'tests/test_cifar_timing_followup.py')
HANDOFF_STATUS = ROOT / 'logs/cifar_timing_followup_handoff_status.json'


def own_sources():
    return {name:t.base.sha(ROOT / name) for name in OWN_SOURCES}


def assert_sources(m, own):
    t.verify_manifest(m)
    if own_sources() != own:
        raise ValueError('Followup controller source changed')


def running_original(old):
    processes = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            args = (path / 'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            continue
        if str(old.resolve()) in args and any(a.endswith('/run_cifar_timing.py') for a in args):
            processes.append(int(path.name))
    return processes


def wait_for_drain(old, pid, own, timeout=21600):
    m = t.base.read(old / 'manifest.json')
    started = time.monotonic()
    while True:
        assert_sources(m, own)
        processes = running_original(old)
        state = t.base.read(old / 'controller_status.json')
        if state.get('pid') != pid:
            raise ValueError('Unexpected original controller identity')
        t.base.write(HANDOFF_STATUS, dict(at=t.now(), pid=os.getpid(), status='waiting_for_natural_drain',
            original_pid=pid, original_processes=processes, original_status=state,
            own_sources=own, output=str(OUTPUT)))
        if not processes:
            if state['status'] != 'incomplete' or state.get('active'):
                raise ValueError('Original controller did not finish its incomplete report normally')
            return
        if time.monotonic() - started > timeout:
            raise TimeoutError('Original drain exceeded six hours; no automatic restart')
        time.sleep(15)


def continuation_plan(old):
    m, rows, counts = t.collect(old)
    t.verify_manifest(m)
    if counts != {'complete':4, 'failed':1, 'pending':175}:
        raise ValueError('Scope changed; expected exactly 4 complete / 1 failed / 175 pending')
    completed = {r['task_id'] for r in rows}
    expected_completed = {b['tasks'][0]['task_id'] for b in m['blocks'][:4]}
    if completed != expected_completed:
        raise ValueError('Unexpected completed tasks; no automatic expansion')
    failed = t.base.read(old / 'tasks' / FAILED_TASK / 'status.json')
    if failed['status'] != 'failed' or 'OutOfMemoryError' not in failed.get('error', ''):
        raise ValueError('The only authorized fresh retry is the identified GPU1 OOM')
    events = [json.loads(line) for line in (old / 'events.jsonl').read_text().splitlines()]
    starts = [e for e in events if e['event'] == 'start']
    exits = [e for e in events if e['event'] == 'exit']
    if Counter(e['task_id'] for e in starts) != Counter(completed | {FAILED_TASK}):
        raise ValueError('Original attempt history differs')
    if Counter(e['task_id'] for e in exits) != Counter(completed | {FAILED_TASK}):
        raise ValueError('Original attempts did not all exit')
    plans = []
    for b in m['blocks']:
        reused = [e for e in starts if e['block_id'] == b['block_id'] and e['task_id'] in completed]
        uuid = reused[0]['uuid'] if reused else None
        if uuid is not None and uuid not in ALLOWED_UUIDS:
            raise ValueError('A reused task was not on an authorized GPU')
        if any(e['returncode'] != 0 for e in exits if e['task_id'] in completed):
            raise ValueError('Reused task had failed exit')
        plans.append(dict(block=b, position=1 if reused else 0, required_uuid=uuid))
    return m, plans, completed, events


def prepare(old, out, own):
    m, plans, completed, events = continuation_plan(old)
    assert_sources(m, own)
    out.mkdir(parents=True, exist_ok=False)
    t.base.write(out / 'manifest.json', m)  # unchanged scientific worker contract
    t.base.write(out / 'followup_manifest.json', dict(at=t.now(), own_sources=own,
        original=str(old), original_manifest_sha256=t.base.sha(old / 'manifest.json'),
        original_fingerprint=m['fingerprint'], authorized_retry=FAILED_TASK,
        reuse=sorted(completed), fresh_tasks=176, allowed_uuids=sorted(ALLOWED_UUIDS),
        maximum_tasks_per_gpu=1, automatic_retries=0,
        authorization='User explicitly allowed the 175 pending + 1 fresh failed task on 2026-11-02.'))
    provenance = out / 'provenance'
    provenance.mkdir()
    shutil.copy2(old / 'events.jsonl', provenance / 'original_events.jsonl')
    shutil.copy2(old / 'analysis/timing_results.json', provenance / 'original_incomplete_report.json')
    shutil.copy2(old / 'tasks' / FAILED_TASK / 'status.json', provenance / 'original_oom_status.json')
    shutil.copy2(old / 'logs' / (FAILED_TASK + '.log'), provenance / 'original_oom.log')
    reused_files = {}
    for task_id in sorted(completed):
        src, dest = old / 'tasks' / task_id, out / 'tasks' / task_id
        dest.mkdir(parents=True)
        for path in src.iterdir():
            if path.is_file() and not path.is_symlink():
                shutil.copy2(path, dest / path.name)
                expected = t.base.sha(path)
                if t.base.sha(dest / path.name) != expected:
                    raise ValueError('Reused artifact differs')
                reused_files[f'{task_id}/{path.name}'] = expected
    t.base.write(provenance / 'reuse_sha256.json', reused_files)
    with (out / 'events.jsonl').open('x') as stream:
        for e in events:
            if e.get('task_id') in completed and e['event'] in ('start','exit'):
                stream.write(json.dumps(e, sort_keys=True) + '\n')
    return m, plans


def finish(out, old, own):
    m = t.base.read(out / 'manifest.json')
    assert_sources(m, own)
    report = t.final_report(out)
    events = [json.loads(line) for line in (out / 'events.jsonl').read_text().splitlines()]
    starts = [e for e in events if e['event']=='start']
    fresh_count = len(starts) - 4
    report['continuation'] = dict(original_directory=str(old), original_failure=FAILED_TASK,
        original_failure_preserved=True, reused_successes=4, planned_fresh_tasks=176,
        fresh_followup_attempts=fresh_count, total_attempts_including_original_oom=5 + fresh_count,
        explicit_fresh_retry_count=sum(e['task_id']==FAILED_TASK for e in starts),
        automatic_retry_count=0, own_sources=own)
    if any(e['uuid'] not in ALLOWED_UUIDS for e in events if e['event']=='start'):
        report['execution_audit']['failures'].append('Followup GPU outside authorized four')
    if sum(e['event']=='start' for e in events) != 180:
        report['execution_audit']['failures'].append('Merged attempt count incomplete')
    report['execution_audit']['passed'] = not report['execution_audit']['failures']
    t.base.write(out / 'analysis/timing_results.json', report)
    return report


def run(old, out, own):
    m, pending = prepare(old, out, own)
    control = open(out / 'controller.lock','a+')
    fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = {}
    stopping, failure, last_telemetry = False, None, 0.
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        t.event(out, 'stop_dispatch', signal=signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    t.event(out, 'followup_start', pid=os.getpid(), authorized_retry=FAILED_TASK)
    try:
        while pending or active:
            snap = t.gpu_snapshot()
            t.base.write(out / 'gpu_status.json', snap)
            if time.monotonic() - last_telemetry >= 60:
                with (out / 'telemetry.jsonl').open('a') as stream:
                    stream.write(json.dumps(snap, sort_keys=True)+'\n')
                last_telemetry = time.monotonic()
            for uuid,item in list(active.items()):
                child = item.get('process')
                if child is not None and child.poll() is not None:
                    task_id = item['block']['tasks'][item['position']]['task_id']
                    t.event(out,'exit',task_id=task_id,block_id=item['block']['block_id'],
                        uuid=uuid,pid=child.pid,returncode=child.returncode,telemetry=snap)
                    item['log'].close()
                    path=out/'tasks'/task_id/'status.json'
                    status=t.base.read(path) if path.exists() else {}
                    if child.returncode!=0 or status.get('status')!='complete':
                        failure=f'{task_id}: exit={child.returncode}, status={status.get("status")}'
                        stopping=True
                    item['position']+=1
                    item['process']=None
                if item.get('process') is None and (item['position']==6 or stopping):
                    item['lock'].close()
                    del active[uuid]
            if not stopping:
                for gpu in snap['gpus']:
                    uuid=gpu['uuid']
                    if uuid not in ALLOWED_UUIDS or gpu['index'] not in ALLOWED_INDICES:
                        continue
                    if uuid in active or gpu['free']<8192:
                        continue
                    eligible=[i for i,p in enumerate(pending) if p['required_uuid'] in (None,uuid)]
                    if not eligible:
                        continue
                    lock=t.acquire_gpu(uuid)
                    if lock:
                        item=pending.pop(eligible[0])
                        active[uuid]=dict(item,lock=lock,process=None)
                for uuid,item in active.items():
                    if item.get('process') is not None or stopping:
                        continue
                    first=t.gpu_snapshot()
                    if not t.admitted(uuid,first,first):
                        continue
                    time.sleep(5)
                    second=t.gpu_snapshot()
                    if not t.admitted(uuid,first,second) or stopping:
                        continue
                    assert_sources(m,own)
                    task=item['block']['tasks'][item['position']]
                    if (out/'tasks'/task['task_id']).exists():
                        raise ValueError('Refusing repeated fresh followup task')
                    log_path=out/'logs'/(task['task_id']+'.log')
                    log_path.parent.mkdir(exist_ok=True)
                    log=log_path.open('x')
                    child=subprocess.Popen([sys.executable,str(ROOT/'run_cifar_timing.py'),
                        '--output',str(out),'--task',task['task_id']],cwd=ROOT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid),stdout=log,stderr=subprocess.STDOUT)
                    item.update(process=child,log=log)
                    t.event(out,'start',task_id=task['task_id'],block_id=item['block']['block_id'],
                        uuid=uuid,pid=child.pid,order=item['position'],admission=[first,second])
            counts=Counter()
            for task in (task for b in m['blocks'] for task in b['tasks']):
                path=out/'tasks'/task['task_id']/'status.json'
                counts[t.base.read(path)['status'] if path.exists() else 'pending']+=1
            t.base.write(out/'controller_status.json',dict(at=t.now(),pid=os.getpid(),
                status='draining' if stopping else 'training',failure=failure,counts=dict(counts),
                pending_blocks=len(pending),active=[dict(uuid=u,block_id=i['block']['block_id'],
                    task_id=i['block']['tasks'][i['position']]['task_id'],pid=i['process'].pid if i.get('process') else None)
                    for u,i in active.items()]))
            if not active and (stopping or not pending):
                break
            time.sleep(10)
        report=finish(out,old,own)
        t.base.write(out/'controller_status.json',dict(at=t.now(),pid=os.getpid(),
            status='complete' if report['execution_audit']['passed'] else 'incomplete',
            failure=failure,counts=report['counts'],active=[],pending_blocks=len(pending)))
        return report['execution_audit']['passed']
    except BaseException as exc:
        t.event(out,'controller_error',error=repr(exc))
        t.base.write(out/'controller_status.json',dict(at=t.now(),pid=os.getpid(),
            status='error_draining',failure=repr(exc)))
        for item in active.values():
            if item.get('process') is not None:
                item['process'].wait()
                item['log'].close()
        raise
    finally:
        for item in active.values():
            item['lock'].close()
        control.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--ready',action='store_true')
    mode.add_argument('--handoff',type=int)
    args=parser.parse_args()
    own=own_sources()
    if not args.ready and args.handoff is None:
        print(json.dumps(dict(reuse=4,pending=175,authorized_fresh_retry=FAILED_TASK,
            fresh_total=176,output=str(OUTPUT),allowed_uuids=sorted(ALLOWED_UUIDS)),indent=2))
        return
    m=t.base.read(OLD/'manifest.json')
    assert_sources(m,own)
    if args.ready:
        print(json.dumps(dict(ready=True,own_sources=own,original_fingerprint=m['fingerprint'],
            original_status=t.base.read(OLD/'controller_status.json')),indent=2))
        return
    if OUTPUT.exists():
        raise ValueError('Followup output already exists; do not launch again')
    with open(ROOT/'logs/cifar_timing_followup_handoff.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            wait_for_drain(OLD,args.handoff,own)
            t.base.write(HANDOFF_STATUS,dict(at=t.now(),pid=os.getpid(),status='taking_over',output=str(OUTPUT)))
            success=run(OLD,OUTPUT,own)
            t.base.write(HANDOFF_STATUS,dict(at=t.now(),pid=os.getpid(),
                status='complete' if success else 'incomplete',output=str(OUTPUT)))
            if not success:
                raise SystemExit(1)
        except Exception as exc:
            t.base.write(HANDOFF_STATUS,dict(at=t.now(),pid=os.getpid(),status='failed',error=repr(exc)))
            raise


if __name__=='__main__':
    main()
