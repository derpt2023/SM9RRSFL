#!/usr/bin/env python3
"""Default: inspect the 48-task plan. --prepare never starts training."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

# Fix the inherited experiment's numerical policy before any NumPy/Torch import.
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import hfcdr_pilot_protocol as protocol


def now():
    return datetime.now(timezone.utc).isoformat()


def run_task(output, data_dir, task_id):
    """One explicit fresh worker, no scheduler, automatic continuation or retry."""
    out = protocol.safe_output(output)
    protocol.check_prepared(out)
    manifest = protocol.read(out / 'manifest.json')
    matches = [t for t in manifest['tasks'] if t['task_id'] == task_id]
    if len(matches) != 1:
        raise ValueError('Task must belong to the prepared first batch')
    task = matches[0]
    folder = out / 'tasks' / task_id
    if folder.resolve() != folder or (out / 'worker.lock').is_symlink():
        raise ValueError('Worker paths cannot traverse symlinks')
    if folder.exists():
        raise FileExistsError('Existing task evidence retained; no retry or resume')
    with (out / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from hfcdr_pilot_data import load_validation, participant_record
        from hfcdr_pilot_runtime import configure_cuda, run_loop, assert_no_cryptography
        environment = configure_cuda(manifest)
        dataset, identity = load_validation(data_dir)
        pairing = protocol.read(out / 'pairing.json')[task['config']['partition']][task['condition']]
        if participant_record(dataset.y_train, task['config']) != pairing:
            raise ValueError('Actual client partition, attacker IDs or poison counts differ')
        folder.parent.mkdir(exist_ok=True)
        folder.mkdir()
        protocol.write(folder / 'task.json', task)
        protocol.write(folder / 'environment.json', environment)
        protocol.write(folder / 'dataset_identity.json', identity)
        protocol.write(folder / 'pairing.json', pairing)
        protocol.write(folder / 'status.json', dict(status='running', pid=os.getpid(), at=now(),
                                                   manifest_fingerprint=manifest['fingerprint']))
        def emit(kind, value):
            if kind in ('binding', 'initial_model'):
                protocol.write(folder / (kind + '.json'), value)
            else:
                with (folder / (kind + 's.jsonl')).open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
                    stream.flush()
        try:
            result = run_loop(dataset, task, device='cuda:0', emit=emit)
            assert_no_cryptography()
            protocol.verify_snapshot(protocol.read(out / 'provenance/protected_files.json'))
            if protocol.plan() != manifest:
                raise ValueError('Scientific sources changed during execution')
            from hfcdr_pilot_report import summarize
            protocol.write(folder / 'result.json', result)
            summary = summarize(result)
            protocol.write(folder / 'summary.json', summary)
            protocol.write(folder / 'status.json', dict(status='complete', at=now(),
                manifest_fingerprint=manifest['fingerprint'], healthy=summary['healthy'],
                result_sha256=protocol.sha(folder / 'result.json')))
        except BaseException as error:
            protocol.write(folder / 'status.json', dict(status='failed', at=now(),
                manifest_fingerprint=manifest['fingerprint'], error=repr(error), automatic_retry=False))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--plan', action='store_true', help='Read-only plan (default)')
    actions.add_argument('--prepare', action='store_true', help='Verify data and write a fresh plan only')
    actions.add_argument('--check', action='store_true', help='Read-only prepared-plan and protected-file audit')
    actions.add_argument('--report', action='store_true', help='Read-only JSON report, including missing/failed cells')
    actions.add_argument('--run-task', metavar='TASK_ID', help='Explicitly TRAIN one prepared task; no batch launch')
    parser.add_argument('--output', type=Path, default=protocol.OUTPUT)
    parser.add_argument('--data-dir', type=Path, default=protocol.ROOT / 'data/cifar10')
    args = parser.parse_args(argv)
    if args.prepare:
        result = protocol.prepare(args.output, args.data_dir)
    elif args.check:
        result = protocol.check_prepared(args.output)
    elif args.report:
        protocol.check_prepared(args.output)
        from hfcdr_pilot_report import report
        result = report(protocol.safe_output(args.output), protocol.read(args.output / 'manifest.json'))
    elif args.run_task:
        run_task(args.output, args.data_dir, args.run_task)
        return 0
    else:
        p = protocol.plan()
        result = dict(p['audit'], fingerprint=p['fingerprint'], output=str(args.output),
                      task_ids=[t['task_id'] for t in p['tasks']])
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
