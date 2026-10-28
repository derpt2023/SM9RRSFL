#!/usr/bin/env python3
"""Read-only live dashboard and timing export for manifest-based CIFAR runs.

The observer never starts, signals, resumes or changes a training process.
Stopping this display leaves the controller and GPU workers running.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import time


CRYPTO_SPANS = ('crypto_setup_wall_seconds', 'crypto_packet_wall_seconds',
                'crypto_audit_wall_seconds', 'crypto_finalize_wall_seconds')
STAGES = ('training_seconds', 'attack_seconds', 'detection_seconds',
          'aggregation_seconds', 'evaluation_seconds', 'hash_seconds',
          'packet_build_seconds', 'sign_seconds', 'verify_seconds') + CRYPTO_SPANS
TERMINAL = ('complete', 'failed', 'numerical_failure', 'partial')


def read(path):
    try:
        return json.loads(path.read_text()), None
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as exc:
        return None, f'{path.name}: {type(exc).__name__}'


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def timing(summary):
    """Missing evidence stays null; parallel per-operation sums are not deducted."""
    total = summary.get('runtime_seconds')
    crypto = summary.get('crypto_wall_seconds')
    excluding = summary.get('runtime_without_crypto_seconds')
    values = (total, crypto, excluding, *(summary.get(k) for k in CRYPTO_SPANS))
    complete = all(finite(v) and v >= 0 for v in values)
    error = None
    if complete and (crypto > total + 1e-6
                     or not math.isclose(sum(summary[k] for k in CRYPTO_SPANS), crypto, abs_tol=1e-6)
                     or not math.isclose(total - crypto, excluding, abs_tol=1e-6)):
        error = 'inconsistent_crypto_wall_accounting'
    return dict(runtime_with_crypto_seconds=total, crypto_wall_seconds=crypto,
        runtime_without_crypto_seconds=excluding,
        stages={k: summary.get(k) for k in STAGES},
        available_and_consistent=bool(complete and error is None), error=error,
        definition='Recorded worker wall time minus non-overlapping measured crypto spans; not a crypto-disabled rerun. '
                   'For the frozen development worker this includes data loading, evaluation, observation and checkpoint I/O; '
                   'its checkpoint_io_seconds field is not a measured exclusion. Shared-GPU wall time is not an isolated speed benchmark.')


def snapshot(output):
    manifest, error = read(output / 'manifest.json')
    if error or not manifest:
        raise ValueError(error or 'No existing manifest; this observer does not start experiments')
    controller, controller_error = read(output / 'controller_status.json')
    controller = controller or {}
    specs = {s['task_id']: s for s in manifest['specs']}
    locations = {n: output / 'tasks' / n for n in specs}
    # A frozen validation matrix may also point to already completed parent cells.
    for name, loc in manifest.get('protocol', {}).get('reuse', {}).items():
        root = output.resolve().parents[1]
        owner = root / loc['output']
        parent, issue = read(owner / 'manifest.json')
        if issue or not parent:
            raise ValueError('Cannot locate reused manifest: ' + name)
        spec = next((s for s in parent['specs'] if s['task_id'] == loc['task']), None)
        if spec is None or name in specs:
            raise ValueError('Invalid or duplicate reused task: ' + name)
        specs[name] = spec
        locations[name] = owner / 'tasks' / loc['task']
    cells = {}
    for name, spec in specs.items():
        folder = locations[name]
        state, issue = read(folder / 'status.json')
        state = state or {}
        row = state.get('final_metrics', {})
        if not row:
            metrics, metric_issue = read(folder / 'metrics.json')
            issue = issue or metric_issue
            rows = (metrics or {}).get('rounds', [])
            row = rows[-1] if rows else {}
        completed = state.get('completed_round')
        sidecar, timing_issue = read(folder / 'timing.json')
        issue = issue or timing_issue
        cells[name] = dict(status='unreadable' if issue else state.get('status', 'pending'),
            read_error=issue, round=completed, rounds=spec['config']['rounds'],
            accuracy=row.get('accuracy'), asr=row.get('source_transfer_asr'),
            elapsed_seconds=state.get('elapsed_seconds'),
            timing=sidecar if sidecar is not None else timing(state.get('summary', {})),
            gpu=controller.get('active', {}).get(name, {}).get('gpu_index'),
            gpu_uuid=controller.get('active', {}).get(name, {}).get('gpu_uuid'),
            method=spec['config']['method'], seed=spec['config']['seed'],
            partition=spec['config']['partition'], malicious_ratio=spec['config']['malicious_ratio'])
    return dict(observed_utc=datetime.now(timezone.utc).isoformat(),
        manifest_fingerprint=manifest['fingerprint'], controller_status=controller.get('status', 'unknown'),
        controller_read_error=controller_error, controller_failure=controller.get('failure'),
        complete=sum(c['status'] == 'complete' for c in cells.values()), total=len(cells),
        failed_or_partial=sum(c['status'] in ('failed', 'partial', 'numerical_failure') for c in cells.values()),
        executed_rounds=sum(max(0, min(c['round'], c['rounds'])) for c in cells.values() if isinstance(c['round'], int)),
        planned_rounds=sum(c['rounds'] for c in cells.values()), cells=cells,
        read_only=True, complete_is_not_health_pass=True)


def duration(seconds):
    if not finite(seconds) or seconds < 0:
        return '--'
    seconds = int(seconds)
    return f'{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}'


def percent(value):
    return f'{100 * value:.2f}%' if finite(value) else '--'


def display(state, eta=None, width=100):
    done, total = state['executed_rounds'], state['planned_rounds']
    fill = int(24 * done / total) if total else 0
    lines = [f"CIFAR [{('#' * fill).ljust(24, '-')}] rounds {done}/{total}",
             f"Tasks {state['complete']}/{state['total']} complete | failed/partial {state['failed_or_partial']} | "
             f"controller {state['controller_status']} | ETA {duration(eta)}",
             'Completed tasks are not a health verdict. Ctrl-C closes this display only.']
    active = [(n, c) for n, c in state['cells'].items() if c['gpu'] is not None or c['status'] == 'unreadable']
    for name, c in active:
        lines.append(f"GPU {c['gpu']} {name} {c['round']}/{c['rounds']} "
                     f"Acc {percent(c['accuracy'])} ASR {percent(c['asr'])} elapsed {duration(c['elapsed_seconds'])}")
    if not active:
        lines.append('No active GPU workers in the latest saved controller state.')
    if state['controller_failure']:
        lines.append('Failure: ' + str(state['controller_failure']))
    if state['controller_read_error']:
        lines.append('Read warning: ' + state['controller_read_error'])
    return '\n'.join(line[:max(40, width)] for line in lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--json', action='store_true', help='Print one complete snapshot including timings')
    parser.add_argument('--refresh', type=float, default=2.)
    parser.add_argument('--log-interval', type=float, default=30.)
    args = parser.parse_args()
    if min(args.refresh, args.log_interval) <= 0:
        parser.error('Refresh intervals must be positive')
    start = time.monotonic(); initial = None; last_log = float('-inf')
    try:
        while True:
            state = snapshot(args.output)
            if args.json:
                print(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False)); return
            if initial is None: initial = state['executed_rounds']
            progress = state['executed_rounds'] - initial
            eta = ((state['planned_rounds'] - state['executed_rounds']) * (time.monotonic() - start) / progress
                   if progress > 0 else None)
            terminal = state['controller_status'] in ('panel_terminal', 'stopped_after_drain', 'failed',
                                                       'blocked', 'interrupted_or_blocked')
            if sys.stdout.isatty() or args.once or terminal or time.monotonic() - last_log >= args.log_interval:
                if sys.stdout.isatty() and not args.once: print('\033[2J\033[H', end='')
                print(display(state, eta, shutil.get_terminal_size((100, 24)).columns), flush=True)
                last_log = time.monotonic()
            if args.once or terminal: return
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print('\nDisplay closed; training was not signalled.')


if __name__ == '__main__':
    main()
