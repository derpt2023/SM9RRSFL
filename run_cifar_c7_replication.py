#!/usr/bin/env python3
"""Fixed C7 clean controls and independent-seed replication on local MPS."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from run_cifar_attack_stability import load_protocol, read_json, write_json, digest, utc
from scripts.experiments.cifar10.attack_stability_worker import execute_task, task_identity

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'configs/cifar10_c7_replication_v1.json'
TERMINAL = {'complete', 'failed', 'numerical_failure'}
PROFILE = {'id': 'C7', 'optimizer': 'adam', 'lr': .001, 'boost': 20.,
           'stealth_steps': 10, 'scope': 'weights'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_replication(path=DEFAULT_CONFIG):
    p = read_json(path)
    expected = {'protocol': 'cifar_c7_replication_v1', 'partition': 'dirichlet',
                'num_clients': 100, 'rounds': 30, 'clean_seed': 301,
                'replication_seed': 401, 'ratios': [.6, .8], 'sm9_workers': 8,
                'profile': PROFILE, 'target_selection': 'original_per_experiment_seed',
                'task_count': 8}
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Configuration differs from the authorized fixed C7 replication')
    out = (ROOT / p['output_dir']).resolve()
    reference = (ROOT / p['reference_dir']).resolve()
    if (not out.is_relative_to(ROOT / 'outputs') or out == reference
            or out.is_relative_to(reference) or reference.is_relative_to(out)):
        raise ValueError('New outputs must be isolated under outputs, outside the reference')
    _, bases, data = load_protocol(ROOT / p['source_protocol'])
    return p, bases, data


def task_specs(protocol, bases):
    rows = []
    for method in ('sm9rrs', 'vert'):
        config = replace(bases[method], partition='dirichlet', malicious_ratio=0.,
                         seed=301, sm9_workers=8)
        rows.append({'task_id': f'clean_{method}_s301', 'config': asdict(config), 'profile': None})
    # Repeat the harder scenario first, without outcome-dependent selection.
    for ratio in (.8, .6):
        for method in ('fedavg', 'sm9rrs', 'vert'):
            config = replace(bases[method], partition='dirichlet', malicious_ratio=ratio,
                             seed=401, sm9_workers=8)
            rows.append({'task_id': f'C7_{method}_dir{int(ratio*100)}_s401',
                         'config': asdict(config), 'profile': dict(protocol['profile'])})
    return rows


def reference_files(protocol):
    directory = ROOT / protocol['reference_dir']
    old = read_json(directory / 'manifest.json')
    if old['fingerprint'] != protocol['reference_fingerprint']:
        raise ValueError('Unexpected reference experiment')
    if digest({k: v for k, v in old.items() if k != 'fingerprint'}) != old['fingerprint']:
        raise ValueError('Reference manifest is corrupt')
    for f, expected in old['sources'].items():
        if sha(ROOT / f) != expected:
            raise ValueError(f'Frozen source changed: {f}')
    selected = read_json(directory / 'recommended_attack.json')
    if selected['profile'] != PROFILE or not selected['audit']['effective']:
        raise ValueError('Reference did not select the expected effective C7')
    paths = [directory / name for name in ('manifest.json', 'recommended_attack.json',
                                           'benchmark_result.json', 'screen_brief.json',
                                           'report.json', 'fedavg_selection.json')]
    tasks = sorted((directory / 'tasks').glob('*/task.json'))
    if len(tasks) != 9:
        raise ValueError('Reference must retain all nine completed tasks')
    for identity_path in tasks:
        task = identity_path.parent
        identity = read_json(identity_path)
        status = read_json(task / 'status.json')
        if identity['root_fingerprint'] != old['fingerprint']:
            raise ValueError(f'Reference root mismatch: {task.name}')
        if digest({k: v for k, v in identity.items() if k != 'fingerprint'}) != identity['fingerprint']:
            raise ValueError(f'Reference task identity corrupt: {task.name}')
        if status['fingerprint'] != identity['fingerprint'] or status['status'] != 'complete' or status['completed_round'] != 30:
            raise ValueError(f'Incomplete reference task: {task.name}')
        paths.extend(task / name for name in ('task.json', 'status.json', 'metrics.json',
                                              'targets.json', 'dataset_identity.json', 'result.json'))
    return old, {str(f.relative_to(ROOT)): sha(f) for f in paths}


def build_manifest(protocol, specs, data, config_path=DEFAULT_CONFIG):
    previous, references = reference_files(protocol)
    sources = dict(previous['sources'])
    for f in (Path(__file__).resolve(), Path(config_path).resolve(),
              ROOT / 'scripts/experiments/cifar10/cifar_c7_replication_report.py'):
        sources[str(f.relative_to(ROOT))] = sha(f)
    identity = {'protocol': protocol, 'source_files': sources, 'reference_files': references,
                'task_specs': specs, 'data_config': data, 'backend': 'torch:mps',
                'official_test_used_for_selection': False,
                'seed_semantics': 'Original selectors: 401 changes initialization, partitions, malicious set, batches and single targets; full data split unchanged.'}
    return {**identity, 'fingerprint': digest(identity)}


def verify_frozen(manifest):
    for field in ('source_files', 'reference_files'):
        for f, expected in manifest[field].items():
            if sha(ROOT / f) != expected:
                raise ValueError(f'Immutable {field} changed: {f}')


def seconds_text(seconds):
    if seconds is None:
        return 'estimating'
    seconds = max(0, round(seconds))
    return f'{seconds//3600}h{seconds%3600//60:02d}m{seconds%60:02d}s'


def task_estimates(protocol, specs):
    reference = ROOT / protocol['reference_dir'] / 'tasks'
    values = {}
    for spec in specs:
        c = spec['config']
        ratio = c['malicious_ratio'] if c['malicious_ratio'] else .8
        old = read_json(reference / f'C7_{c["method"]}_dir{int(ratio*100)}' / 'status.json')
        seconds = old['elapsed_seconds']
        if not c['malicious_ratio'] and c['method'] == 'sm9rrs':
            # Clean has no malicious revocation wave; retain other measured work.
            seconds = max(60, seconds - old['summary']['crypto_audit_wall_seconds'])
        values[spec['task_id']] = seconds
    return values


class ReplicationController:
    def __init__(self, protocol, specs, data, manifest, mode):
        self.p, self.specs, self.data, self.manifest = protocol, specs, data, manifest
        self.out = (ROOT / protocol['output_dir']).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        import fcntl
        self.lock = (self.out / 'controller.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = read_json(self.out / 'manifest.json')
        if old is not None and old != manifest:
            raise ValueError('Replication manifest mismatch; preserve this output and use a reviewed new version')
        if old is None and any(f.name != 'controller.lock' for f in self.out.iterdir()):
            raise ValueError('Refusing unidentified preexisting outputs')
        write_json(self.out / 'manifest.json', manifest)
        self.state = read_json(self.out / 'controller_status.json', {})
        self.elapsed_before = self.state.get('elapsed_seconds', 0.)
        if self.state.get('status') == 'running':
            self.elapsed_before += max(0, time.time() - self.state['updated_epoch'])
        self.start = time.time()
        self.estimates = task_estimates(protocol, specs)
        self.state.update(status='running', pid=os.getpid(), started_utc=self.state.get('started_utc', utc()),
                          planned_tasks=8, planned_rounds=240, current=None,
                          explicit_new_time_limit_seconds=None)
        self.child = None
        self.last_display = 0
        self.live = mode == 'live' or (mode == 'auto' and sys.stdout.isatty())
        self.save()

    def save(self):
        self.state.update(updated_utc=utc(), updated_epoch=time.time(),
                          elapsed_seconds=self.elapsed_before + time.time() - self.start)
        write_json(self.out / 'controller_status.json', self.state)

    def display(self, force=False):
        if not force and time.time() - self.last_display < 20:
            return
        self.last_display = time.time()
        rounds = complete = 0
        remaining = 0.
        current_line = ''
        for spec in self.specs:
            status = read_json(self.out / 'tasks' / spec['task_id'] / 'status.json', {})
            done = max(0, status.get('completed_round', 0))
            rounds += done
            complete += status.get('status') == 'complete'
            if status.get('status') not in TERMINAL:
                used = status.get('elapsed_seconds', 0.)
                # ETA is approximate; retain a rate estimate if reference time is exceeded.
                remaining += max(self.estimates[spec['task_id']] - used,
                                 used / max(done, 1) * (30 - done) if done else 0.)
            if spec['task_id'] == self.state.get('current'):
                c = spec['config']
                current_line = (f"MPS {c['method']} seed={c['seed']} ratio={c['malicious_ratio']} "
                                f"round={done}/30 | " + ('clean' if not spec['profile'] else
                                'C7 Adam lr=.001 boost=20 normal_steps<=10 scope=weights'))
        fill = min(36, int(36 * rounds / 240))
        self.state.update(completed_tasks=complete, completed_rounds=rounds,
                          estimated_remaining_seconds=remaining)
        text = (f"[{'#'*fill}{'-'*(36-fill)}] rounds {rounds}/240 | tasks {complete}/8 | "
                f"elapsed {seconds_text(self.elapsed_before + time.time()-self.start)} | "
                f"ETA ~{seconds_text(remaining)} | {current_line or self.state['status']}")
        print(('\r\033[2K' if self.live else '') + text, end='' if self.live else '\n', flush=True)
        self.save()

    def request(self, spec):
        return {**spec, 'kind': 'fl', 'output_dir': str(self.out / 'tasks'),
                'data_config': self.data, 'device': 'mps',
                'fingerprint': self.manifest['fingerprint'],
                'return_path': str(self.out / 'returns' / f"{spec['task_id']}.json")}

    def run_task(self, spec):
        verify_frozen(self.manifest)
        request = self.request(spec)
        task_dir = self.out / 'tasks' / spec['task_id']
        old = read_json(task_dir / 'status.json')
        if old:
            if read_json(task_dir / 'task.json') != task_identity(request):
                raise ValueError('Task identity mismatch')
            if old['status'] in TERMINAL:
                return old
        self.state['current'] = spec['task_id']
        self.display(True)
        path = self.out / 'requests' / f"{spec['task_id']}.json"
        write_json(path, request)
        returns = Path(request['return_path'])
        returns.unlink(missing_ok=True)
        log_path = self.out / 'logs' / f"{spec['task_id']}.log"
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open('a') as log:
            self.child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                           '--worker-request', str(path)], cwd=ROOT,
                                          stdout=log, stderr=subprocess.STDOUT,
                                          env={**os.environ, 'OMP_NUM_THREADS': '2',
                                               'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
            try:
                while self.child.poll() is None:
                    self.display()
                    time.sleep(1)
                code = self.child.returncode
            except BaseException:
                if self.child.poll() is None:
                    self.child.send_signal(signal.SIGINT)
                try:
                    self.child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.child.terminate()
                    self.child.wait(timeout=10)
                raise
            finally:
                self.child = None
        if not returns.exists():
            raise RuntimeError(f"Worker {spec['task_id']} exited {code} without return; see {log_path}")
        return read_json(returns)

    def snapshot(self):
        from scripts.experiments.cifar10.cifar_c7_replication_report import build_report
        report = build_report(self.out, ROOT / self.p['reference_dir'])
        report.update(fingerprint=self.manifest['fingerprint'], updated_utc=utc(),
                      elapsed_seconds=self.elapsed_before + time.time() - self.start)
        write_json(self.out / 'report.json', report)
        return report

    def run(self):
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError('This local replication requires Apple MPS')
        report = self.snapshot()
        if not report['reference_integrity_valid']:
            raise ValueError('Reference report integrity failed; no new training started')
        for spec in self.specs:
            result = self.run_task(spec)
            self.snapshot()
            if result['status'] == 'partial':
                self.state['status'] = 'interrupted'
                return
        verify_frozen(self.manifest)
        report = self.snapshot()
        accepted = (report['all_complete'] and report['all_numerically_finite']
                    and report['reference_integrity_valid'] and report['all_comparisons_available'])
        self.state['status'] = 'complete' if accepted else 'finished_with_failures'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--display-mode', choices=['auto', 'live', 'log'], default='auto')
    parser.add_argument('--worker-request', type=Path, help=argparse.SUPPRESS)
    opts = parser.parse_args(argv)
    if opts.worker_request:
        request = read_json(opts.worker_request)
        write_json(request['return_path'], execute_task(request))
        return 0
    p, bases, data = load_replication(opts.config)
    specs = task_specs(p, bases)
    if opts.summary:
        out = ROOT / p['output_dir']
        print(json.dumps({'controller': read_json(out / 'controller_status.json'),
                          'report': read_json(out / 'report.json')}, ensure_ascii=False, indent=2))
        return 0
    manifest = build_manifest(p, specs, data, opts.config)
    if opts.dry_run:
        print(json.dumps({'protocol': p, 'tasks': specs, 'data': data,
                          'source_files': len(manifest['source_files']),
                          'reference_files': len(manifest['reference_files']),
                          'fingerprint': manifest['fingerprint'],
                          'planned_tasks': 8, 'planned_rounds': 240,
                          'estimates_seconds': task_estimates(p, specs)}, indent=2))
        return 0
    controller = ReplicationController(p, specs, data, manifest, opts.display_mode)
    old_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        controller.run()
    except KeyboardInterrupt:
        controller.state.update(status='interrupted', reason='complete checkpoints retained')
    except Exception as error:
        controller.state.update(status='failed', error_type=type(error).__name__, error=str(error),
                                traceback=traceback.format_exc())
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        controller.display(True)
        if controller.live:
            print()
        controller.save()
    return 0 if controller.state['status'] == 'complete' else 2


if __name__ == '__main__':
    os.chdir(ROOT)
    raise SystemExit(main())
