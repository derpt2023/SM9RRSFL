#!/usr/bin/env python3
"""Bounded local attack screening; original v14 science and artifacts are unchanged."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'configs/cifar10_attack_stability_v1.json'
TERMINAL = {'complete', 'numerical_failure', 'failed'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def load_protocol(path=DEFAULT_CONFIG):
    from sm9rrsfl import experiments, fair_tuning
    from sm9rrsfl.config_runner import parameters_to_argv
    p = read_json(path)
    if p['protocol'] != 'cifar_attack_stability_v1':
        raise ValueError('Unsupported protocol')
    source = ROOT / p['source_config']
    if hashlib.sha256(source.read_bytes()).hexdigest() != p['source_config_sha256']:
        raise ValueError('Source v14 config changed; use a separately reviewed protocol')
    if (p['rounds'], p['num_clients'], p['seed']) != (30, 100, 301):
        raise ValueError('v1 requires 30 rounds, 100 clients, seed 301')
    if not 0 < p['budget_seconds'] <= 7200 or not 0 < p['probe_budget_seconds'] <= 1800:
        raise ValueError('Budget exceeds the authorized two-hour screen')
    if p['probe_processes'] not in (1, 2, 3) or p['probe_threads'] != 2:
        raise ValueError('At most three two-thread probe workers')
    if p['representative_clients'] != 16:
        raise ValueError('v1 requires sixteen representatives per partition')
    expected = [('sgd', None, 20, 1, 'all'), ('sgd', None, 5, 1, 'all'),
                ('sgd', None, 10, 1, 'all'), ('sgd', None, 20, 5, 'all'),
                ('sgd', None, 20, 10, 'all'), ('sgd', None, 20, 10, 'weights'),
                ('adam', .001, 10, 10, 'weights'), ('adam', .001, 20, 10, 'weights')]
    actual = [(q['optimizer'], q['lr'], q['boost'], q['stealth_steps'], q['scope'])
              for q in p['profiles']]
    if actual != expected or [q['id'] for q in p['profiles']] != [f'C{i}' for i in range(8)]:
        raise ValueError('Profiles differ from the approved v1 screen')
    spec = fair_tuning.load_fair_tuning_config(source)
    args = experiments.parse_args(parameters_to_argv(spec.shared_parameters))
    base = experiments.build_experiment_configs(args)[0]
    configs = {}
    for method in ('fedavg', 'sm9rrs', 'vert'):
        if p['candidate_ids'][method] != f'{method}-001':
            raise ValueError('Defense candidates are fixed during attack selection')
        configs[method] = replace(base, **spec.candidates[method][0], method=method,
                                  rounds=30, num_clients=100, seed=301, device='mps')
    data = {'dataset': args.dataset, 'data_dir': str((ROOT / args.data_dir).resolve()),
            'train_samples': args.train_samples, 'test_samples': args.test_samples,
            'seed': args.seed, 'split_seed': spec.split_seed,
            'validation_fraction': spec.validation_fraction}
    return p, configs, data


def source_manifest(protocol):
    files = sorted(set([Path(__file__).resolve(), ROOT / protocol['source_config']]
                       + list((ROOT / 'sm9rrsfl').rglob('*.py'))
                       + list((ROOT / 'scripts/experiments/cifar10').glob('attack_stability_*.py'))))
    sources = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
               for f in files}
    identity = {'protocol': protocol, 'sources': sources, 'backend': 'torch:mps',
                'official_test_used_for_selection': False}
    return {**identity, 'fingerprint': digest(identity)}


def effectiveness(attacked, clean, policy):
    """Paired six-round evidence; no defense outcome enters attack selection."""
    wanted = list(range(policy['first_round'], policy['last_round'] + 1))
    a = [r for r in attacked if r['round'] in wanted]
    c = [r for r in clean if r['round'] in wanted]
    if [r['round'] for r in a] != wanted or [r['round'] for r in c] != wanted:
        return {'effective': False, 'reason': 'incomplete_paired_attack_window'}
    import math
    fields = ('training_target_success', 'training_target_confidence', 'accuracy',
              'source_transfer_asr', 'legacy_asr')
    if any(not math.isfinite(float(r[k])) for r in a + c for k in fields):
        return {'effective': False, 'reason': 'nonfinite_metrics'}
    if any(r.get('nonfinite_updates', 0) or r.get('cumulative_nonfinite_updates', 0) for r in a + c):
        return {'effective': False, 'reason': 'nonfinite_updates'}
    counts = ('training_target_count', 'source_transfer_count', 'legacy_target_count')
    if any(len({r.get(k) for r in a + c}) != 1 or
           any(not isinstance(r.get(k), int) or r[k] <= 0 for r in a + c) for k in counts):
        return {'effective': False, 'reason': 'invalid_metric_denominators'}
    if any(not 0 <= r[k] <= 1 for r in a + c for k in fields):
        return {'effective': False, 'reason': 'invalid_metric_range'}
    mean = lambda rows, key: math.fsum(r[key] for r in rows) / len(rows)
    hit = mean(a, 'training_target_success')
    conf = mean(a, 'training_target_confidence')
    clean_conf = mean(c, 'training_target_confidence')
    high_clean = (mean(c, 'training_target_success') + 1e-12 >= policy['min_target_hit_rate']
                  and clean_conf + 1e-12 >= policy['clean_high_confidence_threshold'])
    gain = conf - clean_conf
    ok = hit + 1e-12 >= policy['min_target_hit_rate'] and gain + 1e-12 >= policy['min_confidence_gain'] and not high_clean
    return {'effective': ok,
            'reason': 'effective' if ok else ('clean_already_high_confidence_target_error'
                                             if high_clean else 'attack_effect_unproven'),
            'rounds': wanted, 'target_hit_rate': hit, 'target_confidence': conf,
            'clean_target_confidence': clean_conf, 'confidence_gain': gain,
            'clean_accuracy_loss': mean(c, 'accuracy') - mean(a, 'accuracy'),
            'source_transfer_asr': mean(a, 'source_transfer_asr'),
            'legacy_asr': mean(a, 'legacy_asr'),
            'training_target_counts': sorted({r['training_target_count'] for r in a}),
            'source_transfer_counts': sorted({r['source_transfer_count'] for r in a}),
            'legacy_target_counts': sorted({r['legacy_target_count'] for r in a})}


def candidate_probe_job(requests):
    from scripts.experiments.cifar10.attack_stability_probes import run_probe, probe_rankable
    results = []
    for request in requests:
        result = run_probe(request)
        results.append(result)
        if not probe_rankable(result):
            break
    return results


def micro_rank(results):
    from scripts.experiments.cifar10.attack_stability_probes import probe_rankable
    if len(results) != 4 or not all(probe_rankable(r) for r in results):
        return None
    if ({(r['partition'], r['anchor_id']) for r in results} !=
            {(p, a) for p in ('iid', 'dirichlet') for a in ('r0', 'r24')}
            or len({r['profile_id'] for r in results}) != 1):
        return None
    return {'target_hit_rate': sum(r['mean_target_success'] for r in results) / 4,
            'target_confidence': sum(r['mean_target_confidence'] for r in results) / 4,
            'seconds': sum(r['runtime_seconds'] for r in results)}


def rank_key(item):
    r = item['rank']
    return (-r['target_hit_rate'], -r['target_confidence'], r['seconds'], item['profile']['id'])


def worker_dispatch(request):
    kind = request['kind']
    if kind == 'fl':
        from scripts.experiments.cifar10.attack_stability_worker import execute_task
        result = execute_task(request)
    elif kind == 'probe':
        from scripts.experiments.cifar10.attack_stability_probes import run_probe
        result = run_probe(request)
    elif kind == 'benchmark':
        from scripts.experiments.cifar10.attack_stability_probes import sm9_threads_benchmark
        result = sm9_threads_benchmark(request['outdir'], request['deadline_epoch'])
    else:
        raise ValueError(kind)
    write_json(request['return_path'], result)
    return result


class Controller:
    def __init__(self, protocol, configs, data, mode='auto'):
        self.p, self.configs, self.data = protocol, configs, data
        self.out = (ROOT / protocol['output_dir']).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        import fcntl
        self.lock = (self.out / 'controller.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.manifest = source_manifest(protocol)
        old = read_json(self.out / 'manifest.json')
        if old is not None and old != self.manifest:
            raise ValueError('Immutable screen manifest mismatch; preserve this output and use a new version')
        write_json(self.out / 'manifest.json', self.manifest)
        self.state = read_json(self.out / 'controller_status.json', {})
        self.elapsed_before = self.state.get('elapsed_seconds', 0)
        # An unclean controller exit must not reset the already-started time budget.
        if self.state.get('status') == 'running':
            self.elapsed_before += max(0, time.time() - self.state['updated_epoch'])
        self.start = time.time()
        self.deadline = self.start + max(0, protocol['budget_seconds'] - self.elapsed_before)
        self.state.update(status='running', pid=os.getpid(), started_utc=self.state.get('started_utc', utc()),
                          deadline_epoch=self.deadline, current='initializing',
                          sm9_workers=self.state.get('sm9_workers', 1))
        self.live = mode == 'live' or (mode == 'auto' and sys.stdout.isatty())
        self.last_display = 0
        self.child = None
        self.save()

    def save(self):
        self.state.update(updated_utc=utc(), updated_epoch=time.time(),
                          elapsed_seconds=self.elapsed_before + time.time() - self.start)
        write_json(self.out / 'controller_status.json', self.state)

    def remaining(self):
        return max(0, self.deadline - time.time())

    def display(self, force=False):
        if not force and time.time() - self.last_display < 20:
            return
        self.last_display = time.time()
        tasks = [read_json(p) for p in sorted((self.out / 'tasks').glob('*/status.json'))]
        rounds = sum(t.get('completed_round', 0) for t in tasks)
        total = 270  # adaptive maximum: 2 clean + 2 attack selectors + 5 comparison runs.
        elapsed = self.p['budget_seconds'] - self.remaining()
        width = 30
        fill = min(width, int(width * elapsed / self.p['budget_seconds']))
        bar = '[' + '#' * fill + '-' * (width - fill) + ']'
        eta = round(self.remaining())
        current = self.state['current']
        text = f'{bar} budget {elapsed/60:.1f}/120m | remaining {eta//60}m{eta%60:02d}s | FL rounds {rounds}/{total} max | {current}'
        if tasks:
            running = [t for t in tasks if t['status'] not in TERMINAL]
            text += ' | ' + ', '.join(f"{t['task_id']} r{t.get('completed_round',0)}/30" for t in running)
        print(('\r\033[2K' if self.live else '') + text,
              end='' if self.live else '\n', flush=True)
        self.save()

    def subprocess_task(self, name, request):
        request = dict(request)
        request['deadline_epoch'] = min(request.get('deadline_epoch', self.deadline), self.deadline)
        request['return_path'] = str(self.out / 'returns' / f'{name}.json')
        path = self.out / 'requests' / f'{name}.json'
        write_json(path, request)
        return_path = Path(request['return_path'])
        # Return envelopes are disposable; worker-owned immutable evidence is not.
        return_path.unlink(missing_ok=True)
        logs = self.out / 'logs'
        logs.mkdir(exist_ok=True)
        self.state['current'] = f"{name} device={request.get('device','cpu')}"
        self.display(True)
        with (logs / f'{name}.log').open('a') as log:
            self.child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                           '--worker-request', str(path)], cwd=ROOT,
                                          stdout=log, stderr=subprocess.STDOUT,
                                          env={**os.environ, 'OMP_NUM_THREADS': '2',
                                               'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
            interrupted_at = None
            try:
                while self.child.poll() is None:
                    if time.time() >= request['deadline_epoch'] + 5 and interrupted_at is None:
                        self.child.send_signal(signal.SIGINT)
                        interrupted_at = time.time()
                    if interrupted_at and time.time() - interrupted_at > 30:
                        self.child.terminate()
                    self.display()
                    time.sleep(1)
            except BaseException:
                self.child.send_signal(signal.SIGINT)
                try:
                    self.child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.child.terminate()
                    self.child.wait(timeout=10)
                raise
            finally:
                self.child = None
        if return_path.exists():
            return read_json(return_path)
        return {'status': 'failed', 'task_id': name,
                'reason': 'worker_exited_without_return', 'log': str(logs / f'{name}.log')}

    def fl_request(self, task_id, method, partition, ratio, profile=None):
        config = replace(self.configs[method], partition=partition, malicious_ratio=ratio,
                         sm9_workers=self.state['sm9_workers'])
        return {'kind': 'fl', 'task_id': task_id, 'output_dir': str(self.out / 'tasks'),
                'config': asdict(config), 'profile': profile, 'data_config': self.data,
                'device': 'mps', 'fingerprint': self.manifest['fingerprint']}

    def run_fl(self, task_id, method, partition, ratio, profile=None, estimate=0):
        existing = read_json(self.out / 'tasks' / task_id / 'status.json')
        if existing and existing['status'] in TERMINAL:
            return existing
        if self.remaining() < max(30, estimate):
            result = {'status': 'not_started', 'task_id': task_id,
                      'reason': 'insufficient_budget', 'estimated_seconds': estimate,
                      'remaining_seconds': self.remaining()}
            write_json(self.out / 'not_started' / f'{task_id}.json', result)
            return result
        return self.subprocess_task(task_id, self.fl_request(task_id, method, partition, ratio, profile))

    def representatives(self):
        from scripts.experiments.cifar10.attack_stability_probes import _load_data, select_representatives, SELECTION_RULE
        from sm9rrsfl import fl
        from sm9rrsfl.datasets import partition_clients
        old = read_json(self.out / 'representatives.json')
        if old:
            return old
        _, dataset = _load_data(self.data)
        config = self.configs['fedavg']
        malicious = fl._choose_malicious([f'client-{i}' for i in range(100)], .8, config.seed)
        indices = [int(i.split('-')[1]) for i in malicious]
        result = {'rule': SELECTION_RULE, 'malicious_indices': sorted(indices), 'partitions': {}}
        for partition in ('iid', 'dirichlet'):
            parts = partition_clients(dataset.y_train, 100, strategy=partition,
                                      dirichlet_alpha=config.dirichlet_alpha, seed=config.seed)
            chosen = select_representatives(dataset.y_train, parts, indices, 16, 5)
            result['partitions'][partition] = {
                'clients': chosen,
                'metadata': [{'client': i, 'samples': len(parts[i]),
                              'source_samples': int((dataset.y_train[parts[i]] == 5).sum())} for i in chosen]}
        write_json(self.out / 'representatives.json', result)
        return result

    def probe_requests(self, profile, reps, deadline):
        requests = []
        for anchor in (0, 24):
            for partition in ('dirichlet', 'iid'):
                task_id = f"{profile['id']}_{partition}_r{anchor}_cpu"
                requests.append({'kind': 'probe', 'task_id': task_id, 'profile': profile,
                    'outdir': str(self.out / 'probes'), 'data_config': self.data,
                    'config': asdict(replace(self.configs['fedavg'], partition=partition, malicious_ratio=.8)),
                    'anchor_path': str(self.out / 'tasks' / f'clean_{partition}' / f'anchor_r{anchor}.npz'),
                    'anchor_id': f'r{anchor}', 'clients': reps['partitions'][partition]['clients'],
                    'device': 'cpu', 'deadline_epoch': deadline, 'fingerprint': self.manifest['fingerprint']})
        return requests

    def screen_probes(self):
        from scripts.experiments.cifar10.attack_stability_probes import initialize_probe_worker, probe_rankable
        stored = read_json(self.out / 'micro_screen.json')
        if stored is not None:
            return stored
        reps = self.representatives()
        deadline = min(self.deadline, time.time() + self.p['probe_budget_seconds'])
        self.state.update(current='CPU probes: up to 3 processes x 2 threads', probe_deadline_epoch=deadline)
        self.display(True)
        rows = []
        pool = ProcessPoolExecutor(max_workers=self.p['probe_processes'],
                                   mp_context=multiprocessing.get_context('spawn'),
                                   initializer=initialize_probe_worker, initargs=(self.data,))
        futures = {pool.submit(candidate_probe_job, self.probe_requests(q, reps, deadline)): q
                   for q in self.p['profiles']}
        aborted = False
        try:
            while futures:
                if time.time() > deadline + 10:
                    aborted = True
                    for profile in futures.values():
                        rows.append({'profile': profile, 'results': [], 'rank': None,
                                     'status': 'partial', 'reason': 'probe_budget_expired'})
                    break
                done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    q = futures.pop(future)
                    try:
                        results = future.result()
                        row = {'profile': q, 'results': results, 'rank': micro_rank(results)}
                    except Exception as exc:
                        row = {'profile': q, 'results': [], 'rank': None,
                               'status': 'failed', 'error': repr(exc)}
                    rows.append(row)
                    write_json(self.out / 'micro_progress.json', rows)
                    self.state['current'] = f"CPU probes: {len(rows)}/8 candidates returned; latest {q['id']} {'finite' if row['rank'] else 'not qualified'}"
                    self.display(True)
                self.display()
        except BaseException:
            aborted = True
            raise
        finally:
            # Interrupt only this controller's child processes; preserve disk evidence.
            if aborted:
                children = list(pool._processes.values())
                for child in children:
                    if child.is_alive():
                        os.kill(child.pid, signal.SIGINT)
                grace = time.time() + 5
                for child in children:
                    child.join(timeout=max(0, grace - time.time()))
                for child in children:
                    if child.is_alive():
                        child.terminate()
                for child in children:
                    child.join(timeout=5)
            pool.shutdown(wait=not aborted, cancel_futures=True)
        for row in sorted(rows, key=lambda r: r['profile']['id']):
            if row['rank'] is None:
                continue
            worst = max(row['results'], key=lambda r: r['worst_growth_ratio'])
            requests = self.probe_requests(row['profile'], reps, self.deadline)
            req = next(r for r in requests if r['task_id'] == worst['task_id'])
            req.update(task_id=worst['task_id'].replace('_cpu', '_mps'), device='mps',
                       clients=[worst['worst_client_idx']])
            if self.remaining() < 30:
                row['mps_replay'] = {'status': 'not_started', 'reason': 'insufficient_budget'}
            else:
                row['mps_replay'] = self.subprocess_task(req['task_id'], req)
            row['qualified'] = probe_rankable(row['mps_replay'])
            write_json(self.out / 'micro_progress.json', rows)
        write_json(self.out / 'micro_screen.json', rows)
        return rows

    def run(self):
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError('This approved local protocol requires Apple MPS')
        benchmark = read_json(self.out / 'benchmark_result.json')
        if benchmark is None and self.remaining() > 30:
            benchmark = self.subprocess_task('sm9_benchmark', {
                'kind': 'benchmark', 'outdir': str(self.out / 'sm9_benchmark'),
                'deadline_epoch': min(self.deadline, time.time() + 120)})
            write_json(self.out / 'benchmark_result.json', benchmark)
        self.state['sm9_workers'] = (benchmark or {}).get('selected_workers', 1)
        for partition in ('iid', 'dirichlet'):
            result = self.run_fl(f'clean_{partition}', 'fedavg', partition, 0.)
            if result['status'] != 'complete':
                return {'status': 'incomplete', 'reason': 'clean_reference_incomplete', 'task': result}
        rows = self.screen_probes()
        finalists = sorted([r for r in rows if r.get('qualified')], key=rank_key)[:2]
        if not finalists:
            return {'status': 'no_qualified_attack', 'reason': 'no_candidate_completed_cpu_and_mps_screen'}
        clean = read_json(self.out / 'tasks/clean_dirichlet/metrics.json')['rounds']
        validations = []
        for row in finalists:
            q = row['profile']
            task = self.run_fl(f"{q['id']}_fedavg_dir80", 'fedavg', 'dirichlet', .8, q,
                               estimate=180)
            audit = {'effective': False, 'reason': 'incomplete_or_nonfinite_task'}
            if task['status'] == 'complete':
                metrics = read_json(self.out / 'tasks' / task['task_id'] / 'metrics.json')['rounds']
                audit = effectiveness(metrics, clean, self.p['effectiveness'])
            validations.append({'profile': q, 'task': task, 'audit': audit})
            write_json(self.out / 'fedavg_selection.json', validations)
        effective = [v for v in validations if v['audit']['effective']]
        if not effective:
            return {'status': 'no_effective_attack', 'validations': validations,
                    'reason': 'FedAvg six-round attack effectiveness was not established'}
        winner = sorted(effective, key=lambda v: (-v['audit']['target_hit_rate'],
                        -v['audit']['confidence_gain'], v['task']['elapsed_seconds'], v['profile']['id']))[0]
        write_json(self.out / 'recommended_attack.json', {**winner,
            'scope': 'seed301 30-round local MPS screen only', 'selected_using': 'FedAvg Dir80 only',
            'full_150_round_qualification': False})
        comparisons = []
        for method, ratio in [('sm9rrs', .8), ('vert', .8), ('fedavg', .6), ('sm9rrs', .6), ('vert', .6)]:
            task = self.run_fl(f"{winner['profile']['id']}_{method}_dir{int(ratio*100)}",
                               method, 'dirichlet', ratio, winner['profile'],
                               estimate=180 if method == 'fedavg' else 600)
            comparisons.append(task)
            if task['status'] == 'numerical_failure':
                # Failure is evidence against stability, never a reason to switch attacks for Ours.
                break
        return {'status': 'screen_complete' if len(comparisons) == 5 and
                all(c['status'] == 'complete' for c in comparisons) else 'screen_partial',
                'winner': winner, 'comparisons': comparisons, 'validations': validations}

    def finish(self, report):
        report.update(updated_utc=utc(), elapsed_seconds=self.elapsed_before + time.time() - self.start,
                      official_test_used_for_selection=False,
                      main_150_round_target_status='not_evaluated',
                      defense_clean_health_status='not_evaluated',
                      limits='Single seed, 30 rounds and six attack rounds; no 150-round or tail-ten qualification.')
        write_json(self.out / 'report.json', report)
        self.state.update(status=report['status'], current=report['status'])
        self.display(True)
        if self.live:
            print()
        self.save()
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--display-mode', choices=['auto', 'live', 'log'], default='auto')
    parser.add_argument('--worker-request', type=Path, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    if options.worker_request:
        worker_dispatch(read_json(options.worker_request))
        return 0
    protocol, configs, data = load_protocol(options.config)
    if options.dry_run:
        print(json.dumps({'protocol': protocol, 'data_config': data,
                          'configs': {k: asdict(v) for k, v in configs.items()},
                          'upper_work': {'fl_tasks': 9, 'fl_rounds': 270, 'cpu_client_probes': 512,
                                         'mps_client_replays': 8},
                          'execution': 'one MPS worker; at most three two-thread CPU probes'},
                         indent=2, sort_keys=True))
        return 0
    if options.summary:
        out = ROOT / protocol['output_dir']
        print(json.dumps({'controller': read_json(out / 'controller_status.json'),
                          'report': read_json(out / 'report.json'),
                          'candidates': read_json(out / 'micro_screen.json', read_json(out / 'micro_progress.json'))},
                         indent=2, sort_keys=True))
        return 0
    controller = Controller(protocol, configs, data, options.display_mode)
    try:
        report = controller.run()
    except KeyboardInterrupt:
        report = {'status': 'interrupted', 'reason': 'user_interrupt; complete checkpoints preserved'}
    except Exception as exc:
        report = {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc),
                  'traceback': traceback.format_exc()}
    controller.finish(report)
    return 0 if report['status'] in ('screen_complete', 'screen_partial', 'no_effective_attack', 'no_qualified_attack') else 2


if __name__ == '__main__':
    os.chdir(ROOT)
    raise SystemExit(main())
