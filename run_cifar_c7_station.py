#!/usr/bin/env python3
"""Fixed C7 six-method CUDA evaluation, isolated from the original v14 grid."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from sm9rrsfl import experiments, fair_tuning
from sm9rrsfl.config_runner import parameters_to_argv
from sm9rrsfl.datasets import load_image_dataset, stratified_training_three_way_split
from sm9rrsfl.ours_calibration import split_metadata, validate_targeted_split
from sm9rrsfl.tuning_resume import runtime_config_key
from scripts.experiments.cifar10 import fair_tuning_dashboard as ui

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'configs/cifar10_c7_station_150r_v1.json'
PROTOCOL = 'cifar10_c7_station_150r_v1'
PROFILE = dict(id='C7', optimizer='adam', lr=.001, boost=20., stealth_steps=10, scope='weights')
FIXED = {
    'sm9rrs': {'detector_distance_threshold': 2.},
    'vert': dict(vert_history_window=10, vert_projection_dim=256,
                 vert_predict_epochs=5, vert_predict_lr=.001, vert_top_k=20,
                 vert_use_ratio_prior=False),
    'alignins': dict(alignins_sparsity=.3, alignins_tda_radius=1., alignins_mpsa_radius=1.),
    'krum': {}, 'ding13': {}, 'fedavg': {},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_plan(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    p = json.loads(path.read_text())
    if p['protocol'] != PROTOCOL or p['attack_profile'] != PROFILE or p['fixed_parameters'] != FIXED:
        raise ValueError('This version requires the validated fixed C7/Ours/V4/A4 protocol')
    shared, e = p['shared_parameters'], p['evaluation']
    required = dict(dataset='cifar10', num_clients=100, rounds=150,
                    partitions=['iid', 'dirichlet'], ratios=[0., .2, .4, .6, .8],
                    local_epochs=1, batch_size=50, lr=.01, lr_decay=.99,
                    train_samples=50000, test_samples=10000, seed=42,
                    attack='alternating_minimization', attack_boost=20., attack_epochs=1,
                    attack_stealth_steps=10, attack_distance_weight=.0001,
                    attack_source_label=5, attack_target_label=7, attack_start_round=25,
                    K=20, dirichlet_alpha=.5, early_stop=False, eval_interval=1,
                    compute_backend='torch', device='auto', jobs='auto')
    if any(shared.get(k) != v for k, v in required.items()):
        raise ValueError('Shared parameters differ from the declared C7 100-client/150-round matrix')
    if (set(shared['methods']) != set(FIXED) or len(shared['methods']) != 6
            or e != dict(validation_fraction=.15, split_seed=20261013,
                         validation_seeds=[301,302,303], final_seeds=[401,402,403],
                         final_jobs='auto', parameter_selection='fixed_before_validation',
                         official_test_used_for_selection=False,
                         run_final_after_complete_health_failures=True, tail_rounds=10)):
        raise ValueError('Expected all six methods, original disjoint seeds and fixed evaluation policy')
    if p['execution'] != dict(torch_cpu_threads=2, gpu_poll_seconds=1,
                              gpu_max_idle_seconds=600, gpu_oom_retries=1):
        raise ValueError('Unsupported execution policy')
    out = (ROOT / shared['output_dir']).resolve()
    outputs = (ROOT / 'outputs').resolve()
    if out.parent != outputs or not out.name.startswith('cifar10_c7_station_150r_'):
        raise ValueError('Use an independent outputs/cifar10_c7_station_150r_* directory')
    args = experiments.parse_args(parameters_to_argv(shared))
    if not args.resume:
        raise ValueError('Round checkpoints and task resume must remain enabled')
    spec = fair_tuning.FairTuningConfig(
        source=path, name=PROTOCOL, description=p['description'], shared_parameters=shared,
        validation_fraction=e['validation_fraction'], split_seed=e['split_seed'],
        validation_seeds=tuple(e['validation_seeds']), final_seeds=tuple(e['final_seeds']),
        trials_per_tunable_method=1, run_final_evaluation=True, final_jobs='auto',
        max_clean_accuracy_drop=.05, min_round_completion_rate=1., max_nonfinite_updates=0,
        objective=dict(fair_tuning.OBJECTIVE_DEFAULTS), objective_mode='fixed_no_selection',
        formal_ratios=tuple(shared['ratios']), calibration_ratios=tuple(shared['ratios']),
        ratio_schedule=None, auto_ours=False,
        candidates={m: (dict(v),) for m, v in FIXED.items()}, performance_target=None)
    bases = experiments.build_experiment_configs(args)
    validation = fair_tuning.build_validation_tasks(spec, bases)
    selected = {m: SimpleNamespace(candidate_id=f'{m}-001', parameters=v) for m,v in FIXED.items()}
    final = fair_tuning.build_final_tasks(spec, bases, selected)
    tasks = {}
    for phase, rows in (('validation', validation), ('final', final)):
        # Canonical CUDA identity is independent of logical card numbering.
        rows = [replace(t, config=replace(t.config, device='cuda')) for t in rows]
        tasks[phase] = sorted(rows, key=lambda t: (t.config.malicious_ratio != 0,
                              t.config.seed, t.config.partition, t.config.malicious_ratio,
                              tuple(FIXED).index(t.method)))
    if any(len(rows) != 180 for rows in tasks.values()):
        raise ValueError('Expected 180 tasks per phase')
    return p, spec, args, tasks


def source_identity(config):
    paths = set(ROOT.joinpath('sm9rrsfl').rglob('*.py'))
    paths.update(ROOT.joinpath('sm9rrsfl').rglob('*.c'))
    paths.update(ROOT.joinpath('sm9rrsfl').rglob('*.h'))
    # This ignored archive is retired, never imported or built by this protocol.
    paths.discard(ROOT / 'sm9rrsfl/_native_rrs.c')
    paths.update(ROOT / name for name in (
        'run_cifar_c7_station.py', 'fair_tuning_progress_display.py',
        'scripts/experiments/cifar10/fair_tuning_dashboard.py',
        'scripts/experiments/cifar10/cifar_fair_gpu_pool.py',
        'scripts/experiments/cifar10/cifar_fair_diagnostics.py',
        'scripts/experiments/cifar10/c7_station_runtime.py',
        'scripts/experiments/cifar10/c7_station_gpu_pool.py',
        'scripts/experiments/cifar10/c7_station_report.py',
        'scripts/experiments/cifar10/attack_stability_attack.py',
        'scripts/experiments/cifar10/attack_stability_worker.py'))
    paths.add(Path(config).resolve())
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def make_manifest(p, spec, tasks):
    from scripts.experiments.cifar10.c7_station_runtime import task_directory
    out = (ROOT / p['shared_parameters']['output_dir']).resolve()
    rows = [dict(phase=phase, candidate_id=t.candidate_id, method=t.method,
                 config=asdict(t.config), profile=PROFILE if t.config.malicious_ratio else None,
                 task_dir=str(task_directory(out, phase, t.config).relative_to(out)))
            for phase, group in tasks.items() for t in group]
    payload = dict(protocol=p, source_files=source_identity(spec.source), task_specs=rows,
                   official_test_used_for_selection=False, fixed_parameters_before_validation=True)
    return dict(payload, fingerprint=digest(payload))


@contextmanager
def station_dashboard(spec, mode='auto'):
    original = ui.Progress
    class C7Progress(original):
        def lines(self, width=120, height=24):
            if height < 5:
                return super().lines(width, height)
            lines = super().lines(width, height - 1)
            label = 'official-test 10000' if self.phase == 'final' else 'training-holdout 7500'
            lines.insert(4, ui.fit('C7 Adam=.001 boost=20 normal<=10 weights-only | fixed d2/V4/A4 | '+label, width))
            return lines[:height]
    ui.Progress = C7Progress
    try:
        with ui.dashboard(spec, mode=mode) as progress:
            yield progress
    finally:
        ui.Progress = original


def run(p, spec, args, tasks, mode):
    from scripts.experiments.cifar10.c7_station_runtime import c7_station_runtime, runtime_fingerprint_context
    from scripts.experiments.cifar10.c7_station_gpu_pool import gpu_pool
    from scripts.experiments.cifar10.c7_station_report import build_report
    out = (ROOT / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This C7 output is already controlled by another running process') from exc
        manifest = make_manifest(p, spec, tasks)
        identity_path = out / 'protocol_manifest.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != manifest:
                raise ValueError('Frozen C7 protocol/source identity differs; preserve this output and use a new version')
        elif any(x.name != 'controller.lock' for x in out.iterdir()):
            raise ValueError('Refusing nonempty output without its matching C7 manifest')
        else:
            write_json(identity_path, manifest)
        status_path = out / 'controller_status.json'
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        if previous and previous.get('fingerprint') != manifest['fingerprint']:
            raise ValueError('Controller status belongs to a different C7 protocol')
        phases = previous.get('phases', {})
        if not isinstance(phases, dict):
            raise ValueError('Controller phase index is malformed; preserve it for inspection')
        state = dict(status='running', started_utc=utc(), pid=os.getpid(), fingerprint=manifest['fingerprint'],
                     planned_tasks=360, planned_rounds=54000, phases=phases)
        write_json(status_path, state)
        try:
            import torch
            torch.set_num_threads(p['execution']['torch_cpu_threads'])
            data = load_image_dataset(args.dataset, args.data_dir, download=args.download,
                        train_limit=args.train_samples, test_limit=args.test_samples, seed=args.seed)
            split = stratified_training_three_way_split(data, seed=spec.split_seed,
                        train_fraction=.8, calibration_fraction=.15, attack_fraction=.05)
            validate_targeted_split(split, args)
            contract = split_metadata(split, spec.split_seed)
            write_json(out / 'data_contract.json', contract)
            fixed = dict(parameters=FIXED, profile=PROFILE, parameter_selection='none',
                         frozen_before_validation=True, official_test_used_for_selection=False)
            write_json(out / 'fixed_parameters.json', fixed)
            with station_dashboard(spec, mode) as progress, \
                    gpu_pool(on_plan=progress.gpu_plan, on_event=progress.scheduler_event), \
                    c7_station_runtime(out, PROFILE):
                for phase, dataset in (('validation', split.calibration_dataset), ('final', split.main_dataset)):
                    state.update(phase=phase, updated_utc=utc())
                    write_json(status_path, state)
                    prepared, jobs, backend, workers = fair_tuning.prepare_tuning_tasks(
                        dataset, tasks[phase], args, requested_jobs='auto')
                    stage_dir = out / ('validation' if phase == 'validation' else 'final_evaluation')
                    stage_dir.mkdir(exist_ok=True)
                    context = dict(runtime_fingerprint_context(PROFILE),
                                   protocol_fingerprint=manifest['fingerprint'], data_contract=contract,
                                   evaluation_scope='official_test' if phase == 'final' else 'training_validation',
                                   parameter_selection='fixed_before_validation')
                    def snapshot(executions, fingerprint, status):
                        results = [r for _, r in executions]
                        if results:
                            experiments.write_result_files(stage_dir, results)
                        state['phases'][phase] = dict(status=status, completed_tasks=len(results),
                            planned_tasks=180, completed_rounds=sum(r.stopped_round for r in results),
                            fingerprint=fingerprint, jobs=jobs, sm9_workers=workers)
                        state.update(updated_utc=utc())
                        write_json(status_path, state)
                        write_json(out / 'report.json', build_report(out))
                    executions, _ = fair_tuning.execute_resumable_tuning_phase(
                        dataset, prepared, args, output_dir=out, jobs=jobs,
                        backend_description=backend, progress_enabled=True, progress_mode='auto',
                        fingerprint_context=context, on_snapshot=snapshot)
                    if (len(executions) != len(prepared)
                            or any(result.stopped_round != task.config.rounds or result.nonfinite_updates
                                   for task, result in executions)):
                        raise RuntimeError(f'{phase} lacks complete finite execution; preserve evidence before continuing')
            report = build_report(out)
            write_json(out / 'report.json', report)
            if (not report['all_complete'] or not report['all_numerically_finite']
                    or report['report_integrity_errors']):
                raise RuntimeError('C7 report integrity/completion failed; inspect report.json before resuming')
            state.update(status='complete', completed_tasks=360, completed_rounds=54000)
        except BaseException as error:
            state.update(status='interrupted_or_failed', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            state.update(ended_utc=utc())
            write_json(status_path, state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--check-gpus', action='store_true')
    parser.add_argument('--summary', action='store_true', help='Read-only report from existing evidence; no CUDA or training')
    parser.add_argument('--display-mode', choices=('auto','live','log'), default='auto')
    options = parser.parse_args(argv)
    try:
        p, spec, args, tasks = load_plan(options.config)
        out = (ROOT / args.output_dir).resolve()
        if options.summary:
            from scripts.experiments.cifar10.c7_station_report import build_report
            print(json.dumps(build_report(out), sort_keys=True, indent=2, allow_nan=False))
            return
        print('c7_station_plan=' + json.dumps(dict(protocol=PROTOCOL, output_dir=str(out),
            profile=PROFILE, fixed_parameters=FIXED, validation_seeds=list(spec.validation_seeds),
            final_seeds=list(spec.final_seeds), validation_tasks=180, final_tasks=180,
            total_tasks=360, total_rounds=54000, official_test_used_for_selection=False,
            parameter_selection='none; fixed local-validated configuration'), sort_keys=True), flush=True)
        if options.dry_run:
            return
        from scripts.experiments.cifar10.c7_station_gpu_pool import check_gpus
        check_gpus()
        if options.check_gpus:
            return
        run(p, spec, args, tasks, options.display_mode)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    os.chdir(ROOT)
    main()
