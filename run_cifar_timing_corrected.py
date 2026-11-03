#!/usr/bin/env python3
"""User-authorized fresh 180-task timing study with verified scoped algorithms.

The original 193 sources and all historical results remain unchanged. This
adapter reuses the MNIST clock and the original fresh-only block scheduler.
"""
from contextlib import contextmanager
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import run_cifar_timing as t

ROOT = t.ROOT
OUTPUT = ROOT / 'outputs/old/cifar_mnist_timing_corrected150_v1'
HISTORY = 'configs/cifar_timing_corrected_history_v1.json'
OWN = ('run_cifar_timing_corrected.py', 'tests/test_cifar_timing_corrected.py', HISTORY)
PRIOR = {
    'resume_cifar_timing.py': '3f258ed85351d767845baff3f8b44bd1f506e86e865cd515250c9d9bef28f219',
    'tests/test_cifar_timing_followup.py': 'f981db2d9b152c88b6dddc259d274fdd183e57c893e20f47f0d314c04278605c'}
GPU_MAP = {
    '2': 'GPU-f3b82a8d-b1fb-1320-896e-50e1d846ffa7',
    '4': 'GPU-f5517558-d812-fe0c-8a93-654da0b7af75',
    '5': 'GPU-f6948b25-3ecf-1ca1-33b1-bd2618fbccb7',
    '7': 'GPU-6c2e2803-c8f8-7f4d-daa0-625be5dd185e'}
ACCEPTED = {
    'outputs/cifar-best/export/combined_bundle.json': '7a56eec00bd2ef936f342e1f2a22088b674cff11d86bcff1981176c5bb6a6198',
    'outputs/cifar-best/provenance/file_manifest.json': '272c599da8104f86b26a7b8d36572fe5167afc709c127e614841562d7290a449',
    'outputs/cifar-best/provenance/accepted_cifar_paper_bundle.json': '2fd0d9cdad96b9cfacb6ed69da0b482dce4bfab537c118e03b9111810714f1a1'}
_PARENT_MANIFEST, _PARENT_BATCH = t.manifest, t.batch
_PARENT_WORKER, _PARENT_REPORT = t.worker, t.final_report
_PARENT_ADMITTED = t.admitted


def manifest(data_dir):
    value = _PARENT_MANIFEST(data_dir)
    value.pop('fingerprint')
    for name, digest in PRIOR.items():
        if t.base.sha(ROOT / name) != digest:
            raise ValueError('Frozen continuation source changed: ' + name)
    value['sources'].update(PRIOR)
    value['sources'].update({name: t.base.sha(ROOT / name) for name in OWN})
    value.update(kind='cifar_mnist_timing_corrected150_v1',
        authorization='2026-11-03 user: 修正后完整重测180项。',
        correction='Bind the common measured runner to scoped fl.run_experiment; inspect actual checkpoint detectors.',
        fresh_tasks=180, reused_results=0, historical_attempts=96,
        accepted_files=ACCEPTED)
    value['scheduling'].update(max_gpus=4, allowed_index_uuid=GPU_MAP)
    value['fingerprint'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def admitted(uuid, first, second):
    if uuid not in GPU_MAP.values() or not _PARENT_ADMITTED(uuid, first, second):
        return False
    return all(any(g['uuid'] == uuid and GPU_MAP.get(str(g['index'])) == uuid
                   for g in snap['gpus']) for snap in (first, second))


def checkpoint_observation(config, state):
    from cifar_tad_no_prior import PriorFreeTAD
    from cifar_vert_paper_core import PaperVERT
    row = dict(round=int(state['completed_round']))
    for key in ('detector', 'weight_manager', 'ding13_detector', 'vert_defense'):
        obj = state.get(key)
        row[key] = None if obj is None else type(obj).__module__ + '.' + type(obj).__qualname__
    if config.method == 'ding13':
        detector = state.get('ding13_detector')
        if not isinstance(detector, PriorFreeTAD):
            raise ValueError('Measured TAD must execute accepted PriorFreeTAD')
        if row['round'] and detector.last_round != row['round']:
            raise ValueError('Actual TAD evaluation did not reach checkpoint round')
        row['tad_audit'] = dict(detector.last_audit)
    elif state.get('ding13_detector') is not None:
        raise ValueError('Unexpected TAD detector in another method')
    if config.method == 'vert' and not isinstance(state.get('vert_defense'), PaperVERT):
        raise ValueError('Measured VERT must execute accepted PaperVERT')
    return row


def validate_binding(task, binding):
    method = task['config']['method']
    expected = '_run_experiment_prior_free_tad' if method == 'ding13' else 'run_experiment'
    if binding.get('function') != expected or binding.get('method') != method:
        raise ValueError('Wrong measured training function')
    observations = binding.get('checkpoint_observations', [])
    if not binding.get('completed') or not observations or observations[0]['round'] != 0:
        raise ValueError('Incomplete actual algorithm evidence')
    if method in ('ding13', 'vert'):
        if [r['round'] for r in observations] != list(range(task['config']['rounds'] + 1)):
            raise ValueError('Missing actual per-round detector evidence')
        key, value = ('ding13_detector', 'cifar_tad_no_prior.PriorFreeTAD') if method == 'ding13' else ('vert_defense', 'cifar_vert_paper_core.PaperVERT')
        if any(r[key] != value for r in observations):
            raise ValueError('Wrong actual detector class')
    if binding.get('target_view_module') != 'cifar_alternating_target_views':
        raise ValueError('Wrong target attack view implementation')


def measured(dataset, task, folder, fingerprint):
    from sm9rrsfl import experiments, fl
    from scripts.experiments.cifar10 import attack_stability_attack
    saved, saved_write = experiments.run_experiment, experiments._write_round_checkpoint
    binding = dict(method=task['config']['method'], completed=False,
                   checkpoint_observations=[], timing_function=t.DEFINITION['function'])
    def checked_checkpoint(path, config, identity, state, **kwargs):
        # Inspection is inside the public runner's already excluded I/O span.
        row = checkpoint_observation(config, state)
        binding['checkpoint_observations'].append(row)
        return saved_write(path, config, identity, state, **kwargs)
    try:
        with t.scientific_scopes(task):
            experiments.run_experiment = fl.run_experiment
            binding['function'] = experiments.run_experiment.__name__
            binding['target_view_module'] = attack_stability_attack.run_profile_attack.__module__
            if binding['target_view_module'] != 'cifar_alternating_target_views':
                raise ValueError('Scoped target views are incorrect')
            expected = '_run_experiment_prior_free_tad' if task['config']['method'] == 'ding13' else 'run_experiment'
            if binding['function'] != expected:
                raise ValueError('Scoped algorithm binding is incorrect')
            experiments._write_round_checkpoint = checked_checkpoint
            with t.checkpoint_progress(folder, experiments):
                result = experiments.run_measured_experiment(dataset, fl.ExperimentConfig(**task['config']),
                    checkpoint_dir=folder / 'checkpoints', run_fingerprint=fingerprint,
                    task_id=task['task_id'], emit_progress_events=False)
        binding['completed'] = True
        validate_binding(task, binding)
        return result
    finally:
        experiments.run_experiment, experiments._write_round_checkpoint = saved, saved_write
        t.base.write(folder / 'training_binding.json', binding)


def worker(out, task_id):
    # Parent handles all training failures and retains evidence; final extra hash
    # is written before the worker exits, so dispatch cannot accept an early status.
    _PARENT_WORKER(out, task_id)
    folder = out / 'tasks' / task_id
    try:
        validate_binding(t.base.read(folder / 'task.json'), t.base.read(folder / 'training_binding.json'))
        status = t.base.read(folder / 'status.json')
        status['artifacts']['training_binding.json'] = t.base.sha(folder / 'training_binding.json')
        t.base.write(folder / 'status.json', status)
    except BaseException as exc:
        t.base.write(folder / 'status.json', dict(status='failed', error=repr(exc), pid=os.getpid(), at=t.now()))
        raise


def history_snapshot():
    """Validate historical results read-only; no results are reused by this run."""
    contract = t.base.read(ROOT / HISTORY)
    followup = ROOT / contract['source_output_relative']
    if t.base.sha(followup / 'analysis/timing_results.json') != contract['report_sha256']:
        raise ValueError('Historical incomplete report changed')
    files = dict(ACCEPTED)
    for name, expected in ACCEPTED.items():
        if t.base.sha(ROOT / name) != expected:
            raise ValueError('Accepted result changed: ' + name)
    for relative, expected_counts in (
        ('outputs/old/cifar_mnist_timing150_v1', {'complete':4, 'failed':1, 'pending':175}),
        (contract['source_output_relative'], {'complete':95, 'pending':85})):
        out = ROOT / relative
        m, rows, counts = t.collect(out)
        if m != _PARENT_MANIFEST(m['data_dir']) or counts != expected_counts:
            raise ValueError('Historical matrix/source mismatch')
        state = t.base.read(out / 'controller_status.json')
        if state['status'] != 'incomplete' or state.get('active'):
            raise ValueError('Historical controller has not drained')
        # Hash all completed small artifacts and raw result pickles; checkpoints
        # are not copied or described as backed up.
        for path in out.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.log') and 'checkpoints' not in path.parts:
                files[str(path.relative_to(ROOT))] = t.base.sha(path)
        for row in rows:
            folder = out / 'tasks' / row['task_id']
            s = t.base.read(folder / 'status.json')
            for name, digest in s['artifacts'].items():
                files[str((folder / name).relative_to(ROOT))] = digest
    for item in contract['invalid_tasks']:
        status = t.base.read(followup / 'tasks' / item['task_id'] / 'status.json')
        if status['artifacts'] != item['artifacts']:
            raise ValueError('Invalid TAD evidence changed')
    if len(contract['invalid_tasks']) != 17:
        raise ValueError('Expected 17 invalid legacy TAD results')
    return dict(at=t.now(), files=files, contract=contract, prior_actual_attempts=96,
                reused_by_corrected_study=0)


def verify_history_snapshot(snapshot):
    for name, digest in snapshot['files'].items():
        if t.base.sha(ROOT / name) != digest:
            raise ValueError('Historical/accepted artifact changed: ' + name)


def running_study():
    entries = ('run_cifar_timing.py', 'resume_cifar_timing.py', 'run_cifar_timing_corrected.py')
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = (p / 'cmdline').read_bytes().decode().split('\0')
            if any(Path(a).name in entries for a in args if a):
                found.append(int(p.name))
        except (OSError, UnicodeError):
            pass
    return found


def batch(out, data_dir, gpu_indices, max_gpus):
    if not 1 <= max_gpus <= 4 or gpu_indices is not None and (not gpu_indices or gpu_indices - GPU_MAP.keys()):
        raise ValueError('Only physical GPU2/4/5/7, at most four tasks')
    if out.exists():
        raise FileExistsError('Fresh independent output required')
    if running_study():
        raise ValueError('An earlier or duplicate timing study is still running')
    snapshot = history_snapshot()
    original_event = t.event
    def event(out, kind, **fields):
        if kind == 'controller_start':
            t.base.write(out / 'provenance/history_snapshot.json', snapshot)
            shutil.copy2(ROOT / HISTORY, out / 'provenance/invalid_tad_ledger.json')
            source = ROOT / snapshot['contract']['source_output_relative']
            shutil.copy2(source / 'analysis/timing_results.json', out / 'provenance/previous_incomplete_report.json')
        return original_event(out, kind, **fields)
    t.event = event
    try:
        return _PARENT_BATCH(out, data_dir, set(GPU_MAP) if gpu_indices is None else gpu_indices, max_gpus)
    finally:
        t.event = original_event


def final_report(out):
    report = _PARENT_REPORT(out)
    failures = report['execution_audit']['failures']
    snapshot = t.base.read(out / 'provenance/history_snapshot.json')
    verify_history_snapshot(snapshot)
    bindings = []
    for task in (task for b in t.base.read(out / 'manifest.json')['blocks'] for task in b['tasks']):
        folder = out / 'tasks' / task['task_id']
        if (folder / 'status.json').exists() and t.base.read(folder / 'status.json')['status'] == 'complete':
            try:
                binding = t.base.read(folder / 'training_binding.json')
                validate_binding(task, binding)
                bindings.append(dict(task_id=task['task_id'], method=binding['method'],
                                     function=binding['function'], observations=len(binding['checkpoint_observations'])))
            except (OSError, KeyError, ValueError) as exc:
                failures.append('Algorithm binding: ' + task['task_id'] + ': ' + str(exc))
    if len(bindings) != 180:
        failures.append('Actual algorithm evidence incomplete: ' + str(len(bindings)))
    if report['execution_audit']['max_active'] > 4:
        failures.append('More than four simultaneous study tasks')
    report['algorithm_binding_audit'] = dict(passed=len(bindings) == 180 and not failures, tasks=bindings)
    report['history'] = dict(prior_actual_attempts=96, invalid_tad_tasks=snapshot['contract']['invalid_tasks'],
        previous_non_tad_complete=78, previous_pending=85, original_oom='TAD_iid_r80_s2026102801',
        corrected_fresh_attempts=report['execution_audit']['starts'], reused_results=0,
        cumulative_actual_attempts=96 + report['execution_audit']['starts'], automatic_retries=0,
        historical_artifacts_verified=True)
    report['execution_audit']['passed'] = not failures
    t.base.write(out / 'analysis/timing_results.json', report)
    return report


@contextmanager
def configured():
    replacements = dict(manifest=manifest, measured=measured, worker=worker, batch=batch,
                        final_report=final_report, admitted=admitted, __file__=str(Path(__file__).resolve()),
                        DEFAULT_OUTPUT=OUTPUT)
    saved = {key:getattr(t, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(t, key, value)
        yield
    finally:
        for key, value in saved.items():
            setattr(t, key, value)


def main():
    # The original CLI defaults to six. This adapter's default and hard limit
    # are four; explicit unauthorized GPU choices are rejected by batch().
    if '--max-gpus' not in sys.argv and not any(a.startswith('--max-gpus=') for a in sys.argv):
        sys.argv.extend(['--max-gpus', '4'])
    with configured():
        t.main()


if __name__ == '__main__':
    main()
