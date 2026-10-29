#!/usr/bin/env python3
"""One reviewed pre-evaluation VERT supplement OOM recovery; never an automatic retry loop.

Default is inspection only. --apply preserves the entire failed attempt, after
all existing workers drain. --continue starts the original scientific matrix
with the observed unstable GPU quarantined by this independent controller.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil

import run_cifar_vert_kmeans_supplement as formal
import run_cifar_h84_full_validation as full
from run_cifar_cp1_shared_gpu import file_sha

TASK = 'VERT_dirichlet_r20_s2026102801'
FAILURE_SHA = '12198451a193669bf80e72d4449377dc2e48c1e8c7cb58779d3076026256b2d1'
RECOVERY = 'reviewed_pre_eval_oom_v1'
OWN = ('recover_vert_kmeans_pre_eval_oom.py', 'tests/test_vert_kmeans_pre_eval_recovery.py')


def inventory(folder):
    return {str(p.relative_to(folder)): file_sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def classify(state, failure, rows, events, spec, identity):
    """A narrow resource-only gate; metrics values cannot influence this decision."""
    if (spec['task_id'] != TASK or spec['arm'] != 'VERT' or spec['config']['malicious_ratio'] != .2
            or state.get('status') != 'failed' or state.get('completed_round') != -1
            or state.get('error_type') != 'OutOfMemoryError'
            or failure.get('error_type') != 'OutOfMemoryError'
            or failure.get('last_completed_round') != -1
            or failure.get('task_id') != TASK or failure.get('audit') is not None
            or 'CUDA out of memory' not in failure.get('error','')
            or rows != []
            or 'initial_accuracy = _evaluate_accuracy' not in failure.get('traceback','')
            or state.get('fingerprint') != identity['fingerprint']):
        raise ValueError('Only the reviewed, pre-evaluation VERT CUDA OOM is eligible')
    starts = [e for e in events if e.get('event') == 'worker_started' and e.get('task') == TASK]
    exits = [e for e in events if e.get('event') == 'worker_exited' and e.get('task') == TASK]
    if (len(starts) != 1 or len(exits) != 1 or starts[0]['child_pid'] != state['pid']
            or exits[0]['child_pid'] != state['pid'] or exits[0].get('status') != 'failed'
            or starts[0]['admission']['free_mb'] < 8192
            or starts[0]['admission']['uuid'] != exits[0]['gpu_uuid']):
        raise ValueError('Actual launch/exit and GPU admission evidence required')
    return starts[0]['admission']


def inspect(config):
    from scripts.experiments.cifar10.attack_stability_worker import task_identity
    p, specs, manifest = formal.plan(config); out = formal.ROOT / p['output_dir']
    if formal.read(out/'manifest.json') != manifest: raise ValueError('Original formal manifest differs')
    full.verify_files(formal.read(out/'reference_lock.json')['references'])
    recovery = out/'recovery'/RECOVERY
    if recovery.exists(): raise ValueError('Recovery already recorded; never apply twice')
    spec = next(s for s in specs if s['task_id'] == TASK)
    identity = task_identity(dict(spec, fingerprint=manifest['fingerprint'], data_config=manifest['data_config']))
    folder = out/'tasks'/TASK
    if formal.read(folder/'task.json') != identity: raise ValueError('Failed task identity differs')
    state = formal.read(folder/'status.json'); failure = formal.read(folder/'first_failure.json')
    allowed={'worker.lock','task.json','dataset_identity.json','targets.json','status.json','first_failure.json'}
    if {f.name for f in folder.iterdir()} != allowed:
        raise ValueError('Expected pre-evaluation files only; no partial training or missing evidence')
    if file_sha(folder/'first_failure.json') != FAILURE_SHA:
        raise ValueError('Only the specifically reviewed failure is eligible')
    controller = formal.read(out/'controller_status.json')
    event_file = out/controller['audit_run']/'events.jsonl'
    events = [json.loads(line) for line in event_file.read_text().splitlines()]
    device = classify(state, failure, [], events, spec, identity)
    return p, specs, manifest, out, dict(task=TASK, original_manifest_fingerprint=manifest['fingerprint'],
        failed_attempt_sha256=inventory(folder), first_failure_sha256=file_sha(folder/'first_failure.json'),
        scheduler_events=str(event_file.relative_to(out)), scheduler_events_sha256=file_sha(event_file),
        original_worker_log=str((out/'logs'/(TASK+'.log')).relative_to(out)),
        original_worker_log_sha256=file_sha(out/'logs'/(TASK+'.log')),
        failed_device=device, quarantine_gpu_uuid=device['uuid'],
        reason='Reviewed CUDA memory exhaustion during initial evaluation before any completed evaluation/training; preserve every failed artifact; '
               'fresh same-seed/same-configuration attempt, never choose between performance results.',
        allowed_additional_attempts=1, automatic_retry_loop=False,
        live_workers=len(controller.get('active',{})), controller_status=controller['status'])


def apply(config, expected_failure_sha):
    p,specs,manifest,out,record = inspect(config)
    if record['first_failure_sha256'] != expected_failure_sha or expected_failure_sha != FAILURE_SHA:
        raise ValueError('Explicitly reviewed first-failure digest differs')
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        current = formal.read(out/'controller_status.json')
        if current.get('active') or current['status'] != 'blocked':
            raise ValueError('Original controller must finish draining before recovery')
        if any(formal.read(out/'tasks'/s['task_id']/'status.json',{}).get('status') in ('running','partial')
               for s in specs): raise ValueError('Running/partial evidence cannot be retried or displaced')
        if any(s['task_id']!=TASK and formal.read(out/'tasks'/s['task_id']/'status.json',{}).get('status') == 'failed' for s in specs):
            raise ValueError('Any additional failure requires a separate review')
        folder = out/'tasks'/TASK
        if inventory(folder) != record['failed_attempt_sha256']: raise ValueError('Failure files changed')
        dest = out/'recovery'/RECOVERY;dest.mkdir(parents=True)
        record.update(reviewed_utc=formal.utc(), original_directory=str(folder.relative_to(out)),
            preserved_directory=str((dest/'failed_attempt').relative_to(out)),
            recovery_source_sha256={n:formal.sha(formal.ROOT/n) for n in OWN},
            reference_lock_sha256=file_sha(out/'reference_lock.json'),
            operator='Assistant reviewed the actual error and admission evidence under the user-authorized formal run; no outcome-based selection.')
        for path_key,hash_key in (('original_worker_log','original_worker_log_sha256'),
                                   ('scheduler_events','scheduler_events_sha256')):
            source=out/record[path_key]
            if file_sha(source)!=record[hash_key]:raise ValueError('Reviewed log changed')
            shutil.copyfile(source,dest/(path_key+'.log'))
        formal.write(dest/'review.json',record)
        folder.rename(dest/'failed_attempt')
        if inventory(dest/'failed_attempt') != record['failed_attempt_sha256']:
            raise ValueError('Archived evidence changed')
        formal.write(dest/'applied.json',dict(applied_utc=formal.utc(),task=TASK,manifest_unchanged=True))
    return record


def verify_recovery(out, manifest):
    dest=out/'recovery'/RECOVERY;record=formal.read(dest/'review.json')
    if not formal.read(dest/'applied.json'):raise ValueError('Reviewed recovery not applied')
    if (record['original_manifest_fingerprint'] != manifest['fingerprint']
            or inventory(dest/'failed_attempt') != record['failed_attempt_sha256']
            or file_sha(out/'reference_lock.json') != record['reference_lock_sha256']):
        raise ValueError('Recovery lineage or preserved failure changed')
    for name,expected in record['recovery_source_sha256'].items():
        if formal.sha(formal.ROOT/name) != expected:raise ValueError('Recovery controller changed')
    for key in ('original_worker_log','scheduler_events'):
        if file_sha(dest/(key+'.log'))!=record[key+'_sha256']:raise ValueError('Preserved recovery log changed')
    return record


def final_audit(out,manifest):
    record=verify_recovery(out,manifest);dest=out/'recovery'/RECOVERY
    old=dest/'failed_attempt';new=out/'tasks'/TASK
    for name in ('task.json','dataset_identity.json','targets.json'):
        if formal.read(old/name)!=formal.read(new/name):
            raise ValueError('Recovered identity/data/target differs')
    report=formal.read(out/'analysis/combined_results.json')
    if not report['replacement_lineage'][TASK]['same_seed_initial_model_data_targets_malicious_ids']:
        raise ValueError('Original formal-run initial model match required')
    formal.write(out/'analysis/recovery_final_audit.json',dict(
        manifest_fingerprint=manifest['fingerprint'],original_failed_attempt_retained=True,
        recovery_review_sha256=file_sha(dest/'review.json'),
        preserved_failure_sha256=record['failed_attempt_sha256'],
        combined_report_sha256=file_sha(out/'analysis/combined_results.json'),
        scientific_supplement_cells=30,supplement_execution_attempts=31,extra_completed_training_rounds=0,
        same_task_identity_data_targets=True,initial_model_matches_original_formal_run=True,
        failed_attempt_initial_model_observed=False,
        failed_attempt_elapsed_seconds=formal.read(old/'status.json')['elapsed_seconds'],
        timing_note='Pre-evaluation OOM had no saved initial model or completed metric; its cost remains separate.'))


def continue_run(config):
    p,specs,manifest=formal.plan(config);out=formal.ROOT/p['output_dir']
    record=verify_recovery(out,manifest)
    dest=out/'recovery'/RECOVERY
    if (dest/'continuation.json').exists():raise ValueError('One continuation only; inspect any later failure')
    # Original scheduler still refuses failed/partial tasks. No repeated repair.
    raw_snapshot=formal.gpu_snapshot
    formal.gpu_snapshot=lambda:[g for g in raw_snapshot() if g['uuid'] != record['quarantine_gpu_uuid']]
    formal.write(dest/'continuation.json',dict(pid=os.getpid(),started_utc=formal.utc(),
        quarantine_gpu_uuid=record['quarantine_gpu_uuid'],max_gpus=8,
        actual_maximum_visible_gpus=6,all_other_gpu_uuids_dynamic=True,science_and_admission_threshold_unchanged=True))
    try:
        from cifar_vert_kmeans_scheduler import run
        run(config,p,specs,manifest,out,8)
    finally:formal.gpu_snapshot=raw_snapshot
    verify_recovery(out,manifest)
    if formal.read(out/'controller_status.json')['status']=='panel_terminal':
        from cifar_vert_kmeans_report import finish
        finish(p,specs,manifest,out)
        final_audit(out,manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=formal.DEFAULT)
    g=parser.add_mutually_exclusive_group();g.add_argument('--apply',action='store_true');g.add_argument('--continue',dest='continue_run',action='store_true')
    parser.add_argument('--failure-sha256')
    a=parser.parse_args()
    if a.apply:
        if not a.failure_sha256:parser.error('--apply needs the explicitly reviewed first-failure SHA256')
        print(json.dumps(apply(a.config,a.failure_sha256),indent=2))
    elif a.continue_run:continue_run(a.config)
    else:print(json.dumps(inspect(a.config)[-1],indent=2))


if __name__=='__main__':main()
