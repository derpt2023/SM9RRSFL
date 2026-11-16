"""Frozen 936-run contract and read-only data pairing for the full study."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product
from pathlib import Path

from hfcdr_pilot_protocol import ROOT, COMMON, METHOD_SOURCE, read, write, sha, digest, baseline, verify_frozen_sources

CONFIG = ROOT / 'configs/hfcdr_attack_study_v1.json'
OUTPUT = ROOT / 'outputs/hfcdr_attack_study_v1'
BASELINES = {'cifar10': 'configs/reproduction/cifar-best.json',
             'mnist': 'configs/reproduction/mnist_v7_target_fair_tuning.json'}
OWN = ('configs/hfcdr_attack_study_v1.json', 'hfcdr_study_protocol.py',
       'hfcdr_study_runtime.py', 'hfcdr_study_scheduler.py', 'run_hfcdr_attack_study.py',
       'hfcdr_study_report.py', 'tests/test_hfcdr_attack_study.py', 'HFCDR_STUDY.txt',
       'smoke_hfcdr_study_cuda.py',
       'hfcdr_pilot_protocol.py', 'hfcdr_pilot_data.py', 'hfcdr_pilot_runtime.py',
       'hfcdr_pilot_report.py', 'configs/hfcdr_attack_pilot_v1.json')


def policy():
    return read(CONFIG)


def task_specs():
    p = policy()
    base = {d: read(ROOT / path) for d, path in BASELINES.items()}
    conditions = [('clean', 'none', 0., 1.)]
    conditions += [(a['name'], a['attack'], ratio, a['scale'])
                   for ratio in p['ratios'][1:] for a in p['attacks']]
    tasks, groups = [], []
    # Interleave the datasets and partitions so coverage appears early.
    for index, (seed, condition, part, dataset) in enumerate(product(
            p['seeds'], conditions, p['partitions'], p['datasets'])):
        name, attack, ratio, scale = condition
        gid = f'{dataset}_{part}_{name}_r{round(ratio*100):02d}_s{seed}'
        order = p['methods'][index % 6:] + p['methods'][:index % 6]
        members = []
        for method in order:
            old = base[dataset]['methods'][METHOD_SOURCE[method]]['config']
            c = {k: old[k] for k in COMMON}
            prefix = {'hfcdr': ('detector_', 'suspicion_'), 'vert': ('vert_',),
                      'alignins': ('alignins_',)}.get(method, ())
            c.update({k: v for k, v in old.items() if prefix and k.startswith(prefix)})
            c.update(dataset=dataset, method=method, partition=part, seed=seed,
                     attack=attack, scale=scale, malicious_ratio=ratio)
            tid = f'{method}_{gid}'
            tasks.append(dict(task_id=tid, group_id=gid, condition=name, config=c))
            members.append(tid)
        groups.append(dict(group_id=gid, tasks=members))
    return tasks, groups


def audit(tasks, groups):
    expected = task_specs()
    if (tasks, groups) != expected or len(tasks) != 936 or len(groups) != 156:
        raise ValueError('Full fixed 936-task/156-group matrix required')
    if len({t['task_id'] for t in tasks}) != 936:
        raise ValueError('Duplicate tasks')
    for t in tasks:
        c = t['config']
        expected_train = (150, 50, 25) if c['dataset'] == 'cifar10' else (50, 32, 12)
        if (c['rounds'], c['batch_size'], c['attack_start_round']) != expected_train:
            raise ValueError('Training contract changed')
        if any(k.startswith(('sm9', 'dkg', 'crypto')) for k in c):
            raise ValueError('Crypto setting in experiment')
    return dict(passed=True, tasks=936, groups=156, seeds=policy()['seeds'],
        by_dataset=dict(Counter(t['config']['dataset'] for t in tasks)),
        by_method=dict(Counter(t['config']['method'] for t in tasks)),
        by_partition=dict(Counter(t['config']['partition'] for t in tasks)),
        by_condition=dict(Counter(t['condition'] for t in tasks)),
        by_ratio=dict(Counter(str(t['config']['malicious_ratio']) for t in tasks)),
        cryptography='absent', official_test_used_for_selection=False, automatic_retry=False)


def plan():
    b = baseline()
    tasks, groups = task_specs()
    payload = dict(schema=policy()['schema'], policy=policy(), tasks=tasks, groups=groups,
        audit=audit(tasks, groups), numerics=b['numerics'],
        original_environment={k: v for k, v in b['original_environment'].items() if k != 'native_binary_sha256'},
        criteria=read(ROOT / 'configs/hfcdr_attack_pilot_v1.json')['criteria'],
        frozen_sources=verify_frozen_sources(),
        new_sources={name: sha(ROOT / name) for name in (*OWN, *BASELINES.values())})
    return dict(payload, fingerprint=digest(payload))


def safe_output(path):
    raw = Path(path).expanduser()
    out = raw.resolve()
    if raw.is_symlink() or out.parent != (ROOT/'outputs').resolve() or not out.name.startswith('hfcdr_attack_study'):
        raise ValueError('Use a fresh outputs/hfcdr_attack_study* sibling, never a legacy output')
    return out


def load_dataset(dataset, data_root):
    """Verify original split; expose official test only to the evaluator."""
    from hfcdr_pilot_data import array_digest
    from sm9rrsfl.datasets import load_image_dataset, stratified_training_three_way_split
    b = read(ROOT / BASELINES[dataset]); d = b['data_config']
    raw = load_image_dataset(dataset, Path(data_root)/dataset, download=False,
        train_limit=d['train_samples'], test_limit=d['test_samples'], seed=d['seed'])
    frac = .8 if dataset == 'cifar10' else .9
    val = .15 if dataset == 'cifar10' else .05
    split = stratified_training_three_way_split(raw, seed=d['split_seed'],
        train_fraction=frac, calibration_fraction=val, attack_fraction=.05)
    contract = dict(source='original training split only', official_test_used_for_selection=False,
        shared_training_arrays_across_phases=True, split_seed=d['split_seed'],
        train_samples=len(split.train_indices), calibration_samples=len(split.calibration_indices),
        attack_auxiliary_samples=len(split.attack_indices), train_fraction_actual=frac,
        train_indices_digest=array_digest(split.train_indices),
        calibration_indices_digest=array_digest(split.calibration_indices),
        attack_indices_digest=array_digest(split.attack_indices))
    ds = split.main_dataset
    identity = dict(contract=contract, arrays={n: array_digest(getattr(ds, n)) for n in
        ('x_train', 'y_train', 'x_test', 'y_test', 'x_attack', 'y_attack')})
    expected_contract = (b['official_dataset_identity']['contract'] if dataset == 'cifar10'
                         else b['training_data_contract'])
    if contract != expected_contract:
        raise ValueError('Training split differs from frozen baseline: ' + dataset)
    if dataset == 'cifar10' and identity != b['official_dataset_identity']:
        raise ValueError('CIFAR official arrays differ from accepted experiment')
    identity.update(dataset=dataset, evaluation_scope=policy()['evaluation'],
                    accuracy_samples=len(ds.y_test), source_samples=int((ds.y_test == 5).sum()))
    return replace(ds, x_attack=None, y_attack=None), identity


def pairing_key(task):
    c = task['config']
    return f"{c['dataset']}_{c['partition']}_r{round(c['malicious_ratio']*100):02d}_s{c['seed']}"


def prepare(output, data_root):
    from hfcdr_pilot_data import participant_record
    out = safe_output(output)
    if out.exists():
        raise FileExistsError('Fresh study output required; existing evidence retained')
    manifest = plan()
    identities, pairing = {}, {}
    for dataset in policy()['datasets']:
        ds, identities[dataset] = load_dataset(dataset, data_root)
        representatives = {}
        for t in manifest['tasks']:
            if t['config']['dataset'] == dataset:
                representatives.setdefault(pairing_key(t), t)
        for key, task in representatives.items():
            pairing[key] = participant_record(ds.y_train, task['config'])
        del ds
    if len(pairing) != 60:
        raise ValueError('Exactly 60 data/seed/partition/ratio pairings required')
    out.mkdir(parents=True)
    write(out/'manifest.json', manifest)
    write(out/'dataset_identity.json', identities)
    write(out/'pairing.json', pairing)
    write(out/'configuration_audit.json', dict(manifest['audit'], data_verified=True,
          pairings=len(pairing), training_started=False, pending=936))
    return dict(manifest['audit'], fingerprint=manifest['fingerprint'], data_verified=True)


def check(output):
    out = safe_output(output)
    manifest = read(out/'manifest.json')
    if manifest != plan():
        raise ValueError('Frozen source/configuration manifest changed')
    counts = Counter()
    for t in manifest['tasks']:
        folder = out/'tasks'/t['task_id']; status = folder/'status.json'
        counts[read(status)['status'] if status.exists() else 'incomplete' if folder.exists() else 'pending'] += 1
    return dict(fingerprint=manifest['fingerprint'], counts=dict(counts), tasks=936)
