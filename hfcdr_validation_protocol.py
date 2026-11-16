"""Independent validation-only contract. Legacy studies remain read-only."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product
from pathlib import Path
import re

from hfcdr_pilot_protocol import (ROOT, COMMON, METHOD_SOURCE, read, write, sha,
                                 digest, baseline, verify_frozen_sources)

CONFIG = ROOT / 'configs/hfcdr_validation_attack_v1.json'
OUTPUT = ROOT / 'outputs/hfcdr_validation_attack_v1'
BASELINES = {'cifar10': 'configs/reproduction/cifar-best.json',
             'mnist': 'configs/reproduction/mnist_v7_target_fair_tuning.json'}
OWN = ('hfcdr_validation_protocol.py', 'hfcdr_validation_runtime.py',
       'hfcdr_validation_scheduler.py', 'hfcdr_validation_analysis.py',
       'run_hfcdr_validation.py', 'tests/test_hfcdr_validation.py',
       'HFCDR_VALIDATION.txt', 'hfcdr_pilot_protocol.py', 'hfcdr_pilot_data.py',
       'hfcdr_pilot_runtime.py', 'configs/hfcdr_attack_pilot_v1.json')
PREFIXES = {'hfcdr': ('detector_', 'suspicion_'), 'vert': ('vert_',),
            'alignins': ('alignins_',)}


def specification(path=CONFIG):
    path = Path(path).resolve()
    if path.parent != ROOT/'configs' or not path.name.startswith('hfcdr_validation_'):
        raise ValueError('Use a versioned validation specification in configs')
    p = read(path)
    if (p['schema'] != 'hfcdr_validation_v1' or p['official_test_used'] is not False
            or p['evaluation'] != 'training_validation_all_source5'
            or p['attack_auxiliary_used'] is not False or p['cryptography'] != 'absent'
            or p['automatic_retry'] is not False):
        raise ValueError('Validation-only, no-crypto, one-attempt contract required')
    if p['datasets'] != ['cifar10', 'mnist'] or p['partitions'] != ['iid', 'dirichlet']:
        raise ValueError('Both datasets and partitions required')
    seed_sets = [set(p[k]) for k in ('seeds', 'confirmation_seeds', 'formal_seeds')]
    if any(len(p[k]) != len(set(p[k])) for k in ('seeds','confirmation_seeds','formal_seeds')):
        raise ValueError('Duplicate seeds')
    if any(seed_sets[i] & seed_sets[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Selection, confirmation and formal seeds must be disjoint')
    if any(seed_sets[i] & {2026111601,2026111602,2026111603} for i in range(3)):
        raise ValueError('Old exposed seeds cannot be selection/confirmation/formal seeds')
    if p['candidate_budget_upper_bound'] != 4:
        raise ValueError('Predeclared equal maximum budget is four candidates')
    if not p['defense_candidates']:
        raise ValueError('No methods')
    for method, candidates in p['defense_candidates'].items():
        if method not in METHOD_SOURCE or not 1 <= len(candidates) <= 4:
            raise ValueError('Unknown method or candidate budget exceeded')
        if len({c['id'] for c in candidates}) != len(candidates):
            raise ValueError('Duplicate candidates')
        for c in candidates:
            if not re.fullmatch('[a-z0-9_]+', c['id']):
                raise ValueError('Invalid candidate id')
            for dataset, overrides in c['overrides'].items():
                if dataset not in BASELINES or any(not k.startswith(PREFIXES.get(method, ())) for k in overrides):
                    raise ValueError('Candidate may change only its own defense parameters')
        if method in ('fedavg', 'krum', 'tad') and (len(candidates) != 1 or candidates[0]['overrides']):
            raise ValueError('FedAvg/Krum/TAD fixed; no fictitious parameter search')
    if p['phase'] == 'attack_screen' and (p['defense_candidates'] != {'fedavg':[{'id':'fixed','overrides':{}}]}
            or p['ratios'] != [0.,.2,.4,.6,.8] or len(p['seeds']) != 2):
        raise ValueError('Attack screen must contain the complete 104-run FedAvg panel')
    if p['phase'] == 'attack_screen' and p['attacks'] != [
            dict(name='label_flip',attack='label_flip',scale=1.),
            dict(name='scale2',attack='scaled_label_flip',scale=2.),
            dict(name='scale5',attack='scaled_label_flip',scale=5.)]:
        raise ValueError('All three predeclared attack settings required')
    return p


def task_specs(p):
    bases = {d: read(ROOT/path) for d,path in BASELINES.items()}
    conditions = [('clean','none',0.,1.)] + [(a['name'],a['attack'],r,a['scale'])
        for r in p['ratios'][1:] for a in p['attacks']]
    candidates = [(m,c) for m,cs in p['defense_candidates'].items() for c in cs]
    tasks, groups = [], []
    for index,(seed,condition,part,dataset) in enumerate(product(p['seeds'],conditions,p['partitions'],p['datasets'])):
        name,attack,ratio,scale = condition
        if (attack,scale) not in (('none',1.),('label_flip',1.),('scaled_label_flip',2.),('scaled_label_flip',5.)):
            raise ValueError('Unregistered attack configuration')
        gid = f'{dataset}_{part}_{name}_r{round(ratio*100):02d}_s{seed}'
        offset = index % len(candidates)
        order = candidates[offset:]+candidates[:offset]
        members = []
        for method,candidate in order:
            old = bases[dataset]['methods'][METHOD_SOURCE[method]]['config']
            c = {k:old[k] for k in COMMON}
            c.update({k:v for k,v in old.items() if k.startswith(PREFIXES.get(method,()))})
            overrides = candidate['overrides'].get(dataset,{})
            if set(overrides)-set(c):
                raise ValueError('Unknown defense hyperparameter')
            c.update(overrides)
            c.update(dataset=dataset,method=method,partition=part,seed=seed,
                     attack=attack,scale=scale,malicious_ratio=ratio)
            tid = f"{method}_{candidate['id']}_{gid}"
            tasks.append(dict(task_id=tid,group_id=gid,condition=name,candidate_id=candidate['id'],config=c))
            members.append(tid)
        groups.append(dict(group_id=gid,tasks=members))
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicated task identity')
    for t in tasks:
        c = t['config']
        expected = (150,50,25) if c['dataset']=='cifar10' else (50,32,12)
        if (c['rounds'],c['batch_size'],c['attack_start_round']) != expected or (c['attack_source_label'],c['attack_target_label']) != (5,7):
            raise ValueError('Public training/attack contract changed')
    return tasks,groups


def plan(spec=CONFIG):
    spec = Path(spec).resolve(); p = specification(spec)
    tasks,groups = task_specs(p); b = baseline()
    sources = (*OWN,*BASELINES.values(),str(spec.relative_to(ROOT)))
    payload = dict(schema=p['schema'],specification=str(spec.relative_to(ROOT)),policy=p,
        tasks=tasks,groups=groups,numerics=b['numerics'],
        original_environment={k:v for k,v in b['original_environment'].items() if k!='native_binary_sha256'},
        frozen_sources=verify_frozen_sources(),new_sources={n:sha(ROOT/n) for n in sources},
        audit=dict(passed=True,tasks=len(tasks),groups=len(groups),
            by_method=dict(Counter(t['config']['method'] for t in tasks)),
            by_dataset=dict(Counter(t['config']['dataset'] for t in tasks)),
            by_condition=dict(Counter(t['condition'] for t in tasks)),
            official_test_used=False,cryptography='absent',automatic_retry=False))
    return dict(payload,fingerprint=digest(payload))


def safe_output(path):
    raw = Path(path).expanduser(); out = raw.resolve()
    if raw.is_symlink() or out.parent != (ROOT/'outputs').resolve() or not out.name.startswith('hfcdr_validation_'):
        raise ValueError('Use an independent outputs/hfcdr_validation_* directory')
    return out


def load_training_raw(dataset,data_root):
    """Never opens official test files, including during validation preparation."""
    import numpy as np
    from sm9rrsfl import datasets as d
    config = read(ROOT/BASELINES[dataset])['data_config']
    directory = Path(data_root)/dataset
    if dataset == 'mnist':
        x = d._read_idx_images(directory/d.MNIST_FILES['train_images'])
        y = d._read_idx_labels(directory/d.MNIST_FILES['train_labels'])
    elif dataset == 'cifar10':
        batches = [d._read_cifar10_batch(directory/d.CIFAR10_DIR/f'data_batch_{i}') for i in range(1,6)]
        x = d._normalize_cifar10(np.concatenate([b[0] for b in batches],axis=0))
        y = np.concatenate([b[1] for b in batches],axis=0)
    else:
        raise ValueError('Unknown dataset')
    x,y = d._limit_samples(x,y,config['train_samples'],config['seed'])
    return d.ImageDataset(x_train=x,y_train=y,x_test=x[:0],y_test=y[:0],name=dataset)


def load_dataset(dataset,data_root):
    from hfcdr_pilot_data import array_digest
    from sm9rrsfl.datasets import stratified_training_three_way_split
    b = read(ROOT/BASELINES[dataset]); d = b['data_config']
    raw = load_training_raw(dataset,data_root)
    frac,val = (.8,.15) if dataset=='cifar10' else (.9,.05)
    split = stratified_training_three_way_split(raw,seed=d['split_seed'],
        train_fraction=frac,calibration_fraction=val,attack_fraction=.05)
    contract = dict(source='original training split only',official_test_used_for_selection=False,
        shared_training_arrays_across_phases=True,split_seed=d['split_seed'],
        train_samples=len(split.train_indices),calibration_samples=len(split.calibration_indices),
        attack_auxiliary_samples=len(split.attack_indices),train_fraction_actual=frac,
        train_indices_digest=array_digest(split.train_indices),
        calibration_indices_digest=array_digest(split.calibration_indices),
        attack_indices_digest=array_digest(split.attack_indices))
    ds = split.calibration_dataset
    identity = dict(contract=contract,arrays={n:array_digest(getattr(ds,n)) for n in
        ('x_train','y_train','x_test','y_test','x_attack','y_attack')})
    expected = b['expected_dataset_identity']['contract'] if dataset=='cifar10' else b['training_data_contract']
    if contract != expected or (dataset=='cifar10' and identity != b['expected_dataset_identity']):
        raise ValueError('Validation split differs from the original training contract')
    identity.update(dataset=dataset,evaluation_scope='training_validation_all_source5',
        official_test_files_opened=False,attack_auxiliary_used=False,
        accuracy_samples=len(ds.y_test),source_samples=int((ds.y_test==5).sum()))
    return replace(ds,x_attack=None,y_attack=None),identity


def pairing_key(task):
    c = task['config']
    return f"{c['dataset']}_{c['partition']}_r{round(c['malicious_ratio']*100):02d}_s{c['seed']}"


def prepare(output,data_root,spec=CONFIG):
    from hfcdr_pilot_data import participant_record
    out = safe_output(output)
    if out.exists(): raise FileExistsError('Fresh validation output required')
    manifest = plan(spec); identities,pairing = {},{}
    for name in manifest['policy']['datasets']:
        ds,identities[name] = load_dataset(name,data_root)
        for t in manifest['tasks']:
            if t['config']['dataset']==name and pairing_key(t) not in pairing:
                pairing[pairing_key(t)] = participant_record(ds.y_train,t['config'])
        del ds
    out.mkdir(parents=True)
    write(out/'manifest.json',manifest); write(out/'dataset_identity.json',identities)
    write(out/'pairing.json',pairing)
    audit = dict(manifest['audit'],data_verified=True,pairings=len(pairing),training_started=False)
    write(out/'configuration_audit.json',audit)
    return dict(audit,fingerprint=manifest['fingerprint'])


def check(output):
    out = safe_output(output); manifest = read(out/'manifest.json')
    if manifest != plan(ROOT/manifest['specification']):
        raise ValueError('Frozen validation source/configuration manifest changed')
    counts = Counter()
    for t in manifest['tasks']:
        folder = out/'tasks'/t['task_id']; status = folder/'status.json'
        counts[read(status)['status'] if status.exists() else 'incomplete' if folder.exists() else 'pending'] += 1
    return dict(fingerprint=manifest['fingerprint'],counts=dict(counts),tasks=len(manifest['tasks']))
