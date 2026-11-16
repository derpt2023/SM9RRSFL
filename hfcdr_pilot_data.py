"""Read-only validation split and matched client identities; no FL/crypto import."""
from __future__ import annotations

import hashlib
import numpy as np

from hfcdr_pilot_protocol import baseline, digest
from sm9rrsfl.datasets import load_image_dataset, stratified_training_three_way_split, partition_clients


def array_digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    view = memoryview(array).cast('B')
    for offset in range(0, len(view), 8 * 1024 * 1024):
        h.update(view[offset:offset + 8 * 1024 * 1024])
    return h.hexdigest()


def load_validation(data_dir):
    b = baseline()
    d = b['data_config']
    raw = load_image_dataset('cifar10', data_dir, download=False, train_limit=50000,
                             test_limit=10000, seed=d['seed'])
    split = stratified_training_three_way_split(raw, seed=d['split_seed'],
                train_fraction=.8, calibration_fraction=.15, attack_fraction=.05)
    contract = dict(source='original training split only', official_test_used_for_selection=False,
        shared_training_arrays_across_phases=True, split_seed=d['split_seed'],
        train_samples=len(split.train_indices), calibration_samples=len(split.calibration_indices),
        attack_auxiliary_samples=len(split.attack_indices), train_fraction_actual=.8,
        train_indices_digest=array_digest(split.train_indices),
        calibration_indices_digest=array_digest(split.calibration_indices),
        attack_indices_digest=array_digest(split.attack_indices))
    dataset = split.calibration_dataset
    identity = dict(contract=contract, arrays={name: array_digest(getattr(dataset, name))
        for name in ('x_train', 'y_train', 'x_test', 'y_test', 'x_attack', 'y_attack')})
    if identity != b['expected_dataset_identity']:
        raise ValueError('Actual validation data differ from cifar-best')
    return dataset, identity


def participants(labels, config):
    ids = [f'client-{i}' for i in range(config['num_clients'])]
    indices = partition_clients(labels, len(ids), strategy=config['partition'],
        dirichlet_alpha=config['dirichlet_alpha'], seed=config['seed'])
    count = round(len(ids) * config['malicious_ratio'])
    rng = np.random.default_rng(config['seed'] + 17)
    malicious = set(str(i) for i in rng.choice(ids, count, replace=False)) if count else set()
    return ids, indices, malicious


def participant_record(labels, config):
    ids, indices, malicious = participants(labels, config)
    per_client = {i: dict(samples=len(idx), indices_sha256=array_digest(idx),
                         source5_samples=int((labels[idx] == 5).sum()))
                  for i, idx in zip(ids, indices)}
    return dict(clients=per_client, malicious_clients=sorted(malicious),
        partition_sha256=digest(per_client),
        poisoned_samples=sum(per_client[i]['source5_samples'] for i in malicious),
        malicious_without_source5=sorted(i for i in malicious if not per_client[i]['source5_samples']))


def pairing_manifest(dataset, tasks):
    result = {}
    for partition in ('iid', 'dirichlet'):
        relevant = [t for t in tasks if t['config']['partition'] == partition]
        records = {condition: participant_record(dataset.y_train, next(t['config'] for t in relevant
                   if t['condition'] == condition)) for condition in ('clean', 'label_flip', 'scale2', 'scale5')}
        if len({r['partition_sha256'] for r in records.values()}) != 1:
            raise ValueError('Clean/attack client partitions differ')
        if not records['label_flip'] == records['scale2'] == records['scale5']:
            raise ValueError('Attack identities or poisoned samples differ across scales')
        result[partition] = records
    return result
