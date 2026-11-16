"""Fixed, preparation-only-by-default CIFAR pilot contract (stdlib only)."""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'configs/hfcdr_attack_pilot_v1.json'
OUTPUT = ROOT / 'outputs/hfcdr_attack_pilot_v1'
OWN = ('run_hfcdr_attack_pilot.py', 'hfcdr_pilot_protocol.py',
       'hfcdr_pilot_data.py', 'hfcdr_pilot_runtime.py', 'hfcdr_pilot_report.py',
       'configs/hfcdr_attack_pilot_v1.json', 'tests/test_hfcdr_attack_pilot.py',
       'HFCDR_PILOT.txt')
METHOD_SOURCE = dict(hfcdr='sm9rrs', fedavg='fedavg', krum='krum',
                     vert='vert', alignins='alignins', tad='ding13')
COMMON = ('num_clients', 'rounds', 'local_epochs', 'batch_size', 'lr', 'lr_decay',
          'dirichlet_alpha', 'attack_start_round', 'attack_source_label',
          'attack_target_label', 'eval_interval', 'early_stop', 'compute_backend')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def baseline():
    policy = read(CONFIG)
    path = ROOT / policy['baseline']
    if sha(path) != policy['baseline_sha256']:
        raise ValueError('Pinned cifar-best configuration changed')
    return read(path)


def verify_frozen_sources():
    sources = baseline()['scientific_sources']
    bad = [name for name, expected in sources.items() if sha(ROOT / name) != expected]
    if bad:
        raise ValueError('Frozen source changed: ' + ', '.join(bad))
    return dict(sources)


def task_specs():
    p, b = read(CONFIG), baseline()
    tasks = []
    for partition, condition, method in itertools.product(p['partitions'], p['conditions'], p['methods']):
        old = b['methods'][METHOD_SOURCE[method]]['config']
        config = {k: old[k] for k in COMMON}
        if method == 'hfcdr':
            config.update({k: v for k, v in old.items()
                           if k.startswith(('detector_', 'suspicion_'))})
        elif method == 'vert':
            config.update({k: v for k, v in old.items() if k.startswith('vert_')})
        elif method == 'alignins':
            config.update({k: v for k, v in old.items() if k.startswith('alignins_')})
        config.update(method=method, partition=partition, seed=p['seed'],
                      attack=condition['attack'], scale=condition['scale'],
                      malicious_ratio=condition['malicious_ratio'])
        tasks.append(dict(task_id=f"{method}_{partition}_{condition['name']}_s{p['seed']}",
                          condition=condition['name'], config=config))
    return tasks


def audit_tasks(tasks):
    if tasks != task_specs():
        raise ValueError('Tasks differ from the fixed first-batch contract')
    ids = [t['task_id'] for t in tasks]
    if len(tasks) != 48 or len(set(ids)) != 48:
        raise ValueError('Exactly 48 distinct first-seed tasks required')
    for task in tasks:
        c = task['config']
        if (c['rounds'], c['num_clients'], c['batch_size'], c['attack_start_round']) != (150, 100, 50, 25):
            raise ValueError('Public training parameters changed')
        if any(k.startswith(('sm9', 'dkg', 'crypto')) for k in c):
            raise ValueError('Cryptographic configuration in a pure-defense task')
        if c['seed'] != 2026111601 or c['attack'] == 'alternating_minimization':
            raise ValueError('Wrong seed or legacy attack')
    return dict(passed=True, tasks=48, seeds=[2026111601], rounds_per_task=150,
                by_method=dict(Counter(t['config']['method'] for t in tasks)),
                by_partition=dict(Counter(t['config']['partition'] for t in tasks)),
                by_condition=dict(Counter(t['condition'] for t in tasks)),
                training_started=False, evaluation='training_validation_all_source5',
                cryptography='absent', automatic_retry=False)


def plan():
    b, p = baseline(), read(CONFIG)
    tasks = task_specs()
    payload = dict(schema=p['schema'], policy=p, tasks=tasks, audit=audit_tasks(tasks),
                   data_config=b['data_config'], dataset_identity=b['expected_dataset_identity'],
                   numerics=b['numerics'], original_environment={k: v for k, v in
                   b['original_environment'].items() if k != 'native_binary_sha256'},
                   frozen_sources=verify_frozen_sources(),
                   new_sources={name: sha(ROOT / name) for name in OWN})
    return dict(payload, fingerprint=digest(payload))


def safe_output(path):
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ValueError('Output cannot be a symlink')
    out = raw.resolve()
    # All pilot writes are confined to a new, explicitly named sibling.
    if out.parent != (ROOT / 'outputs').resolve() or not out.name.startswith('hfcdr_attack_pilot'):
        raise ValueError('Use a fresh outputs/hfcdr_attack_pilot* directory')
    return out


def protected_snapshot():
    """Hash every current cifar-best file, including the manifest itself."""
    frozen = verify_frozen_sources()
    folder = ROOT / 'outputs/cifar-best'
    inventory = read(folder / 'provenance/current_file_manifest.json')['files']
    actual = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p))
              for p in sorted(folder.rglob('*')) if p.is_file()}
    for relative, entry in inventory.items():
        if actual.get('outputs/cifar-best/' + relative) != entry:
            raise ValueError('Accepted cifar-best artifact changed: ' + relative)
    actual.update({name: dict(bytes=(ROOT / name).stat().st_size, sha256=value)
                   for name, value in frozen.items()})
    return actual


def verify_snapshot(snapshot):
    if protected_snapshot() != snapshot:
        raise ValueError('Protected source/result inventory changed')


def prepare(output, data_dir):
    out = safe_output(output)
    if out.exists():
        raise FileExistsError('Preparation requires a fresh directory; existing evidence is retained')
    manifest = plan()
    protected = protected_snapshot()
    from hfcdr_pilot_data import load_validation, pairing_manifest
    dataset, identity = load_validation(data_dir)
    pairing = pairing_manifest(dataset, manifest['tasks'])
    # No experiment/worker function is imported or called by preparation.
    out.mkdir()
    write(out / 'manifest.json', manifest)
    write(out / 'configuration_audit.json', dict(manifest['audit'],
          dataset_verified=True, accuracy_samples=len(dataset.y_test),
          source5_samples=int((dataset.y_test == 5).sum()),
          protected_files=len(protected), pending=48, running=0, complete=0))
    write(out / 'dataset_identity.json', identity)
    write(out / 'pairing.json', pairing)
    write(out / 'provenance/protected_files.json', protected)
    return read(out / 'configuration_audit.json')


def check_prepared(output, *, verify_results=True):
    out = safe_output(output)
    manifest = read(out / 'manifest.json')
    if manifest != plan():
        raise ValueError('Prepared source, configuration or task manifest changed')
    if read(out / 'dataset_identity.json') != manifest['dataset_identity']:
        raise ValueError('Prepared validation identity changed')
    if verify_results:
        verify_snapshot(read(out / 'provenance/protected_files.json'))
    counts = Counter()
    for task in manifest['tasks']:
        folder = out / 'tasks' / task['task_id']
        status = folder / 'status.json'
        counts[read(status)['status'] if status.exists() else 'incomplete' if folder.exists() else 'pending'] += 1
    configuration = {k: v for k, v in manifest['audit'].items() if k != 'training_started'}
    return dict(fingerprint=manifest['fingerprint'], configuration=configuration,
                counts=dict(counts), training_started=any(k != 'pending' and n for k, n in counts.items()),
                protected_results_checked=verify_results)
