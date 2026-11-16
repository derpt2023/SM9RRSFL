"""Independent six-defense loop. No fl runner, signer, verifier or crypto context."""
from __future__ import annotations

from dataclasses import asdict, replace
import importlib
import json
import os
import platform
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from hfcdr_pilot_data import participants, array_digest
from sm9rrsfl.aggregation import krum, weighted_fedavg
from sm9rrsfl.alignins import AlignInsDefense, aggregate_with_coefficients
from sm9rrsfl.model import init_params, model_spec_for_dataset
from sm9rrsfl.ours_policy import OursParameters
from sm9rrsfl.svd_detector import LongitudinalSVDDetector
from sm9rrsfl.weighting import SuspicionWeightManager, bounded_aggregation_coefficients
from cifar_tad_no_prior import PriorFreeTAD
from cifar_vert_paper_core import PaperVERT


def assert_no_cryptography():
    forbidden = ('sm9rrsfl.crypto', 'sm9rrsfl.sm9_backend', 'sm9rrsfl.threshold',
                 'sm9rrsfl.accumulator', 'sm9rrsfl._native_sm9', 'sm9rrsfl._native_sm3',
                 'sm9rrsfl.fl', 'sm9rrsfl.experiments')
    loaded = [name for name in forbidden if name in sys.modules]
    if loaded:
        raise RuntimeError('Pure pilot must not import legacy FL/crypto modules: ' + ', '.join(loaded))


def cpu(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def finite(value):
    if hasattr(value, 'detach'):
        import torch
        return bool(torch.isfinite(value).all().item())
    return bool(np.isfinite(value).all())


def flip_labels(labels, source=5, target=7):
    if source == target:
        raise ValueError('Distinct source and target required')
    result = np.array(labels, copy=True)
    result[result == source] = target
    return result


def attack_active(config, client, malicious, round_id):
    return config['attack'] != 'none' and client in malicious and round_id >= config['attack_start_round']


def scale_update(update, factor):
    if isinstance(factor, bool) or not np.isfinite(factor) or factor <= 0:
        raise ValueError('Positive finite scale required')
    return update if factor == 1 else update * factor


class LocalTrainer:
    """Swap label tensor views for one client call; preserve frozen SGD exactly."""
    def __init__(self, dataset, indices, spec, device):
        from sm9rrsfl.torch_backend import TorchTrainingContext
        # The reserved auxiliary set is verified, but never exposed to training.
        self.context = TorchTrainingContext(replace(dataset, x_attack=None, y_attack=None),
                                            indices, spec=spec, device=device)
        self.clean_labels = self.context.y_train
        self.poisoned_labels = self.context.torch.as_tensor(flip_labels(dataset.y_train),
            dtype=self.context.torch.long, device=self.context.device)

    def local(self, params, client_idx, *, poisoned, lr, epochs, batch_size, seed):
        ctx = self.context
        previous = ctx.y_train
        try:
            ctx.y_train = self.poisoned_labels if poisoned else self.clean_labels
            return ctx.local_train_delta_resident(params, client_idx=client_idx,
                        lr=lr, epochs=epochs, batch_size=batch_size, seed=seed)
        finally:
            ctx.y_train = previous

    def evaluate(self, params):
        return validation_metrics(self.context, params)


def validation_metrics(context, vector):
    from sm9rrsfl.torch_backend import _torch_params_from_tensor, _torch_forward
    torch = context.torch
    params = _torch_params_from_tensor(torch, context._ensure_global_vector(vector),
                                      context.spec, requires_grad=False, clone=False)
    confusion = np.zeros((10, 10), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(context.y_test), 512):
            logits = _torch_forward(torch, params, context.x_test[start:start + 512], context.spec)
            if not finite(logits):
                raise FloatingPointError('Nonfinite validation logits')
            pred = cpu(logits.argmax(dim=1))
            labels = cpu(context.y_test[start:start + 512])
            confusion += np.bincount(labels * 10 + pred, minlength=100).reshape(10, 10)
    return metrics_from_confusion(confusion)


def metrics_from_confusion(confusion):
    m = np.asarray(confusion)
    if m.shape != (10, 10) or m.sum() <= 0 or m[5].sum() <= 0 or np.any(m < 0):
        raise ValueError('Validation confusion matrix needs source-class observations')
    return dict(accuracy=float(m.trace() / m.sum()), asr=float(m[5, 7] / m[5].sum()),
                source_accuracy=float(m[5, 5] / m[5].sum()),
                accuracy_samples=int(m.sum()), source_samples=int(m[5].sum()),
                confusion=m.tolist())


def normalized(ids, values):
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('Invalid aggregation coefficients')
    total = float(a.sum())
    return dict(zip(ids, (a / total).tolist())) if total > 0 else dict.fromkeys(ids, 0.)


class DirectHFCDR:
    """The frozen R4 detection/weight equations with direct server-side removal."""
    def __init__(self, config, samples, spec):
        p = OursParameters.from_object(SimpleNamespace(**config))
        p.validate()
        self.policy, self.samples = p, dict(samples)
        self.detector = LongitudinalSVDDetector(window_size=config['detector_window'], policy=p,
            num_classes=spec.num_classes, expected_update_size=spec.parameter_size,
            matrix_offset=spec.svd_matrix_offset, matrix_shape=spec.svd_matrix_shape)
        self.weights = SuspicionWeightManager(penalty_factor=p.suspicion_penalty_factor,
            recovery_factor=p.suspicion_recovery_factor, remove_after=p.suspicion_remove_after)
        self.removed = set()

    def coefficients(self, updates, *, round_id, lr):
        if set(updates) & self.removed or set(updates) - self.samples.keys():
            raise ValueError('Unexpected or removed participant')
        decisions = {i: self.detector.evaluate(i, cpu(delta), round_id=round_id, learning_rate=lr)
                     for i, delta in updates.items()}
        suspicious = {i for i, d in decisions.items() if d.would_flag}
        result = self.weights.update(list(updates), suspicious,
            {i for i, d in decisions.items() if d.count_increment},
            recovery_tags={i for i, d in decisions.items() if d.recovery_eligible and i not in suspicious},
            immediate_revocation_tags={i for i, d in decisions.items() if d.immediate_revocation})
        # In the new trusted-server threat model there is no certificate gate.
        for identity in sorted(result.trace_requested_tags):
            self.weights.confirm_revocation(identity)
            self.detector.forget(identity)
            self.removed.add(identity)
        total = sum(self.samples.values())
        nominal = {i: self.samples[i] / total for i in updates}
        coefficients = bounded_aggregation_coefficients(list(updates), nominal, result.weights,
                                                         decisions, self.policy.detector_weight_cap)
        audit = {}
        for identity, decision in decisions.items():
            admitted = False
            if identity not in self.removed:
                admitted = self.detector.commit(identity, admit_history=(
                    coefficients[identity] > 0 and not result.history_frozen))
            audit[identity] = dict(asdict(decision), coefficient=coefficients[identity],
                reliability=self.weights.weights.get(identity, 1.),
                suspicion_count=self.weights.evidence_counts.get(identity, 0.),
                history_admitted=admitted, removed=identity in self.removed)
        return coefficients, dict(clients=audit, history_frozen=result.history_frozen,
                                  newly_removed=sorted(result.trace_requested_tags))


class Server:
    """No malicious identity input; only Krum receives its declared count bound."""
    def __init__(self, config, samples, spec, context, *, krum_bound=0):
        self.config, self.samples, self.context = config, dict(samples), context
        self.method, self.krum_bound = config['method'], krum_bound
        self.removed, self.defense = set(), None
        if self.method == 'hfcdr':
            self.defense = DirectHFCDR(config, samples, spec)
        elif self.method == 'tad':
            self.defense = PriorFreeTAD(list(samples), seed=config['seed'],
                matrix_offset=spec.svd_matrix_offset, matrix_shape=spec.svd_matrix_shape,
                compute_backend=config['compute_backend'], device=context.device if context else 'cpu')
        elif self.method == 'vert':
            self.defense = PaperVERT(list(samples), parameter_size=spec.parameter_size,
                history_window=config['vert_history_window'], projection_dim=config['vert_projection_dim'],
                predict_epochs=config['vert_predict_epochs'], learning_rate=config['vert_predict_lr'],
                top_k=0, malicious_ratio_prior=None, compute_backend=config['compute_backend'],
                device=context.device if context else 'cpu', seed=config['seed'])
        elif self.method == 'alignins':
            self.defense = AlignInsDefense(**{k: v for k, v in config.items() if k.startswith('alignins_')})
        elif self.method not in ('fedavg', 'krum'):
            raise ValueError('Unknown method')

    def binding(self):
        obj = self.defense
        return dict(method=self.method, implementation=(type(obj).__module__ + '.' + type(obj).__qualname__)
                    if obj is not None else 'sm9rrsfl.aggregation.' + self.method,
                    detector=(type(obj.detector).__module__ + '.' + type(obj.detector).__qualname__)
                    if self.method == 'hfcdr' else None, cryptography='absent',
                    krum_count_bound=self.krum_bound if self.method == 'krum' else None)

    def average(self, updates, weights, counts=None):
        values = list(updates.values())
        effective = np.asarray(weights, dtype=np.float64)
        if counts is not None:
            effective *= np.asarray(counts)
        if effective.sum() <= 0:
            return values[0] * 0
        if self.context is not None:
            return self.context.weighted_average(values, weights, sample_counts=counts)
        return weighted_fedavg(values, weights, sample_counts=counts)

    def step(self, params, updates, *, round_id, lr):
        if not updates:
            return params * 0, {}, dict(empty=True)
        if set(updates) & self.removed or set(updates) - self.samples.keys():
            raise ValueError('Invalid active participant set')
        if any(not finite(delta) for delta in updates.values()):
            raise FloatingPointError('Nonfinite upload')
        ids, detail = list(updates), {}
        if self.method == 'hfcdr':
            coef, detail = self.defense.coefficients(updates, round_id=round_id, lr=lr)
            self.removed = set(self.defense.removed)
            aggregate = aggregate_with_coefficients(updates, coef)
        elif self.method == 'tad':
            answer = self.defense.evaluate_round(updates, global_params=params, round_id=round_id)
            self.removed.update(answer.newly_removed)
            weights, counts = [answer.weights[i] for i in ids], [self.samples[i] for i in ids]
            coef = normalized(ids, np.asarray(weights) * counts)
            aggregate = self.average(updates, weights, counts)
            detail = dict(self.defense.last_audit)
        elif self.method == 'vert':
            raw = {i: cpu(v).copy() for i, v in updates.items()}
            answer = self.defense.evaluate_round(raw, round_id=round_id)
            selected = {i: updates[i] for i in ids if i in answer.selected_clients}
            weights = [answer.weights[i] for i in selected]
            coef = dict.fromkeys(ids, 0.)
            coef.update(normalized(list(selected), weights))
            aggregate = self.average(selected, weights) if selected else next(iter(updates.values())) * 0
            self.defense.finalize_round(raw, cpu(aggregate), answer)
            detail = dict(scores=answer.scores, selected=list(answer.selected_clients))
        elif self.method == 'alignins':
            answer = self.defense.evaluate_round(params, updates)
            coef = dict(answer.aggregation_coefficients)
            aggregate = aggregate_with_coefficients(updates, coef)
            detail = dict(selected=list(answer.selected_clients), clip_factors=answer.clip_factors)
        elif self.method == 'krum':
            coef = dict.fromkeys(ids, 0.)
            if len(ids) - self.krum_bound - 2 < 1:
                aggregate = next(iter(updates.values())) * 0
                detail = dict(undefined_neighbor_count=True)
            else:
                if self.context is not None:
                    index, aggregate, scores = self.context.krum_select(list(updates.values()),
                                                                         byzantine_count=self.krum_bound)
                else:
                    answer = krum(list(updates.values()), self.krum_bound)
                    index, aggregate, scores = answer.selected_index, answer.update, answer.scores
                coef[ids[index]] = 1.
                detail = dict(selected=ids[index], scores=np.asarray(scores).tolist())
        else:
            weights = [self.samples[i] for i in ids]
            coef = normalized(ids, weights)
            aggregate = self.average(updates, weights)
        if not finite(aggregate) or any(not np.isfinite(v) or v < 0 for v in coef.values()):
            raise FloatingPointError('Invalid aggregate or coefficients')
        return aggregate, coef, detail


def weight_metrics(samples, malicious, coefficients):
    total = sum(samples.values())
    honest = [i for i in samples if i not in malicious]
    honest_mass = sum(samples[i] / total for i in honest)
    lost = sum(max(0., samples[i] / total - coefficients.get(i, 0.)) for i in honest)
    return dict(honest_weight_loss=lost / honest_mass if honest_mass else 0.,
                malicious_weight_mass=sum(coefficients.get(i, 0.) for i in sorted(malicious)))


def run_loop(dataset, task, *, device, emit, trainer_factory=LocalTrainer):
    """Only explicit worker execution calls this; unit tests inject fixed updates."""
    assert_no_cryptography()
    config = task['config']
    ids, indices, malicious = participants(dataset.y_train, config)
    samples = {i: len(idx) for i, idx in zip(ids, indices)}
    source_counts = {i: int((dataset.y_train[idx] == config['attack_source_label']).sum())
                     for i, idx in zip(ids, indices)}
    spec = model_spec_for_dataset(dataset)
    trainer = trainer_factory(dataset, indices, spec, device)
    context = trainer.context
    # Detector constructor receives no attacker identities or evaluation data.
    defense_config = {k: v for k, v in config.items() if k not in
        ('attack', 'scale', 'malicious_ratio', 'attack_start_round', 'attack_source_label', 'attack_target_label')}
    server = Server(defense_config, samples, spec, context,
                    krum_bound=len(malicious) if config['method'] == 'krum' else 0)
    params = init_params(seed=config['seed'], spec=spec)
    binding = server.binding()
    emit('binding', binding)
    emit('initial_model', dict(sha256=array_digest(params)))
    timers = dict(local_training_seconds=0., defense_aggregation_seconds=0., evaluation_seconds=0.)
    rows = []
    started, callback_seconds = perf_counter(), 0.

    def output(kind, value):
        nonlocal callback_seconds
        t = perf_counter()
        emit(kind, value)
        callback_seconds += perf_counter() - t

    def evaluate(round_id, coefficients, **extra):
        t = perf_counter()
        metrics = trainer.evaluate(params)
        timers['evaluation_seconds'] += perf_counter() - t
        row = dict(round=round_id, **metrics, **extra)
        if round_id:
            row.update(weight_metrics(samples, malicious, coefficients),
                accepted_updates=sum(v > 0 for v in coefficients.values()),
                removed_clients=len(server.removed),
                true_positive_removals=len(server.removed & malicious),
                false_positive_removals=len(server.removed - malicious),
                honest_clients=len(ids) - len(malicious))
        rows.append(row)
        output('round', row)

    evaluate(0, {})
    for round_id in range(1, config['rounds'] + 1):
        if len(server.removed) == len(ids):
            raise RuntimeError('All clients removed; retain incomplete experiment')
        lr = config['lr'] * config['lr_decay'] ** (round_id - 1)
        updates, exposures = {}, {}
        t = perf_counter()
        for idx, identity in enumerate(ids):
            if identity in server.removed:
                continue
            poisoned = attack_active(config, identity, malicious, round_id)
            delta, stats = trainer.local(params, idx, poisoned=poisoned, lr=lr,
                epochs=config['local_epochs'], batch_size=config['batch_size'],
                seed=config['seed'] + round_id * 1009 + idx)
            if stats.samples != samples[identity]:
                raise ValueError('Client sample count changed')
            delta = scale_update(delta, config['scale']) if poisoned else delta
            if not finite(delta):
                # Never silently filter an unstable attack and call it defended.
                raise FloatingPointError(f'Nonfinite upload: round {round_id}, {identity}')
            updates[identity] = delta
            exposures[identity] = dict(poisoned=poisoned,
                labels_flipped=source_counts[identity] if poisoned else 0,
                scale=config['scale'] if poisoned else 1.)
        timers['local_training_seconds'] += perf_counter() - t
        t = perf_counter()
        aggregate, coefficients, detail = server.step(params, updates, round_id=round_id, lr=lr)
        params = context.add_update(params, aggregate) if context else (params + aggregate).astype(np.float32)
        timers['defense_aggregation_seconds'] += perf_counter() - t
        if not finite(params):
            raise FloatingPointError('Nonfinite global model')
        output('audit', dict(round=round_id, binding=binding, coefficients=coefficients,
                            exposures=exposures, defense=detail))
        evaluate(round_id, coefficients, attack_active=round_id >= config['attack_start_round'] and bool(malicious),
                 poisoned_samples=sum(x['labels_flipped'] for x in exposures.values()))
    assert_no_cryptography()
    return dict(task_id=task['task_id'], config=config, binding=binding, records=rows,
                stopped_round=config['rounds'], nonfinite_updates=0, cryptography='absent',
                model_sha256=array_digest(cpu(params)), evaluation_scope='training_validation_all_source5',
                official_test_used=False, timings=dict(timers,
                    runtime_seconds=perf_counter() - started - callback_seconds,
                    artifact_callback_seconds=callback_seconds,
                    definition='initial evaluation plus local training, defense and evaluation; dataset/context setup and artifact callbacks excluded'))


def configure_cuda(manifest):
    """Strict production preflight; never called by plan/prepare/check or tests."""
    assert_no_cryptography()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible:
        raise ValueError('Expose exactly one explicitly selected CUDA GPU')
    expected = manifest['numerics']
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != expected['cublas_workspace_config']:
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must be set before importing Torch')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Pilot workers require exactly one visible CUDA GPU')
    current = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        sklearn=importlib.import_module('sklearn').__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0))
    original = manifest['original_environment']
    differences = {k: (original[k].split()[0] if k == 'python' else original[k], v)
        for k, v in current.items() if v != (original[k].split()[0] if k == 'python' else original[k])}
    if differences:
        raise ValueError('CUDA worker environment differs from cifar-best: ' + repr(differences))
    torch.set_num_threads(expected['torch_threads'])
    torch.use_deterministic_algorithms(expected['deterministic_algorithms'])
    torch.backends.cudnn.deterministic = expected['cudnn_deterministic']
    torch.backends.cudnn.benchmark = expected['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = expected['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = expected['cudnn_allow_tf32']
    free, total = torch.cuda.mem_get_info(0)
    if free < 8192 * 1024 * 1024:
        raise ValueError('At least 8192 MiB free GPU memory required')
    return dict(current=current, numerics=expected, visible_gpu=visible,
                free_bytes=free, total_bytes=total, cryptography='absent')
