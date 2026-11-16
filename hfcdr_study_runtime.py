"""Official-test study loop using the tested pure-defense pilot primitives."""
from __future__ import annotations

from time import perf_counter
import numpy as np

from hfcdr_pilot_data import participants, array_digest
from hfcdr_pilot_runtime import (LocalTrainer, Server, assert_no_cryptography, attack_active,
                                 scale_update, finite, cpu, weight_metrics, configure_cuda)
from sm9rrsfl.model import init_params, model_spec_for_dataset


def run_loop(dataset, task, *, device, emit, trainer_factory=LocalTrainer):
    assert_no_cryptography()
    config = task['config']
    ids, indices, malicious = participants(dataset.y_train, config)
    samples = {i: len(idx) for i, idx in zip(ids, indices)}
    source_counts = {i: int((dataset.y_train[idx] == 5).sum()) for i, idx in zip(ids, indices)}
    spec = model_spec_for_dataset(dataset)
    setup_start = perf_counter()
    trainer = trainer_factory(dataset, indices, spec, device)
    context = trainer.context
    defense_config = {k: v for k, v in config.items() if k not in
        ('attack', 'scale', 'malicious_ratio', 'attack_start_round', 'attack_source_label', 'attack_target_label')}
    server = Server(defense_config, samples, spec, context,
                    krum_bound=len(malicious) if config['method'] == 'krum' else 0)
    params = init_params(seed=config['seed'], spec=spec)
    is_cuda = context is not None and str(context.device).startswith('cuda')
    def sync():
        if is_cuda:
            context.torch.cuda.synchronize(context.device)
    sync()
    if is_cuda:
        context.torch.cuda.reset_peak_memory_stats(context.device)
    setup_seconds = perf_counter() - setup_start
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

    def evaluate(round_id, coefficients, stage, **extra):
        sync(); begin = perf_counter()
        metrics = trainer.evaluate(params)
        sync()
        elapsed = perf_counter() - begin
        timers['evaluation_seconds'] += elapsed
        row = dict(round=round_id, **metrics, **extra,
                   round_timing=dict(stage, evaluation_seconds=elapsed))
        if round_id:
            row.update(weight_metrics(samples, malicious, coefficients),
                accepted_updates=sum(v > 0 for v in coefficients.values()),
                removed_clients=len(server.removed),
                true_positive_removals=len(server.removed & malicious),
                false_positive_removals=len(server.removed - malicious),
                honest_clients=len(ids) - len(malicious))
        rows.append(row)
        output('round', row)

    evaluate(0, {}, dict(local_training_seconds=0., defense_aggregation_seconds=0.))
    for round_id in range(1, config['rounds'] + 1):
        if len(server.removed) == len(ids):
            raise RuntimeError('Scientific collapse: all clients removed')
        lr = config['lr'] * config['lr_decay'] ** (round_id - 1)
        updates, exposures, local_loss, trained_samples = {}, {}, 0., 0
        sync(); begin = perf_counter()
        for idx, identity in enumerate(ids):
            if identity in server.removed:
                continue
            poisoned = attack_active(config, identity, malicious, round_id)
            delta, stats = trainer.local(params, idx, poisoned=poisoned, lr=lr,
                epochs=config['local_epochs'], batch_size=config['batch_size'],
                seed=config['seed'] + round_id * 1009 + idx)
            if stats.samples != samples[identity]:
                raise ValueError('Client sample count changed')
            if not np.isfinite(stats.loss):
                raise FloatingPointError(f'Nonfinite local loss: round {round_id}, {identity}')
            delta = scale_update(delta, config['scale']) if poisoned else delta
            if not finite(delta):
                raise FloatingPointError(f'Nonfinite upload: round {round_id}, {identity}')
            updates[identity] = delta
            trained_samples += stats.samples
            local_loss += stats.loss * stats.samples
            exposures[identity] = dict(poisoned=poisoned,
                labels_flipped=source_counts[identity] if poisoned else 0,
                scale=config['scale'] if poisoned else 1.)
        sync(); local_seconds = perf_counter() - begin
        timers['local_training_seconds'] += local_seconds
        begin = perf_counter()
        aggregate, coefficients, detail = server.step(params, updates, round_id=round_id, lr=lr)
        params = context.add_update(params, aggregate) if context else (params + aggregate).astype(np.float32)
        if not finite(params):
            raise FloatingPointError('Nonfinite global model')
        sync(); defense_seconds = perf_counter() - begin
        timers['defense_aggregation_seconds'] += defense_seconds
        output('audit', dict(round=round_id, binding=binding, coefficients=coefficients,
                            exposures=exposures, defense=detail))
        evaluate(round_id, coefficients,
                 dict(local_training_seconds=local_seconds, defense_aggregation_seconds=defense_seconds),
                 attack_active=round_id >= config['attack_start_round'] and bool(malicious),
                 active_clients=len(updates), trained_samples=trained_samples,
                 local_training_loss=local_loss / trained_samples if trained_samples else 0.,
                 poisoned_samples=sum(x['labels_flipped'] for x in exposures.values()))
    sync()
    runtime_seconds = perf_counter() - started - callback_seconds
    final = cpu(params).copy()
    emit('final_model', final)
    assert_no_cryptography()
    memory = dict(cuda_peak_allocated_mib=None, cuda_peak_reserved_mib=None)
    if is_cuda:
        memory = dict(cuda_peak_allocated_mib=context.torch.cuda.max_memory_allocated(context.device)/2**20,
                      cuda_peak_reserved_mib=context.torch.cuda.max_memory_reserved(context.device)/2**20)
    return dict(task_id=task['task_id'], config=config, binding=binding, records=rows,
        stopped_round=config['rounds'], nonfinite_updates=0, cryptography='absent',
        model_sha256=array_digest(final), evaluation_scope='official_test_all_source5',
        official_test_used=True, official_test_used_for_selection=False,
        memory=memory, timings=dict(timers, runtime_seconds=runtime_seconds,
            context_setup_seconds=setup_seconds, artifact_callback_seconds=callback_seconds,
            definition='Synchronized wall seconds: initial evaluation plus all local training, defense/aggregation and official evaluation; data/context setup and artifact I/O excluded. Shared GPU load is recorded separately.'))
