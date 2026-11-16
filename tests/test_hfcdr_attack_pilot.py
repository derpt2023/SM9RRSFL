"""Contract and numerical integration checks; no real local SGD/CIFAR training."""
from __future__ import annotations

import copy
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import hfcdr_pilot_protocol as p
import hfcdr_pilot_data as data
import hfcdr_pilot_runtime as runtime
import hfcdr_pilot_report as reporting
import run_hfcdr_attack_pilot as cli
from sm9rrsfl.datasets import ImageDataset
from sm9rrsfl.model import ModelSpec, TrainStats, model_spec_for_dataset
from sm9rrsfl.svd_detector import DetectionResult
from sm9rrsfl.torch_backend import TorchTrainingContext

torch.set_num_threads(1)


def task(method='hfcdr', condition='label_flip'):
    return copy.deepcopy(next(t for t in p.task_specs() if t['config']['method'] == method
                              and t['condition'] == condition and t['config']['partition'] == 'iid'))


def toy_dataset():
    y = np.tile(np.arange(10, dtype=np.int64), 6)
    return ImageDataset(x_train=np.zeros((60, 1, 4, 4), dtype=np.float32), y_train=y,
        x_test=np.zeros((20, 1, 4, 4), dtype=np.float32), y_test=np.tile(np.arange(10), 2),
        name='unit', num_classes=10)


class ProtocolTests(unittest.TestCase):
    def test_exact_48_cell_cartesian_product_and_fixed_r4(self):
        tasks = p.task_specs()
        audit = p.audit_tasks(tasks)
        self.assertEqual(audit['by_method'], dict.fromkeys(p.METHOD_SOURCE, 8))
        self.assertEqual(audit['by_partition'], {'iid': 24, 'dirichlet': 24})
        self.assertEqual(audit['by_condition'], dict.fromkeys(('clean', 'label_flip', 'scale2', 'scale5'), 12))
        c = task()['config']
        self.assertEqual((c['detector_window'], c['suspicion_recovery_factor'], c['detector_clip_factor'],
                          c['suspicion_remove_after']), (20, 4., 1., 60))
        self.assertEqual({t['config']['seed'] for t in tasks}, {2026111601})

    def test_only_attack_identity_and_scope_differ_from_baseline(self):
        b = p.baseline()
        for t in p.task_specs():
            c = t['config']; original = b['methods'][p.METHOD_SOURCE[c['method']]]['config']
            for name, value in c.items():
                if name not in ('method', 'partition', 'seed', 'attack', 'scale', 'malicious_ratio'):
                    self.assertEqual(value, original[name], (t['task_id'], name))
            self.assertFalse(any(k in c for k in ('crypto_mode', 'sm9_workers', 'dkg_nodes',
                                                   'attack_boost', 'attack_target_count')))

    def test_mutated_missing_duplicate_and_second_seed_plans_rejected(self):
        for mutate in (lambda ts: ts.pop(), lambda ts: ts.append(ts[0]),
                       lambda ts: ts[0]['config'].update(seed=2026111602),
                       lambda ts: ts[0]['config'].update(rounds=60),
                       lambda ts: ts[0]['config'].update(crypto_mode='sm9')):
            with self.subTest(mutate=mutate):
                tasks = p.task_specs(); mutate(tasks)
                with self.assertRaises(ValueError): p.audit_tasks(tasks)

    def test_126_frozen_scientific_sources_and_new_source_fingerprint(self):
        self.assertEqual(len(p.verify_frozen_sources()), 126)
        plan = p.plan()
        self.assertEqual(plan['fingerprint'], p.digest({k: v for k, v in plan.items() if k != 'fingerprint'}))
        self.assertEqual(plan['dataset_identity']['contract']['train_samples'], 40000)
        self.assertEqual(plan['dataset_identity']['contract']['calibration_samples'], 7500)

    def test_protected_parent_old_result_and_symlink_outputs_rejected(self):
        for path in (p.ROOT, p.ROOT/'outputs', p.ROOT/'outputs/cifar-best',
                     p.ROOT/'outputs/cifar-best/hfcdr_attack_pilot', p.ROOT/'outputs/old/hfcdr_attack_pilot',
                     p.ROOT/'outputs/mnist_v7_target_fair_tuning', p.ROOT/'outputs/unrelated'):
            with self.subTest(path=path), self.assertRaises(ValueError): p.safe_output(path)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'outputs').mkdir()
            (root/'outputs/hfcdr_attack_pilot_link').symlink_to(p.ROOT/'outputs/cifar-best')
            with patch.object(p, 'ROOT', root), self.assertRaises(ValueError):
                p.safe_output(root/'outputs/hfcdr_attack_pilot_link')

    def test_default_plan_never_prepares_or_dispatches(self):
        with patch.object(cli, 'run_task') as run, patch.object(p, 'prepare') as prepare, \
             patch('sys.stdout', new_callable=io.StringIO) as stream:
            self.assertEqual(cli.main([]), 0)
        self.assertFalse(json.loads(stream.getvalue())['training_started'])
        run.assert_not_called(); prepare.assert_not_called()

    def test_prepare_writes_only_plan_and_no_worker(self):
        plan = p.plan(); ds = toy_dataset()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'new'
            with patch.object(p, 'safe_output', return_value=out), patch.object(p, 'plan', return_value=plan), \
                 patch.object(p, 'protected_snapshot', return_value={}), \
                 patch.object(data, 'load_validation', return_value=(ds, plan['dataset_identity'])), \
                 patch.object(data, 'pairing_manifest', return_value={}), patch.object(runtime, 'run_loop') as run:
                audit = p.prepare(out, '/unused')
                self.assertEqual((audit['pending'], audit['running'], audit['complete']), (48, 0, 0))
                self.assertFalse((out/'tasks').exists())
                run.assert_not_called()
                with self.assertRaises(FileExistsError): p.prepare(out, '/unused')

    def test_prepared_source_mutation_fails_before_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); p.write(out/'manifest.json', {'changed': True})
            with patch.object(p, 'safe_output', return_value=out), self.assertRaises(ValueError):
                p.check_prepared(out)

    def test_no_implicit_cpu_training_fallback(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), self.assertRaises(ValueError):
            runtime.configure_cuda(p.plan())

    def test_worker_fails_before_local_training_on_environment_error(self):
        manifest = p.plan()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp).resolve(); p.write(out/'manifest.json', manifest)
            with patch.object(p, 'safe_output', return_value=out), patch.object(p, 'check_prepared'), \
                 patch.object(runtime, 'configure_cuda', side_effect=ValueError('environment mismatch')), \
                 patch.object(runtime, 'run_loop') as run:
                with self.assertRaises(ValueError): cli.run_task(out, '/unused', manifest['tasks'][0]['task_id'])
                run.assert_not_called()
                self.assertFalse((out/'tasks').exists())

    def test_worker_failure_is_preserved_and_cannot_automatically_retry(self):
        manifest = p.plan(); selected = manifest['tasks'][0]; ds = toy_dataset()
        pairing = {'iid': {'clean': data.participant_record(ds.y_train, selected['config'])}}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp).resolve(); p.write(out/'manifest.json', manifest); p.write(out/'pairing.json', pairing)
            with patch.object(p, 'safe_output', return_value=out), patch.object(p, 'check_prepared'), \
                 patch.object(runtime, 'configure_cuda', return_value={}), \
                 patch.object(data, 'load_validation', return_value=(ds, {})), \
                 patch.object(runtime, 'run_loop', side_effect=RuntimeError('fixed test failure')) as run:
                with self.assertRaises(RuntimeError): cli.run_task(out, '/unused', selected['task_id'])
                status = p.read(out/'tasks'/selected['task_id']/'status.json')
                self.assertEqual(status['status'], 'failed'); self.assertFalse(status['automatic_retry'])
                with self.assertRaises(FileExistsError): cli.run_task(out, '/unused', selected['task_id'])
                self.assertEqual(run.call_count, 1)


class PairingAndAttackTests(unittest.TestCase):
    def test_same_partition_and_attackers_across_all_methods_and_scales(self):
        labels = np.tile(np.arange(10, dtype=np.int64), 4000)
        for part in ('iid', 'dirichlet'):
            rows = [t for t in p.task_specs() if t['config']['partition'] == part]
            records = [data.participant_record(labels, t['config']) for t in rows]
            self.assertEqual(len({r['partition_sha256'] for r in records}), 1)
            attacked = [r for r, t in zip(records, rows) if t['condition'] != 'clean']
            self.assertTrue(all(r == attacked[0] for r in attacked))
            self.assertEqual(len(attacked[0]['malicious_clients']), 40)
            if part == 'iid': self.assertEqual({v['samples'] for v in records[0]['clients'].values()}, {400})

    def test_no_source_clients_are_not_replaced(self):
        labels = np.zeros(100, dtype=np.int64)
        c = task()['config']
        record = data.participant_record(labels, c)
        self.assertEqual(record['poisoned_samples'], 0)
        self.assertEqual(record['malicious_without_source5'], record['malicious_clients'])
        self.assertEqual(len(record['malicious_clients']), 40)

    def test_label_flip_changes_only_source_and_never_mutates_original(self):
        original = np.array([1, 5, 7, 5, 9])
        changed = runtime.flip_labels(original)
        np.testing.assert_array_equal(changed, [1, 7, 7, 7, 9])
        np.testing.assert_array_equal(original, [1, 5, 7, 5, 9])

    def test_onset_round24_clean_round25_attack_and_honest_unchanged(self):
        c = task()['config']
        self.assertFalse(runtime.attack_active(c, 'bad', {'bad'}, 24))
        self.assertTrue(runtime.attack_active(c, 'bad', {'bad'}, 25))
        self.assertFalse(runtime.attack_active(c, 'honest', {'bad'}, 25))
        self.assertFalse(runtime.attack_active(task(condition='clean')['config'], 'bad', {'bad'}, 25))

    def test_scaling_complete_delta_identity_direction_and_norm(self):
        for delta in (np.array([1., -2., 3.], dtype=np.float32), torch.tensor([1., -2., 3.])):
            self.assertIs(runtime.scale_update(delta, 1), delta)
            scaled = runtime.scale_update(delta, 5)
            np.testing.assert_array_equal(runtime.cpu(scaled), [5., -10., 15.])
            np.testing.assert_array_equal(runtime.cpu(delta), [1., -2., 3.])
        for scale in (0, -1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError): runtime.scale_update(np.ones(3), scale)

    def test_real_training_context_label_view_is_restored_on_success_and_error(self):
        ds = toy_dataset(); indices = [np.arange(30), np.arange(30, 60)]
        trainer = runtime.LocalTrainer(ds, indices, model_spec_for_dataset(ds), 'cpu')
        clean = trainer.context.y_train
        for poison, fail in ((True, False), (False, False), (True, True)):
            seen = []
            def observe(*args, **kwargs):
                seen.append(runtime.cpu(trainer.context.y_train).copy())
                self.assertEqual(kwargs['client_idx'], 1)
                if fail: raise RuntimeError('unit failure')
                return torch.zeros(3), TrainStats(0., 30)
            with patch.object(trainer.context, 'local_train_delta_resident', side_effect=observe):
                if fail:
                    with self.assertRaises(RuntimeError):
                        trainer.local(np.zeros(3), 1, poisoned=poison, lr=.05, epochs=1, batch_size=50, seed=7)
                else:
                    trainer.local(np.zeros(3), 1, poisoned=poison, lr=.05, epochs=1, batch_size=50, seed=7)
            np.testing.assert_array_equal(seen[0], runtime.flip_labels(ds.y_train) if poison else ds.y_train)
            self.assertIs(trainer.context.y_train, clean)
            np.testing.assert_array_equal(runtime.cpu(clean), ds.y_train)
        self.assertEqual(trainer.context.x_attack.shape[0], 0)


class DefenseTests(unittest.TestCase):
    def spec(self):
        return ModelSpec(input_shape=(1, 4, 4))

    def test_actual_six_bindings_no_legacy_alias_or_crypto(self):
        from cifar_tad_no_prior import PriorFreeTAD
        from cifar_vert_paper_core import PaperVERT
        for method in p.METHOD_SOURCE:
            c = task(method)['config']
            server = runtime.Server(c, {'client-0': 10, 'client-1': 20}, self.spec(), None)
            if method == 'tad': self.assertIs(type(server.defense), PriorFreeTAD)
            if method == 'vert': self.assertIs(type(server.defense), PaperVERT)
            if method == 'hfcdr': self.assertIs(type(server.defense), runtime.DirectHFCDR)
        runtime.assert_no_cryptography()

    def test_real_hfcdr_clean_history_and_immediate_same_round_removal(self):
        spec = self.spec(); c = task()['config']
        defense = runtime.DirectHFCDR(c, {'a': 400, 'b': 400}, spec)
        vector = np.linspace(.001, .01, spec.parameter_size, dtype=np.float32)
        for round_id in range(1, 21):
            coef, detail = defense.coefficients({'a': vector, 'b': vector}, round_id=round_id, lr=.05)
            self.assertEqual(coef, {'a': .5, 'b': .5})
        self.assertIsNotNone(defense.detector._states['a'].anchor)
        anchor = defense.detector._states['b'].anchor
        coef, detail = defense.coefficients({'a': -100 * vector, 'b': vector}, round_id=21, lr=.05)
        self.assertEqual(coef['a'], 0.)
        self.assertIn('a', defense.removed)
        self.assertNotIn('a', defense.detector._states)
        self.assertIs(defense.detector._states['b'].anchor, anchor)
        self.assertFalse(detail['clients']['a']['history_admitted'])
        with self.assertRaises(ValueError):
            defense.coefficients({'a': vector}, round_id=22, lr=.05)

    def test_cumulative_60_round_removal_is_not_bypassed_by_mass_shock(self):
        defense = runtime.DirectHFCDR(task()['config'], {'a': 400}, self.spec())
        d = DetectionResult(False, 'suspicious', would_flag=True, count_increment=True)
        with patch.object(defense.detector, 'evaluate', return_value=d), \
             patch.object(defense.detector, 'commit', return_value=False):
            for round_id in range(1, 61):
                coef, detail = defense.coefficients({'a': np.ones(self.spec().parameter_size)},
                                                   round_id=round_id, lr=.05)
                self.assertEqual(coef['a'], 0.)
                self.assertEqual('a' in defense.removed, round_id == 60)

    def test_hfcdr_clip_and_nominal_cap_not_renormalized_after_other_removal(self):
        defense = runtime.DirectHFCDR(task()['config'], {'a': 100, 'b': 900}, self.spec())
        decision = DetectionResult(True, 'normal', clip_factor=.5)
        with patch.object(defense.detector, 'evaluate', return_value=decision), \
             patch.object(defense.detector, 'commit', return_value=False):
            coef, _ = defense.coefficients({'a': np.ones(self.spec().parameter_size)}, round_id=21, lr=.05)
        # Fixed nominal mass .1, cap 2, clip .5 => .1; surviving mass is not expanded to1.
        self.assertAlmostEqual(coef['a'], .1)

    def test_tad_actual_evaluation_and_zero_mass_no_resurrection(self):
        c = task('tad')['config']; spec = self.spec()
        server = runtime.Server(c, {'a': 10, 'b': 90}, spec, None)
        params = np.zeros(spec.parameter_size, dtype=np.float32)
        updates = {i: np.ones_like(params)*.01 for i in ('a', 'b')}
        for r in range(1, 4): server.step(params, updates, round_id=r, lr=.05)
        self.assertEqual(server.defense.last_round, 3)
        self.assertNotIn('malicious_ratio', vars(server.defense))
        answer = SimpleNamespace(weights={'a': 0., 'b': 0.}, newly_removed=set())
        with patch.object(server.defense, 'evaluate_round', return_value=answer):
            aggregate, coef, _ = server.step(params, updates, round_id=4, lr=.05)
        np.testing.assert_array_equal(aggregate, 0.)
        self.assertEqual(coef, {'a': 0., 'b': 0.})

    def test_fedavg_sample_weights_and_krum_count(self):
        updates = {'a': np.array([0., 0.], dtype=np.float32), 'b': np.array([2., 2.], dtype=np.float32)}
        server = runtime.Server(task('fedavg')['config'], {'a': 1, 'b': 3}, self.spec(), None)
        agg, coef, _ = server.step(np.zeros(2), updates, round_id=1, lr=.05)
        np.testing.assert_array_equal(agg, [1.5, 1.5]); self.assertEqual(coef, {'a': .25, 'b': .75})
        server = runtime.Server(task('krum')['config'], {str(i): 10 for i in range(5)}, self.spec(), None, krum_bound=1)
        values = {str(i): np.array([v], dtype=np.float32) for i, v in enumerate((0., .1, .2, .3, 100.))}
        agg, coef, _ = server.step(np.zeros(1), values, round_id=1, lr=.05)
        self.assertEqual(sum(coef.values()), 1.); self.assertEqual(coef['4'], 0.)

    def test_alignins_applies_native_coefficients_without_rescaling(self):
        server = runtime.Server(task('alignins')['config'], {'a': 1, 'b': 3}, self.spec(), None)
        updates = {'a': np.array([2., 1.], dtype=np.float32), 'b': np.array([20., 10.], dtype=np.float32)}
        params = np.ones(2, dtype=np.float32)
        answer = server.defense.evaluate_round(params, updates)
        aggregate, coef, _ = server.step(params, updates, round_id=1, lr=.05)
        expected = sum((updates[i]*v for i, v in answer.aggregation_coefficients.items()), np.zeros(2))
        np.testing.assert_allclose(aggregate, expected)
        self.assertEqual(coef, answer.aggregation_coefficients)

    def test_nonfinite_upload_fails_not_silently_rejected_as_defense_success(self):
        server = runtime.Server(task('fedavg')['config'], {'a': 1}, self.spec(), None)
        with self.assertRaises(FloatingPointError):
            server.step(np.ones(2), {'a': np.array([float('nan'), 1.])}, round_id=1, lr=.05)


class LoopTests(unittest.TestCase):
    class FixedTrainer:
        calls = []
        def __init__(self, ds, indices, spec, device):
            self.context = None; self.indices = indices; self.spec = spec
        def local(self, params, client_idx, **kw):
            self.calls.append((client_idx, dict(kw)))
            delta = np.full(self.spec.parameter_size, .001 * (client_idx + 1), dtype=np.float32)
            return delta, TrainStats(0., len(self.indices[client_idx]))
        def evaluate(self, params):
            return dict(accuracy=.6, asr=.08, source_accuracy=.5, accuracy_samples=20, source_samples=2)

    def run_fixed(self, method, condition, rounds=26):
        t = task(method, condition)
        t['config'].update(num_clients=6, rounds=rounds, compute_backend='numpy')
        self.FixedTrainer.calls = []; logs = []
        # No local SGD is possible here; VERT only its frozen interfaces and inference.
        with patch('cifar_vert_paper_core.fit_client', return_value=[1.]):
            result = runtime.run_loop(toy_dataset(), t, device='cpu',
                emit=lambda kind, value: logs.append((kind, value)), trainer_factory=self.FixedTrainer)
        return result, logs, list(self.FixedTrainer.calls)

    def test_all_six_real_server_paths_run_without_crypto_or_sgd(self):
        for method in p.METHOD_SOURCE:
            with self.subTest(method=method):
                result, logs, calls = self.run_fixed(method, 'clean', rounds=4)
                self.assertEqual(result['stopped_round'], 4)
                self.assertEqual([r['round'] for r in result['records']], list(range(5)))
                self.assertTrue(calls)
                self.assertTrue(all(not c['poisoned'] for _, c in calls))
                self.assertEqual(result['binding']['cryptography'], 'absent')
                json.dumps(logs, allow_nan=False)
        runtime.assert_no_cryptography()

    def test_real_loop_onset_scale_multiplier_and_random_schedule_pair(self):
        lf, lf_logs, lf_calls = self.run_fixed('fedavg', 'label_flip')
        scaled, sc_logs, sc_calls = self.run_fixed('fedavg', 'scale5')
        self.assertEqual(lf_calls, sc_calls)
        lf_audit = {v['round']: v for k, v in lf_logs if k == 'audit'}
        sc_audit = {v['round']: v for k, v in sc_logs if k == 'audit'}
        self.assertEqual(lf_audit[24], sc_audit[24])
        malicious = [i for i, x in sc_audit[25]['exposures'].items() if x['poisoned']]
        self.assertEqual(len(malicious), 2)
        self.assertTrue(all(sc_audit[25]['exposures'][i]['scale'] == 5 for i in malicious))
        self.assertNotEqual(lf['model_sha256'], scaled['model_sha256'])

    def test_scale_one_is_bitwise_label_flip_reference(self):
        a = task('fedavg'); b = task('fedavg', 'scale2')
        for t in (a, b): t['config'].update(num_clients=6, rounds=26, compute_backend='numpy', scale=1.)
        outcomes = [runtime.run_loop(toy_dataset(), t, device='cpu', emit=lambda *x: None,
                    trainer_factory=self.FixedTrainer) for t in (a, b)]
        self.assertEqual(outcomes[0]['model_sha256'], outcomes[1]['model_sha256'])
        self.assertEqual(outcomes[0]['records'], outcomes[1]['records'])

    def test_removed_clients_are_not_trained_again(self):
        original = runtime.Server.step
        def remove(server, params, updates, **kw):
            result = original(server, params, updates, **kw)
            if kw['round_id'] == 1: server.removed.add('client-0')
            return result
        with patch.object(runtime.Server, 'step', remove):
            _, _, calls = self.run_fixed('fedavg', 'clean', rounds=3)
        self.assertEqual([i for i, _ in calls].count(0), 1)

    def test_validation_uses_all_source_examples_and_true_label_accuracy(self):
        cm = np.eye(10, dtype=int) * 10
        cm[5, 5] = 4; cm[5, 7] = 6
        metrics = runtime.metrics_from_confusion(cm)
        self.assertEqual(metrics['source_samples'], 10)
        self.assertEqual(metrics['asr'], .6)
        self.assertEqual(metrics['accuracy'], .94)


class ReportTests(unittest.TestCase):
    def metrics(self, accuracy=.6, asr=.08):
        return dict(accuracy=accuracy, asr=asr, source_accuracy=.5)

    def test_predeclared_paired_gates(self):
        criteria = p.read(p.CONFIG)['criteria']
        args = [self.metrics(.59, .1), self.metrics(), self.metrics(.45, .48), self.metrics(),
                [self.metrics(.60, .09), self.metrics(.3, .01)], criteria]
        passed = reporting.assess(*args)
        self.assertTrue(passed['performance_pass'])
        self.assertAlmostEqual(passed['values']['excess_asr_reduction'], .95)
        args[0] = self.metrics(.55, .09)
        self.assertFalse(reporting.assess(*args)['checks']['accuracy_competitive'])

    def test_weak_attack_cannot_be_called_passed(self):
        same = self.metrics()
        result = reporting.assess(same, same, same, same, [same], p.read(p.CONFIG)['criteria'])
        self.assertFalse(result['checks']['attack_effective'])
        self.assertFalse(result['performance_pass'])
        self.assertIsNone(result['values']['excess_asr_reduction'])

    def test_missing_runs_stay_in_report_no_fabricated_performance(self):
        with tempfile.TemporaryDirectory() as temp:
            report = reporting.report(Path(temp), p.plan())
        self.assertEqual(len(report['missing']), 48)
        self.assertFalse(report['complete'])
        self.assertTrue(all(c['status'] == 'incomplete' for c in report['comparisons']))

    def test_health_and_tail_windows_require_full_150_rounds(self):
        t = task()
        rows = [dict(round=i, **self.metrics(), false_positive_removals=0, honest_clients=60,
                     accepted_updates=100, honest_weight_loss=0.) for i in range(151)]
        result = dict(task_id=t['task_id'], config=t['config'], records=rows, stopped_round=150,
                      cryptography='absent', official_test_used=False, nonfinite_updates=0)
        self.assertTrue(reporting.summarize(result)['healthy'])
        rows[-1]['false_positive_removals'] = 7
        self.assertFalse(reporting.summarize(result)['healthy'])
        rows.pop()
        self.assertIsNone(reporting.summarize(result)['tail10'])
        self.assertFalse(reporting.summarize(result)['complete'])


if __name__ == '__main__':
    unittest.main()
