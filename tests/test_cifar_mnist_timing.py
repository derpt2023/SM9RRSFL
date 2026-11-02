"""Timing boundaries and fresh-only scheduling, without scientific training."""
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import reproduce as base
import run_cifar_timing as t
from sm9rrsfl import experiments as e
from sm9rrsfl.fl import ExperimentConfig, ExperimentResult, StageTimings


def fake_result(**kwargs):
    return ExperimentResult(ExperimentConfig(), [], .5, .5, 150, (), (), **kwargs)


class ProtocolTests(unittest.TestCase):
    def test_exact_fixed_matrix_and_balanced_order(self):
        p = t.protocol()
        blocks = t.blocks(p)
        self.assertEqual(len(blocks), 30)
        original = {x['task_id']: x for x in base.specs(p, t.METHODS)}
        current = {x['task_id']: x for b in blocks for x in b['tasks']}
        self.assertEqual(current, original)
        self.assertEqual(len(current), 180)
        for b in blocks:
            self.assertEqual(len({(x['config']['seed'], x['config']['partition'],
                x['config']['malicious_ratio']) for x in b['tasks']}), 1)
        for position in range(6):
            self.assertEqual(Counter(b['tasks'][position]['config']['method'] for b in blocks),
                             {method: 5 for method in t.METHODS})

    def test_original_126_sources_and_manifest_tampering(self):
        m = t.manifest(Path('/tmp/data'))
        self.assertEqual(len(m['protocol']['scientific_sources']), 126)
        t.verify_manifest(m)
        m['blocks'][0]['tasks'][0]['config']['rounds'] = 2
        with self.assertRaises(ValueError):
            t.verify_manifest(m)

    def test_common_runner_excludes_checkpoint_and_progress_io(self):
        # Deterministic synthetic clock: no CUDA/model training. Real common runner.
        clock = [0.0]
        def checkpoint(*args, **kwargs):
            clock[0] += 20
        def progress(*args, **kwargs):
            clock[0] += 7
        def run(dataset, config, **kwargs):
            clock[0] += 10
            kwargs['checkpoint_callback']({'completed_round': 0})
            clock[0] += 30
            return fake_result()
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(e, 'perf_counter', side_effect=lambda: clock[0]), \
                patch.object(e, '_load_round_checkpoint', return_value=(None, 0., 0.)), \
                patch.object(e, '_effective_checkpoint_interval', return_value=1), \
                patch.object(e, '_write_round_checkpoint', side_effect=checkpoint), \
                patch.object(e, '_peak_rss_mb', return_value=1), \
                patch.object(e, 'finalize_config_checkpoint'), \
                patch.object(e, 'run_experiment', side_effect=run), \
                patch.object(t, 'scientific_scopes', return_value=nullcontext()), \
                patch.object(base, 'write', side_effect=progress):
            task = t.blocks(t.protocol())[0]['tasks'][0]
            result = t.measured(object(), task, Path(temp), 'test')
        self.assertEqual(result.runtime_seconds, 40)
        self.assertEqual(result.checkpoint_io_seconds, 27)
        self.assertEqual(clock[0], 67)

    def test_progress_wrapper_restored_after_failure(self):
        original = lambda *a, **k: None
        module = SimpleNamespace(_write_round_checkpoint=original)
        with self.assertRaises(RuntimeError):
            with t.checkpoint_progress(Path('/tmp/no-write'), module):
                self.assertIsNot(module._write_round_checkpoint, original)
                raise RuntimeError('test')
        self.assertIs(module._write_round_checkpoint, original)

    def test_uses_wall_crypto_not_parallel_operation_sums(self):
        stages = StageTimings(crypto_setup_wall_seconds=1, crypto_packet_wall_seconds=2,
            crypto_audit_wall_seconds=3, crypto_finalize_wall_seconds=4,
            sign_seconds=1000, verify_seconds=1000)
        record = t.timing_record(fake_result(runtime_seconds=40, checkpoint_io_seconds=27,
                                           stage_timings=stages))
        self.assertEqual(record['runtime_without_crypto_seconds'], 30)
        self.assertEqual(record['checkpoint_io_seconds'], 27)
        self.assertEqual(record['definition'], t.DEFINITION)
        with self.assertRaises(ValueError):
            t.timing_record(fake_result(runtime_seconds=5, stage_timings=stages))
        with self.assertRaises(ValueError):
            t.timing_record(fake_result(runtime_seconds=float('nan')))

    def test_scientific_adapters_restore_and_vert_has_no_forensic_file(self):
        from sm9rrsfl import fl, torch_backend
        from scripts.experiments.cifar10 import attack_stability_attack as a
        original = (fl.VERTDefense, a.run_profile_attack,
                    torch_backend.TorchTrainingContext.alternating_minimization_delta_resident)
        for task in t.blocks(t.protocol())[0]['tasks']:
            with t.scientific_scopes(task):
                self.assertIsNot(a.run_profile_attack, original[1])
                if task['config']['method'] == 'vert':
                    self.assertIsNone(fl.VERTDefense.keywords['audit_path'])
            self.assertEqual((fl.VERTDefense, a.run_profile_attack,
                torch_backend.TorchTrainingContext.alternating_minimization_delta_resident), original)

    def test_admission_requires_both_samples_and_no_quarantine(self):
        def snap(free, uuid='gpu'):
            return {'gpus': [{'uuid': uuid, 'free': free}]}
        self.assertTrue(t.admitted('gpu', snap(8192), snap(8192)))
        for a,b in [(8191,10000),(10000,8191)]:
            self.assertFalse(t.admitted('gpu', snap(a), snap(b)))
        bad = next(iter(base.QUARANTINED))
        self.assertFalse(t.admitted(bad, snap(50000,bad), snap(50000,bad)))

    def test_existing_task_rejected_before_cuda_or_data_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            m = t.manifest(Path('/tmp/data'))
            base.write(out / 'manifest.json', m)
            task = m['blocks'][0]['tasks'][0]
            (out / 'tasks' / task['task_id']).mkdir(parents=True)
            with patch.object(base, 'runtime') as runtime, self.assertRaises(FileExistsError):
                t.worker(out, task['task_id'])
            runtime.assert_not_called()

    def test_controller_same_gpu_serial_block_and_fail_stop(self):
        # Exercise the actual dispatch loop with simulated children, not CUDA.
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                out = Path(temp) / 'fresh'
                m = t.manifest(Path('/tmp/data'))
                m['blocks'] = m['blocks'][:1]
                calls = []
                sample = dict(gpus=[dict(index='2', uuid='gpu', free=9000, utilization=0)])
                def child(command, **kwargs):
                    task_id = command[-1]
                    calls.append((task_id, kwargs['env']['CUDA_VISIBLE_DEVICES']))
                    base.write(out / 'tasks' / task_id / 'status.json',
                               {'status': 'failed' if fail else 'complete'})
                    code = 1 if fail else 0
                    return SimpleNamespace(pid=100 + len(calls), returncode=code,
                                           poll=lambda: code, wait=lambda: code)
                with patch.object(t, 'manifest', return_value=m), \
                        patch.object(t, 'verify_manifest'), \
                        patch.object(t, 'gpu_snapshot', return_value=sample), \
                        patch.object(t, 'acquire_gpu', side_effect=lambda uuid: open(Path(temp) / 'lock', 'a')), \
                        patch.object(t.signal, 'signal'), patch.object(t.time, 'sleep'), \
                        patch.object(t.subprocess, 'Popen', side_effect=child), \
                        patch.object(t, 'final_report', return_value=dict(counts={}, execution_audit={'passed':not fail})):
                    t.batch(out, Path('/tmp/data'), None, 6)
                self.assertEqual(len(calls), 1 if fail else 6)
                self.assertEqual([name for name,gpu in calls],
                                 [x['task_id'] for x in m['blocks'][0]['tasks']][:len(calls)])
                self.assertEqual({gpu for name,gpu in calls}, {'gpu'})
                status = base.read(out / 'controller_status.json')
                self.assertEqual(status['status'], 'incomplete' if fail else 'complete')

    def test_final_audit_preserves_health_failures_and_rejects_duplicate_start(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            m = t.manifest(Path('/tmp/data'))
            base.write(out / 'manifest.json', m)
            for b in m['blocks']:
                for task in b['tasks']:
                    folder = out / 'tasks' / task['task_id']
                    base.write(folder / 'result.json', dict(runtime_seconds=10,
                        runtime_without_crypto_seconds=8, crypto_wall_seconds=2, checkpoint_io_seconds=1))
                    base.write(folder / 'status.json', dict(status='complete',
                        health_reasons=['retained_original_health_failure'],
                        artifacts={'result.json': base.sha(folder / 'result.json')}))
                    sample = {'gpus': [{'uuid':'gpu','free':9000}], 'at':'2026-11-02T00:00:00+00:00'}
                    second = dict(sample, at='2026-11-02T00:00:05+00:00')
                    t.event(out, 'start', uuid='gpu', task_id=task['task_id'],
                            block_id=b['block_id'], admission=[sample,second])
                    t.event(out, 'exit', uuid='gpu', task_id=task['task_id'], returncode=0)
            report = t.final_report(out)
            self.assertTrue(report['execution_audit']['passed'])
            self.assertEqual(len(report['health_failures']), 180)
            self.assertEqual(len(report['aggregate']), 60)
            t.event(out, 'start', uuid='gpu', task_id=task['task_id'],
                    block_id=b['block_id'], admission=[sample,second])
            self.assertFalse(t.final_report(out)['execution_audit']['passed'])


if __name__ == '__main__':
    unittest.main()
