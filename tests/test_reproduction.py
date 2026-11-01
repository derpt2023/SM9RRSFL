"""Protocol fidelity and safe fresh-checkout entrypoints; no model training."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import reproduce as r
from sm9rrsfl.fl import ExperimentConfig, ExperimentResult, RoundRecord


class ProtocolTests(unittest.TestCase):
    def test_complete_fixed_matrices(self):
        for name in r.NAMES:
            p = r.read(r.CONFIGS / (name + '.json'))
            tasks = r.specs(p, sorted(p['methods']))
            self.assertEqual(len(tasks), 180)
            self.assertEqual(len({t['task_id'] for t in tasks}), 180)
            for task in tasks:
                self.assertEqual(asdict(ExperimentConfig(**task['config'])), task['config'])
                self.assertEqual(task['config']['rounds'], p['rounds'])

    def test_invalid_and_duplicate_methods(self):
        p = r.read(r.CONFIGS / 'cifar-best.json')
        for methods in [['Ours'], ['sm9rrs', 'sm9rrs']]:
            with self.assertRaises(ValueError):
                r.specs(p, methods)
        self.assertEqual(len(r.specs(p, ['sm9rrs'])), 30)

    def test_official_reporting_never_relabels_auxiliary_asr_without_replacing_values(self):
        record = RoundRecord('sm9rrs', .2, 0, .5, .5, 100, 0, 0, 0, 0, '',
                             attack_target_success_rate=.9, attack_target_confidence=.8)
        result = ExperimentResult(ExperimentConfig(), [record], .5, .5, 0, (), ())
        rows = [dict(round=0, accuracy=.6, error=.4, source_transfer_asr=.1,
                     source_transfer_confidence=.2)]
        official = r.official_result(result, rows)
        self.assertEqual(official.final_accuracy, .6)
        self.assertEqual(official.records[0].attack_target_success_rate, .1)
        self.assertEqual(official.records[0].attack_target_confidence, .2)
        self.assertEqual(result.records[0].attack_target_success_rate, .9)
        with self.assertRaises(ValueError):
            r.official_result(result, [])
        with self.assertRaises(ValueError):
            r.official_result(result, rows + rows)

    def test_cifar_matches_all_180_original_tasks(self):
        base = r.ROOT / 'outputs/cifar-best'
        if not base.exists():
            self.skipTest('Optional historical result package not present')
        p = r.read(r.CONFIGS / 'cifar-best.json')
        actual = {t['task_id']: t for t in r.specs(p, sorted(p['methods']))}
        for source in r.read(base / 'provenance/task_sources.json').values():
            original = r.read(base / source['raw_folder'] / 'task.json')
            task = actual[source['source_task']]
            self.assertEqual(task['config'], dict(original['config'], device='cuda:0'))
            self.assertEqual(task['profile'], original['profile'])

    def test_mnist_matches_execution_manifest_not_preallocation_summary(self):
        base = r.ROOT / 'outputs/mnist_v7_target_fair_tuning/.tuning_state/final'
        if not base.exists():
            self.skipTest('Optional historical result package not present')
        old = r.read(next(base.glob('*/run_manifest.json')))
        p = r.read(r.CONFIGS / 'mnist_v7_target_fair_tuning.json')
        actual = {json.dumps(t['config'], sort_keys=True) for t in r.specs(p, sorted(p['methods']))}
        expected = {json.dumps(dict(t['config'], device='cuda:0'), sort_keys=True) for t in old['candidates']}
        self.assertEqual(actual, expected)

    def test_existing_results_are_protected(self):
        for path in ['.', 'outputs', 'outputs/old', 'outputs/cifar-best',
                     'outputs/cifar-best/anything', 'outputs/mnist_v7_target_fair_tuning/new']:
            with self.assertRaises(ValueError):
                r.safe_output(r.ROOT / path)

    def test_reproduction_identity_changes_with_data_location_or_parameters(self):
        p = r.read(r.CONFIGS / 'cifar-best.json')
        tasks = r.specs(p, ['sm9rrs'])
        first = r.make_manifest(p, tasks, Path('/tmp/data-a'))
        second = r.make_manifest(p, tasks, Path('/tmp/data-b'))
        self.assertNotEqual(first['fingerprint'], second['fingerprint'])
        changed = copy.deepcopy(tasks)
        changed[0]['config']['suspicion_recovery_factor'] = 2
        self.assertNotEqual(first['fingerprint'], r.make_manifest(p, changed, Path('/tmp/data-a'))['fingerprint'])

    def test_plan_works_without_any_historical_output_or_absolute_station_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = r.read(r.CONFIGS / 'source_files.json')
            for name in set(files) | {'configs/reproduction/source_files.json'}:
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(r.ROOT / name, dest)
            for name in r.NAMES:
                process = subprocess.run([sys.executable, str(root / 'reproduce.py'), name],
                                         cwd=root, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                result = json.loads(process.stdout)
                self.assertEqual(result['tasks'], 180)
                self.assertFalse(result['training_started'])
            self.assertFalse((root / 'outputs').exists())

    def test_fresh_cifar_adapter_preserves_method_and_attack_scopes(self):
        from sm9rrsfl import fl
        from scripts.experiments.cifar10 import attack_stability_worker as worker
        from scripts.experiments.cifar10 import attack_stability_attack as attack
        from cifar_vert_paper_core import PaperVERT
        from cifar_alternating_target_views import run_profile_attack
        import watch_cifar_experiment as watch
        p = r.read(r.CONFIGS / 'cifar-best.json')
        original_vert, original_attack = fl.VERTDefense, attack.run_profile_attack
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1),
            set_num_threads=lambda n: None, use_deterministic_algorithms=lambda b: None,
            backends=SimpleNamespace(cudnn=SimpleNamespace(),
                cuda=SimpleNamespace(matmul=SimpleNamespace())))
        for method in ['sm9rrs', 'vert', 'ding13']:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                task = r.specs(p, [method])[0]
                m = dict(specs=[task], data_dir='/no-data-read-by-this-test', sources={}, fingerprint='test')
                r.write(out / 'manifest.json', m)
                summary = dict(runtime_seconds=10., crypto_wall_seconds=2., runtime_without_crypto_seconds=8.)
                summary.update(dict.fromkeys(watch.CRYPTO_SPANS, .5))
                def execute(request):
                    self.assertEqual(request['config'], task['config'])
                    self.assertEqual(request['profile'], task['profile'])
                    self.assertIs(attack.run_profile_attack, run_profile_attack)
                    if method == 'vert':
                        self.assertIs(fl.VERTDefense.func, PaperVERT)
                    r.write(out / 'tasks' / task['task_id'] / 'result.json', dict(status='complete'))
                    return dict(status='complete', summary=summary)
                with patch.dict(sys.modules, {'torch': fake_torch}), \
                     patch.object(r, 'runtime', return_value={}), \
                     patch.object(r, 'make_manifest', return_value=m), \
                     patch.object(r, 'verify_sources', return_value={}), \
                     patch.object(worker, 'execute_task', side_effect=execute) as trained:
                    r.run_task(SimpleNamespace(allow_environment_difference=True), p, task, out)
                    self.assertEqual(trained.call_count, 1)
                    # A second invocation is refused before the worker or environment is touched.
                    with self.assertRaisesRegex(ValueError, 'Existing task'):
                        r.run_task(SimpleNamespace(allow_environment_difference=True), p, task, out)
                    self.assertEqual(trained.call_count, 1)
                self.assertIs(fl.VERTDefense, original_vert)
                self.assertIs(attack.run_profile_attack, original_attack)


if __name__ == '__main__':
    unittest.main()
