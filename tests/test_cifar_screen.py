import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest import TestCase, mock

import numpy as np

from run_attack_screen import load_screen_data, split_screen_data, summarize
from screen_performance import screen_performance_target
from sm9rrsfl.datasets import make_synthetic_mnist_like, stratified_training_three_way_split
from sm9rrsfl.fl import ExperimentConfig, ExperimentResult, RoundRecord


class ScreenDatasetTest(TestCase):
    def test_legacy_mnist_loader_defaults_are_preserved(self):
        spec = dict(train_samples=1000, split_seed=9)
        with mock.patch('run_attack_screen.load_image_dataset') as loader:
            load_screen_data(spec)
        loader.assert_called_once_with('mnist', 'data/mnist', download=False,
                                       train_limit=1000, test_limit=100, seed=9)

    def test_cifar_requires_explicit_training_settings_before_loading(self):
        spec = dict(dataset='cifar10', train_samples=1000, split_seed=9, training={})
        with mock.patch('run_attack_screen.load_image_dataset') as loader:
            with self.assertRaisesRegex(ValueError, 'explicit training settings'):
                load_screen_data(spec)
        loader.assert_not_called()
        spec['training'] = dict(rounds=100, local_epochs=1, batch_size=50, lr=.05,
            lr_decay=.99, attack_epochs=1, attack_start_round=25, detector_window=20)
        with mock.patch('run_attack_screen.load_image_dataset') as loader:
            load_screen_data(spec)
        self.assertEqual(loader.call_args.args[:2], ('cifar10', 'data/cifar10'))
        spec['candidates'] = {'sm9rrs': {'invalid': {'detector_distance_threshold': 1.25}}}
        with mock.patch('run_attack_screen.load_image_dataset') as loader:
            with self.assertRaisesRegex(ValueError, 'history threshold'):
                load_screen_data(spec)
        loader.assert_not_called()

    def test_default_split_is_identical_and_custom_holdout_excludes_official_test(self):
        data = make_synthetic_mnist_like(train_samples=1000, test_samples=30, seed=9)
        original = stratified_training_three_way_split(data, seed=9,
            train_fraction=.90, calibration_fraction=.05, attack_fraction=.05)
        legacy = split_screen_data(data, dict(split_seed=9))
        for name in ('train_indices', 'calibration_indices', 'attack_indices'):
            np.testing.assert_array_equal(getattr(original, name), getattr(legacy, name))
        larger = split_screen_data(data, dict(split_seed=9, validation_fraction=.20))
        self.assertGreater(len(larger.calibration_indices), len(legacy.calibration_indices))
        self.assertFalse(set(larger.train_indices) & set(larger.calibration_indices))
        self.assertFalse(set(larger.attack_indices) & set(larger.calibration_indices))
        np.testing.assert_array_equal(larger.calibration_dataset.x_test,
                                      data.x_train[larger.calibration_indices])
        self.assertIs(larger.main_dataset.x_test, data.x_test)


class ScreenTargetTest(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.spec = dict(training=dict(num_clients=20), partitions=['iid'], ratios=[.4],
            screen_seeds=[1], confirmation_seeds=[2], attacks={'boost15': {}, 'boost7': {}},
            candidates={'sm9rrs': {'bad': {}, 'good': {}}, 'vert': {'frozen': {}, 'weak': {}}},
            performance_target=dict(accuracy_gap=.005, asr_gap=.01, max_asr=.1, max_peak_asr=.3))
        self.selected = {'sm9rrs': 'bad', 'vert': 'frozen'}

    def save(self, phase, method, name, profile, accuracy=.8, asr=0.):
        ratio = 0. if profile == 'clean' else .4
        seed = 1 if phase == 'screen' else 2
        config = ExperimentConfig(method=method, seed=seed, num_clients=20, rounds=20,
            detector_window=3, attack_start_round=5, malicious_ratio=ratio,
            attack_boost=7. if profile == 'boost7' else 15.)
        records = [RoundRecord(method, ratio, i, accuracy, 1-accuracy, 20, 0, 0, 0, 0, '',
            attack_target_success_rate=asr) for i in range(1, 21)]
        result = ExperimentResult(config, records, accuracy, 1-accuracy, 20, (), ())
        row = dict(phase=phase, method=method, candidate=name, profile=profile,
                   partition='iid', ratio=ratio, seed=seed, **summarize(result))
        path = self.directory / f'{phase}-{method}-{name}-{profile}.json'
        path.write_text(json.dumps(dict(config=asdict(config), summary=row,
                                       rounds=[asdict(r) for r in records])))
        return path

    def populate(self, phase='screen'):
        self.save(phase, 'fedavg', 'reference', 'clean')
        for method, name in [('sm9rrs', 'bad'), ('sm9rrs', 'good'), ('vert', 'frozen')]:
            for profile in ['clean', *self.spec['attacks']]:
                self.save(phase, method, name, profile,
                    asr=.7 if method == 'sm9rrs' and name == 'bad' and profile != 'clean' else 0.)

    def test_selects_protected_candidate_across_profiles_and_keeps_vert_frozen(self):
        self.populate()
        chosen, report = screen_performance_target(self.directory, self.spec, 'screen',
                                                   self.selected, choose=True)
        self.assertEqual(chosen, {'sm9rrs': 'good', 'vert': 'frozen'})
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(set(report['candidates']['good']['profiles']), {'boost15', 'boost7'})
        self.assertEqual(self.selected['sm9rrs'], 'bad')

    def test_missing_profile_and_high_absolute_asr_cannot_pass(self):
        self.populate()
        self.save('screen', 'sm9rrs', 'good', 'boost7', asr=.2)
        _, report = screen_performance_target(self.directory, self.spec, 'screen', self.selected, choose=True)
        self.assertEqual(report['status'], 'unmet')
        (self.directory / 'screen-sm9rrs-good-boost7.json').unlink()
        _, report = screen_performance_target(self.directory, self.spec, 'screen', self.selected, choose=True)
        self.assertFalse(report['candidates']['good']['profiles']['boost7']['structurally_complete'])
        self.assertEqual(report['status'], 'unmet')

    def test_confirmation_never_reselects_and_clean_reference_drop_invalidates(self):
        for method, name in [('sm9rrs', 'bad'), ('vert', 'frozen')]:
            for profile in ['clean', *self.spec['attacks']]:
                self.save('confirmation', method, name, profile)
        self.save('confirmation', 'fedavg', 'reference', 'clean', accuracy=.9)
        chosen, report = screen_performance_target(self.directory, self.spec, 'confirmation', self.selected)
        self.assertEqual(chosen, self.selected)
        self.assertEqual(report['status'], 'unmet')
        self.assertIn('clean_accuracy_drop', report['candidates']['bad']['health_failures'])
        self.assertIn('clean_accuracy_drop', report['vert_health_failures'])
        self.assertFalse(report['parameters_reselected'])
        with self.assertRaisesRegex(ValueError, 'never reselect'):
            screen_performance_target(self.directory, self.spec, 'confirmation', self.selected, choose=True)

    def test_extra_confirmation_candidate_and_missing_reference_cannot_pass(self):
        self.populate('confirmation')
        (self.directory / 'confirmation-fedavg-reference-clean.json').unlink()
        _, report = screen_performance_target(self.directory, self.spec, 'confirmation', self.selected)
        health = report['candidates']['bad']['health_failures']
        self.assertIn('unfrozen_candidate', health)
        self.assertIn('missing_or_duplicate_clean_reference', health)
        self.assertEqual(report['status'], 'unmet')

    def test_no_policy_does_not_apply_new_selection(self):
        chosen, report = screen_performance_target(self.directory, {}, 'screen', self.selected, choose=True)
        self.assertEqual(chosen, self.selected)
        self.assertEqual(report, {'status': 'disabled'})
