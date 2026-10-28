from copy import deepcopy
import math
import unittest

import cifar_h84_formal_report as report
import render_cifar_h84_formal as render


class FormalRenderStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.cells = {}
        for seed, score in zip(render.SEEDS, (.1, .2, .5)):
            self.cells[f'H84_iid_clean_s{seed}'] = dict(status='complete', complete_finite=True,
                config=dict(malicious_ratio=0), tail=dict(accuracy=score, source_transfer_asr=.1),
                final_fp=20 if score == .5 else 0, health=dict(complete_observed_health_pass=score != .5),
                timing=dict(runtime_with_crypto_seconds=10, runtime_without_crypto_seconds=8, crypto_wall_seconds=2))
        self.groups = report.grouped_statistics(self.cells, render.SEEDS)

    def test_health_failure_is_retained_as_an_observed_seed(self):
        render.validate_groups(self.cells, self.groups)
        self.assertEqual(self.groups['H84_iid_clean']['actual_n'], 3)
        self.assertEqual(self.groups['H84_iid_clean']['samples']['accuracy']['values'], [.1, .2, .5])
        pruned = report.grouped_statistics({k:v for k,v in self.cells.items() if v['health']['complete_observed_health_pass']}, render.SEEDS)
        with self.assertRaises(ValueError): render.validate_groups(self.cells, pruned)

    def test_only_sd_rounding_is_tolerated_not_raw_samples_or_means(self):
        perturbed = deepcopy(self.groups)
        sample = perturbed['H84_iid_clean']['samples']['accuracy']
        sample['sample_sd'] = math.nextafter(sample['sample_sd'], math.inf)
        render.validate_groups(self.cells, perturbed)
        for field in ('mean', 'sample_sd'):
            changed = deepcopy(self.groups)
            changed['H84_iid_clean']['samples']['accuracy'][field] += 1e-8
            with self.assertRaises(ValueError): render.validate_groups(self.cells, changed)
        changed = deepcopy(self.groups)
        changed['H84_iid_clean']['samples']['accuracy']['values'][0] += 1e-8
        with self.assertRaises(ValueError): render.validate_groups(self.cells, changed)

    def test_missing_groups_or_zero_filled_n_are_rejected(self):
        missing = deepcopy(self.groups); missing.pop('H84_iid_clean')
        with self.assertRaises(ValueError): render.validate_groups(self.cells, missing)
        wrong_n = deepcopy(self.groups); wrong_n['H84_iid_clean']['actual_n'] = 2
        with self.assertRaises(ValueError): render.validate_groups(self.cells, wrong_n)


if __name__ == '__main__': unittest.main()
