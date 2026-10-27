from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import run_cifar_cp1_cuda_development as base
import run_cifar_w85_history_admission as history
from audit_cifar_cuda_panel import audit_calls


def protocol():
    parent = base.read(base.DEFAULT)
    p = deepcopy(parent)
    p['history_admission_design'] = deepcopy(history.DESIGN)
    p['arms'] = {name: dict(method='sm9rrs', parameters=dict(
        parent['arms']['W85']['parameters'], detector_history_threshold=value))
        for name, value in history.HISTORY.items()}
    p['tasks'] = [dict(id=f'{a}_iid_{s}', arm=a, partition='iid', ratio=r)
                  for a in history.HISTORY for s, r in [('clean', 0.), ('r60', .6), ('r80', .8)]]
    p['output_dir'] = 'outputs/cifar_w85_history_admission150_v1'
    return p, parent


class DesignTests(unittest.TestCase):
    def test_only_history_threshold_differs_from_w85(self):
        p, parent = protocol()
        history.validate_protocol(p, parent)
        reference = dict(parent['base_configs']['sm9rrs'], **parent['arms']['W85']['parameters'], device='cuda:0')
        for spec in base.specs_for(p):
            expected = dict(reference, malicious_ratio=spec['config']['malicious_ratio'],
                            detector_history_threshold=history.HISTORY[spec['arm']])
            self.assertEqual(spec['config'], expected)
            self.assertEqual(spec['profile'], parent['profile'])

    def test_unrequested_detector_and_public_changes_rejected(self):
        for key, value in [('detector_distance_threshold', .9), ('detector_drift_allowance', .8),
                           ('detector_history_confirm', 2), ('suspicion_remove_after', 61)]:
            p, parent = protocol()
            p['arms']['H80']['parameters'][key] = value
            with self.assertRaises(ValueError):
                history.validate_protocol(p, parent)
        for group, key, value in [('profile', 'boost', 14), ('selection_policy', 'attack_asr_gap', .06),
                                  ('numerics', 'cuda_matmul_allow_tf32', True)]:
            p, parent = protocol()
            p[group][key] = value
            with self.assertRaises(ValueError):
                history.validate_protocol(p, parent)

    def test_threshold_order_and_complete_scenario_coverage(self):
        p, parent = protocol()
        p['arms']['H84']['parameters']['detector_history_threshold'] = .85
        with self.assertRaises(ValueError):
            history.validate_protocol(p, parent)
        p, parent = protocol()
        p['tasks'].pop()
        with self.assertRaises(ValueError):
            history.validate_protocol(p, parent)

    def test_prepare_keeps_manifest_and_reference_lock_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'out'
            history.prepare(out, {'fingerprint': 'f'}, {'references': {'old': 'sha'}})
            history.prepare(out, {'fingerprint': 'f'}, {'references': {'old': 'sha'}})
            with self.assertRaises(ValueError):
                history.prepare(out, {'fingerprint': 'f'}, {'references': {'old': 'changed'}})
            other = Path(directory) / 'partial'
            other.mkdir()
            (other / 'unknown').write_text('retain')
            with self.assertRaises(ValueError):
                history.prepare(other, {}, {})

    def test_real_config_and_source_lock(self):
        p, specs, manifest = history.plan(history.DEFAULT)
        self.assertEqual(len(specs), 6)
        self.assertEqual(manifest['maximum_rounds'], 900)
        self.assertEqual(p['selection_policy'], base.read(base.DEFAULT)['selection_policy'])


def complete_cell(acc=.58, asr=.10, fp=0, healthy=True):
    return dict(status='complete', tail=dict(accuracy=acc, source_transfer_asr=asr),
                final_fp=fp, health=dict(complete_observed_health_pass=healthy))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.ref = dict(cells={'W85_iid_' + suffix: dict(summary=complete_cell(fp=10 if suffix == 'clean' else 1))
                               for suffix in ('clean', 'r60', 'r80')})
        self.ref['cells']['FedAvg_iid_clean'] = dict(summary=complete_cell(acc=.59))
        self.cells = {arm + '_iid_' + suffix: complete_cell(fp=2 if arm == 'H80' else 3)
                      for arm in history.HISTORY for suffix in ('clean', 'r60', 'r80')}

    def test_clean_improvement_and_fixed_tie_break_choose_candidate(self):
        result = history.assess(self.cells, self.ref, True)
        self.assertEqual(result['selected_for_full_validation'], 'H80')
        self.assertFalse(result['main_qualified'])

    def test_no_fp_improvement_health_failure_or_attack_regression_rejected(self):
        for change in ('fp', 'health', 'accuracy', 'asr'):
            with self.subTest(change=change):
                cells = deepcopy(self.cells)
                for arm in history.HISTORY:
                    if change == 'fp': cells[arm + '_iid_clean']['final_fp'] = 10
                    if change == 'health': cells[arm + '_iid_clean']['health']['complete_observed_health_pass'] = False
                    if change == 'accuracy': cells[arm + '_iid_r80']['tail']['accuracy'] = .559
                    if change == 'asr': cells[arm + '_iid_r60']['tail']['source_transfer_asr'] = .151
                self.assertEqual(history.assess(cells, self.ref, True)['status'], 'bounded_history_panel_failed')

    def test_missing_and_unaudited_evidence_never_qualifies(self):
        for audited in (False, True):
            cells = deepcopy(self.cells)
            for arm in history.HISTORY:
                cells[arm + '_iid_r80'] = dict(status='running')
            self.assertIsNone(history.assess(cells, self.ref, audited)['selected_for_full_validation'])
        self.assertIsNone(history.assess(self.cells, self.ref, False)['selected_for_full_validation'])


class ActualAttackTests(unittest.TestCase):
    def setUp(self):
        p = base.read(base.DEFAULT)
        self.spec = next(s for s in base.specs_for(p) if s['task_id'] == 'W85_iid_r60')
        cfg, profile = self.spec['config'], self.spec['profile']
        self.targets = dict(training_target_indices=list(range(128)), source_transfer_count=750,
                            official_test_used_for_selection=False, training_target_is_optimizer_input=True,
                            transfer_target_is_optimizer_input=False)
        self.rows = [dict(blacklisted_client_ids=[] if r < 25 else ['client-1']) for r in range(151)]
        from cifar_alternating_target_views import VIEW_SALT
        seed = cfg['seed'] + 25 * 1009 + 1
        rng = np.random.default_rng([seed, VIEW_SALT])
        y, x, flips = rng.integers(0, 9, 128), rng.integers(0, 9, 128), rng.random(128) < .5
        draws = dict(y=y.tolist(), x=x.tolist(), flip=flips.tolist())
        view = dict(protocol=profile['target_transform'], local_call_seed=seed, salt=VIEW_SALT,
                    count=128, pad=4, padding_mode='reflect', flipped=int(flips.sum()),
                    original_auxiliary_indices_sha256=hashlib.sha256(np.arange(128, dtype='<i8').tobytes()).hexdigest(),
                    draws_sha256=hashlib.sha256(json.dumps(draws, sort_keys=True).encode()).hexdigest())
        self.record = dict(task_id=self.spec['task_id'], round=25, client_idx=1, client_id='client-1',
                           source_client='client-1', status='complete', checks_enabled=True, first_failure=None,
                           normal_steps=8, reference_steps=8, profile_id='V2', normal_lr=cfg['lr'] * cfg['lr_decay'] ** 24,
                           target_views=view, **{k: profile[k] for k in ('optimizer', 'normal_optimizer', 'lr', 'boost', 'stealth_steps', 'scope')})

    def check(self, records):
        return audit_calls(records, self.spec, self.targets, ['client-1'], self.rows)

    def test_real_view_seed_and_draws_validate_with_exact_active_coverage(self):
        self.assertEqual(self.check([self.record])['actual_calls'], 1)

    def test_missing_duplicate_wrong_profile_or_view_rejected(self):
        for records in ([], [self.record, self.record], [dict(self.record, optimizer='adam')],
                        [dict(self.record, target_views=dict(self.record['target_views'], draws_sha256='wrong'))],
                        [dict(self.record, normal_lr=.01)]):
            with self.assertRaises(ValueError):
                self.check(records)

    def test_evaluation_images_cannot_be_optimizer_inputs(self):
        self.targets['transfer_target_is_optimizer_input'] = True
        with self.assertRaises(ValueError):
            self.check([self.record])


if __name__ == '__main__':
    unittest.main()
