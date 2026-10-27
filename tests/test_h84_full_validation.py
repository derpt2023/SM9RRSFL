from copy import deepcopy
import unittest

import run_cifar_h84_full_validation as full
import run_cifar_w85_history_admission as history
import run_cifar_cp1_cuda_development as base


def protocol():
    parent = base.read(history.DEFAULT)
    p = deepcopy(parent)
    p['arms'] = {a: dict(method=m, parameters=deepcopy(parent['arms']['H84']['parameters'])
                        if a == 'H84' else {}) for a, m in full.METHODS.items()}
    p.update(validation_design=deepcopy(full.DESIGN), reuse=full.reuse_map(),
             output_dir=full.OUTPUT, tad_reproduction=deepcopy(full.PAPER_CHOICES))
    p['tasks'] = [t for t in full.all_tasks() if t['id'] not in p['reuse']]
    return p, parent


def healthy_cells():
    h = dict(complete_observed_health_pass=True, v14_failure_reasons=[], additional_review_reasons=[])
    return {t['id']: dict(summary=dict(status='complete', complete_finite=True,
            tail=dict(accuracy=.59, source_transfer_asr=.10), health=deepcopy(h))) for t in full.all_tasks()}


def prerequisites():
    gates = {str(i): dict(passed=True) for i in range(8)}
    pairing = {str(i): dict(passed=True, required=True) for i in range(48)}
    return gates, pairing


class ProtocolTests(unittest.TestCase):
    def test_54_fresh_and_six_exact_references_cover_matrix(self):
        p, parent = protocol()
        full.validate(p, parent)
        self.assertEqual(len(p['tasks']), 54)
        self.assertEqual(len(full.all_tasks()), 60)
        self.assertFalse({t['id'] for t in p['tasks']} & set(p['reuse']))
        self.assertEqual({t['id'] for t in p['tasks']} | set(p['reuse']), {t['id'] for t in full.all_tasks()})
        specs = base.specs_for(dict(p, tasks=full.all_tasks()))
        for s in specs:
            if s['arm'] == 'H84':
                self.assertEqual(s['config']['detector_history_threshold'], .84)
                self.assertEqual(s['config']['detector_distance_threshold'], .85)
                self.assertEqual(s['config']['suspicion_remove_after'], 60)

    def test_candidate_attack_baseline_and_missing_task_changes_rejected(self):
        for mutate in (lambda p: p['arms']['H84']['parameters'].update(detector_history_threshold=.80),
                       lambda p: p['arms']['Krum']['parameters'].update(suspicion_remove_after=61),
                       lambda p: p['profile'].update(boost=14),
                       lambda p: p['selection_policy'].update(clean_asr_gap=.03),
                       lambda p: p['tasks'].pop(), lambda p: p['reuse'].pop('H84_iid_clean')):
            p, parent = protocol(); mutate(p)
            with self.assertRaises(ValueError):
                full.validate(p, parent)

    def test_gpu_uuid_may_differ_but_software_cannot(self):
        env = {k: k for k in ('python', 'torch', 'cuda', 'cudnn', 'gpu', 'numerics', 'native_binary_sha256')}
        self.assertEqual(full.environment_key(dict(env, gpu_uuid='a')), full.environment_key(dict(env, gpu_uuid='b')))
        self.assertNotEqual(full.environment_key(env), full.environment_key(dict(env, torch='other')))


class SelectionTests(unittest.TestCase):
    def test_pending_baseline_is_not_silently_excluded(self):
        cells = healthy_cells(); cells['Krum_iid_r80']['summary']['complete_finite'] = False
        with self.assertRaises(ValueError): full.assess(cells, *prerequisites())

    def test_health_exclusion_spans_all_five_ratios_and_preserves_raw(self):
        cells = healthy_cells()
        for label in full.RATIOS:
            cells[f'Krum_iid_{label}']['summary']['tail']['source_transfer_asr'] = 0.
        h = cells['Krum_iid_r80']['summary']['health']
        h.update(complete_observed_health_pass=False, additional_review_reasons=['empty_aggregation'])
        result = full.assess(cells, *prerequisites())
        self.assertTrue(result['development_criteria_passed'])
        for label in full.RATIOS:
            c = result['scenarios']['iid_' + label]
            self.assertNotIn('Krum', c['eligible_methods'])
            self.assertEqual(c['raw_asr_ranks']['Krum'], 1)
            self.assertEqual(c['asr_rank'], 1)
        self.assertTrue(result['method_health']['dirichlet']['Krum']['valid'])

    def test_ours_one_health_failure_removes_distribution_eligibility(self):
        cells = healthy_cells(); h = cells['H84_iid_r60']['summary']['health']
        h.update(complete_observed_health_pass=False, additional_review_reasons=['high_honest_false_revocation_rate_under_attack'])
        result = full.assess(cells, *prerequisites())
        self.assertFalse(result['development_criteria_passed'])
        for label in full.RATIOS:
            self.assertIsNone(result['scenarios']['iid_' + label]['accuracy_rank'])
            self.assertFalse(result['scenarios']['iid_' + label]['passed'])
        self.assertTrue(result['scenarios']['dirichlet_r60']['passed'])

    def test_clean_no_rank_requirement_but_one_and_two_pp_limits(self):
        metrics = {a: dict(accuracy=.60, asr=.10) for a in full.METHODS}
        metrics['H84'] = dict(accuracy=.59, asr=.12)
        result = full.rank_comparison(metrics, list(metrics), True, True)
        self.assertEqual(result['accuracy_rank'], 6)
        self.assertTrue(result['passed'])
        self.assertFalse(full.rank_comparison(metrics, list(metrics), True, False)['passed'])
        metrics['H84']['asr'] = .12001
        self.assertFalse(full.rank_comparison(metrics, list(metrics), True, True)['passed'])

    def test_clean_utility_and_attack_prerequisites_are_not_ignored(self):
        cells = healthy_cells(); cells['H84_iid_clean']['summary']['tail']['accuracy'] = .53
        self.assertFalse(full.assess(cells, *prerequisites())['method_health']['iid']['H84']['valid'])
        gates, pairing = prerequisites(); gates['0']['passed'] = False
        self.assertFalse(full.assess(healthy_cells(), gates, pairing)['development_criteria_passed'])
        gates, pairing = prerequisites(); pairing['0']['passed'] = False
        self.assertFalse(full.assess(healthy_cells(), gates, pairing)['development_criteria_passed'])
        pairing['0']['required'] = False
        self.assertTrue(full.assess(healthy_cells(), gates, pairing)['development_criteria_passed'])


class TadAuditTests(unittest.TestCase):
    def rows(self):
        return [dict(round=0), dict(round=1, blacklisted_client_ids=[], tad_no_prior=dict(
            version='TAD-no-prior-v1', round=1, scores={}, outliers=[], observations={},
            abnormal_counts={}, newly_removed=[], removed=[], weights={'client-0': 1.}))]

    def test_requires_actual_no_prior_observations(self):
        self.assertEqual(full.tad_audit(self.rows())['rounds_verified'], 1)
        rows = self.rows(); rows[1].pop('tad_no_prior')
        with self.assertRaises(ValueError): full.tad_audit(rows)

    def test_score_quota_or_blacklist_tampering_rejected(self):
        rows = self.rows(); rows[1]['tad_no_prior']['scores'] = {'client-0': .8}
        with self.assertRaises(ValueError): full.tad_audit(rows)
        rows = self.rows(); rows[1]['blacklisted_client_ids'] = ['client-0']
        with self.assertRaises(ValueError): full.tad_audit(rows)


if __name__ == '__main__':
    unittest.main()
