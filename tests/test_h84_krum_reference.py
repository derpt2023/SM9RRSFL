from copy import deepcopy
import unittest

import assess_h84_krum_reference as reader
import run_cifar_h84_full_validation as full
from tests.test_h84_full_validation import healthy_cells, prerequisites


def fixture():
    cells = healthy_cells()
    for d in full.PARTITIONS:
        cells[f'Krum_{d}_clean']['summary']['tail']['accuracy'] = .43
    gates, pairing = prerequisites()
    report = dict(status='complete_audited_development', cells=cells, attack_gates=gates,
                  pre_attack_pairing=pairing, assessment=full.assess(cells, gates, pairing))
    amendment = dict(exception=deepcopy(reader.EXCEPTION), post_result_user_revision=True,
        mechanism_assessment='supported_observed_selection_concentration_and_local_overfit')
    return report, amendment


class AmendmentTests(unittest.TestCase):
    def test_only_iid_krum_utility_waived_original_unchanged(self):
        report, amendment = fixture(); before = deepcopy(report)
        result = reader.apply_amendment(report, amendment)
        self.assertEqual(report, before)
        self.assertTrue(result['reference_eligibility']['iid']['Krum']['valid'])
        self.assertFalse(result['reference_eligibility']['dirichlet']['Krum']['valid'])
        self.assertFalse(result['original_strict_assessment']['method_health']['iid']['Krum']['valid'])
        self.assertEqual(result['scenarios']['iid_clean']['raw_metrics']['Krum']['accuracy'], .43)

    def test_krum_revocation_failure_still_excludes(self):
        report, amendment = fixture()
        h = report['cells']['Krum_iid_r80']['summary']['health']
        h.update(complete_observed_health_pass=False, additional_review_reasons=['empty_aggregation'])
        report['assessment'] = full.assess(report['cells'], report['attack_gates'], report['pre_attack_pairing'])
        result = reader.apply_amendment(report, amendment)
        self.assertFalse(result['reference_eligibility']['iid']['Krum']['valid'])
        self.assertEqual(result['reference_eligibility']['iid']['Krum']['reasons'], ['r80:empty_aggregation'])

    def test_asr_third_after_including_krum_fails_without_changing_gaps(self):
        report, amendment = fixture()
        report['cells']['Krum_iid_r20']['summary']['tail']['source_transfer_asr'] = .07
        report['cells']['TAD_iid_r20']['summary']['tail']['source_transfer_asr'] = .09
        report['assessment'] = full.assess(report['cells'], report['attack_gates'], report['pre_attack_pairing'])
        self.assertTrue(report['assessment']['scenarios']['iid_r20']['passed'])
        result = reader.apply_amendment(report, amendment)
        c = result['scenarios']['iid_r20']
        self.assertEqual(c['asr_rank'], 3)
        self.assertAlmostEqual(c['asr_gap_pp'], 3)
        self.assertFalse(c['passed'])
        self.assertFalse(result['formal_plan_condition_met'])

    def test_missing_evidence_and_changed_strict_decision_rejected(self):
        report, amendment = fixture(); report['cells']['TAD_iid_r80']['summary']['status'] = 'running'
        with self.assertRaises(ValueError): reader.apply_amendment(report, amendment)
        report, amendment = fixture(); report['assessment']['prerequisites_pass'] = False
        with self.assertRaises(ValueError): reader.apply_amendment(report, amendment)

    def test_no_other_method_or_health_rule_exemption(self):
        for key, value in [('arm', 'H84'), ('partition', 'dirichlet'), ('waived_reason', 'empty_aggregation')]:
            report, amendment = fixture(); amendment['exception'][key] = value
            with self.assertRaises(ValueError): reader.apply_amendment(report, amendment)


if __name__ == '__main__':
    unittest.main()
