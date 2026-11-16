"""Warmup diagnostics are not silently equated to attack-induced failures."""
from copy import deepcopy
import unittest
from hfcdr_health_context import context


class ContextTests(unittest.TestCase):
    def fixture(self):
        result=dict(task_id='fixture',config=dict(method='krum',attack='label_flip',malicious_ratio=.4,attack_start_round=12),
            records=[dict(round=i,false_positive_removals=0,honest_clients=60) for i in range(1,20)])
        summary=dict(healthy=False,health_reasons=['persistent_honest_weight_starvation'],starvation_rounds=[11],high_false_removal_rounds=[])
        return result,summary

    def test_warmup_flag_is_preserved_but_not_claimed_attack_failure(self):
        result,summary=self.fixture();before=deepcopy((result,summary));r=context(result,summary)
        self.assertTrue(r['starvation_exclusively_before_attack']);self.assertFalse(r['raw_healthy'])
        self.assertEqual(r['starvation_windows']['warmup_only'],[dict(first_round=7,last_round=11)])
        self.assertEqual((result,summary),before);self.assertFalse(r['changes_frozen_health_or_selection'])

    def test_crossing_window_is_not_mislabelled_five_attack_rounds(self):
        result,summary=self.fixture();summary['starvation_rounds']=[12,16]
        r=context(result,summary)
        self.assertEqual(r['starvation_windows']['crosses_onset'],[dict(first_round=8,last_round=12)])
        self.assertEqual(r['starvation_windows']['attack_phase_only'],[dict(first_round=12,last_round=16)])
        self.assertFalse(r['starvation_exclusively_before_attack'])

    def test_no_attack_and_false_removal_denominator_explicit(self):
        result,summary=self.fixture();result['config'].update(attack='none',malicious_ratio=0.)
        for row in result['records']:row.update(honest_clients=100,false_positive_removals=17)
        summary['high_false_removal_rounds']=[4,5,6];r=context(result,summary)
        self.assertEqual(r['high_false_removal_rounds_without_attack'],[4,5,6])
        self.assertEqual(r['max_declared_honest_false_removal_rate'],.17)
        self.assertIsNone(r['attack_start_round']);self.assertEqual(len(r['starvation_windows']['no_attack']),1)


if __name__ == '__main__':unittest.main()
