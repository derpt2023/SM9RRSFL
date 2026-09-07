import tempfile
from pathlib import Path
from dataclasses import asdict
from unittest import TestCase, mock

from run_attack_screen import _execute, select_candidates, summarize, confirmation_report
from sm9rrsfl.fl import ExperimentConfig, ExperimentResult, RoundRecord


class AttackScreenTest(TestCase):
    def result(self):
        config = ExperimentConfig(rounds=2, detector_window=3, attack_start_round=1,
                                  num_clients=10, malicious_ratio=.2)
        records = [RoundRecord('sm9rrs', .2, r, .8, .2, 8, 2, 0, 0, 0, '',
                               attack_target_success_rate=.1) for r in (1, 2)]
        return ExperimentResult(config, records, .8, .2, 2, ('a', 'b'), ())

    def test_cached_completed_run_never_retrains(self):
        result = self.result()
        with tempfile.TemporaryDirectory() as folder:
            task = (str(Path(folder) / 'run.json'), dict(method='sm9rrs'), asdict(result.config))
            with mock.patch('run_attack_screen.run_experiment', return_value=result) as run:
                first = _execute(task, validation=object())
                second = _execute(task, validation=object())
            self.assertEqual(run.call_count, 1)
            self.assertEqual(first, second)

    def test_incomplete_execution_cannot_be_a_feasible_screen_candidate(self):
        result = self.result()
        from dataclasses import replace
        metrics = summarize(replace(result, stopped_round=1))
        self.assertIn('round_completion_rate', metrics['health_reasons'])

    def test_health_failure_cannot_win_even_with_better_accuracy_and_asr(self):
        rows = []
        for method, names in {'fedavg': ['reference'], 'sm9rrs': ['healthy', 'collapsed'],
                              'vert': ['healthy', 'collapsed']}.items():
            for name in names:
                for profile in ['clean', 'attack']:
                    rows.append(dict(method=method, candidate=name, profile=profile,
                        partition='iid', seed=1, final_accuracy=.9,
                        postattack_accuracy=.9 if name == 'collapsed' else .8,
                        postattack_asr=0. if name == 'collapsed' else .1,
                        health_reasons=['all_honest_revoked'] if name == 'collapsed' else []))
        selected, scores = select_candidates(rows,
            {method: {'healthy': {}, 'collapsed': {}} for method in ['sm9rrs', 'vert']})
        self.assertEqual(selected, {'sm9rrs': 'healthy', 'vert': 'healthy'})
        self.assertFalse(scores['sm9rrs/collapsed']['valid'])

    def test_confirmation_clean_failure_invalidates_apparent_attack_win(self):
        rows = []
        for method in ['fedavg', 'sm9rrs', 'vert']:
            for profile in ['clean', 'attack']:
                rows.append(dict(phase='confirmation', method=method, candidate='chosen',
                    profile=profile, partition='iid', seed=2, ratio=0 if profile == 'clean' else .4,
                    final_accuracy=.9, postattack_accuracy=.9 if method == 'sm9rrs' else .8,
                    postattack_asr=0. if method == 'sm9rrs' else .1,
                    health_reasons=['clean_false_revocation_rate']
                        if method == 'sm9rrs' and profile == 'clean' else []))
        audit, comparisons = confirmation_report(rows, {'sm9rrs': 'chosen', 'vert': 'chosen'})
        self.assertEqual(audit['sm9rrs']['status'], 'failed')
        self.assertTrue(comparisons[0]['strictly_better_both'])
        self.assertFalse(comparisons[0]['validated_better_both'])

    def test_absent_confirmation_is_not_a_pass(self):
        audit, comparisons = confirmation_report([], {'sm9rrs': 'chosen', 'vert': 'chosen'})
        self.assertEqual(audit['sm9rrs']['status'], 'not_run')
        self.assertFalse(comparisons)

    def test_missing_seed_cannot_be_reported_as_a_complete_confirmation(self):
        row = dict(phase='confirmation', method='sm9rrs', candidate='chosen', profile='clean',
                   partition='iid', seed=2, ratio=0., final_accuracy=.9, health_reasons=[])
        spec = dict(partitions=['iid'], confirmation_seeds=[2, 3], attacks={'attack': {}}, ratios=[.4])
        audit, _ = confirmation_report([row], {'sm9rrs': 'chosen'}, spec)
        self.assertEqual(audit['sm9rrs']['status'], 'failed')
        self.assertIn('incomplete_confirmation', audit['sm9rrs']['failures'][0]['reasons'])

    def test_cached_config_mismatch_is_rejected(self):
        result = self.result()
        with tempfile.TemporaryDirectory() as folder:
            task = (str(Path(folder) / 'run.json'), dict(method='sm9rrs'), asdict(result.config))
            with mock.patch('run_attack_screen.run_experiment', return_value=result):
                _execute(task, validation=object())
            changed = {**task[2], 'seed': result.config.seed + 1}
            with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                _execute((task[0], task[1], changed), validation=object())

    def test_stopped_before_attack_has_no_attack_tail(self):
        from dataclasses import replace
        result = self.result()
        result = replace(result, config=replace(result.config, attack_start_round=0, detector_window=7))
        metrics = summarize(result)
        self.assertIsNone(metrics['tail_asr'])
