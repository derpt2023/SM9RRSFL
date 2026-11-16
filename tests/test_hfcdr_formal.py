"""Official evaluation is gated, parameters fixed, all 936 runs preserved."""
from copy import deepcopy
from collections import Counter,defaultdict
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hfcdr_validation_protocol as v
import hfcdr_formal_protocol as p
import run_hfcdr_formal as cli
from tests.test_hfcdr_confirmation import fixture


def confirmed():
    manifest,_ = fixture()
    decision = dict(complete=True,formal_allowed=True,fingerprint=manifest['fingerprint'],completed=312)
    return manifest,decision


class FormalTests(unittest.TestCase):
    def test_all936_new_seed_tasks_keep_exact_confirmed_parameters(self):
        manifest,decision = confirmed(); policy = p.make_policy(manifest,decision)
        tasks,groups = p.task_specs(policy)
        self.assertEqual(len(tasks),936);self.assertEqual(len(groups),156)
        self.assertEqual({t['config']['seed'] for t in tasks},{2026111801,2026111802,2026111803})
        reference={(t['config']['method'],t['config']['dataset'],t['config']['partition'],t['condition'],t['config']['malicious_ratio']):t['config'] for t in manifest['tasks']}
        for task in tasks:
            cfg=task['config'];key=(cfg['method'],cfg['dataset'],cfg['partition'],task['condition'],cfg['malicious_ratio'])
            self.assertEqual({k:x for k,x in cfg.items() if k!='seed'},{k:x for k,x in reference[key].items() if k!='seed'})
        for method in policy['methods']:
            self.assertEqual(sum(t['config']['method']==method for t in tasks),156)
        self.assertEqual(sum(t['condition']=='clean' for t in tasks),72)

    def test_each_dataset_has_all_six_positions_balanced_for_every_method(self):
        manifest,decision=confirmed();tasks,groups=p.task_specs(p.make_policy(manifest,decision))
        lookup={t['task_id']:t for t in tasks};per_dataset=defaultdict(Counter);per_domain=defaultdict(Counter)
        for group in groups:
            for position,tid in enumerate(group['tasks']):
                cfg=lookup[tid]['config'];per_dataset[cfg['dataset'],cfg['method']][position]+=1
                per_domain[cfg['dataset'],cfg['partition'],cfg['method']][position]+=1
        self.assertTrue(all(dict(counts)==dict.fromkeys(range(6),13) for counts in per_dataset.values()))
        self.assertTrue(all(set(counts.values())=={6,7} and len(counts)==6 for counts in per_domain.values()))

    def test_failed_or_incomplete_confirmation_cannot_make_formal_policy(self):
        manifest,decision=confirmed()
        for key,value in [('formal_allowed',False),('complete',False),('completed',311),('fingerprint','wrong')]:
            bad=dict(decision);bad[key]=value
            with self.assertRaises(ValueError):p.make_policy(manifest,bad)

    def test_no_extra_candidate_or_missing_method_can_enter_formal(self):
        manifest,decision=confirmed();bad=deepcopy(manifest)
        bad['policy']['defense_candidates']['hfcdr']*=2
        with self.assertRaises(ValueError):p.make_policy(bad,decision)
        bad=deepcopy(manifest);bad['policy']['defense_candidates'].pop('tad')
        with self.assertRaises(ValueError):p.make_policy(bad,decision)

    def test_preparation_checks_gate_before_opening_official_data(self):
        with patch.object(p,'gate',side_effect=ValueError('confirmation failed')),patch.object(p,'load_dataset') as load:
            with self.assertRaisesRegex(ValueError,'confirmation failed'):p.prepare(p.OUTPUT,'/unused')
        load.assert_not_called()

    def test_default_pending_plan_does_not_open_official_data_or_launch(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(p,'CONFIG',Path(tmp)/'absent.json'),\
             patch.object(p,'load_dataset') as load,patch.object(p,'prepare') as prepare,\
             patch('sys.stdout',new_callable=io.StringIO) as out:
            cli.main([])
            self.assertIn('936',out.getvalue());self.assertIn('"official_test_opened": false',out.getvalue())
        load.assert_not_called();prepare.assert_not_called()

    def test_formal_batch_rechecks_qualified_confirmation_before_dispatch(self):
        with patch.object(p,'gate',side_effect=ValueError('failed')),\
             patch('hfcdr_formal_scheduler.batch') as batch:
            with self.assertRaises(ValueError):cli.main(['--batch'])
        batch.assert_not_called()

    def test_only_new_output_prefix_allowed(self):
        self.assertEqual(p.safe_output(p.OUTPUT),p.OUTPUT)
        for bad in (p.ROOT/'outputs/cifar-best',p.ROOT/'outputs/hfcdr_attack_study_v1',p.ROOT/'outputs/hfcdr_validation_attack_v1'):
            with self.assertRaises(ValueError):p.safe_output(bad)

    def test_source_guards_include_confirmation_and_frozen_choice_evidence(self):
        self.assertIn('outputs/hfcdr_validation_confirmation_v1/confirmation.json',p.OWN)
        self.assertIn('outputs/hfcdr_validation_defense_v1/selection.json',p.OWN)
        self.assertIn('hfcdr_study_runtime.py',p.OWN)


if __name__ == '__main__':unittest.main()
