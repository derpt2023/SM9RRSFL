"""New-seed full-matrix configuration and prespecified confirmation gates."""
from copy import deepcopy
import io
import unittest
from unittest.mock import patch

import hfcdr_validation_protocol as p
import hfcdr_validation_selection as s
import hfcdr_confirmation as c
import run_hfcdr_confirmation as cli


def fixture():
    policy = p.specification(s.SPEC)
    choice = dict(confirmation_allowed=True, defense_fingerprint='fixture', decisions=[
        dict(method=m, dataset=d, selected=cs[-1]['id'] if d == 'mnist' else cs[0]['id'])
        for m,cs in policy['defense_candidates'].items() for d in policy['datasets']])
    rules = p.read(c.RULES)
    spec = c.make_spec(choice,policy,rules); tasks,groups = p.task_specs(spec)
    return dict(fingerprint='confirmation_fixture',policy=spec,tasks=tasks,groups=groups), choice


def results(manifest):
    summaries = {}
    for t in manifest['tasks']:
        config = t['config']; clean = config['attack'] == 'none'; method = config['method']
        acc,asr = (.9,.01) if clean else (.89,.03) if method == 'hfcdr' else (.7,.4) if method == 'fedavg' else (.9,.05)
        summaries[t['task_id']] = dict(task_id=t['task_id'],config=config,complete=True,healthy=method!='tad',
            tail10=dict(accuracy=acc,asr=asr))
    return dict(complete=True,completed=312,summaries=summaries)


class ConfirmationTests(unittest.TestCase):
    def test_one_selected_configuration_shared_across_full312_new_seed(self):
        manifest,_ = fixture(); tasks = manifest['tasks']
        self.assertEqual(len(tasks),312); self.assertEqual(len(manifest['groups']),52)
        self.assertEqual({t['config']['seed'] for t in tasks},{2026111703})
        for method in c.METHODS:
            self.assertEqual(sum(t['config']['method']==method for t in tasks),52)
            for dataset in ('cifar10','mnist'):
                variants = [{k:v for k,v in t['config'].items() if k.startswith(p.PREFIXES.get(method,()))}
                    for t in tasks if t['config']['method']==method and t['config']['dataset']==dataset]
                self.assertTrue(all(v==variants[0] for v in variants))
        mnist = next(t['config'] for t in tasks if t['config']['method']=='alignins' and t['config']['dataset']=='mnist')
        self.assertEqual(mnist['alignins_tda_radius'],2.)

    def test_missing_or_unregistered_choice_cannot_build_confirmation(self):
        _,choice = fixture(); policy = p.specification(s.SPEC); rules = p.read(c.RULES)
        choice['decisions'][0]['selected']='invented'
        with self.assertRaises(ValueError): c.make_spec(choice,policy,rules)
        choice['confirmation_allowed']=False
        with self.assertRaises(ValueError): c.make_spec(choice,policy,rules)

    def test_frozen_controls_health_failure_is_retained_not_retuned(self):
        manifest,_ = fixture(); outcome = c.compare(manifest,results(manifest))
        self.assertTrue(outcome['formal_allowed'])
        self.assertEqual(len(outcome['comparisons']),48)
        self.assertTrue(all(not cell['peer_health']['tad'] for cell in outcome['comparisons']))

    def test_weak20pct_is_not_evidence_but_weak40pct_blocks_formal(self):
        manifest,_ = fixture(); assessment = results(manifest)
        for summary in assessment['summaries'].values():
            if summary['config']['method']=='fedavg' and summary['config']['malicious_ratio']==.2:
                summary['tail10']['asr']=.06
        outcome = c.compare(manifest,assessment)
        self.assertTrue(outcome['formal_allowed'])
        self.assertTrue(all(not x['relative_reduction_applicable'] for x in outcome['comparisons'] if x['malicious_ratio']==.2))
        for summary in assessment['summaries'].values():
            if summary['config']['method']=='fedavg' and summary['config']['malicious_ratio']==.4:
                summary['tail10']['asr']=.06
        self.assertFalse(c.compare(manifest,assessment)['formal_allowed'])

    def test_unhealthy_or_uncompetitive_ours_and_missing_runs_block_formal(self):
        manifest,_ = fixture()
        for key,value in [('healthy',False),('accuracy',.86),('asr',.09)]:
            assessment=results(manifest)
            summary=next(s for s in assessment['summaries'].values() if s['config']['method']=='hfcdr' and s['config']['attack']!='none')
            if key=='healthy': summary[key]=value
            else: summary['tail10'][key]=value
            self.assertFalse(c.compare(manifest,assessment)['formal_allowed'])
        assessment=results(manifest);assessment['complete']=False;assessment['completed']=311
        assessment['summaries'].pop(next(iter(assessment['summaries'])))
        self.assertFalse(c.compare(manifest,assessment)['formal_allowed'])

    def test_default_does_not_build_or_prepare_or_dispatch(self):
        with patch.object(c,'rules',return_value=p.read(c.RULES)),patch.object(c,'build') as build,\
             patch.object(c,'prepare') as prepare,patch('sys.stdout',new_callable=io.StringIO):cli.main([])
        build.assert_not_called();prepare.assert_not_called()


if __name__ == '__main__': unittest.main()
