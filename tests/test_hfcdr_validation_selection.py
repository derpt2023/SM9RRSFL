"""Original-method settings, identical budgets and upstream gating tests."""
import copy
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import hfcdr_validation_protocol as p
import hfcdr_validation_selection as s
import run_hfcdr_defense_selection as cli
from hfcdr_pilot_runtime import Server


class ContractTests(unittest.TestCase):
    def test_four_candidates_equal_56_runs_fixed_methods_skip_search(self):
        policy=p.specification(s.SPEC);ts,groups=p.task_specs(policy)
        self.assertEqual(len(ts),616);self.assertEqual(len(groups),56)
        for method,candidates in policy['defense_candidates'].items():
            self.assertEqual(len(candidates),4 if method in ('hfcdr','alignins') else 1)
            for candidate in candidates:
                selected=[t for t in ts if t['config']['method']==method and t['candidate_id']==candidate['id']]
                self.assertEqual(len(selected),56)
        self.assertEqual({t['config']['seed'] for t in ts},{2026111701,2026111702})
        for g in groups:
            members=[t for t in ts if t['task_id'] in g['tasks']]
            shared=[{k:v for k,v in t['config'].items() if k in (*p.COMMON,'dataset','partition','seed','attack','scale','malicious_ratio')} for t in members]
            self.assertTrue(all(x==shared[0] for x in shared))

    def test_actual_candidate_constructor_binding_and_paper_defaults(self):
        ts=p.task_specs(p.specification(s.SPEC))[0];seen=set()
        spec=SimpleNamespace(parameter_size=1000,num_classes=10,svd_matrix_offset=0,svd_matrix_shape=(10,10))
        samples={f'client-{i}':10 for i in range(6)}
        for t in ts:
            c=t['config'];key=(c['method'],c['dataset'],t['candidate_id'])
            if key in seen:continue
            seen.add(key);server=Server(c,samples,spec,None,krum_bound=0)
            if c['method']=='vert':
                v=server.defense
                self.assertEqual((v.history_window,v.predict_epochs,v.learning_rate),(10,5,.01))
                self.assertEqual(v.projection_dim,128 if c['dataset']=='mnist' else 256)
                self.assertIsNone(v.malicious_ratio_prior)
            elif c['method']=='alignins':
                v=server.defense;self.assertEqual(v.alignins_sparsity,.3)
                self.assertIn((v.alignins_tda_radius,v.alignins_mpsa_radius),((1.,1.),(.5,.5),(1.,.5),(2.,2.)))
            elif c['method']=='hfcdr':
                self.assertEqual(server.defense.policy.detector_clip_factor,c['detector_clip_factor'])
                self.assertEqual(server.defense.policy.suspicion_recovery_factor,c['suspicion_recovery_factor'])
            self.assertEqual(server.binding()['cryptography'],'absent')
        self.assertEqual(len(seen),22)

    def test_upstream_failed_performance_blocks_prepare_and_training(self):
        spec=p.specification(s.SPEC)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p.write(root/'manifest.json',{'fingerprint':spec['entry_gate']['attack_manifest_fingerprint']})
            p.write(root/'controller.json',dict(status='complete',active=[]))
            with patch.object(p,'safe_output',return_value=root),patch.object(p,'check'),\
                 patch.object(s,'report',return_value=dict(complete=True,next_stage_allowed=False)),\
                 patch.object(p,'prepare') as prepare,patch.object(s,'verify_selection_sources'):
                with self.assertRaisesRegex(ValueError,'gate failed'):cli.prepare(root,root/'new','/unused')
                prepare.assert_not_called()

    def test_incomplete_upstream_cannot_pass_by_process_exit(self):
        spec=p.specification(s.SPEC)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p.write(root/'manifest.json',{'fingerprint':spec['entry_gate']['attack_manifest_fingerprint']})
            p.write(root/'controller.json',dict(status='incomplete',active=[]))
            with patch.object(p,'safe_output',return_value=root),patch.object(p,'check'),self.assertRaises(ValueError):
                s.attack_gate(root,spec)

    def test_default_only_plans_and_cannot_dispatch(self):
        with patch.object(cli,'prepare') as prepare,patch.object(s,'select') as select,\
             patch('sys.stdout',new_callable=io.StringIO):cli.main([])
        prepare.assert_not_called();select.assert_not_called()

    def test_frozen_selection_source_mismatch_fails(self):
        spec=p.specification(s.SPEC)
        with patch.object(p,'sha',return_value='wrong'),self.assertRaises(ValueError):s.verify_selection_sources(spec)


class SelectionTests(unittest.TestCase):
    def exercise(self,all_ours_bad=False):
        manifest=p.plan(s.SPEC);attack=p.plan();summaries={};avg={}
        for t in manifest['tasks']:
            c=t['config'];cs=manifest['policy']['defense_candidates'][c['method']]
            index=next(i for i,v in enumerate(cs) if v['id']==t['candidate_id'])
            # Candidate2 has the strongest attacked score but fails clean retention.
            acc,asr=(.8,.25) if index==0 else (.8,.05) if index in (1,3) else (.9,.01)
            if c['attack']=='none':acc,asr=(.7,.01) if index==2 else (.8,.01)
            healthy=c['method']!='tad' and not (all_ours_bad and c['method']=='hfcdr')
            summaries[t['task_id']]=dict(task_id=t['task_id'],config=c,healthy=healthy,
                health_reasons=[] if healthy else ['fixture_health'],tail10=dict(accuracy=acc,asr=asr))
        for t in attack['tasks']:
            avg[t['task_id']]=dict(task_id=t['task_id'],config=t['config'],tail10=dict(accuracy=.8,asr=.01))
        with tempfile.TemporaryDirectory() as tmp:
            ap=Path(tmp)/'attack';dp=Path(tmp)/'defense';gate=dict(all_pass=True,attack_fingerprint=attack['fingerprint'])
            for out,m in ((ap,attack),(dp,manifest)):
                p.write(out/'manifest.json',m);p.write(out/'dataset_identity.json',dict(shared=True))
                p.write(out/'pairing.json',dict(pair=dict(shared=True)))
            p.write(dp/'entry_gate.json',gate)
            def reports(out,m):return dict(complete=True,summaries=summaries if Path(out)==dp else avg)
            with patch.object(p,'safe_output',side_effect=lambda x:Path(x)),patch.object(p,'check'),\
                 patch.object(s,'attack_gate',return_value=gate),patch.object(s,'report',side_effect=reports),\
                 patch('audit_hfcdr_validation.audit',return_value=dict(all_results_complete=True)):
                return s.select(ap,dp)

    def test_each_method_selects_same_utility_clean_gate_tie_order_fixed_failure_retained(self):
        result=self.exercise()
        self.assertTrue(result['confirmation_allowed'])
        for d in result['decisions']:
            if d['method']=='hfcdr':self.assertEqual(d['selected'],'recovery')
            if d['method']=='alignins':self.assertEqual(d['selected'],'tight')
            if d['method']=='tad':
                self.assertEqual(d['selected'],'fixed');self.assertTrue(d['fixed_failure_retained'])
            for c in d['candidates']:self.assertEqual((c['full_runs'],c['attacked_runs']),(28,24))

    def test_no_healthy_candidate_leaves_null_without_budget_expansion(self):
        result=self.exercise(all_ours_bad=True)
        self.assertFalse(result['confirmation_allowed'])
        self.assertTrue(all(d['selected'] is None for d in result['decisions'] if d['method']=='hfcdr'))
        self.assertFalse(result['automatic_search_expansion'])


if __name__=='__main__':unittest.main()
