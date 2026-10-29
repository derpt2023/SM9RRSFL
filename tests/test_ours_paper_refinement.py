"""Paired identity, empirical non-regression, validation isolation and execution safety."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_cifar_ours_paper_refinement as run
import cifar_ours_paper_refinement_report as report
import cifar_ours_paper_refinement_scheduler as scheduler

class RefinementTests(unittest.TestCase):
    def test_matrix_only_paper_parameters_changes_and_old_sources_preserved(self):
        from run_cifar_vert_paper_formal import plan
        p,specs,m=run.plan();old=plan()[2]
        self.assertEqual(len(specs),180);self.assertEqual(m['maximum_rounds'],27000)
        self.assertEqual({s['config']['seed'] for s in specs},set(run.SEEDS))
        for f,h in old['sources'].items():self.assertEqual(m['sources'][f],h)
        lookup={(s['arm'],s['config']['seed'],s['config']['partition'],s['config']['malicious_ratio']):s for s in specs}
        self.assertEqual(len(lookup),180)
        for key,s in lookup.items():
            if key[0] in run.CANDIDATES:
                original=lookup[('H84',*key[1:])]
                self.assertEqual(s['config'],dict(original['config'],**run.CANDIDATES[key[0]]))
                self.assertEqual(s['profile'],original['profile'])
        self.assertFalse(m['official_test_used_for_selection']);self.assertTrue(m['prior_test_results_seen'])
    def test_local_training_rejected(self):
        with patch.object(run,'ROOT',Path('/tmp/local')),self.assertRaises(RuntimeError):run.require_station()
    def test_protocol_cannot_expand_or_use_test(self):
        p=run.read(run.DEFAULT)
        for key,v in [('seeds',[2026102801]),('evaluation_scope','official_test'),('candidates',{'anything':{}}),('automatic_retry',True)]:
            with patch.object(run,'read',return_value=dict(p,**{key:v})),self.assertRaises(ValueError):run.plan()
    def fixture(self):
        _,specs,_=run.plan();cells={}
        for s in specs:
            arm=s['arm'];attack=bool(s['config']['malicious_ratio']);part=s['config']['partition']
            acc=.60 if not attack else .55
            if arm in run.CANDIDATES and part=='iid':acc+=.01
            if arm=='VERT' and part=='iid':acc+=.012
            asr=.06
            cells[s['task_id']]=dict(arm=arm,config=s['config'],complete_finite=True,
                health=dict(complete_observed_health_pass=True),tail=dict(accuracy=acc,source_transfer_asr=asr),
                attack_period=dict(accuracy=acc,source_transfer_asr=asr))
        return dict(cells=cells,evaluation_scope='training_validation_7500',official_test_used_for_selection=False,
            execution_audited=True,attack_pairing={str(i):dict(classification_prefix_equal=True,round24_model_equal=True) for i in range(144)},
            fedavg_attack_gates={str(i):dict(passed=True) for i in range(24)})
    def test_selection_retains_every_seed_and_non_iid_failure(self):
        r=self.fixture();self.assertIsNotNone(report.assess(r)['selected'])
        c=r['cells']['R2_dirichlet_r80_s2026102803'];c['tail']['source_transfer_asr']+=.021
        d=report.assess(r);self.assertFalse(d['candidates']['R2']['eligible'])
        self.assertTrue(any('2026102803' in x for x in d['candidates']['R2']['failures']))
    def test_no_partial_or_test_selection_or_silent_missing_seed(self):
        r=self.fixture();next(iter(r['cells'].values()))['complete_finite']=False
        self.assertIsNone(report.assess(r)['selected'])
        r=self.fixture();r['official_test_used_for_selection']=True
        with self.assertRaises(ValueError):report.assess(r)
        r=self.fixture();r['execution_audited']=False
        self.assertIsNone(report.assess(r)['selected'])
    def test_attack_period_regression_and_health_cannot_be_hidden_by_tail(self):
        r=self.fixture()
        for arm in run.CANDIDATES:
            r['cells'][arm+'_dirichlet_r20_s2026102801']['attack_period']['source_transfer_asr']+=.03
        self.assertIsNone(report.assess(r)['selected'])
        r=self.fixture()
        for arm in run.CANDIDATES:r['cells'][arm+'_iid_clean_s2026102802']['health']['complete_observed_health_pass']=False
        self.assertIsNone(report.assess(r)['selected'])
    def test_summary_pairs_within_each_seed(self):
        _,specs,m=run.plan();seen=[]
        def fake(p,subset,m,out):
            seen.append({s['config']['seed'] for s in subset})
            return dict(status='incomplete',**{k:{} for k in ('cells','attack_pairing','fedavg_attack_gates','input_sha256')})
        with patch.object(run.base,'summarize',side_effect=fake),patch.object(run,'write'):
            run.summarize({},specs,m,Path('/unused'))
        self.assertEqual(seen,[{s} for s in run.SEEDS])
    def test_quarantine_and_gpu_limit(self):
        with patch.object(run.base,'gpu_snapshot',return_value=[{'uuid':s} for s in (*run.QUARANTINE,'safe')]):
            self.assertEqual(run.gpu_snapshot(),[{'uuid':'safe'}])
        with self.assertRaises(ValueError):scheduler.run(None,None,None,None,None,7)
    def test_original_reliability_and_capping_formulas_are_not_replaced(self):
        from sm9rrsfl.weighting import bounded_aggregation_coefficients
        from types import SimpleNamespace
        d={'a':SimpleNamespace(accepted=True,would_flag=False,clip_factor=.5)}
        # Absolute cap and clipping leave missing mass; do not renormalize to one.
        got=bounded_aggregation_coefficients(['a'],{'a':.1},{'a':.5},d,weight_cap=2.)
        self.assertAlmostEqual(got['a'],.05)

from tests import test_cp1_dynamic_gpu as pool_suite

class RefinementSchedulerTests(pool_suite.PoolTests):
    def setUp(self):
        for name,value in (('pool',scheduler),('base',run)):
            patcher=patch.object(pool_suite,name,value);patcher.start();self.addCleanup(patcher.stop)
        super().setUp()
    def run_pool(self,specs=None,max_gpus=6,launch=None,snapshot=None):
        return super().run_pool(specs,max_gpus,launch,snapshot)
    def test_eight_concurrent_distinct_gpu_tasks_and_completed_reuse(self):
        # This successor has a stricter six-GPU ceiling than the original suite.
        path=self.out/'tasks/task_0/status.json'
        run.write(path,dict(status='complete'));before=path.read_bytes()
        self.run_pool()
        self.assertEqual(self.peak,6);self.assertEqual(len(self.started),12)
        self.assertEqual(len({name for name,gpu in self.started}),12)
        self.assertNotIn('task_0',dict(self.started));self.assertEqual(path.read_bytes(),before)
        self.assertEqual(run.read(self.out/'controller_status.json')['status'],'panel_terminal')
        self.started.clear()
        with patch.object(scheduler.os,'getpid',return_value=999999):self.run_pool()
        self.assertEqual(self.started,[])

if __name__=='__main__':unittest.main()
