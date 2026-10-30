"""Paired identity, empirical non-regression, validation isolation and execution safety."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_cifar_ours_only as run
import cifar_ours_only_report as report
import cifar_ours_only_scheduler as scheduler

class OursOnlyTests(unittest.TestCase):
    def test_matrix_reuses_original_worker_identities_and_only_runs_ours(self):
        p,specs,m,scope=run.plan();old_p,old_specs,old_m=run.legacy.plan()
        self.assertEqual(m,old_m);self.assertEqual(len(m['sources']),110)
        self.assertEqual(len(specs),120);self.assertEqual(len(scope['sources']),5)
        self.assertEqual(specs,[s for s in old_specs if s['arm'] in run.ARMS])
        self.assertEqual({s['config']['method'] for s in specs},{'sm9rrs'})
        self.assertEqual(len(scope['omitted_from_future_dispatch']),60)
        lookup={(s['arm'],s['config']['seed'],s['config']['partition'],s['config']['malicious_ratio']):s for s in specs}
        for key,s in lookup.items():
            if key[0] in run.CANDIDATES:
                original=lookup[('H84',*key[1:])]
                self.assertEqual(s['config'],dict(original['config'],**run.CANDIDATES[key[0]]))
                self.assertEqual(s['profile'],original['profile'])
        self.assertFalse(scope['official_test_used_for_selection'])
    def test_local_training_rejected(self):
        with patch.object(run.legacy,'ROOT',Path('/tmp/local')),self.assertRaises(RuntimeError):run.legacy.require_station()
    def test_protocol_cannot_expand_or_use_test(self):
        p=run.read(run.DEFAULT)
        for key,v in [('permitted_arms',['H84','VERT']),('other_methods_retrained',True),
                      ('worker_identity_unchanged',False),('selection_rule',{})]:
            with patch.object(run,'read',return_value=dict(p,**{key:v})),self.assertRaises(ValueError):run.plan()
    def fixture(self):
        _,specs,_,_=run.plan();cells={}
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
            execution_audited=True,attack_pairing={str(i):dict(classification_prefix_equal=True,round24_model_equal=True) for i in range(96)},
            fedavg_attack_gates={})
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
        p,specs,m,_=run.plan();seen=[]
        def fake(p,subset,m,out):
            seen.append({s['config']['seed'] for s in subset})
            return dict(status='incomplete',**{k:{} for k in ('cells','attack_pairing','fedavg_attack_gates','input_sha256')})
        with patch.object(run.legacy.base,'summarize',side_effect=fake),patch.object(run,'write'),patch.object(run.legacy,'write'):
            run.summarize(p,specs,m,Path('/unused'))
        self.assertEqual(seen,[{s} for s in run.SEEDS])
    def test_quarantine_and_gpu_limit(self):
        with patch.object(run.legacy.base,'gpu_snapshot',return_value=[{'uuid':s} for s in (*run.QUARANTINE,'safe')]):
            self.assertEqual(run.gpu_snapshot(),[{'uuid':'safe'}])
        with self.assertRaises(ValueError):scheduler.run(None,None,None,None,None,7)
    def test_original_reliability_and_capping_formulas_are_not_replaced(self):
        from sm9rrsfl.weighting import bounded_aggregation_coefficients
        from types import SimpleNamespace
        d={'a':SimpleNamespace(accepted=True,would_flag=False,clip_factor=.5)}
        # Absolute cap and clipping leave missing mass; do not renormalize to one.
        got=bounded_aggregation_coefficients(['a'],{'a':.1},{'a':.5},d,weight_cap=2.)
        self.assertAlmostEqual(got['a'],.05)

class ScopeAndExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.out=self.root/'out';self.out.mkdir()
        self.specs=[dict(task_id='ours')];self.m=dict(fingerprint='parent',sources={},specs=self.specs+[dict(task_id='old_vert')])
        self.scope=dict(parent_fingerprint='parent',fingerprint='scope',sources={},tasks=['ours'],omitted_from_future_dispatch=['old_vert'])
        run.write(self.out/'manifest.json',self.m)
        run.write(self.out/'controller_status.json',dict(status='stopped_after_drain',active={},failure=None,pid=None))
        for name in ('ours','old_vert'):
            run.write(self.out/'tasks'/name/'status.json',dict(status='complete'))
            run.write(self.out/'tasks'/name/'task.json',dict(identity=name))
        patcher=patch.object(run,'ROOT',self.root);patcher.start();self.addCleanup(patcher.stop)
    def test_clean_transition_preserves_old_tasks_and_manifest(self):
        before={str(f):f.read_bytes() for f in self.out.rglob('*.json')}
        run.prepare_scope(self.out,self.m,self.scope)
        for f,data in before.items():self.assertEqual(Path(f).read_bytes(),data)
        d=run.read(self.out/run.SCOPE_DIR/'transition.json')
        self.assertEqual(d['retained_ours'],['ours']);self.assertEqual(d['excluded_completed'],['old_vert'])
        manifest=(self.out/run.SCOPE_DIR/'manifest.json').read_bytes()
        run.prepare_scope(self.out,self.m,self.scope)
        self.assertEqual(manifest,(self.out/run.SCOPE_DIR/'manifest.json').read_bytes())
    def test_transition_never_ignores_active_failure_or_partial(self):
        state=run.read(self.out/'controller_status.json')
        for change in (dict(active={'ours':1}),dict(failure='OOM'),dict(status='training')):
            run.write(self.out/'controller_status.json',dict(state,**change))
            with self.assertRaises(ValueError):run.prepare_scope(self.out,self.m,self.scope)
        run.write(self.out/'controller_status.json',state)
        for status in ('failed','running','numerical_failure'):
            run.write(self.out/'tasks/old_vert/status.json',dict(status=status))
            with self.assertRaises(ValueError):run.prepare_scope(self.out,self.m,self.scope)
    def events(self,dirname,names):
        import json
        f=self.out/dirname/'run/events.jsonl';f.parent.mkdir(parents=True,exist_ok=True)
        rows=[]
        for i,n in enumerate(names):
            rows += [dict(event='worker_started',task=n,child_pid=i,admission=dict(uuid='safe',free_mb=8192)),
                     dict(event='worker_exited',task=n,child_pid=i,gpu_uuid='safe',returncode=0,status='complete')]
        f.write_text(''.join(json.dumps(x)+'\n' for x in rows));return f
    def execution_fixture(self):
        run.prepare_scope(self.out,self.m,self.scope)
        run.write(self.out/'controller_status.json',dict(status='panel_terminal',active={},pending=[],failure=None))
        self.events('scheduler_ours_paper_refinement_v1',['old_vert'])
        return self.events('scheduler_ours_only_v1',['ours'])
    def test_audit_counts_historical_controls_separately_and_rejects_repeat(self):
        f=self.execution_fixture();a=report.execution_audit(self.out,self.specs,self.m)
        self.assertEqual(a['ours_attempts'],1);self.assertEqual(a['historical_control_attempts'],1)
        self.assertFalse(a['historical_controls_used_for_selection'])
        f.write_text(f.read_text()*2)
        with self.assertRaisesRegex(ValueError,'Non-single'):report.execution_audit(self.out,self.specs,self.m)
    def test_audit_rejects_new_non_ours_dispatch_or_missing_event(self):
        f=self.execution_fixture();self.events('scheduler_ours_only_v1',['ours','old_vert'])
        with self.assertRaisesRegex(ValueError,'non-Ours'):report.execution_audit(self.out,self.specs,self.m)
        f.write_text('')
        with self.assertRaisesRegex(ValueError,'coverage'):report.execution_audit(self.out,self.specs,self.m)

from tests import test_cp1_dynamic_gpu as pool_suite

class OursOnlySchedulerTests(pool_suite.PoolTests):
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

    def test_excluded_completed_controls_are_preserved_but_never_dispatched(self):
        self.manifest['specs']=self.specs+[dict(task_id='old_vert'),dict(task_id='unrun_fedavg')]
        run.write(self.out/'manifest.json',self.manifest)
        run.write(self.out/'tasks/old_vert/status.json',dict(status='complete'))
        self.policy['legacy_worker_config']=str(self.root/'frozen_worker.json')
        def launch(args,**kw):
            self.assertEqual(args[2],str(self.root/'run_cifar_ours_paper_refinement.py'))
            self.assertEqual(args[4],str(self.root/'frozen_worker.json'))
            return self.launch(args,**kw)
        self.run_pool(launch=launch)
        self.assertEqual({n for n,g in self.started},{s['task_id'] for s in self.specs})
        self.assertEqual(run.read(self.out/'tasks/old_vert/status.json'),dict(status='complete'))
        self.assertFalse((self.out/'tasks/unrun_fedavg').exists())

if __name__=='__main__':unittest.main()
