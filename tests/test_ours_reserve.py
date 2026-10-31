"""Bounded reserve identity, retained failures, strict pairing and controller safety."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import run_cifar_ours_reserve as run
import cifar_ours_reserve_report as report
import cifar_ours_reserve_scheduler as scheduler


class ReserveTests(unittest.TestCase):
    def test_exact_90_new_30_reused_and_only_declared_parameters(self):
        p,s,m=run.plan();_,old,old_m,scope=run.first.plan()
        self.assertEqual(len(s),90);self.assertEqual(len(m['reused_h84_specs']),30)
        self.assertEqual(len(m['sources']),120);self.assertEqual(m['maximum_rounds'],13500)
        self.assertEqual(m['reused_h84_specs'],[x for x in old if x['arm']=='H84'])
        self.assertEqual({x['arm'] for x in s},{'R4','C150','R2C150'})
        self.assertTrue(all(x['config']['method']=='sm9rrs' for x in s))
        self.assertFalse({x['task_id'] for x in old}&{x['task_id'] for x in s})
        lookup={x['task_id']:x for x in m['reused_h84_specs']}
        for x in s:
            b=lookup['H84_'+x['task_id'].split('_',1)[1]]
            self.assertEqual(x['config'],dict(b['config'],**run.CANDIDATES[x['arm']]))
            self.assertEqual(x['profile'],b['profile'])
        self.assertEqual(p['frozen_source_sha256'],dict(old_m['sources'],**scope['sources']))
        self.assertNotEqual(p['output_dir'],run.PARENT)
        self.assertFalse(m['official_test_used_for_selection'])
    def test_local_training_rejected(self):
        with patch.object(run.first.legacy,'ROOT',Path('/tmp/local')),self.assertRaises(RuntimeError):run.first.legacy.require_station()
    def test_cannot_expand_change_public_settings_or_use_test(self):
        p=run.read(run.DEFAULT);read=run.read
        for key,v in [('rounds',80),('seeds',[1]),('candidates',{'R8':{}}),('selection_rule',{}),('output_dir',run.PARENT),('evaluation_scope','official_test'),('gpu_admission',{}),('extra',True),('frozen_source_sha256',{})]:
            def altered(path,default=None):return dict(p,**{key:v}) if Path(path)==run.DEFAULT else read(path,default)
            with patch.object(run,'read',side_effect=altered),self.assertRaises(ValueError):run.plan()
    def test_parent_metadata_and_all_baselines_validate_against_actual_archive(self):
        p,s,m=run.plan();a=run.ROOT/'outputs/cifar_ours_only_validation_complete_20261031'
        if not a.exists():a=run.ROOT/run.PARENT
        if not (a/'analysis/ours_only_results.json').exists():self.skipTest('First-panel completion evidence unavailable')
        original=run.read(a/'analysis/ours_only_results.json')
        self.assertEqual(run.sha(a/'analysis/ours_only_results.json'),run.REPORT_SHA)
        run.validate_parent_report(original,m['reused_h84_specs'])
        for transform in [lambda r:r['attack_pairing'].pop(run.FAILED_PAIR),lambda r:r['attack_pairing'][run.FAILED_PAIR].update(round24_model_equal=True),lambda r:r['decision'].update(selected='R2'),lambda r:r.update(official_test_used_for_selection=True),lambda r:r['cells']['H84_iid_clean_s2026102801']['health'].update(complete_observed_health_pass=False),lambda r:r['cells']['H84_iid_clean_s2026102801']['config'].update(seed=1)]:
            r=deepcopy(original);transform(r)
            with self.assertRaises(ValueError):run.validate_parent_report(r,m['reused_h84_specs'])
    def fixture(self):
        _,new,m=run.plan();cells={}
        for s in m['reused_h84_specs']+new:
            arm=s['arm'];cfg=s['config'];acc=.6 if not cfg['malicious_ratio'] else .55
            if arm!='H84' and cfg['partition']=='iid':acc+=.01
            cells[s['task_id']]=dict(status='complete',arm=arm,config=cfg,complete_finite=True,
                health=dict(complete_observed_health_pass=True),tail=dict(accuracy=acc,source_transfer_asr=.06),
                attack_period=dict(accuracy=acc,source_transfer_asr=.06))
        return dict(cells=cells,evaluation_scope='training_validation_7500',official_test_used_for_selection=False,
            execution_audited=True,attack_pairing={n:dict(classification_prefix_equal=True,round24_model_equal=True) for n,c in cells.items() if c['config']['malicious_ratio']},
            h84_warmup_pairing={n:True for n,c in cells.items() if c['arm']!='H84'},
            prior_panel=dict(report_sha256=run.REPORT_SHA,selected=None,original_pairing_gate_passed=False,
                failed_pairs={run.FAILED_PAIR:dict(classification_prefix_equal=False,round24_model_equal=False)}))
    def test_all_seeds_health_and_full_attack_guardrails(self):
        r=self.fixture();self.assertIsNotNone(report.assess(r)['selected'])
        for arm in run.CANDIDATES:
            r['cells'][arm+'_dirichlet_r80_s2026102803']['attack_period']['source_transfer_asr']+=.021
        d=report.assess(r);self.assertIsNone(d['selected'])
        self.assertIn('budget exhausted',d['next_step'])
        self.assertTrue(all(any('2026102803_dirichlet_0.8:non_iid_full_attack_asr'==x for x in a['failures']) for a in d['candidates'].values()))
        r=self.fixture()
        for arm in run.CANDIDATES:r['cells'][arm+'_iid_clean_s2026102802']['health']['complete_observed_health_pass']=False
        self.assertIsNone(report.assess(r)['selected'])
    def test_exact_iid_gain_boundary_and_clean_preservation(self):
        r=self.fixture()
        for n,c in r['cells'].items():
            if c['arm']!='H84' and c['config']['partition']=='iid':c['tail']['accuracy']-=.005
        self.assertIsNotNone(report.assess(r)['selected'])
        for n,c in r['cells'].items():
            if c['arm']!='H84' and c['config']['partition']=='iid':c['tail']['accuracy']-=.00001
        self.assertIsNone(report.assess(r)['selected'])
    def test_missing_or_failed_new_prefix_or_h84_warmup_blocks_selection(self):
        for target in ('attack_pairing','h84_warmup_pairing'):
            r=self.fixture();r[target].pop(next(iter(r[target])))
            self.assertIsNone(report.assess(r)['selected'])
        r=self.fixture();r['attack_pairing']['R4_iid_r40_s2026102802']['round24_model_equal']=False
        self.assertIsNone(report.assess(r)['selected'])
        r=self.fixture();r['h84_warmup_pairing']['R4_iid_clean_s2026102802']=False
        self.assertIsNone(report.assess(r)['selected'])
        r=self.fixture();key=next(iter(r['attack_pairing']));r['attack_pairing']['fake']=r['attack_pairing'].pop(key)
        self.assertIsNone(report.assess(r)['selected'])
    def test_previous_failure_cannot_be_removed_or_requalified(self):
        for field,value in [('failed_pairs',{}),('selected','R2'),('original_pairing_gate_passed',True)]:
            r=self.fixture();r['prior_panel'][field]=value
            with self.assertRaises(ValueError):report.assess(r)
    def test_missing_nonfinite_or_test_data_never_selected(self):
        r=self.fixture();r['cells'].pop(next(iter(r['cells'])))
        self.assertIsNone(report.assess(r)['selected'])
        r=self.fixture();next(iter(r['cells'].values()))['complete_finite']=False
        self.assertIsNone(report.assess(r)['selected'])
        r=self.fixture();r['evaluation_scope']='official_test'
        with self.assertRaises(ValueError):report.assess(r)
    def test_partial_finish_retains_failure_without_running_any_audit_or_worker(self):
        r=self.fixture();r['cells']['R4_iid_r40_s2026102803'].update(status='numerical_failure',complete_finite=False)
        with patch.object(run,'summarize',return_value=r),patch.object(run,'write') as write,patch.object(report,'execution_audit') as audit:
            got=report.finish({},[],{},Path('/unused'))
        self.assertEqual(got['status'],'incomplete_or_failed_validation_retained');audit.assert_not_called();self.assertIsNone(got['decision']['selected']);write.assert_called_once()
    def test_summary_writes_only_new_output_and_retains_prior_failure(self):
        p,s,m=run.plan();old=self.fixture();prior=dict(old,cells={n:c for n,c in old['cells'].items() if c['arm']=='H84'})
        prior['attack_pairing'][run.FAILED_PAIR]=dict(classification_prefix_equal=False,round24_model_equal=False);prior['decision']={'selected':None};prior['execution_audits']={};prior['execution_summary']={'historical_control_attempts':2}
        new=dict(old,cells={n:c for n,c in old['cells'].items() if c['arm']!='H84'},attack_pairing={n:v for n,v in old['attack_pairing'].items() if not n.startswith('H84_')})
        out=Path('/new/output')
        with patch.object(run,'parent_evidence',return_value=(prior,{})),patch.object(run.first.legacy,'summarize',return_value=new) as summary,patch.object(run,'write') as write:
            got=run.summarize(p,s,m,out)
        self.assertEqual(len(got['cells']),120);self.assertEqual(summary.call_args.args[3],out)
        self.assertEqual(write.call_args.args[0],out/'analysis/reserve_progress.json')
        self.assertFalse(got['prior_panel']['original_pairing_gate_passed'])


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.out=self.root/'out';self.out.mkdir();self.specs=[{'task_id':'R4_task'},{'task_id':'C150_task'}];self.m=dict(fingerprint='x',reused_h84_specs=[{'task_id':'H84_task'}])
        patcher=patch.object(run,'ROOT',self.root);patcher.start();self.addCleanup(patcher.stop)
        run.write(self.out/'controller_status.json',dict(status='panel_terminal',active={},pending=[],failure=None))
    def events(self):
        f=self.out/'scheduler_ours_reserve_v1/run/events.jsonl';f.parent.mkdir(parents=True,exist_ok=True);rows=[]
        for i,s in enumerate(self.specs):
            rows.extend([dict(event='worker_started',task=s['task_id'],child_pid=i,admission=dict(uuid='safe',free_mb=8192)),dict(event='worker_exited',task=s['task_id'],child_pid=i,gpu_uuid='safe',returncode=0,status='complete')])
        f.write_text(''.join(json.dumps(x)+'\n' for x in rows));return f,rows
    def test_90_scope_does_not_retrain_baselines_or_retry(self):
        f,rows=self.events();a=report.execution_audit(self.out,self.specs,self.m);self.assertEqual(a['new_attempts'],2);self.assertFalse(a['reference_training_repeated'])
        rows[0]['task']='H84_task';f.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        with self.assertRaisesRegex(ValueError,'Non-reserve'):report.execution_audit(self.out,self.specs,self.m)
        f,_=self.events();f.write_text(f.read_text()*2)
        with self.assertRaisesRegex(ValueError,'Non-single'):report.execution_audit(self.out,self.specs,self.m)
    def test_incomplete_draining_failed_or_quarantined_rejected(self):
        f,rows=self.events()
        for key,value in [('active',{'x':1}),('pending',['x']),('failure','OOM'),('status','draining')]:
            run.write(self.out/'controller_status.json',dict(status='panel_terminal',active={},pending=[],failure=None,**{}))
            state=run.read(self.out/'controller_status.json');state[key]=value;run.write(self.out/'controller_status.json',state)
            with self.assertRaises(ValueError):report.execution_audit(self.out,self.specs,self.m)
        run.write(self.out/'controller_status.json',dict(status='panel_terminal',active={},pending=[],failure=None))
        rows[0]['admission']['uuid']=run.QUARANTINE[0];f.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        with self.assertRaisesRegex(ValueError,'admission'):report.execution_audit(self.out,self.specs,self.m)

from tests import test_cp1_dynamic_gpu as pool_suite

class ReserveSchedulerTests(pool_suite.PoolTests):
    def setUp(self):
        for name,value in (('pool',scheduler),('base',run)):
            patcher=patch.object(pool_suite,name,value);patcher.start();self.addCleanup(patcher.stop)
        super().setUp()
    def run_pool(self,specs=None,max_gpus=6,launch=None,snapshot=None):return super().run_pool(specs,max_gpus,launch,snapshot)
    def test_eight_concurrent_distinct_gpu_tasks_and_completed_reuse(self):
        path=self.out/'tasks/task_0/status.json';run.write(path,dict(status='complete'));before=path.read_bytes();self.run_pool()
        self.assertEqual(self.peak,6);self.assertEqual(len(self.started),12);self.assertNotIn('task_0',dict(self.started));self.assertEqual(path.read_bytes(),before)
        self.started.clear()
        with patch.object(scheduler.os,'getpid',return_value=999999):self.run_pool()
        self.assertEqual(self.started,[])
    def test_worker_entry_and_process_environment_are_fixed(self):
        self.policy['process_environment']=run.PROCESS_ENV
        def launch(args,**kwargs):
            self.assertEqual(Path(args[2]).name,'run_cifar_ours_reserve.py')
            self.assertTrue(all(kwargs['env'][k]==v for k,v in run.PROCESS_ENV.items()))
            return self.launch(args,**kwargs)
        self.run_pool(specs=self.specs[:1],launch=launch)


class ReadOnlyIntegrityTests(unittest.TestCase):
    def test_warmup_gate_includes_round20_but_allows_post_intervention_difference(self):
        rows=[dict(round=i,accuracy=.5,source_transfer_asr=.1,non_source_accuracy=.5,aux128_confidence=.7) for i in range(151)]
        ref=deepcopy(rows);self.assertTrue(report.same_warmup(rows,ref))
        rows[21]['accuracy']+=.1;self.assertTrue(report.same_warmup(rows,ref))
        rows[1]['aux128_confidence']+=1e-14;self.assertFalse(report.same_warmup(rows,ref))
        rows=deepcopy(ref);rows[20]['accuracy']+=.001;self.assertFalse(report.same_warmup(rows,ref))
        with self.assertRaises(ValueError):report.same_warmup(rows[:20],ref)
    def test_report_repair_cannot_overlap_live_controller_or_training(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);run.write(out/'controller_status.json',dict(status='panel_terminal',active={},pending=[],failure=None,pid=123))
            with patch.object(run.os,'kill',return_value=None),self.assertRaisesRegex(ValueError,'still alive'):run.require_finished_controller(out)
            with patch.object(run.os,'kill',side_effect=ProcessLookupError):run.require_finished_controller(out)
            run.write(out/'controller_status.json',dict(status='training',active={'x':1},pending=[],failure=None))
            with self.assertRaisesRegex(ValueError,'terminal'):run.require_finished_controller(out)
    def test_parent_bytes_are_never_rewritten_and_changes_rejected(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(run,'ROOT',Path(temp)):
            root=Path(temp);parent=root/run.PARENT
            run.write(parent/'manifest.json',dict(fingerprint=run.PARENT_FP,sources={}))
            run.write(parent/'ours_only_scope_v1/manifest.json',dict(fingerprint=run.SCOPE_FP,sources={}))
            raw=parent/'tasks/saved.raw';raw.parent.mkdir(parents=True);raw.write_bytes(b'original scientific evidence')
            r=dict(input_sha256={str(raw.relative_to(root)):run.sha(raw)})
            run.write(root/run.REPORT,r);run.write(parent/'reference_lock.json',{'immutable':True})
            paths=[run.REPORT,run.PARENT+'/manifest.json',run.PARENT+'/ours_only_scope_v1/manifest.json',run.PARENT+'/reference_lock.json']
            p=dict(parent_metadata_sha256={n:run.sha(root/n) for n in paths},frozen_source_sha256={})
            before={str(f):f.read_bytes() for f in root.rglob('*') if f.is_file()}
            with patch.object(run,'validate_parent_report',side_effect=lambda r,s:r):
                run.parent_evidence(p,{'reused_h84_specs':[]})
                self.assertEqual(before,{str(f):f.read_bytes() for f in root.rglob('*') if f.is_file()})
                raw.write_bytes(b'altered')
                with self.assertRaisesRegex(ValueError,'evidence changed'):run.parent_evidence(p,{'reused_h84_specs':[]})
                (root/run.REPORT).write_text('{}')
                with self.assertRaisesRegex(ValueError,'metadata changed'):run.parent_evidence(p,{'reused_h84_specs':[]},full=False)

if __name__=='__main__':unittest.main()
