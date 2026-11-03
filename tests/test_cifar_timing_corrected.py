from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_cifar_timing as old
import run_cifar_timing_corrected as fixed
from sm9rrsfl import experiments, fl
from sm9rrsfl.datasets import make_synthetic_mnist_like
from cifar_tad_no_prior import PriorFreeTAD


class BindingTests(unittest.TestCase):
    def task(self, ratio=0., method='ding13'):
        config=fl.ExperimentConfig(method=method,attack='none',num_clients=5,
            malicious_ratio=ratio,rounds=4,local_epochs=1,batch_size=20,
            early_stop=False,compute_backend='numpy',device='cpu',seed=27)
        return dict(task_id='synthetic_test',config=asdict(config),
            profile=old.protocol()['methods']['ding13']['profile'])

    def test_reproduces_the_stale_function_alias_without_training(self):
        saved=experiments.run_experiment
        with old.scientific_scopes(self.task()):
            self.assertIs(experiments.run_experiment,saved)
            self.assertIsNot(experiments.run_experiment,fl.run_experiment)
            self.assertEqual(fl.run_experiment.__name__,'_run_experiment_prior_free_tad')

    def test_real_common_runner_calls_prior_free_tad_and_ratio_is_not_an_input(self):
        dataset=make_synthetic_mnist_like(train_samples=100,test_samples=30,seed=19)
        audits=[];results=[]
        evaluate=PriorFreeTAD.evaluate_round
        def observed(instance,*args,**kwargs):
            answer=evaluate(instance,*args,**kwargs)
            audits.append(instance.last_audit)
            return answer
        original_function=experiments.run_experiment
        with tempfile.TemporaryDirectory() as temp,patch.object(PriorFreeTAD,'evaluate_round',observed):
            for i,ratio in enumerate((0.,.8)):
                folder=Path(temp)/str(i);folder.mkdir()
                result=fixed.measured(dataset,self.task(ratio),folder,'unit-test')
                results.append(result)
                record=old.base.read(folder/'training_binding.json')
                self.assertTrue(record['completed'])
                self.assertTrue(all(x['ding13_detector']=='cifar_tad_no_prior.PriorFreeTAD' for x in record['checkpoint_observations']))
                self.assertEqual(result.stopped_round,4)
                self.assertGreater(result.runtime_seconds,0)
                self.assertGreater(result.checkpoint_io_seconds,0)
                self.assertIs(experiments.run_experiment,original_function)
        self.assertEqual(len(audits),8)
        self.assertEqual(audits[:4],audits[4:])
        self.assertEqual([r.accuracy for r in results[0].records], [r.accuracy for r in results[1].records])

    def test_binding_and_checkpoint_hooks_restore_on_error(self):
        function=experiments.run_experiment;writer=experiments._write_round_checkpoint
        with tempfile.TemporaryDirectory() as temp,patch.object(experiments,'run_measured_experiment',side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):fixed.measured(None,self.task(),Path(temp),'unit-test')
        self.assertIs(experiments.run_experiment,function)
        self.assertIs(experiments._write_round_checkpoint,writer)
        self.assertEqual(fl.run_experiment.__name__,'run_experiment')

    def test_real_vert_common_runner_observes_paper_vert(self):
        dataset=make_synthetic_mnist_like(train_samples=100,test_samples=30,seed=19)
        task=self.task(method='vert')
        task['config'].update(vert_top_k=0,vert_use_ratio_prior=False)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            result=fixed.measured(dataset,task,folder,'unit-test')
            self.assertEqual(result.stopped_round,4)
            record=old.base.read(folder/'training_binding.json')
            self.assertEqual([r['round'] for r in record['checkpoint_observations']],list(range(5)))
            self.assertTrue(all(r['vert_defense']=='cifar_vert_paper_core.PaperVERT' for r in record['checkpoint_observations']))

    def test_real_other_methods_preserve_function_and_restore(self):
        dataset=make_synthetic_mnist_like(train_samples=100,test_samples=30,seed=19)
        saved=experiments.run_experiment
        with tempfile.TemporaryDirectory() as temp:
            for method in ('fedavg','krum','alignins','sm9rrs'):
                folder=Path(temp)/method;folder.mkdir()
                result=fixed.measured(dataset,self.task(method=method),folder,'unit-test')
                self.assertEqual(result.stopped_round,4)
                binding=old.base.read(folder/'training_binding.json')
                self.assertEqual(binding['function'],'run_experiment')
                self.assertTrue(binding['completed'])
                self.assertIs(experiments.run_experiment,saved)

    def test_wrong_actual_tad_or_vert_detector_rejected(self):
        for method in ('ding13','vert'):
            config=fl.ExperimentConfig(method=method)
            with self.assertRaises(ValueError):
                fixed.checkpoint_observation(config,{'completed_round':0})

    def test_tad_missing_round_or_bad_target_view_rejected(self):
        dataset=make_synthetic_mnist_like(train_samples=100,test_samples=30,seed=19)
        task=self.task()
        with tempfile.TemporaryDirectory() as temp:
            fixed.measured(dataset,task,Path(temp),'unit-test')
            binding=old.base.read(Path(temp)/'training_binding.json')
        fixed.validate_binding(task,binding)
        import copy
        missing=copy.deepcopy(binding);missing['checkpoint_observations'].pop(2)
        with self.assertRaises(ValueError):fixed.validate_binding(task,missing)
        wrong=copy.deepcopy(binding);wrong['target_view_module']='old_implementation'
        with self.assertRaises(ValueError):fixed.validate_binding(task,wrong)


class CorrectedProtocolTests(unittest.TestCase):
    def test_matrix_unchanged_and_193_original_sources_preserved(self):
        parent=old.manifest(Path('/tmp/data'))
        current=fixed.manifest(Path('/tmp/data'))
        self.assertEqual(current['blocks'],parent['blocks'])
        self.assertEqual(current['protocol'],parent['protocol'])
        self.assertEqual(current['timing'],parent['timing'])
        self.assertEqual(len(parent['sources']),191)
        self.assertEqual(len(current['sources']),196)
        for name,digest in parent['sources'].items():self.assertEqual(current['sources'][name],digest)
        self.assertEqual(current['reused_results'],0)
        self.assertEqual(current['fresh_tasks'],180)
        self.assertEqual(current['scheduling']['max_gpus'],4)
        original=old.manifest
        with fixed.configured():
            old.verify_manifest(current)
            current['blocks'][0]['tasks'][0]['config']['rounds']=1
            with self.assertRaises(ValueError):old.verify_manifest(current)
        self.assertIs(old.manifest,original)

    def test_gpu_index_uuid_and_both_free_samples_required(self):
        index,uuid=next(iter(fixed.GPU_MAP.items()))
        def snap(index=index,free=8192):return {'gpus':[dict(index=index,uuid=uuid,free=free)]}
        self.assertTrue(fixed.admitted(uuid,snap(),snap()))
        self.assertFalse(fixed.admitted(uuid,snap(free=8191),snap()))
        self.assertFalse(fixed.admitted(uuid,snap(),snap(free=8191)))
        self.assertFalse(fixed.admitted(uuid,snap(index='1'),snap()))
        self.assertFalse(fixed.admitted('wrong',snap(),snap()))

    def test_forbidden_resource_scope_rejected_before_dispatch(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(fixed,'_PARENT_BATCH') as launch:
            for indices,count in [(None,6),({'1'},1),(set(),1),({'2','6'},2)]:
                with self.assertRaises(ValueError):fixed.batch(Path(temp)/'new',Path('/tmp/data'),indices,count)
            launch.assert_not_called()

    def test_existing_output_and_running_study_rejected(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(fixed,'history_snapshot') as history:
            with self.assertRaises(FileExistsError):fixed.batch(Path(temp),Path('/tmp/data'),None,4)
            with patch.object(fixed,'running_study',return_value=[42]):
                with self.assertRaises(ValueError):fixed.batch(Path(temp)/'new',Path('/tmp/data'),None,4)
            history.assert_not_called()

    def test_complete_six_method_block_uses_corrected_worker_and_same_gpu(self):
        from types import SimpleNamespace
        for fail in (False,True):
            with tempfile.TemporaryDirectory() as temp:
                out=Path(temp)/'new';calls=[]
                m=fixed.manifest(Path('/tmp/data'));m['blocks']=m['blocks'][:1]
                index,uuid=next(iter(fixed.GPU_MAP.items()))
                sample=dict(at='2026-11-03T00:00:00+00:00',gpus=[dict(index=index,uuid=uuid,free=9000)],compute_apps=[])
                def child(args,**kw):
                    task_id=args[-1];calls.append((task_id,kw['env']['CUDA_VISIBLE_DEVICES']))
                    self.assertTrue(args[1].endswith('/run_cifar_timing_corrected.py'))
                    code=1 if fail else 0
                    old.base.write(out/'tasks'/task_id/'status.json',dict(status='failed' if fail else 'complete'))
                    return SimpleNamespace(pid=42,returncode=code,poll=lambda:code,wait=lambda:code)
                with fixed.configured(),patch.object(old,'manifest',return_value=m),patch.object(old,'verify_manifest'), \
                     patch.object(old,'gpu_snapshot',return_value=sample), \
                     patch.object(old,'acquire_gpu',side_effect=lambda uuid:open(Path(temp)/'lock','a')), \
                     patch.object(old.signal,'signal'),patch.object(old.time,'sleep'), \
                     patch.object(old.subprocess,'Popen',side_effect=child), \
                     patch.object(old,'final_report',return_value=dict(counts={},execution_audit={'passed':not fail})):
                    fixed._PARENT_BATCH(out,Path('/tmp/data'),{index},4)
                self.assertEqual(len(calls),1 if fail else 6)
                self.assertEqual({gpu for name,gpu in calls},{uuid})
                self.assertEqual([name for name,gpu in calls],[t['task_id'] for t in m['blocks'][0]['tasks']][:len(calls)])

    def test_history_ledger_carries_17_invalid_and_96_actual_attempts(self):
        ledger=old.base.read(fixed.ROOT/fixed.HISTORY)
        self.assertEqual(len(ledger['invalid_tasks']),17)
        self.assertTrue(all(not t['eligible_for_final_tad_comparison'] for t in ledger['invalid_tasks']))
        self.assertEqual(ledger['historical_actual_attempts'],96)
        self.assertEqual(ledger['reused_results'],0)
        with tempfile.TemporaryDirectory() as temp,patch.object(fixed,'ROOT',Path(temp)):
            f=Path(temp)/'artifact';f.write_text('changed')
            with self.assertRaises(ValueError):fixed.verify_history_snapshot({'files':{'artifact':'wrong'}})

    def test_final_report_requires_all_actual_bindings_and_retains_history(self):
        import copy
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);m=fixed.manifest(Path('/tmp/data'))
            old.base.write(out/'manifest.json',m)
            ledger=old.base.read(fixed.ROOT/fixed.HISTORY)
            old.base.write(out/'provenance/history_snapshot.json',dict(files={},contract=ledger))
            for task in (t for b in m['blocks'] for t in b['tasks']):
                method=task['config']['method'];folder=out/'tasks'/task['task_id']
                rows=[dict(round=n,ding13_detector='cifar_tad_no_prior.PriorFreeTAD' if method=='ding13' else None,
                           vert_defense='cifar_vert_paper_core.PaperVERT' if method=='vert' else None) for n in range(151)]
                binding=dict(method=method,completed=True,checkpoint_observations=rows,
                             function='_run_experiment_prior_free_tad' if method=='ding13' else 'run_experiment',
                             target_view_module='cifar_alternating_target_views')
                old.base.write(folder/'training_binding.json',binding)
                old.base.write(folder/'status.json',dict(status='complete'))
            template=dict(execution_audit=dict(passed=True,failures=[],starts=180,max_active=4),health_failures=['retained'])
            with patch.object(fixed,'_PARENT_REPORT',side_effect=lambda out:copy.deepcopy(template)):
                report=fixed.final_report(out)
                self.assertTrue(report['algorithm_binding_audit']['passed'])
                self.assertTrue(report['execution_audit']['passed'])
                self.assertEqual(report['history']['cumulative_actual_attempts'],276)
                self.assertEqual(report['history']['reused_results'],0)
                self.assertEqual(len(report['history']['invalid_tad_tasks']),17)
                self.assertEqual(report['health_failures'],['retained'])
                tad=next(t for b in m['blocks'] for t in b['tasks'] if t['config']['method']=='ding13')
                path=out/'tasks'/tad['task_id']/'training_binding.json'
                broken=old.base.read(path);broken['checkpoint_observations'].pop();old.base.write(path,broken)
                report=fixed.final_report(out)
                self.assertFalse(report['execution_audit']['passed'])
                self.assertFalse(report['algorithm_binding_audit']['passed'])
                self.assertEqual(len(report['algorithm_binding_audit']['tasks']),179)

    def test_checked_callback_stays_inside_common_clock_exclusion(self):
        from tests.test_cifar_mnist_timing import fake_result
        from contextlib import nullcontext
        clock=[0.]
        def inspect(*args):clock[0]+=3;return {'round':0}
        def write(*args,**kwargs):clock[0]+=20
        def run(dataset,config,**kwargs):
            clock[0]+=10;kwargs['checkpoint_callback']({'completed_round':0});clock[0]+=30
            return fake_result()
        run.__name__='run_experiment'
        task=old.blocks(old.protocol())[0]['tasks'][-1]
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(fl,'run_experiment',run), \
             patch.object(fixed,'checkpoint_observation',side_effect=inspect),patch.object(fixed,'validate_binding'), \
             patch.object(experiments,'perf_counter',side_effect=lambda:clock[0]), \
             patch.object(experiments,'_load_round_checkpoint',return_value=(None,0.,0.)), \
             patch.object(experiments,'_effective_checkpoint_interval',return_value=1), \
             patch.object(experiments,'_write_round_checkpoint',side_effect=write), \
             patch.object(experiments,'_peak_rss_mb',return_value=1),patch.object(experiments,'finalize_config_checkpoint'):
            result=fixed.measured(None,task,Path(temp),'unit-test')
        self.assertEqual(result.runtime_seconds,40)
        self.assertEqual(result.checkpoint_io_seconds,23)


if __name__=='__main__':unittest.main()
