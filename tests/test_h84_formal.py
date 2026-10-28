from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run_cifar_h84_formal as formal
import cifar_h84_formal_report as report
from scripts.experiments.cifar10 import attack_stability_worker as w


class FormalTests(unittest.TestCase):
    def setUp(self):
        self.p = formal.read(formal.DEFAULT)
        self.parent = formal.read(formal.ROOT / self.p['development_config'])

    def test_fresh_matrix_only_changes_seed_and_keeps_all_failed_baselines(self):
        specs = formal.specs_for(self.p)
        old = formal.development.specs_for(dict(self.parent, tasks=formal.full.all_tasks()))
        self.assertEqual(len(specs), 180)
        self.assertEqual(len({s['task_id'] for s in specs}), 180)
        for i, seed in enumerate(self.p['seeds']):
            for before, after in zip(old, specs[i*60:(i+1)*60]):
                expected = deepcopy(before['config']); expected['seed'] = seed
                self.assertEqual(after['config'], expected)
                self.assertEqual(before['profile'], after['profile'])
        self.assertEqual(self.p['acceptance']['original_passed_scenarios'], 8)

    def test_refuses_search_old_seed_and_development_overwrite(self):
        for changes in (dict(seeds=[2026102201]), dict(seeds=[1, 1]), dict(retune_on_test=True),
                        dict(result_pruning=True), dict(reuse_development_cells=True),
                        dict(output_dir='outputs/cifar_h84_full_validation150_v1')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                formal.specs_for(dict(self.p, **changes))

    def test_new_test_loader_preserves_training_and_auxiliary_identity(self):
        train=np.zeros((40000,1)); auxiliary=np.ones((2500,1))
        dataset=SimpleNamespace(x_train=train,y_train=np.arange(40000)%10,
            x_attack=auxiliary,y_attack=np.arange(2500)%10,
            x_test=np.ones((10000,1))*2,y_test=np.arange(10000)%10)
        validation=SimpleNamespace(**{**vars(dataset),'x_test':np.zeros((7500,1)), 'y_test':np.arange(7500)%10})
        split=SimpleNamespace(main_dataset=dataset,calibration_dataset=validation)
        contract=dict(split='unchanged'); expected=w._dataset_identity(validation,contract)
        with patch.object(w,'load_image_dataset',return_value=object()), \
             patch.object(w,'stratified_training_three_way_split',return_value=split), \
             patch.object(w,'split_metadata',return_value=contract):
            actual,_=formal.official_data(self.parent['data_config'], expected)
            self.assertIs(actual.x_train, train);self.assertIs(actual.x_attack,auxiliary)
            self.assertEqual(len(actual.y_test),10000)
            changed=deepcopy(expected);changed['arrays']['x_train']='wrong'
            with self.assertRaises(ValueError):formal.official_data(self.parent['data_config'],changed)

    def test_formal_target_audit_never_accepts_development_denominator(self):
        cfg=dict(attack_start_round=25)
        spec=dict(config=cfg,profile={})
        targets=dict(training_target_indices=list(range(128)), source_transfer_count=1000,
            official_test_used_for_selection=False,training_target_is_optimizer_input=True,
            transfer_target_is_optimizer_input=False,evaluation_scope='official_test',evaluation_count=10000)
        self.assertTrue(report.audit_calls([],spec,targets,[],[])['complete_coverage'])
        for changes in (dict(source_transfer_count=750),dict(transfer_target_is_optimizer_input=True),
                        dict(evaluation_scope='training_validation'),dict(official_test_used_for_selection=True)):
            with self.assertRaises(ValueError):report.audit_calls([],spec,dict(targets,**changes),[],[])

    def test_grouped_statistics_keep_actual_n_and_failed_health_results(self):
        cell=dict(status='complete',complete_finite=True,config=dict(malicious_ratio=0),
            tail=dict(accuracy=.6,source_transfer_asr=.1),final_fp=20,
            health=dict(complete_observed_health_pass=False),timing=dict(
                runtime_with_crypto_seconds=10,runtime_without_crypto_seconds=8,crypto_wall_seconds=2))
        cells={'H84_iid_clean_s1':cell, 'H84_iid_clean_s2':dict(status='numerical_failure')}
        group=report.grouped_statistics(cells,[1,2])['H84_iid_clean']
        self.assertEqual(group['actual_n'],1);self.assertFalse(group['all_seeds_complete'])
        self.assertEqual(group['samples']['accuracy']['values'],[.6])
        self.assertIsNone(group['samples']['accuracy']['sample_sd'])
        self.assertEqual(group['samples']['false_revocation_rate']['mean'],.2)

    def test_instrumentation_labels_test_and_restores_hooks_on_failure(self):
        from run_cifar_targeted_optimization import metric_scope
        originals=(w._load_dataset,w._atomic,w._write_json,w._write_pickle,w.RoundMetricEvaluator)
        expected=dict(fake='dataset')
        fake=SimpleNamespace(y_test=np.arange(10000)%10)
        class FakeEvaluator:
            def __init__(self,dataset,config):self.dataset=dataset;self.config=config
            def target_manifest(self):
                return dict(official_test_used_for_selection=False,legacy_validation_indices=[0],
                    transfer_validation_indices=list(range(1000)))
        with tempfile.TemporaryDirectory() as tmp, \
             patch('run_cifar_targeted_optimization.metric_scope',return_value=nullcontext()), \
             patch.object(w,'RoundMetricEvaluator',FakeEvaluator), \
             patch.object(formal,'official_data',return_value=(fake,{})), \
             patch.object(w,'_dataset_identity',return_value=expected):
            with self.assertRaisesRegex(RuntimeError,'intentional'):
                with formal.formal_observation(self.p,{},dict(dataset_identity=expected),Path(tmp)) as measurements:
                    self.assertIs(w._load_dataset({})[0],fake)
                    evaluator=w.RoundMetricEvaluator(fake,SimpleNamespace())
                    self.assertEqual(evaluator.target_manifest()['evaluation_scope'],'official_test')
                    w._write_pickle(Path(tmp)/'checkpoint.pickle',dict(state=1))
                    w._write_pickle(Path(tmp)/'result.pickle',dict(result=1))
                    w._write_json(Path(tmp)/'result.json',dict(qualification='short-screen'))
                    self.assertGreater(measurements['at_result_boundary']['checkpoint_write_seconds'],0)
                    self.assertIn('formal evaluation',json.loads((Path(tmp)/'result.json').read_text())['qualification'])
                    raise RuntimeError('intentional')
        self.assertEqual(originals,(w._load_dataset,w._atomic,w._write_json,w._write_pickle,w.RoundMetricEvaluator))

    def test_scheduler_binds_formal_worker_and_retains_cross_batch_gpu_locks(self):
        import cifar_h84_formal_scheduler as scheduler
        self.assertIs(scheduler.base,formal)
        source=Path(scheduler.__file__).read_text()
        self.assertIn("worker='run_cifar_h84_formal.py'",source)
        self.assertIn("outputs/.codex_gpu_locks",source)
        self.assertNotIn("worker='run_cifar_cp1_cuda_development.py'",source)
        rows=[dict(uuid='GPU-'+str(i),index=i,free_mb=9000,util=100,name='NVIDIA GeForce RTX 4090 D') for i in range(8)]
        counts,ready=scheduler.ready_gpus(rows,self.p['gpu_admission'],{},set())
        self.assertEqual(ready,[])
        _,ready=scheduler.ready_gpus(rows,self.p['gpu_admission'],counts,{'GPU-3'})
        self.assertEqual(len(ready),7)

    def test_real_cpu_one_round_observes_metrics_and_crypto_wall_fields(self):
        from sm9rrsfl.datasets import ImageDataset
        from sm9rrsfl.fl import ExperimentConfig
        from contextlib import redirect_stdout
        import io
        data=ImageDataset(x_train=np.zeros((20,1,8,8),np.float32),y_train=np.arange(20)%10,
            x_test=np.zeros((10000,1,8,8),np.float32),y_test=np.arange(10000)%10,
            x_attack=np.zeros((128,1,8,8),np.float32),y_attack=np.full(128,5),name='mnist')
        cfg=ExperimentConfig(method='fedavg',rounds=1,num_clients=2,malicious_ratio=0,
            compute_backend='torch',device='cpu',local_epochs=1,batch_size=10,eval_interval=1,
            early_stop=False,attack_target_count=128,attack_source_label=5,attack_target_label=7)
        identity=w._dataset_identity(data,{})
        with tempfile.TemporaryDirectory() as tmp, patch.object(formal,'official_data',return_value=(data,{})):
            req=dict(task_id='toy',config=asdict(cfg),profile=None,fingerprint='synthetic-fixture',
                data_config={},output_dir=tmp,device='cpu')
            with redirect_stdout(io.StringIO()), formal.formal_observation(self.p,{},dict(dataset_identity=identity),Path(tmp)) as observed:
                state=w.execute_task(req)
            self.assertEqual(state['status'],'complete',state.get('error'))
            row=state['final_metrics']
            self.assertEqual(row['source_transfer_count'],1000)
            self.assertTrue(0 <= row['source_transfer_asr'] <= 1)
            self.assertIn('cumulative_stage_timings',row)
            self.assertTrue(formal.timing(state['summary'])['available_and_consistent'])
            self.assertGreater(observed['at_result_boundary']['checkpoint_write_seconds'],0)
            targets=formal.read(Path(tmp)/'toy/targets.json')
            self.assertEqual(targets['evaluation_scope'],'official_test')
            self.assertEqual(targets['training_target_count'],128)
            self.assertFalse(targets['transfer_target_is_optimizer_input'])


from tests import test_cp1_dynamic_gpu as pool_suite
import cifar_h84_formal_scheduler as formal_pool


class FormalSchedulerTests(pool_suite.PoolTests):
    """Apply the concurrency, draining, locking and real-child suite to the new controller."""
    def setUp(self):
        for name,value in (('pool',formal_pool),('base',formal)):
            patcher=patch.object(pool_suite,name,value)
            patcher.start();self.addCleanup(patcher.stop)
        super().setUp()


if __name__=='__main__':unittest.main()
