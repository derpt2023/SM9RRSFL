import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_cifar_cp1_cuda_development as runner


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.p=runner.read(runner.DEFAULT)

    def test_only_declared_ours_policy_changes(self):
        specs=runner.specs_for(self.p)
        self.assertEqual(len(specs),12)
        parent=runner.read(runner.ROOT/self.p['parent_config'])
        for s in specs:
            base=parent['cells']['iid_clean_'+s['config']['method']]['config']
            expected=dict(base,**self.p['arms'][s['arm']]['parameters'],device='cuda:0',
                partition=s['config']['partition'],malicious_ratio=s['config']['malicious_ratio'])
            self.assertEqual(expected,s['config'])
            self.assertEqual(s['profile'],parent['cells']['iid_r60_sm9rrs']['profile'])

    def test_attack_training_and_baseline_edits_rejected(self):
        for key,value in [('attack_boost',14.),('lr',.04),('rounds',149),('seed',2026102202)]:
            with self.subTest(key=key):
                p=copy.deepcopy(self.p);p['base_configs']['sm9rrs'][key]=value
                with self.assertRaises(ValueError):runner.specs_for(p)
        p=copy.deepcopy(self.p);p['profile']['lr']=.02
        with self.assertRaises(ValueError):runner.specs_for(p)
        p=copy.deepcopy(self.p);p['arms']['W85']['parameters']['attack_boost']=14.
        with self.assertRaises(ValueError):runner.specs_for(p)

    def test_latest_user_policy_and_numerics_required(self):
        for group,key,value in [('selection_policy','clean_asr_gap',.01),
                ('selection_policy','attack_max_rank',3),('numerics','cuda_matmul_allow_tf32',True)]:
            with self.subTest(key=key):
                p=copy.deepcopy(self.p);p[group][key]=value
                with self.assertRaises(ValueError):runner.specs_for(p)

    def test_no_data_dependent_or_prior_based_parameter_choice(self):
        p=copy.deepcopy(self.p);p['arms']['FedAvg']['parameters']={'detector_distance_threshold':.8}
        with self.assertRaises(ValueError):runner.specs_for(p)
        p=copy.deepcopy(self.p);p['tasks'].append(copy.deepcopy(p['tasks'][0]))
        with self.assertRaises(ValueError):runner.specs_for(p)


class SafetyTests(unittest.TestCase):
    def test_gpu_memory_utilization_model_and_pin(self):
        policy=dict(min_free_mb=16384,max_utilization=5)
        good=dict(index=0,uuid='GPU-a',name='NVIDIA GeForce RTX 4090 D',free_mb=20000,util=0)
        rows=[good,dict(good,uuid='GPU-b',util=6),dict(good,uuid='GPU-c',free_mb=16000),
              dict(good,uuid='GPU-d',name='Different GPU')]
        self.assertEqual(runner.qualified_gpus(rows,policy),[good])
        self.assertEqual(runner.qualified_gpus(rows,policy,'GPU-b'),[])

    def test_partial_and_orphan_are_never_automatically_retried(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'task'
            self.assertEqual(runner.guard_task_directory(p),'new')
            p.mkdir();(p/'checkpoint.pickle').write_bytes(b'preserve')
            with self.assertRaises(ValueError):runner.guard_task_directory(p)
            for status in ['partial','running']:
                (p/'status.json').write_text(json.dumps(dict(status=status)))
                with self.assertRaises(ValueError):runner.guard_task_directory(p)
            for status in ['complete','failed','numerical_failure']:
                (p/'status.json').write_text(json.dumps(dict(status=status)))
                self.assertEqual(runner.guard_task_directory(p),status)
            self.assertEqual((p/'checkpoint.pickle').read_bytes(),b'preserve')

    def test_data_identity_fails_before_training_and_hook_restores(self):
        from scripts.experiments.cifar10 import attack_stability_worker as w
        original=w.RoundMetricEvaluator
        with patch.object(w,'_load_dataset',return_value=('data','contract')),patch.object(w,'_dataset_identity',return_value={'wrong':1}):
            loader=w._load_dataset
            with self.assertRaises(ValueError):
                with runner.observation_scope({'right':1}):w._load_dataset({})
            self.assertIs(w._load_dataset,loader)
        self.assertIs(w.RoundMetricEvaluator,original)

    def test_controller_launches_only_after_stable_idle_and_keeps_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'outputs/panel';(root/'outputs').mkdir()
            s=dict(task_id='x');m=dict(fingerprint='f',sources={})
            p=dict(gpu_admission=dict(min_free_mb=16384,max_utilization=5,consecutive_samples=3,
                sample_interval_seconds=0,maximum_wait_seconds=10),numerics=dict(cublas_workspace_config=':4096:8'))
            good=dict(index=0,uuid='GPU-a',name='NVIDIA GeForce RTX 4090 D',free_mb=22000,util=0)
            snapshots=[[good],[dict(good,util=50)],[good],[good],[good]]
            calls=[]
            class Child:
                pid=123
                def wait(self,timeout=None):
                    (out/'tasks/x').mkdir(parents=True)
                    runner.write(out/'tasks/x/status.json',dict(status='complete'))
                    return 0
            def launch(*args,**kw):
                calls.append(kw['env']['CUDA_VISIBLE_DEVICES'])
                return Child()
            with patch.object(runner,'ROOT',root),patch.object(runner,'gpu_snapshot',side_effect=snapshots) as snap, \
                    patch.object(runner.subprocess,'Popen',side_effect=launch),patch.object(runner,'summarize'):
                runner.batch(root/'config.json',p,[s],m,out)
            self.assertEqual(snap.call_count,5)
            self.assertEqual(calls,['GPU-a'])
            self.assertEqual(runner.read(out/'manifest.json'),m)
            self.assertEqual(runner.read(out/'controller_status.json')['status'],'panel_terminal')


if __name__=='__main__':unittest.main()
