"""Validation isolation, fixed gates, pairing and one-attempt regressions."""
from collections import Counter
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import hfcdr_validation_protocol as p
import hfcdr_validation_runtime as runtime
import hfcdr_validation_analysis as analysis
import hfcdr_validation_scheduler as scheduler
import run_hfcdr_validation as cli
from hfcdr_pilot_data import participant_record
from sm9rrsfl.datasets import ImageDataset
from sm9rrsfl.model import TrainStats


def tasks():
    return p.task_specs(p.specification())[0]


def toy():
    return ImageDataset(x_train=np.zeros((60,1,4,4),np.float32),y_train=np.tile(np.arange(10),6),
        x_test=np.zeros((20,1,4,4),np.float32),y_test=np.tile(np.arange(10),2),name='unit')


class FixedTrainer:
    calls=[]
    def __init__(self,dataset,indices,spec,device):
        self.context=None;self.indices=indices;self.spec=spec
    def local(self,params,client_idx,**kw):
        self.calls.append((client_idx,kw))
        return np.full(self.spec.parameter_size,.001*(client_idx+1),np.float32),TrainStats(.2,len(self.indices[client_idx]))
    def evaluate(self,params):
        return dict(accuracy=.6,asr=.08,source_accuracy=.5,accuracy_samples=20,source_samples=2)


class ProtocolTests(unittest.TestCase):
    def test_104_panel_exact_training_contract_and_separate_seeds(self):
        manifest=p.plan();ts=manifest['tasks']
        self.assertEqual(len(ts),104)
        self.assertEqual(Counter(t['condition'] for t in ts),dict(clean=8,label_flip=32,scale2=32,scale5=32))
        self.assertEqual(Counter(t['config']['dataset'] for t in ts),dict(cifar10=52,mnist=52))
        self.assertEqual({t['config']['method'] for t in ts},{'fedavg'})
        for t in ts:
            c=t['config'];old=p.read(p.ROOT/p.BASELINES[c['dataset']])['methods']['fedavg']['config']
            for name in p.COMMON:self.assertEqual(c[name],old[name])
            self.assertFalse(any(k.startswith(('crypto','sm9','dkg')) for k in c))
        self.assertEqual(manifest['fingerprint'],p.digest({k:v for k,v in manifest.items() if k!='fingerprint'}))
        self.assertEqual(len(manifest['frozen_sources']),126)

    def test_pairings_identical_across_all_attacks_for_each_ratio(self):
        labels=np.tile(np.arange(10),500)
        for part in ('iid','dirichlet'):
            for ratio in (.2,.4,.6,.8):
                ts=[t for t in tasks() if t['config']['dataset']=='mnist' and
                    t['config']['seed']==2026111701 and t['config']['partition']==part and
                    t['config']['malicious_ratio']==ratio]
                records=[participant_record(labels,t['config']) for t in ts]
                self.assertTrue(all(x==records[0] for x in records))

    def test_no_training_from_default_and_no_legacy_output(self):
        with patch.object(cli,'run_task') as worker,patch.object(p,'prepare') as prepare,\
                patch.object(scheduler,'batch') as batch,patch('sys.stdout',new_callable=io.StringIO) as stream:
            cli.main([])
        self.assertFalse(json.loads(stream.getvalue())['training_started'])
        worker.assert_not_called();prepare.assert_not_called();batch.assert_not_called()
        for name in ('cifar-best','hfcdr_attack_study_v1','old/hfcdr_validation_x'):
            with self.assertRaises(ValueError):p.safe_output(p.ROOT/'outputs'/name)

    def test_reject_test_access_seed_overlap_and_fictitious_tuning(self):
        original=p.read(p.CONFIG)
        cases=[]
        c=copy.deepcopy(original);c['official_test_used']=True;cases.append(c)
        c=copy.deepcopy(original);c['formal_seeds'][0]=c['seeds'][0];cases.append(c)
        c=copy.deepcopy(original);c['defense_candidates']['fedavg'][0]['overrides']={'mnist':{'lr':.1}};cases.append(c)
        for c in cases:
            with patch.object(p,'read',return_value=c),self.assertRaises(ValueError):p.specification()


class DataTests(unittest.TestCase):
    def test_mnist_loader_opens_only_training_files(self):
        with patch('sm9rrsfl.datasets._read_idx_images',return_value=toy().x_train) as images,\
             patch('sm9rrsfl.datasets._read_idx_labels',return_value=toy().y_train) as labels:
            ds=p.load_training_raw('mnist','/unused')
        self.assertEqual(len(ds.y_test),0)
        self.assertEqual(images.call_count,1);self.assertEqual(labels.call_count,1)
        self.assertEqual(images.call_args.args[0].name,'train-images-idx3-ubyte.gz')
        self.assertEqual(labels.call_args.args[0].name,'train-labels-idx1-ubyte.gz')

    def test_cifar_loader_opens_five_train_batches_only(self):
        x=np.zeros((10,3,32,32),np.float32);y=np.arange(10)
        with patch('sm9rrsfl.datasets._read_cifar10_batch',return_value=(x,y)) as batch:
            ds=p.load_training_raw('cifar10','/unused')
        self.assertEqual([c.args[0].name for c in batch.call_args_list],[f'data_batch_{i}' for i in range(1,6)])
        self.assertEqual(len(ds.y_train),50);self.assertEqual(len(ds.y_test),0)


class RuntimeTests(unittest.TestCase):
    def test_validation_scope_and_both_attack_onsets(self):
        for dataset,start in [('mnist',12),('cifar10',25)]:
            outputs=[]
            for condition in ('label_flip','scale5'):
                t=copy.deepcopy(next(t for t in tasks() if t['config']['dataset']==dataset and
                    t['condition']==condition and t['config']['malicious_ratio']==.4))
                t['config'].update(num_clients=6,rounds=start,compute_backend='numpy')
                events=[];FixedTrainer.calls=[]
                result=runtime.run_loop(toy(),t,device='cpu',emit=lambda k,v:events.append((k,v)),trainer_factory=FixedTrainer)
                self.assertFalse(result['official_test_used'])
                self.assertEqual(result['evaluation_scope'],'training_validation_all_source5')
                self.assertTrue(analysis.summarize(result)['healthy'])
                audits={v['round']:v for k,v in events if k=='audit'}
                self.assertFalse(any(v['poisoned'] for v in audits[start-1]['exposures'].values()))
                self.assertEqual(sum(v['poisoned'] for v in audits[start]['exposures'].values()),2)
                outputs.append((result,events,FixedTrainer.calls))
            self.assertEqual(outputs[0][2],outputs[1][2])
            self.assertNotEqual(outputs[0][0]['model_sha256'],outputs[1][0]['model_sha256'])
        runtime.assert_no_cryptography()

    def test_official_result_is_rejected_by_validation_summarizer(self):
        t=copy.deepcopy(tasks()[0]);t['config'].update(num_clients=6,rounds=12)
        result=runtime.run_loop(toy(),t,device='cpu',emit=lambda *_:None,trainer_factory=FixedTrainer)
        result['official_test_used']=True
        self.assertIn('wrong_experiment_scope',analysis.summarize(result)['health_reasons'])

    def test_worker_preflight_failure_preserved_and_duplicate_refused(self):
        manifest=p.plan();t=manifest['tasks'][0]
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp).resolve();p.write(out/'manifest.json',manifest)
            with patch.object(p,'safe_output',return_value=out),patch.object(p,'check'),\
                 patch.object(runtime,'configure_cuda',side_effect=ValueError('fixture')),\
                 patch.object(runtime,'run_loop') as run:
                with self.assertRaises(ValueError):cli.run_task(out,'/unused',t['task_id'])
                self.assertEqual(p.read(out/'tasks'/t['task_id']/'status.json')['status'],'failed')
                with self.assertRaises(FileExistsError):cli.run_task(out,'/unused',t['task_id'])
                run.assert_not_called()


class GateTests(unittest.TestCase):
    def fixtures(self):
        m=p.plan();s={}
        for t in m['tasks']:
            s[t['task_id']]=dict(task_id=t['task_id'],config=t['config'],complete=True,healthy=True,
                tail10=dict(accuracy=.8,asr=.05 if t['condition']=='clean' else .3))
        return m,s

    def test_paired_excess_gate_and_all_cells_required(self):
        m,s=self.fixtures();r=analysis.attack_gates(s,m)
        self.assertTrue(r['next_stage_allowed']);self.assertEqual(len(r['cells']),48)
        target=next(t for t in m['tasks'] if t['condition']=='label_flip' and t['config']['malicious_ratio']==.4)
        s[target['task_id']]['tail10']['asr']=.06
        r=analysis.attack_gates(s,m)
        self.assertFalse(r['next_stage_allowed'])
        self.assertFalse(r['attack_decisions'][0]['pass_gate'])
        self.assertTrue(r['attack_decisions'][1]['pass_gate'])

    def test_one_good_seed_cannot_hide_weak_seed_and_missing_not_zero(self):
        m,s=self.fixtures()
        target=next(t for t in m['tasks'] if t['condition']=='scale5' and t['config']['malicious_ratio']==.8)
        s[target['task_id']]['tail10']['asr']=.149
        self.assertFalse(analysis.attack_gates(s,m)['next_stage_allowed'])
        del s[target['task_id']]
        cells=analysis.attack_gates(s,m)['cells']
        self.assertTrue(any(c['status']=='incomplete' for c in cells))

    def test_paired_clean_asr_prevents_high_background_false_success(self):
        m,s=self.fixtures()
        for value in s.values():value['tail10']['asr']=.8
        r=analysis.attack_gates(s,m)
        self.assertFalse(r['next_stage_allowed'])
        self.assertTrue(all(c['mean_excess_asr']==0 for c in r['cells']))


class SchedulerTests(unittest.TestCase):
    def test_dynamic_total_same_gpu_and_one_attempt(self):
        m=p.plan();m['tasks']=m['tasks'][:2];m['groups']=m['groups'][:2]
        clock=[0.];launches=[]
        def sleep(v):clock[0]+=v
        rows=[dict(index=i,uuid=f'gpu{i}',free_mib=24000,total_mib=24564,utilization_pct=0,name='fixture') for i in range(2)]
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);p.write(out/'manifest.json',m)
            class Process:
                def __init__(self,command,**kw):
                    self.pid=1000+len(launches);self.returncode=0
                    tid=command[command.index('--run-task')+1];launches.append(tid)
                    self.command=command
                    p.write(out/'tasks'/tid/'status.json',dict(status='complete'))
                def poll(self):return 0
            with patch.object(p,'safe_output',return_value=out),patch.object(p,'check'),\
                 patch.object(scheduler,'gpu_snapshot',return_value=rows),\
                 patch.object(scheduler.time,'sleep',side_effect=sleep),\
                 patch.object(scheduler.time,'monotonic',side_effect=lambda:clock[0]),\
                 patch.object(scheduler.subprocess,'Popen',side_effect=Process):
                scheduler.batch(out,'/unused')
                self.assertEqual(p.read(out/'controller.json')['status'],'complete')
                with self.assertRaises(FileExistsError):scheduler.batch(out,'/unused')
            self.assertEqual(len(launches),2);self.assertEqual(len(set(launches)),2)


if __name__=='__main__':unittest.main()
