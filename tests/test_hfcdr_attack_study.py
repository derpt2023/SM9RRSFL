"""Full matrix, official evaluation, scheduler and numerical-loop regressions."""
from __future__ import annotations

from collections import Counter
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import hfcdr_study_protocol as p
import hfcdr_study_runtime as runtime
import hfcdr_study_scheduler as scheduler
import hfcdr_study_report as reporting
import run_hfcdr_attack_study as cli
from hfcdr_pilot_data import participant_record
from sm9rrsfl.datasets import ImageDataset
from sm9rrsfl.model import TrainStats


def task(dataset='cifar10', method='fedavg', condition='label_flip', ratio=.4):
    return copy.deepcopy(next(t for t in p.task_specs()[0] if t['config']['dataset']==dataset
        and t['config']['method']==method and t['condition']==condition
        and t['config']['malicious_ratio']==ratio and t['config']['partition']=='iid'))


def toy():
    y=np.tile(np.arange(10,dtype=np.int64),6)
    return ImageDataset(x_train=np.zeros((60,1,4,4),dtype=np.float32),y_train=y,
        x_test=np.zeros((20,1,4,4),dtype=np.float32),y_test=np.tile(np.arange(10),2),name='unit')


class ProtocolTests(unittest.TestCase):
    def test_complete_936_matrix_three_seeds_and_shared_clean(self):
        tasks,groups=p.task_specs(); audit=p.audit(tasks,groups)
        self.assertEqual(len(tasks),936);self.assertEqual(len(groups),156)
        self.assertEqual(audit['by_dataset'],dict(cifar10=468,mnist=468))
        self.assertEqual(audit['by_condition'],dict(clean=72,label_flip=288,scale2=288,scale5=288))
        self.assertEqual(audit['by_method'],dict.fromkeys(p.policy()['methods'],156))
        self.assertEqual(Counter(t['config']['seed'] for t in tasks),dict.fromkeys(p.policy()['seeds'],312))
        self.assertEqual(Counter(str(t['config']['malicious_ratio']) for t in tasks),
                         {'0.0':72,'0.2':216,'0.4':216,'0.6':216,'0.8':216})
        self.assertEqual({p.pairing_key(t) for t in tasks}.__len__(),60)

    def test_dataset_specific_hyperparameters_are_unchanged(self):
        for t in p.task_specs()[0]:
            c=t['config'];old=p.read(p.ROOT/p.BASELINES[c['dataset']])['methods'][p.METHOD_SOURCE[c['method']]]['config']
            for name,value in c.items():
                if name not in ('dataset','method','partition','seed','attack','scale','malicious_ratio'):
                    self.assertEqual(value,old[name],(t['task_id'],name))
            self.assertFalse(any(k.startswith(('crypto','sm9','dkg')) for k in c))
        c=task('mnist','hfcdr')['config']
        self.assertEqual((c['detector_window'],c['detector_clip_factor'],c['suspicion_recovery_factor']),(7,3.,1.25))
        c=task('cifar10','hfcdr')['config']
        self.assertEqual((c['detector_window'],c['detector_clip_factor'],c['suspicion_recovery_factor']),(20,1.,4.))

    def test_balanced_six_method_order_rotation_and_group_membership(self):
        tasks,groups=p.task_specs();by_id={t['task_id']:t for t in tasks}
        positions=Counter()
        for group in groups:
            members=[by_id[i] for i in group['tasks']]
            self.assertEqual({t['config']['method'] for t in members},set(p.policy()['methods']))
            self.assertEqual({t['group_id'] for t in members},{group['group_id']})
            for i,t in enumerate(members):positions[i,t['config']['method']]+=1
        self.assertEqual(set(positions.values()),{26})

    def test_manifest_hash_and_matrix_mutation_refusal(self):
        plan=p.plan()
        self.assertEqual(plan['fingerprint'],p.digest({k:v for k,v in plan.items() if k!='fingerprint'}))
        self.assertEqual(len(plan['frozen_sources']),126)
        tasks,groups=p.task_specs();tasks[0]['config']['rounds']=1
        with self.assertRaises(ValueError):p.audit(tasks,groups)

    def test_six_methods_and_three_attacks_share_pairing_for_every_ratio(self):
        labels=np.tile(np.arange(10,dtype=np.int64),500)
        tasks=p.task_specs()[0]
        for ratio in p.policy()['ratios']:
            for part in p.policy()['partitions']:
                relevant=[t for t in tasks if t['config']['dataset']=='mnist' and
                    t['config']['seed']==2026111601 and t['config']['partition']==part
                    and t['config']['malicious_ratio']==ratio]
                records=[participant_record(labels,t['config']) for t in relevant]
                self.assertTrue(all(r==records[0] for r in records))
                self.assertEqual(len(records[0]['malicious_clients']),round(100*ratio))

    def test_legacy_or_nested_or_symlink_output_refused(self):
        for name in ('cifar-best','mnist_v7_target_fair_tuning','old/hfcdr_attack_study_v1','hfcdr_attack_pilot_v1'):
            with self.assertRaises(ValueError):p.safe_output(p.ROOT/'outputs'/name)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'outputs').mkdir()
            (root/'outputs/hfcdr_attack_study_link').symlink_to(p.ROOT/'outputs/cifar-best')
            with patch.object(p,'ROOT',root),self.assertRaises(ValueError):
                p.safe_output(root/'outputs/hfcdr_attack_study_link')

    def test_default_has_no_training_prepare_or_dispatch(self):
        with patch.object(cli,'run_task') as worker,patch.object(p,'prepare') as prepare,\
             patch.object(scheduler,'batch') as batch,patch('sys.stdout',new_callable=io.StringIO) as stream:
            cli.main([])
        self.assertFalse(json.loads(stream.getvalue())['training_started'])
        worker.assert_not_called();prepare.assert_not_called();batch.assert_not_called()


class LoopTests(unittest.TestCase):
    class FixedTrainer:
        calls=[]
        def __init__(self,dataset,indices,spec,device):
            self.context=None;self.indices=indices;self.spec=spec
        def local(self,params,client_idx,**kw):
            self.calls.append((client_idx,kw))
            return np.full(self.spec.parameter_size,.001*(client_idx+1),dtype=np.float32),TrainStats(.2,len(self.indices[client_idx]))
        def evaluate(self,params):
            return dict(accuracy=.6,asr=.08,source_accuracy=.5,accuracy_samples=20,source_samples=2)

    def fixed(self,dataset='mnist',method='fedavg',condition='label_flip',rounds=13):
        t=task(dataset,method,condition)
        t['config'].update(num_clients=6,rounds=rounds,compute_backend='numpy')
        logs=[];self.FixedTrainer.calls=[]
        with patch('cifar_vert_paper_core.fit_client',return_value=[1.]):
            result=runtime.run_loop(toy(),t,device='cpu',emit=lambda k,v:logs.append((k,v)),trainer_factory=self.FixedTrainer)
        return result,logs,list(self.FixedTrainer.calls)

    def test_actual_six_bindings_on_both_dataset_configurations(self):
        for d in ('cifar10','mnist'):
            for method in p.policy()['methods']:
                with self.subTest(dataset=d,method=method):
                    result,logs,calls=self.fixed(d,method,rounds=4)
                    self.assertEqual(result['evaluation_scope'],'official_test_all_source5')
                    self.assertTrue(result['official_test_used'])
                    self.assertFalse(result['official_test_used_for_selection'])
                    self.assertEqual(len(result['records']),5)
                    self.assertEqual(result['binding']['cryptography'],'absent')
                    self.assertEqual(len([v for k,v in logs if k=='final_model']),1)
        runtime.assert_no_cryptography()

    def test_both_onsets_scale_identity_and_exact_common_randomness(self):
        for d,start in [('cifar10',25),('mnist',12)]:
            lf,ll,lc=self.fixed(d,condition='label_flip',rounds=start)
            scaled,sl,sc=self.fixed(d,condition='scale5',rounds=start)
            self.assertEqual(lc,sc)
            a={v['round']:v for k,v in ll if k=='audit'}
            b={v['round']:v for k,v in sl if k=='audit'}
            self.assertEqual(a[start-1],b[start-1])
            self.assertEqual(sum(v['poisoned'] for v in b[start]['exposures'].values()),2)
            self.assertNotEqual(lf['model_sha256'],scaled['model_sha256'])

    def test_timing_stage_totals_round_traces_and_final_model(self):
        result,logs,calls=self.fixed()
        timing=result['timings']
        for metric in ('local_training_seconds','defense_aggregation_seconds','evaluation_seconds'):
            self.assertAlmostEqual(timing[metric],sum(r['round_timing'][metric] for r in result['records']))
            self.assertGreater(timing[metric],0)
        self.assertGreaterEqual(timing['runtime_seconds']+1e-6,sum(timing[k] for k in
            ('local_training_seconds','defense_aggregation_seconds','evaluation_seconds')))
        final=next(v for k,v in logs if k=='final_model')
        self.assertEqual(runtime.array_digest(final),result['model_sha256'])


class ExecutionTests(unittest.TestCase):
    def test_worker_preflight_failure_recorded_once_and_retry_refused(self):
        manifest=p.plan();t=manifest['tasks'][0]
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp).resolve();p.write(out/'manifest.json',manifest)
            with patch.object(p,'safe_output',return_value=out),patch.object(p,'check'),\
                 patch.object(runtime,'configure_cuda',side_effect=ValueError('fixture mismatch')),\
                 patch.object(runtime,'run_loop') as run:
                with self.assertRaises(ValueError):cli.run_task(out,'/unused',t['task_id'])
                status=p.read(out/'tasks'/t['task_id']/'status.json')
                self.assertEqual(status['failure_type'],'execution')
                with self.assertRaises(FileExistsError):cli.run_task(out,'/unused',t['task_id'])
                run.assert_not_called()

    def test_memory_admission_capacity_and_start_spacing(self):
        rows=[dict(index=i,free_mib=free,uuid=str(i)) for i,free in enumerate((24000,8000,17000))]
        c=p.policy()['scheduler']
        self.assertEqual(scheduler.eligible_gpu(rows,[],{},c,100)['index'],0)
        self.assertEqual(scheduler.eligible_gpu(rows,[dict(gpu=0)],{},c,100)['index'],2)
        self.assertEqual(scheduler.eligible_gpu(rows,[dict(gpu=0),dict(gpu=2)],{0:90},c,100)['index'],2)
        self.assertIsNone(scheduler.eligible_gpu(rows,[dict(gpu=0),dict(gpu=0),dict(gpu=2),dict(gpu=2)],{},c,100))

    def test_failed_science_is_distinct_from_infrastructure_and_no_status_success(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            self.assertEqual(scheduler.task_outcome(folder,0)[0],'execution_failure')
            p.write(folder/'status.json',dict(status='failed',failure_type='scientific'))
            self.assertEqual(scheduler.task_outcome(folder,1)[0],'scientific_failure')
            p.write(folder/'status.json',dict(status='complete'))
            self.assertEqual(scheduler.task_outcome(folder,0)[0],'complete')
            self.assertEqual(scheduler.task_outcome(folder,1)[0],'execution_failure')

    def test_existing_controller_cannot_launch_any_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);p.write(out/'manifest.json',p.plan());p.write(out/'controller.json',{})
            with patch.object(p,'safe_output',return_value=out),patch.object(p,'check'),\
                 patch.object(scheduler.subprocess,'Popen') as launch:
                with self.assertRaises(FileExistsError):scheduler.batch(out,'/unused')
                launch.assert_not_called()

    def test_actual_scheduler_probes_order_group_gpu_and_one_attempt(self):
        manifest=p.plan();manifest['groups']=manifest['groups'][:2]
        clock=[0.]; launches=[]
        def sleep(value):clock[0]+=value
        rows=[dict(index=i,uuid=f'gpu{i}',free_mib=24000,total_mib=24564,utilization_pct=0,name='fixture') for i in range(2)]
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);p.write(out/'manifest.json',manifest)
            class Process:
                def __init__(self,command,**kwargs):
                    self.pid=1000+len(launches);self.returncode=0
                    tid=command[command.index('--run-task')+1]
                    launches.append((tid,kwargs['env']['CUDA_VISIBLE_DEVICES']))
                    p.write(out/'tasks'/tid/'status.json',dict(status='complete'))
                def poll(self):return 0
            with patch.object(p,'safe_output',return_value=out),patch.object(p,'check'),\
                 patch.object(scheduler,'gpu_snapshot',return_value=rows),\
                 patch.object(scheduler.time,'sleep',side_effect=sleep),\
                 patch.object(scheduler.time,'monotonic',side_effect=lambda:clock[0]),\
                 patch.object(scheduler.subprocess,'Popen',side_effect=Process),patch.object(scheduler.signal,'signal'):
                scheduler.batch(out,'/unused')
            self.assertEqual(len(launches),12)
            self.assertEqual(len({i for i,g in launches}),12)
            for group in manifest['groups']:
                observed=[(i,g) for i,g in launches if i in group['tasks']]
                self.assertEqual([i for i,g in observed],group['tasks'])
                self.assertEqual(len({g for i,g in observed}),1)
            events=[json.loads(line) for line in (out/'scheduler_events.jsonl').read_text().splitlines()]
            self.assertTrue(all(e['admission_elapsed_seconds']>=5 for e in events if e['kind']=='start'))
            self.assertEqual(p.read(out/'controller.json')['completed'],12)


class ReportTests(unittest.TestCase):
    def result(self,dataset='mnist'):
        t=task(dataset);c=t['config']
        rows=[dict(round=i,accuracy=.6,asr=.08,source_accuracy=.5,false_positive_removals=0,
              honest_clients=60,accepted_updates=100,honest_weight_loss=0.) for i in range(c['rounds']+1)]
        return dict(task_id=t['task_id'],config=c,records=rows,stopped_round=c['rounds'],
            cryptography='absent',official_test_used=True,official_test_used_for_selection=False,
            evaluation_scope='official_test_all_source5',nonfinite_updates=0,
            timings=dict(runtime_seconds=10.,local_training_seconds=6.,defense_aggregation_seconds=3.,evaluation_seconds=1.))

    def test_tail10_is_dataset_specific_and_incomplete_never_passes(self):
        for dataset in ('mnist','cifar10'):
            result=self.result(dataset);result['records'][-1]['accuracy']=.7
            summary=reporting.summarize(result)
            self.assertAlmostEqual(summary['tail10']['accuracy'],.61)
            self.assertTrue(summary['healthy'])
            result['records'].pop()
            self.assertIsNone(reporting.summarize(result)['tail10'])
            self.assertFalse(reporting.summarize(result)['complete'])

    def test_wrong_scope_and_bad_health_are_not_hidden(self):
        result=self.result();result['official_test_used_for_selection']=True
        self.assertIn('wrong_experiment_scope',reporting.summarize(result)['health_reasons'])
        result=self.result();result['records'][-1]['false_positive_removals']=7
        self.assertIn('honest_false_removal_above_10pct',reporting.summarize(result)['health_reasons'])

    def test_empty_full_report_keeps_every_cell_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            report=reporting.report(Path(temp),p.plan())
        self.assertEqual(len(report['missing']),936)
        self.assertEqual(len(report['comparisons']),144)
        self.assertFalse(report['complete'])
        self.assertTrue(all(c['status']=='incomplete' for c in report['comparisons']))


if __name__=='__main__':
    unittest.main()
