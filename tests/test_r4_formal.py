"""Resource-only handoff, concurrent CPU workers, and immutable task evidence."""
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cifar_r4_formal_scheduler as pack
import cifar_r4_formal_report as report
import run_cifar_r4_formal as run
import render_cifar_r4_formal as render
from types import SimpleNamespace


class PackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root/'outputs/panel'
        self.out.mkdir(parents=True)
        self.m = dict(fingerprint='same', sources={}, reused_h84_specs=[])
        self.specs = [dict(task_id=f'R4_task_{i}') for i in range(8)]
        self.p = dict(numerics=dict(cublas_workspace_config=':4096:8'),
                      process_environment=dict(PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python', OMP_NUM_THREADS='2'))
        self.settings = dict(pack.POLICY, sample_interval_seconds=.01, same_gpu_launch_interval_seconds=.03,
                             maximum_wait_seconds=1)
        self.policy = dict(worker='run_cifar_r4_formal.py', adapter_sha256={},
                           parameters=self.settings, quarantine_gpu_uuids=['bad'])
        self.gpus = [dict(uuid=f'GPU-{i}', index=i, name='NVIDIA GeForce RTX 4090 D',
                          free_mb=22000, util=50) for i in range(2)]
        pack.base.write(self.out/'manifest.json', self.m)
        pack.base.write(self.out/'controller_status.json', dict(status='stopped_after_drain',
            active={}, pending=[], failure=None, pid=99999999))
        self.started, self.live = [], {}
        self.peak, self.peak_gpu = 0, Counter()
        self.real_popen = subprocess.Popen

    def launch(self, args, **kwargs):
        name = args[-1]
        uuid = kwargs['env']['CUDA_VISIBLE_DEVICES']
        self.assertEqual(kwargs['env']['OMP_NUM_THREADS'], '2')
        self.assertEqual(kwargs['env']['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        self.started.append((name, uuid))
        self.live[name] = uuid
        self.peak = max(self.peak, len(self.live))
        for gpu,count in Counter(self.live.values()).items():
            self.peak_gpu[gpu] = max(count, self.peak_gpu[gpu])
        script = ('import json,os,pathlib,sys,time; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True); '
                  'start=time.time(); time.sleep(.25); '
                  '(p/"status.json").write_text(json.dumps(dict(status="complete",start=start,end=time.time(),'
                  'gpu=os.environ["CUDA_VISIBLE_DEVICES"],omp=os.environ["OMP_NUM_THREADS"])))')
        child = self.real_popen([sys.executable, '-c', script, str(self.out/'tasks'/name)], **kwargs)
        original_poll = child.poll
        def poll():
            code = original_poll()
            if code is not None:
                self.live.pop(name, None)
            return code
        child.poll = poll
        return child

    def run_pool(self, launch=None, snapshot=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(pack.base, 'ROOT', self.root))
            stack.enter_context(patch.object(pack.base, 'summarize'))
            stack.enter_context(patch.object(pack.base, 'QUARANTINE', ['bad']))
            stack.enter_context(patch.object(pack.base, 'gpu_snapshot', side_effect=snapshot or (lambda:self.gpus)))
            stack.enter_context(patch.object(pack, 'POLICY', self.settings))
            stack.enter_context(patch.object(pack.subprocess, 'Popen', side_effect=launch or self.launch))
            pack.run(self.root/'config.json', self.p, self.specs, self.m, self.out, self.policy)

    def test_real_workers_pack_six_on_two_gpus_single_attempt_and_preserve_completed(self):
        folder = self.out/'tasks'/self.specs[-1]['task_id']
        pack.base.write(folder/'status.json', dict(status='complete'))
        (folder/'checkpoint.pickle').write_bytes(b'opaque checkpoint')
        before = {p.name:p.read_bytes() for p in folder.iterdir()}
        self.run_pool()
        self.assertEqual(self.peak, 6)
        self.assertEqual(dict(self.peak_gpu), {'GPU-0':3, 'GPU-1':3})
        self.assertEqual(len(self.started), 7)
        self.assertEqual(len({n for n,g in self.started}), 7)
        self.assertEqual(before, {p.name:p.read_bytes() for p in folder.iterdir()})
        self.assertEqual(pack.base.read(self.out/'manifest.json'), self.m)
        with patch.object(pack.base, 'ROOT', self.root):
            audited = pack.audit_packing(self.out)
            self.assertEqual(next(iter(audited.values()))['peak_tasks'], 6)
        rows = [pack.base.read(self.out/'tasks'/n/'status.json') for n,g in self.started]
        self.assertTrue(all(r['omp']=='2' for r in rows))
        first = [r for r in rows if r['gpu']=='GPU-0'][:3]
        self.assertLess(max(r['start'] for r in first), min(r['end'] for r in first))

    def test_recheck_rejects_memory_drop_without_reserving_or_launching(self):
        self.specs = self.specs[:1]
        calls = 0
        def snapshot():
            nonlocal calls
            calls += 1
            return [dict(self.gpus[0], free_mb=100 if calls==3 else 22000)]
        self.run_pool(snapshot=snapshot)
        self.assertEqual(len(self.started), 1)
        self.assertGreaterEqual(calls, 6)

    def test_cooldown_slots_quarantine_and_consecutive_samples(self):
        rows = [self.gpus[0], dict(self.gpus[1], uuid='bad')]
        with patch.object(pack.base, 'QUARANTINE', ['bad']):
            counts,ready = pack.ready_gpus(rows, {}, {}, {}, 100)
            self.assertFalse(ready)
            counts,ready = pack.ready_gpus(rows, {}, {}, counts, 105)
            self.assertEqual(ready, [self.gpus[0]])
            self.assertEqual(pack.ready_gpus(rows, {}, {'GPU-0':100}, counts, 159), ({},[]))
            active = {str(i):{'gpu':self.gpus[0]} for i in range(3)}
            self.assertEqual(pack.ready_gpus(rows, active, {}, counts, 200), ({},[]))
            self.assertEqual(pack.ready_gpus([dict(self.gpus[0], free_mb=8191)], {}, {}, counts, 200), ({},[]))

    def test_partial_and_failed_are_preserved_without_launch(self):
        for status in ('running', 'failed', 'numerical_failure'):
            pack.base.write(self.out/'tasks'/self.specs[0]['task_id']/'status.json', dict(status=status))
            with self.assertRaises(ValueError):
                self.run_pool()
            self.assertEqual(self.started, [])

    def test_controller_lock_and_gpu_lock_are_respected(self):
        with (self.out/'controller.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):self.run_pool()
        with patch.object(pack.base, 'ROOT', self.root):
            handle = pack.old.take_gpu_lock('GPU-0')
            try:self.assertIsNone(pack.old.take_gpu_lock('GPU-0'))
            finally:handle.close()

    def test_gpu_lock_is_held_until_last_shared_worker_exits(self):
        def launch(args, **kw):
            uuid = kw['env']['CUDA_VISIBLE_DEVICES']
            self.assertIsNone(pack.old.take_gpu_lock(uuid))
            return self.launch(args, **kw)
        self.run_pool(launch=launch)
        with patch.object(pack.base, 'ROOT', self.root):
            for row in self.gpus:
                handle = pack.old.take_gpu_lock(row['uuid'])
                self.assertIsNotNone(handle)
                handle.close()

    def test_failure_drains_without_retry(self):
        first = True
        def launch(args, **kw):
            nonlocal first
            child = self.launch(args, **kw)
            if first:
                first = False
                original = child.poll
                def fail():
                    code = original()
                    return 7 if code is not None else None
                child.poll = fail
            return child
        with self.assertRaisesRegex(RuntimeError, 'exit=7'):
            self.run_pool(launch=launch)
        self.assertEqual(len(self.started), 6)
        self.assertEqual(self.live, {})
        self.assertEqual(pack.base.read(self.out/'controller_status.json')['status'], 'blocked')

    def test_signal_only_drains_current_workers(self):
        def launch(args, **kw):
            child = self.launch(args, **kw)
            os.kill(os.getpid(), signal.SIGTERM)
            return child
        self.run_pool(launch=launch)
        self.assertEqual(len(self.started), 1)
        self.assertEqual(pack.base.read(self.out/'controller_status.json')['status'], 'stopped_after_drain')

    def test_retained_mutation_detected(self):
        folder = self.out/'tasks'/self.specs[0]['task_id']
        pack.base.write(folder/'status.json', dict(status='complete'))
        evidence = pack.retain(self.out, self.specs)
        (folder/'status.json').write_text('{"status":"changed"}')
        with self.assertRaisesRegex(ValueError, 'Retained'):
            pack.verify_retained(self.out, evidence)

    def test_live_controller_prevents_takeover(self):
        with patch.object(pack,'command',return_value=[b'python3']):
            with self.assertRaisesRegex(ValueError,'exited'):
                self.run_pool()
        self.assertEqual(self.started,[])


def fixture():
    from tests.test_vert_kmeans_render import fixture as previous_fixture
    b,_,old=previous_fixture()
    parent=dict(report=old,trajectories=b['trajectories'])
    for c in old['cells'].values():c['config']['suspicion_recovery_factor']=1.25
    raw=json.dumps(parent).encode()
    new={n:deepcopy(c) for n,c in old['cells'].items() if c['arm']=='H84'}
    audits={}
    for name,c in new.items():
        c['config'].update(run.PARAMETERS);c.update(candidate='R4',source_task=name.replace('H84_','R4_',1))
        audits[name]=dict(summary=c,attack=deepcopy(old['audited_cells'][name]['attack']),actual_weights_verified=True,
            mechanism=dict(actual_rule_count_coverage_and_mass_verified=True))
    pairs={str(s):{n:deepcopy(v) for n,v in old['per_seed'][str(s)]['pairing'].items() if n.startswith('H84_')} for s in run.SEEDS}
    warmup={c['source_task']:True for c in new.values()}
    r=report.combine(old,new,audits,pairs,warmup,run.SEEDS)
    r.update(fingerprint='fixture',replacement_lineage={n:dict(same_initial_model_data_targets_malicious_ids=True,
        only_changed_parameter=run.PARAMETERS) for n in new})
    bundle=dict(schema='user-requested-r4-official-export-v1',report=r,manifest=dict(fingerprint='fixture',
        sources={str(i):'hash' for i in range(126)},protocol=dict(parameters=run.PARAMETERS,authorization=run.AUTHORIZATION)),
        source_verified=True,all_inputs_unchanged=True,accepted_baseline_sha256=render.sha(raw),trajectories=deepcopy(parent['trajectories']))
    seal(bundle)
    return bundle,raw,old,new,audits,pairs,warmup


def seal(b):b['combined_report_sha256']=render.sha((json.dumps(b['report'],sort_keys=True,indent=2,allow_nan=False)+'\n').encode())


class FormalTests(unittest.TestCase):
    def test_same30_seeds_and_only_recovery_changed(self):
        p,specs,m=run.plan();_,previous,_=run.formal.plan()
        previous=[s for s in previous if s['arm']=='H84']
        self.assertEqual(len(specs),30);self.assertEqual(len(m['sources']),126)
        self.assertEqual({s['config']['seed'] for s in specs},set(run.SEEDS))
        for s,old in zip(specs,previous):
            expected=deepcopy(old);expected.update(arm='R4',task_id=old['task_id'].replace('H84_','R4_',1))
            expected['config'].update(suspicion_recovery_factor=4.)
            self.assertEqual(s,expected)
            self.assertEqual(s['config']['detector_clip_factor'],1.)
        self.assertEqual(m['data_config'],run.formal.plan()[2]['data_config'])
        self.assertFalse(m['official_test_used_for_selection'])
        self.assertTrue(m['user_requested_after_validation_failure'])
        self.assertFalse(run.AUTHORIZATION['validation_selection_passed'])
        self.assertEqual(p['output_dir'],run.OUTPUT)

    def test_scope_mutation_rejected(self):
        original=run.read(run.DEFAULT)
        changes=dict(parameters={'suspicion_recovery_factor':5.},seeds=[1,2,3],rounds=30,retune_on_test=True,
            output_dir='outputs/cifar-paper',quarantine_gpu_uuids=[],process_environment={},other_five_retrained=True,
            automatic_retry=True,pinned_reference_sha256={},frozen_source_sha256={})
        for k,v in changes.items():
            with self.subTest(k=k),tempfile.TemporaryDirectory() as d:
                path=Path(d)/'bad.json';path.write_text(json.dumps(dict(original,**{k:v})))
                with self.assertRaises(ValueError):run.plan(path)

    def test_prior_failed_decision_is_required_not_relabelled_as_pass(self):
        r=dict(status='complete_audited_reserve_validation',execution_audited=True,
            decision=dict(selected=None,candidates={'R4':dict(eligible=False,failures=['mean_IID_accuracy_gain_below_0.5pp'])}),
            cells={str(i):{} for i in range(120)},attack_pairing={str(i):dict(classification_prefix_equal=True,round24_model_equal=True) for i in range(96)},
            h84_warmup_pairing={str(i):True for i in range(90)},prior_panel=dict(original_pairing_gate_passed=False))
        run.validate_prior(r)
        for mutate in (lambda r:r['decision'].update(selected='R4'),lambda r:r['prior_panel'].update(original_pairing_gate_passed=True),
                       lambda r:r['h84_warmup_pairing'].update({'0':False})):
            bad=deepcopy(r);mutate(bad)
            with self.assertRaises(ValueError):run.validate_prior(bad)

    def test_actual_weights_coverage_identity_finite_and_values(self):
        rows=[dict(round=i,honest_weight_loss=.1,malicious_weight_mass=0.) for i in range(151)]
        actual=[dict(r,task_id='R4_t',weights={'client-0':.2}) for r in rows[1:]]
        ds=[SimpleNamespace(round=i,client_id='client-0',aggregation_weight=.2) for i in range(1,151)]
        self.assertTrue(report.audit_weights(actual,rows,ds,'R4_t'))
        for mutate in (lambda a:a.pop(),lambda a:a[0].update(task_id='other'),lambda a:a[0].update(weights={'client-0':.3}),
                       lambda a:a[0].update(honest_weight_loss=float('nan'))):
            bad=deepcopy(actual);mutate(bad)
            with self.assertRaises(ValueError):report.audit_weights(bad,rows,ds,'R4_t')

    def test_merge_all30_and_preserve_other150_failures(self):
        b,raw,old,new,audits,pairs,warmup=fixture()
        for n,c in old['cells'].items():
            if c['arm']!='H84':self.assertEqual(c,b['report']['cells'][n])
        name=next(iter(new));new[name]['health']['complete_observed_health_pass']=False
        r=report.combine(old,new,audits,pairs,warmup,run.SEEDS)
        self.assertFalse(r['cells'][name]['health']['complete_observed_health_pass'])
        self.assertEqual(r['groups']['H84_iid_clean']['actual_n'],3)
        for bad in ({},dict(new,unexpected=new[name])):
            with self.assertRaises(ValueError):report.combine(old,bad,audits,pairs,warmup,run.SEEDS)
        with self.assertRaises(ValueError):report.combine(old,new,audits,pairs,{},run.SEEDS)
        warmup[next(iter(warmup))]=False
        r=report.combine(old,new,audits,pairs,warmup,run.SEEDS)
        self.assertFalse(r['integrity_checks_passed']);self.assertEqual(len(r['cells']),180)

    def test_render_validates_raw_curves_source_and_recomputed_comparison(self):
        b,raw,*_=fixture()
        with patch.object(run,'BASELINE_SHA',render.sha(raw)):
            self.assertIs(render.validate(b,raw),b['report'])
            for mutate in (lambda b:b['trajectories']['FedAvg_iid_clean_s2026102801'][3].update(accuracy=.8),
                           lambda b:b['report'].update(validation_selection_passed=True),
                           lambda b:b['report']['paired_ours_changes'][next(iter(b['report']['paired_ours_changes']))].update(iid_or_non_iid_raw_accuracy_gap_pp=-1),
                           lambda b:b['report']['cells']['H84_iid_clean_s2026102801']['config'].update(suspicion_recovery_factor=2)):
                bad=deepcopy(b);mutate(bad);seal(bad)
                with self.assertRaises(ValueError):render.validate(bad,raw)

    def test_fresh_controller_and_results_metadata(self):
        with tempfile.TemporaryDirectory() as d:self.assertEqual(pack.check_previous(Path(d))['status'],'fresh_panel')
        from scripts.experiments.cifar10 import attack_stability_worker as w
        captured=[]
        with patch.object(w,'_write_json',side_effect=lambda path,v:captured.append(v)):
            before=w._write_json
            with run.r4_metadata_scope():w._write_json(Path('result.json'),{'qualification':'old'})
            self.assertIs(w._write_json,before)
        self.assertEqual(captured[0]['user_authorization'],run.AUTHORIZATION)
        self.assertIn('validation selection failed',captured[0]['qualification'])

    def test_single_execution_no_extra_methods_or_retries(self):
        specs=[dict(task_id=f'R4_{i}') for i in range(30)]
        events=[]
        for i,s in enumerate(specs):
            events+=[dict(event='worker_started',task=s['task_id'],child_pid=i,admission=dict(uuid='good',free_mb=9000)),
                dict(event='worker_exited',task=s['task_id'],child_pid=i,gpu_uuid='good',returncode=0,status='complete')]
        with tempfile.TemporaryDirectory() as d,patch.object(run,'ROOT',Path(d)):
            out=Path(d)/'out';run.write(out/'controller_status.json',dict(status='panel_terminal',active={},pending=[],failure=None))
            f=out/'scheduler_r4_formal_v1'/'run'/'events.jsonl';f.parent.mkdir(parents=True)
            f.write_text('\n'.join(json.dumps(e) for e in events))
            self.assertEqual(report.execution(out,specs)['execution_attempts'],30)
            for mutate in (lambda e:e.append(e[0]),lambda e:e[-1].update(returncode=1),lambda e:e[0]['admission'].update(free_mb=8000)):
                bad=deepcopy(events);mutate(bad);f.write_text('\n'.join(json.dumps(e) for e in bad))
                with self.assertRaises(ValueError):report.execution(out,specs)


if __name__=='__main__':unittest.main()
