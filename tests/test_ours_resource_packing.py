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

import run_cifar_ours_resource_packing as pack
import cifar_ours_reserve_report as report


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
        self.policy = dict(worker='run_cifar_ours_reserve.py', adapter_sha256={},
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
            # Original execution audit accepts the same event identity/8GiB checks.
            got = report.execution_audit(self.out, self.specs[:-1], self.m)
            self.assertEqual(got['new_attempts'], 7)
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

    def test_only_exact_controller_can_receive_signal(self):
        state = dict(pid=123, status='training', failure=None)
        good = [b'python3', b'-u', b'/repo/run_cifar_ours_reserve.py', b'--batch']
        with patch.object(pack, 'command', return_value=good):
            self.assertEqual(pack.validate_old_controller(state,123),good)
            with self.assertRaises(ValueError):pack.validate_old_controller(state,124)
        for args in ([b'python',b'-u',b'other.py',b'--batch'],good+[b'--task',b'x']):
            with patch.object(pack,'command',return_value=args),self.assertRaises(ValueError):
                pack.validate_old_controller(state,123)

    def test_original_120_sources_and_task_matrix_unchanged(self):
        p,s,m = pack.base.plan()
        self.assertEqual(len(m['sources']),120)
        self.assertEqual(len(s),90)
        self.assertFalse(set(pack.SOURCES)&set(m['sources']))
        self.assertEqual(p['seeds'],[2026102801,2026102802,2026102803])
        self.assertEqual(pack.POLICY['min_free_mb'],8192)
        self.assertEqual(pack.POLICY['same_gpu_launch_interval_seconds'],60)

    def test_handoff_signals_only_validated_controller_once_and_waits_for_exit(self):
        state = dict(pid=123, status='training', failure=None)
        pack.base.write(self.out/'controller_status.json', state)
        args = [b'python3',b'-u',b'run_cifar_ours_reserve.py',b'--batch']
        with patch.object(pack,'validate_old_controller',return_value=args) as validate, \
                patch.object(pack.os,'kill') as kill, patch.object(pack,'command',side_effect=[args,[]]), \
                patch.object(pack.time,'sleep') as sleep, patch.object(pack,'check_drained') as drained:
            pack.wait_for_drain(self.out,123,self.policy)
            kill.assert_called_once_with(123,signal.SIGTERM)
            self.assertEqual(validate.call_count,2)
            sleep.assert_called_once_with(5)
            drained.assert_called_once_with(self.out)
        with patch.object(pack,'validate_old_controller',return_value=args),patch.object(pack.os,'kill') as kill:
            with self.assertRaisesRegex(ValueError,'already requested'):
                pack.wait_for_drain(self.out,123,self.policy)
            kill.assert_not_called()

    def test_live_controller_prevents_takeover(self):
        with patch.object(pack,'command',return_value=[b'python3']):
            with self.assertRaisesRegex(ValueError,'exited'):
                self.run_pool()
        self.assertEqual(self.started,[])


if __name__=='__main__':unittest.main()
