import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import run_cifar_cp1_cuda_development as base
import run_cifar_cp1_dynamic_gpu as pool


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'outputs/panel'
        self.out.mkdir(parents=True)
        self.manifest = dict(fingerprint='unchanged', sources={})
        base.write(self.out / 'manifest.json', self.manifest)
        self.policy = dict(gpu_admission=dict(min_free_mb=16384, max_utilization=5,
                           consecutive_samples=3, sample_interval_seconds=0, maximum_wait_seconds=20),
                           numerics=dict(cublas_workspace_config=':4096:8'))
        for name in pool.SOURCES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('immutable adapter')
        self.specs = [dict(task_id='task_' + str(i)) for i in range(13)]
        self.gpus = [dict(index=i, uuid='GPU-' + str(i), name='NVIDIA GeForce RTX 4090 D',
                         free_mb=22000, util=0) for i in range(8)]
        self.started = []
        self.live = {}
        self.peak = 0

    def launch(self, args, **kwargs):
        task = args[-1]
        gpu = kwargs['env']['CUDA_VISIBLE_DEVICES']
        self.assertNotIn(gpu, self.live.values())
        self.assertEqual(kwargs['env']['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
        self.started.append((task, gpu))
        self.live[task] = gpu
        self.peak = max(self.peak, len(self.live))
        owner = self

        class Child:
            pid = 1000 + len(owner.started)
            remaining = 3
            code = None

            def poll(self):
                if self.code is not None:
                    return self.code
                self.remaining -= 1
                if self.remaining > 0:
                    return None
                return self.wait()

            def wait(self, timeout=None):
                self.code = 0
                owner.live.pop(task, None)
                base.write(owner.out / 'tasks' / task / 'status.json', dict(status='complete'))
                return 0

        return Child()

    def run_pool(self, specs=None, max_gpus=8, launch=None, snapshot=None):
        with patch.object(base, 'ROOT', self.root), patch.object(base, 'summarize'), \
                patch.object(base, 'gpu_snapshot', side_effect=snapshot or (lambda: self.gpus)), \
                patch.object(pool.subprocess, 'Popen', side_effect=launch or self.launch):
            pool.run(self.root / 'config.json', self.policy, self.specs if specs is None else specs,
                     self.manifest, self.out, max_gpus)

    def test_eight_concurrent_distinct_gpu_tasks_and_completed_reuse(self):
        path = self.out / 'tasks/task_0/status.json'
        base.write(path, dict(status='complete'))
        before = path.read_bytes()
        self.run_pool()
        self.assertEqual(self.peak, 8)
        self.assertEqual(len(self.started), 12)
        self.assertEqual(len({name for name, gpu in self.started}), 12)
        self.assertNotIn('task_0', dict(self.started))
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(base.read(self.out / 'manifest.json'), self.manifest)
        state = base.read(self.out / 'controller_status.json')
        self.assertEqual(state['status'], 'panel_terminal')
        self.assertEqual(state['active'], {})
        self.assertIsNone(state['pinned_gpu_uuid'])
        # The same evidence is reused on a second invocation, with no launches.
        self.started.clear()
        # Separate invocation timestamp (the real process uses a distinct PID).
        with patch.object(pool.os, 'getpid', return_value=999999):
            self.run_pool()
        self.assertEqual(self.started, [])

    def test_maximum_and_busy_gpu4_are_respected(self):
        self.gpus[4]['util'] = 90
        self.run_pool(max_gpus=3)
        self.assertEqual(self.peak, 3)
        self.assertNotIn('GPU-4', [gpu for _, gpu in self.started])
        self.assertEqual(len(self.started), 13)

    def test_consecutive_samples_reset_and_occupied_not_eligible(self):
        counts = {}
        for rows in ([self.gpus[0]], [dict(self.gpus[0], util=6)], [self.gpus[0]], [self.gpus[0]]):
            counts, ready = pool.ready_gpus(rows, self.policy['gpu_admission'], counts, set())
            self.assertEqual(ready, [])
        counts, ready = pool.ready_gpus([self.gpus[0]], self.policy['gpu_admission'], counts, set())
        self.assertEqual(ready, [self.gpus[0]])
        counts, ready = pool.ready_gpus([self.gpus[0]], self.policy['gpu_admission'], counts, {'GPU-0'})
        self.assertEqual((counts, ready), ({}, []))

    def test_partial_fails_before_any_launch_and_is_preserved(self):
        path = self.out / 'tasks/task_3/checkpoint.pickle'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'real partial evidence')
        with self.assertRaisesRegex(ValueError, 'Partial'):
            self.run_pool()
        self.assertEqual(self.started, [])
        self.assertEqual(path.read_bytes(), b'real partial evidence')

    def test_existing_failed_is_not_retried_or_silently_ignored(self):
        base.write(self.out / 'tasks/task_0/status.json', dict(status='failed'))
        with self.assertRaisesRegex(ValueError, 'failure retained'):
            self.run_pool()
        self.assertEqual(self.started, [])

    def test_worker_failure_drains_other_worker_and_blocks_new_dispatch(self):
        self.gpus = self.gpus[:2]

        def launch(args, **kw):
            child = self.launch(args, **kw)
            task = args[-1]
            if task == 'task_0':
                def fail():
                    self.live.pop(task, None)
                    base.write(self.out / 'tasks' / task / 'status.json', dict(status='running'))
                    return 7
                child.poll = fail
            else:
                child.remaining = 6
            return child

        with self.assertRaisesRegex(RuntimeError, 'exit=7'):
            self.run_pool(launch=launch)
        self.assertEqual(len(self.started), 2)
        self.assertEqual(self.live, {})
        self.assertEqual(base.read(self.out / 'tasks/task_1/status.json')['status'], 'complete')
        self.assertEqual(base.read(self.out / 'controller_status.json')['status'], 'blocked')

    def test_stop_signal_drains_without_starting_more(self):
        self.gpus = self.gpus[:2]

        def launch(args, **kw):
            child = self.launch(args, **kw)
            os.kill(os.getpid(), signal.SIGTERM)
            return child

        self.run_pool(launch=launch)
        self.assertEqual(len(self.started), 1)
        self.assertEqual(self.live, {})
        self.assertEqual(base.read(self.out / 'controller_status.json')['status'], 'stopped_after_drain')

    def test_controller_and_gpu_locks_prevent_overlap(self):
        import fcntl
        with (self.out / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_pool()
        with patch.object(base, 'ROOT', self.root):
            first = pool.take_gpu_lock('GPU-0')
            try:
                self.assertIsNone(pool.take_gpu_lock('GPU-0'))
            finally:
                first.close()
        self.assertEqual(self.started, [])

    def test_launch_recheck_rejects_gpu_that_became_busy(self):
        calls = 0
        self.gpus = self.gpus[:1]

        def snapshot():
            nonlocal calls
            calls += 1
            return [dict(self.gpus[0], util=100 if calls == 4 else 0)]

        self.run_pool(specs=self.specs[:1], snapshot=snapshot)
        self.assertEqual(len(self.started), 1)
        self.assertGreaterEqual(calls, 8)

    def test_invalid_capacity_identity_or_existing_evidence_rejected(self):
        for capacity in (0, 9):
            with self.assertRaises(ValueError):
                self.run_pool(max_gpus=capacity)
        base.write(self.out / 'manifest.json', dict(fingerprint='changed'))
        with self.assertRaisesRegex(ValueError, 'manifest'):
            self.run_pool()
        self.assertEqual(self.started, [])
        path = self.out / 'kept'
        path.write_text('before')
        hashes = {'kept': pool.file_sha(path)}
        path.write_text('after')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            pool.verify_retained(self.out, hashes)

    def test_gpu_query_failure_drains_running_tasks(self):
        def snapshot():
            if self.started:
                raise RuntimeError('nvidia query failed')
            return self.gpus[:1]

        with self.assertRaisesRegex(RuntimeError, 'nvidia query failed'):
            self.run_pool(snapshot=snapshot)
        self.assertEqual(len(self.started), 1)
        self.assertEqual(self.live, {})

    def test_real_subprocesses_overlap_on_distinct_cards_and_keep_env(self):
        import subprocess
        import sys
        real_popen = subprocess.Popen
        self.gpus = self.gpus[:2]
        self.policy['gpu_admission']['sample_interval_seconds'] = .02
        script = (
            'import json,os,pathlib,sys,time; '
            'folder=pathlib.Path(sys.argv[1]); folder.mkdir(parents=True); '
            'start=time.time(); time.sleep(.3); '
            'data=dict(status="complete",start=start,end=time.time(),'
            'gpu=os.environ["CUDA_VISIBLE_DEVICES"],cublas=os.environ["CUBLAS_WORKSPACE_CONFIG"]); '
            '(folder/"status.json").write_text(json.dumps(data))'
        )

        def launch(args, **kwargs):
            return real_popen([sys.executable, '-c', script, str(self.out / 'tasks' / args[-1])], **kwargs)

        self.run_pool(specs=self.specs[:3], launch=launch)
        records = [base.read(self.out / 'tasks' / s['task_id'] / 'status.json') for s in self.specs[:3]]
        self.assertNotEqual(records[0]['gpu'], records[1]['gpu'])
        self.assertLess(max(r['start'] for r in records[:2]), min(r['end'] for r in records[:2]))
        earlier = next(r for r in records[:2] if r['gpu'] == records[2]['gpu'])
        self.assertGreater(records[2]['start'], earlier['end'])
        self.assertTrue(all(r['cublas'] == ':4096:8' for r in records))
        self.assertEqual(base.read(self.out / 'controller_status.json')['status'], 'panel_terminal')


if __name__ == '__main__':
    unittest.main()
