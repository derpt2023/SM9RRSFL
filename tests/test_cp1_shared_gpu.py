import unittest
from unittest.mock import patch

import run_cifar_cp1_cuda_development as base
import run_cifar_cp1_shared_gpu as shared
from tests import test_cp1_dynamic_gpu as dynamic_tests


class SharedPoolTests(dynamic_tests.PoolTests):
    """Run the original concurrency/drain/real-process safety suite on this adapter."""
    def setUp(self):
        patcher = patch.object(dynamic_tests, 'pool', shared)
        patcher.start()
        self.addCleanup(patcher.stop)
        super().setUp()

    def test_busy_compute_is_admitted_only_with_enough_memory(self):
        self.policy = shared.shared_parameters(self.policy)
        self.policy['gpu_admission']['sample_interval_seconds'] = 0
        self.gpus = [dict(g, util=100, free_mb=9000 if g['index'] in (0, 2) else 8000)
                     for g in self.gpus]
        self.run_pool(specs=self.specs[:4])
        self.assertEqual(self.peak, 2)
        self.assertEqual({gpu for _, gpu in self.started}, {'GPU-0', 'GPU-2'})
        policy = base.read(self.out / 'scheduler_shared_v1/policy.json')
        self.assertTrue(policy['shared_gpu_authorized'])
        self.assertTrue(policy['utilization_is_advisory'])
        self.assertEqual(policy['gpu_admission']['min_free_mb'], 8192)

    def test_previous_scheduler_audit_is_untouched(self):
        path = self.out / 'scheduler_dynamic_v1/policy.json'
        base.write(path, {'original': 'immutable prior admission'})
        before = path.read_bytes()
        self.run_pool(specs=self.specs[:1])
        self.assertEqual(path.read_bytes(), before)
        self.assertTrue((self.out / 'scheduler_shared_v1/policy.json').exists())


class ScientificIdentityTests(unittest.TestCase):
    def test_only_runtime_admission_changes_original_protocol_not_mutated(self):
        p, specs, manifest = base.plan(base.DEFAULT)
        original_hash = base.digest(p)
        runtime = shared.shared_parameters(p)
        self.assertEqual(base.digest(p), original_hash)
        self.assertEqual(runtime['gpu_admission'], shared.SHARED_ADMISSION)
        self.assertEqual({k: v for k, v in runtime.items() if k != 'gpu_admission'},
                         {k: v for k, v in p.items() if k != 'gpu_admission'})
        self.assertEqual(base.specs_for(runtime), specs)
        self.assertEqual(manifest['protocol'], p)
        base.verify(manifest)


if __name__ == '__main__':
    unittest.main()
