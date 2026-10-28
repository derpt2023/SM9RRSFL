from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import recover_cifar_h84_initial_oom as recovery


class RecoveryTests(unittest.TestCase):
    def evidence(self):
        spec=dict(task_id=recovery.TASK,arm='VERT',config=dict(malicious_ratio=0))
        identity=dict(fingerprint='f')
        state=dict(status='failed',completed_round=0,error_type='OutOfMemoryError',fingerprint='f',pid=10)
        failure=dict(error_type='OutOfMemoryError',last_completed_round=0,task_id=recovery.TASK,
                     audit=None,error='CUDA out of memory. Tried to allocate 14 MiB')
        rows=[dict(round=0,accuracy=.1,source_transfer_asr=.2)]
        events=[dict(event='worker_started',task=recovery.TASK,child_pid=10,
                     admission=dict(uuid='GPU6',free_mb=17287)),
                dict(event='worker_exited',task=recovery.TASK,child_pid=10,gpu_uuid='GPU6',status='failed')]
        return state,failure,rows,events,spec,identity

    def test_round_zero_cuda_oom_is_supported_independent_of_metric_values(self):
        args=list(self.evidence())
        self.assertEqual(recovery.classify(*args)['uuid'],'GPU6')
        args[2][0].update(accuracy=1.,source_transfer_asr=0.)
        self.assertEqual(recovery.classify(*args)['uuid'],'GPU6')

    def test_partial_numerical_later_round_or_other_task_is_not_retried(self):
        cases=[(0,dict(status='partial')),(0,dict(completed_round=1)),(1,dict(error_type='ValueError')),
               (1,dict(audit=dict(nonfinite=True))),(4,dict(task_id='H84_iid_clean_s2026102801'))]
        for index,change in cases:
            with self.subTest(change=change),self.assertRaises(ValueError):
                args=list(self.evidence());args[index].update(change);recovery.classify(*args)

    def test_duplicate_or_mismatched_launch_is_not_accepted(self):
        args=list(self.evidence());args[3].append(deepcopy(args[3][0]))
        with self.assertRaises(ValueError):recovery.classify(*args)

    def test_apply_preserves_all_bytes_and_never_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'outputs/formal';task=out/'tasks'/recovery.TASK;task.mkdir(parents=True)
            recovery.formal.write(task/'status.json',dict(status='failed'))
            (task/'checkpoint.pickle').write_bytes(b'original round zero checkpoint')
            recovery.formal.write(out/'controller_status.json',dict(status='blocked',active={}))
            recovery.formal.write(out/'reference_lock.json',dict(immutable=True))
            for name in recovery.OWN:
                path=root/name;path.parent.mkdir(exist_ok=True);path.write_text('immutable recovery')
            originals=recovery.inventory(task)
            (out/'old_worker.log').write_text('original worker error')
            (out/'old_scheduler.log').write_text('original scheduler event')
            record=dict(first_failure_sha256='reviewed',failed_attempt_sha256=originals,
                original_manifest_fingerprint='f',quarantine_gpu_uuid='GPU6',
                original_worker_log='old_worker.log',original_worker_log_sha256=recovery.file_sha(out/'old_worker.log'),
                scheduler_events='old_scheduler.log',scheduler_events_sha256=recovery.file_sha(out/'old_scheduler.log'))
            with patch.object(recovery,'inspect',return_value=({},[dict(task_id=recovery.TASK)],dict(fingerprint='f'),out,record)),patch.object(recovery.formal,'ROOT',root):
                recovery.apply(Path('config'),'reviewed')
                self.assertFalse(task.exists())
                self.assertEqual(recovery.inventory(out/'recovery'/recovery.RECOVERY/'failed_attempt'),originals)
                recovery.verify_recovery(out,dict(fingerprint='f'))
                with self.assertRaises((ValueError,FileExistsError)):
                    recovery.apply(Path('config'),'reviewed')

    def test_active_workers_block_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);recovery.formal.write(out/'controller_status.json',dict(status='draining_after_failure',active={'other':{}}))
            with patch.object(recovery,'inspect',return_value=({},[],{},out,dict(first_failure_sha256='f'))):
                with self.assertRaisesRegex(ValueError,'draining'):recovery.apply(Path('cfg'),'f')
            self.assertFalse((out/'recovery').exists())


if __name__=='__main__':unittest.main()
