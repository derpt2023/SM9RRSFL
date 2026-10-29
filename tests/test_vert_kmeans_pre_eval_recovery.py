from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import recover_vert_kmeans_pre_eval_oom as r


class RecoveryTests(unittest.TestCase):
    def evidence(self):
        state=dict(status='failed',completed_round=-1,error_type='OutOfMemoryError',fingerprint='fixed',pid=66508)
        failure=dict(error_type='OutOfMemoryError',last_completed_round=-1,task_id=r.TASK,audit=None,
                     error='CUDA out of memory',traceback='initial_accuracy = _evaluate_accuracy')
        events=[dict(event='worker_started',task=r.TASK,child_pid=66508,admission=dict(free_mb=16634,uuid='gpu3')),
                dict(event='worker_exited',task=r.TASK,child_pid=66508,status='failed',gpu_uuid='gpu3')]
        spec=dict(task_id=r.TASK,arm='VERT',config=dict(malicious_ratio=.2))
        return [state,failure,[],events,spec,dict(fingerprint='fixed')]

    def test_only_actual_pre_eval_failure_passes(self):
        self.assertEqual(r.classify(*self.evidence())['uuid'],'gpu3')
        for index,key,value in [(0,'completed_round',0),(0,'status','partial'),(1,'error_type','RuntimeError'),
                                (1,'traceback','training'),(1,'audit',{}),(4,'task_id','other')]:
            e=self.evidence();e[index][key]=value
            with self.assertRaises(ValueError):r.classify(*e)
        e=self.evidence();e[2]=[dict(round=0)]
        with self.assertRaises(ValueError):r.classify(*e)

    def test_requires_actual_matching_launch_and_exit(self):
        for mutate in [lambda e:e[3].pop(), lambda e:e[3].append(deepcopy(e[3][0])),
                       lambda e:e[3][0]['admission'].update(free_mb=8000),
                       lambda e:e[3][1].update(child_pid=2),lambda e:e[3][1].update(gpu_uuid='other')]:
            e=self.evidence();mutate(e)
            with self.assertRaises(ValueError):r.classify(*e)

    def test_unreviewed_digest_is_rejected_before_mutation(self):
        with patch.object(r,'inspect',return_value=(None,None,None,None,dict(first_failure_sha256=r.FAILURE_SHA))):
            with self.assertRaises(ValueError):r.apply(None,'unreviewed')

    def test_retained_bytes_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);d=out/'recovery'/r.RECOVERY;f=d/'failed_attempt';f.mkdir(parents=True)
            (f/'status.json').write_text('original')
            (out/'reference_lock.json').write_text('reference')
            (d/'original_worker_log.log').write_text('worker')
            (d/'scheduler_events.log').write_text('events')
            record=dict(original_manifest_fingerprint='x',failed_attempt_sha256=r.inventory(f),
                reference_lock_sha256=r.file_sha(out/'reference_lock.json'),recovery_source_sha256={},
                original_worker_log_sha256=r.file_sha(d/'original_worker_log.log'),
                scheduler_events_sha256=r.file_sha(d/'scheduler_events.log'))
            r.formal.write(d/'review.json',record);r.formal.write(d/'applied.json',dict(applied=True))
            self.assertEqual(r.verify_recovery(out,dict(fingerprint='x')),record)
            (f/'status.json').write_text('changed')
            with self.assertRaises(ValueError):r.verify_recovery(out,dict(fingerprint='x'))

    def test_duplicate_continuation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'recovery'/r.RECOVERY;d.mkdir(parents=True);(d/'continuation.json').write_text('{}')
            with patch.object(r.formal,'plan',return_value=(dict(output_dir=tmp),[],{})),patch.object(r,'verify_recovery',return_value={}):
                with self.assertRaises(ValueError):r.continue_run(None)

if __name__=='__main__':unittest.main()
