"""Bounded explicit OOM continuation, no training or GPU calls."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_cifar_timing as t
import resume_cifar_timing as f


def original_fixture(root):
    old=root/'original'
    old.mkdir()
    m=t.manifest(Path('/tmp/data'))
    t.base.write(old/'manifest.json',m)
    for block,uuid in zip(m['blocks'][:4],sorted(f.ALLOWED_UUIDS)):
        task=block['tasks'][0]
        folder=old/'tasks'/task['task_id']
        t.base.write(folder/'result.json',dict(runtime_seconds=10,runtime_without_crypto_seconds=8,
            crypto_wall_seconds=2,checkpoint_io_seconds=1))
        t.base.write(folder/'status.json',dict(status='complete',health_reasons=[],
            artifacts={'result.json':t.base.sha(folder/'result.json')}))
        t.event(old,'start',task_id=task['task_id'],block_id=block['block_id'],uuid=uuid)
        t.event(old,'exit',task_id=task['task_id'],uuid=uuid,returncode=0)
    t.base.write(old/'tasks'/f.FAILED_TASK/'status.json',dict(status='failed',error="OutOfMemoryError('test')"))
    t.event(old,'start',task_id=f.FAILED_TASK,block_id='block_04',uuid='gpu1')
    t.event(old,'exit',task_id=f.FAILED_TASK,uuid='gpu1',returncode=1)
    log=old/'logs'/(f.FAILED_TASK+'.log')
    log.parent.mkdir();log.write_text('original OOM')
    t.base.write(old/'analysis/timing_results.json',{'counts':{'complete':4,'failed':1,'pending':175}})
    return old,m


class FollowupTests(unittest.TestCase):
    def test_exact_scope_and_original_gpu_assignment(self):
        with tempfile.TemporaryDirectory() as temp:
            old,m=original_fixture(Path(temp))
            _,plans,completed,_=f.continuation_plan(old)
            self.assertEqual(len(completed),4)
            self.assertEqual(sum(6-p['position'] for p in plans),176)
            self.assertEqual([p['position'] for p in plans[:5]],[1,1,1,1,0])
            self.assertEqual({p['required_uuid'] for p in plans[:4]},f.ALLOWED_UUIDS)
            self.assertTrue(all(p['required_uuid'] is None for p in plans[4:]))

    def test_different_failure_or_additional_attempt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            old,m=original_fixture(Path(temp))
            path=old/'tasks'/f.FAILED_TASK/'status.json'
            t.base.write(path,dict(status='failed',error='Different error'))
            with self.assertRaises(ValueError):f.continuation_plan(old)
            t.base.write(path,dict(status='failed',error='OutOfMemoryError'))
            t.event(old,'start',task_id=f.FAILED_TASK,block_id='block_04',uuid='gpu1')
            with self.assertRaises(ValueError):f.continuation_plan(old)

    def test_reuse_preserves_old_files_and_has_new_empty_retry_slot(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old,m=original_fixture(root)
            before={str(p.relative_to(old)):t.base.sha(p) for p in old.rglob('*') if p.is_file()}
            out=root/'followup'
            f.prepare(old,out,f.own_sources())
            after={str(p.relative_to(old)):t.base.sha(p) for p in old.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            self.assertEqual(t.base.read(out/'manifest.json'),m)
            self.assertFalse((out/'tasks'/f.FAILED_TASK).exists())
            self.assertEqual(len(list((out/'tasks').iterdir())),4)
            self.assertEqual((out/'provenance/original_oom.log').read_text(),'original OOM')
            self.assertEqual(len((out/'events.jsonl').read_text().splitlines()),8)
            with self.assertRaises(FileExistsError):f.prepare(old,out,f.own_sources())

    def test_tampered_completed_result_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            old,m=original_fixture(Path(temp))
            path=old/'tasks'/m['blocks'][0]['tasks'][0]['task_id']/'result.json'
            path.write_text('{}')
            with self.assertRaises(ValueError):f.continuation_plan(old)

    def test_incomplete_report_uses_actual_not_planned_attempt_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old,m=original_fixture(root);out=root/'followup'
            f.prepare(old,out,f.own_sources())
            with patch.object(t,'final_report',return_value={'execution_audit':{'failures':['incomplete']}}):
                report=f.finish(out,old,f.own_sources())
            self.assertEqual(report['continuation']['fresh_followup_attempts'],0)
            self.assertEqual(report['continuation']['total_attempts_including_original_oom'],5)
            self.assertEqual(report['continuation']['explicit_fresh_retry_count'],0)
            self.assertFalse(report['execution_audit']['passed'])

    def test_handoff_waits_for_workers_and_normal_final_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old,m=original_fixture(root)
            t.base.write(old/'controller_status.json',dict(pid=42,status='incomplete',active=[]))
            with patch.object(f,'HANDOFF_STATUS',root/'handoff.json'), \
                    patch.object(f,'running_original',side_effect=[[42,43],[]]), \
                    patch.object(f.time,'sleep') as sleep:
                f.wait_for_drain(old,42,f.own_sources())
                sleep.assert_called_once_with(15)
            t.base.write(old/'controller_status.json',dict(pid=42,status='error_draining',active=[]))
            with patch.object(f,'HANDOFF_STATUS',root/'handoff.json'), \
                    patch.object(f,'running_original',return_value=[]),self.assertRaises(ValueError):
                f.wait_for_drain(old,42,f.own_sources())

    def test_dispatches_only_remaining_tasks_on_allowed_cards(self):
        for failed in (False,True):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);out=root/'followup'
                m=t.manifest(Path('/tmp/data'));block=m['blocks'][0]
                uuid=sorted(f.ALLOWED_UUIDS)[0]
                def prepare(*args):
                    out.mkdir();t.base.write(out/'manifest.json',m)
                    return m,[dict(block=block,position=1,required_uuid=uuid)]
                calls=[]
                def child(command,**kwargs):
                    task_id=command[-1];calls.append((task_id,kwargs['env']['CUDA_VISIBLE_DEVICES']))
                    t.base.write(out/'tasks'/task_id/'status.json',{'status':'failed' if failed else 'complete'})
                    code=1 if failed else 0
                    return SimpleNamespace(pid=100+len(calls),returncode=code,poll=lambda:code,wait=lambda:code)
                snap={'gpus':[dict(index='1',uuid='gpu1',free=20000),dict(index='2',uuid=uuid,free=10000)]}
                with patch.object(f,'prepare',side_effect=prepare),patch.object(f,'assert_sources'), \
                        patch.object(t,'gpu_snapshot',return_value=snap),patch.object(f.time,'sleep'), \
                        patch.object(f.signal,'signal'),patch.object(f.subprocess,'Popen',side_effect=child), \
                        patch.object(t,'acquire_gpu',side_effect=lambda u:open(root/'lock','a')), \
                        patch.object(f,'finish',return_value={'counts':{},'execution_audit':{'passed':not failed}}):
                    self.assertEqual(f.run(root/'original',out,f.own_sources()),not failed)
                self.assertEqual(len(calls),1 if failed else 5)
                self.assertEqual({u for _,u in calls},{uuid})
                self.assertEqual(calls[0][0],block['tasks'][1]['task_id'])


if __name__=='__main__':
    unittest.main()
