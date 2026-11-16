"""Independent scheduler evidence checks; no scheduler or training is started."""
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import tempfile
import unittest

import audit_hfcdr_attack_study as a
import hfcdr_study_protocol as p


class AuditTests(unittest.TestCase):
    def events(self):
        manifest=p.plan();group=manifest['groups'][0];events=[]
        at=datetime(2026,11,16,tzinfo=timezone.utc)
        for i,tid in enumerate(group['tasks']):
            start=dict(kind='start',at=(at+timedelta(seconds=61*i)).isoformat(),
                task_id=tid,group_id=group['group_id'],pid=i+100,gpu=2,
                admission_elapsed_seconds=5.1,admission_before=dict(free_mib=9000),
                admission_after=dict(free_mib=8900))
            events.extend([start,dict(kind='exit',task_id=tid,pid=i+100,gpu=2)])
        return manifest,events

    def audit(self,manifest,events,suffix=''):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'scheduler_events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events)+suffix)
            return a.inspect_scheduler(path,manifest)

    def test_complete_unique_same_gpu_rotated_sequence(self):
        manifest,events=self.events();r=self.audit(manifest,events)
        self.assertEqual((r['started'],r['exited'],r['peak_workers']),(6,6,1))
        self.assertEqual(r['minimum_admission_free_mib'],8900)

    def test_repeated_task_or_wrong_order_refused(self):
        manifest,events=self.events()
        with self.assertRaises(ValueError):self.audit(manifest,events+[events[0]])
        manifest,events=self.events();events[0]['task_id']=manifest['groups'][0]['tasks'][1]
        with self.assertRaises(ValueError):self.audit(manifest,events)

    def test_gpu_migration_or_memory_admission_or_spacing_refused(self):
        for mutation in ('gpu','memory','spacing'):
            manifest,events=self.events()
            if mutation=='gpu':events[2]['gpu']=3
            if mutation=='memory':events[2]['admission_after']['free_mib']=8191
            if mutation=='spacing':events[2]['at']=events[0]['at']
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.audit(manifest,events)

    def test_exit_must_match_real_started_pid(self):
        manifest,events=self.events();events[1]['pid']=999
        with self.assertRaises(ValueError):self.audit(manifest,events)

    def test_live_partial_last_event_is_not_fabricated(self):
        manifest,events=self.events();r=self.audit(manifest,events,suffix='{"kind":"start"')
        self.assertEqual(r['started'],6)


if __name__=='__main__':unittest.main()
