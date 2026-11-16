"""Read-only validation audit and plot integrity regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_hfcdr_validation as a
import hfcdr_validation_protocol as p
import render_hfcdr_validation as r
from tests import test_hfcdr_study_audit as existing


class ValidationSchedulerTests(existing.AuditTests):
    # The event format is identical, but execute the independent new reader.
    def audit(self,manifest,events,suffix=''):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'scheduler_events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events)+suffix)
            return a.inspect_scheduler(path,manifest)


class PlotTests(unittest.TestCase):
    def test_two_seed_mean_and_sd_exact_and_missing_not_imputed(self):
        rows=[]
        for s,v in [(1,.4),(2,.6)]:
            rows.append(dict(task_id=str(s),config=dict(rounds=1),records=[dict(round=i,accuracy=v) for i in (0,1)]))
        stats=r.curve_statistics(rows,'accuracy',2)
        self.assertEqual(stats['mean'],[50.,50.])
        self.assertAlmostEqual(stats['sd'][0],14.142135623730951)
        self.assertIsNone(r.curve_statistics(rows[:1],'accuracy',2))

    def test_clean_reference_shared_and_plot_scope_is_validation(self):
        clean=dict(task_id='clean',config=dict(rounds=1),records=[dict(round=i,accuracy=.6) for i in (0,1)])
        lookup={('mnist','iid','clean',0.,1):clean}
        self.assertIs(r.series(lookup,'mnist','iid','scale5',0.,[1])[0],clean)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            name=r.draw(lookup,folder,'mnist','iid',[1],'1','accuracy')
            data=p.read(folder/(name+'.plot.json'))
            self.assertEqual(data['series'][0]['stats']['mean'],[60.,60.])
            self.assertEqual(data['seeds'],[1])
            self.assertIn('Validation accuracy',(folder/(name+'.svg')).read_text())
            self.assertNotIn('Official test',(folder/(name+'.svg')).read_text())


if __name__=='__main__':unittest.main()
