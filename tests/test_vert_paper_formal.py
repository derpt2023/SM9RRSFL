from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_cifar_vert_paper_formal as run
import cifar_vert_paper_formal_report as report


class FormalTests(unittest.TestCase):
    def test_same30_formal_cells_with_only_selected_config_changes_and_frozen100(self):
        p,specs,m=run.plan()
        _,old,old_manifest=run.previous.plan()
        self.assertEqual(len(specs),30);self.assertEqual(m['maximum_rounds'],4500)
        self.assertEqual(m['evaluation_scope'],'official_test');self.assertFalse(m['official_test_used_for_selection'])
        for s,o in zip(specs,old):
            expected=deepcopy(o);expected['config'].update(run.PARAMETERS)
            self.assertEqual(s,expected)
        self.assertEqual({s['config']['seed'] for s in specs},{2026102801,2026102802,2026102803})
        self.assertEqual(len(p['amendment']['validation_frozen_sources']),100)
        for name,h in p['amendment']['validation_frozen_sources'].items():self.assertEqual(m['sources'][name],h)
        for name,h in old_manifest['sources'].items():self.assertEqual(m['sources'][name],h)
        self.assertNotEqual(p['output_dir'],run.previous.plan()[0]['output_dir'])

    def test_refuses_new_learning_rate_seed_rank_or_test_retuning(self):
        original=run.read(run.DEFAULT)
        changes=[('parameters',dict(run.PARAMETERS,vert_predict_lr=.003)),('seeds',[1,2,3]),
                 ('rank_target',2),('retune_on_formal_test',True),('rounds',30),
                 ('quarantine_gpu_uuids',[]),('process_environment',{})]
        for key,value in changes:
            a=deepcopy(original);a[key]=value
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'bad.json';path.write_text(json.dumps(a))
                with self.assertRaises(ValueError):run.plan(path)

    def test_validation_evidence_digest_is_mandatory(self):
        a=run.read(run.DEFAULT)
        with patch.object(run,'sha',return_value='changed'):
            with self.assertRaises(ValueError):run.selected_validation(a)
        a['validation_report_sha256']='changed'
        with self.assertRaises(ValueError):run.selected_validation(a)

    def test_selected_validation_cannot_be_incomplete_or_official_test(self):
        a=run.read(run.DEFAULT);valid=run.read(run.ROOT/a['validation_report'])
        for key,value in [('status','incomplete'),('official_test_used_for_selection',True),
                          ('evaluation_scope','official_test'),('execution_audits',{})]:
            bad=deepcopy(valid);bad[key]=value
            with patch.object(run,'sha',return_value=run.VALIDATION_SHA),patch.object(run,'read',return_value=bad):
                with self.assertRaises(ValueError):run.selected_validation(a)
        bad=deepcopy(valid);next(iter(bad['cells'].values()))['complete_finite']=False
        with patch.object(run,'sha',return_value=run.VALIDATION_SHA),patch.object(run,'read',return_value=bad):
            with self.assertRaises(ValueError):run.selected_validation(a)

    def test_both_quarantines_and_cli_cap_are_kept(self):
        with patch.object(run.formal,'gpu_snapshot',return_value=[dict(uuid=x) for x in (*run.QUARANTINE,'GPU-ok')]):
            self.assertEqual(run.gpu_snapshot(),[dict(uuid='GPU-ok')])
        with patch('sys.argv',['run','--max-gpus','7']):
            with self.assertRaises(ValueError):run.main()

    def test_all180_merge_does_not_mix_missing_old_vert_and_other150_are_unchanged(self):
        from tests import test_vert_kmeans_supplement as original
        with patch.object(original,'report',report),patch.object(original,'run',run):
            original.SupplementTests('test_merge_requires_all_new_vert_and_preserves_other_five').test_merge_requires_all_new_vert_and_preserves_other_five()

    def test_actual_weights_audit_rejects_missing_identity_weights_nonfinite(self):
        decisions=[dict(task_id='t',round=i,weights={'a':1.}) for i in range(1,151)]
        actual=[dict(task_id='t',round=i,weights={'a':1.},honest_weight_loss=.1,malicious_weight_mass=0.) for i in range(1,151)]
        metrics=[dict(round=i,honest_weight_loss=.1,malicious_weight_mass=0.) for i in range(151)]
        self.assertTrue(report.audit_actual_weights(decisions,actual,metrics,'t'))
        with self.assertRaises(ValueError):report.audit_actual_weights(decisions,actual[:-1],metrics,'t')
        for key,value in [('task_id','other'),('weights',{'b':1.}),('honest_weight_loss',float('nan'))]:
            bad=deepcopy(actual);bad[20][key]=value
            with self.assertRaises(ValueError):report.audit_actual_weights(decisions,bad,metrics,'t')


from tests import test_cp1_dynamic_gpu as pool_suite
import cifar_vert_paper_formal_scheduler as pool


class FormalSchedulerTests(pool_suite.PoolTests):
    def setUp(self):
        for name,value in (('pool',pool),('base',run)):
            p=patch.object(pool_suite,name,value);p.start();self.addCleanup(p.stop)
        super().setUp()


if __name__=='__main__':unittest.main()
