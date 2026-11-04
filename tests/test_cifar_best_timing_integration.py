import copy,json,unittest
from pathlib import Path
import cifar_best_timing_integration as c
import integrate_cifar_best_timing as integration

ROOT=Path(__file__).resolve().parents[1]

class MainTimingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=ROOT/'outputs/cifar-best';cls.report=json.loads((p/'export/combined_bundle.json').read_text())['report']
        cls.payload=integration.make_payload(ROOT/'outputs/old/cifar_timing_corrected_complete_20261104/raw')
        cls.summary=json.loads((p/'final_evaluation/summary.json').read_text())

    def test_all_performance_and_health_values_survive_overlay(self):
        before=copy.deepcopy(self.report);after=c.overlay(before,self.payload)
        self.assertEqual(integration.performance(before),integration.performance(after));self.assertEqual(before,self.report)
        for name,item in self.payload['cells'].items():
            self.assertEqual(after['cells'][name]['timing'],item['timing'])
            self.assertEqual(after['audited_cells'][name]['summary']['timing'],item['timing'])

    def test_r4_mapping_and_scenario_identity_cannot_be_swapped(self):
        for field,value in [('seed',2026102802),('malicious_ratio',.2),('method','fedavg'),('suspicion_recovery_factor',1.25)]:
            bad=copy.deepcopy(self.payload);bad['cells']['H84_iid_clean_s2026102801']['config'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):c.overlay(self.report,bad)

    def test_summary_changes_only_time_columns_and_tags(self):
        after=c.merge_summary(self.summary,self.payload)
        self.assertEqual(c.projection(after),c.projection(self.summary))
        self.assertEqual(c.projection_sha(after),c.projection_sha(self.summary))
        for row in after:
            item=self.payload['cells'][row['task_id']]
            for key in row:
                if key.endswith('_seconds'):
                    expected=item['summary']['checkpoint_io_seconds'] if key=='measured_checkpoint_write_seconds' else item['summary'][key]
                    self.assertEqual(row[key],expected)

    def test_all_sixty_runtime_statistics_come_from_new_three_seeds(self):
        after=c.overlay(self.report,self.payload)
        for name,g in after['groups'].items():
            self.assertEqual(g['samples']['accuracy'],self.report['groups'][name]['samples']['accuracy'])
            self.assertEqual(g['samples']['asr'],self.report['groups'][name]['samples']['asr'])
            self.assertEqual(g['samples']['runtime_with_crypto_seconds']['mean'],self.payload['aggregate'][name]['runtime_seconds']['mean'])

    def test_rejects_missing_task_and_wrong_source_digest(self):
        for kind in ('missing','digest'):
            bad=copy.deepcopy(self.payload)
            if kind=='missing':bad['cells'].pop(next(iter(bad['cells'])))
            else:bad['report_sha256']='0'*64
            with self.assertRaises(ValueError):c.overlay(self.report,bad)

if __name__=='__main__':unittest.main()
