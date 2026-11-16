"""Statistical completeness, candidate coverage and one-shot archive safeguards."""
from pathlib import Path
import tempfile
import unittest
import zipfile

import hfcdr_validation_protocol as p
import hfcdr_validation_selection as s
import render_hfcdr_defense_validation as r
import finish_hfcdr_defense_validation as f


class DeliveryTests(unittest.TestCase):
    def test_missing_replicate_not_imputed_and_sample_sd(self):
        self.assertIsNone(r.scalar([.8], 2)['mean'])
        self.assertEqual(r.scalar([.8], 2)['n'], 1)
        stats = r.scalar([1., 3.], 2)
        self.assertEqual(stats['mean'], 2.)
        self.assertAlmostEqual(stats['sd'], 2**.5)

    def test_all_22_candidates_have_identical_panel_coverage_and_health_retained(self):
        m = p.plan(s.SPEC); summaries = {}
        for t in m['tasks']:
            summaries[t['task_id']] = dict(task_id=t['task_id'], healthy=t['config']['method'] != 'tad',
                tail10=dict(accuracy=.8, asr=.2), timings=dict(runtime_seconds=30.))
        total = 0
        for d in m['policy']['datasets']:
            for part in m['policy']['partitions']:
                data = r.plot_data(m, dict(summaries=summaries), d, part, 'accuracy')
                self.assertEqual(len(data['panels']), 7)
                for panel in data['panels']:
                    self.assertEqual(len(panel['cells']), 11)
                    for cell in panel['cells']:
                        self.assertEqual(cell['stats']['mean'], 80.)
                        self.assertEqual(cell['stats']['n'], 2)
                        self.assertEqual(cell['healthy_n'], 0 if cell['method'] == 'tad' else 2)
                        total += len(cell['tasks'])
        self.assertEqual(total, 616)

    def test_controller_identifies_defense_entry_not_attack_entry(self):
        self.assertTrue(f.matching_command(['/usr/bin/python3', 'run_hfcdr_defense_selection.py', '--batch'], p.ROOT))
        self.assertFalse(f.matching_command(['/usr/bin/python3', 'run_hfcdr_validation.py', '--batch'], p.ROOT))
        self.assertFalse(f.matching_command(['/usr/bin/python3', 'run_hfcdr_defense_selection.py', '--select'], p.ROOT))

    def test_archive_is_hashed_excludes_live_files_and_preserves_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'selection.json').write_text('{"confirmation_allowed":false}')
            (root/'finalizer.log').write_text('live')
            receipt = f.package(root)
            self.assertEqual(receipt['files'], 1)
            self.assertEqual(receipt['sha256'], p.sha(receipt['archive']))
            with zipfile.ZipFile(receipt['archive']) as archive:
                self.assertEqual(set(archive.namelist()), {'selection.json', 'delivery_manifest.json'})
            with self.assertRaises(FileExistsError): f.package(root)


if __name__ == '__main__': unittest.main()
