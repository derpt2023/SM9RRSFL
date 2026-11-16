"""Completion never trains, and confirmation reports keep n=1 explicit."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import finish_hfcdr_confirmed_phase as f
import render_hfcdr_confirmation as r
import hfcdr_validation_protocol as p
from tests.test_hfcdr_confirmation import fixture


class ConfirmedDeliveryTests(unittest.TestCase):
    def test_controller_identity_is_phase_specific(self):
        for program in ('run_hfcdr_confirmation.py','run_hfcdr_formal.py'):
            self.assertTrue(f.matching_command(['/usr/bin/python3',program,'--batch'],p.ROOT,program))
            self.assertFalse(f.matching_command(['/usr/bin/python3',program,'--plan'],p.ROOT,program))
        self.assertFalse(f.matching_command(['/usr/bin/python3','run_hfcdr_defense_selection.py','--batch'],p.ROOT,'run_hfcdr_confirmation.py'))

    def test_frozen_decision_accepts_identical_evidence_but_never_replaces_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'confirmation.json';decision=dict(formal_allowed=False)
            f.freeze_decision(path,decision);before=path.read_bytes();f.freeze_decision(path,decision)
            self.assertEqual(path.read_bytes(),before)
            with self.assertRaises(ValueError):f.freeze_decision(path,dict(formal_allowed=True))
            self.assertEqual(path.read_bytes(),before)

    def test_phase_archives_are_distinct_hashed_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'result.json').write_text('{"fixture":true}')
            for phase in ('confirmation','formal'):
                receipt=f.package(root,phase)
                self.assertIn(phase,Path(receipt['archive']).name)
                self.assertEqual(receipt['sha256'],p.sha(receipt['archive']))
                with zipfile.ZipFile(receipt['archive']) as z:self.assertEqual(set(z.namelist()),{'result.json','delivery_manifest.json'})
                with self.assertRaises(FileExistsError):f.package(root,phase)

    def test_incomplete_confirmation_is_n1_validation_and_cannot_claim_qualification(self):
        manifest,_=fixture();summary=dict(complete=False,completed=0,summaries={})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();p.write(root/'manifest.json',manifest)
            with patch.object(p,'safe_output',return_value=root),patch.object(p,'check'),\
                 patch.object(r,'report',return_value=summary),patch.object(r,'curves',return_value=None),\
                 patch.object(r,'resources',return_value=None):
                result=r.render(root,root/'report')
            self.assertEqual(result['statistical_replicates'],1);self.assertFalse(result['official_test_used'])
            self.assertFalse(p.read(root/'report/confirmation.json')['formal_allowed'])
            self.assertIn('n=1',(root/'report/report.html').read_text())


if __name__=='__main__':unittest.main()
