"""Completion pipeline safeguards; no processes or training are launched."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import finish_hfcdr_attack_study as f
import hfcdr_study_protocol as p


class FinishTests(unittest.TestCase):
    def test_live_or_disappeared_controller_not_marked_complete(self):
        c=dict(status='training',active=['task'])
        self.assertEqual(f.terminal_state(c,True),'waiting')
        self.assertEqual(f.terminal_state(c,False),'unexpected_controller_exit')
        self.assertEqual(f.terminal_state(dict(status='complete',active=[]),False),'terminal')
        self.assertEqual(f.terminal_state(dict(status='incomplete',active=[]),False),'terminal')

    def test_original_preservation_detects_change_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'original.json';path.write_text('original')
            expected={'original.json':dict(bytes=path.stat().st_size,sha256=p.sha(path))}
            self.assertTrue(f.preservation(root,expected)['unchanged'])
            path.write_text('changed')
            self.assertEqual(f.preservation(root,expected)['changed'],['original.json'])
            path.unlink()
            self.assertEqual(f.preservation(root,expected)['missing'],['original.json'])

    def test_delivery_is_hashed_self_excluding_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'manifest.json').write_text('{"fixture":true}')
            (root/'finalizer.log').write_text('live log excluded')
            receipt=f.package(root)
            self.assertEqual(receipt['files'],1)
            archive=Path(receipt['archive']);self.assertEqual(receipt['sha256'],p.sha(archive))
            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(set(bundle.namelist()),{'manifest.json','delivery_manifest.json'})
                inventory=json.loads(bundle.read('delivery_manifest.json'))
                self.assertEqual(inventory['manifest.json']['sha256'],p.sha(root/'manifest.json'))
            with self.assertRaises(FileExistsError):f.package(root)


if __name__=='__main__':unittest.main()
