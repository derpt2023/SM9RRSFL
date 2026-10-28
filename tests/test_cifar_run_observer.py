import json
from pathlib import Path
import tempfile
import unittest

import watch_cifar_experiment as observer


class ObserverTests(unittest.TestCase):
    def test_parallel_operation_sums_are_not_deducted(self):
        summary = dict(runtime_seconds=10., crypto_wall_seconds=2.5,
                       runtime_without_crypto_seconds=7.5, sign_seconds=100., verify_seconds=200.)
        summary.update(dict(zip(observer.CRYPTO_SPANS, (.5, 1.5, .25, .25))))
        result = observer.timing(summary)
        self.assertTrue(result['available_and_consistent'])
        self.assertEqual(result['runtime_without_crypto_seconds'], 7.5)

    def test_missing_or_inconsistent_times_never_become_zero(self):
        result = observer.timing({})
        self.assertFalse(result['available_and_consistent'])
        self.assertIsNone(result['runtime_without_crypto_seconds'])
        s = dict(runtime_seconds=2., crypto_wall_seconds=3., runtime_without_crypto_seconds=0.)
        s.update(dict.fromkeys(observer.CRYPTO_SPANS, .75))
        self.assertEqual(observer.timing(s)['error'], 'inconsistent_crypto_wall_accounting')

    def test_running_missing_and_failed_cells_display_truthfully_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = dict(task_id='a', config=dict(rounds=150, method='sm9rrs', seed=1,
                        partition='iid', malicious_ratio=.2))
            (root / 'manifest.json').write_text(json.dumps(dict(fingerprint='f', specs=[spec, dict(spec, task_id='b')])))
            (root / 'controller_status.json').write_text(json.dumps(dict(status='training', active={'a':dict(gpu_index=3,gpu_uuid='G')})))
            task = root / 'tasks/a';task.mkdir(parents=True)
            (task / 'status.json').write_text(json.dumps(dict(status='running',completed_round=12,elapsed_seconds=4)))
            (task / 'metrics.json').write_text(json.dumps(dict(rounds=[dict(round=12,accuracy=.6,source_transfer_asr=.1)])))
            before = {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            state = observer.snapshot(root)
            self.assertEqual((state['complete'],state['total'],state['executed_rounds']), (0,2,12))
            self.assertIsNone(state['cells']['b']['accuracy'])
            self.assertIn('Acc 60.00% ASR 10.00%',observer.display(state,width=200))
            self.assertEqual(before,{p:p.read_bytes() for p in root.rglob('*') if p.is_file()})
            (task / 'status.json').write_text('{broken')
            self.assertEqual(observer.snapshot(root)['cells']['a']['status'],'unreadable')

    def test_missing_manifest_does_not_create_a_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):observer.snapshot(Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()),[])


if __name__ == '__main__':
    unittest.main()
