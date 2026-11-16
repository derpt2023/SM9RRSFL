"""Plot statistics and numeric artifact validation, without any experiment run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import render_hfcdr_attack_study as r


def sample(seed):
    return dict(task_id=f'fixture{seed}',config=dict(rounds=50),
        records=[dict(round=i,accuracy=.5+seed*.01,asr=.05+seed*.01) for i in range(51)],
        timings=dict(runtime_seconds=100+seed),memory=dict(cuda_peak_allocated_mib=1000+seed))


class VisualizationTests(unittest.TestCase):
    def test_mean_and_sample_sd_are_across_complete_seeds(self):
        result=r.curve_statistics([sample(i) for i in (1,2,3)],'accuracy',3)
        np.testing.assert_allclose(result['mean'],52.)
        np.testing.assert_allclose(result['sd'],1.)
        self.assertEqual(result['rounds'],list(range(51)))
        self.assertIsNone(r.curve_statistics([sample(1),sample(2)],'accuracy',3))

    def test_incomplete_horizon_never_interpolated(self):
        results=[sample(i) for i in (1,2,3)];results[1]['records'].pop()
        with self.assertRaises(ValueError):r.curve_statistics(results,'asr',3)

    def test_zero_percent_references_same_clean_run(self):
        result=sample(1)
        mapping={('mnist','iid','clean',0.,'hfcdr',1):result}
        for condition in r.ATTACK_LABELS:
            found=r.lookup(mapping,'mnist','iid',condition,0.,'hfcdr',[1])
            self.assertIs(found[0],result)

    def test_saved_svg_and_json_contain_exact_values(self):
        mapping={('mnist','iid','clean' if ratio==0 else 'label_flip',ratio,m,seed):sample(seed)
                 for ratio in r.RATIOS for m in r.METHODS for seed in (1,2,3)}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            name=r.curves(mapping,folder,'mnist','iid','label_flip','mean','asr',[1,2,3])
            sidecar=json.loads((folder/(name+'.plot.json')).read_text())
            self.assertEqual(len(sidecar['series']),30)
            self.assertEqual({s['ratio'] for s in sidecar['series']},set(r.RATIOS))
            for s in sidecar['series']:
                np.testing.assert_allclose(s['stats']['mean'],7.)
            svg=ET.parse(folder/(name+'.svg')).getroot()
            embedded=next(x.text for x in svg.iter() if x.tag.endswith('description'))
            self.assertEqual(json.loads(embedded),sidecar)
            self.assertGreater((folder/(name+'.png')).stat().st_size,10000)

    def test_missing_resource_cells_are_not_zero_measurements(self):
        mapping={('mnist','iid','clean',0.,'hfcdr',1):sample(1)}
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            name=r.resources(mapping,folder,'mnist','iid','scale2',1,[1,2,3])
            evidence=json.loads((folder/(name+'.plot.json')).read_text())['bars']
            self.assertEqual(sum(x['plotted'] for x in evidence),2)
            self.assertTrue(all(x['values']==[] for x in evidence if not x['plotted']))


if __name__=='__main__':unittest.main()
