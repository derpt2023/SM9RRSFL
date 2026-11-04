import copy
import math
import statistics
import unittest
import xml.etree.ElementTree as ET
import render_cifar_timing as r


def matrix():
    rows=[]
    for i,(s,p,q,m) in enumerate((s,p,q,m) for s in r.SEEDS for p in r.PARTITIONS for q in r.RATIOS for m in r.METHODS):
        total=100.+i; crypto=30. if m=='sm9rrs' else 0.
        rows.append(dict(task_id=f'{s}-{p}-{q}-{m}',seed=s,partition=p,malicious_ratio=q,method=m,
                         runtime_seconds=total,runtime_without_crypto_seconds=total-crypto,
                         crypto_wall_seconds=crypto,checkpoint_io_seconds=5.,health_reasons=['retained'] if i==0 else []))
    return rows


class TimingRenderTests(unittest.TestCase):
    def test_matrix_rejects_missing_duplicate_and_nonfinite(self):
        rows=matrix();r.validate_rows(rows)
        for bad in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):r.validate_rows(bad)
        for value in (math.nan,-1.):
            bad=copy.deepcopy(rows);bad[0]['runtime_seconds']=value
            with self.assertRaises(ValueError):r.validate_rows(bad)

    def test_crypto_subtraction_is_checked_without_deducting_io_twice(self):
        rows=matrix();r.validate_rows(rows)
        bad=copy.deepcopy(rows);bad[0]['runtime_without_crypto_seconds']-=5.
        with self.assertRaises(ValueError):r.validate_rows(bad)

    def test_sample_sd_and_health_failure_are_preserved(self):
        rows=matrix();group=r.aggregate(rows)
        self.assertEqual(len(group),60)
        a=next(g for g in group if g['method']=='sm9rrs' and g['partition']=='iid' and g['malicious_ratio']==0)
        self.assertEqual(a['n'],3);self.assertEqual(a['runtime_seconds']['mean'],160.)
        self.assertEqual(a['runtime_seconds']['sample_sd'],60.)
        self.assertEqual(rows[0]['health_reasons'],['retained'])

    def test_every_chart_bar_exactly_matches_seed_or_mean(self):
        rows=matrix()
        for seed in r.SEEDS+(None,):
            for metric in r.METRICS[:2]:
                svg=r.chart(rows,'iid',metric,400.,seed);root=ET.fromstring(svg)
                metadata=r.json.loads(root.find('{*}metadata').text)
                bars=[b for b in root.findall('{*}rect') if b.get('data-value')]
                self.assertEqual(len(bars),30)
                for bar,item in zip(bars,metadata['bars']):
                    source=[x[metric] for x in rows if x['task_id'] in item['tasks']]
                    self.assertEqual(float(bar.get('data-value')),statistics.mean(source))
                    self.assertEqual(item['n'],3 if seed is None else 1)
                    self.assertEqual(bar.get('fill'),r.style.METHOD_COLORS[item['method']])
                self.assertEqual(metadata['definition'],r.NOTE)

if __name__=='__main__':unittest.main()
