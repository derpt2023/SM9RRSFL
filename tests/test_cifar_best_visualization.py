import copy
import json
import math
import unittest
import xml.etree.ElementTree as ET

import render_cifar_best as render


def timing():
    return dict(available_and_consistent=True, error=None,
                runtime_with_crypto_seconds=100., runtime_without_crypto_seconds=60.,
                crypto_wall_seconds=40., stages={
                    **dict.fromkeys(render.CRYPTO_SPANS, 10.),
                    'sign_seconds': 250., 'verify_seconds': 500.})


def metadata(svg):
    return json.loads(ET.fromstring(svg).find('{*}metadata').text)


class CifarBestVisualizationTests(unittest.TestCase):
    def test_parallel_operation_sums_are_never_subtracted(self):
        self.assertEqual(render.checked_timing(timing())['runtime_without_crypto_seconds'], 60.)

    def test_missing_inconsistent_or_nonfinite_timing_is_rejected(self):
        for key, value in [('runtime_without_crypto_seconds', 0.),
                           ('runtime_with_crypto_seconds', math.nan),
                           ('crypto_wall_seconds', -1.), ('available_and_consistent', False)]:
            t = timing(); t[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                render.checked_timing(t)
        t = timing(); t['stages'][render.CRYPTO_SPANS[0]] = 11.
        with self.assertRaises(ValueError):
            render.checked_timing(t)

    def test_official_asr_keeps_every_round_and_every_method(self):
        bundle = {'trajectories': {}}
        for arm in render.ARMS:
            for tag in render.RATIOS:
                bundle['trajectories'][f'{arm}_iid_{tag}_s1'] = [
                    dict(round=i, accuracy=i / 300, source_transfer_asr=i / (1000 if tag == 'clean' else 200),
                         attack_target_success_rate=.99) for i in range(151)]
        before = copy.deepcopy(bundle)
        for metric, count in [('accuracy', 30), ('source_transfer_asr', 30)]:
            svg = render.curve_plot(bundle, 1, 'iid', metric)
            data = metadata(svg)
            self.assertEqual(len(data['series']), count)
            self.assertEqual(len(ET.fromstring(svg).findall('.//{*}polyline')), count)
            for series in data['series']:
                self.assertEqual([p[0] for p in series['points']], list(range(151)))
                expected = (.15 if series['ratio'] == 0 else .75) if metric == 'source_transfer_asr' else .5
                self.assertEqual(series['points'][-1][1], expected)
            if metric == 'source_transfer_asr':
                self.assertNotIn('>测试准确率</text>', svg)
                self.assertNotIn('data-series="accuracy"', svg)
                self.assertEqual({s['ratio'] for s in data['series']}, {0., .2, .4, .6, .8})
                self.assertEqual(len([s for s in data['series'] if s['ratio'] == 0]), 6)
                self.assertIn('>背景转移率</text>', svg)
                self.assertIn('(a) 0% 恶意节点（背景转移率）', svg)
        self.assertEqual(bundle, before)

    def test_both_runtime_charts_have_all_thirty_exact_bars(self):
        report = {'cells': {f'{arm}_iid_{tag}_s1': {'timing': timing()}
                            for arm in render.ARMS for tag in render.RATIOS}}
        for metric, expected in [('runtime_with_crypto_seconds', 100.),
                                 ('runtime_without_crypto_seconds', 60.)]:
            svg = render.time_plot(report, 1, 'iid', metric, 130.)
            data = metadata(svg)
            self.assertEqual(len(data['bars']), 30)
            self.assertEqual({bar['value'] for bar in data['bars']}, {expected})
            self.assertEqual(data['y_max'], 130.)
            bars = [r for r in ET.fromstring(svg).findall('{*}rect') if r.get('data-task')]
            self.assertEqual(len(bars), 30)
            self.assertEqual({float(r.get('data-value')) for r in bars}, {expected})

    def test_unmeasured_memory_does_not_become_zero_memory_bars(self):
        svg = render.memory_plot(1, 'iid')
        self.assertIsNone(metadata(svg)['plotted_value'])
        self.assertFalse(metadata(svg)['available'])
        self.assertIn('未记录', svg)
        self.assertEqual(len(ET.fromstring(svg).findall('{*}rect')), 1)  # background only


if __name__ == '__main__':
    unittest.main()
