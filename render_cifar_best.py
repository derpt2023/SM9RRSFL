"""Redraw the accepted CIFAR-BEST package in the MNIST per-seed SVG layout.

Read-only inputs; the output directory must be new. No training or pickle loads.
The historical scientific renderer and visualization helpers remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import render_cifar_r4_formal as accepted
from sm9rrsfl import visualization as style
from watch_cifar_experiment import CRYPTO_SPANS

ROOT = Path(__file__).resolve().parent
SOURCE_SHA = '7a56eec00bd2ef936f342e1f2a22088b674cff11d86bcff1981176c5bb6a6198'
SUMMARY_SHA = 'da4a90c173f75efe226d6fc6cb748a51be181254d9324379b1816bbbe42787f6'
PARTITIONS = {
    'iid': ('iid', 'IID 独立同分布'),
    'dirichlet': ('dirichlet_alpha_0_5', 'Non-IID 非独立同分布（Dirichlet α=0.5）'),
}
ARMS = ('H84', 'VERT', 'AlignIns', 'TAD', 'Krum', 'FedAvg')
LABELS = dict(zip(ARMS, ('Ours', 'VERT', 'AlignIns', 'TAD', 'Krum', 'FedAvg')))
RATIOS = {'clean': 0., 'r20': .2, 'r40': .4, 'r60': .6, 'r80': .8}
TIMING_NOTE = ('耗时为共享 GPU 上的实测墙钟秒数，包含数据加载、评估、观测和断点 I/O。'
               '扣除密码协议 = 总耗时 − 四段互不重叠的密码协议墙钟时间；'
               '不扣除并行操作累计时间，不代表关闭密码学重跑或独占速度测试。')
SCOPE_NOTE = ('Ours 为固定 R4，其他五法逐项复用 cifar-paper。准确率使用官方 test 10000 张，'
              'ASR 使用其中 source 5 的 1000 张，非辅助目标 ASR。'
              '曲线保留全部 0–150 轮，未平滑、插值或按健康状态删样本；攻击从第 25 轮开始。'
              'ASR 图与 MNIST 一样展示四个攻击比例，clean 背景转移率仍保留在原表格中。')
SVG_NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG_NS)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + '\n', encoding='utf-8')


def checked_timing(timing):
    """Validate disjoint wall spans, never overlapping sign/verify worker sums."""
    keys = ('runtime_with_crypto_seconds', 'runtime_without_crypto_seconds', 'crypto_wall_seconds')
    numbers = [timing[k] for k in keys] + [timing['stages'][k] for k in CRYPTO_SPANS]
    if (not timing.get('available_and_consistent') or timing.get('error')
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or v < 0 for v in numbers)):
        raise ValueError('Missing or invalid wall-time evidence')
    total, excluding, crypto = numbers[:3]
    if (crypto > total or not math.isclose(sum(numbers[3:]), crypto, rel_tol=0, abs_tol=1e-6)
            or not math.isclose(total - crypto, excluding, rel_tol=0, abs_tol=1e-6)):
        raise ValueError('Crypto subtraction is inconsistent')
    return dict(zip(keys, numbers[:3]))


def load_inputs(package, expected_sha256):
    source = package / 'export/combined_bundle.json'
    baseline = package / 'provenance/accepted_cifar_paper_bundle.json'
    summary = package / 'final_evaluation/summary.json'
    files = {p: p.read_bytes() for p in (source, baseline, summary)}
    if expected_sha256 != SOURCE_SHA or sha(files[source]) != expected_sha256:
        raise ValueError('Exact accepted CIFAR-BEST bundle required')
    if sha(files[summary]) != SUMMARY_SHA:
        raise ValueError('Exact accepted summary required for memory evidence')
    bundle = json.loads(files[source])
    report = accepted.validate(bundle, files[baseline])
    summaries = json.loads(files[summary])
    by_id = {row['task_id']: row for row in summaries}
    if len(summaries) != 180 or set(by_id) != set(report['cells']):
        raise ValueError('Summary coverage differs')
    for name, cell in report['cells'].items():
        timing = checked_timing(cell['timing'])
        row = by_id[name]
        comparisons = {
            'final_accuracy': cell['final']['accuracy'],
            'final_attack_target_success_rate': cell['final']['source_transfer_asr'],
            'runtime_seconds': timing['runtime_with_crypto_seconds'],
            'runtime_without_crypto_seconds': timing['runtime_without_crypto_seconds'],
            'crypto_wall_seconds': timing['crypto_wall_seconds'],
        }
        if any(row[k] != value for k, value in comparisons.items()):
            raise ValueError('Summary differs from verified bundle: ' + name)
        if row['peak_memory_mb'] != 0:
            raise ValueError('Unexpected memory evidence; review the measurement definition')
    return bundle, report, {p: sha(raw) for p, raw in files.items()}


def curve_plot(bundle, seed, partition, metric):
    is_asr = metric == 'source_transfer_asr'
    title = PARTITIONS[partition][1] + ('：不同恶意节点比例下的攻击成功率（ASR）变化'
                                     if is_asr else '：不同恶意节点比例下的模型收敛/准确率对比')
    panels, evidence = [], []
    for tag, ratio in RATIOS.items():
        if is_asr and ratio == 0:
            continue
        series = {}
        for arm in ARMS:
            name = f'{arm}_{partition}_{tag}_s{seed}'
            points = [(r['round'], r[metric]) for r in bundle['trajectories'][name]]
            series[LABELS[arm]] = points
            evidence.append(dict(task=name, method=LABELS[arm], ratio=ratio, points=points))
        panels.append((ratio, series))
    svg = style._accuracy_comparison_chart(title=title, panels=panels, target_rounds=150)
    # Reuse the exact MNIST geometry/markers; only change metric labels and provenance.
    if is_asr:
        svg = svg.replace('data-series="accuracy"', 'data-series="source_transfer_asr"')
        svg = svg.replace('>测试准确率</text>', '>攻击成功率 (ASR)</text>')
    svg = re.sub(r'<desc>.*?</desc>', '<desc>' + escape(SCOPE_NOTE) + '</desc>', svg)
    return with_metadata(svg, dict(metric=metric, seed=seed, partition=partition, series=evidence))


def with_metadata(svg, data):
    text = escape(json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
    return svg.replace('</svg>', '<metadata id="plot-data">' + text + '</metadata>\n</svg>')


def time_plot(report, seed, partition, metric, y_max):
    without = metric == 'runtime_without_crypto_seconds'
    title = PARTITIONS[partition][1] + ('：训练/防御时间开销（扣除密码协议）'
                                     if without else '：端到端时间开销（含密码协议）')
    y_label = '扣除密码协议后的运行时间（秒）' if without else '端到端运行时间（秒）'
    values, evidence = {}, []
    for tag, ratio in RATIOS.items():
        group = {}
        for arm in ARMS:
            name = f'{arm}_{partition}_{tag}_s{seed}'
            timing = checked_timing(report['cells'][name]['timing'])
            group[LABELS[arm]] = timing[metric]
            evidence.append(dict(task=name, method=LABELS[arm], ratio=ratio,
                                 value=timing[metric], **timing))
        values[f'{100 * ratio:.0f}%'] = group
    svg = style._grouped_bar_chart(title=title, values=values, x_label='恶意节点比例',
                                   y_label=y_label, y_max=y_max)
    root = ET.fromstring(svg)
    # CIFAR durations have more digits: rotate value labels to avoid collisions.
    for el in root.findall(f'{{{SVG_NS}}}text'):
        if el.get('font-size') == '10':
            el.text = f'{float(el.text):.1f}'
            el.set('transform', f'rotate(-60 {el.get("x")} {el.get("y")})')
            el.set('text-anchor', 'start')
    bars = [el for el in root.findall(f'{{{SVG_NS}}}rect') if el.get('height') not in ('100%', '14')]
    if len(bars) != len(evidence):
        raise ValueError('Unexpected MNIST bar template')
    for el, item in zip(bars, evidence):
        el.set('data-task', item['task'])
        el.set('data-value', repr(item['value']))
        ET.SubElement(el, '{' + SVG_NS + '}title').text = f'{item["method"]}: {item["value"]:.6f} s'
    return with_metadata(ET.tostring(root, encoding='unicode'),
                         dict(metric=metric, unit='seconds', seed=seed, partition=partition,
                              y_max=y_max, definition=TIMING_NOTE, bars=evidence))


def memory_plot(seed, partition):
    title = PARTITIONS[partition][1] + '：内存开销记录'
    svg = '\n'.join([style._svg_open(920, 480), style._chart_title(title, 920),
        '<text x="460" y="205" text-anchor="middle" font-size="25" fill="#374151">峰值内存：未记录</text>',
        '<text x="460" y="250" text-anchor="middle" font-size="16" fill="#64748B">原实验 180 项的字段均为占位值 0，不能解释为实际占用 0 MB。</text>',
        '<text x="460" y="285" text-anchor="middle" font-size="16" fill="#64748B">原始字段保留于 summary.json；本图不绘制虚假的零值柱。</text>', '</svg>'])
    return with_metadata(svg, dict(metric='peak_memory_mb', seed=seed, partition=partition,
                                  available=False, raw_value=0., plotted_value=None))


def dashboard(title, generated, intro):
    page = style._dashboard_html(generated, has_baseline=True)
    page = page.replace('联邦学习防御实验可视化', escape(title))
    page = re.sub(r'<p>本页面由实验脚本自动生成.*?</p>', intro, page)
    # Preserve the MNIST card layout while making full-size SVGs easy to download.
    page = re.sub(r'(<img src="([^"]+)"[^>]*>)',
                  lambda m: m[1] + f'<p><a href="{m[2]}">打开 / 下载 SVG</a></p>', page)
    return page


def png_exports(output, browser):
    """Optional standard browser export; use an isolated temporary profile."""
    with tempfile.TemporaryDirectory(prefix='cifar-best-png-') as profile:
        for source in sorted(output.glob('seed_*/plots/*.svg')):
            root = ET.parse(source).getroot()
            target = source.with_suffix('.png')
            command = [str(browser), '--headless', '--no-first-run', '--no-default-browser-check',
                       '--disable-background-networking', '--hide-scrollbars', '--force-device-scale-factor=1',
                       '--user-data-dir=' + profile, '--screenshot=' + str(target.resolve()),
                       f'--window-size={root.get("width")},{root.get("height")}', source.resolve().as_uri()]
            run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            if run.returncode or not target.exists():
                raise RuntimeError('PNG browser export failed: ' + source.name)


def render(package, output, expected_sha256=SOURCE_SHA, browser=None):
    bundle, report, inputs = load_inputs(package, expected_sha256)
    output.mkdir(parents=True, exist_ok=False)
    all_plots, runtime_plots, asr_plots = [], [], []
    nav = []
    for seed in accepted.SEEDS:
        seed_dir = output / f'seed_{seed}'
        (seed_dir / 'plots').mkdir(parents=True)
        generated = []
        for partition, (slug, label) in PARTITIONS.items():
            y_max = max(c['timing']['runtime_with_crypto_seconds'] for c in report['cells'].values()
                        if c['config']['seed'] == seed and c['config']['partition'] == partition) * 1.30
            plots = [
                ('accuracy_comparison', '模型收敛 / 准确率', curve_plot(bundle, seed, partition, 'accuracy')),
                ('asr_comparison', '攻击成功率（ASR）', curve_plot(bundle, seed, partition, 'source_transfer_asr')),
                ('runtime_without_crypto', '扣除密码协议的时间开销', time_plot(report, seed, partition, 'runtime_without_crypto_seconds', y_max)),
                ('runtime_overhead', '端到端时间开销（含密码协议）', time_plot(report, seed, partition, 'runtime_with_crypto_seconds', y_max)),
                ('memory_overhead', '峰值内存（未记录）', memory_plot(seed, partition)),
            ]
            for name, caption, svg in plots:
                path = seed_dir / 'plots' / f'{slug}_{name}.svg'
                path.write_text(svg, encoding='utf-8')
                generated.append((label + ' - ' + caption, path.relative_to(seed_dir).as_posix()))
                all_plots.append(path.relative_to(output).as_posix())
                entry = (f'seed {seed} · {label} - {caption}', path.relative_to(output).as_posix())
                if name.startswith('runtime_'):
                    runtime_plots.append(entry)
                if name == 'asr_comparison':
                    asr_plots.append(entry)
        intro = (f'<p>seed {seed} · 6 方法 × 2 分布 × 5 比例 · 每项 150 轮。</p>'
                 f'<p>{SCOPE_NOTE}</p><p>{TIMING_NOTE}</p>'
                 '<p><a href="../visualizations.html">全部 seed</a> · <a href="../report.html">完整原值、健康与排名报告</a></p>')
        (seed_dir / 'visualizations.html').write_text(dashboard(f'CIFAR-BEST · seed {seed}', generated, intro), encoding='utf-8')
        nav.append(f'<li><a href="seed_{seed}/visualizations.html">seed {seed}：准确率 / ASR / 两种耗时 / 内存记录</a></li>')
    overview = (f'<p>{SCOPE_NOTE}</p><ul>{"".join(nav)}</ul>'
                '<p><a href="runtime_visualizations.html">全部耗时图</a> · '
                '<a href="asr_visualizations.html">全部 ASR 图</a> · '
                '<a href="report.html">完整原值、健康与排名报告</a></p>'
                f'<p>{TIMING_NOTE}</p>')
    fair_plots = [item for item in runtime_plots if item[1].endswith('runtime_without_crypto.svg')]
    for name, title, plots in [('visualizations', 'CIFAR-BEST · MNIST 格式可视化', fair_plots),
                                ('runtime_visualizations', 'CIFAR-BEST · 耗时可视化', runtime_plots),
                                ('asr_visualizations', 'CIFAR-BEST · 官方测试 ASR', asr_plots)]:
        (output / f'{name}.html').write_text(dashboard(title, plots, overview), encoding='utf-8')
    # Keep the complete historical tables and disclosures, replace only figure blocks.
    accepted.document(report, output, expected_sha256)
    page = (output / 'report.html').read_text()
    for seed in accepted.SEEDS:
        page = re.sub(r'<figure><img src="tail10_s' + str(seed) + r'\.png".*?</figure>',
                      f'<p><a href="seed_{seed}/visualizations.html">本 seed 全部图（MNIST 格式）</a></p>', page)
        for old_name, new_names in {
            'trajectories_iid': ['iid_accuracy_comparison', 'iid_asr_comparison'],
            'trajectories_dirichlet': ['dirichlet_alpha_0_5_accuracy_comparison', 'dirichlet_alpha_0_5_asr_comparison'],
            'timing': ['iid_runtime_without_crypto', 'iid_runtime_overhead',
                       'dirichlet_alpha_0_5_runtime_without_crypto', 'dirichlet_alpha_0_5_runtime_overhead'],
        }.items():
            replacement = ''.join(f'<figure><img src="seed_{seed}/plots/{n}.svg" alt="{n}">'
                                  f'<figcaption><a href="seed_{seed}/plots/{n}.svg">SVG</a></figcaption></figure>' for n in new_names)
            page = re.sub(r'<figure><img src="' + old_name + '_s' + str(seed) + r'\.png".*?</figure>', replacement, page)
    page = page.replace('<h1>', '<p><a href="visualizations.html">可视化首页</a> · <a href="runtime_visualizations.html">全部耗时图</a></p><h1>', 1)
    (output / 'report.html').write_text(page, encoding='utf-8')
    write_json(output / 'combined_results.json', report)
    if browser:
        png_exports(output, browser)
    for path, expected in inputs.items():
        if sha(path.read_bytes()) != expected:
            raise ValueError('Input changed during rendering: ' + str(path))
    manifest = dict(schema='cifar-best-mnist-layout-v1', created_utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=expected_sha256, baseline_sha256=accepted.base.BASELINE_SHA,
                    summary_sha256=SUMMARY_SHA, renderer_sha256=sha(Path(__file__).read_bytes()),
                    frozen_style_sha256=sha(Path(style.__file__).read_bytes()),
                    figures=all_plots, svg_count=len(all_plots), png_count=len(list(output.rglob('*.png'))),
                    curves='raw per-seed rounds 0–150; no smoothing or cross-seed averaging',
                    timing_definition=TIMING_NOTE, timing_cells_validated=180,
                    memory_available=False, all180_cells_validated=True, all30_ours_replaced=True,
                    other150_values_and_curves_unchanged=True, old_top20_plotted=False,
                    mnist_palette=style.METHOD_COLORS, visual_inspection_pending=True,
                    outputs={p.relative_to(output).as_posix(): sha(p.read_bytes())
                             for p in sorted(output.rglob('*')) if p.is_file()})
    write_json(output / 'render_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=ROOT / 'outputs/cifar-best')
    parser.add_argument('--output', type=Path, required=True, help='New directory; never overwrite accepted data')
    parser.add_argument('--expected-sha256', default=SOURCE_SHA)
    parser.add_argument('--png-browser', type=Path, help='Optional Chrome/Chromium executable for PNG exports')
    args = parser.parse_args()
    result = render(args.package, args.output, args.expected_sha256, args.png_browser)
    print(json.dumps(dict(output=str(args.output), svg_count=result['svg_count'], png_count=result['png_count'])))


if __name__ == '__main__':
    main()
