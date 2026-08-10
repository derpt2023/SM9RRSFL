"""Dependency-free SVG/HTML visualizations for experiment outputs."""

from __future__ import annotations

from collections import defaultdict
from html import escape
import math
from pathlib import Path

from .fl import ExperimentResult


METHOD_LABELS = {
    "sm9rrs": "Ours",
    "vert": "VERT",
    "fedredefense": "FedREDefense",
    "krum": "Krum",
    "ding13": "TAD",
    "fedavg": "FedAvg",
}

METHOD_COLORS = {
    "sm9rrs": "#8DBAD9",
    "vert": "#E99092",
    "fedredefense": "#8ADDE4",
    "krum": "#DCDD8F",
    "ding13": "#FEBD85",
    "fedavg": "#C4A9A2",
}

THEME_INK = "#374151"
THEME_TICK = "#64748B"
THEME_GRID = "#D1D5DB"
THEME_GRID_LIGHT = "#E5E7EB"

METHOD_ORDER = ["sm9rrs", "vert", "fedredefense", "ding13", "krum", "fedavg"]

LINE_STYLES = {
    "Ours": "",
    "VERT": "7 3",
    "FedREDefense": "2 3",
    "Krum": "8 4",
    "TAD": "3 4",
    "FedAvg": "10 3 3 3",
}


def generate_visualizations(results: list[ExperimentResult], output_dir: str | Path) -> Path:
    """Write SVG plots and an HTML dashboard, then return the dashboard path."""

    out = Path(output_dir)
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[str, str]] = []
    baseline_found = False
    scenario_groups = _group_by_partition_and_clients(results)
    single_scenario = len(scenario_groups) == 1
    multiple_client_counts = len({result.config.num_clients for result in results}) > 1

    for scenario_key, scenario_results in scenario_groups:
        scenario_label = _scenario_label(scenario_results[0], include_clients=multiple_client_counts)
        if single_scenario:
            prefix = ""
        elif multiple_client_counts:
            prefix = f"{_scenario_slug(scenario_results[0])}_"
        else:
            prefix = f"{_partition_slug(scenario_results[0])}_"
        title_prefix = "" if single_scenario else f"{scenario_label}："

        ratios = sorted({result.config.malicious_ratio for result in scenario_results})
        if any(_is_ratio(ratio, 0.0) for ratio in ratios):
            baseline_found = True

        panels = []
        for ratio in ratios:
            subset = [
                result
                for result in scenario_results
                if _is_ratio(result.config.malicious_ratio, ratio)
            ]
            panels.append((ratio, _accuracy_series(subset, pad_early_stop=False)))

        _remove_legacy_accuracy_plots(plot_dir, prefix)
        accuracy_path = plot_dir / f"{prefix}accuracy_comparison.svg"
        _write_text(
            accuracy_path,
            _accuracy_comparison_chart(
                title=f"{title_prefix}不同恶意节点比例下的模型收敛/准确率对比",
                panels=panels,
                target_rounds=_target_rounds(scenario_results),
            ),
        )
        generated.append(
            (
                f"{scenario_label} - 不同恶意节点比例的收敛曲线",
                accuracy_path.relative_to(out).as_posix(),
            )
        )

        fair_runtime_path = plot_dir / f"{prefix}runtime_without_crypto.svg"
        _write_text(
            fair_runtime_path,
            _grouped_bar_chart(
                title=f"{title_prefix}各方案的训练/防御时间开销对比（扣除密码协议）",
                values=_metric_by_ratio_and_method(
                    scenario_results,
                    "runtime_without_crypto_seconds",
                ),
                x_label="恶意节点比例",
                y_label="扣除密码协议后的运行时间（秒）",
            ),
        )
        generated.append(
            (
                f"{scenario_label} - 公平时间开销对比（扣除密码协议）",
                fair_runtime_path.relative_to(out).as_posix(),
            )
        )

        runtime_path = plot_dir / f"{prefix}runtime_overhead.svg"
        _write_text(
            runtime_path,
            _grouped_bar_chart(
                title=f"{title_prefix}各方案的端到端时间开销对比（含密码协议）",
                values=_metric_by_ratio_and_method(scenario_results, "runtime_seconds"),
                x_label="恶意节点比例",
                y_label="端到端运行时间（秒）",
            ),
        )
        generated.append(
            (
                f"{scenario_label} - 端到端时间开销对比（含密码协议）",
                runtime_path.relative_to(out).as_posix(),
            )
        )

        memory_path = plot_dir / f"{prefix}memory_overhead.svg"
        _write_text(
            memory_path,
            _grouped_bar_chart(
                title=f"{title_prefix}各方案的内存开销对比",
                values=_metric_by_ratio_and_method(scenario_results, "peak_memory_mb"),
                x_label="恶意节点比例",
                y_label="峰值内存（MB）",
            ),
        )
        generated.append((f"{scenario_label} - 内存开销对比", memory_path.relative_to(out).as_posix()))

    partition_groups = _group_by_partition(results)
    single_partition = len(partition_groups) == 1
    for group_key, partition_results in partition_groups:
        client_counts = {result.config.num_clients for result in partition_results}
        if len(client_counts) < 2:
            continue
        partition_label = _partition_label(partition_results[0])
        prefix = "" if single_partition else f"{_partition_slug(partition_results[0])}_"
        for ratio in sorted({result.config.malicious_ratio for result in partition_results}):
            subset = [
                result
                for result in partition_results
                if _is_ratio(result.config.malicious_ratio, ratio)
            ]
            ratio_label = _format_ratio(ratio)

            accuracy_path = plot_dir / f"{prefix}client_count_accuracy_ratio_{_ratio_slug(ratio)}.svg"
            _write_text(
                accuracy_path,
                _grouped_bar_chart(
                    title=f"{partition_label}：恶意比例 {ratio_label} 时不同客户端数量的最终准确率对比",
                    values=_metric_by_clients_and_method(subset, "final_accuracy"),
                    x_label="客户端数量",
                    y_label="最终测试准确率",
                    y_max=1.0,
                ),
            )
            generated.append(
                (
                    f"{partition_label} - 恶意比例 {ratio_label} 客户端数量/准确率对比",
                    accuracy_path.relative_to(out).as_posix(),
                )
            )

            runtime_path = plot_dir / f"{prefix}client_count_runtime_ratio_{_ratio_slug(ratio)}.svg"
            _write_text(
                runtime_path,
                _grouped_bar_chart(
                    title=f"{partition_label}：恶意比例 {ratio_label} 时不同客户端数量的公平时间开销（扣除密码协议）",
                    values=_metric_by_clients_and_method(
                        subset,
                        "runtime_without_crypto_seconds",
                    ),
                    x_label="客户端数量",
                    y_label="扣除密码协议后的运行时间（秒）",
                ),
            )
            generated.append(
                (
                    f"{partition_label} - 恶意比例 {ratio_label} 客户端数量/公平时间开销对比",
                    runtime_path.relative_to(out).as_posix(),
                )
            )

            end_to_end_path = (
                plot_dir
                / f"{prefix}client_count_runtime_end_to_end_ratio_{_ratio_slug(ratio)}.svg"
            )
            _write_text(
                end_to_end_path,
                _grouped_bar_chart(
                    title=f"{partition_label}：恶意比例 {ratio_label} 时不同客户端数量的端到端时间开销（含密码协议）",
                    values=_metric_by_clients_and_method(subset, "runtime_seconds"),
                    x_label="客户端数量",
                    y_label="端到端运行时间（秒）",
                ),
            )
            generated.append(
                (
                    f"{partition_label} - 恶意比例 {ratio_label} 客户端数量/端到端时间开销",
                    end_to_end_path.relative_to(out).as_posix(),
                )
            )

            memory_path = plot_dir / f"{prefix}client_count_memory_ratio_{_ratio_slug(ratio)}.svg"
            _write_text(
                memory_path,
                _grouped_bar_chart(
                    title=f"{partition_label}：恶意比例 {ratio_label} 时不同客户端数量的内存开销对比",
                    values=_metric_by_clients_and_method(subset, "peak_memory_mb"),
                    x_label="客户端数量",
                    y_label="峰值内存（MB）",
                ),
            )
            generated.append(
                (
                    f"{partition_label} - 恶意比例 {ratio_label} 客户端数量/内存开销对比",
                    memory_path.relative_to(out).as_posix(),
                )
            )

    dashboard = out / "visualizations.html"
    _write_text(dashboard, _dashboard_html(generated, has_baseline=baseline_found))
    return dashboard


def _accuracy_series(
    results: list[ExperimentResult],
    *,
    pad_early_stop: bool = True,
) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = {}
    for result in sorted(results, key=lambda item: _method_index(item.config.method)):
        label = _method_label(result.config.method)
        points = [(record.round, record.accuracy) for record in result.records]
        if (
            pad_early_stop
            and result.config.early_stop
            and result.stopped_round < result.config.rounds
        ):
            points.append((result.config.rounds, result.final_accuracy))
        series[label] = points
    return series


def _target_rounds(results: list[ExperimentResult]) -> int:
    return max((result.config.rounds for result in results), default=1)


def _metric_by_ratio_and_method(
    results: list[ExperimentResult],
    metric: str,
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    for result in results:
        ratio_label = _format_ratio(result.config.malicious_ratio)
        values[ratio_label][_method_label(result.config.method)] = float(getattr(result, metric))
    return dict(sorted(values.items(), key=lambda item: _ratio_sort_key(item[0])))


def _metric_by_clients_and_method(
    results: list[ExperimentResult],
    metric: str,
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    for result in results:
        client_label = str(result.config.num_clients)
        values[client_label][_method_label(result.config.method)] = float(getattr(result, metric))
    return dict(sorted(values.items(), key=lambda item: int(item[0])))


def _group_by_partition(
    results: list[ExperimentResult],
) -> list[tuple[tuple[str, float], list[ExperimentResult]]]:
    groups: dict[tuple[str, float], list[ExperimentResult]] = defaultdict(list)
    for result in results:
        alpha = result.config.dirichlet_alpha if result.config.partition == "dirichlet" else 0.0
        groups[(result.config.partition, alpha)].append(result)
    return sorted(groups.items(), key=lambda item: _partition_sort_key(item[0]))


def _group_by_partition_and_clients(
    results: list[ExperimentResult],
) -> list[tuple[tuple[str, float, int], list[ExperimentResult]]]:
    groups: dict[tuple[str, float, int], list[ExperimentResult]] = defaultdict(list)
    for result in results:
        alpha = result.config.dirichlet_alpha if result.config.partition == "dirichlet" else 0.0
        groups[(result.config.partition, alpha, result.config.num_clients)].append(result)
    return sorted(groups.items(), key=lambda item: (*_partition_sort_key(item[0][:2]), item[0][2]))


def _accuracy_comparison_chart(
    *,
    title: str,
    panels: list[tuple[float, dict[str, list[tuple[float, float]]]]],
    target_rounds: int,
) -> str:
    """Render one paper-style panel per malicious-node ratio.

    Every SVG polyline contains all recorded round/accuracy pairs. Markers are
    thinned only for readability; the underlying curves are never smoothed,
    averaged, or interpolated.
    """

    panel_count = max(1, len(panels))
    outer_left, outer_right, gap, panel_width = 24, 20, 16, 330
    width = outer_left + outer_right + panel_count * panel_width + (panel_count - 1) * gap
    methods = sorted(
        {method for _, series in panels for method in series},
        key=_method_label_index,
    )
    legend_columns = min(
        max(1, len(methods)),
        max(1, int((width - 60) // 180)),
    )
    legend_rows = max(1, math.ceil(len(methods) / legend_columns))
    plot_top = 76 + (legend_rows - 1) * 22
    plot_height = 242
    plot_bottom = plot_top + plot_height
    height = plot_bottom + 98
    max_x = max(
        target_rounds,
        max(
            (x for _, series in panels for values in series.values() for x, _ in values),
            default=1.0,
        ),
        1.0,
    )

    parts = [
        _svg_open(width, height),
        f'<title>{escape(title)}</title>',
        '<desc>图中每条曲线均使用 rounds.csv 保存的逐轮测试准确率原始值，未进行平滑、平均或插值。</desc>',
        (
            '<style>text { font-family: Arial, "Helvetica Neue", "PingFang SC", '
            '"Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }</style>'
        ),
        _chart_title(title, width, font_size=22),
    ]

    legend_item_width = min(210.0, (width - 60) / legend_columns)
    legend_block_width = legend_columns * legend_item_width
    legend_start_x = (width - legend_block_width) / 2
    for idx, method in enumerate(methods):
        row, column = divmod(idx, legend_columns)
        y = 53 + row * 22
        x = legend_start_x + column * legend_item_width
        color = _color_for_label(method)
        stroke_dash = LINE_STYLES.get(method, "")
        dash_attr = f' stroke-dasharray="{stroke_dash}"' if stroke_dash else ""
        parts.append(
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 28:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="{2.2 if method == "Ours" else 1.7}"{dash_attr}/>'
        )
        parts.append(_marker(method, x + 14, y, color, filled=True))
        parts.append(
            f'<text x="{x + 36:.1f}" y="{y + 5:.1f}" font-size="14.5" '
            f'font-weight="700" fill="{THEME_INK}">{escape(method)}</text>'
        )

    for panel_idx, (ratio, series) in enumerate(panels):
        panel_x = outer_left + panel_idx * (panel_width + gap)
        left = panel_x + 54
        plot_width = panel_width - 64

        def sx(value: float) -> float:
            return left + (value / max_x) * plot_width

        def sy(value: float) -> float:
            bounded = min(max(value, 0.0), 1.0)
            return plot_top + (1.0 - bounded) * plot_height

        letter = chr(ord("a") + panel_idx)
        ratio_label = _format_ratio(ratio)
        parts.append(f'<g id="panel-{letter}" data-malicious-ratio="{ratio:g}">')
        parts.append(f'<title>({letter}) 恶意节点比例 {escape(ratio_label)}</title>')

        for tick in range(6):
            value = tick / 5
            y = sy(value)
            parts.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" '
                f'stroke="{THEME_GRID}" stroke-width="0.8" opacity="0.45"/>'
            )
            parts.append(
                f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-size="13.5" font-weight="600" fill="{THEME_TICK}">{value:.1f}</text>'
            )
        for tick in range(6):
            value = max_x * tick / 5
            x = sx(value)
            parts.append(
                f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bottom}" '
                f'stroke="{THEME_GRID_LIGHT}" stroke-width="0.7" opacity="0.45"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{plot_bottom + 18}" text-anchor="middle" '
                f'font-size="13.5" font-weight="600" fill="{THEME_TICK}">{value:.0f}</text>'
            )

        parts.extend(
            [
                f'<rect x="{left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" '
                f'fill="none" stroke="{THEME_INK}" stroke-width="1.4"/>',
                (
                    f'<text x="{left + plot_width / 2:.1f}" y="{plot_bottom + 41}" '
                    f'text-anchor="middle" font-size="15" font-weight="700" '
                    f'fill="{THEME_INK}">训练轮次</text>'
                ),
                (
                    f'<text transform="translate({panel_x + 14} {plot_top + plot_height / 2:.1f}) '
                    'rotate(-90)" text-anchor="middle" font-size="15" font-weight="700" '
                    f'fill="{THEME_INK}">测试准确率</text>'
                ),
            ]
        )

        for method, points in sorted(series.items(), key=lambda item: _method_label_index(item[0])):
            color = _color_for_label(method)
            stroke_dash = LINE_STYLES.get(method, "")
            dash_attr = f' stroke-dasharray="{stroke_dash}"' if stroke_dash else ""
            real_points = points
            real_text = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in real_points)
            if real_text:
                stroke_width = 2.2 if method == "Ours" else 1.65
                opacity = 1.0 if method == "Ours" else 0.9
                parts.append(
                    f'<polyline data-series="accuracy" data-method="{escape(method)}" '
                    f'data-malicious-ratio="{ratio:g}" points="{real_text}" fill="none" '
                    f'stroke="{color}" stroke-width="{stroke_width}"{dash_attr} opacity="{opacity}"/>'
                )
                marker_step = max(1, math.ceil((len(real_points) - 1) / 6))
                marker_indices = set(range(0, len(real_points), marker_step))
                if real_points:
                    marker_indices.add(len(real_points) - 1)
                for point_idx in sorted(marker_indices):
                    x, y = real_points[point_idx]
                    parts.append(
                        _marker(
                            method,
                            sx(x),
                            sy(y),
                            color,
                            filled=True,
                        )
                    )

        parts.append(
            f'<text x="{left + plot_width / 2:.1f}" y="{plot_bottom + 73}" '
            f'text-anchor="middle" font-size="15.5" font-weight="700" fill="{THEME_INK}">'
            f'({letter}) 恶意节点比例 {escape(ratio_label)}</text>'
        )
        parts.append("</g>")

    parts.append("</svg>")
    return "\n".join(parts)


def _line_chart(
    *,
    title: str,
    series: dict[str, list[tuple[float, float]]],
    target_rounds: int,
    x_label: str,
    y_label: str,
    y_min: float,
    y_max: float,
) -> str:
    width, height = 920, 480
    left, right, top, bottom = 80, 170, 56, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_x = max(target_rounds, max((x for values in series.values() for x, _ in values), default=1.0))
    max_x = max(max_x, 1.0)

    def sx(x: float) -> float:
        return left + (x / max_x) * plot_w

    def sy(y: float) -> float:
        bounded = min(max(y, y_min), y_max)
        return top + (y_max - bounded) / (y_max - y_min) * plot_h

    parts = [_svg_open(width, height), _chart_title(title, width)]
    parts.append(_axes(left, top, plot_w, plot_h, x_label, y_label, width, height))
    for tick in range(0, 6):
        value = y_min + (y_max - y_min) * tick / 5
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{THEME_GRID_LIGHT}"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="{THEME_TICK}">{value:.1f}</text>')
    for tick in range(0, 6):
        value = max_x * tick / 5
        x = sx(value)
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-size="12" fill="{THEME_TICK}">{value:.0f}</text>')

    for idx, (label, points) in enumerate(series.items()):
        color = _color_for_label(label)
        stroke_dash = LINE_STYLES.get(label, "")
        dash_attr = f' stroke-dasharray="{stroke_dash}"' if stroke_dash else ""
        real_points, padded_points = _split_padded_points(points, target_rounds)
        real_text = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in real_points)
        padded_text = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in padded_points)
        if real_text:
            parts.append(f'<polyline points="{real_text}" fill="none" stroke="{color}" stroke-width="2.7"{dash_attr} opacity="0.92"/>')
            for x, y in real_points:
                parts.append(_marker(label, sx(x), sy(y), color, filled=True))
        if len(padded_points) > 1:
            parts.append(
                f'<polyline points="{padded_text}" fill="none" stroke="{color}" '
                f'stroke-width="2.2" stroke-dasharray="5 5" opacity="0.62"/>'
            )
        legend_y = top + 24 + idx * 24
        parts.append(f'<rect x="{width - right + 24}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{width - right + 44}" y="{legend_y + 2}" font-size="13" fill="{THEME_INK}">{escape(label)}</text>')

    parts.append(
        f'<text x="{left}" y="{height - 8}" font-size="11" fill="{THEME_TICK}">'
        "虚线平台段表示该方法已达到误差阈值并提前停止，图中延伸最终准确率便于横向比较。"
        "</text>"
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _grouped_bar_chart(
    *,
    title: str,
    values: dict[str, dict[str, float]],
    x_label: str,
    y_label: str,
    y_max: float | None = None,
) -> str:
    width, height = 920, 480
    left, right, top, bottom = 84, 170, 56, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    methods = sorted({method for group in values.values() for method in group}, key=_method_label_index)
    max_value = max((value for group in values.values() for value in group.values()), default=1.0)
    max_value = max(max_value, 1e-9)
    y_max = y_max or max_value * 1.15
    groups = list(values.items())
    group_w = plot_w / max(1, len(groups))
    bar_w = min(34.0, group_w / max(1, len(methods) + 1))

    def sy(value: float) -> float:
        return top + (1.0 - value / y_max) * plot_h

    parts = [_svg_open(width, height), _chart_title(title, width)]
    parts.append(_axes(left, top, plot_w, plot_h, x_label, y_label, width, height))
    for tick in range(0, 6):
        value = y_max * tick / 5
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{THEME_GRID_LIGHT}"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="{THEME_TICK}">{value:.2f}</text>')

    for group_idx, (ratio_label, group_values) in enumerate(groups):
        center = left + group_idx * group_w + group_w / 2
        start = center - (len(methods) * bar_w) / 2
        for method_idx, method in enumerate(methods):
            value = group_values.get(method, 0.0)
            x = start + method_idx * bar_w
            y = sy(value)
            color = _color_for_label(method)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 4:.1f}" '
                f'height="{top + plot_h - y:.1f}" fill="{color}"/>'
            )
            if value > 0:
                parts.append(
                    f'<text x="{x + (bar_w - 4) / 2:.1f}" y="{y - 5:.1f}" '
                    f'text-anchor="middle" font-size="10" fill="{THEME_INK}">{value:.2f}</text>'
                )
        parts.append(f'<text x="{center:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="12" fill="{THEME_TICK}">{escape(ratio_label)}</text>')

    for idx, method in enumerate(methods):
        color = _color_for_label(method)
        legend_y = top + 24 + idx * 24
        parts.append(f'<rect x="{width - right + 24}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{width - right + 44}" y="{legend_y + 2}" font-size="13" fill="{THEME_INK}">{escape(method)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def _dashboard_html(generated: list[tuple[str, str]], *, has_baseline: bool) -> str:
    note = "" if has_baseline else "<p class=\"note\">未检测到 0% 恶意节点实验结果，因此未生成无恶意节点收敛曲线。运行时请包含 <code>--ratios 0.00</code>。</p>"
    figures = "\n".join(
        f'<section><h2>{escape(title)}</h2><img src="{escape(path)}" alt="{escape(title)}"></section>'
        for title, path in generated
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>联邦学习防御实验可视化</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #0f172a; background: #f8fafc; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    h2 {{ font-size: 18px; margin: 0 0 12px; }}
    p {{ color: #475569; }}
    section {{ margin-top: 24px; padding: 18px; background: white; border: 1px solid #e2e8f0; border-radius: 8px; }}
    img {{ display: block; max-width: 100%; height: auto; }}
    code {{ background: #e2e8f0; padding: 2px 5px; border-radius: 4px; }}
    .note {{ padding: 12px; background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; }}
  </style>
</head>
<body>
<main>
  <h1>联邦学习防御实验可视化</h1>
  <p>本页面由实验脚本自动生成，数据来自同目录下的 <code>summary.csv</code> 与 <code>rounds.csv</code>。</p>
  {note}
  {figures}
</main>
</body>
</html>
"""


def _svg_open(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<rect width="100%" height="100%" fill="white"/>'
    )


def _chart_title(title: str, width: int, *, font_size: int = 18) -> str:
    return f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="{font_size}" font-weight="700" fill="{THEME_INK}">{escape(title)}</text>'


def _axes(
    left: int,
    top: int,
    plot_w: int,
    plot_h: int,
    x_label: str,
    y_label: str,
    width: int,
    height: int,
) -> str:
    return "\n".join(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="{THEME_INK}" stroke-width="1.4"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="{THEME_INK}" stroke-width="1.4"/>',
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 20}" text-anchor="middle" font-size="13" fill="{THEME_INK}">{escape(x_label)}</text>',
            f'<text transform="translate(22 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="13" fill="{THEME_INK}">{escape(y_label)}</text>',
        ]
    )


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _method_index(method: str) -> int:
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method)
    return len(METHOD_ORDER)


def _method_label_index(label: str) -> int:
    reverse = {value: key for key, value in METHOD_LABELS.items()}
    return _method_index(reverse.get(label, label))


def _color_for_label(label: str) -> str:
    reverse = {value: key for key, value in METHOD_LABELS.items()}
    method = reverse.get(label, label)
    return METHOD_COLORS.get(method, THEME_TICK)


def _format_ratio(ratio: float) -> str:
    return f"{ratio * 100:.0f}%"


def _ratio_slug(ratio: float) -> str:
    return f"{int(round(ratio * 100)):03d}"


def _ratio_sort_key(label: str) -> float:
    return float(label.rstrip("%"))


def _partition_label(result: ExperimentResult) -> str:
    if result.config.partition == "iid":
        return "IID 独立同分布"
    if result.config.partition == "dirichlet":
        return f"Non-IID 非独立同分布（Dirichlet alpha={result.config.dirichlet_alpha:g}）"
    return result.config.partition


def _partition_slug(result: ExperimentResult) -> str:
    if result.config.partition == "iid":
        return "iid"
    if result.config.partition == "dirichlet":
        alpha = str(result.config.dirichlet_alpha).replace(".", "_")
        return f"dirichlet_alpha_{alpha}"
    return result.config.partition.replace(" ", "_")


def _scenario_label(result: ExperimentResult, *, include_clients: bool) -> str:
    label = _partition_label(result)
    if include_clients:
        return f"{label}（客户端数={result.config.num_clients}）"
    return label


def _scenario_slug(result: ExperimentResult) -> str:
    return f"{_partition_slug(result)}_clients_{result.config.num_clients:03d}"


def _partition_sort_key(key: tuple[str, float]) -> tuple[int, str, float]:
    partition, alpha = key
    order = 0 if partition == "iid" else 1
    return (order, partition, alpha)


def _is_ratio(value: float, expected: float) -> bool:
    return abs(value - expected) < 1e-9


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _remove_legacy_accuracy_plots(plot_dir: Path, prefix: str) -> None:
    """Remove superseded one-ratio accuracy SVGs before writing the panel figure."""

    legacy_paths = [plot_dir / f"{prefix}accuracy_baseline.svg"]
    legacy_paths.extend(plot_dir.glob(f"{prefix}accuracy_ratio_*.svg"))
    for path in legacy_paths:
        if path.exists():
            path.unlink()


def _split_padded_points(
    points: list[tuple[float, float]],
    target_rounds: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if len(points) >= 2 and points[-1][0] == target_rounds and points[-2][0] != target_rounds:
        return points[:-1], [points[-2], points[-1]]
    return points, []


def _marker(label: str, x: float, y: float, color: str, *, filled: bool) -> str:
    fill = color if filled else "white"
    if label == "VERT":
        points = (
            f"{x:.1f},{y - 4.2:.1f} {x + 4.2:.1f},{y:.1f} "
            f"{x:.1f},{y + 4.2:.1f} {x - 4.2:.1f},{y:.1f}"
        )
        return (
            f'<polygon points="{points}" fill="{fill}" stroke="{color}" '
            'stroke-width="1.4"/>'
        )
    if label == "FedREDefense":
        return (
            f'<path d="M {x - 4:.1f} {y:.1f} H {x + 4:.1f} '
            f'M {x:.1f} {y - 4:.1f} V {y + 4:.1f}" '
            f'stroke="{color}" stroke-width="1.8"/>'
        )
    if label == "TAD":
        return (
            f'<rect x="{x - 3.5:.1f}" y="{y - 3.5:.1f}" width="7" height="7" '
            f'fill="{fill}" stroke="{color}" stroke-width="1.4"/>'
        )
    if label == "Krum":
        points = f"{x:.1f},{y - 4:.1f} {x + 4:.1f},{y + 3.5:.1f} {x - 4:.1f},{y + 3.5:.1f}"
        return f'<polygon points="{points}" fill="{fill}" stroke="{color}" stroke-width="1.4"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{fill}" stroke="{color}" stroke-width="1.2"/>'
