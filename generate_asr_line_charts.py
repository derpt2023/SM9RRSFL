#!/usr/bin/env python3
"""生成 ASR 逐轮折线图，风格与 accuracy_comparison 一致。

从 rounds.csv 读取逐轮 attack_target_success_rate 数据，
为每个 seed 目录生成多面板折线图，并更新 visualizations.html。
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from html import escape
from pathlib import Path


# 复用 visualization.py 中的常量
METHOD_LABELS = {
    "sm9rrs": "Ours",
    "vert": "VERT",
    "alignins": "AlignIns",
    "krum": "Krum",
    "ding13": "TAD",
    "fedavg": "FedAvg",
}

METHOD_COLORS = {
    "sm9rrs": "#8DBAD9",
    "vert": "#E99092",
    "alignins": "#8ADDE4",
    "krum": "#DCDD8F",
    "ding13": "#FEBD85",
    "fedavg": "#C4A9A2",
}

THEME_INK = "#374151"
THEME_TICK = "#64748B"
THEME_GRID = "#D1D5DB"
THEME_GRID_LIGHT = "#E5E7EB"

METHOD_ORDER = ["sm9rrs", "vert", "alignins", "ding13", "krum", "fedavg"]

LINE_STYLES = {
    "Ours": "",
    "VERT": "7 3",
    "AlignIns": "2 3",
    "Krum": "8 4",
    "TAD": "3 4",
    "FedAvg": "10 3 3 3",
}


def read_rounds_csv(csv_path: Path) -> list[dict]:
    """读取 rounds.csv 文件。"""
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def extract_asr_series(
    rows: list[dict],
    partition: str,
    alpha: float,
    num_clients: int,
    seed: int,
) -> dict[float, dict[str, list[tuple[int, float]]]]:
    """提取指定场景的ASR逐轮数据。

    返回: {malicious_ratio: {method_label: [(round, asr), ...]}}
    """
    data: dict[float, dict[str, list[tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in rows:
        if (
            row["partition"] != partition
            or float(row["dirichlet_alpha"]) != alpha
            or int(row["num_clients"]) != num_clients
            or int(row["seed"]) != seed
        ):
            continue

        ratio = float(row["malicious_ratio"])
        method = row["method"]
        round_num = int(row["round"])
        asr_value = row["attack_target_success_rate"]

        # 跳过空值或非攻击轮次
        if not asr_value or asr_value == "":
            continue

        asr = float(asr_value)
        method_label = METHOD_LABELS.get(method, method)
        data[ratio][method_label].append((round_num, asr))

    # 排序每个series的点
    for ratio_data in data.values():
        for method_label in ratio_data:
            ratio_data[method_label].sort(key=lambda x: x[0])

    return dict(data)


def method_label_index(label: str) -> int:
    """获取方法标签的排序索引。"""
    reverse = {v: k for k, v in METHOD_LABELS.items()}
    method = reverse.get(label, label)
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method)
    return len(METHOD_ORDER)


def color_for_label(label: str) -> str:
    """获取方法标签对应的颜色。"""
    reverse = {v: k for k, v in METHOD_LABELS.items()}
    method = reverse.get(label, label)
    return METHOD_COLORS.get(method, THEME_TICK)


def marker(label: str, x: float, y: float, color: str, *, filled: bool) -> str:
    """生成标记点（与visualization.py保持一致）。"""
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
    if label == "AlignIns":
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


def generate_asr_line_chart(
    *,
    title: str,
    panels: list[tuple[float, dict[str, list[tuple[int, float]]]]],
    target_rounds: int,
) -> str:
    """生成 ASR 多面板折线图（与 accuracy_comparison_chart 风格一致）。"""
    panel_count = max(1, len(panels))
    outer_left, outer_right, gap, panel_width = 24, 20, 16, 330
    width = outer_left + outer_right + panel_count * panel_width + (panel_count - 1) * gap

    methods = sorted(
        {method for _, series in panels for method in series},
        key=method_label_index,
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
            default=1,
        ),
        1,
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<title>{escape(title)}</title>',
        '<desc>图中每条曲线均使用 rounds.csv 保存的逐轮攻击成功率（ASR）原始值，未进行平滑、平均或插值。</desc>',
        (
            '<style>text { font-family: Arial, "Helvetica Neue", "PingFang SC", '
            '"Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }</style>'
        ),
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="22" '
        f'font-weight="700" fill="{THEME_INK}">{escape(title)}</text>',
    ]

    # 图例
    legend_item_width = min(210.0, (width - 60) / legend_columns)
    legend_block_width = legend_columns * legend_item_width
    legend_start_x = (width - legend_block_width) / 2

    for idx, method in enumerate(methods):
        row, column = divmod(idx, legend_columns)
        y = 53 + row * 22
        x = legend_start_x + column * legend_item_width
        color = color_for_label(method)
        stroke_dash = LINE_STYLES.get(method, "")
        dash_attr = f' stroke-dasharray="{stroke_dash}"' if stroke_dash else ""
        parts.append(
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 28:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="{2.2 if method == "Ours" else 1.7}"{dash_attr}/>'
        )
        parts.append(marker(method, x + 14, y, color, filled=True))
        parts.append(
            f'<text x="{x + 36:.1f}" y="{y + 5:.1f}" font-size="14.5" '
            f'font-weight="700" fill="{THEME_INK}">{escape(method)}</text>'
        )

    # 各个面板
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
        ratio_label = f"{ratio * 100:.0f}%"
        parts.append(f'<g id="panel-{letter}" data-malicious-ratio="{ratio:g}">')
        parts.append(f'<title>({letter}) 恶意节点比例 {escape(ratio_label)}</title>')

        # Y轴网格线和刻度
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

        # X轴网格线和刻度
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

        # 坐标轴边框
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
                    f'fill="{THEME_INK}">攻击成功率 (ASR)</text>'
                ),
            ]
        )

        # 绘制折线
        for method, points in sorted(series.items(), key=lambda item: method_label_index(item[0])):
            color = color_for_label(method)
            stroke_dash = LINE_STYLES.get(method, "")
            dash_attr = f' stroke-dasharray="{stroke_dash}"' if stroke_dash else ""

            if not points:
                continue

            points_text = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
            stroke_width = 2.2 if method == "Ours" else 1.65
            opacity = 1.0 if method == "Ours" else 0.9

            parts.append(
                f'<polyline data-series="asr" data-method="{escape(method)}" '
                f'data-malicious-ratio="{ratio:g}" points="{points_text}" fill="none" '
                f'stroke="{color}" stroke-width="{stroke_width}"{dash_attr} opacity="{opacity}"/>'
            )

            # 标记点（适当稀疏化）
            marker_step = max(1, math.ceil((len(points) - 1) / 6))
            marker_indices = set(range(0, len(points), marker_step))
            if points:
                marker_indices.add(len(points) - 1)

            for point_idx in sorted(marker_indices):
                x, y = points[point_idx]
                parts.append(marker(method, sx(x), sy(y), color, filled=True))

        # 面板标题
        parts.append(
            f'<text x="{left + plot_width / 2:.1f}" y="{plot_bottom + 73}" '
            f'text-anchor="middle" font-size="15.5" font-weight="700" fill="{THEME_INK}">'
            f'({letter}) 恶意节点比例 {escape(ratio_label)}</text>'
        )
        parts.append("</g>")

    parts.append("</svg>")
    return "\n".join(parts)


def partition_label(partition: str, alpha: float) -> str:
    """生成分区标签。"""
    if partition == "iid":
        return "IID 独立同分布"
    if partition == "dirichlet":
        return f"Non-IID 非独立同分布（Dirichlet alpha={alpha:g}）"
    return partition


def partition_slug(partition: str, alpha: float) -> str:
    """生成分区文件名slug。"""
    if partition == "iid":
        return "iid"
    if partition == "dirichlet":
        alpha_str = str(alpha).replace(".", "_")
        return f"dirichlet_alpha_{alpha_str}"
    return partition.replace(" ", "_")


def update_html(html_path: Path, new_sections: list[tuple[str, str]]) -> None:
    """更新 visualizations.html，添加 ASR 图表链接。"""
    if not html_path.exists():
        print(f"  警告: {html_path} 不存在，跳过更新")
        return

    content = html_path.read_text(encoding="utf-8")

    # 在</main>之前插入新sections
    new_html_sections = "\n".join(
        f'<section><h2>{escape(title)}</h2><img src="{escape(path)}" alt="{escape(title)}"></section>'
        for title, path in new_sections
    )

    insert_marker = "</main>"
    if insert_marker in content:
        content = content.replace(insert_marker, f"{new_html_sections}\n{insert_marker}")
        html_path.write_text(content, encoding="utf-8")
        print(f"  ✓ 更新 HTML: {html_path.name}")
    else:
        print(f"  警告: 无法在 {html_path.name} 中找到插入点")


def generate_asr_charts_for_experiment(experiment_dir: Path) -> None:
    """为实验目录生成所有 ASR 折线图。"""
    rounds_csv = experiment_dir / "final_evaluation" / "rounds.csv"

    if not rounds_csv.exists():
        print(f"错误: 找不到 {rounds_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"读取数据: {rounds_csv}")
    rows = read_rounds_csv(rounds_csv)

    # 提取所有场景和seed
    scenarios = set()
    seeds = set()
    for row in rows:
        partition = row["partition"]
        alpha = float(row["dirichlet_alpha"])
        num_clients = int(row["num_clients"])
        seed = int(row["seed"])
        scenarios.add((partition, alpha, num_clients))
        seeds.add(seed)

    print(f"找到 {len(scenarios)} 个场景, {len(seeds)} 个 seed")

    # 为每个seed生成图表
    for seed in sorted(seeds):
        seed_dir = experiment_dir / "final_evaluation" / f"seed_{seed}"
        plot_dir = seed_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n处理 seed_{seed}:")
        new_html_sections = []

        for partition, alpha, num_clients in sorted(scenarios):
            scenario_label = partition_label(partition, alpha)
            prefix = partition_slug(partition, alpha)

            # 提取ASR数据
            asr_data = extract_asr_series(rows, partition, alpha, num_clients, seed)

            # 只保留有攻击的场景（ratio > 0）
            panels = []
            for ratio in sorted(asr_data.keys()):
                if ratio > 1e-9:  # 跳过ratio=0
                    panels.append((ratio, asr_data[ratio]))

            if not panels:
                print(f"  跳过 {scenario_label}: 没有攻击轮次ASR数据")
                continue

            # 确定目标轮数
            target_rounds = max(
                (int(row["round"]) for row in rows
                 if row["partition"] == partition
                 and float(row["dirichlet_alpha"]) == alpha
                 and int(row["num_clients"]) == num_clients
                 and int(row["seed"]) == seed),
                default=150
            )

            # 生成SVG
            svg_path = plot_dir / f"{prefix}_asr_comparison.svg"
            svg_content = generate_asr_line_chart(
                title=f"{scenario_label}：不同恶意节点比例下的攻击成功率（ASR）变化",
                panels=panels,
                target_rounds=target_rounds,
            )

            svg_path.write_text(svg_content, encoding="utf-8")
            print(f"  ✓ 生成: {svg_path.relative_to(seed_dir)}")

            new_html_sections.append(
                (
                    f"{scenario_label} - 不同恶意节点比例的ASR变化",
                    f"plots/{prefix}_asr_comparison.svg",
                )
            )

        # 更新HTML
        html_path = seed_dir / "visualizations.html"
        if new_html_sections:
            update_html(html_path, new_html_sections)


def main():
    if len(sys.argv) != 2:
        print("用法: python generate_asr_line_charts.py <experiment_dir>")
        print("示例: python generate_asr_line_charts.py outputs/mnist_v7_target_fair_tuning")
        sys.exit(1)

    experiment_dir = Path(sys.argv[1])
    if not experiment_dir.is_dir():
        print(f"错误: {experiment_dir} 不是有效的目录", file=sys.stderr)
        sys.exit(1)

    print(f"正在为实验生成 ASR 折线图: {experiment_dir}\n")
    generate_asr_charts_for_experiment(experiment_dir)
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
