#!/usr/bin/env python3
"""生成 ASR (Attack Success Rate) 可视化图表。

从 aggregate.csv 读取数据，生成柱状图展示不同方法在不同恶意节点比例下的ASR对比。
"""

from __future__ import annotations

import csv
import json
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


def read_aggregate_csv(csv_path: Path) -> list[dict]:
    """读取 aggregate.csv 文件。"""
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def group_by_partition_and_clients(
    rows: list[dict],
) -> dict[tuple[str, float, int], list[dict]]:
    """按 partition、dirichlet_alpha、num_clients 分组。"""
    groups: dict[tuple[str, float, int], list[dict]] = defaultdict(list)
    for row in rows:
        partition = row["partition"]
        alpha = float(row["dirichlet_alpha"])
        num_clients = int(row["num_clients"])
        groups[(partition, alpha, num_clients)].append(row)
    return groups


def extract_asr_by_ratio_and_method(rows: list[dict]) -> dict[str, dict[str, float]]:
    """提取按恶意比例和方法组织的 ASR 数据。

    返回: {ratio_label: {method_label: asr_value}}
    """
    data: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        ratio = float(row["malicious_ratio"])
        # 跳过干净场景（ratio=0.0），因为ASR在此场景下没有意义
        if ratio < 1e-9:
            continue

        method = row["method"]
        asr_mean = float(row["attack_success_rate_mean"])

        ratio_label = f"{ratio * 100:.0f}%"
        method_label = METHOD_LABELS.get(method, method)

        data[ratio_label][method_label] = asr_mean

    # 按比例排序
    return dict(sorted(data.items(), key=lambda x: float(x[0].rstrip("%"))))


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


def generate_asr_bar_chart(
    *,
    title: str,
    values: dict[str, dict[str, float]],
    x_label: str,
    y_label: str,
    y_max: float = 1.0,
) -> str:
    """生成 ASR 柱状图（SVG）。"""
    width, height = 920, 480
    left, right, top, bottom = 84, 170, 56, 86
    plot_w = width - left - right
    plot_h = height - top - bottom

    methods = sorted(
        {method for group in values.values() for method in group},
        key=method_label_index,
    )

    groups = list(values.items())
    group_w = plot_w / max(1, len(groups))
    bar_w = min(34.0, group_w / max(1, len(methods) + 1))

    def sy(value: float) -> float:
        """将 ASR 值转换为 y 坐标。"""
        return top + (1.0 - value / y_max) * plot_h

    # 开始构建 SVG
    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
    )
    parts.append('<rect width="100%" height="100%" fill="white"/>')

    # 标题
    parts.append(
        '<style>text { font-family: Arial, "Helvetica Neue", "PingFang SC", '
        '"Microsoft YaHei", "Noto Sans CJK SC", sans-serif; }</style>'
    )
    parts.append(
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" '
        f'font-size="18" font-weight="700" fill="{THEME_INK}">{escape(title)}</text>'
    )

    # 坐标轴
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="{THEME_INK}" stroke-width="1.4"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" '
        f'stroke="{THEME_INK}" stroke-width="1.4"/>'
    )

    # Y 轴刻度和网格线
    for tick in range(0, 6):
        value = y_max * tick / 5
        y = sy(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="{THEME_GRID_LIGHT}"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12" fill="{THEME_TICK}">{value:.2f}</text>'
        )

    # 绘制柱状图
    for group_idx, (ratio_label, group_values) in enumerate(groups):
        center = left + group_idx * group_w + group_w / 2
        start = center - (len(methods) * bar_w) / 2

        for method_idx, method in enumerate(methods):
            value = group_values.get(method, 0.0)
            x = start + method_idx * bar_w
            y = sy(value)
            color = color_for_label(method)

            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 4:.1f}" '
                f'height="{top + plot_h - y:.1f}" fill="{color}"/>'
            )

            if value > 0:
                parts.append(
                    f'<text x="{x + (bar_w - 4) / 2:.1f}" y="{y - 5:.1f}" '
                    f'text-anchor="middle" font-size="10" fill="{THEME_INK}">'
                    f'{value:.3f}</text>'
                )

        # X 轴标签
        parts.append(
            f'<text x="{center:.1f}" y="{top + plot_h + 24}" text-anchor="middle" '
            f'font-size="12" fill="{THEME_TICK}">{escape(ratio_label)}</text>'
        )

    # 坐标轴标签
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 20}" text-anchor="middle" '
        f'font-size="13" fill="{THEME_INK}">{escape(x_label)}</text>'
    )
    parts.append(
        f'<text transform="translate(22 {top + plot_h / 2:.1f}) rotate(-90)" '
        f'text-anchor="middle" font-size="13" fill="{THEME_INK}">{escape(y_label)}</text>'
    )

    # 图例
    for idx, method in enumerate(methods):
        color = color_for_label(method)
        legend_y = top + 24 + idx * 24
        parts.append(
            f'<rect x="{width - right + 24}" y="{legend_y - 10}" '
            f'width="14" height="14" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{width - right + 44}" y="{legend_y + 2}" '
            f'font-size="13" fill="{THEME_INK}">{escape(method)}</text>'
        )

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


def generate_asr_visualizations(experiment_dir: Path) -> None:
    """为指定实验目录生成 ASR 可视化。"""
    aggregate_csv = experiment_dir / "final_evaluation" / "aggregate.csv"

    if not aggregate_csv.exists():
        print(f"错误: 找不到 {aggregate_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"读取数据: {aggregate_csv}")
    rows = read_aggregate_csv(aggregate_csv)

    # 按场景分组
    groups = group_by_partition_and_clients(rows)

    plot_dir = experiment_dir / "final_evaluation" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []

    for (partition, alpha, num_clients), group_rows in groups.items():
        scenario_label = partition_label(partition, alpha)
        prefix = partition_slug(partition, alpha)

        # 提取 ASR 数据
        asr_data = extract_asr_by_ratio_and_method(group_rows)

        if not asr_data:
            print(f"  跳过 {scenario_label}: 没有攻击场景数据")
            continue

        # 生成 ASR 柱状图
        asr_path = plot_dir / f"{prefix}_asr_comparison.svg"
        svg_content = generate_asr_bar_chart(
            title=f"{scenario_label}：各方案的攻击成功率（ASR）对比",
            values=asr_data,
            x_label="恶意节点比例",
            y_label="攻击成功率 (ASR)",
            y_max=1.0,
        )

        asr_path.write_text(svg_content, encoding="utf-8")
        generated_files.append((scenario_label, asr_path))
        print(f"  ✓ 生成: {asr_path.relative_to(experiment_dir)}")

    # 更新或创建 HTML 索引
    html_path = experiment_dir / "final_evaluation" / "asr_visualizations.html"
    html_content = generate_html_index(generated_files, experiment_dir)
    html_path.write_text(html_content, encoding="utf-8")
    print(f"\n✓ HTML 索引: {html_path.relative_to(experiment_dir)}")

    # 生成元数据
    metadata = {
        "generated_files": [str(p.relative_to(experiment_dir)) for _, p in generated_files],
        "num_scenarios": len(generated_files),
    }
    meta_path = experiment_dir / "final_evaluation" / "asr_visualization_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ 元数据: {meta_path.relative_to(experiment_dir)}")


def generate_html_index(
    generated_files: list[tuple[str, Path]],
    experiment_dir: Path,
) -> str:
    """生成 HTML 索引页面。"""
    figures = "\n".join(
        f'<section><h2>{escape(label)}</h2>'
        f'<img src="{escape(str(path.relative_to(experiment_dir / "final_evaluation")))}" '
        f'alt="{escape(label)}"></section>'
        for label, path in generated_files
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>攻击成功率 (ASR) 可视化</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           color: #0f172a; background: #f8fafc; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; }}
    h2 {{ font-size: 18px; margin: 0 0 12px; }}
    p {{ color: #475569; line-height: 1.6; }}
    section {{ margin-top: 24px; padding: 18px; background: white;
               border: 1px solid #e2e8f0; border-radius: 8px; }}
    img {{ display: block; max-width: 100%; height: auto; }}
    code {{ background: #e2e8f0; padding: 2px 5px; border-radius: 4px; font-family: monospace; }}
    .note {{ padding: 12px; background: #fef3c7; border: 1px solid #fcd34d;
             border-radius: 8px; margin-top: 16px; }}
  </style>
</head>
<body>
<main>
  <h1>攻击成功率 (ASR) 可视化</h1>
  <p>本页面展示各防御方案在不同恶意节点比例下的攻击成功率（Attack Success Rate, ASR）对比。</p>
  <p>数据来源：<code>final_evaluation/aggregate.csv</code> 中的 <code>attack_success_rate_mean</code> 字段。</p>
  <div class="note">
    <strong>说明：</strong>ASR 越低表示防御效果越好。Ours (SM9RRS) 方案应在各比例下保持较低的 ASR。
  </div>
  {figures}
</main>
</body>
</html>
"""


def main():
    if len(sys.argv) != 2:
        print("用法: python generate_asr_visualization.py <experiment_dir>")
        print("示例: python generate_asr_visualization.py outputs/mnist_v7_target_fair_tuning")
        sys.exit(1)

    experiment_dir = Path(sys.argv[1])
    if not experiment_dir.is_dir():
        print(f"错误: {experiment_dir} 不是有效的目录", file=sys.stderr)
        sys.exit(1)

    print(f"正在为实验生成 ASR 可视化: {experiment_dir}")
    generate_asr_visualizations(experiment_dir)
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
