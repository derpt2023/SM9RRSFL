#!/usr/bin/env python3
"""
生成运行时间可视化图表 - 使用与原visualization.py完全相同的SVG生成方式
完全避免matplotlib，直接生成SVG，确保中文正常显示
"""

import pandas as pd
from pathlib import Path
import sys
from html import escape

# 与原visualization.py完全一致的配置
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


def _method_label_index(label: str) -> int:
    """获取方法标签的排序索引"""
    reverse = {value: key for key, value in METHOD_LABELS.items()}
    method = reverse.get(label, label)
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method)
    return len(METHOD_ORDER)


def _color_for_label(label: str) -> str:
    """获取方法标签对应的颜色"""
    reverse = {value: key for key, value in METHOD_LABELS.items()}
    method = reverse.get(label, label)
    return METHOD_COLORS.get(method, THEME_TICK)


def _svg_open(width: int, height: int) -> str:
    """SVG开始标签"""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<rect width="100%" height="100%" fill="white"/>'
    )


def _chart_title(title: str, width: int, font_size: int = 18) -> str:
    """图表标题"""
    return f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="{font_size}" font-weight="700" fill="{THEME_INK}">{escape(title)}</text>'


def _axes(left: int, top: int, plot_w: int, plot_h: int, x_label: str, y_label: str, width: int, height: int) -> str:
    """坐标轴"""
    return "\n".join([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="{THEME_INK}" stroke-width="1.4"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="{THEME_INK}" stroke-width="1.4"/>',
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 20}" text-anchor="middle" font-size="13" fill="{THEME_INK}">{escape(x_label)}</text>',
        f'<text transform="translate(22 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="13" fill="{THEME_INK}">{escape(y_label)}</text>',
    ])


def _grouped_bar_chart(title: str, values: dict, x_label: str, y_label: str, y_max: float = None) -> str:
    """生成分组柱状图SVG - 与原visualization.py完全一致"""
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

    # 网格线和y轴刻度
    for tick in range(0, 6):
        value = y_max * tick / 5
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{THEME_GRID_LIGHT}"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="{THEME_TICK}">{value:.1f}</text>')

    # 绘制柱状图
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
                    f'text-anchor="middle" font-size="10" fill="{THEME_INK}">{value:.1f}</text>'
                )
        parts.append(f'<text x="{center:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="12" fill="{THEME_TICK}">{escape(ratio_label)}</text>')

    # 图例
    for idx, method in enumerate(methods):
        color = _color_for_label(method)
        legend_y = top + 24 + idx * 24
        parts.append(f'<rect x="{width - right + 24}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{width - right + 44}" y="{legend_y + 2}" font-size="13" fill="{THEME_INK}">{escape(method)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def generate_runtime_charts_for_seed(seed_dir: Path):
    """为单个seed目录生成运行时间图表"""

    seed_name = seed_dir.name
    print(f"\n处理 {seed_name}...")

    # 读取summary.csv
    summary_file = seed_dir.parent / 'summary.csv'
    if not summary_file.exists():
        print(f"❌ 找不到 {summary_file}")
        return False

    df = pd.read_csv(summary_file)

    # 筛选该seed的数据
    seed_num = int(seed_name.split('_')[1])
    df_seed = df[df['seed'] == seed_num].copy()

    if df_seed.empty:
        print(f"❌ 在summary.csv中找不到seed={seed_num}的数据")
        return False

    print(f"✓ 读取到 {len(df_seed)} 条记录")

    # 为IID和Non-IID场景分别生成图表
    for partition_type in ['iid', 'dirichlet']:
        df_partition = df_seed[df_seed['partition'] == partition_type].copy()

        if df_partition.empty:
            print(f"⚠️  没有{partition_type}场景的数据")
            continue

        # 生成两种图表
        for metric_name, metric_col, y_label, title_suffix in [
            ('runtime_overhead', 'runtime_seconds', '端到端运行时间（秒）', '各方案的端到端时间开销对比（含密码协议）'),
            ('runtime_without_crypto', 'runtime_without_crypto_seconds', '扣除密码协议后的运行时间（秒）', '各方案的训练/防御时间开销对比（扣除密码协议）')
        ]:
            generate_single_chart(
                df_partition,
                partition_type,
                metric_name,
                metric_col,
                y_label,
                title_suffix,
                seed_dir
            )

    return True


def generate_single_chart(df, partition_type, metric_name, metric_col, y_label, title_suffix, seed_dir):
    """生成单个运行时间图表"""

    # 准备数据
    methods = ['sm9rrs', 'vert', 'alignins', 'ding13', 'krum', 'fedavg']
    malicious_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]

    # 构建values字典 - 与原visualization.py的_metric_by_ratio_and_method返回格式一致
    values = {}
    for ratio in malicious_ratios:
        ratio_label = f"{int(ratio * 100)}%"
        values[ratio_label] = {}

        for method in methods:
            records = df[(df['method'] == method) & (df['malicious_ratio'] == ratio)]
            if not records.empty:
                value = records[metric_col].values[0]
                method_label = METHOD_LABELS[method]
                values[ratio_label][method_label] = value

    # 生成标题
    if partition_type == 'dirichlet':
        scenario_label = "Non-IID 非独立同分布（Dirichlet alpha=0.5）"
        alpha_str = 'alpha_0_5'
        filename = f'dirichlet_{alpha_str}_{metric_name}.svg'
    else:
        scenario_label = "IID 独立同分布"
        filename = f'{partition_type}_{metric_name}.svg'

    title = f"{scenario_label}：{title_suffix}"

    # 生成SVG
    svg_content = _grouped_bar_chart(
        title=title,
        values=values,
        x_label="恶意节点比例",
        y_label=y_label
    )

    # 保存文件
    plots_dir = seed_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    output_path = plots_dir / filename

    output_path.write_text(svg_content, encoding='utf-8')
    print(f"  ✓ 生成 {filename}")


def main():
    if len(sys.argv) < 2:
        print("用法: python generate_runtime_svg.py <实验目录>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])

    if not exp_dir.exists():
        print(f"❌ 实验目录不存在: {exp_dir}")
        sys.exit(1)

    final_eval_dir = exp_dir / 'final_evaluation'

    if not final_eval_dir.exists():
        print(f"❌ final_evaluation目录不存在: {final_eval_dir}")
        sys.exit(1)

    print(f"开始为 {exp_dir.name} 生成运行时间可视化图表...")

    # 查找所有seed目录
    seed_dirs = sorted([d for d in final_eval_dir.iterdir() if d.is_dir() and d.name.startswith('seed_')])

    if not seed_dirs:
        print("❌ 未找到seed_*目录")
        sys.exit(1)

    print(f"找到 {len(seed_dirs)} 个seed目录")

    success_count = 0
    for seed_dir in seed_dirs:
        if generate_runtime_charts_for_seed(seed_dir):
            success_count += 1

    print(f"\n✅ 完成! 成功处理 {success_count}/{len(seed_dirs)} 个seed")


if __name__ == '__main__':
    main()
