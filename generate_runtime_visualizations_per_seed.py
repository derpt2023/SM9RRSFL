#!/usr/bin/env python3
"""
生成运行时间可视化图表 - 每个seed独立生成
与现有的accuracy图表保持一致的数据源和风格
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 方法名称映射（与accuracy图表保持一致）
METHOD_DISPLAY_NAMES = {
    'sm9rrs': 'Ours (SM9RRS)',
    'vert': 'VERT',
    'alignins': 'AlignIns',
    'ding13': 'TAD',
    'krum': 'Krum',
    'fedavg': 'FedAvg'
}

# 颜色方案（与accuracy图表保持一致）
METHOD_COLORS = {
    'sm9rrs': '#2E86AB',
    'vert': '#A23B72',
    'alignins': '#F18F01',
    'ding13': '#C73E1D',
    'krum': '#6A994E',
    'fedavg': '#BC4B51'
}

def generate_runtime_chart_for_seed(seed_dir: Path):
    """为单个seed目录生成运行时间图表"""

    seed_name = seed_dir.name
    print(f"\n处理 {seed_name}...")

    # 读取该seed的summary.csv
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
        alpha_str = 'alpha_0_5' if partition_type == 'dirichlet' else ''

        df_partition = df_seed[df_seed['partition'] == partition_type].copy()

        if df_partition.empty:
            print(f"⚠️  没有{partition_type}场景的数据")
            continue

        # 生成两种图表
        for metric_name, metric_col in [
            ('runtime_overhead', 'runtime_seconds'),
            ('runtime_without_crypto', 'runtime_without_crypto_seconds')
        ]:
            generate_single_chart(
                df_partition,
                partition_type,
                metric_name,
                metric_col,
                seed_dir,
                alpha_str
            )

    # 更新该seed目录的visualizations.html
    update_html_for_seed(seed_dir)

    return True

def generate_single_chart(df, partition_type, metric_name, metric_col, seed_dir, alpha_str):
    """生成单个运行时间图表"""

    # 提取数据
    methods = ['sm9rrs', 'vert', 'alignins', 'ding13', 'krum', 'fedavg']
    malicious_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]

    data = {}
    for method in methods:
        method_data = []
        for ratio in malicious_ratios:
            records = df[(df['method'] == method) & (df['malicious_ratio'] == ratio)]
            if not records.empty:
                value = records[metric_col].values[0]
                method_data.append(value)
            else:
                method_data.append(np.nan)
        data[method] = method_data

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(malicious_ratios))
    width = 0.13

    for i, method in enumerate(methods):
        offset = (i - 2.5) * width
        values = data[method]

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=METHOD_DISPLAY_NAMES[method],
            color=METHOD_COLORS[method],
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

    # 设置标签和标题
    scenario_name = 'Non-IID (Dirichlet α=0.5)' if partition_type == 'dirichlet' else 'IID'

    if metric_name == 'runtime_overhead':
        title = f'运行时间开销对比 ({scenario_name})'
        ylabel = '运行时间 (秒)'
    else:
        title = f'运行时间开销对比 - 不含密码学 ({scenario_name})'
        ylabel = '运行时间 (秒，不含密码学)'

    ax.set_xlabel('恶意节点比例', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(r*100)}%' for r in malicious_ratios])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存文件
    plots_dir = seed_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    if partition_type == 'dirichlet':
        filename = f'dirichlet_{alpha_str}_{metric_name}.svg'
    else:
        filename = f'{partition_type}_{metric_name}.svg'

    output_path = plots_dir / filename
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  ✓ 生成 {filename}")

def update_html_for_seed(seed_dir: Path):
    """更新单个seed目录的visualizations.html"""

    html_file = seed_dir / 'visualizations.html'

    if not html_file.exists():
        print(f"⚠️  {html_file} 不存在，跳过HTML更新")
        return

    # 读取现有HTML
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # 检查是否已经有运行时间图表的section
    if '<h2>运行时间开销</h2>' in html_content or '<h2>Runtime Overhead</h2>' in html_content:
        print(f"  ℹ️  HTML中已存在运行时间图表section，跳过更新")
        return

    # 在ASR section之后插入运行时间section
    runtime_section = '''
    <h2>运行时间开销</h2>
    <div class="visualization-grid">
        <div class="visualization-item">
            <h3>IID - 运行时间 (含密码学)</h3>
            <object data="plots/iid_runtime_overhead.svg" type="image/svg+xml"></object>
        </div>
        <div class="visualization-item">
            <h3>IID - 运行时间 (不含密码学)</h3>
            <object data="plots/iid_runtime_without_crypto.svg" type="image/svg+xml"></object>
        </div>
        <div class="visualization-item">
            <h3>Non-IID (α=0.5) - 运行时间 (含密码学)</h3>
            <object data="plots/dirichlet_alpha_0_5_runtime_overhead.svg" type="image/svg+xml"></object>
        </div>
        <div class="visualization-item">
            <h3>Non-IID (α=0.5) - 运行时间 (不含密码学)</h3>
            <object data="plots/dirichlet_alpha_0_5_runtime_without_crypto.svg" type="image/svg+xml"></object>
        </div>
    </div>
'''

    # 在</body>之前插入
    html_content = html_content.replace('</body>', runtime_section + '\n</body>')

    # 写回文件
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"  ✓ 更新 {html_file.name}")

def main():
    if len(sys.argv) < 2:
        print("用法: python generate_runtime_visualizations_per_seed.py <实验目录>")
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
        if generate_runtime_chart_for_seed(seed_dir):
            success_count += 1

    print(f"\n✅ 完成! 成功处理 {success_count}/{len(seed_dirs)} 个seed")

if __name__ == '__main__':
    main()
