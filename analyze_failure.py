#!/usr/bin/env python3
"""
快速分析 CIFAR-10 v9 多方法对比实验中 sm9rrs 在 IID ratio=0.4 失效的原因
"""

import pandas as pd
import sys
import os

def analyze_failure():
    """分析 IID ratio=0.4 失效原因"""

    print("="*80)
    print("分析 sm9rrs 在 IID ratio=0.4 下失效的原因")
    print("="*80)

    # 读取汇总数据
    summary_path = 'outputs/cifar10_improved_v9_full_quick_validation/summary.csv'
    try:
        summary = pd.read_csv(summary_path)
    except FileNotFoundError:
        print(f"错误：找不到 {summary_path} 文件")
        sys.exit(1)

    # 检查列名
    print("\n【CSV 列名】")
    print(list(summary.columns))

    # 标准化列名
    if 'malicious_ratio' in summary.columns and 'ratio' not in summary.columns:
        summary['ratio'] = summary['malicious_ratio']

    # 筛选 sm9rrs 的数据
    sm9rrs_data = summary[summary['method'] == 'sm9rrs'].copy()

    print("\n【sm9rrs 所有场景的 ASR 表现】")
    print("-"*80)
    for _, row in sm9rrs_data.iterrows():
        print(f"{row['partition']:10s} ratio={row['ratio']:.1f}  "
              f"ASR={row['target_asr']*100:.1f}%  Acc={row['final_acc']*100:.1f}%  "
              f"撤销={row['blacklisted']:2.0f}个")

    # 重点分析 IID 0.4
    iid_04 = sm9rrs_data[
        (sm9rrs_data['partition'] == 'iid') &
        (sm9rrs_data['ratio'] == 0.4)
    ]

    if len(iid_04) == 0:
        print("\n错误：找不到 IID ratio=0.4 的数据")
        sys.exit(1)

    print("\n【IID ratio=0.4 详细信息】")
    print("-"*80)
    row = iid_04.iloc[0]
    print(f"ASR: {row['target_asr']*100:.1f}%")
    print(f"Accuracy: {row['final_acc']*100:.1f}%")
    print(f"撤销客户端数: {row['blacklisted']:.0f}/100")
    print(f"运行时间: {row['runtime_s']:.1f}秒")
    if 'crypto_s' in row:
        print(f"Crypto时间: {row['crypto_s']:.1f}秒")
    if 'train_s' in row:
        print(f"训练时间: {row['train_s']:.1f}秒")

    # 对比其他方法在 IID 0.4 的表现
    print("\n【所有方法在 IID ratio=0.4 的 ASR 对比】")
    print("-"*80)
    iid_04_all = summary[
        (summary['partition'] == 'iid') &
        (summary['ratio'] == 0.4)
    ].sort_values('target_asr')

    for _, row in iid_04_all.iterrows():
        asr_pct = row['target_asr'] * 100
        acc_pct = row['final_acc'] * 100
        status = "✅" if row['target_asr'] < 0.15 else "❌"
        print(f"{row['method']:10s}  ASR={asr_pct:6.1f}%  "
              f"Acc={acc_pct:5.1f}%  撤销={row['blacklisted']:2.0f}  {status}")

    # 对比单独运行的成功案例
    print("\n【对比分析】")
    print("-"*80)
    print("单独运行 ratio=0.4 (seed=2026101302):")
    print("  - ASR: 10.5% ✅")
    print("  - Accuracy: 52.51%")
    print("  - 撤销: 39个客户端")
    print()
    print("多方法对比 IID ratio=0.4 (seed=2026101301):")
    sm9rrs_iid04 = iid_04.iloc[0]
    print(f"  - ASR: {sm9rrs_iid04['target_asr']*100:.1f}% ❌")
    print(f"  - Accuracy: {sm9rrs_iid04['final_acc']*100:.1f}%")
    print(f"  - 撤销: {sm9rrs_iid04['blacklisted']:.0f}个客户端")
    print()
    print("关键差异：")
    print("  1. Seed 不同：2026101302 vs 2026101301")
    print(f"  2. ASR 差异巨大：10.5% vs {sm9rrs_iid04['target_asr']*100:.1f}%")
    print(f"  3. 撤销数量差异：39 vs {sm9rrs_iid04['blacklisted']:.0f}")
    print()
    print("结论：方案对客户端划分（seed）高度敏感，鲁棒性不足！")

    # 所有场景汇总对比
    print("\n【所有方法所有场景 ASR 汇总】")
    print("-"*80)

    # 创建透视表
    pivot = summary[summary['ratio'] > 0].pivot_table(
        values='target_asr',
        index='method',
        columns=['partition', 'ratio'],
        aggfunc='mean'
    )

    print("\nASR (%) 对比表：")
    print(pivot * 100)

    # 计算平均 ASR
    print("\n【各方法平均 ASR（所有攻击场景）】")
    print("-"*80)
    avg_asr = summary[summary['ratio'] > 0].groupby('method')['target_asr'].mean().sort_values()
    for method, asr in avg_asr.items():
        status = "✅" if asr < 0.20 else "⚠️" if asr < 0.40 else "❌"
        print(f"{method:10s}  平均 ASR={asr*100:5.1f}%  {status}")

    # 检查是否有诊断文件
    print("\n【诊断文件检查】")
    print("-"*80)
    diag_files = []
    for root, dirs, files in os.walk('outputs/cifar10_improved_v9_full_quick_validation/'):
        for f in files:
            if 'diagnostics' in f:
                diag_files.append(os.path.join(root, f))

    if diag_files:
        print(f"找到 {len(diag_files)} 个诊断文件：")
        for f in diag_files[:3]:
            print(f"  {f}")
        if len(diag_files) > 3:
            print(f"  ... 还有 {len(diag_files)-3} 个文件")
    else:
        print("⚠️ 未找到诊断文件")

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == '__main__':
    analyze_failure()
