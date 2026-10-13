#!/usr/bin/env python3
"""
快速分析 CIFAR-10 v9 多方法对比实验中 sm9rrs 在 IID ratio=0.4 失效的原因
"""

import pandas as pd
import sys

def analyze_failure():
    """分析 IID ratio=0.4 失效原因"""

    print("="*80)
    print("分析 sm9rrs 在 IID ratio=0.4 下失效的原因")
    print("="*80)

    # 读取汇总数据
    try:
        summary = pd.read_csv('outputs/cifar10_improved_v9_full_quick_validation/summary.csv')
    except FileNotFoundError:
        print("错误：找不到 summary.csv 文件")
        sys.exit(1)

    # 筛选 sm9rrs 的数据
    sm9rrs_data = summary[summary['method'] == 'sm9rrs'].copy()

    print("\n【sm9rrs 所有场景的 ASR 表现】")
    print("-"*80)
    for _, row in sm9rrs_data.iterrows():
        print(f"{row['partition']:10s} ratio={row['ratio']:.1f}  "
              f"ASR={row['target_asr']:.1%}  Acc={row['final_acc']:.1%}  "
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
    print(f"ASR: {row['target_asr']:.1%}")
    print(f"Accuracy: {row['final_acc']:.1%}")
    print(f"撤销客户端数: {row['blacklisted']:.0f}/100")
    print(f"运行时间: {row['runtime_s']:.1f}秒")
    print(f"Crypto时间: {row['crypto_s']:.1f}秒")
    print(f"训练时间: {row['train_s']:.1f}秒")

    # 对比其他方法在 IID 0.4 的表现
    print("\n【所有方法在 IID ratio=0.4 的 ASR 对比】")
    print("-"*80)
    iid_04_all = summary[
        (summary['partition'] == 'iid') &
        (summary['ratio'] == 0.4)
    ].sort_values('target_asr')

    for _, row in iid_04_all.iterrows():
        status = "✅" if row['target_asr'] < 0.15 else "❌"
        print(f"{row['method']:10s}  ASR={row['target_asr']:6.1%}  "
              f"Acc={row['final_acc']:5.1%}  撤销={row['blacklisted']:2.0f}  {status}")

    # 对比单独运行的成功案例
    print("\n【对比分析】")
    print("-"*80)
    print("单独运行 ratio=0.4 (seed=2026101302):")
    print("  - ASR: 10.5% ✅")
    print("  - Accuracy: 52.51%")
    print("  - 撤销: 39个客户端")
    print()
    print("多方法对比 IID ratio=0.4 (seed=2026101301):")
    print(f"  - ASR: {row['target_asr']:.1%} ❌")
    print(f"  - Accuracy: {row['final_acc']:.1%}")
    print(f"  - 撤销: {row['blacklisted']:.0f}个客户端")
    print()
    print("关键差异：")
    print("  1. Seed 不同：2026101302 vs 2026101301")
    print("  2. ASR 差异巨大：10.5% vs 80.5%")
    print("  3. 撤销数量差异：39 vs 34")
    print()
    print("结论：方案对客户端划分（seed）高度敏感，鲁棒性不足！")

    # 检查是否有诊断文件
    print("\n【诊断文件检查】")
    print("-"*80)
    import os
    diag_files = []
    for root, dirs, files in os.walk('outputs/cifar10_improved_v9_full_quick_validation/'):
        for f in files:
            if 'diagnostics' in f:
                diag_files.append(os.path.join(root, f))

    if diag_files:
        print(f"找到 {len(diag_files)} 个诊断文件：")
        for f in diag_files[:5]:
            print(f"  {f}")
        if len(diag_files) > 5:
            print(f"  ... 还有 {len(diag_files)-5} 个文件")
    else:
        print("⚠️ 未找到诊断文件，无法深入分析失效机制")
        print("可能需要单独运行 IID ratio=0.4 并保存详细诊断")

if __name__ == '__main__':
    analyze_failure()
