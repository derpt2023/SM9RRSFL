#!/usr/bin/env python3
"""
快速分析当前运行的多方法对比结果，并给出下一步建议
"""

import pandas as pd
import sys
from pathlib import Path

def analyze_current_results():
    """分析 cifar10_v9_balanced_all_methods 的结果"""

    summary_path = Path("outputs/cifar10_v9_balanced_all_methods/summary.csv")

    if not summary_path.exists():
        print("❌ 实验还未完成，summary.csv 不存在")
        return False

    df = pd.read_csv(summary_path)

    # 过滤攻击场景
    attack_df = df[df['malicious_ratio'] > 0].copy()

    print("=" * 80)
    print("📊 多方法对比结果（attack_boost=20, balanced 配置）")
    print("=" * 80)

    # 按方法汇总
    method_stats = attack_df.groupby('method').agg({
        'final_attack_target_success_rate': ['mean', 'std', 'min', 'max'],
        'final_accuracy': 'mean'
    }).round(4)

    print("\n各方法平均表现:")
    print("-" * 80)
    for method in method_stats.index:
        asr_mean = method_stats.loc[method, ('final_attack_target_success_rate', 'mean')]
        asr_std = method_stats.loc[method, ('final_attack_target_success_rate', 'std')]
        acc_mean = method_stats.loc[method, ('final_accuracy', 'mean')]

        rank = "🥇" if asr_mean < 0.12 else "🥈" if asr_mean < 0.15 else "🥉" if asr_mean < 0.20 else "⚠️"

        print(f"{method:12s} {rank}  ASR: {asr_mean*100:5.1f}% ± {asr_std*100:4.1f}%  Acc: {acc_mean*100:5.1f}%")

    # 详细场景对比
    print("\n\n详细场景对比:")
    print("-" * 80)
    print(f"{'方法':<12s} {'分布':<10s} {'ratio':<6s} {'ASR':<8s} {'Accuracy':<10s} {'撤销':<10s}")
    print("-" * 80)

    for _, row in attack_df.sort_values(['partition', 'malicious_ratio', 'method']).iterrows():
        print(f"{row['method']:<12s} {row['partition']:<10s} {row['malicious_ratio']:<6.1f} "
              f"{row['final_attack_target_success_rate']*100:6.1f}% {row['final_accuracy']*100:8.1f}% "
              f"{row['blacklisted_clients']:<10s}")

    # 排名
    print("\n\n" + "=" * 80)
    print("🏆 最终排名（按平均 ASR）")
    print("=" * 80)

    rankings = attack_df.groupby('method')['final_attack_target_success_rate'].mean().sort_values()

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"]
    for i, (method, asr) in enumerate(rankings.items()):
        medal = medals[i] if i < len(medals) else "  "
        print(f"{medal} {i+1}. {method:12s}  平均 ASR: {asr*100:5.1f}%")

    # 分析 sm9rrs 排名
    sm9rrs_rank = list(rankings.index).index('sm9rrs') + 1 if 'sm9rrs' in rankings.index else None
    sm9rrs_asr = rankings.get('sm9rrs', None)

    print("\n\n" + "=" * 80)
    print("🎯 决策建议")
    print("=" * 80)

    if sm9rrs_rank is None:
        print("❌ sm9rrs 结果未找到")
        return False

    print(f"\nsm9rrs 排名: 第 {sm9rrs_rank} 名")
    print(f"sm9rrs 平均 ASR: {sm9rrs_asr*100:.1f}%")

    # 检查 VERT
    vert_asr = rankings.get('vert', None)
    if vert_asr and vert_asr > 0.30:
        print(f"\n⚠️  VERT 表现很差 (ASR={vert_asr*100:.1f}%)，需要用优化参数重跑！")
        print("   运行: python3 run_experiments_from_config.py --config configs/cifar10_vert_optimized_test.json")

    # 给出建议
    print("\n" + "-" * 80)

    if sm9rrs_rank <= 2:
        print("✅ 建议: sm9rrs 表现优秀！")
        print("\n下一步:")
        print("1. 补充 ratio=0.6 和 0.8 的实验")
        print("   运行: python3 run_experiments_from_config.py --config configs/cifar10_v9_balanced_supplement.json")
        print("\n2. 然后进行 150 轮正式实验")

    elif sm9rrs_rank == 3:
        print("✅ 建议: sm9rrs 表现良好（第三名）")
        print("\n可选:")
        print("- 尝试微调检测器参数")
        print("- 或直接用当前结果撰写论文（前三名已经很好）")
        print("\n无论如何，先补充 ratio=0.6 和 0.8:")
        print("运行: python3 run_experiments_from_config.py --config configs/cifar10_v9_balanced_supplement.json")

    else:
        print("⚠️  建议: sm9rrs 排名较后，考虑进一步优化")
        print("\n选项:")
        print("1. 调整检测器参数（detector_subspace_dim, thresholds）")
        print("2. 或调整论文角度，强调 MNIST 优势和加密特性")

    # 对比基准
    print("\n" + "-" * 80)
    print("📊 与 MNIST 表现对比:")
    print("-" * 80)

    mnist_results = {
        'sm9rrs': '< 2%',
        'vert': '2.34%',
        'krum': '~10%',
        'alignins': '~11%'
    }

    for method in ['sm9rrs', 'vert', 'krum', 'alignins']:
        if method in rankings:
            cifar_asr = rankings[method] * 100
            mnist_asr = mnist_results.get(method, 'N/A')
            print(f"{method:12s}  MNIST: {mnist_asr:>7s}  →  CIFAR-10: {cifar_asr:5.1f}%")

    return True

if __name__ == "__main__":
    success = analyze_current_results()
    sys.exit(0 if success else 1)
