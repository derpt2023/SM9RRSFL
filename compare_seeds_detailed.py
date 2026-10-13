#!/usr/bin/env python3
"""
对比两个 seed 的诊断数据，找出为什么一个成功一个失败
"""

import pandas as pd
import numpy as np

def compare_seeds():
    print("="*80)
    print("对比分析：seed=2026101301 (失败) vs seed=2026101302 (成功)")
    print("="*80)

    # 读取两个实验的诊断数据
    try:
        # 失败的实验
        fail_diag = pd.read_csv('outputs/cifar10_debug_seed1/sm9rrs_diagnostics.csv')
        print(f"\n✅ 读取失败实验诊断数据：{len(fail_diag)} 行")
    except:
        print("\n❌ 找不到 seed=2026101301 的诊断数据")
        return

    try:
        # 成功的实验
        succ_diag = pd.read_csv('outputs/cifar10_improved_v9_quick_validation/ratio0.4/sm9rrs_diagnostics.csv')
        print(f"✅ 读取成功实验诊断数据：{len(succ_diag)} 行")
    except:
        print("❌ 找不到 seed=2026101302 的诊断数据")
        return

    # 关键轮次：首攻击轮 (round 12)
    print("\n" + "="*80)
    print("【首攻击轮分析】round=12")
    print("="*80)

    fail_r12 = fail_diag[fail_diag['round'] == 12]
    succ_r12 = succ_diag[succ_diag['round'] == 12]

    print(f"\n失败实验 round 12: {len(fail_r12)} 个客户端")
    print(f"成功实验 round 12: {len(succ_r12)} 个客户端")

    # 分析恶意节点的 novelty_score
    print("\n【恶意节点 novelty 分布】")

    fail_mal = fail_r12[fail_r12['is_malicious'] == True]
    fail_hon = fail_r12[fail_r12['is_malicious'] == False]
    succ_mal = succ_r12[succ_r12['is_malicious'] == True]
    succ_hon = succ_r12[succ_r12['is_malicious'] == False]

    print(f"\n失败实验 (seed=2026101301):")
    print(f"  恶意节点数: {len(fail_mal)}")
    print(f"  恶意 novelty 均值: {fail_mal['novelty_score'].mean():.4f}")
    print(f"  恶意 novelty 最大: {fail_mal['novelty_score'].max():.4f}")
    print(f"  恶意 novelty 最小: {fail_mal['novelty_score'].min():.4f}")
    print(f"  诚实 novelty 均值: {fail_hon['novelty_score'].mean():.4f}")
    print(f"  诚实 novelty 最大: {fail_hon['novelty_score'].max():.4f}")

    print(f"\n成功实验 (seed=2026101302):")
    print(f"  恶意节点数: {len(succ_mal)}")
    print(f"  恶意 novelty 均值: {succ_mal['novelty_score'].mean():.4f}")
    print(f"  恶意 novelty 最大: {succ_mal['novelty_score'].max():.4f}")
    print(f"  恶意 novelty 最小: {succ_mal['novelty_score'].min():.4f}")
    print(f"  诚实 novelty 均值: {succ_hon['novelty_score'].mean():.4f}")
    print(f"  诚实 novelty 最大: {succ_hon['novelty_score'].max():.4f}")

    # 分析接纳率
    print("\n【首攻击轮接纳情况】")

    fail_mal_accepted = fail_mal['aggregation_accepted'].sum()
    fail_hon_accepted = fail_hon['aggregation_accepted'].sum()
    succ_mal_accepted = succ_mal['aggregation_accepted'].sum()
    succ_hon_accepted = succ_hon['aggregation_accepted'].sum()

    print(f"\n失败实验:")
    print(f"  恶意接纳: {fail_mal_accepted}/{len(fail_mal)} ({fail_mal_accepted/len(fail_mal)*100:.1f}%)")
    print(f"  诚实接纳: {fail_hon_accepted}/{len(fail_hon)} ({fail_hon_accepted/len(fail_hon)*100:.1f}%)")

    print(f"\n成功实验:")
    print(f"  恶意接纳: {succ_mal_accepted}/{len(succ_mal)} ({succ_mal_accepted/len(succ_mal)*100:.1f}%)")
    print(f"  诚实接纳: {succ_hon_accepted}/{len(succ_hon)} ({succ_hon_accepted/len(succ_hon)*100:.1f}%)")

    # 分析阈值
    print("\n【关键发现】")

    fail_threshold = fail_r12['detector_distance_threshold'].iloc[0] if len(fail_r12) > 0 else None
    succ_threshold = succ_r12['detector_distance_threshold'].iloc[0] if len(succ_r12) > 0 else None

    if fail_threshold:
        print(f"\n检测阈值: {fail_threshold}")

    fail_mal_above = (fail_mal['novelty_score'] > fail_threshold).sum() if fail_threshold else 0
    succ_mal_above = (succ_mal['novelty_score'] > succ_threshold).sum() if succ_threshold else 0

    print(f"\n失败实验: {fail_mal_above}/{len(fail_mal)} 恶意节点 novelty 超过阈值")
    print(f"成功实验: {succ_mal_above}/{len(succ_mal)} 恶意节点 novelty 超过阈值")

    # 关键差异
    print("\n" + "="*80)
    print("【核心差异总结】")
    print("="*80)

    if fail_mal['novelty_score'].mean() < succ_mal['novelty_score'].mean():
        diff = succ_mal['novelty_score'].mean() - fail_mal['novelty_score'].mean()
        print(f"\n⚠️  失败 seed 的恶意更新 novelty 比成功 seed 低 {diff:.4f}")
        print("   原因：客户端数据分布导致恶意更新不够显著")
        print("   解决：提高 attack_boost，使恶意更新更明显")

    if fail_mal_accepted > succ_mal_accepted:
        print(f"\n⚠️  失败 seed 接纳了更多恶意节点：{fail_mal_accepted} vs {succ_mal_accepted}")
        print("   原因：novelty 不够高，没有触发检测")
        print("   解决：降低 detector_distance_threshold 或提高 attack_boost")

    # 查看后续轮次的累积效应
    print("\n" + "="*80)
    print("【累积撤销分析】")
    print("="*80)

    fail_revoked = fail_diag.groupby('round')['revoked'].sum()
    succ_revoked = succ_diag.groupby('round')['revoked'].sum()

    print(f"\n失败实验总撤销: {fail_revoked.sum()} 次")
    print(f"成功实验总撤销: {succ_revoked.sum()} 次")

    # 查看 round 12-20 的撤销情况
    print(f"\n首攻击阶段 (round 12-20) 撤销:")
    print(f"  失败实验: {fail_revoked[12:21].sum()} 次")
    print(f"  成功实验: {succ_revoked[12:21].sum()} 次")

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == '__main__':
    compare_seeds()
