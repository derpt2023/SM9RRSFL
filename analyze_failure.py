#!/usr/bin/env python3
"""
快速分析 CIFAR-10 v9 多方法对比实验中 sm9rrs 在 IID ratio=0.4 失效的原因
"""

import pandas as pd
import numpy as np
import sys
import os

def safe_float(val, default=0.0):
    """安全转换为浮点数，处理 NaN"""
    if pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

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

    # 标准化列名（创建简短别名）
    summary['ratio'] = summary['malicious_ratio']
    summary['asr'] = summary['final_attack_target_success_rate']
    summary['acc'] = summary['final_accuracy']
    summary['blacklisted'] = summary['blacklisted_clients'].fillna(0)
    summary['runtime_s'] = summary['runtime_seconds']

    # 筛选 sm9rrs 的数据
    sm9rrs_data = summary[summary['method'] == 'sm9rrs'].copy()

    print("\n【sm9rrs 所有场景的 ASR 表现】")
    print("-"*80)
    for _, row in sm9rrs_data.iterrows():
        ratio_val = safe_float(row['ratio'])
        asr_val = safe_float(row['asr'])
        acc_val = safe_float(row['acc'])
        bl_val = safe_float(row['blacklisted'])

        print(f"{row['partition']:10s} ratio={ratio_val:.1f}  "
              f"ASR={asr_val*100:5.1f}%  Acc={acc_val*100:5.1f}%  "
              f"撤销={bl_val:2.0f}个")

    # 重点分析 IID 0.4
    iid_04 = sm9rrs_data[
        (sm9rrs_data['partition'] == 'iid') &
        (sm9rrs_data['ratio'] == 0.4)
    ]

    if len(iid_04) == 0:
        print("\n⚠️ 未找到 IID ratio=0.4 的数据")
        print("可能的原因：")
        print("  1. 实验尚未完成")
        print("  2. 数据文件不完整")
        print("  3. 配置文件中未包含此场景")
    else:
        print("\n【IID ratio=0.4 详细信息】")
        print("-"*80)
        row = iid_04.iloc[0]
        print(f"ASR: {safe_float(row['asr'])*100:.1f}%")
        print(f"Accuracy: {safe_float(row['acc'])*100:.1f}%")
        print(f"撤销客户端数: {safe_float(row['blacklisted']):.0f}/{safe_float(row['num_clients']):.0f}")
        print(f"运行时间: {safe_float(row['runtime_s']):.1f}秒")
        if 'crypto_wall_seconds' in summary.columns:
            print(f"Crypto时间: {safe_float(row.get('crypto_wall_seconds', 0)):.1f}秒")
        if 'training_seconds' in summary.columns:
            print(f"训练时间: {safe_float(row.get('training_seconds', 0)):.1f}秒")

    # 对比其他方法在 IID 0.4 的表现
    print("\n【所有方法在 IID ratio=0.4 的 ASR 对比】")
    print("-"*80)
    print(f"{'方法':<12} {'ASR':>8} {'Accuracy':>10} {'撤销':>6} {'状态':>4}")
    print("-"*80)

    iid_04_all = summary[
        (summary['partition'] == 'iid') &
        (summary['ratio'] == 0.4)
    ].sort_values('asr')

    if len(iid_04_all) == 0:
        print("⚠️ 未找到任何方法在 IID ratio=0.4 的数据")
    else:
        for _, row in iid_04_all.iterrows():
            asr_pct = safe_float(row['asr']) * 100
            acc_pct = safe_float(row['acc']) * 100
            bl_val = safe_float(row['blacklisted'])
            status = "✅" if row['asr'] < 0.15 else "⚠️" if row['asr'] < 0.40 else "❌"
            print(f"{row['method']:<12} {asr_pct:7.1f}% {acc_pct:9.1f}% {bl_val:5.0f} {status:>4}")

    # 对比单独运行的成功案例（如果 IID 0.4 数据存在）
    if len(iid_04) > 0:
        print("\n【对比分析：单独运行 vs 多方法对比】")
        print("-"*80)
        print("单独运行 ratio=0.4 (seed=2026101302):")
        print("  - ASR: 10.5% ✅")
        print("  - Accuracy: 52.51%")
        print("  - 撤销: 39个客户端")
        print()
        print("多方法对比 IID ratio=0.4 (seed=2026101301):")
        sm9rrs_iid04 = iid_04.iloc[0]
        asr_val = safe_float(sm9rrs_iid04['asr'])
        acc_val = safe_float(sm9rrs_iid04['acc'])
        bl_val = safe_float(sm9rrs_iid04['blacklisted'])

        print(f"  - ASR: {asr_val*100:.1f}% {'✅' if asr_val < 0.15 else '❌'}")
        print(f"  - Accuracy: {acc_val*100:.1f}%")
        print(f"  - 撤销: {bl_val:.0f}个客户端")
        print()
        print("【关键差异】")
        print("  1. Seed 不同：2026101302 (成功) vs 2026101301 (当前)")
        print(f"  2. ASR 差异：10.5% vs {asr_val*100:.1f}% (相差 {abs(10.5 - asr_val*100):.1f} 个百分点)")
        print(f"  3. 撤销数量：39 vs {bl_val:.0f}")
        print()
        if asr_val > 0.50:
            print("⚠️  结论：方案对客户端划分（seed）高度敏感，鲁棒性严重不足！")
        else:
            print("✅ 结论：虽然有差异，但两个 seed 的 ASR 都在可接受范围")

    # 所有场景汇总对比
    print("\n【所有方法所有场景 ASR 汇总表】")
    print("="*80)

    # 只看攻击场景（ratio > 0）
    attack_data = summary[summary['ratio'] > 0].copy()

    if len(attack_data) == 0:
        print("⚠️ 未找到任何攻击场景数据（ratio > 0）")
    else:
        # 按方法和场景分组
        print("\n按分布类型和恶意比例的 ASR (%)：")
        print("-"*80)

        for partition in ['iid', 'dirichlet']:
            part_data = attack_data[attack_data['partition'] == partition]
            if len(part_data) == 0:
                continue

            print(f"\n{partition.upper()} 分布:")
            print(f"{'方法':<12} {'r=0.2':>8} {'r=0.4':>8} {'平均':>8}")
            print("-"*50)

            for method in ['sm9rrs', 'vert', 'alignins', 'krum', 'ding13', 'fedavg']:
                method_data = part_data[part_data['method'] == method]

                if len(method_data) == 0:
                    continue

                r02_data = method_data[method_data['ratio'] == 0.2]
                r04_data = method_data[method_data['ratio'] == 0.4]

                r02_val = safe_float(r02_data['asr'].values[0]) * 100 if len(r02_data) > 0 else float('nan')
                r04_val = safe_float(r04_data['asr'].values[0]) * 100 if len(r04_data) > 0 else float('nan')

                if not np.isnan(r02_val) and not np.isnan(r04_val):
                    avg = (r02_val + r04_val) / 2
                    print(f"{method:<12} {r02_val:7.1f}% {r04_val:7.1f}% {avg:7.1f}%")
                elif not np.isnan(r02_val):
                    print(f"{method:<12} {r02_val:7.1f}%    -       -")
                elif not np.isnan(r04_val):
                    print(f"{method:<12}    -    {r04_val:7.1f}%    -")

        # 计算总体平均 ASR
        print("\n【各方法总体平均 ASR（所有攻击场景）】")
        print("="*80)
        avg_asr = attack_data.groupby('method')['asr'].mean().sort_values()

        print(f"{'方法':<12} {'平均 ASR':>12} {'状态':>6}")
        print("-"*35)
        for method, asr in avg_asr.items():
            status = "✅ 优秀" if asr < 0.20 else "⚠️  一般" if asr < 0.40 else "❌ 失败"
            print(f"{method:<12} {asr*100:11.1f}% {status:>6}")

        # 关键发现
        print("\n【关键发现】")
        print("="*80)

        sm9rrs_avg = avg_asr.get('sm9rrs', float('nan'))
        alignins_avg = avg_asr.get('alignins', float('nan'))

        if not np.isnan(sm9rrs_avg) and not np.isnan(alignins_avg):
            if alignins_avg < sm9rrs_avg:
                diff = (sm9rrs_avg - alignins_avg) * 100
                print(f"⚠️  AlignIns 平均 ASR 比 sm9rrs 低 {diff:.1f} 个百分点")
                print(f"   sm9rrs: {sm9rrs_avg*100:.1f}%  vs  alignins: {alignins_avg*100:.1f}%")
                print()
                print("   这意味着 AlignIns 在整体上更鲁棒！")
            else:
                diff = (alignins_avg - sm9rrs_avg) * 100
                print(f"✅ sm9rrs 平均 ASR 比 AlignIns 低 {diff:.1f} 个百分点")
                print(f"   sm9rrs: {sm9rrs_avg*100:.1f}%  vs  alignins: {alignins_avg*100:.1f}%")

            print()
            print("【建议】")
            if sm9rrs_avg < 0.20:
                print("✅ sm9rrs 整体表现良好（平均 ASR < 20%）")
                print("   建议：运行多 seed 正式实验以验证稳定性")
            elif sm9rrs_avg < 0.40:
                print("⚠️  sm9rrs 表现一般（平均 ASR 20-40%）")
                print("   建议：调整参数提高性能，或运行多 seed 确认")
            else:
                print("❌ sm9rrs 整体表现不佳（平均 ASR > 40%）")
                print("   建议：重新审视方案或调整关键参数")

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)

if __name__ == '__main__':
    analyze_failure()
