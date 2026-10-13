#!/usr/bin/env python3
"""
对比 boost=25 下各方法的表现
"""

import pandas as pd
import numpy as np

def analyze_all_methods():
    print("="*80)
    print("多方法对比分析 (attack_boost=25)")
    print("="*80)

    try:
        df = pd.read_csv('outputs/cifar10_v9_boost25_all_methods/summary.csv')
        print(f"\n✅ 读取数据：{len(df)} 行")
    except:
        print("\n❌ 数据文件不存在，请先运行实验")
        return

    # 提取关键列
    methods = df['method'].unique()

    print("\n" + "="*80)
    print("【各方法 ASR 对比】")
    print("="*80)

    # 创建对比表
    results = []
    for method in methods:
        method_df = df[df['method'] == method]
        attack_df = method_df[method_df['malicious_ratio'] > 0]

        if len(attack_df) == 0:
            continue

        avg_asr = attack_df['final_attack_target_success_rate'].mean() * 100
        iid_04 = attack_df[(attack_df['partition'] == 'iid') &
                           (attack_df['malicious_ratio'] == 0.4)]['final_attack_target_success_rate']
        iid_04_asr = iid_04.values[0] * 100 if len(iid_04) > 0 else np.nan

        results.append({
            'method': method,
            'avg_asr': avg_asr,
            'iid_04': iid_04_asr
        })

    results_df = pd.DataFrame(results).sort_values('avg_asr')

    print("\n方法          平均 ASR    IID 0.4 ASR")
    print("-" * 50)
    for _, row in results_df.iterrows():
        rank = "🥇" if _ == 0 else "🥈" if _ == 1 else "🥉" if _ == 2 else "  "
        print(f"{rank} {row['method']:12s}  {row['avg_asr']:6.1f}%      {row['iid_04']:6.1f}%")

    print("\n" + "="*80)
    print("【详细场景对比】")
    print("="*80)

    for partition in ['iid', 'dirichlet']:
        print(f"\n{partition.upper()} 分布:")
        print("-" * 60)
        print("方法          r=0.2 ASR    r=0.4 ASR")
        print("-" * 60)

        for method in sorted(methods):
            method_df = df[(df['method'] == method) & (df['partition'] == partition)]
            r02 = method_df[method_df['malicious_ratio'] == 0.2]['final_attack_target_success_rate']
            r04 = method_df[method_df['malicious_ratio'] == 0.4]['final_attack_target_success_rate']

            r02_val = r02.values[0] * 100 if len(r02) > 0 else np.nan
            r04_val = r04.values[0] * 100 if len(r04) > 0 else np.nan

            print(f"{method:12s}  {r02_val:6.1f}%       {r04_val:6.1f}%")

    print("\n" + "="*80)
    print("【结论】")
    print("="*80)

    sm9rrs_rank = results_df[results_df['method'] == 'sm9rrs'].index[0] + 1
    best_method = results_df.iloc[0]['method']
    best_asr = results_df.iloc[0]['avg_asr']
    sm9rrs_asr = results_df[results_df['method'] == 'sm9rrs']['avg_asr'].values[0]

    if sm9rrs_rank == 1:
        print(f"\n🎉 sm9rrs 是最优方法！平均 ASR = {sm9rrs_asr:.1f}%")
    elif sm9rrs_rank == 2:
        diff = sm9rrs_asr - best_asr
        print(f"\n🥈 sm9rrs 排名第二，仅比 {best_method} 高 {diff:.1f} 个百分点")
        print(f"   {best_method}: {best_asr:.1f}% vs sm9rrs: {sm9rrs_asr:.1f}%")
    else:
        diff = sm9rrs_asr - best_asr
        print(f"\n⚠️  sm9rrs 排名第 {sm9rrs_rank}，比最优的 {best_method} 高 {diff:.1f} 个百分点")

    print("\n" + "="*80)

if __name__ == '__main__':
    analyze_all_methods()
