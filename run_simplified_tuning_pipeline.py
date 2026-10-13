#!/usr/bin/env python3
"""
简化版 TPE 调优流程 - 使用项目现有的 fair_tuning 框架

流程:
1. sm9rrs 网格搜索调优（2-3 小时）
2. VERT 修复（使用 MNIST 最优参数）
3. 最终完整实验（150 轮）
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, desc):
    """执行命令并显示进度"""
    print(f"\n{'='*80}")
    print(f"▶️  {desc}")
    print(f"{'='*80}")
    print(f"命令: {cmd}\n")

    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n❌ 失败！耗时 {elapsed/60:.1f} 分钟")
        return False

    print(f"\n✅ 完成！耗时 {elapsed/60:.1f} 分钟")
    return True


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              CIFAR-10 调优简化流程 - 基于 fair_tuning                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

流程:
  阶段 1: sm9rrs 网格搜索       (~2-3 小时, 20 候选配置)
  阶段 2: VERT 修复             (~10 分钟)
  阶段 3: 最终完整实验          (~3-4 小时, 150 轮)

总耗时: 约 6-7 小时

是否继续？[Y/n]: """)

    response = input().strip().lower()
    if response and response != 'y':
        print("已取消")
        sys.exit(0)

    # 阶段 1: sm9rrs 网格搜索
    success = run_command(
        "python3 run_fair_tuning_from_config.py --config configs/cifar10_v10_sm9rrs_grid_tuning.json",
        "阶段 1/3: sm9rrs 网格搜索调优（2-3 小时）"
    )

    if not success:
        print("\n❌ sm9rrs 调优失败")
        sys.exit(1)

    # 检查调优结果
    tuning_output = Path("outputs/cifar10_v10_sm9rrs_focused_tuning")
    summary_file = tuning_output / "tuning_summary.json"

    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)

        best_config = summary.get('sm9rrs', {}).get('best_config', {})
        best_score = summary.get('sm9rrs', {}).get('best_score', {})

        print(f"\n{'='*80}")
        print("📊 sm9rrs 调优结果:")
        print(f"{'='*80}")
        print(f"最优 ASR: {best_score.get('mean_asr', 'N/A')}")
        print(f"最优配置关键参数:")
        print(f"  attack_boost: {best_config.get('attack_boost', 'N/A')}")
        print(f"  detector_distance_threshold: {best_config.get('detector_distance_threshold', 'N/A')}")
        print(f"  detector_subspace_dim: {best_config.get('detector_subspace_dim', 'N/A')}")
    else:
        print(f"\n⚠️  未找到调优总结文件: {summary_file}")

    # 阶段 2: VERT 修复
    print(f"\n{'='*80}")
    print("准备运行 VERT 优化...")
    print(f"{'='*80}")
    print("\n继续？[Y/n]: ")

    response = input().strip().lower()
    if response and response != 'y':
        print(f"\n✅ sm9rrs 调优已完成")
        print("   稍后可手动运行 VERT 优化:")
        print("   python3 run_experiments_from_config.py --config configs/cifar10_vert_optimized_test.json")
        sys.exit(0)

    success = run_command(
        "python3 run_experiments_from_config.py --config configs/cifar10_vert_optimized_test.json",
        "阶段 2/3: VERT 优化（10 分钟）"
    )

    if not success:
        print("\n⚠️  VERT 优化失败，但可以继续")

    # 阶段 3: 询问是否运行最终实验
    print(f"\n{'='*80}")
    print("准备运行最终完整实验（150 轮，所有方法）")
    print("预计耗时: 3-4 小时")
    print(f"{'='*80}")
    print("\n继续？[Y/n]: ")

    response = input().strip().lower()
    if response and response != 'y':
        print(f"\n✅ 调优已完成！")
        print("\n下一步: 根据 sm9rrs 最优参数创建最终实验配置")
        print("   查看: outputs/cifar10_v10_sm9rrs_focused_tuning/tuning_summary.json")
        sys.exit(0)

    print("\n⚠️  最终实验配置需要手动创建（基于调优结果）")
    print("   请查看 outputs/cifar10_v10_sm9rrs_focused_tuning/ 中的最优参数")


if __name__ == "__main__":
    main()
