#!/usr/bin/env python3
"""
优化版调优流程 - 基于今天的实验发现

使用优化后的网格搜索配置 (v11)，目标：ASR < 7.0%
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
║           CIFAR-10 优化调优流程 - 基于今天的成功经验 (v11)                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

今天的关键发现:
  ✅ boost=20 + threshold=2.0 → 平均 ASR=13.5% (balanced 配置)
  ✅ IID 0.4 从 80.5% → 11.0% (巨大改善)
  ✅ sm9rrs 准确率 53.6% (所有方法中最高)

新配置改进:
  - attack_boost: 20-24 (窄化到最优区间)
  - threshold: 1.7-2.3 (围绕成功的 2.0)
  - subspace_dim: 固定为 3 (减少搜索空间)
  - 30 个 trials (比之前多 50%)

目标: 将 sm9rrs 从 8.6% 优化到 ≤ 7.0%

流程:
  阶段 1: sm9rrs 优化调优       (~9 小时, 30 trials)
  阶段 2: VERT 修复             (~10 分钟)
  阶段 3: 补充实验 (ratio 0.6, 0.8) (~40 分钟)

总耗时: 约 10 小时 (建议挂机过夜)

是否继续？[Y/n]: """)

    response = input().strip().lower()
    if response and response != 'y':
        print("已取消")
        sys.exit(0)

    # 阶段 1: sm9rrs 优化调优
    print("\n" + "="*80)
    print("📊 配置详情:")
    print("="*80)
    print("  attack_boost: [20, 21, 22, 23, 24]")
    print("  detector_distance_threshold: [1.7, 1.9, 2.1, 2.3]")
    print("  detector_subspace_dim: [3] (固定)")
    print("  trials: 30 个候选配置")
    print("  预计耗时: 9 小时")
    print("\n继续？[Y/n]: ")

    response = input().strip().lower()
    if response and response != 'y':
        print("已取消")
        sys.exit(0)

    # 设置 PYTHONPATH 以找到 fair_tuning_progress_display 模块
    cmd = (
        "export PYTHONPATH=\"${PYTHONPATH}:$(pwd)/scripts/experiments/mnist\" && "
        "python3 run_fair_tuning_from_config.py --config configs/cifar10_v11_optimized_tuning.json"
    )
    success = run_command(
        cmd,
        "阶段 1/3: sm9rrs 优化调优（9 小时）"
    )

    if not success:
        print("\n❌ sm9rrs 调优失败")
        sys.exit(1)

    # 检查调优结果
    tuning_output = Path("outputs/cifar10_v11_optimized_tuning")
    summary_file = tuning_output / "tuning_summary.json"

    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)

        best_config = summary.get('sm9rrs', {}).get('best_config', {})
        best_score = summary.get('sm9rrs', {}).get('best_score', {})

        print(f"\n{'='*80}")
        print("🎉 sm9rrs 调优结果:")
        print(f"{'='*80}")

        mean_asr = best_score.get('mean_asr', -1)
        if mean_asr >= 0:
            print(f"最优平均 ASR: {mean_asr*100:.2f}%")

            if mean_asr <= 0.070:
                print("✅ 优秀！已达到目标 (≤7.0%)")
            elif mean_asr <= 0.080:
                print("✅ 良好！接近目标")
            else:
                print("⚠️  仍需改进")

        print(f"\n最优配置关键参数:")
        print(f"  attack_boost: {best_config.get('attack_boost', 'N/A')}")
        print(f"  detector_distance_threshold: {best_config.get('detector_distance_threshold', 'N/A')}")
        print(f"  detector_subspace_dim: {best_config.get('detector_subspace_dim', 'N/A')}")
        print(f"  detector_reject_threshold: {best_config.get('detector_reject_threshold', 'N/A')}")
    else:
        print(f"\n⚠️  未找到调优总结文件: {summary_file}")

    # 阶段 2: VERT 修复
    print(f"\n{'='*80}")
    print("准备运行 VERT 优化（使用 MNIST 最优参数）...")
    print("预计耗时: 10 分钟")
    print(f"{'='*80}")
    print("\n继续？[Y/n]: ")

    response = input().strip().lower()
    if response and response != 'y':
        print(f"\n✅ sm9rrs 调优已完成")
        print("\n下一步建议:")
        print("  1. 查看调优结果: cat outputs/cifar10_v11_optimized_tuning/tuning_summary.json")
        print("  2. 运行 VERT 优化: python3 run_experiments_from_config.py --config configs/cifar10_vert_optimized_test.json")
        print("  3. 补充实验: python3 run_experiments_from_config.py --config configs/cifar10_v9_balanced_supplement.json")
        sys.exit(0)

    success = run_command(
        "python3 run_experiments_from_config.py --config configs/cifar10_vert_optimized_test.json",
        "阶段 2/3: VERT 优化（10 分钟）"
    )

    if not success:
        print("\n⚠️  VERT 优化失败，但可以继续")

    # 阶段 3: 补充实验
    print(f"\n{'='*80}")
    print("准备补充实验（ratio=0.6, 0.8）")
    print("使用 balanced 配置 (boost=20, threshold=2.0)")
    print("预计耗时: 40 分钟")
    print(f"{'='*80}")
    print("\n继续？[Y/n]: ")

    response = input().strip().lower()
    if response and response != 'y':
        print(f"\n✅ 前两阶段已完成！")
        print("\n稍后可手动运行补充实验:")
        print("  python3 run_experiments_from_config.py --config configs/cifar10_v9_balanced_supplement.json")
        sys.exit(0)

    success = run_command(
        "python3 run_experiments_from_config.py --config configs/cifar10_v9_balanced_supplement.json",
        "阶段 3/3: 补充实验 (ratio 0.6, 0.8) - 40 分钟"
    )

    # 完成总结
    print(f"\n{'='*80}")
    print("🎉 所有阶段完成！")
    print(f"{'='*80}")
    print("\n下一步:")
    print("  1. 查看最优参数: cat outputs/cifar10_v11_optimized_tuning/tuning_summary.json")
    print("  2. 基于最优参数创建最终实验配置（150 轮，所有方法）")
    print("  3. 运行最终对比实验")
    print("\n需要帮助创建最终配置吗？联系助手！")


if __name__ == "__main__":
    main()
