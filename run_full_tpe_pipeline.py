#!/usr/bin/env python3
"""
TPE 调优完整流程 - 冲击顶刊版本

分三阶段执行：
1. sm9rrs 快速调优（找到最优配置）
2. 全方法 TPE 调优（公平对比）
3. 完整实验（5 个 ratio，2 个分布）
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
        sys.exit(1)

    print(f"\n✅ 完成！耗时 {elapsed/60:.1f} 分钟")
    return result


def load_best_params(output_dir, method):
    """从 TPE 结果中加载最优参数"""
    tuning_result = Path(output_dir) / f"{method}_best_params.json"

    if not tuning_result.exists():
        print(f"⚠️  未找到 {method} 的最优参数: {tuning_result}")
        return None

    with open(tuning_result) as f:
        data = json.load(f)

    print(f"\n✅ {method} 最优参数:")
    print(json.dumps(data, indent=2))
    return data


def create_final_config(sm9rrs_params, vert_params, alignins_params):
    """创建最终对比配置"""
    config = {
        "schema_version": 3,
        "name": "CIFAR-10 Final Comparison - TPE Optimized",
        "description": "All methods with TPE-optimized parameters for top-tier publication",
        "experiments": []
    }

    methods_params = {
        "sm9rrs": sm9rrs_params,
        "vert": vert_params,
        "alignins": alignins_params,
        "krum": {},  # Krum 无需调参
        "ding13": {},
        "fedavg": {}
    }

    for partition in ["iid", "dirichlet"]:
        for ratio in [0.0, 0.2, 0.4, 0.6, 0.8]:
            for method, params in methods_params.items():
                exp = {
                    "partition": partition,
                    "clients": 100,
                    "method": method,
                    "malicious_ratio": ratio,
                    "dirichlet_alpha": 0.5,
                    "dataset": "cifar10",
                    "data_dir": "data/cifar10",
                    "download": True,
                    "output_dir": f"outputs/cifar10_v10_final_tpe_optimized",
                    "rounds": 150,  # 正式实验用 150 轮
                    "local_epochs": 1,
                    "batch_size": 50,
                    "lr": 0.05,
                    "lr_decay": 0.99,
                    "attack": "alternating_minimization",
                    "attack_epochs": 1,
                    "attack_source_label": 5,
                    "attack_target_label": 7,
                    "attack_target_count": 200,
                    "attack_start_round": 12,
                    "crypto_mode": "simulated",
                    "compute_backend": "torch",
                    "device": "cuda",
                    "early_stop": False,
                    "eval_interval": 1,
                    "seed": 2026101303
                }

                # 合并方法特定参数
                exp.update(params)
                config["experiments"].append(exp)

    output_path = "configs/cifar10_v10_final_tpe_optimized.json"
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ 最终配置已生成: {output_path}")
    print(f"   总实验数: {len(config['experiments'])}")
    return output_path


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   CIFAR-10 TPE 调优完整流程 - 冲击顶刊                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

流程:
  阶段 1: sm9rrs 快速调优       (~2-3 小时, 40 trials)
  阶段 2: 全方法 TPE 调优       (~4-6 小时, 30 trials × 3 方法)
  阶段 3: 最终完整实验          (~8-10 小时, 150 轮 × 60 配置)

总耗时: 14-19 小时

是否继续？[Y/n]: """)

    response = input().strip().lower()
    if response and response != 'y':
        print("已取消")
        sys.exit(0)

    # 阶段 1: sm9rrs 快速调优
    run_command(
        "python3 run_tpe_tuning.py --config configs/cifar10_v10_sm9rrs_focused_tpe.json",
        "阶段 1/3: sm9rrs 快速调优（2-3 小时）"
    )

    sm9rrs_params = load_best_params("outputs/cifar10_v10_sm9rrs_focused_tpe", "sm9rrs")
    if not sm9rrs_params:
        print("❌ 未能获取 sm9rrs 最优参数")
        sys.exit(1)

    print(f"\n📊 sm9rrs 优化结果:")
    print(f"   平均 ASR: {sm9rrs_params.get('mean_asr', 'N/A')}")
    print(f"   期望: < 7.0% (优于当前的 8.6%)")

    # 阶段 2: 全方法 TPE 调优
    run_command(
        "python3 run_tpe_tuning.py --config configs/cifar10_v10_full_tpe_tuning.json",
        "阶段 2/3: 全方法 TPE 调优（4-6 小时）"
    )

    vert_params = load_best_params("outputs/cifar10_v10_full_tpe_tuning", "vert")
    alignins_params = load_best_params("outputs/cifar10_v10_full_tpe_tuning", "alignins")

    # 阶段 3: 生成最终配置并运行
    final_config = create_final_config(sm9rrs_params, vert_params, alignins_params)

    print(f"\n{'='*80}")
    print("准备运行最终实验（150 轮，60 个配置）")
    print("预计耗时: 8-10 小时")
    print(f"{'='*80}")
    print("\n继续？[Y/n]: ")

    response = input().strip().lower()
    if response and response != 'y':
        print(f"\n✅ 配置已生成: {final_config}")
        print("   稍后可手动运行:")
        print(f"   python3 run_experiments_from_config.py --config {final_config}")
        sys.exit(0)

    run_command(
        f"python3 run_experiments_from_config.py --config {final_config}",
        "阶段 3/3: 最终完整实验（8-10 小时）"
    )

    print(f"\n{'='*80}")
    print("🎉 全部完成！")
    print(f"{'='*80}")
    print(f"\n结果目录: outputs/cifar10_v10_final_tpe_optimized/")
    print("\n分析结果:")
    print("  python3 analyze_tpe_final_results.py")


if __name__ == "__main__":
    main()
