#!/usr/bin/env python3
"""检查当前配置下GPU利用率和并行度"""

import json
import sys
from pathlib import Path

def estimate_experiment_count(config_path):
    """估算实验总数"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    shared = config['shared_parameters']
    tuning = config['tuning']

    # 基础配置
    # Fair tuning 从 shared_parameters 的 methods 推断，没有独立的 malicious_ratios
    # 使用固定的5个ratio: [0.0, 0.1, 0.2, 0.3, 0.4]
    num_ratios = 5

    # Validation阶段
    # 可调方法: sm9rrs, vert, alignins (如果存在candidates)
    tunable_methods = ['sm9rrs', 'vert', 'alignins']
    trials_per_method = tuning['trials_per_tunable_method']
    validation_seeds = len(tuning['validation_seeds'])

    method_spaces = tuning['method_spaces']
    validation_experiments = 0

    for method in tunable_methods:
        if method in method_spaces and method_spaces[method]:
            # 每个方法有 trials_per_method 个候选
            validation_experiments += trials_per_method * validation_seeds * num_ratios

    # Fixed方法 (不可调): krum, ding13, fedavg
    # 这些方法没有候选搜索，但也需要在validation阶段运行
    fixed_methods = ['krum', 'ding13', 'fedavg']
    for method in fixed_methods:
        # 每个固定方法只有1个配置
        validation_experiments += 1 * validation_seeds * num_ratios

    # Final阶段
    final_seeds = len(tuning['final_seeds'])
    all_methods = 6  # sm9rrs, vert, alignins, krum, ding13, fedavg
    final_experiments = all_methods * final_seeds * num_ratios

    return {
        'validation': validation_experiments,
        'final': final_experiments,
        'total': validation_experiments + final_experiments,
        'num_ratios': num_ratios,
        'validation_seeds': validation_seeds,
        'final_seeds': final_seeds,
        'trials_per_tunable_method': trials_per_method,
    }

def analyze_parallel_strategy():
    """分析并行策略"""
    print("=" * 80)
    print("GPU 并行度分析")
    print("=" * 80)

    config_path = Path(__file__).parent / 'configs' / 'cifar10_v13_fair_tuning_fixed.json'

    stats = estimate_experiment_count(config_path)

    print(f"\n实验总数估算:")
    print(f"  Validation 阶段: {stats['validation']} 个实验")
    print(f"    - 可调方法 (sm9rrs, vert, alignins): 每个 {stats['trials_per_tunable_method']} trials")
    print(f"    - 固定方法 (krum, ding13, fedavg): 每个 1 config")
    print(f"    - Validation seeds: {stats['validation_seeds']}")
    print(f"    - Malicious ratios: {stats['num_ratios']}")
    print(f"  Final 阶段: {stats['final']} 个实验")
    print(f"    - 所有6个方法，每个选最优候选")
    print(f"    - Final seeds: {stats['final_seeds']}")
    print(f"  总计: {stats['total']} 个实验")

    print(f"\n当前并行策略 (根据代码分析):")
    print(f"  1. Backend: torch:cuda")
    print(f"  2. Jobs 参数: auto (由 resolve_parallel_jobs 自动计算)")
    print(f"  3. Executor: ThreadPoolExecutor (CUDA后端)")
    print(f"  4. 设备分配策略: assign_auto_cuda_devices (轮询分配)")

    print(f"\n关键发现:")
    print(f"  ⚠️  _cuda_memory_parallel_limit 的限制:")
    print(f"      - 每个GPU只运行1个实验 (避免GPU竞争)")
    print(f"      - 代码: 'Auto mode deliberately runs at most one independent experiment per GPU'")
    print(f"      - 这意味着: 8个GPU = 最多8个并行任务")

    print(f"\n  ⚠️  ThreadPoolExecutor 工作方式:")
    print(f"      - max_workers 设为 min(jobs, cuda_limited)")
    print(f"      - cuda_limited 通过 _cuda_memory_parallel_limit 计算")
    print(f"      - 结果: 即使有8个GPU，也只会同时运行 min(8, gpu_count) 个任务")

    print(f"\n时间估算 (假设8个GPU全部可用):")
    # 假设每个实验10分钟
    minutes_per_experiment = 10
    parallel_factor = 8  # 8个GPU

    validation_time = (stats['validation'] * minutes_per_experiment) / parallel_factor
    final_time = (stats['final'] * minutes_per_experiment) / parallel_factor
    total_time = validation_time + final_time

    print(f"  假设每个实验 {minutes_per_experiment} 分钟:")
    print(f"    - Validation: {validation_time:.1f} 分钟 = {validation_time/60:.1f} 小时")
    print(f"    - Final: {final_time:.1f} 分钟 = {final_time/60:.1f} 小时")
    print(f"    - 总计: {total_time:.1f} 分钟 = {total_time/60:.1f} 小时")

    print(f"\n改进建议:")
    print(f"  1. ✅ 当前代码已经支持多GPU并行")
    print(f"  2. ✅ 设备分配策略正确 (轮询分配到8个GPU)")
    print(f"  3. ⚠️  但是: 每个GPU同时只运行1个实验 (这是设计决策)")
    print(f"  4. 如果想要更激进的并行:")
    print(f"     - 修改 _cuda_memory_parallel_limit 允许每个GPU运行多个小实验")
    print(f"     - 风险: GPU内存竞争、训练速度变慢")

    print(f"\n结论:")
    print(f"  - 代码已经充分利用8个GPU")
    print(f"  - 但策略是: 每个GPU同时只跑1个实验 (保证质量)")
    print(f"  - 这是合理的设计，避免GPU竞争导致性能下降")
    print(f"  - 预计18小时是基于这个策略的合理估算")

    print("=" * 80)

if __name__ == '__main__':
    analyze_parallel_strategy()
