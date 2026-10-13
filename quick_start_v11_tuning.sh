#!/bin/bash
# 快速启动优化调优 - 解决 PYTHONPATH 问题

cd /3251901002/SM9RRSFL || exit 1

# 设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts/experiments/mnist"

echo "=================================="
echo "🚀 启动 CIFAR-10 优化调优 (v11)"
echo "=================================="
echo ""
echo "配置:"
echo "  - attack_boost: [20, 21, 22, 23, 24]"
echo "  - threshold: [1.7, 1.9, 2.1, 2.3]"
echo "  - 30 trials"
echo "  - 预计 9 小时"
echo ""
echo "开始时间: $(date)"
echo ""

# 运行调优
python3 run_fair_tuning_from_config.py --config configs/cifar10_v11_optimized_tuning.json

echo ""
echo "完成时间: $(date)"
echo ""
echo "查看结果:"
echo "  cat outputs/cifar10_v11_optimized_tuning/tuning_summary.json"
