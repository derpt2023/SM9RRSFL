#!/bin/bash
# 终止所有实验进程并启动新的调优

cd /3251901002/SM9RRSFL || exit 1

echo "🛑 终止现有进程..."
pkill -9 -f "python.*experiment"
pkill -9 -f "python.*tuning"
pkill -9 -f "run_fair_tuning"
pkill -9 -f "run_experiments"
sleep 2

echo "✅ 进程已清理"
echo ""

# 设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts/experiments/mnist"

echo "=================================="
echo "🚀 启动 CIFAR-10 优化调优 (v11)"
echo "=================================="
echo ""
echo "修复:"
echo "  ✅ attack_start_round: 12 → 25 (必须 > K=20)"
echo ""
echo "配置:"
echo "  - attack_boost: [20, 21, 22, 23, 24]"
echo "  - threshold: [1.7, 1.9, 2.1, 2.3]"
echo "  - 30 trials"
echo "  - 预计 9 小时"
echo ""
echo "开始时间: $(date)"
echo ""

# 拉取最新配置
git pull

# 运行调优
python3 run_fair_tuning_from_config.py --config configs/cifar10_v11_optimized_tuning.json

echo ""
echo "完成时间: $(date)"
echo ""
echo "查看结果:"
echo "  cat outputs/cifar10_v11_optimized_tuning/tuning_summary.json"
