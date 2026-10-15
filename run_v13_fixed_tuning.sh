#!/bin/bash
set -e

echo "🛑 终止现有进程..."
pkill -9 -f "run_fair_tuning_from_config.py" 2>/dev/null || true
pkill -9 -f "python.*fair_tuning" 2>/dev/null || true
sleep 2
echo "✅ 进程已清理"
echo ""

cd "$(dirname "$0")"

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║              CIFAR-10 公平调优 v13 - 修复 Krum 约束问题                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

🔧 修复:
  - ratios: [0.0, 0.1, 0.2, 0.4, 0.6] (移除 0.8)
  - 原因: Krum 要求 n-f-2 ≥ 1，10客户端时 ratio=0.8 不满足
  - attack_boost: 20.0 (共享参数，不调优)
  - attack_start_round: 25 (> K=20)

调优参数:
  sm9rrs:   detector_distance_threshold [2.0..3.1] (12 values)
            detector_subspace_dim 固定为 3
  VERT:     window × dim × epochs × lr (采样 12 个)
  AlignIns: sparsity × radii (采样 12 个)

预计耗时:
  - 验证: 12+12+12 = 36 trials × 5场景 × 10分钟 ≈ 30 小时
  - 最终: 6方法 × 5ratio × 2partition × 150轮 ≈ 15 小时
  - 总计: 45 小时

学术价值:
  ⭐⭐⭐⭐⭐ 符合一区论文标准
  - 公平对比 (有参数的方法都调优)
  - 方法一致 (与 MNIST v7 相同)
  - Krum 约束兼容

EOF

echo ""
echo "开始时间: $(date)"
echo ""
read -p "是否开始 45 小时的完整调优？[Y/n]: " answer
if [[ "$answer" =~ ^[Nn] ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "▶️  启动调优..."
echo ""

export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts/experiments/mnist"
python3 run_fair_tuning_from_config.py --config configs/cifar10_v13_fair_tuning_fixed.json

echo ""
echo "完成时间: $(date)"
echo ""
echo "📊 查看结果:"
echo "  cat outputs/cifar10_v13_fair_tuning_fixed/tuning_summary.json"
echo "  cat outputs/cifar10_v13_fair_tuning_fixed/best_parameters.json"
