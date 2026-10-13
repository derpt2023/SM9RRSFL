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
║              CIFAR-10 公平调优 v13 - 修复共享参数错误                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

✅ 修复:
  - attack_boost: 20.0 (共享参数，不调优)
  - attack_start_round: 25 (> K=20)
  - 只调优方法特定参数

调优参数:
  sm9rrs:   detector_distance_threshold [1.7, 1.9, 2.1, 2.3, 2.5] (5 values)
            detector_subspace_dim 固定为 3
  VERT:     window × dim × epochs × lr (3×3×2×2 = 36 组合，TPE 采样 12 个)
  AlignIns: sparsity × radii (3×3×3 = 27 组合，TPE 采样 18 个)

基于今天的发现:
  ✅ boost=20 效果好 (平均 ASR=13.5%)
  ✅ threshold 在 1.7-2.5 范围内搜索（围绕成功的 2.0）
  ✅ dim=3 已验证有效

预计耗时:
  - 验证: 5+12+18 = 35 trials × 10分钟 = 6 小时
  - 最终: 6方法 × 5ratio × 2partition × 150轮 = 约12小时
  - 总计: 18 小时

学术价值:
  ⭐⭐⭐⭐⭐ 符合一区论文标准
  - 公平对比 (有参数的方法都调优)
  - 方法一致 (与 MNIST v7 相同)
  - 参数基于实证 (今天的测试)

EOF

echo ""
echo "开始时间: $(date)"
echo ""
read -p "是否开始 18 小时的完整调优？[Y/n]: " answer
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
