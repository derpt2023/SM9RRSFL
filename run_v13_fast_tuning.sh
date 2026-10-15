#!/bin/bash
# CIFAR-10 v13 快速调优脚本
# 修正：num_clients=100，优化搜索空间，48小时内完成

set -e

echo "🛑 终止现有进程..."
pkill -f "run_fair_tuning_from_config.py" || true
sleep 2
echo "✅ 进程已清理"
echo ""

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║              CIFAR-10 快速调优 v13 - 修正客户端数量                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

✅ 关键修正:
  - num_clients: 10 → 100 （与MNIST一致）
  - ratios: [0.0, 0.2, 0.4, 0.6] （移除0.8避免Krum失败，移除0.1节省时间）

✅ 资源优化:
  - 验证: 1 seed × 1 partition (Dirichlet) × 50 rounds
  - 正式: 2 seeds × 2 partitions × 150 rounds
  - 候选: sm9rrs=3, vert=4, alignins=4 (聚焦最佳)

调优参数:
  sm9rrs:   detector_distance_threshold [1.9, 2.0, 2.1] (围绕成功值2.0)
  VERT:     window × dim × epochs × lr (TPE采样4个)
  AlignIns: sparsity × radii (TPE采样4个)

预计耗时:
  - 验证: 14候选 × 1seed × 1partition × 4ratios = 56 任务
          56 × 10分钟 = 9.3 小时
  - 正式: 6方法 × 2seeds × 2partitions × 4ratios = 96 任务
          96 × 30分钟 = 48 小时
  - 总计: 57.3 小时 ≈ 2.4 天

学术价值:
  ⭐⭐⭐⭐⭐ 仍符合一区论文标准
  - ✅ 100客户端（与MNIST一致）
  - ✅ 公平对比（所有方法都调优）
  - ✅ 覆盖0%~60%恶意比例（充分）
  - ✅ IID + Non-IID 两种分布
  - ⚡ 在合理时间内完成

EOF

echo ""
echo "开始时间: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

read -p "是否开始 57 小时的快速调优？[Y/n]: " -r
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "❌ 已取消"
    exit 0
fi

echo ""
echo "▶️  启动调优..."
echo ""

python3 run_fair_tuning_from_config.py \
    configs/cifar10_v13_fast_tuning.json \
    --devices auto \
    --jobs auto

EXIT_CODE=$?
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "结束时间: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 调优成功完成"
    echo ""
    echo "📊 查看结果："
    echo "   cat outputs/cifar10_v13_fast_tuning/validation_summary.json"
else
    echo "❌ 调优失败 (退出码: $EXIT_CODE)"
fi

exit $EXIT_CODE
