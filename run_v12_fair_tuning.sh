#!/bin/bash
# CIFAR-10 公平调优 v12 - 遵循 MNIST v7 方法论

cd /3251901002/SM9RRSFL || exit 1

echo "🛑 终止现有进程..."
pkill -9 -f "python.*experiment"
pkill -9 -f "python.*tuning"
sleep 2
echo "✅ 进程已清理"
echo ""

git pull

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║              CIFAR-10 公平调优 v12 - 一区论文标准                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

遵循 MNIST v7 方法论:
  ✅ sm9rrs:   调优 (boost, threshold, dim)  - 20 trials
  ✅ VERT:     调优 (window, epochs, lr)     - 12 trials
  ✅ AlignIns: 调优 (sparsity, radii)        - 18 trials
  ❌ Krum:     固定参数 (无可调参数)
  ❌ Ding13:   固定参数 (无可调参数)
  ❌ FedAvg:   固定参数 (基线)

调优阶段:
  1. 验证阶段: 每个方法找最优参数 (ratio=0.2, 50轮)
  2. 最终阶段: 用最优参数跑完整实验 (5个ratio, 150轮)

预计耗时:
  - 验证: 50 trials × 10分钟 = 8-9 小时
  - 最终: 6方法 × 5ratio × 2partition × 150轮 = 约12小时
  - 总计: 20-21 小时

学术价值:
  ⭐⭐⭐⭐⭐ 完全符合一区论文标准
  - 公平对比 (有参数的方法都调优)
  - 方法一致 (与 MNIST v7 相同)
  - 可重现 (完整搜索空间)

EOF

echo ""
echo "开始时间: $(date)"
echo ""
read -p "是否开始 20 小时的完整调优？[Y/n]: " confirm
if [[ ! "$confirm" =~ ^[Yy]?$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "▶️  启动调优..."
echo ""

export PYTHONPATH="${PYTHONPATH}:$(pwd)/scripts/experiments/mnist"

python3 run_fair_tuning_from_config.py --config configs/cifar10_v12_fair_tuning_correct.json 2>&1 | tee tuning_v12.log

echo ""
echo "完成时间: $(date)"
echo ""
echo "📊 查看结果:"
echo "  cat outputs/cifar10_v12_fair_tuning/tuning_summary.json"
echo "  cat outputs/cifar10_v12_fair_tuning/best_parameters.json"
