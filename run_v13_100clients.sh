#!/bin/bash

# CIFAR-10 v13 公平调优 - 100客户端版本
# ratios = [0.0, 0.2, 0.4, 0.6, 0.8]
# 基于 v13 fixed 成功配置

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║              CIFAR-10 公平调优 v13 - 100客户端 (5比例)                       ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 终止现有进程
echo "🛑 终止现有进程..."
pkill -f "run_fair_tuning_from_config.py" || true
sleep 2
echo "✅ 进程已清理"
echo ""

# 显示配置
echo "配置参数:"
echo "  客户端数量: 100"
echo "  恶意比例: [0.0, 0.2, 0.4, 0.6, 0.8]"
echo "  轮次: 150"
echo "  攻击参数: boost=20.0, start_round=25"
echo ""
echo "调优参数:"
echo "  sm9rrs:   detector_distance_threshold [2.0, 2.1, ..., 3.1] (12 values)"
echo "            detector_subspace_dim 固定为 3"
echo "  VERT:     window × epochs × lr (3×2×2 = 12 组合)"
echo "            projection_dim 固定为 20, top_k 固定为 5"
echo "  AlignIns: sparsity × tda_radius (4×3 = 12 组合)"
echo "            mpsa_radius 固定为 0.1"
echo ""
echo "预计耗时:"
echo "  - 验证: (12+12+12+1+1+1)×5ratios×3seeds×2partitions = 1170 trials"
echo "  - 最终: 6方法×5ratios×3seeds×2partitions = 180 trials"
echo "  - 每trial约150轮，100客户端"
echo "  - 总计: 约 200-300 小时 (7张4090D并行)"
echo ""

# 确认
echo "开始时间: $(date)"
echo ""
read -p "是否开始 200-300 小时的完整调优？[Y/n]: " confirm
confirm=${confirm:-Y}

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "❌ 已取消"
    exit 1
fi

echo ""
echo "▶️  启动调优..."
echo ""

# 运行
export EMIT_PROGRESS_EVENTS=1
python3 -u run_fair_tuning_from_config.py \
    --config configs/cifar10_v13_100clients.json \
    2>&1 | python3 fair_tuning_gpu_display.py

echo ""
echo "✅ 调优完成"
echo "结束时间: $(date)"
