#!/bin/bash
# CIFAR v13 远端快速恢复脚本
# 生成时间: 2026-10-15
# 问题: Krum约束失败 (n=10, f=8, n-f-2=0)
# 修复: V4.8.13 添加 ratios=[0.0, 0.1, 0.2, 0.4, 0.6]

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  CIFAR v13 远端恢复 - Krum约束修复"
echo "=========================================="
echo ""

# 1. 检查当前目录
if [ ! -f "run_v13_fixed_tuning.sh" ]; then
    echo "❌ 错误: 不在项目根目录 /3251901002/SM9RRSFL"
    echo "请先执行: cd /3251901002/SM9RRSFL"
    exit 1
fi

echo "✅ 当前目录正确"
echo ""

# 2. 同步代码
echo "步骤1: 同步代码..."
git fetch origin
BEFORE=$(git rev-parse HEAD)
git pull origin main
AFTER=$(git rev-parse HEAD)

if [ "$BEFORE" = "$AFTER" ]; then
    echo "⚠️  代码已是最新 ($AFTER)"
else
    echo "✅ 代码已更新: $BEFORE -> $AFTER"
fi
echo ""

# 3. 验证修复
echo "步骤2: 验证修复..."
if grep -q '"ratios": \[0.0, 0.1, 0.2, 0.4, 0.6\]' configs/cifar10_v13_fair_tuning_fixed.json; then
    echo "✅ ratios字段存在且正确（排除0.8）"
else
    echo "❌ 错误: ratios字段缺失或不正确"
    echo "请检查配置文件: configs/cifar10_v13_fair_tuning_fixed.json"
    exit 1
fi

# 验证Krum兼容性
python3 << 'EOF'
import json
with open('configs/cifar10_v13_fair_tuning_fixed.json') as f:
    config = json.load(f)

n = config['shared_parameters']['num_clients']
ratios = config['shared_parameters']['ratios']

print(f"\n配置验证:")
print(f"  num_clients = {n}")
print(f"  ratios = {ratios}")
print(f"\nKrum兼容性检查:")

all_ok = True
for r in ratios:
    f_val = int(n * r)
    constraint = n - f_val - 2
    status = "✅" if constraint >= 1 else "❌"
    print(f"  ratio={r}: f={f_val}, n-f-2={constraint} {status}")
    if constraint < 1:
        all_ok = False

if not all_ok:
    print("\n❌ 错误: 存在不兼容的ratio配置")
    exit(1)
else:
    print("\n✅ 所有ratio配置均与Krum兼容")
EOF

if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

# 4. 清理旧输出
echo "步骤3: 清理旧输出..."
if [ -d "outputs/cifar10_v13_fair_tuning" ]; then
    BACKUP_DIR="outputs/cifar10_v13_fair_tuning_failed_20261015"
    if [ -d "$BACKUP_DIR" ]; then
        echo "⚠️  备份目录已存在: $BACKUP_DIR"
        echo "请手动处理旧输出后重新运行此脚本"
        exit 1
    fi

    mv outputs/cifar10_v13_fair_tuning "$BACKUP_DIR"
    echo "✅ 旧输出已备份到: $BACKUP_DIR"

    # 统计备份内容
    TASK_COUNT=$(find "$BACKUP_DIR" -name "task_*.json" 2>/dev/null | wc -l)
    BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
    echo "   └─ 包含 $TASK_COUNT 个任务, 大小 $BACKUP_SIZE"
else
    echo "✅ 无旧输出需要清理"
fi
echo ""

# 5. 最终确认
echo "=========================================="
echo "准备就绪！"
echo "=========================================="
echo ""
echo "预计耗时: 约30小时"
echo "任务总数: 120 (30验证 + 90最终)"
echo "并行GPU: 8"
echo ""
echo "监控命令:"
echo "  tail -f outputs/cifar10_v13_fair_tuning/run_*.log"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "是否立即启动实验？"
read -p "输入 Y 继续，其他键取消: " CONFIRM

if [ "$CONFIRM" != "Y" ] && [ "$CONFIRM" != "y" ]; then
    echo ""
    echo "已取消。手动启动命令:"
    echo "  bash run_v13_fixed_tuning.sh"
    exit 0
fi

echo ""
echo "=========================================="
echo "  启动实验..."
echo "=========================================="
echo ""

# 6. 启动实验
bash run_v13_fixed_tuning.sh

echo ""
echo "=========================================="
echo "实验已启动！"
echo "=========================================="
echo ""
echo "关键里程碑:"
echo "  [ ] 验证阶段完成 (~5小时)"
echo "  [ ] 最终阶段开始"
echo "  [ ] 全部完成 (~30小时)"
echo ""
echo "失败时检查:"
echo "  grep 'error\|Error\|ERROR' outputs/cifar10_v13_fair_tuning/run_*.log"
echo ""
