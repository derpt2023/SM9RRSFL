#!/bin/bash
# CIFAR v13 断点恢复脚本
# 用途：从Krum失败处恢复，利用已完成的762个任务

set -e

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║              CIFAR v13 断点恢复 - 从Krum失败处继续                           ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 检查当前目录
if [ ! -f "configs/cifar10_v13_fair_tuning_fixed.json" ]; then
    echo "❌ 错误：找不到配置文件"
    echo "   请在项目根目录运行此脚本"
    exit 1
fi

# 检查输出目录是否存在
OUTPUT_DIR="outputs/cifar10_v13_fair_tuning"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ 错误：输出目录不存在: $OUTPUT_DIR"
    echo "   可能需要先同步远端数据"
    exit 1
fi

# 检查.tuning_state是否存在
STATE_DIR="$OUTPUT_DIR/.tuning_state/validation"
if [ ! -d "$STATE_DIR" ]; then
    echo "⚠️  警告：未找到断点数据: $STATE_DIR"
    echo "   将从头开始运行（不使用恢复）"
    USE_RESUME=""
else
    # 统计已完成任务
    CHECKPOINT_COUNT=$(find "$STATE_DIR" -name "result_*.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "✅ 找到断点数据"
    echo "   - 状态目录: $STATE_DIR"
    echo "   - 已完成任务: $CHECKPOINT_COUNT"
    echo ""
    USE_RESUME="--resume"
fi

# 验证配置中的ratios
echo "📋 验证配置..."
RATIOS=$(python3 -c "
import json
with open('configs/cifar10_v13_fair_tuning_fixed.json') as f:
    config = json.load(f)
    ratios = config.get('ratios')
    if ratios:
        print(ratios)
    else:
        print('未设置（将使用默认值）')
" 2>/dev/null)

if [[ "$RATIOS" == *"0.8"* ]] || [[ "$RATIOS" == "未设置"* ]]; then
    echo "❌ 错误：配置仍包含ratio=0.8或未显式设置ratios"
    echo "   当前ratios: $RATIOS"
    echo ""
    echo "   请确保configs/cifar10_v13_fair_tuning_fixed.json中有："
    echo '   "ratios": [0.0, 0.1, 0.2, 0.4, 0.6]'
    exit 1
fi

echo "   ✅ ratios已正确配置: $RATIOS"
echo ""

# 运行Krum兼容性检查
if [ -f "scripts/verify_krum_compatibility.py" ]; then
    echo "🔍 运行Krum兼容性检查..."
    python3 scripts/verify_krum_compatibility.py
    if [ $? -ne 0 ]; then
        echo "❌ Krum兼容性检查失败"
        exit 1
    fi
    echo ""
else
    echo "⚠️  跳过Krum兼容性检查（脚本不存在）"
    echo ""
fi

# 显示恢复计划
echo "📊 恢复计划："
if [ -n "$USE_RESUME" ]; then
    echo "   - 模式: 断点恢复"
    echo "   - 已完成: ~$CHECKPOINT_COUNT 任务"
    echo "   - 新配置: 585任务（39候选 × 3seeds × 5ratios）"
    echo "   - 预计补跑: ~41任务（约1小时）"
else
    echo "   - 模式: 完整运行（无断点）"
    echo "   - 任务数: 585（39候选 × 3seeds × 5ratios）"
    echo "   - 预计耗时: 约13小时"
fi
echo ""

# 询问确认
read -p "是否继续？[Y/n]: " -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "❌ 已取消"
    exit 0
fi

# 记录开始时间
START_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "⏰ 开始时间: $START_TIME"
echo ""

# 执行恢复
echo "▶️  启动恢复..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 run_fair_tuning_from_config.py \
    configs/cifar10_v13_fair_tuning_fixed.json \
    --output "$OUTPUT_DIR" \
    $USE_RESUME \
    --devices auto \
    --jobs auto

EXIT_CODE=$?

# 记录结束时间
END_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏰ 结束时间: $END_TIME"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 恢复成功完成"
    echo ""
    echo "📊 查看结果："
    echo "   python3 -c \"
import json
with open('$OUTPUT_DIR/validation_summary.json') as f:
    data = json.load(f)
    print(f'完成任务: {len(data[\\\"executions\\\"])}')
    print(f'候选方法: {data[\\\"candidate_counts\\\"]}')
    ratios = sorted(set(e['config']['malicious_ratio'] for e in data['executions']))
    print(f'实际比例: {ratios}')
\""
else
    echo "❌ 恢复失败 (退出码: $EXIT_CODE)"
    echo ""
    echo "🔍 故障排除："
    echo "   1. 查看错误日志"
    echo "   2. 检查GPU可用性"
    echo "   3. 验证配置文件"
    echo "   4. 查看 docs/CIFAR-v13-断点恢复指南-2026-10-15.md"
fi

exit $EXIT_CODE
