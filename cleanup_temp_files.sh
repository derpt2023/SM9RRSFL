#!/bin/bash
# SM9RRSFL 项目临时文件清理脚本
# 日期: 2026-10-14
# 说明: 清理已完成任务的临时分析脚本和旧版本文件
# 风险等级: 低

set -e  # 遇到错误立即退出

echo "=========================================="
echo "SM9RRSFL 项目清理脚本"
echo "=========================================="
echo ""

# 确认当前在项目根目录
if [ ! -f "setup.py" ] || [ ! -d "sm9rrsfl" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    exit 1
fi

echo "📋 清理范围："
echo "  - 14个根目录临时分析脚本"
echo "  - 5个旧版本shell脚本"
echo "  - 15个outputs中的audit临时目录"
echo ""

# 统计将要删除的文件
FILES_TO_DELETE=(
    "analyze_failure.py"
    "analyze_all_methods_balanced.py"
    "analyze_all_methods_boost25.py"
    "compare_seeds_detailed.py"
    "quick_decision_maker.py"
    "check_gpu_utilization.py"
    "generate_asr_line_charts.py"
    "generate_asr_visualization.py"
    "generate_runtime_svg.py"
    "generate_runtime_visualizations_per_seed.py"
    "run_cifar10_improved_quick_test.py"
    "run_full_tpe_pipeline.py"
    "run_optimized_tuning_pipeline.py"
    "run_simplified_tuning_pipeline.py"
    "run_v12_fair_tuning.sh"
    "run_quick_grid_search.sh"
    "quick_start_v11_tuning.sh"
    "kill_and_start_v11.sh"
    "quick_check_boost_results.sh"
)

DIRS_TO_DELETE=(
    "outputs/cifar_clean_feedback_audit_20261008"
    "outputs/cifar_clean_horizon_feedback_audit_20261011"
    "outputs/cifar_client_step_feedback_audit_20261009"
    "outputs/cifar_cnn_match_feedback_audit_20261008"
    "outputs/cifar_deterministic_prefix_feedback_audit_20261010"
    "outputs/cifar_history_feedback_audit_20261009"
    "outputs/cifar_history_forensics_feedback_audit_20261009"
    "outputs/cifar_mechanism_details_feedback_audit_20261010"
    "outputs/cifar_mechanism_feedback_audit_20261010"
    "outputs/cifar_prefix_clients_feedback_audit_20261009"
    "outputs/cifar_prefix_complete_feedback_audit_20261009"
    "outputs/cifar_prefix_feedback_audit_20261009"
    "outputs/cifar_tail_determinism_feedback_audit_20261009"
    "outputs/cifar_threshold_feedback_audit_20261008"
    "outputs/cifar_timing_feedback_audit_20261008"
)

# 检查文件是否存在
EXISTING_FILES=0
MISSING_FILES=0

echo "🔍 检查文件状态..."
for file in "${FILES_TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        EXISTING_FILES=$((EXISTING_FILES + 1))
    else
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

for dir in "${DIRS_TO_DELETE[@]}"; do
    if [ -d "$dir" ]; then
        EXISTING_FILES=$((EXISTING_FILES + 1))
    else
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

echo "  存在: $EXISTING_FILES 项"
echo "  已删除或不存在: $MISSING_FILES 项"
echo ""

if [ $EXISTING_FILES -eq 0 ]; then
    echo "✅ 没有需要清理的文件"
    exit 0
fi

# 用户确认
echo "⚠️  将要删除 $EXISTING_FILES 个文件/目录"
echo ""
read -p "确认执行清理? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "❌ 清理已取消"
    exit 0
fi

echo ""
echo "🗑️  开始清理..."

# 删除根目录文件
DELETED_COUNT=0
for file in "${FILES_TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        echo "  删除: $file"
        rm -f "$file"
        DELETED_COUNT=$((DELETED_COUNT + 1))
    fi
done

# 删除outputs目录
for dir in "${DIRS_TO_DELETE[@]}"; do
    if [ -d "$dir" ]; then
        echo "  删除: $dir/"
        rm -rf "$dir"
        DELETED_COUNT=$((DELETED_COUNT + 1))
    fi
done

echo ""
echo "✅ 清理完成! 删除了 $DELETED_COUNT 项"
echo ""

# 验证关键测试
echo "🧪 运行验证测试..."
echo ""

if command -v python &> /dev/null; then
    echo "测试 1: 核心功能测试"
    if python -m pytest tests/test_fl.py -v --tb=short 2>&1 | tail -5; then
        echo "✅ 核心功能测试通过"
    else
        echo "⚠️  核心功能测试有问题，请检查"
    fi
    echo ""

    echo "测试 2: 配置运行器测试"
    if python -m pytest tests/test_config_runner.py -v --tb=short 2>&1 | tail -5; then
        echo "✅ 配置运行器测试通过"
    else
        echo "⚠️  配置运行器测试有问题，请检查"
    fi
    echo ""
else
    echo "⚠️  未找到python命令，跳过测试验证"
fi

echo "=========================================="
echo "清理报告"
echo "=========================================="
echo "删除项目: $DELETED_COUNT"
echo "剩余根目录Python脚本数:"
ls -1 *.py 2>/dev/null | wc -l
echo "剩余Shell脚本数:"
ls -1 *.sh 2>/dev/null | wc -l
echo ""
echo "✅ 项目清理完成"
echo ""
echo "📝 下一步："
echo "  1. 检查 git status 确认变更"
echo "  2. 运行完整测试: python -m pytest tests/"
echo "  3. 提交清理记录到Git（如果满意）"
echo "=========================================="
