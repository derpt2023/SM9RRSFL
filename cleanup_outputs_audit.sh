#!/bin/bash
# 清理outputs中的audit临时目录
# 这些目录是2026-10-08至10-11期间完成的诊断任务输出
# 诊断结论已记录在AGENTS.md中，可以安全删除

echo "=========================================="
echo "清理 outputs 中的 audit 临时目录"
echo "=========================================="
echo ""
echo "将要删除15个目录（约2.6MB）："
echo ""

DIRS=(
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

# 检查目录是否存在
EXISTING=0
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir"
        EXISTING=$((EXISTING + 1))
    else
        echo "  ✗ $dir (不存在)"
    fi
done

echo ""
echo "找到 $EXISTING 个目录"
echo ""

if [ $EXISTING -eq 0 ]; then
    echo "没有需要删除的目录"
    exit 0
fi

read -p "确认删除这些目录? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "取消删除"
    exit 0
fi

echo ""
echo "开始删除..."

DELETED=0
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "删除: $dir"
        rm -rf "$dir"
        DELETED=$((DELETED + 1))
    fi
done

echo ""
echo "✅ 完成！删除了 $DELETED 个目录"
echo "=========================================="
