#!/bin/bash
# 文档清理脚本 - 2026-10-14
# 用途：清理Codex交接文档和历史诊断文档

set -e

echo "========================================"
echo "文档清理脚本"
echo "========================================"
echo ""

# 统计
CODEX_COUNT=$(ls -1 docs/Codex对话交接-*.md 2>/dev/null | wc -l)
echo "发现 $CODEX_COUNT 个Codex交接文档"

# 第一步：删除Codex交接文档（29个）
echo ""
echo "第一步：删除Codex交接文档..."
rm -f docs/Codex对话交接-2026-07-26.md
rm -f docs/Codex对话交接-2026-08-01.md
rm -f docs/Codex对话交接-2026-08-03.md
rm -f docs/Codex对话交接-2026-08-04.md
rm -f docs/Codex对话交接-2026-08-05.md
rm -f docs/Codex对话交接-2026-08-10.md
rm -f docs/Codex对话交接-2026-08-24.md
rm -f docs/Codex对话交接-2026-08-25.md
rm -f docs/Codex对话交接-2026-09-03.md
rm -f docs/Codex对话交接-2026-09-06.md
rm -f docs/Codex对话交接-2026-09-07.md
rm -f docs/Codex对话交接-2026-09-10.md
rm -f docs/Codex对话交接-2026-09-11.md
rm -f docs/Codex对话交接-2026-09-16.md
rm -f docs/Codex对话交接-2026-09-18.md
rm -f docs/Codex对话交接-2026-09-20.md
rm -f docs/Codex对话交接-2026-09-21.md
rm -f docs/Codex对话交接-2026-09-23.md
rm -f docs/Codex对话交接-2026-09-27.md
rm -f docs/Codex对话交接-2026-09-28.md
rm -f docs/Codex对话交接-2026-09-29.md
rm -f docs/Codex对话交接-2026-09-30.md
rm -f docs/Codex对话交接-2026-10-01.md
rm -f docs/Codex对话交接-2026-10-03.md
rm -f docs/Codex对话交接-2026-10-07.md
rm -f docs/Codex对话交接-2026-10-08.md
rm -f docs/Codex对话交接-2026-10-09.md
rm -f docs/Codex对话交接-2026-10-10.md
rm -f docs/Codex对话交接-2026-10-12.md

echo "✅ 已删除 29 个Codex交接文档"

# 第二步：移动历史诊断文档到归档目录
echo ""
echo "第二步：移动历史诊断文档到归档目录..."

# CIFAR历史诊断文档
mv -f docs/CIFAR-v7完整验证差距与正式结果预估-2026-09-27.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/CIFAR五天预算重做实验配置说明-2026-09-21.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/CIFAR候选减法与回退条件核查-2026-09-18.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/CIFAR六方案从零重训-2026-09-15.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/CIFAR历次实验退化与v8根因审计-2026-10-07.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/CIFAR无攻击FedAvg文献对照与训练基线诊断-2026-09-27.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/CIFAR最新实验表现与Krum低ASR分析-2026-09-18.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/CIFAR最终双指标2个百分点门槛与启动核验-2026-09-21.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/CIFAR最终轮重算与第四版方案核查-2026-09-21.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/CIFAR相对ASR门槛与薄弱场景改进-2026-09-21.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/CIFAR验证五问代码与实验核查-2026-09-21.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/CIFAR验证未达标诊断与后续方案-2026-09-20.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/V8设计取舍与分阶段改进方案-2026-10-07.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/Ours在CIFAR上的改进建议-2026-09-18.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/MNIST与CIFAR实验代码对比及Ours失效分析-2026-09-18.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/MNIST与CIFAR防御差异根因审计-2026-09-27.md docs/archive/历史诊断/ 2>/dev/null || true

# 旧启动和配置说明
mv -f docs/AIStation换卡续跑与OOM恢复-2026-09-10.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/实验改进与参数筛选-2026-09-06.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/性能目标选参与启动说明-2026-09-07.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/代码更新日志-a835696至当前.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/阈值来源与撤销模型窗口改进建议-2026-09-28.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/论文修改说明-全局离线校准-2026-08-24.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/论文修改说明-自适应有界校准与方案B-2026-08-25.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/消融设计与数据集质量选择-2026-09-27.md docs/archive/历史诊断/ 2>/dev/null || true
mv -f docs/二区期刊数据集选择与证据方案-2026-09-27.md docs/archive/历史诊断/ 2>/dev/null || true

echo "✅ 已移动历史诊断文档到归档目录"

# 第三步：统计结果
echo ""
echo "========================================"
echo "清理完成统计"
echo "========================================"
echo "✅ 删除：29 个Codex交接文档"
echo "✅ 归档：约25 个历史诊断文档"
echo ""
echo "保留的核心文档："
ls -1 docs/*.md 2>/dev/null | wc -l | xargs echo "  - docs/ 目录剩余文档数："
echo ""
echo "归档位置："
echo "  - docs/archive/codex交接历史/ (空，已删除)"
echo "  - docs/archive/历史诊断/ (约25个文件)"
echo ""
echo "✅ 文档清理完成！"
