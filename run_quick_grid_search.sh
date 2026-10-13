#!/bin/bash
# 快速网格搜索 - 只优化 sm9rrs 的 boost 和 threshold

cd /3251901002/SM9RRSFL || exit 1

echo "🛑 终止现有进程..."
pkill -9 -f "python.*experiment"
pkill -9 -f "python.*tuning"
sleep 2
echo "✅ 进程已清理"
echo ""

git pull

echo "=================================="
echo "🚀 sm9rrs 快速网格搜索"
echo "=================================="
echo ""
echo "参数组合:"
echo "  - attack_boost: [20, 21, 22, 23, 24] (5个值)"
echo "  - threshold: [1.7, 1.9, 2.1, 2.3] (4个值)"
echo "  - 总共: 5 × 4 = 20 个配置"
echo "  - 每个配置: ~10分钟"
echo "  - 预计总耗时: 2-3 小时"
echo ""
echo "目标: 找到最优参数，使 ASR < 7.0%"
echo ""
echo "开始时间: $(date)"
echo ""

python3 run_experiments_from_config.py --config configs/cifar10_sm9rrs_quick_grid.json

echo ""
echo "完成时间: $(date)"
echo ""
echo "🔍 分析结果:"
python3 << 'PYTHON_SCRIPT'
import json
import glob
from pathlib import Path

results_dir = Path("outputs/cifar10_sm9rrs_quick_grid")
if not results_dir.exists():
    print("❌ 结果目录不存在")
    exit(1)

# 收集所有实验结果
results = []
for summary_file in results_dir.glob("**/summary.json"):
    try:
        with open(summary_file) as f:
            data = json.load(f)
            config_name = summary_file.parent.name

            # 提取参数
            boost = None
            threshold = None
            for part in config_name.split("_"):
                if part.startswith("boost"):
                    boost = float(part.replace("boost", ""))
                elif part.startswith("thresh"):
                    threshold = float(part.replace("thresh", ""))

            if boost and threshold:
                asr = data.get("attack_success_rate", 1.0)
                acc = data.get("final_accuracy", 0.0)
                results.append({
                    "boost": boost,
                    "threshold": threshold,
                    "asr": asr,
                    "acc": acc
                })
    except:
        pass

if not results:
    print("❌ 没有找到有效结果")
    exit(1)

# 排序：ASR 最低优先
results.sort(key=lambda x: (x["asr"], -x["acc"]))

print("\n" + "="*80)
print("📊 网格搜索结果（按 ASR 排序）")
print("="*80)
print(f"{'Boost':<8} {'Threshold':<12} {'ASR':<10} {'Accuracy':<10} {'Status'}")
print("-"*80)

for r in results[:10]:  # 显示前10
    status = "✅ 优秀" if r["asr"] < 0.07 else ("🔥 良好" if r["asr"] < 0.10 else "⚠️  一般")
    print(f"{r['boost']:<8.1f} {r['threshold']:<12.1f} {r['asr']*100:<9.2f}% {r['acc']*100:<9.2f}% {status}")

print("\n🏆 最优配置:")
best = results[0]
print(f"  attack_boost = {best['boost']}")
print(f"  detector_distance_threshold = {best['threshold']}")
print(f"  ASR = {best['asr']*100:.2f}%")
print(f"  Accuracy = {best['acc']*100:.2f}%")
print("\n保存到: best_sm9rrs_params.json")

with open("best_sm9rrs_params.json", "w") as f:
    json.dump(best, f, indent=2)

PYTHON_SCRIPT

echo ""
echo "✅ 完成！查看 best_sm9rrs_params.json"
