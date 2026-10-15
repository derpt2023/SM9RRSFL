#!/bin/bash
# 停止Fair Tuning实验

echo "🛑 正在停止实验..."

# 查找并终止相关进程
pkill -f "run_fair_tuning_from_config.py"
pkill -f "fair_tuning_progress_display.py"

# 等待进程结束
sleep 2

# 检查是否还有残留进程
REMAINING=$(pgrep -f "run_fair_tuning_from_config.py" | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo "⚠️  发现残留进程，强制终止..."
    pkill -9 -f "run_fair_tuning_from_config.py"
    sleep 1
fi

echo "✅ 实验已停止"
echo ""
echo "检查GPU状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader,nounits
