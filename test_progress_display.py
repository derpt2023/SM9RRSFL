#!/usr/bin/env python3
"""测试进度显示是否工作"""
import sys
import os

print(f"Python版本: {sys.version}", flush=True)
print(f"是否TTY: {sys.stdout.isatty()}", flush=True)
print(f"TERM: {os.environ.get('TERM', 'not set')}", flush=True)
print(f"终端大小: {os.get_terminal_size() if sys.stdout.isatty() else 'N/A'}", flush=True)

# 测试导入
try:
    from fair_tuning_gpu_display import gpu_progress_display
    print("✅ fair_tuning_gpu_display 导入成功", flush=True)
except ImportError as e:
    print(f"❌ fair_tuning_gpu_display 导入失败: {e}", flush=True)

try:
    from fair_tuning_progress_display import progress_display
    print("✅ fair_tuning_progress_display 导入成功", flush=True)
except ImportError as e:
    print(f"❌ fair_tuning_progress_display 导入失败: {e}", flush=True)

# 测试基本显示
print("\n测试基本输出（应该立即看到）：", flush=True)
for i in range(3):
    print(f"  行 {i+1}/3", flush=True)

print("\n✅ 测试完成", flush=True)
