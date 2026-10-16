#!/usr/bin/env python3
"""测试增强的进度显示：配置进度条、总轮数进度条、GPU轮次进度条"""

import json
import time

# 模拟多GPU并行训练的事件序列
events = [
    # GPU 0 开始
    {"type": "START", "partition": "iid", "clients": 100, "method": "sm9rrs", "ratio": 0.2, "rounds": 150, "device": "cuda:0"},
    {"type": "ROUND", "round": 10, "device": "cuda:0"},

    # GPU 1 开始
    {"type": "START", "partition": "dirichlet", "clients": 100, "method": "VERT", "ratio": 0.4, "rounds": 150, "device": "cuda:1"},
    {"type": "ROUND", "round": 5, "device": "cuda:1"},

    # GPU 0 继续
    {"type": "ROUND", "round": 20, "device": "cuda:0"},
    {"type": "ROUND", "round": 8, "device": "cuda:1"},

    # GPU 2 开始
    {"type": "START", "partition": "iid", "clients": 100, "method": "AlignIns", "ratio": 0.6, "rounds": 150, "device": "cuda:2"},

    # 多轮更新
    {"type": "ROUND", "round": 30, "device": "cuda:0"},
    {"type": "ROUND", "round": 15, "device": "cuda:1"},
    {"type": "ROUND", "round": 3, "device": "cuda:2"},

    {"type": "ROUND", "round": 45, "device": "cuda:0"},
    {"type": "ROUND", "round": 22, "device": "cuda:1"},
    {"type": "ROUND", "round": 8, "device": "cuda:2"},

    # GPU 0 完成
    {"type": "ROUND", "round": 150, "device": "cuda:0"},
    {"type": "FINISH", "device": "cuda:0"},

    # GPU 3 开始
    {"type": "START", "partition": "dirichlet", "clients": 100, "method": "Krum", "ratio": 0.0, "rounds": 150, "device": "cuda:3"},

    # 继续训练
    {"type": "ROUND", "round": 50, "device": "cuda:1"},
    {"type": "ROUND", "round": 25, "device": "cuda:2"},
    {"type": "ROUND", "round": 10, "device": "cuda:3"},
]

print("测试增强的进度显示...\n", flush=True)
print("预期效果：", flush=True)
print("1. 配置进度条显示整体完成进度", flush=True)
print("2. 总轮数进度条显示所有GPU的总轮数进度", flush=True)
print("3. 每个GPU显示自己的轮次进度条", flush=True)
print("\n开始模拟事件流：\n", flush=True)

for i, event in enumerate(events):
    print(f"progress_event={json.dumps(event)}", flush=True)
    time.sleep(0.5)  # 模拟训练间隔

    if i == 5:
        print("\n=== 应该看到3个GPU并行工作 ===\n", flush=True)
    elif i == 12:
        print("\n=== GPU 0 完成，应该从列表中移除 ===\n", flush=True)

print("\n\n测试完成！", flush=True)
print("检查项：", flush=True)
print("✓ 配置进度条随任务完成更新", flush=True)
print("✓ 总轮数进度条显示所有GPU累计轮数", flush=True)
print("✓ 每个GPU显示 [####------] round/total percent 格式", flush=True)
print("✓ 刷新时不会向上移动行", flush=True)
