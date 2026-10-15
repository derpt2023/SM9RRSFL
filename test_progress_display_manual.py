#!/usr/bin/env python3
"""手动测试进度显示器的事件解析"""

import json
import sys
import time

# 模拟事件序列
events = [
    {"type": "START", "partition": "iid", "clients": 100, "method": "sm9rrs", "ratio": 0.2, "device": "cuda:0"},
    {"type": "ROUND", "round": 1, "device": "cuda:0"},
    {"type": "ROUND", "round": 2, "device": "cuda:0"},
    {"type": "START", "partition": "dirichlet", "clients": 100, "method": "VERT", "ratio": 0.4, "device": "cuda:1"},
    {"type": "ROUND", "round": 3, "device": "cuda:0"},
    {"type": "ROUND", "round": 1, "device": "cuda:1"},
    {"type": "ROUND", "round": 4, "device": "cuda:0"},
    {"type": "ROUND", "round": 2, "device": "cuda:1"},
    {"type": "FINISH", "device": "cuda:0"},
    {"type": "ROUND", "round": 3, "device": "cuda:1"},
    {"type": "FINISH", "device": "cuda:1"},
]

print("开始测试进度事件...\n")

for event in events:
    # 输出与 experiments.py 相同的格式
    print(f"progress_event={json.dumps(event)}", flush=True)
    time.sleep(0.3)  # 模拟训练间隔

print("\n测试完成！")
