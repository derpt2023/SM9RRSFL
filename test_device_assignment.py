#!/usr/bin/env python3
"""测试设备自动分配功能"""

import json
from pathlib import Path

def test_device_auto_assignment():
    """验证配置文件使用auto而不是固定cuda"""
    config_path = Path("configs/cifar10_v13_100clients.json")
    with open(config_path) as f:
        config = json.load(f)

    device = config.get("shared_parameters", {}).get("device")
    assert device == "auto", f"Expected device='auto', got '{device}'"
    print(f"✓ 设备配置正确: {device}")
    print("  这将允许系统自动分配多个GPU")

if __name__ == "__main__":
    test_device_auto_assignment()
    print("\n测试通过！")
