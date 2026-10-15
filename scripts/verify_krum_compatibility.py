#!/usr/bin/env python3
"""验证 CIFAR v13 配置的 Krum 兼容性"""

import json
import sys
from pathlib import Path


def check_krum_constraint(num_clients: int, malicious_ratio: float) -> tuple[bool, str]:
    """检查 Krum 约束: n - f - 2 >= 1"""
    malicious_count = int(num_clients * malicious_ratio + 0.5)
    neighbor_count = num_clients - malicious_count - 2

    if neighbor_count < 1:
        return False, (
            f"❌ Krum 不满足约束: n={num_clients}, f={malicious_count}, "
            f"n-f-2={neighbor_count} < 1"
        )
    return True, (
        f"✓ ratio={malicious_ratio:.1f}: n={num_clients}, f={malicious_count}, "
        f"n-f-2={neighbor_count}"
    )


def main():
    config_path = Path("configs/cifar10_v13_fair_tuning_fixed.json")

    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        print("请确认当前目录为项目根目录")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    shared = config.get("shared_parameters", {})
    num_clients = shared.get("num_clients")
    methods = shared.get("methods", [])
    ratios = shared.get("ratios")

    print("=" * 70)
    print("CIFAR v13 Krum 约束验证")
    print("=" * 70)
    print()

    # 检查配置完整性
    if num_clients is None:
        print("❌ 缺少 num_clients 配置")
        sys.exit(1)
    print(f"客户端数量: {num_clients}")

    if "krum" not in methods:
        print("⚠️  配置中未包含 Krum 方法，无需检查约束")
        sys.exit(0)
    print(f"方法列表: {', '.join(methods)}")
    print()

    # 检查 ratios
    if ratios is None:
        print("⚠️  未显式设置 ratios，将使用框架默认值")
        print("   默认值: [0.0, 0.1, 0.2, 0.4, 0.45, 0.6, 0.8]")
        print("   ⚠️  在 num_clients=10 时，ratio=0.8 会导致 Krum 失败！")
        ratios = [0.0, 0.1, 0.2, 0.4, 0.45, 0.6, 0.8]
    else:
        print(f"✅ 显式设置 ratios: {ratios}")
    print()

    # 验证每个 ratio
    print("Krum 约束检查 (n - f - 2 >= 1):")
    print("-" * 70)

    all_pass = True
    for ratio in ratios:
        passed, message = check_krum_constraint(num_clients, ratio)
        print(message)
        if not passed:
            all_pass = False

    print()
    print("=" * 70)
    if all_pass:
        print("✅ 所有 ratio 均满足 Krum 约束")
        print("配置可以安全运行")
        sys.exit(0)
    else:
        print("❌ 存在不满足 Krum 约束的 ratio")
        print()
        print("修复建议:")
        print("  1. 在 configs/cifar10_v13_fair_tuning_fixed.json 的")
        print("     shared_parameters 中添加:")
        print('     "ratios": [0.0, 0.1, 0.2, 0.4, 0.6]')
        print()
        print("  2. 或增加客户端数量（不推荐，会改变实验设置）:")
        max_safe_ratio = (num_clients - 3) / num_clients
        print(f"     当前 {num_clients} 客户端最大安全 ratio: {max_safe_ratio:.2f}")
        print(f"     至少需要 {max([13, num_clients])} 客户端才能安全使用 ratio=0.8")
        sys.exit(1)


if __name__ == "__main__":
    main()
