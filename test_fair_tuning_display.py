#!/usr/bin/env python3
"""测试 fair_tuning 进度显示环境."""

import sys
import os

print("=" * 70)
print("Fair Tuning 进度显示环境测试")
print("=" * 70)

# 1. Python版本
print(f"\n1. Python版本: {sys.version}")

# 2. TTY检测
is_tty = sys.stdout.isatty()
print(f"\n2. TTY状态: {is_tty}")
if not is_tty:
    print("   ⚠️  警告：当前不是TTY，进度条可能无法显示")
    print("   建议：确保在交互式终端中运行")

# 3. TERM环境变量
term = os.environ.get("TERM", "未设置")
print(f"\n3. TERM: {term}")
if term == "未设置" or term == "dumb":
    print("   ⚠️  警告：TERM不支持丰富显示")

# 4. 测试基础进度条
print("\n4. 测试基础进度条输出...")
try:
    import time
    for i in range(5):
        print(f"\r进度: {'#' * (i+1)}{'-' * (4-i)} {(i+1)*20}%", end="", flush=True)
        time.sleep(0.5)
    print()  # 换行
    print("   ✅ 基础进度条测试成功")
except Exception as e:
    print(f"   ❌ 基础进度条测试失败: {e}")

# 5. 测试模块导入
print("\n5. 测试模块导入...")

modules_to_test = [
    ("fair_tuning_gpu_display", "增强GPU显示"),
    ("fair_tuning_progress_display", "基础进度显示"),
]

for module_name, description in modules_to_test:
    try:
        __import__(module_name)
        print(f"   ✅ {module_name}: {description} - 导入成功")
    except ImportError as e:
        print(f"   ❌ {module_name}: {description} - 导入失败")
        print(f"      错误: {e}")
    except Exception as e:
        print(f"   ⚠️  {module_name}: 导入时出现异常: {e}")

# 6. 检查CUDA环境
print("\n6. 检查CUDA环境...")
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {cuda_available}")
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"   CUDA设备数: {device_count}")
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            print(f"     cuda:{i} - {name}")
except ImportError:
    print("   ⚠️  PyTorch未安装")
except Exception as e:
    print(f"   ❌ CUDA检测失败: {e}")

# 7. 检查配置文件
print("\n7. 检查配置文件...")
config_path = "configs/cifar10_v13_100clients.json"
try:
    import json
    with open(config_path) as f:
        config = json.load(f)

    # 检查关键字段
    has_partitions = "partitions" in config.get("shared_parameters", {})
    checkpoint_interval = config.get("shared_parameters", {}).get("checkpoint_interval")

    print(f"   配置文件: {config_path}")
    print(f"   使用 'partitions' (复数): {'✅' if has_partitions else '❌'}")
    print(f"   checkpoint_interval: {checkpoint_interval} {'✅' if checkpoint_interval == 0 else '⚠️'}")
except FileNotFoundError:
    print(f"   ❌ 配置文件不存在: {config_path}")
except Exception as e:
    print(f"   ❌ 配置文件解析失败: {e}")

# 8. 输出缓冲测试
print("\n8. 测试输出缓冲...")
print("   写入无换行输出...", end="", flush=True)
sys.stdout.flush()
import time
time.sleep(1)
print(" 完成")
print("   ✅ flush()机制工作正常")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)

print("\n💡 建议:")
print("  - 如果TTY为False，在交互式终端中运行")
print("  - 如果模块导入失败，检查Python路径")
print("  - 如果基础进度条无法显示，检查终端类型")
print("  - 运行实验时注意观察是否有 'START <task_id> device=<device>' 格式的输出")
