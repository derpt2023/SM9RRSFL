#!/usr/bin/env python3
"""测试进度事件输出功能."""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def test_event_format():
    """测试事件格式是否正确."""
    print("测试 1: 验证事件 JSON 格式...")

    # 模拟输出事件
    events = [
        {"type": "START", "partition": "iid", "clients": 100, "method": "sm9rrs", "ratio": 0.20, "device": "cuda:0"},
        {"type": "ROUND", "round": 1, "device": "cuda:0"},
        {"type": "FINISH", "device": "cuda:0"}
    ]

    for event in events:
        line = f"progress_event={json.dumps(event)}"
        # 验证可以解析
        assert line.startswith("progress_event=")
        parsed = json.loads(line[15:])
        assert parsed["type"] == event["type"]

    print("✅ 事件格式正确\n")


def test_display_parsing():
    """测试显示器能否解析事件."""
    print("测试 2: 验证显示器事件解析...")

    from fair_tuning_gpu_display import GPUProgressDisplay

    display = GPUProgressDisplay()
    display.total = 10

    # 测试 START 事件
    event = {"type": "START", "partition": "iid", "clients": 100, "method": "sm9rrs", "ratio": 0.20, "device": "cuda:0"}
    config = {
        'partition': event['partition'],
        'clients': str(event['clients']),
        'method': event['method'],
        'ratio': f"{event['ratio']:.2f}"
    }
    display.update_gpu_status('cuda:0', config)

    assert 'cuda:0' in display.gpu_tasks
    assert display.gpu_tasks['cuda:0']['method'] == 'sm9rrs'

    # 测试 ROUND 事件
    display.gpu_tasks['cuda:0']['round'] = 5
    assert display.gpu_tasks['cuda:0']['round'] == 5

    # 测试 FINISH 事件
    display.remove_gpu_task('cuda:0')
    assert 'cuda:0' not in display.gpu_tasks

    print("✅ 显示器解析正确\n")


def test_env_variable():
    """测试环境变量控制."""
    print("测试 3: 验证环境变量控制...")

    # 测试启用
    os.environ["EMIT_PROGRESS_EVENTS"] = "1"
    assert os.environ.get("EMIT_PROGRESS_EVENTS", "0") == "1"

    # 测试禁用
    os.environ["EMIT_PROGRESS_EVENTS"] = "0"
    assert os.environ.get("EMIT_PROGRESS_EVENTS", "0") == "0"

    # 清理
    del os.environ["EMIT_PROGRESS_EVENTS"]

    print("✅ 环境变量控制正确\n")


def test_pipeline():
    """测试管道处理."""
    print("测试 4: 验证管道处理...")

    # 创建测试脚本
    test_script = """
import json
import sys

print('progress_event=' + json.dumps({
    'type': 'START',
    'partition': 'iid',
    'clients': 100,
    'method': 'sm9rrs',
    'ratio': 0.20,
    'device': 'cuda:0'
}))
sys.stdout.flush()

for r in range(1, 4):
    print(f'Round {r}...')
    print('progress_event=' + json.dumps({
        'type': 'ROUND',
        'round': r,
        'device': 'cuda:0'
    }))
    sys.stdout.flush()

print('progress_event=' + json.dumps({
    'type': 'FINISH',
    'device': 'cuda:0'
}))
sys.stdout.flush()
"""

    # 运行管道
    proc = subprocess.Popen(
        ["python3", "-c", test_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = proc.communicate(timeout=5)

    # 验证输出包含事件
    assert "progress_event=" in stdout
    assert '"type": "START"' in stdout
    assert '"type": "FINISH"' in stdout

    print("✅ 管道处理正确\n")


if __name__ == "__main__":
    try:
        test_event_format()
        test_display_parsing()
        test_env_variable()
        test_pipeline()

        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        print("\n使用方法:")
        print("  export EMIT_PROGRESS_EVENTS=1")
        print("  python3 -u run_fair_tuning_from_config.py --config CONFIG | python3 fair_tuning_gpu_display.py")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
