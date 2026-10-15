#!/usr/bin/env python3
"""Test dynamic refresh of GPU progress display."""

import json
import sys
import time

# Simulate progress events
events = [
    # START events for multiple GPUs
    {"type": "START", "partition": "iid", "clients": 100, "method": "sm9rrs", "ratio": 0.2, "device": "cuda:0"},
    {"type": "START", "partition": "dirichlet", "clients": 100, "method": "VERT", "ratio": 0.4, "device": "cuda:1"},
    {"type": "START", "partition": "iid", "clients": 100, "method": "AlignIns", "ratio": 0.6, "device": "cuda:2"},
]

# Print initial events
for event in events:
    print(f"progress_event={json.dumps(event)}", flush=True)
    time.sleep(0.5)

# Simulate rounds progressing
for round_num in range(1, 21):
    time.sleep(2)  # Wait 2 seconds per round

    # Update all GPUs
    for idx, device in enumerate(["cuda:0", "cuda:1", "cuda:2"]):
        # Each GPU progresses at different rates
        actual_round = round_num + idx * 5
        event = {"type": "ROUND", "round": actual_round, "device": device}
        print(f"progress_event={json.dumps(event)}", flush=True)

# Finish tasks
for device in ["cuda:0", "cuda:1", "cuda:2"]:
    event = {"type": "FINISH", "device": device}
    print(f"progress_event={json.dumps(event)}", flush=True)
    time.sleep(0.5)

print("Test completed!")
