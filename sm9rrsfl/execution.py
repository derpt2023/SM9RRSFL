"""Bounded device-affine scheduling shared by calibration and final runs."""

from collections import defaultdict, deque
from concurrent.futures import as_completed, wait, FIRST_COMPLETED


def completed_pairs(futures):
    if isinstance(futures, dict):
        for future in as_completed(futures):
            yield future, futures[future]
    else:
        yield from futures


def device_affine_futures(executor, tasks, *, submit, device_key, jobs):
    """Refill free devices, not the next global index in a greedy queue.

    With eight jobs/eight GPUs there is exactly one active job per GPU.
    Even after resume leaves work on only one GPU, its slot is not multiplied.
    Completion and durable snapshot writes remain in the caller/parent.
    """
    if jobs < 1:
        raise ValueError("jobs must be positive")
    queues = defaultdict(deque)
    for task in tasks:
        queues[device_key(task)].append(task)
    if not queues:
        return
    devices = deque(queues)
    per_device = 1
    active = defaultdict(int)
    pending = {}

    def fill():
        for _ in range(len(devices) * per_device):
            if len(pending) >= jobs:
                return
            key = devices[0]
            devices.rotate(-1)
            if queues[key] and active[key] < per_device:
                task = queues[key].popleft()
                pending[submit(executor, task)] = (key, task)
                active[key] += 1

    fill()
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            key, task = pending.pop(future)
            active[key] -= 1
            yield future, task
        fill()
