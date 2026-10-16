"""Use every eligible visible GPU in v14 without changing scientific configs."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from functools import wraps
import json
import os

from sm9rrsfl import cuda_execution, experiments, fair_tuning


def inspect_gpus():
    devices = experiments.available_cuda_devices()
    if not devices:
        raise RuntimeError("v14 requires visible CUDA GPUs; check PyTorch/CUDA and CUDA_VISIBLE_DEVICES")
    info = experiments.cuda_device_memory_info()
    if info is None or any(device not in {row[0] for row in info} for device in devices):
        raise RuntimeError("cannot inspect all visible GPUs' memory; no GPU work was scheduled")
    return devices, info


def check_gpus():
    """Read-only hardware inventory; no dataset download or experiment output."""
    devices, info = inspect_gpus()
    result = {
        "visible_devices": list(devices),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unmasked"),
        "cpu_slots": experiments.available_cpu_count(),
        "host_memory_mb": round(experiments._physical_memory_mb()),
        "gpu_memory_mb": [{"device": device, "free": round(free), "total": round(total)}
                          for device, free, total in info],
        "policy": "all eligible visible GPUs, one task per GPU; CPU slot count is not a GPU job cap",
        "capacity": "per-task GPU and host memory checked again after loading the dataset",
    }
    print("gpu_inventory=" + json.dumps(result, sort_keys=True), flush=True)
    return result


class Pool:
    def __init__(self, on_plan=None):
        self.devices = ()
        self.jobs = 0
        self.on_plan = on_plan

    def prepare(self, dataset, tasks, args, *, requested_jobs=None, spread_cuda_devices=True):
        if not tasks:
            return tasks, 1, "torch:cuda", args.sm9_workers
        if (args.compute_backend != "torch" or args.device.strip().lower() != "auto"
                or str(args.jobs).lower() != "auto"
                or requested_jobs is not None and str(requested_jobs).lower() != "auto"):
            raise RuntimeError("v14 GPU pool requires compute_backend=torch, device=auto and jobs/final_jobs=auto")
        visible, info = inspect_gpus()
        configs = [task.config for task in tasks]
        required_mb = experiments._estimated_cuda_worker_memory_mb(dataset, configs)
        host_worker_mb = experiments._estimated_parallel_worker_memory_mb(dataset, configs)
        host_memory_mb = experiments._physical_memory_mb()
        host_slots = int(host_memory_mb * .75 // host_worker_mb)
        if host_slots < 1:
            raise RuntimeError("insufficient host memory for one v14 GPU task; increase the container memory allocation")
        fraction = experiments.CUDA_MEMORY_SAFETY_FRACTION
        by_device = {device: (free, total) for device, free, total in info}
        # Keep busy but sufficiently large GPUs in the pool so they can join later.
        self.devices = tuple(d for d in visible if by_device[d][1] * fraction >= required_mb)
        if not self.devices:
            raise RuntimeError("no visible GPU has sufficient total memory for one v14 task")
        self.jobs = min(len(tasks), len(self.devices), host_slots)
        workers = (experiments.resolve_sm9_workers("auto", max(c.num_clients for c in configs),
                   parallel_jobs=self.jobs) if args._sm9_workers_auto else args.sm9_workers)
        prepared = [replace(task, config=replace(task.config, sm9_workers=workers,
                    device=self.devices[index % len(self.devices)])) for index, task in enumerate(tasks)]
        ready = [d for d in self.devices if by_device[d][0] * fraction >= required_mb]
        report = {
            "phase": tasks[0].phase,
            "visible_devices": list(visible),
            "pool_devices": list(self.devices),
            "ready_now": ready,
            "waiting_for_memory": [d for d in self.devices if d not in ready],
            "excluded_total_memory": [d for d in visible if d not in self.devices],
            "target_jobs": self.jobs,
            "cpu_slots": experiments.available_cpu_count(),
            "cpu_slot_cap_applied": False,
            "host_slots": host_slots,
            "host_memory_limited": host_slots < min(len(tasks), len(self.devices)),
            "estimated_host_worker_mb": round(host_worker_mb),
            "required_gpu_free_mb": round(required_mb / fraction),
            "poll_seconds": 1,
            "max_idle_seconds": 600,
            "gpu_memory_mb": [{"device": d, "free": round(f), "total": round(t)} for d, f, t in info],
        }
        print("gpu_pool_plan=" + json.dumps(report, sort_keys=True), flush=True)
        if self.on_plan is not None:
            self.on_plan(report)
        return prepared, self.jobs, "torch:cuda", workers

    @staticmethod
    def ready_devices(required_mb):
        # A failed runtime query must not be interpreted as all cards being free.
        info = experiments.cuda_device_memory_info()
        if info is None:
            return ()
        return tuple(device for device, free, _ in info
                     if free * experiments.CUDA_MEMORY_SAFETY_FRACTION >= required_mb)


@contextmanager
def gpu_pool(*, on_plan=None):
    """Scope v14 resource policy; retain original queue, OOM recovery and commits."""
    pool = Pool(on_plan)
    original_prepare = fair_tuning.prepare_tuning_tasks
    original_queue = fair_tuning.run_cuda_tasks
    original_ready = cuda_execution.cuda_devices_with_capacity

    @wraps(original_queue)
    def queue(tasks, **kwargs):
        # The original runner derives devices from task assignments. Override that
        # narrower set, especially when only a few uncached tasks remain on resume.
        kwargs.update(devices=pool.devices, jobs=min(pool.jobs, len(tasks)),
                      poll_seconds=1., max_idle_seconds=600.)
        return original_queue(tasks, **kwargs)

    fair_tuning.prepare_tuning_tasks = pool.prepare
    fair_tuning.run_cuda_tasks = queue
    cuda_execution.cuda_devices_with_capacity = pool.ready_devices
    try:
        yield pool
    finally:
        fair_tuning.prepare_tuning_tasks = original_prepare
        fair_tuning.run_cuda_tasks = original_queue
        cuda_execution.cuda_devices_with_capacity = original_ready
