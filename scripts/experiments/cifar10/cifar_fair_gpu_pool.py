"""Use every eligible visible GPU in v14 without changing scientific configs."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from functools import wraps
import json
import math
import os

from sm9rrsfl import cuda_execution, experiments, fair_tuning
from sm9rrsfl.torch_backend import _torch_module


def probe_gpu_memory():
    """Keep successful logical-device queries when another GPU cannot be queried.

    Querying an almost full GPU can itself raise CUDA OOM. Never substitute
    zero/free memory for that device or discard the other devices' readings.
    """
    devices = experiments.available_cuda_devices()
    if not devices:
        return (), (), {}
    torch = _torch_module()
    info, failures = [], {}
    for device in devices:
        try:
            index = int(device.split(":")[1])
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            free, total = float(free_bytes) / (1024 * 1024), float(total_bytes) / (1024 * 1024)
            if not (math.isfinite(free) and math.isfinite(total) and 0 <= free <= total and total > 0):
                raise ValueError("invalid free/total GPU memory values")
            info.append((device, free, total))
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            failures[device] = {"error_type": type(exc).__name__, "error": str(exc)}
    return devices, tuple(info), failures


def inspect_gpus():
    devices, info, failures = probe_gpu_memory()
    if not devices:
        raise RuntimeError("v14 requires visible CUDA GPUs; check PyTorch/CUDA and CUDA_VISIBLE_DEVICES")
    if not info:
        raise RuntimeError("no GPU memory query succeeded; no GPU work was scheduled; "
                           "query_failures=" + json.dumps(failures, sort_keys=True))
    return devices, info, failures


def check_gpus():
    """Read-only hardware inventory; no dataset download or experiment output."""
    devices, info, failures = inspect_gpus()
    result = {
        "visible_devices": list(devices),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unmasked"),
        "cpu_slots": experiments.available_cpu_count(),
        "host_memory_mb": round(experiments._physical_memory_mb()),
        "gpu_memory_mb": [{"device": device, "free": round(free), "total": round(total)}
                          for device, free, total in info],
        "memory_query_status": "partial" if failures else "complete",
        "query_failures": failures,
        "policy": "all eligible visible GPUs, one task per GPU; CPU slot count is not a GPU job cap",
        "capacity": "per-task GPU and host memory checked again after loading the dataset",
    }
    print("gpu_inventory=" + json.dumps(result, sort_keys=True), flush=True)
    return result


class Pool:
    def __init__(self, on_plan=None, on_event=None):
        self.devices = ()
        self.jobs = 0
        self.on_plan = on_plan
        self.on_event = on_event
        self.query_failures = {}
        self.required_free_mb = None

    def prepare(self, dataset, tasks, args, *, requested_jobs=None, spread_cuda_devices=True):
        if not tasks:
            return tasks, 1, "torch:cuda", args.sm9_workers
        if (args.compute_backend != "torch" or args.device.strip().lower() != "auto"
                or str(args.jobs).lower() != "auto"
                or requested_jobs is not None and str(requested_jobs).lower() != "auto"):
            raise RuntimeError("v14 GPU pool requires compute_backend=torch, device=auto and jobs/final_jobs=auto")
        visible, info, failures = inspect_gpus()
        self.query_failures = failures
        configs = [task.config for task in tasks]
        required_mb = experiments._estimated_cuda_worker_memory_mb(dataset, configs)
        host_worker_mb = experiments._estimated_parallel_worker_memory_mb(dataset, configs)
        host_memory_mb = experiments._physical_memory_mb()
        host_slots = int(host_memory_mb * .75 // host_worker_mb)
        if host_slots < 1:
            raise RuntimeError("insufficient host memory for one v14 GPU task; increase the container memory allocation")
        fraction = experiments.CUDA_MEMORY_SAFETY_FRACTION
        self.required_free_mb = required_mb / fraction
        by_device = {device: (free, total) for device, free, total in info}
        # Keep busy but sufficiently large GPUs in the pool so they can join later.
        # Unknown-memory cards remain candidates for a later successful query,
        # but ready_devices never admits them while their query is failing.
        self.devices = tuple(d for d in visible
                             if d not in by_device or by_device[d][1] * fraction >= required_mb)
        if not self.devices:
            raise RuntimeError("no visible GPU has sufficient total memory for one v14 task")
        self.jobs = min(len(tasks), len(self.devices), host_slots)
        workers = (experiments.resolve_sm9_workers("auto", max(c.num_clients for c in configs),
                   parallel_jobs=self.jobs) if args._sm9_workers_auto else args.sm9_workers)
        prepared = [replace(task, config=replace(task.config, sm9_workers=workers,
                    device=self.devices[index % len(self.devices)])) for index, task in enumerate(tasks)]
        ready = [d for d in self.devices if d in by_device and by_device[d][0] * fraction >= required_mb]
        report = {
            "phase": tasks[0].phase,
            "visible_devices": list(visible),
            "pool_devices": list(self.devices),
            "ready_now": ready,
            "waiting_for_memory": [d for d in self.devices if d in by_device and d not in ready],
            "waiting_for_memory_query": [d for d in self.devices if d not in by_device],
            "query_failures": failures,
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

    def memory_info(self):
        _, info, failures = probe_gpu_memory()
        if failures != self.query_failures:
            print("gpu_memory_status=" + json.dumps({
                "query_failures": failures,
                "queries_recovered": sorted(set(self.query_failures) - set(failures)),
            }, sort_keys=True), flush=True)
            self.query_failures = failures
        if self.on_plan is not None and self.devices:
            self.on_plan({
                "pool_devices": list(self.devices), "target_jobs": self.jobs,
                "required_gpu_free_mb": self.required_free_mb,
                "gpu_memory_mb": [{"device": d, "free": f, "total": t} for d, f, t in info],
                "query_failures": failures,
            })
        return info

    def ready_devices(self, required_mb):
        # Only successful current readings admit work; one bad card cannot
        # block other cards or be treated as free by the scheduler.
        self.required_free_mb = required_mb / experiments.CUDA_MEMORY_SAFETY_FRACTION
        info = self.memory_info()
        return tuple(device for device, free, _ in info
                     if free * experiments.CUDA_MEMORY_SAFETY_FRACTION >= required_mb)


@contextmanager
def gpu_pool(*, on_plan=None, on_event=None):
    """Scope v14 resource policy; retain original queue, OOM recovery and commits."""
    pool = Pool(on_plan, on_event)
    original_prepare = fair_tuning.prepare_tuning_tasks
    original_queue = fair_tuning.run_cuda_tasks
    original_ready = cuda_execution.cuda_devices_with_capacity
    original_memory = experiments.cuda_device_memory_info

    @wraps(original_queue)
    def queue(tasks, **kwargs):
        # The original runner derives devices from task assignments. Override that
        # narrower set, especially when only a few uncached tasks remain on resume.
        kwargs.update(devices=pool.devices, jobs=min(pool.jobs, len(tasks)),
                      poll_seconds=1., max_idle_seconds=600.)
        run = kwargs["run"]

        def execute(task):
            try:
                return run(task)
            except Exception as error:
                from .cifar_fair_diagnostics import clear_failure_frames
                clear_failure_frames(error)
                raise
            finally:
                # The queue admits work using driver free memory. PyTorch's
                # unused cache is excluded from that reading, so a finished
                # task could otherwise prevent the next task on its own GPU.
                # No other v14 task is active on this device at this point.
                cuda_execution.release_cuda_cache(task.config.device)

        kwargs["run"] = execute
        event = kwargs.get("event")
        if event is not None and pool.on_event is not None:
            def observe(kind, task, **details):
                result = event(kind, task, **details)
                pool.on_event(kind, task, **details)
                return result
            kwargs["event"] = observe
        return original_queue(tasks, **kwargs)

    fair_tuning.prepare_tuning_tasks = pool.prepare
    fair_tuning.run_cuda_tasks = queue
    cuda_execution.cuda_devices_with_capacity = pool.ready_devices
    # Keep the original runner's resource log consistent with this queue's
    # partial inventory. The generic reader itself remains unchanged on disk.
    experiments.cuda_device_memory_info = pool.memory_info
    try:
        yield pool
    finally:
        fair_tuning.prepare_tuning_tasks = original_prepare
        fair_tuning.run_cuda_tasks = original_queue
        cuda_execution.cuda_devices_with_capacity = original_ready
        experiments.cuda_device_memory_info = original_memory
