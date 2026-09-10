"""Bounded CUDA recovery for independent tuning experiments.

Scheduling may change; training batches, evaluation batches and parameters do
not. Successful results are committed even when another worker fails.
"""

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
import gc
from time import monotonic, sleep
import traceback

from .experiments import cuda_devices_with_capacity
from .tuning_resume import runtime_config_key


class CudaResourcesUnavailable(RuntimeError):
    pass


def is_cuda_oom(error):
    message = str(error).lower()
    return "cuda out of memory" in message or "cuda error: out of memory" in message


def release_cuda_cache(device):
    """Only discard this process's unused allocator cache on an idle device."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    except (ImportError, RuntimeError, ValueError):
        # Cleanup cannot hide the original OOM or become the reason to retry.
        pass


def run_cuda_tasks(tasks, *, devices, jobs, required_mb, run, commit, event,
                   max_retries=1, max_idle_seconds=60., cooldown_seconds=10.,
                   poll_seconds=5.):
    """Use at most one worker per allowed device; retry OOM from its checkpoint.

    A failed card cools down while another allowed card can take the pending
    task. When no card is usable, bounded waiting ends with a resumable error.
    Non-OOM exceptions are never treated as a successful or skippable run.
    """
    queue = deque(tasks)
    devices = tuple(dict.fromkeys(devices))
    if not devices or jobs < 1:
        raise ValueError("CUDA execution requires allowed devices and positive jobs")
    attempts = {}
    cooldown = {}
    pending = {}
    fatal = None
    idle_since = None
    last_wait_log = None
    with ThreadPoolExecutor(max_workers=min(jobs, len(devices))) as pool:
        while queue or pending:
            if fatal is None and queue:
                ready = set(cuda_devices_with_capacity(required_mb))
                occupied = {t.config.device for t in pending.values()}
                now = monotonic()
                for device in devices:
                    if len(pending) >= jobs or not queue:
                        break
                    canonical = "cuda:0" if device == "cuda" else device
                    if device in occupied or canonical not in ready or cooldown.get(device, 0.) > now:
                        continue
                    # Keep previous placement when available, so normal runs
                    # retain their original device assignment and ordering.
                    task = next((t for t in queue if t.config.device == device), queue[0])
                    queue.remove(task)
                    if task.config.device != device:
                        event("cuda_task_reassigned", task, previous_device=task.config.device,
                              next_device=device)
                    task = replace(task, config=replace(task.config, device=device),
                                   checkpoint_config=task.checkpoint_config or task.config)
                    key = runtime_config_key(task.config)
                    attempts[key] = attempts.get(key, 0) + 1
                    pending[pool.submit(run, task)] = task
                    occupied.add(device)
                    idle_since = None
            if not pending:
                if fatal is not None:
                    break
                now = monotonic()
                if idle_since is None:
                    idle_since = now
                if last_wait_log is None or now - last_wait_log >= 15.:
                    event("cuda_waiting_for_memory", None, allowed_devices=list(devices),
                          pending_tasks=len(queue), estimated_worker_mb=required_mb,
                          idle_seconds=round(now - idle_since, 1))
                    last_wait_log = now
                if now - idle_since >= max_idle_seconds:
                    fatal = CudaResourcesUnavailable(
                        f"no allowed CUDA device became available within {max_idle_seconds:g}s; "
                        "completed results and round checkpoints are retained. Resume with idle GPUs.")
                    break
                sleep(min(poll_seconds, max_idle_seconds - (now - idle_since)))
                continue
            done, _ = wait(pending, timeout=poll_seconds, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    if is_cuda_oom(error):
                        key = runtime_config_key(task.config)
                        retry = attempts[key] <= max_retries and fatal is None
                        event("cuda_oom", task, attempt=attempts[key], retry=retry,
                              error=str(error))
                        # Future tracebacks otherwise retain the failed model,
                        # local gradients and tensors, defeating any retry.
                        traceback.clear_frames(error.__traceback__)
                        error.__traceback__ = None
                        release_cuda_cache(task.config.device)
                        cooldown[task.config.device] = monotonic() + cooldown_seconds
                        if retry:
                            queue.appendleft(task)
                        elif fatal is None:
                            fatal = CudaResourcesUnavailable(
                                f"CUDA OOM retry limit reached for {task.candidate_id} "
                                f"seed={task.config.seed}; preserve outputs and resume on an idle GPU.")
                    else:
                        event("cuda_worker_failed", task, error_type=type(error).__name__, error=str(error))
                        fatal = error if fatal is None else fatal
                else:
                    # Snapshot failures are fatal too, but drain other workers
                    # so their successful work can still be committed.
                    try:
                        commit(task, result)
                    except Exception as error:
                        fatal = error if fatal is None else fatal
            if fatal is not None:
                queue.clear()
    if fatal is not None:
        raise fatal
