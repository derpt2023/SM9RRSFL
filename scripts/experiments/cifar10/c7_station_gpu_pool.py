"""Independent, task-aware CUDA admission for the fixed C7 station study.

The accounting is a conservative estimate, not a measured CUDA peak or a
guarantee against another process consuming memory after admission. No batch,
attack, model, or aggregation arithmetic is changed by this module.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import replace
import json
import math
import os
from time import monotonic, sleep
import traceback

from sm9rrsfl import cuda_execution, experiments, fair_tuning
from sm9rrsfl.tuning_resume import runtime_config_key
from .cifar_fair_gpu_pool import probe_gpu_memory


MIB = 1024 * 1024


def check_gpus():
    """Inventory only; visible-but-unreadable cards may recover during the run."""
    devices, info, failures = probe_gpu_memory()
    if not devices:
        raise RuntimeError("C7 station requires visible CUDA GPUs; check PyTorch/CUDA and CUDA_VISIBLE_DEVICES")
    result = {
        "visible_devices": list(devices),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unmasked"),
        "cpu_slots": experiments.available_cpu_count(),
        "host_memory_mb": round(experiments._physical_memory_mb()),
        "gpu_memory_mb": [{"device": d, "free": round(f), "total": round(t)} for d, f, t in info],
        "memory_query_status": "unavailable" if not info else "partial" if failures else "complete",
        "query_failures": failures,
        "policy": "one task per eligible visible GPU; task-specific memory admission; CPU slots do not cap GPU jobs",
        "capacity": "estimates checked after loading the dataset; unknown readings never admit a task",
        "startup_behavior": "run waits dynamically up to 600 seconds if no pending task fits; checkpoints retained on timeout",
    }
    print("gpu_inventory=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def _array_bytes(array, itemsize):
    if array is None:
        return 0
    size = getattr(array, "size", None)
    if size is None:
        size = math.prod(array.shape)
    return int(size) * itemsize


def estimate_task_memory(dataset, config):
    """Return explicit MiB allowances for this CNN task's peak live tensors.

    Images are resident float32 and labels int64. Clients train sequentially:
    three Adam optimizers need six moment vectors for ONE client, not N clients.
    Twelve vectors additionally reserve optimizer parameters/step temporaries.
    Krum retains the original updates plus its stack and cdist workspaces. The
    latter is backend-dependent, so four further N-by-P arrays are reserved.
    """
    params = experiments._parameter_size_for_dataset(dataset)
    vector_mb = params * 4 / MIB
    clients = config.num_clients
    resident_mb = sum(
        _array_bytes(getattr(dataset, name, None), size)
        for name, size in (("x_train", 4), ("y_train", 8), ("x_test", 4),
                           ("y_test", 8), ("x_attack", 4), ("y_attack", 8))
    ) / MIB
    updates_mb = clients * vector_mb
    attacking = config.malicious_ratio > 0 and config.attack not in {"none", ""}
    components = {
        "resident_train_eval_attack": resident_mb,
        "all_client_updates": updates_mb,
        "model_training_scratch": 16 * vector_mb,
        "c7_adam_moments": 6 * vector_mb if attacking else 0.,
        "c7_adam_extra_parameters_and_steps": 12 * vector_mb if attacking else 0.,
        "aggregation_workspace": (4 if config.method == "krum" else 1) * updates_mb,
        "vert_projection_and_predictor": 0.,
        "cuda_workspace_allowance": 768.,
    }
    projection_kind = None
    if config.method == "vert":
        dimension = min(config.vert_projection_dim, params)
        dense_mb = dimension * vector_mb
        if dense_mb <= 256.:
            projection_kind = "dense"
            projection_mb = dense_mb + dimension * 4 / MIB
            coefficient_dim = params
        else:
            projection_kind = "signed_sketch"
            projection_mb = params * (8 + 4) / MIB + dimension * 4 / MIB
            coefficient_dim = dimension
        predictor_mb = (3 * dimension ** 2 + 3 * dimension + 2 * coefficient_dim) * 4 / MIB
        # Predictor, its moments, gradients, and autograd workspaces.
        components["vert_projection_and_predictor"] = projection_mb + 8 * predictor_mb
    estimated_mb = max(1024., sum(components.values()))
    return {
        "method": config.method,
        "malicious_ratio": config.malicious_ratio,
        "parameter_count": params,
        "num_clients": clients,
        "components_mb": components,
        "vert_projection_kind": projection_kind,
        "estimated_worker_mb": estimated_mb,
        "required_gpu_free_mb": estimated_mb / experiments.CUDA_MEMORY_SAFETY_FRACTION,
        "safety_fraction": experiments.CUDA_MEMORY_SAFETY_FRACTION,
        "status": "estimate_not_measured_peak_or_guarantee",
    }


class Pool:
    def __init__(self, on_plan=None, on_event=None):
        self.devices = ()
        self.jobs = 0
        self.on_plan = on_plan
        self.on_event = on_event
        self.query_failures = {}
        self.required_by_config = {}
        self.memory_bills = []
        self.required_free_mb = None

    def prepare(self, dataset, tasks, args, *, requested_jobs=None, spread_cuda_devices=True):
        if not tasks:
            return tasks, 1, "torch:cuda", args.sm9_workers
        if (args.compute_backend != "torch" or args.device.strip().lower() != "auto"
                or str(args.jobs).lower() != "auto"
                or requested_jobs is not None and str(requested_jobs).lower() != "auto"):
            raise RuntimeError("C7 station GPU pool requires compute_backend=torch, device=auto and jobs/final_jobs=auto")
        visible, info, failures = probe_gpu_memory()
        if not visible:
            raise RuntimeError("C7 station requires visible CUDA GPUs")
        self.query_failures = failures
        configs = [task.config for task in tasks]
        bills = [estimate_task_memory(dataset, config) for config in configs]
        self.required_by_config = {
            runtime_config_key(config): bill["estimated_worker_mb"]
            for config, bill in zip(configs, bills)
        }
        # The identical shapes/algorithms at different seeds need one public bill.
        self.memory_bills = list({json.dumps(bill, sort_keys=True): bill for bill in bills}.values())
        smallest = min(self.required_by_config.values())
        fraction = experiments.CUDA_MEMORY_SAFETY_FRACTION
        self.required_free_mb = smallest / fraction
        by_device = {device: (free, total) for device, free, total in info}
        self.devices = tuple(device for device in visible
                             if device not in by_device or by_device[device][1] * fraction >= smallest)
        if not self.devices:
            raise RuntimeError("no visible GPU has sufficient total memory for any C7 station task")
        # Unknown cards can recover; known cards must collectively fit every task.
        if all(device in by_device for device in self.devices):
            largest_capacity = max(by_device[device][1] * fraction for device in self.devices)
            if any(required > largest_capacity for required in self.required_by_config.values()):
                raise RuntimeError("some C7 station tasks exceed every visible GPU's total-memory allowance")
        host_worker_mb = experiments._estimated_parallel_worker_memory_mb(dataset, configs)
        host_memory_mb = experiments._physical_memory_mb()
        host_slots = int(host_memory_mb * .75 // host_worker_mb)
        if host_slots < 1:
            raise RuntimeError("insufficient host memory for one C7 station GPU task")
        self.jobs = min(len(tasks), len(self.devices), host_slots)
        workers = (experiments.resolve_sm9_workers("auto", max(c.num_clients for c in configs),
                   parallel_jobs=self.jobs) if args._sm9_workers_auto else args.sm9_workers)
        prepared = [replace(task, config=replace(task.config, sm9_workers=workers,
                    device=self.devices[index % len(self.devices)])) for index, task in enumerate(tasks)]
        self.required_by_config = {
            runtime_config_key(task.config): bill["estimated_worker_mb"]
            for task, bill in zip(prepared, bills)
        }
        ready = [device for device in self.devices
                 if device in by_device and by_device[device][0] * fraction >= smallest]
        report = {
            "phase": tasks[0].phase, "visible_devices": list(visible),
            "pool_devices": list(self.devices), "ready_now": ready,
            "ready_now_semantics": "can_fit_at_least_one_task_not_every_method",
            "waiting_for_memory": [d for d in self.devices if d in by_device and d not in ready],
            "waiting_for_memory_query": [d for d in self.devices if d not in by_device],
            "query_failures": failures,
            "excluded_total_memory": [d for d in visible if d not in self.devices],
            "target_jobs": self.jobs, "cpu_slots": experiments.available_cpu_count(),
            "cpu_slot_cap_applied": False, "host_slots": host_slots,
            "host_memory_limited": host_slots < min(len(tasks), len(self.devices)),
            "estimated_host_worker_mb": round(host_worker_mb),
            "required_gpu_free_mb": self.required_free_mb,
            "required_gpu_free_mb_semantics": "minimum_across_tasks; admission_is_task_specific",
            "memory_bills": self.memory_bills, "poll_seconds": 1, "max_idle_seconds": 600,
            "gpu_memory_mb": [{"device": d, "free": f, "total": t} for d, f, t in info],
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

    def required_mb(self, task):
        key = runtime_config_key(task.config)
        if key not in self.required_by_config:
            raise ValueError("CUDA task was not included in this pool's prepared phase")
        return self.required_by_config[key]

    def queue(self, tasks, *, run, commit, event, devices=None, jobs=None, required_mb=None,
              max_retries=1, max_idle_seconds=600., cooldown_seconds=10., poll_seconds=1.):
        """Drain successes on failure; OOM retries retain original checkpoint IDs."""
        if not tasks:
            return
        if not self.devices or self.jobs < 1:
            raise ValueError("prepare the C7 station GPU pool before running tasks")
        for task in tasks:
            self.required_mb(task)
        pending_tasks = deque(tasks)
        attempts, cooldown, pending = {}, {}, {}
        fatal = None
        idle_since = None
        last_wait_log = None
        concurrency = min(self.jobs, len(tasks), len(self.devices))

        def emit(kind, task, **details):
            if event is not None:
                event(kind, task, **details)
            if self.on_event is not None:
                self.on_event(kind, task, **details)

        def execute(task):
            try:
                return run(task)
            except Exception as error:
                # Drop failed frames before empty_cache so retry does not retain
                # another model through Future traceback references.
                from .cifar_fair_diagnostics import clear_failure_frames
                clear_failure_frames(error)
                raise
            finally:
                cuda_execution.release_cuda_cache(task.config.device)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            while pending_tasks or pending:
                if fatal is None and pending_tasks:
                    # One inventory per poll, reused for all device/task matches.
                    self.required_free_mb = min(self.required_mb(task) for task in pending_tasks) / experiments.CUDA_MEMORY_SAFETY_FRACTION
                    free_by_device = {d: free for d, free, _ in self.memory_info()}
                    occupied = {task.config.device for task in pending.values()}
                    now = monotonic()
                    for device in self.devices:
                        if len(pending) >= concurrency or not pending_tasks:
                            break
                        if (device in occupied or device not in free_by_device
                                or cooldown.get(device, 0.) > now):
                            continue
                        capacity = free_by_device[device] * experiments.CUDA_MEMORY_SAFETY_FRACTION
                        fitting = [task for task in pending_tasks if self.required_mb(task) <= capacity]
                        if not fitting:
                            continue
                        task = next((task for task in fitting if task.config.device == device), fitting[0])
                        pending_tasks.remove(task)
                        if task.config.device != device:
                            emit("cuda_task_reassigned", task, previous_device=task.config.device, next_device=device)
                        task = replace(task, config=replace(task.config, device=device),
                                       checkpoint_config=task.checkpoint_config or task.config)
                        key = runtime_config_key(task.config)
                        attempts[key] = attempts.get(key, 0) + 1
                        pending[executor.submit(execute, task)] = task
                        occupied.add(device)
                        idle_since = None
                if not pending:
                    if fatal is not None:
                        break
                    now = monotonic()
                    if idle_since is None:
                        idle_since = now
                    if last_wait_log is None or now - last_wait_log >= 15.:
                        emit("cuda_waiting_for_memory", None, allowed_devices=list(self.devices),
                             pending_tasks=len(pending_tasks),
                             estimated_worker_mb=min(self.required_mb(task) for task in pending_tasks),
                             idle_seconds=round(now - idle_since, 1))
                        last_wait_log = now
                    if now - idle_since >= max_idle_seconds:
                        fatal = cuda_execution.CudaResourcesUnavailable(
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
                        if cuda_execution.is_cuda_oom(error):
                            key = runtime_config_key(task.config)
                            retry = attempts[key] <= max_retries and fatal is None
                            emit("cuda_oom", task, attempt=attempts[key], retry=retry, error=str(error))
                            traceback.clear_frames(error.__traceback__)
                            error.__traceback__ = None
                            cuda_execution.release_cuda_cache(task.config.device)
                            cooldown[task.config.device] = monotonic() + cooldown_seconds
                            if retry:
                                pending_tasks.appendleft(task)
                            elif fatal is None:
                                fatal = cuda_execution.CudaResourcesUnavailable(
                                    f"CUDA OOM retry limit reached for {task.candidate_id} "
                                    f"seed={task.config.seed}; preserve outputs and resume on an idle GPU.")
                        else:
                            emit("cuda_worker_failed", task, error_type=type(error).__name__, error=str(error))
                            fatal = error if fatal is None else fatal
                    else:
                        try:
                            commit(task, result)
                        except Exception as error:
                            fatal = error if fatal is None else fatal
                if fatal is not None:
                    pending_tasks.clear()
        if fatal is not None:
            raise fatal


@contextmanager
def gpu_pool(*, on_plan=None, on_event=None):
    """Install only execution hooks, restoring their exact enclosing callables."""
    pool = Pool(on_plan, on_event)
    original_prepare = fair_tuning.prepare_tuning_tasks
    original_queue = fair_tuning.run_cuda_tasks
    original_memory = experiments.cuda_device_memory_info
    fair_tuning.prepare_tuning_tasks = pool.prepare
    fair_tuning.run_cuda_tasks = pool.queue
    experiments.cuda_device_memory_info = pool.memory_info
    try:
        yield pool
    finally:
        fair_tuning.prepare_tuning_tasks = original_prepare
        fair_tuning.run_cuda_tasks = original_queue
        experiments.cuda_device_memory_info = original_memory
