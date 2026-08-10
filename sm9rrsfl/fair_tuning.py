"""Fair, dataset-specific hyperparameter search for every comparison method.

The tuner keeps training/attack/evaluation settings shared across methods,
selects method-specific defense parameters on a holdout split of the training
set, and only then runs the unified main evaluation on the untouched test set.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass, replace
from itertools import product
import json
from pathlib import Path
from statistics import fmean, pstdev
import sys
from typing import Any, Iterable

import numpy as np

from .config_runner import ConfigError, parameters_to_argv
from .datasets import ImageDataset, load_image_dataset
from .experiments import (
    ProgressReporter,
    assign_auto_cuda_devices,
    available_cuda_devices,
    build_experiment_configs,
    parallel_executor_kind,
    parse_args,
    print_resource_plan,
    resolve_parallel_jobs,
    resolve_sm9_workers,
    run_measured_experiment,
    write_result_files,
)
from .fl import ExperimentConfig, ExperimentResult, malicious_client_count
from .model import describe_compute_backend
from .visualization import generate_visualizations


TUNING_SCHEMA_VERSION = 1
ALL_METHODS = (
    "sm9rrs",
    "vert",
    "fedredefense",
    "krum",
    "ding13",
    "fedavg",
)
METHOD_TUNABLE_PARAMETERS = {
    "sm9rrs": frozenset(
        {
            "detector_window",
            "z_threshold",
            "suspicion_penalty_factor",
            "suspicion_recovery_factor",
            "suspicion_remove_after",
        }
    ),
    "vert": frozenset(
        {
            "vert_history_window",
            "vert_projection_dim",
            "vert_predict_epochs",
            "vert_predict_lr",
            "vert_top_k",
            "vert_use_ratio_prior",
        }
    ),
    "fedredefense": frozenset(
        {
            "fedre_threshold",
            "fedre_initial_iterations",
            "fedre_max_iterations",
            "fedre_synthetic_steps",
            "fedre_images_per_class",
            "fedre_image_lr",
            "fedre_label_lr",
            "fedre_teacher_lr",
            "fedre_teacher_lr_lr",
        }
    ),
    "krum": frozenset(),
    "ding13": frozenset(),
    "fedavg": frozenset(),
}
ROOT_KEYS = {
    "schema_version",
    "name",
    "description",
    "shared_parameters",
    "tuning",
}
TUNING_KEYS = {
    "algorithm",
    "validation_fraction",
    "split_seed",
    "validation_seeds",
    "final_seeds",
    "trials_per_tunable_method",
    "require_finite_updates",
    "run_final_evaluation",
    "objective",
    "method_spaces",
}
OBJECTIVE_DEFAULTS = {
    "clean_accuracy_weight": 0.25,
    "robust_accuracy_weight": 0.50,
    "attack_success_weight": 0.20,
    "false_positive_weight": 0.05,
}


@dataclass(frozen=True)
class FairTuningConfig:
    source: Path
    name: str
    description: str
    shared_parameters: dict[str, Any]
    validation_fraction: float
    split_seed: int
    validation_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    trials_per_tunable_method: int
    require_finite_updates: bool
    run_final_evaluation: bool
    objective: dict[str, float]
    candidates: dict[str, tuple[dict[str, Any], ...]]


@dataclass(frozen=True)
class TrialScore:
    method: str
    candidate_id: str
    parameters: dict[str, Any]
    score: float
    valid: bool
    clean_accuracy: float
    robust_accuracy: float
    attack_success_rate: float
    false_positive_rate: float
    nonfinite_updates: int
    result_count: int

    def row(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "candidate_id": self.candidate_id,
            "parameters": json.dumps(
                self.parameters,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "score": self.score,
            "valid": self.valid,
            "clean_accuracy": self.clean_accuracy,
            "robust_accuracy": self.robust_accuracy,
            "attack_success_rate": self.attack_success_rate,
            "false_positive_rate": self.false_positive_rate,
            "nonfinite_updates": self.nonfinite_updates,
            "result_count": self.result_count,
        }


@dataclass(frozen=True)
class TuningExperimentTask:
    """One independently executable validation or final-evaluation config."""

    phase: str
    candidate_id: str
    method: str
    config: ExperimentConfig


_TUNING_WORKER_DATASET: ImageDataset | None = None


class FairTuningError(ValueError):
    """The requested search would violate the declared fairness protocol."""


def load_fair_tuning_config(path: str | Path) -> FairTuningConfig:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FairTuningError(f"cannot read tuning configuration {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FairTuningError(
            f"invalid JSON in {source} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise FairTuningError("tuning configuration root must be a JSON object")
    unknown_root = sorted(set(payload) - ROOT_KEYS)
    if unknown_root:
        raise FairTuningError(f"unknown tuning root key: {unknown_root[0]}")
    if payload.get("schema_version") != TUNING_SCHEMA_VERSION:
        raise FairTuningError(
            f"tuning schema_version must be {TUNING_SCHEMA_VERSION}"
        )

    shared = payload.get("shared_parameters")
    tuning = payload.get("tuning")
    if not isinstance(shared, dict):
        raise FairTuningError("shared_parameters must be a JSON object")
    if not isinstance(tuning, dict):
        raise FairTuningError("tuning must be a JSON object")
    unknown_tuning = sorted(set(tuning) - TUNING_KEYS)
    if unknown_tuning:
        raise FairTuningError(f"unknown tuning key: {unknown_tuning[0]}")
    if tuning.get("algorithm", "grid") != "grid":
        raise FairTuningError("only deterministic exhaustive grid search is supported")

    try:
        args = parse_args(parameters_to_argv(shared))
    except (ConfigError, SystemExit) as exc:
        raise FairTuningError("invalid shared experiment parameters") from exc
    if tuple(args.methods) != ALL_METHODS and set(args.methods) != set(ALL_METHODS):
        raise FairTuningError(
            "shared_parameters.methods must contain all six methods exactly once"
        )
    if len(args.methods) != len(ALL_METHODS):
        raise FairTuningError("shared_parameters.methods contains duplicate methods")
    if not any(abs(ratio) < 1e-12 for ratio in args.ratios):
        raise FairTuningError("ratios must include a 0% clean-control scenario")
    if not any(ratio > 0.0 for ratio in args.ratios):
        raise FairTuningError("ratios must include at least one attacked scenario")
    if not shared.get("early_stop") is False:
        raise FairTuningError(
            "shared_parameters.early_stop must be false so every candidate gets the same rounds"
        )
    if args.eval_interval != 1:
        raise FairTuningError(
            "shared_parameters.eval_interval must be 1 for complete stability auditing"
        )
    if not args.output_dir:
        raise FairTuningError(
            "shared_parameters.output_dir is required to avoid overwriting a main run"
        )

    validation_fraction = _finite_float(
        tuning.get("validation_fraction", 0.1),
        "validation_fraction",
    )
    if not 0.0 < validation_fraction < 0.5:
        raise FairTuningError("validation_fraction must be in (0, 0.5)")
    split_seed = _integer(tuning.get("split_seed", 2026), "split_seed")
    validation_seeds = _seed_list(tuning.get("validation_seeds"), "validation_seeds")
    final_seeds = _seed_list(tuning.get("final_seeds"), "final_seeds")
    if set(validation_seeds) & set(final_seeds):
        raise FairTuningError("validation_seeds and final_seeds must be disjoint")
    budget = _integer(
        tuning.get("trials_per_tunable_method"),
        "trials_per_tunable_method",
    )
    if budget < 1:
        raise FairTuningError("trials_per_tunable_method must be at least 1")
    require_finite = tuning.get("require_finite_updates", True)
    run_final = tuning.get("run_final_evaluation", True)
    if not isinstance(require_finite, bool) or not isinstance(run_final, bool):
        raise FairTuningError(
            "require_finite_updates and run_final_evaluation must be boolean"
        )

    objective = dict(OBJECTIVE_DEFAULTS)
    objective_payload = tuning.get("objective", {})
    if not isinstance(objective_payload, dict):
        raise FairTuningError("objective must be a JSON object")
    unknown_objective = sorted(set(objective_payload) - set(OBJECTIVE_DEFAULTS))
    if unknown_objective:
        raise FairTuningError(f"unknown objective weight: {unknown_objective[0]}")
    for key, value in objective_payload.items():
        objective[key] = _finite_float(value, key)
    if any(value < 0.0 for value in objective.values()):
        raise FairTuningError("objective weights must be non-negative")
    if sum(objective.values()) <= 0.0:
        raise FairTuningError("at least one objective weight must be positive")

    spaces = tuning.get("method_spaces")
    if not isinstance(spaces, dict) or set(spaces) != set(ALL_METHODS):
        raise FairTuningError("method_spaces must contain exactly the six methods")
    candidates: dict[str, tuple[dict[str, Any], ...]] = {}
    for method in ALL_METHODS:
        space = spaces[method]
        if not isinstance(space, dict):
            raise FairTuningError(f"method_spaces.{method} must be a JSON object")
        unknown = sorted(set(space) - METHOD_TUNABLE_PARAMETERS[method])
        if unknown:
            raise FairTuningError(
                f"{method} cannot tune shared or foreign parameter '{unknown[0]}'"
            )
        if METHOD_TUNABLE_PARAMETERS[method] and not space:
            raise FairTuningError(
                f"{method} must declare a non-empty defense-parameter search space"
            )
        method_candidates = tuple(_grid_candidates(space, method))
        if METHOD_TUNABLE_PARAMETERS[method] and len(method_candidates) != budget:
            raise FairTuningError(
                f"{method} has {len(method_candidates)} candidates; every tunable method "
                f"must have exactly trials_per_tunable_method={budget}"
            )
        for candidate in method_candidates:
            merged = dict(shared)
            merged.update(candidate)
            try:
                parse_args(parameters_to_argv(merged))
            except (ConfigError, SystemExit) as exc:
                raise FairTuningError(
                    f"invalid {method} candidate: {candidate}"
                ) from exc
        candidates[method] = method_candidates

    return FairTuningConfig(
        source=source,
        name=str(payload.get("name") or source.stem),
        description=str(payload.get("description") or ""),
        shared_parameters=dict(shared),
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        validation_seeds=validation_seeds,
        final_seeds=final_seeds,
        trials_per_tunable_method=budget,
        require_finite_updates=require_finite,
        run_final_evaluation=run_final,
        objective=objective,
        candidates=candidates,
    )


def _grid_candidates(space: dict[str, Any], method: str) -> Iterable[dict[str, Any]]:
    if not space:
        yield {}
        return
    names = sorted(space)
    values: list[list[Any]] = []
    for name in names:
        choices = space[name]
        if not isinstance(choices, list) or not choices:
            raise FairTuningError(
                f"method_spaces.{method}.{name} must be a non-empty JSON array"
            )
        if any(isinstance(value, (list, dict)) or value is None for value in choices):
            raise FairTuningError(
                f"method_spaces.{method}.{name} contains an invalid value"
            )
        if len({json.dumps(value, sort_keys=True) for value in choices}) != len(choices):
            raise FairTuningError(
                f"method_spaces.{method}.{name} contains duplicate values"
            )
        values.append(choices)
    for combination in product(*values):
        yield dict(zip(names, combination))


def make_validation_dataset(
    dataset: ImageDataset,
    *,
    fraction: float,
    seed: int,
) -> ImageDataset:
    """Create a deterministic stratified holdout from training data only."""

    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    validation_indices: list[np.ndarray] = []
    for label in range(dataset.num_classes):
        indices = np.flatnonzero(dataset.y_train == label)
        if len(indices) < 2:
            raise FairTuningError(
                f"class {label} needs at least two training samples for a holdout split"
            )
        shuffled = rng.permutation(indices)
        validation_count = min(len(indices) - 1, max(1, int(round(len(indices) * fraction))))
        validation_indices.append(shuffled[:validation_count])
        train_indices.append(shuffled[validation_count:])
    train = rng.permutation(np.concatenate(train_indices))
    validation = rng.permutation(np.concatenate(validation_indices))
    return ImageDataset(
        x_train=dataset.x_train[train].copy(),
        y_train=dataset.y_train[train].copy(),
        x_test=dataset.x_train[validation].copy(),
        y_test=dataset.y_train[validation].copy(),
        # Preserve the exact name because it selects the CIFAR architecture and
        # also remains the task identifier in the SM9-RRS protocol.
        name=dataset.name,
        input_shape=dataset.input_shape,
        num_classes=dataset.num_classes,
    )


def score_trial(
    method: str,
    candidate_id: str,
    parameters: dict[str, Any],
    results: list[ExperimentResult],
    *,
    objective: dict[str, float],
    require_finite_updates: bool,
) -> TrialScore:
    clean = [result for result in results if abs(result.config.malicious_ratio) < 1e-12]
    attacked = [result for result in results if result.config.malicious_ratio > 0.0]
    if not clean or not attacked:
        raise FairTuningError("each trial needs clean and attacked validation scenarios")
    clean_accuracy = fmean(result.final_accuracy for result in clean)
    robust_accuracy = fmean(result.final_accuracy for result in attacked)
    attack_rates = [
        result.records[-1].attack_target_success_rate
        for result in attacked
        if result.records and result.records[-1].attack_target_success_rate is not None
    ]
    attack_success = fmean(float(value) for value in attack_rates) if attack_rates else 0.0
    false_positive_rates = []
    for result in results:
        final_record = result.records[-1] if result.records else None
        honest = result.config.num_clients - malicious_client_count(
            result.config.num_clients,
            result.config.malicious_ratio,
        )
        false_positive_rates.append(
            (final_record.false_positive_revocations if final_record is not None else 0)
            / max(1, honest)
        )
    false_positive_rate = fmean(false_positive_rates)
    nonfinite_updates = sum(result.nonfinite_updates for result in results)
    valid = not require_finite_updates or nonfinite_updates == 0
    score = (
        objective["clean_accuracy_weight"] * clean_accuracy
        + objective["robust_accuracy_weight"] * robust_accuracy
        - objective["attack_success_weight"] * attack_success
        - objective["false_positive_weight"] * false_positive_rate
    )
    if not valid:
        score = float("-inf")
    return TrialScore(
        method=method,
        candidate_id=candidate_id,
        parameters=parameters,
        score=score,
        valid=valid,
        clean_accuracy=clean_accuracy,
        robust_accuracy=robust_accuracy,
        attack_success_rate=attack_success,
        false_positive_rate=false_positive_rate,
        nonfinite_updates=nonfinite_updates,
        result_count=len(results),
    )


def select_best_trials(trials: list[TrialScore]) -> dict[str, TrialScore]:
    selected: dict[str, TrialScore] = {}
    for method in ALL_METHODS:
        method_trials = [trial for trial in trials if trial.method == method and trial.valid]
        if not method_trials:
            raise FairTuningError(
                f"no finite valid candidate remains for {method}; fix the shared attack/training "
                "configuration instead of selecting an apparently good NaN run"
            )
        selected[method] = max(
            method_trials,
            key=lambda trial: (trial.score, tuple(sorted(trial.parameters.items()))),
        )
    return selected


def build_validation_tasks(
    spec: FairTuningConfig,
    base_configs: list[ExperimentConfig],
) -> list[TuningExperimentTask]:
    """Expand every candidate over the identical validation seeds/scenarios."""

    tasks: list[TuningExperimentTask] = []
    for method in ALL_METHODS:
        method_bases = [config for config in base_configs if config.method == method]
        for index, parameters in enumerate(spec.candidates[method], start=1):
            candidate_id = f"{method}-{index:03d}"
            for seed in spec.validation_seeds:
                for base in method_bases:
                    tasks.append(
                        TuningExperimentTask(
                            phase="validation",
                            candidate_id=candidate_id,
                            method=method,
                            config=replace(base, seed=seed, **parameters),
                        )
                    )
    return tasks


def build_final_tasks(
    spec: FairTuningConfig,
    base_configs: list[ExperimentConfig],
    selected: dict[str, TrialScore],
) -> list[TuningExperimentTask]:
    """Expand selected candidates over independent final-evaluation seeds."""

    tasks: list[TuningExperimentTask] = []
    for method in ALL_METHODS:
        method_bases = [config for config in base_configs if config.method == method]
        selected_trial = selected[method]
        for seed in spec.final_seeds:
            for base in method_bases:
                tasks.append(
                    TuningExperimentTask(
                        phase="final",
                        candidate_id=selected_trial.candidate_id,
                        method=method,
                        config=replace(
                            base,
                            seed=seed,
                            **selected_trial.parameters,
                        ),
                    )
                )
    return tasks


def prepare_tuning_tasks(
    dataset: ImageDataset,
    tasks: list[TuningExperimentTask],
    args: argparse.Namespace,
    *,
    requested_jobs: str | int | None = None,
    spread_cuda_devices: bool = True,
) -> tuple[list[TuningExperimentTask], int, str, int]:
    """Resolve accelerator, safe concurrency, SM9 threads, and CUDA placement."""

    if not tasks:
        return [], 1, describe_compute_backend(args.compute_backend, args.device), args.sm9_workers
    backend_description = describe_compute_backend(args.compute_backend, args.device)
    job_request = args.jobs if requested_jobs is None else requested_jobs
    jobs = resolve_parallel_jobs(
        job_request,
        dataset,
        [task.config for task in tasks],
        args,
    )
    jobs = min(jobs, len(tasks))
    sm9_workers = args.sm9_workers
    if args._sm9_workers_auto:
        sm9_workers = resolve_sm9_workers(
            "auto",
            max(task.config.num_clients for task in tasks),
            parallel_jobs=jobs,
        )
    configs = [replace(task.config, sm9_workers=sm9_workers) for task in tasks]
    if spread_cuda_devices:
        configs = assign_auto_cuda_devices(configs, backend_description, args.device)
    prepared = [
        replace(task, config=config)
        for task, config in zip(tasks, configs)
    ]
    return prepared, jobs, backend_description, sm9_workers


def execute_tuning_tasks(
    dataset: ImageDataset,
    tasks: list[TuningExperimentTask],
    *,
    jobs: int,
    backend_description: str,
    progress_enabled: bool,
    progress_mode: str,
) -> list[tuple[TuningExperimentTask, ExperimentResult]]:
    """Execute independent configs with the main runner's executor semantics."""

    progress = ProgressReporter(
        total=len(tasks),
        enabled=progress_enabled,
        stream=sys.stdout,
        mode=progress_mode,
    )
    completed: list[tuple[TuningExperimentTask, ExperimentResult]] = []
    executor_kind = parallel_executor_kind(backend_description, jobs)
    print(
        f"tuning_phase={tasks[0].phase if tasks else 'empty'} "
        f"configurations={len(tasks)} jobs={jobs} executor={executor_kind}",
        flush=True,
    )
    try:
        if jobs <= 1:
            for task in tasks:
                progress.start_config(task.config)
                result = run_measured_experiment(dataset, task.config)
                completed.append((task, result))
                progress.finish_config(task.config)
            return completed

        progress.start_parallel(jobs, len(tasks))
        if executor_kind == "thread":
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(_run_tuning_task_in_thread, dataset, task): task
                    for task in tasks
                }
                _consume_tuning_futures(futures, completed, progress)
            return completed

        try:
            with ProcessPoolExecutor(
                max_workers=jobs,
                initializer=_init_tuning_worker_dataset,
                initargs=(dataset,),
            ) as executor:
                futures = {
                    executor.submit(_run_tuning_task_in_worker, task): task
                    for task in tasks
                }
                _consume_tuning_futures(futures, completed, progress)
        except PermissionError as exc:
            print(
                f"tuning_process_pool_unavailable={exc}; falling back to thread pool",
                flush=True,
            )
            finished = {task for task, _result in completed}
            remaining = [task for task in tasks if task not in finished]
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(_run_tuning_task_in_thread, dataset, task): task
                    for task in remaining
                }
                _consume_tuning_futures(futures, completed, progress)
        return completed
    finally:
        progress.close()


def _init_tuning_worker_dataset(dataset: ImageDataset) -> None:
    global _TUNING_WORKER_DATASET
    _TUNING_WORKER_DATASET = dataset


def _run_tuning_task_in_worker(task: TuningExperimentTask) -> ExperimentResult:
    if _TUNING_WORKER_DATASET is None:
        raise RuntimeError("tuning worker dataset was not initialized")
    return run_measured_experiment(_TUNING_WORKER_DATASET, task.config)


def _run_tuning_task_in_thread(
    dataset: ImageDataset,
    task: TuningExperimentTask,
) -> ExperimentResult:
    return run_measured_experiment(dataset, task.config)


def _consume_tuning_futures(
    futures,
    completed: list[tuple[TuningExperimentTask, ExperimentResult]],
    progress: ProgressReporter,
) -> None:
    for future in as_completed(futures):
        task = futures[future]
        completed.append((task, future.result()))
        progress.finish_config(task.config)


def run_fair_tuning(spec: FairTuningConfig) -> dict[str, TrialScore]:
    args = parse_args(parameters_to_argv(spec.shared_parameters))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_image_dataset(
        args.dataset,
        args.data_dir,
        download=args.download,
        train_limit=args.train_samples,
        test_limit=args.test_samples,
        seed=args.seed,
    )
    validation_dataset = make_validation_dataset(
        dataset,
        fraction=spec.validation_fraction,
        seed=spec.split_seed,
    )
    base_configs = build_experiment_configs(args)
    validation_tasks = build_validation_tasks(spec, base_configs)
    (
        validation_tasks,
        validation_jobs,
        backend_description,
        validation_sm9_workers,
    ) = prepare_tuning_tasks(validation_dataset, validation_tasks, args)
    print_resource_plan(
        backend_description,
        validation_dataset,
        [task.config for task in validation_tasks],
        jobs=validation_jobs,
        sm9_workers=validation_sm9_workers,
        requested_jobs=args.jobs,
        cuda_devices=(
            available_cuda_devices()
            if backend_description == "torch:cuda"
            else ()
        ),
    )
    validation_executions = execute_tuning_tasks(
        validation_dataset,
        validation_tasks,
        jobs=validation_jobs,
        backend_description=backend_description,
        progress_enabled=not args.no_progress,
        progress_mode=args.progress_mode,
    )
    validation_executions.sort(key=lambda item: _tuning_task_sort_key(item[0]))
    results_by_candidate: dict[str, list[ExperimentResult]] = {}
    validation_rows: list[dict[str, Any]] = []
    for task, result in validation_executions:
        results_by_candidate.setdefault(task.candidate_id, []).append(result)
        validation_rows.append(
            {
                "candidate_id": task.candidate_id,
                **result.summary_dict(),
            }
        )

    trial_scores: list[TrialScore] = []
    for method in ALL_METHODS:
        for index, parameters in enumerate(spec.candidates[method], start=1):
            candidate_id = f"{method}-{index:03d}"
            trial = score_trial(
                method,
                candidate_id,
                parameters,
                results_by_candidate.get(candidate_id, []),
                objective=spec.objective,
                require_finite_updates=spec.require_finite_updates,
            )
            trial_scores.append(trial)
            print(
                f"tuning_candidate_complete={candidate_id} valid={trial.valid} "
                f"score={trial.score:.6f}",
                flush=True,
            )

    selected = select_best_trials(trial_scores)
    _write_csv(output_dir / "tuning_trials.csv", [trial.row() for trial in trial_scores])
    _write_csv(output_dir / "validation_results.csv", validation_rows)
    best_payload = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "dataset": args.dataset,
        "selection_data": "stratified holdout from training set only",
        "official_test_used_for_selection": False,
        "validation_fraction": spec.validation_fraction,
        "split_seed": spec.split_seed,
        "validation_seeds": list(spec.validation_seeds),
        "final_seeds": list(spec.final_seeds),
        "objective": spec.objective,
        "trials_per_tunable_method": spec.trials_per_tunable_method,
        "require_finite_updates": spec.require_finite_updates,
        "shared_parameters": spec.shared_parameters,
        "selected": {
            method: {
                "candidate_id": trial.candidate_id,
                "parameters": trial.parameters,
                "validation_score": trial.score,
            }
            for method, trial in selected.items()
        },
    }
    _write_json(output_dir / "best_parameters.json", best_payload)

    if spec.run_final_evaluation:
        # Candidate search is safe to parallelize because timing is not part of
        # the selection objective.  The final paper-facing timing run remains
        # serial and pinned to the default accelerator to avoid resource
        # contention or cross-GPU differences biasing method comparisons.
        final_tasks = build_final_tasks(spec, base_configs, selected)
        (
            final_tasks,
            final_jobs,
            final_backend_description,
            final_sm9_workers,
        ) = prepare_tuning_tasks(
            dataset,
            final_tasks,
            args,
            requested_jobs=1,
            spread_cuda_devices=False,
        )
        print("tuning_final_timing_policy=serial", flush=True)
        print_resource_plan(
            final_backend_description,
            dataset,
            [task.config for task in final_tasks],
            jobs=final_jobs,
            sm9_workers=final_sm9_workers,
            requested_jobs=1,
            cuda_devices=(
                available_cuda_devices()
                if final_backend_description == "torch:cuda"
                else ()
            ),
        )
        final_executions = execute_tuning_tasks(
            dataset,
            final_tasks,
            jobs=final_jobs,
            backend_description=final_backend_description,
            progress_enabled=not args.no_progress,
            progress_mode=args.progress_mode,
        )
        final_executions.sort(key=lambda item: _tuning_task_sort_key(item[0]))
        final_results = [result for _task, result in final_executions]
        final_dir = output_dir / "final_evaluation"
        final_dir.mkdir(parents=True, exist_ok=True)
        write_result_files(final_dir, final_results)
        _write_final_aggregate(final_dir / "aggregate.csv", final_results)
        if not args.no_visualizations:
            # Generate one dashboard per seed.  Combining repeated seeds in the
            # standard line chart would silently overwrite equal method/ratio
            # labels; aggregate.csv is authoritative for the paper table.
            for seed in spec.final_seeds:
                generate_visualizations(
                    [result for result in final_results if result.config.seed == seed],
                    final_dir / f"seed_{seed}",
                )
    _write_json(output_dir / "tuning_manifest.json", best_payload)
    return selected


def _tuning_task_sort_key(task: TuningExperimentTask) -> tuple[Any, ...]:
    config = task.config
    return (
        task.phase,
        task.candidate_id,
        config.seed,
        config.partition,
        config.dirichlet_alpha,
        config.num_clients,
        config.malicious_ratio,
        config.method,
    )


def _write_final_aggregate(path: Path, results: list[ExperimentResult]) -> None:
    grouped: dict[tuple[Any, ...], list[ExperimentResult]] = {}
    for result in results:
        key = (
            result.config.partition,
            result.config.dirichlet_alpha,
            result.config.num_clients,
            result.config.method,
            result.config.malicious_ratio,
        )
        grouped.setdefault(key, []).append(result)
    rows: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items(), key=lambda item: item[0]):
        accuracy = [item.final_accuracy for item in items]
        attack_success = [
            item.records[-1].attack_target_success_rate
            for item in items
            if item.records and item.records[-1].attack_target_success_rate is not None
        ]
        rows.append(
            {
                "partition": key[0],
                "dirichlet_alpha": key[1],
                "num_clients": key[2],
                "method": key[3],
                "malicious_ratio": key[4],
                "seeds": len(items),
                "final_accuracy_mean": fmean(accuracy),
                "final_accuracy_std": pstdev(accuracy) if len(accuracy) > 1 else 0.0,
                "attack_success_rate_mean": (
                    fmean(float(value) for value in attack_success)
                    if attack_success
                    else ""
                ),
                "runtime_end_to_end_mean": fmean(item.runtime_seconds for item in items),
                "runtime_without_crypto_mean": fmean(
                    item.runtime_without_crypto_seconds for item in items
                ),
                "crypto_wall_mean": fmean(
                    item.stage_timings.crypto_wall_seconds for item in items
                ),
                "nonfinite_updates_total": sum(item.nonfinite_updates for item in items),
            }
        )
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    temporary.replace(path)


def _seed_list(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise FairTuningError(f"{name} must be a non-empty JSON array")
    seeds = tuple(_integer(item, name) for item in value)
    if len(set(seeds)) != len(seeds):
        raise FairTuningError(f"{name} contains duplicate seeds")
    return seeds


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FairTuningError(f"{name} must be an integer")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FairTuningError(f"{name} must be numeric")
    number = float(value)
    if not np.isfinite(number):
        raise FairTuningError(f"{name} must be finite")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate fairness constraints and print candidate counts without training.",
    )
    args = parser.parse_args(argv)
    try:
        spec = load_fair_tuning_config(args.config)
    except FairTuningError as exc:
        parser.error(str(exc))
    print(f"tuning_config={spec.source}", flush=True)
    print(
        "candidate_counts="
        + json.dumps(
            {method: len(candidates) for method, candidates in spec.candidates.items()},
            sort_keys=True,
        ),
        flush=True,
    )
    print(
        f"validation_seeds={list(spec.validation_seeds)} final_seeds={list(spec.final_seeds)} "
        "official_test_used_for_selection=false",
        flush=True,
    )
    resolved_args = parse_args(parameters_to_argv(spec.shared_parameters))
    print(
        "tuning_execution_request="
        f"compute_backend={resolved_args.compute_backend} device={resolved_args.device} "
        f"jobs={resolved_args.jobs} "
        f"progress={'disabled' if resolved_args.no_progress else resolved_args.progress_mode} "
        "final_timing_jobs=1",
        flush=True,
    )
    if args.dry_run:
        return
    try:
        selected = run_fair_tuning(spec)
    except (ConfigError, FairTuningError) as exc:
        parser.error(str(exc))
    print(
        "selected_parameters="
        + json.dumps(
            {method: trial.parameters for method, trial in selected.items()},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])


__all__ = [
    "ALL_METHODS",
    "FairTuningConfig",
    "FairTuningError",
    "METHOD_TUNABLE_PARAMETERS",
    "TrialScore",
    "TuningExperimentTask",
    "build_final_tasks",
    "build_validation_tasks",
    "execute_tuning_tasks",
    "load_fair_tuning_config",
    "make_validation_dataset",
    "prepare_tuning_tasks",
    "run_fair_tuning",
    "score_trial",
    "select_best_trials",
]
