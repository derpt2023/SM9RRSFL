import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sm9rrsfl.datasets import make_synthetic_mnist_like
from sm9rrsfl.fair_tuning import (
    ALL_METHODS,
    FairTuningError,
    TuningExperimentTask,
    build_validation_tasks,
    execute_resumable_tuning_phase,
    execute_tuning_tasks,
    load_fair_tuning_config,
    make_validation_dataset,
    prepare_tuning_tasks,
    score_trial,
    select_best_trials,
)
from sm9rrsfl.fl import ExperimentConfig, ExperimentResult, RoundRecord
from sm9rrsfl.experiments import build_experiment_configs, parse_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FairTuningTest(unittest.TestCase):
    def test_example_enforces_all_methods_and_equal_tunable_budget(self):
        spec = load_fair_tuning_config(
            PROJECT_ROOT / "configs" / "fair_tuning.example.json"
        )

        self.assertEqual(set(spec.candidates), set(ALL_METHODS))
        self.assertEqual(len(spec.candidates["sm9rrs"]), 4)
        self.assertEqual(len(spec.candidates["vert"]), 4)
        self.assertEqual(len(spec.candidates["fedredefense"]), 4)
        self.assertEqual(len(spec.candidates["fedavg"]), 1)
        self.assertTrue(set(spec.validation_seeds).isdisjoint(spec.final_seeds))
        self.assertTrue(spec.require_clean_acceptance)
        self.assertEqual(
            {
                candidate["fedre_teacher_lr"]
                for candidate in spec.candidates["fedredefense"]
            },
            {5.0, 6.0},
        )

    def test_validation_tasks_expand_every_candidate_seed_and_scenario(self):
        spec = load_fair_tuning_config(
            PROJECT_ROOT / "configs" / "fair_tuning.example.json"
        )
        args = parse_args([
            "--methods",
            *ALL_METHODS,
            "--ratios",
            "0.0",
            "0.4",
            "--partitions",
            "iid",
            "--num-clients",
            "20",
        ])
        tasks = build_validation_tasks(spec, build_experiment_configs(args))

        expected_candidates = sum(len(items) for items in spec.candidates.values())
        self.assertEqual(
            len(tasks),
            expected_candidates * len(spec.validation_seeds) * 2,
        )
        self.assertEqual({task.phase for task in tasks}, {"validation"})

    def test_task_preparation_reuses_accelerator_resource_planning(self):
        dataset = make_synthetic_mnist_like(train_samples=40, test_samples=10, seed=5)
        args = parse_args(
            [
                "--compute-backend",
                "auto",
                "--device",
                "auto",
                "--jobs",
                "4",
                "--sm9-workers",
                "auto",
            ]
        )
        tasks = [
            TuningExperimentTask(
                phase="validation",
                candidate_id=f"sm9rrs-{index:03d}",
                method="sm9rrs",
                config=ExperimentConfig(
                    method="sm9rrs",
                    compute_backend="auto",
                    device="auto",
                    seed=index,
                ),
            )
            for index in (1, 2)
        ]
        with (
            mock.patch(
                "sm9rrsfl.fair_tuning.describe_compute_backend",
                return_value="torch:cuda",
            ),
            mock.patch(
                "sm9rrsfl.fair_tuning.resolve_parallel_jobs",
                return_value=2,
            ),
            mock.patch(
                "sm9rrsfl.fair_tuning.resolve_sm9_workers",
                return_value=1,
            ),
            mock.patch(
                "sm9rrsfl.fair_tuning.cuda_devices_with_capacity",
                return_value=("cuda:0", "cuda:1"),
            ),
        ):
            prepared, jobs, backend, sm9_workers = prepare_tuning_tasks(
                dataset,
                tasks,
                args,
            )

        self.assertEqual(jobs, 2)
        self.assertEqual(backend, "torch:cuda")
        self.assertEqual(sm9_workers, 1)
        self.assertEqual(
            [task.config.device for task in prepared],
            ["cuda:0", "cuda:1"],
        )

    def test_task_execution_reports_configuration_progress(self):
        dataset = make_synthetic_mnist_like(train_samples=40, test_samples=10, seed=6)
        tasks = [
            TuningExperimentTask(
                phase="validation",
                candidate_id=f"fedavg-{index:03d}",
                method="fedavg",
                config=ExperimentConfig(method="fedavg", seed=index),
            )
            for index in (1, 2)
        ]
        with (
            mock.patch(
                "sm9rrsfl.fair_tuning.run_measured_experiment",
                side_effect=lambda _dataset, config, **_kwargs: _result(
                    config.malicious_ratio,
                    0.5,
                    0,
                    config.method,
                ),
            ),
            mock.patch("sys.stdout", new=io.StringIO()) as output,
        ):
            completed = execute_tuning_tasks(
                dataset,
                tasks,
                jobs=1,
                backend_description="numpy",
                progress_enabled=True,
                progress_mode="log",
            )

        self.assertEqual(len(completed), 2)
        self.assertIn("2/2", output.getvalue())
        self.assertIn("eta=", output.getvalue())

    def test_accelerator_tasks_use_parallel_thread_queue(self):
        dataset = make_synthetic_mnist_like(train_samples=40, test_samples=10, seed=7)
        tasks = [
            TuningExperimentTask(
                phase="validation",
                candidate_id=f"vert-{index:03d}",
                method="vert",
                config=ExperimentConfig(method="vert", seed=index),
            )
            for index in (1, 2)
        ]
        with (
            mock.patch(
                "sm9rrsfl.fair_tuning.run_measured_experiment",
                side_effect=lambda _dataset, config, **_kwargs: _result(
                    config.malicious_ratio,
                    0.5,
                    0,
                    config.method,
                ),
            ) as run,
            mock.patch("sys.stdout", new=io.StringIO()) as output,
        ):
            completed = execute_tuning_tasks(
                dataset,
                tasks,
                jobs=2,
                backend_description="torch:mps",
                progress_enabled=True,
                progress_mode="log",
            )

        self.assertEqual(len(completed), 2)
        self.assertEqual(run.call_count, 2)
        self.assertIn("executor=thread", output.getvalue())

    def test_resumable_phase_skips_atomically_committed_tasks(self):
        dataset = make_synthetic_mnist_like(train_samples=40, test_samples=10, seed=8)
        args = parse_args(["--methods", *ALL_METHODS, "--no-early-stop"])
        tasks = [
            TuningExperimentTask(
                phase="validation",
                candidate_id=f"fedavg-{index:03d}",
                method="fedavg",
                config=ExperimentConfig(
                    method="fedavg",
                    num_clients=10,
                    rounds=1,
                    early_stop=False,
                    seed=index,
                ),
            )
            for index in (1, 2)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            snapshots = []
            with mock.patch(
                "sm9rrsfl.fair_tuning.run_measured_experiment",
                side_effect=lambda _dataset, config, **_kwargs: _result(
                    config.malicious_ratio,
                    0.5,
                    0,
                    config.method,
                    config=config,
                ),
            ) as first_run:
                completed, fingerprint = execute_resumable_tuning_phase(
                    dataset,
                    tasks,
                    args,
                    output_dir=output_dir,
                    jobs=1,
                    backend_description="numpy",
                    progress_enabled=False,
                    progress_mode="log",
                    on_snapshot=lambda executions, _fingerprint, status: snapshots.append(
                        (len(executions), status)
                    ),
                )
            self.assertEqual(first_run.call_count, 2)
            self.assertEqual(len(completed), 2)
            self.assertTrue(fingerprint)
            self.assertEqual(
                snapshots,
                [(0, "running"), (1, "running"), (2, "running"), (2, "complete")],
            )

            with mock.patch(
                "sm9rrsfl.fair_tuning.run_measured_experiment",
                side_effect=AssertionError("completed tuning tasks must be skipped"),
            ) as resumed_run:
                resumed, resumed_fingerprint = execute_resumable_tuning_phase(
                    dataset,
                    tasks,
                    args,
                    output_dir=output_dir,
                    jobs=1,
                    backend_description="numpy",
                    progress_enabled=False,
                    progress_mode="log",
                )
            self.assertEqual(resumed_run.call_count, 0)
            self.assertEqual(len(resumed), 2)
            self.assertEqual(resumed_fingerprint, fingerprint)

    def test_shared_training_parameter_cannot_be_tuned_per_method(self):
        payload = json.loads(
            (PROJECT_ROOT / "configs" / "fair_tuning.example.json").read_text(
                encoding="utf-8"
            )
        )
        payload["tuning"]["method_spaces"]["sm9rrs"] = {"lr": [0.01, 0.02]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(FairTuningError, "cannot tune shared"):
                load_fair_tuning_config(path)

    def test_validation_split_uses_training_samples_and_leaves_test_unused(self):
        dataset = make_synthetic_mnist_like(
            train_samples=250,
            test_samples=40,
            seed=19,
        )
        validation = make_validation_dataset(dataset, fraction=0.2, seed=91)

        self.assertEqual(validation.name, dataset.name)
        self.assertEqual(
            len(validation.y_train) + len(validation.y_test),
            len(dataset.y_train),
        )
        self.assertNotEqual(len(validation.y_test), len(dataset.y_test))

    def test_nonfinite_trial_is_never_selected_even_with_high_accuracy(self):
        bad = score_trial(
            "fedavg",
            "bad",
            {},
            [_result(0.0, 0.99, 1), _result(0.4, 0.99, 1)],
            objective={
                "clean_accuracy_weight": 0.25,
                "robust_accuracy_weight": 0.5,
                "attack_success_weight": 0.2,
                "false_positive_weight": 0.05,
            },
            require_finite_updates=True,
        )
        good = score_trial(
            "fedavg",
            "good",
            {},
            [_result(0.0, 0.7, 0), _result(0.4, 0.6, 0)],
            objective={
                "clean_accuracy_weight": 0.25,
                "robust_accuracy_weight": 0.5,
                "attack_success_weight": 0.2,
                "false_positive_weight": 0.05,
            },
            require_finite_updates=True,
        )
        trials = []
        for method in ALL_METHODS:
            if method == "fedavg":
                trials.extend([bad, good])
            else:
                trials.append(
                    score_trial(
                        method,
                        f"{method}-only",
                        {},
                        [_result(0.0, 0.7, 0, method), _result(0.4, 0.6, 0, method)],
                        objective={
                            "clean_accuracy_weight": 0.25,
                            "robust_accuracy_weight": 0.5,
                            "attack_success_weight": 0.2,
                            "false_positive_weight": 0.05,
                        },
                        require_finite_updates=True,
                    )
                )

        selected = select_best_trials(trials)

        self.assertFalse(bad.valid)
        self.assertEqual(selected["fedavg"].candidate_id, "good")

    def test_candidate_rejecting_every_clean_client_is_invalid(self):
        trial = score_trial(
            "fedredefense",
            "collapsed",
            {"fedre_teacher_lr": 0.1},
            [
                _result(0.0, 0.1, 0, "fedredefense", accepted_updates=0),
                _result(0.4, 0.1, 0, "fedredefense", accepted_updates=0),
            ],
            objective={
                "clean_accuracy_weight": 0.25,
                "robust_accuracy_weight": 0.5,
                "attack_success_weight": 0.2,
                "false_positive_weight": 0.05,
            },
            require_finite_updates=True,
            require_clean_acceptance=True,
        )

        self.assertFalse(trial.valid)
        self.assertEqual(trial.clean_acceptance_rate, 0.0)


def _result(
    ratio,
    accuracy,
    nonfinite,
    method="fedavg",
    *,
    accepted_updates=10,
    config=None,
):
    config = config or ExperimentConfig(
        method=method,
        malicious_ratio=ratio,
        num_clients=10,
    )
    record = RoundRecord(
        method,
        ratio,
        1,
        accuracy,
        1.0 - accuracy,
        accepted_updates,
        nonfinite,
        0,
        0,
        0,
        "",
        attack_target_success_rate=0.1 if ratio > 0.0 else 0.0,
        nonfinite_updates=nonfinite,
    )
    return ExperimentResult(
        config=config,
        records=[record],
        final_accuracy=accuracy,
        final_error=1.0 - accuracy,
        stopped_round=1,
        malicious_clients=tuple(),
        blacklisted_clients=tuple(),
        nonfinite_updates=nonfinite,
    )


if __name__ == "__main__":
    unittest.main()
