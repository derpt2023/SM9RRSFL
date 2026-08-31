import json
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sm9rrsfl.config_runner import parameters_to_argv
from sm9rrsfl.datasets import make_synthetic_mnist_like
from sm9rrsfl.fair_tuning import (
    ALL_METHODS,
    FairTuningError,
    FairTuningConfig,
    TuningExperimentTask,
    _learn_unified_objective_weights,
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
OBJECTIVE = {
    "clean_accuracy_weight": 0.25,
    "robust_accuracy_weight": 0.50,
    "attack_success_weight": 0.20,
    "honest_weight_loss_weight": 0.05,
}


def _set_four_candidate_tunable_spaces(payload):
    payload["tuning"]["trials_per_tunable_method"] = 4
    payload["tuning"]["method_spaces"]["vert"] = {
        "vert_history_window": [5, 10],
        "vert_predict_epochs": [3, 5],
    }
    payload["tuning"]["method_spaces"]["alignins"] = {
        "alignins_sparsity": [0.3],
        "alignins_tda_radius": [0.5, 1.0],
        "alignins_mpsa_radius": [0.5, 1.0],
    }


class FairTuningTest(unittest.TestCase):
    def test_example_enforces_all_methods_and_equal_tunable_budget(self):
        spec = load_fair_tuning_config(
            PROJECT_ROOT / "configs" / "fair_tuning.example.json"
        )

        self.assertEqual(set(spec.candidates), set(ALL_METHODS))
        self.assertTrue(spec.auto_ours)
        self.assertEqual(len(spec.candidates["sm9rrs"]), 0)
        self.assertEqual(len(spec.candidates["vert"]), 12)
        self.assertEqual(len(spec.candidates["alignins"]), 12)
        self.assertEqual(len(spec.candidates["fedavg"]), 1)
        self.assertEqual(spec.trials_per_tunable_method, 12)
        self.assertEqual(spec.formal_ratios, (0.0, 0.2, 0.4, 0.6, 0.8))
        self.assertEqual(spec.calibration_ratios, (0.0, 0.1, 0.3, 0.5, 0.7))
        self.assertEqual(spec.objective_mode, "learned_leave_one_attacked_ratio_out")
        self.assertTrue(set(spec.validation_seeds).isdisjoint(spec.final_seeds))
        self.assertAlmostEqual(spec.max_clean_accuracy_drop, 0.05)
        self.assertEqual(spec.final_jobs, "auto")
        self.assertEqual(
            {
                candidate["alignins_sparsity"]
                for candidate in spec.candidates["alignins"]
            },
            {0.1, 0.3, 0.5},
        )

    def test_v3_detector_hyperparameters_are_valid_ours_only_search_axes(self):
        payload = json.loads(
            (PROJECT_ROOT / "configs" / "fair_tuning.example.json").read_text(
                encoding="utf-8"
            )
        )
        payload["tuning"]["method_spaces"]["sm9rrs"] = {
            "detector_subspace_dim": [2],
            "detector_gap_threshold": [0.1],
            "detector_adjacent_threshold": [2.5, 3.0],
            "detector_anchor_threshold": [2.5, 3.0],
            "detector_drift_memory": [0.9],
            "detector_drift_allowance": [1.0],
            "detector_drift_threshold": [5.0],
            "suspicion_count_max": [3],
        }
        payload["shared_parameters"]["ours_parameter_mode"] = "fixed"
        _set_four_candidate_tunable_spaces(payload)
        payload["tuning"]["objective"] = {
            "clean_accuracy_weight": 0.25,
            "robust_accuracy_weight": 0.5,
            "attack_success_weight": 0.2,
            "honest_weight_loss_weight": 0.05,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v3-grid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            spec = load_fair_tuning_config(path)

        self.assertEqual(len(spec.candidates["sm9rrs"]), 4)
        self.assertTrue(
            all(
                candidate["detector_subspace_dim"] == 2
                and candidate["suspicion_count_max"] == 3
                for candidate in spec.candidates["sm9rrs"]
            )
        )
        self.assertEqual(
            parse_args(parameters_to_argv(spec.shared_parameters)).detector_decision_rule,
            "any",
        )

    def test_fixed_objective_requires_complete_normalized_weights(self):
        template = json.loads(
            (PROJECT_ROOT / "configs" / "fair_tuning.example.json").read_text(
                encoding="utf-8"
            )
        )
        valid = json.loads(json.dumps(template))
        valid["tuning"]["objective"] = dict(OBJECTIVE)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "valid-fixed-objective.json"
            path.write_text(json.dumps(valid), encoding="utf-8")
            spec = load_fair_tuning_config(path)
        self.assertEqual(spec.objective_mode, "fixed")
        self.assertEqual(spec.objective, OBJECTIVE)

        invalid_cases = (
            (
                {
                    "clean_accuracy_weight": 0.25,
                    "robust_accuracy_weight": 0.50,
                    "attack_success_weight": 0.20,
                },
                "must contain all four weights",
            ),
            (
                {
                    "clean_accuracy_weight": 0.25,
                    "robust_accuracy_weight": 0.75,
                    "attack_success_weight": 0.20,
                    "honest_weight_loss_weight": 0.05,
                },
                "must sum to 1",
            ),
        )
        for objective, error in invalid_cases:
            payload = json.loads(json.dumps(template))
            payload["tuning"]["objective"] = objective
            with self.subTest(objective=objective):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "invalid-fixed-objective.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(FairTuningError, error):
                        load_fair_tuning_config(path)

    def test_hard_constraint_defaults_and_ranges_are_validated(self):
        template = json.loads(
            (PROJECT_ROOT / "configs" / "fair_tuning.example.json").read_text(
                encoding="utf-8"
            )
        )
        template["tuning"].pop("max_clean_accuracy_drop", None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "defaults.json"
            path.write_text(json.dumps(template), encoding="utf-8")
            spec = load_fair_tuning_config(path)

        self.assertAlmostEqual(spec.max_clean_accuracy_drop, 0.05)

        invalid_cases = (
            ("max_clean_accuracy_drop", -0.01, "must be in \\[0, 1\\]"),
            ("max_clean_accuracy_drop", 1.01, "must be in \\[0, 1\\]"),
            ("final_jobs", 0, "must be 'auto' or a positive integer"),
        )
        for key, value, error in invalid_cases:
            payload = json.loads(json.dumps(template))
            payload["tuning"][key] = value
            with self.subTest(key=key, value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "invalid-hard-constraint.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(FairTuningError, error):
                        load_fair_tuning_config(path)

        legacy = json.loads(json.dumps(template))
        legacy["tuning"]["max_clean_false_positive_rate"] = 0.01
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy-hard-constraint.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            with self.assertRaisesRegex(FairTuningError, "unknown tuning key"):
                load_fair_tuning_config(path)

    def test_detector_decision_rule_is_fixed_shared_algorithm_semantics(self):
        payload = json.loads(
            (PROJECT_ROOT / "configs" / "fair_tuning.example.json").read_text(
                encoding="utf-8"
            )
        )
        payload["tuning"]["method_spaces"]["sm9rrs"] = {
            "detector_decision_rule": ["any"],
            "detector_anchor_threshold": [2.5, 3.0],
        }
        payload["shared_parameters"]["ours_parameter_mode"] = "fixed"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid-rule-grid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(FairTuningError, "cannot tune shared or foreign"):
                load_fair_tuning_config(path)

    def test_tuning_detector_window_requires_one_fixed_safe_attack_round(self):
        base_payload = json.loads(
            (PROJECT_ROOT / "configs" / "fair_tuning.example.json").read_text(
                encoding="utf-8"
            )
        )
        base_payload["shared_parameters"].pop("K", None)
        base_payload["shared_parameters"]["ours_parameter_mode"] = "fixed"
        _set_four_candidate_tunable_spaces(base_payload)
        base_payload["tuning"]["method_spaces"]["sm9rrs"] = {
            "detector_window": [7, 10],
            "detector_anchor_threshold": [2.5, 3.0],
        }

        invalid_cases = (
            (0, "requires an explicit shared attack_start_round"),
            (11, "must be at least max\\(detector_window\\) \\+ 2"),
        )
        for attack_start_round, error in invalid_cases:
            payload = json.loads(json.dumps(base_payload))
            payload["shared_parameters"]["attack_start_round"] = attack_start_round
            with self.subTest(attack_start_round=attack_start_round):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "invalid-window-grid.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(FairTuningError, error):
                        load_fair_tuning_config(path)

        base_payload["shared_parameters"]["attack_start_round"] = 12
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "valid-window-grid.json"
            path.write_text(json.dumps(base_payload), encoding="utf-8")
            spec = load_fair_tuning_config(path)

        args = parse_args(
            [
                "--methods",
                *ALL_METHODS,
                "--ratios",
                "0.0",
                "0.4",
                "--num-clients",
                "20",
                "--attack-start-round",
                "12",
            ]
        )
        tasks = build_validation_tasks(spec, build_experiment_configs(args))
        ours_tasks = [task for task in tasks if task.method == "sm9rrs"]
        self.assertEqual({task.config.detector_window for task in ours_tasks}, {7, 10})
        self.assertEqual({task.config.attack_start_round for task in ours_tasks}, {12})

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
            expected_candidates
            * len(spec.validation_seeds)
            * len(spec.calibration_ratios),
        )
        self.assertEqual({task.phase for task in tasks}, {"validation"})
        self.assertEqual(
            {task.config.malicious_ratio for task in tasks},
            set(spec.calibration_ratios),
        )

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
            objective=OBJECTIVE,
        )
        good = score_trial(
            "fedavg",
            "good",
            {},
            [_result(0.0, 0.7, 0), _result(0.4, 0.6, 0)],
            objective=OBJECTIVE,
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
                        objective=OBJECTIVE,
                    )
                )

        selected = select_best_trials(trials)

        self.assertFalse(bad.valid)
        self.assertEqual(selected["fedavg"].candidate_id, "good")

    def test_honest_weight_loss_replaces_permanent_false_positive_score(self):
        trial = score_trial(
            "alignins",
            "weighted-harm",
            {},
            [
                _result(0.0, 0.8, 0, "alignins", honest_weight_loss=0.2),
                _result(0.4, 0.7, 0, "alignins", honest_weight_loss=0.4),
            ],
            objective=OBJECTIVE,
        )

        self.assertTrue(trial.valid)
        self.assertAlmostEqual(trial.honest_weight_loss, 0.3)
        self.assertIn("honest_weight_loss", trial.row())
        self.assertNotIn("false_positive_rate", trial.row())

    def test_matched_fedavg_clean_accuracy_drop_is_a_gate(self):
        clean = _result(0.0, 0.70, 0, "sm9rrs")
        attacked = _result(0.4, 0.65, 0, "sm9rrs")
        trial = score_trial(
            "sm9rrs",
            "clean-drop",
            {},
            [clean, attacked],
            objective=OBJECTIVE,
            clean_accuracy_reference={
                (
                    clean.config.partition,
                    clean.config.dirichlet_alpha,
                    clean.config.num_clients,
                    clean.config.seed,
                ): 0.80
            },
            max_clean_accuracy_drop=0.05,
        )

        self.assertFalse(trial.valid)
        self.assertAlmostEqual(trial.worst_clean_accuracy_drop, 0.10)

    def test_clean_accuracy_gate_distinguishes_dirichlet_alphas(self):
        clean = _result(
            0.0,
            0.70,
            0,
            "sm9rrs",
            config=ExperimentConfig(
                method="sm9rrs",
                malicious_ratio=0.0,
                partition="dirichlet",
                dirichlet_alpha=0.3,
                num_clients=10,
                seed=7,
            ),
        )
        attacked = _result(
            0.4,
            0.65,
            0,
            "sm9rrs",
            config=replace(clean.config, malicious_ratio=0.4),
        )
        trial = score_trial(
            "sm9rrs",
            "alpha-specific-clean-control",
            {},
            [clean, attacked],
            objective=OBJECTIVE,
            clean_accuracy_reference={
                ("dirichlet", 0.5, 10, 7): 0.70,
            },
        )

        self.assertFalse(trial.valid)
        self.assertAlmostEqual(trial.worst_clean_accuracy_drop, 1.0)

    def test_high_asr_is_scored_instead_of_being_a_hard_constraint(self):
        trial = score_trial(
            "sm9rrs",
            "high-asr",
            {},
            [
                _result(0.0, 0.8, 0, "sm9rrs"),
                _result(0.4, 0.7, 0, "sm9rrs", attack_success=0.95),
            ],
            objective=OBJECTIVE,
        )

        self.assertTrue(trial.valid)
        self.assertAlmostEqual(trial.attack_success_rate, 0.95)

    def test_missing_attack_success_is_not_silently_treated_as_zero(self):
        missing = score_trial(
            "sm9rrs",
            "missing-asr",
            {},
            [
                _result(0.0, 0.8, 0, "sm9rrs"),
                _result(0.4, 0.7, 0, "sm9rrs", attack_success=None),
            ],
            objective=OBJECTIVE,
        )

        self.assertFalse(missing.valid)
        self.assertEqual(missing.attack_success_rate, 1.0)

    def test_nonfinite_accuracy_or_attack_metric_invalidates_trial(self):
        bad_accuracy = replace(
            _result(0.0, 0.8, 0, "sm9rrs"),
            final_accuracy=float("nan"),
        )
        invalid_accuracy = score_trial(
            "sm9rrs",
            "nan-accuracy",
            {},
            [bad_accuracy, _result(0.4, 0.7, 0, "sm9rrs")],
            objective=OBJECTIVE,
        )
        invalid_asr = score_trial(
            "sm9rrs",
            "nan-asr",
            {},
            [
                _result(0.0, 0.8, 0, "sm9rrs"),
                _result(
                    0.4,
                    0.7,
                    0,
                    "sm9rrs",
                    attack_success=float("nan"),
                ),
            ],
            objective=OBJECTIVE,
        )

        self.assertFalse(invalid_accuracy.valid)
        self.assertEqual(invalid_accuracy.score, float("-inf"))
        self.assertFalse(invalid_asr.valid)
        self.assertEqual(invalid_asr.attack_success_rate, 1.0)

    def test_completion_and_nonfinite_thresholds_are_method_neutral(self):
        incomplete_attack = _result(
            0.4,
            0.7,
            0,
            "sm9rrs",
            config=ExperimentConfig(
                method="sm9rrs",
                malicious_ratio=0.4,
                num_clients=10,
                rounds=2,
                attack_start_round=1,
            ),
            stopped_round=1,
        )
        trial = score_trial(
            "sm9rrs",
            "incomplete",
            {},
            [_result(0.0, 0.8, 0, "sm9rrs"), incomplete_attack],
            objective=OBJECTIVE,
        )

        self.assertFalse(trial.valid)
        self.assertFalse(trial.all_runs_completed)
        self.assertFalse(trial.row()["all_runs_completed"])

        relaxed = score_trial(
            "sm9rrs",
            "incomplete-relaxed",
            {},
            [_result(0.0, 0.8, 0, "sm9rrs"), incomplete_attack],
            objective=OBJECTIVE,
            min_round_completion_rate=0.5,
        )
        self.assertTrue(relaxed.valid)

    def test_leave_one_ratio_out_does_not_prefilter_with_held_out_metrics(self):
        candidates = {
            "sm9rrs": ({"detector_subspace_dim": 1}, {"detector_subspace_dim": 2}),
            "vert": ({"vert_history_window": 5},),
            "alignins": ({"alignins_sparsity": 0.3},),
            "krum": ({},),
            "ding13": ({},),
            "fedavg": ({},),
        }
        spec = _minimal_spec(candidates)
        results = {
            "fedavg-001": [
                _result(0.0, 0.80, 0, "fedavg"),
                _result(0.1, 0.70, 0, "fedavg"),
                _result(0.3, 0.70, 0, "fedavg"),
            ],
            "sm9rrs-001": [
                _result(0.0, 0.80, 0, "sm9rrs"),
                _result(0.1, 0.10, 0, "sm9rrs", attack_success=1.0),
                _result(0.3, 0.95, 0, "sm9rrs", attack_success=0.0),
            ],
            "sm9rrs-002": [
                _result(0.0, 0.79, 0, "sm9rrs"),
                _result(0.1, 0.70, 0, "sm9rrs", attack_success=0.2),
                _result(0.3, 0.70, 0, "sm9rrs", attack_success=0.2),
            ],
        }
        for method in ("vert", "alignins"):
            results[f"{method}-001"] = [
                _result(0.0, 0.79, 0, method),
                _result(0.1, 0.70, 0, method),
                _result(0.3, 0.70, 0, method),
            ]

        with mock.patch(
            "sm9rrsfl.fair_tuning.objective_weight_grid",
            return_value=(OBJECTIVE,),
        ):
            _weights, learning = _learn_unified_objective_weights(spec, results)

        fold = next(
            item for item in learning["folds"] if item["held_out_ratio"] == 0.1
        )
        ours = next(item for item in fold["selected"] if item["method"] == "sm9rrs")
        self.assertEqual(ours["candidate_id"], "sm9rrs-001")
        self.assertTrue(ours["held_out_valid"])
        self.assertEqual(ours["attack_success_rate"], 1.0)


def _minimal_spec(candidates):
    return FairTuningConfig(
        source=Path("test-fair-tuning.json"),
        name="test",
        description="",
        shared_parameters={},
        validation_fraction=0.1,
        split_seed=1,
        validation_seeds=(1,),
        final_seeds=(2,),
        trials_per_tunable_method=max(
            len(candidates[method]) for method in ("sm9rrs", "vert", "alignins")
        ),
        run_final_evaluation=False,
        final_jobs="auto",
        max_clean_accuracy_drop=0.05,
        min_round_completion_rate=1.0,
        max_nonfinite_updates=0,
        objective=dict(OBJECTIVE),
        objective_mode="learned_leave_one_attacked_ratio_out",
        formal_ratios=(0.0, 0.2, 0.4),
        calibration_ratios=(0.0, 0.1, 0.3),
        ratio_schedule=None,
        auto_ours=False,
        preinvalid_candidates=(),
        candidates=candidates,
    )


def _result(
    ratio,
    accuracy,
    nonfinite,
    method="fedavg",
    *,
    accepted_updates=10,
    false_positive_revocations=0,
    attack_success="default",
    honest_weight_loss=0.0,
    malicious_weight_mass=None,
    stopped_round=None,
    config=None,
):
    accepted_by_round = (
        list(accepted_updates)
        if isinstance(accepted_updates, (list, tuple))
        else [accepted_updates]
    )
    config = config or ExperimentConfig(
        method=method,
        malicious_ratio=ratio,
        num_clients=10,
        rounds=len(accepted_by_round),
        attack_start_round=1,
    )
    attack_success_value = (
        (0.1 if ratio > 0.0 else 0.0)
        if attack_success == "default"
        else attack_success
    )
    malicious_weight_mass_value = (
        float(ratio)
        if malicious_weight_mass is None
        else float(malicious_weight_mass)
    )
    records = [
        RoundRecord(
            method=method,
            malicious_ratio=ratio,
            round=round_id,
            accuracy=accuracy,
            error=1.0 - accuracy,
            accepted_updates=accepted,
            rejected_updates=max(0, config.num_clients - accepted),
            blacklisted_clients=false_positive_revocations,
            true_positive_revocations=0,
            false_positive_revocations=false_positive_revocations,
            krum_selected_client="",
            attack_target_success_rate=attack_success_value,
            nonfinite_updates=nonfinite,
            attack_active=ratio > 0.0,
            honest_weight_loss=honest_weight_loss,
            malicious_weight_mass=malicious_weight_mass_value,
        )
        for round_id, accepted in enumerate(accepted_by_round, start=1)
    ]
    return ExperimentResult(
        config=config,
        records=records,
        final_accuracy=accuracy,
        final_error=1.0 - accuracy,
        stopped_round=(len(records) if stopped_round is None else stopped_round),
        malicious_clients=tuple(),
        blacklisted_clients=tuple(),
        nonfinite_updates=nonfinite,
    )


if __name__ == "__main__":
    unittest.main()
