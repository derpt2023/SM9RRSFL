import argparse
import csv
import io
import json
import os
import tempfile
from threading import Event
import unittest
from pathlib import Path
from unittest import mock

from sm9rrsfl.experiments import (
    ProgressReporter,
    _format_duration,
    assign_auto_cuda_devices,
    build_experiment_configs,
    confirm_matching_checkpoints,
    default_output_dir,
    finalize_config_checkpoint,
    load_archived_results,
    load_completed_results_snapshot,
    parse_args,
    parallel_executor_kind,
    read_results,
    resolve_sm9_workers,
    resolve_parallel_jobs,
    resolve_output_dir,
    run_measured_experiment,
    write_result_files,
)
from sm9rrsfl.datasets import make_synthetic_mnist_like
from sm9rrsfl.fl import (
    ExperimentConfig,
    experiment_config_error,
    malicious_client_count,
    run_experiment,
)


class ExperimentOutputDirTest(unittest.TestCase):
    def test_matching_checkpoint_prompts_for_resume_or_restart(self):
        dataset = make_synthetic_mnist_like(train_samples=40, test_samples=10, seed=18)
        config = ExperimentConfig(
            method="fedavg",
            num_clients=4,
            rounds=1,
            early_stop=False,
            seed=18,
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / ".checkpoints"
            run_measured_experiment(
                dataset,
                config,
                checkpoint_dir=checkpoint_dir,
                run_fingerprint="prompt-test",
                retain_success_checkpoint=True,
            )
            checkpoint = next(checkpoint_dir.glob("*.pickle"))

            with mock.patch("sys.stdout", new=io.StringIO()) as output:
                resume_counts = confirm_matching_checkpoints(
                    checkpoint_dir,
                    [config],
                    "prompt-test",
                    input_func=lambda _prompt: "Y",
                    interactive=True,
                )
            self.assertEqual(resume_counts, {"found": 1, "resumed": 1, "restarted": 0})
            self.assertTrue(checkpoint.exists())
            self.assertIn("已选择 Y", output.getvalue())

            mismatch_counts = confirm_matching_checkpoints(
                checkpoint_dir,
                [config],
                "different-fingerprint",
                interactive=False,
            )
            self.assertEqual(mismatch_counts["found"], 0)

            with mock.patch("sys.stdout", new=io.StringIO()) as output:
                restart_counts = confirm_matching_checkpoints(
                    checkpoint_dir,
                    [config],
                    "prompt-test",
                    input_func=lambda _prompt: "N",
                    interactive=True,
                )
            self.assertEqual(restart_counts, {"found": 1, "resumed": 0, "restarted": 1})
            self.assertFalse(checkpoint.exists())
            self.assertEqual(len(list((checkpoint_dir / "discarded").glob("*.pickle"))), 1)
            self.assertIn("已选择 N", output.getvalue())

    def test_matching_archived_run_is_recovered_by_fingerprint(self):
        dataset = make_synthetic_mnist_like(train_samples=20, test_samples=10, seed=17)
        config = ExperimentConfig(
            method="fedavg",
            num_clients=2,
            rounds=1,
            early_stop=False,
            seed=17,
        )
        result = run_experiment(dataset, config)
        fingerprint = "abcdef123456" + "0" * 52
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            archived_dir = output_dir / ".stale" / fingerprint[:12]
            archived_dir.mkdir(parents=True)
            write_result_files(archived_dir, [result])

            recovered = load_archived_results(output_dir, fingerprint, [config])

        self.assertIsNotNone(recovered)
        recovered_results, recovered_path = recovered
        self.assertEqual(len(recovered_results), 1)
        self.assertEqual(recovered_results[0].config, config)
        self.assertEqual(recovered_path.name, fingerprint[:12])

    def test_terminal_checkpoint_replays_completed_worker_before_parent_commit(self):
        dataset = make_synthetic_mnist_like(train_samples=40, test_samples=10, seed=16)
        config = ExperimentConfig(
            method="fedavg",
            malicious_ratio=0.0,
            num_clients=4,
            rounds=1,
            early_stop=False,
            seed=16,
        )
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / ".checkpoints"
            first = run_measured_experiment(
                dataset,
                config,
                checkpoint_dir=checkpoint_dir,
                run_fingerprint="two-phase-test",
                retain_success_checkpoint=True,
            )
            self.assertEqual(len(list(checkpoint_dir.glob("*.pickle"))), 1)

            with mock.patch(
                "sm9rrsfl.experiments.run_experiment",
                side_effect=AssertionError("terminal result should be replayed"),
            ):
                replayed = run_measured_experiment(
                    dataset,
                    config,
                    checkpoint_dir=checkpoint_dir,
                    run_fingerprint="two-phase-test",
                    retain_success_checkpoint=True,
                )
            finalize_config_checkpoint(checkpoint_dir, config, "two-phase-test")

            self.assertEqual(first.records, replayed.records)
            self.assertEqual(len(list(checkpoint_dir.glob("*.pickle"))), 0)

    def test_invalid_numeric_arguments_fail_during_preflight(self):
        invalid_commands = (
            ["--ratios", "1.0"],
            ["--client-counts", "0"],
            ["--rounds", "0"],
            ["--batch-size", "0"],
            ["--lr", "0"],
            ["--dirichlet-alpha", "0"],
            ["--attack-scale", "0"],
            ["--attack-boost", "0"],
            ["--attack-epochs", "0"],
            ["--attack-stealth-steps", "0"],
            ["--attack-distance-weight", "-1"],
            ["--attack-source-label", "-1"],
            ["--attack-target-label", "10"],
            ["--attack-target-count", "0"],
            [
                "--attack",
                "alternating_minimization",
                "--attack-source-label",
                "5",
                "--attack-target-label",
                "5",
            ],
            ["--attack-start-round", "-1"],
            ["--K", "1"],
            ["--C_tol", "0"],
            ["--vert-history-window", "1"],
            ["--vert-projection-dim", "1"],
            ["--vert-predict-epochs", "0"],
            ["--vert-predict-lr", "0"],
            ["--vert-top-k", "-1"],
            ["--vert-use-ratio-prior", "--vert-top-k", "1"],
            ["--fedre-threshold", "0"],
            ["--fedre-initial-iterations", "0"],
            ["--fedre-max-iterations", "0"],
            ["--fedre-synthetic-steps", "0"],
            ["--fedre-images-per-class", "0"],
            ["--fedre-image-lr", "0"],
            ["--fedre-label-lr", "0"],
            ["--fedre-teacher-lr", "0"],
            ["--fedre-teacher-lr-lr", "0"],
        )
        for command in invalid_commands:
            with self.subTest(command=command):
                with mock.patch("sys.stderr", new=io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(command)

    def test_krum_validation_uses_general_formula_across_boundaries(self):
        ratios = (0.0, 0.25, 0.49, 0.5, 0.6, 0.8, 0.95)
        for clients in range(1, 31):
            for ratio in ratios:
                config = ExperimentConfig(
                    method="krum",
                    num_clients=clients,
                    malicious_ratio=ratio,
                )
                malicious = malicious_client_count(clients, ratio)
                expected_invalid = clients < 3 or clients - malicious - 2 < 1
                self.assertEqual(
                    experiment_config_error(config) is not None,
                    expected_invalid,
                    msg=f"clients={clients}, ratio={ratio}, f={malicious}",
                )

    def test_invalid_krum_grid_points_are_detected_without_changing_manifest_grid(self):
        args = parse_args(
            [
                "--methods",
                "sm9rrs",
                "krum",
                "ding13",
                "fedavg",
                "--ratios",
                "0",
                "0.1",
                "0.2",
                "0.4",
                "0.6",
                "0.8",
                "--partitions",
                "iid",
                "dirichlet",
                "--client-counts",
                "10",
                "20",
                "50",
            ]
        )
        configs = build_experiment_configs(args)
        invalid = [config for config in configs if experiment_config_error(config)]

        self.assertEqual(len(configs), 144)
        self.assertEqual(len(invalid), 2)
        self.assertTrue(all(config.method == "krum" for config in invalid))
        self.assertTrue(all(config.num_clients == 10 for config in invalid))
        self.assertTrue(all(config.malicious_ratio == 0.8 for config in invalid))

    def test_sm9_mnist_default_keeps_main_output_directory(self):
        self.assertEqual(default_output_dir("mnist", "sm9"), Path("outputs/mnist"))

    def test_simulated_mnist_default_uses_separate_output_directory(self):
        self.assertEqual(
            default_output_dir("mnist", "simulated"),
            Path("outputs/mnist_simulated"),
        )

    def test_cifar10_default_uses_dataset_specific_directory(self):
        self.assertEqual(default_output_dir("cifar10", "sm9"), Path("outputs/cifar10"))
        self.assertEqual(
            default_output_dir("cifar10", "simulated"),
            Path("outputs/cifar10_simulated"),
        )

    def test_explicit_output_directory_overrides_mode_default(self):
        args = argparse.Namespace(
            dataset="mnist",
            crypto_mode="simulated",
            output_dir="outputs/custom",
        )

        self.assertEqual(resolve_output_dir(args), Path("outputs/custom"))

    def test_cifar10_clean_baseline_preset(self):
        args = parse_args(["--cifar10-clean-baseline", "--rounds", "5"])

        self.assertEqual(args.dataset, "cifar10")
        self.assertEqual(args.methods, ["fedavg"])
        self.assertEqual(args.ratios, [0.0])
        self.assertEqual(args.attack, "none")
        self.assertEqual(args.partitions, ["iid"])
        self.assertIsNone(args.train_samples)
        self.assertIsNone(args.test_samples)
        self.assertEqual(args.rounds, 5)
        self.assertEqual(resolve_output_dir(args), Path("outputs/cifar10_clean_baseline"))

    def test_cifar10_clean_baseline_uses_cifar_training_defaults(self):
        args = parse_args(["--cifar10-clean-baseline"])

        self.assertEqual(args.rounds, 300)
        self.assertEqual(args.local_epochs, 5)
        self.assertEqual(args.batch_size, 50)
        self.assertEqual(args.lr, 0.05)
        self.assertEqual(args.lr_decay, 0.99)

    def test_compute_backend_defaults_to_auto(self):
        args = parse_args([])

        self.assertEqual(
            args.methods,
            [
                "sm9rrs",
                "vert",
                "fedredefense",
                "krum",
                "ding13",
                "fedavg",
            ],
        )
        self.assertEqual(args.compute_backend, "auto")
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.jobs, "auto")
        self.assertEqual(args.rounds, 30)
        self.assertEqual(args.local_epochs, 1)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.lr, 0.05)
        self.assertEqual(args.lr_decay, 1.0)
        self.assertEqual(args.attack, "alternating_minimization")
        self.assertEqual(args.attack_boost, 10.0)
        self.assertEqual(args.attack_epochs, 10)
        self.assertEqual(args.attack_stealth_steps, 10)
        self.assertEqual(args.attack_distance_weight, 1e-4)
        self.assertEqual(args.attack_source_label, 5)
        self.assertEqual(args.attack_target_label, 7)
        self.assertEqual(args.attack_target_count, 1)
        self.assertEqual(args.vert_history_window, 10)
        self.assertEqual(args.vert_projection_dim, 128)
        self.assertEqual(args.vert_predict_epochs, 5)
        self.assertEqual(args.vert_predict_lr, 1e-2)
        self.assertEqual(args.vert_top_k, 0)
        self.assertFalse(args.vert_use_ratio_prior)
        self.assertEqual(args.fedre_threshold, 0.6)
        self.assertEqual(args.fedre_initial_iterations, 800)
        self.assertEqual(args.fedre_max_iterations, 2000)
        self.assertEqual(args.fedre_synthetic_steps, 5)

    def test_alternating_alias_and_parameters_map_to_config(self):
        args = parse_args(
            [
                "--attack",
                "alternating",
                "--attack-boost",
                "12",
                "--attack-epochs",
                "3",
                "--attack-stealth-steps",
                "4",
                "--attack-distance-weight",
                "0.002",
                "--attack-source-label",
                "2",
                "--attack-target-label",
                "9",
                "--attack-target-count",
                "6",
            ]
        )
        config = build_experiment_configs(args)[0]

        self.assertEqual(args.attack, "alternating_minimization")
        self.assertEqual(config.attack, "alternating_minimization")
        self.assertEqual(config.attack_boost, 12.0)
        self.assertEqual(config.attack_epochs, 3)
        self.assertEqual(config.attack_stealth_steps, 4)
        self.assertEqual(config.attack_distance_weight, 0.002)
        self.assertEqual(config.attack_source_label, 2)
        self.assertEqual(config.attack_target_label, 9)
        self.assertEqual(config.attack_target_count, 6)

    def test_attack_scale_is_rejected_for_alternating_minimization(self):
        stderr = io.StringIO()
        with mock.patch("sys.stderr", new=stderr), self.assertRaises(SystemExit):
            parse_args(
                [
                    "--attack",
                    "alternating_minimization",
                    "--attack-scale",
                    "100",
                ]
            )
        self.assertIn(
            "--attack-scale does not control alternating minimization",
            stderr.getvalue(),
        )

        sign_flip = parse_args(
            ["--attack", "sign_flip", "--attack-scale", "100"]
        )
        self.assertEqual(sign_flip.attack_scale, 100.0)

    def test_cifar10_uses_paper_training_defaults(self):
        args = parse_args(["--dataset", "cifar10"])

        self.assertEqual(args.rounds, 300)
        self.assertEqual(args.local_epochs, 5)
        self.assertEqual(args.batch_size, 50)
        self.assertEqual(args.lr, 0.05)
        self.assertEqual(args.lr_decay, 0.99)

    def test_explicit_training_arguments_override_dataset_defaults(self):
        args = parse_args(
            [
                "--dataset",
                "cifar10",
                "--rounds",
                "7",
                "--local-epochs",
                "2",
                "--batch-size",
                "16",
                "--lr",
                "0.02",
                "--lr-decay",
                "0.95",
            ]
        )

        self.assertEqual(args.rounds, 7)
        self.assertEqual(args.local_epochs, 2)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.lr, 0.02)
        self.assertEqual(args.lr_decay, 0.95)

    def test_gpu_backend_arguments_are_parsed(self):
        args = parse_args(["--compute-backend", "torch", "--device", "mps"])

        self.assertEqual(args.compute_backend, "torch")
        self.assertEqual(args.device, "mps")

    def test_eval_interval_and_sm9_workers_are_parsed(self):
        args = parse_args(["--eval-interval", "5", "--sm9-workers", "3"])

        self.assertEqual(args.eval_interval, 5)
        self.assertEqual(args.sm9_workers, 3)

    def test_paper_k_and_c_tol_cli_names_map_to_internal_config_fields(self):
        args = parse_args(["--K", "10", "--C_tol", "6"])
        config = build_experiment_configs(args)[0]

        self.assertEqual(args.detector_window, 10)
        self.assertEqual(args.suspicion_remove_after, 6)
        self.assertEqual(config.detector_window, 10)
        self.assertEqual(config.suspicion_remove_after, 6)

        legacy = parse_args(
            ["--detector-window", "8", "--suspicion-remove-after", "5"]
        )
        self.assertEqual(legacy.detector_window, 8)
        self.assertEqual(legacy.suspicion_remove_after, 5)

        lowercase = parse_args(["--k", "7", "--c-tol", "4"])
        self.assertEqual(lowercase.detector_window, 7)
        self.assertEqual(lowercase.suspicion_remove_after, 4)

    def test_vert_ratio_prior_flag_maps_to_every_experiment_config(self):
        args = parse_args(
            [
                "--methods",
                "vert",
                "--ratios",
                "0.2",
                "0.6",
                "--client-counts",
                "20",
                "100",
                "--vert-use-ratio-prior",
            ]
        )
        configs = build_experiment_configs(args)

        self.assertEqual(len(configs), 4)
        self.assertTrue(all(config.vert_use_ratio_prior for config in configs))
        self.assertTrue(all(config.vert_top_k == 0 for config in configs))

    def test_sm9_workers_auto_is_bounded_by_client_count(self):
        args = parse_args(["--num-clients", "4", "--sm9-workers", "auto"])

        self.assertGreaterEqual(args.sm9_workers, 1)
        self.assertLessEqual(args.sm9_workers, min(4, os.cpu_count() or 1))

    def test_parallel_jobs_integer_is_capped_by_config_count(self):
        args = parse_args(["--jobs", "8"])

        self.assertEqual(resolve_parallel_jobs(args.jobs, object(), [object(), object()], args), 2)

    def test_auto_jobs_uses_two_thread_workers_for_single_gpu(self):
        args = parse_args(["--jobs", "auto"])
        dataset = make_synthetic_mnist_like(train_samples=20, test_samples=10, seed=2)
        configs = [ExperimentConfig(), ExperimentConfig(method="fedavg")]

        with mock.patch(
            "sm9rrsfl.experiments.describe_compute_backend",
            return_value="torch:cuda",
        ):
            jobs = resolve_parallel_jobs("auto", dataset, configs, args)

        self.assertEqual(jobs, 2)
        self.assertEqual(parallel_executor_kind("torch:cuda", jobs), "thread")

    def test_auto_jobs_uses_all_available_cpu_slots_when_cuda_memory_allows(self):
        args = parse_args(["--jobs", "auto"])
        dataset = make_synthetic_mnist_like(train_samples=20, test_samples=10, seed=2)
        configs = [ExperimentConfig(seed=index) for index in range(8)]

        with (
            mock.patch(
                "sm9rrsfl.experiments.describe_compute_backend",
                return_value="torch:cuda",
            ),
            mock.patch("sm9rrsfl.experiments.available_cpu_count", return_value=4),
            mock.patch("sm9rrsfl.experiments._physical_memory_mb", return_value=64 * 1024),
            mock.patch("sm9rrsfl.experiments._cuda_memory_parallel_limit", return_value=6),
        ):
            jobs = resolve_parallel_jobs("auto", dataset, configs, args)

        self.assertEqual(jobs, 4)

    def test_auto_sm9_workers_share_cpu_budget_with_parallel_experiments(self):
        with mock.patch("sm9rrsfl.experiments.available_cpu_count", return_value=4):
            workers = resolve_sm9_workers("auto", 100, parallel_jobs=4)

        self.assertEqual(workers, 1)

    def test_auto_cuda_assignment_round_robins_visible_devices(self):
        configs = [ExperimentConfig(seed=index) for index in range(5)]
        with mock.patch(
            "sm9rrsfl.experiments.available_cuda_devices",
            return_value=("cuda:0", "cuda:1"),
        ):
            assigned = assign_auto_cuda_devices(configs, "torch:cuda", "auto")

        self.assertEqual(
            [config.device for config in assigned],
            ["cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0"],
        )
        self.assertTrue(all(config.device == "auto" for config in configs))

    def test_internal_indexed_cuda_device_is_accepted(self):
        from sm9rrsfl.torch_backend import _normalize_device

        self.assertEqual(_normalize_device("cuda:3"), "cuda:3")

    def test_explicit_jobs_is_honored_on_mps_and_uses_threads(self):
        args = parse_args(["--jobs", "4"])
        configs = [ExperimentConfig(), ExperimentConfig(method="fedavg")]
        with mock.patch(
            "sm9rrsfl.experiments.describe_compute_backend",
            return_value="torch:mps",
        ):
            jobs = resolve_parallel_jobs("4", object(), configs, args)

        self.assertEqual(jobs, 2)
        self.assertEqual(parallel_executor_kind("torch:mps", jobs), "thread")

    def test_cpu_parallel_jobs_use_processes(self):
        self.assertEqual(parallel_executor_kind("numpy", 4), "process")
        self.assertEqual(parallel_executor_kind("torch:cpu", 2), "process")
        self.assertEqual(parallel_executor_kind("numpy", 1), "serial")

    def test_incremental_checkpoint_round_trip_includes_stage_timings(self):
        dataset = make_synthetic_mnist_like(train_samples=20, test_samples=10, seed=4)
        result = run_experiment(
            dataset,
            ExperimentConfig(
                method="fedavg",
                num_clients=2,
                rounds=1,
                early_stop=False,
                seed=4,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_result_files(output_dir, [result])
            snapshot = load_completed_results_snapshot(output_dir)
            restored = read_results(output_dir / "summary.csv", output_dir / "rounds.csv")
            with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(len(restored), 1)
        self.assertIsNotNone(snapshot)
        self.assertEqual(len(snapshot), 1)
        self.assertIn("training_seconds", row)
        self.assertGreaterEqual(restored[0].stage_timings.training_seconds, 0.0)

    def test_sm9rrs_client_diagnostics_are_written(self):
        dataset = make_synthetic_mnist_like(
            train_samples=40,
            test_samples=10,
            seed=44,
        )
        result = run_experiment(
            dataset,
            ExperimentConfig(
                method="sm9rrs",
                malicious_ratio=0.0,
                num_clients=2,
                rounds=2,
                local_epochs=1,
                batch_size=8,
                crypto_mode="simulated",
                early_stop=False,
                seed=44,
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_result_files(output_dir, [result])
            with (output_dir / "sm9rrs_diagnostics.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 4)
        self.assertEqual({row["method"] for row in rows}, {"sm9rrs"})
        self.assertIn("z_sigma", rows[0])
        self.assertIn("z_direction", rows[0])
        self.assertIn("weight_after_penalty_recovery", rows[0])
        self.assertIn("aggregation_weight", rows[0])
        self.assertIn("count_after", rows[0])
        self.assertEqual(
            {row["client_id"] for row in rows},
            {"client-0", "client-1"},
        )

    def test_unexpected_second_round_failure_resumes_from_atomic_checkpoint(self):
        from sm9rrsfl import fl as fl_module

        dataset = make_synthetic_mnist_like(train_samples=80, test_samples=20, seed=15)
        config = ExperimentConfig(
            method="fedavg",
            malicious_ratio=0.0,
            num_clients=4,
            rounds=3,
            local_epochs=1,
            batch_size=16,
            early_stop=False,
            seed=15,
        )
        uninterrupted = run_experiment(dataset, config)
        original_fedavg = fl_module._fedavg
        calls = 0

        def fail_during_second_round(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected second-round failure")
            return original_fedavg(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            checkpoint_dir = output_dir / ".checkpoints"
            with mock.patch.object(fl_module, "_fedavg", side_effect=fail_during_second_round):
                with self.assertRaisesRegex(RuntimeError, "injected second-round failure"):
                    run_measured_experiment(
                        dataset,
                        config,
                        checkpoint_dir=checkpoint_dir,
                        run_fingerprint="test-run",
                    )

            with (output_dir / "last_failure.json").open(encoding="utf-8") as handle:
                failure = json.load(handle)
            self.assertEqual(failure["last_completed_round"], 1)
            self.assertTrue(failure["checkpoint_exists"])
            self.assertEqual(len(list(checkpoint_dir.glob("*.pickle"))), 1)

            resumed = run_measured_experiment(
                dataset,
                config,
                checkpoint_dir=checkpoint_dir,
                run_fingerprint="test-run",
            )
            with (output_dir / "last_failure.json").open(encoding="utf-8") as handle:
                resolved_failure = json.load(handle)

        self.assertEqual(
            [record.accuracy for record in resumed.records],
            [record.accuracy for record in uninterrupted.records],
        )
        self.assertTrue(resolved_failure["resolved"])
        self.assertEqual(len(list(checkpoint_dir.glob("*.pickle"))), 0)

    def test_progress_can_be_disabled(self):
        args = parse_args(["--no-progress"])

        self.assertTrue(args.no_progress)

    def test_progress_reporter_prints_eta(self):
        stream = io.StringIO()
        progress = ProgressReporter(total=2, stream=stream)
        progress.finish_config(ExperimentConfig(method="fedavg", malicious_ratio=0.0))
        progress.close()

        output = stream.getvalue()
        self.assertIn("1/2", output)
        self.assertIn("eta=", output)
        self.assertIn("complete", output)

    def test_tty_progress_refreshes_without_a_completed_configuration(self):
        refreshed = Event()

        class TTYStream(io.StringIO):
            def __init__(self):
                super().__init__()
                self.refresh_count = 0

            def isatty(self):
                return True

            def write(self, value):
                if value.startswith("\r"):
                    self.refresh_count += 1
                    if self.refresh_count >= 2:
                        refreshed.set()
                return super().write(value)

        stream = TTYStream()
        progress = ProgressReporter(
            total=2,
            stream=stream,
            refresh_interval=0.01,
        )
        try:
            self.assertTrue(refreshed.wait(0.5))
            progress.start_parallel(2, 2)
        finally:
            progress.close()

        self.assertGreaterEqual(stream.refresh_count, 3)
        self.assertIn("running 2 configurations with 2 workers", stream.getvalue())
        self.assertFalse(progress._refresh_thread.is_alive())

    def test_format_duration(self):
        self.assertEqual(_format_duration(5), "5s")
        self.assertEqual(_format_duration(65), "1m05s")
        self.assertEqual(_format_duration(3661), "1h01m01s")


if __name__ == "__main__":
    unittest.main()
