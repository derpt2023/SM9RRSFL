from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts.experiments.cifar10 import cifar_fair_diagnostics as diag
from scripts.experiments.cifar10 import fair_tuning_dashboard as ui
from sm9rrsfl import experiments, fair_tuning
from sm9rrsfl.fl import ExperimentConfig
import run_cifar_v14_with_progress as launcher


ROOT = Path(__file__).resolve().parents[1]


class FailureEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.checkpoints = self.folder / ".checkpoints"
        self.checkpoints.mkdir()
        self.config = ExperimentConfig(method="sm9rrs", device="cuda:3", compute_backend="torch")
        self.torch = SimpleNamespace(cuda=SimpleNamespace(
            current_device=lambda: 0, mem_get_info=lambda device: (200, 24000),
            memory_allocated=lambda device: 100, memory_reserved=lambda device: 500,
            memory_stats=lambda device: {"num_ooms": 2, "unrelated": 10}))
        self.stream = io.StringIO()
        self.redirect = redirect_stdout(self.stream)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def failure(self):
        # Use actual traceback frames with the real training module/function
        # names and scalar metadata, without starting training or CUDA.
        namespace = {"__name__": "sm9rrsfl.torch_backend"}
        exec("def local_train_delta_resident():\n"
             "    client_idx, batch_size, samples, start, loss_batches = 19, 50, 701, 700, 14\n"
             "    raise RuntimeError('cuDNN error: CUDNN_STATUS_INTERNAL_ERROR')\n", namespace)
        fl_namespace = {"__name__": "sm9rrsfl.fl", "train": namespace["local_train_delta_resident"]}
        exec("def run_experiment():\n"
             "    round_id, client_idx, identity, attack_active = 26, 19, 'client-19', False\n"
             "    train()\n", fl_namespace)
        try:
            fl_namespace["run_experiment"]()
        except RuntimeError as error:
            return error

    def record(self, error=None):
        with mock.patch.object(diag, "_torch_module", return_value=self.torch), \
                mock.patch.object(diag, "runtime_info", return_value={"torch": "fixture"}):
            return diag.record_failure(error or self.failure(), self.config,
                                       checkpoint_dir=self.checkpoints, task_id="validation_sm9rrs-001")

    def test_exact_training_location_and_memory_are_saved_before_frames_are_cleared(self):
        error = self.failure()
        report = self.record(error)
        self.assertEqual(report["context"]["round_id"], 26)
        self.assertEqual(report["context"]["identity"], "client-19")
        self.assertEqual(report["context"]["start"], 700)
        self.assertEqual(report["context"]["samples"], 701)
        self.assertEqual(report["cuda_memory"]["reserved_bytes"], 500)
        self.assertEqual(report["cuda_memory"]["allocator_stats"], {"num_ooms": 2})
        diag.clear_failure_frames(error)
        self.assertIn("CUDNN_STATUS_INTERNAL_ERROR", report["traceback"])
        self.assertIn("source_sha256", report["code_at_start"])
        self.assertEqual(json.loads((self.folder / "gpu_failures.jsonl").read_text()), report)

    def test_failure_memory_query_cannot_replace_original_cudnn_error(self):
        def broken(device):
            raise RuntimeError("CUDA query failed")
        self.torch.cuda.mem_get_info = broken
        report = self.record()
        self.assertIn("CUDNN_STATUS_INTERNAL_ERROR", report["error"])
        self.assertEqual(report["cuda_memory"]["free_total_bytes"]["error"], "CUDA query failed")

    def test_real_cpu_training_failure_captures_round_client_batch_and_checkpoint(self):
        import torch
        from sm9rrsfl.datasets import make_synthetic_mnist_like
        dataset = make_synthetic_mnist_like(train_samples=30, test_samples=10, seed=7)
        config = ExperimentConfig(method="fedavg", num_clients=2, rounds=1, seed=301,
            compute_backend="torch", device="cpu", crypto_mode="simulated", early_stop=False)
        item = fair_tuning.TuningExperimentTask("validation", "fedavg-001", "fedavg", config)
        spec = fair_tuning.load_fair_tuning_config(ROOT / "configs/cifar10_v14_fair_grid.json")
        flags = diag.runtime_info()
        with mock.patch.object(torch.Tensor, "backward", side_effect=RuntimeError("injected backward failure")), \
                ui.dashboard(spec) as progress:
            progress.register([item])
            with self.assertRaisesRegex(RuntimeError, "injected backward failure"):
                fair_tuning.run_measured_experiment(dataset, config, checkpoint_dir=self.checkpoints,
                    run_fingerprint="fixture", task_id="validation_fedavg-001")
        row = json.loads((self.folder / "gpu_failures.jsonl").read_text())
        self.assertEqual(row["context"]["round_id"], 1)
        self.assertEqual(row["context"]["client_idx"], 0)
        self.assertEqual(row["context"]["start"], 0)
        self.assertEqual(row["context"]["loss"]["shape"], [])
        self.assertEqual(row["context"]["parameters"][0]["device"], "cpu")
        self.assertEqual(diag.runtime_info(), flags)
        saved = json.loads((self.folder / "last_failure.json").read_text())
        self.assertEqual(saved["last_completed_round"], 0)
        self.assertTrue(Path(saved["checkpoint_path"]).exists())

    def test_concurrent_failures_remain_separate_json_records(self):
        with mock.patch.object(diag, "_torch_module", return_value=self.torch), \
                mock.patch.object(diag, "runtime_info", return_value={}):
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda i: diag.record_failure(self.failure(), self.config,
                    checkpoint_dir=self.checkpoints, task_id=str(i)), range(8)))
        rows = [json.loads(line) for line in (self.folder / "gpu_failures.jsonl").read_text().splitlines()]
        self.assertEqual({row["task_id"] for row in rows}, {str(i) for i in range(8)})

    def test_diagnose_reads_json_only_and_does_not_treat_old_record_as_current(self):
        state = self.folder / ".tuning_state" / "validation" / "fingerprint"
        state.mkdir(parents=True)
        (state / "last_failure.json").write_text(json.dumps({"timestamp_utc": "old", "resolved": True}))
        (state / "gpu_failures.jsonl").write_text('{"attempt": 1}\n{"attempt": 2}\n')
        (state / "cuda_recovery.jsonl").write_text('{"event": "cuda_worker_failed"}\n')
        with mock.patch.object(diag, "_torch_module", side_effect=AssertionError("no CUDA inspection")), \
                mock.patch.object(fair_tuning, "run_fair_tuning") as run:
            report = diag.diagnose(self.folder)
        run.assert_not_called()
        self.assertTrue(report["states"][0]["last_failure"]["resolved"])
        self.assertEqual(len(report["states"][0]["gpu_failures.jsonl"]), 2)
        self.assertIn("Historical", report["note"])

    def test_launcher_diagnose_does_not_load_data_or_start_experiment(self):
        with mock.patch.object(diag, "diagnose", return_value={}) as diagnose, \
                mock.patch.object(fair_tuning, "run_fair_tuning") as run, \
                mock.patch.object(fair_tuning, "load_image_dataset") as load:
            launcher.main(["--diagnose"])
        diagnose.assert_called_once()
        run.assert_not_called()
        load.assert_not_called()

    def test_diagnostic_write_error_does_not_mask_training_exception(self):
        spec = fair_tuning.load_fair_tuning_config(ROOT / "configs/cifar10_v14_fair_grid.json")
        item = fair_tuning.TuningExperimentTask("validation", "sm9rrs-001", "sm9rrs", self.config)
        with mock.patch.object(fair_tuning, "run_measured_experiment", side_effect=self.failure()), \
                mock.patch.object(diag, "record_failure", side_effect=OSError("disk full")), \
                ui.dashboard(spec) as progress:
            progress.register([item])
            with self.assertRaisesRegex(RuntimeError, "CUDNN_STATUS_INTERNAL_ERROR"):
                fair_tuning.run_measured_experiment(None, self.config, checkpoint_dir=self.checkpoints)
            self.assertEqual(progress.failed_attempts, 1)


if __name__ == "__main__":
    unittest.main()
