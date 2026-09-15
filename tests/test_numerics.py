import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from sm9rrsfl import numerics


def fake_torch(*, initialized=False):
    torch = SimpleNamespace(
        __version__="2.test.cpu", version=SimpleNamespace(cuda=None, hip=None),
        cuda=SimpleNamespace(is_initialized=mock.Mock(return_value=initialized),
                             device_count=mock.Mock(return_value=0),
                             get_device_properties=mock.Mock()),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(deterministic=False, benchmark=True,
                                  allow_tf32=True, version=lambda: None),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))),
        use_deterministic_algorithms=mock.Mock(),
        set_float32_matmul_precision=mock.Mock(),
        are_deterministic_algorithms_enabled=lambda: True,
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
        get_float32_matmul_precision=lambda: "highest",
        get_num_threads=lambda: 1, get_num_interop_threads=lambda: 2,
        __config__=SimpleNamespace(show=lambda: "test BLAS build"),
    )
    return torch


class MetadataImportTest(unittest.TestCase):
    def test_import_alone_preserves_environment_and_does_not_load_torch(self):
        # Import the existing package first: its long-standing thread defaults
        # are separate from this module's read-only contract.
        code = (
            "import os, sys; import sm9rrsfl; before = dict(os.environ); "
            "import sm9rrsfl.numerics; "
            "assert 'torch' not in sys.modules; "
            "assert dict(os.environ) == before; "
            "assert sm9rrsfl.numerics.__all__ == ['numerical_environment']; "
            "assert not hasattr(sm9rrsfl.numerics, 'configure_strict_numerics')"
        )
        subprocess.run([sys.executable, "-c", code],
                       cwd=Path(__file__).resolve().parents[1], check=True,
                       capture_output=True, text=True)

class NumericalEnvironmentTest(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("_nvidia_information", {"available": False, "gpus": []}),
            ("_native_sm9_information", {"available": False}),
        ):
            patcher = mock.patch.object(numerics, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_cpu_metadata_is_json_serializable_and_does_not_initialize_cuda(self):
        torch = fake_torch()
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "3",
                                          "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            result = numerics.numerical_environment(device="cpu", torch_module=torch)
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["requested_device"], "cpu")
        self.assertEqual(result["torch"]["version"], "2.test.cpu")
        self.assertIsNone(result["torch"]["cuda_version"])
        self.assertFalse(result["torch"]["cuda_initialized"])
        self.assertEqual(result["environment"]["OMP_NUM_THREADS"], "3")
        self.assertTrue(result["numpy"]["available"])
        self.assertIn("version", result["python"])
        self.assertEqual(result["torch"]["num_interop_threads"], 2)
        self.assertEqual(result["torch"]["build_configuration"], "test BLAS build")
        torch.cuda.device_count.assert_not_called()
        torch.cuda.get_device_properties.assert_not_called()
        torch.use_deterministic_algorithms.assert_not_called()

    def test_snapshot_preserves_all_flags_and_operator_environment(self):
        torch = fake_torch()
        for workspace in (None, ":16:8", "operator-defined"):
            with self.subTest(workspace=workspace), mock.patch.dict(os.environ, {}, clear=True):
                if workspace is not None:
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
                os.environ["CUDA_VISIBLE_DEVICES"] = "7"
                before = dict(os.environ)
                result = numerics.numerical_environment("cuda:0", torch_module=torch)
                self.assertEqual(dict(os.environ), before)
                self.assertEqual(result["environment"]["CUBLAS_WORKSPACE_CONFIG"], workspace)
                self.assertFalse(torch.backends.cudnn.deterministic)
                self.assertTrue(torch.backends.cudnn.benchmark)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                torch.use_deterministic_algorithms.assert_not_called()
                torch.set_float32_matmul_precision.assert_not_called()
                torch.cuda.device_count.assert_not_called()
                torch.cuda.get_device_properties.assert_not_called()

    def test_optional_torch_query_failure_is_recorded_as_null(self):
        torch = fake_torch()
        torch.backends.cudnn.version = mock.Mock(side_effect=RuntimeError("missing library"))
        torch.get_num_threads = mock.Mock(side_effect=OSError("query failed"))
        result = numerics.numerical_environment(torch_module=torch)
        self.assertIsNone(result["torch"]["cudnn_version"])
        self.assertIsNone(result["torch"]["num_threads"])

    def test_initialized_cuda_metadata_records_logical_device_identity(self):
        torch = fake_torch(initialized=True)
        torch.version.cuda = "12.test"
        torch.cuda.device_count.return_value = 1
        torch.cuda.get_device_properties.return_value = SimpleNamespace(
            name="test GPU", total_memory=24 * 1024 ** 3, major=8, minor=9,
            uuid="GPU-example")
        result = numerics.numerical_environment(device="cuda:0", torch_module=torch)
        device = result["torch"]["logical_cuda_devices"][0]
        self.assertEqual(device["logical_index"], 0)
        self.assertEqual(device["compute_capability"], [8, 9])
        self.assertEqual(device["uuid"], "GPU-example")
        self.assertEqual(result["torch"]["logical_cuda_devices_status"], "queried")
        torch.use_deterministic_algorithms.assert_not_called()
        torch.set_float32_matmul_precision.assert_not_called()
        json.dumps(result, allow_nan=False)

    def test_initialized_cuda_properties_failure_is_reported(self):
        torch = fake_torch(initialized=True)
        torch.cuda.device_count.side_effect = RuntimeError("driver unavailable")
        result = numerics.numerical_environment("cuda:0", torch_module=torch)
        self.assertEqual(result["torch"]["logical_cuda_devices"], [])
        self.assertEqual(result["torch"]["logical_cuda_devices_status"], "RuntimeError")

    def test_missing_torch_is_reported_without_failing_metadata(self):
        with mock.patch.object(numerics, "_load_torch", side_effect=RuntimeError("missing")):
            result = numerics.numerical_environment()
        self.assertEqual(result["torch"], {"available": False})
        self.assertIn("numpy", result)


class OptionalMetadataTest(unittest.TestCase):
    def test_nvidia_unavailable_does_not_spawn_command(self):
        with mock.patch.object(numerics.shutil, "which", return_value=None), \
                mock.patch.object(numerics.subprocess, "run") as run:
            result = numerics._nvidia_information()
        self.assertFalse(result["available"])
        run.assert_not_called()

    def test_nvidia_failures_are_optional(self):
        for failure in (OSError("missing"), subprocess.TimeoutExpired("nvidia-smi", 2)):
            with self.subTest(failure=failure), \
                    mock.patch.object(numerics.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
                    mock.patch.object(numerics.subprocess, "run", side_effect=failure):
                result = numerics._nvidia_information()
            self.assertFalse(result["available"])
            self.assertEqual(result["reason"], type(failure).__name__)

    def test_nvidia_readonly_query_parses_physical_identity(self):
        response = SimpleNamespace(returncode=0, stdout="0, GPU-id, Test GPU, 123.4, 24576\ninvalid\n")
        with mock.patch.object(numerics.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
                mock.patch.object(numerics.subprocess, "run", return_value=response) as run:
            result = numerics._nvidia_information()
        self.assertEqual(result["gpus"][0]["uuid"], "GPU-id")
        self.assertEqual(len(result["gpus"]), 1)
        self.assertEqual(run.call_args.args[0], [
            "/usr/bin/nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits"])
        self.assertEqual(run.call_args.kwargs["timeout"], 2)

    def test_numpy_optional_build_query_failure_is_recorded_as_null(self):
        np = SimpleNamespace(__version__="test", __config__=SimpleNamespace(
            get_info=mock.Mock(side_effect=RuntimeError("unsupported query"))))
        with mock.patch.object(numerics.importlib, "import_module", return_value=np):
            result = numerics._numpy_information()
        self.assertTrue(result["available"])
        self.assertTrue(all(value is None for value in result["build_configuration"].values()))

    def test_missing_numpy_is_optional(self):
        with mock.patch.object(numerics.importlib, "import_module", side_effect=ImportError):
            self.assertEqual(numerics._numpy_information(), {"available": False})


if __name__ == "__main__":
    unittest.main()
