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


class StrictNumericsTest(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_import_alone_does_not_load_torch_or_set_workspace(self):
        code = (
            "import os, sys; import sm9rrsfl.numerics; "
            "assert 'torch' not in sys.modules; "
            "assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ"
        )
        subprocess.run([sys.executable, "-c", code],
                       cwd=Path(__file__).resolve().parents[1], check=True,
                       capture_output=True, text=True)

    def test_strict_configuration_applies_before_first_cuda_work(self):
        torch = fake_torch()
        def verify_workspace(*args, **kwargs):
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        torch.use_deterministic_algorithms.side_effect = verify_workspace
        result = numerics.configure_strict_numerics(torch_module=torch)
        torch.use_deterministic_algorithms.assert_called_once_with(True, warn_only=False)
        torch.set_float32_matmul_precision.assert_called_once_with("highest")
        self.assertTrue(torch.backends.cudnn.deterministic)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(result["cuda_matmul_allow_tf32"])
        torch.cuda.device_count.assert_not_called()
        torch.cuda.get_device_properties.assert_not_called()

    def test_different_or_empty_preexisting_workspace_is_not_overwritten(self):
        for value in (":16:8", "invalid", ""):
            with self.subTest(value=value):
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = value
                torch = fake_torch()
                with self.assertRaisesRegex(ValueError, "CUBLAS_WORKSPACE_CONFIG"):
                    numerics.configure_strict_numerics(torch_module=torch)
                self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], value)
                torch.use_deterministic_algorithms.assert_not_called()

    def test_late_configuration_without_workspace_requires_restart(self):
        torch = fake_torch(initialized=True)
        with self.assertRaisesRegex(RuntimeError, "already initialized.*restart"):
            numerics.configure_strict_numerics(torch_module=torch)
        self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", os.environ)
        torch.use_deterministic_algorithms.assert_not_called()

    def test_repeat_call_with_correct_environment_supports_initialized_worker(self):
        torch = fake_torch()
        first = numerics.configure_strict_numerics(torch_module=torch)
        torch.cuda.is_initialized.return_value = True
        second = numerics.configure_strict_numerics(torch_module=torch)
        self.assertEqual(first, second)
        self.assertEqual(torch.use_deterministic_algorithms.call_count, 2)

    def test_tf32_cannot_be_enabled_under_strict_protocol(self):
        torch = fake_torch()
        with self.assertRaisesRegex(ValueError, "tf32=False"):
            numerics.configure_strict_numerics(tf32=True, torch_module=torch)
        self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", os.environ)
        torch.use_deterministic_algorithms.assert_not_called()


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
        json.dumps(result, allow_nan=False)

    def test_missing_torch_is_reported_without_failing_metadata(self):
        with mock.patch.object(numerics, "_load_torch", side_effect=RuntimeError("missing")):
            result = numerics.numerical_environment()
        self.assertEqual(result["torch"], {"available": False})
        self.assertIn("numpy", result)


if __name__ == "__main__":
    unittest.main()
