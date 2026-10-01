import gc
import importlib.util
import pickle
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from sm9rrsfl import model
import fashion_resnet_gn as resnet_gn
from sm9rrsfl.datasets import ImageDataset


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
class FashionResNetGNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.previous_threads)

    def tearDown(self):
        resnet_gn.uninstall_runtime()
        gc.collect()

    def dataset(self):
        rng = np.random.default_rng(4)
        return ImageDataset(
            rng.normal(size=(2, 1, 28, 28)).astype(np.float32), np.array([0, 1]),
            rng.normal(size=(2, 1, 28, 28)).astype(np.float32), np.array([0, 1]),
            name="fashion_mnist", x_attack=rng.normal(size=(1, 1, 28, 28)).astype(np.float32),
            y_attack=np.array([0]),
        )

    def test_layout_initialization_and_flat_roundtrip(self):
        from sm9rrsfl import torch_backend as backend
        spec = resnet_gn.SPEC
        self.assertEqual(spec.parameter_size, 11172810)
        self.assertEqual(spec.svd_matrix_shape, (512, 10))
        self.assertEqual(spec.svd_matrix_offset + 5130, spec.parameter_size)
        rng_state = self.torch.get_rng_state().clone()
        first = resnet_gn.init_params(seed=3)
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, resnet_gn.init_params(seed=3))
        self.assertFalse(np.array_equal(first, resnet_gn.init_params(seed=4)))
        self.assertTrue(self.torch.equal(rng_state, self.torch.get_rng_state()))
        with resnet_gn.runtime():
            parts = backend._torch_params_from_tensor(self.torch, self.torch.from_numpy(first),
                                                       spec, requires_grad=False, clone=False)
            np.testing.assert_array_equal(backend._torch_vector_from_params(parts), first)
            for (name, _), part in zip(resnet_gn.parameter_layout(), parts):
                if name.endswith("gn_weight"):
                    self.assertTrue(self.torch.all(part == 1).item())
                elif name.endswith("gn_bias") or name == "classifier.bias":
                    self.assertTrue(self.torch.all(part == 0).item())
            with self.assertRaises((ValueError, RuntimeError)):
                backend._torch_params_from_tensor(self.torch, self.torch.zeros(3), spec,
                                                   requires_grad=False, clone=False)

    def test_forward_and_gradients_match_independent_module(self):
        from sm9rrsfl import torch_backend as backend
        torch, nn = self.torch, self.torch.nn

        class Block(nn.Module):
            def __init__(self, incoming, outgoing, stride):
                super().__init__()
                self.first = nn.Sequential(nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False),
                                           nn.GroupNorm(2, outgoing), nn.ReLU())
                self.second = nn.Sequential(nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False),
                                            nn.GroupNorm(2, outgoing))
                self.shortcut = (nn.Sequential(nn.Conv2d(incoming, outgoing, 1, stride, bias=False),
                                               nn.GroupNorm(2, outgoing))
                                 if incoming != outgoing else nn.Identity())

            def forward(self, x):
                return torch.relu(self.second(self.first(x)) + self.shortcut(x))

        class Reference(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1, bias=False), nn.GroupNorm(2, 64), nn.ReLU())
                blocks = []
                incoming = 64
                for outgoing in (64, 128, 256, 512):
                    blocks.extend((Block(incoming, outgoing, 1 if incoming == outgoing else 2),
                                   Block(outgoing, outgoing, 1)))
                    incoming = outgoing
                self.blocks = nn.Sequential(*blocks)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Linear(512, 10)

            def forward(self, x):
                return self.classifier(self.pool(self.blocks(self.stem(x))).flatten(1))

        with resnet_gn.runtime():
            reference = Reference()
            self.assertEqual(list(reference.buffers()), [])
            vector = torch.from_numpy(resnet_gn.init_params(seed=6))
            params = backend._torch_params_from_tensor(torch, vector, resnet_gn.SPEC,
                                                        requires_grad=True, clone=True)
            module_params = list(reference.parameters())
            self.assertEqual(len(params), len(module_params))
            with torch.no_grad():
                for index, (target, source) in enumerate(zip(module_params, params)):
                    target.copy_(source.T if index == len(params) - 2 else source)
            x = torch.from_numpy(self.dataset().x_train)
            actual = backend._torch_forward(torch, params, x, resnet_gn.SPEC)
            expected = reference(x)
            self.assertEqual(tuple(actual.shape), (2, 10))
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            target = torch.tensor([0, 1])
            nn.functional.cross_entropy(actual, target).backward()
            nn.functional.cross_entropy(expected, target).backward()
            for index, (part, ref) in enumerate(zip(params, module_params)):
                self.assertIsNotNone(part.grad)
                self.assertTrue(torch.isfinite(part.grad).all().item())
                expected_grad = ref.grad.T if index == len(params) - 2 else ref.grad
                torch.testing.assert_close(part.grad, expected_grad, rtol=2e-4, atol=2e-6)
            with self.assertRaisesRegex(ValueError, "NCHW"):
                backend._torch_forward(torch, params, x[:, :, :24, :24], resnet_gn.SPEC)

    def test_classifier_layout_used_by_both_detectors(self):
        from sm9rrsfl.svd_detector import LongitudinalSVDDetector
        from sm9rrsfl.ding13_detector import Ding13TrajectoryDetector
        from sm9rrsfl.ours_policy import OursParameters
        spec = resnet_gn.SPEC
        vector = np.zeros(spec.parameter_size, dtype=np.float32)
        weights = np.tile(np.arange(1, 11, dtype=np.float32), (512, 1))
        vector[spec.svd_matrix_offset:-10] = weights.ravel()
        vector[-10:] = np.arange(10, dtype=np.float32)
        detector = LongitudinalSVDDetector(window_size=3, policy=OursParameters(),
            num_classes=10, expected_update_size=spec.parameter_size,
            matrix_offset=spec.svd_matrix_offset, matrix_shape=spec.svd_matrix_shape)
        feature, _ = detector._extract(vector, 1.)
        classes = np.concatenate((weights.mean(axis=0), vector[-10:])).astype(np.float64)
        classes *= np.sqrt(20) / np.linalg.norm(classes)
        np.testing.assert_allclose(feature[-20:], classes, rtol=1e-6, atol=1e-7)
        ding = Ding13TrajectoryDetector(["client-0"], contamination=0.,
            matrix_offset=spec.svd_matrix_offset, matrix_shape=spec.svd_matrix_shape)
        values = ding._singular_values(vector)
        expected = np.linalg.svd(weights.T @ weights, compute_uv=False)
        np.testing.assert_allclose(values, expected, rtol=1e-5, atol=2e-4)

    def test_resident_training_attack_and_evaluation(self):
        from sm9rrsfl import torch_backend as backend
        dataset = self.dataset()
        with resnet_gn.runtime():
            context = backend.TorchTrainingContext(dataset, [np.array([0, 1])],
                                                   spec=resnet_gn.SPEC, device="cpu")
            params = resnet_gn.init_params(seed=8)
            kwargs = dict(client_idx=0, lr=.001, epochs=1, batch_size=2, seed=9)
            delta, stats = context.local_train_delta_resident(params, **kwargs)
            repeat, _ = context.local_train_delta_resident(params, **kwargs)
            self.assertTrue(self.torch.equal(delta, repeat))
            self.assertEqual(stats.samples, 2)
            self.assertTrue(np.isfinite(stats.loss))
            self.assertTrue(self.torch.isfinite(delta).all().item())
            self.assertGreater(float(self.torch.linalg.vector_norm(delta)), 0.)
            direct, _ = model.local_train_delta(params, dataset.x_train, dataset.y_train,
                spec=resnet_gn.SPEC, compute_backend="torch", device="cpu",
                lr=.001, epochs=1, batch_size=2, seed=9)
            np.testing.assert_allclose(context.to_numpy(delta), direct, rtol=1e-5, atol=1e-7)
            del repeat, direct
            attack, attack_stats = context.alternating_minimization_delta_resident(params,
                client_idx=0, target_indices=np.array([0]), target_label=1, lr=.001,
                attack_epochs=1, batch_size=2, stealth_steps=1, boost=2., distance_weight=.0001, seed=10)
            self.assertEqual(attack_stats.samples, 2)
            self.assertTrue(self.torch.isfinite(attack).all().item())
            self.assertFalse(self.torch.equal(attack, delta))
            acc = context.accuracy(params, batch_size=1)
            direct_acc = model.accuracy(params, dataset.x_test, dataset.y_test,
                spec=resnet_gn.SPEC, compute_backend="torch", device="cpu", batch_size=1)
            self.assertEqual(acc, direct_acc)
            success, confidence = context.targeted_metrics(params, target_indices=np.array([0]), target_label=1)
            self.assertTrue(0 <= success <= 1 and 0 <= confidence <= 1)

    def test_runtime_preserves_legacy_models_and_nested_cifar_adapter(self):
        import cifar_resnet_gn as cifar
        from sm9rrsfl import fl, torch_backend as backend
        original_selector = model.model_spec_for_dataset
        original_forward = backend._torch_forward
        cifar_dataset = SimpleNamespace(name="cifar10", input_shape=(3, 32, 32), num_classes=10)
        mnist_dataset = SimpleNamespace(name="mnist", input_shape=(1, 28, 28), num_classes=10)
        old_specs = [original_selector(dataset) for dataset in (cifar_dataset, mnist_dataset)]
        inputs = [self.torch.zeros(2, 3, 32, 32), self.torch.zeros(2, 1, 28, 28)]
        vectors = [model.init_params(seed=5, spec=spec) for spec in old_specs]
        parts = [backend._torch_params_from_tensor(self.torch, self.torch.from_numpy(vector), spec,
                 requires_grad=False, clone=False) for spec, vector in zip(old_specs, vectors)]
        before = [original_forward(self.torch, p, x, s) for p, x, s in zip(parts, inputs, old_specs)]
        with resnet_gn.runtime():
            self.assertIs(model.model_spec_for_dataset(self.dataset()), resnet_gn.SPEC)
            self.assertIs(fl.model_spec_for_dataset(self.dataset()), resnet_gn.SPEC)
            self.assertEqual(model.model_spec_for_dataset(cifar_dataset), old_specs[0])
            self.assertEqual(model.model_spec_for_dataset(mnist_dataset), old_specs[1])
            with resnet_gn.runtime():
                self.assertTrue(resnet_gn.runtime_installed())
            for p, x, s, vector, expected in zip(parts, inputs, old_specs, vectors, before):
                self.assertTrue(self.torch.equal(backend._torch_forward(self.torch, p, x, s), expected))
                np.testing.assert_array_equal(model.init_params(seed=5, spec=s), vector)
        self.assertIs(model.model_spec_for_dataset, original_selector)
        self.assertIs(backend._torch_forward, original_forward)

        # Compose with an already-active CIFAR v8 adapter without replacing it.
        with cifar.runtime():
            cifar_selector = model.model_spec_for_dataset
            cifar_vector = model.init_params(seed=7, spec=cifar.SPEC)
            p = backend._torch_params_from_tensor(self.torch, self.torch.from_numpy(cifar_vector),
                cifar.SPEC, requires_grad=False, clone=False)
            expected = backend._torch_forward(self.torch, p, inputs[0], cifar.SPEC)
            with resnet_gn.runtime():
                self.assertIs(model.model_spec_for_dataset(cifar_dataset), cifar.SPEC)
                self.assertIs(model.model_spec_for_dataset(self.dataset()), resnet_gn.SPEC)
                self.assertTrue(self.torch.equal(backend._torch_forward(self.torch, p, inputs[0], cifar.SPEC), expected))
                np.testing.assert_array_equal(model.init_params(seed=7, spec=cifar.SPEC), cifar_vector)
                with self.assertRaisesRegex(RuntimeError, "uninstall it first"):
                    cifar.uninstall_runtime()
            self.assertIs(model.model_spec_for_dataset, cifar_selector)
            self.assertTrue(cifar.runtime_installed())
        self.assertIs(model.model_spec_for_dataset, original_selector)
        self.assertIs(backend._torch_forward, original_forward)

    def test_numpy_rejected_and_fl_loop_uses_new_model(self):
        from sm9rrsfl import fl
        dataset = self.dataset()
        with resnet_gn.runtime():
            config = fl.ExperimentConfig(method="fedavg", num_clients=1, rounds=1,
                malicious_ratio=0., attack="none", batch_size=2, lr=.001,
                compute_backend="numpy", device="cpu", early_stop=False)
            with self.assertRaisesRegex(ValueError, "explicit compute_backend"):
                fl.run_experiment(dataset, config)
            for name in ("accuracy", "targeted_metrics", "local_train_delta", "predict",
                         "vector_to_params", "alternating_minimization_delta"):
                with self.subTest(api=name), self.assertRaisesRegex(ValueError, "NumPy"):
                    getattr(model, name)(None, spec=resnet_gn.SPEC)
            from dataclasses import replace
            checkpoints = []
            result = fl.run_experiment(dataset, replace(config, compute_backend="torch", checkpoint_interval=1),
                                        checkpoint_callback=checkpoints.append)
            self.assertEqual(result.config.rounds, 1)
            self.assertTrue(checkpoints)
            self.assertEqual(checkpoints[-1]["params"].shape, (resnet_gn.SPEC.parameter_size,))

    def test_import_has_no_runtime_side_effects(self):
        subprocess.run([sys.executable, "-c", """
from sm9rrsfl import model, fl, torch_backend
before = (model.model_spec_for_dataset, fl.model_spec_for_dataset, model.init_params,
          fl.init_params, torch_backend._torch_forward, torch_backend._parameter_shapes)
import fashion_resnet_gn
after = (model.model_spec_for_dataset, fl.model_spec_for_dataset, model.init_params,
         fl.init_params, torch_backend._torch_forward, torch_backend._parameter_shapes)
assert before == after
assert not fashion_resnet_gn.runtime_installed()
assert fashion_resnet_gn.protocol_descriptor()['dataset'] == 'fashion_mnist'
"""], check=True, capture_output=True, text=True)

    def test_six_methods_accept_same_fashion_shape_and_task_identity(self):
        from dataclasses import replace
        from sm9rrsfl import fl
        original = self.dataset()
        # Three clients are sufficient for the unmodified clean Krum branch.
        dataset = replace(original, x_train=np.tile(original.x_train, (3, 1, 1, 1)),
                          y_train=np.tile(original.y_train, 3))
        config = fl.ExperimentConfig(num_clients=3, rounds=1, malicious_ratio=0.,
            attack="none", batch_size=2, lr=.001, compute_backend="torch", device="cpu",
            crypto_mode="simulated", early_stop=False, checkpoint_interval=1,
            vert_projection_dim=8, vert_predict_epochs=1)
        with resnet_gn.runtime():
            for method in ("sm9rrs", "vert", "alignins", "krum", "ding13", "fedavg"):
                with self.subTest(method=method):
                    checkpoints = []
                    result = fl.run_experiment(dataset, replace(config, method=method),
                                               checkpoint_callback=checkpoints.append)
                    self.assertEqual([r.round for r in result.records], [0, 1])
                    self.assertTrue(np.isfinite(result.final_accuracy))
                    state = checkpoints[-1]
                    self.assertEqual(state["params"].shape, (resnet_gn.SPEC.parameter_size,))
                    self.assertEqual(state["params"].dtype, np.float32)
                    if method == "sm9rrs":
                        self.assertTrue(result.diagnostics)
                        # Client tags are cryptographic pseudonyms; the task
                        # identity is preserved separately in crypto state.
                        self.assertEqual(len({d.task_tag for d in result.diagnostics}), 3)
                        self.assertIn("fashion_mnist", state["crypto_state"].finalized_task_ids)
                    del checkpoints, result, state
                    gc.collect()

    def test_serialized_round_checkpoint_resume_matches_uninterrupted(self):
        from sm9rrsfl import fl

        class InterruptedForTest(RuntimeError):
            pass

        config = fl.ExperimentConfig(method="fedavg", num_clients=1, rounds=3,
            malicious_ratio=0., attack="none", batch_size=2, lr=.001,
            compute_backend="torch", device="cpu", early_stop=False, checkpoint_interval=1)
        dataset = self.dataset()
        saved = []

        def interrupt_after_round_one(state):
            if state["completed_round"] == 1:
                saved.append(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
                raise InterruptedForTest()

        with resnet_gn.runtime():
            with self.assertRaises(InterruptedForTest):
                fl.run_experiment(dataset, config, checkpoint_callback=interrupt_after_round_one)
            checkpoint = pickle.loads(saved.pop())
            self.assertEqual(checkpoint["params"].shape, (resnet_gn.SPEC.parameter_size,))
            resumed_states = []
            resumed = fl.run_experiment(dataset, config, resume_state=checkpoint,
                                         checkpoint_callback=resumed_states.append)
            self.assertEqual([state["completed_round"] for state in resumed_states], [2, 3])
            uninterrupted_states = []
            uninterrupted = fl.run_experiment(dataset, config,
                                               checkpoint_callback=uninterrupted_states.append)
            self.assertEqual(resumed.records, uninterrupted.records)
            np.testing.assert_array_equal(resumed_states[-1]["params"], uninterrupted_states[-1]["params"])


if __name__ == "__main__":
    unittest.main()
