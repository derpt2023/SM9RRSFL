"""Fashion input integrity and split isolation, using synthetic IDX archives."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
import gzip
import hashlib
import io
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

import fashion_dataset as fashion
from sm9rrsfl.datasets import ImageDataset, stratified_training_three_way_split
from sm9rrsfl.ours_calibration import OursCalibrationError


def idx_blob(kind, count, *, magic=None, rows=28, cols=28, payload=None):
    if kind.endswith("images"):
        header = struct.pack(">IIII", 2051 if magic is None else magic, count, rows, cols)
        if payload is None:
            payload = np.arange(count * rows * cols).astype(np.uint8).tobytes()
    else:
        header = struct.pack(">II", 2049 if magic is None else magic, count)
        if payload is None:
            payload = np.tile(np.arange(10, dtype=np.uint8), count // 10).tobytes()
    return gzip.compress(header + payload, mtime=0)


class DatasetFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "fashion"
        self.directory.mkdir()
        self.blobs = {kind: idx_blob(kind, 200 if kind.startswith("train_") else 100)
                      for kind in fashion.FASHION_FILES}
        self.files = {kind: (filename, hashlib.md5(self.blobs[kind]).hexdigest())
                      for kind, (filename, _) in fashion.FASHION_FILES.items()}
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(fashion, "FASHION_FILES", self.files))
        stack.enter_context(mock.patch.object(fashion, "TRAIN_SAMPLES", 200))
        stack.enter_context(mock.patch.object(fashion, "TEST_SAMPLES", 100))
        self.spec = {"dataset": {"name": "fashion_mnist", "data_dir": str(self.directory),
                        "download": False, "train_samples": 200, "test_samples": 100,
                        "validation_fraction": .05, "split_seed": 20261001},
                     "shared_parameters": {"attack": "alternating_minimization",
                        "attack_source_label": 5, "attack_target_label": 7, "attack_target_count": 1}}

    def cache(self):
        for kind, (filename, _) in self.files.items():
            (self.directory / filename).write_bytes(self.blobs[kind])

    def replace_blob(self, kind, blob):
        self.blobs[kind] = blob
        filename = self.files[kind][0]
        self.files[kind] = (filename, hashlib.md5(blob).hexdigest())

    def network(self, url, **kwargs):
        filename = url.rsplit("/", 1)[-1]
        kind = next(key for key, (name, _) in self.files.items() if name == filename)
        self.assertTrue(url.startswith("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/"))
        return io.BytesIO(self.blobs[kind])


class SourceTests(unittest.TestCase):
    def test_official_content_identity_and_semantics_are_fixed(self):
        self.assertEqual((fashion.TRAIN_SAMPLES, fashion.TEST_SAMPLES), (60000, 10000))
        self.assertEqual({kind: pair[1] for kind, pair in fashion.FASHION_FILES.items()}, {
            "train_images": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
            "train_labels": "25c81989df183df01b3e8a0aad5dffbe",
            "test_images": "bef4ecab320f06d8554ea6380940ec79",
            "test_labels": "bb300cfdad3c16e7a12a480ee83cd310"})
        self.assertEqual((fashion.LABELS[5], fashion.LABELS[7]), ("Sandal", "Sneaker"))


class CacheTests(DatasetFixture):
    def test_valid_cache_loads_native_nchw_unit_float_and_integral_labels(self):
        self.cache()
        with mock.patch.object(fashion, "urlopen", side_effect=AssertionError("network forbidden")):
            data = fashion.load_fashion_mnist(self.directory)
        self.assertEqual((data.name, data.input_shape, data.num_classes), ("fashion_mnist", (1, 28, 28), 10))
        self.assertEqual(data.x_train.shape, (200, 1, 28, 28))
        self.assertEqual(data.x_test.shape, (100, 1, 28, 28))
        self.assertEqual(data.x_train.dtype, np.float32)
        self.assertEqual(data.y_train.dtype, np.int64)
        self.assertEqual((data.x_train.min(), data.x_train.max()), (0., 1.))
        self.assertEqual(data.x_train[0, 0, 0, 1], np.float32(1) / np.float32(255))

    def test_wrong_dataset_or_corruption_is_not_overwritten_or_downloaded(self):
        path = self.directory / self.files["train_images"][0]
        path.write_bytes(b"same IDX filename, but wrong MNIST content")
        before = path.read_bytes()
        with mock.patch.object(fashion, "urlopen") as download:
            with self.assertRaisesRegex(ValueError, "checksum mismatch.*MNIST"):
                fashion.load_fashion_mnist(self.directory, download=True)
        download.assert_not_called()
        self.assertEqual(path.read_bytes(), before)

    def test_missing_files_without_download_fail(self):
        with self.assertRaisesRegex(FileNotFoundError, "archive is missing"):
            fashion.load_fashion_mnist(self.directory)

    def test_verified_download_then_cached_load_fetches_each_archive_once(self):
        with mock.patch.object(fashion, "urlopen", side_effect=self.network) as network:
            first = fashion.load_fashion_mnist(self.directory, download=True)
            second = fashion.load_fashion_mnist(self.directory, download=True)
        self.assertEqual(network.call_count, 4)
        np.testing.assert_array_equal(first.x_train, second.x_train)
        self.assertEqual(list(self.directory.glob("*.partial")), [])
        for kind, (filename, _) in self.files.items():
            self.assertEqual((self.directory / filename).read_bytes(), self.blobs[kind])

    def test_failed_download_checksum_never_publishes_target(self):
        with mock.patch.object(fashion, "urlopen", return_value=io.BytesIO(b"truncated response")):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                fashion.load_fashion_mnist(self.directory, download=True)
        self.assertFalse((self.directory / self.files["train_images"][0]).exists())
        self.assertEqual(list(self.directory.glob("*.partial")), [])

    def test_structurally_invalid_download_never_publishes_even_with_matching_checksum(self):
        self.replace_blob("train_images", idx_blob("train_images", 199))
        with mock.patch.object(fashion, "urlopen", side_effect=self.network):
            with self.assertRaisesRegex(ValueError, "header/count/shape"):
                fashion.load_fashion_mnist(self.directory, download=True)
        self.assertFalse((self.directory / self.files["train_images"][0]).exists())
        self.assertEqual(list(self.directory.glob("*.partial")), [])

    def test_interrupted_download_cleans_only_own_partial_and_can_resume(self):
        stale = self.directory / ".previous-crash.partial"
        stale.write_bytes(b"preserve unrelated file")
        with mock.patch.object(fashion, "urlopen", side_effect=OSError("connection lost")):
            with self.assertRaisesRegex(OSError, "connection lost"):
                fashion.load_fashion_mnist(self.directory, download=True)
        self.assertEqual(list(self.directory.glob("*.partial")), [stale])
        with mock.patch.object(fashion, "urlopen", side_effect=self.network):
            fashion.load_fashion_mnist(self.directory, download=True)
        self.assertEqual(stale.read_bytes(), b"preserve unrelated file")

    def test_concurrent_loaders_share_cache_without_duplicate_downloads(self):
        calls = []
        call_lock = threading.Lock()
        def slow_network(url, **kwargs):
            with call_lock:
                calls.append(url)
            time.sleep(.01)
            return self.network(url, **kwargs)
        with mock.patch.object(fashion, "urlopen", side_effect=slow_network):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: fashion.load_fashion_mnist(self.directory, download=True), range(2)))
        self.assertEqual(len(calls), 4)
        np.testing.assert_array_equal(results[0].x_train, results[1].x_train)

    def test_symlink_archives_and_directory_are_rejected(self):
        self.cache()
        path = self.directory / self.files["train_images"][0]
        saved = self.directory / "outside-copy.gz"
        path.rename(saved)
        path.symlink_to(saved)
        with self.assertRaisesRegex(ValueError, "regular file"):
            fashion.load_fashion_mnist(self.directory)
        alias = self.directory.parent / "alias"
        alias.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "directory must not be a symlink"):
            fashion.load_fashion_mnist(alias)

    def test_invalid_headers_payloads_and_labels_rejected_after_checksum(self):
        cases = [
            ("train_images", idx_blob("train_images", 200, magic=2049), "header/count/shape"),
            ("train_images", idx_blob("train_images", 200, rows=27), "header/count/shape"),
            ("train_images", idx_blob("train_images", 200, payload=b"short"), "payload length"),
            ("train_images", idx_blob("train_images", 200, payload=bytes(200 * 784 + 1)), "payload length"),
            ("train_labels", idx_blob("train_labels", 200, payload=bytes([10]) * 200), "outside"),
            ("train_labels", idx_blob("train_labels", 200, payload=bytes(200)), "class counts"),
            ("train_images", b"not gzip", "invalid gzip"),
        ]
        for kind, blob, message in cases:
            with self.subTest(message=message):
                old_blob, old_entry = self.blobs[kind], self.files[kind]
                self.replace_blob(kind, blob)
                self.cache()
                with self.assertRaisesRegex(ValueError, message):
                    fashion.load_fashion_mnist(self.directory)
                self.blobs[kind], self.files[kind] = old_blob, old_entry


class SplitTests(DatasetFixture):
    def test_contract_has_source_hashes_target_semantics_and_reproducible_disjoint_split(self):
        self.cache()
        split, contract = fashion.load_split(self.spec)
        repeat, repeated_contract = fashion.load_split(deepcopy(self.spec))
        self.assertEqual(contract, repeated_contract)
        self.assertEqual((len(split.train_indices), len(split.calibration_indices), len(split.attack_indices)), (180, 10, 10))
        groups = [set(indices) for indices in (split.train_indices, split.calibration_indices, split.attack_indices)]
        self.assertEqual(len(set.union(*groups)), 200)
        self.assertTrue(all(not groups[i].intersection(groups[j]) for i in range(3) for j in range(i)))
        self.assertIs(split.main_dataset.x_train, split.calibration_dataset.x_train)
        self.assertIs(split.main_dataset.x_attack, split.calibration_dataset.x_attack)
        self.assertEqual(len(split.main_dataset.x_test), 100)
        self.assertEqual(len(split.calibration_dataset.x_test), 10)
        self.assertEqual(contract["split_seed"], self.spec["dataset"]["split_seed"])
        self.assertFalse(contract["official_test_used_for_selection"])
        self.assertEqual(contract["targeted_attack"]["source_name"], "Sandal")
        self.assertEqual(contract["targeted_attack"]["target_name"], "Sneaker")
        self.assertEqual(contract["targeted_attack"]["available_source_samples"],
                         {"attack_auxiliary": 1, "validation": 1, "official_evaluation": 10})
        self.assertFalse(contract["preprocessing"]["estimated_normalization_statistics"])
        for kind, blob in self.blobs.items():
            self.assertEqual(contract["source_checksums"][kind]["sha256"], hashlib.sha256(blob).hexdigest())
        self.assertEqual(contract["validation_arrays"]["x_train"], contract["final_arrays"]["x_train"])
        self.assertEqual(contract["final_array_layout"]["x_train"], {"shape": [180, 1, 28, 28], "dtype": "float32"})
        np.testing.assert_array_equal(split.train_indices, repeat.train_indices)

    def test_changed_seed_changes_identity_but_not_partition_counts(self):
        self.cache()
        first, first_contract = fashion.load_split(self.spec)
        self.spec["dataset"]["split_seed"] += 1
        second, second_contract = fashion.load_split(self.spec)
        self.assertFalse(np.array_equal(first.train_indices, second.train_indices))
        self.assertNotEqual(first_contract["train_indices_digest"], second_contract["train_indices_digest"])
        self.assertEqual(first_contract["train_samples"], second_contract["train_samples"])

    def test_official_test_changes_do_not_change_training_or_validation(self):
        self.cache()
        _, first = fashion.load_split(self.spec)
        self.replace_blob("test_images", idx_blob("test_images", 100, payload=bytes(100 * 784)))
        self.cache()
        _, second = fashion.load_split(self.spec)
        self.assertEqual(first["validation_arrays"], second["validation_arrays"])
        self.assertEqual(first["train_indices_digest"], second["train_indices_digest"])
        self.assertNotEqual(first["final_arrays"]["x_test"], second["final_arrays"]["x_test"])

    def test_invalid_spec_and_insufficient_attack_samples_fail_before_training(self):
        self.cache()
        for field, value in (("name", "mnist"), ("train_samples", 100), ("validation_fraction", .1),
                             ("attack_fraction", .1), ("split_seed", True)):
            broken = deepcopy(self.spec)
            broken["dataset"][field] = value
            with self.subTest(field=field), self.assertRaises((ValueError, OursCalibrationError)):
                fashion.load_split(broken)
        for field, value in (("attack_source_label", 10), ("attack_target_label", 5),
                             ("attack_target_count", 0), ("attack_target_count", 200)):
            broken = deepcopy(self.spec)
            broken["shared_parameters"][field] = value
            with self.subTest(field=field), self.assertRaises((ValueError, OursCalibrationError)):
                fashion.load_split(broken)

    def test_complete_balanced_training_cardinality_is_54000_3000_3000(self):
        # Tiny spatial tensors exercise the same existing stratifier without
        # allocating the full image corpus; this is a synthetic cardinality test.
        dataset = ImageDataset(np.zeros((60000, 1, 1, 1), dtype=np.float32),
                    np.repeat(np.arange(10), 6000), np.zeros((10000, 1, 1, 1), dtype=np.float32),
                    np.repeat(np.arange(10), 1000), name="fashion_mnist")
        split = stratified_training_three_way_split(dataset, seed=20261001)
        self.assertEqual((len(split.train_indices), len(split.calibration_indices), len(split.attack_indices)), (54000, 3000, 3000))
        self.assertEqual(np.count_nonzero(split.calibration_dataset.y_test == 5), 300)
        self.assertEqual(np.count_nonzero(split.main_dataset.y_attack == 5), 300)


if __name__ == "__main__":
    unittest.main()
