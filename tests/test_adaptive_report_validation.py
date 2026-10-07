"""Report-boundary regression tests; no training or CUDA is required."""
from copy import deepcopy
from dataclasses import asdict, replace
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cifar_adaptive_reporting as cifar_reporting
import fashion_adaptive_reporting as fashion_reporting
import run_cifar_six_from_scratch as base
from adaptive_report_validation import aggregation_weight_tolerance
from tests.test_cifar_six_pipeline import synthetic_run


REPORTERS = (cifar_reporting, fashion_reporting)


def make_run(**changes):
    config = base.fl.ExperimentConfig(method="vert", rounds=150, num_clients=100,
                                     malicious_ratio=.8, detector_window=10,
                                     attack_start_round=12)
    run = synthetic_run(config)
    if changes:
        run = replace(run, records=[replace(row, **changes) if row.round == 7 else row
                                    for row in run.records])
    return run


def aggregate_mass(count):
    clients = [str(index) for index in range(count)]
    coefficients = base.fl._normalized_client_coefficients(clients, [.1] * count)
    _, mass = base.fl._aggregation_weight_diagnostics(
        {identity: 1 for identity in clients}, set(clients), coefficients)
    return mass


class AggregationToleranceTests(unittest.TestCase):
    def test_real_aggregation_roundoff_and_original_health_are_preserved(self):
        # The real normalizer and diagnostic sum, not a hand-written 1+epsilon.
        mass = aggregate_mass(9)
        self.assertGreater(mass, 1.)
        self.assertLessEqual(mass, 1. + 1e-9)
        run = make_run(malicious_weight_mass=mass)
        before = deepcopy(asdict(run))
        health = base.metrics(run)
        self.assertTrue(health["healthy"])
        for reporting in REPORTERS:
            original = reporting._inspect_result
            with self.subTest(reporter=reporting.__name__):
                with self.assertRaisesRegex(reporting.ReportIntegrityError, "malicious_weight_mass"):
                    original(run, 150)
                with aggregation_weight_tolerance(reporting) as audit:
                    self.assertEqual(reporting._inspect_result(run, 150), (True, None, health))
                self.assertIs(reporting._inspect_result, original)
                self.assertEqual(len(audit), 1)
                entry = audit[0]
                self.assertEqual(entry["raw_value"], mass)
                self.assertEqual(entry["round"], 7)
                self.assertEqual(entry["field"], "malicious_weight_mass")
                self.assertEqual(entry["config"], asdict(run.config))
                self.assertEqual(entry["semantic_config_sha256"],
                                 base.digest(base.semantic_config(run.config)))
                self.assertTrue(entry["raw_value_preserved"])
                self.assertEqual(asdict(run), before)

    def test_ten_point_one_terms_keep_the_actual_sum_without_adjustment(self):
        # Ten 0.1 terms happen to round down on this path. They must not be
        # changed to one or described as an upper-bound tolerance event.
        mass = aggregate_mass(10)
        self.assertEqual(mass, sum([.1] * 10))
        self.assertLess(mass, 1.)
        run = make_run(malicious_weight_mass=mass)
        for reporting in REPORTERS:
            expected = reporting._inspect_result(run, 150)
            with aggregation_weight_tolerance(reporting) as audit:
                self.assertEqual(reporting._inspect_result(run, 150), expected)
            self.assertEqual(audit, [])
            self.assertEqual(run.records[7].malicious_weight_mass, mass)

    def test_both_diagnostics_include_boundary_and_preserve_health_failures(self):
        run = make_run(honest_weight_loss=1 + 1e-9, malicious_weight_mass=1 + 1e-9)
        run = replace(run, nonfinite_updates=1)
        expected = base.metrics(run)
        self.assertFalse(expected["healthy"])
        for reporting in REPORTERS:
            with aggregation_weight_tolerance(reporting) as audit:
                self.assertEqual(reporting._inspect_result(run, 150), (True, None, expected))
            self.assertEqual({row["field"] for row in audit},
                             {"honest_weight_loss", "malicious_weight_mass"})
            self.assertTrue(all(row["raw_value"] == 1 + 1e-9 for row in audit))

    def test_true_invalid_weights_remain_fatal(self):
        for reporting in REPORTERS:
            for field in ("honest_weight_loss", "malicious_weight_mass"):
                for value in (-1e-16, math.nextafter(1 + 1e-9, math.inf),
                              float("nan"), float("inf"), "1.0"):
                    with self.subTest(reporter=reporting.__name__, field=field, value=value):
                        with aggregation_weight_tolerance(reporting) as audit:
                            with self.assertRaisesRegex(reporting.ReportIntegrityError, field):
                                reporting._inspect_result(make_run(**{field: value}), 150)
                        self.assertEqual(audit, [])

    def test_accuracy_asr_and_confidence_bounds_are_not_relaxed(self):
        for reporting in REPORTERS:
            for field in ("accuracy", "attack_target_success_rate", "attack_target_confidence"):
                with self.subTest(reporter=reporting.__name__, field=field):
                    run = make_run(malicious_weight_mass=aggregate_mass(9),
                                   **{field: math.nextafter(1., math.inf)})
                    with aggregation_weight_tolerance(reporting) as audit:
                        with self.assertRaisesRegex(reporting.ReportIntegrityError, field):
                            reporting._inspect_result(run, 150)
                    self.assertEqual(audit, [])

    def test_original_integrity_checks_and_incomplete_policy_remain(self):
        run = make_run(malicious_weight_mass=aggregate_mass(9))
        broken = (
            replace(run, final_accuracy=.9),
            replace(run, records=run.records[:-1]),
            replace(run, runtime_seconds=-1),
            replace(run, records=[replace(row, method="fedavg") if row.round == 2 else row
                                  for row in run.records]),
        )
        for reporting in REPORTERS:
            for bad in broken:
                with aggregation_weight_tolerance(reporting) as audit:
                    with self.assertRaises(reporting.ReportIntegrityError):
                        reporting._inspect_result(bad, 150)
                self.assertEqual(audit, [])
            partial = replace(run, stopped_round=29, records=run.records[:30])
            with aggregation_weight_tolerance(reporting) as audit:
                self.assertEqual(reporting._inspect_result(partial, 150),
                                 (False, "incomplete_rounds", None))
            self.assertEqual(audit, [])

    def test_health_is_computed_from_raw_run_and_invalid_reasons_still_fail(self):
        run = make_run(malicious_weight_mass=aggregate_mass(9))
        original_metrics = base.metrics
        seen = []

        def metrics(raw):
            seen.append(raw)
            return {**original_metrics(raw), "healthy": False,
                    "reasons": ["invalid_weight_metrics"]}

        for reporting in REPORTERS:
            with patch.object(base, "metrics", side_effect=metrics):
                with aggregation_weight_tolerance(reporting) as audit:
                    with self.assertRaisesRegex(reporting.ReportIntegrityError, "invalid scientific metrics"):
                        reporting._inspect_result(run, 150)
            self.assertEqual(audit, [])
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(raw is run for raw in seen))

    def test_nested_context_and_exception_restore_inspectors(self):
        original = cifar_reporting._inspect_result
        fashion_original = fashion_reporting._inspect_result
        run = make_run(malicious_weight_mass=aggregate_mass(9))
        with self.assertRaisesRegex(RuntimeError, "test abort"):
            with aggregation_weight_tolerance(cifar_reporting):
                outer = cifar_reporting._inspect_result
                with aggregation_weight_tolerance(cifar_reporting) as inner_audit:
                    self.assertTrue(cifar_reporting._inspect_result(run, 150)[0])
                self.assertEqual(len(inner_audit), 1)
                self.assertIs(cifar_reporting._inspect_result, outer)
                self.assertIs(fashion_reporting._inspect_result, fashion_original)
                raise RuntimeError("test abort")
        self.assertIs(cifar_reporting._inspect_result, original)

    def test_both_full_report_exports_retain_exact_raw_values_and_sources(self):
        spec = json.loads(Path("configs/cifar10_six_relative_best_five_day_v7.json").read_text())
        spec["schema_version"] = 8
        spec["shared_parameters"].update(rounds=150, detector_window=10, attack_start_round=12)
        spec["final"] = {"seeds": [4101], "scenarios": [
            {"partition": "iid", "malicious_ratio": .8}]}
        selected = {method: rows[0]["candidate_id"] for method, rows in spec["candidates"].items()}
        tasks = [dict(task, fingerprint=f"task-{index}")
                 for index, task in enumerate(base.build_tasks(spec, "final", selected))]
        mass = aggregate_mass(9)
        results, statuses = {}, []
        for task in tasks:
            run = synthetic_run(base.fl.ExperimentConfig(**task["config"]))
            if task["method"] == "vert":
                run = replace(run, records=[replace(row, malicious_weight_mass=mass)
                              if row.round == 7 else row for row in run.records], nonfinite_updates=1)
            results.setdefault(task["candidate"]["candidate_id"], []).append(run)
            statuses.append({"task_id": task["task_id"], "status": "complete"})
        validation = {"status": "needs_ours_target_development"}
        before = deepcopy(results)
        for reporting in REPORTERS:
            with self.subTest(reporter=reporting.__name__), tempfile.TemporaryDirectory() as directory:
                report_spec = deepcopy(spec)
                if reporting is fashion_reporting:
                    report_spec["protocol"] = "fashion-mnist-resnet18-gn-tpe-v1"
                    report_spec["dataset"]["name"] = "fashion_mnist"
                    report_spec["model"] = {"architecture": "fashion_resnet18_gn", "group_norm_groups": 2}
                output = Path(directory)
                for task in tasks:
                    folder = output / "tasks" / task["task_id"]
                    folder.mkdir(parents=True)
                    (folder / "task.json").write_text(json.dumps(task))
                hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in output.rglob("task.json")}
                with aggregation_weight_tolerance(reporting) as audit:
                    with patch.object(reporting, "_plots", return_value=[]):
                        summary = reporting.write_final_report(
                            output, report_spec, selected, results, tasks, statuses, validation, {"answer": "Y"})
                self.assertEqual(len(audit), 1)
                self.assertEqual(summary["status"], "completed_with_health_failures")
                self.assertEqual(summary["methods"]["vert"]["healthy_runs"], 0)
                self.assertEqual(summary["selected"], selected)
                self.assertFalse(summary["parameters_reselected"])
                with (output / "final_results" / "rounds.csv").open(newline="") as handle:
                    row = next(row for row in csv.DictReader(handle)
                               if row["method"] == "vert" and row["round"] == "7")
                self.assertEqual(float(row["malicious_weight_mass"]), mass)
                self.assertGreater(float(row["malicious_weight_mass"]), 1.)
                self.assertEqual(float(row["accuracy"]), .8)
                self.assertEqual(float(row["attack_target_success_rate"]), .05)
                self.assertTrue(all(hashlib.sha256(path.read_bytes()).hexdigest() == digest
                                    for path, digest in hashes.items()))
                self.assertEqual(results, before)


if __name__ == "__main__":
    unittest.main()
