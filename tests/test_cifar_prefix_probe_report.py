"""Synthetic three-round boundary evidence; no GPU, dataset or training reads."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import cifar_prefix_probe_report as report
from sm9rrsfl.fl import ExperimentConfig
import tests.test_cifar_history_forensics as client_fixture


def fingerprint(name, shape=None, dtype="<f4"):
    return {"sha256": hashlib.sha256(name.encode()).hexdigest(), "dtype": dtype, "shape": [20] if shape is None else shape}


def payload(task):
    config = task["config"]
    ids = ["client-" + str(i) for i in range(config["num_clients"])]
    def model(rd):
        return fingerprint("model" + str(rd))
    def record(rd):
        return {"round": rd, "accuracy": .2 + rd * .01, "error": .8 - rd * .01,
            "attack_target_success_rate": .1, "attack_target_confidence": .09,
            "false_positive_revocations": 0, "true_positive_revocations": 0,
            "malicious_weight_mass": 0., "honest_weight_loss": 0., "accepted_updates": len(ids) if rd else 0,
            "rejected_updates": 0, "blacklisted_clients": 0, "nonfinite_updates": 0}
    fixture = client_fixture.HistoryForensicsTests()
    rounds, checkpoints, evaluations = [], [], []
    for rd in range(4):
        checkpoints.append({"round": rd, "model": model(rd), "record": record(rd),
                            "diagnostic_count": len(ids) if rd else 0, "callback_count": 2 if rd == 3 else 1})
        for kind in ("accuracy", "target"):
            size = 4 if kind == "accuracy" else config["attack_target_count"]
            evaluations.append({"round": rd, "kind": kind, "model_input": model(rd),
                "batches": [{"batch": 0, "logits": fingerprint("logits" + str(rd) + kind, [size, 10]),
                             "predictions": fingerprint("predictions" + str(rd) + kind, [size], "<i8")}],
                "value": record(rd)["accuracy"] if kind == "accuracy" else [.1, .09]})
        if not rd:
            continue
        clients, diagnostics = [], []
        for index, cid in enumerate(ids):
            clients.append({"client_id": cid, "samples": 5, "model_input": model(rd - 1),
                "update": fingerprint("update" + str(rd) + cid), "stats": {"loss": .25, "samples": 5},
                "training_seed": config["seed"] + rd * 1009 + index,
                "learning_rate": config["lr"] * config["lr_decay"] ** (rd - 1), "epochs": 1,
                "batch_size": config["batch_size"], "epoch_indices": [fingerprint("epoch" + str(rd) + cid, [5], "<i8")],
                "minibatch_sizes_per_epoch": [5]})
            d = asdict(fixture.diagnostic(rd, cid, malicious=False))
            d.pop("task_tag")
            d.update(decision_reason="clean_warmup", aggregation_weight=1 / len(ids))
            diagnostics.append(d)
        rounds.append({"round": rd, "clients": clients,
            "candidate_order": list(ids), "verified_order": list(ids), "aggregate_order": list(ids),
            "coefficients": {"order": list(ids), "by_client": dict.fromkeys(ids, 1 / len(ids))},
            "aggregate": fingerprint("aggregate" + str(rd)), "post_model": model(rd), "record": record(rd),
            "diagnostics": diagnostics})
    return {"schema": report.SCHEMA, "task_id": task["task_id"], "task_fingerprint": task["fingerprint"],
        "configuration": deepcopy(config), "observation_contract": {"epoch_indices": "reconstructed"},
        "data": {k: None if k.endswith("attack") else fingerprint(k, [4 if k.endswith("test") else 10] + ([] if k.startswith("y") else [3, 32, 32])) for k in report.DATA_FIELDS},
        "model_spec": {"kind": "synthetic-cnn", "num_classes": 10}, "initial_model": model(0),
        "partition": {"strategy": config["partition"], "seed": config["seed"], "alpha": config["dirichlet_alpha"],
            "clients": [{"client_id": cid, "samples": 5, "indices": fingerprint(config["partition"] + cid, [5], "<i8")} for cid in ids]},
        "rounds": rounds, "evaluations": evaluations, "checkpoints": checkpoints}


class PrefixProbeReportTests(unittest.TestCase):
    def setUp(self):
        self.output = Path("/nonexistent/read-only-prefix-test")
        self.uuid = "GPU-synthetic-physical-id"
        self.metadata = {"actual_compute_device": {"name": "synthetic GPU", "uuid": self.uuid, "logical_device": "cuda:0"},
            "requested_device": "cuda:0", "environment": {"CUDA_VISIBLE_DEVICES": self.uuid},
            "nvidia": {"gpus": [{"uuid": self.uuid, "name": "synthetic GPU", "driver_version": "550"}]},
            "torch": {"logical_cuda_devices": [{"logical_index": 0, "uuid": self.uuid, "name": "synthetic GPU"}],
                      "version": "test", "deterministic_algorithms": False}}
        self.environment = report.matched.normalized_environment(self.metadata)
        self.tasks, self.artifacts = [], {}
        for partition in ("iid", "dirichlet"):
            for repeat in (1, 2, 3):
                task_id = "prefix_" + partition + "_repeat" + str(repeat)
                config = ExperimentConfig(method="sm9rrs", rounds=3, num_clients=2, malicious_ratio=0., partition=partition,
                    detector_window=20, attack_start_round=25, seed=2026093001, local_epochs=1, batch_size=32, lr_decay=.99)
                task = {"task_id": task_id, "partition": partition, "repeat": repeat,
                    "fingerprint": hashlib.sha256(task_id.encode()).hexdigest(), "config": asdict(config)}
                self.tasks.append(task)
                self.artifacts[task_id] = {"status": "complete", "task_fingerprint": task["fingerprint"],
                    "observations": payload(task), "environment": deepcopy(self.metadata), "gpu_uuid": self.uuid,
                    "wall_seconds": 1.25, "artifact_fingerprint": "f" * 64,
                    "fresh_start": True, "checkpoints_used": False}
        historical = {p: [deepcopy(c["record"]) for c in self.artifacts["prefix_" + p + "_repeat1"]["observations"]["checkpoints"]]
                      for p in ("iid", "dirichlet")}
        self.manifest = {"fingerprint": "manifest-fp", "protocol": "cifar-prefix-probe-v1", "same_gpu_uuid": self.uuid,
            "source_sha256": {"original.py": "x"}, "reference": {"historical_prefix": historical,
            "execution_environment": deepcopy(self.environment)}}
        self.protocol = SimpleNamespace(read_study=mock.Mock(return_value=(self.manifest, self.tasks)),
            source_hashes=mock.Mock(return_value=self.manifest["source_sha256"]),
            load_completed=mock.Mock(side_effect=lambda output, task: self.artifacts.get(task["task_id"])))
        patcher = mock.patch.dict(sys.modules, {"cifar_prefix_probe_protocol": self.protocol})
        patcher.start()
        self.addCleanup(patcher.stop)

    def observations(self, partition="iid", repeat=2):
        return self.artifacts["prefix_" + partition + "_repeat" + str(repeat)]["observations"]

    def summarize(self):
        return report.summarize(self.output)

    def pair(self, result, partition="iid", repeat=2):
        return next(p for p in result["pairs"] if p["partition"] == partition and p["later_repeat"] == repeat)

    def test_six_complete_four_pairs_equal_without_health_or_cause_claim(self):
        summary = self.summarize()
        self.assertEqual((summary["complete_tasks"], summary["available_pairs"]), (6, 4))
        self.assertTrue(summary["all_pairs_equal"])
        self.assertTrue(all(row["old_H0_comparison"]["equal"] for row in summary["rows"]))
        for field in ("formal_qualification_assessed", "three_round_health_assessed", "automatic_next_stage", "cuda_cause_identified"):
            self.assertFalse(summary["decision"][field])
        self.protocol.read_study.assert_called_once_with(self.output, current_sources=True)

    def test_data_partition_and_initial_model_are_earlier_than_later_changes(self):
        observations = self.observations()
        observations["data"]["y_train"] = fingerprint("changed labels", [10], "<i8")
        observations["rounds"][0]["clients"][0]["update"] = fingerprint("changed update")
        result = self.pair(self.summarize())
        self.assertEqual(result["first_divergence"]["stage"], "data")
        observations["data"] = deepcopy(self.observations(repeat=1)["data"])
        observations["partition"]["clients"][0]["indices"] = fingerprint("changed partition", [5], "<i8")
        self.assertEqual(self.pair(self.summarize())["first_divergence"]["stage"], "partition")
        observations["partition"] = deepcopy(self.observations(repeat=1)["partition"])
        observations["initial_model"] = fingerprint("changed initial")
        observations["checkpoints"][0]["model"] = deepcopy(observations["initial_model"])
        for evaluation in observations["evaluations"]:
            if evaluation["round"] == 0:
                evaluation["model_input"] = deepcopy(observations["initial_model"])
        for client in observations["rounds"][0]["clients"]:
            client["model_input"] = deepcopy(observations["initial_model"])
        self.assertEqual(self.pair(self.summarize())["first_divergence"]["stage"], "initial_model")

    def test_actual_client_input_update_coefficients_aggregate_and_post_boundaries(self):
        for stage in ("client_inputs_and_reconstructed_order", "client_update", "coefficients", "aggregate", "post_model"):
            with self.subTest(stage=stage):
                task = self.tasks[1]
                self.artifacts[task["task_id"]]["observations"] = payload(task)
                observations = self.observations()
                rd = observations["rounds"][1]
                if stage == "client_inputs_and_reconstructed_order":
                    rd["clients"][0]["epoch_indices"][0] = fingerprint("different reconstructed order", [5], "<i8")
                elif stage == "client_update":
                    rd["clients"][0]["update"] = fingerprint("different delta")
                elif stage == "coefficients":
                    rd["coefficients"]["by_client"]["client-0"] = .45
                    rd["diagnostics"][0]["aggregation_weight"] = .45
                else:
                    rd[stage] = fingerprint("different " + stage)
                    if stage == "post_model":
                        observations["checkpoints"][2]["model"] = deepcopy(rd[stage])
                        for evaluation in observations["evaluations"]:
                            if evaluation["round"] == 2:
                                evaluation["model_input"] = deepcopy(rd[stage])
                        for client in observations["rounds"][2]["clients"]:
                            client["model_input"] = deepcopy(rd[stage])
                first = self.pair(self.summarize())["first_divergence"]
                self.assertEqual((first["stage"], first["round"]), (stage, 2))

    def test_evaluation_only_difference_not_inferred_as_training_or_cuda_cause(self):
        self.observations()["evaluations"][0]["batches"][0]["logits"] = fingerprint("changed logits", [4, 10])
        summary = self.summarize()
        first = self.pair(summary)["first_divergence"]
        self.assertEqual((first["stage"], first["round"]), ("evaluation_accuracy", 0))
        self.assertFalse(summary["decision"]["cuda_cause_identified"])

    def test_missing_task_and_failed_load_never_count_as_equal(self):
        self.artifacts.pop(self.tasks[1]["task_id"])
        summary = self.summarize()
        self.assertEqual(summary["complete_tasks"], 5)
        self.assertFalse(self.pair(summary)["available"])
        self.assertIsNone(self.pair(summary)["equal"])
        self.assertIsNone(summary["all_pairs_equal"])
        self.protocol.load_completed.side_effect = ValueError("failed worker artifact")
        summary = self.summarize()
        self.assertEqual(summary["complete_tasks"], 0)
        self.assertTrue(all(p["equal"] is None for p in summary["pairs"]))

    def test_missing_round_client_eval_fingerprint_or_diagnostic_invalidates_task(self):
        changes = [lambda p: p["rounds"].pop(), lambda p: p["rounds"][0]["clients"].pop(),
            lambda p: p["evaluations"].pop(), lambda p: p["rounds"][0]["aggregate"].pop("sha256"),
            lambda p: p["rounds"][0]["diagnostics"][0].pop("novelty_score"),
            lambda p: p["checkpoints"].pop()]
        for change in changes:
            with self.subTest(change=change):
                self.artifacts[self.tasks[1]["task_id"]]["observations"] = payload(self.tasks[1])
                change(self.observations())
                summary = self.summarize()
                self.assertEqual(summary["rows"][1]["status"], "invalid_evidence")
                self.assertIsNone(self.pair(summary)["equal"])

    def test_nonfinite_values_and_bad_boolean_do_not_emit_success_looking_json(self):
        for change in (lambda p: p["rounds"][0]["clients"][0]["stats"].update(loss=float("nan")),
                       lambda p: p["rounds"][0]["diagnostics"][0].update(history_admitted=1)):
            self.artifacts[self.tasks[1]["task_id"]]["observations"] = payload(self.tasks[1])
            change(self.observations())
            summary = self.summarize()
            self.assertEqual(summary["rows"][1]["status"], "invalid_evidence")
            json.dumps(summary, allow_nan=False)

    def test_cross_boundary_misalignment_invalidates_instead_of_comparing(self):
        changes = [
            lambda p: p["rounds"][0]["clients"][0].update(model_input=fingerprint("unrelated model")),
            lambda p: p["evaluations"][0].update(model_input=fingerprint("unrelated evaluation")),
            lambda p: p["evaluations"][0].update(value=.55),
            lambda p: p["evaluations"][0]["batches"][0].update(logits=fingerprint("bad class count", [4, 9])),
            lambda p: p["evaluations"][0]["batches"][0].update(predictions=fingerprint("bad sample count", [3], "<i8")),
            lambda p: p["evaluations"][1]["batches"][0].update(logits=fingerprint("wrong target coverage", [2, 10]), predictions=fingerprint("wrong coverage", [2], "<i8")),
            lambda p: p["rounds"][0]["diagnostics"][0].update(aggregation_weight=.123),
            lambda p: p["rounds"][0]["record"].update(nonfinite_updates=1),
            lambda p: p["rounds"][0]["clients"][0].update(update=fingerprint("wrong delta shape", [21])),
        ]
        for change in changes:
            with self.subTest(change=change):
                self.artifacts[self.tasks[1]["task_id"]]["observations"] = payload(self.tasks[1])
                change(self.observations())
                summary = self.summarize()
                self.assertEqual(summary["rows"][1]["status"], "invalid_evidence")
                self.assertIsNone(self.pair(summary)["equal"])

    def test_configuration_only_allows_logical_device_change(self):
        self.observations()["configuration"]["device"] = "cuda:0"
        self.assertEqual(self.summarize()["complete_tasks"], 6)
        self.observations()["configuration"]["sm9_workers"] += 1
        self.assertEqual(self.summarize()["rows"][1]["status"], "invalid_evidence")

    def test_artifact_must_establish_fresh_start_without_checkpoints(self):
        artifact = self.artifacts[self.tasks[1]["task_id"]]
        for key, bad in (("fresh_start", False), ("checkpoints_used", True)):
            with self.subTest(key=key):
                old = artifact[key]
                artifact[key] = bad
                summary = self.summarize()
                self.assertEqual(summary["rows"][1]["status"], "invalid_evidence")
                self.assertIsNone(self.pair(summary)["equal"])
                artifact[key] = old

    def test_evidence_or_source_change_during_read_blocks_all_comparisons(self):
        with mock.patch.object(report, "evidence_hashes", side_effect=[{"file": "before"}, {"file": "after"}]):
            summary = self.summarize()
        self.assertEqual(summary["status"], "changed_during_read")
        self.assertIsNone(summary["all_pairs_equal"])
        self.assertTrue(all(p["equal"] is None and not p["available"] for p in summary["pairs"]))
        self.protocol.source_hashes.side_effect = [{"source": "before"}, {"source": "after"}]
        summary = self.summarize()
        self.assertEqual(summary["status"], "changed_during_read")
        self.assertFalse(summary["input_evidence_unchanged"])

    def test_wrong_gpu_environment_or_artifact_task_identity_blocks_pair(self):
        original = deepcopy(self.artifacts[self.tasks[1]["task_id"]])
        for kind in ("gpu", "environment", "identity"):
            with self.subTest(kind=kind):
                self.artifacts[self.tasks[1]["task_id"]] = deepcopy(original)
                artifact = self.artifacts[self.tasks[1]["task_id"]]
                if kind == "gpu":
                    artifact["gpu_uuid"] = "GPU-another"
                elif kind == "environment":
                    artifact["environment"]["actual_compute_device"]["name"] = "another GPU"
                else:
                    artifact["observations"]["task_fingerprint"] = "wrong"
                summary = self.summarize()
                self.assertEqual(summary["rows"][1]["status"], "invalid_evidence")
                self.assertFalse(self.pair(summary)["available"])

    def test_duplicate_repeats_and_duplicate_task_ids_do_not_form_complete_matrix(self):
        self.tasks[1]["repeat"] = 1
        summary = self.summarize()
        self.assertNotEqual(summary["status"], "complete")
        self.assertIsNone(summary["all_pairs_equal"])
        self.assertFalse(self.pair(summary)["available"])

    def test_old_H0_scalar_difference_is_separate_from_equal_fresh_repeats(self):
        self.manifest["reference"]["historical_prefix"]["iid"][2]["accuracy"] = .23
        summary = self.summarize()
        self.assertTrue(summary["all_pairs_equal"])
        old = summary["rows"][0]["old_H0_comparison"]
        self.assertFalse(old["equal"])
        self.assertEqual(old["first_divergence"], {"round": 2, "field": "accuracy", "probe": .22, "old_H0": .23})
        self.manifest["reference"]["historical_prefix"]["iid"] = []
        self.assertIsNone(self.summarize()["rows"][0]["old_H0_comparison"]["equal"])

    def test_source_or_reference_rejection_is_copyable_and_no_probe_or_write_occurs(self):
        before = deepcopy(self.artifacts)
        self.protocol.read_study.side_effect = ValueError("frozen source/reference changed")
        with mock.patch("builtins.open", side_effect=AssertionError("file write/read")), \
                mock.patch.object(report.timing.base, "load_split", side_effect=AssertionError("dataset")), \
                mock.patch.object(report.timing.base.experiments, "run_measured_experiment", side_effect=AssertionError("train")):
            summary = self.summarize()
        self.assertEqual(summary["status"], "unavailable_or_invalid_study")
        self.assertEqual(before, self.artifacts)
        self.protocol.load_completed.assert_not_called()
        stream = io.StringIO()
        with redirect_stdout(stream):
            report.print_summary(summary)
        self.assertIn("frozen source/reference changed", stream.getvalue())

    def test_complete_compact_output_has_all_six_tasks_and_four_pairs_under_50KB(self):
        summary = self.summarize()
        stream = io.StringIO()
        with redirect_stdout(stream):
            report.print_summary(summary)
        text = stream.getvalue()
        lines = text.splitlines()
        self.assertEqual(lines[0], "=== CIFAR_PREFIX_PROBE_BEGIN ===")
        self.assertEqual(lines[-1], "=== CIFAR_PREFIX_PROBE_END ===")
        rows = [json.loads(line) for line in lines[1:-1]]
        self.assertEqual(sum(r["type"] == "task" for r in rows), 6)
        self.assertEqual(sum(r["type"] == "pair" for r in rows), 4)
        self.assertLess(len(text.encode()), 50000)
        self.assertNotIn("task_tag", text)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", text)


if __name__ == "__main__":
    unittest.main()
