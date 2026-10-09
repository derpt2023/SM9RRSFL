"""Read-only six-artifact client comparison and complete 100-client denominators."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
import unittest
from unittest import mock

import diagnose_cifar_prefix_clients as entry
import cifar_prefix_probe_protocol as protocol
import cifar_prefix_probe_report as report
from sm9rrsfl.fl import ExperimentConfig
import tests.test_cifar_prefix_probe_protocol as identities
import tests.test_cifar_prefix_probe_report as observations

base, runtime = protocol.base, protocol.runtime


def task(partition="dirichlet", repeat=1):
    identity = "prefix_" + partition + "_repeat" + str(repeat)
    config = asdict(ExperimentConfig(method="sm9rrs", rounds=3, num_clients=100,
        partition=partition, malicious_ratio=0., seed=2026093001, local_epochs=1,
        batch_size=50, detector_window=20, attack_start_round=25, lr=.05, lr_decay=.99))
    return {"task_id": identity, "partition": partition, "repeat": repeat,
            "fingerprint": hashlib.sha256(identity.encode()).hexdigest(), "config": config}


def payload(item):
    result = observations.payload(item)
    # Include full final batches and equal one-sample tails reached after a
    # different number of complete batches; every group uses all 100 clients.
    counts = [100, 150, 51, 101] + [50] * 96
    batch_size = item["config"]["batch_size"]
    for index, count in enumerate(counts):
        client = result["partition"]["clients"][index]
        client["samples"] = count
        client["indices"] = observations.fingerprint("indices-" + str(index), [count], "<i8")
        for rd in result["rounds"]:
            current = rd["clients"][index]
            current["samples"] = count
            current["stats"]["samples"] = count
            current["epoch_indices"] = [observations.fingerprint("epoch-" + str(rd["round"]) + "-" + str(index), [count], "<i8")]
            current["minibatch_sizes_per_epoch"] = [min(batch_size, count - start) for start in range(0, count, batch_size)]
    report.validate_observations(result, item)
    return result


def changed_global_after_round(value, rd, name):
    """Keep all duplicated model observations valid when modeling propagation."""
    fp = observations.fingerprint(name)
    value["rounds"][rd - 1]["post_model"] = deepcopy(fp)
    value["checkpoints"][rd]["model"] = deepcopy(fp)
    for evaluation in value["evaluations"]:
        if evaluation["round"] == rd:
            evaluation["model_input"] = deepcopy(fp)
    if rd < 3:
        for client in value["rounds"][rd]["clients"]:
            client["model_input"] = deepcopy(fp)


class ClientComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [task(repeat=r) for r in (1, 2, 3)]
        self.payloads = [payload(t) for t in self.tasks]

    def analyze(self):
        for value, item in zip(self.payloads, self.tasks):
            report.validate_observations(value, item)
        return entry.analyze_partition("dirichlet", self.payloads)

    def test_all_three_pairs_include_repeat2_against_repeat3(self):
        self.payloads[1]["rounds"][0]["clients"][19]["update"] = observations.fingerprint("different-update")
        self.payloads[2]["rounds"][0]["clients"][19]["update"] = observations.fingerprint("different-update")
        result = self.analyze()
        pairs = {(r["earlier_repeat"], r["later_repeat"]): r for r in result["pairs"]}
        self.assertEqual(set(pairs), {(1, 2), (1, 3), (2, 3)})
        self.assertEqual(pairs[(1, 2)]["update_differing_ids"], ["client-19"])
        self.assertEqual(pairs[(1, 3)]["update_differing_ids"], ["client-19"])
        self.assertEqual(pairs[(2, 3)]["update_differing_ids"], [])
        self.assertTrue(all(r["clients"] == 100 for r in pairs.values()))

    def test_equal_loss_does_not_hide_different_update_hash_or_claim_magnitude(self):
        self.payloads[1]["rounds"][0]["clients"][19]["update"] = observations.fingerprint("different-update")
        result = self.analyze()
        client = next(c for c in result["clients"] if c["client_id"] == "client-19")
        self.assertEqual(client["loss"], [.25, .25, .25])
        self.assertEqual(len(set(client["update_sha256"])), 2)
        self.assertTrue(client["inputs_equal"])
        pair = result["pairs"][0]
        self.assertEqual(pair["loss_differing_ids"], [])
        self.assertEqual(pair["update_differing_ids"], ["client-19"])
        for field in ("max_abs_error", "relative_l2", "error_magnitude"):
            self.assertNotIn(field, client)

    def test_round1_changes_and_later_unequal_model_inputs_are_separate(self):
        later = self.payloads[1]
        later["rounds"][0]["clients"][19]["update"] = observations.fingerprint("first-different-update")
        changed_global_after_round(later, 1, "later-round1-model")
        later["rounds"][1]["clients"][0]["update"] = observations.fingerprint("propagated-update")
        result = self.analyze()
        self.assertEqual(result["pairs"][0]["update_differing_ids"], ["client-19"])
        self.assertEqual(result["pairs"][0]["input_differing_ids"], [])
        by_round = {row["round"]: row for row in result["propagation"]}
        self.assertEqual(set(by_round), {2, 3})
        round2 = next(p for p in by_round[2]["pairs"] if (p["earlier_repeat"], p["later_repeat"]) == (1, 2))
        self.assertEqual(round2["update_differing_ids"], ["client-0"])
        self.assertEqual(len(round2["input_differing_ids"]), 100)

    def test_tail_group_denominators_include_unchanged_clients_and_full_final_batches(self):
        self.payloads[1]["rounds"][0]["clients"][2]["update"] = observations.fingerprint("tail-difference")
        result = self.analyze()
        groups = {(g["remainder"], g["last_batch_size"]): g for g in result["tail_groups"]}
        self.assertEqual(set(groups), {(0, 50), (1, 1)})
        self.assertEqual(sum(g["clients"] for g in groups.values()), 100)
        self.assertEqual(groups[(0, 50)]["clients"], 98)
        self.assertEqual(groups[(1, 1)]["clients"], 2)
        self.assertEqual(groups[(0, 50)]["any_update_different"], 0)
        self.assertEqual(groups[(1, 1)]["any_update_different"], 1)
        first, second = result["clients"][2:4]
        self.assertEqual((first["remainder"], second["remainder"]), (1, 1))
        self.assertEqual((first["last_batch_size"], second["last_batch_size"]), (1, 1))
        self.assertNotEqual(first["batch_count"], second["batch_count"])
        self.assertNotEqual(first["minibatch_sizes"], second["minibatch_sizes"])

    def test_reconstructed_order_difference_is_explicit_and_not_equal_inputs(self):
        self.payloads[1]["rounds"][0]["clients"][19]["epoch_indices"][0] = observations.fingerprint("different-order", [50], "<i8")
        self.payloads[1]["rounds"][0]["clients"][19]["update"] = observations.fingerprint("different-output")
        result = self.analyze()
        client = result["clients"][19]
        self.assertFalse(client["inputs_equal"])
        self.assertTrue(any("epoch_indices" in f for f in client["different_input_fields"]))
        self.assertEqual(result["pairs"][0]["input_differing_ids"], ["client-19"])

    def test_global_data_mismatch_marks_every_client_input_as_unequal(self):
        field = self.payloads[1]["data"]["x_train"]
        self.payloads[1]["data"]["x_train"] = observations.fingerprint("different-training-data", field["shape"], field["dtype"])
        result = self.analyze()
        self.assertFalse(result["common"]["global_input_equal"])
        self.assertTrue(all(not c["inputs_equal"] for c in result["clients"]))
        self.assertTrue(all(any(f.startswith("global.") for f in c["different_input_fields"]) for c in result["clients"]))
        self.assertEqual(len(result["pairs"][0]["input_differing_ids"]), 100)
        self.assertEqual(result["pairs"][0]["update_differing_ids"], [])

    def test_cross_repeat_batch_shape_mismatch_is_counted_outside_matched_tail_groups(self):
        value = self.payloads[1]
        value["partition"]["clients"][19].update(samples=51, indices=observations.fingerprint("indices19", [51], "<i8"))
        for rd in value["rounds"]:
            client = rd["clients"][19]
            client.update(samples=51, epoch_indices=[observations.fingerprint("epoch19", [51], "<i8")],
                          minibatch_sizes_per_epoch=[50, 1])
            client["stats"]["samples"] = 51
        result = self.analyze()
        self.assertEqual(result["unmatched_batch_shape_ids"], ["client-19"])
        self.assertEqual(sum(g["clients"] for g in result["tail_groups"]), 99)
        self.assertEqual(len(result["clients"]), 100)
        self.assertFalse(result["clients"][19]["inputs_equal"])


class SerializedClientReaderTests(identities.PrefixFixture):
    def setUp(self):
        super().setUp()
        self.metadata = self.worker_metadata()
        # The old fixture predates device inventories. Give the new synthetic
        # probe a coherent recorded inventory/profile; frozen old90 files and
        # their real hashes remain unchanged and are read by the actual reader.
        reference = deepcopy(self.reference)
        reference["execution_environment"] = report.matched.normalized_environment(self.metadata)
        self.manifest = protocol.build_manifest(reference, identities.GPU)
        self.tasks = protocol.build_tasks(self.manifest)
        base.write_json(self.output / "manifest.json", self.manifest)
        base.write_json(self.output / "task_plans/prefix.json", {
            "manifest_fingerprint": self.manifest["fingerprint"], "tasks": self.tasks})
        for item in self.tasks:
            folder = self.output / "tasks" / item["task_id"]
            base.write_json(folder / "task.json", item)
            artifact = {k: v for k, v in self.completion(item).items() if k != "artifact_fingerprint"}
            artifact["observations"] = payload(item)
            base.write_json(folder / "completed.json", protocol.seal_artifact(artifact))

    def hashes(self):
        return {str(file): hashlib.sha256(file.read_bytes()).hexdigest()
                for root in [self.output, *self.paths.values()] for file in root.rglob("*") if file.is_file()}

    def guards(self):
        stack = self.readonly_guards()
        for obj, name in ((base, "write_json"), (base, "immutable_json"), (base.fl, "run_experiment")):
            stack.enter_context(mock.patch.object(obj, name, side_effect=AssertionError("forbidden " + name)))
        return stack

    def change_artifact(self, mutate):
        path = self.output / "tasks" / self.tasks[4]["task_id"] / "completed.json"
        artifact = runtime.read_json(path)
        mutate(artifact)
        artifact.pop("artifact_fingerprint")
        base.write_json(path, protocol.seal_artifact(artifact))

    def test_real_manifest_six_sealed_artifacts_and_90_reference_chain_are_readonly(self):
        before = self.hashes()
        with self.guards():
            result = entry.summarize(self.output)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["partitions"]), 2)
        self.assertEqual(sum(len(p["clients"]) for p in result["partitions"]), 200)
        self.assertEqual(sum(len(p["pairs"]) for p in result["partitions"]), 6)
        self.assertEqual(before, self.hashes())

    def test_complete_JSONL_keeps_all_clients_three_hashes_losses_and_six_pairs(self):
        raw_loss = .25000000000000006
        self.change_artifact(lambda a: a["observations"]["rounds"][0]["clients"][19]["stats"].update(loss=raw_loss))
        before, capture = self.hashes(), io.StringIO()
        with self.guards(), redirect_stdout(capture):
            self.assertEqual(entry.main(["--output", str(self.output)]), 0)
        self.assertEqual(before, self.hashes())
        text = capture.getvalue()
        lines = text.splitlines()
        self.assertEqual(lines[0], "=== CIFAR_PREFIX_CLIENTS_BEGIN ===")
        self.assertEqual(lines[-1], "=== CIFAR_PREFIX_CLIENTS_END ===")
        rows = [json.loads(line) for line in lines[1:-1]]
        self.assertEqual(len(rows), 217)
        clients = [r for r in rows if r["type"] == "client"]
        self.assertEqual(len(clients), 200)
        pairs = [r for r in rows if r["type"] == "pair"]
        self.assertEqual(len(pairs), 6)
        columns = next(r["client_values"] for r in rows if r["type"] == "legend")
        decoded = [(r["partition"], dict(zip(columns, r["values"]))) for r in clients]
        for _, client in decoded:
            self.assertEqual(len(client["update_sha256"]), 3)
            self.assertTrue(all(len(sha) == 64 for sha in client["update_sha256"]))
            self.assertEqual(len(client["loss"]), 3)
        chosen = next(c for partition, c in decoded if partition == "dirichlet" and c["client_id"] == "client-19")
        self.assertEqual(chosen["loss"][1], raw_loss)
        self.assertNotIn("task_tag", text)
        self.assertLess(len(text.encode()), 125000)
        print("CLIENT_DIAG_COMPLETE_BYTES=" + str(len(text.encode())), flush=True)

    def test_missing_artifact_is_invalid_without_partial_partitions(self):
        (self.output / "tasks" / self.tasks[4]["task_id"] / "completed.json").unlink()
        with self.guards():
            result = entry.summarize(self.output)
        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result.get("partitions"))

    def test_bad_environment_and_invalid_observations_are_rejected(self):
        self.change_artifact(lambda a: a["environment"]["actual_compute_device"].update(name="other GPU"))
        with self.guards():
            bad_environment = entry.summarize(self.output)
        self.assertEqual(bad_environment["status"], "invalid")
        self.assertFalse(bad_environment.get("partitions"))
        self.change_artifact(lambda a: a.update(environment=deepcopy(self.metadata)))
        self.change_artifact(lambda a: a["observations"]["rounds"][0]["clients"].pop())
        with self.guards():
            incomplete = entry.summarize(self.output)
        self.assertEqual(incomplete["status"], "invalid")
        self.assertFalse(incomplete.get("partitions"))

    def test_upstream_reference_mutation_is_rejected_without_GPU_or_training(self):
        path = self.paths["history"] / "execution_environment.json"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.guards():
                result = entry.summarize(self.output)
            self.assertEqual(result["status"], "invalid")
            self.assertFalse(result.get("partitions"))
        finally:
            path.write_bytes(original)

    def test_artifact_change_during_read_invalidates_all_partitions(self):
        target = self.output / "tasks" / self.tasks[-1]["task_id"] / "completed.json"
        original, actual = target.read_bytes(), protocol.load_completed
        def mutate_after_read(output, item):
            result = actual(output, item)
            if item["task_id"] == self.tasks[-1]["task_id"]:
                target.write_bytes(original + b"\n")
            return result
        try:
            with self.guards(), mock.patch.object(protocol, "load_completed", side_effect=mutate_after_read):
                result = entry.summarize(self.output)
            self.assertEqual(result["status"], "invalid")
            self.assertFalse(result.get("partitions"))
        finally:
            target.write_bytes(original)

    def test_CLI_failure_returns2_and_one_copyable_JSONL_frame(self):
        (self.output / "tasks" / self.tasks[4]["task_id"] / "completed.json").unlink()
        capture = io.StringIO()
        with self.guards(), redirect_stdout(capture):
            self.assertEqual(entry.main(["--output", str(self.output)]), 2)
        lines = capture.getvalue().splitlines()
        self.assertEqual(lines[0], "=== CIFAR_PREFIX_CLIENTS_BEGIN ===")
        self.assertEqual(lines[-1], "=== CIFAR_PREFIX_CLIENTS_END ===")
        values = [json.loads(line) for line in lines[1:-1]]
        self.assertEqual(values[0]["status"], "invalid")
        self.assertNotIn("task_tag", capture.getvalue())


if __name__ == "__main__":
    unittest.main()
