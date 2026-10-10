"""Numerical drilldown on public synthetic evidence, with no training or files."""
from copy import deepcopy
import json
import unittest
from unittest import mock

import cifar_mechanism_details as details


def fixture():
    config = {"num_clients": 3, "malicious_ratio": 0., "detector_distance_threshold": 1.5,
        "detector_drift_threshold": 2., "detector_reject_threshold": 3., "detector_history_threshold": 1.,
        "detector_reference_budget": 2., "detector_drift_allowance": 1.25, "suspicion_remove_after": 3}
    payload = {"rounds": [], "history_events": [], "actual_batches": [], "singleton_batches": []}
    for rd in range(21, 31):
        row = {"round": rd, "clients": [], "diagnostics": [], "blacklisted_before": [],
            "coefficients": {"by_client": {}}, "record": {"honest_weight_loss": .1}}
        for i in range(3):
            cid = "client-" + str(i)
            decision = {"accepted": True, "reason": "normal", "would_flag": False, "count_increment": False,
                "novelty_score": 1., "anchor_score": .5, "signed_score": .5, "class_score": .5,
                "cumulative_drift": .2, "norm_score": 0., "clip_factor": 1., "history_eligible": True,
                "trusted_history_size": 20, "normal_cluster_count": 2, "immediate_revocation": False, "recovery_eligible": True}
            diagnostic = {k: v for k, v in decision.items() if k not in ("accepted", "reason", "would_flag")}
            diagnostic.update(client_id=cid, decision_reason="normal", suspicious=False, round=rd,
                weight_before=.8, weight_after_penalty_recovery=.9, aggregation_weight=.3, count_before=.5, count_after=.25,
                trace_requested=False, trace_pending=False, revoked=False, aggregation_accepted=True, history_admitted=True, history_frozen=False)
            row["clients"].append({"client_id": cid, "update_finite": True, "model_input": {"sha256": "model" + str(rd - 1)},
                "update": {"sha256": "update" + str(rd) + cid}, "stats": {"loss": .1}})
            row["diagnostics"].append(diagnostic)
            row["coefficients"]["by_client"][cid] = .3
            state = {"history": {"sha256": "history" + str(rd - 1)}, "history_size": 20,
                "normal": {"sha256": "normal" + str(rd - 1)}, "anchor": {"sha256": "anchor"},
                "norm_limit": 1., "drift": .2, "clean_streak": 5, "recovery_streak": 5}
            after = deepcopy(state)
            after.update(history={"sha256": "history" + str(rd)}, normal={"sha256": "normal" + str(rd)})
            payload["history_events"].append({"round": rd, "client_id": cid, "decision": decision,
                "before_evaluate": deepcopy(state), "after_evaluate": deepcopy(state), "forget": None,
                "commit": {"requested_admission": True, "admitted": True, "before": deepcopy(state), "after": after}})
        payload["rounds"].append(row)
    h0, h1 = deepcopy(payload), deepcopy(payload)
    for event in h1["history_events"]:
        if event["round"] >= 25:
            event["commit"]["admitted"] = False
            event["commit"]["after"] = deepcopy(event["commit"]["before"])
    for row in h1["rounds"]:
        if row["round"] >= 25:
            for d in row["diagnostics"]:
                d["history_admitted"] = False
    return h1, h0, config


def event(payload, rd=26, cid="client-0"):
    return next(e for e in payload["history_events"] if e["round"] == rd and e["client_id"] == cid)


def row(payload, rd=26):
    return next(r for r in payload["rounds"] if r["round"] == rd)


def change(payload, *, rd=26, cid="client-0", **updates):
    e = event(payload, rd, cid)
    d = next(d for d in row(payload, rd)["diagnostics"] if d["client_id"] == cid)
    mapping = {"reason": "decision_reason", "would_flag": "suspicious", "accepted": "aggregation_accepted"}
    for k, v in updates.items():
        e["decision"][k] = v
        d[mapping.get(k, k)] = v


class MechanismDetailsTests(unittest.TestCase):
    def setUp(self):
        self.h1, self.h0, self.config = fixture()

    def analyze(self):
        return details.analyze_pair(self.h1, self.h0, self.config)

    def test_round25_suppression_records_actual_admission_and_public_state_changes(self):
        result = self.analyze()["round25"]
        self.assertTrue(result["common_precommit_equal"])
        self.assertEqual(result["H0"]["admitted"], 3)
        self.assertEqual(result["H1"]["admitted"], 0)
        self.assertEqual(result["paired_suppressed_admission_ids"], ["client-0", "client-1", "client-2"])
        self.assertEqual(result["postcommit_state_differing_ids"]["normal"], result["paired_suppressed_admission_ids"])
        self.assertEqual(result["postcommit_state_differing_ids"]["anchor"], [])
        self.assertNotIn("sha256", json.dumps(result))

    def test_threshold_equality_is_not_exceedance_and_raw_margins_are_exact(self):
        change(self.h0, novelty_score=1.5, cumulative_drift=2.)
        change(self.h1, novelty_score=1.75, cumulative_drift=2.1, accepted=False, would_flag=True, count_increment=True, reason="suspicious")
        value = self.analyze()
        flips = value["round26"]["decisions"]["threshold_flips"]
        self.assertEqual(flips["warning"][0]["H0"]["margin"], 0.)
        self.assertFalse(flips["warning"][0]["H0"]["exceeds"])
        self.assertEqual(flips["warning"][0]["H1"]["margin"], .25)
        self.assertEqual(flips["drift"][0]["H0"]["margin"], 0.)
        self.assertEqual(flips["strong"], [])
        case = value["flip_cases"][0]
        self.assertEqual(case["H1"]["decision"]["accepted"], False)
        self.assertEqual(case["H1"]["removal_margins"]["after_count_minus_limit"], -2.75)
        self.assertEqual([t["round"] for t in case["trajectory"]], list(range(21, 31)))

    def test_equal_warning_totals_expose_swapped_client_sets_and_pre_evaluate_states(self):
        change(self.h0, cid="client-0", novelty_score=2.)
        change(self.h1, cid="client-1", novelty_score=2.)
        before = event(self.h1)["before_evaluate"]
        before["history"]["sha256"] = "different-history"
        before["normal"]["sha256"] = "different-live-model"
        before["drift"] = .5
        before["clean_streak"] = 4
        value = self.analyze()["round26"]
        self.assertEqual(value["decisions"]["threshold_counts"]["warning"], {"H0": 1, "H1": 1, "denominator": 3})
        sets = value["decisions"]["threshold_sets"]["warning"]
        self.assertEqual(sets["common_exceeding_ids"], [])
        self.assertEqual(sets["H0_only_exceeding_ids"], ["client-0"])
        self.assertEqual(sets["H1_only_exceeding_ids"], ["client-1"])
        fields = value["pre_evaluate_state"]["fields"]
        self.assertEqual(fields["history"]["differing_client_ids"], ["client-0"])
        self.assertEqual(fields["normal"]["differing_client_ids"], ["client-0"])
        self.assertEqual(fields["anchor"]["differing_client_ids"], [])
        self.assertEqual(fields["clean_streak"]["changed_scalars"], [["client-0", 5, 4, -1]])
        self.assertAlmostEqual(fields["drift"]["changed_scalars"][0][3], .3)
        self.assertNotIn("sha256", json.dumps(value["pre_evaluate_state"]))

    def test_strong_equal_threshold_and_overlap_counts(self):
        change(self.h0, novelty_score=3.)
        change(self.h1, novelty_score=3.5, immediate_revocation=True)
        result = self.analyze()["round26"]["decisions"]
        self.assertEqual(result["threshold_counts"]["warning"], {"H0": 1, "H1": 1, "denominator": 3})
        self.assertEqual(result["threshold_counts"]["strong"], {"H0": 0, "H1": 1, "denominator": 3})
        self.assertEqual(result["threshold_flips"]["strong"][0]["H0"]["margin"], 0)

    def test_continuous_score_changes_are_not_claimed_as_discrete_decision_flips(self):
        change(self.h1, novelty_score=1.1, signed_score=.6)
        value = self.analyze()
        decision = value["round26"]["decisions"]
        self.assertEqual(decision["continuous_score_changed_clients"], 1)
        self.assertEqual(decision["continuous_changes_without_discrete_flip_clients"], 1)
        self.assertEqual(decision["flip_client_ids"], [])
        self.assertEqual(value["flip_cases"], [])
        self.assertEqual(decision["score_deltas"]["novelty_score"]["n"], 3)
        self.assertAlmostEqual(decision["score_deltas"]["novelty_score"]["max_abs"], .1)
        self.assertEqual(decision["score_deltas"]["novelty_score"]["p50"], 0.)

    def test_signed_quantiles_and_history_count_only_change_are_distinct(self):
        change(self.h1, cid="client-0", signed_score=-.5)
        change(self.h1, cid="client-2", signed_score=1.5)
        value = self.analyze()["round26"]["decisions"]
        scalar = value["score_deltas"]["signed_score"]
        self.assertEqual((scalar["min"], scalar["p50"], scalar["max"], scalar["max_abs"], scalar["mean"]), (-1., 0., 1., 1., 0.))
        self.assertAlmostEqual(scalar["p90"], .8)
        self.assertAlmostEqual(scalar["p99"], .98)
        self.h1, self.h0, self.config = fixture()
        change(self.h1, normal_cluster_count=1)
        value = self.analyze()["round26"]["decisions"]
        self.assertEqual(value["changed_decision_clients"], 1)
        self.assertEqual(value["changed_decision_client_ids"], ["client-0"])
        self.assertEqual(value["continuous_score_changed_clients"], 0)
        self.assertEqual(value["continuous_score_changed_client_ids"], [])

    def test_first_decision_difference_follows_actual_event_order_not_client_sort(self):
        for payload in (self.h0, self.h1):
            selected = [e for e in payload["history_events"] if e["round"] == 26]
            payload["history_events"] = [e for e in payload["history_events"] if e["round"] != 26] + list(reversed(selected))
        change(self.h1, cid="client-0", anchor_score=.7)
        change(self.h1, cid="client-2", norm_score=.3)
        first = self.analyze()["round26"]["decisions"]["first_decision_difference"]
        self.assertEqual((first["client_id"], first["H1_event_index"], first["H0_event_index"]), ("client-2", 0, 0))
        self.assertTrue(first["actual_common_order_certified"])

    def test_reordered_coefficient_mapping_is_equal_and_weight_reliability_is_separate(self):
        row(self.h1)["coefficients"]["by_client"] = dict(reversed(list(row(self.h1)["coefficients"]["by_client"].items())))
        row(self.h1)["diagnostics"][0]["weight_after_penalty_recovery"] = .7
        result = self.analyze()["round26"]["coefficients"]
        self.assertEqual(result["delta_l1"], 0)
        self.assertEqual(result["changed_clients"], 0)
        self.assertEqual(result["reliability"]["weight_after_penalty_recovery"]["changed_client_ids"], ["client-0"])
        self.assertAlmostEqual(result["reliability"]["weight_after_penalty_recovery"]["delta"]["min"], -.2)

    def test_coefficient_deltas_support_and_honest_weight_loss_use_actual_maps(self):
        row(self.h0)["coefficients"]["by_client"] = {"client-0": .1, "client-1": 0., "client-2": .5}
        row(self.h1)["coefficients"]["by_client"] = {"client-2": .2, "client-1": .2, "client-0": 0.}
        row(self.h1)["record"]["honest_weight_loss"] = .3
        value = self.analyze()["round26"]["coefficients"]
        self.assertAlmostEqual(value["delta_l1"], .6)
        self.assertAlmostEqual(value["delta_max_abs"], .3)
        self.assertEqual(value["max_abs_delta_client_ids"], ["client-2"])
        self.assertEqual(value["max_abs_delta_example"]["client_id"], "client-2")
        self.assertEqual(value["max_abs_delta_example"]["H0"], .5)
        self.assertEqual(value["max_abs_delta_example"]["H1"], .2)
        self.assertAlmostEqual(value["delta_signed_sum"], -.2)
        self.assertEqual(value["support_gained_common_ids"], ["client-1"])
        self.assertEqual(value["support_lost_common_ids"], ["client-0"])
        self.assertEqual(value["changed_coefficients"][0], ["client-0", .1, 0., -.1])
        self.assertAlmostEqual(value["honest_weight_loss"]["delta"], .2)

    def test_missing_revoked_and_nonfinite_clients_are_not_imputed_zero(self):
        r = row(self.h1)
        r["blacklisted_before"] = ["client-2"]
        r["clients"] = r["clients"][:2]
        r["clients"][1]["update_finite"] = False
        r["diagnostics"] = r["diagnostics"][:1]
        r["coefficients"]["by_client"] = {"client-0": .3}
        self.h1["history_events"] = [e for e in self.h1["history_events"] if e["round"] != 26 or e["client_id"] == "client-0"]
        value = self.analyze()["round26"]
        self.assertEqual(value["coverage"]["H1"]["active_clients"], 2)
        self.assertEqual(value["coverage"]["H1"]["nonfinite_client_ids"], ["client-1"])
        self.assertEqual(value["decisions"]["common_clients"], 1)
        self.assertEqual(value["coefficients"]["common_clients"], 1)
        self.assertFalse(value["coefficients"]["identity_coverage_equal"])
        self.assertEqual(value["coefficients"]["delta_l1"], 0)
        self.assertEqual(value["coefficients"]["H0_only_client_ids"], ["client-1", "client-2"])

    def test_zero_observations_and_missing_round_have_none_numeric_statistics(self):
        for payload in (self.h0, self.h1):
            r = row(payload)
            for c in r["clients"]:
                c["update_finite"] = False
            r["diagnostics"] = []
            r["coefficients"]["by_client"] = {}
            payload["history_events"] = [e for e in payload["history_events"] if e["round"] != 26]
            payload["rounds"] = [r for r in payload["rounds"] if r["round"] != 30]
        value = self.analyze()
        self.assertEqual(value["round26"]["decisions"]["score_deltas"]["novelty_score"]["n"], 0)
        self.assertIsNone(value["round26"]["decisions"]["score_deltas"]["novelty_score"]["max_abs"])
        self.assertIsNone(value["round26"]["coefficients"]["delta_l1"])
        self.assertIsNone(value["round26"]["coefficients"]["H0"]["sum"])
        self.assertFalse(value["rounds27_30"][-1]["coefficients"]["available"])
        self.assertIsNone(value["rounds27_30"][-1]["changed_decision_clients"])

    def test_later_round_local_mismatch_is_exposed_without_same_input_causal_claim(self):
        row(self.h1, 27)["clients"][0]["model_input"]["sha256"] = "changed"
        change(self.h1, rd=27, novelty_score=2.)
        result = self.analyze()["rounds27_30"][0]
        self.assertFalse(result["local_training_equality"]["equal"])
        self.assertEqual(result["local_training_equality"]["different_client_ids"], ["client-0"])
        self.assertEqual(result["scope"], "post_intervention_observations_no_same_input_causal_attribution")
        self.assertEqual(result["changed_decision_clients"], 1)

    def test_case_trajectories_mark_post_revocation_and_missing_rounds_explicitly(self):
        change(self.h1, novelty_score=2., history_eligible=False, recovery_eligible=False)
        r = row(self.h1, 27)
        r["blacklisted_before"] = ["client-0"]
        r["clients"] = r["clients"][1:]
        r["diagnostics"] = r["diagnostics"][1:]
        self.h1["rounds"] = [r for r in self.h1["rounds"] if r["round"] != 30]
        case = self.analyze()["flip_cases"][0]
        self.assertEqual(case["trajectory"][6]["H1"], {"status": "revoked_before_round", "values": None})
        self.assertEqual(case["trajectory"][-1]["H1"], {"status": "round_unavailable", "values": None})

    def test_reject_nonfinite_or_duplicate_scalar_evidence(self):
        for edit in (lambda: event(self.h1)["decision"].update(novelty_score=float("nan")),
                     lambda: self.h1["history_events"].append(deepcopy(event(self.h1))),
                     lambda: row(self.h1)["coefficients"]["by_client"].update({"client-0": float("inf")})):
            self.h1, self.h0, self.config = fixture()
            edit()
            with self.assertRaises(ValueError):
                self.analyze()

    def test_pure_readonly_finite_json_preserves_inputs_and_never_exports_hashes_or_task_tags(self):
        change(self.h1, novelty_score=2.)
        before = deepcopy((self.h1, self.h0, self.config))
        for d in row(self.h1)["diagnostics"]:
            d["task_tag"] = "private-never-export"
        before = deepcopy((self.h1, self.h0, self.config))
        with mock.patch("builtins.open", side_effect=AssertionError("unexpected file I/O")):
            value = self.analyze()
        text = json.dumps(value, allow_nan=False)
        self.assertEqual((self.h1, self.h0, self.config), before)
        for secret in ("sha256", "private-never-export", "task_tag"):
            self.assertNotIn(secret, text)
        self.assertFalse(value["limits"]["tensor_distance_inferred"])
        self.assertFalse(value["limits"]["training_started"])


if __name__ == "__main__":
    unittest.main()
