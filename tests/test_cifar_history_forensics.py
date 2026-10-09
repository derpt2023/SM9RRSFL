"""Client evidence semantics using synthetic records, without training or files."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import pickle
from types import SimpleNamespace
import unittest
from unittest import mock

import cifar_history_forensics as report
from sm9rrsfl.fl import ClientDiagnosticRecord, ExperimentConfig
from tests.test_cifar_six_pipeline import synthetic_run


class HistoryForensicsTests(unittest.TestCase):
    def task(self, ratio=.7, arm="H0", partition="iid"):
        config = ExperimentConfig(method="sm9rrs", num_clients=100, rounds=150, malicious_ratio=ratio,
            partition=partition, detector_window=20, attack_start_round=25, seed=2026093001,
            detector_distance_threshold=1.25, detector_drift_threshold=6., detector_reject_threshold=6.,
            suspicion_remove_after=5)
        return {"task_id": "forensic_" + arm + "_" + partition + "_" + str(ratio), "arm": arm,
                "config": asdict(config), "candidate": {"variant": "original" if arm == "H0" else "Ours-FrozenHistory-v1"}}

    def diagnostic(self, round_id=25, client="client-0", malicious=True, **values):
        row = dict(round=round_id, client_id=client, task_tag="PRIVATE_TAG_NEVER_EXPORT", is_malicious=malicious,
            decision_reason="normal", suspicious=False, count_increment=False, weight_before=1.,
            weight_after_penalty_recovery=1., aggregation_weight=.01, count_before=0., count_after=0.,
            trace_requested=False, trace_pending=False, revoked=False, novelty_score=.5, anchor_score=.7,
            signed_score=.2, class_score=.3, cumulative_drift=0., clip_factor=1., aggregation_accepted=True,
            history_eligible=True, history_admitted=True, history_frozen=False, immediate_revocation=False,
            trusted_history_size=20, normal_cluster_count=2, attack_active=malicious and round_id >= 25,
            recovery_eligible=True, norm_score=.1)
        row.update(values)
        if row["revoked"]:
            row.update(aggregation_accepted=False, aggregation_weight=0., history_admitted=False)
        return ClientDiagnosticRecord(**row)

    def result(self, task, diagnostics, nonfinite=0):
        result = synthetic_run(ExperimentConfig(**task["config"]), .60, .10, nonfinite)
        malicious = tuple("client-" + str(i) for i in range(round(100 * task["config"]["malicious_ratio"])))
        events = [d for d in diagnostics if d.revoked]
        records = [replace(r, false_positive_revocations=sum(d.round <= r.round and not d.is_malicious for d in events),
            true_positive_revocations=sum(d.round <= r.round and d.is_malicious for d in events),
            blacklisted_clients=sum(d.round <= r.round for d in events)) for r in result.records]
        return replace(result, records=records, diagnostics=diagnostics, malicious_clients=malicious,
                       blacklisted_clients=tuple(d.client_id for d in events))

    def analyze(self, task, diagnostics, **kwargs):
        return report.analyze_task(self.result(task, diagnostics, **kwargs), task)

    def group(self, analysis, name="first25_29", role="malicious"):
        return next(w for w in analysis["windows"] if w["name"] == name)["groups"][role]

    def test_five_of_seventy_observations_use_five_not_nominal_population(self):
        task = self.task()
        observations = [self.diagnostic(client="client-" + str(i), history_admitted=i < 2) for i in range(5)]
        analysis = self.analyze(task, observations)
        group = self.group(analysis)
        self.assertEqual(analysis["status"], "valid")
        self.assertEqual(group["counts"]["original_clients"], 70)
        self.assertEqual(group["counts"]["remaining_client_rounds"], 350)
        self.assertEqual(group["counts"]["observed"], 5)
        self.assertEqual(group["counts"]["gap"], 345)
        self.assertEqual(group["accepted_rate"], 1.)
        self.assertEqual(group["admitted_rate"], .4)

    def test_empty_group_scores_and_rates_are_unavailable_not_zero(self):
        analysis = self.analyze(self.task(), [])
        group = self.group(analysis)
        self.assertEqual(group["counts"]["observed"], 0)
        self.assertEqual(group["counts"]["gap"], 350)
        self.assertIsNone(group["admitted_rate"])
        self.assertEqual(group["scores"]["novelty"], {"n": 0, "p50": None, "p90": None, "p99": None, "max": None})
        self.assertEqual(group["status"], "no_observed_records")

    def test_clean_window_and_history_eligibility_are_separate_from_admission(self):
        task = self.task(ratio=0., arm="H1")
        analysis = self.analyze(task, [self.diagnostic(client="client-90", malicious=False,
            history_eligible=True, history_admitted=False)])
        group = self.group(analysis, role="honest")
        self.assertEqual((group["counts"]["history_eligible"], group["counts"]["admitted"]), (1, 0))
        self.assertEqual(self.group(analysis)["status"], "not_applicable_no_original_clients")
        self.assertEqual([w["name"] for w in analysis["windows"]], [w[0] for w in report.WINDOWS])

    def test_strict_threshold_equality_and_overlapping_strong_trigger_counts(self):
        task = self.task()
        observations = [self.diagnostic(client="client-" + str(i), novelty_score=score, cumulative_drift=drift)
            for i, (score, drift) in enumerate(((1.25, 6.), (2., 0.), (.5, 7.), (7., 8.), (6., 0.)))]
        group = self.group(self.analyze(task, observations))
        self.assertEqual(group["triggers"], {"warning_only": 2, "drift_only": 1, "both": 1, "neither": 1, "strong": 1})
        self.assertEqual(sum(group["triggers"][k] for k in report.TRIGGERS[:4]), 5)
        self.assertEqual(group["decision_reasons"], {"normal": 5})

    def test_all_six_score_quantiles_use_linear_interpolation_on_raw_values(self):
        task = self.task()
        observations = [self.diagnostic(client="client-" + str(i), **{field: float(i) for field in report.SCORES.values()})
                        for i in range(4)]
        for score in self.group(self.analyze(task, observations))["scores"].values():
            self.assertEqual(score["n"], 4)
            self.assertEqual(score["p50"], 1.5)
            self.assertAlmostEqual(score["p90"], 2.7)
            self.assertAlmostEqual(score["p99"], 2.97)
            self.assertEqual(score["max"], 3.)

    def test_revocation_paths_are_immediate_count_or_both_and_events_not_summed_cumulative(self):
        task = self.task()
        observations = []
        for i, (immediate, count, rd) in enumerate(((True, 1., 24), (False, 5., 25), (True, 5., 122))):
            observations.append(self.diagnostic(rd, "client-" + str(i), revoked=True, immediate_revocation=immediate,
                count_increment=True, count_after=count, novelty_score=7. if immediate else 2., trace_requested=True))
        analysis = self.analyze(task, observations)
        summary = analysis["revocations"]["malicious"]
        self.assertEqual(summary["count"], 3)
        self.assertEqual((summary["first_round"], summary["last_round"]), (24, 122))
        self.assertEqual((summary["immediate"], summary["nonimmediate"]), (2, 1))
        self.assertEqual(summary["paths"], {"immediate_only": 1, "count_only": 1, "both": 1, "neither": 0})
        self.assertEqual(summary["round_blocks"], {"1_20": 0, "21_24": 1, "25_29": 1, "30_120": 0, "121_150": 1})
        self.assertTrue(analysis["revocation_consistency"]["consistent"])
        self.assertEqual(analysis["health"]["TP"], 3)

    def test_prior_revoked_are_excluded_from_remaining_while_unobserved_survivors_stay_gap(self):
        task = self.task()
        observations = [self.diagnostic(24, revoked=True, immediate_revocation=True, count_increment=True,
                                       count_after=1., novelty_score=7., trace_requested=True)]
        analysis = self.analyze(task, observations)
        group = self.group(analysis)
        self.assertEqual(group["counts"]["remaining_client_rounds"], 69 * 5)
        self.assertEqual(group["counts"]["gap"], 69 * 5)
        self.assertEqual(group["counts"]["revoked"], 0)

    def test_revoked_event_counter_or_blacklist_mismatch_is_invalid_not_silently_healthy(self):
        task = self.task()
        result = self.result(task, [self.diagnostic(revoked=True)])
        for mutation in (replace(result, blacklisted_clients=()), replace(result,
            records=[replace(r, true_positive_revocations=0) for r in result.records])):
            with self.subTest(mutation=mutation.blacklisted_clients):
                analysis = report.analyze_task(mutation, task)
                self.assertEqual(analysis["status"], "invalid_evidence")
                self.assertFalse(analysis["revocation_consistency"]["consistent"])
                self.assertEqual(analysis["revocations"]["malicious"]["count"], 1)

    def test_observation_after_permanent_revocation_is_invalid(self):
        task = self.task()
        analysis = self.analyze(task, [self.diagnostic(25, revoked=True), self.diagnostic(26)])
        self.assertEqual(analysis["status"], "invalid_evidence")
        self.assertIn("after permanent revocation", analysis["error"])

    def test_missing_invalid_boolean_and_nonfinite_scores_never_become_zero(self):
        task = self.task()
        for field, value in (("history_eligible", None), ("immediate_revocation", 0),
                             ("norm_score", float("nan")), ("anchor_score", float("inf")), ("signed_score", None)):
            with self.subTest(field=field):
                d = SimpleNamespace(**asdict(self.diagnostic()))
                if value is None:
                    delattr(d, field)
                else:
                    setattr(d, field, value)
                analysis = self.analyze(task, [d])
                self.assertEqual(analysis["status"], "invalid_evidence")
                self.assertEqual(analysis["windows"], [])
                self.assertIn(field, analysis["error"])
                json.dumps(report.compact_task_analysis(analysis), allow_nan=False)

    def test_original_clean_health_failure_and_nonfinite_health_failure_are_preserved(self):
        task = self.task(ratio=0.)
        events = [self.diagnostic(80, "client-" + str(i), malicious=False, revoked=True) for i in range(11)]
        analysis = self.analyze(task, events)
        self.assertEqual(analysis["status"], "valid")
        self.assertFalse(analysis["health"]["healthy"])
        self.assertIn("clean_false_revocation_rate", analysis["health"]["reasons"])
        self.assertEqual((analysis["health"]["FP"], analysis["health"]["original_honest"]), (11, 100))
        analysis = self.analyze(self.task(), [], nonfinite=1)
        self.assertEqual(analysis["status"], "valid")
        self.assertFalse(analysis["health"]["healthy"])
        self.assertEqual(analysis["health"]["nonfinite_updates"], 1)

    def test_cases_are_sorted_nonrepresentative_recent_observations_not_interpolated_rounds(self):
        task = self.task(ratio=0.)
        observations = []
        for index, revoked_round in ((3, 120), (2, 80), (1, 80), (4, 130)):
            for rd in list(range(21, 31)) + [revoked_round]:
                observations.append(self.diagnostic(rd, "client-" + str(index), malicious=False, revoked=rd == revoked_round))
        analysis = self.analyze(task, list(reversed(observations)))
        cases = analysis["revocations"]["honest"]["examples"]
        self.assertEqual([c["client_id"] for c in cases], ["client-1", "client-2", "client-3"])
        self.assertEqual(len(cases[0]["observations"]), 8)
        self.assertEqual(cases[0]["observations"][-1]["round"], 80)
        compact = report.compact_task_analysis(analysis, detailed=True)
        self.assertEqual(compact["honest_case"]["client_id"], "client-1")
        self.assertEqual([o[0] for o in compact["honest_case"]["observations"]], [27, 28, 29, 30, 80])

    def test_pure_function_does_not_mutate_input_read_files_or_expose_task_tag(self):
        task = self.task(ratio=0.)
        result = self.result(task, [self.diagnostic(25, "client-90", malicious=False, revoked=True)])
        before = pickle.dumps((result, task))
        with mock.patch("builtins.open", side_effect=AssertionError("file read")), \
                mock.patch.object(report.timing.base, "load_split", side_effect=AssertionError("data")), \
                mock.patch.object(report.timing.base.experiments, "run_measured_experiment", side_effect=AssertionError("train")):
            analysis = report.analyze_task(result, task)
            compact = report.compact_task_analysis(analysis, detailed=True)
        self.assertEqual(before, pickle.dumps((result, task)))
        text = json.dumps([analysis, compact, report.COMPACT_LEGEND], allow_nan=False)
        self.assertNotIn("PRIVATE_TAG_NEVER_EXPORT", text)
        self.assertNotIn('"task_tag"', text)

    def test_missing_snapshot_or_wrong_configuration_is_explicit(self):
        task = self.task()
        self.assertEqual(report.analyze_task(None, task)["status"], "unavailable")
        result = self.result(task, [])
        task["config"]["seed"] += 1
        self.assertEqual(report.analyze_task(result, task)["status"], "invalid_evidence")

    def test_partial_rounds_are_retained_as_partial_not_completed_or_zero_filled(self):
        task = self.task()
        result = self.result(task, [self.diagnostic(25)])
        partial = replace(result, stopped_round=30, records=result.records[:31])
        analysis = report.analyze_task(partial, task)
        self.assertFalse(analysis["health"]["healthy"])
        self.assertIn("incomplete_rounds", analysis["health"]["reasons"])
        window = next(w for w in analysis["windows"] if w["name"] == "last121_150")
        self.assertFalse(window["complete_round_records"])
        self.assertEqual(window["available_rounds"], 0)
        self.assertIsNone(window["groups"]["malicious"]["counts"]["remaining_client_rounds"])

    def test_focus6_plus_brief6_preserve_raw_numbers_and_fit_under_50KB(self):
        records = [report.COMPACT_LEGEND]
        for arm in ("H0", "H1"):
            for partition in ("iid", "dirichlet"):
                for ratio in (0., .1, .7):
                    task = self.task(ratio, arm, partition)
                    observations = [self.diagnostic(rd, "client-99", malicious=False,
                        novelty_score=1.2345678901234567, anchor_score=2.345678901234567,
                        cumulative_drift=3.456789012345678, norm_score=.456789012345678,
                        signed_score=.567890123456789, class_score=.67890123456789,
                        revoked=rd == 150) for rd in range(1, 151)]
                    analysis = self.analyze(task, observations)
                    detailed = ratio == .7 or (partition == "dirichlet" and ratio == 0)
                    records.append(report.compact_task_analysis(analysis, detailed=detailed))
        text = "\n".join(json.dumps(r, separators=(",", ":"), allow_nan=False) for r in records)
        self.assertEqual(sum(r.get("detailed") is True for r in records), 6)
        self.assertLess(len(text.encode()), 50000)
        self.assertIn("1.2345678901234567", text)
        for row in records[1:]:
            self.assertEqual(len(row["windows"]), 3 if row["detailed"] else 1)
            if row["detailed"]:
                self.assertEqual(len(row["honest_case"]["observations"]), 5)
            else:
                self.assertNotIn("honest_case", row)


if __name__ == "__main__":
    unittest.main()
