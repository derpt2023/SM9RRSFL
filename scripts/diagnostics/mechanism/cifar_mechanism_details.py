"""Pure public-scalar drilldown for the frozen clean history-mechanism evidence.

Inputs are observations already authenticated and validated by the independent
reader. This module opens no files, trains no models and infers no distances
from fingerprints. Every signed difference is H1 minus H0 on common clients.
"""
import math

SCORES = ("novelty_score", "anchor_score", "signed_score", "class_score", "cumulative_drift", "norm_score", "clip_factor")
DECISION_FLAGS = ("accepted", "would_flag", "count_increment", "history_eligible", "immediate_revocation", "recovery_eligible")
DECISION_FIELDS = ("accepted", "reason", "would_flag", "count_increment", *SCORES,
    "history_eligible", "trusted_history_size", "normal_cluster_count", "immediate_revocation", "recovery_eligible")
DIAGNOSTIC_FIELDS = ("decision_reason", "suspicious", "count_increment", *SCORES,
    "weight_before", "weight_after_penalty_recovery", "aggregation_weight", "count_before", "count_after",
    "trace_requested", "trace_pending", "revoked", "aggregation_accepted", "history_eligible", "history_admitted",
    "history_frozen", "immediate_revocation", "trusted_history_size", "normal_cluster_count", "recovery_eligible")
TRAJECTORY_FIELDS = ("novelty_score", "cumulative_drift", "anchor_score", "norm_score", "clip_factor",
    "weight_before", "weight_after_penalty_recovery", "aggregation_weight", "count_before", "count_after",
    "suspicious", "count_increment", "aggregation_accepted", "history_eligible", "history_admitted", "recovery_eligible", "revoked")
THRESHOLDS = {"warning": ("novelty_score", "detector_distance_threshold"),
              "drift": ("cumulative_drift", "detector_drift_threshold"),
              "strong": ("novelty_score", "detector_reject_threshold")}
NUMERIC_DECISIONS = (*SCORES, "trusted_history_size", "normal_cluster_count")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _finite_json(value):
    if isinstance(value, dict):
        return all(isinstance(k, str) and _finite_json(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite_json(v) for v in value)
    return value is None or type(value) in (str, bool) or _number(value)


def _index(rows, key="client_id"):
    _require(isinstance(rows, list), "public observation list is required")
    out = {}
    for row in rows:
        identity = row.get(key)
        _require((type(identity) is int if key == "round" else isinstance(identity, str) and bool(identity))
                 and identity not in out, "missing or duplicate public " + key)
        out[identity] = row
    return out


def _ids(values):
    return sorted(values, key=lambda cid: (int(cid.split("-")[-1]) if cid.startswith("client-") and cid.split("-")[-1].isdigit() else 10**9, cid))


def _summary(values):
    values = list(values)
    _require(all(_number(v) for v in values), "nonfinite or nonnumeric scalar comparison")
    if not values:
        return {"n": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None, "max_abs": None, "mean": None}
    ordered = sorted(values)
    def quantile(q):
        position = (len(ordered) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    return {"n": len(values), "min": ordered[0], "p50": quantile(.5), "p90": quantile(.9),
        "p99": quantile(.99), "max": ordered[-1], "max_abs": max(map(abs, values)), "mean": math.fsum(values) / len(values)}


def _scalar_pair(h1, h0):
    _require(_number(h1) and _number(h0), "missing or invalid paired scalar")
    return {"H0": h0, "H1": h1, "delta": h1 - h0}


def _select(value, fields):
    _require(isinstance(value, dict) and all(f in value for f in fields), "incomplete public scalar record")
    return {f: value[f] for f in fields}


def _rounds(payload):
    _require(isinstance(payload, dict), "observation payload is required")
    return _index(payload.get("rounds", []), "round")


def _events(payload, rd):
    selected = [e for e in payload.get("history_events", []) if e.get("round") == rd]
    _index(selected)
    return selected


def _coverage(row, config):
    if row is None:
        return {"available": False, "active_clients": None, "trained_clients": None,
            "observed_diagnostics": None, "nonfinite_updates": None, "missing_active_ids": None}
    clients = _index(row.get("clients", []))
    diagnostics = _index(row.get("diagnostics", []))
    starting_blacklist = row.get("blacklisted_before")
    _require(isinstance(starting_blacklist, list) and len(set(starting_blacklist)) == len(starting_blacklist), "invalid public starting blacklist")
    _require(all(type(c.get("update_finite")) is bool for c in clients.values()), "missing actual update finiteness")
    expected_active = {"client-" + str(i) for i in range(config["num_clients"])} - set(starting_blacklist)
    return {"available": True, "active_clients": len(expected_active), "trained_clients": len(clients),
        "observed_diagnostics": len(diagnostics), "nonfinite_updates": sum(not c["update_finite"] for c in clients.values()),
        "revoked_before_ids": _ids(starting_blacklist), "missing_active_ids": _ids(expected_active - set(diagnostics)),
        "untrained_active_ids": _ids(expected_active - set(clients)),
        "nonfinite_client_ids": _ids(cid for cid, c in clients.items() if not c["update_finite"]),
        "revoked_this_round_ids": _ids(cid for cid, d in diagnostics.items() if d["revoked"])}


def _local_equality(h1, h0, rd):
    a, b = _rounds(h1).get(rd), _rounds(h0).get(rd)
    if a is None or b is None:
        return {"available": False, "equal": None, "common_client_count": None, "different_client_ids": None}
    ac, bc = _index(a["clients"]), _index(b["clients"])
    common = _ids(set(ac) & set(bc))
    def extra(payload, name, cid):
        return [r for r in payload.get(name, []) if r.get("round") == rd and r.get("client_id") == cid]
    different = [cid for cid in common if ac[cid] != bc[cid] or any(extra(h1, name, cid) != extra(h0, name, cid)
        for name in ("actual_batches", "singleton_batches"))]
    return {"available": True, "equal": not different and list(ac) == list(bc)
            and a["blacklisted_before"] == b["blacklisted_before"], "common_client_count": len(common),
        "different_client_ids": different, "H0_only_client_ids": _ids(set(bc) - set(ac)), "H1_only_client_ids": _ids(set(ac) - set(bc)),
        "starting_active_order_equal": list(ac) == list(bc), "starting_blacklist_equal": a["blacklisted_before"] == b["blacklisted_before"],
        "scope": "exact public local inputs, raw/returned update hashes, statistics and recorded actual batch/singleton fields; no vector distance"}


def _coefficient_comparison(a, b):
    if a is None or b is None:
        return {"available": False, "common_clients": None, "delta_l1": None, "delta_max_abs": None,
            "delta_signed_sum": None, "H0": None, "H1": None, "honest_weight_loss": None, "reliability": None}
    av, bv = a.get("coefficients", {}).get("by_client"), b.get("coefficients", {}).get("by_client")
    _require(isinstance(av, dict) and isinstance(bv, dict), "missing actual per-client coefficient map")
    _require(all(isinstance(cid, str) and _number(v) and 0 <= v <= 1 + 1e-9 for m in (av, bv) for cid, v in m.items()), "invalid actual coefficient map")
    common = _ids(set(av) & set(bv))
    delta = {cid: av[cid] - bv[cid] for cid in common}
    max_delta = max(map(abs, delta.values())) if common else None
    max_ids = [cid for cid in common if max_delta and abs(delta[cid]) == max_delta]
    def arm(values):
        return {"observed_clients": len(values), "positive_support": _ids(cid for cid, v in values.items() if v > 0),
                "sum": math.fsum(values.values()) if values else None}
    ad, bd = _index(a["diagnostics"]), _index(b["diagnostics"])
    both = _ids(set(ad) & set(bd))
    reliability = {}
    for f in ("weight_before", "weight_after_penalty_recovery"):
        reliability[f] = {"delta": _summary(ad[cid][f] - bd[cid][f] for cid in both),
            "changed_client_ids": [cid for cid in both if ad[cid][f] != bd[cid][f]]}
    _require(all(_number(d[f]) for ds in (ad, bd) for d in ds.values() for f in ("weight_before", "weight_after_penalty_recovery")), "missing reliability values")
    return {"available": True, "common_clients": len(common), "identity_coverage_equal": set(av) == set(bv),
        "H0_only_client_ids": _ids(set(bv) - set(av)), "H1_only_client_ids": _ids(set(av) - set(bv)),
        "delta_l1": math.fsum(abs(v) for v in delta.values()) if common else None,
        "delta_max_abs": max_delta, "max_abs_delta_client_ids": max_ids,
        "max_abs_delta_example": None if not max_ids else {"client_id": max_ids[0], **_scalar_pair(av[max_ids[0]], bv[max_ids[0]])},
        "delta_signed_sum": math.fsum(delta.values()) if common else None,
        "changed_clients": len([v for v in delta.values() if v != 0]),
        "changed_client_ids": [cid for cid in common if delta[cid] != 0],
        "changed_coefficients": [[cid, bv[cid], av[cid], delta[cid]] for cid in common if delta[cid] != 0],
        "changed_coefficient_order": ["client_id", "H0", "H1", "delta"],
        "support_gained_common_ids": [cid for cid in common if av[cid] > 0 and bv[cid] == 0],
        "support_lost_common_ids": [cid for cid in common if av[cid] == 0 and bv[cid] > 0],
        "H0": arm(bv), "H1": arm(av), "honest_weight_loss": _scalar_pair(a["record"]["honest_weight_loss"], b["record"]["honest_weight_loss"]),
        "reliability_common_clients": len(both), "reliability": reliability,
        "scope": "delta norms use common observed coefficients only; absent clients are not filled with zero; reliability is the weight-manager value before the final sample/cap/clip coefficient"}


def _thresholds(decision, config):
    return {name: {"value": decision[field], "threshold": config[key], "margin": decision[field] - config[key],
                   "exceeds": decision[field] > config[key]} for name, (field, key) in THRESHOLDS.items()}


def _case_arm(event, diagnostic, config):
    if event is None or diagnostic is None:
        return None
    decision = _select(event["decision"], DECISION_FIELDS)
    diagnostic = _select(diagnostic, DIAGNOSTIC_FIELDS)
    return {"decision": decision, "diagnostic": diagnostic, "strict_thresholds": _thresholds(decision, config),
        "admission_margins": {"novelty_minus_history_threshold": decision["novelty_score"] - config["detector_history_threshold"],
            "anchor_minus_reference_budget": decision["anchor_score"] - config["detector_reference_budget"],
            "drift_minus_allowance": decision["cumulative_drift"] - config["detector_drift_allowance"],
            "novelty_minus_allowance": decision["novelty_score"] - config["detector_drift_allowance"],
            "clip_minus_one": decision["clip_factor"] - 1},
        "removal_margins": {"before_count_minus_limit": diagnostic["count_before"] - config["suspicion_remove_after"],
            "after_count_minus_limit": diagnostic["count_after"] - config["suspicion_remove_after"],
            "count_route_requested": diagnostic["count_increment"] and diagnostic["count_after"] >= config["suspicion_remove_after"],
            "immediate_route_requested": diagnostic["immediate_revocation"]}}


def _decision_comparison(h1, h0, rd, config):
    ae, be = _events(h1, rd), _events(h0, rd)
    am, bm = _index(ae), _index(be)
    a, b = _rounds(h1).get(rd), _rounds(h0).get(rd)
    if a is None or b is None:
        return {"available": False, "common_clients": None, "first_decision_difference": None,
                "score_deltas": None, "threshold_flips": None, "flag_flips": None, "flip_client_ids": []}
    common = [e["client_id"] for e in ae if e["client_id"] in bm]
    order_equal = list(am) == list(bm)
    changed, continuous, flag_flips, threshold_flips = [], [], {}, {}
    for cid in common:
        for event in (am[cid], bm[cid]):
            decision = _select(event["decision"], DECISION_FIELDS)
            _require(all(type(decision[f]) is bool for f in DECISION_FLAGS) and isinstance(decision["reason"], str), "invalid detector flags/reason")
            _require(all(_number(decision[f]) for f in NUMERIC_DECISIONS), "nonfinite detector scores")
        x, y = am[cid]["decision"], bm[cid]["decision"]
        if x != y:
            changed.append(cid)
        if any(x[f] != y[f] for f in SCORES):
            continuous.append(cid)
    first_id = changed[0] if changed else None
    for field in (*DECISION_FLAGS, "reason"):
        flag_flips[field] = [{"client_id": cid, "H0": bm[cid]["decision"][field], "H1": am[cid]["decision"][field]}
                            for cid in common if am[cid]["decision"][field] != bm[cid]["decision"][field]]
    for name, (field, key) in THRESHOLDS.items():
        threshold_flips[name] = [{"client_id": cid, "H0": _thresholds(bm[cid]["decision"], config)[name],
                                  "H1": _thresholds(am[cid]["decision"], config)[name]}
            for cid in common if (am[cid]["decision"][field] > config[key]) != (bm[cid]["decision"][field] > config[key])]
    flip_ids = set(row["client_id"] for rows in (*flag_flips.values(), *threshold_flips.values()) for row in rows)
    return {"available": True, "common_clients": len(common), "event_order_equal": order_equal,
        "H0_event_count": len(be), "H1_event_count": len(ae),
        "H0_only_client_ids": _ids(set(bm) - set(am)), "H1_only_client_ids": _ids(set(am) - set(bm)),
        "first_decision_difference": None if first_id is None else {"client_id": first_id,
            "H1_event_index": list(am).index(first_id), "H0_event_index": list(bm).index(first_id),
            "actual_common_order_certified": order_equal,
            "different_fields": [f for f in DECISION_FIELDS if am[first_id]["decision"][f] != bm[first_id]["decision"][f]],
            "H0": _select(bm[first_id]["decision"], DECISION_FIELDS), "H1": _select(am[first_id]["decision"], DECISION_FIELDS)},
        "score_deltas": {f: _summary(am[cid]["decision"][f] - bm[cid]["decision"][f] for cid in common) for f in SCORES},
        "changed_decision_clients": len(changed), "changed_decision_client_ids": changed,
        "continuous_score_changed_clients": len(continuous), "continuous_score_changed_client_ids": continuous,
        "continuous_changes_without_discrete_flip_clients": len(set(continuous) - flip_ids),
        "threshold_flips": threshold_flips, "flag_flips": flag_flips, "flip_client_ids": [cid for cid in common if cid in flip_ids],
        "threshold_sets": {name: {
            "common_exceeding_ids": [cid for cid in common if am[cid]["decision"][field] > config[key] and bm[cid]["decision"][field] > config[key]],
            "H0_only_exceeding_ids": [cid for cid in common if bm[cid]["decision"][field] > config[key] and not am[cid]["decision"][field] > config[key]],
            "H1_only_exceeding_ids": [cid for cid in common if am[cid]["decision"][field] > config[key] and not bm[cid]["decision"][field] > config[key]],
            "scope": "common observed clients only; set differences are threshold-status swaps, not missing client coverage"}
            for name, (field, key) in THRESHOLDS.items()},
        "threshold_counts": {name: {"H0": sum(bm[cid]["decision"][field] > config[key] for cid in common),
            "H1": sum(am[cid]["decision"][field] > config[key] for cid in common), "denominator": len(common)}
            for name, (field, key) in THRESHOLDS.items()}}


def _pre_evaluate_state_comparison(h1, h0, rd):
    if rd not in _rounds(h1) or rd not in _rounds(h0):
        return {"available": False, "common_clients": None, "fields": None}
    am, bm = _index(_events(h1, rd)), _index(_events(h0, rd))
    common = [cid for cid in am if cid in bm]
    presence = [cid for cid in common if (am[cid]["before_evaluate"] is None) != (bm[cid]["before_evaluate"] is None)]
    both = [cid for cid in common if am[cid]["before_evaluate"] is not None and bm[cid]["before_evaluate"] is not None]
    scalar_fields = ("history_size", "norm_limit", "drift", "clean_streak", "recovery_streak", "last_round")
    fields = {}
    for f in ("history", "normal", "anchor", *scalar_fields):
        # Missing state fields stay explicit; never replace absent values with zero.
        observed = [cid for cid in both if f in am[cid]["before_evaluate"] and f in bm[cid]["before_evaluate"]]
        changed = [cid for cid in observed if am[cid]["before_evaluate"][f] != bm[cid]["before_evaluate"][f]]
        values = {"common_field_observations": len(observed), "differing_client_ids": changed,
            "missing_field_ids": [cid for cid in both if cid not in observed]}
        if f in scalar_fields:
            values["scalar_deltas"] = _summary(am[cid]["before_evaluate"][f] - bm[cid]["before_evaluate"][f] for cid in observed)
            values["changed_scalars"] = [[cid, bm[cid]["before_evaluate"][f], am[cid]["before_evaluate"][f],
                am[cid]["before_evaluate"][f] - bm[cid]["before_evaluate"][f]] for cid in changed]
        fields[f] = values
    return {"available": True, "common_clients": len(common), "both_state_observed_clients": len(both),
        "state_presence_differing_ids": presence, "fields": fields,
        "changed_scalar_order": ["client_id", "H0", "H1", "delta"],
        "scope": "public state immediately before detector.evaluate; fingerprint comparisons report equality only"}


def _intervention(h1, h0):
    ae, be = _events(h1, 25), _events(h0, 25)
    am, bm = _index(ae), _index(be)
    available = 25 in _rounds(h1) and 25 in _rounds(h0)
    common = [e["client_id"] for e in ae if e["client_id"] in bm]
    def arm(events, present):
        if not present:
            return None
        commits = [e["commit"] for e in events if e.get("commit") is not None]
        return {"observed_events": len(events), "actual_commits": len(commits),
            "requested_admission": sum(c["requested_admission"] for c in commits), "admitted": sum(c["admitted"] for c in commits),
            "history_changed_ids": [e["client_id"] for e in events if e.get("commit") is not None
                and e["commit"]["before"]["history"] != e["commit"]["after"]["history"]],
            "live_normal_changed_ids": [e["client_id"] for e in events if e.get("commit") is not None
                and e["commit"]["before"]["normal"] != e["commit"]["after"]["normal"]],
            "forgotten_ids": [e["client_id"] for e in events if e.get("forget") is not None]}
    state_changes = {f: [] for f in ("history", "history_size", "normal", "anchor", "norm_limit", "drift", "clean_streak", "recovery_streak")}
    suppressed = []
    pre_equal = True
    for cid in common:
        a, b = am[cid], bm[cid]
        pre_equal = pre_equal and all(a.get(f) == b.get(f) for f in ("before_evaluate", "after_evaluate", "decision", "forget"))
        ac, bc = a.get("commit"), b.get("commit")
        if ac is not None and bc is not None:
            pre_equal = pre_equal and ac["before"] == bc["before"] and ac["requested_admission"] == bc["requested_admission"]
            if bc["admitted"] and not ac["admitted"]:
                suppressed.append(cid)
            for field in state_changes:
                if ac["after"][field] != bc["after"][field]:
                    state_changes[field].append(cid)
        elif ac != bc:
            pre_equal = False
    return {"available": available, "common_clients": len(common) if available else None,
        "event_order_equal": list(am) == list(bm) if available else None,
        "common_precommit_equal": pre_equal and list(am) == list(bm) if available and common else None,
        "H0": arm(be, 25 in _rounds(h0)), "H1": arm(ae, 25 in _rounds(h1)),
        "paired_suppressed_admission_ids": suppressed, "postcommit_state_differing_ids": state_changes,
        "scope": "state equality from recorded public fingerprints/scalars; no history/model vector distance"}


def analyze_pair(h1_payload, h0_payload, task_config):
    """Drill down authenticated observations without I/O or protocol changes."""
    config = dict(task_config)
    for field in (*[k for _, k in THRESHOLDS.values()], "detector_history_threshold", "detector_reference_budget",
                  "detector_drift_allowance", "suspicion_remove_after"):
        _require(_number(config.get(field)), "missing or invalid public threshold: " + field)
    _require(type(config.get("num_clients")) is int and config["num_clients"] > 0, "invalid original client population")
    _require(config.get("malicious_ratio") == 0, "details require the declared clean panel")
    rounds1, rounds0 = _rounds(h1_payload), _rounds(h0_payload)
    comparison = _decision_comparison(h1_payload, h0_payload, 26, config)
    diagnostics1 = {rd: _index(r["diagnostics"]) for rd, r in rounds1.items()}
    diagnostics0 = {rd: _index(r["diagnostics"]) for rd, r in rounds0.items()}
    event1, event0 = _index(_events(h1_payload, 26)), _index(_events(h0_payload, 26))
    cases = []
    for cid in comparison["flip_client_ids"]:
        trajectories = []
        for rd in range(21, 31):
            def point(rows, ds):
                if rd not in rows:
                    return {"status": "round_unavailable", "values": None}
                d = ds.get(rd, {}).get(cid)
                if d is not None:
                    return {"status": "observed", "values": [d[f] for f in TRAJECTORY_FIELDS]}
                clients = _index(rows[rd]["clients"])
                status = "revoked_before_round" if cid in rows[rd]["blacklisted_before"] else (
                    "nonfinite_update" if cid in clients and not clients[cid]["update_finite"] else "missing_observation")
                return {"status": status, "values": None}
            trajectories.append({"round": rd, "H0": point(rounds0, diagnostics0), "H1": point(rounds1, diagnostics1)})
        cases.append({"client_id": cid, "round": 26,
            "H0": _case_arm(event0.get(cid), diagnostics0.get(26, {}).get(cid), config),
            "H1": _case_arm(event1.get(cid), diagnostics1.get(26, {}).get(cid), config), "trajectory": trajectories})
    later = []
    for rd in range(27, 31):
        decision = _decision_comparison(h1_payload, h0_payload, rd, config)
        later.append({"round": rd, "scope": "post_intervention_observations_no_same_input_causal_attribution",
            "local_training_equality": _local_equality(h1_payload, h0_payload, rd),
            "coverage": {"H0": _coverage(rounds0.get(rd), config), "H1": _coverage(rounds1.get(rd), config)},
            "changed_decision_clients": decision.get("changed_decision_clients"),
            "threshold_counts": decision.get("threshold_counts"), "flip_client_ids": decision["flip_client_ids"],
            "coefficients": _coefficient_comparison(rounds1.get(rd), rounds0.get(rd))})
    value = {"schema": "cifar-clean-history-mechanism-details-v1", "direction": "H1_minus_H0",
        "round25": _intervention(h1_payload, h0_payload),
        "round26": {"pre_evaluate_state": _pre_evaluate_state_comparison(h1_payload, h0_payload, 26),
            "local_training_equality": _local_equality(h1_payload, h0_payload, 26),
            "coverage": {"H0": _coverage(rounds0.get(26), config), "H1": _coverage(rounds1.get(26), config)},
            "decisions": comparison, "coefficients": _coefficient_comparison(rounds1.get(26), rounds0.get(26))},
        "flip_cases": cases, "case_trajectory_fields": list(TRAJECTORY_FIELDS), "rounds27_30": later,
        "limits": {"training_started": False, "tensor_distance_inferred": False, "performance_improvement_assessed": False,
            "formal_qualification_assessed": False, "quantiles": "linear interpolation of observed signed scalar deltas on common clients",
            "thresholds": "warning/drift/strong all use strict >; zero margin is not an exceedance; strong overlaps warning",
            "cases": "all round26 strict-threshold or discrete decision/flag flips; continuous-only changes summarized over all common clients",
            "causality": "authentication, repeated trajectories and common-condition eligibility are established by the independent reader; later changed local inputs invalidate same-input comparisons"}}
    _require(_finite_json(value), "details contain nonfinite or non-public JSON values")
    return value
