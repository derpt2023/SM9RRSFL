"""Pure forensic summaries of existing client records; no model or checkpoint use."""
from collections import Counter, defaultdict
from math import floor

import cifar_timing_report as timing

WINDOWS = (("pre21_24", 21, 24), ("first25_29", 25, 29),
           ("last121_150", 121, 150), ("all25_150", 25, 150))
SCORES = {"novelty": "novelty_score", "anchor": "anchor_score", "drift": "cumulative_drift",
          "norm": "norm_score", "class": "class_score", "signed": "signed_score"}
TRIGGERS = ("warning_only", "drift_only", "both", "neither", "strong")
PATHS = ("immediate_only", "count_only", "both", "neither")
BLOCKS = (("1_20", 1, 20), ("21_24", 21, 24), ("25_29", 25, 29),
          ("30_120", 30, 120), ("121_150", 121, 150))
FLAGS = ("suspicious", "count_increment", "history_eligible", "immediate_revocation",
         "recovery_eligible", "history_frozen", "trace_requested", "trace_pending")
CASE_FIELDS = ("round", "decision_reason", "novelty_score", "anchor_score", "cumulative_drift",
    "norm_score", "class_score", "signed_score", "weight_before", "weight_after_penalty_recovery",
    "aggregation_weight", "count_before", "count_after", "clip_factor", "suspicious", "count_increment",
    "immediate_revocation", "trace_requested", "trace_pending", "revoked", "history_eligible",
    "history_admitted", "history_frozen", "recovery_eligible")
COUNTS = ("original_clients", "remaining_client_rounds", "observed", "gap", "accepted", "history_eligible",
          "admitted", "suspicious", "count_increment", "immediate", "trace_requested", "trace_pending",
          "revoked", "recovery_eligible", "history_frozen")

COMPACT_LEGEND = {
    "type": "forensics_legend", "schema": "cifar-history-forensics-v1",
    "groups": "M/H use original malicious/honest labels for offline diagnostics only",
    "group_counts": list(COUNTS), "triggers": list(TRIGGERS),
    "trigger_semantics": "warning/drift use strict score>warning and drift>h on raw values; first four categories partition observed records; strong=score>reject is overlapping, not a guessed main cause",
    "scores": {"order": ["novelty", "anchor", "drift"], "values": ["n", "p50", "p90", "max"]},
    "quantiles": "linear interpolation on existing client observations; not counterfactual training, optimization or a new threshold",
    "window": "[start,end,requested_rounds,available_rounds,complete]; weights=[malicious_n,mean,max,honest_loss_n,mean,max]",
    "revocations": {"summary": "[count,first_round,last_round,immediate,nonimmediate]",
        "blocks": [b[0] for b in BLOCKS], "paths": list(PATHS)},
    "case_fields": list(CASE_FIELDS),
    "case_selection": "earliest honest revocation by (round,client_id), last up to five actual observations; deliberately nonrepresentative example, not five consecutive rounds",
    "coverage": "observed=verified finite online updates; gap excludes clients revoked before that round; no observations means rates/quantiles unavailable, not zero behavior",
    "scope": "default only all25_150 counts; detailed adds 25_29 and 121_150 with three score summaries and one honest case; full helper also retains21_24, all six score fields, p99 and at most three honest cases/eight observations",
    "limits": "no task tags, keys, checkpoints or model vectors; no first model divergence or post-revocation recovery counterfactual is inferred; original health rules remain unchanged"
}


def _quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return {"n": 0, "p50": None, "p90": None, "p99": None, "max": None}
    def at(q):
        location = (len(ordered) - 1) * q
        lower = floor(location)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)
    return {"n": len(ordered), "p50": at(.5), "p90": at(.9), "p99": at(.99), "max": ordered[-1]}


def _triggers(observations, config):
    counts = dict.fromkeys(TRIGGERS, 0)
    for d in observations:
        warning = d.novelty_score > config.detector_distance_threshold
        drift = d.cumulative_drift > config.detector_drift_threshold
        counts["both" if warning and drift else "warning_only" if warning else "drift_only" if drift else "neither"] += 1
        counts["strong"] += d.novelty_score > config.detector_reject_threshold
    return counts


def _validate_details(result):
    by_client = defaultdict(list)
    for d in result.diagnostics:
        if not isinstance(d.client_id, str) or not d.client_id:
            raise ValueError("missing or invalid public client identity")
        for field in FLAGS:
            if type(getattr(d, field, None)) is not bool:
                raise ValueError("missing or invalid client diagnostic flag: " + field)
        for field in (*SCORES.values(), "weight_before", "weight_after_penalty_recovery", "count_before", "count_after", "clip_factor"):
            if not timing.finite(getattr(d, field, None)):
                raise ValueError("missing or nonfinite client diagnostic number: " + field)
        if not isinstance(d.decision_reason, str) or not d.decision_reason:
            raise ValueError("missing or invalid client decision reason")
        by_client[d.client_id].append(d)
    for observations in by_client.values():
        observations.sort(key=lambda d: d.round)
        revoked = False
        for d in observations:
            if revoked:
                raise ValueError("client observation occurs after permanent revocation")
            revoked = d.revoked
    return by_client


def _window(result, records, diagnostics, name, start, end):
    source = timing.mechanism_window(result, records, diagnostics, start, end - start + 1)
    output = {"name": name, "start": start, "end": end,
        "requested_rounds": end - start + 1, "available_rounds": len(source["rounds_observed"]),
        "complete_round_records": source["complete_round_records"], "groups": {}}
    for field in ("malicious_weight_mass", "honest_weight_loss"):
        output[field] = {k: source[field][k] for k in ("n", "mean", "max")}
    for role, malicious in (("malicious", True), ("honest", False)):
        original = source["client_diagnostics"][role]
        observations = [d for rd in range(start, end + 1) for d in diagnostics.get(rd, []) if d.is_malicious == malicious]
        n = len(observations)
        counts = {"original_clients": original["original_clients"],
            "remaining_client_rounds": original["remaining_client_rounds"], "observed": n,
            "gap": original["unobserved_remaining_client_rounds"],
            "accepted": sum(d.aggregation_accepted for d in observations),
            "admitted": sum(d.history_admitted for d in observations),
            "revoked": sum(d.revoked for d in observations),
            **{field: sum(getattr(d, field) for d in observations) for field in FLAGS}}
        counts["immediate"] = counts.pop("immediate_revocation")
        output["groups"][role] = {"status": original["status"], "counts": counts,
            "accepted_rate": counts["accepted"] / n if n else None,
            "history_eligible_rate": counts["history_eligible"] / n if n else None,
            "admitted_rate": counts["admitted"] / n if n else None,
            "triggers": _triggers(observations, result.config),
            "decision_reasons": dict(sorted(Counter(d.decision_reason for d in observations).items())),
            "scores": {key: _quantiles([getattr(d, field) for d in observations]) for key, field in SCORES.items()}}
    return output


def _revocations(result, records, by_client):
    output = {}
    events = sorted([d for d in result.diagnostics if d.revoked], key=lambda d: (d.round, d.client_id))
    for role, malicious in (("malicious", True), ("honest", False)):
        selected = [d for d in events if d.is_malicious == malicious]
        paths = dict.fromkeys(PATHS, 0)
        for d in selected:
            immediate = d.immediate_revocation
            counted = d.count_increment and d.count_after >= result.config.suspicion_remove_after
            paths["both" if immediate and counted else "immediate_only" if immediate else "count_only" if counted else "neither"] += 1
        examples = []
        if role == "honest":
            for d in selected[:3]:
                recent = [o for o in by_client[d.client_id] if o.round <= d.round][-8:]
                examples.append({"client_id": d.client_id, "revoked_round": d.round,
                    "observations_available_before_revocation": len(by_client[d.client_id]),
                    "observations": [{field: getattr(o, field) for field in CASE_FIELDS} for o in recent]})
        output[role] = {"count": len(selected), "first_round": selected[0].round if selected else None,
            "last_round": selected[-1].round if selected else None,
            "round_blocks": {name: sum(start <= d.round <= end for d in selected) for name, start, end in BLOCKS},
            "immediate": sum(d.immediate_revocation for d in selected),
            "nonimmediate": sum(not d.immediate_revocation for d in selected), "paths": paths,
            "trigger_types": _triggers(selected, result.config),
            "decision_reasons": dict(sorted(Counter(d.decision_reason for d in selected).items())),
            "examples": examples, "examples_are_nonrepresentative": True}
    mismatches = []
    for rd, record in records.items():
        honest = sum(d.round <= rd and not d.is_malicious for d in events)
        malicious = sum(d.round <= rd and d.is_malicious for d in events)
        if honest != record.false_positive_revocations or malicious != record.true_positive_revocations:
            mismatches.append({"round": rd, "diagnostic_FP": honest, "record_FP": record.false_positive_revocations,
                "diagnostic_TP": malicious, "record_TP": record.true_positive_revocations})
    final_ids = set(result.blacklisted_clients)
    diagnosed_ids = {d.client_id for d in events}
    consistency = {"consistent": not mismatches and final_ids == diagnosed_ids,
        "checked_rounds": len(records), "mismatched_rounds": len(mismatches),
        "first_counter_mismatch": mismatches[0] if mismatches else None,
        "blacklist_matches_observed_revocations": final_ids == diagnosed_ids,
        "diagnostic_FP": output["honest"]["count"], "diagnostic_TP": output["malicious"]["count"],
        "record_FP": result.records[-1].false_positive_revocations,
        "record_TP": result.records[-1].true_positive_revocations}
    return output, consistency


def analyze_task(result, task):
    """Summarize supplied evidence without loading data, files, CUDA or model state."""
    config = task["config"]
    output = {"task_id": task["task_id"], "arm": task.get("arm"), "partition": config["partition"],
        "seed": config["seed"], "ratio": config["malicious_ratio"], "status": "unavailable",
        "health": None, "windows": [], "revocations": None, "revocation_consistency": None}
    if result is None:
        return {**output, "error": "completed snapshot unavailable; no observations imputed"}
    try:
        if timing.base.semantic_config(result.config) != timing.base.semantic_config(config):
            raise ValueError("snapshot scientific configuration differs from task")
        if result.config.method != "sm9rrs":
            raise ValueError("history forensics expects original or declared SM9 history variant")
        records = timing._records(result)
        diagnostics = timing._diagnostics(result, records)
        output["health"] = {**timing.base.metrics(result),
            "original_honest": config["num_clients"] - len(result.malicious_clients),
            "original_malicious": len(result.malicious_clients),
            "FP": result.records[-1].false_positive_revocations, "TP": result.records[-1].true_positive_revocations,
            "nonfinite_updates": result.nonfinite_updates, "stopped_round": result.stopped_round}
        by_client = _validate_details(result)
        output["windows"] = [_window(result, records, diagnostics, name, start, end) for name, start, end in WINDOWS]
        output["revocations"], output["revocation_consistency"] = _revocations(result, records, by_client)
        output["status"] = "valid" if output["revocation_consistency"]["consistent"] else "invalid_evidence"
        if output["status"] != "valid":
            output["error"] = "actual revocation diagnostics disagree with cumulative records or final blacklist"
    except Exception as exc:
        output.update(status="invalid_evidence", error=str(exc))
    return output


def _compact_group(group, *, scores):
    output = {"c": [group["counts"][field] for field in COUNTS], "status": group["status"],
        "tr": [group["triggers"][field] for field in TRIGGERS], "reasons": group["decision_reasons"]}
    if scores:
        output["s"] = [[group["scores"][name][field] for field in ("n", "p50", "p90", "max")]
                       for name in ("novelty", "anchor", "drift")]
    return output


def compact_task_analysis(analysis, detailed=False):
    """Keep original numbers; detailed defaults to two score windows and one case."""
    output = {"type": "client_forensics", "id": analysis["task_id"], "arm": analysis.get("arm"),
        "p": analysis["partition"], "seed": analysis["seed"], "ratio": analysis["ratio"],
        "status": analysis["status"], "detailed": detailed}
    if "error" in analysis:
        output["error"] = analysis["error"]
    health = analysis.get("health")
    if health is not None:
        output["health"] = {key: health[key] for key in
            ("healthy", "reasons", "original_honest", "original_malicious", "FP", "TP", "nonfinite_updates", "stopped_round")}
    selected = {"all25_150", "first25_29", "last121_150"} if detailed else {"all25_150"}
    windows = []
    for window in analysis.get("windows", []):
        if window["name"] not in selected:
            continue
        row = {"name": window["name"], "r": [window[field] for field in
            ("start", "end", "requested_rounds", "available_rounds", "complete_round_records")],
            "weights": [window[metric][field] for metric in ("malicious_weight_mass", "honest_weight_loss")
                        for field in ("n", "mean", "max")]}
        for role, label in (("malicious", "M"), ("honest", "H")):
            row[label] = _compact_group(window["groups"][role], scores=detailed and window["name"] != "all25_150")
        windows.append(row)
    output["windows"] = windows
    if analysis.get("revocations") is not None:
        revoked = {}
        for role, label in (("malicious", "M"), ("honest", "H")):
            value = analysis["revocations"][role]
            revoked[label] = {"n": [value[field] for field in ("count", "first_round", "last_round", "immediate", "nonimmediate")],
                "blocks": [value["round_blocks"][name] for name, _, _ in BLOCKS],
                "paths": [value["paths"][name] for name in PATHS],
                "tr": [value["trigger_types"][name] for name in TRIGGERS], "reasons": value["decision_reasons"]}
        output["revocations"] = revoked
        if detailed and analysis["revocations"]["honest"]["examples"]:
            example = analysis["revocations"]["honest"]["examples"][0]
            output["honest_case"] = {"client_id": example["client_id"], "revoked_round": example["revoked_round"],
                "observations_available_before_revocation": example["observations_available_before_revocation"],
                "observations": [[row[field] for field in CASE_FIELDS] for row in example["observations"][-5:]]}
    output["revocation_consistency"] = analysis.get("revocation_consistency")
    return output
