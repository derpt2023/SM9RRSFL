"""Freeze a focused, costed search; historical scores never qualify a new run."""
import json
from pathlib import Path

from cifar_expanded_candidates import run_budget

REPO = Path(__file__).resolve().parent
BASE = REPO / "configs/cifar10_six_relative_asr_v6.json"
TIMING = REPO / "configs/cifar10_five_day_timing_reference.json"
OUTPUT = REPO / "configs/cifar10_six_relative_best_five_day_v7.json"


def estimate_cost(spec, timing):
    """Use each known cost, the slowest candidate mean for new configurations.

    No expected acceleration from more revocations is credited. Reserve one
    complete candidate per method for formal evaluation, with a full-run TAD
    allowance because the observed TAD mean includes early loss of clients.
    """
    recorded = timing["candidates"]
    allowances = {
        method: max(v["mean_seconds"] for c, v in recorded.items()
                    if c.startswith(method + "-"))
        for method in spec["candidates"]
    }
    allowances["ding13"] = max(allowances["ding13"], recorded["ding13-v10-001"]["max_seconds"])
    per_validation = len(spec["validation"]["seeds"]) * len(spec["validation"]["scenarios"])
    validation = sum(per_validation * recorded.get(c["candidate_id"],
                     {"mean_seconds": allowances[m]})["mean_seconds"]
                     for m, cs in spec["candidates"].items() for c in cs)
    # Impute the one missing historical TAD task at a full observed task cost.
    for m, cs in spec["candidates"].items():
        for c in cs:
            old = recorded.get(c["candidate_id"])
            if old and old["n"] < per_validation:
                validation += (per_validation - old["n"]) * (allowances[m] - old["mean_seconds"])
    per_formal = len(spec["final"]["seeds"]) * len(spec["final"]["scenarios"])
    formal = per_formal * sum(allowances.values())
    measured = sum(v["sum_seconds"] for v in recorded.values())
    overhead = max(1., timing["wall_seconds"] * timing["gpu_lanes"] / measured)
    divisor = timing["gpu_lanes"] * 3600
    v_hours, f_hours = validation / divisor * overhead, formal / divisor * overhead
    return {
        "reference": str(TIMING.relative_to(REPO)),
        "historical_validation_wall_hours": timing["wall_seconds"] / 3600,
        "reference_gpu_lanes": timing["gpu_lanes"], "gpu_name": timing["gpu_name"],
        "new_candidate_seconds_by_method": allowances,
        "observed_overhead_factor": overhead,
        "validation_hours": v_hours, "formal_reserve_hours": f_hours,
        "training_hours": v_hours + f_hours, "slowdown_allowance_fraction": .20,
        "setup_and_report_reserve_hours": 1.,
        "planned_total_hours_with_margin": (v_hours + f_hours) * 1.20 + 1.,
        "limitation": "Observational estimate, not a completion guarantee; assumes matching hardware, seven available lanes, CPU/SM9 throughput and no extra contention. No formal performance informs the search.",
    }


def build_spec_v6():
    spec = json.loads(BASE.read_text(encoding="utf-8"))
    anchor = next(c for c in spec["candidates"]["sm9rrs"] if c["candidate_id"] == "sm9rrs-v10-014")
    drift = {"detector_drift_allowance": .85, "detector_drift_threshold": 1.0}
    history = {"detector_history_threshold": .8, "detector_history_confirm": 3}
    combined = {**drift, **history}
    lower_warning = {**combined, "detector_distance_threshold": 1.10}
    probes = [
        ("A: accumulate small persistent deviations", drift),
        ("B: stricter trusted-history admission", history),
        ("C: A plus B", combined),
        ("D: C plus warning 1.10", lower_warning),
        ("D with allowance .75", {**lower_warning, "detector_drift_allowance": .75}),
        ("D with allowance .95", {**lower_warning, "detector_drift_allowance": .95}),
        ("D with drift threshold .75", {**lower_warning, "detector_drift_threshold": .75}),
        ("D with drift threshold 1.5", {**lower_warning, "detector_drift_threshold": 1.5}),
        ("C with warning 1.20", {**combined, "detector_distance_threshold": 1.20}),
        ("C with history threshold .90", {**combined, "detector_history_threshold": .90}),
        ("C with history threshold .70", {**combined, "detector_history_threshold": .70}),
        ("C with four history confirmations", {**combined, "detector_history_confirm": 4}),
        ("D with longer drift memory", {**lower_warning, "detector_drift_memory": .90}),
        ("D with slower permanent revocation", {**lower_warning, "suspicion_remove_after": 7}),
    ]
    for index, (reason, delta) in enumerate(probes, 101):
        spec["candidates"]["sm9rrs"].append({
            "candidate_id": f"sm9rrs-v10-{index:03d}", "variant": "original",
            "parameters": {**anchor["parameters"], **delta},
            "origin": "014_focused_development_20260921", "shortlist_reason": reason,
        })
    vert = next(c for c in spec["candidates"]["vert"] if c["candidate_id"] == "vert-v10-015")
    for index, delta in enumerate(({"vert_history_window": 7}, {"vert_predict_epochs": 10}), 101):
        spec["candidates"]["vert"].append({
            "candidate_id": f"vert-v10-{index}", "variant": "original",
            "parameters": {**vert["parameters"], **delta},
            "origin": "015_neighborhood_20260921", "shortlist_reason": "Lower-cost history/epoch interpolation; unchanged VERT algorithm",
        })
    for index, sparsity in enumerate((.1, .3), 101):
        spec["candidates"]["alignins"].append({
            "candidate_id": f"alignins-v10-{index}", "variant": "original",
            "parameters": {"alignins_sparsity": sparsity, "alignins_tda_radius": .75, "alignins_mpsa_radius": .75},
            "origin": "radius_interpolation_20260921", "shortlist_reason": "Intermediate existing radius controls; unchanged AlignIns algorithm",
        })
    spec.update(name="CIFAR-10 focused relative-ASR search with five-day runtime estimate",
                output_dir="outputs/cifar10_six_relative_asr_five_day_v6")
    spec["search_design"] = {
        "parent_configuration": str(BASE.relative_to(REPO)),
        "all_parent_candidates_preserved": True, "baseline_candidates_unchanged": False,
        "old_candidates_and_order_preserved": True,
        "new_candidates": {m: [c["candidate_id"] for c in cs if int(c["candidate_id"].rsplit("-", 1)[1]) >= 101]
                           for m, cs in spec["candidates"].items()},
        "selection_and_promotion_rules_unchanged": True,
        "historical_validation_used_for_search_design": True,
        "historical_scores_reused_for_new_qualification": False,
        "formal_results_used_for_selection": False,
        "validation_evidence": "outputs/cifar10_six_mnist_gate_compact_plus_v4; development evidence only, not cached qualification",
        "unequal_search_budget": "28 Ours / 6 VERT / 6 AlignIns / 1 Krum / 1 TAD / 1 FedAvg; not equal tuning compute or an exhaustive optimum",
        "fixed_method_reason": "Krum, TAD and FedAvg expose no method-specific tuning controls; no duplicate configurations added",
        "no_performance_guarantee": "Neither qualification nor best formal performance is guaranteed. Reused validation seeds are development data, not independent confirmation.",
    }
    spec["run_budget"] = run_budget(spec)
    spec["runtime_estimate"] = estimate_cost(spec, json.loads(TIMING.read_text(encoding="utf-8")))
    if spec["runtime_estimate"]["planned_total_hours_with_margin"] >= 120:
        raise ValueError("search plus formal reserve exceeds the five-day planning budget")
    spec["protocol_note"] = (
        "Freeze all 43 candidates before 1290 validation tasks; if Ours qualifies, run all 180 formal tasks. "
        "Keep v6 final-round relative minimum ASR, VERT accuracy comparison, health, Score and fallback rules. "
        "Retain all 25 parent candidates; add 14 Ours around 014 and two each VERT/AlignIns interpolations. "
        "Public training/attack parameters, splits, validation/formal seeds and core algorithms are unchanged. "
        "Historical validation informs development, never new qualification. Validation seeds have prior search history. "
        "Five days is a planning estimate only; training and reports have no wall-time deadline. "
        "Keep per-round checkpoints and ordinary resume. Use the progress wrapper for final HTML/SVG/PDF."
    )
    return spec


def build_spec():
    from cifar_relative_best_gate import ASR_TARGET, ACCURACY_TARGET, PERFORMANCE_TARGET
    spec = build_spec_v6()
    spec.update(schema_version=7, name="CIFAR-10 final Accuracy and ASR within two points of selected-method best",
                output_dir="outputs/cifar10_six_relative_best_five_day_v7",
                asr_target=dict(ASR_TARGET), accuracy_target=dict(ACCURACY_TARGET),
                performance_target=dict(PERFORMANCE_TARGET))
    spec["search_design"].update(parent_configuration="configs/cifar10_six_relative_asr_five_day_v6.json",
        selection_and_promotion_rules_unchanged=False,
        performance_gate_change="final Accuracy gap to maximum <= .02; final ASR gap to minimum <= .02; equality passes",
        candidates_unchanged_from_parent=True)
    spec["protocol_note"] = (
        "Freeze the same 43 candidates before 1290 validation tasks; run 180 formal tasks only if a healthy Ours qualifies. "
        "Each task uses round 100: best available fixed-selected method Accuracy minus Ours <= .02; "
        "each attacked task additionally requires Ours ASR minus the available fixed-selected minimum <= .02. "
        "The two best references may be different methods; equality passes. Clean ASR remains diagnostic only. "
        "Keep health, final Score, baseline fallback, mean-dual preference, candidates, shared parameters and seeds. "
        "Incomplete baseline comparisons remain explicitly incomplete without a separate veto. "
        "No wall-time cutoff. Keep round checkpoints, resume and automatic HTML/SVG/PDF via the progress wrapper. "
        "Runtime estimate is conditional, and validation qualification does not guarantee formal superiority."
    )
    return spec


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build_spec(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)
