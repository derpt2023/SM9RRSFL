"""Predeclared CIFAR v4 search space; no experiment results are read here.

The historical rows freeze *parameters*, not historical scores or a chosen
winner. Every candidate must be evaluated again on the v4 validation split.
Only existing defense-specific interfaces vary; public training, attack and
detector warm-up settings remain those of the full CIFAR v3 protocol.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
import json
from pathlib import Path


REPO = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO / "configs/cifar10_six_mnist_gate_v4.json"
METHODS = ("sm9rrs", "vert", "alignins", "krum", "ding13", "fedavg")
EXPECTED_CANDIDATE_COUNTS = {
    "sm9rrs": 42, "vert": 24, "alignins": 18,
    "krum": 1, "ding13": 1, "fedavg": 1,
}

# These 16 values per row reproduce the 12 strategies in the historical
# MNIST tuning_trials.csv. Keeping this table in source avoids requiring an
# ignored outputs/ directory on the training station.
OURS_POLICY_FIELDS = (
    "detector_subspace_dim", "detector_normal_clusters",
    "detector_distance_threshold", "detector_reject_threshold",
    "detector_drift_memory", "detector_drift_allowance",
    "detector_drift_threshold", "detector_history_confirm",
    "detector_history_threshold", "detector_recovery_confirm",
    "detector_reference_budget", "detector_clip_factor",
    "detector_weight_cap", "suspicion_penalty_factor",
    "suspicion_recovery_factor", "suspicion_remove_after",
)
MNIST_OURS_ROWS = (
    (2, 2, 3., 6., .8, 2.5, 6., 2, 2., 2, 3.5, 2., 2., .5, 1.25, 2),
    (2, 2, 3., 6., .8, 2.5, 6., 2, 2., 2, 3.5, 2., 2., .5, 1.25, 3),
    (2, 2, 3., 6., .8, 2.5, 6., 2, 2., 2, 3.5, 2., 2., .5, 1.25, 1),
    (2, 2, 3., 6., .8, 2.5, 6., 2, 2., 2, 3.5, 2., 2., .5, 1.25, 5),
    (2, 2, 2.5, 5., .8, 2., 8., 2, 1.75, 2, 3., 2., 2., .75, 1.1, 5),
    (3, 2, 2.5, 5., .9, 2., 8., 3, 1.75, 3, 3.5, 2., 2., .25, 1.25, 1),
    (1, 1, 3., 6., .8, 2.5, 4., 2, 2., 2, 2.5, 2., 2., .75, 1.1, 3),
    (2, 1, 3., 6., .9, 2.5, 4., 3, 2., 3, 3., 2., 2., .25, 1.25, 5),
    (3, 1, 3., 6., .8, 2.5, 6., 2, 2., 2, 3.5, 3., 2.5, .5, 1.1, 3),
    (1, 2, 3.5, 7., .9, 3., 6., 3, 2.5, 3, 2.5, 3., 2.5, .25, 1.25, 5),
    (2, 2, 3.5, 7., .8, 3., 8., 2, 2.5, 2, 3., 3., 2.5, .5, 1.1, 1),
    (3, 2, 3.5, 7., .9, 3., 8., 3, 2.5, 3, 3.5, 3., 2.5, .75, 1.25, 3),
)
CIFAR_OURS_ANCHOR = dict(zip(OURS_POLICY_FIELDS, (
    2, 2, 1.75, 6., .8, 1.25, 6., 2, 1., 2, 3.5, 2., 2., .5, 1.25, 3,
)))

# Each bridge changes exactly one field of the CIFAR anchor. These are
# interpretable sensitivity probes, not an exhaustive Cartesian product.
# In particular, history_threshold < warning_threshold and
# drift_allowance <= warning_threshold must remain valid.
OURS_BRIDGES = (
    ("detector_distance_threshold", 2.),
    ("detector_distance_threshold", 3.),
    ("detector_distance_threshold", 3.5),
    ("detector_subspace_dim", 1),
    ("detector_subspace_dim", 3),
    ("detector_subspace_dim", 4),
    ("detector_normal_clusters", 1),
    ("detector_normal_clusters", 3),
    ("detector_history_threshold", 1.5),
    ("detector_drift_allowance", 1.5),
    ("detector_drift_threshold", 4.),
    ("detector_drift_threshold", 8.),
    ("detector_reject_threshold", 5.),
    ("detector_reject_threshold", 7.),
    ("suspicion_penalty_factor", .25),
    ("suspicion_penalty_factor", .75),
    ("detector_weight_cap", 2.5),
    ("suspicion_remove_after", 1),
    ("suspicion_remove_after", 2),
    ("suspicion_recovery_factor", 1.1),
    ("detector_history_confirm", 3),
    ("detector_recovery_confirm", 3),
    ("detector_reference_budget", 2.5),
    ("detector_clip_factor", 3.),
)


def build_candidates() -> dict[str, list[dict]]:
    """Return independent, deterministic dictionaries of all 87 strategies."""
    candidates = {method: [] for method in METHODS}

    def add(method, parameters, origin):
        candidates[method].append({
            "candidate_id": f"{method}-v10-{len(candidates[method]) + 1:03d}",
            "variant": "original", "parameters": dict(parameters),
            "origin": origin,
        })

    for index, values in enumerate(MNIST_OURS_ROWS, start=1):
        add("sm9rrs", zip(OURS_POLICY_FIELDS, values), f"mnist_historical_{index:03d}")
    for warning, remove in product((1.25, 1.75, 2.5), (3, 5)):
        add("sm9rrs", {**CIFAR_OURS_ANCHOR,
                       "detector_distance_threshold": warning,
                       "suspicion_remove_after": remove}, "cifar_v3_historical")
    for name, value in OURS_BRIDGES:
        add("sm9rrs", {**CIFAR_OURS_ANCHOR, name: value}, f"cifar_anchor_one_factor:{name}")

    for history, epochs, rate in product((5, 7, 10, 20), (5, 20, 50), (.0005, .001)):
        # The complete policy explicitly prevents a change in projection,
        # top-k selection, or attacker-ratio knowledge as a side effect.
        add("vert", {"vert_history_window": history,
                     "vert_predict_epochs": epochs, "vert_predict_lr": rate,
                     "vert_projection_dim": 128, "vert_top_k": 0,
                     "vert_use_ratio_prior": False}, "historical_union_and_window_epoch_extension")

    for mpsa, sparsity, tda in product((.5, 1.), (.1, .3, .5), (.5, 1.)):
        add("alignins", {"alignins_sparsity": sparsity,
                         "alignins_tda_radius": tda,
                         "alignins_mpsa_radius": mpsa}, "mnist_historical_including_cifar_v3")
    for sparsity, (tda, mpsa) in product((.1, .3, .5), ((1.5, 1.), (1., 1.5))):
        add("alignins", {"alignins_sparsity": sparsity,
                         "alignins_tda_radius": tda,
                         "alignins_mpsa_radius": mpsa}, "one_radius_extension")
    for method in ("krum", "ding13", "fedavg"):
        add(method, {}, "fixed_original_method_no_exposed_defense_hyperparameters")
    return candidates


def run_budget(spec: dict) -> dict:
    counts = {method: len(items) for method, items in spec["candidates"].items()}
    per_candidate = len(spec["validation"]["seeds"]) * len(spec["validation"]["scenarios"])
    validation = sum(counts.values()) * per_candidate
    final = len(counts) * len(spec["final"]["seeds"]) * len(spec["final"]["scenarios"])
    return {"candidate_counts": counts, "total_candidates": sum(counts.values()),
            "validation_runs_per_candidate": per_candidate,
            "validation_runs": validation, "formal_runs_if_ours_qualifies": final,
            "maximum_total_runs": validation + final}


def build_spec(base_spec: dict | None = None) -> dict:
    """Build the new protocol from the tracked CIFAR shared configuration.

    This generator is a development utility. The independent v4 entry point
    consumes the checked-in JSON and freezes it in its experiment manifest.
    """
    if base_spec is None:
        base_spec = json.loads((REPO / "configs/cifar10_six_630_mean_v3.json").read_text())
    spec = deepcopy(base_spec)
    spec.update(
        schema_version=4,
        name="CIFAR-10 expanded validation with the MNIST per-scenario performance target",
        candidates=build_candidates(),
        output_dir="outputs/cifar10_six_mnist_gate_v4",
    )
    spec["validation"]["seeds"] = [1001, 1002, 1003]
    spec["final"]["seeds"] = [1101, 1102, 1103]
    spec["performance_target"] = {
        "accuracy_gap": .02, "asr_gap": .01, "max_asr": .05,
        "max_peak_asr": .20, "tail_rounds": 10,
    }
    spec.pop("mean_dual_gate", None)
    spec["promotion"] = {
        "required_healthy_methods": ["sm9rrs"],
        "performance_target_is_gate": True,
        "attack_effectiveness_is_gate": False,
        "missing_clean_reference": "report_unassessed_without_baseline_veto",
        "require_mean_dual_best": False,
    }
    spec["fallback_candidates"] = {
        method: items[0]["candidate_id"] for method, items in spec["candidates"].items()
        if method != "sm9rrs"
    }
    spec["search_design"] = {
        "version": "mnist_cifar_union_and_predeclared_bridges_v4",
        "objective_policy": "retain_cifar_v3_weights_0.25_0.50_0.20_0.05_without_relearning",
        "mnist_alignment_scope": "performance_target_thresholds_and_per_seed_per_scenario_windows; the common Score remains the CIFAR v3 formula",
        "historical_scores_reused": False,
        "official_test_used_for_selection": False,
        "shared_training_and_attack_protocol": "unchanged_from_cifar_v3",
        "ours": "12 complete MNIST policies + 6 complete CIFAR v3 policies + 24 single-field probes around the CIFAR anchor",
        "vert": "history (5,7,10,20) x predictor epochs (5,20,50) x predictor learning rate (0.0005,0.001); fixed projection 128, top_k 0 and no ratio prior",
        "alignins": "12 historical policies + 6 single-radius extensions; sparsity (0.1,0.3,0.5), original radii (0.5,1.0), one radius at a time extended to 1.5",
        "fixed_methods": ["krum", "ding13", "fedavg"],
        "fixed_method_reason": "The existing method-specific tuning interface has no hyperparameters; repeated identical candidates are not distinct choices.",
        "budget_limitation": "This unequal, predeclared search space is not a claim of equal tuning compute or an exhaustive optimum. Ours has more coupled controls; all counts are reported.",
        "no_performance_guarantee": "An expanded search cannot guarantee an Ours pass, formal generalization, or best performance among six methods.",
    }
    spec["run_budget"] = run_budget(spec)
    spec["protocol_note"] = (
        "New experiment identity: do not mix historical MNIST/CIFAR scores or formal results into this search. "
        "Evaluate 87 predeclared candidates on 3 validation seeds and 10 scenarios (2610 runs). "
        "Keep the CIFAR v3 common training, attack, split, and original six algorithms. "
        "Retain CIFAR v3 common Score weights (0.25,0.50,0.20,0.05), without relearning them. "
        "Require Ours health and the MNIST per-seed/per-scenario target: accuracy gap to selected VERT <=0.02, "
        "ASR gap <=0.01, attack/tail/final ASR <=0.05, and attack-window peak ASR <=0.20; tail has 10 rounds. "
        "Other methods select their highest-Score healthy candidate, or their highest raw-Score complete "
        "scorable candidate if none is healthy, retaining unqualified status and reasons. "
        "Only if no candidate is scorable use the predeclared first candidate. Baseline health alone cannot "
        "veto a qualified Ours; missing comparator evidence must remain visibly unassessed. "
        "The new validation seeds are 1001/1002/1003 and formal seeds are 1101/1102/1103. "
        "The same train split and official test set have prior experiment history; new seeds do not make "
        "the official test set unseen. If Ours qualifies, freeze the six choices and run 180 formal tasks. "
        "Neither validation targets nor a broader search guarantee formal performance or an Ours victory."
    )
    return spec


if __name__ == "__main__":
    # Explicit invocation regenerates only this new protocol, never an old run.
    DEFAULT_CONFIG.write_text(json.dumps(build_spec(), ensure_ascii=False, indent=2) + "\n")
    print(DEFAULT_CONFIG)
