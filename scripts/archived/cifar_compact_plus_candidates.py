"""Add four Ours policies to compact v4 without changing its selection rules.

All original candidates remain unchanged. Historical validation only informs
the shortlist; each added policy must pass fresh full validation. The runner
reads the generated JSON, not this generator or any local experiment outputs.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import cifar_compact_candidates as compact
import cifar_expanded_candidates as expanded


REPO = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO / "configs/cifar10_six_mnist_gate_compact_plus_v4.json"
ADDITIONAL_OURS = (
    (10, "MNIST validation third-ranked complete healthy policy; complement the retained top policy, without assuming CIFAR transfer"),
    (11, "MNIST validation second-ranked complete healthy policy; together with 010 and retained 012 covers the historical top three"),
    (16, "CIFAR v3 004: change only anchor revocation threshold 3 to 5; better observed raw Score/ASR but one nonfinite update, so must requalify"),
    (39, "Unmeasured probe: change only trusted-history confirmation 2 to 3; delay admission without claiming it prevents persistent attacks"),
)


def build_spec() -> dict:
    parent = compact.build_spec()
    spec = deepcopy(parent)
    full = {c["candidate_id"]: c for c in expanded.build_candidates()["sm9rrs"]}
    for index, reason in ADDITIONAL_OURS:
        candidate = deepcopy(full[f"sm9rrs-v10-{index:03d}"])
        candidate["shortlist_reason"] = reason
        spec["candidates"]["sm9rrs"].append(candidate)
    spec.update(
        name="CIFAR-10 compact plus four Ours candidates with unchanged v4 rules",
        output_dir="outputs/cifar10_six_mnist_gate_compact_plus_v4",
    )
    spec["run_budget"] = expanded.run_budget(spec)
    full_budget = expanded.run_budget(expanded.build_spec())
    validation_runs = spec["run_budget"]["validation_runs"]
    extra_runs = validation_runs - parent["run_budget"]["validation_runs"]
    spec["search_design"].update({
        "version": "compact_plus_four_ours_v4",
        "parent_configuration": "configs/cifar10_six_mnist_gate_compact_v4.json",
        "ours": "All 10 compact Ours policies plus 2 complete MNIST top-three policies, delayed revocation, and delayed trusted-history admission; 14 total",
        "additional_ours_candidates": [f"sm9rrs-v10-{i:03d}" for i, _ in ADDITIONAL_OURS],
        "all_parent_candidates_preserved": True,
        "baseline_candidates_unchanged": True,
        "selection_and_promotion_rules_unchanged": True,
        "extra_validation_runs_vs_compact": extra_runs,
        "extra_training_rounds_vs_compact": extra_runs * spec["shared_parameters"]["rounds"],
        "rounds_saved_vs_full_validation": (full_budget["validation_runs"] - validation_runs) * spec["shared_parameters"]["rounds"],
        "validation_run_reduction_fraction": 1 - validation_runs / full_budget["validation_runs"],
        "performance_limit": "More eligible choices may improve the selected validation result; no guarantee of passing, global optimality or formal superiority. Keep the existing mean-dual preference and Score order.",
        "runtime_caveat": "The extra 120 validation runs are all Ours; their cost can exceed the 19.05% task-count increase. Historical concurrent runtimes are not a wall-time guarantee.",
    })
    spec["protocol_note"] = (
        "Preserve the compact v4 rules and all 21 original candidates; append four Ours policies "
        "010, 011, 016 and 039 from the existing full v4 pool. "
        "Ours 14, VERT 4, AlignIns 4, and Krum/TAD/FedAvg one each: "
        "25 candidates x 3 validation seeds x 10 scenarios = 750 validation runs / 75000 training rounds. "
        "Qualified Ours permits the unchanged 180 formal runs, totaling at most 930 runs / 93000 training rounds. "
        "Keep all shared training/attack settings, health and MNIST targets, common Score, seeds, "
        "per-method baseline fallback and mean-dual selection preference unchanged. "
        "Historical validation informs candidate design only; every added candidate must newly qualify. "
        "The delayed-revocation candidate was historically unhealthy; do not erase its failure or transfer its score. "
        "Formal results never select parameters. A larger candidate set does not guarantee better test results. "
        "Use this separate output identity; retain compact 630-run and full 2610-run studies unchanged."
    )
    return spec


if __name__ == "__main__":
    DEFAULT_CONFIG.write_text(json.dumps(build_spec(), ensure_ascii=False, indent=2) + "\n")
    print(DEFAULT_CONFIG)
