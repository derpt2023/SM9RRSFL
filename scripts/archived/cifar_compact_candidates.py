"""A validation-informed subset of the frozen 87-candidate CIFAR v4 space.

Historical validation observations inform the development shortlist only.
They never replace a new candidate's 30 validation runs or qualify Ours.
This generator needs tracked source/config files, not local outputs/ data.
The training runner consumes the resulting JSON, preserving its existing
source identity and the full v4 configuration for separate experiments.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import cifar_expanded_candidates as expanded


REPO = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO / "configs/cifar10_six_mnist_gate_compact_v4.json"

# Order also defines the explicit first-candidate fallback for each baseline
# when no complete scorable candidate exists. These are complete v4 policies,
# never combinations of individual historical winners' fields.
SHORTLIST = {
    "sm9rrs": (
        (15, "CIFAR v3 highest healthy Score; reference anchor"),
        (13, "CIFAR v3 highest raw Score and lowest mean ASR; historically unhealthy, must requalify"),
        (14, "CIFAR v3 second raw Score; paired revocation threshold 5 versus 3; historically unhealthy"),
        (2, "MNIST healthy balanced q=2, warning=3 complete policy"),
        (7, "MNIST healthy q=1, one-cluster complete policy"),
        (12, "MNIST highest validation Score complete policy, q=3 and warning=3.5"),
        (23, "Unmeasured probe: change only anchor subspace rank from 2 to 3"),
        (29, "Unmeasured probe: change only drift threshold from 6 to 4"),
        (34, "Unmeasured probe: change only reliability penalty factor from .5 to .75"),
        (41, "Unmeasured probe: change only reference budget from 3.5 to 2.5"),
    ),
    "vert": (
        (14, "CIFAR v3 highest healthy Score: history=10, epochs=5, lr=.001"),
        (16, "CIFAR v3 highest raw Score: history=10, epochs=20, lr=.001; historically unhealthy"),
        (15, "Unmeasured lower-lr counterpart of the preceding candidate"),
        (3, "MNIST highest validation Score: history=5, epochs=20, lr=.0005"),
    ),
    "alignins": (
        (8, "CIFAR v3 highest raw Score; historically unhealthy"),
        (12, "CIFAR v3 second raw Score, higher sparsity and fewer nonfinite updates"),
        (9, "CIFAR v3 29 of 30 healthy runs; retain a near-healthy middle-sparsity policy"),
        (1, "MNIST validation observations rank first under the current CIFAR Score weights"),
    ),
    "krum": ((1, "Fixed original method; no exposed method-specific search parameter"),),
    "ding13": ((1, "Fixed original method; no exposed method-specific search parameter"),),
    "fedavg": ((1, "Fixed original method; no exposed method-specific search parameter"),),
}


def build_spec() -> dict:
    full = expanded.build_spec()
    spec = deepcopy(full)
    for method, choices in SHORTLIST.items():
        available = {c["candidate_id"]: c for c in full["candidates"][method]}
        spec["candidates"][method] = []
        for index, reason in choices:
            candidate = deepcopy(available[f"{method}-v10-{index:03d}"])
            candidate["shortlist_reason"] = reason
            spec["candidates"][method].append(candidate)
    spec.update(
        name="CIFAR-10 compact validation with unchanged MNIST per-scenario targets",
        output_dir="outputs/cifar10_six_mnist_gate_compact_v4",
    )
    spec["fallback_candidates"] = {
        m: cs[0]["candidate_id"] for m, cs in spec["candidates"].items() if m != "sm9rrs"
    }
    spec["run_budget"] = expanded.run_budget(spec)
    spec["search_design"] = {
        **full["search_design"],
        "version": "compact_validation_informed_subset_v4",
        "parent_configuration": "configs/cifar10_six_mnist_gate_v4.json",
        "historical_validation_used_for_search_design": True,
        "historical_scores_reused_for_new_qualification": False,
        "validation_evidence": [
            "CIFAR v3: outputs/cifar10_six_630_near_vert_v3/validation_summary.json",
            "MNIST: outputs/mnist_v7_target_fair_tuning/tuning_trials.csv",
        ],
        "evidence_availability": "Provenance only; historical outputs are not required to generate or run this configuration.",
        "ours": "3 CIFAR historical policies + 3 MNIST complete policies + 4 unmeasured single-field probes",
        "vert": "CIFAR healthy-best and raw-best, a lower-lr probe, and MNIST validation-best; omit epochs=50 and history=20",
        "alignins": "CIFAR raw-best and runner-up, a 29/30-healthy policy, and MNIST best under CIFAR weights; omit radius=1.5 probes",
        "pruning_limit": "A priority shortlist, not proof that removed candidates are inferior on new seeds or that retained candidates satisfy the targets.",
        "rounds_saved_vs_full_validation": 198000,
        "validation_run_reduction_fraction": 1 - 630 / 2610,
        "runtime_caveat": "Training-round reduction is not a wall-time guarantee; methods, policy choices and machine contention have different costs.",
    }
    spec["protocol_note"] = (
        "Compact, separately frozen subset of the full v4 search: 21 candidates x 3 validation seeds "
        "x 10 scenarios x 100 rounds = 630 validation runs / 63000 training rounds. "
        "Ours 10, VERT 4, AlignIns 4, and Krum/TAD/FedAvg one each. "
        "Historical validation only informs this development shortlist; no historical score, "
        "formal outcome or shortened run substitutes for fresh validation qualification. "
        "Keep v4 shared training/attack, health and MNIST targets, Score, seeds and scenario matrices. "
        "Ours must qualify before all six frozen selections enter 180 formal runs. "
        "Baseline fallback is per method: healthy highest Score, otherwise complete highest raw Score, "
        "otherwise the explicitly declared first candidate; retain failed/unassessed labels. "
        "Ours failure prevents formal execution even when baseline fallback selections were calculated. "
        "Keep full v4 available separately; never combine their output directories. "
        "No claim of equal per-method search budget, global optimum, guaranteed pass or formal victory."
    )
    return spec


if __name__ == "__main__":
    DEFAULT_CONFIG.write_text(json.dumps(build_spec(), ensure_ascii=False, indent=2) + "\n")
    print(DEFAULT_CONFIG)
