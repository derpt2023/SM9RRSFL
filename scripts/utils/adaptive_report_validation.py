"""Report-only compatibility for aggregation-weight roundoff in schema 8.

The frozen workers and health scorer already accept aggregation diagnostics in
``[0, 1 + 1e-9]``. Their frozen reporters instead require ``[0, 1]``. This adapter
reconciles only that discrepancy without changing a scientific source, result,
checkpoint, selection, or performance gate. Accuracy, ASR and confidence keep
the reporters' original strict bounds.

Only the inspector receives temporary dataclass copies with tolerated weight
values set to one. The report still receives the original runs, and returned
health is recomputed from those originals. Consequently exported CSV values
and result objects retain their exact observed values. Successful inspections
append their tolerated observations to the yielded audit list.

Use in a single-threaded report-recovery process. Installation is temporary and
process-local; nested contexts restore the preceding inspector in LIFO order.
Importing this module does not install an adapter or load an experiment.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from functools import wraps
import math

import run_cifar_six_from_scratch as base


AGGREGATION_WEIGHT_TOLERANCE = 1e-9
AGGREGATION_WEIGHT_FIELDS = ("honest_weight_loss", "malicious_weight_mass")


@contextmanager
def aggregation_weight_tolerance(reporting_module):
    """Temporarily adapt a CIFAR/Fashion reporter; yield raw-value audit rows.

    Callers must still verify manifest, task and snapshot identities before
    reporting. This adapter is not an identity verifier and never makes an
    incomplete or unhealthy run healthy. Invalid values remain fatal, including
    negative weights, nonfinite values and weights above ``1 + 1e-9``.
    """
    original = reporting_module._inspect_result
    audit = []

    @wraps(original)
    def inspect(run, rounds):
        # Preserve the original incomplete-run policy, including its lack of
        # final-round eligibility. Do not audit unvalidated partial records.
        if run.stopped_round != rounds:
            return original(run, rounds)

        records, tolerated = [], []
        for row in run.records:
            replacements = {}
            for field in AGGREGATION_WEIGHT_FIELDS:
                value = getattr(row, field)
                if (isinstance(value, (int, float)) and math.isfinite(value)
                        and 1 < value <= 1 + AGGREGATION_WEIGHT_TOLERANCE):
                    replacements[field] = 1.0
                    tolerated.append((row.round, field, value))
            records.append(replace(row, **replacements) if replacements else row)

        validation_run = replace(run, records=records) if tolerated else run
        complete, reason, health = original(validation_run, rounds)
        if not complete:
            return complete, reason, health

        # Never use the copied diagnostics to classify training health. In
        # particular, retain the original health failures and invalid_* guard.
        health = base.metrics(run)
        if any(reason.startswith("invalid_") for reason in health["reasons"]):
            raise reporting_module.ReportIntegrityError(
                "finished result has invalid scientific metrics")
        if tolerated:
            config = asdict(run.config)
            config_sha256 = base.digest(base.semantic_config(run.config))
            for round_index, field, value in tolerated:
                audit.append({
                    "reporting_module": reporting_module.__name__,
                    "semantic_config_sha256": config_sha256,
                    "config": dict(config),
                    "round": round_index,
                    "field": field,
                    "raw_value": value,
                    "excess_above_one": value - 1,
                    "tolerance": AGGREGATION_WEIGHT_TOLERANCE,
                    "validation_copy_value": 1.0,
                    "raw_value_preserved": True,
                    "health_from_original_result": True,
                })
        return complete, reason, health

    reporting_module._inspect_result = inspect
    try:
        yield audit
    finally:
        reporting_module._inspect_result = original
