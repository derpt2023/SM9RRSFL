"""Explicit, process-local history ablation for the independent CNN panel.

The original detector class and its pickle identity stay unchanged. A frozen
run uses the original evaluate/weighting/aggregation/revocation paths; only
post-decision history admission is suppressed from a predeclared public round.
The same round applies to clean tasks. No client truth labels are inspected.
"""
from contextlib import contextmanager

from sm9rrsfl.svd_detector import LongitudinalSVDDetector


ORIGINAL_VARIANT = "original"
FROZEN_VARIANT = "Ours-FrozenHistory-v1"
_active = False


@contextmanager
def history_runtime(variant, *, freeze_start_round):
    """Apply one variant for one worker, including checkpoint deserialization.

Calling the original commit with admit_history=False would also erase a clean
streak. Instead this ablation preserves evaluate's streak/recovery/drift state
and performs the original commit's finalization without admitting/refitting.
An original aggregation guard can still reset the clean streak as before.

Existing ClientDiagnosticRecord.history_frozen describes the weight manager's
own guard; it is NOT repurposed. The new task identity records this intervention
and commit's actual False return records history_admitted=False.
"""
    global _active
    if variant == ORIGINAL_VARIANT:
        if freeze_start_round is not None:
            raise ValueError("original variant must not declare a history freeze")
    elif variant == FROZEN_VARIANT:
        if (isinstance(freeze_start_round, bool)
                or not isinstance(freeze_start_round, int) or freeze_start_round < 1):
            raise ValueError("frozen history requires a positive public round")
    else:
        raise ValueError("unknown history ablation variant")
    if _active:
        raise RuntimeError("history runtime contexts cannot be nested")

    original_commit = LongitudinalSVDDetector.commit

    def frozen_commit(self, tag, *, admit_history):
        if freeze_start_round <= self.window_size:
            raise ValueError("history freezing must start after trusted warmup")
        state = self._states[tag]
        if state.pending is None:
            raise RuntimeError("no pending observation")
        current, _, _, _ = state.pending
        if current < freeze_start_round:
            return original_commit(self, tag, admit_history=admit_history)
        # Do not change history, live normal, immutable anchor or norm limit.
        # evaluate has already updated drift, clean_streak and recovery_streak.
        if not admit_history:
            state.clean_streak = 0
        state.last_round = current
        state.pending = None
        return False

    _active = True
    try:
        if variant == FROZEN_VARIANT:
            LongitudinalSVDDetector.commit = frozen_commit
        yield
    finally:
        LongitudinalSVDDetector.commit = original_commit
        _active = False
