"""Locate legacy tuning caches without treating CUDA placement as new data."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from .experiments import load_completed_results_snapshot, read_run_manifest


def runtime_config_key(config):
    payload = asdict(config) if not isinstance(config, dict) else dict(config)
    # CUDA-to-CPU/MPS changes are deliberately NOT normalized. All training,
    # evaluation, crypto mode and detector settings remain part of identity.
    device = payload.get("device")
    if isinstance(device, str) and re.fullmatch(r"cuda(?::\d+)?", device):
        payload["device"] = "cuda"
    payload.pop("sm9_workers", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def phase_identity(manifest, state_root):
    payload = json.loads(json.dumps(manifest))
    payload.pop("fingerprint", None)
    payload["configs"] = [json.loads(runtime_config_key(c)) for c in payload["configs"]]
    for candidate in payload.get("candidates", []):
        candidate["config"] = json.loads(runtime_config_key(candidate["config"]))
    context = payload.get("tuning_context", {})
    reference = context.get("selected_validation_fingerprint")
    if payload.get("tuning_phase") == "final" and isinstance(reference, str):
        if re.fullmatch(r"[a-f0-9]{64}", reference):
            parent = read_run_manifest(Path(state_root) / "validation" / reference)
            if (parent and parent.get("fingerprint") == reference
                    and parent.get("tuning_phase") == "validation"):
                context["selected_validation_fingerprint"] = phase_identity(parent, state_root)
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def locate_tuning_state(output_dir, manifest):
    """Keep the richest compatible legacy cache and its original fingerprint.

    Keeping its manifest and checkpoint filenames avoids copying multi-GiB
    round states or rewriting historical device/runtime provenance. The final
    phase also verifies the training-holdout identity of its validation parent.
    """
    root = Path(output_dir) / ".tuning_state"
    phase_dir = root / manifest["tuning_phase"]
    requested = phase_dir / manifest["fingerprint"]
    identity = phase_identity(manifest, root)
    best = None
    best_rank = None
    for directory in sorted(phase_dir.iterdir()) if phase_dir.exists() else ():
        if not directory.is_dir():
            continue
        previous = read_run_manifest(directory)
        if not previous or previous.get("fingerprint") != directory.name:
            continue
        try:
            compatible = phase_identity(previous, root) == identity
        except (KeyError, TypeError, ValueError):
            compatible = False
        if not compatible:
            continue
        allowed = {runtime_config_key(c) for c in previous["configs"]}
        results = load_completed_results_snapshot(directory) or []
        results = [r for r in results if runtime_config_key(r.config) in allowed]
        count = len({runtime_config_key(r.config) for r in results})
        checkpoint_count = sum(1 for _ in (directory / ".checkpoints").glob("*.pickle"))
        rank = (count, checkpoint_count, directory == requested)
        if best_rank is None or rank > best_rank:
            best = directory, previous, results
            best_rank = rank
    if best is not None:
        return best
    if requested.exists() and read_run_manifest(requested) is not None:
        raise ValueError(f"incompatible tuning manifest at {requested}; preserve it for inspection")
    return requested, manifest, []
