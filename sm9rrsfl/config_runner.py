"""Load experiment CLI parameters from a versioned JSON configuration file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any

from .experiments import main as run_experiments
from .experiments import parse_args


CONFIG_SCHEMA_VERSION = 1
LIST_PARAMETERS = {"methods", "ratios", "client_counts", "partitions"}
ROOT_KEYS = {"schema_version", "name", "description", "parameters"}
BOOLEAN_ALIASES = {
    "early_stop": ("no_early_stop", True),
    "visualizations": ("no_visualizations", True),
    "progress": ("no_progress", True),
    "no_resume": ("resume", True),
}


class ConfigError(ValueError):
    """The JSON file cannot be represented by the experiment CLI."""


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Read and validate the versioned JSON wrapper."""

    config_path = Path(path).expanduser()
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read configuration file {config_path}: {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"invalid JSON in {config_path} at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConfigError("configuration root must be a JSON object")

    unknown_root = sorted(set(payload) - ROOT_KEYS)
    if unknown_root:
        raise ConfigError(f"unknown configuration root key: {unknown_root[0]}")
    schema_version = payload.get("schema_version")
    if schema_version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            "configuration schema_version must be "
            f"{CONFIG_SCHEMA_VERSION}, got {schema_version!r}"
        )
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ConfigError("configuration 'parameters' must be a JSON object")
    return parameters


def parameters_to_argv(parameters: dict[str, Any]) -> list[str]:
    """Convert JSON parameter names to the existing experiment CLI syntax."""

    defaults = vars(parse_args([]))
    known_parameters = set(defaults)
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in parameters.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ConfigError("every parameter name must be a non-empty string")
        key = raw_key.replace("-", "_")
        if key in BOOLEAN_ALIASES:
            canonical, invert = BOOLEAN_ALIASES[key]
            if not isinstance(raw_value, bool):
                raise ConfigError(f"parameter '{raw_key}' must be boolean")
            key = canonical
            value = not raw_value if invert else raw_value
        else:
            value = raw_value
        if key in normalized:
            raise ConfigError(f"parameter '{key}' is configured more than once")
        normalized[key] = value

    unknown = sorted(set(normalized) - known_parameters)
    if unknown:
        spelling = unknown[0].replace("_", "-")
        raise ConfigError(f"unknown experiment parameter: {spelling}")

    argv: list[str] = []
    for key, value in normalized.items():
        if value is None:
            continue
        option = f"--{key.replace('_', '-')}"
        default = defaults[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"parameter '{key}' must be boolean")
            if key == "resume":
                if not value:
                    argv.append("--no-resume")
            elif value:
                argv.append(option)
            continue

        if key in LIST_PARAMETERS:
            if not isinstance(value, list) or not value:
                raise ConfigError(f"parameter '{key}' must be a non-empty JSON array")
            if any(
                item is None or isinstance(item, (bool, list, dict))
                for item in value
            ):
                raise ConfigError(f"parameter '{key}' contains an invalid array value")
            argv.append(option)
            argv.extend(str(item) for item in value)
            continue

        if isinstance(value, (bool, list, dict)):
            raise ConfigError(f"parameter '{key}' must be a scalar value")
        if isinstance(value, str) and not value:
            raise ConfigError(f"parameter '{key}' must not be an empty string")
        argv.extend((option, str(value)))
    return argv


def config_file_to_argv(path: str | Path) -> list[str]:
    """Load one JSON configuration and return validated experiment arguments."""

    parameters = load_experiment_config(path)
    argv = parameters_to_argv(parameters)
    # Reuse the authoritative parser for choices, ranges, aliases, and presets.
    parse_args(argv)
    return argv


def main(
    argv: list[str] | None = None,
    *,
    default_config: str | Path | None = None,
) -> None:
    """Run the configured experiment or print its resolved arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(default_config) if default_config is not None else None,
        required=default_config is None,
        help="Path to a schema_version=1 JSON experiment configuration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the configuration and print resolved parameters without training.",
    )
    runner_args = parser.parse_args(argv)
    config_path = Path(runner_args.config).expanduser().resolve()
    try:
        experiment_argv = config_file_to_argv(config_path)
    except ConfigError as exc:
        parser.error(str(exc))

    command = [
        sys.executable,
        "-m",
        "sm9rrsfl.experiments",
        *experiment_argv,
    ]
    print(f"config_file={config_path}", flush=True)
    print(f"experiment_command={shlex.join(command)}", flush=True)
    if runner_args.dry_run:
        resolved = vars(parse_args(experiment_argv))
        print(
            "resolved_parameters="
            + json.dumps(resolved, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        return
    run_experiments(experiment_argv)


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "ConfigError",
    "config_file_to_argv",
    "load_experiment_config",
    "main",
    "parameters_to_argv",
]
