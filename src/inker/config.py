"""
Configuration utilities for the INKER replication repository.

This module provides helper functions for loading the repository's YAML
configuration file and resolving project paths defined inside that
configuration.

The configuration file is expected to contain sections such as:

    paths:
        project_dir: ...
        results_dir: ...
        detector_dir: ...

    detector:
        hidden_layers: ...

Main responsibilities
---------------------
1. Load a YAML configuration file.
2. Resolve relative paths with respect to the configured project directory.
3. Optionally create output directories when they do not already exist.
4. Return the hidden layers used by the confidence representation detector.

This file does NOT contain model-loading, training, generation, confidence
scoring, or retrieval-trigger logic. It only handles configuration values
shared by the rest of the repository.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


ConfigDict = Dict[str, Any]
PathLike = Union[str, Path]


def load_config(config_path: PathLike) -> ConfigDict:
    """
    Load a YAML experiment configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.

    ValueError
        If the YAML file is empty or does not contain a dictionary.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(
            f"Configuration file is empty: {config_path}"
        )

    if not isinstance(config, dict):
        raise ValueError(
            "The YAML configuration must contain a dictionary "
            f"at the top level: {config_path}"
        )

    return config


def resolve_project_path(
    config: ConfigDict,
    path_key: str,
    create_if_missing: bool = False,
) -> Path:
    """
    Resolve a configured project path.

    Paths stored under ``config["paths"]`` may be either:

    - absolute paths, or
    - paths relative to ``config["paths"]["project_dir"]``.

    Example
    -------
    If the configuration contains:

        paths:
            project_dir: /content/drive/MyDrive/inker-confidence-detector
            results_dir: results

    then:

        resolve_project_path(config, "results_dir")

    returns:

        /content/drive/MyDrive/inker-confidence-detector/results

    Parameters
    ----------
    config:
        Parsed configuration dictionary.

    path_key:
        Name of the path inside ``config["paths"]``.

    create_if_missing:
        If True, create the resolved directory and any missing parent
        directories.

    Returns
    -------
    pathlib.Path
        Fully resolved path.

    Raises
    ------
    KeyError
        If ``paths``, ``project_dir``, or the requested ``path_key``
        is missing from the configuration.
    """

    if "paths" not in config:
        raise KeyError(
            "Missing required 'paths' section in configuration."
        )

    paths_config = config["paths"]

    if "project_dir" not in paths_config:
        raise KeyError(
            "Missing required 'paths.project_dir' configuration value."
        )

    if path_key not in paths_config:
        raise KeyError(
            f"Missing required path configuration: 'paths.{path_key}'"
        )

    project_dir = Path(paths_config["project_dir"]).expanduser()
    configured_path = Path(paths_config[path_key]).expanduser()

    if configured_path.is_absolute():
        resolved_path = configured_path
    else:
        resolved_path = project_dir / configured_path

    if create_if_missing:
        resolved_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    return resolved_path


def detector_layers(config: ConfigDict) -> List[int]:
    """
    Return the hidden layers used by the confidence detector.

    The layers are read from:

        config["detector"]["hidden_layers"]

    Returns
    -------
    list[int]
        Hidden-layer indices used during representation extraction.

    Raises
    ------
    KeyError
        If the detector configuration or hidden-layer list is missing.
    """

    if "detector" not in config:
        raise KeyError(
            "Missing required 'detector' section in configuration."
        )

    if "hidden_layers" not in config["detector"]:
        raise KeyError(
            "Missing required 'detector.hidden_layers' configuration value."
        )

    return list(config["detector"]["hidden_layers"])
