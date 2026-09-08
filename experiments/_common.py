"""
Shared utilities for INKER experiment scripts.

This module centralizes setup logic that would otherwise be repeated across:

    - train_detector.py
    - evaluate_detector.py
    - run_confidence_only.py
    - run_test_suite.py
    - run_inker_trigger.py
    - and future experiment scripts.

Its responsibilities are intentionally limited to:

    1. locating the repository root,
    2. making ``src/`` importable when scripts are run directly,
    3. loading the YAML configuration,
    4. reading an optional Hugging Face access token,
    5. translating YAML dtype names into ``torch.dtype`` objects,
    6. loading the configured model and tokenizer.

It does NOT:
    - build datasets,
    - train confidence directions,
    - score examples,
    - calculate query complexity,
    - or perform retrieval triggering.

Keeping this shared setup in one place ensures that all experiments use the
same model, tokenizer, quantization, padding, and numerical settings.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch


# ==========================================================================
# Repository paths
# ==========================================================================

# experiments/common.py
#        │
#        └── parent       -> experiments/
#             └── parent -> repository root
#
# Therefore:
#
#     parents[1]
#
# is the repository root.

REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_DIR = (
    REPO_ROOT
    / "src"
)

DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "default.yaml"
)


# ==========================================================================
# Make src/ importable when experiment files are executed directly.
# ==========================================================================

src_path = str(
    SRC_DIR
)

if src_path not in sys.path:
    sys.path.insert(
        0,
        src_path,
    )


# Imports from the project package must come after src/ is added to sys.path.
from inker.config import (  # noqa: E402
    detector_layers,
    load_config,
    resolve_project_path,
)

from inker.model import (  # noqa: E402
    load_model_and_tokenizer,
)


# ==========================================================================
# Configuration
# ==========================================================================

def get_config(
    config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Load an experiment configuration.

    Parameters
    ----------
    config_path:
        Optional path to a YAML configuration file.

        If omitted, the repository default is used:

            configs/default.yaml

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """

    path = (
        Path(config_path)
        if config_path is not None
        else DEFAULT_CONFIG
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {path}"
        )

    return load_config(
        path
    )


# ==========================================================================
# Hugging Face authentication
# ==========================================================================

def get_hf_token() -> Optional[str]:
    """
    Return the Hugging Face access token from the environment.

    The function reads:

        HF_TOKEN

    and returns None if it is not defined.

    Passing None is normally fine when:

        - the model is public, or
        - ``huggingface_hub.login()`` was already used, or
        - credentials are cached locally.

    Important
    ---------
    Hugging Face tokens should never be hard-coded in repository files.
    """

    token = os.getenv(
        "HF_TOKEN"
    )

    if token is not None:
        token = token.strip()

    return (
        token
        if token
        else None
    )


# ==========================================================================
# Dtype conversion
# ==========================================================================

DTYPE_MAPPING = {
    "float16":
        torch.float16,

    "fp16":
        torch.float16,

    "bfloat16":
        torch.bfloat16,

    "bf16":
        torch.bfloat16,

    "float32":
        torch.float32,

    "fp32":
        torch.float32,
}


def dtype_from_name(
    name: Optional[str],
) -> Optional[torch.dtype]:
    """
    Convert a configuration dtype name into ``torch.dtype``.

    Supported values include:

        float16
        fp16
        bfloat16
        bf16
        float32
        fp32

    ``None`` is also accepted and returned unchanged. This allows the model
    loader to choose a hardware-appropriate dtype automatically.

    Examples
    --------
    ``"float16"`` -> ``torch.float16``

    ``"bf16"`` -> ``torch.bfloat16``

    ``None`` -> ``None``
    """

    if name is None:
        return None

    if not isinstance(
        name,
        str,
    ):
        raise TypeError(
            "compute_dtype must be a string or None."
        )

    normalized = (
        name
        .strip()
        .lower()
    )

    if normalized not in DTYPE_MAPPING:
        raise ValueError(
            f"Unsupported compute_dtype: {name!r}. "
            f"Supported values are: "
            f"{sorted(DTYPE_MAPPING.keys())}"
        )

    return DTYPE_MAPPING[
        normalized
    ]


# ==========================================================================
# Model configuration
# ==========================================================================

def load_configured_model(
    config: Dict[str, Any],
):
    """
    Load the tokenizer and language model described by the configuration.

    Expected YAML section
    ---------------------
    A configuration typically contains:

        model:
          name: mistralai/Mistral-7B-Instruct-v0.1
          load_in_4bit: true
          compute_dtype: float16
          use_double_quant: true
          quant_type: nf4
          padding_side: left
          pad_token_id: 0
          device_map: auto

    The same model configuration should be used during:

        detector training
        detector evaluation
        confidence-only testing
        live INKER triggering

    because the learned representation directions are tied to the model's
    hidden-state space.

    Returns
    -------
    tokenizer, model
        Configured tokenizer and causal language model.
    """

    if "model" not in config:
        raise KeyError(
            "Configuration does not contain a 'model' section."
        )

    model_config = config[
        "model"
    ]

    if not isinstance(
        model_config,
        dict,
    ):
        raise TypeError(
            "config['model'] must be a dictionary."
        )

    required_keys = {
        "name",
        "load_in_4bit",
        "padding_side",
    }

    missing = (
        required_keys
        - set(model_config.keys())
    )

    if missing:
        raise KeyError(
            "Model configuration is missing required keys: "
            f"{sorted(missing)}"
        )

    return load_model_and_tokenizer(
        model_name=model_config[
            "name"
        ],

        hf_token=get_hf_token(),

        load_in_4bit=bool(
            model_config[
                "load_in_4bit"
            ]
        ),

        compute_dtype=dtype_from_name(
            model_config.get(
                "compute_dtype"
            )
        ),

        use_double_quant=bool(
            model_config.get(
                "use_double_quant",
                True,
            )
        ),

        quant_type=model_config.get(
            "quant_type",
            "nf4",
        ),

        padding_side=model_config.get(
            "padding_side",
            "left",
        ),

        pad_token_id=model_config.get(
            "pad_token_id",
            0,
        ),

        device_map=model_config.get(
            "device_map",
            "auto",
        ),
    )


# ==========================================================================
# Convenience helpers
# ==========================================================================

def get_detector_layers(
    config: Dict[str, Any],
) -> list[int]:
    """
    Return the detector layers defined by the configuration.

    This is a small convenience wrapper around:

        inker.config.detector_layers(...)
    """

    layers = detector_layers(
        config
    )

    if not layers:
        raise ValueError(
            "Detector layer configuration produced an empty layer list."
        )

    return [
        int(layer)
        for layer in layers
    ]


def get_project_path(
    config: Dict[str, Any],
    path_key: str,
) -> Path:
    """
    Resolve a configured project-relative path.

    This wrapper keeps experiment scripts concise and ensures paths are
    consistently returned as ``Path`` objects.

    Example
    -------
    If the configuration contains:

        paths:
          results: results

    then:

        get_project_path(config, "results")

    returns the resolved repository path for that entry.
    """

    path = resolve_project_path(
        config,
        path_key,
    )

    return Path(
        path
    )
