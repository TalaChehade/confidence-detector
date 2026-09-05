from pathlib import Path
import os
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inker.config import load_config, resolve_project_path, detector_layers
from inker.model import load_model_and_tokenizer


DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def get_config(config_path=None):
    return load_config(
        config_path or DEFAULT_CONFIG
    )


def get_hf_token():
    # Optional. If you already used huggingface_hub.login(), passing None is
    # fine because Transformers can use cached credentials.
    return os.getenv("HF_TOKEN")


def dtype_from_name(name):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if name not in mapping:
        raise ValueError(
            f"Unsupported compute_dtype: {name}"
        )

    return mapping[name]


def load_configured_model(config):
    m = config["model"]

    return load_model_and_tokenizer(
        model_name=m["name"],
        hf_token=get_hf_token(),
        load_in_4bit=m["load_in_4bit"],
        compute_dtype=dtype_from_name(
            m["compute_dtype"]
        ),
        use_double_quant=m[
            "use_double_quant"
        ],
        padding_side=m["padding_side"],
        pad_token_id=m["pad_token_id"],
    )
