"""
Model and tokenizer loading utilities for the INKER replication.

This module is responsible for loading the base causal language model used by:

    - hidden-state extraction,
    - confidence-direction training,
    - confidence scoring,
    - and live token-by-token generation.

The default model is:

    mistralai/Mistral-7B-Instruct-v0.1

which is the model used by the current INKER confidence-detector replication.

The loader supports:

    1. 4-bit bitsandbytes quantization for memory-efficient Colab inference.
    2. Full / half precision loading when quantization is disabled.
    3. Left-padding configuration required by the hidden-state extractor.
    4. Safe pad-token configuration.
    5. Optional Hugging Face authentication.

Important
---------
The confidence detector is sensitive to the model from which its hidden
representations were learned.

A representation reader trained using one model should NOT be used with a
different model architecture or checkpoint unless that change is intentional
and the detector is retrained.

For example:

    detector trained on:
        mistralai/Mistral-7B-Instruct-v0.1

should normally be evaluated using:

        mistralai/Mistral-7B-Instruct-v0.1

as well.

Similarly, tokenizer configuration must remain consistent between:

    dataset construction
    hidden-state extraction
    detector training
    evaluation
    live generation

because tokenization changes the hidden representations being measured.
"""

from __future__ import annotations

from typing import Optional

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


DEFAULT_MODEL_NAME = (
    "mistralai/Mistral-7B-Instruct-v0.1"
)


def _select_compute_dtype(
    requested_dtype: Optional[torch.dtype] = None,
) -> torch.dtype:
    """
    Select the compute dtype used by the quantized model.

    If the caller explicitly supplies a dtype, that value is returned.

    Otherwise:

        - bfloat16 is preferred on CUDA hardware that supports it;
        - float16 is used on other CUDA devices;
        - float32 is used on CPU.

    Returns
    -------
    torch.dtype
        Compute dtype suitable for the current hardware.
    """

    if requested_dtype is not None:
        return requested_dtype

    if torch.cuda.is_available():

        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    return torch.float32


def _configure_padding(
    tokenizer,
    padding_side: str,
    pad_token_id: Optional[int],
):
    """
    Configure tokenizer padding consistently.

    The confidence-detector training pipeline uses left padding because
    ``hidden_states.py`` extracts:

        rep_token = -1

    as the representative position.

    With left padding:

        [PAD PAD t1 t2 t3]

    the final tensor position is always the last real token.

    Parameters
    ----------
    tokenizer:
        Hugging Face tokenizer.

    padding_side:
        Either ``"left"`` or ``"right"``.

    pad_token_id:
        Explicit padding-token ID.

        If None and the tokenizer has no pad token, its EOS token is reused
        as the padding token.
    """

    if padding_side not in {
        "left",
        "right",
    }:
        raise ValueError(
            "padding_side must be either 'left' or 'right'."
        )

    tokenizer.padding_side = (
        padding_side
    )

    # ------------------------------------------------------------------
    # Preferred behavior:
    #
    # If the user explicitly specifies a pad token ID, preserve it.
    # This allows exact reproduction of the original notebook where
    # pad_token_id=0 was used.
    # ------------------------------------------------------------------

    if pad_token_id is not None:

        if (
            pad_token_id < 0
            or pad_token_id >= len(tokenizer)
        ):
            raise ValueError(
                f"pad_token_id={pad_token_id} is outside the tokenizer "
                f"vocabulary range [0, {len(tokenizer) - 1}]."
            )

        tokenizer.pad_token_id = int(
            pad_token_id
        )

    # ------------------------------------------------------------------
    # Otherwise, if the tokenizer already has a pad token, leave it alone.
    #
    # If it does not, reuse EOS.
    #
    # Reusing EOS avoids expanding the tokenizer vocabulary and therefore
    # avoids needing to resize the language-model embedding matrix.
    # ------------------------------------------------------------------

    elif tokenizer.pad_token_id is None:

        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither a pad token nor an EOS token. "
                "A padding token must be configured explicitly."
            )

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    return tokenizer


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    hf_token: Optional[str] = None,
    load_in_4bit: bool = True,
    compute_dtype: Optional[torch.dtype] = None,
    use_double_quant: bool = True,
    quant_type: str = "nf4",
    padding_side: str = "left",
    pad_token_id: Optional[int] = 0,
    device_map: str = "auto",
):
    """
    Load the tokenizer and causal language model used by INKER.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier.

        Default:

            mistralai/Mistral-7B-Instruct-v0.1

    hf_token:
        Optional Hugging Face access token.

        This is only required if the selected model requires authentication
        or gated access.

    load_in_4bit:
        If True, load model weights using bitsandbytes 4-bit quantization.

        This is useful for running Mistral-7B on limited-memory GPUs such as
        those commonly available in Google Colab.

    compute_dtype:
        Compute dtype used by the quantized layers.

        If None:

            CUDA + BF16 support -> torch.bfloat16
            CUDA otherwise      -> torch.float16
            CPU                 -> torch.float32

        To reproduce the original notebook exactly, explicitly use:

            compute_dtype=torch.float16

    use_double_quant:
        Enable nested / double quantization.

        This further reduces memory usage for 4-bit models.

    quant_type:
        Bitsandbytes 4-bit quantization type.

        Default:

            "nf4"

    padding_side:
        Tokenizer padding direction.

        The INKER confidence detector requires:

            "left"

        when ``rep_token=-1`` is used by ``hidden_states.py``.

    pad_token_id:
        Explicit tokenizer padding-token ID.

        The default remains:

            0

        to preserve compatibility with the original replication notebook.

        Set to None if you prefer the loader to keep the tokenizer's native
        padding token, or reuse EOS if no padding token exists.

    device_map:
        Hugging Face device placement strategy.

        Default:

            "auto"

        which allows Accelerate / Transformers to place the model on the
        available device automatically.

    Returns
    -------
    tokenizer:
        Configured Hugging Face tokenizer.

    model:
        Loaded causal language model in evaluation mode.

    Raises
    ------
    RuntimeError
        If 4-bit loading is requested without CUDA.

    ValueError
        If incompatible tokenizer or quantization settings are supplied.
    """

    # ------------------------------------------------------------------
    # 1. Validate quantized loading.
    #
    # bitsandbytes 4-bit loading is primarily intended for supported
    # accelerator environments. For the current Colab workflow we expect
    # CUDA.
    # ------------------------------------------------------------------

    if (
        load_in_4bit
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "load_in_4bit=True was requested, but CUDA is not available. "
            "Either enable a GPU runtime or call "
            "load_model_and_tokenizer(load_in_4bit=False)."
        )

    if quant_type not in {
        "nf4",
        "fp4",
    }:
        raise ValueError(
            "quant_type must be either 'nf4' or 'fp4'."
        )

    # ------------------------------------------------------------------
    # 2. Load tokenizer.
    # ------------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        token=hf_token,
    )

    tokenizer = _configure_padding(
        tokenizer=tokenizer,
        padding_side=padding_side,
        pad_token_id=pad_token_id,
    )

    # ------------------------------------------------------------------
    # 3. Select compute dtype.
    # ------------------------------------------------------------------

    selected_dtype = (
        _select_compute_dtype(
            compute_dtype
        )
    )

    # ------------------------------------------------------------------
    # 4. Prepare model-loading arguments shared by quantized and
    #    non-quantized modes.
    # ------------------------------------------------------------------

    model_kwargs = {
        "device_map":
            device_map,

        "token":
            hf_token,
    }

    # ------------------------------------------------------------------
    # 5. Configure 4-bit loading.
    #
    # Current Hugging Face / bitsandbytes configuration uses:
    #
    #     load_in_4bit
    #     bnb_4bit_compute_dtype
    #     bnb_4bit_quant_type
    #     bnb_4bit_use_double_quant
    #
    # NF4 is retained as the explicit repository default.
    # ------------------------------------------------------------------

    if load_in_4bit:

        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,

                bnb_4bit_compute_dtype=(
                    selected_dtype
                ),

                bnb_4bit_use_double_quant=(
                    use_double_quant
                ),

                bnb_4bit_quant_type=(
                    quant_type
                ),
            )
        )

        model_kwargs[
            "quantization_config"
        ] = (
            quantization_config
        )

    # ------------------------------------------------------------------
    # 6. Non-quantized loading.
    #
    # Explicitly supply the chosen dtype when loading regular weights.
    # ------------------------------------------------------------------

    else:

        model_kwargs[
            "torch_dtype"
        ] = (
            selected_dtype
        )

    # ------------------------------------------------------------------
    # 7. Load the language model.
    # ------------------------------------------------------------------

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_name,
            **model_kwargs,
        )
    )

    # ------------------------------------------------------------------
    # 8. Keep tokenizer and model generation configuration consistent.
    # ------------------------------------------------------------------

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    if hasattr(
        model,
        "generation_config",
    ):
        model.generation_config.pad_token_id = (
            tokenizer.pad_token_id
        )

    # ------------------------------------------------------------------
    # Hidden states are requested explicitly by hidden_states.py and
    # generation.py when needed, so there is no need to force them for
    # every model invocation globally.
    #
    # This avoids unnecessary hidden-state memory during ordinary passes.
    # ------------------------------------------------------------------

    model.config.output_hidden_states = (
        False
    )

    model.eval()

    return (
        tokenizer,
        model,
    )
