"""
Hidden-state extraction utilities for the INKER confidence detector.

This module converts batches of text prompts into transformer hidden-state
representations used by the confidence-direction learning stage.

Its main function:

    batched_string_to_hiddens(...)

takes a collection of strings and, for every requested transformer layer,
extracts the hidden representation of one representative token position.

In the current INKER confidence-detector replication, that representative
position is:

    rep_token = -1

meaning the final tensor position.

Because the tokenizer is configured with LEFT PADDING, the final tensor
position corresponds to the final real token for every sequence in the batch,
regardless of the sequence's original length.

Example
-------
Suppose a batch contains:

    sequence A: 5 tokens
    sequence B: 8 tokens

With left padding:

    A: [PAD PAD PAD t1 t2 t3 t4 t5]
    B: [t1  t2  t3  t4 t5 t6 t7 t8]

The tensor position:

    -1

therefore corresponds to:

    t5 for sequence A
    t8 for sequence B

which is exactly the last real token representation required by the
representation-reading procedure.

Output dimensions
-----------------
If:

    N = number of input texts
    D = model hidden dimension

then for every requested layer l:

    output[l].shape = (N, D)

For Mistral-7B:

    D = 4096

so, for example, 1000 prompts produce:

    output[layer].shape = (1000, 4096)

These representations are later consumed by ``direction.py`` to construct
paired hidden-state differences and learn PCA confidence directions.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm


@torch.no_grad()
def batched_string_to_hiddens(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    layers: Sequence[int],
    batch_size: int = 32,
    rep_token: int = -1,
    max_length: int = 512,
    show_progress: bool = True,
) -> Dict[int, np.ndarray]:
    """
    Extract one representative hidden state per text and transformer layer.

    Parameters
    ----------
    texts:
        Input strings whose representations should be extracted.

    tokenizer:
        Hugging Face tokenizer associated with ``model``.

        For the current INKER detector, the tokenizer must use:

            tokenizer.padding_side = "left"

        when ``rep_token=-1``.

    model:
        Transformer language model.

        The model must support:

            output_hidden_states=True

    layers:
        Hidden-state indices to extract.

        Hugging Face causal language models generally return:

            outputs.hidden_states[0]
                embedding output

            outputs.hidden_states[1]
                transformer block 1 output

            ...

        Therefore the supplied indices must correspond to the same layer
        convention used throughout training and inference.

    batch_size:
        Number of texts processed in each forward pass.

    rep_token:
        Sequence position used as the representative token.

        The INKER confidence detector uses:

            rep_token = -1

        together with left padding, meaning the last real token.

    max_length:
        Maximum tokenized sequence length.

        Longer inputs are truncated by the tokenizer.

    show_progress:
        Whether to display a tqdm progress bar.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping:

            layer -> hidden-state matrix

        where each matrix has shape:

            (number_of_texts, hidden_dimension)

    Raises
    ------
    ValueError
        If:
        - no texts are provided,
        - no layers are supplied,
        - batch_size is invalid,
        - max_length is invalid,
        - or ``rep_token=-1`` is used without left padding.

    IndexError
        If a requested hidden-state layer does not exist.

    Notes
    -----
    This function intentionally extracts representations after tokenization
    and truncation.

    Therefore the representation corresponds to the final token that remains
    after the ``max_length`` constraint has been applied.
    """

    # ------------------------------------------------------------------
    # 1. Validate arguments.
    # ------------------------------------------------------------------

    if len(texts) == 0:
        raise ValueError(
            "texts is empty. At least one input string is required."
        )

    if len(layers) == 0:
        raise ValueError(
            "layers is empty. At least one hidden-state layer is required."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if max_length <= 0:
        raise ValueError(
            "max_length must be greater than zero."
        )

    # ------------------------------------------------------------------
    # The current detector relies on this relationship:
    #
    #     left padding + rep_token=-1
    #
    #             ↓
    #
    #     last tensor position = last real token
    #
    # With right padding, -1 would often select PAD instead.
    # ------------------------------------------------------------------

    if (
        rep_token == -1
        and getattr(
            tokenizer,
            "padding_side",
            None,
        ) != "left"
    ):
        raise ValueError(
            "rep_token=-1 requires tokenizer.padding_side='left' "
            "for batched last-token hidden-state extraction. "
            f"Current padding_side={getattr(tokenizer, 'padding_side', None)!r}."
        )

    model.eval()

    # ------------------------------------------------------------------
    # 2. Prepare one output list per requested layer.
    #
    # Before concatenation:
    #
    #     out[layer] =
    #
    #         [
    #             batch_0_hidden_states,
    #             batch_1_hidden_states,
    #             ...
    #         ]
    #
    # where each batch array has shape:
    #
    #     (batch_size_for_this_batch, hidden_dimension)
    # ------------------------------------------------------------------

    out: Dict[int, list[np.ndarray]] = {
        int(layer): []
        for layer in layers
    }

    # ------------------------------------------------------------------
    # 3. Process texts batch by batch.
    # ------------------------------------------------------------------

    iterator = range(
        0,
        len(texts),
        batch_size,
    )

    iterator = tqdm(
        iterator,
        desc="Extracting hidden states",
        disable=not show_progress,
    )

    for start_index in iterator:

        batch = texts[
            start_index:
            start_index + batch_size
        ]

        # --------------------------------------------------------------
        # Left padding is handled by the tokenizer configuration.
        #
        # Example:
        #
        # short:
        #     [PAD PAD PAD x x x x]
        #
        # long:
        #     [x   x   x   x x x x]
        #
        # Therefore position -1 is a real token for both.
        # --------------------------------------------------------------

        inputs = tokenizer(
            list(batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(
            model.device
        )

        # --------------------------------------------------------------
        # We need hidden states only.
        #
        # use_cache=False avoids storing KV caches that are useful for
        # autoregressive generation but unnecessary during bulk
        # representation extraction.
        # --------------------------------------------------------------

        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

        hidden_states = (
            outputs.hidden_states
        )

        if hidden_states is None:
            raise RuntimeError(
                "The model did not return hidden states even though "
                "output_hidden_states=True was requested."
            )

        num_hidden_state_tensors = len(
            hidden_states
        )

        # --------------------------------------------------------------
        # 4. Extract the selected representative token from each layer.
        #
        # If:
        #
        #     B = current batch size
        #     D = hidden dimension
        #
        # then:
        #
        #     hidden_states[layer].shape
        #         = (B, sequence_length, D)
        #
        # Selecting:
        #
        #     [:, rep_token, :]
        #
        # gives:
        #
        #     (B, D)
        # --------------------------------------------------------------

        for layer in layers:

            # Support both positive and normal Python negative indices.
            if not (
                -num_hidden_state_tensors
                <= layer
                < num_hidden_state_tensors
            ):
                raise IndexError(
                    f"Requested hidden-state layer {layer}, but the model "
                    f"returned only {num_hidden_state_tensors} hidden-state "
                    "tensors."
                )

            layer_hidden = (
                hidden_states[layer][
                    :,
                    rep_token,
                    :
                ]
                .float()
                .cpu()
                .numpy()
            )

            out[
                int(layer)
            ].append(
                layer_hidden
            )

    # ------------------------------------------------------------------
    # 5. Concatenate all batches.
    #
    # Example:
    #
    #     batch 1 -> (32, 4096)
    #     batch 2 -> (32, 4096)
    #     batch 3 -> (12, 4096)
    #
    # becomes:
    #
    #     (76, 4096)
    #
    # for each requested layer.
    # ------------------------------------------------------------------

    result: Dict[int, np.ndarray] = {}

    for layer, batches in out.items():

        if not batches:
            raise RuntimeError(
                f"No hidden states were collected for layer {layer}."
            )

        result[layer] = np.vstack(
            batches
        )

        if result[layer].shape[0] != len(texts):
            raise RuntimeError(
                f"Layer {layer} produced {result[layer].shape[0]} "
                f"representations for {len(texts)} input texts."
            )

    return result
