"""
Learn the internal confidence representation directions used by INKER.

This module implements the representation-reading procedure used to construct
the confidence detector from paired confident/unconfident training examples.

For each selected transformer layer, the procedure is:

    1. Extract one hidden-state representation for every training prompt.
    2. Group representations according to the original contrastive pairs.
    3. Compute a difference vector for each pair.
    4. Mean-center the difference vectors.
    5. Fit one-component PCA.
    6. Use the original pair labels to determine which orientation of the
       PCA component corresponds to higher confidence.

The result for every layer consists of:

    directions[layer]
        The first PCA component.

    H_train_means[layer]
        Mean of the paired hidden-state difference vectors.

    signs[layer]
        Either +1 or -1, indicating which PCA orientation corresponds
        to the confident side of the contrastive pairs.

Important
---------
PCA directions have arbitrary sign.

If ``v`` is a valid PCA component, then ``-v`` represents the same principal
axis. Therefore, PCA alone cannot tell us which direction means "confident".

The training labels are used only AFTER PCA to orient that axis.

The returned direction itself is NOT multiplied by its sign. The sign is
stored separately and must be applied exactly once during scoring:

    score = ((h - H_mean) @ direction) * sign

This separation prevents accidentally applying the orientation twice.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.decomposition import PCA

from .hidden_states import batched_string_to_hiddens


def get_directions(
    train_texts: Sequence[str],
    train_labels: Sequence[Sequence[bool]],
    tokenizer: Any,
    model: Any,
    layers: Sequence[int],
    n_difference: int = 1,
    batch_size: int = 32,
    rep_token: int = -1,
    max_length: int = 512,
) -> Dict[str, Dict[int, np.ndarray | int]]:
    """
    Learn one confidence representation direction per transformer layer.

    Parameters
    ----------
    train_texts:
        Flattened training prompts.

        The expected structure is:

            [
                pair_0_text_0,
                pair_0_text_1,
                pair_1_text_0,
                pair_1_text_1,
                ...
            ]

        Each consecutive pair corresponds to one confident/unconfident
        contrastive example.

    train_labels:
        Pair-level labels indicating which member of each pair is confident.

        Example:

            [
                [True, False],
                [False, True],
                ...
            ]

        The ordering may differ from pair to pair because training pairs are
        randomly shuffled during dataset preparation.

    tokenizer:
        Hugging Face tokenizer used by the language model.

    model:
        Language model from which hidden states are extracted.

    layers:
        Transformer layers used by the confidence detector.

        INKER's replication configuration typically selects a fixed subset
        of model layers.

    n_difference:
        Number of successive difference operations.

        The INKER confidence detector implemented in this repository uses:

            n_difference = 1

        corresponding to one confident/unconfident paired difference.

    batch_size:
        Number of prompts processed simultaneously during hidden-state
        extraction.

    rep_token:
        Token position whose hidden representation is used.

        ``-1`` means the final represented token in the tokenized sequence,
        consistent with the existing representation-reading pipeline.

    max_length:
        Maximum tokenizer sequence length.

    Returns
    -------
    dict
        Dictionary containing:

            {
                "directions": {
                    layer: PCA_direction
                },

                "H_train_means": {
                    layer: paired_difference_mean
                },

                "signs": {
                    layer: +1 or -1
                }
            }

    Notes
    -----
    Suppose:

        N = number of contrastive training pairs
        D = model hidden dimension

    Since every pair contains two prompts:

        number of training texts = 2N

    Hidden-state extraction for one layer therefore produces:

        H.shape = (2N, D)

    Pairwise differencing produces:

        DeltaH.shape = (N, D)

    where:

        DeltaH_i = H_(2i) - H_(2i+1)

    PCA is then fitted to the centered DeltaH matrix.
    """

    # ------------------------------------------------------------------
    # 1. Validate the paired training structure.
    # ------------------------------------------------------------------

    if len(train_texts) == 0:
        raise ValueError(
            "train_texts is empty. At least one contrastive pair is required."
        )

    if len(train_texts) % 2 != 0:
        raise ValueError(
            "train_texts must contain an even number of prompts because "
            "every confident example must be paired with one unconfident "
            "example."
        )

    expected_pairs = len(train_texts) // 2

    if len(train_labels) != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} pair-level labels for "
            f"{len(train_texts)} training texts, but received "
            f"{len(train_labels)}."
        )

    for pair_index, pair_labels in enumerate(train_labels):

        if len(pair_labels) != 2:
            raise ValueError(
                f"Training pair {pair_index} has {len(pair_labels)} labels. "
                "Every contrastive pair must contain exactly two labels."
            )

        if sum(bool(label) for label in pair_labels) != 1:
            raise ValueError(
                f"Training pair {pair_index} must contain exactly one "
                "confident label (True)."
            )

    if len(layers) == 0:
        raise ValueError(
            "At least one transformer layer must be supplied."
        )

    # The current INKER/CtrlA representation procedure uses exactly
    # one difference between the two members of each pair.
    if n_difference != 1:
        raise ValueError(
            "This INKER confidence-detector implementation expects "
            "n_difference=1. Additional difference operations would change "
            "the pairing structure and no longer correspond to the "
            "documented detector."
        )

    # ------------------------------------------------------------------
    # 2. Extract hidden-state representations.
    #
    # For N training pairs and hidden dimension D:
    #
    #     hidden_states[layer].shape = (2N, D)
    #
    # Example:
    #
    #     row 0 -> first member of pair 0
    #     row 1 -> second member of pair 0
    #     row 2 -> first member of pair 1
    #     row 3 -> second member of pair 1
    #     ...
    # ------------------------------------------------------------------

    hidden_states = batched_string_to_hiddens(
        train_texts,
        tokenizer=tokenizer,
        model=model,
        layers=layers,
        batch_size=batch_size,
        rep_token=rep_token,
        max_length=max_length,
    )

    # ------------------------------------------------------------------
    # 3. Compute one paired difference vector per contrastive pair.
    #
    # If:
    #
    #     H.shape = (2N, D)
    #
    # then:
    #
    #     H[::2].shape  = (N, D)
    #     H[1::2].shape = (N, D)
    #
    # and:
    #
    #     DeltaH = H[::2] - H[1::2]
    #
    # gives:
    #
    #     DeltaH.shape = (N, D)
    #
    # Note that the first member is NOT always the confident member because
    # the dataset preparation step randomly reorders training pairs.
    #
    # That does not cause a problem: the PCA stage discovers the common
    # contrastive axis, while the later sign-orientation stage uses the
    # labels to determine which orientation means "confident".
    # ------------------------------------------------------------------

    relative_hidden_states: Dict[int, np.ndarray] = {}

    for layer in layers:

        layer_hidden = hidden_states[layer]

        if layer_hidden.shape[0] != len(train_texts):
            raise ValueError(
                f"Layer {layer} returned {layer_hidden.shape[0]} hidden "
                f"representations for {len(train_texts)} training texts."
            )

        relative_hidden_states[layer] = (
            layer_hidden[::2]
            - layer_hidden[1::2]
        )

    # ------------------------------------------------------------------
    # 4. Learn one PCA confidence axis per layer.
    #
    # For every layer:
    #
    #     DeltaH ∈ R^(N x D)
    #
    # First calculate:
    #
    #                   1
    #     H_mean = ----------- Σ DeltaH_i
    #                   N
    #
    # Then center:
    #
    #     X = DeltaH - H_mean
    #
    # and fit PCA with one component.
    #
    # The first component is the direction explaining the largest amount
    # of variance in the paired confidence contrasts.
    # ------------------------------------------------------------------

    directions: Dict[int, np.ndarray] = {}
    H_train_means: Dict[int, np.ndarray] = {}

    for layer in layers:

        H_train = relative_hidden_states[layer]

        if H_train.shape[0] < 2:
            raise ValueError(
                f"Layer {layer} contains only {H_train.shape[0]} paired "
                "difference vectors. PCA requires more training examples."
            )

        # Shape:
        #
        #     H_mean.shape = (1, D)
        #
        # keepdims=True is intentionally preserved because it makes later
        # broadcasting explicit and consistent with the original notebook.
        H_mean = H_train.mean(
            axis=0,
            keepdims=True,
        )

        H_train_means[layer] = H_mean

        centered_differences = (
            H_train
            - H_mean
        )

        pca = PCA(
            n_components=1,
            whiten=False,
        )

        pca.fit(
            centered_differences
        )

        # Shape:
        #
        #     direction.shape = (D,)
        directions[layer] = (
            pca.components_[0].copy()
        )

    # ------------------------------------------------------------------
    # 5. Determine PCA orientation.
    #
    # PCA identifies an AXIS, not a semantic direction.
    #
    # Therefore:
    #
    #     v
    #
    # and:
    #
    #     -v
    #
    # are mathematically equivalent PCA solutions.
    #
    # We use the known training labels to determine whether larger or
    # smaller projections correspond to confident examples.
    # ------------------------------------------------------------------

    signs: Dict[int, int] = {}

    for layer in layers:

        # --------------------------------------------------------------
        # Project the ORIGINAL individual prompt representations onto
        # the learned PCA axis.
        #
        # hidden_states[layer]:
        #
        #     (2N, D)
        #
        # H_train_means[layer]:
        #
        #     (1, D)
        #
        # directions[layer]:
        #
        #     (D,)
        #
        # therefore:
        #
        # centered:
        #
        #     (2N, D)
        #
        # proj:
        #
        #     (2N,)
        # --------------------------------------------------------------

        centered = (
            hidden_states[layer]
            - H_train_means[layer]
        )

        projections = (
            centered
            @ directions[layer]
        )

        # --------------------------------------------------------------
        # Reconstruct the original two-element pair structure.
        #
        # Example:
        #
        # projections =
        #
        #     [p0a, p0b, p1a, p1b, ...]
        #
        # becomes:
        #
        #     [
        #         [p0a, p0b],
        #         [p1a, p1b],
        #         ...
        #     ]
        # --------------------------------------------------------------

        pair_projections = []

        start = 0

        for pair_labels in train_labels:

            pair_length = len(
                pair_labels
            )

            pair_projections.append(
                projections[
                    start:
                    start + pair_length
                ]
            )

            start += pair_length

        # --------------------------------------------------------------
        # Check how often the known confident member has:
        #
        #     the smaller projection
        #
        # versus:
        #
        #     the larger projection.
        #
        # Example:
        #
        # Pair projections:
        #
        #     [-0.8, +1.1]
        #
        # Pair labels:
        #
        #     [False, True]
        #
        # The confident member is at index 1 and has the maximum
        # projection, so this pair votes for positive orientation.
        # --------------------------------------------------------------

        confident_is_min = np.mean([
            pair_output[pair_labels.index(True)]
            == np.min(pair_output)
            for pair_output, pair_labels in zip(
                pair_projections,
                train_labels,
            )
        ])

        confident_is_max = np.mean([
            pair_output[pair_labels.index(True)]
            == np.max(pair_output)
            for pair_output, pair_labels in zip(
                pair_projections,
                train_labels,
            )
        ])

        # --------------------------------------------------------------
        # If confident examples more often occupy the maximum side:
        #
        #     sign = +1
        #
        # If confident examples more often occupy the minimum side:
        #
        #     sign = -1
        #
        # This makes the final signed score increase with confidence.
        # --------------------------------------------------------------

        orientation = np.sign(
            confident_is_max
            - confident_is_min
        )

        # A perfect tie is unlikely. For deterministic behavior,
        # preserve the original convention and choose +1.
        signs[layer] = (
            1
            if orientation == 0
            else int(orientation)
        )

    # ------------------------------------------------------------------
    # 6. Return the learned representation reader.
    #
    # The directions remain unsigned.
    #
    # During scoring, use:
    #
    #     ((h - H_mean) @ direction) * sign
    #
    # and apply ``sign`` exactly once.
    # ------------------------------------------------------------------

    return {
        "directions": directions,
        "H_train_means": H_train_means,
        "signs": signs,
    }
