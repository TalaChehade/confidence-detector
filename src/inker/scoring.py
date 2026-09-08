"""
Offline confidence-detector scoring and evaluation utilities.

This module evaluates the representation reader learned by ``direction.py``.

It is used primarily for detector-level experiments such as:

    - train/eval/test ROC-AUC,
    - confident-vs-unconfident pairwise accuracy,
    - per-topic accuracy,
    - and inspection of raw confidence scores.

This module evaluates COMPLETE TEXT REPRESENTATIONS.

It is separate from ``generation.py``, which applies the same learned
representation reader to generated tokens during live autoregressive
generation.

Confidence scoring
------------------
For text representation h^(l) at transformer layer l:

    score^(l)
        = sign^(l)
          * ((h^(l) - mu^(l)) @ v^(l))

where:

    mu^(l)
        Stored paired-difference mean learned during detector training.

    v^(l)
        PCA confidence direction.

    sign^(l)
        Orientation indicating which side of the PCA axis corresponds to
        higher confidence.

The final detector score is the mean across selected layers:

    score = mean_l(score^(l))

Higher scores are therefore intended to correspond to greater confidence.

Evaluation pair structure
-------------------------
The evaluation and test datasets are expected to preserve pairs in this order:

    [
        confident_0,
        unconfident_0,

        confident_1,
        unconfident_1,

        ...
    ]

Thus the corresponding labels are:

    [1, 0, 1, 0, ...]

For every pair:

    score(confident) > score(unconfident)

is counted as a correct pairwise prediction.

This module does NOT:
    - construct training pairs,
    - learn PCA directions,
    - causally normalize token confidence,
    - calculate query complexity E,
    - compute K(t_i),
    - or perform live retrieval triggering.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .hidden_states import batched_string_to_hiddens


# ==========================================================================
# Internal validation
# ==========================================================================

def _validate_rep_reader(
    rep_reader: Dict[str, Any],
    layers: Sequence[int],
) -> None:
    """
    Validate that the representation reader contains all required objects.

    Expected structure:

        rep_reader = {
            "directions": {
                layer: ...
            },
            "H_train_means": {
                layer: ...
            },
            "signs": {
                layer: ...
            },
        }

    Every requested layer must exist in all three components.
    """

    required_keys = {
        "directions",
        "H_train_means",
        "signs",
    }

    missing_keys = (
        required_keys
        - set(rep_reader.keys())
    )

    if missing_keys:
        raise KeyError(
            "rep_reader is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    if len(layers) == 0:
        raise ValueError(
            "At least one detector layer must be supplied."
        )

    for layer in layers:

        for component in required_keys:

            if layer not in rep_reader[component]:
                raise KeyError(
                    f"Layer {layer} is missing from "
                    f"rep_reader['{component}']."
                )


# ==========================================================================
# Raw text scoring
# ==========================================================================

def score_texts(
    texts: Sequence[str],
    rep_reader: Dict[str, Any],
    tokenizer: Any,
    model: Any,
    layers: Sequence[int],
    batch_size: int = 32,
    rep_token: int = -1,
    max_length: int = 512,
    return_per_layer: bool = False,
):
    """
    Score complete texts using the learned confidence representation reader.

    Parameters
    ----------
    texts:
        Texts to score.

    rep_reader:
        Learned representation reader returned by ``direction.get_directions``.

    tokenizer:
        Hugging Face tokenizer.

    model:
        Base causal language model used to produce hidden states.

    layers:
        Transformer layers used by the detector.

    batch_size:
        Hidden-state extraction batch size.

    rep_token:
        Representative sequence position.

        The detector normally uses:

            rep_token = -1

        together with left padding.

    max_length:
        Maximum tokenized sequence length.

    return_per_layer:
        If False, return only the final mean score for each text.

        If True, return:

            {
                "scores": ...,
                "per_layer_scores": ...
            }

        which is useful for layer-level diagnostics.

    Returns
    -------
    np.ndarray
        By default, one scalar confidence score per text:

            shape = (N,)

    or dict
        If ``return_per_layer=True``.

    Notes
    -----
    Suppose:

        N = number of texts
        D = hidden dimension
        L = number of detector layers

    For each layer:

        H^(l).shape = (N, D)

    Projection produces:

        score^(l).shape = (N,)

    Stacking all detector layers gives:

        layer_scores.shape = (L, N)

    and averaging over layers gives:

        scores.shape = (N,)
    """

    if len(texts) == 0:
        raise ValueError(
            "texts is empty. At least one text is required."
        )

    _validate_rep_reader(
        rep_reader=rep_reader,
        layers=layers,
    )

    # ------------------------------------------------------------------
    # 1. Extract representative hidden states.
    #
    # For each layer:
    #
    #     hidden_states[layer].shape = (N, D)
    # ------------------------------------------------------------------

    hidden_states = batched_string_to_hiddens(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        layers=layers,
        batch_size=batch_size,
        rep_token=rep_token,
        max_length=max_length,
    )

    per_layer_scores = {}

    # ------------------------------------------------------------------
    # 2. Project every text representation onto each learned confidence
    #    direction.
    # ------------------------------------------------------------------

    for layer in layers:

        layer_hidden = (
            hidden_states[layer]
        )

        train_mean = (
            np.asarray(
                rep_reader[
                    "H_train_means"
                ][layer]
            )
            .reshape(1, -1)
        )

        direction = (
            np.asarray(
                rep_reader[
                    "directions"
                ][layer]
            )
            .reshape(-1)
        )

        sign = float(
            rep_reader[
                "signs"
            ][layer]
        )

        # --------------------------------------------------------------
        # Validate dimensions.
        #
        # layer_hidden:
        #
        #     (N, D)
        #
        # train_mean:
        #
        #     (1, D)
        #
        # direction:
        #
        #     (D,)
        # --------------------------------------------------------------

        hidden_dim = (
            layer_hidden.shape[1]
        )

        if train_mean.shape[1] != hidden_dim:
            raise ValueError(
                f"Layer {layer}: hidden dimension is {hidden_dim}, "
                f"but H_train_means has dimension "
                f"{train_mean.shape[1]}."
            )

        if direction.shape[0] != hidden_dim:
            raise ValueError(
                f"Layer {layer}: hidden dimension is {hidden_dim}, "
                f"but confidence direction has dimension "
                f"{direction.shape[0]}."
            )

        # --------------------------------------------------------------
        # Center representations using the same stored training mean used
        # by the detector.
        #
        # Broadcasting:
        #
        #     (N, D) - (1, D)
        #
        #             ↓
        #
        #          (N, D)
        # --------------------------------------------------------------

        centered = (
            layer_hidden
            - train_mean
        )

        # --------------------------------------------------------------
        # Project onto the PCA confidence direction.
        #
        #     (N, D) @ (D,)
        #
        #          ↓
        #
        #        (N,)
        # --------------------------------------------------------------

        projection = (
            centered
            @ direction
        )

        # --------------------------------------------------------------
        # Orient PCA axis so larger values correspond to greater
        # confidence.
        # --------------------------------------------------------------

        signed_scores = (
            sign
            * projection
        )

        per_layer_scores[
            int(layer)
        ] = (
            signed_scores.astype(
                np.float64,
                copy=False,
            )
        )

    # ------------------------------------------------------------------
    # 3. Average scores across detector layers.
    #
    # Stack:
    #
    #     (L, N)
    #
    # Mean across layers:
    #
    #     (N,)
    # ------------------------------------------------------------------

    stacked_scores = np.stack(
        [
            per_layer_scores[
                int(layer)
            ]
            for layer in layers
        ],
        axis=0,
    )

    scores = np.mean(
        stacked_scores,
        axis=0,
    )

    if return_per_layer:

        return {
            "scores":
                scores,

            "per_layer_scores":
                per_layer_scores,
        }

    return scores


# ==========================================================================
# Split-level evaluation
# ==========================================================================

def evaluate(
    split_name: str,
    texts: Sequence[str],
    rep_reader: Dict[str, Any],
    tokenizer: Any,
    model: Any,
    layers: Sequence[int],
    batch_size: int = 32,
    rep_token: int = -1,
    max_length: int = 512,
    verbose: bool = True,
):
    """
    Evaluate the confidence detector on an ordered contrastive split.

    Expected ordering
    -----------------
    ``texts`` must be arranged as:

        [
            confident_0,
            unconfident_0,
            confident_1,
            unconfident_1,
            ...
        ]

    Thus:

        labels = [1, 0, 1, 0, ...]

    Two metrics are reported.

    ROC-AUC
    -------
    Measures how well the raw detector score ranks confident examples above
    unconfident examples globally.

    Pairwise accuracy
    -----------------
    Measures how often:

        score(confident_i)
            >
        score(unconfident_i)

    for the original contrastive pair i.

    Returns
    -------
    dict
        {
            "auc": ...,
            "pairwise_accuracy": ...,
            "scores": ...,
            "pair_scores": ...
        }

    Notes
    -----
    A trailing unmatched text is NOT silently discarded.

    Evaluation data should contain complete pairs. If an odd number of texts
    is encountered, that indicates a dataset or alignment problem and an
    exception is raised.
    """

    if len(texts) == 0:
        raise ValueError(
            f"{split_name}: evaluation split is empty."
        )

    if len(texts) % 2 != 0:
        raise ValueError(
            f"{split_name}: received {len(texts)} texts. "
            "Evaluation requires complete confident/unconfident pairs, "
            "so the number of texts must be even."
        )

    n_pairs = (
        len(texts)
        // 2
    )

    # ------------------------------------------------------------------
    # Score all texts.
    # ------------------------------------------------------------------

    scores = score_texts(
        texts=texts,
        rep_reader=rep_reader,
        tokenizer=tokenizer,
        model=model,
        layers=layers,
        batch_size=batch_size,
        rep_token=rep_token,
        max_length=max_length,
    )

    if scores.shape[0] != len(texts):
        raise RuntimeError(
            f"{split_name}: received {scores.shape[0]} scores "
            f"for {len(texts)} texts."
        )

    # ------------------------------------------------------------------
    # Labels correspond to:
    #
    #     confident   -> 1
    #     unconfident -> 0
    #
    # repeated once per pair.
    # ------------------------------------------------------------------

    labels = np.tile(
        np.array(
            [1, 0],
            dtype=np.int64,
        ),
        n_pairs,
    )

    # ------------------------------------------------------------------
    # Global ranking quality.
    # ------------------------------------------------------------------

    auc = roc_auc_score(
        labels,
        scores,
    )

    # ------------------------------------------------------------------
    # Restore pair structure:
    #
    #     scores:
    #         [c0, u0, c1, u1, ...]
    #
    # becomes:
    #
    #         [
    #             [c0, u0],
    #             [c1, u1],
    #             ...
    #         ]
    #
    # Shape:
    #
    #     (N_pairs, 2)
    # ------------------------------------------------------------------

    pair_scores = scores.reshape(
        n_pairs,
        2,
    )

    pair_correct = (
        pair_scores[:, 0]
        >
        pair_scores[:, 1]
    )

    pairwise_acc = float(
        np.mean(
            pair_correct
        )
    )

    if verbose:

        print(
            f"{split_name}: "
            f"AUC={auc:.4f} | "
            f"pairwise accuracy "
            f"(confident > unconfident)="
            f"{pairwise_acc:.4f}"
        )

    return {
        "auc":
            float(
                auc
            ),

        "pairwise_accuracy":
            pairwise_acc,

        "scores":
            scores,

        "pair_scores":
            pair_scores,

        "pair_correct":
            pair_correct,
    }


# ==========================================================================
# Per-topic evaluation
# ==========================================================================

def per_topic_breakdown(
    split_name: str,
    pair_scores: np.ndarray,
    pair_topics: Sequence[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute pairwise confidence accuracy independently for every topic.

    Parameters
    ----------
    split_name:
        Human-readable split name used for console output.

    pair_scores:
        Array of shape:

            (N_pairs, 2)

        where:

            pair_scores[i, 0]
                confidence score of confident example i

            pair_scores[i, 1]
                confidence score of unconfident example i

    pair_topics:
        One topic label per contrastive pair.

        Therefore:

            len(pair_topics)
                ==
            len(pair_scores)

    verbose:
        Whether to print the resulting dataframe.

    Returns
    -------
    pandas.DataFrame
        Columns:

            topic
            n_pairs
            n_correct
            pairwise_accuracy

        sorted from worst to best pairwise accuracy.

    Important
    ---------
    A mismatch between ``pair_scores`` and ``pair_topics`` raises an error.

    The previous implementation silently truncated both to their minimum
    length, which could hide dataset alignment bugs.
    """

    pair_scores = np.asarray(
        pair_scores
    )

    if pair_scores.ndim != 2:
        raise ValueError(
            "pair_scores must be a two-dimensional array."
        )

    if pair_scores.shape[1] != 2:
        raise ValueError(
            "pair_scores must have shape (N_pairs, 2). "
            f"Received shape {pair_scores.shape}."
        )

    if len(pair_scores) != len(pair_topics):
        raise ValueError(
            f"{split_name}: pair_scores contains "
            f"{len(pair_scores)} pairs, but pair_topics contains "
            f"{len(pair_topics)} topic labels."
        )

    if len(pair_scores) == 0:
        return pd.DataFrame(
            columns=[
                "topic",
                "n_pairs",
                "n_correct",
                "pairwise_accuracy",
            ]
        )

    rows = []

    unique_topics = sorted(
        set(
            pair_topics
        )
    )

    for topic in unique_topics:

        indices = [
            i
            for i, current_topic
            in enumerate(
                pair_topics
            )
            if current_topic == topic
        ]

        topic_scores = (
            pair_scores[
                indices
            ]
        )

        correct = (
            topic_scores[:, 0]
            >
            topic_scores[:, 1]
        )

        n_correct = int(
            np.sum(
                correct
            )
        )

        accuracy = float(
            np.mean(
                correct
            )
        )

        rows.append({
            "topic":
                topic,

            "n_pairs":
                len(
                    indices
                ),

            "n_correct":
                n_correct,

            "pairwise_accuracy":
                accuracy,
        })

    df = pd.DataFrame(
        rows
    )

    df = (
        df
        .sort_values(
            by=[
                "pairwise_accuracy",
                "topic",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    if verbose:

        print(
            f"\n{split_name} per-topic "
            "pairwise accuracy "
            "(worst first):"
        )

        display_df = (
            df.copy()
        )

        display_df[
            "pairwise_accuracy"
        ] = (
            display_df[
                "pairwise_accuracy"
            ]
            .round(4)
        )

        print(
            display_df.to_string(
                index=False
            )
        )

    return df
