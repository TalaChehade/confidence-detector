import numpy as np
from sklearn.decomposition import PCA

from .hidden_states import batched_string_to_hiddens


def get_directions(
    train_texts,
    train_labels,
    tokenizer,
    model,
    layers,
    n_difference=1,
    batch_size=32,
    rep_token=-1,
    max_length=512,
):
    """
    Learn one paired-difference PCA confidence direction per layer.

    This preserves the original implementation:
      1. Hidden states for training texts.
      2. Pairwise subtraction [::2] - [1::2].
      3. Mean-center those differences.
      4. Fit one-component PCA.
      5. Determine PCA orientation using the original training pairs.

    IMPORTANT:
    directions are NOT multiplied by sign here. The sign is stored separately
    and is applied exactly once during scoring.
    """
    hidden_states = batched_string_to_hiddens(
        train_texts,
        tokenizer=tokenizer,
        model=model,
        layers=layers,
        batch_size=batch_size,
        rep_token=rep_token,
        max_length=max_length,
    )

    relative_hidden_states = {
        k: np.copy(v)
        for k, v in hidden_states.items()
    }

    for layer in layers:
        for _ in range(n_difference):
            relative_hidden_states[layer] = (
                relative_hidden_states[layer][::2]
                - relative_hidden_states[layer][1::2]
            )

    directions = {}
    H_train_means = {}

    for layer in layers:
        H_train = relative_hidden_states[layer]

        # keepdims=True is preserved from the original notebook.
        H_mean = H_train.mean(axis=0, keepdims=True)
        H_train_means[layer] = H_mean

        pca = PCA(
            n_components=1,
            whiten=False,
        ).fit(H_train - H_mean)

        directions[layer] = pca.components_[0]

    signs = {}

    for layer in layers:
        centered = (
            hidden_states[layer]
            - H_train_means[layer]
        )

        proj = centered @ directions[layer]

        pca_outputs_comp = []
        start = 0

        for pair_labels in train_labels:
            pca_outputs_comp.append(
                proj[start:start + len(pair_labels)]
            )
            start += len(pair_labels)

        outputs_min = np.mean([
            o[pl.index(True)] == min(o)
            for o, pl in zip(
                pca_outputs_comp,
                train_labels,
            )
        ])

        outputs_max = np.mean([
            o[pl.index(True)] == max(o)
            for o, pl in zip(
                pca_outputs_comp,
                train_labels,
            )
        ])

        sign = np.sign(
            outputs_max - outputs_min
        )

        signs[layer] = (
            1 if sign == 0 else sign
        )

    return {
        "directions": directions,
        "H_train_means": H_train_means,
        "signs": signs,
    }
