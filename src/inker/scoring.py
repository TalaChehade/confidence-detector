import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .hidden_states import batched_string_to_hiddens


def score_texts(
    texts,
    rep_reader,
    tokenizer,
    model,
    layers,
    batch_size=32,
    rep_token=-1,
    max_length=512,
):
    """Score texts with the learned confidence directions."""
    hidden_states = batched_string_to_hiddens(
        texts,
        tokenizer=tokenizer,
        model=model,
        layers=layers,
        batch_size=batch_size,
        rep_token=rep_token,
        max_length=max_length,
    )

    per_layer_scores = []

    for layer in layers:
        centered = (
            hidden_states[layer]
            - rep_reader["H_train_means"][layer]
        )

        proj = (
            centered
            @ rep_reader["directions"][layer]
        )

        per_layer_scores.append(
            rep_reader["signs"][layer] * proj
        )

    return np.mean(
        per_layer_scores,
        axis=0,
    )


def evaluate(
    split_name,
    texts,
    rep_reader,
    tokenizer,
    model,
    layers,
    batch_size=32,
    rep_token=-1,
    max_length=512,
    verbose=True,
):
    """
    Reproduce the original evaluation:
      * discard a trailing unpaired text, if present;
      * labels are [1, 0] for each complete pair;
      * compute ROC-AUC;
      * compute pairwise confident > unconfident accuracy.
    """
    n_pairs = len(texts) // 2
    texts = texts[:n_pairs * 2]

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

    labels = np.array(
        [1, 0] * n_pairs
    )

    auc = roc_auc_score(
        labels,
        scores,
    )

    pair_scores = scores.reshape(
        -1,
        2,
    )

    pairwise_acc = np.mean(
        pair_scores[:, 0]
        > pair_scores[:, 1]
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
        "auc": float(auc),
        "pairwise_accuracy": float(pairwise_acc),
        "scores": scores,
        "pair_scores": pair_scores,
    }


def per_topic_breakdown(
    split_name,
    pair_scores,
    pair_topics_slice,
    verbose=True,
):
    """Original per-topic pairwise-accuracy diagnostic."""
    n = min(
        len(pair_scores),
        len(pair_topics_slice),
    )

    pair_scores = pair_scores[:n]
    topics_slice = pair_topics_slice[:n]

    rows = []

    for topic in sorted(set(topics_slice)):
        idx = [
            i
            for i, t in enumerate(topics_slice)
            if t == topic
        ]

        if not idx:
            continue

        sub = pair_scores[idx]

        acc = np.mean(
            sub[:, 0] > sub[:, 1]
        )

        rows.append({
            "topic": topic,
            "n_pairs": len(idx),
            "pairwise_accuracy": round(
                float(acc),
                4,
            ),
        })

    df = pd.DataFrame(rows).sort_values(
        "pairwise_accuracy"
    )

    if verbose:
        print(
            f"\n{split_name} per-topic pairwise "
            f"accuracy (worst first):"
        )
        print(df.to_string(index=False))

    return df
