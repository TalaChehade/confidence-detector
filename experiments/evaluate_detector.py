"""
Evaluate the trained INKER confidence representation detector.

This experiment evaluates the representation reader produced by:

    experiments/train_detector.py

on the evaluation and test splits of the INKER confidence dataset.

The evaluation reports:

    1. ROC-AUC
    2. confident-vs-unconfident pairwise accuracy
    3. per-topic pairwise accuracy

The resulting CSV files are saved in the configured replication-results
directory.

Pipeline
--------
The experiment performs:

    configuration
        ↓
    load Mistral model/tokenizer
        ↓
    reconstruct INKER confident/unconfident pairs
        ↓
    recreate EXACTLY the same train/eval/test split used during training
        ↓
    load trained representation reader
        ↓
    score eval/test texts
        ↓
    calculate:
        ROC-AUC
        pairwise accuracy
        per-topic accuracy
        ↓
    save CSV results

Important
---------
The train/eval/test split MUST be reconstructed with exactly the same:

    - dataset,
    - tokenizer,
    - pair construction,
    - train/eval/test ratios,
    - and random seed

used during detector training.

Otherwise evaluation may accidentally test on examples used for training or
produce results that cannot be compared with the trained detector.

This script does NOT retrain the representation reader.
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from _common import (
    get_config,
    get_detector_layers,
    get_project_path,
    load_configured_model,
)

from inker.dataset import (
    build_inker_pairs,
    make_split,
)

from inker.scoring import (
    evaluate,
    per_topic_breakdown,
)


# ==========================================================================
# Reproducibility
# ==========================================================================

def set_random_seed(
    seed: int,
) -> None:
    """
    Seed Python, NumPy, and PyTorch random-number generators.

    Although evaluation itself is deterministic, dataset construction and
    splitting may use randomness. Using the same seed as training is
    therefore essential for reconstructing exactly the same split.
    """

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ==========================================================================
# Representation reader loading
# ==========================================================================

def load_representation_reader(
    reader_path: Path,
    layers: list[int],
) -> Dict[str, Any]:
    """
    Load and validate the trained confidence representation reader.

    Expected structure
    ------------------
    The pickle file must contain:

        directions
        H_train_means
        signs

    for every detector layer requested by the current configuration.
    """

    if not reader_path.exists():

        raise FileNotFoundError(
            "Representation reader not found:\n"
            f"    {reader_path}\n\n"
            "Run experiments/train_detector.py first."
        )

    with reader_path.open(
        "rb"
    ) as file:

        rep_reader = pickle.load(
            file
        )

    if not isinstance(
        rep_reader,
        dict,
    ):

        raise TypeError(
            "The loaded representation reader must be a dictionary."
        )

    required_keys = {
        "directions",
        "H_train_means",
        "signs",
    }

    missing_keys = (
        required_keys
        - set(
            rep_reader.keys()
        )
    )

    if missing_keys:

        raise KeyError(
            "Representation reader is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    # ------------------------------------------------------------------
    # Make sure the representation reader was trained for every layer
    # requested by this evaluation configuration.
    # ------------------------------------------------------------------

    for layer in layers:

        for component in required_keys:

            if layer not in rep_reader[
                component
            ]:

                raise KeyError(
                    f"Representation reader does not contain layer "
                    f"{layer} in '{component}'.\n"
                    "Training and evaluation must use the same detector "
                    "layer configuration."
                )

    return rep_reader


# ==========================================================================
# Main evaluation
# ==========================================================================

def main(
    config_path: str | Path | None = None,
) -> None:
    """
    Run detector evaluation.
    """

    # ------------------------------------------------------------------
    # 1. Load configuration.
    # ------------------------------------------------------------------

    config = get_config(
        config_path
    )

    seed = int(
        config[
            "experiment"
        ][
            "seed"
        ]
    )

    set_random_seed(
        seed
    )

    # ------------------------------------------------------------------
    # 2. Resolve project paths.
    # ------------------------------------------------------------------

    dataset_path = get_project_path(
        config,
        "dataset",
    )

    reader_path = get_project_path(
        config,
        "representation_reader",
    )

    result_dir = get_project_path(
        config,
        "replication_results",
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # 3. Detector configuration.
    # ------------------------------------------------------------------

    layers = get_detector_layers(
        config
    )

    detector_config = config[
        "detector"
    ]

    batch_size = int(
        detector_config[
            "batch_size"
        ]
    )

    rep_token = int(
        detector_config[
            "rep_token"
        ]
    )

    max_length = int(
        detector_config[
            "max_length"
        ]
    )

    # ------------------------------------------------------------------
    # 4. Read split ratios from configuration when available.
    #
    # Keeping these values centralized prevents train_detector.py and
    # evaluate_detector.py from silently creating different datasets.
    #
    # If your current default.yaml does not yet contain these values, the
    # original 70 / 15 / 15 ratios are preserved.
    # ------------------------------------------------------------------

    split_config = config.get(
        "split",
        {}
    )

    train_ratio = float(
        split_config.get(
            "train_ratio",
            0.70,
        )
    )

    eval_ratio = float(
        split_config.get(
            "eval_ratio",
            0.15,
        )
    )

    test_ratio = float(
        split_config.get(
            "test_ratio",
            0.15,
        )
    )

    ratio_sum = (
        train_ratio
        + eval_ratio
        + test_ratio
    )

    if not np.isclose(
        ratio_sum,
        1.0,
    ):

        raise ValueError(
            "Dataset split ratios must sum to 1.0. "
            f"Received train={train_ratio}, "
            f"eval={eval_ratio}, "
            f"test={test_ratio}, "
            f"sum={ratio_sum}."
        )

    # ------------------------------------------------------------------
    # 5. Load model and tokenizer.
    #
    # These must match the configuration used during detector training.
    # ------------------------------------------------------------------

    tokenizer, model = (
        load_configured_model(
            config
        )
    )

    # ------------------------------------------------------------------
    # 6. Reconstruct INKER confident/unconfident pairs.
    #
    # build_inker_pairs currently returns:
    #
    #     honest_statements
    #     untruthful_statements
    #     topics
    #     pair_topics
    #
    # "honest" / "untruthful" are legacy CtrlA-compatible names.
    # Conceptually, in the INKER replication they correspond to:
    #
    #     confident
    #     unconfident
    # ------------------------------------------------------------------

    (
        honest_statements,
        untruthful_statements,
        _topics,
        pair_topics,
    ) = build_inker_pairs(
        dataset_path,
        tokenizer,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # 7. Recreate EXACTLY the same dataset split used for training.
    #
    # IMPORTANT FIX:
    #
    # The old version used:
    #
    #     seed=0
    #
    # here even though the experiment seed had already been loaded from
    # config.
    #
    # That could cause evaluation to use a different split from training.
    #
    # We now use:
    #
    #     seed=seed
    #
    # everywhere.
    # ------------------------------------------------------------------

    dataset = make_split(
        honest_statements,
        untruthful_statements,
        pair_topics,
        train_ratio=train_ratio,
        eval_ratio=eval_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # 8. Validate reconstructed split.
    # ------------------------------------------------------------------

    for split_name in (
        "train",
        "eval",
        "test",
    ):

        if split_name not in dataset:

            raise KeyError(
                f"Dataset split '{split_name}' was not produced."
            )

        split = dataset[
            split_name
        ]

        if "data" not in split:

            raise KeyError(
                f"Dataset split '{split_name}' does not contain 'data'."
            )

        if "topics" not in split:

            raise KeyError(
                f"Dataset split '{split_name}' does not contain 'topics'."
            )

        # Every pair contributes exactly two texts.
        if len(
            split["data"]
        ) % 2 != 0:

            raise ValueError(
                f"{split_name} contains an odd number of texts: "
                f"{len(split['data'])}."
            )

        expected_pairs = (
            len(
                split[
                    "data"
                ]
            )
            // 2
        )

        if len(
            split[
                "topics"
            ]
        ) != expected_pairs:

            raise ValueError(
                f"{split_name}: found {expected_pairs} text pairs "
                f"but {len(split['topics'])} topic labels."
            )

    # ------------------------------------------------------------------
    # 9. Load the trained representation reader.
    # ------------------------------------------------------------------

    rep_reader = (
        load_representation_reader(
            reader_path=reader_path,
            layers=layers,
        )
    )

    # ------------------------------------------------------------------
    # 10. Arguments shared by eval and test.
    # ------------------------------------------------------------------

    scoring_args = {
        "rep_reader":
            rep_reader,

        "tokenizer":
            tokenizer,

        "model":
            model,

        "layers":
            layers,

        "batch_size":
            batch_size,

        "rep_token":
            rep_token,

        "max_length":
            max_length,
    }

    # ==================================================================
    # Evaluation split
    # ==================================================================

    eval_result = evaluate(
        split_name="Eval",
        texts=dataset[
            "eval"
        ][
            "data"
        ],
        **scoring_args,
    )

    # ==================================================================
    # Test split
    # ==================================================================

    test_result = evaluate(
        split_name="Test",
        texts=dataset[
            "test"
        ][
            "data"
        ],
        **scoring_args,
    )

    # ==================================================================
    # Overall metrics
    # ==================================================================

    metrics_df = pd.DataFrame([
        {
            "split":
                "eval",

            "n_pairs":
                len(
                    dataset[
                        "eval"
                    ][
                        "topics"
                    ]
                ),

            "n_texts":
                len(
                    dataset[
                        "eval"
                    ][
                        "data"
                    ]
                ),

            "roc_auc":
                eval_result[
                    "auc"
                ],

            "pairwise_accuracy":
                eval_result[
                    "pairwise_accuracy"
                ],
        },
        {
            "split":
                "test",

            "n_pairs":
                len(
                    dataset[
                        "test"
                    ][
                        "topics"
                    ]
                ),

            "n_texts":
                len(
                    dataset[
                        "test"
                    ][
                        "data"
                    ]
                ),

            "roc_auc":
                test_result[
                    "auc"
                ],

            "pairwise_accuracy":
                test_result[
                    "pairwise_accuracy"
                ],
        },
    ])

    metrics_path = (
        result_dir
        / "replication_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # ==================================================================
    # Per-topic evaluation
    # ==================================================================

    eval_topic_df = (
        per_topic_breakdown(
            split_name="Eval",
            pair_scores=eval_result[
                "pair_scores"
            ],
            pair_topics=dataset[
                "eval"
            ][
                "topics"
            ],
        )
    )

    test_topic_df = (
        per_topic_breakdown(
            split_name="Test",
            pair_scores=test_result[
                "pair_scores"
            ],
            pair_topics=dataset[
                "test"
            ][
                "topics"
            ],
        )
    )

    eval_topic_path = (
        result_dir
        / "eval_per_topic.csv"
    )

    test_topic_path = (
        result_dir
        / "test_per_topic.csv"
    )

    eval_topic_df.to_csv(
        eval_topic_path,
        index=False,
    )

    test_topic_df.to_csv(
        test_topic_path,
        index=False,
    )

    # ==================================================================
    # Summary
    # ==================================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "INKER CONFIDENCE DETECTOR EVALUATION COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nSeed: {seed}"
    )

    print(
        f"Detector layers: {layers}"
    )

    print(
        "\nSplit sizes:"
    )

    print(
        f"  Eval: "
        f"{len(dataset['eval']['topics'])} pairs "
        f"({len(dataset['eval']['data'])} texts)"
    )

    print(
        f"  Test: "
        f"{len(dataset['test']['topics'])} pairs "
        f"({len(dataset['test']['data'])} texts)"
    )

    print(
        "\nMetrics:"
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {metrics_path}"
    )

    print(
        f"  {eval_topic_path}"
    )

    print(
        f"  {test_topic_path}"
    )

    print(
        "=" * 80
    )


# ==========================================================================
# CLI
# ==========================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained INKER confidence representation detector."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional path to a YAML configuration file. "
            "If omitted, configs/default.yaml is used."
        ),
    )

    args = parser.parse_args()

    main(
        args.config
    )
