"""
Train the INKER internal confidence representation detector.

This experiment trains the confidence detector used by the INKER IE-KRT
retrieval-trigger mechanism.

The detector learns a confidence-related direction in the hidden-state space
of the base language model.

Pipeline
--------
The training process is:

    INKER confident/unconfident statements
                ↓
        build contrastive pairs
                ↓
       topic-stratified splitting
                ↓
     hidden-state extraction
                ↓
       pair differences

        Delta h_i^(l)
        = h_(2i)^(l) - h_(2i+1)^(l)

                ↓
         mean centering
                ↓
          PCA, 1 component
                ↓
      confidence direction v^(l)
                ↓
        direction orientation
                ↓
        saved rep_reader.pkl

The saved representation reader contains, for every detector layer:

    directions
        PCA confidence direction v^(l)

    H_train_means
        Mean pair-difference vector used for centering

    signs
        Orientation sign used so that higher projected values correspond
        consistently to the confident side of the contrastive pairs

This trained representation reader is later used by:

    scoring.py
        for offline confidence-detector evaluation

    generation.py
        for live token-level confidence estimation

    run_confidence_only.py
        for the confidence-only baseline

    run_inker_trigger.py
        for the full IE-KRT trigger experiment

Important
---------
The dataset split used here MUST be reconstructed identically by
evaluate_detector.py.

Therefore training and evaluation must use the same:

    - dataset,
    - pair construction,
    - split ratios,
    - and random seed.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
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

from inker.directions import (
    get_directions,
)


# ==========================================================================
# Reproducibility
# ==========================================================================

def set_random_seed(
    seed: int,
) -> None:
    """
    Seed Python, NumPy, and PyTorch random-number generators.

    Dataset construction and splitting use randomness, so the same seed must
    also be used later by evaluate_detector.py when reconstructing the split.
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
# Validation
# ==========================================================================

def validate_dataset_split(
    dataset: Dict[str, Any],
) -> None:
    """
    Validate the train/eval/test structure returned by ``make_split``.

    Expected structure
    ------------------
    Each split must contain:

        data
            Flattened text sequence:

                confident_0
                unconfident_0
                confident_1
                unconfident_1
                ...

        labels
            One pair-level label entry per confident/unconfident pair.

        topics
            One topic label per pair.

    Therefore:

        len(data) = 2 * len(labels)
        len(labels) = len(topics)
    """

    required_splits = (
        "train",
        "eval",
        "test",
    )

    for split_name in required_splits:

        if split_name not in dataset:

            raise KeyError(
                f"Dataset is missing split {split_name!r}."
            )

        split = dataset[
            split_name
        ]

        for required_key in (
            "data",
            "labels",
            "topics",
        ):

            if required_key not in split:

                raise KeyError(
                    f"Dataset split {split_name!r} is missing "
                    f"{required_key!r}."
                )

        n_texts = len(
            split[
                "data"
            ]
        )

        n_labels = len(
            split[
                "labels"
            ]
        )

        n_topics = len(
            split[
                "topics"
            ]
        )

        if n_texts % 2 != 0:

            raise ValueError(
                f"{split_name}: expected an even number of texts, "
                f"but received {n_texts}."
            )

        n_pairs = (
            n_texts
            // 2
        )

        if n_labels != n_pairs:

            raise ValueError(
                f"{split_name}: found {n_pairs} text pairs but "
                f"{n_labels} pair-level labels."
            )

        if n_topics != n_pairs:

            raise ValueError(
                f"{split_name}: found {n_pairs} text pairs but "
                f"{n_topics} topic labels."
            )

        if n_pairs == 0:

            raise ValueError(
                f"{split_name}: split contains zero pairs."
            )


def validate_representation_reader(
    rep_reader: Dict[str, Any],
    layers: list[int],
) -> None:
    """
    Validate the representation reader before saving it.
    """

    if not isinstance(
        rep_reader,
        dict,
    ):

        raise TypeError(
            "get_directions() must return a dictionary."
        )

    required_components = {
        "directions",
        "H_train_means",
        "signs",
    }

    missing_components = (
        required_components
        - set(
            rep_reader.keys()
        )
    )

    if missing_components:

        raise KeyError(
            "Representation reader is missing required components: "
            f"{sorted(missing_components)}"
        )

    for layer in layers:

        for component in required_components:

            if layer not in rep_reader[
                component
            ]:

                raise KeyError(
                    f"Representation reader is missing layer "
                    f"{layer} in {component!r}."
                )


# ==========================================================================
# Main
# ==========================================================================

def main(
    config_path: str | Path | None = None,
) -> None:
    """
    Train and save the INKER confidence representation reader.
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
    # 2. Resolve paths.
    # ------------------------------------------------------------------

    dataset_path = get_project_path(
        config,
        "dataset",
    )

    reader_path = get_project_path(
        config,
        "representation_reader",
    )

    reader_path.parent.mkdir(
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

    n_difference = int(
        detector_config[
            "n_difference"
        ]
    )

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
    # 4. Dataset split configuration.
    #
    # The same values must be used in evaluate_detector.py.
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
    # 5. Load base language model and tokenizer.
    # ------------------------------------------------------------------

    print(
        "Loading base language model..."
    )

    tokenizer, model = (
        load_configured_model(
            config
        )
    )

    # ------------------------------------------------------------------
    # 6. Build INKER contrastive confident/unconfident pairs.
    #
    # The variable names honest_statements / untruthful_statements are
    # retained for compatibility with the original CtrlA-inspired code.
    #
    # In the INKER replication they conceptually mean:
    #
    #     confident_statements
    #     unconfident_statements
    # ------------------------------------------------------------------

    (
        honest_statements,
        untruthful_statements,
        topics,
        pair_topics,
    ) = build_inker_pairs(
        dataset_path,
        tokenizer,
        seed=seed,
    )

    n_pairs_total = len(
        honest_statements
    )

    if len(
        untruthful_statements
    ) != n_pairs_total:

        raise ValueError(
            "Confident and unconfident statement counts differ:\n"
            f"    confident:   {n_pairs_total}\n"
            f"    unconfident: {len(untruthful_statements)}"
        )

    if len(
        pair_topics
    ) != n_pairs_total:

        raise ValueError(
            "Pair-topic count does not match the number of contrastive pairs."
        )

    print(
        "\nDataset:"
    )

    print(
        f"  Topics: "
        f"{len(topics)}"
    )

    print(
        f"  Total pairs before split: "
        f"{n_pairs_total}"
    )

    # ------------------------------------------------------------------
    # 7. Create train/eval/test split.
    #
    # IMPORTANT:
    #
    # Use the SAME experiment seed here and in evaluate_detector.py.
    #
    # The old implementation used seed=0 here, which could silently create
    # a different split from other experiment scripts.
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

    validate_dataset_split(
        dataset
    )

    # ------------------------------------------------------------------
    # 8. Report split sizes.
    # ------------------------------------------------------------------

    print(
        "\nSplit sizes:"
    )

    for split_name in (
        "train",
        "eval",
        "test",
    ):

        split = dataset[
            split_name
        ]

        n_pairs = len(
            split[
                "labels"
            ]
        )

        n_texts = len(
            split[
                "data"
            ]
        )

        n_topics = len(
            set(
                split[
                    "topics"
                ]
            )
        )

        print(
            f"  {split_name:<5}: "
            f"{n_pairs:>5} pairs | "
            f"{n_texts:>5} texts | "
            f"{n_topics:>2} topics"
        )

    # ------------------------------------------------------------------
    # 9. Train confidence directions.
    # ------------------------------------------------------------------

    print(
        "\nFitting INKER Appendix-B-style "
        "confidence representation detector..."
    )

    print(
        f"  Layers: {layers}"
    )

    print(
        f"  n_difference: {n_difference}"
    )

    print(
        f"  batch_size: {batch_size}"
    )

    print(
        f"  rep_token: {rep_token}"
    )

    print(
        f"  max_length: {max_length}"
    )

    rep_reader = get_directions(
        train_texts=dataset[
            "train"
        ][
            "data"
        ],

        train_labels=dataset[
            "train"
        ][
            "labels"
        ],

        tokenizer=tokenizer,
        model=model,
        layers=layers,
        n_difference=n_difference,
        batch_size=batch_size,
        rep_token=rep_token,
        max_length=max_length,
    )

    # ------------------------------------------------------------------
    # 10. Validate representation reader.
    # ------------------------------------------------------------------

    validate_representation_reader(
        rep_reader=rep_reader,
        layers=layers,
    )

    # ------------------------------------------------------------------
    # 11. Save representation reader.
    # ------------------------------------------------------------------

    with reader_path.open(
        "wb"
    ) as file:

        pickle.dump(
            rep_reader,
            file,
        )

    # ------------------------------------------------------------------
    # 12. Save training metadata.
    #
    # This gives us an explicit record of the configuration that produced
    # the representation reader.
    #
    # It is useful later when checking replication reproducibility.
    # ------------------------------------------------------------------

    metadata_path = (
        reader_path.parent
        / (
            reader_path.stem
            + "_metadata.json"
        )
    )

    metadata = {
        "seed":
            seed,

        "dataset_path":
            str(
                dataset_path
            ),

        "representation_reader":
            str(
                reader_path
            ),

        "detector_layers":
            layers,

        "n_difference":
            n_difference,

        "batch_size":
            batch_size,

        "rep_token":
            rep_token,

        "max_length":
            max_length,

        "split": {
            "train_ratio":
                train_ratio,

            "eval_ratio":
                eval_ratio,

            "test_ratio":
                test_ratio,
        },

        "dataset_statistics": {
            "num_topics":
                len(
                    topics
                ),

            "total_pairs":
                n_pairs_total,

            "train_pairs":
                len(
                    dataset[
                        "train"
                    ][
                        "labels"
                    ]
                ),

            "eval_pairs":
                len(
                    dataset[
                        "eval"
                    ][
                        "labels"
                    ]
                ),

            "test_pairs":
                len(
                    dataset[
                        "test"
                    ][
                        "labels"
                    ]
                ),

            "train_topics":
                len(
                    set(
                        dataset[
                            "train"
                        ][
                            "topics"
                        ]
                    )
                ),

            "eval_topics":
                len(
                    set(
                        dataset[
                            "eval"
                        ][
                            "topics"
                        ]
                    )
                ),

            "test_topics":
                len(
                    set(
                        dataset[
                            "test"
                        ][
                            "topics"
                        ]
                    )
                ),
        },
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # 13. Final summary.
    # ------------------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CONFIDENCE DETECTOR TRAINING COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "\nRepresentation reader:"
    )

    print(
        f"  {reader_path}"
    )

    print(
        "\nTraining metadata:"
    )

    print(
        f"  {metadata_path}"
    )

    print(
        "\nThe representation reader can now be used by:"
    )

    print(
        "  - evaluate_detector.py"
    )

    print(
        "  - run_confidence_only.py"
    )

    print(
        "  - run_inker_trigger.py"
    )

    print(
        "  - run_test_suite.py"
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
            "Train the INKER internal confidence representation detector."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML configuration path. "
            "Defaults to configs/default.yaml."
        ),
    )

    args = parser.parse_args()

    main(
        args.config
    )
