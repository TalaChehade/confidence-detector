"""
Dataset preparation utilities for the INKER confidence detector.

This module builds the contrastive confident/unconfident statement pairs used
to learn the internal confidence representation direction.

It is responsible only for the CONFIDENCE-DETECTOR training dataset.

Dataset format
--------------
The input JSON file is expected to contain two top-level dictionaries:

    {
        "confident": {
            "topic_1": [...],
            "topic_2": [...],
            ...
        },

        "unconfident": {
            "topic_1": [...],
            "topic_2": [...],
            ...
        }
    }

For each topic, confident statement i is paired with unconfident statement i:

    confident[i]  <->  unconfident[i]

Each statement is tokenized and converted into multiple prefix truncations.
Matching confident/unconfident truncations are then wrapped with the INKER
instruction templates:

    [INST] Pretend you're a confident person making statements about the world.
    [/INST] <confident prefix>

and:

    [INST] Pretend you're an unconfident person making statements about the world.
    [/INST] <unconfident prefix>

These contrastive pairs are later used to learn the confidence representation
direction from hidden states.

Splitting strategy
------------------
Pairs are split independently WITHIN every topic.

This ensures that train, evaluation, and test sets contain examples from every
topic, provided that each topic contains at least three generated pairs.

Pairs are never broken across splits.

Training pairs:
    The two members of each pair are randomly reordered. A corresponding
    pair-level Boolean label records which member is the confident example.

Evaluation/test pairs:
    The order is kept fixed as:

        [confident, unconfident]

    with labels:

        [1, 0]

This makes pairwise evaluation straightforward.

Important
---------
The returned training ``data`` is flattened:

    [pair0_text0, pair0_text1, pair1_text0, pair1_text1, ...]

while ``labels`` remains pair-level:

    [[True, False], [False, True], ...]

This is intentional because the direction-learning stage operates on
contrastive pairs after hidden-state extraction.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


# ---------------------------------------------------------------------------
# Prompt template used to construct confident/unconfident examples.
# ---------------------------------------------------------------------------

USER_TAG = "[INST]"
ASSISTANT_TAG = "[/INST]"

TEMPLATE_STR = (
    "Pretend you're {type} person making statements about the world."
)

POS_TAG = "a confident"
NEG_TAG = "an unconfident"


def _load_statements_json(
    statements_path: str | Path,
) -> Dict[str, Any]:
    """
    Load the confident/unconfident statement JSON file.

    UTF-8 is attempted first. Windows-1252 is used as a fallback because
    some versions of the original generated dataset were stored using that
    encoding.

    Parameters
    ----------
    statements_path:
        Path to the JSON statement dataset.

    Returns
    -------
    dict
        Parsed JSON dictionary.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.

    ValueError
        If the required ``confident`` or ``unconfident`` sections are absent.
    """

    statements_path = Path(statements_path)

    if not statements_path.exists():
        raise FileNotFoundError(
            f"Statement dataset not found: {statements_path}"
        )

    try:
        with statements_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except UnicodeDecodeError:
        with statements_path.open(
            "r",
            encoding="windows-1252",
        ) as file:
            data = json.load(file)

    if "confident" not in data:
        raise ValueError(
            "Dataset is missing the top-level 'confident' section."
        )

    if "unconfident" not in data:
        raise ValueError(
            "Dataset is missing the top-level 'unconfident' section."
        )

    return data


def build_inker_pairs(
    statements_path: str | Path,
    tokenizer: Any,
    user_tag: str = USER_TAG,
    assistant_tag: str = ASSISTANT_TAG,
    pos_tag: str = POS_TAG,
    neg_tag: str = NEG_TAG,
    seed: int = 0,
) -> Tuple[
    List[str],
    List[str],
    List[str],
    List[str],
]:
    """
    Build truncated confident/unconfident contrastive statement pairs.

    For every topic:

    1. Pair confident statement i with unconfident statement i.
    2. Tokenize both statements.
    3. Produce progressively longer prefixes.
    4. Pair confident and unconfident prefixes with the same truncation index.
    5. Wrap each prefix with its corresponding confidence instruction.
    6. Record the topic associated with each generated pair.

    The final five tokens of each original statement are excluded from the
    truncation range, matching the previous INKER replication procedure.

    Parameters
    ----------
    statements_path:
        Path to the JSON file containing the confident and unconfident
        statements grouped by topic.

    tokenizer:
        Hugging Face tokenizer used by the target language model.

    user_tag:
        Opening instruction tag.

    assistant_tag:
        Closing instruction tag.

    pos_tag:
        Text inserted into the template for confident examples.

    neg_tag:
        Text inserted into the template for unconfident examples.

    seed:
        Reserved random seed for reproducibility.

        The current pair-construction procedure itself is deterministic, but
        the argument is retained for compatibility with the previous pipeline
        and future sampling extensions.

    Returns
    -------
    confident_statements:
        Flattened list of truncated confident prompts.

    unconfident_statements:
        Flattened list of corresponding truncated unconfident prompts.

    topics:
        List of all topic names found in the dataset.

    pair_topics:
        Topic corresponding to each confident/unconfident pair.

    Raises
    ------
    ValueError
        If a topic is missing from one side of the dataset.
    """

    random.seed(seed)

    data = _load_statements_json(
        statements_path
    )

    confident_statements: List[str] = []
    unconfident_statements: List[str] = []
    pair_topics: List[str] = []

    topics = list(
        data["confident"].keys()
    )

    for topic in topics:

        if topic not in data["unconfident"]:
            raise ValueError(
                f"Topic '{topic}' exists in 'confident' but not "
                "in 'unconfident'."
            )

        conf_list = data["confident"][topic]
        unconf_list = data["unconfident"][topic]

        if len(conf_list) != len(unconf_list):
            raise ValueError(
                f"Topic '{topic}' has {len(conf_list)} confident "
                f"statements but {len(unconf_list)} unconfident statements. "
                "The two lists must have equal length."
            )

        for confident_statement, unconfident_statement in zip(
            conf_list,
            unconf_list,
        ):
            confident_tokens = tokenizer.tokenize(
                confident_statement
            )

            unconfident_tokens = tokenizer.tokenize(
                unconfident_statement
            )

            # -----------------------------------------------------------
            # Build progressively longer statement prefixes.
            #
            # Example:
            #
            # tokens = [t1, t2, t3, ..., tn]
            #
            # produces:
            #
            # [t1]
            # [t1, t2]
            # [t1, t2, t3]
            # ...
            #
            # while excluding the last five token positions.
            # -----------------------------------------------------------

            confident_truncations = [
                tokenizer.convert_tokens_to_string(
                    confident_tokens[:idx]
                )
                for idx in range(
                    1,
                    len(confident_tokens) - 5,
                )
            ]

            unconfident_truncations = [
                tokenizer.convert_tokens_to_string(
                    unconfident_tokens[:idx]
                )
                for idx in range(
                    1,
                    len(unconfident_tokens) - 5,
                )
            ]

            # -----------------------------------------------------------
            # zip() intentionally keeps only truncation positions that
            # exist on BOTH sides of the confident/unconfident pair.
            #
            # This prevents a longer statement from contributing unmatched
            # prefixes.
            # -----------------------------------------------------------

            for confident_prefix, unconfident_prefix in zip(
                confident_truncations,
                unconfident_truncations,
            ):
                confident_prompt = (
                    f"{user_tag} "
                    f"{TEMPLATE_STR.format(type=pos_tag)} "
                    f"{assistant_tag} "
                    f"{confident_prefix}"
                )

                unconfident_prompt = (
                    f"{user_tag} "
                    f"{TEMPLATE_STR.format(type=neg_tag)} "
                    f"{assistant_tag} "
                    f"{unconfident_prefix}"
                )

                confident_statements.append(
                    confident_prompt
                )

                unconfident_statements.append(
                    unconfident_prompt
                )

                pair_topics.append(
                    topic
                )

    return (
        confident_statements,
        unconfident_statements,
        topics,
        pair_topics,
    )


def _flatten_pairs(
    pairs: Sequence[Sequence[str]],
) -> List[str]:
    """
    Flatten a list of two-element statement pairs.

    Example
    -------
    Input:

        [
            ["conf_1", "unconf_1"],
            ["conf_2", "unconf_2"],
        ]

    Output:

        [
            "conf_1",
            "unconf_1",
            "conf_2",
            "unconf_2",
        ]
    """

    return [
        statement
        for pair in pairs
        for statement in pair
    ]


def make_split(
    honest_statements: Sequence[str],
    untruthful_statements: Sequence[str],
    pair_topics: Sequence[str],
    train_ratio: float = 0.70,
    eval_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """
    Create topic-stratified train, evaluation, and test splits.

    Each confident/unconfident pair remains intact throughout the split.

    Every topic is split independently, ensuring that train, evaluation,
    and test sets all contain examples from every topic whenever at least
    three pairs are available for that topic.

    Parameters
    ----------
    honest_statements:
        Confident prompts.

        The name is retained for compatibility with the existing repository.
        Conceptually, these are ``confident_statements``.

    untruthful_statements:
        Unconfident prompts.

        The name is retained for compatibility with the existing repository.
        Conceptually, these are ``unconfident_statements``.

    pair_topics:
        Topic associated with each confident/unconfident pair.

    train_ratio:
        Fraction of each topic assigned to training.

    eval_ratio:
        Fraction of each topic assigned to evaluation.

    test_ratio:
        Fraction of each topic assigned to testing.

    seed:
        Random seed used for all pair-level shuffling.

    Returns
    -------
    dict
        Dictionary containing ``train``, ``eval``, and ``test`` splits.

        Training format:

            {
                "data": [
                    text_0,
                    text_1,
                    text_2,
                    text_3,
                    ...
                ],

                "labels": [
                    [True, False],
                    [False, True],
                    ...
                ],

                "topics": [
                    topic_for_pair_0,
                    topic_for_pair_1,
                    ...
                ]
            }

        Evaluation/test format:

            data:
                flattened [confident, unconfident] pairs

            labels:
                [[1, 0], [1, 0], ...]

            topics:
                one topic entry per pair

    Raises
    ------
    ValueError
        If:
        - split ratios do not sum to one,
        - input lengths do not match,
        - or a topic contains fewer than three pairs.
    """

    # ------------------------------------------------------------------
    # 1. Validate arguments.
    # ------------------------------------------------------------------

    if abs(
        train_ratio
        + eval_ratio
        + test_ratio
        - 1.0
    ) > 1e-8:
        raise ValueError(
            "train_ratio + eval_ratio + test_ratio must equal 1.0."
        )

    if not (
        len(honest_statements)
        == len(untruthful_statements)
        == len(pair_topics)
    ):
        raise ValueError(
            "honest_statements, untruthful_statements, and pair_topics "
            "must contain the same number of elements."
        )

    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 2. Group COMPLETE confident/unconfident pairs by topic.
    #
    # No statement is split independently.
    # ------------------------------------------------------------------

    topic_pairs: Dict[str, List[List[str]]] = defaultdict(
        list
    )

    for confident, unconfident, topic in zip(
        honest_statements,
        untruthful_statements,
        pair_topics,
    ):
        topic_pairs[topic].append(
            [
                confident,
                unconfident,
            ]
        )

    # ------------------------------------------------------------------
    # 3. Containers for the final splits.
    # ------------------------------------------------------------------

    train_pairs: List[List[str]] = []
    train_labels: List[List[bool]] = []
    train_topics: List[str] = []

    eval_pairs: List[List[str]] = []
    eval_topics: List[str] = []

    test_pairs: List[List[str]] = []
    test_topics: List[str] = []

    # ------------------------------------------------------------------
    # 4. Split EACH topic independently.
    # ------------------------------------------------------------------

    for topic, pairs in topic_pairs.items():

        pairs = pairs.copy()

        rng.shuffle(
            pairs
        )

        n_pairs = len(
            pairs
        )

        # We require at least one pair in train, eval, and test.
        if n_pairs < 3:
            raise ValueError(
                f"Topic '{topic}' contains only {n_pairs} pairs. "
                "At least 3 pairs are required to create "
                "train/eval/test splits."
            )

        n_train = int(
            round(
                n_pairs * train_ratio
            )
        )

        n_eval = int(
            round(
                n_pairs * eval_ratio
            )
        )

        # --------------------------------------------------------------
        # Guarantee at least one pair remains for each split.
        # --------------------------------------------------------------

        n_train = max(
            1,
            min(
                n_train,
                n_pairs - 2,
            ),
        )

        n_eval = max(
            1,
            min(
                n_eval,
                n_pairs - n_train - 1,
            ),
        )

        n_test = (
            n_pairs
            - n_train
            - n_eval
        )

        topic_train = pairs[
            :n_train
        ]

        topic_eval = pairs[
            n_train:
            n_train + n_eval
        ]

        topic_test = pairs[
            n_train + n_eval:
        ]

        # --------------------------------------------------------------
        # TRAIN
        #
        # Each original pair starts as:
        #
        #     [confident, unconfident]
        #
        # We randomly reorder the two members and record which position
        # contains the confident statement.
        #
        # Example:
        #
        # shuffled:
        #     [unconfident, confident]
        #
        # labels:
        #     [False, True]
        #
        # This prevents the confidence direction learner from exploiting
        # a fixed positional ordering.
        # --------------------------------------------------------------

        for pair in topic_train:

            shuffled_pair = pair.copy()

            confident_statement = pair[0]

            rng.shuffle(
                shuffled_pair
            )

            labels = [
                statement == confident_statement
                for statement in shuffled_pair
            ]

            train_pairs.append(
                shuffled_pair
            )

            train_labels.append(
                labels
            )

            train_topics.append(
                topic
            )

        # --------------------------------------------------------------
        # EVALUATION
        #
        # Keep deterministic ordering:
        #
        #     [confident, unconfident]
        #
        # so the corresponding label is:
        #
        #     [1, 0]
        # --------------------------------------------------------------

        for pair in topic_eval:

            eval_pairs.append(
                pair
            )

            eval_topics.append(
                topic
            )

        # --------------------------------------------------------------
        # TEST
        #
        # Same deterministic pair ordering as evaluation.
        # --------------------------------------------------------------

        for pair in topic_test:

            test_pairs.append(
                pair
            )

            test_topics.append(
                topic
            )

    # ------------------------------------------------------------------
    # 5. Shuffle complete pairs ACROSS topics.
    #
    # We shuffle at pair level rather than text level so confident and
    # unconfident members are never separated.
    # ------------------------------------------------------------------

    train_combined = list(
        zip(
            train_pairs,
            train_labels,
            train_topics,
        )
    )

    rng.shuffle(
        train_combined
    )

    train_pairs = [
        pair
        for pair, _, _ in train_combined
    ]

    train_labels = [
        labels
        for _, labels, _ in train_combined
    ]

    train_topics = [
        topic
        for _, _, topic in train_combined
    ]

    eval_combined = list(
        zip(
            eval_pairs,
            eval_topics,
        )
    )

    rng.shuffle(
        eval_combined
    )

    eval_pairs = [
        pair
        for pair, _ in eval_combined
    ]

    eval_topics = [
        topic
        for _, topic in eval_combined
    ]

    test_combined = list(
        zip(
            test_pairs,
            test_topics,
        )
    )

    rng.shuffle(
        test_combined
    )

    test_pairs = [
        pair
        for pair, _ in test_combined
    ]

    test_topics = [
        topic
        for _, topic in test_combined
    ]

    # ------------------------------------------------------------------
    # 6. Flatten the text pairs.
    #
    # Example:
    #
    # [
    #     [text_1, text_2],
    #     [text_3, text_4],
    # ]
    #
    # becomes:
    #
    # [
    #     text_1,
    #     text_2,
    #     text_3,
    #     text_4,
    # ]
    #
    # The hidden-state extraction code expects this flattened format.
    # ------------------------------------------------------------------

    train_data = _flatten_pairs(
        train_pairs
    )

    eval_data = _flatten_pairs(
        eval_pairs
    )

    test_data = _flatten_pairs(
        test_pairs
    )

    # ------------------------------------------------------------------
    # 7. Evaluation/test labels.
    #
    # Because every eval/test pair is ordered:
    #
    #     [confident, unconfident]
    #
    # the labels are always:
    #
    #     [1, 0]
    # ------------------------------------------------------------------

    eval_labels = [
        [1, 0]
        for _ in eval_pairs
    ]

    test_labels = [
        [1, 0]
        for _ in test_pairs
    ]

    # ------------------------------------------------------------------
    # 8. Diagnostics.
    # ------------------------------------------------------------------

    print(
        f"Topics: {len(topic_pairs)}"
    )

    print(
        f"Train pairs: {len(train_pairs)} | "
        f"texts: {len(train_data)}"
    )

    print(
        f"Eval pairs: {len(eval_pairs)} | "
        f"texts: {len(eval_data)}"
    )

    print(
        f"Test pairs: {len(test_pairs)} | "
        f"texts: {len(test_data)}"
    )

    print(
        f"Train topics: {len(set(train_topics))}"
    )

    print(
        f"Eval topics: {len(set(eval_topics))}"
    )

    print(
        f"Test topics: {len(set(test_topics))}"
    )

    # Sanity check: every topic should appear in all three splits.
    expected_topics = set(
        topic_pairs.keys()
    )

    if set(train_topics) != expected_topics:
        raise RuntimeError(
            "Training split does not contain every dataset topic."
        )

    if set(eval_topics) != expected_topics:
        raise RuntimeError(
            "Evaluation split does not contain every dataset topic."
        )

    if set(test_topics) != expected_topics:
        raise RuntimeError(
            "Test split does not contain every dataset topic."
        )

    # ------------------------------------------------------------------
    # 9. Return the existing repository interface unchanged.
    # ------------------------------------------------------------------

    return {
        "train": {
            "data": train_data,
            "labels": train_labels,
            "topics": train_topics,
        },

        "eval": {
            "data": eval_data,
            "labels": eval_labels,
            "topics": eval_topics,
        },

        "test": {
            "data": test_data,
            "labels": test_labels,
            "topics": test_topics,
        },
    }
