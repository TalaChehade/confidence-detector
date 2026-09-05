import json
import random
import numpy as np


USER_TAG = "[INST]"
ASSISTANT_TAG = "[/INST]"
TEMPLATE_STR = "Pretend you're {type} person making statements about the world."
POS_TAG = "a confident"
NEG_TAG = "an unconfident"


def build_inker_pairs(
    statements_path,
    tokenizer,
    user_tag=USER_TAG,
    assistant_tag=ASSISTANT_TAG,
    pos_tag=POS_TAG,
    neg_tag=NEG_TAG,
    seed=0,
):
    """
    Build the confident/unconfident truncated statement pairs.

    Algorithmically identical to the original Colab implementation.
    """
    random.seed(seed)

    with open(statements_path, "r", encoding="windows-1252") as f:
        data = json.loads(f.read())

    honest_statements = []
    untruthful_statements = []
    pair_topics = []

    topics = list(data["confident"].keys())

    for topic in topics:
        conf_list = data["confident"][topic]
        unconf_list = data["unconfident"][topic]

        for c_stmt, u_stmt in zip(conf_list, unconf_list):
            c_tokens = tokenizer.tokenize(c_stmt)
            u_tokens = tokenizer.tokenize(u_stmt)

            c_truncations = [
                tokenizer.convert_tokens_to_string(c_tokens[:idx])
                for idx in range(1, len(c_tokens) - 5)
            ]
            u_truncations = [
                tokenizer.convert_tokens_to_string(u_tokens[:idx])
                for idx in range(1, len(u_tokens) - 5)
            ]

            for c_trunc, u_trunc in zip(c_truncations, u_truncations):
                honest_statements.append(
                    f"{user_tag} {TEMPLATE_STR.format(type=pos_tag)} "
                    f"{assistant_tag} {c_trunc}"
                )
                untruthful_statements.append(
                    f"{user_tag} {TEMPLATE_STR.format(type=neg_tag)} "
                    f"{assistant_tag} {u_trunc}"
                )
                pair_topics.append(topic)

    return honest_statements, untruthful_statements, topics, pair_topics


def make_split(
    honest_statements,
    untruthful_statements,
    pair_topics,
    train_ratio=0.70,
    eval_ratio=0.15,
    test_ratio=0.15,
    seed=0,
):
    """
    Topic-stratified split.

    Ensures that train, eval, and test each receive confident/unconfident
    pairs from every topic, as long as the topic has enough examples.

    Each pair is kept intact:
        (confident_i, unconfident_i)

    The order of the two statements inside training pairs is randomized,
    exactly as in the original training procedure.

    Eval/test pairs remain ordered as:
        [confident, unconfident]

    so evaluation can use labels:
        [1, 0]
    """

    import random
    import numpy as np
    from collections import defaultdict

    if abs(
        train_ratio + eval_ratio + test_ratio - 1.0
    ) > 1e-8:
        raise ValueError(
            "train_ratio + eval_ratio + test_ratio must equal 1.0"
        )

    rng = random.Random(seed)

    # ---------------------------------------------------------
    # 1. Group complete confident/unconfident pairs by topic
    # ---------------------------------------------------------

    topic_pairs = defaultdict(list)

    for honest, untruthful, topic in zip(
        honest_statements,
        untruthful_statements,
        pair_topics,
    ):
        topic_pairs[topic].append(
            [honest, untruthful]
        )

    # ---------------------------------------------------------
    # 2. Containers for final splits
    # ---------------------------------------------------------

    train_pairs = []
    train_labels = []
    train_topics = []

    eval_pairs = []
    eval_topics = []

    test_pairs = []
    test_topics = []

    # ---------------------------------------------------------
    # 3. Split EACH topic independently
    # ---------------------------------------------------------

    for topic, pairs in topic_pairs.items():

        pairs = pairs.copy()
        rng.shuffle(pairs)

        n = len(pairs)

        # Require enough examples to put at least one pair
        # in every split.
        if n < 3:
            raise ValueError(
                f"Topic '{topic}' only has {n} pairs. "
                "At least 3 are required for train/eval/test."
            )

        n_train = int(
            round(n * train_ratio)
        )

        n_eval = int(
            round(n * eval_ratio)
        )

        # Make sure all three splits contain at least one pair.
        n_train = max(
            1,
            min(n_train, n - 2),
        )

        n_eval = max(
            1,
            min(n_eval, n - n_train - 1),
        )

        n_test = (
            n
            - n_train
            - n_eval
        )

        # -----------------------------------------------------
        # Topic-specific slices
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # TRAIN
        #
        # Preserve the original behavior:
        # randomly shuffle the two members of each pair and
        # remember which one is the confident statement.
        # -----------------------------------------------------

        for pair in topic_train:

            pair = pair.copy()

            confident_statement = pair[0]

            rng.shuffle(pair)

            labels = [
                statement == confident_statement
                for statement in pair
            ]

            train_pairs.append(
                pair
            )

            train_labels.append(
                labels
            )

            train_topics.append(
                topic
            )

        # -----------------------------------------------------
        # EVAL
        #
        # Keep:
        #     [confident, unconfident]
        # -----------------------------------------------------

        for pair in topic_eval:

            eval_pairs.append(
                pair
            )

            eval_topics.append(
                topic
            )

        # -----------------------------------------------------
        # TEST
        # -----------------------------------------------------

        for pair in topic_test:

            test_pairs.append(
                pair
            )

            test_topics.append(
                topic
            )

    # ---------------------------------------------------------
    # 4. Shuffle pairs ACROSS topics
    #
    # Important:
    # shuffle at pair level, not text level.
    # Otherwise confident/unconfident pairs would be broken.
    # ---------------------------------------------------------

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
        x[0]
        for x in train_combined
    ]

    train_labels = [
        x[1]
        for x in train_combined
    ]

    train_topics = [
        x[2]
        for x in train_combined
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
        x[0]
        for x in eval_combined
    ]

    eval_topics = [
        x[1]
        for x in eval_combined
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
        x[0]
        for x in test_combined
    ]

    test_topics = [
        x[1]
        for x in test_combined
    ]

    # ---------------------------------------------------------
    # 5. Flatten pairs
    #
    # The detector expects:
    #
    # [text1, text2, text1, text2, ...]
    # ---------------------------------------------------------

    train_data = np.concatenate(
        train_pairs
    ).tolist()

    eval_data = np.concatenate(
        eval_pairs
    ).tolist()

    test_data = np.concatenate(
        test_pairs
    ).tolist()

    # ---------------------------------------------------------
    # 6. Labels for eval/test
    #
    # Since eval/test pair order is:
    #     [confident, unconfident]
    #
    # labels are:
    #     [1, 0]
    # ---------------------------------------------------------

    eval_labels = [
        [1, 0]
        for _ in eval_pairs
    ]

    test_labels = [
        [1, 0]
        for _ in test_pairs
    ]

    # ---------------------------------------------------------
    # 7. Diagnostics
    # ---------------------------------------------------------

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
        f"Train topics: "
        f"{len(set(train_topics))}"
    )

    print(
        f"Eval topics: "
        f"{len(set(eval_topics))}"
    )

    print(
        f"Test topics: "
        f"{len(set(test_topics))}"
    )

    return {

        "train": {
            "data":
                train_data,

            "labels":
                train_labels,

            "topics":
                train_topics,
        },

        "eval": {
            "data":
                eval_data,

            "labels":
                eval_labels,

            "topics":
                eval_topics,
        },

        "test": {
            "data":
                test_data,

            "labels":
                test_labels,

            "topics":
                test_topics,
        },
    }
