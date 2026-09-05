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
    n_train=512,
):
    """
    Reproduce the original training/evaluation/test split exactly.

    NOTE:
    The shifted eval/test construction is intentionally unusual:
        honest_statements[:-1] is paired with untruthful_statements[1:].

    It is preserved because the goal is faithful replication rather than
    redesigning the experiment.
    """
    combined_data = [
        [h, u]
        for h, u in zip(honest_statements, untruthful_statements)
    ]

    random.shuffle(combined_data)
    train_data = combined_data[:n_train]

    train_labels = []

    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])

    train_data = np.concatenate(train_data).tolist()

    reshaped_data = np.array([
        [h, u]
        for h, u in zip(
            honest_statements[:-1],
            untruthful_statements[1:],
        )
    ]).flatten()

    eval_data = reshaped_data[n_train:n_train * 2].tolist()
    test_data = reshaped_data[-300:-1].tolist()

    n_reshaped_pairs = len(honest_statements) - 1
    reshaped_pair_topics = pair_topics[:n_reshaped_pairs]

    eval_topics = reshaped_pair_topics[n_train // 2:n_train]
    test_topics = reshaped_pair_topics[-150:-1]

    print(f"Train data: {len(train_data)}")
    print(f"Eval data: {len(eval_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        "train": {
            "data": train_data,
            "labels": train_labels,
        },
        # These fields are preserved for compatibility with the original
        # notebook, although evaluate() constructs its labels directly.
        "eval": {
            "data": eval_data,
            "labels": [[1, 0]] * len(eval_data),
            "topics": eval_topics,
        },
        "test": {
            "data": test_data,
            "labels": [[1, 0]] * len(test_data),
            "topics": test_topics,
        },
    }
