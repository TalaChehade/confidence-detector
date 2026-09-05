import argparse
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch

from _common import (
    get_config,
    load_configured_model,
    resolve_project_path,
    detector_layers,
)

from inker.dataset import (
    build_inker_pairs,
    make_split,
)

from inker.scoring import (
    evaluate,
    per_topic_breakdown,
)


def main(config_path=None):
    config = get_config(config_path)

    seed = config["experiment"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_path = resolve_project_path(
        config,
        "dataset",
    )
    reader_path = resolve_project_path(
        config,
        "representation_reader",
    )
    result_dir = resolve_project_path(
        config,
        "replication_results",
    )

    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    if not os.path.exists(reader_path):
        raise FileNotFoundError(
            f"Representation reader not found: "
            f"{reader_path}\n"
            "Run train_detector.py first."
        )

    tokenizer, model = (
        load_configured_model(config)
    )

    honest_statements, untruthful_statements, topics, pair_topics = (
        build_inker_pairs(
            dataset_path,
            tokenizer,
            seed=seed,
        )
    )

    dataset = make_split(
    honest_statements,
    untruthful_statements,
    pair_topics,
    train_ratio=0.70,
    eval_ratio=0.15,
    test_ratio=0.15,
    seed=0,
)

    with open(reader_path, "rb") as f:
        rep_reader = pickle.load(f)

    common = dict(
        rep_reader=rep_reader,
        tokenizer=tokenizer,
        model=model,
        layers=detector_layers(config),
        batch_size=config[
            "detector"
        ]["batch_size"],
        rep_token=config[
            "detector"
        ]["rep_token"],
        max_length=config[
            "detector"
        ]["max_length"],
    )

    eval_result = evaluate(
        "Eval",
        dataset["eval"]["data"],
        **common,
    )

    test_result = evaluate(
        "Test",
        dataset["test"]["data"],
        **common,
    )

    metrics_df = pd.DataFrame([
        {
            "split": "eval",
            "roc_auc": eval_result["auc"],
            "pairwise_accuracy":
                eval_result[
                    "pairwise_accuracy"
                ],
        },
        {
            "split": "test",
            "roc_auc": test_result["auc"],
            "pairwise_accuracy":
                test_result[
                    "pairwise_accuracy"
                ],
        },
    ])

    metrics_df.to_csv(
        os.path.join(
            result_dir,
            "replication_metrics.csv",
        ),
        index=False,
    )

    eval_topic_df = per_topic_breakdown(
        "Eval",
        eval_result["pair_scores"],
        dataset["eval"]["topics"],
    )

    test_topic_df = per_topic_breakdown(
        "Test",
        test_result["pair_scores"],
        dataset["test"]["topics"],
    )

    eval_topic_df.to_csv(
        os.path.join(
            result_dir,
            "eval_per_topic.csv",
        ),
        index=False,
    )

    test_topic_df.to_csv(
        os.path.join(
            result_dir,
            "test_per_topic.csv",
        ),
        index=False,
    )

    print(
        f"\nSaved evaluation results to:\n"
        f"{result_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
    )
    args = parser.parse_args()
    main(args.config)
