import argparse
import os
import pickle
import random
import numpy as np
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
from inker.directions import get_directions


def main(config_path=None):
    config = get_config(config_path)

    seed = config["experiment"]["seed"]

    # The original dataset builder uses Python random.
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

    os.makedirs(
        os.path.dirname(reader_path),
        exist_ok=True,
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

    print(
        f"Topics: {len(topics)} | "
        f"Total pairs before split: "
        f"{len(honest_statements)}"
    )

    dataset = make_split(
        honest_statements,
        untruthful_statements,
        pair_topics,
        n_train=config["detector"]["n_train"],
    )

    print(
        "Fitting INKER-recipe "
        "(Appendix B) confidence detector..."
    )

    rep_reader = get_directions(
        train_texts=dataset["train"]["data"],
        train_labels=dataset["train"]["labels"],
        tokenizer=tokenizer,
        model=model,
        layers=detector_layers(config),
        n_difference=config[
            "detector"
        ]["n_difference"],
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

    with open(reader_path, "wb") as f:
        pickle.dump(rep_reader, f)

    print(
        f"Confidence detector trained successfully.\n"
        f"Saved rep_reader to:\n{reader_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
    )
    args = parser.parse_args()
    main(args.config)
