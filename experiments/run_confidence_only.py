import argparse
import os
import pickle
import pandas as pd

from _common import (
    get_config,
    load_configured_model,
    resolve_project_path,
    detector_layers,
)

from inker.generation import (
    answer_with_confidence_only,
)


QUESTIONS = [
    {
        "question":
            "What is the capital of France?",
        "expected_answer": "Paris",
    },
    {
        "question":
            "Who wrote Romeo and Juliet?",
        "expected_answer":
            "William Shakespeare",
    },
    {
        "question":
            "What is the largest planet "
            "in the Solar System?",
        "expected_answer": "Jupiter",
    },
]


def main(config_path=None):
    config = get_config(config_path)

    reader_path = resolve_project_path(
        config,
        "representation_reader",
    )
    result_dir = resolve_project_path(
        config,
        "confidence_only_results",
    )

    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    with open(reader_path, "rb") as f:
        rep_reader = pickle.load(f)

    tokenizer, model = (
        load_configured_model(config)
    )

    threshold = config[
        "confidence"
    ]["threshold"]

    summary_rows = []
    token_rows = []

    for question_id, case in enumerate(
        QUESTIONS
    ):
        record = answer_with_confidence_only(
            question=case["question"],
            expected_answer=case.get(
                "expected_answer"
            ),
            tokenizer=tokenizer,
            model=model,
            rep_reader=rep_reader,
            layers=detector_layers(config),
            threshold=threshold,
            max_new_tokens=config[
                "generation"
            ]["max_new_tokens"],
            repetition_penalty=config[
                "generation"
            ]["repetition_penalty"],
        )

        summary_rows.append({
            "question_id": question_id,
            "question": record["question"],
            "answer":
                record["answer_text"],
            "expected_answer":
                record["expected_answer"],
            "threshold":
                record["threshold"],
            "mean_confidence":
                record["mean_confidence"],
            "min_confidence":
                record["min_confidence"],
            "max_confidence":
                record["max_confidence"],
            "num_content_tokens":
                record["num_content_tokens"],
            "num_low_confidence_tokens":
                record[
                    "num_low_confidence_tokens"
                ],
            "would_trigger":
                record["would_trigger"],
        })

        for entry in record[
            "token_entries"
        ]:
            token_rows.append({
                "question_id": question_id,
                "question":
                    record["question"],
                "answer":
                    record["answer_text"],
                "token_index":
                    entry["token_index"],
                "token": entry["token"],
                "raw_score":
                    entry["raw_score"],
                "m_tilde":
                    entry.get("m_tilde"),
                "confidence":
                    entry.get("confidence"),
                "status":
                    entry.get("status"),
                "is_content":
                    entry["is_content"],
            })

    pd.DataFrame(
        summary_rows
    ).to_csv(
        os.path.join(
            result_dir,
            "confidence_only_questions.csv",
        ),
        index=False,
    )

    pd.DataFrame(
        token_rows
    ).to_csv(
        os.path.join(
            result_dir,
            "confidence_only_tokens.csv",
        ),
        index=False,
    )

    print(
        f"Saved confidence-only results to:\n"
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
