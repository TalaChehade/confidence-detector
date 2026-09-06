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

from inker.complexity import load_complexity_evaluator

from inker.generation import (
    answer_with_confidence,
)


TEST_SUITE = [
    {
        "question":
            "In what city is the company that Fastjet Tanzania "
            "was originally founded as a part of prior to "
            "rebranding based?",
        "category": "paper_case_study",
        "expected_answer": "Nairobi",
    },
    {
        "question":
            "Stephen Smith appears on ESPN First Take alongside "
            "which HBO boxing commentator?",
        "category": "paper_case_study",
        "expected_answer": "Kellerman",
    },
    {
        "question":
            "What is the capital of France?",
        "category": "high_conf_correct",
        "expected_answer": "Paris",
    },
    {
        "question":
            "What is the chemical symbol for water?",
        "category": "high_conf_correct",
        "expected_answer": "H2O",
    },
    {
        "question":
            "Who played the villain in the original 1984 "
            "Terminator movie?",
        "category": "high_conf_wrong_target",
        "expected_answer": None,
    },
    {
        "question":
            "Which HBO boxing analyst co-hosted ESPN's "
            "First Take with Stephen A. Smith?",
        "category": "high_conf_wrong_target",
        "expected_answer": "Kellerman",
    },
    {
        "question":
            "How many senses do humans have?",
        "category": "high_conf_wrong_target",
        "expected_answer": None,
    },
    {
        "question":
            "Does the oldest section of the Great Wall of China "
            "predate the Qin dynasty?",
        "category": "high_conf_wrong_target",
        "expected_answer": None,
    },
    {
        "question":
            "What are the dimensions of the Xiaomi SU7?",
        "category": "numeric_spec",
        "expected_answer": None,
    },
    {
        "question":
            "What is the population of Lebanon as of 2024?",
        "category": "numeric_spec",
        "expected_answer": None,
    },
    {
        "question":
            "Who is the last president of the United States?",
        "category": "ambiguous_recency",
        "expected_answer": None,
    },
    {
        "question":
            "What is the capital of the fictional country "
            "Gorgonzolia?",
        "category": "low_conf_expected",
        "expected_answer": None,
    },
    {
        "question":
            "What year did Guns N' Roses perform a promo for "
            "a movie starring Arnold Schwarzenegger as a former "
            "New York Police detective?",
        "category": "multihop",
        "expected_answer": "1999",
    },
]


def main(config_path=None, eva_model_path=None):
    config = get_config(config_path)

    if not eva_model_path:
        raise ValueError(
            "A fine-tuned Eva checkpoint is required. Run "
            "train_complexity_evaluator.py, then pass --eva-model."
        )
    complexity_fn = load_complexity_evaluator(eva_model_path)

    reader_path = resolve_project_path(
        config,
        "representation_reader",
    )
    result_dir = resolve_project_path(
        config,
        "full_k_results",
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

    rows = []
    token_rows = []

    for test_id, case in enumerate(
        TEST_SUITE
    ):
        record = answer_with_confidence(
            question=case["question"],
            tokenizer=tokenizer,
            model=model,
            rep_reader=rep_reader,
            layers=detector_layers(config),
            complexity_fn=complexity_fn,
            threshold=threshold,
            max_new_tokens=config[
                "generation"
            ]["max_new_tokens"],
            repetition_penalty=config[
                "generation"
            ]["repetition_penalty"],
            verbose=False,
        )

        exp = case.get(
            "expected_answer"
        )

        # Preserved from the original notebook as a convenience field.
        # It is NOT a robust semantic correctness metric.
        auto_correct = (
            exp.lower()
            in record[
                "answer_text"
            ].lower()
            if exp is not None
            else None
        )

        rows.append({
            "test_id": test_id,
            "category": case["category"],
            "question": case["question"],
            "answer":
                record["answer_text"],
            "expected": exp,
            "auto_correct": auto_correct,
            "E": record["E"],
            "mean_m_tilde":
                record["mean_m_tilde"],
            "min_m_tilde":
                record["min_m_tilde"],
            "max_K":
                record["max_K"],
            "would_trigger_full_K":
                record[
                    "would_trigger_full"
                ],
            "would_trigger_confidence_only":
                record[
                    "would_trigger_confidence_only"
                ],
        })

        for entry in record[
            "token_entries"
        ]:
            token_rows.append({
                "test_id": test_id,
                "category":
                    case["category"],
                "question":
                    case["question"],
                "answer":
                    record["answer_text"],
                "token_index":
                    entry["token_index"],
                "token": entry["token"],
                "raw_score":
                    entry["raw_score"],
                "m_tilde":
                    entry.get("m_tilde"),
                "s_i": entry["s_i"],
                "K":
                    entry.get("K"),
                "is_content":
                    entry["is_content"],
            })

    results_df = pd.DataFrame(rows)
    token_df = pd.DataFrame(token_rows)

    results_path = os.path.join(
        result_dir,
        "test_suite_questions.csv",
    )
    token_path = os.path.join(
        result_dir,
        "test_suite_tokens.csv",
    )

    results_df.to_csv(
        results_path,
        index=False,
    )
    token_df.to_csv(
        token_path,
        index=False,
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved results to:\n"
        f"{result_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
    )
    parser.add_argument(
        "--eva-model",
        required=True,
        help="Path to the fine-tuned three-class Eva checkpoint.",
    )
    args = parser.parse_args()
    main(args.config, args.eva_model)
