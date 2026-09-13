"""
Run the confidence-only baseline experiment.

This experiment evaluates the trained internal confidence detector without
using external query complexity.

For each generated token, the pipeline is:

    question
        ↓
    Mistral generation
        ↓
    hidden-state confidence score m_i
        ↓
    causal normalization
        ↓
    normalized confidence m_tilde_i
        ↓
    compare with confidence threshold

A token is considered low-confidence when:

    m_tilde_i < confidence_threshold

This experiment is intentionally different from the full INKER IE-KRT rule:

    K(t_i) = (E - m_tilde_i) * s_i

because it ignores query complexity E.

Purpose
-------
The confidence-only experiment provides a baseline for determining whether
combining external query complexity with internal confidence actually improves
retrieval-trigger behavior.

It can later be compared against:

    1. confidence-only triggering,
    2. complexity-only triggering,
    3. full INKER K(t_i).

This script does NOT:
    - calculate query complexity E,
    - calculate K(t_i),
    - perform document retrieval,
    - or implement IE-KQF.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from _common import (
    get_config,
    get_detector_layers,
    get_project_path,
    load_configured_model,
)

from inker.generation import (
    answer_with_confidence_only,
)


# ==========================================================================
# Default small sanity-check question set
# ==========================================================================

DEFAULT_QUESTIONS = [
    {
        "question":
            "What is the capital of France?",

        "expected_answer":
            "Paris",
    },
    {
        "question":
            "Who wrote Romeo and Juliet?",

        "expected_answer":
            "William Shakespeare",
    },
    {
        "question":
            "What is the largest planet in the Solar System?",

        "expected_answer":
            "Jupiter",
    },
]


# ==========================================================================
# Representation reader
# ==========================================================================

def load_representation_reader(
    reader_path: Path,
    layers: Sequence[int],
) -> Dict[str, Any]:
    """
    Load and validate the trained confidence representation reader.
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
            "Representation reader must be a dictionary."
        )

    required_components = {
        "directions",
        "H_train_means",
        "signs",
    }

    missing = (
        required_components
        - set(
            rep_reader.keys()
        )
    )

    if missing:

        raise KeyError(
            "Representation reader is missing required components: "
            f"{sorted(missing)}"
        )

    for layer in layers:

        for component in required_components:

            if layer not in rep_reader[
                component
            ]:

                raise KeyError(
                    f"Representation reader does not contain "
                    f"layer {layer} in '{component}'."
                )

    return rep_reader


# ==========================================================================
# Question loading
# ==========================================================================

def load_questions(
    questions_path: str | Path | None = None,
) -> List[Dict[str, str | None]]:
    """
    Load confidence-only evaluation questions.

    If no file is supplied, the three built-in sanity-check questions are
    returned.

    Supported file formats
    ----------------------
    CSV
        Expected columns:

            question

        and optionally:

            expected_answer

    JSON
        Either:

            [
                {
                    "question": "...",
                    "expected_answer": "..."
                }
            ]

        or simply:

            [
                "question one",
                "question two"
            ]

    JSONL
        One JSON object per line.

    TXT
        One question per non-empty line.
        Expected answers are unavailable in this format.
    """

    if questions_path is None:

        return [
            dict(
                item
            )
            for item in DEFAULT_QUESTIONS
        ]

    path = Path(
        questions_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Question file not found: {path}"
        )

    suffix = (
        path.suffix
        .lower()
    )

    cases: List[
        Dict[str, str | None]
    ] = []

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    if suffix == ".csv":

        df = pd.read_csv(
            path
        )

        if "question" not in df.columns:

            raise KeyError(
                "CSV question file must contain a 'question' column."
            )

        for _, row in df.iterrows():

            question = row.get(
                "question"
            )

            if pd.isna(
                question
            ):
                continue

            expected_answer = row.get(
                "expected_answer"
            )

            if pd.isna(
                expected_answer
            ):
                expected_answer = None

            cases.append({
                "question":
                    str(
                        question
                    ).strip(),

                "expected_answer":
                    (
                        str(
                            expected_answer
                        ).strip()
                        if expected_answer is not None
                        else None
                    ),
            })

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    elif suffix == ".json":

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        if not isinstance(
            data,
            list,
        ):

            raise ValueError(
                "JSON question file must contain a list."
            )

        for item in data:

            if isinstance(
                item,
                str,
            ):

                cases.append({
                    "question":
                        item.strip(),

                    "expected_answer":
                        None,
                })

            elif isinstance(
                item,
                dict,
            ):

                if "question" not in item:

                    raise KeyError(
                        "Each JSON question object must contain 'question'."
                    )

                cases.append({
                    "question":
                        str(
                            item[
                                "question"
                            ]
                        ).strip(),

                    "expected_answer":
                        (
                            str(
                                item[
                                    "expected_answer"
                                ]
                            ).strip()
                            if item.get(
                                "expected_answer"
                            ) is not None
                            else None
                        ),
                })

            else:

                raise TypeError(
                    "JSON entries must be strings or dictionaries."
                )

    # ------------------------------------------------------------------
    # JSONL
    # ------------------------------------------------------------------

    elif suffix == ".jsonl":

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                item = json.loads(
                    line
                )

                if isinstance(
                    item,
                    str,
                ):

                    cases.append({
                        "question":
                            item.strip(),

                        "expected_answer":
                            None,
                    })

                elif isinstance(
                    item,
                    dict,
                ):

                    if "question" not in item:

                        raise KeyError(
                            f"Line {line_number} is missing 'question'."
                        )

                    cases.append({
                        "question":
                            str(
                                item[
                                    "question"
                                ]
                            ).strip(),

                        "expected_answer":
                            (
                                str(
                                    item[
                                        "expected_answer"
                                    ]
                                ).strip()
                                if item.get(
                                    "expected_answer"
                                ) is not None
                                else None
                            ),
                    })

                else:

                    raise TypeError(
                        f"Line {line_number}: expected string "
                        "or dictionary."
                    )

    # ------------------------------------------------------------------
    # TXT
    # ------------------------------------------------------------------

    elif suffix == ".txt":

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                question = (
                    line.strip()
                )

                if not question:
                    continue

                cases.append({
                    "question":
                        question,

                    "expected_answer":
                        None,
                })

    else:

        raise ValueError(
            "Unsupported question-file format. "
            "Supported extensions: .csv, .json, .jsonl, .txt"
        )

    cases = [
        case
        for case in cases
        if case[
            "question"
        ]
    ]

    if not cases:

        raise ValueError(
            f"No usable questions found in {path}."
        )

    return cases


# ==========================================================================
# Small numeric helpers
# ==========================================================================

def safe_mean(
    values,
) -> float:
    """
    Return a numeric mean or NaN for an empty collection.
    """

    values = [
        float(value)
        for value in values
        if value is not None
        and np.isfinite(
            float(value)
        )
    ]

    if not values:
        return float(
            "nan"
        )

    return float(
        np.mean(
            values
        )
    )


def safe_min(
    values,
) -> float:
    """
    Return a numeric minimum or NaN for an empty collection.
    """

    values = [
        float(value)
        for value in values
        if value is not None
        and np.isfinite(
            float(value)
        )
    ]

    if not values:
        return float(
            "nan"
        )

    return float(
        np.min(
            values
        )
    )


def safe_max(
    values,
) -> float:
    """
    Return a numeric maximum or NaN for an empty collection.
    """

    values = [
        float(value)
        for value in values
        if value is not None
        and np.isfinite(
            float(value)
        )
    ]

    if not values:
        return float(
            "nan"
        )

    return float(
        np.max(
            values
        )
    )


# ==========================================================================
# Main experiment
# ==========================================================================

def main(
    config_path: str | Path | None = None,
    questions_path: str | Path | None = None,
    num_questions: int | None = None,
) -> None:
    """
    Run confidence-only generation and save question/token-level results.
    """

    # ------------------------------------------------------------------
    # 1. Configuration.
    # ------------------------------------------------------------------

    config = get_config(
        config_path
    )

    layers = get_detector_layers(
        config
    )

    reader_path = get_project_path(
        config,
        "representation_reader",
    )

    result_dir = get_project_path(
        config,
        "confidence_only_results",
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # 2. Load trained representation reader.
    # ------------------------------------------------------------------

    rep_reader = load_representation_reader(
        reader_path=reader_path,
        layers=layers,
    )

    # ------------------------------------------------------------------
    # 3. Load Mistral model/tokenizer.
    # ------------------------------------------------------------------

    tokenizer, model = (
        load_configured_model(
            config
        )
    )

    # ------------------------------------------------------------------
    # 4. Configuration values.
    # ------------------------------------------------------------------

    confidence_threshold = float(
        config[
            "confidence"
        ][
            "threshold"
        ]
    )

    generation_config = config[
        "generation"
    ]

    max_new_tokens = int(
        generation_config[
            "max_new_tokens"
        ]
    )

    repetition_penalty = float(
        generation_config[
            "repetition_penalty"
        ]
    )

    # ------------------------------------------------------------------
    # 5. Load questions.
    # ------------------------------------------------------------------

    cases = load_questions(
        questions_path
    )

    if num_questions is not None:

        if num_questions <= 0:

            raise ValueError(
                "--num-questions must be greater than zero."
            )

        cases = cases[
            :num_questions
        ]

    print(
        f"Evaluating {len(cases)} questions "
        "with the confidence-only baseline..."
    )

    # ------------------------------------------------------------------
    # 6. Run generation.
    # ------------------------------------------------------------------

    summary_rows = []
    token_rows = []

    for question_id, case in enumerate(
        cases
    ):

        question = (
            case[
                "question"
            ]
        )

        expected_answer = (
            case.get(
                "expected_answer"
            )
        )

        print(
            f"\n[{question_id + 1}/{len(cases)}] "
            f"{question}"
        )

        record = (
            answer_with_confidence_only(
                question=question,
                expected_answer=expected_answer,
                tokenizer=tokenizer,
                model=model,
                rep_reader=rep_reader,
                layers=layers,
                confidence_threshold=confidence_threshold,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                verbose=False,
            )
        )

        # --------------------------------------------------------------
        # Support either final naming used by generation.py:
        #
        #     token_entries
        #
        # or:
        #
        #     token_history
        # --------------------------------------------------------------

        token_entries = record.get(
            "token_entries",
            record.get(
                "token_history",
                [],
            ),
        )

        # --------------------------------------------------------------
        # Compute summaries from token records rather than depending on
        # duplicated summary fields in generation.py.
        # --------------------------------------------------------------

        content_entries = [
            entry
            for entry in token_entries
            if int(
                entry.get(
                    "s_i",
                    0,
                )
            ) == 1
        ]

        confidence_values = [
            entry.get(
                "m_tilde"
            )
            for entry in content_entries
        ]

        low_confidence_entries = [
            entry
            for entry in content_entries
            if (
                entry.get(
                    "m_tilde"
                ) is not None
                and float(
                    entry[
                        "m_tilde"
                    ]
                ) < confidence_threshold
            )
        ]

        would_trigger = bool(
            low_confidence_entries
        )

        answer_text = record.get(
            "answer_text",
            record.get(
                "generated_text",
                "",
            ),
        )

        # --------------------------------------------------------------
        # Question-level record.
        # --------------------------------------------------------------

        summary_rows.append({
            "question_id":
                question_id,

            "question":
                question,

            "answer":
                answer_text,

            "expected_answer":
                expected_answer,

            "confidence_threshold":
                confidence_threshold,

            "mean_confidence":
                safe_mean(
                    confidence_values
                ),

            "min_confidence":
                safe_min(
                    confidence_values
                ),

            "max_confidence":
                safe_max(
                    confidence_values
                ),

            "num_generated_tokens":
                len(
                    token_entries
                ),

            "num_content_tokens":
                len(
                    content_entries
                ),

            "num_low_confidence_tokens":
                len(
                    low_confidence_entries
                ),

            "would_trigger_confidence_only":
                would_trigger,
        })

        # --------------------------------------------------------------
        # Token-level records.
        # --------------------------------------------------------------

        for fallback_index, entry in enumerate(
            token_entries
        ):

            m_tilde = entry.get(
                "m_tilde"
            )

            s_i = int(
                entry.get(
                    "s_i",
                    0,
                )
            )

            is_content = (
                s_i == 1
            )

            low_confidence = bool(
                is_content
                and m_tilde is not None
                and float(
                    m_tilde
                ) < confidence_threshold
            )

            token_rows.append({
                "question_id":
                    question_id,

                "question":
                    question,

                "answer":
                    answer_text,

                "token_index":
                    entry.get(
                        "token_index",
                        fallback_index,
                    ),

                "token":
                    entry.get(
                        "token",
                        "",
                    ),

                "raw_confidence":
                    entry.get(
                        "raw_confidence",
                        entry.get(
                            "raw_score",
                            entry.get(
                                "m_i"
                            ),
                        ),
                    ),

                "m_tilde":
                    m_tilde,

                "s_i":
                    s_i,

                "is_content":
                    is_content,

                "low_confidence":
                    low_confidence,
            })

    # ------------------------------------------------------------------
    # 7. Save CSV outputs.
    # ------------------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    token_df = pd.DataFrame(
        token_rows
    )

    summary_path = (
        result_dir
        / "confidence_only_questions.csv"
    )

    token_path = (
        result_dir
        / "confidence_only_tokens.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    token_df.to_csv(
        token_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # 8. Print summary.
    # ------------------------------------------------------------------

    n_questions = len(
        summary_df
    )

    n_triggered = int(
        summary_df[
            "would_trigger_confidence_only"
        ].sum()
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "CONFIDENCE-ONLY BASELINE SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"\nQuestions evaluated: "
        f"{n_questions}"
    )

    print(
        f"Confidence threshold: "
        f"{confidence_threshold:.4f}"
    )

    if n_questions > 0:

        print(
            f"Questions with at least one "
            f"low-confidence content token: "
            f"{n_triggered}/{n_questions} "
            f"({100.0 * n_triggered / n_questions:.1f}%)"
        )

    print(
        f"\nQuestion-level results:\n"
        f"  {summary_path}"
    )

    print(
        f"\nToken-level results:\n"
        f"  {token_path}"
    )

    print(
        "\n"
        + "=" * 80
    )


# ==========================================================================
# CLI
# ==========================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run the INKER internal-confidence-only baseline."
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

    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help=(
            "Optional CSV, JSON, JSONL, or TXT question file. "
            "If omitted, three built-in sanity-check questions are used."
        ),
    )

    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help=(
            "Optional maximum number of questions to evaluate."
        ),
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        questions_path=args.questions,
        num_questions=args.num_questions,
    )
