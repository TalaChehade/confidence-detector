"""
Run the qualitative INKER IE-KRT test suite.

This experiment evaluates the live INKER retrieval-trigger mechanism on a
small hand-curated collection of questions designed to probe different
behaviors, including:

    - paper-inspired case studies,
    - simple factual questions,
    - potentially overconfident answers,
    - numeric/specification questions,
    - ambiguous or recency-sensitive questions,
    - deliberately unanswerable / fictional questions,
    - and multihop questions.

For each question, the experiment combines:

    1. External query complexity E
       estimated using the Adaptive-RAG-style T5-Large evaluator.

    2. Internal token confidence m_tilde_i
       estimated from the trained confidence representation reader.

    3. Token-content mask s_i.

The live INKER activation is:

    K(t_i) = (E - m_tilde_i) * s_i

Retrieval is triggered when:

    K(t_i) > tau

for the first qualifying generated token.

Purpose
-------
This is a qualitative / behavioral diagnostic test suite.

It is NOT intended to be treated as a statistically meaningful benchmark
accuracy evaluation.

The manually assigned categories are useful for examining whether the trigger
behaves sensibly across different kinds of questions.

Important
---------
This script implements IE-KRT:

    "When should retrieval occur?"

It does NOT yet perform retrieval after a trigger.

The full IE-KQF / retrieval / evidence-injection pipeline will be implemented
separately.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch

from _common import (
    get_config,
    get_detector_layers,
    get_project_path,
    load_configured_model,
)

from inker.complexity import (
    AdaptiveRAGComplexityEvaluator,
    DEFAULT_ADAPTIVE_RAG_MODEL,
)

from inker.generation import (
    answer_with_confidence,
)


# ==========================================================================
# Qualitative test suite
# ==========================================================================

TEST_SUITE = [
    {
        "question":
            "In what city is the company that Fastjet Tanzania "
            "was originally founded as a part of prior to "
            "rebranding based?",

        "category":
            "paper_case_study",

        "expected_answer":
            "Nairobi",
    },
    {
        "question":
            "Stephen Smith appears on ESPN First Take alongside "
            "which HBO boxing commentator?",

        "category":
            "paper_case_study",

        "expected_answer":
            "Kellerman",
    },
    {
        "question":
            "What is the capital of France?",

        "category":
            "high_conf_correct",

        "expected_answer":
            "Paris",
    },
    {
        "question":
            "What is the chemical symbol for water?",

        "category":
            "high_conf_correct",

        "expected_answer":
            "H2O",
    },
    {
        "question":
            "Who played the villain in the original 1984 "
            "Terminator movie?",

        "category":
            "high_conf_wrong_target",

        "expected_answer":
            None,
    },
    {
        "question":
            "Which HBO boxing analyst co-hosted ESPN's "
            "First Take with Stephen A. Smith?",

        "category":
            "high_conf_wrong_target",

        "expected_answer":
            "Kellerman",
    },
    {
        "question":
            "How many senses do humans have?",

        "category":
            "high_conf_wrong_target",

        "expected_answer":
            None,
    },
    {
        "question":
            "Does the oldest section of the Great Wall of China "
            "predate the Qin dynasty?",

        "category":
            "high_conf_wrong_target",

        "expected_answer":
            None,
    },
    {
        "question":
            "What are the dimensions of the Xiaomi SU7?",

        "category":
            "numeric_spec",

        "expected_answer":
            None,
    },
    {
        "question":
            "What is the population of Lebanon as of 2024?",

        "category":
            "numeric_spec",

        "expected_answer":
            None,
    },
    {
        "question":
            "Who is the last president of the United States?",

        "category":
            "ambiguous_recency",

        "expected_answer":
            None,
    },
    {
        "question":
            "What is the capital of the fictional country "
            "Gorgonzolia?",

        "category":
            "low_conf_expected",

        "expected_answer":
            None,
    },
    {
        "question":
            "What year did Guns N' Roses perform a promo for "
            "a movie starring Arnold Schwarzenegger as a former "
            "New York Police detective?",

        "category":
            "multihop",

        "expected_answer":
            "1999",
    },
]


# ==========================================================================
# Reproducibility
# ==========================================================================

def set_random_seed(
    seed: int,
) -> None:
    """
    Seed Python-adjacent numerical libraries used by this experiment.
    """

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
# Representation reader
# ==========================================================================

def load_representation_reader(
    reader_path: Path,
    layers: Sequence[int],
) -> Dict[str, Any]:
    """
    Load and validate the trained internal-confidence representation reader.
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
# Numeric helpers
# ==========================================================================

def safe_mean(
    values: Iterable[float],
) -> float:
    """
    Return the mean of valid values, or NaN if none exist.
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
    values: Iterable[float],
) -> float:
    """
    Return the minimum of valid values, or NaN if none exist.
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
    values: Iterable[float],
) -> float:
    """
    Return the maximum of valid values, or NaN if none exist.
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
# Correctness convenience field
# ==========================================================================

def simple_expected_answer_match(
    expected_answer: str | None,
    generated_text: str,
) -> bool | None:
    """
    Perform a simple case-insensitive substring match.

    Important
    ---------
    This is only a convenience diagnostic.

    It is NOT a robust semantic correctness metric.

    Examples
    --------
    expected = "Paris"
    generated = "The capital of France is Paris."

        -> True

    But the same substring-based method may fail on:

        - synonyms,
        - paraphrases,
        - alternate valid answer forms,
        - numeric formatting,
        - partially correct answers,
        - or answers containing the expected string in an incorrect context.

    Therefore this field should not be reported as benchmark accuracy.
    """

    if expected_answer is None:
        return None

    expected = (
        expected_answer
        .strip()
        .lower()
    )

    generated = (
        generated_text
        .strip()
        .lower()
    )

    if not expected:
        return None

    return expected in generated


# ==========================================================================
# Main
# ==========================================================================

def main(
    config_path: str | Path | None = None,
    complexity_model: str = DEFAULT_ADAPTIVE_RAG_MODEL,
    num_tests: int | None = None,
) -> None:
    """
    Run the qualitative live IE-KRT test suite.
    """

    # ------------------------------------------------------------------
    # 1. Configuration.
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

    layers = get_detector_layers(
        config
    )

    # ------------------------------------------------------------------
    # 2. Paths.
    # ------------------------------------------------------------------

    reader_path = get_project_path(
        config,
        "representation_reader",
    )

    result_dir = get_project_path(
        config,
        "full_k_results",
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # 3. Thresholds.
    #
    # Confidence-only and full retrieval thresholds are separate.
    # ------------------------------------------------------------------

    confidence_config = config.get(
        "confidence",
        {}
    )

    trigger_config = config.get(
        "trigger",
        {}
    )

    generation_config = config.get(
        "generation",
        {}
    )

    confidence_threshold = float(
        confidence_config.get(
            "threshold",
            0.5,
        )
    )

    trigger_threshold = float(
        trigger_config.get(
            "threshold",
            0.5,
        )
    )

    stop_on_trigger = bool(
        trigger_config.get(
            "stop_on_trigger",
            True,
        )
    )

    max_new_tokens = int(
        generation_config.get(
            "max_new_tokens",
            60,
        )
    )

    repetition_penalty = float(
        generation_config.get(
            "repetition_penalty",
            1.1,
        )
    )

    # ------------------------------------------------------------------
    # 4. Load trained internal confidence detector.
    # ------------------------------------------------------------------

    rep_reader = (
        load_representation_reader(
            reader_path=reader_path,
            layers=layers,
        )
    )

    # ------------------------------------------------------------------
    # 5. Load base LLM.
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
    # 6. Load Adaptive-RAG-style complexity evaluator.
    #
    # No locally trained Eva checkpoint is required for the primary
    # replication.
    # ------------------------------------------------------------------

    print(
        "Loading query-complexity evaluator..."
    )

    print(
        f"  {complexity_model}"
    )

    complexity_evaluator = (
        AdaptiveRAGComplexityEvaluator(
            model_name=complexity_model
        )
    )

    # ------------------------------------------------------------------
    # 7. Select tests.
    # ------------------------------------------------------------------

    test_cases = list(
        TEST_SUITE
    )

    if num_tests is not None:

        if num_tests <= 0:

            raise ValueError(
                "--num-tests must be greater than zero."
            )

        test_cases = test_cases[
            :num_tests
        ]

    print(
        f"\nRunning {len(test_cases)} qualitative tests..."
    )

    # ------------------------------------------------------------------
    # 8. Run.
    # ------------------------------------------------------------------

    question_rows: List[
        Dict[str, Any]
    ] = []

    token_rows: List[
        Dict[str, Any]
    ] = []

    for test_id, case in enumerate(
        test_cases
    ):

        question = case[
            "question"
        ]

        category = case[
            "category"
        ]

        expected_answer = case.get(
            "expected_answer"
        )

        print(
            f"\n[{test_id + 1}/{len(test_cases)}]"
        )

        print(
            f"Category: {category}"
        )

        print(
            f"Question: {question}"
        )

        # --------------------------------------------------------------
        # Evaluate query complexity once.
        #
        # We cache this result so generation.py does not need to run the
        # T5 evaluator a second time.
        # --------------------------------------------------------------

        complexity = (
            complexity_evaluator.evaluate(
                question
            )
        )

        def complexity_fn(
            _question: str,
            cached_complexity=complexity,
        ):
            return cached_complexity

        # --------------------------------------------------------------
        # Live IE-KRT generation.
        # --------------------------------------------------------------

        record = (
            answer_with_confidence(
                question=question,
                tokenizer=tokenizer,
                model=model,
                rep_reader=rep_reader,
                layers=layers,
                complexity_fn=complexity_fn,
                trigger_threshold=trigger_threshold,
                confidence_threshold=confidence_threshold,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                stop_on_trigger=stop_on_trigger,
                verbose=False,
            )
        )

        # --------------------------------------------------------------
        # Support either token-history naming while repository cleanup is
        # still in progress.
        # --------------------------------------------------------------

        token_entries = record.get(
            "token_entries",
            record.get(
                "token_history",
                [],
            ),
        )

        answer_text = record.get(
            "answer_text",
            record.get(
                "generated_text",
                "",
            ),
        )

        # --------------------------------------------------------------
        # Derive metrics from token-level records.
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

        m_tilde_values = [
            entry.get(
                "m_tilde"
            )
            for entry in content_entries
        ]

        K_values = [
            entry.get(
                "K"
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

        full_triggered = bool(
            record.get(
                "triggered",
                record.get(
                    "would_trigger_full",
                    False,
                ),
            )
        )

        confidence_only_triggered = bool(
            low_confidence_entries
        )

        trigger_token = (
            record.get(
                "trigger_token"
            )
        )

        trigger_index = (
            record.get(
                "trigger_index"
            )
        )

        auto_correct = (
            simple_expected_answer_match(
                expected_answer=expected_answer,
                generated_text=answer_text,
            )
        )

        # --------------------------------------------------------------
        # Question-level result.
        # --------------------------------------------------------------

        question_rows.append({
            "test_id":
                test_id,

            "category":
                category,

            "question":
                question,

            "answer":
                answer_text,

            "expected":
                expected_answer,

            "simple_expected_match":
                auto_correct,

            "complexity_class":
                complexity.predicted_class,

            "p_A":
                complexity.p_A,

            "p_B":
                complexity.p_B,

            "p_C":
                complexity.p_C,

            "E":
                complexity.E,

            "confidence_threshold":
                confidence_threshold,

            "trigger_threshold":
                trigger_threshold,

            "num_generated_tokens":
                len(
                    token_entries
                ),

            "num_content_tokens":
                len(
                    content_entries
                ),

            "mean_m_tilde":
                safe_mean(
                    m_tilde_values
                ),

            "min_m_tilde":
                safe_min(
                    m_tilde_values
                ),

            "max_m_tilde":
                safe_max(
                    m_tilde_values
                ),

            "max_K":
                safe_max(
                    K_values
                ),

            "num_low_confidence_content_tokens":
                len(
                    low_confidence_entries
                ),

            "retrieval_triggered_full_K":
                full_triggered,

            "retrieval_triggered_confidence_only":
                confidence_only_triggered,

            "trigger_token":
                trigger_token,

            "trigger_index":
                trigger_index,
        })

        # --------------------------------------------------------------
        # Token-level result.
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
                "test_id":
                    test_id,

                "category":
                    category,

                "question":
                    question,

                "answer":
                    answer_text,

                "E":
                    complexity.E,

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

                "K":
                    entry.get(
                        "K"
                    ),

                "is_content":
                    is_content,

                "low_confidence":
                    low_confidence,

                "triggered":
                    bool(
                        entry.get(
                            "triggered",
                            False,
                        )
                    ),
            })

        print(
            f"  E = {complexity.E:.4f}"
        )

        print(
            f"  Full K trigger = {full_triggered}"
        )

        print(
            f"  Confidence-only trigger = "
            f"{confidence_only_triggered}"
        )

        if trigger_token is not None:

            print(
                f"  Trigger token = {trigger_token!r}"
            )

    # ------------------------------------------------------------------
    # 9. DataFrames.
    # ------------------------------------------------------------------

    results_df = pd.DataFrame(
        question_rows
    )

    token_df = pd.DataFrame(
        token_rows
    )

    # ------------------------------------------------------------------
    # 10. Save.
    # ------------------------------------------------------------------

    results_path = (
        result_dir
        / "test_suite_questions.csv"
    )

    token_path = (
        result_dir
        / "test_suite_tokens.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    token_df.to_csv(
        token_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # 11. Print question-level table.
    # ------------------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "QUALITATIVE TEST SUITE RESULTS"
    )

    print(
        "=" * 100
    )

    display_columns = [
        "test_id",
        "category",
        "complexity_class",
        "E",
        "mean_m_tilde",
        "max_K",
        "retrieval_triggered_full_K",
        "retrieval_triggered_confidence_only",
        "trigger_token",
        "simple_expected_match",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # 12. Category summary.
    # ------------------------------------------------------------------

    if not results_df.empty:

        category_summary = (
            results_df
            .groupby(
                "category",
                dropna=False,
            )
            .agg(
                n_tests=(
                    "test_id",
                    "count",
                ),

                mean_E=(
                    "E",
                    "mean",
                ),

                mean_max_K=(
                    "max_K",
                    "mean",
                ),

                full_trigger_rate=(
                    "retrieval_triggered_full_K",
                    "mean",
                ),

                confidence_only_trigger_rate=(
                    "retrieval_triggered_confidence_only",
                    "mean",
                ),
            )
            .reset_index()
        )

        category_path = (
            result_dir
            / "test_suite_by_category.csv"
        )

        category_summary.to_csv(
            category_path,
            index=False,
        )

        print(
            "\nCategory summary:"
        )

        print(
            category_summary.to_string(
                index=False
            )
        )

    else:

        category_path = None

    # ------------------------------------------------------------------
    # 13. Final paths.
    # ------------------------------------------------------------------

    print(
        "\nSaved results:"
    )

    print(
        f"  {results_path}"
    )

    print(
        f"  {token_path}"
    )

    if category_path is not None:

        print(
            f"  {category_path}"
        )


# ==========================================================================
# CLI
# ==========================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run the qualitative live INKER IE-KRT test suite."
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
        "--complexity-model",
        type=str,
        default=DEFAULT_ADAPTIVE_RAG_MODEL,
        help=(
            "Adaptive-RAG-style T5 complexity model. "
            f"Default: {DEFAULT_ADAPTIVE_RAG_MODEL}"
        ),
    )

    parser.add_argument(
        "--num-tests",
        type=int,
        default=None,
        help=(
            "Optional number of test-suite questions to run."
        ),
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        complexity_model=args.complexity_model,
        num_tests=args.num_tests,
    )
