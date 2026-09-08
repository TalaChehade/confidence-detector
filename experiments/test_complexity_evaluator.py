"""
Test the Adaptive-RAG-style query complexity evaluator.

This experiment evaluates ONLY the external query-complexity component used by
the INKER IE-KRT replication.

The evaluator predicts one of the Adaptive-RAG retrieval-complexity classes:

    A = no retrieval
    B = single-step retrieval
    C = multi-step / iterative retrieval

The replication then converts the class probabilities into a continuous
external complexity score:

    A -> 0.0
    B -> 0.5
    C -> 1.0

and defines:

    E = 0.0 * P(A)
        + 0.5 * P(B)
        + 1.0 * P(C)

therefore:

    E = 0.5 * P(B) + P(C)

Purpose
-------
This file is a component-level sanity check.

It answers questions such as:

    - Does the Adaptive-RAG reproduction load correctly?
    - What class does it predict for simple and difficult queries?
    - Are P(A), P(B), and P(C) sensible?
    - Does the continuous replication score E increase for queries that
      appear to require more retrieval/reasoning?

Important
---------
The manually assigned labels:

    low
    medium
    high

in this script are qualitative expectations created for diagnostic purposes.

They are NOT official Adaptive-RAG labels and should NOT be reported as
ground-truth benchmark annotations.

This script intentionally does NOT run Mistral generation or the confidence
detector. Full integration is tested separately by:

    experiments/run_inker_trigger.py
    experiments/run_test_suite.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from _common import (
    get_config,
    get_project_path,
)

from inker.complexity import (
    AdaptiveRAGComplexityEvaluator,
    DEFAULT_ADAPTIVE_RAG_MODEL,
)


# ==========================================================================
# Qualitative sanity-check queries
# ==========================================================================

TEST_QUERIES = [
    # ------------------------------------------------------------------
    # Lower expected retrieval complexity
    # ------------------------------------------------------------------
    {
        "query":
            "What is the capital of France?",

        "expected_level":
            "low",

        "notes":
            "Simple factual question likely answerable parametrically.",
    },
    {
        "query":
            "Who wrote Romeo and Juliet?",

        "expected_level":
            "low",

        "notes":
            "Simple well-known factual question.",
    },
    {
        "query":
            "What is 2 + 2?",

        "expected_level":
            "low",

        "notes":
            "Very simple arithmetic.",
    },
    {
        "query":
            "What color is the sky?",

        "expected_level":
            "low",

        "notes":
            "Simple general-knowledge question.",
    },

    # ------------------------------------------------------------------
    # Medium expected retrieval complexity
    # ------------------------------------------------------------------
    {
        "query":
            "How do plants perform photosynthesis?",

        "expected_level":
            "medium",

        "notes":
            "Explanatory question requiring multiple related facts.",
    },
    {
        "query":
            "What are the causes of climate change?",

        "expected_level":
            "medium",

        "notes":
            "Broad explanatory question with several contributing factors.",
    },
    {
        "query":
            "Explain the theory of evolution.",

        "expected_level":
            "medium",

        "notes":
            "Conceptual explanation rather than a single factual lookup.",
    },
    {
        "query":
            "Describe the water cycle.",

        "expected_level":
            "medium",

        "notes":
            "Multi-stage explanatory process.",
    },

    # ------------------------------------------------------------------
    # Higher expected retrieval / reasoning complexity
    # ------------------------------------------------------------------
    {
        "query":
            "Who was the first president of the United States "
            "and what were his major achievements?",

        "expected_level":
            "high",

        "notes":
            "Requires entity identification plus additional information.",
    },
    {
        "query":
            "Compare and contrast the causes of World War I "
            "and World War II.",

        "expected_level":
            "high",

        "notes":
            "Requires retrieving and comparing multiple sets of facts.",
    },
    {
        "query":
            "How does the greenhouse effect contribute to climate change "
            "and what are the long-term consequences?",

        "expected_level":
            "high",

        "notes":
            "Multi-part causal question.",
    },
    {
        "query":
            "Explain the relationship between supply and demand in economics "
            "and how it affects market prices.",

        "expected_level":
            "high",

        "notes":
            "Requires explaining a relationship and its consequences.",
    },
]


# ==========================================================================
# Helpers
# ==========================================================================

def qualitative_level_to_rank(
    level: str,
) -> int:
    """
    Map qualitative diagnostic labels to an ordinal rank.

    This mapping exists ONLY for summary statistics.

        low    -> 0
        medium -> 1
        high   -> 2

    It must not be confused with the Adaptive-RAG A/B/C labels.
    """

    mapping = {
        "low": 0,
        "medium": 1,
        "high": 2,
    }

    if level not in mapping:

        raise ValueError(
            f"Unknown qualitative complexity level: {level!r}"
        )

    return mapping[
        level
    ]


# ==========================================================================
# Main evaluator test
# ==========================================================================

def test_complexity_evaluator(
    config_path: str | Path | None = None,
    model_name: str = DEFAULT_ADAPTIVE_RAG_MODEL,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate the Adaptive-RAG complexity model on diagnostic queries.

    Parameters
    ----------
    config_path:
        Optional YAML configuration path.

    model_name:
        Hugging Face model identifier for the Adaptive-RAG-style evaluator.

    verbose:
        Whether to print detailed predictions.

    Returns
    -------
    pandas.DataFrame
        One row per diagnostic query.
    """

    # ------------------------------------------------------------------
    # 1. Configuration.
    # ------------------------------------------------------------------

    config = get_config(
        config_path
    )

    # ------------------------------------------------------------------
    # 2. Load pretrained Adaptive-RAG reproduction.
    #
    # The primary replication does not require retraining T5-Large.
    # ------------------------------------------------------------------

    if verbose:

        print(
            "Loading Adaptive-RAG query-complexity evaluator..."
        )

        print(
            f"  Model: {model_name}"
        )

    evaluator = (
        AdaptiveRAGComplexityEvaluator(
            model_name=model_name
        )
    )

    # ------------------------------------------------------------------
    # 3. Evaluate diagnostic queries.
    # ------------------------------------------------------------------

    results: List[
        Dict
    ] = []

    if verbose:

        print(
            "\n"
            + "=" * 100
        )

        print(
            "ADAPTIVE-RAG COMPLEXITY EVALUATOR SANITY CHECK"
        )

        print(
            "=" * 100
        )

    for test_id, case in enumerate(
        TEST_QUERIES
    ):

        query = case[
            "query"
        ]

        expected_level = case[
            "expected_level"
        ]

        complexity = (
            evaluator.evaluate(
                query
            )
        )

        row = {
            "test_id":
                test_id,

            "query":
                query,

            "qualitative_expected_level":
                expected_level,

            "qualitative_expected_rank":
                qualitative_level_to_rank(
                    expected_level
                ),

            "notes":
                case.get(
                    "notes"
                ),

            "predicted_class":
                complexity.predicted_class,

            "p_A_no_retrieval":
                complexity.p_A,

            "p_B_single_step":
                complexity.p_B,

            "p_C_multi_step":
                complexity.p_C,

            "complexity_score_E":
                complexity.E,

            "query_word_count":
                len(
                    query.split()
                ),
        }

        results.append(
            row
        )

        if verbose:

            print(
                f"\n[{test_id + 1}/{len(TEST_QUERIES)}]"
            )

            print(
                f"Query:\n  {query}"
            )

            print(
                f"Qualitative expectation: "
                f"{expected_level}"
            )

            print(
                "Adaptive-RAG probabilities:"
            )

            print(
                f"  P(A) = {complexity.p_A:.6f}"
            )

            print(
                f"  P(B) = {complexity.p_B:.6f}"
            )

            print(
                f"  P(C) = {complexity.p_C:.6f}"
            )

            print(
                f"Predicted class: "
                f"{complexity.predicted_class}"
            )

            print(
                f"Continuous E: "
                f"{complexity.E:.6f}"
            )

            print(
                "-" * 100
            )

    # ------------------------------------------------------------------
    # 4. Results DataFrame.
    # ------------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # ------------------------------------------------------------------
    # 5. Sanity checks.
    # ------------------------------------------------------------------

    probability_sums = (
        results_df[
            [
                "p_A_no_retrieval",
                "p_B_single_step",
                "p_C_multi_step",
            ]
        ]
        .sum(
            axis=1
        )
        .to_numpy()
    )

    if not np.allclose(
        probability_sums,
        1.0,
        atol=1e-5,
    ):

        raise ValueError(
            "Adaptive-RAG class probabilities do not sum to 1."
        )

    E_values = (
        results_df[
            "complexity_score_E"
        ]
        .to_numpy()
    )

    if not np.all(
        (
            E_values >= 0.0
        )
        & (
            E_values <= 1.0
        )
    ):

        raise ValueError(
            "At least one complexity score E lies outside [0, 1]."
        )

    # ------------------------------------------------------------------
    # 6. Qualitative group summary.
    # ------------------------------------------------------------------

    level_summary = (
        results_df
        .groupby(
            [
                "qualitative_expected_rank",
                "qualitative_expected_level",
            ],
            as_index=False,
        )
        .agg(
            n_queries=(
                "test_id",
                "count",
            ),

            mean_E=(
                "complexity_score_E",
                "mean",
            ),

            std_E=(
                "complexity_score_E",
                "std",
            ),

            min_E=(
                "complexity_score_E",
                "min",
            ),

            max_E=(
                "complexity_score_E",
                "max",
            ),

            mean_p_A=(
                "p_A_no_retrieval",
                "mean",
            ),

            mean_p_B=(
                "p_B_single_step",
                "mean",
            ),

            mean_p_C=(
                "p_C_multi_step",
                "mean",
            ),
        )
        .sort_values(
            "qualitative_expected_rank"
        )
    )

    # ------------------------------------------------------------------
    # 7. Save.
    # ------------------------------------------------------------------

    result_dir = get_project_path(
        config,
        "complexity_eval_results",
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        result_dir
        / "complexity_test_queries.csv"
    )

    summary_path = (
        result_dir
        / "complexity_test_by_expected_level.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    level_summary.to_csv(
        summary_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # 8. Print summary.
    # ------------------------------------------------------------------

    if verbose:

        print(
            "\n"
            + "=" * 100
        )

        print(
            "SUMMARY BY QUALITATIVE EXPECTED LEVEL"
        )

        print(
            "=" * 100
        )

        print(
            "\n"
            + level_summary[
                [
                    "qualitative_expected_level",
                    "n_queries",
                    "mean_E",
                    "std_E",
                    "min_E",
                    "max_E",
                    "mean_p_A",
                    "mean_p_B",
                    "mean_p_C",
                ]
            ]
            .to_string(
                index=False
            )
        )

        print(
            "\nSaved:"
        )

        print(
            f"  {results_path}"
        )

        print(
            f"  {summary_path}"
        )

    return results_df


# ==========================================================================
# CLI
# ==========================================================================

def main() -> None:
    """
    Command-line entry point.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Test the Adaptive-RAG-style external query complexity evaluator."
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
        "--model",
        type=str,
        default=DEFAULT_ADAPTIVE_RAG_MODEL,
        help=(
            "Adaptive-RAG-style T5 model identifier. "
            f"Default: {DEFAULT_ADAPTIVE_RAG_MODEL}"
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress detailed per-query output."
        ),
    )

    args = parser.parse_args()

    test_complexity_evaluator(
        config_path=args.config,
        model_name=args.model,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
