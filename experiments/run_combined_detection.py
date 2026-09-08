"""
Run the live INKER IE-KRT retrieval-trigger experiment.

This experiment combines the two signals used by the INKER retrieval trigger:

    1. External query complexity E
       --------------------------------
       Estimated using an Adaptive-RAG-style T5-Large complexity evaluator.

    2. Internal token confidence m_tilde_i
       ------------------------------------
       Estimated from Mistral hidden representations using the trained
       confidence representation reader.

For every generated token t_i:

    K(t_i) = (E - m_tilde_i) * s_i

where:

    E
        Query-level external complexity score.

    m_tilde_i
        Causally normalized internal confidence.

    s_i
        Binary content-token mask.

Retrieval is triggered when:

    K(t_i) > tau

for the first qualifying generated token.

Because ``generation.py`` performs token-by-token autoregressive generation,
the trigger is evaluated LIVE. If ``stop_on_trigger=True``, generation stops
as soon as the first retrieval trigger is detected.

Important
---------
This script evaluates IE-KRT:

    "When should retrieval occur?"

It does NOT yet implement the complete INKER RAG pipeline.

Specifically, after a trigger this experiment does not yet:

    - formulate the retrieval query,
    - search an external corpus,
    - rerank retrieved evidence,
    - inject evidence into the prompt,
    - or resume/restart generation.

Those operations belong to the IE-KQF and retrieval portions of INKER.

Complexity evaluator
--------------------
The primary replication uses the pretrained third-party Adaptive-RAG
T5-Large reproduction:

    LenckCuak/Adaptive-RAG

rather than retraining Adaptive-RAG from scratch.

The conversion from A/B/C probabilities into continuous query complexity E
is the replication assumption implemented in ``inker.complexity``:

    A -> 0.0
    B -> 0.5
    C -> 1.0

and:

    E = 0.0 P(A) + 0.5 P(B) + 1.0 P(C)

      = 0.5 P(B) + P(C)

This assumption must be documented when reporting replication results.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
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
# Reproducibility
# ==========================================================================

def set_random_seed(
    seed: int,
) -> None:
    """
    Seed Python, NumPy, and PyTorch.
    """

    random.seed(
        seed
    )

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
                    f"layer {layer} in '{component}'.\n"
                    "Training and live evaluation must use the same "
                    "detector layers."
                )

    return rep_reader


# ==========================================================================
# Questions
# ==========================================================================

def load_questions_from_file(
    path: Path,
    question_column: str = "question",
) -> List[str]:
    """
    Load evaluation questions from CSV, JSON, JSONL, or TXT.

    Supported formats
    -----------------
    CSV
        Requires a column such as:

            question

    JSON
        Either:

            [
                {"question": "..."},
                ...
            ]

        or:

            [
                "question one",
                "question two"
            ]

    JSONL
        One JSON object per line.

    TXT
        One question per non-empty line.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Question file not found: {path}"
        )

    suffix = (
        path.suffix
        .lower()
    )

    questions: List[str] = []

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    if suffix == ".csv":

        df = pd.read_csv(
            path
        )

        if question_column not in df.columns:

            raise KeyError(
                f"CSV does not contain question column "
                f"{question_column!r}. "
                f"Available columns: {list(df.columns)}"
            )

        questions = (
            df[
                question_column
            ]
            .dropna()
            .astype(str)
            .tolist()
        )

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

                questions.append(
                    item
                )

            elif isinstance(
                item,
                dict,
            ):

                if question_column not in item:

                    raise KeyError(
                        f"JSON object is missing "
                        f"{question_column!r}."
                    )

                questions.append(
                    str(
                        item[
                            question_column
                        ]
                    )
                )

            else:

                raise TypeError(
                    "JSON question entries must be strings "
                    "or dictionaries."
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

                line = (
                    line.strip()
                )

                if not line:
                    continue

                item = json.loads(
                    line
                )

                if isinstance(
                    item,
                    str,
                ):

                    questions.append(
                        item
                    )

                elif isinstance(
                    item,
                    dict,
                ):

                    if question_column not in item:

                        raise KeyError(
                            f"Line {line_number}: missing "
                            f"{question_column!r}."
                        )

                    questions.append(
                        str(
                            item[
                                question_column
                            ]
                        )
                    )

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

            questions = [
                line.strip()
                for line in file
                if line.strip()
            ]

    else:

        raise ValueError(
            "Unsupported question-file format. "
            "Supported extensions are: "
            ".csv, .json, .jsonl, .txt"
        )

    # ------------------------------------------------------------------
    # Final cleanup.
    # ------------------------------------------------------------------

    questions = [
        question.strip()
        for question in questions
        if question.strip()
    ]

    if not questions:

        raise ValueError(
            f"No usable questions found in {path}."
        )

    return questions


# ==========================================================================
# Complexity evaluator
# ==========================================================================

def build_complexity_evaluator(
    model_name: str = DEFAULT_ADAPTIVE_RAG_MODEL,
):
    """
    Load the Adaptive-RAG-style query complexity evaluator.

    Returns
    -------
    evaluator
        ``AdaptiveRAGComplexityEvaluator`` instance.

    complexity_fn
        Callable accepted by ``generation.answer_with_confidence``.

        The callable receives a question and returns the complexity result
        produced by the evaluator.
    """

    evaluator = (
        AdaptiveRAGComplexityEvaluator(
            model_name=model_name
        )
    )

    def complexity_fn(
        question: str,
    ):
        return evaluator.evaluate(
            question
        )

    return (
        evaluator,
        complexity_fn,
    )


# ==========================================================================
# Evaluation
# ==========================================================================

def evaluate_questions(
    questions: Sequence[str],
    tokenizer: Any,
    model: Any,
    rep_reader: Dict[str, Any],
    layers: Sequence[int],
    complexity_fn,
    trigger_threshold: float,
    confidence_threshold: float,
    max_new_tokens: int,
    repetition_penalty: float,
    stop_on_trigger: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run live IE-KRT generation over a collection of questions.

    Each question receives:

        E
            query complexity

        m_i
            raw internal confidence per generated token

        m_tilde_i
            causal normalized confidence

        s_i
            token content mask

        K(t_i)
            retrieval activation

        triggered
            whether retrieval fired
    """

    results: List[
        Dict[str, Any]
    ] = []

    total = len(
        questions
    )

    for index, question in enumerate(
        questions,
        start=1,
    ):

        if verbose:

            print(
                f"\n[{index}/{total}] "
                f"{question}"
            )

        result = answer_with_confidence(
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

        results.append(
            result
        )

        if verbose:

            print(
                f"    E = "
                f"{float(result['E']):.4f}"
            )

            print(
                "    Retrieval triggered = "
                f"{bool(result.get('triggered', False))}"
            )

            trigger_token = (
                result.get(
                    "trigger_token"
                )
            )

            if trigger_token is not None:

                print(
                    "    Trigger token = "
                    f"{trigger_token!r}"
                )

    return results


# ==========================================================================
# Result conversion
# ==========================================================================

def _get_token_entries(
    result: Dict[str, Any],
) -> list:
    """
    Retrieve token-level records from a generation result.

    ``generation.py`` may expose them as ``token_entries`` or
    ``token_history`` depending on the final repository naming.

    This small compatibility helper prevents CSV export from depending on
    only one of those names.
    """

    if "token_entries" in result:

        return result[
            "token_entries"
        ]

    if "token_history" in result:

        return result[
            "token_history"
        ]

    return []


def _safe_mean(
    values: Iterable[float],
) -> float:
    """
    Mean of a possibly empty sequence.
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


def _safe_min(
    values: Iterable[float],
) -> float:
    """
    Minimum of a possibly empty sequence.
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


def _safe_max(
    values: Iterable[float],
) -> float:
    """
    Maximum of a possibly empty sequence.
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


def results_to_dataframes(
    results: Sequence[Dict[str, Any]],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Convert live generation results into question-level and token-level tables.
    """

    question_rows = []
    token_rows = []

    for question_index, result in enumerate(
        results
    ):

        token_entries = (
            _get_token_entries(
                result
            )
        )

        m_tilde_values = [
            entry.get(
                "m_tilde"
            )
            for entry
            in token_entries
        ]

        K_values = [
            entry.get(
                "K"
            )
            for entry
            in token_entries
        ]

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

        complexity_result = (
            result.get(
                "complexity"
            )
        )

        # --------------------------------------------------------------
        # Complexity class/probabilities may either be available in a
        # nested dictionary or only as E.
        # --------------------------------------------------------------

        complexity_class = None
        p_A = np.nan
        p_B = np.nan
        p_C = np.nan

        if isinstance(
            complexity_result,
            dict,
        ):

            complexity_class = (
                complexity_result.get(
                    "predicted_class"
                )
            )

            p_A = complexity_result.get(
                "p_A",
                np.nan,
            )

            p_B = complexity_result.get(
                "p_B",
                np.nan,
            )

            p_C = complexity_result.get(
                "p_C",
                np.nan,
            )

        triggered = bool(
            result.get(
                "triggered",
                result.get(
                    "would_trigger_full",
                    False,
                ),
            )
        )

        trigger_token = (
            result.get(
                "trigger_token"
            )
        )

        trigger_index = (
            result.get(
                "trigger_index"
            )
        )

        question_rows.append({
            "question_index":
                question_index,

            "question":
                result.get(
                    "question",
                    "",
                ),

            "generated_text":
                result.get(
                    "answer_text",
                    result.get(
                        "generated_text",
                        "",
                    ),
                ),

            "complexity_class":
                complexity_class,

            "p_A":
                p_A,

            "p_B":
                p_B,

            "p_C":
                p_C,

            "E":
                float(
                    result[
                        "E"
                    ]
                ),

            "n_generated_tokens":
                len(
                    token_entries
                ),

            "n_content_tokens":
                len(
                    content_entries
                ),

            "mean_m_tilde":
                _safe_mean(
                    m_tilde_values
                ),

            "min_m_tilde":
                _safe_min(
                    m_tilde_values
                ),

            "max_m_tilde":
                _safe_max(
                    m_tilde_values
                ),

            "max_K":
                _safe_max(
                    K_values
                ),

            "retrieval_triggered":
                triggered,

            "trigger_token":
                trigger_token,

            "trigger_index":
                trigger_index,
        })

        # --------------------------------------------------------------
        # Token-level rows.
        # --------------------------------------------------------------

        for fallback_index, entry in enumerate(
            token_entries
        ):

            token_rows.append({
                "question_index":
                    question_index,

                "question":
                    result.get(
                        "question",
                        "",
                    ),

                "E":
                    float(
                        result[
                            "E"
                        ]
                    ),

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
                            "m_i"
                        ),
                    ),

                "m_tilde":
                    entry.get(
                        "m_tilde"
                    ),

                "s_i":
                    int(
                        entry.get(
                            "s_i",
                            0,
                        )
                    ),

                "K":
                    entry.get(
                        "K"
                    ),

                "triggered":
                    bool(
                        entry.get(
                            "triggered",
                            False,
                        )
                    ),
            })

    question_df = pd.DataFrame(
        question_rows
    )

    token_df = pd.DataFrame(
        token_rows
    )

    return (
        question_df,
        token_df,
    )


# ==========================================================================
# Saving
# ==========================================================================

def save_results(
    results: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Save question-level, token-level, and complete JSON results.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    question_df, token_df = (
        results_to_dataframes(
            results
        )
    )

    question_path = (
        output_dir
        / "inker_trigger_questions.csv"
    )

    token_path = (
        output_dir
        / "inker_trigger_tokens.csv"
    )

    json_path = (
        output_dir
        / "inker_trigger_results.json"
    )

    question_df.to_csv(
        question_path,
        index=False,
    )

    token_df.to_csv(
        token_path,
        index=False,
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            list(
                results
            ),
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return (
        question_df,
        token_df,
    )


# ==========================================================================
# Summary
# ==========================================================================

def print_summary(
    question_df: pd.DataFrame,
    token_df: pd.DataFrame,
    trigger_threshold: float,
) -> None:
    """
    Print high-level IE-KRT experiment statistics.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "INKER IE-KRT LIVE TRIGGER SUMMARY"
    )

    print(
        "=" * 80
    )

    n_questions = len(
        question_df
    )

    print(
        f"\nQuestions evaluated: "
        f"{n_questions}"
    )

    print(
        f"Generated/scored tokens: "
        f"{len(token_df)}"
    )

    print(
        f"Retrieval threshold tau: "
        f"{trigger_threshold:.4f}"
    )

    if n_questions == 0:

        print(
            "\nNo questions were evaluated."
        )

        return

    # ------------------------------------------------------------------
    # Complexity.
    # ------------------------------------------------------------------

    print(
        "\nComplexity score E:"
    )

    print(
        f"  Mean: "
        f"{question_df['E'].mean():.4f}"
    )

    print(
        f"  Std:  "
        f"{question_df['E'].std():.4f}"
    )

    print(
        f"  Min:  "
        f"{question_df['E'].min():.4f}"
    )

    print(
        f"  Max:  "
        f"{question_df['E'].max():.4f}"
    )

    # ------------------------------------------------------------------
    # Retrieval triggers.
    # ------------------------------------------------------------------

    n_triggered = int(
        question_df[
            "retrieval_triggered"
        ].sum()
    )

    trigger_rate = (
        n_triggered
        / n_questions
    )

    print(
        "\nRetrieval triggers:"
    )

    print(
        f"  Triggered: "
        f"{n_triggered}/{n_questions} "
        f"({100.0 * trigger_rate:.1f}%)"
    )

    print(
        f"  Not triggered: "
        f"{n_questions - n_triggered}/{n_questions}"
    )

    # ------------------------------------------------------------------
    # Token statistics.
    # ------------------------------------------------------------------

    if not token_df.empty:

        content_tokens = token_df[
            token_df[
                "s_i"
            ] == 1
        ]

        print(
            "\nToken-level statistics:"
        )

        print(
            f"  Total tokens: "
            f"{len(token_df)}"
        )

        print(
            f"  Content tokens: "
            f"{len(content_tokens)}"
        )

        if not content_tokens.empty:

            print(
                f"  Mean normalized confidence: "
                f"{content_tokens['m_tilde'].mean():.4f}"
            )

            print(
                f"  Mean K(t_i): "
                f"{content_tokens['K'].mean():.4f}"
            )

            print(
                f"  Maximum K(t_i): "
                f"{content_tokens['K'].max():.4f}"
            )

            # Note:
            #
            # K > 0 does NOT mean retrieval.
            # Retrieval specifically requires:
            #
            #     K > tau
            #
            above_threshold = (
                content_tokens[
                    "K"
                ]
                > trigger_threshold
            )

            print(
                f"  Content tokens with K > tau: "
                f"{int(above_threshold.sum())}"
            )

    print(
        "\n"
        + "=" * 80
    )


# ==========================================================================
# Main
# ==========================================================================

def main(
    config_path: str | Path | None = None,
    questions_path: str | Path | None = None,
    question: str | None = None,
    num_questions: int | None = None,
    complexity_model: str = DEFAULT_ADAPTIVE_RAG_MODEL,
) -> None:
    """
    Run the live INKER retrieval-trigger experiment.
    """

    # ------------------------------------------------------------------
    # 1. Configuration and seed.
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

    # ------------------------------------------------------------------
    # 2. Detector configuration.
    # ------------------------------------------------------------------

    layers = get_detector_layers(
        config
    )

    detector_config = config[
        "detector"
    ]

    # ------------------------------------------------------------------
    # Retrieval trigger and confidence-only thresholds are intentionally
    # separate concepts.
    # ------------------------------------------------------------------

    trigger_config = config.get(
        "trigger",
        {}
    )

    confidence_config = config.get(
        "confidence",
        {}
    )

    trigger_threshold = float(
        trigger_config.get(
            "threshold",
            0.5,
        )
    )

    confidence_threshold = float(
        confidence_config.get(
            "threshold",
            0.5,
        )
    )

    generation_config = config[
        "generation"
    ]

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

    stop_on_trigger = bool(
        trigger_config.get(
            "stop_on_trigger",
            True,
        )
    )

    # ------------------------------------------------------------------
    # 3. Load Mistral.
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
    # 4. Load confidence representation reader.
    # ------------------------------------------------------------------

    reader_path = get_project_path(
        config,
        "representation_reader",
    )

    rep_reader = (
        load_representation_reader(
            reader_path=reader_path,
            layers=layers,
        )
    )

    # ------------------------------------------------------------------
    # 5. Load Adaptive-RAG complexity evaluator.
    # ------------------------------------------------------------------

    print(
        "Loading query-complexity evaluator..."
    )

    print(
        f"  {complexity_model}"
    )

    (
        _complexity_evaluator,
        complexity_fn,
    ) = build_complexity_evaluator(
        model_name=complexity_model
    )

    # ------------------------------------------------------------------
    # 6. Load real questions.
    #
    # DO NOT use the confidence-detector contrastive statement dataset
    # here. That dataset exists to train/evaluate the internal detector,
    # not to evaluate query complexity or INKER retrieval behavior.
    # ------------------------------------------------------------------

    if question is not None:

        questions = [
            question.strip()
        ]

    elif questions_path is not None:

        questions = (
            load_questions_from_file(
                Path(
                    questions_path
                )
            )
        )

    else:

        raise ValueError(
            "Provide either:\n"
            "  --question \"...\"\n"
            "or\n"
            "  --questions path/to/questions.csv"
        )

    if num_questions is not None:

        if num_questions <= 0:

            raise ValueError(
                "--num-questions must be greater than zero."
            )

        questions = questions[
            :num_questions
        ]

    print(
        f"\nEvaluating "
        f"{len(questions)} questions..."
    )

    # ------------------------------------------------------------------
    # 7. Live IE-KRT evaluation.
    # ------------------------------------------------------------------

    results = evaluate_questions(
        questions=questions,
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
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 8. Save.
    # ------------------------------------------------------------------

    result_dir = get_project_path(
        config,
        "full_k_results",
    )

    question_df, token_df = (
        save_results(
            results=results,
            output_dir=result_dir,
        )
    )

    # ------------------------------------------------------------------
    # 9. Summary.
    # ------------------------------------------------------------------

    print_summary(
        question_df=question_df,
        token_df=token_df,
        trigger_threshold=trigger_threshold,
    )

    print(
        "\nSaved results to:"
    )

    print(
        f"  {result_dir}"
    )

    print(
        "\nFiles:"
    )

    print(
        "  inker_trigger_questions.csv"
    )

    print(
        "  inker_trigger_tokens.csv"
    )

    print(
        "  inker_trigger_results.json"
    )


# ==========================================================================
# CLI
# ==========================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run the live INKER IE-KRT "
            "confidence + query-complexity retrieval trigger."
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

    input_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    input_group.add_argument(
        "--question",
        type=str,
        default=None,
        help=(
            "Evaluate one question directly."
        ),
    )

    input_group.add_argument(
        "--questions",
        type=str,
        default=None,
        help=(
            "Question file: CSV, JSON, JSONL, or TXT."
        ),
    )

    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help=(
            "Optional limit on number of questions evaluated."
        ),
    )

    parser.add_argument(
        "--complexity-model",
        type=str,
        default=DEFAULT_ADAPTIVE_RAG_MODEL,
        help=(
            "Adaptive-RAG-style T5 model identifier. "
            "Default: LenckCuak/Adaptive-RAG"
        ),
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        questions_path=args.questions,
        question=args.question,
        num_questions=args.num_questions,
        complexity_model=args.complexity_model,
    )
