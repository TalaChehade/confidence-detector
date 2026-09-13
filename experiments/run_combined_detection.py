"""
Run the live INKER IE-KRT retrieval-trigger experiment.

This experiment combines the two signals used by the INKER retrieval trigger:

    1. External query complexity E
       ---------------------------
       Estimated using a T5-Large Adaptive-RAG-style query-complexity
       classifier adapted with LoRA and loaded in 4-bit NF4.

    2. Internal token confidence m_tilde_i
       -----------------------------------
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

Because generation.py performs token-by-token autoregressive generation,
the trigger is evaluated LIVE.

If stop_on_trigger=True, generation stops when the first retrieval trigger
is detected.

Important
---------
This script evaluates IE-KRT:

    "When should retrieval occur?"

It does NOT yet implement the complete INKER RAG pipeline.

After a trigger it does not yet:

    - formulate the retrieval query,
    - search an external corpus,
    - rerank retrieved evidence,
    - inject evidence into the prompt,
    - or resume generation.

Those operations belong to the IE-KQF and retrieval stages.

External complexity evaluator
-----------------------------
The current proof-of-concept uses:

    T5-Large
        +
    4-bit NF4 base-model loading
        +
    locally trained Adaptive-RAG LoRA adapter

The classifier predicts:

    A = no retrieval
    B = single-step retrieval
    C = multi-step retrieval

The class probabilities are converted into a continuous external
complexity score using the replication assumption:

    A -> 0.0
    B -> 0.5
    C -> 1.0

therefore:

    E = 0.0 * P(A)
        + 0.5 * P(B)
        + 1.0 * P(C)

      = 0.5 * P(B) + P(C)

This mapping is a replication assumption and should not be presented
as an exact formula released by the INKER authors.
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
)

from inker.generation import (
    answer_with_confidence,
)


# ==========================================================================
# Reproducibility
# ==========================================================================

def set_random_seed(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    with reader_path.open("rb") as file:
        rep_reader = pickle.load(file)

    if not isinstance(rep_reader, dict):
        raise TypeError(
            "Representation reader must be a dictionary."
        )

    required_components = {
        "directions",
        "H_train_means",
        "signs",
    }

    missing = required_components - set(rep_reader.keys())

    if missing:
        raise KeyError(
            "Representation reader is missing required components: "
            f"{sorted(missing)}"
        )

    for layer in layers:
        for component in required_components:
            if layer not in rep_reader[component]:
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
        Requires a question column.

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
        One JSON object or string per line.

    TXT
        One question per non-empty line.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Question file not found: {path}"
        )

    suffix = path.suffix.lower()

    questions: List[str] = []

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    if suffix == ".csv":

        df = pd.read_csv(path)

        if question_column not in df.columns:
            raise KeyError(
                f"CSV does not contain question column "
                f"{question_column!r}. "
                f"Available columns: {list(df.columns)}"
            )

        questions = (
            df[question_column]
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
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(
                "JSON question file must contain a list."
            )

        for item in data:

            if isinstance(item, str):
                questions.append(item)

            elif isinstance(item, dict):

                if question_column not in item:
                    raise KeyError(
                        f"JSON object is missing "
                        f"{question_column!r}."
                    )

                questions.append(
                    str(item[question_column])
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

                line = line.strip()

                if not line:
                    continue

                item = json.loads(line)

                if isinstance(item, str):
                    questions.append(item)

                elif isinstance(item, dict):

                    if question_column not in item:
                        raise KeyError(
                            f"Line {line_number}: missing "
                            f"{question_column!r}."
                        )

                    questions.append(
                        str(item[question_column])
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
    # Final cleanup
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
    config: dict,
    adapter_path: str | None = None,
):
    """
    Load the T5-Large + 4-bit + LoRA Adaptive-RAG complexity evaluator.

    Parameters
    ----------
    config:
        Repository YAML configuration.

    adapter_path:
        Optional CLI override.

        If omitted, complexity.adapter_path is read from the config.

    Returns
    -------
    evaluator:
        AdaptiveRAGComplexityEvaluator instance.

    complexity_fn:
        Callable accepted by generation.answer_with_confidence().
    """

    evaluator = (
        AdaptiveRAGComplexityEvaluator.from_config(
            config=config,
            adapter_path=adapter_path,
        )
    )

    def complexity_fn(
        question: str,
    ):
        return evaluator.evaluate(
            question
        )

    return evaluator, complexity_fn


# ==========================================================================
# Evaluation
# ==========================================================================

def evaluate_questions(
    questions: Sequence[str],
    tokenizer: Any,
    model: Any,
    rep_reader: Dict[str, Any],
    layers: Sequence[int],
    complexity_evaluator: AdaptiveRAGComplexityEvaluator,
    trigger_threshold: float,
    confidence_threshold: float,
    max_new_tokens: int,
    repetition_penalty: float,
    stop_on_trigger: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run live IE-KRT generation over a collection of questions.

    For each question:

        1. Compute Adaptive-RAG complexity once.

        2. Obtain:
               P(A)
               P(B)
               P(C)
               E

        3. Cache that complexity result.

        4. Generate tokens using Mistral.

        5. For every generated token compute:
               m_i
               m_tilde_i
               s_i
               K(t_i)

        6. Trigger when:
               K(t_i) > tau
    """

    results: List[Dict[str, Any]] = []

    total = len(questions)

    for index, question in enumerate(
        questions,
        start=1,
    ):

        if verbose:
            print(
                f"\n[{index}/{total}] {question}"
            )

        # --------------------------------------------------------------
        # External complexity is question-level.
        #
        # Compute it ONCE rather than re-running T5 during token
        # generation.
        # --------------------------------------------------------------

        complexity = complexity_evaluator.evaluate(
            question
        )

        if verbose:

            print(
                f"    Complexity class = "
                f"{complexity.predicted_class}"
            )

            print(
                f"    P(A) = {complexity.p_A:.4f}"
            )

            print(
                f"    P(B) = {complexity.p_B:.4f}"
            )

            print(
                f"    P(C) = {complexity.p_C:.4f}"
            )

            print(
                f"    E = {complexity.E:.4f}"
            )

        # --------------------------------------------------------------
        # Cache result.
        #
        # answer_with_confidence expects a callable, but E is constant
        # for the entire generated answer.
        # --------------------------------------------------------------

        def complexity_fn(
            _question: str,
            cached=complexity,
        ):
            return cached

        # --------------------------------------------------------------
        # Live token-by-token IE-KRT generation.
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Ensure the complete complexity result is saved.
        # --------------------------------------------------------------

        result["complexity"] = (
            complexity.to_dict()
        )

        result["E"] = float(
            complexity.E
        )

        results.append(result)

        if verbose:

            print(
                "    Retrieval triggered = "
                f"{bool(result.get('triggered', False))}"
            )

            trigger_token = result.get(
                "trigger_token"
            )

            if trigger_token is not None:

                print(
                    "    Trigger token = "
                    f"{trigger_token!r}"
                )

    return results


# ==========================================================================
# Result helpers
# ==========================================================================

def _get_token_entries(
    result: Dict[str, Any],
) -> list:
    """
    Retrieve token-level records from a generation result.

    Temporary compatibility is retained for:

        token_entries
        token_history
    """

    if "token_entries" in result:
        return result["token_entries"]

    if "token_history" in result:
        return result["token_history"]

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
        and np.isfinite(float(value))
    ]

    if not values:
        return float("nan")

    return float(
        np.mean(values)
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
        and np.isfinite(float(value))
    ]

    if not values:
        return float("nan")

    return float(
        np.min(values)
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
        and np.isfinite(float(value))
    ]

    if not values:
        return float("nan")

    return float(
        np.max(values)
    )


# ==========================================================================
# Result conversion
# ==========================================================================

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

        token_entries = _get_token_entries(
            result
        )

        m_tilde_values = [
            entry.get("m_tilde")
            for entry in token_entries
        ]

        K_values = [
            entry.get("K")
            for entry in token_entries
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

        complexity_result = result.get(
            "complexity"
        )

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

        trigger_token = result.get(
            "trigger_token"
        )

        trigger_index = result.get(
            "trigger_index"
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
                    result["E"]
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
        # Token-level rows
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
                        result["E"]
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
            list(results),
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
    # Complexity
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
    # Complexity class counts
    # ------------------------------------------------------------------

    if "complexity_class" in question_df.columns:

        print(
            "\nAdaptive-RAG predicted classes:"
        )

        counts = (
            question_df[
                "complexity_class"
            ]
            .value_counts(
                dropna=False
            )
        )

        for label, count in counts.items():

            print(
                f"  {label}: {count}"
            )

    # ------------------------------------------------------------------
    # Retrieval triggers
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
    # Token statistics
    # ------------------------------------------------------------------

    if not token_df.empty:

        content_tokens = token_df[
            token_df["s_i"] == 1
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

            # K > 0 does NOT mean retrieval.
            #
            # Retrieval specifically requires:
            #
            #     K > tau

            above_threshold = (
                content_tokens["K"]
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
    complexity_adapter: str | None = None,
) -> None:
    """
    Run the live INKER retrieval-trigger experiment.
    """

    # ------------------------------------------------------------------
    # 1. Configuration and seed
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
    # 2. Detector configuration
    # ------------------------------------------------------------------

    layers = get_detector_layers(
        config
    )

    trigger_config = config.get(
        "trigger",
        {},
    )

    confidence_config = config.get(
        "confidence",
        {},
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

    generation_config = config.get(
        "generation",
        {},
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

    stop_on_trigger = bool(
        trigger_config.get(
            "stop_on_trigger",
            True,
        )
    )

    # ------------------------------------------------------------------
    # 3. Load questions FIRST
    #
    # This is done before loading the large models so bad input paths are
    # detected immediately.
    # ------------------------------------------------------------------

    if question is not None:

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "--question cannot be empty."
            )

        questions = [
            cleaned_question
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
        f"\nQuestions to evaluate: "
        f"{len(questions)}"
    )

    # ------------------------------------------------------------------
    # 4. Load Mistral
    # ------------------------------------------------------------------

    print(
        "\nLoading base language model..."
    )

    tokenizer, model = (
        load_configured_model(
            config
        )
    )

    # ------------------------------------------------------------------
    # 5. Load confidence representation reader
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

    print(
        "✓ Confidence representation reader loaded."
    )

    # ------------------------------------------------------------------
    # 6. Load Adaptive-RAG complexity evaluator
    # ------------------------------------------------------------------

    complexity_config = config.get(
        "complexity",
        {},
    )

    configured_adapter = (
        complexity_config.get(
            "adapter_path"
        )
    )

    resolved_adapter = (
        complexity_adapter
        if complexity_adapter is not None
        else configured_adapter
    )

    if not resolved_adapter:
        raise ValueError(
            "No Adaptive-RAG LoRA adapter configured.\n\n"
            "Set:\n"
            "  complexity.adapter_path\n\n"
            "in configs/default.yaml or pass:\n"
            "  --complexity-adapter PATH"
        )

    print(
        "\nLoading query-complexity evaluator..."
    )

    print(
        f"  Base model: "
        f"{complexity_config.get('base_model_name', 't5-large')}"
    )

    print(
        f"  Adapter: {resolved_adapter}"
    )

    print(
        f"  4-bit: "
        f"{complexity_config.get('load_in_4bit', True)}"
    )

    complexity_evaluator, _ = (
        build_complexity_evaluator(
            config=config,
            adapter_path=resolved_adapter,
        )
    )

    # ------------------------------------------------------------------
    # 7. Live IE-KRT evaluation
    # ------------------------------------------------------------------

    print(
        f"\nEvaluating "
        f"{len(questions)} questions..."
    )

    results = evaluate_questions(
        questions=questions,
        tokenizer=tokenizer,
        model=model,
        rep_reader=rep_reader,
        layers=layers,
        complexity_evaluator=complexity_evaluator,
        trigger_threshold=trigger_threshold,
        confidence_threshold=confidence_threshold,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        stop_on_trigger=stop_on_trigger,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 8. Save
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
    # 9. Summary
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
            "confidence + Adaptive-RAG complexity retrieval trigger."
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
        "--complexity-adapter",
        type=str,
        default=None,
        help=(
            "Optional override for the Adaptive-RAG LoRA adapter. "
            "May point to an extracted adapter folder or ZIP. "
            "If omitted, complexity.adapter_path is read from config."
        ),
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        questions_path=args.questions,
        question=args.question,
        num_questions=args.num_questions,
        complexity_adapter=args.complexity_adapter,
    )
