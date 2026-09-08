"""
Run the live INKER IE-KRT retrieval-trigger experiment.

This script connects all components required for INKER's retrieval-trigger
mechanism:

    1. Adaptive-RAG-style external query-complexity evaluator.
    2. Trained internal confidence representation reader.
    3. Live token-by-token Mistral generation.
    4. Causal confidence normalization.
    5. Content-token masking.
    6. INKER activation:

           K(t_i) = (E - m_tilde_i) * s_i

    7. Retrieval decision:

           K(t_i) > tau

Unlike the earlier prototype version of this script, no artificial token
confidence values are used. Confidence is obtained directly from the trained
representation reader during live autoregressive generation.

Important
---------
This experiment implements IE-KRT:

    "When should external retrieval be triggered?"

It does NOT yet implement the complete RAG pipeline after triggering.

In particular, this script does not yet:

    - formulate a retrieval query,
    - search an external corpus,
    - rerank retrieved documents,
    - inject retrieved evidence,
    - or resume generation after retrieval.

Those operations belong to the later IE-KQF / retrieval integration stage.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Sequence

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
# CLI
# ==========================================================================

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run the live INKER IE-KRT retrieval-trigger experiment."
        )
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to evaluate.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML configuration file. "
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
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional retrieval threshold override. "
            "If omitted, config['trigger']['threshold'] is used."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Optional generation-length override."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional JSON output path."
        ),
    )

    parser.add_argument(
        "--no-stop-on-trigger",
        action="store_true",
        help=(
            "Continue generation even after a trigger. "
            "By default, live generation stops at the first trigger."
        ),
    )

    return parser.parse_args()


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
# Formatting helpers
# ==========================================================================

def print_complexity_result(
    question: str,
    complexity,
) -> None:
    """
    Print the external query-complexity result.
    """

    print(
        "\n"
        + "=" * 80
    )

    print(
        "QUERY COMPLEXITY"
    )

    print(
        "=" * 80
    )

    print(
        f"\nQuestion:\n{question}"
    )

    print(
        "\nAdaptive-RAG class:"
    )

    print(
        f"  {complexity.predicted_class}"
    )

    print(
        "\nClass probabilities:"
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
        "\nReplication conversion:"
    )

    print(
        "  A -> 0.0"
    )

    print(
        "  B -> 0.5"
    )

    print(
        "  C -> 1.0"
    )

    print(
        "\nTherefore:"
    )

    print(
        "  E = 0.0*P(A) + 0.5*P(B) + 1.0*P(C)"
    )

    print(
        f"  E = 0.5*{complexity.p_B:.6f} "
        f"+ {complexity.p_C:.6f}"
    )

    print(
        f"  E = {complexity.E:.6f}"
    )


def print_token_results(
    result: Dict[str, Any],
    threshold: float,
) -> None:
    """
    Print live token-level confidence and retrieval activation.
    """

    token_entries = result.get(
        "token_entries",
        result.get(
            "token_history",
            [],
        ),
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LIVE TOKEN-LEVEL IE-KRT"
    )

    print(
        "=" * 80
    )

    print(
        f"\nRetrieval threshold tau = {threshold:.4f}\n"
    )

    header = (
        f"{'Idx':>5}"
        f"{'Token':<24}"
        f"{'m_i':>12}"
        f"{'m_tilde':>12}"
        f"{'s_i':>8}"
        f"{'K(t_i)':>12}"
        f"{'Trigger':>12}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for fallback_index, entry in enumerate(
        token_entries
    ):

        token_index = entry.get(
            "token_index",
            fallback_index,
        )

        token = entry.get(
            "token",
            "",
        )

        raw_confidence = entry.get(
            "raw_confidence",
            entry.get(
                "raw_score",
                entry.get(
                    "m_i"
                ),
            ),
        )

        m_tilde = entry.get(
            "m_tilde"
        )

        s_i = int(
            entry.get(
                "s_i",
                0,
            )
        )

        K = entry.get(
            "K"
        )

        triggered = bool(
            entry.get(
                "triggered",
                False,
            )
        )

        raw_text = (
            f"{float(raw_confidence):.4f}"
            if raw_confidence is not None
            else "N/A"
        )

        normalized_text = (
            f"{float(m_tilde):.4f}"
            if m_tilde is not None
            else "N/A"
        )

        K_text = (
            f"{float(K):.4f}"
            if K is not None
            else "N/A"
        )

        print(
            f"{token_index:>5}"
            f"{repr(token):<24}"
            f"{raw_text:>12}"
            f"{normalized_text:>12}"
            f"{s_i:>8d}"
            f"{K_text:>12}"
            f"{str(triggered):>12}"
        )


# ==========================================================================
# Main
# ==========================================================================

def main():
    """
    Run one complete live IE-KRT experiment.
    """

    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load configuration.
    # ------------------------------------------------------------------

    config = get_config(
        args.config
    )

    layers = get_detector_layers(
        config
    )

    # ------------------------------------------------------------------
    # 2. Resolve thresholds and generation settings.
    # ------------------------------------------------------------------

    trigger_config = config.get(
        "trigger",
        {}
    )

    confidence_config = config.get(
        "confidence",
        {}
    )

    generation_config = config.get(
        "generation",
        {}
    )

    trigger_threshold = (
        float(
            args.threshold
        )
        if args.threshold is not None
        else float(
            trigger_config.get(
                "threshold",
                0.5,
            )
        )
    )

    confidence_threshold = float(
        confidence_config.get(
            "threshold",
            0.5,
        )
    )

    max_new_tokens = (
        int(
            args.max_new_tokens
        )
        if args.max_new_tokens is not None
        else int(
            generation_config.get(
                "max_new_tokens",
                60,
            )
        )
    )

    repetition_penalty = float(
        generation_config.get(
            "repetition_penalty",
            1.1,
        )
    )

    stop_on_trigger = (
        not args.no_stop_on_trigger
    )

    # ------------------------------------------------------------------
    # 3. Load Mistral model/tokenizer.
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
    # 4. Load trained confidence detector.
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
        f"  {args.complexity_model}"
    )

    complexity_evaluator = (
        AdaptiveRAGComplexityEvaluator(
            model_name=args.complexity_model
        )
    )

    # ------------------------------------------------------------------
    # 6. Compute query complexity explicitly once.
    #
    # We evaluate it here so that:
    #
    #     - the A/B/C probabilities can be displayed,
    #     - E can be printed,
    #     - and the same exact E is then reused during generation.
    # ------------------------------------------------------------------

    complexity = (
        complexity_evaluator.evaluate(
            args.question
        )
    )

    print_complexity_result(
        question=args.question,
        complexity=complexity,
    )

    # ------------------------------------------------------------------
    # 7. Complexity callable used by generation.py.
    #
    # Since complexity has already been evaluated, return the cached result.
    #
    # This avoids running T5 twice for the same question.
    # ------------------------------------------------------------------

    def complexity_fn(
        _question: str,
    ):
        return complexity

    # ------------------------------------------------------------------
    # 8. Run live token-by-token generation and confidence detection.
    # ------------------------------------------------------------------

    result = (
        answer_with_confidence(
            question=args.question,
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

    # ------------------------------------------------------------------
    # 9. Token-level display.
    # ------------------------------------------------------------------

    print_token_results(
        result=result,
        threshold=trigger_threshold,
    )

    # ------------------------------------------------------------------
    # 10. Final decision.
    # ------------------------------------------------------------------

    retrieval_triggered = bool(
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

    answer_text = result.get(
        "answer_text",
        result.get(
            "generated_text",
            "",
        ),
    )

    print(
        "\n"
        + "-" * 80
    )

    print(
        "\nGenerated text:"
    )

    print(
        answer_text
    )

    print(
        "\nFinal retrieval decision: "
        + (
            "RETRIEVE"
            if retrieval_triggered
            else "DO NOT RETRIEVE"
        )
    )

    if retrieval_triggered:

        print(
            f"Trigger token: "
            f"{trigger_token!r}"
        )

        print(
            f"Trigger index: "
            f"{trigger_index}"
        )

    # ------------------------------------------------------------------
    # 11. Optional JSON output.
    # ------------------------------------------------------------------

    if args.output is not None:

        output_path = Path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = {
            "question":
                args.question,

            "complexity":
                complexity.to_dict(),

            "trigger_threshold":
                trigger_threshold,

            "confidence_threshold":
                confidence_threshold,

            "stop_on_trigger":
                stop_on_trigger,

            "result":
                result,
        }

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                output,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        print(
            f"\nSaved complete result to:\n"
            f"  {output_path}"
        )


# ==========================================================================
# Entry point
# ==========================================================================

if __name__ == "__main__":
    main()
