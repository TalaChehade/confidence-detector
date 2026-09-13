
"""
Run the live INKER IE-KRT retrieval-trigger experiment.

This script connects all components required for INKER's retrieval-trigger
mechanism:

    1. Adaptive-RAG-style external query-complexity evaluator.

       Current proof-of-concept implementation:

           T5-Large
               +
           4-bit NF4 quantization
               +
           locally trained Adaptive-RAG LoRA adapter

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

External complexity
-------------------
The Adaptive-RAG classifier predicts:

    A = no retrieval
    B = single-step retrieval
    C = multi-step retrieval

For the current INKER replication, these probabilities are converted into
continuous external complexity using:

    E = 0.0 * P(A)
        + 0.5 * P(B)
        + 1.0 * P(C)

which simplifies to:

    E = 0.5 * P(B) + P(C)

The A/B/C -> E mapping is a replication assumption rather than a claim about
an exact continuous calibration formula released by the INKER authors.
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
)

from inker.generation import (
    answer_with_confidence,
)


# =============================================================================
# CLI
# =============================================================================

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
        "--complexity-adapter",
        type=str,
        default=None,
        help=(
            "Optional override for the locally trained Adaptive-RAG "
            "LoRA adapter path. "
            "May point to an adapter directory or ZIP file. "
            "If omitted, complexity.adapter_path is read from the config."
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


# =============================================================================
# Representation reader
# =============================================================================

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
                    f"layer {layer} in '{component}'.\n"
                    "Training and live generation must use the "
                    "same detector layers."
                )

    return rep_reader


# =============================================================================
# Complexity helpers
# =============================================================================

def complexity_to_dict(
    complexity,
) -> Dict[str, float | str]:
    """
    Convert a complexity result into a JSON-safe dictionary.

    This helper avoids requiring ComplexityResult itself to implement
    a to_dict() method.
    """

    return {
        "predicted_class":
            str(
                complexity.predicted_class
            ),

        "p_A":
            float(
                complexity.p_A
            ),

        "p_B":
            float(
                complexity.p_B
            ),

        "p_C":
            float(
                complexity.p_C
            ),

        "E":
            float(
                complexity.E
            ),
    }


def build_complexity_evaluator(
    config: Dict[str, Any],
    adapter_path: str | None = None,
):
    """
    Construct the external Adaptive-RAG complexity evaluator.

    The evaluator is built from:

        t5-large
            +
        optional 4-bit NF4 loading
            +
        locally trained LoRA adapter

    Parameters
    ----------
    config:
        Complete repository configuration.

    adapter_path:
        Optional CLI override for complexity.adapter_path.
    """

    complexity_config = config.get(
        "complexity",
        {},
    )

    configured_adapter = complexity_config.get(
        "adapter_path"
    )

    resolved_adapter = (
        adapter_path
        if adapter_path is not None
        else configured_adapter
    )

    if not resolved_adapter:

        raise ValueError(
            "No Adaptive-RAG LoRA adapter path is configured.\n\n"
            "Either set:\n\n"
            "    complexity:\n"
            "      adapter_path: /path/to/adapter.zip\n\n"
            "inside configs/default.yaml, or pass:\n\n"
            "    --complexity-adapter /path/to/adapter.zip"
        )

    evaluator = (
        AdaptiveRAGComplexityEvaluator.from_config(
            config=config,
            adapter_path=resolved_adapter,
        )
    )

    return evaluator, resolved_adapter


# =============================================================================
# Formatting helpers
# =============================================================================

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


# =============================================================================
# Main
# =============================================================================

def main():
    """
    Run one complete live IE-KRT experiment.
    """

    args = parse_args()

    # -------------------------------------------------------------------------
    # 1. Load configuration
    # -------------------------------------------------------------------------

    config = get_config(
        args.config
    )

    layers = get_detector_layers(
        config
    )

    # -------------------------------------------------------------------------
    # 2. Resolve thresholds and generation settings
    # -------------------------------------------------------------------------

    trigger_config = config.get(
        "trigger",
        {},
    )

    confidence_config = config.get(
        "confidence",
        {},
    )

    generation_config = config.get(
        "generation",
        {},
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

    configured_stop_on_trigger = bool(
        trigger_config.get(
            "stop_on_trigger",
            True,
        )
    )

    stop_on_trigger = (
        False
        if args.no_stop_on_trigger
        else configured_stop_on_trigger
    )

    # -------------------------------------------------------------------------
    # 3. Resolve and validate confidence representation reader
    # -------------------------------------------------------------------------

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
        "Confidence representation reader loaded."
    )

    print(
        f"  {reader_path}"
    )

    # -------------------------------------------------------------------------
    # 4. Load external query-complexity evaluator
    # -------------------------------------------------------------------------
    #
    # Query complexity is question-level and needs to be computed only once.
    # It is therefore calculated before live token generation.
    # -------------------------------------------------------------------------

    complexity_config = config.get(
        "complexity",
        {},
    )

    print(
        "\nLoading query-complexity evaluator..."
    )

    print(
        "  Base model: "
        f"{complexity_config.get('base_model_name', 't5-large')}"
    )

    print(
        "  4-bit: "
        f"{complexity_config.get('load_in_4bit', True)}"
    )

    complexity_evaluator, resolved_adapter = (
        build_complexity_evaluator(
            config=config,
            adapter_path=args.complexity_adapter,
        )
    )

    print(
        f"  Adapter: {resolved_adapter}"
    )

    print(
        "Query-complexity evaluator loaded."
    )

    # -------------------------------------------------------------------------
    # 5. Compute query complexity exactly once
    # -------------------------------------------------------------------------
    #
    # We compute E before generation so:
    #
    #     - A/B/C probabilities can be displayed,
    #     - E can be displayed,
    #     - and the exact same E is reused for every generated token.
    # -------------------------------------------------------------------------

    complexity = (
        complexity_evaluator.evaluate(
            args.question
        )
    )

    print_complexity_result(
        question=args.question,
        complexity=complexity,
    )

    # -------------------------------------------------------------------------
    # 6. Cache complexity for generation.py
    # -------------------------------------------------------------------------

    def complexity_fn(
        _question: str,
        cached_complexity=complexity,
    ):
        """
        Return the already-computed query complexity.

        External complexity E is question-level, so it must not be recomputed
        independently for every generated token.
        """

        return cached_complexity

    # -------------------------------------------------------------------------
    # 7. Load Mistral model/tokenizer
    # -------------------------------------------------------------------------

    print(
        "\nLoading base language model..."
    )

    tokenizer, model = (
        load_configured_model(
            config
        )
    )

    print(
        "Base language model loaded."
    )

    # -------------------------------------------------------------------------
    # 8. Run live IE-KRT generation
    # -------------------------------------------------------------------------

    print(
        "\nStarting live IE-KRT generation..."
    )

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

    # -------------------------------------------------------------------------
    # 9. Attach external complexity explicitly to result
    # -------------------------------------------------------------------------

    result[
        "complexity"
    ] = complexity_to_dict(
        complexity
    )

    result[
        "E"
    ] = float(
        complexity.E
    )

    # -------------------------------------------------------------------------
    # 10. Token-level display
    # -------------------------------------------------------------------------

    print_token_results(
        result=result,
        threshold=trigger_threshold,
    )

    # -------------------------------------------------------------------------
    # 11. Final decision
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # 12. Optional JSON output
    # -------------------------------------------------------------------------

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
                complexity_to_dict(
                    complexity
                ),

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


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()
