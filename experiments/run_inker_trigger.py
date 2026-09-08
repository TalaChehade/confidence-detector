# experiments/run_inker_trigger.py

"""
Small end-to-end test of the INKER retrieval-trigger mechanism.

Current purpose:

1. Load the pretrained Adaptive-RAG complexity evaluator.
2. Evaluate a question.
3. Obtain P(A), P(B), P(C).
4. Convert those probabilities into continuous E.
5. Use example token confidence scores.
6. Compute:

       K(t_i) = (E - m_tilde_i) * s_i

7. Determine which tokens would trigger retrieval.

Later, the example confidence values in this script will be replaced by the
real confidence scores produced by the repository's confidence detector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inker.complexity import AdaptiveRAGComplexityEvaluator
from inker.trigger import (
    evaluate_sequence_activation,
    sequence_triggers_retrieval,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Run the INKER retrieval-trigger test."
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to evaluate.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Retrieval activation threshold.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="LenckCuak/Adaptive-RAG",
        help="Adaptive-RAG T5 complexity model.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # ------------------------------------------------------------------
    # 1. External query-complexity evaluator
    # ------------------------------------------------------------------

    evaluator = AdaptiveRAGComplexityEvaluator(
        model_name=args.model
    )

    complexity = evaluator.evaluate(
        args.question
    )

    # ------------------------------------------------------------------
    # 2. Display complexity result
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("QUERY COMPLEXITY")
    print("=" * 80)

    print(f"\nQuestion:\n{args.question}")

    print(
        f"\nPredicted Adaptive-RAG class: "
        f"{complexity.predicted_class}"
    )

    print("\nClass probabilities:")
    print(f"  P(A) = {complexity.p_A:.6f}")
    print(f"  P(B) = {complexity.p_B:.6f}")
    print(f"  P(C) = {complexity.p_C:.6f}")

    print("\nReplication conversion:")
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

    # ------------------------------------------------------------------
    # 3. Temporary example confidence information
    #
    # THESE WILL LATER BE REPLACED BY THE REAL OUTPUT FROM
    # YOUR CONFIDENCE DETECTOR.
    # ------------------------------------------------------------------

    tokens = [
        "The",
        " Olympics",
        " were",
        " hosted",
        " by",
        " France",
    ]

    normalized_confidences = [
        0.72,
        0.81,
        0.63,
        0.58,
        0.71,
        0.31,
    ]

    content_masks = [
        0,
        1,
        0,
        1,
        0,
        1,
    ]

    # ------------------------------------------------------------------
    # 4. Compute INKER activations
    # ------------------------------------------------------------------

    activations = evaluate_sequence_activation(
        tokens=tokens,
        confidences=normalized_confidences,
        content_masks=content_masks,
        E=complexity.E,
        threshold=args.threshold,
    )

    # ------------------------------------------------------------------
    # 5. Print token-level results
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TOKEN-LEVEL INKER ACTIVATION")
    print("=" * 80)

    print(
        f"\nThreshold tau = {args.threshold}\n"
    )

    header = (
        f"{'Token':<20}"
        f"{'m_tilde':>12}"
        f"{'s_i':>8}"
        f"{'K(t_i)':>14}"
        f"{'Trigger':>12}"
    )

    print(header)
    print("-" * len(header))

    for item in activations:

        print(
            f"{repr(item.token):<20}"
            f"{item.confidence:>12.4f}"
            f"{item.content_mask:>8d}"
            f"{item.activation:>14.4f}"
            f"{str(item.triggered):>12}"
        )

    retrieval_triggered = sequence_triggers_retrieval(
        activations
    )

    print("\n" + "-" * 80)

    print(
        "Final retrieval decision: "
        + (
            "RETRIEVE"
            if retrieval_triggered
            else "DO NOT RETRIEVE"
        )
    )

    # ------------------------------------------------------------------
    # 6. Optional JSON output
    # ------------------------------------------------------------------

    if args.output is not None:

        output = {
            "question": args.question,
            "complexity": complexity.to_dict(),
            "threshold": args.threshold,
            "tokens": [
                item.to_dict()
                for item in activations
            ],
            "retrieval_triggered": retrieval_triggered,
        }

        output_path = Path(args.output)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                output,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"\nSaved results to: "
            f"{output_path}"
        )


if __name__ == "__main__":
    main()
