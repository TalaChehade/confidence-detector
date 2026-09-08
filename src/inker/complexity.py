# src/inker/complexity.py

"""
Query-complexity evaluator used by the INKER replication.

The original Adaptive-RAG classifier predicts one of three query-complexity
classes:

    A -> no retrieval
    B -> single-step retrieval
    C -> multi-step retrieval

For the INKER replication, we need a continuous external complexity score E
that can be compared with the normalized internal confidence score.

IMPORTANT:
The INKER paper does not clearly specify the exact conversion from the
Adaptive-RAG-style A/B/C output to continuous E.

We therefore use the following explicit replication assumption:

    A -> 0.0
    B -> 0.5
    C -> 1.0

and compute the probability-weighted expected complexity:

    E = 0.0 * P(A) + 0.5 * P(B) + 1.0 * P(C)

      = 0.5 * P(B) + P(C)

This preserves the uncertainty of the classifier instead of using only the
hard predicted class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_ADAPTIVE_RAG_MODEL = "LenckCuak/Adaptive-RAG"


@dataclass
class ComplexityResult:
    """
    Output produced by the Adaptive-RAG complexity evaluator.
    """

    predicted_class: str
    p_A: float
    p_B: float
    p_C: float
    E: float

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "predicted_class": self.predicted_class,
            "p_A": self.p_A,
            "p_B": self.p_B,
            "p_C": self.p_C,
            "E": self.E,
        }


class AdaptiveRAGComplexityEvaluator:
    """
    Evaluate query complexity using a pretrained Adaptive-RAG T5 classifier.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier or local path.

    device:
        Device on which the model will run.
        If None:
            CUDA is used when available,
            otherwise CPU.

    max_input_length:
        Maximum number of input tokens given to T5.
    """

    LABELS = ("A", "B", "C")

    # Ordinal values used by this replication.
    LABEL_VALUES = {
        "A": 0.0,
        "B": 0.5,
        "C": 1.0,
    }

    def __init__(
        self,
        model_name: str = DEFAULT_ADAPTIVE_RAG_MODEL,
        device: Optional[str] = None,
        max_input_length: int = 384,
    ) -> None:

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.max_input_length = max_input_length

        print(f"Loading Adaptive-RAG complexity evaluator: {model_name}")
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    @staticmethod
    def build_prompt(question: str) -> str:
        """
        Build the query-complexity classification prompt.

        This follows the usage described for the available pretrained
        Adaptive-RAG checkpoint.
        """

        return (
            "Classify the complexity of the following query: "
            f"{question.strip()}"
        )

    # ------------------------------------------------------------------
    # Sequence likelihood
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _label_log_likelihood(
        self,
        encoded_question: Dict[str, torch.Tensor],
        label: str,
    ) -> torch.Tensor:
        """
        Compute log P(label | question).

        Rather than looking only at a single decoder token, we score the
        complete target sequence produced by the tokenizer.

        This is more robust because T5 is a sequence-to-sequence model.
        """

        target = self.tokenizer(
            label,
            return_tensors="pt",
            add_special_tokens=True,
        )

        labels = target["input_ids"].to(self.device)

        outputs = self.model(
            input_ids=encoded_question["input_ids"],
            attention_mask=encoded_question["attention_mask"],
            labels=labels,
        )

        logits = outputs.logits

        # logits:
        # [batch_size, target_length, vocabulary_size]
        #
        # labels:
        # [batch_size, target_length]

        log_probs = F.log_softmax(logits, dim=-1)

        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)

        # Ignore padding if any exists.
        valid_mask = labels.ne(self.tokenizer.pad_token_id)

        sequence_log_prob = (
            token_log_probs * valid_mask
        ).sum(dim=-1)

        return sequence_log_prob.squeeze(0)

    # ------------------------------------------------------------------
    # Complexity evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, question: str) -> ComplexityResult:
        """
        Evaluate one question.

        Returns
        -------
        ComplexityResult
            predicted_class
            p_A
            p_B
            p_C
            E
        """

        prompt = self.build_prompt(question)

        encoded_question = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )

        encoded_question = {
            key: value.to(self.device)
            for key, value in encoded_question.items()
        }

        # --------------------------------------------------------------
        # Score each allowed Adaptive-RAG output.
        # --------------------------------------------------------------

        label_scores = []

        for label in self.LABELS:
            score = self._label_log_likelihood(
                encoded_question,
                label,
            )

            label_scores.append(score)

        label_scores = torch.stack(label_scores)

        # Convert the three sequence scores into probabilities restricted
        # to the allowed classes A/B/C.
        probabilities = F.softmax(
            label_scores,
            dim=0,
        )

        p_A = probabilities[0].item()
        p_B = probabilities[1].item()
        p_C = probabilities[2].item()

        predicted_index = torch.argmax(probabilities).item()

        predicted_class = self.LABELS[predicted_index]

        # --------------------------------------------------------------
        # Replication assumption:
        #
        # A = 0
        # B = 0.5
        # C = 1
        #
        # E = expectation over the ordinal complexity scale.
        # --------------------------------------------------------------

        E = (
            self.LABEL_VALUES["A"] * p_A
            + self.LABEL_VALUES["B"] * p_B
            + self.LABEL_VALUES["C"] * p_C
        )

        return ComplexityResult(
            predicted_class=predicted_class,
            p_A=p_A,
            p_B=p_B,
            p_C=p_C,
            E=E,
        )


# ----------------------------------------------------------------------
# Simple standalone test
# ----------------------------------------------------------------------

if __name__ == "__main__":

    evaluator = AdaptiveRAGComplexityEvaluator()

    examples = [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        (
            "Which country hosted the Olympic Games immediately after "
            "the country whose capital is Athens?"
        ),
    ]

    for question in examples:

        result = evaluator.evaluate(question)

        print("\n" + "=" * 80)
        print("QUESTION")
        print("=" * 80)
        print(question)

        print("\nComplexity prediction:")
        print(f"  class = {result.predicted_class}")
        print(f"  P(A)  = {result.p_A:.4f}")
        print(f"  P(B)  = {result.p_B:.4f}")
        print(f"  P(C)  = {result.p_C:.4f}")
        print(f"  E     = {result.E:.4f}")
