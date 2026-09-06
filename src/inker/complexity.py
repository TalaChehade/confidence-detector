"""Paper-aligned Eva query-complexity evaluator.

Eva is the Adaptive-RAG three-way T5 classifier: A=no retrieval,
B=single-step retrieval, and C=multi-step retrieval. It must be loaded from
a fine-tuned checkpoint; this module deliberately contains no heuristic or
pre-trained-model fallback.
"""

from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


CLASS_LABELS = ("A", "B", "C")
CLASS_TO_COMPLEXITY = {"A": 0.0, "B": 0.5, "C": 1.0}


class ComplexityEvaluator:
    """Fine-tuned T5-Large Eva evaluator used by the INKER activation score."""

    def __init__(self, model_path, device=None):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Fine-tuned Eva checkpoint not found: {path}. "
                "Run experiments/prepare_adaptive_rag_data.py and "
                "experiments/train_complexity_evaluator.py first."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = str(path)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device)
        except Exception as exc:
            raise ValueError(f"Could not load fine-tuned Eva checkpoint at {path}: {exc}") from exc
        self.model.eval()
        self._class_token_ids = {
            label: self._single_token_id(label) for label in CLASS_LABELS
        }

    def _single_token_id(self, label):
        ids = self.tokenizer(label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(
                f"Eva label {label!r} must tokenize to one token; got {ids}."
            )
        return ids[0]

    def predict(self, question, max_length=384):
        """Return the predicted class and its A/B/C probabilities."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        encoded = self.tokenizer(
            question, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)
        decoder_start = torch.full(
            (1, 1), self.model.config.decoder_start_token_id,
            dtype=torch.long, device=self.device,
        )
        with torch.no_grad():
            logits = self.model(**encoded, decoder_input_ids=decoder_start).logits[0, 0]
        class_logits = torch.stack([logits[self._class_token_ids[label]] for label in CLASS_LABELS])
        probabilities = torch.softmax(class_logits, dim=0).cpu().tolist()
        probs = dict(zip(CLASS_LABELS, map(float, probabilities)))
        label = max(probs, key=probs.get)
        return {"label": label, "probabilities": probs}

    def estimate_complexity(self, question, max_length=384):
        """Return static E in [0, 1], the expected normalized A/B/C level."""
        prediction = self.predict(question, max_length=max_length)
        return sum(
            CLASS_TO_COMPLEXITY[label] * probability
            for label, probability in prediction["probabilities"].items()
        )

    def __call__(self, question):
        return self.estimate_complexity(question)


def load_complexity_evaluator(model_path, device=None):
    """Load a fine-tuned Eva checkpoint; no heuristic fallback is available."""
    return ComplexityEvaluator(model_path, device=device)
