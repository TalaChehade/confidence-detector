import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

MULTIHOP_CUES = {
    "and",
    "who",
    "which",
    "before",
    "after",
    "same",
    "both",
    "compare",
    "than",
}

DEFAULT_EVA_MODEL_NAME = "google-t5/t5-large"


class ComplexityEvaluator:
    """
    Eva: A lightweight query complexity evaluator using T5-Large.
    
    This evaluator estimates a complexity score E for a given query Q.
    A higher complexity score indicates that the input query is more complex
    and is more likely to require retrieval.
    
    The model is fine-tuned on an open-source corpus that does not overlap
    with the test queries.
    """
    
    def __init__(
        self,
        model_name=DEFAULT_EVA_MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu",
        hf_token=None,
    ):
        """
        Initialize the complexity evaluator.
        
        Args:
            model_name: HuggingFace model ID for the T5-Large based evaluator
            device: Device to load the model on (cuda or cpu)
            hf_token: HuggingFace API token for accessing private models
        """
        self.device = device
        self.model_name = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True,
            )
            self.model.to(device)
            self.model.eval()
        except Exception as e:
            raise ValueError(
                f"Failed to load Eva model '{model_name}'. "
                f"Error: {e}. "
                f"You may need to train the model first or use the rule-based proxy."
            )
    
    def estimate_complexity(
        self,
        question,
        max_length=512,
        batch_size=1,
    ):
        """
        Estimate complexity score E for a query.
        
        Args:
            question: The input query string
            max_length: Maximum token length for the input
            batch_size: Batch size for processing (for compatibility)
        
        Returns:
            Complexity score E in range [0, 1]
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                question,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Convert logits to probability using softmax
            probs = torch.softmax(logits, dim=-1)
            
            # Assuming binary classification:
            # class 0: simple, class 1: complex
            # Get probability of complex class
            E = probs[0, 1].item() if probs.shape[-1] == 2 else probs[0, -1].item()
        
        return round(float(E), 4)
    
    def __call__(self, question):
        """Make the evaluator callable."""
        return self.estimate_complexity(question)


def estimate_complexity_proxy(question):
    """
    Rule-based E proxy from the original notebook.

    This is a fallback stand-in for INKER's trained T5-large Eva evaluator
    when the trained model is not available.
    """
    words = question.lower().split()
    n_words = len(words)

    cue_hits = sum(
        1
        for w in words
        if w.strip("?,.") in MULTIHOP_CUES
    )

    length_component = min(
        n_words / 25.0,
        1.0,
    )

    cue_component = min(
        cue_hits / 3.0,
        1.0,
    )

    E = (
        0.5 * length_component
        + 0.5 * cue_component
    )

    return round(
        min(max(E, 0.0), 1.0),
        4,
    )


def load_complexity_evaluator(
    model_name=DEFAULT_EVA_MODEL_NAME,
    use_proxy=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    hf_token=None,
):
    """
    Load the complexity evaluator.
    
    Args:
        model_name: HuggingFace model ID for Eva
        use_proxy: If True, use the rule-based proxy instead of the model
        device: Device to load the model on
        hf_token: HuggingFace API token
    
    Returns:
        A callable that takes a question string and returns a complexity score E
    """
    if use_proxy:
        return estimate_complexity_proxy
    
    try:
        eva = ComplexityEvaluator(
            model_name=model_name,
            device=device,
            hf_token=hf_token,
        )
        return eva
    except Exception as e:
        print(f"Warning: Failed to load Eva model. Falling back to rule-based proxy.")
        print(f"Error details: {e}")
        return estimate_complexity_proxy
