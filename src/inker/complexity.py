import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class ComplexityEvaluator:
    """
    Eva: A lightweight query complexity evaluator using fine-tuned T5-Large.
    
    This evaluator estimates a complexity score E for a given query Q.
    A higher complexity score indicates that the input query is more complex
    and is more likely to require retrieval.
    
    The model MUST be fine-tuned on an open-source corpus that does not overlap
    with the test queries using the hyperparameters specified in the INKER paper:
    - Learning rate: 3e-5
    - Max sequence length: 384
    - Training batch size: 32
    - Evaluation batch size: 100
    - Optimizer: AdamW with weight decay 0.01
    - Number of training epochs: 15
    
    Train using: python experiments/train_complexity_evaluator.py
    """
    
    def __init__(
        self,
        model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the complexity evaluator with a fine-tuned model.
        
        Args:
            model_path: Path to fine-tuned Eva model directory
            device: Device to load the model on (cuda or cpu)
        
        Raises:
            FileNotFoundError: If model_path does not exist
            ValueError: If model fails to load
        """
        import os
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Eva model not found at: {model_path}\n"
                f"Please train the model first using:\n"
                f"  python experiments/train_complexity_evaluator.py"
            )
        
        self.device = device
        self.model_path = model_path
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(device)
            self.model.eval()
        except Exception as e:
            raise ValueError(
                f"Failed to load fine-tuned Eva model from: {model_path}\n"
                f"Error: {e}"
            )
    
    def estimate_complexity(
        self,
        question,
        max_length=384,  # As specified in INKER paper
    ):
        """
        Estimate complexity score E for a query.
        
        Args:
            question: The input query string
            max_length: Maximum token length (384 as per INKER paper)
        
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
            
            # Binary classification: class 0 = simple, class 1 = complex
            # Return probability of complex class
            E = probs[0, 1].item() if probs.shape[-1] == 2 else probs[0, -1].item()
        
        return round(float(E), 4)
    
    def __call__(self, question):
        """Make the evaluator callable."""
        return self.estimate_complexity(question)


def load_complexity_evaluator(model_path):
    """
    Load the fine-tuned complexity evaluator (Eva).
    
    Args:
        model_path: Path to fine-tuned Eva model directory
    
    Returns:
        ComplexityEvaluator instance
    
    Raises:
        FileNotFoundError: If model not found
        ValueError: If model fails to load
    """
    return ComplexityEvaluator(model_path)
