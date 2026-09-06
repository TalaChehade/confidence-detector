"""
Train the Eva complexity evaluator using INKER paper specifications.

This script fine-tunes T5-Large for query complexity estimation using the exact
hyperparameters from the INKER paper:
- Learning rate: 3e-5
- Max sequence length: 384
- Document stride: 128
- Training batch size: 32
- Evaluation batch size: 100
- Optimizer: AdamW with weight decay 0.01
- Number of training epochs: 15

The model is trained to classify queries as simple (0) or complex (1), where
complex queries are more likely to require retrieval augmentation.

Dataset Requirements:
The training dataset should be in JSON format with queries labeled by complexity:
{
    "train": [
        {"query": "What is the capital of France?", "complexity": 0},
        {"query": "Compare photosynthesis and cellular respiration.", "complexity": 1},
        ...
    ],
    "validation": [...]
}

The corpus should be from open-source data with NO overlap with test queries.
"""

import argparse
import json
import os
import random
import numpy as np
import torch
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset, DatasetDict


# INKER Paper Hyperparameters
INKER_HYPERPARAMETERS = {
    "learning_rate": 3e-5,
    "max_seq_length": 384,
    "doc_stride": 128,
    "train_batch_size": 32,
    "eval_batch_size": 100,
    "weight_decay": 0.01,
    "num_epochs": 15,
    "optimizer": "adamw",  # AdamW is default in transformers.Trainer
}

DEFAULT_MODEL_NAME = "google-t5/t5-large"


def load_dataset_from_json(json_path):
    """
    Load complexity dataset from JSON file.
    
    Expected format:
    {
        "train": [
            {"query": str, "complexity": int (0 or 1)},
            ...
        ],
        "validation": [...]
    }
    
    Args:
        json_path: Path to JSON file
    
    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Dataset not found at {json_path}\n"
            f"Please provide a JSON file with 'train' and 'validation' keys"
        )
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if "train" not in data or "validation" not in data:
        raise ValueError(
            "Dataset JSON must contain 'train' and 'validation' keys"
        )
    
    train_dataset = Dataset.from_dict({
        "text": [ex["query"] for ex in data["train"]],
        "label": [ex["complexity"] for ex in data["train"]],
    })
    
    val_dataset = Dataset.from_dict({
        "text": [ex["query"] for ex in data["validation"]],
        "label": [ex["complexity"] for ex in data["validation"]],
    })
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
    })


def preprocess_function(examples, tokenizer, max_length):
    """Preprocess examples for model training."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def compute_metrics(eval_pred):
    """Compute metrics during training."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }


def main(
    dataset_path,
    output_dir=None,
    model_name=DEFAULT_MODEL_NAME,
    seed=42,
    hf_token=None,
):
    """
    Train Eva complexity evaluator using INKER paper hyperparameters.
    
    Args:
        dataset_path: Path to JSON training dataset
        output_dir: Directory to save trained model (default: models/eva)
        model_name: Base model name (default: google-t5/t5-large)
        seed: Random seed
        hf_token: HuggingFace API token
    """
    if output_dir is None:
        output_dir = "models/eva"
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"\n{'='*80}")
    print(f"Training Eva Complexity Evaluator")
    print(f"{'='*80}")
    print(f"\nHyperparameters (from INKER paper):")
    for param, value in INKER_HYPERPARAMETERS.items():
        print(f"  {param}: {value}")
    print(f"\nBase model: {model_name}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset_from_json(dataset_path)
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['validation'])}")
    
    # Load tokenizer and model
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,  # Binary classification: simple (0) vs complex (1)
        token=hf_token,
    )
    
    print(f"  Model loaded: {model_name}")
    print(f"  Model parameters: {model.num_parameters():,}")
    
    # Preprocess dataset
    print("\nPreprocessing dataset...")
    processed_dataset = dataset.map(
        lambda x: preprocess_function(
            x,
            tokenizer,
            INKER_HYPERPARAMETERS["max_seq_length"],
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # Training arguments (using INKER hyperparameters)
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=INKER_HYPERPARAMETERS["learning_rate"],
        per_device_train_batch_size=INKER_HYPERPARAMETERS["train_batch_size"],
        per_device_eval_batch_size=INKER_HYPERPARAMETERS["eval_batch_size"],
        weight_decay=INKER_HYPERPARAMETERS["weight_decay"],
        num_train_epochs=INKER_HYPERPARAMETERS["num_epochs"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        push_to_hub=False,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\nStarting training...")
    print(f"{'='*80}")
    train_result = trainer.train()
    print(f"{'='*80}\n")
    
    # Evaluate
    print("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    
    print(f"\nValidation Results:")
    for metric, value in eval_results.items():
        if metric.startswith("eval_"):
            metric_name = metric.replace("eval_", "")
            print(f"  {metric_name}: {value:.4f}")
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✓ Training complete!")
    print(f"  Model saved to: {output_dir}")
    print(f"  You can now use this model with:")
    print(f"    from inker.complexity import load_complexity_evaluator")
    print(f"    eva = load_complexity_evaluator('{output_dir}')")
    
    return trainer, train_result, eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Eva complexity evaluator (INKER paper)"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to JSON training dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/eva",
        help="Directory to save trained model (default: models/eva)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Base model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    main(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_name=args.model,
        seed=args.seed,
    )
