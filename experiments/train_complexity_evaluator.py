"""Fine-tune Eva exactly as the Adaptive-RAG three-class T5 classifier.

Input is the JSON produced by ``prepare_adaptive_rag_data.py``.  It combines
the official silver labels (the cheapest strategy that answered correctly) and
the official inductive-bias labels (B for single-hop, C for multi-hop).
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


INKER_HYPERPARAMETERS = {
    "learning_rate": 3e-5,
    "max_seq_length": 384,
    "doc_stride": 128,
    "train_batch_size": 32,
    "eval_batch_size": 100,
    "weight_decay": 0.01,
    "num_epochs": 15,
}
DEFAULT_MODEL_NAME = "google-t5/t5-large"
LABELS = ("A", "B", "C")
LABEL_ALIASES = {
    "0": "A", "a": "A", "no_retrieval": "A", "no-retrieval": "A", "zero": "A",
    "1": "B", "b": "B", "single_step": "B", "single-step": "B", "single": "B",
    "2": "C", "c": "C", "multi_step": "C", "multi-step": "C", "multi": "C",
}


def normalize_label(value):
    """Normalize official A/B/C labels and unambiguous numeric aliases."""
    label = LABEL_ALIASES.get(str(value).strip().lower())
    if label is None:
        raise ValueError(f"Invalid Eva label {value!r}; expected A, B, or C")
    return label


def _example_to_pair(example):
    question = example.get("question", example.get("query"))
    label = example.get("answer", example.get("complexity", example.get("label")))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Each Eva example requires a non-empty 'question' (or 'query')")
    return question, normalize_label(label)


def load_dataset_from_json(json_path):
    """Load the official ``train``/``validation`` JSON structure."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Eva dataset not found: {json_path}")
    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not {"train", "validation"}.issubset(data):
        raise ValueError("Eva dataset must contain 'train' and 'validation' arrays")

    def make_split(rows):
        pairs = [_example_to_pair(row) for row in rows]
        return Dataset.from_dict({"question": [x[0] for x in pairs], "label": [x[1] for x in pairs]})

    dataset = DatasetDict(train=make_split(data["train"]), validation=make_split(data["validation"]))
    if not set(dataset["train"]["label"]).issuperset(LABELS):
        raise ValueError("Training split must contain all three Eva labels A, B, and C")
    return dataset


def tokenize(examples, tokenizer):
    model_inputs = tokenizer(
        examples["question"], truncation=True, max_length=INKER_HYPERPARAMETERS["max_seq_length"]
    )
    labels = tokenizer(text_target=examples["label"], truncation=True, max_length=2)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_predictions = [text.strip().upper() for text in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
    decoded_labels = [text.strip().upper() for text in tokenizer.batch_decode(labels, skip_special_tokens=True)]
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": accuracy_score(decoded_labels, decoded_predictions),
        "f1_macro": f1_score(decoded_labels, decoded_predictions, labels=list(LABELS), average="macro", zero_division=0),
    }


def main(dataset_path, output_dir="models/eva", model_name=DEFAULT_MODEL_NAME, seed=42, hf_token=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = load_dataset_from_json(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=hf_token)
    processed = dataset.map(
        lambda batch: tokenize(batch, tokenizer), batched=True, remove_columns=dataset["train"].column_names
    )
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=INKER_HYPERPARAMETERS["learning_rate"],
        per_device_train_batch_size=INKER_HYPERPARAMETERS["train_batch_size"],
        per_device_eval_batch_size=INKER_HYPERPARAMETERS["eval_batch_size"],
        weight_decay=INKER_HYPERPARAMETERS["weight_decay"],
        num_train_epochs=INKER_HYPERPARAMETERS["num_epochs"],
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1_macro", greater_is_better=True,
        predict_with_generate=True, generation_max_length=2,
        fp16=torch.cuda.is_available(), seed=seed, logging_steps=50, report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=processed["train"], eval_dataset=processed["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model), tokenizer=tokenizer,
        compute_metrics=lambda prediction: compute_metrics(prediction, tokenizer),
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved fine-tuned three-class Eva checkpoint to {output_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the three-class Adaptive-RAG Eva T5-Large model")
    parser.add_argument("dataset", help="JSON from prepare_adaptive_rag_data.py")
    parser.add_argument("--output-dir", default="models/eva")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.dataset, args.output_dir, args.model, args.seed)
