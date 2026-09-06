"""Fine-tune Eva exactly as the Adaptive-RAG three-class T5 classifier.

Input is the JSON produced by ``prepare_adaptive_rag_data.py``.  It combines
the official silver labels (the cheapest strategy that answered correctly) and
the official inductive-bias labels (B for single-hop, C for multi-hop).
"""

import argparse
import json
import math
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


def main(
    dataset_path,
    output_dir="models/eva",
    model_name=DEFAULT_MODEL_NAME,
    seed=42,
    hf_token=None,
    train_micro_batch_size=1,
    eval_micro_batch_size=4,
    gradient_checkpointing=True,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = load_dataset_from_json(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=hf_token)
    if gradient_checkpointing:
        # A T4-class 16 GB GPU cannot hold T5-Large with a physical batch of
        # 32 at length 384. Checkpointing trades compute for activation memory.
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    processed = dataset.map(
        lambda batch: tokenize(batch, tokenizer), batched=True, remove_columns=dataset["train"].column_names
    )
    gradient_accumulation_steps = math.ceil(
        INKER_HYPERPARAMETERS["train_batch_size"] / train_micro_batch_size
    )
    effective_train_batch_size = train_micro_batch_size * gradient_accumulation_steps
    print(
        "Eva batch configuration: "
        f"micro-batch={train_micro_batch_size}, "
        f"gradient accumulation={gradient_accumulation_steps}, "
        f"effective train batch={effective_train_batch_size}; "
        f"eval micro-batch={eval_micro_batch_size}"
    )
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=INKER_HYPERPARAMETERS["learning_rate"],
        per_device_train_batch_size=train_micro_batch_size,
        per_device_eval_batch_size=eval_micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        weight_decay=INKER_HYPERPARAMETERS["weight_decay"],
        num_train_epochs=INKER_HYPERPARAMETERS["num_epochs"],
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1_macro", greater_is_better=True,
        predict_with_generate=True, generation_max_length=2,
        fp16=torch.cuda.is_available(), seed=seed, logging_steps=50, report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=processed["train"], eval_dataset=processed["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model), processing_class=tokenizer,
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
    parser.add_argument(
        "--train-micro-batch-size", type=int, default=1,
        help="Physical GPU batch size; gradient accumulation retains effective batch 32 (default: 1).",
    )
    parser.add_argument(
        "--eval-micro-batch-size", type=int, default=4,
        help="Physical GPU evaluation batch size for memory-limited runtimes (default: 4).",
    )
    parser.add_argument(
        "--no-gradient-checkpointing", action="store_true",
        help="Disable activation checkpointing; requires substantially more GPU memory.",
    )
    args = parser.parse_args()
    if args.train_micro_batch_size < 1 or args.eval_micro_batch_size < 1:
        parser.error("micro-batch sizes must be positive")
    if INKER_HYPERPARAMETERS["train_batch_size"] % args.train_micro_batch_size:
        parser.error("--train-micro-batch-size must divide the paper batch size (32)")
    main(
        args.dataset,
        args.output_dir,
        args.model,
        args.seed,
        train_micro_batch_size=args.train_micro_batch_size,
        eval_micro_batch_size=args.eval_micro_batch_size,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
