"""
Combined confidence detector + complexity evaluator experiment.

This script runs the full INKER pipeline combining:
1. Confidence detector (trained representation direction)
2. Complexity evaluator (Eva - fine-tuned T5-Large)
3. Activation score calculation: K(t_i) = (E - m_tilde_i) * s_i

The activation score K(t_i) integrates query complexity E with token-level
confidence m_tilde_i to determine whether retrieval is needed. A positive K
indicates high activation (need for retrieval), while negative K indicates
the model is confident in its generation.

Requirements:
- Trained confidence detector (from train_detector.py)
- Fine-tuned Eva model (from train_complexity_evaluator.py)
"""

import argparse
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch

from _common import (
    get_config,
    load_configured_model,
    resolve_project_path,
    detector_layers,
)

from inker.complexity import load_complexity_evaluator
from inker.generation import answer_with_confidence


def evaluate_with_complexity(
    questions,
    tokenizer,
    model,
    rep_reader,
    layers,
    complexity_fn,
    threshold=0.5,
    max_new_tokens=60,
    repetition_penalty=1.1,
    verbose=True,
):
    """
    Evaluate questions using combined confidence detector and complexity evaluator.
    
    Args:
        questions: List of question strings
        tokenizer: Model tokenizer
        model: Language model
        rep_reader: Trained representation reader (confidence detector)
        layers: Detector layers to use
        complexity_fn: Callable for complexity evaluation
        threshold: Decision threshold
        max_new_tokens: Max tokens to generate
        repetition_penalty: Generation penalty
        verbose: Print results
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    for i, question in enumerate(questions):
        if verbose and i > 0 and i % 5 == 0:
            print(f"Processed {i}/{len(questions)} questions...")
        
        result = answer_with_confidence(
            question=question,
            tokenizer=tokenizer,
            model=model,
            rep_reader=rep_reader,
            layers=layers,
            complexity_fn=complexity_fn,
            threshold=threshold,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            verbose=False,
        )
        
        results.append(result)
    
    return results


def export_results_to_csv(results, output_dir):
    """
    Export results to CSV files for analysis.
    
    Args:
        results: List of result dictionaries from evaluation
        output_dir: Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Question-level results
    question_results = []
    
    # Token-level results
    token_results = []
    
    for result in results:
        question_results.append({
            "question": result["question"],
            "answer_text": result["answer_text"],
            "E": result["E"],
            "mean_m_tilde": result["mean_m_tilde"],
            "min_m_tilde": result["min_m_tilde"],
            "max_K": result["max_K"],
            "would_trigger_full": result["would_trigger_full"],
            "would_trigger_confidence_only": result["would_trigger_confidence_only"],
        })
        
        for entry in result["token_entries"]:
            token_results.append({
                "question": result["question"],
                "E": result["E"],
                "token": entry.get("token", ""),
                "token_index": entry.get("token_index", -1),
                "s_i": entry.get("s_i", 0),
                "m_tilde": entry.get("m_tilde", None),
                "K": entry.get("K", None),
                "is_content": entry.get("is_content", False),
                "skip": entry.get("skip", True),
            })
    
    # Save to CSV
    questions_df = pd.DataFrame(question_results)
    tokens_df = pd.DataFrame(token_results)
    
    questions_csv = os.path.join(output_dir, "combined_questions.csv")
    tokens_csv = os.path.join(output_dir, "combined_tokens.csv")
    
    questions_df.to_csv(questions_csv, index=False)
    tokens_df.to_csv(tokens_csv, index=False)
    
    return questions_df, tokens_df


def print_summary(questions_df, tokens_df):
    """Print summary statistics from results."""
    print("\n" + "="*80)
    print("Combined Detector Results Summary")
    print("="*80)
    
    print(f"\nTotal questions evaluated: {len(questions_df)}")
    print(f"Total tokens processed: {len(tokens_df)}")
    
    # Complexity statistics
    print(f"\n{'Complexity Score E Statistics':^80}")
    print("-" * 80)
    print(f"  Mean: {questions_df['E'].mean():.4f}")
    print(f"  Std:  {questions_df['E'].std():.4f}")
    print(f"  Min:  {questions_df['E'].min():.4f}")
    print(f"  Max:  {questions_df['E'].max():.4f}")
    
    # Trigger statistics
    print(f"\n{'Trigger Statistics':^80}")
    print("-" * 80)
    full_triggers = questions_df["would_trigger_full"].sum()
    conf_only_triggers = questions_df["would_trigger_confidence_only"].sum()
    print(f"  Full K(t_i) triggers: {full_triggers}/{len(questions_df)} ({100*full_triggers/len(questions_df):.1f}%)")
    print(f"  Confidence-only triggers: {conf_only_triggers}/{len(questions_df)} ({100*conf_only_triggers/len(questions_df):.1f}%)")
    
    # Token-level statistics
    content_tokens = tokens_df[tokens_df["is_content"] == True]
    if len(content_tokens) > 0:
        print(f"\n{'Token-Level Statistics (Content Tokens Only)':^80}")
        print("-" * 80)
        print(f"  Total content tokens: {len(content_tokens)}")
        print(f"  Mean m_tilde: {content_tokens['m_tilde'].mean():.4f}")
        print(f"  Mean K(t_i): {content_tokens['K'].mean():.4f}")
        print(f"  Positive K (high activation): {(content_tokens['K'] > 0).sum()}/{len(content_tokens)}")
    
    print("\n" + "="*80 + "\n")


def main(config_path=None, eva_model_path=None, num_questions=None):
    """
    Main experiment runner.
    
    Args:
        config_path: Path to config file
        eva_model_path: Path to fine-tuned Eva model directory
        num_questions: Number of questions to evaluate (for testing)
    """
    config = get_config(config_path)
    
    seed = config["experiment"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load models
    print("Loading models...")
    tokenizer, model = load_configured_model(config)
    
    reader_path = resolve_project_path(config, "representation_reader")
    if not os.path.exists(reader_path):
        raise FileNotFoundError(
            f"Representation reader not found: {reader_path}\n"
            "Run train_detector.py first."
        )
    
    with open(reader_path, "rb") as f:
        rep_reader = pickle.load(f)
    
    # Load complexity evaluator
    print("Loading complexity evaluator (Eva)...")
    if eva_model_path is None:
        raise ValueError(
            "Eva model path is required. "
            "Please provide path using --eva-model or train first:\n"
            "  python experiments/train_complexity_evaluator.py <dataset.json>"
        )
    
    if not os.path.exists(eva_model_path):
        raise FileNotFoundError(
            f"Eva model not found at: {eva_model_path}\n"
            f"Please train the model first:\n"
            f"  python experiments/train_complexity_evaluator.py <dataset.json>"
        )
    
    try:
        complexity_fn = load_complexity_evaluator(eva_model_path)
        method = "Fine-tuned T5-Large Eva"
    except Exception as e:
        print(f"Error loading Eva model: {e}")
        raise
    
    print(f"Using complexity evaluator: {method}")
    print(f"  Model path: {eva_model_path}")
    
    # Load test questions
    print("Loading test questions...")
    from inker.dataset import build_inker_pairs, make_split
    
    dataset_path = resolve_project_path(config, "dataset")
    honest_statements, untruthful_statements, topics, pair_topics = (
        build_inker_pairs(
            dataset_path,
            tokenizer,
            seed=seed,
        )
    )
    
    dataset = make_split(
        honest_statements,
        untruthful_statements,
        pair_topics,
        train_ratio=0.70,
        eval_ratio=0.15,
        test_ratio=0.15,
        seed=0,
    )
    
    # Use test split for evaluation
    test_texts = dataset["test"]["data"]
    if num_questions:
        test_texts = test_texts[:num_questions]
    
    print(f"Evaluating {len(test_texts)} questions with combined detector...\n")
    
    # Evaluate
    results = evaluate_with_complexity(
        questions=test_texts,
        tokenizer=tokenizer,
        model=model,
        rep_reader=rep_reader,
        layers=detector_layers(config),
        complexity_fn=complexity_fn,
        threshold=config["confidence"]["threshold"],
        max_new_tokens=config["generation"]["max_new_tokens"],
        repetition_penalty=config["generation"]["repetition_penalty"],
        verbose=True,
    )
    
    # Save results
    result_dir = resolve_project_path(
        config,
        "full_k_results",
        create_if_missing=True,
    )
    
    questions_df, tokens_df = export_results_to_csv(results, result_dir)
    
    print(f"Results saved to: {result_dir}")
    print(f"  - combined_questions.csv")
    print(f"  - combined_tokens.csv")
    
    # Print summary
    print_summary(questions_df, tokens_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run combined confidence detector + complexity evaluator"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--eva-model",
        type=str,
        required=True,
        help="Path to fine-tuned Eva model directory (required)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all)",
    )
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        eva_model_path=args.eva_model,
        num_questions=args.num_questions,
    )
