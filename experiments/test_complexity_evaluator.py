"""
Test and validate the complexity evaluator (Eva).

This script tests the query complexity evaluator which estimates a complexity
score E for a given query. The complexity evaluator uses a fine-tuned T5-Large
model trained with hyperparameters from the INKER paper.

A higher complexity score indicates that the input query is more complex and is
more likely to require retrieval. This serves as a static metric throughout the
entire generation process.

Before running this script, you must train the Eva model:
    python experiments/train_complexity_evaluator.py <path_to_dataset.json>
"""

import argparse
import os
import pandas as pd
import numpy as np

from _common import get_config, resolve_project_path
from inker.complexity import load_complexity_evaluator


def get_test_queries():
    """
    Get a diverse set of test queries for complexity evaluation.
    
    Returns:
        List of tuples: (query, expected_complexity_level)
    """
    return [
        # Simple queries (low complexity)
        ("What is the capital of France?", "low"),
        ("Who wrote Romeo and Juliet?", "low"),
        ("What is 2 + 2?", "low"),
        ("What color is the sky?", "low"),
        ("Is water wet?", "low"),
        
        # Medium complexity queries
        ("How do plants perform photosynthesis?", "medium"),
        ("What are the causes of climate change?", "medium"),
        ("Explain the theory of evolution.", "medium"),
        ("What is the process of how vaccines work?", "medium"),
        ("Describe the water cycle.", "medium"),
        
        # High complexity queries (multihop, reasoning)
        ("Who was the first president of the United States and what were his major achievements?", "high"),
        ("Compare and contrast the causes of World War I and World War II.", "high"),
        ("What are the similarities and differences between prokaryotes and eukaryotes before and after their evolution?", "high"),
        ("How does the greenhouse effect contribute to climate change and what are the long-term consequences?", "high"),
        ("Explain the relationship between supply and demand in economics and how it affects market prices.", "high"),
    ]


def test_complexity_evaluator(config_path=None, eva_model_path=None, verbose=True):
    """
    Test the complexity evaluator on various queries.
    
    Args:
        config_path: Path to config file
        eva_model_path: Path to fine-tuned Eva model directory
        verbose: Print detailed results
    """
    config = get_config(config_path)
    
    # Load the complexity evaluator
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
    
    # Get test queries
    test_queries = get_test_queries()
    
    # Evaluate complexity for each query
    results = []
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Complexity Evaluator Test ({method})")
        print(f"{'='*80}\n")
    
    for query, expected_level in test_queries:
        prediction = complexity_fn.predict(query)
        E = complexity_fn(query)
        
        results.append({
            "query": query,
            "expected_level": expected_level,
            "complexity_score_E": E,
            "predicted_class": prediction["label"],
            "p_A_no_retrieval": prediction["probabilities"]["A"],
            "p_B_single_step": prediction["probabilities"]["B"],
            "p_C_multi_step": prediction["probabilities"]["C"],
            "query_length": len(query.split()),
        })
        
        if verbose:
            print(f"Query: {query}")
            print(
                f"Expected Level: {expected_level:>8} | Class: {prediction['label']} "
                f"| Complexity Score E: {E:.4f}"
            )
            print("-" * 80)
    
    # Create DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Statistics by complexity level
    if verbose:
        print(f"\n{'='*80}")
        print("Summary Statistics by Expected Complexity Level")
        print(f"{'='*80}\n")
        
        for level in ["low", "medium", "high"]:
            level_data = results_df[results_df["expected_level"] == level]["complexity_score_E"]
            if len(level_data) > 0:
                print(f"{level.upper():>10}: "
                      f"mean E = {level_data.mean():.4f}, "
                      f"min E = {level_data.min():.4f}, "
                      f"max E = {level_data.max():.4f}, "
                      f"std E = {level_data.std():.4f}")
    
    # Save results
    result_dir = resolve_project_path(config, "complexity_eval_results", create_if_missing=True)
    os.makedirs(result_dir, exist_ok=True)
    
    csv_path = os.path.join(result_dir, f"complexity_test_{method.lower().replace(' ', '_')}.csv")
    results_df.to_csv(csv_path, index=False)
    
    if verbose:
        print(f"\nResults saved to: {csv_path}\n")
    
    return results_df


def test_complexity_with_generation(config_path=None, eva_model_path=None, verbose=True):
    """
    Test complexity evaluator integrated with the generation pipeline.
    
    This demonstrates how complexity scores work together with the confidence
    detector to produce activation scores K(ti) = (E - m_tilde_i) * s_i.
    """
    from _common import load_configured_model, detector_layers
    from inker.generation import answer_with_confidence
    import pickle
    
    config = get_config(config_path)
    
    # Load model and reader
    tokenizer, model = load_configured_model(config)
    reader_path = resolve_project_path(config, "representation_reader")
    
    if not os.path.exists(reader_path):
        print(f"Error: Representation reader not found at {reader_path}")
        print("Please run train_detector.py first.")
        return None
    
    with open(reader_path, "rb") as f:
        rep_reader = pickle.load(f)
    
    # Load complexity evaluator
    if eva_model_path is None:
        raise ValueError(
            "Eva model path is required. "
            "Please provide path using --eva-model or train first:\n"
            "  python experiments/train_complexity_evaluator.py <dataset.json>"
        )
    
    if not os.path.exists(eva_model_path):
        raise FileNotFoundError(
            f"Eva model not found at: {eva_model_path}"
        )
    
    try:
        complexity_fn = load_complexity_evaluator(eva_model_path)
        method = "Fine-tuned T5-Large Eva"
    except Exception as e:
        print(f"Error loading Eva model: {e}")
        raise
    
    # Test queries
    test_questions = [
        "What is the capital of France?",
        "Compare and contrast photosynthesis and cellular respiration.",
        "How does artificial intelligence work in large language models?",
    ]
    
    result_dir = resolve_project_path(config, "complexity_generation_results", create_if_missing=True)
    os.makedirs(result_dir, exist_ok=True)
    
    results = []
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Complexity Evaluator + Generation Test ({method})")
        print(f"{'='*80}\n")
    
    for question in test_questions:
        result = answer_with_confidence(
            question=question,
            tokenizer=tokenizer,
            model=model,
            rep_reader=rep_reader,
            layers=detector_layers(config),
            complexity_fn=complexity_fn,
            threshold=0.5,
            verbose=verbose,
        )
        
        results.append(result)
    
    # Save detailed results
    csv_path = os.path.join(result_dir, f"generation_test_{method.lower().replace(' ', '_')}.csv")
    
    export_results = []
    for result in results:
        for entry in result["token_entries"]:
            export_results.append({
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
    
    export_df = pd.DataFrame(export_results)
    export_df.to_csv(csv_path, index=False)
    
    if verbose:
        print(f"Detailed results saved to: {csv_path}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test the complexity evaluator (Eva)"
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
        "--with-generation",
        action="store_true",
        help="Test complexity evaluator integrated with generation pipeline",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed results (default: True)",
    )
    
    args = parser.parse_args()
    
    # Test complexity evaluator alone
    test_complexity_evaluator(
        config_path=args.config,
        eva_model_path=args.eva_model,
        verbose=args.verbose,
    )
    
    # Test complexity evaluator with generation if requested
    if args.with_generation:
        test_complexity_with_generation(
            config_path=args.config,
            eva_model_path=args.eva_model,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
