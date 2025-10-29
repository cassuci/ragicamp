#!/usr/bin/env python3
"""
Script to run Gemma 2B baseline evaluation.

This script specifically evaluates the Gemma 2B model without any retrieval
as a baseline for comparison with RAG approaches.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ragicamp.agents.direct_llm import DirectLLMAgent
from ragicamp.datasets.hotpotqa import HotpotQADataset
from ragicamp.datasets.nq import NaturalQuestionsDataset
from ragicamp.datasets.triviaqa import TriviaQADataset
from ragicamp.evaluation.evaluator import Evaluator
from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric
from ragicamp.models.huggingface import HuggingFaceModel


def main():
    """Run Gemma 2B baseline evaluation."""
    parser = argparse.ArgumentParser(
        description="Run Gemma 2B baseline evaluation (no retrieval)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["natural_questions", "hotpotqa", "triviaqa"],
        default="natural_questions",
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split (train/validation/test)"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of examples to evaluate (default: 100)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization (saves memory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/gemma2b_baseline_results.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Gemma 2B Baseline Evaluation (No Retrieval)")
    print("=" * 70)
    print(f"\nDataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Examples: {args.num_examples}")
    print(f"Device: {args.device}")
    print(f"8-bit quantization: {args.load_in_8bit}")
    print("\n" + "=" * 70 + "\n")
    
    # Create model
    print("Loading Gemma 2B model...")
    try:
        model = HuggingFaceModel(
            model_name="google/gemma-2-2b-it",
            device=args.device,
            load_in_8bit=args.load_in_8bit
        )
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nNote: You may need to:")
        print("  1. Accept the Gemma license on HuggingFace")
        print("  2. Login with: huggingface-cli login")
        print("  3. Use --load-in-8bit if you have memory issues")
        sys.exit(1)
    
    # Create agent
    print("Creating DirectLLM agent...")
    agent = DirectLLMAgent(
        name="gemma_2b_baseline",
        model=model,
        system_prompt="You are a helpful AI assistant. Answer questions accurately and concisely based on your knowledge."
    )
    print("✓ Agent created\n")
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "natural_questions":
        dataset = NaturalQuestionsDataset(split=args.split)
    elif args.dataset == "hotpotqa":
        dataset = HotpotQADataset(split=args.split)
    elif args.dataset == "triviaqa":
        dataset = TriviaQADataset(split=args.split)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Limit to requested number of examples
    if args.num_examples < len(dataset):
        dataset.examples = dataset.examples[:args.num_examples]
    
    print(f"✓ Loaded {len(dataset)} examples\n")
    
    # Create metrics
    print("Setting up metrics...")
    metrics = [
        ExactMatchMetric(),
        F1Metric()
    ]
    print("✓ Metrics: Exact Match, F1\n")
    
    # Create evaluator
    evaluator = Evaluator(
        agent=agent,
        dataset=dataset,
        metrics=metrics
    )
    
    # Run evaluation
    print("=" * 70)
    print("Starting Evaluation")
    print("=" * 70 + "\n")
    
    results = evaluator.evaluate(
        save_predictions=True,
        output_path=args.output
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nDataset: {args.dataset}")
    print(f"Examples evaluated: {results['num_examples']}")
    print(f"\nMetrics:")
    for metric_name, score in results.items():
        if metric_name not in ["num_examples", "agent_name", "dataset_name"]:
            if isinstance(score, float):
                print(f"  {metric_name}: {score:.4f}")
            else:
                print(f"  {metric_name}: {score}")
    
    print(f"\n✓ Results saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()

