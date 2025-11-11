#!/usr/bin/env python3
"""
Example: Using LLM-as-a-Judge for QA Evaluation

This script demonstrates how to use GPT-4 as a judge to evaluate answer quality
with categorical judgments (correct/partially correct/incorrect).
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragicamp.agents.direct_llm import DirectLLMAgent
from ragicamp.datasets.nq import NaturalQuestionsDataset
from ragicamp.evaluation.evaluator import Evaluator
from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric
from ragicamp.metrics.llm_judge_qa import LLMJudgeQAMetric
from ragicamp.models.huggingface import HuggingFaceModel
from ragicamp.models.openai import OpenAIModel


def main():
    """Run evaluation with LLM-as-a-judge."""
    
    print("=" * 70)
    print("LLM-as-a-Judge Evaluation Example")
    print("=" * 70)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("\nThis example requires GPT-4 for LLM-as-a-judge evaluation.")
        sys.exit(1)
    
    # 1. Create the model to evaluate (Gemma 2B)
    print("\n1. Loading model to evaluate (Gemma 2B)...")
    model = HuggingFaceModel(
        model_name="google/gemma-2-2b-it",
        device="cuda",
        load_in_8bit=True
    )
    
    # 2. Create agent
    print("2. Creating DirectLLM agent...")
    agent = DirectLLMAgent(
        name="gemma_2b_with_llm_judge",
        model=model,
        system_prompt="You are a helpful assistant. Answer questions accurately and concisely."
    )
    
    # 3. Load dataset (small sample)
    print("3. Loading Natural Questions dataset...")
    dataset = NaturalQuestionsDataset(split="validation")
    dataset.filter_with_answers()
    dataset.examples = dataset.examples[:10]  # Small sample for demo
    print(f"   Using {len(dataset)} examples")
    
    # 4. Create GPT-4 judge
    print("\n4. Creating GPT-4 judge...")
    judge_model = OpenAIModel(
        model_name="gpt-4o",  # or "gpt-4-turbo", "gpt-4"
        temperature=0.0  # Deterministic judgments
    )
    
    # 5. Create metrics
    print("5. Setting up metrics...")
    metrics = [
        # Standard metrics
        ExactMatchMetric(),
        F1Metric(),
        
        # LLM judge - Binary (correct/incorrect)
        LLMJudgeQAMetric(
            judge_model=judge_model,
            judgment_type="binary"
        ),
        
        # Uncomment for ternary (correct/partial/incorrect)
        # LLMJudgeQAMetric(
        #     judge_model=judge_model,
        #     judgment_type="ternary"
        # ),
    ]
    
    # 6. Create evaluator
    evaluator = Evaluator(
        agent=agent,
        dataset=dataset,
        metrics=metrics
    )
    
    # 7. Run evaluation
    print("\n" + "=" * 70)
    print("Running Evaluation (this will take a few minutes)")
    print("=" * 70 + "\n")
    
    results = evaluator.evaluate(
        save_predictions=True,
        output_path="outputs/llm_judge_evaluation.json"
    )
    
    # 8. Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nStandard Metrics:")
    print(f"  Exact Match:        {results['exact_match']:.4f}")
    print(f"  F1 Score:           {results['f1']:.4f}")
    
    print(f"\nLLM Judge (GPT-4):")
    print(f"  Average Score:      {results['llm_judge_qa']:.4f}")
    print(f"  Correct:            {results['llm_judge_qa_correct']:.4f} ({results['llm_judge_qa_correct']*100:.1f}%)")
    print(f"  Incorrect:          {results['llm_judge_qa_incorrect']:.4f} ({results['llm_judge_qa_incorrect']*100:.1f}%)")
    
    if 'llm_judge_qa_partial' in results:
        print(f"  Partially Correct:  {results['llm_judge_qa_partial']:.4f} ({results['llm_judge_qa_partial']*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("\n💡 Tip: Check outputs/llm_judge_evaluation.json for detailed per-question judgments")
    print("=" * 70)


if __name__ == "__main__":
    main()

