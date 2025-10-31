#!/usr/bin/env python3
"""
Script to evaluate a trained FixedRAG agent.

Memory-efficient two-pass approach:
1. Load retriever, retrieve all contexts, save them
2. Unload retriever, load LLM, generate answers
"""

import argparse
import gc
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
from tqdm import tqdm

from ragicamp.agents.fixed_rag import FixedRAGAgent
from ragicamp.datasets.hotpotqa import HotpotQADataset
from ragicamp.datasets.nq import NaturalQuestionsDataset
from ragicamp.datasets.triviaqa import TriviaQADataset
from ragicamp.evaluation.evaluator import Evaluator
from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric
from ragicamp.models.huggingface import HuggingFaceModel

# Optional metrics
try:
    from ragicamp.metrics.bertscore import BERTScoreMetric
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

try:
    from ragicamp.metrics.bleurt_metric import BLEURTMetric
    BLEURT_AVAILABLE = True
except ImportError:
    BLEURT_AVAILABLE = False


def retrieve_all_contexts(retriever, dataset, top_k, output_path):
    """Pass 1: Retrieve contexts for all queries (memory efficient).
    
    This runs BEFORE loading the LLM to save memory.
    Returns a list of (query, retrieved_docs, references) tuples.
    """
    print("\n" + "=" * 70)
    print("PASS 1: Retrieving contexts for all queries")
    print("=" * 70)
    print(f"Using retriever to fetch top-{top_k} documents per query")
    print("(LLM not loaded yet to save memory)\n")
    
    contexts = []
    
    for example in tqdm(dataset.examples, desc="Retrieving"):
        query = example.question
        
        # Retrieve documents
        retrieved_docs = retriever.retrieve(query, top_k=top_k)
        
        # Store the retrieved documents
        contexts.append({
            "question_id": getattr(example, "id", None),
            "question": query,
            "retrieved_docs": [
                {
                    "text": doc.text,
                    "score": doc.score if hasattr(doc, 'score') else None,
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                }
                for doc in retrieved_docs
            ],
            "references": example.answers
        })
    
    # Save intermediate results
    cache_file = Path(output_path).parent / (Path(output_path).stem + "_contexts.json")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_file, 'w') as f:
        json.dump(contexts, f, indent=2)
    
    print(f"\n✓ Retrieved contexts for {len(contexts)} queries")
    print(f"✓ Saved to: {cache_file}")
    
    return contexts, str(cache_file)


def collect_rag_details(agent, dataset, output_path):
    """Collect detailed RAG pipeline information for each query.
    
    This captures:
    - Retrieved documents (text, scores, metadata)
    - Prompts sent to the LLM
    - Generated answers
    - Any intermediate steps
    """
    print("\n" + "=" * 70)
    print("Collecting RAG Pipeline Details")
    print("=" * 70 + "\n")
    
    details = []
    
    for example in tqdm(dataset.examples, desc="Processing questions"):
        query = example.question
        
        # Get agent response (this includes context with retrieved docs)
        response = agent.answer(query)
        
        # Extract retrieved documents info
        retrieved_docs = []
        for doc in response.context.retrieved_docs:
            retrieved_docs.append({
                "text": doc.text,
                "score": doc.score if hasattr(doc, 'score') else None,
                "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
            })
        
        # Build the detail record
        detail = {
            "question_id": getattr(example, "id", None),
            "question": query,
            "answer": response.answer,
            "retrieved_documents": retrieved_docs,
            "num_retrieved": len(retrieved_docs),
            "references": example.answers,
            "intermediate_steps": response.context.intermediate_steps,
            "metadata": response.context.metadata
        }
        
        details.append(detail)
    
    # Save details to file
    output_file = Path(output_path).parent / (Path(output_path).stem + "_details.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(details, f, indent=2)
    
    print(f"\n✓ Saved RAG details to: {output_file}")
    print(f"  Total queries: {len(details)}")
    print(f"  Avg documents per query: {sum(d['num_retrieved'] for d in details) / len(details):.1f}")
    
    return str(output_file)


def main():
    """Run FixedRAG evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained FixedRAG agent"
    )
    parser.add_argument(
        "--retriever-artifact",
        type=str,
        required=True,
        help="Name of the retriever artifact to load (e.g., 'wikipedia_small')"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-2-2b-it",
        help="HuggingFace model to use (default: google/gemma-2-2b-it)"
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
        default="outputs/fixed_rag_results.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--filter-no-answer",
        action="store_true",
        help="Filter out questions without explicit answers"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["exact_match", "f1"],
        choices=["exact_match", "f1", "bertscore", "bleurt"],
        help="Metrics to compute (default: exact_match f1)"
    )
    parser.add_argument(
        "--bertscore-model",
        type=str,
        default="microsoft/deberta-base-mnli",
        help="Model for BERTScore (default: deberta-base-mnli)"
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        default=True,
        help="Save detailed RAG pipeline information (retrieved docs, prompts, etc.)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FixedRAG Agent Evaluation (Memory-Efficient Two-Pass)")
    print("=" * 70)
    print(f"\nRetriever artifact: {args.retriever_artifact}")
    print(f"Top-K documents: {args.top_k}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Examples: {args.num_examples}")
    print(f"Device: {args.device}")
    print(f"\n💡 Two-pass approach to avoid OOM:")
    print(f"   1. Load retriever → retrieve contexts → unload")
    print(f"   2. Load LLM → generate answers")
    print("\n" + "=" * 70 + "\n")
    
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
    
    print(f"Initial dataset size: {len(dataset)} examples")
    
    # Filter questions without explicit answers if requested
    if args.filter_no_answer:
        print("\nFiltering questions without explicit answers...")
        dataset.filter_with_answers()
    
    # Limit to requested number of examples
    if args.num_examples < len(dataset):
        dataset.examples = dataset.examples[:args.num_examples]
    
    print(f"✓ Final dataset size: {len(dataset)} examples\n")
    
    # ========================================================================
    # PASS 1: Load retriever and retrieve all contexts
    # ========================================================================
    
    print(f"Loading retriever: {args.retriever_artifact}...")
    try:
        from ragicamp.retrievers.dense import DenseRetriever
        retriever = DenseRetriever.load_index(args.retriever_artifact)
        print("✓ Retriever loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading retriever: {e}")
        print(f"\nMake sure you've indexed Wikipedia first:")
        print(f"  make index-wiki-small  # Quick test")
        print(f"  make index-wiki-simple # For evaluation")
        sys.exit(1)
    
    # Retrieve contexts for all queries
    contexts, contexts_file = retrieve_all_contexts(
        retriever, 
        dataset, 
        args.top_k, 
        args.output
    )
    
    # Free retriever from memory
    print("\n🗑️  Freeing retriever from memory...")
    del retriever
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ Retriever unloaded\n")
    
    # ========================================================================
    # PASS 2: Load LLM and generate answers
    # ========================================================================
    
    print(f"Loading LLM: {args.model_name}...")
    try:
        model = HuggingFaceModel(
            model_name=args.model_name,
            device=args.device,
            load_in_8bit=args.load_in_8bit
        )
        print("✓ LLM loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading LLM: {e}")
        sys.exit(1)
    
    # Create a minimal retriever that uses pre-retrieved contexts
    from ragicamp.retrievers.base import Document, Retriever
    
    class PreRetrievedRetriever(Retriever):
        """Dummy retriever that returns pre-retrieved contexts."""
        def __init__(self, contexts_map):
            super().__init__(name="pre_retrieved")
            self.contexts_map = contexts_map
        
        def retrieve(self, query, top_k=5, **kwargs):
            # Look up pre-retrieved docs
            if query in self.contexts_map:
                doc_dicts = self.contexts_map[query]
                return [
                    Document(
                        id=f"doc_{i}",
                        text=d["text"],
                        metadata=d.get("metadata", {})
                    )
                    for i, d in enumerate(doc_dicts)
                ]
            return []
        
        def index_documents(self, documents):
            """Not needed for pre-retrieved contexts."""
            pass
    
    # Create map of query -> retrieved docs
    contexts_map = {ctx["question"]: ctx["retrieved_docs"] for ctx in contexts}
    pre_retriever = PreRetrievedRetriever(contexts_map)
    
    # Create FixedRAG agent with pre-retrieved contexts
    print(f"Creating FixedRAG agent with pre-retrieved contexts...")
    agent = FixedRAGAgent(
        name="fixed_rag_eval",
        model=model,
        retriever=pre_retriever,
        top_k=args.top_k
    )
    print("✓ Agent created successfully\n")
    
    # Create metrics
    print("Setting up metrics...")
    metrics = []
    metric_names = []
    
    if "exact_match" in args.metrics:
        metrics.append(ExactMatchMetric())
        metric_names.append("Exact Match")
    
    if "f1" in args.metrics:
        metrics.append(F1Metric())
        metric_names.append("F1")
    
    if "bertscore" in args.metrics:
        if not BERTSCORE_AVAILABLE:
            print("⚠ BERTScore requested but not installed. Skipping...")
        else:
            print(f"  Loading BERTScore model: {args.bertscore_model}...")
            metrics.append(BERTScoreMetric(model_type=args.bertscore_model))
            metric_names.append("BERTScore")
    
    if "bleurt" in args.metrics:
        if not BLEURT_AVAILABLE:
            print("⚠ BLEURT requested but not installed. Skipping...")
        else:
            print(f"  Loading BLEURT...")
            metrics.append(BLEURTMetric())
            metric_names.append("BLEURT")
    
    print(f"✓ Metrics: {', '.join(metric_names)}\n")
    
    # Create evaluator
    evaluator = Evaluator(
        agent=agent,
        dataset=dataset,
        metrics=metrics
    )
    
    # Run evaluation
    print("=" * 70)
    print("Starting Metrics Evaluation")
    print("=" * 70 + "\n")
    print("ℹ️  Note: Retrieved contexts already saved to:")
    print(f"   {contexts_file}\n")
    
    results = evaluator.evaluate(
        save_predictions=True,
        output_path=args.output
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nAgent: {agent.name}")
    print(f"Dataset: {args.dataset}")
    print(f"Examples evaluated: {results['num_examples']}")
    print(f"\nMetrics:")
    for metric_name, score in results.items():
        if metric_name not in ["num_examples", "agent_name", "dataset_name"]:
            if isinstance(score, float):
                print(f"  {metric_name}: {score:.4f}")
            else:
                print(f"  {metric_name}: {score}")
    
    print("=" * 70)
    print(f"\n💡 Output files:")
    print(f"   Metrics summary: {args.output.replace('.json', '_summary.json')}")
    print(f"   Predictions:     {args.output.replace('.json', '_predictions.json')}")
    print(f"   Contexts:        {contexts_file}")
    print(f"\n💡 Compare with baseline:")
    print(f"   Baseline: outputs/gemma_2b_baseline_summary.json")
    print(f"   FixedRAG: {args.output.replace('.json', '_summary.json')}")
    print(f"\n💡 Memory usage:")
    print(f"   ✓ Two-pass approach prevented OOM errors")
    print(f"   ✓ Only one model in memory at a time")
    print("=" * 70)


if __name__ == "__main__":
    main()

