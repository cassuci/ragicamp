#!/usr/bin/env python3
"""
Script to train (prepare) a FixedRAG agent.

This indexes documents and saves the agent configuration for later use.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ragicamp.agents.fixed_rag import FixedRAGAgent
from ragicamp.datasets.nq import NaturalQuestionsDataset
from ragicamp.models.huggingface import HuggingFaceModel
from ragicamp.retrievers.base import Document
from ragicamp.retrievers.dense import DenseRetriever


def main():
    """Train a FixedRAG agent (index documents + save config)."""
    parser = argparse.ArgumentParser(
        description="Train a FixedRAG agent for Natural Questions"
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default="fixed_rag_nq_v1",
        help="Name for the agent artifact (default: fixed_rag_nq_v1)"
    )
    parser.add_argument(
        "--retriever-name",
        type=str,
        default="wikipedia_nq_v1",
        help="Name for the retriever artifact (default: wikipedia_nq_v1)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=None,
        help="Limit number of documents to index (for testing)"
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip indexing (use existing retriever artifact)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Training FixedRAG Agent")
    print("=" * 70)
    print(f"\nAgent name: {args.agent_name}")
    print(f"Retriever name: {args.retriever_name}")
    print(f"Top-K: {args.top_k}")
    print("\n" + "=" * 70 + "\n")
    
    # Step 1: Create or load retriever
    if args.skip_indexing:
        print(f"Loading existing retriever: {args.retriever_name}")
        retriever = DenseRetriever.load_index(args.retriever_name)
    else:
        print("Step 1: Indexing documents...")
        print("-" * 70)
        
        # Load dataset
        print("Loading Natural Questions dataset...")
        dataset = NaturalQuestionsDataset(split="train")
        print(f"✓ Loaded {len(dataset)} examples\n")
        
        # Extract documents
        print("Extracting documents...")
        documents = []
        seen_questions = set()
        
        for i, example in enumerate(dataset.examples):
            if args.num_docs and i >= args.num_docs:
                break
            
            # Use question as document (placeholder)
            # TODO: Load actual Wikipedia articles
            if example.question not in seen_questions:
                doc = Document(
                    id=f"nq_doc_{i}",
                    text=example.question,  # Placeholder
                    metadata={
                        "source": "natural_questions",
                        "example_id": example.id,
                        "answers": example.answers
                    }
                )
                documents.append(doc)
                seen_questions.add(example.question)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1} examples...")
        
        print(f"✓ Extracted {len(documents)} documents\n")
        
        # Create and index retriever
        print("Creating and indexing retriever...")
        retriever = DenseRetriever(
            name="wikipedia_nq_retriever",
            embedding_model=args.embedding_model,
            index_type="flat"
        )
        retriever.index_documents(documents)
        
        # Save retriever
        retriever.save_index(args.retriever_name)
        print()
    
    # Step 2: Create agent (without model for now)
    print("Step 2: Creating FixedRAG agent...")
    print("-" * 70)
    
    # Create a placeholder model (not saved)
    print("Note: Model is not saved, you'll need to provide it when loading")
    model = None  # Will be provided at inference time
    
    # Create agent
    agent = FixedRAGAgent(
        name="fixed_rag_wikipedia",
        model=model,  # Placeholder
        retriever=retriever,
        top_k=args.top_k,
        system_prompt="You are a helpful assistant. Use the provided context to answer questions accurately.",
        context_template="Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    
    # Save agent config
    agent.save(args.agent_name, args.retriever_name)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nAgent: {args.agent_name}")
    print(f"Retriever: {args.retriever_name}")
    print(f"Documents indexed: {len(retriever.documents)}")
    print(f"\nTo use this agent:")
    print(f"  from ragicamp.agents.fixed_rag import FixedRAGAgent")
    print(f"  from ragicamp.models.huggingface import HuggingFaceModel")
    print(f"  ")
    print(f"  model = HuggingFaceModel('google/gemma-2-2b-it')")
    print(f"  agent = FixedRAGAgent.load('{args.agent_name}', model)")
    print(f"  response = agent.answer('Your question here')")
    print("=" * 70)


if __name__ == "__main__":
    main()

