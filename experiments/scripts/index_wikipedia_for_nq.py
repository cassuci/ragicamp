#!/usr/bin/env python3
"""
Script to index Wikipedia documents for Natural Questions dataset.

This creates a retriever artifact that can be reused across experiments.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ragicamp.datasets.nq import NaturalQuestionsDataset
from ragicamp.retrievers.base import Document
from ragicamp.retrievers.dense import DenseRetriever


def main():
    """Index Wikipedia articles for Natural Questions."""
    parser = argparse.ArgumentParser(
        description="Index Wikipedia documents for Natural Questions"
    )
    parser.add_argument(
        "--artifact-name",
        type=str,
        default="wikipedia_nq_v1",
        help="Name for the retriever artifact (default: wikipedia_nq_v1)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        choices=["flat", "ivf"],
        help="FAISS index type (default: flat)"
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=None,
        help="Limit number of documents to index (for testing)"
    )
    parser.add_argument(
        "--use-contexts",
        action="store_true",
        help="Use question contexts as documents (if available)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Indexing Wikipedia for Natural Questions")
    print("=" * 70)
    print(f"\nArtifact name: {args.artifact_name}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Index type: {args.index_type}")
    print("\n" + "=" * 70 + "\n")
    
    # Load Natural Questions dataset
    print("Loading Natural Questions dataset...")
    dataset = NaturalQuestionsDataset(split="train")
    print(f"✓ Loaded {len(dataset)} examples\n")
    
    # Extract documents
    # Note: NQ open-domain doesn't include contexts, so we'll create
    # placeholder documents. In practice, you'd load from Wikipedia dump.
    print("Extracting documents...")
    
    documents = []
    seen_questions = set()
    
    for i, example in enumerate(dataset.examples):
        if args.num_docs and i >= args.num_docs:
            break
        
        # Use question as document for now (placeholder)
        # In production, you'd load actual Wikipedia articles
        if example.question not in seen_questions:
            doc = Document(
                id=f"nq_doc_{i}",
                text=example.question,  # Placeholder - use actual Wikipedia text
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
    
    print(f"✓ Extracted {len(documents)} unique documents\n")
    
    # Create retriever
    print(f"Creating DenseRetriever with {args.embedding_model}...")
    retriever = DenseRetriever(
        name="wikipedia_nq_retriever",
        embedding_model=args.embedding_model,
        index_type=args.index_type
    )
    print("✓ Retriever created\n")
    
    # Index documents
    print("Indexing documents (this may take a while)...")
    retriever.index_documents(documents)
    print("✓ Documents indexed\n")
    
    # Save index
    print(f"Saving index as '{args.artifact_name}'...")
    artifact_path = retriever.save_index(args.artifact_name)
    
    print("\n" + "=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)
    print(f"\nArtifact saved to: {artifact_path}")
    print(f"Documents indexed: {len(documents)}")
    print(f"\nTo use this retriever:")
    print(f"  from ragicamp.retrievers.dense import DenseRetriever")
    print(f"  retriever = DenseRetriever.load_index('{args.artifact_name}')")
    print("=" * 70)


if __name__ == "__main__":
    main()

