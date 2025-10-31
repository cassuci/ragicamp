#!/usr/bin/env python3
"""
Demonstration of new architecture improvements.

Shows how to use:
1. DocumentCorpus abstraction
2. ExperimentConfig system
3. OutputManager for results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ragicamp.corpus import WikipediaCorpus, CorpusConfig
from ragicamp.config import ExperimentConfig, create_fixed_rag_config
from ragicamp.output import OutputManager


def demo_corpus():
    """Demonstrate DocumentCorpus usage."""
    print("\n" + "=" * 70)
    print("1. DocumentCorpus Abstraction")
    print("=" * 70)
    
    # Create corpus configuration
    config = CorpusConfig(
        name="wikipedia_simple_demo",
        source="wikimedia/wikipedia",
        version="20231101.simple",
        max_docs=5  # Just 5 docs for demo
    )
    
    print(f"\nCorpus: {config}")
    
    # Initialize corpus
    corpus = WikipediaCorpus(config)
    
    # Load documents
    print("\nLoading documents:")
    for i, doc in enumerate(corpus.load(), 1):
        print(f"\n  Document {i}:")
        print(f"    ID: {doc.id}")
        print(f"    Title: {doc.metadata.get('title', 'N/A')}")
        print(f"    Text length: {len(doc.text)} chars")
        print(f"    Preview: {doc.text[:100]}...")
    
    # Show corpus info
    print("\n  Corpus info:")
    for key, value in corpus.get_info().items():
        print(f"    {key}: {value}")
    
    print("\n  ✓ No answer information in documents!")
    print("  ✓ Clean separation from QA datasets!")


def demo_config():
    """Demonstrate configuration system."""
    print("\n" + "=" * 70)
    print("2. ExperimentConfig System")
    print("=" * 70)
    
    # Create config programmatically
    config = create_fixed_rag_config(dataset="natural_questions")
    
    print(f"\nExperiment: {config.name}")
    print(f"Corpus: {config.corpus}")
    print(f"Retriever: top_k={config.retriever.top_k}, model={config.retriever.embedding_model}")
    print(f"Model: {config.model.name}, 8-bit={config.model.load_in_8bit}")
    print(f"Evaluation: {config.evaluation.dataset}, n={config.evaluation.num_examples}")
    
    # Save to YAML
    yaml_path = "outputs/demo_config.yaml"
    config.to_yaml(yaml_path)
    print(f"\n✓ Saved to: {yaml_path}")
    
    # Load back from YAML
    loaded_config = ExperimentConfig.from_yaml(yaml_path)
    print(f"✓ Loaded back: {loaded_config.name}")
    
    # Show as dict
    print("\n  Config as dict:")
    config_dict = config.to_dict()
    for key in ["name", "corpus", "model"]:
        print(f"    {key}: {config_dict[key]}")


def demo_output_manager():
    """Demonstrate OutputManager."""
    print("\n" + "=" * 70)
    print("3. OutputManager")
    print("=" * 70)
    
    mgr = OutputManager()
    
    # Create demo experiment
    print("\nCreating demo experiment directory...")
    exp_dir = mgr.create_experiment_dir("demo_experiment_v1")
    print(f"  Created: {exp_dir}")
    
    # Save demo experiment
    from ragicamp.config import create_baseline_config
    config = create_baseline_config()
    results = {
        "exact_match": 0.45,
        "f1": 0.52,
        "num_examples": 10
    }
    
    print("\nSaving experiment artifacts...")
    mgr.save_experiment(config, results, exp_dir)
    
    # List experiments
    print("\nListing all experiments:")
    experiments = mgr.list_experiments(limit=5)
    for exp in experiments:
        print(f"  - {exp['experiment_name']}")
        print(f"    Dataset: {exp.get('dataset', 'N/A')}")
        print(f"    Time: {exp.get('timestamp', 'N/A')[:19]}")
    
    # Show directory structure
    print("\n  Directory structure:")
    print(f"    {exp_dir}/")
    for file in exp_dir.iterdir():
        print(f"      ├── {file.name}")
    
    print("\n  ✓ Clean, organized output structure!")
    print("  ✓ Easy to compare experiments!")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print(" " * 20 + "NEW ARCHITECTURE DEMONSTRATION")
    print("=" * 80)
    print("\nShowing the new abstractions:")
    print("  1. DocumentCorpus - Clean document sources (no data leakage)")
    print("  2. ExperimentConfig - Simple, type-safe configuration")
    print("  3. OutputManager - Organized experiment results")
    
    try:
        demo_corpus()
        demo_config()
        demo_output_manager()
        
        print("\n" + "=" * 80)
        print(" " * 30 + "DEMO COMPLETE!")
        print("=" * 80)
        print("\n✅ All new abstractions working!")
        print("\n📝 Next steps:")
        print("  1. Use index_corpus.py to index Wikipedia")
        print("  2. Create experiment configs in experiments/configs/")
        print("  3. Run experiments and compare results")
        print("\n" + "=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

