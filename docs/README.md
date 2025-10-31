# RAGiCamp Documentation

Complete documentation for the RAGiCamp framework.

## 📚 Main Documentation

- **[Getting Started](GETTING_STARTED.md)** - Quick start guide for new users
- **[Architecture](ARCHITECTURE.md)** - System design and components
- **[Agents Guide](AGENTS.md)** - Complete guide to agents (DirectLLM, FixedRAG, BanditRAG, MDPRAG)
- **[Usage Guide](USAGE.md)** - Detailed usage examples
- **[Quick Reference](QUICK_REFERENCE.md)** - Command cheat sheet
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions

## 📖 Guides

- **[Output Structure](guides/OUTPUT_STRUCTURE.md)** - Understanding result files
- **[Metrics Guide](guides/METRICS_GUIDE.md)** - Available metrics and usage
- **[Normalization Guide](guides/NORMALIZATION_GUIDE.md)** - Text normalization for evaluation
- **[Gemma 2B Quickstart](guides/gemma2b_quickstart.md)** - Using Gemma 2B model
- **[Answer Filtering](guides/ANSWER_FILTERING_UPDATE.md)** - Filtering strategies

## 🛠️ Development

- **[Easy Fixes Summary](development/EASY_FIXES_SUMMARY.md)** - Recent type safety improvements
- **[Refactoring Summary](development/REFACTORING_SUMMARY.md)** - Major refactorings
- **[Update Summary](development/UPDATE_SUMMARY.md)** - Recent updates
- **[Project Summary](development/SUMMARY.md)** - Overall project status

## 🚀 Quick Links

### Training & Indexing
```bash
# Quick test with 1000 documents
make train-fixed-rag-small

# Full Wikipedia indexing
make train-fixed-rag

# List saved artifacts
make list-artifacts
```

### Evaluation
```bash
# Quick test (10 examples)
make run-gemma2b

# Full evaluation (100 examples)
make run-gemma2b-full

# With BERTScore
make run-bertscore
```

### Agent Usage
```python
from ragicamp.agents.fixed_rag import FixedRAGAgent
from ragicamp.models.huggingface import HuggingFaceModel

# Load pre-trained agent
model = HuggingFaceModel('google/gemma-2-2b-it')
agent = FixedRAGAgent.load('fixed_rag_nq_v1', model)

# Get answer
response = agent.answer('What is machine learning?')
print(response.answer)
```

## 📦 Artifact Structure

```
artifacts/
├── retrievers/          # Saved retriever indices
│   ├── wikipedia_nq_v1/
│   │   ├── index.faiss  # FAISS index
│   │   ├── documents.pkl # Document store
│   │   └── config.json  # Retriever config
│   └── ...
└── agents/             # Saved agent configs
    ├── fixed_rag_nq_v1/
    │   └── config.json # Agent config (references retriever)
    └── ...
```

## 🔗 External Resources

- [RAGiCamp GitHub](https://github.com/yourusername/ragicamp)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Natural Questions Dataset](https://ai.google.com/research/NaturalQuestions)

