# RAGiCamp 🏕️

A modular, production-ready framework for experimenting with Retrieval-Augmented Generation (RAG), from simple baselines to complex RL-based adaptive systems.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🎯 **Clean Abstractions** - Well-defined interfaces for agents, models, retrievers, metrics
- 🔄 **Multiple RAG Strategies** - DirectLLM, FixedRAG, BanditRAG, MDP-based agents
- 📊 **Rich Evaluation** - EM, F1, BERTScore, BLEURT, LLM-as-a-judge
- 🏋️ **Training Support** - Built-in training for adaptive agents with RL
- 💾 **Artifact Management** - Save/load trained agents and indices
- 📈 **Multiple Datasets** - Natural Questions, HotpotQA, TriviaQA

## 🚀 Quick Start

```bash
# Install dependencies
make install

# Train a FixedRAG agent (quick test - 1000 docs)
make train-fixed-rag-small

# Evaluate baseline
make run-gemma2b

# List saved artifacts
make list-artifacts
```

## 💡 Usage Example

```python
from ragicamp.agents.fixed_rag import FixedRAGAgent
from ragicamp.models.huggingface import HuggingFaceModel

# Load model
model = HuggingFaceModel('google/gemma-2-2b-it')

# Load pre-trained agent with retriever
agent = FixedRAGAgent.load('fixed_rag_nq_v1', model)

# Get answer
response = agent.answer('What is the capital of France?')
print(response.answer)
```

## 🏗️ Architecture

```
ragicamp/
├── src/ragicamp/           # Core framework
│   ├── agents/             # RAG strategies (DirectLLM, FixedRAG, BanditRAG, MDPRAG)
│   ├── models/             # LLM interfaces (HuggingFace, OpenAI)
│   ├── retrievers/         # Retrieval systems (Dense, Sparse)
│   ├── datasets/           # QA datasets (NQ, HotpotQA, TriviaQA)
│   ├── metrics/            # Evaluation metrics
│   ├── policies/           # Decision policies (Bandits, MDP)
│   ├── training/           # Training utilities
│   ├── evaluation/         # Evaluation utilities
│   └── utils/              # Formatting, prompts, artifacts
├── experiments/            # Configs and scripts
├── docs/                   # Documentation
├── artifacts/              # Saved models and indices
└── outputs/                # Evaluation results
```

## 📚 Documentation

- **[Getting Started](docs/GETTING_STARTED.md)** - Installation and first steps
- **[Architecture](docs/ARCHITECTURE.md)** - System design
- **[Usage Guide](docs/USAGE.md)** - Detailed examples
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Command cheat sheet
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues

See [docs/](docs/) for complete documentation.

## 🎯 Training & Inference Workflow

### 1. Train (Index Documents)
```bash
# Index Wikipedia documents for Natural Questions
make train-fixed-rag-small  # Quick test (1000 docs)
# or
make train-fixed-rag        # Full dataset

# This creates:
# - artifacts/retrievers/wikipedia_nq_v1/  (FAISS index + documents)
# - artifacts/agents/fixed_rag_nq_v1/      (agent config)
```

### 2. Inference (Use Trained Agent)
```python
from ragicamp.agents.fixed_rag import FixedRAGAgent
from ragicamp.models.huggingface import HuggingFaceModel

# Load model (not saved in artifacts)
model = HuggingFaceModel('google/gemma-2-2b-it')

# Load agent (automatically loads retriever)
agent = FixedRAGAgent.load('fixed_rag_nq_v1', model)

# Answer questions
response = agent.answer('When was Python created?')
```

### 3. Evaluate
```python
from ragicamp.evaluation.evaluator import Evaluator
from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric

evaluator = Evaluator(agent, dataset, [ExactMatchMetric(), F1Metric()])
results = evaluator.evaluate(save_predictions=True)
```

## 🛠️ Available Commands

```bash
# Setup
make install              # Install dependencies
make setup                # Full setup with BLEURT

# Training
make train-fixed-rag      # Train FixedRAG (full)
make train-fixed-rag-small# Quick test (1000 docs)
make list-artifacts       # List saved artifacts

# Evaluation
make run-gemma2b          # Quick test (10 examples)
make run-gemma2b-full     # Full eval (100 examples)
make run-bertscore        # With BERTScore
make run-all-metrics      # All metrics

# Development
make test                 # Run tests
make lint                 # Lint code
make format               # Format code
make clean                # Clean generated files
```

## 🔬 Supported Agents

| Agent | Description | Training | Use Case |
|-------|-------------|----------|----------|
| **DirectLLM** | No retrieval, direct LLM | ❌ No | Baseline comparison |
| **FixedRAG** | Standard RAG, fixed params | ✅ Index docs | Production RAG |
| **BanditRAG** | Adaptive param selection | ✅ RL training | Optimize retrieval |
| **MDPRAG** | Sequential decision making | ✅ RL training | Complex reasoning |

## 📊 Supported Metrics

- **Exact Match** - Normalized exact matching
- **F1 Score** - Token-level F1
- **BERTScore** - Semantic similarity
- **BLEURT** - Learned evaluation metric
- **LLM-as-a-Judge** - LLM-based evaluation

## 🤝 Contributing

Contributions welcome! This is a research framework designed for experimentation.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with:
- [HuggingFace Transformers](https://huggingface.co/transformers)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [BERTScore](https://github.com/Tiiiger/bert_score)

---

**Ready to start?** Run `make help` to see all available commands!
