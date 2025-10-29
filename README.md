# RAGiCamp 🏕️

A modular framework for experimenting with Retrieval-Augmented Generation (RAG) approaches, from simple baselines to complex agentic systems.

## Overview

RAGiCamp provides a flexible, extensible architecture for:
- **Testing multiple RAG strategies**: from direct LLM queries to sophisticated MDP-based agents
- **Supporting diverse datasets**: NQ, HotpotQA, TriviaQA, and more
- **Evaluating with various metrics**: BERTScore, BLEURT, LLM-as-a-judge, etc.
- **Training and evaluating agents**: built-in support for bandit and MDP-based RAG policies

## Architecture

The framework is built on clean abstractions that separate concerns:

- **Agents**: Different RAG strategies (direct LLM, fixed RAG, bandit-based, MDP-based)
- **Datasets**: Unified interface for QA datasets
- **Models**: LLM interfaces (HuggingFace, OpenAI, etc.)
- **Retrievers**: Document retrieval systems (dense, sparse, hybrid)
- **Metrics**: Evaluation metrics for answer quality
- **Policies**: Decision-making strategies for adaptive RAG (bandits, MDPs)
- **Training**: Utilities for training adaptive agents
- **Evaluation**: Comprehensive evaluation pipelines

## Quick Start

```bash
# Install dependencies with uv
uv sync

# Run a baseline experiment
uv run python experiments/scripts/run_experiment.py \
    --config experiments/configs/baseline_direct.yaml \
    --mode eval

# Or run the Gemma 2B baseline
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100
```

## Project Structure

```
ragicamp/
├── src/ragicamp/          # Core framework
│   ├── agents/            # RAG strategies
│   ├── datasets/          # Dataset loaders
│   ├── models/            # LLM interfaces
│   ├── retrievers/        # Retrieval systems
│   ├── metrics/           # Evaluation metrics
│   ├── policies/          # Decision policies (bandit/MDP)
│   ├── training/          # Training utilities
│   └── evaluation/        # Evaluation utilities
├── experiments/           # Experiment configs and scripts
├── tests/                 # Unit tests
└── notebooks/             # Exploration notebooks
```

## Experiment Pipeline

1. **Baseline 1**: Direct LLM (no retrieval)
2. **Baseline 2**: Fixed RAG with predefined parameters
3. **Bandit-based RAG**: Adaptive parameter selection
4. **MDP-based RAG**: Iterative action selection with state tracking

## Contributing

This is an experimental research framework. Feel free to add new datasets, models, metrics, or RAG strategies!

