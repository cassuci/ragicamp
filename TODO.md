# RAGiCamp TODO

This file tracks features and improvements to be implemented.

## Core Features

- [x] Base abstractions (agents, models, retrievers, metrics, policies)
- [x] Baseline agents (DirectLLM, FixedRAG)
- [x] Adaptive agents (Bandit, MDP)
- [x] Dataset loaders (NQ, HotpotQA, TriviaQA)
- [x] Basic metrics (EM, F1)
- [x] Training and evaluation utilities
- [x] Configuration-driven experiments

## High Priority

- [ ] Add BLEURT metric implementation
- [ ] Implement document corpus loading utilities
- [ ] Add more retriever implementations (hybrid, reranking)
- [ ] Improve MDP state representation
- [ ] Add more bandit algorithms (Thompson Sampling, etc.)
- [ ] Add logging and experiment tracking (Weights & Biases, MLflow)

## Medium Priority

- [ ] Implement more sophisticated MDP policies (PPO, A2C)
- [ ] Add support for custom document corpora
- [ ] Implement query reformulation strategies
- [ ] Add caching for retrieval results
- [ ] Multi-turn conversation support
- [ ] Add visualization utilities for results

## Low Priority

- [ ] Distributed training support
- [ ] API server for deploying agents
- [ ] Web UI for experiments
- [ ] More dataset loaders (MS MARCO, SQuAD, etc.)
- [ ] Integration with vector databases (Pinecone, Weaviate)
- [ ] Support for multimodal retrieval

## Documentation

- [x] README with overview
- [x] Architecture documentation
- [x] Usage guide
- [ ] API reference documentation
- [ ] Tutorial notebooks
- [ ] Example use cases
- [ ] Paper reproduction guides

## Testing

- [ ] Unit tests for all components
- [ ] Integration tests
- [ ] End-to-end experiment tests
- [ ] Performance benchmarks
- [ ] CI/CD pipeline

## Code Quality

- [ ] Type hints throughout
- [ ] Docstring coverage
- [ ] Linting configuration
- [ ] Code formatting (black, isort)
- [ ] Pre-commit hooks

