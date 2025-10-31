# RAGiCamp Architecture

This document describes the architecture and design principles of RAGiCamp.

## Core Design Principles

1. **Modularity**: Each component (agents, models, retrievers, metrics) is independent and interchangeable
2. **Abstraction**: Clear base classes define contracts, specialized classes implement details
3. **Extensibility**: Easy to add new datasets, models, agents, or metrics
4. **Separation of Concerns**: Training, evaluation, and agent logic are cleanly separated

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Experiment Layer                      │
│  (configs, scripts, notebooks for running experiments)   │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────────┐
│              Orchestration Layer                         │
│  ┌──────────┐              ┌──────────┐                 │
│  │ Trainer  │              │Evaluator │                 │
│  └──────────┘              └──────────┘                 │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────────┐
│                   Agent Layer                            │
│  ┌────────────┐  ┌─────────┐  ┌──────────┐             │
│  │DirectLLM   │  │FixedRAG │  │BanditRAG │             │
│  └────────────┘  └─────────┘  └──────────┘             │
│  ┌────────────┐                                         │
│  │  MDPRAG    │                                         │
│  └────────────┘                                         │
└─────────┬──────────────────────┬─────────────┬──────────┘
          │                      │             │
┌─────────┴─────┐    ┌──────────┴─────┐  ┌───┴──────────┐
│  Model Layer  │    │Retriever Layer │  │Policy Layer  │
│ ┌───────────┐ │    │ ┌────────────┐ │  │┌───────────┐ │
│ │HuggingFace│ │    │ │   Dense    │ │  ││  Bandits  │ │
│ │  OpenAI   │ │    │ │   Sparse   │ │  ││    MDP    │ │
│ └───────────┘ │    │ └────────────┘ │  │└───────────┘ │
└───────────────┘    └────────────────┘  └──────────────┘

┌─────────────────────────────────────────────────────────┐
│                  Support Layers                          │
│  ┌───────────┐    ┌─────────┐    ┌─────────┐            │
│  │ Datasets  │    │ Metrics │    │  Utils  │            │
│  │  - NQ     │    │ - EM/F1 │    │         │            │
│  │  - HotpotQA│   │ - BERT  │    │         │            │
│  │  - TriviaQA│   │ - LLM   │    │         │            │
│  └───────────┘    └─────────┘    └─────────┘            │
└─────────────────────────────────────────────────────────┘
```

## Component Descriptions

### Agents (`ragicamp.agents`)

Agents are the core decision-making components that answer questions. All agents inherit from `RAGAgent` base class.

- **DirectLLMAgent**: Baseline that directly queries LLM without retrieval
- **FixedRAGAgent**: Standard RAG with fixed parameters (top_k, etc.)
- **BanditRAGAgent**: Uses bandit policies to adaptively select RAG parameters
- **MDPRAGAgent**: Treats RAG as sequential decision process with state tracking

**Key abstractions:**
- `RAGAgent.answer(query)` → `RAGResponse`
- `RAGContext`: Stores query, retrieved docs, and intermediate steps
- `RAGResponse`: Contains answer, context, and metadata

### Models (`ragicamp.models`)

Unified interface for different LLM providers.

- **Base class**: `LanguageModel`
  - `generate(prompt)`: Text generation
  - `get_embeddings(texts)`: Get embeddings
  - `count_tokens(text)`: Token counting

- **Implementations**:
  - `HuggingFaceModel`: Local HF transformers models
  - `OpenAIModel`: OpenAI API models

### Retrievers (`ragicamp.retrievers`)

Document retrieval systems.

- **Base class**: `Retriever`
  - `retrieve(query, top_k)`: Find relevant documents
  - `index_documents(docs)`: Index document corpus

- **Implementations**:
  - `DenseRetriever`: Neural embeddings + FAISS
  - `SparseRetriever`: TF-IDF based retrieval

### Policies (`ragicamp.policies`)

Decision policies for adaptive RAG.

- **Base class**: `Policy`
  - `select_action(context)`: Choose action/parameters
  - `update(feedback)`: Learn from rewards

- **Implementations**:
  - Bandits: `EpsilonGreedyBandit`, `UCBBandit`
  - MDP: `QLearningMDPPolicy`, `RandomMDPPolicy`

### Datasets (`ragicamp.datasets`)

Loaders for QA datasets.

- **Base class**: `QADataset`
  - Standard interface for all datasets
  - `QAExample` dataclass for examples

- **Implementations**: NQ, HotpotQA, TriviaQA

### Metrics (`ragicamp.metrics`)

Evaluation metrics.

- **Base class**: `Metric`
  - `compute(predictions, references)`: Compute score

- **Implementations**:
  - `ExactMatchMetric`, `F1Metric`
  - `BERTScoreMetric`
  - `LLMJudgeMetric`

### Training & Evaluation

- **Trainer**: Handles training loop for adaptive agents
  - Generates answers
  - Computes rewards
  - Updates policies

- **Evaluator**: Runs evaluation on datasets
  - Supports multiple metrics
  - Can compare multiple agents
  - Saves predictions and results

## Data Flow

### Evaluation Flow
```
Dataset → Agent.answer(query)
           ├→ Model.generate()
           ├→ Retriever.retrieve() (if RAG)
           └→ Policy.select_action() (if adaptive)
       → RAGResponse
       → Metrics.compute()
       → Results
```

### Training Flow (Adaptive Agents)
```
Dataset → Agent.answer(query)
       → Compute Reward (using metrics)
       → Policy.update(reward)
       → Repeat
```

## Extension Points

Adding new components is straightforward:

1. **New Agent**: Inherit from `RAGAgent`, implement `answer()`
2. **New Model**: Inherit from `LanguageModel`, implement `generate()` and `get_embeddings()`
3. **New Retriever**: Inherit from `Retriever`, implement `retrieve()` and `index_documents()`
4. **New Dataset**: Inherit from `QADataset`, implement `load()`
5. **New Metric**: Inherit from `Metric`, implement `compute()`
6. **New Policy**: Inherit from `Policy`, implement `select_action()` and `update()`

## Configuration-Driven Experiments

Experiments are defined via YAML configs that specify:
- Agent type and parameters
- Model configuration
- Retriever settings
- Policy configuration
- Dataset selection
- Metrics to compute

This allows rapid experimentation without code changes.

## Future Enhancements

Potential areas for extension:
- More retrieval strategies (hybrid, reranking)
- More sophisticated policies (PPO, actor-critic)
- Multi-turn conversations
- External knowledge bases
- Logging and monitoring integration
- Distributed training

