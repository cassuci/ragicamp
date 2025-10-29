# Getting Started with RAGiCamp

Welcome to RAGiCamp! This guide will help you get started quickly.

## What is RAGiCamp?

RAGiCamp is a modular framework for experimenting with Retrieval-Augmented Generation (RAG) approaches, from simple baselines to complex adaptive systems using bandits and reinforcement learning.

## Installation

```bash
# Navigate to the repository
cd ragicamp

# Install dependencies
pip install -e .

# Optional: Install additional dependencies
pip install -e ".[dev,metrics,viz]"
```

## 5-Minute Quickstart

### 1. Simple Example

Create a file `demo.py`:

```python
import sys
sys.path.insert(0, "src")

from ragicamp.agents.direct_llm import DirectLLMAgent

# Mock model for demonstration
class SimpleModel:
    def __init__(self):
        self.model_name = "demo"
    
    def generate(self, prompt, **kwargs):
        return "Paris"  # Example answer
    
    def get_embeddings(self, texts):
        import numpy as np
        return np.random.randn(len(texts), 384).tolist()

# Create agent
model = SimpleModel()
agent = DirectLLMAgent(name="demo", model=model)

# Ask question
response = agent.answer("What is the capital of France?")
print(f"Answer: {response.answer}")
```

Run it:
```bash
python demo.py
```

### 2. Run a Baseline Experiment

```bash
python experiments/scripts/run_experiment.py \
    --config experiments/configs/baseline_direct.yaml \
    --mode eval
```

## Core Concepts

### Agents
Different RAG strategies:
- **DirectLLM**: No retrieval, just LLM
- **FixedRAG**: Standard RAG with fixed parameters
- **BanditRAG**: Adaptive parameter selection
- **MDPRAG**: Iterative decision-making

### Models
Unified LLM interface:
- HuggingFace transformers
- OpenAI API
- Easy to extend to others

### Retrievers
Document retrieval:
- Dense (embeddings + FAISS)
- Sparse (TF-IDF)

### Datasets
QA datasets:
- Natural Questions
- HotpotQA
- TriviaQA

### Metrics
Evaluation:
- Exact Match, F1
- BERTScore
- LLM-as-a-judge

### Policies
For adaptive agents:
- Bandits (Epsilon-Greedy, UCB)
- MDP (Q-Learning)

## Typical Workflow

```
1. Define experiment in YAML config
   ↓
2. Run baseline experiments
   ↓
3. Train adaptive agents
   ↓
4. Compare results
   ↓
5. Iterate and improve
```

## Project Structure

```
ragicamp/
├── src/ragicamp/           # Core framework code
│   ├── agents/             # RAG strategies
│   ├── models/             # LLM interfaces
│   ├── retrievers/         # Retrieval systems
│   ├── datasets/           # Dataset loaders
│   ├── metrics/            # Evaluation metrics
│   ├── policies/           # Decision policies
│   ├── training/           # Training utilities
│   └── evaluation/         # Evaluation utilities
├── experiments/            # Experiment configs & scripts
│   ├── configs/            # YAML configurations
│   └── scripts/            # Python scripts
├── tests/                  # Unit tests
└── notebooks/              # Jupyter notebooks
```

## Key Design Principles

1. **Modularity**: Swap any component easily
2. **Abstraction**: Clean interfaces, specialized implementations
3. **Configuration**: Define experiments in YAML
4. **Extensibility**: Easy to add new components

## Example Use Cases

### Baseline Comparison
Compare direct LLM vs standard RAG:
```bash
python experiments/scripts/compare_baselines.py
```

### Adaptive RAG Training
Train a bandit agent to learn optimal retrieval parameters:
```bash
python experiments/scripts/run_experiment.py \
    --config experiments/configs/bandit_rag.yaml \
    --mode train
```

### MDP-based Iterative RAG
Train an agent that decides when to retrieve, reformulate, or answer:
```bash
python experiments/scripts/run_experiment.py \
    --config experiments/configs/mdp_rag.yaml \
    --mode train
```

## Next Steps

1. **Read the docs**:
   - `USAGE.md` - Detailed usage guide
   - `ARCHITECTURE.md` - System design
   - `TODO.md` - Planned features

2. **Explore examples**:
   - Check `experiments/configs/` for example configurations
   - Run `experiments/scripts/compare_baselines.py`
   - Open `notebooks/quickstart.ipynb`

3. **Customize**:
   - Add your own datasets
   - Implement custom agents
   - Create new metrics
   - Design novel policies

4. **Experiment**:
   - Try different model sizes
   - Test various retrieval strategies
   - Tune policy hyperparameters
   - Compare on different datasets

## Need Help?

- Check `USAGE.md` for detailed examples
- Read source code - it's well-documented!
- Look at configuration files in `experiments/configs/`
- Explore test files in `tests/` for usage patterns

## Contributing

This is a research framework - feel free to:
- Add new datasets, models, or agents
- Implement new metrics
- Improve existing components
- Share your experiments!

Happy experimenting! 🏕️

