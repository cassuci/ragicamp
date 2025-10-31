# New Architecture Guide

**Last Updated:** October 31, 2024

This document explains the new architecture improvements and how to use them.

## Overview

We've added three major abstractions to improve code quality, prevent bugs, and make experimentation easier:

1. **DocumentCorpus** - Clean document sources (prevents data leakage)
2. **ExperimentConfig** - Type-safe configuration system
3. **OutputManager** - Organized experiment results

## 1. DocumentCorpus

### Purpose

Provide documents for retrieval **WITHOUT answer information**. This prevents data leakage where the retrieval system has access to training answers.

### Key Concept

**Corpus** (documents for retrieval) ≠ **Dataset** (QA pairs for evaluation)

- `corpus/` modules provide clean documents for indexing
- `datasets/` modules provide QA pairs for testing
- Never mix the two!

### Usage

```python
from ragicamp.corpus import WikipediaCorpus, CorpusConfig

# 1. Configure corpus
config = CorpusConfig(
    name="wikipedia_simple",
    source="wikimedia/wikipedia",
    version="20231101.simple",
    max_docs=10000  # Optional: limit for testing
)

# 2. Create corpus
corpus = WikipediaCorpus(config)

# 3. Load documents
for doc in corpus.load():
    print(f"Title: {doc.metadata['title']}")
    print(f"Text: {doc.text[:100]}...")
    # doc has NO answer information!
```

### Available Corpora

Currently available:
- `WikipediaCorpus` - Wikipedia articles (Simple or full English)

Coming soon:
- `PubMedCorpus` - Medical literature
- `ArXivCorpus` - Research papers
- `CustomCorpus` - Your own documents

### Indexing with Corpus

```bash
# New way (recommended)
make index-corpus

# Or manually:
python experiments/scripts/index_corpus.py \
    --corpus-name wikipedia_simple \
    --corpus-version 20231101.simple \
    --max-docs 10000 \
    --artifact-name my_retriever
```

## 2. ExperimentConfig

### Purpose

Single source of truth for experiment configuration. Type-safe, serializable, and version-controlled.

### Components

```python
@dataclass
class ExperimentConfig:
    name: str                        # Experiment identifier
    corpus: CorpusConfig            # Document source
    retriever: RetrieverConfig      # Retrieval settings
    model: ModelConfig              # LLM settings
    evaluation: EvaluationConfig    # Eval settings
```

### Usage

**Option A: Load from YAML (Recommended)**

```python
from ragicamp.config import ExperimentConfig

# Load configuration
config = ExperimentConfig.from_yaml("experiments/configs/my_experiment.yaml")

# All settings are typed and validated
print(f"Using model: {config.model.name}")
print(f"Top-K: {config.retriever.top_k}")
print(f"8-bit: {config.model.load_in_8bit}")
```

**Option B: Create programmatically**

```python
from ragicamp.config import create_fixed_rag_config

# Use preset
config = create_fixed_rag_config(dataset="natural_questions")

# Or build from scratch
config = ExperimentConfig(
    name="my_experiment",
    corpus=CorpusConfig(...),
    retriever=RetrieverConfig(...),
    model=ModelConfig(...),
    evaluation=EvaluationConfig(...)
)
```

### Example YAML

```yaml
# experiments/configs/fixed_rag_nq.yaml
name: "fixed_rag_nq_v1"

corpus:
  name: "wikipedia_simple"
  source: "wikimedia/wikipedia"
  version: "20231101.simple"
  max_docs: 10000

retriever:
  type: "dense"
  embedding_model: "all-MiniLM-L6-v2"
  top_k: 3

model:
  name: "google/gemma-2-2b-it"
  load_in_8bit: true
  device: "cuda"

evaluation:
  dataset: "natural_questions"
  num_examples: 100
  metrics: ["exact_match", "f1"]
```

### Benefits

- ✅ Type safety (no typos, IDE autocomplete)
- ✅ Validation (catch errors early)
- ✅ Reproducibility (save with results)
- ✅ Version control (track changes)
- ✅ Easy experimentation (change one value, re-run)

## 3. OutputManager

### Purpose

Organize experiment results in a clean, discoverable structure.

### Directory Structure

```
outputs/
├── experiments/
│   ├── baseline_direct_llm_v1/
│   │   ├── config.yaml         # Exact config used
│   │   ├── metadata.json       # Timestamp, git hash, etc
│   │   ├── predictions.json    # All predictions
│   │   └── results.json        # Metrics summary
│   │
│   ├── fixed_rag_nq_v1/
│   │   ├── config.yaml
│   │   ├── metadata.json
│   │   ├── contexts.json       # Retrieved contexts (RAG only)
│   │   ├── predictions.json
│   │   └── results.json
│   │
│   └── fixed_rag_nq_v2/
│       └── ...
│
└── comparisons/
    ├── nq_all_experiments.json
    └── by_metric/
        ├── exact_match.json
        └── f1.json
```

### Usage

```python
from ragicamp.output import OutputManager

mgr = OutputManager()

# Create experiment directory
exp_dir = mgr.create_experiment_dir("my_experiment_v1")

# Run experiment...
results = run_evaluation(config)

# Save everything
mgr.save_experiment(config, results, exp_dir)

# List all experiments
experiments = mgr.list_experiments()
for exp in experiments:
    print(f"{exp['experiment_name']}: {exp['dataset']}")

# Filter by dataset
nq_experiments = mgr.list_experiments(dataset="natural_questions")

# Compare experiments
mgr.print_comparison(
    ["baseline_v1", "fixed_rag_v1", "fixed_rag_v2"],
    metrics=["exact_match", "f1"]
)
```

### Makefile Commands

```bash
# List all experiments
make list-experiments

# Compare specific experiments
make compare-experiments EXPERIMENTS='exp1 exp2 exp3'

# Clean old experiments (keep last 10)
make clean-old-experiments
```

## Complete Workflow Example

### 1. Create Experiment Config

```yaml
# experiments/configs/my_rag_experiment.yaml
name: "my_rag_experiment_v1"

corpus:
  name: "wikipedia_simple"
  source: "wikimedia/wikipedia"
  version: "20231101.simple"
  max_docs: 10000

retriever:
  type: "dense"
  embedding_model: "all-MiniLM-L6-v2"
  top_k: 3

model:
  name: "google/gemma-2-2b-it"
  load_in_8bit: true

evaluation:
  dataset: "natural_questions"
  num_examples: 100
  metrics: ["exact_match", "f1"]
```

### 2. Index Corpus

```bash
make index-wiki-small
```

### 3. Run Experiment

```python
from ragicamp.config import ExperimentConfig
from ragicamp.output import OutputManager

# Load config
config = ExperimentConfig.from_yaml("configs/my_rag_experiment.yaml")

# Create output directory
mgr = OutputManager()
exp_dir = mgr.create_experiment_dir(config.name)

# Run evaluation (your evaluation code here)
results = run_evaluation(config)

# Save everything
mgr.save_experiment(config, results, exp_dir)
```

### 4. Compare Results

```bash
make compare-experiments EXPERIMENTS='baseline_v1 my_rag_experiment_v1'
```

## Migration Guide

### Before (Old Way)

```python
# Hard-coded configuration
dataset = load_dataset("wikimedia/wikipedia", "20231101.simple")
retriever = DenseRetriever("my_retriever", "all-MiniLM-L6-v2")

# Index documents (with potential data leakage!)
for item in dataset:
    retriever.index(Document(text=item["question"]))  # WRONG!

# Scattered output files
with open("results.json", 'w') as f:
    json.dump(results, f)
```

### After (New Way)

```python
# Configuration
config = ExperimentConfig.from_yaml("configs/my_experiment.yaml")

# Load corpus (no data leakage!)
corpus = WikipediaCorpus(config.corpus)

# Index documents
retriever = DenseRetriever(...)
for doc in corpus.load():
    retriever.index(doc)  # doc has NO answers

# Organized outputs
mgr = OutputManager()
exp_dir = mgr.create_experiment_dir(config.name)
mgr.save_experiment(config, results, exp_dir)
```

## Best Practices

### DO ✅

1. **Always use DocumentCorpus for indexing**
   - Prevents data leakage
   - Explicit about what's being indexed

2. **Always use ExperimentConfig**
   - Type safety
   - Reproducibility
   - Easy to modify

3. **Always use OutputManager**
   - Clean organization
   - Easy comparisons
   - Full provenance

4. **Separate corpus from dataset**
   - Corpus = documents for retrieval (no answers)
   - Dataset = QA pairs for evaluation (with answers)

5. **Save config with results**
   - Know exactly what was run
   - Can reproduce later

### DON'T ❌

1. **Don't index QA dataset questions**
   - Massive data leakage!
   - Use DocumentCorpus instead

2. **Don't hard-code configuration**
   - Hard to reproduce
   - Hard to compare
   - Use ExperimentConfig

3. **Don't scatter output files**
   - Hard to find
   - Hard to compare
   - Use OutputManager

4. **Don't mix corpus and dataset**
   - Keep them separate!
   - Different purposes

## Demo

Try the new architecture:

```bash
make demo-architecture
```

This will demonstrate:
1. Loading corpus without answers
2. Type-safe configuration
3. Organized output management

## Summary

The new architecture provides:

1. **Safety** - Prevents data leakage
2. **Clarity** - Clear separation of concerns
3. **Reusability** - Easy to add new corpora/datasets
4. **Organization** - Clean, discoverable outputs
5. **Reproducibility** - Full experiment tracking

See `docs/development/ARCHITECTURE_REVIEW.md` for full technical details.

