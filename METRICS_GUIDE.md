# Metrics Guide

RAGiCamp supports multiple evaluation metrics for assessing answer quality.

## Available Metrics

### 1. Exact Match (EM)
**Type**: String matching  
**Range**: 0.0 - 1.0  
**Description**: Binary metric - 1 if prediction exactly matches any reference (after normalization)

**Pros**:
- Simple and interpretable
- Fast to compute
- Good for factoid questions

**Cons**:
- Very strict - minor variations count as wrong
- Doesn't capture semantic similarity

### 2. F1 Score
**Type**: Token overlap  
**Range**: 0.0 - 1.0  
**Description**: Harmonic mean of precision and recall at token level

**Pros**:
- More lenient than exact match
- Rewards partial answers
- Fast to compute

**Cons**:
- Token-based, not semantic
- Sensitive to word choice

### 3. BERTScore
**Type**: Semantic similarity (neural)  
**Range**: 0.0 - 1.0  
**Description**: Uses BERT embeddings to measure semantic similarity

**Pros**:
- Captures semantic meaning
- Robust to paraphrasing
- Correlates well with human judgment

**Cons**:
- Slower (requires model inference)
- Requires additional dependencies

**Models**: 
- `microsoft/deberta-base-mnli` (default - good speed/accuracy tradeoff)
- `microsoft/deberta-xlarge-mnli` (more accurate but slower)
- `roberta-large-mnli` (alternative)

### 4. BLEURT
**Type**: Learned metric  
**Range**: Variable (checkpoint-dependent)  
**Description**: Trained to predict human judgments of text quality

**Pros**:
- State-of-the-art correlation with human judgment
- Handles fluency and naturalness
- Good for generation tasks

**Cons**:
- Slowest metric
- Large model downloads
- More complex setup

**Checkpoints**:
- `BLEURT-20` (default - recommended)
- `bleurt-large-512` (alternative)

## Installation

### Default Metrics (EM, F1)
```bash
# Already included with base installation
uv sync
```

### Advanced Metrics (BERTScore, BLEURT)
```bash
# Install optional metrics dependencies
uv sync --extra metrics
```

This installs:
- `bert-score` - For BERTScore
- `bleurt` - For BLEURT (from Google Research)

## Usage

### Command-Line

**Use default metrics (EM, F1)**:
```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100
```

**Add BERTScore**:
```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100 \
    --metrics exact_match f1 bertscore
```

**Add BLEURT**:
```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100 \
    --metrics exact_match f1 bleurt
```

**Use all metrics**:
```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100 \
    --metrics exact_match f1 bertscore bleurt
```

**Custom BERTScore model**:
```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100 \
    --metrics bertscore \
    --bertscore-model microsoft/deberta-xlarge-mnli
```

### Make Commands

```bash
# Default metrics (EM, F1)
make run-gemma2b-full

# With BERTScore
make run-gemma2b-bertscore

# With BLEURT
make run-gemma2b-bleurt

# All metrics
make run-gemma2b-all-metrics
```

### Programmatic

```python
from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric
from ragicamp.metrics.bertscore import BERTScoreMetric
from ragicamp.metrics.bleurt import BLEURTMetric
from ragicamp.evaluation.evaluator import Evaluator

# Create metrics
metrics = [
    ExactMatchMetric(),
    F1Metric(),
    BERTScoreMetric(model_type="microsoft/deberta-base-mnli"),
    BLEURTMetric(checkpoint="BLEURT-20")
]

# Evaluate
evaluator = Evaluator(agent=agent, dataset=dataset, metrics=metrics)
results = evaluator.evaluate()

print(results)
# {
#   "exact_match": 0.23,
#   "f1": 0.34,
#   "bertscore_precision": 0.78,
#   "bertscore_recall": 0.81,
#   "bertscore_f1": 0.79,
#   "bleurt": 0.42
# }
```

## Results Format

When using multiple metrics, results include all scores:

```json
{
  "results": {
    "exact_match": 0.23,
    "f1": 0.34,
    "bertscore_precision": 0.78,
    "bertscore_recall": 0.81,
    "bertscore_f1": 0.79,
    "bleurt": 0.42,
    "num_examples": 100
  },
  "predictions": [...]
}
```

## Performance Comparison

| Metric | Speed | Memory | Quality | Use Case |
|--------|-------|--------|---------|----------|
| Exact Match | ⚡⚡⚡ | 💾 | ⭐⭐ | Factoid QA |
| F1 | ⚡⚡⚡ | 💾 | ⭐⭐⭐ | General QA |
| BERTScore | ⚡⚡ | 💾💾 | ⭐⭐⭐⭐ | Semantic matching |
| BLEURT | ⚡ | 💾💾💾 | ⭐⭐⭐⭐⭐ | Human-like judgment |

**Estimated time for 100 examples** (on GPU):
- EM + F1: ~2 minutes
- + BERTScore: ~5 minutes
- + BLEURT: ~10 minutes

## Recommendations

### For Quick Iteration
Use default metrics:
```bash
--metrics exact_match f1
```

### For Better Evaluation
Add BERTScore (good speed/quality tradeoff):
```bash
--metrics exact_match f1 bertscore
```

### For Publication/Final Results
Use all metrics:
```bash
--metrics exact_match f1 bertscore bleurt
```

### For Different Tasks

**Factoid QA** (e.g., "What year..."):
- Exact Match + F1 (fast and appropriate)

**Open-ended QA** (e.g., "How does..."):
- BERTScore + BLEURT (captures semantic quality)

**Generation Tasks**:
- BLEURT (best for fluency and naturalness)

## Troubleshooting

### BERTScore Issues

**Error: "CUDA out of memory"**
- Use smaller model: `--bertscore-model microsoft/deberta-base-mnli`
- Or use CPU (slower): The model will automatically fall back

**Error: "No module named 'bert_score'"**
```bash
uv sync --extra metrics
```

### BLEURT Issues

**Error: "Failed to load BLEURT checkpoint"**
- First time download may take a while
- Checkpoint size: ~1.5GB for BLEURT-20
- Check internet connection

**Error: "No module named 'bleurt'"**
```bash
uv sync --extra metrics
```

**BLEURT is very slow**
- This is expected - BLEURT is the slowest metric
- Reduce dataset size: `--num-examples 10`
- Consider using only BERTScore for faster feedback

## Example Workflows

### Workflow 1: Quick Development Cycle
```bash
# Fast iteration with default metrics
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 10 \
    --metrics exact_match f1
```

### Workflow 2: Thorough Evaluation
```bash
# Full evaluation with semantic metrics
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100 \
    --metrics exact_match f1 bertscore \
    --filter-no-answer
```

### Workflow 3: Publication-Ready
```bash
# All metrics for comprehensive analysis
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100 \
    --metrics exact_match f1 bertscore bleurt \
    --filter-no-answer
```

## Interpreting Results

### Exact Match = 0.23, F1 = 0.34
- Model gets exact answer 23% of the time
- Has 34% token overlap on average
- Significant room for improvement

### + BERTScore F1 = 0.79
- Answers are semantically similar to references
- Model understands the question but uses different words
- Better than EM/F1 suggest

### + BLEURT = 0.42
- Moderate quality by human standards
- Consider fluency, not just accuracy
- BLEURT < 0 is poor, > 0.5 is good

## Custom Metrics

You can add your own metrics by extending the `Metric` base class:

```python
from ragicamp.metrics.base import Metric

class MyCustomMetric(Metric):
    def __init__(self):
        super().__init__(name="my_metric")
    
    def compute(self, predictions, references, **kwargs):
        # Your metric logic here
        scores = []
        for pred, ref in zip(predictions, references):
            score = your_metric_function(pred, ref)
            scores.append(score)
        return sum(scores) / len(scores)
```

## Summary

✅ **Default metrics** (EM, F1): Fast, good for factoid QA  
✅ **BERTScore**: Best speed/quality tradeoff for semantic evaluation  
✅ **BLEURT**: Most human-like, best for final evaluation  
✅ **Mix and match**: Use `--metrics` to choose what you need  

Start with default metrics, add BERTScore for better evaluation, use BLEURT for publication-ready results.

