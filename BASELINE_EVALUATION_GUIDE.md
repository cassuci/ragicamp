# Baseline Evaluation Guide - Natural Questions

Complete guide for running baseline (no-retrieval) inference on Natural Questions with comprehensive metrics.

## 🎯 Quick Start

### Recommended Command (All Metrics)

```bash
cd /home/gabriel_frontera_cloudwalk_io/ragicamp

uv run python experiments/scripts/run_gemma2b_baseline.py \
  --dataset natural_questions \
  --num-examples 100 \
  --device cuda \
  --filter-no-answer \
  --metrics exact_match f1 bertscore bleurt \
  --load-in-8bit \
  --output outputs/nq_baseline_full.json
```

### Fast Test (10 examples)

```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
  --dataset natural_questions \
  --num-examples 10 \
  --device cuda \
  --filter-no-answer \
  --metrics exact_match f1 \
  --load-in-8bit
```

---

## 📊 Available Metrics

| Metric | Description | Speed | Use Case | Command Flag |
|--------|-------------|-------|----------|--------------|
| **Exact Match** | Normalized exact string match (SQuAD-style) | ⚡ Fast | Quick baseline | `exact_match` |
| **F1 Score** | Token-level F1 with normalization | ⚡ Fast | Standard QA metric | `f1` |
| **BERTScore** | Neural semantic similarity (DeBERTa) | 🐢 Slow | Semantic evaluation | `bertscore` |
| **BLEURT** | Learned evaluation metric | 🐢 Slow | Human-correlated quality | `bleurt` |
| **LLM Judge** | LLM-based evaluation | 🐌 Very Slow | Qualitative assessment | (custom) |

### Metric Details

#### 1. Exact Match (EM)
- **What it measures**: Binary match after normalization
- **Normalization**: Lowercase, remove punctuation, articles (a/an/the)
- **Best for**: Strict factual accuracy
- **Output**: `exact_match: 0.0-1.0`

#### 2. F1 Score
- **What it measures**: Token-level precision and recall
- **Normalization**: Same as EM
- **Best for**: Partial credit for overlapping tokens
- **Output**: `f1: 0.0-1.0`

#### 3. BERTScore
- **What it measures**: Semantic similarity using contextualized embeddings
- **Model**: DeBERTa (configurable)
- **Best for**: Capturing paraphrases and semantic equivalence
- **Output**: `bertscore_precision`, `bertscore_recall`, `bertscore_f1`

#### 4. BLEURT
- **What it measures**: Learned metric trained to correlate with human judgments
- **Checkpoint**: BLEURT-20 or smaller variants
- **Best for**: Overall answer quality
- **Output**: `bleurt: -2.0 to 1.0` (higher is better)

---

## 🔧 Command Options

### Basic Options

```bash
--dataset natural_questions     # Dataset to use (nq, hotpotqa, triviaqa)
--split validation               # Dataset split
--num-examples 100               # Number of examples to evaluate
--device cuda                    # Device (cuda/cpu)
--output outputs/results.json    # Output path
```

### Model Options

```bash
--load-in-8bit                   # Use 8-bit quantization (saves ~50% memory)
--max-tokens 128                 # Maximum tokens to generate
--temperature 0.7                # Generation temperature (0.0-1.0)
```

### Dataset Filtering

```bash
--filter-no-answer               # Remove questions without explicit answers
```

### Metric Selection

```bash
# Fast metrics only
--metrics exact_match f1

# All metrics
--metrics exact_match f1 bertscore bleurt

# Specific metric configuration
--bertscore-model microsoft/deberta-base-mnli   # Faster BERTScore model
--bleurt-checkpoint BLEURT-20-D3                # Smaller BLEURT checkpoint
```

---

## 📁 Output Files

The evaluation saves **3 JSON files**:

### 1. `natural_questions_questions.json`
Dataset questions (reusable across experiments)
```json
{
  "dataset_name": "natural_questions",
  "num_questions": 100,
  "questions": [
    {
      "id": "nq_validation_0",
      "question": "when did the us break away from england",
      "expected_answer": "1776",
      "all_acceptable_answers": ["1776", "July 4, 1776"]
    }
  ]
}
```

### 2. `gemma_2b_baseline_predictions.json`
Predictions with per-question metrics
```json
{
  "agent_name": "gemma_2b_baseline",
  "dataset_name": "natural_questions",
  "timestamp": "2025-11-11T10:30:00",
  "num_examples": 100,
  "predictions": [
    {
      "question_id": "nq_validation_0",
      "question": "when did the us break away from england",
      "prediction": "The United States declared independence in 1776.",
      "metrics": {
        "exact_match": 0.0,
        "f1": 0.667,
        "bertscore_f1": 0.892,
        "bleurt": 0.45
      }
    }
  ]
}
```

### 3. `gemma_2b_baseline_summary.json`
Overall metrics and statistics
```json
{
  "agent_name": "gemma_2b_baseline",
  "dataset_name": "natural_questions",
  "num_examples": 100,
  "overall_metrics": {
    "exact_match": 0.34,
    "f1": 0.48,
    "bertscore_f1": 0.85,
    "bleurt": 0.38
  },
  "metric_statistics": {
    "exact_match": {
      "mean": 0.34,
      "min": 0.0,
      "max": 1.0,
      "std": 0.474
    }
  }
}
```

---

## 🚀 Usage Examples

### Example 1: Quick Test (10 examples, EM + F1)

```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
  --dataset natural_questions \
  --num-examples 10 \
  --metrics exact_match f1 \
  --load-in-8bit
```

**Expected time**: 2-3 minutes on GPU

### Example 2: Full Evaluation (100 examples, all metrics)

```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
  --dataset natural_questions \
  --num-examples 100 \
  --metrics exact_match f1 bertscore bleurt \
  --load-in-8bit \
  --output outputs/nq_baseline_full.json
```

**Expected time**: 20-30 minutes on GPU

### Example 3: CPU Evaluation (limited metrics)

```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
  --dataset natural_questions \
  --num-examples 10 \
  --device cpu \
  --metrics exact_match f1 \
  --output outputs/nq_baseline_cpu.json
```

**Expected time**: 30-60 minutes on CPU

### Example 4: Custom Python Script

See `experiments/scripts/custom_baseline_eval.py` for a fully customizable example.

```bash
uv run python experiments/scripts/custom_baseline_eval.py
```

---

## 🔍 Advanced: LLM-as-a-Judge

For qualitative evaluation, you can add LLM-as-a-judge:

```python
from ragicamp.metrics.llm_judge import LLMJudgeMetric
from ragicamp.models.openai import OpenAIModel

# Create judge model
judge_model = OpenAIModel("gpt-4")

# Add to metrics
llm_judge = LLMJudgeMetric(
    judge_model=judge_model,
    criteria="accuracy",
    scale=10
)

metrics = [
    ExactMatchMetric(),
    F1Metric(),
    BERTScoreMetric(),
    BLEURTMetric(),
    llm_judge
]
```

---

## 📊 Analyzing Results

### Load and Analyze Predictions

```python
import json
import pandas as pd

# Load predictions
with open('outputs/gemma_2b_baseline_predictions.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame([
    {
        'question': p['question'],
        'prediction': p['prediction'],
        **p['metrics']
    }
    for p in data['predictions']
])

# Find low-scoring questions
low_em = df[df['exact_match'] < 0.5]
print(f"Questions with low EM: {len(low_em)}")

# Correlation between metrics
print(df[['exact_match', 'f1', 'bertscore_f1', 'bleurt']].corr())
```

### Compare Multiple Runs

```python
# Load multiple summaries
with open('outputs/gemma_2b_baseline_summary.json') as f:
    baseline = json.load(f)

with open('outputs/fixed_rag_summary.json') as f:
    rag = json.load(f)

# Compare
for metric in ['exact_match', 'f1', 'bertscore_f1']:
    baseline_score = baseline['overall_metrics'][metric]
    rag_score = rag['overall_metrics'][metric]
    improvement = (rag_score - baseline_score) / baseline_score * 100
    print(f"{metric:20s}: {baseline_score:.3f} → {rag_score:.3f} ({improvement:+.1f}%)")
```

---

## 🐛 Troubleshooting

### BLEURT Checkpoint Not Found

```bash
# Download BLEURT checkpoint manually
mkdir -p ~/.cache/bleurt
cd ~/.cache/bleurt
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D3.zip
unzip BLEURT-20-D3.zip
```

### Out of Memory

```bash
# Use 8-bit quantization
--load-in-8bit

# Or reduce batch size (for BERTScore)
--bertscore-model microsoft/deberta-base-mnli  # Use smaller model

# Or use CPU
--device cpu
```

### Slow Evaluation

```bash
# Start with fast metrics only
--metrics exact_match f1

# Reduce examples
--num-examples 10

# Use smaller models
--bertscore-model microsoft/deberta-base-mnli
--bleurt-checkpoint BLEURT-20-D3
```

---

## 📚 Additional Resources

- **Metrics Guide**: `docs/guides/METRICS_GUIDE.md`
- **Output Structure**: `docs/guides/OUTPUT_STRUCTURE.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Full Documentation**: `docs/USAGE.md`

---

## ✅ Summary Checklist

- [ ] Install dependencies: `cd ragicamp && uv sync`
- [ ] Choose metrics: EM, F1 (fast) + BERTScore, BLEURT (slow but better)
- [ ] Run evaluation: See command examples above
- [ ] Check outputs: 3 JSON files in `outputs/`
- [ ] Analyze results: Use pandas for per-question analysis

**Ready to go!** Everything is implemented - just run the commands! 🚀

