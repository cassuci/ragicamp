# LLM-as-a-Judge Quick Start Guide

## 🎯 Your Use Case: Binary Performance Evaluation

You want GPT-4 to classify answers as **correct** or **incorrect** for building performance metrics.

---

## 🚀 Quick Start

### 1. Set API Key

```bash
export OPENAI_API_KEY='your-openai-api-key'
```

### 2. Run the Example

```bash
cd /home/gabriel_frontera_cloudwalk_io/ragicamp

# Install if needed
uv sync

# Run LLM judge evaluation
uv run python examples/llm_judge_evaluation.py
```

This will:
- Evaluate 10 Natural Questions examples
- Use Gemma 2B to generate answers
- Use GPT-4 to judge if answers are correct/incorrect
- Output binary classification results

---

## 📊 Output

You'll get results like:

```
Standard Metrics:
  Exact Match:        0.3000
  F1 Score:           0.4500

LLM Judge (GPT-4):
  Average Score:      0.7000
  Correct:            0.7000 (70.0%)
  Incorrect:          0.3000 (30.0%)
```

**Key insight:** LLM Judge often shows higher accuracy than EM/F1 because it understands semantic correctness.

---

## 💻 Code Integration

### Basic Usage

```python
from ragicamp.metrics.llm_judge_qa import LLMJudgeQAMetric
from ragicamp.models.openai import OpenAIModel

# Create GPT-4 judge
judge_model = OpenAIModel("gpt-4o", temperature=0.0)

# Binary judge (correct/incorrect)
llm_judge = LLMJudgeQAMetric(
    judge_model=judge_model,
    judgment_type="binary"  # This gives you binary classification
)

# Add to metrics
metrics = [
    ExactMatchMetric(),
    F1Metric(),
    llm_judge  # Your binary performance metric
]
```

### Results Structure

```python
results = evaluator.evaluate()

# Access binary classification results
print(f"Correct rate: {results['llm_judge_qa_correct']}")
print(f"Incorrect rate: {results['llm_judge_qa_incorrect']}")
print(f"Average score: {results['llm_judge_qa']}")  # 0.0-1.0
```

---

## 🎛️ Judgment Types

### Binary (Recommended for you)
```python
judgment_type="binary"
```
**Output:** correct (1.0) or incorrect (0.0)

**Use when:** You need clean binary classification for performance metrics

### Ternary (More nuanced)
```python
judgment_type="ternary"
```
**Output:** correct (1.0), partially_correct (0.5), or incorrect (0.0)

**Use when:** You want to distinguish partially correct answers

---

## 💰 Cost Estimates

| Model | Cost per question | Speed | When to use |
|-------|------------------|-------|-------------|
| **GPT-4o** | ~$0.01-0.03 | 2-3s | Best quality |
| **GPT-4-turbo** | ~$0.02-0.03 | 2-3s | High quality |
| **GPT-4o-mini** | ~$0.001 | 1-2s | Budget option |

**For 100 questions:**
- GPT-4o: $1-3
- GPT-4o-mini: $0.10

---

## 🎯 Use Cases

### 1. Binary Performance Metrics ✅ (Your goal)
```python
# Evaluate model performance with binary labels
llm_judge = LLMJudgeQAMetric(judge_model, judgment_type="binary")

# Get clean correct/incorrect labels
results = evaluator.evaluate()
accuracy = results['llm_judge_qa_correct']
```

### 2. Error Analysis
```python
# Find questions where EM/F1 disagree with LLM judge
for i, (em, llm) in enumerate(zip(em_scores, llm_scores)):
    if em == 0 and llm == 1.0:
        print(f"Question {i}: EM missed but LLM says correct")
```

### 3. Quality Control
```python
# Sample 10% of production traffic for LLM judgment
import random
sample = random.sample(predictions, len(predictions) // 10)
llm_scores = judge.compute(sample, references, questions)
```

---

## 📁 Output Files

The evaluation creates detailed JSON files:

### `*_predictions.json`
```json
{
  "predictions": [
    {
      "question": "when did ww2 end?",
      "prediction": "World War 2 ended in 1945",
      "metrics": {
        "exact_match": 0.0,
        "f1": 0.5,
        "llm_judge_qa": 1.0  // ✅ Binary: correct
      }
    }
  ]
}
```

### `*_summary.json`
```json
{
  "overall_metrics": {
    "exact_match": 0.30,
    "f1": 0.45,
    "llm_judge_qa": 0.70,           // Average score
    "llm_judge_qa_correct": 0.70,   // 70% correct
    "llm_judge_qa_incorrect": 0.30  // 30% incorrect
  }
}
```

---

## 🔧 Advanced Options

### Use Different Models

```python
# GPT-4o (recommended)
judge = OpenAIModel("gpt-4o", temperature=0.0)

# GPT-4o-mini (cheaper)
judge = OpenAIModel("gpt-4o-mini", temperature=0.0)

# GPT-4-turbo
judge = OpenAIModel("gpt-4-turbo-preview", temperature=0.0)
```

### Custom Prompting

The prompts are designed for QA evaluation but you can extend:

```python
# See src/ragicamp/metrics/llm_judge_qa.py
# Modify _create_judgment_prompt() for custom logic
```

---

## ✅ Comparison: Standard Metrics vs LLM Judge

### Example 1: Semantic Match

```
Question: "Who invented the telephone?"
Reference: "Alexander Graham Bell"
Prediction: "Bell invented the telephone"

Exact Match: 0.0 ❌
F1 Score:    0.25
LLM Judge:   1.0 ✅ (correct - same meaning)
```

### Example 2: Wrong but Similar

```
Question: "When did WWII end?"
Reference: "1945"
Prediction: "The war ended around the mid-1940s"

Exact Match: 0.0 ❌
F1 Score:    0.0 ❌
LLM Judge:   0.5 (partial) or 0.0 (incorrect - not precise enough)
```

---

## 📖 Full Documentation

- **Detailed guide**: `docs/METRICS_RECOMMENDATIONS.md`
- **Implementation**: `src/ragicamp/metrics/llm_judge_qa.py`
- **Example script**: `examples/llm_judge_evaluation.py`

---

## 🎯 Next Steps

1. **Try it:**
   ```bash
   export OPENAI_API_KEY='your-key'
   uv run python examples/llm_judge_evaluation.py
   ```

2. **Integrate into your workflow:**
   - Add to evaluation metrics
   - Use binary labels for performance analysis
   - Compare with EM/F1 results

3. **Analyze results:**
   - Check `outputs/llm_judge_evaluation.json`
   - Compare LLM judgments with automatic metrics
   - Identify edge cases where metrics disagree

---

## 💡 Pro Tips

1. **Use temperature=0.0** for consistent judgments
2. **Always include questions** in the evaluation (required for context)
3. **Sample for cost control** - don't need to judge every answer
4. **Compare with EM/F1** - use all three metrics together
5. **Check edge cases** - where LLM judge differs from automatic metrics

---

**You now have a high-quality binary classifier for building performance metrics!** 🎉

# Using LLM Judge via Config Files

## 🚀 Quick Start

### 1. Set your OpenAI API key

```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

### 2. Run evaluation with LLM judge

```bash
cd /home/gabriel_frontera_cloudwalk_io/ragicamp

# Binary judge (correct/incorrect) - Recommended
make eval-with-llm-judge

# Budget version (GPT-4o-mini, 10x cheaper)
make eval-with-llm-judge-mini

# Ternary judge (correct/partial/incorrect)
make eval-with-llm-judge-ternary
```

---

## 📋 Available Configs

### 1. Binary Judge (Recommended for Performance Metrics)

**File:** `experiments/configs/nq_baseline_with_llm_judge.yaml`

**Command:**
```bash
make eval-with-llm-judge
```

**Features:**
- Uses GPT-4o for high quality
- Binary classification: correct (1.0) or incorrect (0.0)
- 20 examples (~$0.50-1.00 cost)
- Perfect for building performance metrics

**Output metrics:**
- `llm_judge_qa`: Average score (0.0-1.0)
- `llm_judge_qa_correct`: Proportion marked as correct
- `llm_judge_qa_incorrect`: Proportion marked as incorrect

---

### 2. Budget Version (GPT-4o-mini)

**File:** `experiments/configs/nq_baseline_with_llm_judge_mini.yaml`

**Command:**
```bash
make eval-with-llm-judge-mini
```

**Features:**
- Uses GPT-4o-mini (10x cheaper)
- Binary classification
- 50 examples (~$0.05-0.10 cost)
- Good for larger evaluations on a budget

---

### 3. Ternary Judge (More Nuanced)

**File:** `experiments/configs/nq_baseline_with_llm_judge_ternary.yaml`

**Command:**
```bash
make eval-with-llm-judge-ternary
```

**Features:**
- Uses GPT-4o
- Three categories: correct / partially_correct / incorrect
- 20 examples (~$0.50-1.00 cost)
- Better for understanding partial correctness

**Output metrics:**
- `llm_judge_qa`: Average score
- `llm_judge_qa_correct`: Fully correct answers
- `llm_judge_qa_partial`: Partially correct answers
- `llm_judge_qa_incorrect`: Incorrect answers

---

## 🎛️ Config Format

Here's the structure for adding LLM judge to any config:

```yaml
# Your agent and model config
agent:
  type: direct_llm
  name: "my_agent"
  system_prompt: "..."

model:
  type: huggingface
  model_name: "google/gemma-2-2b-it"
  device: "cuda"
  load_in_8bit: true

dataset:
  name: natural_questions
  split: validation
  num_examples: 20
  filter_no_answer: true

# LLM Judge Configuration
judge_model:
  type: openai
  model_name: "gpt-4o"  # or "gpt-4o-mini" for lower cost
  temperature: 0.0      # Keep at 0.0 for consistency

# Metrics including LLM judge
metrics:
  - exact_match
  - f1
  - name: llm_judge_qa
    params:
      judgment_type: "binary"  # or "ternary"

output:
  save_predictions: true
  output_path: "outputs/my_results.json"
```

---

## 💰 Cost Estimates

| Config | Examples | Model | Estimated Cost |
|--------|----------|-------|----------------|
| `nq_baseline_with_llm_judge.yaml` | 20 | GPT-4o | $0.50-1.00 |
| `nq_baseline_with_llm_judge_mini.yaml` | 50 | GPT-4o-mini | $0.05-0.10 |
| `nq_baseline_with_llm_judge_ternary.yaml` | 20 | GPT-4o | $0.50-1.00 |

**Tip:** Start with the mini version to test, then use full GPT-4o for final evaluation.

---

## 📊 Output Structure

### Summary File (`*_summary.json`)

```json
{
  "overall_metrics": {
    "exact_match": 0.30,
    "f1": 0.45,
    "llm_judge_qa": 0.70,           // Average: 70% correct
    "llm_judge_qa_correct": 0.70,   // 70% marked as correct
    "llm_judge_qa_incorrect": 0.30  // 30% marked as incorrect
  }
}
```

### Predictions File (`*_predictions.json`)

```json
{
  "predictions": [
    {
      "question": "when did ww2 end?",
      "prediction": "World War 2 ended in 1945",
      "metrics": {
        "exact_match": 0.0,
        "f1": 0.5,
        "llm_judge_qa": 1.0  // ✅ Marked as correct!
      }
    }
  ]
}
```

---

## 🔧 Customization

### Change Model

Edit the config file:

```yaml
judge_model:
  model_name: "gpt-4-turbo"  # or "gpt-4", "gpt-4o-mini"
```

### Change Number of Examples

```yaml
dataset:
  num_examples: 50  # Adjust based on budget
```

### Change Judgment Type

```yaml
metrics:
  - name: llm_judge_qa
    params:
      judgment_type: "ternary"  # or "binary"
```

---

## 🎯 Best Practices

### 1. **Start Small**
```bash
# Test with 5-10 examples first
# Edit config: num_examples: 5
make eval-with-llm-judge
```

### 2. **Use Mini for Iteration**
```bash
# Use GPT-4o-mini during development
make eval-with-llm-judge-mini
```

### 3. **Final Eval with GPT-4o**
```bash
# Use full GPT-4o for final/production evaluation
make eval-with-llm-judge
```

### 4. **Always Include Standard Metrics**
```yaml
metrics:
  - exact_match  # Fast, strict
  - f1           # Fast, partial credit
  - name: llm_judge_qa  # Semantic correctness
```

---

## 🐛 Troubleshooting

### Error: "OPENAI_API_KEY not set"

```bash
# Set the API key
export OPENAI_API_KEY='your-key-here'

# Verify it's set
echo $OPENAI_API_KEY
```

### Error: "Rate limit exceeded"

**Solution:** Reduce number of examples or add delay:

```yaml
dataset:
  num_examples: 10  # Reduce from 20
```

### High Costs

**Solutions:**
1. Use GPT-4o-mini: `make eval-with-llm-judge-mini`
2. Reduce examples: Edit config `num_examples: 10`
3. Sample strategically: Evaluate subset, extrapolate

---

## 📈 Comparing Results

### Standard Metrics vs LLM Judge

```bash
# Run both
make eval-baseline-quick      # EM + F1 (free, fast)
make eval-with-llm-judge      # +LLM judge (paid, slow)

# Compare in outputs/
ls outputs/
# - gemma_2b_baseline_quick_summary.json
# - gemma_2b_with_llm_judge_summary.json
```

### Analysis Example

```python
import json

# Load results
with open('outputs/gemma_2b_baseline_quick_summary.json') as f:
    baseline = json.load(f)

with open('outputs/gemma_2b_with_llm_judge_summary.json') as f:
    with_judge = json.load(f)

# Compare
print(f"Exact Match:  {baseline['overall_metrics']['exact_match']:.2%}")
print(f"F1 Score:     {baseline['overall_metrics']['f1']:.2%}")
print(f"LLM Judge:    {with_judge['overall_metrics']['llm_judge_qa']:.2%}")
```

---

## ✅ Quick Reference

| Task | Command |
|------|---------|
| **Binary judge (recommended)** | `make eval-with-llm-judge` |
| **Budget version** | `make eval-with-llm-judge-mini` |
| **Ternary classification** | `make eval-with-llm-judge-ternary` |
| **Set API key** | `export OPENAI_API_KEY='...'` |
| **Check results** | `cat outputs/*_summary.json` |

---

## 🎯 Next Steps

1. **Set API key:**
   ```bash
   export OPENAI_API_KEY='your-key'
   ```

2. **Test with mini:**
   ```bash
   make eval-with-llm-judge-mini
   ```

3. **Check results:**
   ```bash
   cat outputs/gemma_2b_with_llm_judge_mini_summary.json
   ```

4. **Analyze:**
   - Compare LLM judge scores with EM/F1
   - Identify questions where metrics disagree
   - Use for building performance metrics

**You now have binary correctness labels for all your answers!** 🎉

