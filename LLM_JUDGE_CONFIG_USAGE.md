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

