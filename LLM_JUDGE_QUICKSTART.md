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

