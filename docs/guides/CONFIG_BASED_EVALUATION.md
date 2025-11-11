# Config-Based Evaluation Guide

## 🎯 Why Use Configs?

✅ **Reproducible** - Same config = same experiment  
✅ **Shareable** - Easy to share experiment settings  
✅ **Version Control** - Track experiment changes in git  
✅ **No Code Changes** - Switch approaches by changing config files  
✅ **Compare Easily** - Run different approaches with same settings  

---

## 🚀 Quick Start

### 1. Quick Test (10 examples, fast metrics)

```bash
cd /home/gabriel_frontera_cloudwalk_io/ragicamp

uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/nq_baseline_gemma2b_quick.yaml \
  --mode eval
```

**Time**: ~2-3 minutes  
**Metrics**: Exact Match, F1

---

### 2. Full Baseline Evaluation (100 examples, all metrics)

```bash
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/nq_baseline_gemma2b_full.yaml \
  --mode eval
```

**Time**: ~20-25 minutes  
**Metrics**: Exact Match, F1, BERTScore, BLEURT

---

### 3. Maximum Quality (100 examples, best metric models)

```bash
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/nq_baseline_gemma2b_all_metrics.yaml \
  --mode eval
```

**Time**: ~30-40 minutes  
**Metrics**: All metrics with highest-quality checkpoints

---

## 📋 Available Configs

| Config File | Agent | Examples | Metrics | Use Case |
|-------------|-------|----------|---------|----------|
| `nq_baseline_gemma2b_quick.yaml` | DirectLLM | 10 | EM, F1 | Quick test |
| `nq_baseline_gemma2b_full.yaml` | DirectLLM | 100 | EM, F1, BERT, BLEURT | Standard eval |
| `nq_baseline_gemma2b_all_metrics.yaml` | DirectLLM | 100 | All (best quality) | Full eval |
| `nq_fixed_rag_gemma2b.yaml` | FixedRAG | 100 | All | RAG comparison |
| `baseline_direct.yaml` | DirectLLM | 100 | EM, F1 | Flan-T5 baseline |
| `baseline_rag.yaml` | FixedRAG | 100 | EM, F1 | Flan-T5 RAG |

---

## 🔧 Config File Structure

### Basic Config Format

```yaml
# Agent configuration
agent:
  type: direct_llm  # Options: direct_llm, fixed_rag, bandit_rag, mdp_rag
  name: "my_agent_name"
  system_prompt: "Your system prompt here"

# Model configuration
model:
  type: huggingface
  model_name: "google/gemma-2-2b-it"
  device: "cuda"
  load_in_8bit: true  # Save memory

# Dataset configuration
dataset:
  name: natural_questions  # Options: natural_questions, hotpotqa, triviaqa
  split: validation
  num_examples: 100
  filter_no_answer: true  # Remove questions without answers

# Metrics configuration
metrics:
  - exact_match
  - f1
  - name: bertscore
    params:
      model_type: "microsoft/deberta-base-mnli"
  - name: bleurt
    params:
      checkpoint: "BLEURT-20-D3"

# Output configuration
output:
  save_predictions: true
  output_path: "outputs/my_results.json"
```

---

## 📊 Switching Between Approaches

### Baseline → RAG Comparison

**Step 1**: Run baseline

```bash
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/nq_baseline_gemma2b_full.yaml \
  --mode eval
```

**Step 2**: Run RAG (requires indexed corpus)

```bash
# First, index corpus (once)
make index-wiki-small

# Then run RAG evaluation
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/nq_fixed_rag_gemma2b.yaml \
  --mode eval
```

**Step 3**: Compare results

```bash
# Results are in outputs/
ls outputs/

# Output files:
# - nq_baseline_gemma2b_full.json
# - nq_fixed_rag_gemma2b.json
```

---

## 🎛️ Customizing Configs

### Change Number of Examples

```yaml
dataset:
  num_examples: 50  # Test on 50 examples instead
```

### Use Different Model

```yaml
model:
  model_name: "meta-llama/Llama-2-7b-chat-hf"  # Use Llama instead
  device: "cuda"
  load_in_8bit: true
```

### Select Specific Metrics

```yaml
# Fast metrics only
metrics:
  - exact_match
  - f1

# Or all metrics
metrics:
  - exact_match
  - f1
  - bertscore
  - bleurt
```

### Adjust BERTScore Model

```yaml
metrics:
  - name: bertscore
    params:
      model_type: "microsoft/deberta-xlarge-mnli"  # Better but slower
```

### Adjust BLEURT Checkpoint

```yaml
metrics:
  - name: bleurt
    params:
      checkpoint: "BLEURT-20"  # Better quality, larger download
```

---

## 📁 Output Structure

Each evaluation creates **3 JSON files**:

### 1. Dataset Questions
`{dataset}_questions.json` - Reusable across experiments

```json
{
  "dataset_name": "natural_questions",
  "num_questions": 100,
  "questions": [...]
}
```

### 2. Predictions with Per-Question Metrics
`{agent_name}_predictions.json`

```json
{
  "agent_name": "gemma_2b_baseline",
  "predictions": [
    {
      "question": "when did...",
      "prediction": "The answer is...",
      "metrics": {
        "exact_match": 0.0,
        "f1": 0.667,
        "bertscore_f1": 0.892
      }
    }
  ]
}
```

### 3. Summary with Overall Metrics
`{agent_name}_summary.json`

```json
{
  "agent_name": "gemma_2b_baseline",
  "overall_metrics": {
    "exact_match": 0.34,
    "f1": 0.48,
    "bertscore_f1": 0.85
  }
}
```

---

## 🔄 Workflow Example

### Comparing Multiple Approaches

```bash
cd /home/gabriel_frontera_cloudwalk_io/ragicamp

# 1. Quick test to verify everything works
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/nq_baseline_gemma2b_quick.yaml \
  --mode eval

# 2. Full baseline evaluation
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/nq_baseline_gemma2b_full.yaml \
  --mode eval

# 3. Index corpus for RAG (one-time)
make index-wiki-small

# 4. RAG evaluation
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/nq_fixed_rag_gemma2b.yaml \
  --mode eval

# 5. Compare results
ls outputs/
# You now have:
# - gemma_2b_baseline_summary.json
# - gemma_2b_fixed_rag_summary.json
```

---

## 💡 Advanced: Creating Your Own Config

### Template

```bash
# Copy existing config
cp experiments/configs/nq_baseline_gemma2b_full.yaml \
   experiments/configs/my_experiment.yaml

# Edit with your settings
vim experiments/configs/my_experiment.yaml

# Run
uv run python experiments/scripts/run_experiment.py \
  --config experiments/configs/my_experiment.yaml \
  --mode eval
```

### Example: Different Dataset

```yaml
# Try HotpotQA instead
dataset:
  name: hotpotqa  # Changed from natural_questions
  split: validation
  num_examples: 50
```

### Example: CPU Evaluation

```yaml
model:
  device: "cpu"  # Changed from cuda
  load_in_8bit: false  # Not supported on CPU
```

---

## 🐛 Troubleshooting

### Config Not Found

```bash
# Use absolute path
uv run python experiments/scripts/run_experiment.py \
  --config /full/path/to/config.yaml \
  --mode eval
```

### Missing Metrics

```
⚠️  Skipping BERTScore (not installed)
```

**Solution**: Install metrics dependencies

```bash
cd /home/gabriel_frontera_cloudwalk_io/ragicamp
uv sync  # Installs all dependencies including metrics
```

### Out of Memory

**Solution**: Enable quantization in config

```yaml
model:
  load_in_8bit: true  # Reduces memory by ~50%
```

---

## 📊 Analyzing Results

### Load and Compare

```python
import json
import pandas as pd

# Load baseline
with open('outputs/gemma_2b_baseline_summary.json') as f:
    baseline = json.load(f)

# Load RAG
with open('outputs/gemma_2b_fixed_rag_summary.json') as f:
    rag = json.load(f)

# Compare
metrics = ['exact_match', 'f1', 'bertscore_f1', 'bleurt']
comparison = pd.DataFrame({
    'Baseline': [baseline['overall_metrics'][m] for m in metrics],
    'RAG': [rag['overall_metrics'][m] for m in metrics],
}, index=metrics)

print(comparison)
# Compute improvement
comparison['Improvement %'] = (
    (comparison['RAG'] - comparison['Baseline']) / 
    comparison['Baseline'] * 100
)
print(comparison)
```

---

## ✅ Best Practices

1. **Start with quick config** to verify setup
2. **Use descriptive names** in configs (include model, dataset, settings)
3. **Version control configs** - commit to git
4. **Document changes** - add comments in YAML
5. **Reuse configs** - copy and modify for new experiments
6. **Compare consistently** - use same num_examples for fair comparison

---

## 📚 Summary

| Task | Command |
|------|---------|
| **Quick test** | `run_experiment.py --config nq_baseline_gemma2b_quick.yaml --mode eval` |
| **Full baseline** | `run_experiment.py --config nq_baseline_gemma2b_full.yaml --mode eval` |
| **RAG eval** | `run_experiment.py --config nq_fixed_rag_gemma2b.yaml --mode eval` |
| **Custom config** | Copy existing config, edit, and run |

**Key Advantage**: Change RAG approach by just changing the config file! 🚀

