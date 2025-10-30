# RAGiCamp Quick Reference

## Installation
```bash
cd ragicamp
uv sync
```

## Run Gemma 2B Baseline

### Quick Test (10 examples, filtered)
```bash
make run-gemma2b
```

### Full Evaluation (100 examples, filtered)
```bash
make run-gemma2b-full
```

### Custom Options
```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100 \
    --device cuda \
    --filter-no-answer \
    --output outputs/my_results.json
```

## Results JSON Structure

```json
{
  "results": {
    "exact_match": 0.23,
    "f1": 0.34,
    "num_examples": 100
  },
  "predictions": [
    {
      "id": "nq_validation_0",
      "question": "who sang i think we're alone now",
      "prediction": "Tiffany",
      "expected_answer": "Tiffany",           // ← Primary expected answer
      "all_acceptable_answers": ["Tiffany"],  // ← All valid variants
      "references": ["Tiffany"],              // ← Backward compatible
      "metadata": {"agent_type": "direct_llm"}
    }
  ]
}
```

## Filtering Questions

### Command-line
```bash
# With filtering (recommended)
--filter-no-answer

# Without filtering
# (omit the flag)
```

### Programmatic
```python
from ragicamp.datasets.nq import NaturalQuestionsDataset

# Method 1: Filter in-place
dataset = NaturalQuestionsDataset(split="validation")
dataset.filter_with_answers()

# Method 2: Get filtered list
dataset = NaturalQuestionsDataset(split="validation")
filtered = dataset.get_examples_with_answers(n=100)
```

## Common Commands

```bash
make help                      # Show all commands
make install                   # Install dependencies
make install-all               # Install with all extras
make run-gemma2b               # Quick test (10 examples)
make run-gemma2b-full          # Full run (100 examples)
make run-gemma2b-cpu           # Run on CPU
make run-gemma2b-8bit          # With 8-bit quantization
make run-gemma2b-bertscore     # With BERTScore metric
make run-gemma2b-bleurt        # With BLEURT metric
make run-gemma2b-all-metrics   # All metrics (EM, F1, BERT, BLEURT)
make clean                     # Clean outputs
```

## Datasets

- `natural_questions` - Google's Natural Questions
- `hotpotqa` - Multi-hop reasoning questions
- `triviaqa` - Trivia questions

## Device Options

- `cuda` - GPU (default, fastest)
- `cpu` - CPU (slower, no GPU needed)

## Memory Options

- Default: Full precision (~4GB GPU RAM)
- `--load-in-8bit`: 8-bit quantization (~2GB GPU RAM)

## Metrics

### Default (Fast)
```bash
--metrics exact_match f1
```

### With BERTScore (Semantic)
```bash
--metrics exact_match f1 bertscore
```

### With BLEURT (Best Quality)
```bash
--metrics exact_match f1 bleurt
```

### All Metrics
```bash
--metrics exact_match f1 bertscore bleurt
```

**Note**: BERTScore and BLEURT require:
```bash
uv sync --extra metrics
```

## Documentation

- `QUICK_START_GEMMA.md` - 3-step quick start
- `GEMMA2B_QUICKSTART.md` - Comprehensive guide
- `METRICS_GUIDE.md` - **NEW: Metrics guide**
- `ANSWER_FILTERING_UPDATE.md` - Filtering guide
- `USAGE.md` - Detailed usage
- `ARCHITECTURE.md` - System design

## Troubleshooting

**Error: License not accepted**
- Visit: https://huggingface.co/google/gemma-2-2b-it
- Click "Agree and access repository"
- Run: `uv run huggingface-cli login`

**Out of memory**
- Use: `--load-in-8bit`
- Or: `--device cpu`
- Or: Reduce `--num-examples`

## Example Workflow

```bash
# 1. Install
uv sync

# 2. Accept license & login (first time only)
uv run huggingface-cli login

# 3. Run quick test
make run-gemma2b

# 4. Check results
cat outputs/gemma2b_baseline_results.json | jq '.predictions[0]'

# 5. Run full evaluation
make run-gemma2b-full

# 6. Compare with other datasets
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset hotpotqa --num-examples 100 --filter-no-answer
```

## Git Commits

```
d2b1cd5 Add expected answer to results and question filtering
50c3957 Add comprehensive update summary
fa06364 Add quick start guide for Gemma 2B baseline
322eb2a Switch to uv package manager and add Gemma 2B baseline
```

---

**Need help?** Check the documentation files listed above or run `make help`

