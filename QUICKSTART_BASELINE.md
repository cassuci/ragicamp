# 🚀 Quick Start: Baseline Evaluation on Natural Questions

## ⚡ TL;DR - Copy & Paste

```bash
# Navigate to ragicamp
cd /home/gabriel_frontera_cloudwalk_io/ragicamp

# Option 1: Fast test (10 examples, ~3 minutes)
uv run python experiments/scripts/run_gemma2b_baseline.py \
  --dataset natural_questions \
  --num-examples 10 \
  --device cuda \
  --filter-no-answer \
  --metrics exact_match f1 bertscore bleurt \
  --load-in-8bit

# Option 2: Full evaluation (100 examples, ~25 minutes)
uv run python experiments/scripts/run_gemma2b_baseline.py \
  --dataset natural_questions \
  --num-examples 100 \
  --device cuda \
  --filter-no-answer \
  --metrics exact_match f1 bertscore bleurt \
  --load-in-8bit \
  --output outputs/nq_baseline_full.json
```

## 📊 What You Get

3 JSON files saved in `outputs/`:
1. **`natural_questions_questions.json`** - Dataset (reusable)
2. **`gemma_2b_baseline_predictions.json`** - All predictions + per-question metrics
3. **`gemma_2b_baseline_summary.json`** - Overall scores + statistics

## 📈 Metrics Computed

| Metric | What it measures | Range |
|--------|------------------|-------|
| **exact_match** | Exact string match (normalized) | 0.0-1.0 |
| **f1** | Token overlap (precision + recall) | 0.0-1.0 |
| **bertscore_f1** | Semantic similarity (neural) | 0.0-1.0 |
| **bleurt** | Learned quality metric | -2.0 to 1.0 |

## ✅ That's It!

**No implementation needed** - everything is already ready to use! 

See `BASELINE_EVALUATION_GUIDE.md` for detailed documentation.

