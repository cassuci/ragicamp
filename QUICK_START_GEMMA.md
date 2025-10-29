# 🚀 Quick Start: Run Gemma 2B Baseline in 3 Steps

Get started with RAGiCamp by running the Gemma 2B baseline evaluation!

## Prerequisites

1. **Accept Gemma License** (one-time):
   - Visit: https://huggingface.co/google/gemma-2-2b-it
   - Click "Agree and access repository"

2. **Login to HuggingFace** (one-time):
   ```bash
   uv run huggingface-cli login
   ```

## Step 1: Install Dependencies

```bash
cd ragicamp
uv sync
```

This installs all dependencies in ~30 seconds using `uv`!

## Step 2: Run Quick Test (10 examples)

```bash
make run-gemma2b
```

Or manually:
```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 10 \
    --device cuda
```

## Step 3: Run Full Evaluation (100 examples)

```bash
make run-gemma2b-full
```

Or manually:
```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100 \
    --device cuda
```

## Results

You'll see output like:

```
======================================================================
RESULTS
======================================================================

Dataset: natural_questions
Examples evaluated: 100

Metrics:
  exact_match: 0.2300
  f1: 0.3456

✓ Results saved to: outputs/gemma2b_baseline_results.json
======================================================================
```

## More Options

### Run on CPU (no GPU needed)
```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 10 \
    --device cpu
```

### Save Memory with 8-bit Quantization
```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100 \
    --load-in-8bit
```

### Try Different Datasets
```bash
# HotpotQA (multi-hop reasoning)
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset hotpotqa \
    --num-examples 50

# TriviaQA
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset triviaqa \
    --num-examples 50
```

## Available Make Commands

```bash
make help                # Show all commands
make install             # Install dependencies
make run-gemma2b         # Quick test (10 examples)
make run-gemma2b-full    # Full run (100 examples)
make run-gemma2b-cpu     # Run on CPU
make run-gemma2b-8bit    # Run with 8-bit quantization
```

## Next Steps

1. **Compare with RAG**: Run FixedRAG baseline and compare
2. **Try Adaptive Agents**: Experiment with bandit or MDP agents
3. **Analyze Results**: Load the JSON output for detailed analysis
4. **Read More**: Check `GEMMA2B_QUICKSTART.md` for full documentation

---

**Estimated Time:**
- Quick test (10 examples): ~1-2 minutes
- Full evaluation (100 examples): ~5-10 minutes on GPU

Enjoy! 🎉

