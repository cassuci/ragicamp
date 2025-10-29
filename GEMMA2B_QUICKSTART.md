# Gemma 2B Baseline Quick Start

This guide shows how to quickly run the Gemma 2B baseline evaluation.

## What is this?

This evaluates Google's **Gemma 2 2B Instruct** model as a baseline for RAG comparison. The model answers questions **without any retrieval or context**, relying solely on its pretrained knowledge.

## Prerequisites

1. **Install dependencies:**
   ```bash
   cd ragicamp
   uv sync
   ```

2. **Accept Gemma license** (first time only):
   - Go to: https://huggingface.co/google/gemma-2-2b-it
   - Click "Agree and access repository"

3. **Login to HuggingFace** (first time only):
   ```bash
   uv run huggingface-cli login
   ```
   Enter your HuggingFace token when prompted.

## Quick Run

### Option 1: Using the Dedicated Script (Recommended)

```bash
# Basic run with 100 examples
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100

# Run on CPU (slower but no GPU needed)
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 10 \
    --device cpu

# Use 8-bit quantization (saves ~50% memory)
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100 \
    --load-in-8bit

# Try different datasets
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset hotpotqa \
    --num-examples 50

uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset triviaqa \
    --num-examples 50
```

### Option 2: Using Config File

```bash
uv run python experiments/scripts/run_experiment.py \
    --config experiments/configs/gemma2b_baseline.yaml \
    --mode eval
```

## Script Options

```
--dataset          Dataset to use (natural_questions, hotpotqa, triviaqa)
--split            Dataset split (train, validation, test) [default: validation]
--num-examples     Number of examples to evaluate [default: 100]
--device           Device to use (cuda, cpu) [default: cuda]
--load-in-8bit     Use 8-bit quantization to save memory
--output           Output path for results [default: outputs/gemma2b_baseline_results.json]
--max-tokens       Maximum tokens to generate [default: 128]
--temperature      Generation temperature [default: 0.7]
```

## Example Usage

### Quick Test (10 examples on CPU)
```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 10 \
    --device cpu
```

### Full Evaluation (100 examples on GPU)
```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100 \
    --device cuda
```

### Memory-Efficient Run (8-bit quantization)
```bash
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100 \
    --load-in-8bit
```

### Multiple Datasets Comparison
```bash
# Natural Questions
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100 \
    --output outputs/gemma2b_nq.json

# HotpotQA
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset hotpotqa \
    --num-examples 100 \
    --output outputs/gemma2b_hotpot.json

# TriviaQA
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset triviaqa \
    --num-examples 100 \
    --output outputs/gemma2b_trivia.json
```

## Output

The script will:
1. Load the Gemma 2B model
2. Create a DirectLLM agent (no retrieval)
3. Load the specified dataset
4. Generate answers for all examples
5. Compute metrics (Exact Match, F1)
6. Save detailed results to JSON

Example output:
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

## Results File

The JSON output contains:
- Overall metrics (Exact Match, F1)
- Per-example predictions and references
- Metadata (agent name, dataset, etc.)

Example:
```json
{
  "results": {
    "exact_match": 0.23,
    "f1": 0.3456,
    "num_examples": 100
  },
  "predictions": [
    {
      "id": "nq_validation_0",
      "question": "who sang the song i think we're alone now",
      "prediction": "Tiffany",
      "references": ["Tiffany"]
    },
    ...
  ]
}
```

## Troubleshooting

### Error: "You need to accept the license"
- Visit https://huggingface.co/google/gemma-2-2b-it
- Click "Agree and access repository"
- Run `uv run huggingface-cli login`

### Error: Out of memory
- Use `--load-in-8bit` flag
- Reduce `--num-examples`
- Use `--device cpu` (slower but uses less memory)

### Error: Model not found
- Make sure you're logged in: `uv run huggingface-cli login`
- Check internet connection (model downloads on first run)

### Slow generation
- Use GPU: `--device cuda`
- Reduce dataset size: `--num-examples 10`
- Use smaller max_tokens: `--max-tokens 64`

## Next Steps

After running the baseline, you can:

1. **Compare with RAG**: Run the FixedRAG baseline and compare scores
2. **Try adaptive RAG**: Train bandit or MDP agents
3. **Analyze results**: Load the JSON output and analyze per-example performance
4. **Tune parameters**: Adjust temperature, max_tokens, etc.

## Programmatic Usage

You can also use the components directly in Python:

```python
from ragicamp.agents.direct_llm import DirectLLMAgent
from ragicamp.models.huggingface import HuggingFaceModel
from ragicamp.datasets.nq import NaturalQuestionsDataset
from ragicamp.evaluation.evaluator import Evaluator
from ragicamp.metrics.exact_match import ExactMatchMetric, F1Metric

# Create model
model = HuggingFaceModel(
    model_name="google/gemma-2-2b-it",
    device="cuda",
    load_in_8bit=False
)

# Create agent
agent = DirectLLMAgent(name="gemma2b", model=model)

# Load dataset
dataset = NaturalQuestionsDataset(split="validation")
dataset.examples = dataset.examples[:100]

# Evaluate
evaluator = Evaluator(
    agent=agent,
    dataset=dataset,
    metrics=[ExactMatchMetric(), F1Metric()]
)

results = evaluator.evaluate(
    save_predictions=True,
    output_path="results.json"
)

print(results)
```

## Performance Notes

- **GPU**: ~2-3 minutes for 100 examples
- **CPU**: ~15-20 minutes for 100 examples
- **Memory**: ~4GB GPU RAM (full precision), ~2GB (8-bit)

Enjoy experimenting! 🚀

