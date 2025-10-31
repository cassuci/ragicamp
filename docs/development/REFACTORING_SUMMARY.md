# Output Structure Refactoring Summary

## What Changed

### Before (5 files per run) ❌
```
outputs/
  gemma2b_baseline_results.json           # 100KB - Full results
  gemma2b_baseline_results_metrics.json   # 2KB - Metrics (redundant)
  gemma2b_baseline_results_metrics.txt    # 1KB - Text (redundant)
  gemma2b_baseline_results_per_question.json  # 80KB - Per-Q
  gemma2b_baseline_results_per_question.csv   # 50KB - CSV
```

**Total: 233KB, 5 files**

### After (3 files per run) ✅
```
outputs/
  natural_questions_questions.json    # 30KB - Questions (reusable!)
  gemma_2b_baseline_predictions.json  # 90KB - Predictions + metrics
  gemma_2b_baseline_summary.json      # 3KB - Summary + stats
```

**Total: 123KB for first run, then ~93KB per additional agent (questions reused!)**

## Key Benefits

### 1. Separation of Concerns
- **Questions**: Dataset only (reusable)
- **Predictions**: Agent predictions + per-question metrics
- **Summary**: Overall metrics + statistics

### 2. Reusability
- Questions file created once per dataset
- Shared across all agent runs
- 30KB saved per additional agent

### 3. Async-Friendly
- Each agent writes to its own predictions file
- No conflicts when running in parallel
- Enables distributed evaluation

### 4. Cleaner Code
- Removed ~100 lines of redundant saving logic
- Added `_compute_metric_statistics()` for stats
- More maintainable structure

### 5. Better Analysis
- Per-question metrics inline with predictions
- Statistics (mean, min, max, std) in summary
- Easy to compare agents

## Migration Path

### Old Format Loading
```python
# Old format (still works with old files)
with open('outputs/gemma2b_baseline_results.json') as f:
    old_data = json.load(f)
    predictions = old_data['predictions']
    metrics = old_data['results']
```

### New Format Loading
```python
# Load questions (once)
with open('outputs/natural_questions_questions.json') as f:
    questions = json.load(f)

# Load agent predictions
with open('outputs/gemma_2b_baseline_predictions.json') as f:
    predictions = json.load(f)

# Load summary
with open('outputs/gemma_2b_baseline_summary.json') as f:
    summary = json.load(f)

# Quick access
print(f"F1: {summary['overall_metrics']['f1']:.4f}")
print(f"Range: {summary['metric_statistics']['f1']['min']:.2f}-"
      f"{summary['metric_statistics']['f1']['max']:.2f}")
```

## Code Changes

### evaluator.py

#### Added: `_compute_metric_statistics()`
Computes mean, min, max, std for each metric across questions.

#### Refactored: `_save_results()`
- Removed: 80+ lines of redundant file writing
- Added: Clean 3-file output structure
- Added: Metric statistics computation
- Added: Path-based file naming

**Before:**
```python
def _save_results(...):
    # Save main results
    with open(output_path, 'w') as f: ...
    # Save metrics JSON
    with open(metrics_path, 'w') as f: ...
    # Save metrics TXT
    with open(txt_path, 'w') as f: ...
    # Save per-question JSON
    with open(per_question_path, 'w') as f: ...
    # Save per-question CSV
    with open(csv_path, 'w') as f: ...
```

**After:**
```python
def _save_results(...):
    # 1. Questions (reusable)
    with open(questions_path, 'w') as f: ...
    # 2. Predictions + metrics
    with open(predictions_path, 'w') as f: ...
    # 3. Summary + stats
    with open(summary_path, 'w') as f: ...
```

## Async Example

```python
import asyncio
from ragicamp.evaluation import Evaluator

async def evaluate_agent(agent, dataset, metrics):
    """Evaluate one agent (async-safe)."""
    evaluator = Evaluator(
        agent=agent,
        dataset=dataset,
        metrics=metrics,
        output_path=f"outputs/{agent.name}_results.json"
    )
    return await evaluator.evaluate_async()

# Run 3 agents in parallel
agents = [gemma_2b, gpt4, llama3]
results = await asyncio.gather(*[
    evaluate_agent(agent, dataset, metrics) for agent in agents
])

# Creates:
# - natural_questions_questions.json (shared)
# - gemma_2b_baseline_predictions.json
# - gemma_2b_baseline_summary.json
# - gpt4_baseline_predictions.json
# - gpt4_baseline_summary.json
# - llama3_baseline_predictions.json
# - llama3_baseline_summary.json
```

## Documentation

- **OUTPUT_STRUCTURE.md** - Complete guide to new format
- **examples/new_output_structure/** - Example files
- **examples/analyze_per_question_metrics.py** - Analysis script

## Testing

No changes to evaluation API:
```bash
make run-gemma2b-full  # Works exactly the same!
```

Just cleaner output! 🎉
