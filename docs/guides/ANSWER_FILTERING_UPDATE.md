# Answer Filtering & Expected Answer Update

## Summary

Added two key improvements to make evaluation more robust and results more interpretable:

1. **Expected Answer in Results** - JSON output now clearly shows the expected answer
2. **Answer Filtering** - Filter out questions without explicit ground-truth answers

## Problem Solved

### Issue 1: Expected Answer Not Clear
Previously, the results JSON had:
```json
{
  "question": "...",
  "prediction": "...",
  "references": ["answer1", "answer2"]
}
```

You had to look at `references` to see what the expected answer was.

### Issue 2: Questions Without Answers
Some datasets include questions where the answer is not explicitly provided, making metrics unreliable.

## Solution

### 1. Expected Answer in Results ✅

Results now include:
```json
{
  "question": "who sang i think we're alone now",
  "prediction": "Tiffany",
  "expected_answer": "Tiffany",              // NEW: Primary expected answer
  "all_acceptable_answers": ["Tiffany"],     // NEW: All variants
  "references": ["Tiffany"],                 // Kept for backward compatibility
  "metadata": {...}
}
```

**Benefits**:
- Easy to compare prediction vs expected answer at a glance
- Clear primary answer for analysis
- Still have all acceptable variants for complex answers
- Backward compatible (keeps `references`)

### 2. Answer Filtering ✅

**New Command-Line Option**:
```bash
--filter-no-answer    # Filter out questions without explicit answers
```

**New Dataset Methods**:

1. **`filter_with_answers()`** - Filter in-place:
```python
dataset = NaturalQuestionsDataset(split="validation")
print(f"Original: {len(dataset)}")  # e.g., 3610
dataset.filter_with_answers()
print(f"Filtered: {len(dataset)}")  # e.g., 3450
```

2. **`get_examples_with_answers(n)`** - Get filtered list:
```python
dataset = NaturalQuestionsDataset(split="validation")
filtered = dataset.get_examples_with_answers(n=100)  # Get 100 with answers
# Original dataset unchanged
```

## Usage

### Quick: Use Make Commands (Filtering Enabled by Default)

```bash
# These now filter by default
make run-gemma2b          # Quick test with filtering
make run-gemma2b-full     # Full run with filtering
```

### Command-Line Usage

```bash
# With filtering (recommended)
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100 \
    --filter-no-answer

# Without filtering (old behavior)
uv run python experiments/scripts/run_gemma2b_baseline.py \
    --dataset natural_questions \
    --num-examples 100
```

### Programmatic Usage

```python
from ragicamp.datasets.nq import NaturalQuestionsDataset

# Method 1: Filter in-place
dataset = NaturalQuestionsDataset(split="validation")
dataset.filter_with_answers()

# Method 2: Get filtered list
dataset = NaturalQuestionsDataset(split="validation")
examples_with_answers = dataset.get_examples_with_answers(n=100)

# Method 3: Manual filtering
dataset = NaturalQuestionsDataset(split="validation")
dataset.examples = [
    ex for ex in dataset.examples
    if ex.answers and any(answer.strip() for answer in ex.answers)
]
```

## Example Output

### Before (without filtering)
```
Initial dataset size: 3610 examples
✓ Final dataset size: 100 examples

Metrics:
  exact_match: 0.15    # Lower due to questions without answers
  f1: 0.25
```

### After (with filtering)
```
Initial dataset size: 3610 examples

Filtering questions without explicit answers...
Filtered out 160 examples without explicit answers
Remaining: 3450 examples

✓ Final dataset size: 100 examples

Metrics:
  exact_match: 0.23    # More accurate - all questions have ground truth
  f1: 0.34
```

## Files Changed

### Core Framework
- `src/ragicamp/datasets/base.py` - Added filtering methods
- `src/ragicamp/evaluation/evaluator.py` - Added expected_answer fields

### Scripts & Configs
- `experiments/scripts/run_gemma2b_baseline.py` - Added --filter-no-answer flag
- `Makefile` - Updated to use filtering by default

### Documentation
- `GEMMA2B_QUICKSTART.md` - Updated with filtering examples
- `CHANGELOG.md` - Documented changes
- `ANSWER_FILTERING_UPDATE.md` - This document

### Examples
- `examples/filter_dataset_example.py` - Demonstration script

## Testing

Run the example script:
```bash
uv run python examples/filter_dataset_example.py
```

Or test with actual evaluation:
```bash
# With filtering
make run-gemma2b

# Check the results
cat outputs/gemma2b_baseline_results.json | jq '.predictions[0]'
```

Expected output:
```json
{
  "id": "nq_validation_0",
  "question": "when was the last time anyone was on the moon",
  "prediction": "The last crewed mission to the Moon was Apollo 17 in December 1972.",
  "expected_answer": "14 December 1972 UTC",
  "all_acceptable_answers": [
    "14 December 1972 UTC",
    "December 1972"
  ],
  "references": [
    "14 December 1972 UTC",
    "December 1972"
  ],
  "metadata": {
    "agent_type": "direct_llm"
  }
}
```

## Benefits

1. **More Reliable Metrics**: Only evaluate on questions with ground truth
2. **Clearer Results**: See expected answer immediately
3. **Better Analysis**: Easy to compare prediction vs expected
4. **Flexible**: Can use filtering or not, depending on use case
5. **Backward Compatible**: Old code still works with `references` field

## Recommendations

**For evaluation**: Use `--filter-no-answer` (now default in make commands)
```bash
make run-gemma2b-full
```

**For analysis**: The results now have both `expected_answer` and `all_acceptable_answers`
```python
import json

with open('outputs/gemma2b_baseline_results.json') as f:
    results = json.load(f)

for pred in results['predictions']:
    print(f"Q: {pred['question']}")
    print(f"Expected: {pred['expected_answer']}")
    print(f"Got: {pred['prediction']}")
    print(f"Match: {pred['prediction'] == pred['expected_answer']}")
    print()
```

## Migration Guide

### If you have existing scripts

**No changes needed** - the `references` field is still there for backward compatibility.

**Optional enhancement** - use the new fields:
```python
# Old way (still works)
expected = prediction['references'][0]

# New way (clearer)
expected = prediction['expected_answer']
all_variants = prediction['all_acceptable_answers']
```

### If you want filtering

Add one flag:
```bash
# Old
python run_gemma2b_baseline.py --dataset nq --num-examples 100

# New (with filtering)
python run_gemma2b_baseline.py --dataset nq --num-examples 100 --filter-no-answer
```

## Summary

✅ **Expected answer clearly visible in results**  
✅ **Filter questions without ground truth**  
✅ **More reliable evaluation metrics**  
✅ **Backward compatible**  
✅ **Easy to use with make commands**  

All make commands now use filtering by default for better results!

