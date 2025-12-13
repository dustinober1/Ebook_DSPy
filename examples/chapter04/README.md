# Chapter 4: Evaluation Examples

This directory contains working code examples for Chapter 4: Evaluation.

## Prerequisites

Before running these examples, ensure you have:

1. **Python 3.9+** installed
2. **DSPy** installed: `pip install dspy-ai`
3. **API key** configured (OpenAI, Anthropic, etc.)

### Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install dspy-ai python-dotenv

# Set up your API key
echo "OPENAI_API_KEY=your-key-here" > .env
```

## Examples

### 01_basic_evaluation.py

**Basic Evaluation Workflows**

Demonstrates fundamental evaluation concepts:
- Creating simple evaluation datasets
- Defining basic metrics (exact match, contains, flexible)
- Running evaluations with `dspy.Evaluate`
- Manual evaluation loops

```bash
python 01_basic_evaluation.py
```

Key concepts:
- `dspy.Example` class and `with_inputs()`
- `dspy.Evaluate` basic usage
- Boolean vs numeric metrics

---

### 02_custom_metrics.py

**Designing Custom Metrics**

Shows how to create various types of metrics:
- Boolean metrics (exact match, normalized match)
- Numeric metrics (word overlap, Jaccard similarity)
- Composite metrics (multi-aspect evaluation)
- Domain-specific metrics (sentiment, entity extraction)
- Trace-aware metrics for optimization

```bash
python 02_custom_metrics.py
```

Key concepts:
- Metric function anatomy (example, pred, trace)
- Combining multiple quality dimensions
- Using trace parameter for optimization

---

### 03_dataset_creation.py

**Creating and Managing Datasets**

Demonstrates dataset creation and management:
- Manual dataset creation
- Loading from JSON/CSV files
- Train/dev/test splitting
- Stratified splitting for balanced classes
- Dataset validation and quality checks
- Versioning and saving datasets

```bash
python 03_dataset_creation.py
```

Key concepts:
- `dspy.Example` creation patterns
- Data splitting strategies
- Deduplication and validation
- Dataset versioning

---

### 04_evaluation_loops.py

**Advanced Evaluation Workflows**

Shows comprehensive evaluation patterns:
- Basic and detailed evaluation
- Parallel vs sequential performance
- Manual evaluation loops
- Error analysis and categorization
- A/B testing between modules
- Comprehensive evaluation reports

```bash
python 04_evaluation_loops.py
```

Key concepts:
- `return_outputs=True` for detailed results
- Category-wise performance analysis
- Error categorization
- A/B testing workflow

---

## Quick Reference

### Creating Examples

```python
import dspy

# Basic example
example = dspy.Example(
    question="What is 2+2?",
    answer="4"
).with_inputs("question")

# Multiple inputs
example = dspy.Example(
    context="...",
    question="...",
    answer="..."
).with_inputs("context", "question")
```

### Defining Metrics

```python
def my_metric(example, pred, trace=None):
    """
    Args:
        example: dspy.Example with expected outputs
        pred: Prediction from module
        trace: None (evaluation) or object (optimization)

    Returns:
        bool or float (0.0 to 1.0)
    """
    # Your scoring logic
    score = example.answer in pred.answer

    if trace is not None:
        # During optimization, return boolean
        return score >= 0.8

    return score
```

### Running Evaluation

```python
import dspy

evaluate = dspy.Evaluate(
    devset=devset,
    metric=my_metric,
    num_threads=8,
    display_progress=True,
    return_outputs=True  # For detailed results
)

result = evaluate(module)
print(f"Score: {result.score}%")
```

## Common Patterns

### Flexible Match Metric

```python
def flexible_match(example, pred, trace=None):
    expected = example.answer.lower().strip()
    predicted = pred.answer.lower().strip()

    if expected == predicted:
        return 1.0
    if expected in predicted or predicted in expected:
        return 0.8
    return 0.0
```

### Dataset Splitting

```python
import random

random.Random(42).shuffle(data)  # Fixed seed
trainset = data[:int(len(data)*0.6)]
devset = data[int(len(data)*0.6):int(len(data)*0.8)]
testset = data[int(len(data)*0.8):]
```

### Error Analysis

```python
for example, pred, score in result.results:
    if score < 0.8:
        print(f"FAIL: {example.question}")
        print(f"  Expected: {example.answer}")
        print(f"  Got: {pred.answer}")
```

## Troubleshooting

### "No module named 'dspy'"
```bash
pip install dspy-ai
```

### API Key Errors
```bash
# Create .env file
echo "OPENAI_API_KEY=sk-..." > .env

# Or export directly
export OPENAI_API_KEY=sk-...
```

### Rate Limiting
Reduce `num_threads` in evaluation:
```python
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=2)
```

## Next Steps

After completing these examples:

1. **Practice**: Try the exercises in `exercises/chapter04/`
2. **Apply**: Create datasets and metrics for your own tasks
3. **Continue**: Move to Chapter 5 (Optimizers) to use evaluation for improvement

## Resources

- [DSPy Documentation](https://dspy.ai)
- [Chapter 4 Content](../../src/04-evaluation/)
- [Chapter 4 Exercises](../../exercises/chapter04/)
