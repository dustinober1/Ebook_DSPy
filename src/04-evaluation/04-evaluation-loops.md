# Evaluation Loops

## Prerequisites

- **Chapter 1-3**: DSPy Fundamentals, Signatures, and Modules
- **Previous Sections**: Creating Datasets, Defining Metrics
- **Required Knowledge**: Basic Python iteration concepts
- **Difficulty Level**: Intermediate
- **Estimated Reading Time**: 30 minutes

## Learning Objectives

By the end of this section, you will be able to:
- Use the DSPy Evaluate class for systematic evaluation
- Run parallel evaluations for better performance
- Track and analyze evaluation results
- Integrate evaluations with MLflow for experiment tracking
- Build evaluation workflows into your development process

## The Evaluate Class

DSPy's `Evaluate` class provides a powerful, systematic way to assess module performance.

### Basic Usage

```python
import dspy

# Setup: module and data
module = dspy.Predict("question -> answer")
devset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="What is 3*3?", answer="9").with_inputs("question"),
    # ... more examples
]

# Define metric
def accuracy(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# Create evaluator
evaluate = dspy.Evaluate(
    devset=devset,
    metric=accuracy
)

# Run evaluation
score = evaluate(module)
print(f"Accuracy: {score}%")
```

### Evaluate Parameters

```python
evaluate = dspy.Evaluate(
    devset=devset,           # Dataset to evaluate on
    metric=metric,           # Metric function
    num_threads=8,           # Parallel threads (default: 1)
    display_progress=True,   # Show progress bar
    display_table=5,         # Show N example results
    return_all_scores=False, # Return individual scores
    return_outputs=False,    # Return predictions
    provide_traceback=False, # Show errors
)
```

### Understanding the Output

```python
# Basic usage - returns aggregate score
score = evaluate(module)
print(f"Score: {score}%")  # e.g., "Score: 87.5%"

# With return_all_scores - returns Result object
result = dspy.Evaluate(
    devset=devset,
    metric=metric,
    return_all_scores=True
)(module)

print(f"Aggregate: {result.score}%")
print(f"Individual scores: {result.scores}")  # List of per-example scores

# With return_outputs - includes predictions
result = dspy.Evaluate(
    devset=devset,
    metric=metric,
    return_outputs=True
)(module)

# Access detailed results
for example, prediction, score in result.results:
    print(f"Q: {example.question}")
    print(f"Expected: {example.answer}")
    print(f"Got: {prediction.answer}")
    print(f"Score: {score}")
    print("---")
```

## Parallel Evaluation

Speed up evaluation with multi-threading:

### Setting Thread Count

```python
# Single-threaded (slow but deterministic)
evaluate_slow = dspy.Evaluate(
    devset=devset,
    metric=metric,
    num_threads=1
)

# Multi-threaded (faster)
evaluate_fast = dspy.Evaluate(
    devset=devset,
    metric=metric,
    num_threads=16  # Adjust based on API rate limits
)

# Compare times
import time

start = time.time()
score_slow = evaluate_slow(module)
slow_time = time.time() - start

start = time.time()
score_fast = evaluate_fast(module)
fast_time = time.time() - start

print(f"Single-threaded: {slow_time:.2f}s")
print(f"Multi-threaded: {fast_time:.2f}s")
print(f"Speedup: {slow_time/fast_time:.1f}x")
```

### Thread Count Guidelines

| API Provider | Recommended Threads | Notes |
|-------------|---------------------|-------|
| OpenAI Free Tier | 2-4 | Conservative rate limits |
| OpenAI Paid | 8-16 | Higher limits |
| Anthropic | 4-8 | Check your tier |
| Local LLM | CPU cores | Limited by hardware |
| Azure OpenAI | 8-20 | Depends on deployment |

```python
# Detect optimal thread count
import os

# Conservative default
num_threads = min(8, os.cpu_count() or 4)

evaluate = dspy.Evaluate(
    devset=devset,
    metric=metric,
    num_threads=num_threads
)
```

## Progress Tracking

Monitor evaluation progress in real-time:

### Progress Bar

```python
# Enable progress bar
evaluate = dspy.Evaluate(
    devset=devset,
    metric=metric,
    num_threads=8,
    display_progress=True  # Shows progress bar
)

score = evaluate(module)
# Output: Progress bar with ETA and current score
```

### Display Table

```python
# Show example results table
evaluate = dspy.Evaluate(
    devset=devset,
    metric=metric,
    display_progress=True,
    display_table=5  # Show first 5 results
)

score = evaluate(module)
# Output: Table showing questions, expected answers, predictions, scores
```

## Manual Evaluation Loops

For more control, write manual evaluation loops:

### Basic Loop

```python
import dspy

def manual_evaluate(module, devset, metric):
    """Simple manual evaluation loop."""
    scores = []

    for example in devset:
        # Get prediction
        pred = module(**example.inputs())

        # Calculate score
        score = metric(example, pred)
        scores.append(score)

    # Aggregate
    avg_score = sum(scores) / len(scores) if scores else 0
    return avg_score * 100  # Return as percentage

# Usage
score = manual_evaluate(qa_module, devset, accuracy_metric)
print(f"Accuracy: {score:.1f}%")
```

### Loop with Detailed Tracking

```python
import dspy
from collections import defaultdict

def detailed_evaluate(module, devset, metric):
    """Evaluation with detailed tracking."""
    results = {
        'scores': [],
        'predictions': [],
        'errors': [],
        'by_category': defaultdict(list)
    }

    for i, example in enumerate(devset):
        try:
            # Get prediction
            pred = module(**example.inputs())

            # Calculate score
            score = metric(example, pred)

            # Store results
            results['scores'].append(score)
            results['predictions'].append({
                'example': example.toDict(),
                'prediction': pred.toDict() if hasattr(pred, 'toDict') else str(pred),
                'score': score
            })

            # Track by category if available
            if hasattr(example, 'category'):
                results['by_category'][example.category].append(score)

        except Exception as e:
            results['errors'].append({
                'index': i,
                'example': example.toDict(),
                'error': str(e)
            })
            results['scores'].append(0)

    # Calculate statistics
    results['stats'] = {
        'total': len(devset),
        'errors': len(results['errors']),
        'avg_score': sum(results['scores']) / len(results['scores']) if results['scores'] else 0,
        'min_score': min(results['scores']) if results['scores'] else 0,
        'max_score': max(results['scores']) if results['scores'] else 0,
    }

    # Category breakdown
    for category, scores in results['by_category'].items():
        results['stats'][f'avg_{category}'] = sum(scores) / len(scores)

    return results

# Usage
results = detailed_evaluate(qa_module, devset, metric)
print(f"Overall accuracy: {results['stats']['avg_score']*100:.1f}%")
print(f"Errors: {results['stats']['errors']}")
```

### Async Evaluation Loop

For I/O-bound operations:

```python
import asyncio
import dspy

async def async_evaluate(module, devset, metric, max_concurrent=10):
    """Async evaluation for I/O-bound modules."""
    semaphore = asyncio.Semaphore(max_concurrent)
    scores = []

    async def evaluate_one(example):
        async with semaphore:
            # Note: Requires async-compatible module
            pred = await module.aforward(**example.inputs())
            return metric(example, pred)

    tasks = [evaluate_one(ex) for ex in devset]
    scores = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    valid_scores = [s for s in scores if isinstance(s, (int, float, bool))]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0

    return avg * 100

# Usage (in async context)
# score = await async_evaluate(module, devset, metric)
```

## MLflow Integration

Track experiments with MLflow:

### Basic MLflow Logging

```python
import dspy
import mlflow

# Configure MLflow
mlflow.set_experiment("dspy-qa-evaluation")

# Run evaluation with logging
with mlflow.start_run(run_name="qa_module_v1"):
    # Log parameters
    mlflow.log_param("module_type", "Predict")
    mlflow.log_param("model", "gpt-4")
    mlflow.log_param("dataset_size", len(devset))

    # Run evaluation
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=8,
        display_progress=True
    )
    score = evaluate(qa_module)

    # Log metrics
    mlflow.log_metric("accuracy", score)

    print(f"Run logged with accuracy: {score}%")
```

### Comprehensive MLflow Tracking

```python
import dspy
import mlflow
import json

def evaluate_with_mlflow(module, devset, metric, run_name, tags=None):
    """Full evaluation with MLflow tracking."""

    with mlflow.start_run(run_name=run_name):
        # Log tags
        if tags:
            mlflow.set_tags(tags)

        # Log dataset info
        mlflow.log_param("dataset_size", len(devset))

        # Run evaluation
        evaluate = dspy.Evaluate(
            devset=devset,
            metric=metric,
            num_threads=16,
            display_progress=True,
            return_outputs=True
        )
        result = evaluate(module)

        # Log aggregate metrics
        mlflow.log_metric("accuracy", result.score)

        # Log detailed results
        detailed_results = []
        for example, pred, score in result.results:
            detailed_results.append({
                "input": example.inputs(),
                "expected": example.toDict(),
                "predicted": pred.toDict() if hasattr(pred, 'toDict') else str(pred),
                "score": score
            })

        mlflow.log_table(
            data={
                "Question": [r["input"].get("question", "") for r in detailed_results],
                "Expected": [r["expected"].get("answer", "") for r in detailed_results],
                "Predicted": [r["predicted"].get("answer", "") if isinstance(r["predicted"], dict) else r["predicted"] for r in detailed_results],
                "Score": [r["score"] for r in detailed_results],
            },
            artifact_file="evaluation_results.json"
        )

        # Log error analysis
        failures = [r for r in detailed_results if not r["score"]]
        if failures:
            mlflow.log_metric("failure_count", len(failures))

        return result.score

# Usage
score = evaluate_with_mlflow(
    module=qa_module,
    devset=devset,
    metric=metric,
    run_name="qa_v1_gpt4",
    tags={"version": "1.0", "model": "gpt-4"}
)
```

## Evaluation Workflows

### Development Workflow

```python
import dspy

def development_evaluation(module, devset, metric):
    """Quick evaluation during development."""
    # Use small subset for speed
    mini_devset = devset[:20]

    evaluate = dspy.Evaluate(
        devset=mini_devset,
        metric=metric,
        num_threads=4,
        display_progress=True,
        display_table=5  # See examples
    )

    score = evaluate(module)
    print(f"\n[Dev] Quick check: {score:.1f}%")
    return score

# Fast iteration loop
for iteration in range(5):
    # Make changes to module...
    score = development_evaluation(module, devset, metric)
    if score > 90:
        print("Target reached!")
        break
```

### Pre-Commit Evaluation

```python
import dspy

def pre_commit_evaluation(module, devset, metric, threshold=80):
    """Run before committing changes."""
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=8,
        display_progress=True
    )

    score = evaluate(module)

    if score < threshold:
        raise ValueError(
            f"Evaluation score {score:.1f}% below threshold {threshold}%"
        )

    print(f"[Pre-commit] PASSED with {score:.1f}%")
    return score

# Use in CI/CD or pre-commit hook
pre_commit_evaluation(module, devset, metric, threshold=85)
```

### A/B Testing Workflow

```python
import dspy

def compare_modules(module_a, module_b, devset, metric, names=("A", "B")):
    """Compare two module versions."""
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=8,
        display_progress=True
    )

    print(f"Evaluating {names[0]}...")
    score_a = evaluate(module_a)

    print(f"\nEvaluating {names[1]}...")
    score_b = evaluate(module_b)

    # Report
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    print(f"{names[0]}: {score_a:.1f}%")
    print(f"{names[1]}: {score_b:.1f}%")
    print(f"Difference: {score_b - score_a:+.1f}%")

    if score_b > score_a:
        print(f"\n{names[1]} is better by {score_b - score_a:.1f} points")
    elif score_a > score_b:
        print(f"\n{names[0]} is better by {score_a - score_b:.1f} points")
    else:
        print("\nBoth modules perform equally")

    return score_a, score_b

# Compare baseline vs optimized
baseline = dspy.Predict("question -> answer")
optimized = optimizer.compile(baseline, trainset=trainset)

compare_modules(baseline, optimized, testset, metric, ("Baseline", "Optimized"))
```

## Error Analysis

Understanding failures is as important as measuring success:

### Categorizing Errors

```python
import dspy
from collections import defaultdict

def error_analysis(module, devset, metric):
    """Analyze evaluation errors."""
    errors = defaultdict(list)
    successes = []

    evaluate = dspy.Evaluate(
        devset=devset,
        metric=metric,
        return_outputs=True
    )
    result = evaluate(module)

    for example, pred, score in result.results:
        if not score:  # Failed
            # Categorize the error
            if len(pred.answer) == 0:
                errors['empty_response'].append((example, pred))
            elif len(pred.answer) < 10:
                errors['too_short'].append((example, pred))
            elif example.answer.lower() not in pred.answer.lower():
                errors['wrong_answer'].append((example, pred))
            else:
                errors['other'].append((example, pred))
        else:
            successes.append((example, pred))

    # Report
    print("ERROR ANALYSIS")
    print("="*50)
    print(f"Total: {len(devset)}")
    print(f"Success: {len(successes)} ({100*len(successes)/len(devset):.1f}%)")
    print(f"Failures: {len(devset) - len(successes)}")
    print("\nError breakdown:")
    for error_type, examples in errors.items():
        print(f"  {error_type}: {len(examples)} ({100*len(examples)/len(devset):.1f}%)")

    return errors

# Run analysis
errors = error_analysis(qa_module, devset, metric)

# Examine specific error types
print("\nExamples of wrong answers:")
for example, pred in errors['wrong_answer'][:3]:
    print(f"  Q: {example.question}")
    print(f"  Expected: {example.answer}")
    print(f"  Got: {pred.answer}")
    print()
```

## Summary

Evaluation loops are your systematic approach to measuring quality:

1. **Use dspy.Evaluate** for standard evaluation needs
2. **Enable parallel execution** for faster evaluation
3. **Track results with MLflow** for experiment management
4. **Build evaluation into workflows** (development, pre-commit, A/B testing)
5. **Analyze errors** to understand failure patterns

### Key Takeaways

1. **dspy.Evaluate** provides comprehensive evaluation capabilities
2. **Parallel execution** speeds up evaluation significantly
3. **Progress tracking** keeps you informed during long evaluations
4. **MLflow integration** enables experiment tracking
5. **Error analysis** reveals improvement opportunities

## Next Steps

- [Next Section: Best Practices](./05-best-practices.md) - Evaluation best practices
- [Exercises](./06-exercises.md) - Practice evaluation skills
- [Examples](../../examples/chapter04/) - See evaluation code

## Further Reading

- [DSPy Evaluate Documentation](https://dspy.ai/api/evaluation/evaluate)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [A/B Testing Best Practices](https://www.optimizely.com/optimization-glossary/ab-testing/)
