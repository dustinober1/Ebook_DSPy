# Best Practices

## Prerequisites

- **Chapter 1-3**: DSPy Fundamentals, Signatures, and Modules
- **Previous Sections**: Why Evaluation Matters through Evaluation Loops
- **Required Knowledge**: Understanding of previous evaluation concepts
- **Difficulty Level**: Intermediate-Advanced
- **Estimated Reading Time**: 25 minutes

## Learning Objectives

By the end of this section, you will understand:
- Best practices for dataset curation and management
- Metric design principles that lead to better optimization
- How to avoid data leakage and other common pitfalls
- Techniques for reproducible evaluation
- Continuous evaluation strategies for production systems

## Dataset Best Practices

### 1. Ensure Representative Data

Your evaluation data must reflect real-world usage:

```python
import dspy

# BAD: Biased dataset
biased_dataset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="What is 3+3?", answer="6").with_inputs("question"),
    dspy.Example(question="What is 4+4?", answer="8").with_inputs("question"),
    # All simple arithmetic - not representative!
]

# GOOD: Diverse dataset covering real use cases
representative_dataset = [
    # Simple questions
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    # Complex reasoning
    dspy.Example(question="If a train leaves at 3pm traveling 60mph, how far does it travel in 2 hours?",
                 answer="120 miles").with_inputs("question"),
    # Ambiguous questions
    dspy.Example(question="What's the best programming language?",
                 answer="It depends on the use case").with_inputs("question"),
    # Edge cases
    dspy.Example(question="What is infinity minus infinity?",
                 answer="undefined").with_inputs("question"),
]
```

### 2. Include Edge Cases

Systematically test boundary conditions:

```python
def create_comprehensive_dataset(base_examples):
    """Add edge cases to base dataset."""
    dataset = list(base_examples)

    # Add edge cases
    edge_cases = [
        # Empty input
        dspy.Example(question="", answer="Please provide a question").with_inputs("question"),

        # Very long input
        dspy.Example(question="What is " + " ".join(["the"] * 100) + " answer?",
                     answer="Please clarify your question").with_inputs("question"),

        # Special characters
        dspy.Example(question="What's the meaning of @#$%?",
                     answer="Those are special characters").with_inputs("question"),

        # Multiple languages (if relevant)
        dspy.Example(question="Qu'est-ce que c'est?",
                     answer="This means 'What is it?' in French").with_inputs("question"),

        # Numbers and symbols
        dspy.Example(question="Calculate 1,234.56 + 7,890.12",
                     answer="9124.68").with_inputs("question"),
    ]

    dataset.extend(edge_cases)
    return dataset
```

### 3. Balance Your Dataset

Ensure fair representation across categories:

```python
from collections import Counter

def analyze_dataset_balance(dataset, category_field='category'):
    """Check dataset balance across categories."""
    categories = [getattr(ex, category_field, 'unknown') for ex in dataset]
    counts = Counter(categories)

    print("Dataset Balance Analysis")
    print("=" * 40)
    total = len(dataset)
    for category, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        bar = "#" * int(pct / 2)
        print(f"{category:20} {count:5} ({pct:5.1f}%) {bar}")

    # Warn about imbalance
    max_count = max(counts.values())
    min_count = min(counts.values())
    if max_count > 5 * min_count:
        print("\nWARNING: Dataset is significantly imbalanced!")

    return counts

# Check your dataset
analyze_dataset_balance(dataset)
```

### 4. Version Your Datasets

Track dataset changes over time:

```python
import json
import hashlib
from datetime import datetime

def save_dataset_with_metadata(dataset, filepath, version_info):
    """Save dataset with versioning metadata."""
    # Create hashable representation
    data_str = json.dumps([ex.toDict() for ex in dataset], sort_keys=True)
    data_hash = hashlib.md5(data_str.encode()).hexdigest()[:8]

    metadata = {
        "version": version_info.get("version", "1.0"),
        "created": datetime.now().isoformat(),
        "hash": data_hash,
        "size": len(dataset),
        "description": version_info.get("description", ""),
        "changes": version_info.get("changes", []),
    }

    output = {
        "metadata": metadata,
        "data": [ex.toDict() for ex in dataset]
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved dataset v{metadata['version']} ({data_hash}) with {len(dataset)} examples")
    return metadata

# Usage
save_dataset_with_metadata(
    dataset,
    "data/qa_dataset_v2.json",
    {
        "version": "2.0",
        "description": "Added edge cases and multi-hop questions",
        "changes": ["Added 50 edge cases", "Added 100 multi-hop questions"]
    }
)
```

## Metric Design Best Practices

### 1. Measure What Actually Matters

```python
# BAD: Metric measures proxy, not actual goal
def bad_metric(example, pred, trace=None):
    # Length doesn't indicate quality!
    return len(pred.answer) > 50

# GOOD: Metric measures actual goal
def good_metric(example, pred, trace=None):
    # Check factual correctness
    correct = example.answer.lower() in pred.answer.lower()
    # Check completeness
    complete = all(
        key_point.lower() in pred.answer.lower()
        for key_point in example.key_points
    )
    return correct and complete
```

### 2. Make Metrics Robust

Handle variations and edge cases:

```python
def robust_metric(example, pred, trace=None):
    """Robust metric with proper handling."""
    # Handle missing attributes
    expected = getattr(example, 'answer', None)
    predicted = getattr(pred, 'answer', None)

    if expected is None or predicted is None:
        return 0.0

    # Normalize for comparison
    def normalize(text):
        if not isinstance(text, str):
            text = str(text)
        # Lowercase, strip whitespace, remove punctuation
        text = text.lower().strip()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        return ' '.join(text.split())  # Normalize whitespace

    expected_norm = normalize(expected)
    predicted_norm = normalize(predicted)

    # Flexible matching
    if expected_norm == predicted_norm:
        return 1.0
    elif expected_norm in predicted_norm:
        return 0.8
    elif predicted_norm in expected_norm:
        return 0.6
    else:
        return 0.0
```

### 3. Use Appropriate Granularity

```python
# TOO COARSE: Only binary
def coarse_metric(example, pred, trace=None):
    return pred.answer == example.answer  # Only 0 or 1

# TOO FINE: Over-engineered
def fine_metric(example, pred, trace=None):
    score = 0
    score += 0.1 if pred.answer else 0
    score += 0.1 if len(pred.answer) > 10 else 0
    score += 0.1 if len(pred.answer) > 50 else 0
    # ... 20 more tiny adjustments
    return score  # Hard to interpret

# JUST RIGHT: Meaningful granularity
def balanced_metric(example, pred, trace=None):
    # Core correctness (most weight)
    correct = example.answer.lower() in pred.answer.lower()

    # Quality bonus
    well_formed = pred.answer.strip().endswith('.')
    appropriate_length = 20 <= len(pred.answer) <= 200

    if correct:
        base = 0.8
        bonus = 0.1 * well_formed + 0.1 * appropriate_length
        return base + bonus
    else:
        return 0.0
```

### 4. Test Your Metrics

Validate metric behavior before use:

```python
def test_metric(metric, test_cases):
    """Test metric on known cases."""
    print("Metric Test Results")
    print("=" * 60)

    all_passed = True
    for i, case in enumerate(test_cases):
        example = case['example']
        pred = case['pred']
        expected = case['expected_score']

        actual = metric(example, pred)

        # Allow small floating point differences
        passed = abs(actual - expected) < 0.01
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"Test {i+1}: {status}")
        print(f"  Input: {example.question[:50]}...")
        print(f"  Expected score: {expected}, Actual: {actual}")

    print("=" * 60)
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed

# Define test cases
test_cases = [
    {
        'example': dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
        'pred': type('Pred', (), {'answer': "4"})(),
        'expected_score': 1.0
    },
    {
        'example': dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
        'pred': type('Pred', (), {'answer': "The answer is 4."})(),
        'expected_score': 0.8  # Partial match
    },
    {
        'example': dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
        'pred': type('Pred', (), {'answer': "5"})(),
        'expected_score': 0.0
    },
]

test_metric(robust_metric, test_cases)
```

## Avoiding Data Leakage

### What is Data Leakage?

Data leakage occurs when information from the test set influences training:

```python
# DANGEROUS: Data leakage example
all_data = load_all_examples()

# WRONG: Test data overlaps with training
trainset = all_data[:800]
testset = all_data[:200]  # BUG! Overlaps with trainset

# Module sees test examples during training
# Test score will be artificially high
```

### Prevention Strategies

```python
import random
from typing import Tuple, List

def safe_split(data: List, train_ratio=0.6, dev_ratio=0.2, seed=42) -> Tuple[List, List, List]:
    """Safely split data without leakage."""
    # Make a copy to avoid modifying original
    data = list(data)

    # Shuffle with fixed seed
    random.Random(seed).shuffle(data)

    # Calculate split points
    n = len(data)
    train_end = int(n * train_ratio)
    dev_end = int(n * (train_ratio + dev_ratio))

    # Split
    trainset = data[:train_end]
    devset = data[train_end:dev_end]
    testset = data[dev_end:]

    # Verify no overlap
    train_ids = {id(ex) for ex in trainset}
    dev_ids = {id(ex) for ex in devset}
    test_ids = {id(ex) for ex in testset}

    assert len(train_ids & dev_ids) == 0, "Train/dev overlap!"
    assert len(train_ids & test_ids) == 0, "Train/test overlap!"
    assert len(dev_ids & test_ids) == 0, "Dev/test overlap!"

    print(f"Safe split: Train={len(trainset)}, Dev={len(devset)}, Test={len(testset)}")

    return trainset, devset, testset
```

### Content-Based Deduplication

Prevent near-duplicate leakage:

```python
def content_aware_split(data, key_field='question', similarity_threshold=0.9):
    """Split data ensuring no similar content across splits."""
    from difflib import SequenceMatcher

    def similar(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    # Build groups of similar items
    groups = []
    assigned = set()

    for i, ex1 in enumerate(data):
        if i in assigned:
            continue

        group = [i]
        key1 = getattr(ex1, key_field, '')

        for j, ex2 in enumerate(data[i+1:], i+1):
            if j in assigned:
                continue
            key2 = getattr(ex2, key_field, '')

            if similar(key1, key2) >= similarity_threshold:
                group.append(j)
                assigned.add(j)

        groups.append(group)
        assigned.add(i)

    # Shuffle groups (not individual items)
    random.shuffle(groups)

    # Assign groups to splits
    train_groups = groups[:int(len(groups) * 0.6)]
    dev_groups = groups[int(len(groups) * 0.6):int(len(groups) * 0.8)]
    test_groups = groups[int(len(groups) * 0.8):]

    trainset = [data[i] for g in train_groups for i in g]
    devset = [data[i] for g in dev_groups for i in g]
    testset = [data[i] for g in test_groups for i in g]

    return trainset, devset, testset
```

## Reproducibility

### 1. Fix Random Seeds

```python
import random
import numpy as np

def set_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    # If using PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# Always set seeds at the start
set_seeds(42)
```

### 2. Log All Configuration

```python
import json
import dspy

def log_evaluation_config(module, devset, metric, filepath="eval_config.json"):
    """Log complete evaluation configuration."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "module": {
            "type": type(module).__name__,
            "signature": str(getattr(module, 'signature', 'unknown')),
        },
        "dataset": {
            "size": len(devset),
            "fields": list(devset[0].toDict().keys()) if devset else [],
        },
        "metric": {
            "name": metric.__name__,
            "doc": metric.__doc__,
        },
        "environment": {
            "dspy_version": dspy.__version__,
            "lm": str(dspy.settings.lm) if dspy.settings.lm else "not configured",
        }
    }

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

    return config
```

### 3. Version Control Everything

```python
# Include in your .gitignore
# - build/
# - __pycache__/
# - .env

# Track in version control:
# - data/datasets/*.json (versioned datasets)
# - configs/*.json (evaluation configs)
# - results/*.json (evaluation results)
```

## Continuous Evaluation

### Scheduled Evaluation

```python
import schedule
import time

def daily_evaluation():
    """Run daily evaluation on production module."""
    # Load latest module
    module = load_production_module()

    # Sample recent data
    recent_data = sample_production_logs(n=100)
    devset = convert_to_examples(recent_data)

    # Run evaluation
    evaluate = dspy.Evaluate(devset=devset, metric=metric)
    score = evaluate(module)

    # Log results
    log_to_monitoring(score)

    # Alert if degradation
    if score < THRESHOLD:
        send_alert(f"Performance degradation: {score}%")

# Schedule daily at 3 AM
schedule.every().day.at("03:00").do(daily_evaluation)

# Keep running
while True:
    schedule.run_pending()
    time.sleep(60)
```

### Regression Testing

```python
def regression_test(new_module, baseline_module, devset, metric, tolerance=2.0):
    """Ensure new module doesn't regress vs baseline."""
    evaluate = dspy.Evaluate(devset=devset, metric=metric)

    baseline_score = evaluate(baseline_module)
    new_score = evaluate(new_module)

    regression = baseline_score - new_score

    print(f"Baseline: {baseline_score:.1f}%")
    print(f"New: {new_score:.1f}%")
    print(f"Change: {new_score - baseline_score:+.1f}%")

    if regression > tolerance:
        raise ValueError(
            f"Regression detected! New module is {regression:.1f}% worse than baseline"
        )

    return new_score, baseline_score
```

## Summary Checklist

Use this checklist for every evaluation:

### Dataset Checklist
- [ ] Data is representative of real usage
- [ ] Edge cases are included
- [ ] Data is balanced across categories
- [ ] No duplicates or near-duplicates
- [ ] Train/dev/test splits are clean (no leakage)
- [ ] Dataset is versioned and documented

### Metric Checklist
- [ ] Metric measures actual goal
- [ ] Metric handles edge cases gracefully
- [ ] Metric is tested on known cases
- [ ] Metric behavior is documented
- [ ] trace parameter is used correctly

### Evaluation Checklist
- [ ] Random seeds are fixed
- [ ] Configuration is logged
- [ ] Results are reproducible
- [ ] Error analysis is performed
- [ ] Comparison to baseline is done

### Production Checklist
- [ ] Continuous evaluation is set up
- [ ] Regression tests are in place
- [ ] Alerts are configured for degradation
- [ ] Results are tracked over time

## Key Takeaways

1. **Representative data** is the foundation of meaningful evaluation
2. **Robust metrics** handle edge cases and variations
3. **Data leakage** invalidates all your results - prevent it!
4. **Reproducibility** requires fixing seeds and logging config
5. **Continuous evaluation** catches production issues early

## Next Steps

- [Exercises](./06-exercises.md) - Practice evaluation skills
- [Examples](../../examples/chapter04/) - See best practices in action
- [Chapter 5: Optimizers](../05-optimizers/00-chapter-intro.md) - Use evaluation for optimization

## Further Reading

- [ML Testing Best Practices](https://developers.google.com/machine-learning/testing-debugging)
- [Data Leakage in ML](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [Reproducible ML Research](https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf)
