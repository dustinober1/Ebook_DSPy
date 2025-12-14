# Exercises

## Prerequisites

- **All Previous Sections**: Complete understanding of Chapter 4 content
- **Working DSPy Setup**: Configured with API key
- **Python Environment**: With dspy installed
- **Difficulty Level**: Intermediate-Advanced
- **Estimated Completion Time**: 2-3 hours

## Overview

These exercises will help you solidify your understanding of DSPy evaluation. Each exercise builds on concepts from the chapter, progressing from basic to more advanced applications.

---

## Exercise 1: Creating a Quality Dataset

**Difficulty**: ⭐⭐ Intermediate

### Objective
Create a well-structured evaluation dataset for a sentiment analysis task.

### Requirements

1. Create a dataset of at least 30 examples with the following fields:
   - `text`: The review/comment text
   - `sentiment`: Expected sentiment (positive, negative, neutral)
   - `confidence`: Expected confidence level (0.0-1.0)

2. Ensure:
   - Balanced distribution across sentiment classes
   - Mix of easy and difficult examples
   - At least 5 edge cases (sarcasm, mixed sentiment, etc.)

3. Properly split into train (60%), dev (20%), test (20%)

### Starter Code

```python
import dspy
import random

def create_sentiment_dataset():
    """
    Create a sentiment analysis dataset.

    Returns:
        Tuple of (trainset, devset, testset)
    """
    # TODO: Create examples
    examples = []

    # Easy positive examples
    examples.append(
        dspy.Example(
            text="This product is amazing! Best purchase ever!",
            sentiment="positive",
            confidence=0.95
        ).with_inputs("text")
    )

    # TODO: Add more examples (at least 30 total)
    # Include:
    # - Positive examples (10+)
    # - Negative examples (10+)
    # - Neutral examples (5+)
    # - Edge cases (5+)

    # TODO: Shuffle with fixed seed

    # TODO: Split into train/dev/test

    return trainset, devset, testset


# Test your implementation
trainset, devset, testset = create_sentiment_dataset()

print(f"Train: {len(trainset)}")
print(f"Dev: {len(devset)}")
print(f"Test: {len(testset)}")

# Verify balance
from collections import Counter
train_sentiments = Counter(ex.sentiment for ex in trainset)
print(f"Train distribution: {train_sentiments}")
```

### Expected Output

```
Train: 18
Dev: 6
Test: 6
Train distribution: Counter({'positive': 6, 'negative': 6, 'neutral': 6})
```

### Hints

<details>
<summary>Hint 1: Edge Cases to Include</summary>

- Sarcastic comments: "Oh great, another broken product. Just what I needed."
- Mixed sentiment: "The food was delicious but the service was terrible."
- Questions: "Is this product worth the price?"
- Very short texts: "Meh."
- Emoji-heavy: "Love it! 😍🎉"
</details>

<details>
<summary>Hint 2: Balancing the Dataset</summary>

Create examples in a loop for each category:
```python
positive_texts = [...]  # 10 positive texts
negative_texts = [...]  # 10 negative texts
neutral_texts = [...]   # 5 neutral texts

for text in positive_texts:
    examples.append(dspy.Example(
        text=text, sentiment="positive", confidence=0.9
    ).with_inputs("text"))
```
</details>

---

## Exercise 2: Designing a Custom Metric

**Difficulty**: ⭐⭐ Intermediate

### Objective
Design a comprehensive metric for evaluating a question-answering system.

### Requirements

1. Create a metric that evaluates:
   - **Correctness** (40%): Does the answer contain the expected information?
   - **Completeness** (30%): Does the answer address all parts of the question?
   - **Conciseness** (20%): Is the answer appropriately brief?
   - **Format** (10%): Is the answer well-formatted?

2. The metric should:
   - Return a float between 0 and 1
   - Handle the `trace` parameter correctly
   - Be robust to missing fields

### Starter Code

```python
import dspy

def qa_quality_metric(example, pred, trace=None):
    """
    Comprehensive QA quality metric.

    Args:
        example: dspy.Example with 'question', 'answer', 'key_points'
        pred: Prediction with 'answer'
        trace: Optional trace for optimization

    Returns:
        float: Quality score between 0 and 1
    """
    # TODO: Implement correctness check (40% weight)
    # Check if expected answer is contained in prediction
    correctness_score = 0.0

    # TODO: Implement completeness check (30% weight)
    # Check if all key_points from example are addressed
    completeness_score = 0.0

    # TODO: Implement conciseness check (20% weight)
    # Penalize overly long or short answers
    conciseness_score = 0.0

    # TODO: Implement format check (10% weight)
    # Check for proper punctuation, no repeated words, etc.
    format_score = 0.0

    # Combine scores
    final_score = (
        0.4 * correctness_score +
        0.3 * completeness_score +
        0.2 * conciseness_score +
        0.1 * format_score
    )

    # Handle trace parameter
    if trace is not None:
        # During optimization, be stricter
        return final_score >= 0.7

    return final_score


# Test the metric
example = dspy.Example(
    question="What are the benefits of exercise?",
    answer="improves health, boosts mood, increases energy",
    key_points=["health", "mood", "energy"]
).with_inputs("question")

# Create mock predictions to test
class MockPred:
    def __init__(self, answer):
        self.answer = answer

# Good prediction
good_pred = MockPred("Exercise improves health, boosts mood, and increases energy levels.")
print(f"Good prediction score: {qa_quality_metric(example, good_pred)}")

# Partial prediction
partial_pred = MockPred("Exercise is good for health.")
print(f"Partial prediction score: {qa_quality_metric(example, partial_pred)}")

# Bad prediction
bad_pred = MockPred("I don't know.")
print(f"Bad prediction score: {qa_quality_metric(example, bad_pred)}")
```

### Expected Output

```
Good prediction score: 0.85-0.95
Partial prediction score: 0.4-0.6
Bad prediction score: 0.0-0.2
```

### Hints

<details>
<summary>Hint 1: Correctness Check</summary>

```python
expected = example.answer.lower()
predicted = pred.answer.lower()
correctness_score = 1.0 if expected in predicted else (
    0.5 if any(word in predicted for word in expected.split()) else 0.0
)
```
</details>

<details>
<summary>Hint 2: Completeness Check</summary>

```python
key_points = getattr(example, 'key_points', [])
if key_points:
    found = sum(1 for kp in key_points if kp.lower() in pred.answer.lower())
    completeness_score = found / len(key_points)
else:
    completeness_score = 1.0  # No key points to check
```
</details>

<details>
<summary>Hint 3: Conciseness Check</summary>

```python
word_count = len(pred.answer.split())
if 10 <= word_count <= 100:
    conciseness_score = 1.0
elif word_count < 5:
    conciseness_score = 0.3
elif word_count > 200:
    conciseness_score = 0.5
else:
    conciseness_score = 0.8
```
</details>

---

## Exercise 3: Running Systematic Evaluation

**Difficulty**: ⭐⭐⭐ Intermediate-Advanced

### Objective
Build a complete evaluation pipeline with detailed analysis.

### Requirements

1. Create a function that:
   - Takes a module, dataset, and metric
   - Runs evaluation with progress tracking
   - Returns detailed results including:
     - Aggregate score
     - Per-category breakdown (if available)
     - Error analysis
     - Best and worst performing examples

2. The function should handle errors gracefully

### Starter Code

```python
import dspy
from collections import defaultdict

def comprehensive_evaluation(module, devset, metric, category_field=None):
    """
    Run comprehensive evaluation with detailed analysis.

    Args:
        module: DSPy module to evaluate
        devset: Evaluation dataset
        metric: Metric function
        category_field: Optional field name for category breakdown

    Returns:
        dict: Detailed evaluation results
    """
    results = {
        'aggregate_score': 0.0,
        'total_examples': len(devset),
        'by_category': {},
        'errors': [],
        'best_examples': [],
        'worst_examples': [],
        'all_scores': []
    }

    # TODO: Iterate through dataset
    for example in devset:
        try:
            # TODO: Get prediction
            pass

            # TODO: Calculate score
            pass

            # TODO: Store results
            pass

        except Exception as e:
            # TODO: Handle errors
            pass

    # TODO: Calculate aggregate score

    # TODO: Category breakdown (if category_field provided)

    # TODO: Find best and worst examples

    # TODO: Generate summary

    return results


def print_evaluation_report(results):
    """Pretty print evaluation results."""
    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    print(f"\nAggregate Score: {results['aggregate_score']*100:.1f}%")
    print(f"Total Examples: {results['total_examples']}")
    print(f"Errors: {len(results['errors'])}")

    if results['by_category']:
        print("\nBy Category:")
        for cat, scores in results['by_category'].items():
            avg = sum(scores) / len(scores) if scores else 0
            print(f"  {cat}: {avg*100:.1f}% ({len(scores)} examples)")

    print("\nTop 3 Best Examples:")
    for ex, score in results['best_examples'][:3]:
        print(f"  Score: {score:.2f} - {str(ex)[:50]}...")

    print("\nTop 3 Worst Examples:")
    for ex, score in results['worst_examples'][:3]:
        print(f"  Score: {score:.2f} - {str(ex)[:50]}...")

    print("=" * 60)


# Test your implementation
# (You'll need a working module and dataset)
```

### Hints

<details>
<summary>Hint 1: Storing Individual Results</summary>

```python
example_results = []
for example in devset:
    pred = module(**example.inputs())
    score = metric(example, pred)
    example_results.append({
        'example': example,
        'prediction': pred,
        'score': score
    })
```
</details>

<details>
<summary>Hint 2: Finding Best/Worst</summary>

```python
sorted_results = sorted(example_results, key=lambda x: x['score'], reverse=True)
results['best_examples'] = [(r['example'], r['score']) for r in sorted_results[:5]]
results['worst_examples'] = [(r['example'], r['score']) for r in sorted_results[-5:]]
```
</details>

---

## Exercise 4: Preventing Data Leakage

**Difficulty**: ⭐⭐⭐ Advanced

### Objective
Implement a data splitting function that prevents various forms of data leakage.

### Requirements

1. Create a function that:
   - Splits data into train/dev/test sets
   - Removes exact duplicates
   - Groups similar items (by content similarity)
   - Ensures no similar items appear across splits
   - Returns statistics about what was removed/grouped

### Starter Code

```python
import dspy
import random
from collections import defaultdict
from difflib import SequenceMatcher

def safe_data_split(
    data,
    key_field='question',
    similarity_threshold=0.85,
    train_ratio=0.6,
    dev_ratio=0.2,
    seed=42
):
    """
    Split data while preventing various forms of leakage.

    Args:
        data: List of dspy.Example objects
        key_field: Field to use for similarity comparison
        similarity_threshold: Threshold for considering items similar
        train_ratio: Fraction for training set
        dev_ratio: Fraction for dev set
        seed: Random seed

    Returns:
        tuple: (trainset, devset, testset, stats)
    """
    stats = {
        'original_count': len(data),
        'duplicates_removed': 0,
        'similarity_groups': 0,
        'final_counts': {}
    }

    # TODO: Step 1 - Remove exact duplicates
    unique_data = []
    seen = set()

    # TODO: Step 2 - Group similar items
    # Items in the same group should go to the same split

    # TODO: Step 3 - Shuffle groups (not individual items)

    # TODO: Step 4 - Assign groups to splits

    # TODO: Step 5 - Flatten groups back to lists

    # Update stats
    stats['final_counts'] = {
        'train': len(trainset),
        'dev': len(devset),
        'test': len(testset)
    }

    return trainset, devset, testset, stats


def verify_no_leakage(trainset, devset, testset, key_field='question', threshold=0.85):
    """Verify no similar items across splits."""
    def similar(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def get_key(ex):
        return getattr(ex, key_field, '')

    issues = []

    # Check train vs dev
    for train_ex in trainset:
        for dev_ex in devset:
            sim = similar(get_key(train_ex), get_key(dev_ex))
            if sim >= threshold:
                issues.append(f"Train-Dev similarity {sim:.2f}: {get_key(train_ex)[:30]}...")

    # Check train vs test
    for train_ex in trainset:
        for test_ex in testset:
            sim = similar(get_key(train_ex), get_key(test_ex))
            if sim >= threshold:
                issues.append(f"Train-Test similarity {sim:.2f}: {get_key(train_ex)[:30]}...")

    # Check dev vs test
    for dev_ex in devset:
        for test_ex in testset:
            sim = similar(get_key(dev_ex), get_key(test_ex))
            if sim >= threshold:
                issues.append(f"Dev-Test similarity {sim:.2f}: {get_key(dev_ex)[:30]}...")

    return issues


# Test with sample data that has duplicates and similar items
test_data = [
    dspy.Example(question="What is machine learning?", answer="...").with_inputs("question"),
    dspy.Example(question="What is machine learning?", answer="...").with_inputs("question"),  # Duplicate
    dspy.Example(question="What is ML?", answer="...").with_inputs("question"),  # Similar
    dspy.Example(question="Explain machine learning", answer="...").with_inputs("question"),  # Similar
    # Add more varied examples...
]

trainset, devset, testset, stats = safe_data_split(test_data)
print(f"Stats: {stats}")

issues = verify_no_leakage(trainset, devset, testset)
print(f"Leakage issues found: {len(issues)}")
for issue in issues[:5]:
    print(f"  - {issue}")
```

### Hints

<details>
<summary>Hint 1: Grouping Similar Items</summary>

```python
groups = []
assigned = set()

for i, ex1 in enumerate(unique_data):
    if i in assigned:
        continue

    group = [ex1]
    key1 = get_key(ex1)

    for j, ex2 in enumerate(unique_data[i+1:], i+1):
        if j in assigned:
            continue
        key2 = get_key(ex2)

        if similar(key1, key2) >= similarity_threshold:
            group.append(ex2)
            assigned.add(j)

    groups.append(group)
    assigned.add(i)
```
</details>

---

## Exercise 5: Building an Evaluation Dashboard

**Difficulty**: ⭐⭐⭐⭐ Advanced

### Objective
Create a function that generates a comprehensive evaluation report suitable for stakeholder review.

### Requirements

1. Create a report that includes:
   - Executive summary with key metrics
   - Performance breakdown by category
   - Trend analysis (if historical data provided)
   - Error categorization and examples
   - Recommendations based on findings

2. Output should be in Markdown format for easy sharing

### Starter Code

```python
import dspy
from datetime import datetime
from collections import Counter

def generate_evaluation_report(
    module_name: str,
    evaluation_results: dict,
    historical_results: list = None,
    output_path: str = None
):
    """
    Generate a comprehensive evaluation report.

    Args:
        module_name: Name of the module being evaluated
        evaluation_results: Results from comprehensive_evaluation()
        historical_results: Optional list of past evaluation results
        output_path: Optional path to save the report

    Returns:
        str: Markdown-formatted report
    """
    report = []

    # Header
    report.append(f"# Evaluation Report: {module_name}")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"\n**Dataset Size**: {evaluation_results['total_examples']} examples")

    # TODO: Executive Summary
    report.append("\n## Executive Summary\n")
    # Add overall score, pass/fail status, key findings

    # TODO: Performance Metrics
    report.append("\n## Performance Metrics\n")
    # Add detailed metrics table

    # TODO: Category Breakdown
    if evaluation_results.get('by_category'):
        report.append("\n## Performance by Category\n")
        # Add category breakdown table

    # TODO: Trend Analysis (if historical data available)
    if historical_results:
        report.append("\n## Trend Analysis\n")
        # Show performance over time

    # TODO: Error Analysis
    report.append("\n## Error Analysis\n")
    # Categorize and show example errors

    # TODO: Recommendations
    report.append("\n## Recommendations\n")
    # Based on findings, suggest improvements

    # Join report
    full_report = "\n".join(report)

    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(full_report)
        print(f"Report saved to {output_path}")

    return full_report


# Example usage
sample_results = {
    'aggregate_score': 0.82,
    'total_examples': 500,
    'by_category': {
        'factual': [0.9, 0.85, 0.88, 0.92],
        'reasoning': [0.7, 0.65, 0.72, 0.68],
        'creative': [0.8, 0.75, 0.82, 0.78]
    },
    'errors': [
        {'type': 'wrong_answer', 'count': 45},
        {'type': 'incomplete', 'count': 30},
        {'type': 'off_topic', 'count': 15}
    ],
    'best_examples': [],
    'worst_examples': []
}

report = generate_evaluation_report(
    module_name="QA Module v2.1",
    evaluation_results=sample_results,
    output_path="eval_report.md"
)
print(report)
```

---

## Solutions

Complete solutions are available in the `exercises/chapter04/solutions/` directory.

Each solution includes:
- Full working code
- Detailed comments explaining the approach
- Test cases to verify correctness
- Discussion of alternative approaches

### Solution Files

- `exercise01_solution.py` - Creating Quality Datasets
- `exercise02_solution.py` - Designing Custom Metrics
- `exercise03_solution.py` - Systematic Evaluation
- `exercise04_solution.py` - Preventing Data Leakage
- `exercise05_solution.py` - Evaluation Dashboard

---

## Self-Assessment

After completing these exercises, you should be able to:

- [ ] Create balanced, representative datasets with proper splits
- [ ] Design metrics that capture multiple quality dimensions
- [ ] Run comprehensive evaluations with detailed analysis
- [ ] Prevent data leakage in your evaluation pipeline
- [ ] Generate stakeholder-ready evaluation reports

## Next Steps

- Review the [Chapter 4 Examples](../../examples/chapter04/)
- Move on to [Chapter 5: Optimizers](../05-optimizers/00-chapter-intro.md)
- Practice with your own datasets and metrics
