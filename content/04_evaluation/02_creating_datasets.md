# Creating Datasets

## Prerequisites

- **Chapter 1-3**: DSPy Fundamentals, Signatures, and Modules
- **Previous Section**: Why Evaluation Matters
- **Required Knowledge**: Basic Python data structures (lists, dictionaries)
- **Difficulty Level**: Intermediate
- **Estimated Reading Time**: 30 minutes

## Learning Objectives

By the end of this section, you will be able to:
- Create DSPy Examples with inputs and expected outputs
- Use the `with_inputs()` method correctly
- Load datasets from various sources
- Properly split data into train/dev/test sets
- Ensure data quality for reliable evaluation

## The Example Class

DSPy uses the `Example` class to represent individual data points for training and evaluation.

### Basic Example Creation

```python
import dspy

# Create a simple example
example = dspy.Example(
    question="What is the capital of France?",
    answer="Paris"
)

# Access fields
print(example.question)  # "What is the capital of France?"
print(example.answer)    # "Paris"
```

### The with_inputs() Method

The `with_inputs()` method is **critical**—it tells DSPy which fields are inputs vs. expected outputs:

```python
import dspy

# Create example and mark which fields are inputs
example = dspy.Example(
    question="What is the capital of France?",
    answer="Paris"
).with_inputs("question")

# Now DSPy knows:
# - "question" is an INPUT (given to the module)
# - "answer" is an OUTPUT (expected result for evaluation)

# Access input fields
print(example.inputs())  # {"question": "What is the capital of France?"}

# Access all fields including labels
print(example.toDict())  # {"question": "...", "answer": "Paris"}
```

### Multiple Inputs

For signatures with multiple inputs:

```python
import dspy

# Example with multiple input fields
example = dspy.Example(
    context="The Eiffel Tower is located in Paris, France.",
    question="Where is the Eiffel Tower?",
    answer="Paris, France"
).with_inputs("context", "question")

# Both context and question are inputs
# answer is the expected output
print(example.inputs())
# {"context": "The Eiffel Tower is...", "question": "Where is..."}
```

### Multiple Outputs

Examples can have multiple expected outputs:

```python
import dspy

# Example with multiple output fields
example = dspy.Example(
    review="Great product! Fast shipping, excellent quality.",
    sentiment="positive",
    confidence=0.95,
    key_points=["quality", "shipping speed"]
).with_inputs("review")

# review is input
# sentiment, confidence, key_points are expected outputs
```

## Creating Datasets

### Manual Dataset Creation

For small datasets, create examples directly:

```python
import dspy

# Create a list of examples
dataset = [
    dspy.Example(
        question="What is the capital of France?",
        answer="Paris"
    ).with_inputs("question"),

    dspy.Example(
        question="What is the capital of Japan?",
        answer="Tokyo"
    ).with_inputs("question"),

    dspy.Example(
        question="What is the capital of Brazil?",
        answer="Brasilia"
    ).with_inputs("question"),

    # ... more examples
]

print(f"Dataset size: {len(dataset)}")
```

### From Python Dictionaries

Convert existing data structures:

```python
import dspy

# Data from your application
raw_data = [
    {"q": "What is 2+2?", "a": "4"},
    {"q": "What is 3*3?", "a": "9"},
    {"q": "What is 10/2?", "a": "5"},
]

# Convert to DSPy Examples
dataset = [
    dspy.Example(question=item["q"], answer=item["a"]).with_inputs("question")
    for item in raw_data
]
```

### From JSON Files

Load datasets from JSON:

```python
import dspy
import json

# Load from JSON file
with open("data/qa_dataset.json", "r") as f:
    raw_data = json.load(f)

# Convert to Examples
dataset = [
    dspy.Example(**item).with_inputs("question")
    for item in raw_data
]

# Example JSON structure:
# [
#     {"question": "What is AI?", "answer": "Artificial Intelligence"},
#     {"question": "What is ML?", "answer": "Machine Learning"}
# ]
```

### From CSV Files

Load datasets from CSV:

```python
import dspy
import csv

# Load from CSV
dataset = []
with open("data/qa_dataset.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        example = dspy.Example(
            question=row["question"],
            answer=row["answer"]
        ).with_inputs("question")
        dataset.append(example)
```

### From Hugging Face Datasets

DSPy's DataLoader integrates with Hugging Face:

```python
import dspy
from dspy.datasets import DataLoader

# Load from Hugging Face Hub
loader = DataLoader()
raw_data = loader.from_huggingface(
    dataset_name="squad",
    split="train",
    fields=("question", "context", "answers"),
    input_keys=("question", "context"),
    trust_remote_code=True
)

# Process into examples
dataset = [
    dspy.Example(
        question=item.question,
        context=item.context,
        answer=item.answers["text"][0]  # First answer
    ).with_inputs("question", "context")
    for item in raw_data[:1000]  # First 1000 examples
]
```

### Using Built-in Datasets

DSPy includes some built-in datasets:

```python
from dspy.datasets import MATH, HotPotQA

# MATH dataset for mathematical reasoning
math_data = MATH(subset='algebra')
print(f"Train: {len(math_data.train)}, Dev: {len(math_data.dev)}")

# Access examples
example = math_data.train[0]
print(f"Question: {example.question}")
print(f"Answer: {example.answer}")

# HotPotQA for multi-hop reasoning
hotpot = HotPotQA()
```

## Train/Dev/Test Splits

Proper data splitting is essential for valid evaluation.

### Why Split Data?

| Split | Purpose | Usage |
|-------|---------|-------|
| **Training** | Optimize prompts/demonstrations | Used by optimizer |
| **Development** | Tune hyperparameters, iterate | Used during development |
| **Test** | Final unbiased evaluation | Used once at the end |

### Basic Splitting

```python
import dspy
import random

# Load your data
data = load_all_examples()  # Your data loading function

# Shuffle for randomness
random.Random(42).shuffle(data)  # Fixed seed for reproducibility

# Split into sets
trainset = data[:200]      # 200 for training
devset = data[200:500]     # 300 for development
testset = data[500:1000]   # 500 for testing

print(f"Train: {len(trainset)}, Dev: {len(devset)}, Test: {len(testset)}")
```

### Stratified Splitting

For classification tasks, maintain class balance:

```python
import dspy
import random
from collections import defaultdict

def stratified_split(data, train_ratio=0.6, dev_ratio=0.2):
    """Split data while maintaining class distribution."""
    # Group by label
    by_label = defaultdict(list)
    for example in data:
        by_label[example.label].append(example)

    trainset, devset, testset = [], [], []

    for label, examples in by_label.items():
        random.shuffle(examples)
        n = len(examples)
        train_end = int(n * train_ratio)
        dev_end = int(n * (train_ratio + dev_ratio))

        trainset.extend(examples[:train_end])
        devset.extend(examples[train_end:dev_end])
        testset.extend(examples[dev_end:])

    # Shuffle each set
    random.shuffle(trainset)
    random.shuffle(devset)
    random.shuffle(testset)

    return trainset, devset, testset

# Usage
trainset, devset, testset = stratified_split(data)
```

### Time-Based Splitting

For time-series data, respect temporal order:

```python
import dspy
from datetime import datetime

# Sort by timestamp
data.sort(key=lambda x: x.timestamp)

# Use older data for training, newer for testing
cutoff_train = datetime(2024, 1, 1)
cutoff_dev = datetime(2024, 6, 1)

trainset = [ex for ex in data if ex.timestamp < cutoff_train]
devset = [ex for ex in data if cutoff_train <= ex.timestamp < cutoff_dev]
testset = [ex for ex in data if ex.timestamp >= cutoff_dev]
```

## Data Quality

High-quality data is essential for meaningful evaluation.

### Quality Checklist

```python
def validate_dataset(dataset, required_fields):
    """Validate dataset quality."""
    issues = []

    for i, example in enumerate(dataset):
        # Check required fields exist
        for field in required_fields:
            if not hasattr(example, field) or getattr(example, field) is None:
                issues.append(f"Example {i}: Missing field '{field}'")

        # Check for empty strings
        for field in required_fields:
            value = getattr(example, field, "")
            if isinstance(value, str) and len(value.strip()) == 0:
                issues.append(f"Example {i}: Empty '{field}'")

        # Check inputs are marked
        if not example.inputs():
            issues.append(f"Example {i}: No inputs marked (use with_inputs())")

    return issues

# Validate your dataset
issues = validate_dataset(dataset, ["question", "answer"])
if issues:
    print("Data quality issues found:")
    for issue in issues[:10]:  # Show first 10
        print(f"  - {issue}")
else:
    print("Dataset passed validation!")
```

### Cleaning Data

```python
import dspy

def clean_example(example):
    """Clean and normalize an example."""
    return dspy.Example(
        question=example.question.strip(),
        answer=example.answer.strip().lower()
    ).with_inputs("question")

# Clean entire dataset
cleaned_dataset = [clean_example(ex) for ex in dataset]
```

### Removing Duplicates

```python
def deduplicate(dataset, key_field="question"):
    """Remove duplicate examples based on a field."""
    seen = set()
    unique = []

    for example in dataset:
        key = getattr(example, key_field)
        if key not in seen:
            seen.add(key)
            unique.append(example)

    print(f"Removed {len(dataset) - len(unique)} duplicates")
    return unique

dataset = deduplicate(dataset)
```

## Complete Dataset Pipeline

Here's a full example of creating a quality dataset:

```python
import dspy
import json
import random

def create_qa_dataset(filepath, seed=42):
    """
    Create a complete QA dataset from JSON file.

    Args:
        filepath: Path to JSON file with question/answer pairs
        seed: Random seed for reproducibility

    Returns:
        Tuple of (trainset, devset, testset)
    """
    # 1. Load raw data
    with open(filepath, "r") as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} raw examples")

    # 2. Convert to Examples
    dataset = []
    for item in raw_data:
        # Skip invalid entries
        if not item.get("question") or not item.get("answer"):
            continue

        example = dspy.Example(
            question=item["question"].strip(),
            answer=item["answer"].strip()
        ).with_inputs("question")

        dataset.append(example)

    print(f"Created {len(dataset)} valid examples")

    # 3. Remove duplicates
    seen_questions = set()
    unique_dataset = []
    for ex in dataset:
        if ex.question not in seen_questions:
            seen_questions.add(ex.question)
            unique_dataset.append(ex)

    dataset = unique_dataset
    print(f"After deduplication: {len(dataset)} examples")

    # 4. Shuffle with fixed seed
    random.Random(seed).shuffle(dataset)

    # 5. Split into train/dev/test (60/20/20)
    n = len(dataset)
    train_end = int(n * 0.6)
    dev_end = int(n * 0.8)

    trainset = dataset[:train_end]
    devset = dataset[train_end:dev_end]
    testset = dataset[dev_end:]

    print(f"Split: Train={len(trainset)}, Dev={len(devset)}, Test={len(testset)}")

    # 6. Validate
    for split_name, split_data in [("train", trainset), ("dev", devset), ("test", testset)]:
        for ex in split_data:
            assert ex.inputs(), f"Example in {split_name} missing inputs"
            assert ex.question, f"Example in {split_name} missing question"
            assert ex.answer, f"Example in {split_name} missing answer"

    print("Validation passed!")

    return trainset, devset, testset


# Usage
trainset, devset, testset = create_qa_dataset("data/questions.json")
```

## Best Practices

### 1. Always Use with_inputs()

```python
# WRONG - Evaluation won't work correctly
example = dspy.Example(question="...", answer="...")

# CORRECT - Inputs clearly marked
example = dspy.Example(question="...", answer="...").with_inputs("question")
```

### 2. Use Fixed Random Seeds

```python
# WRONG - Different results each run
random.shuffle(data)

# CORRECT - Reproducible shuffling
random.Random(42).shuffle(data)
```

### 3. Validate Before Using

```python
# Always check your data
assert len(trainset) > 0, "Empty training set!"
assert all(ex.inputs() for ex in trainset), "Missing inputs!"
```

### 4. Document Your Datasets

```python
# Create dataset info
dataset_info = {
    "name": "QA Dataset v1",
    "created": "2024-01-15",
    "source": "internal QA logs",
    "train_size": len(trainset),
    "dev_size": len(devset),
    "test_size": len(testset),
    "fields": ["question", "answer"],
    "input_fields": ["question"]
}
```

## Summary

Creating quality datasets involves:

1. **Using the Example class** to structure your data
2. **Marking inputs with `with_inputs()`** to distinguish inputs from outputs
3. **Loading from various sources** (JSON, CSV, Hugging Face)
4. **Proper train/dev/test splitting** to prevent data leakage
5. **Ensuring data quality** through validation and cleaning

### Key Takeaways

1. **`with_inputs()` is essential** - Always mark which fields are inputs
2. **Separate your splits** - Never overlap train and test data
3. **Use fixed seeds** - Ensure reproducibility
4. **Validate your data** - Catch issues early
5. **Document everything** - Future you will thank present you

## Next Steps

- [Next Section: Defining Metrics](./03-defining-metrics.md) - Learn to create evaluation metrics
- [Evaluation Loops](./04-evaluation-loops.md) - Run systematic evaluations
- [Examples](../../examples/chapter04/) - See working code

## Further Reading

- [DSPy Example Class Documentation](https://dspy.ai/api/data/Example)
- [Hugging Face Datasets Library](https://huggingface.co/docs/datasets)
- [Best Practices for ML Datasets](https://developers.google.com/machine-learning/data-prep)
