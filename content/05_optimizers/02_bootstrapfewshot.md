# BootstrapFewShot: Automatic Few-Shot Example Generation

## Introduction

BootstrapFewShot is one of DSPy's most powerful optimizers. It automatically generates and selects high-quality few-shot examples to improve your program's performance. Instead of manually crafting examples, BootstrapFewShot discovers the optimal demonstrations for your specific task.

## Weak Supervision and the annotate() Method

A key innovation from the Demonstrate-Search-Predict paper is the concept of **weak supervision** - the ability to train models without hand-labeled intermediate steps. BootstrapFewShot implements this through the `annotate()` functionality, which allows:

1. **Automatic annotation of reasoning chains** without manual step-by-step labeling
2. **Bootstrapping demonstrations** from minimal supervision
3. **Training with only input-output pairs** (no intermediate reasoning needed)

### The annotate() Mechanism

```python
from dspy.teleprompter import BootstrapFewShot

# Traditional approach requires manually annotated reasoning
traditional_training = [
    dspy.Example(
        question="What is 15 * 23?",
        reasoning="Step 1: 15 * 20 = 300\nStep 2: 15 * 3 = 45\nStep 3: 300 + 45 = 345",
        answer="345"
    ),
    # ... many more with detailed reasoning
]

# With weak supervision (annotate), you only need:
weak_supervision_training = [
    dspy.Example(question="What is 15 * 23?", answer="345"),
    dspy.Example(question="What is 12 * 17?", answer="204"),
    # ... just input-output pairs
]

# BootstrapFewShot will automatically generate the reasoning!
```

### How annotate() Works

1. **Teacher-Student Framework**:
   - A teacher model generates full demonstrations
   - The student learns from these generated examples
   - Only final outputs need to be verified

2. **Automatic Reasoning Generation**:
   ```python
   class MathSolver(dspy.Module):
       def __init__(self):
           super().__init__()
           self.solve = dspy.ChainOfThought("question -> answer")

       def forward(self, question):
           result = self.solve(question=question)
           return dspy.Prediction(
               answer=result.answer,
               reasoning=result.rationale  # Automatically generated!
           )
   ```

3. **Filtering by Ground Truth**:
   - Generated demonstrations are validated against known outputs
   - Only high-quality demonstrations are kept
   - Poor generations are automatically discarded

## How BootstrapFewShot Works

### The Bootstrap Process

1. **Initial Generation**: Uses the unoptimized program to generate candidate examples
2. **Quality Filtering**: Evaluates generated examples using your metric
3. **Example Selection**: Chooses the best examples based on performance
4. **Iterative Refinement**: Repeats the process to improve example quality

### Key Components

```python
from dspy.teleprompter import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=your_evaluation_metric,     # How to measure success
    max_bootstrapped_demos=8,          # Maximum examples to generate
    max_labeled_demos=4,               # Maximum labeled examples to include
    max_rounds=2                       # Number of bootstrap rounds
)
```

## Basic Usage

### Simple Example

```python
import dspy
from dspy.teleprompter import BootstrapFewShot

# 1. Define your program
class SimpleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.generate(question=question)

# 2. Define evaluation metric
def exact_match(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# 3. Prepare training data
trainset = [
    dspy.Example(question="What is 2+2?", answer="4"),
    dspy.Example(question="What is the capital of France?", answer="Paris"),
    # ... more examples
]

# 4. Create optimizer and compile
optimizer = BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=4)
compiled_qa = optimizer.compile(SimpleQA(), trainset=trainset)

# 5. Use the compiled program
result = compiled_qa(question="What is 3+3?")
print(result.answer)  # Should be "6"
```

## Advanced Configuration

### Customizing the Bootstrap Process

```python
optimizer = BootstrapFewShot(
    metric=your_metric,
    max_bootstrapped_demos=16,      # Generate more examples
    max_labeled_demos=8,            # Include more labeled examples
    max_rounds=4,                   # More refinement rounds
    max_sample_errors=5             # Maximum errors to sample from
)
```

### Using with Chain of Thought

```python
class CoTQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        result = self.generate(question=question)
        return dspy.Prediction(
            answer=result.answer,
            reasoning=result.rationale
        )

# Bootstrap with Chain of Thought
optimizer = BootstrapFewShot(
    metric=exact_match,
    max_bootstrapped_demos=8,
    teacher_settings=dict(lm=dspy.settings.lm)  # Use same LM for generation
)

compiled_cot = optimizer.compile(CoTQA(), trainset=trainset)
```

### Weak Supervision Example: Complex Reasoning

```python
class ComplexReasoning(dspy.Module):
    def __init__(self):
        super().__init__()
        # Multi-step reasoning task
        self.reason = dspy.ChainOfThought(
            "context, question -> reasoning_steps, answer"
        )

    def forward(self, context, question):
        result = self.reason(context=context, question=question)
        return dspy.Prediction(
            answer=result.answer,
            reasoning_steps=result.rationale  # Will be auto-generated!
        )

# Training data with ONLY inputs and outputs (weak supervision)
reasoning_trainset = [
    dspy.Example(
        context="Alice is taller than Bob. Bob is taller than Charlie.",
        question="Who is the tallest?",
        answer="Alice"
    ),
    dspy.Example(
        context="All mammals are animals. Dogs are mammals.",
        question="Are dogs animals?",
        answer="Yes"
    ),
    # ... more examples without manually written reasoning steps
]

# BootstrapFewShot automatically generates the reasoning steps!
optimizer = BootstrapFewShot(
    metric=exact_match,
    max_bootstrapped_demos=6,
    max_labeled_demos=2  # Keep 2 original examples for stability
)

# The magic: annotate() happens automatically during compilation
compiled_reasoner = optimizer.compile(
    ComplexReasoning(),
    trainset=reasoning_trainset
)

# The compiled model now has high-quality demonstrations
# with automatically generated reasoning steps!
```

### Benefits of Weak Supervision

1. **Reduced Annotation Cost**:
   - No need to write detailed reasoning chains
   - Only final answers need verification
   - Scales to thousands of examples easily

2. **Consistent Quality**:
   - Generated reasoning follows consistent patterns
   - Avoids human annotation inconsistencies
   - Maintains formatting automatically

3. **Rapid Prototyping**:
   - Test new tasks with minimal data preparation
   - Iterate quickly on task definitions
   - Focus on problem formulation, not annotation

4. **Better Coverage**:
   - Generates diverse reasoning strategies
   - Discovers multiple solution paths
   - Reduces annotation bias

## Working with Different Task Types

### Classification Tasks

```python
class TextClassifier(dspy.Module):
    def __init__(self, categories):
        super().__init__()
        self.classify = dspy.Predict(
            f"text, categories[{','.join(categories)}] -> classification"
        )

    def forward(self, text):
        return self.classify(text=text)

# Custom metric for classification
def classification_metric(example, pred, trace=None):
    return example.category.lower() == pred.classification.lower()

categories = ["positive", "negative", "neutral"]
trainset = [
    dspy.Example(text="I love this!", category="positive"),
    dspy.Example(text="This is terrible.", category="negative"),
    # ... more examples
]

optimizer = BootstrapFewShot(metric=classification_metric)
classifier = optimizer.compile(TextClassifier(categories), trainset=trainset)
```

### Multi-Modal Tasks

```python
class ImageCaptioner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.caption = dspy.Predict("image_description -> caption")

    def forward(self, image_description):
        return self.caption(image_description=image_description)

# Bootstrap with image descriptions
image_trainset = [
    dspy.Example(
        image_description="A cat sitting on a windowsill",
        caption="A cat sits on a windowsill looking outside"
    ),
    # ... more examples
]

optimizer = BootstrapFewShot(metric=rouge_score)
captioner = optimizer.compile(ImageCaptioner(), trainset=image_trainset)
```

## BootstrapFewShot Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | Callable | Required | Function to evaluate example quality |
| `max_bootstrapped_demos` | int | 8 | Maximum generated examples |
| `max_labeled_demos` | int | 4 | Maximum human-labeled examples |
| `max_rounds` | int | 2 | Number of bootstrap iterations |
| `max_sample_errors` | int | None | Max error examples to use |

### Advanced Parameters

```python
optimizer = BootstrapFewShot(
    metric=complex_metric,
    max_bootstrapped_demos=16,
    max_labeled_demos=8,
    max_rounds=4,
    max_sample_errors=10,
    learner_class=dspy.teleprompter.BootstrapFewShot,  # Custom learner
    teacher_settings=dict(temperature=0.7),  # Teacher LM settings
    promptgen=None,  # Custom prompt generator
    calibrate=False,  # Calibration mode
    require_metadata=False,  # Metadata requirements
    require_guidance=False,  # Guidance requirements
    language_model=dspy.settings.lm  # Custom LM
)
```

## Metrics for BootstrapFewShot

### Exact Match Metrics

```python
def exact_match_metric(example, pred, trace=None):
    """Simple exact string match."""
    return str(example.answer).lower() == str(pred.answer).lower()

def fuzzy_match(example, pred, trace=None):
    """Fuzzy matching with some tolerance."""
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, example.answer, pred.answer).ratio()
    return similarity > 0.9
```

### Semantic Metrics

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(example, pred, trace=None):
    """Semantic similarity using embeddings."""
    emb1 = model.encode(str(example.answer))
    emb2 = model.encode(str(pred.answer))
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity > 0.8
```

### Task-Specific Metrics

```python
def qa_f1_metric(example, pred, trace=None):
    """F1 score for QA tasks."""
    from collections import Counter

    pred_tokens = Counter(str(pred.answer).lower().split())
    true_tokens = Counter(str(example.answer).lower().split())

    common = pred_tokens & true_tokens
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(true_tokens) if true_tokens else 0

    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)
```

## Best Practices

### 1. Data Quality
```python
# Ensure high-quality training examples
def clean_dataset(dataset):
    cleaned = []
    for example in dataset:
        if len(str(example.answer).strip()) > 0:
            cleaned.append(example)
    return cleaned

trainset = clean_dataset(raw_trainset)
```

### 2. Balanced Examples
```python
# Balance different types of examples
from collections import defaultdict

def balance_dataset(dataset, field):
    """Balance examples by field values."""
    groups = defaultdict(list)
    for example in dataset:
        groups[getattr(example, field)].append(example)

    min_count = min(len(group) for group in groups.values())
    balanced = []
    for group in groups.values():
        balanced.extend(group[:min_count])

    return balanced

# Balance by category
balanced_trainset = balance_dataset(trainset, 'category')
```

### 3. Progressive Compilation
```python
# Start with fewer examples, gradually increase
def progressive_compile(program, trainset):
    results = []
    for num_examples in [4, 8, 12, 16]:
        subset = trainset[:num_examples]
        optimizer = BootstrapFewShot(
            metric=your_metric,
            max_bootstrapped_demos=num_examples
        )
        compiled = optimizer.compile(program, trainset=subset)

        # Evaluate on validation set
        score = evaluate(compiled, valset)
        results.append((num_examples, compiled, score))

    # Return best performing version
    best = max(results, key=lambda x: x[2])
    return best[1]
```

## Common Pitfalls and Solutions

### Pitfall 1: Overfitting
```python
# Problem: Too many bootstrapped examples
optimizer = BootstrapFewShot(max_bootstrapped_demos=50)  # Too many

# Solution: Use reasonable limits
optimizer = BootstrapFewShot(max_bootstrapped_demos=8)  # Better
```

### Pitfall 2: Poor Metric Definition
```python
# Problem: Metric doesn't reflect actual performance
def bad_metric(example, pred):
    return len(pred.answer) > 10  # Bad metric

# Solution: Use meaningful metrics
def good_metric(example, pred):
    return semantic_similarity(example, pred) > 0.8
```

### Pitfall 3: Insufficient Data Diversity
```python
# Problem: All examples are similar
similar_examples = [
    dspy.Example(question="What is 2+2?", answer="4"),
    dspy.Example(question="What is 3+3?", answer="6"),
    # ... all simple math
]

# Solution: Include diverse examples
diverse_examples = [
    dspy.Example(question="What is 2+2?", answer="4"),
    dspy.Example(question="What is the capital of France?", answer="Paris"),
    dspy.Example(question="Explain photosynthesis", answer="Process by which plants..."),
    # ... diverse tasks
]
```

## Evaluating Results

```python
# Compare baseline vs compiled
baseline = SimpleQA()
compiled = optimizer.compile(SimpleQA(), trainset=trainset)

# Evaluate both
baseline_score = evaluate(baseline, testset)
compiled_score = evaluate(compiled, testset)

print(f"Baseline accuracy: {baseline_score:.2%}")
print(f"Compiled accuracy: {compiled_score:.2%}")
print(f"Improvement: {compiled_score - baseline_score:.2%}")
```

## Key Takeaways

1. BootstrapFewShot automatically generates high-quality few-shot examples
2. It improves performance by discovering optimal demonstrations
3. Proper metric definition is crucial for success
4. Data quality and diversity matter more than quantity
5. Always validate compiled programs on held-out data
6. Start with simple configurations and iterate

## Next Steps

In the next section, we'll explore MIPRO, an advanced optimizer that goes beyond example selection to optimize instructions and demonstrations together.