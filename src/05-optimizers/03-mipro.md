# MIPRO: Multi-step Instruction and Demonstration Optimization

## Introduction

MIPRO (Multi-step Instruction and demonstration PRompt Optimization) is DSPy's most advanced optimizer. Unlike BootstrapFewShot which only optimizes examples, MIPRO simultaneously optimizes both the instructions (prompts) and demonstrations (examples) to achieve superior performance.

## What Makes MIPRO Special?

### Dual Optimization
1. **Instruction Optimization**: Rewrites and refines natural language instructions
2. **Demonstration Optimization**: Selects and generates optimal examples
3. **Joint Optimization**: Optimizes instructions and examples together

### Multi-Step Process
MIPRO uses an iterative approach to progressively improve your program:
1. Analyze current performance
2. Identify weak points
3. Generate improvements
4. Evaluate and select best versions
5. Repeat until convergence

## Basic MIPRO Usage

### Simple Example

```python
import dspy
from dspy.teleprompter import MIPRO

# 1. Define your program
class AdvancedQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.generate(question=question)

# 2. Define evaluation metric
def answer_em(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# 3. Prepare data
trainset = [
    dspy.Example(question="What causes rain?", answer="Condensation of water vapor"),
    dspy.Example(question="Why is the sky blue?", answer="Rayleigh scattering of light"),
    # ... more examples
]

# 4. Create MIPRO optimizer
optimizer = MIPRO(
    metric=answer_em,
    num_candidates=10,      # Generate 10 candidate instructions
    init_temperature=1.0     # Start with high creativity
)

# 5. Compile the program
compiled_qa = optimizer.compile(
    AdvancedQA(),
    trainset=trainset,
    num_trials=3,          # Run optimization 3 times
    max_bootstrapped_demos=8
)

# 6. Use the optimized program
result = compiled_qa(question="How do airplanes fly?")
print(result.answer)
```

## Advanced Configuration

### Customizing MIPRO Parameters

```python
optimizer = MIPRO(
    metric=your_metric,
    num_candidates=20,          # More instruction candidates
    init_temperature=1.2,       # Higher initial creativity
    verbose=True,               # Show optimization progress
    auto="medium",             # Auto mode: "light", "medium", "heavy"
    adapt_temperature=True,    # Adapt temperature during optimization
    logic_history=True         # Track optimization history
)
```

### Multi-Objective Optimization

```python
def multi_metric(example, pred, trace=None):
    """Combines multiple metrics."""
    accuracy = exact_match(example, pred)
    efficiency = length_penalty(pred)
    coherence = coherence_score(pred)

    # Weighted combination
    return 0.5 * accuracy + 0.3 * efficiency + 0.2 * coherence

optimizer = MIPRO(metric=multi_metric, num_candidates=15)
```

## MIPRO Optimization Strategies

### 1. Instruction Evolution

MIPRO evolves instructions through multiple generations:

```python
# Generation 0: Original instruction
original_inst = "Answer the question based on your knowledge."

# Generation 1: MIPRO variations
gen1_variations = [
    "Carefully analyze the question and provide a precise answer.",
    "Think step by step before giving your final answer.",
    "Consider the context and nuances of the question.",
    # ... more variations
]

# Generation 2: Refined instructions
gen2_variations = [
    "Analyze the question step-by-step, consider all relevant information, and provide a precise, accurate answer.",
    "Break down the question into components, reason about each, then synthesize a comprehensive answer.",
    # ... even better instructions
]
```

### 2. Demonstration Synthesis

MIPRO can create synthetic demonstrations:

```python
class SyntheticExampleGenerator:
    def __init__(self, lm):
        self.lm = lm

    def generate_example(self, instruction, topic):
        """Generate a new example based on instruction."""
        prompt = f"""
        Instruction: {instruction}

        Generate a high-quality example for this instruction about: {topic}

        Example:
        """
        return self.lm.generate(prompt)

# MIPRO uses this internally to create diverse examples
```

### 3. Joint Optimization

```python
# MIPRO evaluates instruction-example pairs together
def evaluate_pair(instruction, examples, test_set):
    """Evaluate how well instruction and examples work together."""
    temp_program = dspy.Predict(instruction)
    temp_program.demos = examples

    score = 0
    for test_example in test_set:
        pred = temp_program(**test_example.inputs())
        score += evaluate_metric(test_example, pred)

    return score / len(test_set)
```

## Using MIPRO with Complex Programs

### Multi-Module Programs

```python
class RAGSystem(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# MIPRO optimizes both modules
optimizer = MIPRO(metric=answer_em, num_candidates=15)
optimized_rag = optimizer.compile(
    RAGSystem(),
    trainset=trainset,
    max_bootstrapped_demos=5
)
```

### Custom Module Optimization

```python
class CustomAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict("text -> entities, sentiment")
        self.summarizer = dspy.Predict("text, entities, sentiment -> summary")

    def forward(self, text):
        extracted = self.extractor(text=text)
        return self.summarizer(
            text=text,
            entities=extracted.entities,
            sentiment=extracted.sentiment
        )

# MIPRO with custom evaluation
def analyzer_metric(example, pred, trace=None):
    entity_f1 = calculate_f1(example.entities, pred.entities)
    sentiment_match = example.sentiment == pred.sentiment
    summary_rouge = rouge_score(example.summary, pred.summary)

    return 0.4 * entity_f1 + 0.3 * sentiment_match + 0.3 * summary_rouge

optimizer = MIPRO(metric=analyzer_metric, num_candidates=12)
analyzer = optimizer.compile(CustomAnalyzer(), trainset=trainset)
```

## MIPRO Parameters Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | Callable | Required | Evaluation function |
| `num_candidates` | int | 10 | Number of instruction candidates |
| `init_temperature` | float | 1.0 | Initial creativity temperature |
| `verbose` | bool | False | Show optimization details |
| `auto` | str | None | Auto mode: "light", "medium", "heavy" |

### Advanced Parameters

```python
optimizer = MIPRO(
    metric=complex_metric,
    num_candidates=20,
    init_temperature=1.2,
    verbose=True,
    auto="heavy",
    adapt_temperature=True,
    logic_history=True,
    breadth=10,               # Search breadth
    depth=3,                  # Search depth
    max_labeled_demos=4,      # Max labeled examples
    max_bootstrapped_demos=8, # Max generated examples
    temperature_range=(0.7, 1.3),  # Temperature bounds
    instruction_penalty=0.1,  # Penalize long instructions
    example_diversity=0.2     # Encourage diverse examples
)
```

## MIPRO Auto Modes

### Light Mode
```python
# Quick optimization for simple tasks
optimizer = MIPRO(auto="light")
# Equivalent to:
optimizer = MIPRO(num_candidates=5, init_temperature=0.8)
```

### Medium Mode
```python
# Balanced optimization
optimizer = MIPRO(auto="medium")
# Equivalent to:
optimizer = MIPRO(num_candidates=10, init_temperature=1.0)
```

### Heavy Mode
```python
# Extensive optimization for complex tasks
optimizer = MIPRO(auto="heavy")
# Equivalent to:
optimizer = MIPRO(num_candidates=20, init_temperature=1.2)
```

## Monitoring MIPRO Optimization

### Progress Tracking

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

# MIPRO will log:
# - Generation 1 instructions
# - Performance scores
# - Best candidates
# - Convergence information

optimizer = MIPRO(metric=your_metric, verbose=True)
```

### Custom Callbacks

```python
class MIPROTracker:
    def __init__(self):
        self.generation = 0
        self.best_score = 0
        self.history = []

    def __call__(self, program, metrics, traces):
        self.generation += 1
        current_score = metrics['score']

        if current_score > self.best_score:
            self.best_score = current_score

        self.history.append({
            'generation': self.generation,
            'score': current_score,
            'best': self.best_score
        })

        print(f"Gen {self.generation}: Score={current_score:.3f}, Best={self.best_score:.3f}")

tracker = MIPROTracker()
optimizer = MIPRO(metric=your_metric, callbacks=[tracker])
```

## Best Practices

### 1. Start with Good Instructions

```python
# Bad: Too vague
vague_instruction = "Answer the question."

# Good: Specific and clear
good_instruction = """
Analyze the question carefully, break it down into key components,
provide a comprehensive answer that addresses all aspects of the question.
"""

# Even better: Include examples of desired behavior
best_instruction = """
When answering questions:
1. Identify the core question being asked
2. Consider relevant context and background information
3. Provide a clear, direct answer
4. Include supporting details or explanations when helpful
5. Ensure the answer is accurate and complete
"""
```

### 2. Use Appropriate Temperature

```python
# For well-defined tasks with clear answers
optimizer = MIPRO(init_temperature=0.7, num_candidates=8)

# For creative or open-ended tasks
optimizer = MIPRO(init_temperature=1.3, num_candidates=15)

# For mixed tasks (most common case)
optimizer = MIPRO(init_temperature=1.0, num_candidates=10)
```

### 3. Provide Diverse Training Data

```python
# Ensure coverage of different question types
diverse_trainset = []

# Factual questions
diverse_trainset.extend(factual_questions)

# Reasoning questions
diverse_trainset.extend(reasoning_questions)

# Opinion questions
diverse_trainset.extend(opinion_questions)

# Domain-specific questions
diverse_trainset.extend(domain_questions)
```

### 4. Evaluate Progressively

```python
def progressive_evaluation(program, optimizer, trainset, valset):
    """Evaluate at different stages of optimization."""
    results = []

    for num_trials in [1, 3, 5, 10]:
        compiled = optimizer.compile(
            program,
            trainset=trainset,
            num_trials=num_trials
        )

        score = evaluate(compiled, valset)
        results.append((num_trials, score))

        print(f"Trials: {num_trials}, Score: {score:.3f}")

    return results
```

## Common Pitfalls and Solutions

### Pitfall 1: Over-optimization
```python
# Problem: Too many candidates leading to diminishing returns
optimizer = MIPRO(num_candidates=50)  # May overfit

# Solution: Use reasonable limits and monitor performance
optimizer = MIPRO(num_candidates=15, auto="medium")
```

### Pitfall 2: Inadequate Evaluation Metric
```python
# Problem: Metric doesn't capture important aspects
def simple_metric(example, pred):
    return example.answer in pred.answer  # Too simple

# Solution: Use comprehensive metrics
def comprehensive_metric(example, pred):
    accuracy = exact_match(example, pred)
    completeness = coverage_score(example, pred)
    clarity = clarity_score(pred)
    return 0.5 * accuracy + 0.3 * completeness + 0.2 * clarity
```

### Pitfall 3: Poor Training Data Quality
```python
# Problem: Inconsistent or incorrect labels
noisy_data = [
    dspy.Example(question="What is 2+2?", answer="5"),  # Wrong!
    # ... more noisy examples
]

# Solution: Clean and validate data
def clean_data(data):
    cleaned = []
    for example in data:
        if validate_example(example):
            cleaned.append(example)
    return cleaned

clean_trainset = clean_data(raw_data)
```

## Comparing MIPRO with Other Optimizers

```python
from dspy.teleprompter import BootstrapFewShot, MIPRO

# Compare optimizers on same task
def compare_optimizers(program, trainset, testset):
    optimizers = {
        'Baseline': None,
        'BootstrapFewShot': BootstrapFewShot(metric=exact_match),
        'MIPRO': MIPRO(metric=exact_match, num_candidates=10)
    }

    results = {}

    for name, optimizer in optimizers.items():
        if optimizer:
            compiled = optimizer.compile(program, trainset=trainset)
        else:
            compiled = program  # Baseline

        score = evaluate(compiled, testset)
        results[name] = score

    return results

results = compare_optimizers(my_program, trainset, testset)
print("Optimization Results:")
for name, score in results.items():
    print(f"{name}: {score:.3f}")
```

## Key Takeaways

1. MIPRO optimizes both instructions and examples simultaneously
2. It uses an evolutionary approach to progressively improve programs
3. MIPRO achieves superior performance on complex tasks
4. Proper metric design is crucial for successful optimization
5. Start with good instructions and diverse training data
6. Monitor optimization progress to avoid overfitting

## Next Steps

In the next section, we'll explore KNNFewShot, an optimizer that uses similarity-based example selection for efficient optimization.