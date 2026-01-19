# Chapter 4: Evaluation

Evaluation is the foundation of building reliable DSPy applications. This chapter teaches you how to measure, validate, and systematically improve your LLM programs through rigorous evaluation practices.

---

## What You'll Learn

By the end of this chapter, you will:

- Understand why evaluation is critical for DSPy optimization
- Create and manage datasets for training, validation, and testing
- Design effective metrics that capture task-specific quality
- Run evaluation loops to measure and track performance
- Apply best practices for reliable, reproducible evaluations

---

## Chapter Overview

This chapter covers the complete evaluation workflow in DSPy:

### [Why Evaluation Matters](01-why-evaluation-matters.md)
Understand the critical role of evaluation in building reliable AI systems.

### [Creating Datasets](02-creating-datasets.md)
Learn to build, structure, and manage datasets using DSPy's Example class.

### [Defining Metrics](03-defining-metrics.md)
Design metrics that accurately measure what matters for your task.

### [Evaluation Loops](04-evaluation-loops.md)
Run systematic evaluations and integrate them into your development workflow.

### [Best Practices](05-best-practices.md)
Follow proven patterns for reliable, reproducible evaluations.

### [Exercises](06-exercises.md)
Practice with 5 hands-on evaluation exercises.

### [Structured Prompting](07-structured-prompting.md)
Learn to create robust, consistent prompts for evaluation tasks. Note that while this overlaps with prompt engineering, it is crucial here for building reliable evaluation criteria.

### [LLM-as-a-Judge](08-llm-as-a-judge.md)
Implement advanced evaluation using LLMs to assess complex outputs.

### [Human-Aligned Evaluation](09-human-aligned-evaluation.md)
Build evaluation systems that correlate with human judgment and real-world priorities.

---

## Prerequisites

Before starting this chapter, ensure you have:

- **Chapter 1-3**: Completed fundamentals, signatures, and modules
- **Working DSPy setup** with API keys configured
- **Basic statistics knowledge** (averages, percentages)
- **Understanding of train/test splits** in machine learning

> **New to evaluation concepts?** This chapter explains everything you need!

---

## Difficulty Level

**Level**: ⭐⭐⭐ Intermediate-Advanced

This chapter introduces concepts that bridge traditional software testing with machine learning evaluation. Understanding these patterns is essential for production DSPy applications.

---

## Estimated Time

**Total time**: 4-5 hours

- Reading: 1.5-2 hours
- Running examples: 1 hour
- Exercises: 1.5-2 hours

---

## The Evaluation Imperative

Without evaluation, you're flying blind:

### Without Evaluation
```python
# How good is this? No idea!
qa = dspy.Predict("question -> answer")
result = qa(question="What is the capital of France?")
print(result.answer)  # "Paris" - but is it always right?
```

### With Evaluation
```python
import dspy

# Define what "good" means
def accuracy(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# Measure systematically
evaluate = dspy.Evaluate(
    devset=test_data,
    metric=accuracy,
    num_threads=8,
    display_progress=True
)

# Know exactly how good it is
score = evaluate(qa)
print(f"Accuracy: {score}%")  # "Accuracy: 87.5%"
```

---

## The Evaluation-Optimization Connection

In DSPy, evaluation isn't just for measuring - it's the engine that drives optimization:

```
┌─────────────────────────────────────────────────────────────┐
│                  DSPy Optimization Loop                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Dataset  ──▶  Metric  ──▶  Optimizer  ──▶  Better Module │
│      │            │             │                │          │
│   Examples    Scoring      Prompt           Improved        │
│   for eval    function    refinement       performance      │
│                                                             │
│   "What to    "How to     "How to          "The result"    │
│    test"      measure"    improve"                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key insight**: The quality of your optimization is bounded by the quality of your evaluation.

---

## Key Concepts Preview

### 1. **Datasets with Examples**
DSPy uses the `Example` class to create structured evaluation data:

```python
example = dspy.Example(
    question="What is the capital of France?",
    answer="Paris"
).with_inputs("question")
```

### 2. **Custom Metrics**
Define what success means for your specific task:

```python
def semantic_match(example, pred, trace=None):
    # Your logic for determining correctness
    return pred.answer.lower() in example.answer.lower()
```

### 3. **The Evaluate Class**
Run systematic evaluations with parallel processing:

```python
evaluate = dspy.Evaluate(
    devset=devset,
    metric=metric,
    num_threads=16,
    display_progress=True
)
```

### 4. **Train/Dev/Test Splits**
Proper data partitioning prevents overfitting:

```python
trainset = data[:200]    # For optimization
devset = data[200:500]   # For development
testset = data[500:]     # For final evaluation
```

---

## Chapter Outline

```
Chapter 4: Evaluation
│
├── Why Evaluation Matters
│   ├── The evaluation imperative
│   ├── Evaluation vs optimization
│   ├── Types of evaluation
│   └── Common pitfalls
│
├── Creating Datasets
│   ├── The Example class
│   ├── with_inputs() method
│   ├── Loading from files/APIs
│   ├── Train/dev/test splits
│   └── Data quality
│
├── Defining Metrics
│   ├── Metric function anatomy
│   ├── Built-in metrics
│   ├── Custom metrics
│   ├── Composite metrics
│   └── The trace parameter
│
├── Evaluation Loops
│   ├── The Evaluate class
│   ├── Parallel evaluation
│   ├── Progress tracking
│   ├── Result analysis
│   └── MLflow integration
│
├── Best Practices
│   ├── Dataset curation
│   ├── Metric design
│   ├── Avoiding data leakage
│   ├── Reproducibility
│   └── Continuous evaluation
│
└── Exercises
    ├── 5 practical exercises
    ├── Metric design challenges
    └── Complete solutions

├── Advanced Topics
    ├── [Structured Prompting](07-structured-prompting.md)
    ├── [LLM-as-a-Judge](08-llm-as-a-judge.md)
    └── [Human-Aligned Evaluation](09-human-aligned-evaluation.md)
```

---

## Code Examples

This chapter includes comprehensive examples in `examples/chapter04/`:

- `01_basic_evaluation.py` - Simple evaluation workflows
- `02_custom_metrics.py` - Designing custom metrics
- `03_dataset_creation.py` - Building evaluation datasets
- `04_evaluation_loops.py` - Running systematic evaluations
- `05_mlflow_integration.py` - Tracking experiments

All examples include detailed comments and sample data!

---

## Real-World Applications

Evaluation powers production systems:

### Quality Assurance
```python
# Ensure responses meet quality standards
def quality_metric(example, pred, trace=None):
    checks = [
        len(pred.answer) >= 50,           # Minimum length
        pred.confidence >= 0.7,            # Confidence threshold
        not contains_hallucination(pred)   # Factuality check
    ]
    return all(checks)
```

### A/B Testing
```python
# Compare two module versions
score_v1 = evaluate(module_v1)
score_v2 = evaluate(module_v2)
print(f"V1: {score_v1}%, V2: {score_v2}%")
```

### Continuous Monitoring
```python
# Track performance over time
with mlflow.start_run():
    score = evaluate(production_module)
    mlflow.log_metric("accuracy", score)
```

---

## Key Takeaways (Preview)

By chapter end, you'll understand:

1. **Evaluation enables optimization** - Without metrics, no improvement
2. **Datasets must be representative** - Garbage in, garbage out
3. **Metrics should capture intent** - Measure what actually matters
4. **Systematic evaluation scales** - Use parallel processing
5. **Best practices prevent mistakes** - Avoid common pitfalls

---

## Learning Approach

This chapter emphasizes practical evaluation skills:

1. **Understand the why** - Motivation for rigorous evaluation
2. **Master the tools** - Example, Evaluate, metrics
3. **Design for your tasks** - Custom metrics and datasets
4. **Build habits** - Best practices for every project

> **Tip**: Good evaluation practices separate amateur from professional DSPy users!

---

## Getting Help

As you work through this chapter:

- **Metric design questions?** See Defining Metrics section
- **Dataset issues?** Check Creating Datasets patterns
- **Performance problems?** Review Evaluation Loops optimization
- **Code errors?** Check examples in `examples/chapter04/`

---

## Let's Begin!

Ready to master DSPy evaluation? Start with [Why Evaluation Matters](01-why-evaluation-matters.md) to understand the foundation.

**Remember**: The best DSPy programs are built on solid evaluation foundations. Time invested here multiplies your effectiveness throughout the entire framework!
