# Choosing Optimizers: Decision Guide and Trade-offs

## Introduction

DSPy offers multiple optimization strategies, each with distinct strengths and ideal use cases. This chapter provides a comprehensive guide to help you select the right optimizer for your specific needs.

## Quick Reference Guide

| Optimizer | Best For | Data Requirements | Speed | Performance | Complexity |
|-----------|-----------|-------------------|-------|-------------|------------|
| **None (Baseline)** | Simple tasks, quick prototyping | None | Fastest | Baseline | Low |
| **BootstrapFewShot** | General improvement | 10-100 examples | Fast | Good | Medium |
| **KNNFewShot** | Dynamic context, large datasets | 100+ examples | Medium | Good | Medium |
| **MIPRO** | Maximum performance | 20-200 examples | Slow | Excellent | High |
| **Fine-Tuning** | Domain-specific, cost-sensitive | 1000+ examples | Very Slow | Excellent | Very High |

## Decision Framework

### Step 1: Analyze Your Constraints

```python
class OptimizationConstraints:
    def __init__(self):
        # Data constraints
        self.num_examples = None
        self.data_quality = None  # high, medium, low
        self.data_diversity = None  # high, medium, low

        # Resource constraints
        self.time_budget = None  # minutes, hours, days
        self.compute_budget = None  # CPU, single GPU, multi-GPU
        self.memory_limit = None  # GB

        # Performance requirements
        self.target_accuracy = None  # percentage
        self.latency_requirement = None  # ms, seconds
        self.inference_frequency = None  # per day, per hour, per minute

        # Task characteristics
        self.task_complexity = None  # simple, moderate, complex
        self.domain_specificity = None  # general, specialized
        self.explanation_needed = False

def analyze_constraints():
    """Interactive constraint analysis."""
    constraints = OptimizationConstraints()

    print("=== Optimization Constraint Analysis ===\n")

    # Data questions
    constraints.num_examples = int(input(
        "How many training examples do you have? "
    ))

    print("\nData quality (1=low, 2=medium, 3=high):")
    constraints.data_quality = input(
        "How accurate/clean is your data? "
    )

    # Resource questions
    print("\nTime budget:")
    print("1. Minutes (quick prototype)")
    print("2. Hours (reasonable effort)")
    print("3. Days (extensive optimization)")
    time_choice = input("Your time budget? ")
    time_mapping = {"1": "minutes", "2": "hours", "3": "days"}
    constraints.time_budget = time_mapping.get(time_choice, "hours")

    # Performance questions
    constraints.target_accuracy = float(input(
        "\nWhat's your target accuracy improvement (%)? "
    ))

    # Task complexity
    print("\nTask complexity:")
    print("1. Simple (e.g., basic classification)")
    print("2. Moderate (e.g., QA with reasoning)")
    print("3. Complex (e.g., multi-step reasoning)")
    complexity_choice = input("Your task complexity? ")
    complexity_mapping = {"1": "simple", "2": "moderate", "3": "complex"}
    constraints.task_complexity = complexity_mapping.get(complexity_choice, "moderate")

    return constraints

# Example usage
constraints = analyze_constraints()
```

### Step 2: Optimizer Recommendations

```python
def recommend_optimizer(constraints):
    """Provide optimizer recommendations based on constraints."""
    recommendations = []

    # Rule-based recommendations
    if constraints.num_examples < 10:
        recommendations.append({
            "optimizer": "None (Baseline)",
            "reason": "Insufficient data for optimization",
            "confidence": "High"
        })

    elif constraints.time_budget == "minutes":
        recommendations.append({
            "optimizer": "BootstrapFewShot",
            "config": {"max_bootstrapped_demos": 4},
            "reason": "Fast optimization with minimal setup",
            "confidence": "High"
        })

    elif constraints.num_examples > 100 and constraints.task_complexity != "complex":
        recommendations.append({
            "optimizer": "KNNFewShot",
            "config": {"k": 5},
            "reason": "Efficient with large datasets",
            "confidence": "High"
        })

    if constraints.task_complexity == "complex" and constraints.target_accuracy > 10:
        recommendations.append({
            "optimizer": "MIPRO",
            "config": {"num_candidates": 15, "auto": "medium"},
            "reason": "Best for complex tasks requiring maximum performance",
            "confidence": "High"
        })

    if constraints.domain_specificity == "specialized" and constraints.num_examples > 1000:
        recommendations.append({
            "optimizer": "Fine-Tuning",
            "config": {"use_qlora": True, "epochs": 3},
            "reason": "Optimal for domain-specific applications",
            "confidence": "Medium"
        })

    if constraints.inference_frequency == "per minute" and constraints.compute_budget == "CPU":
        recommendations.append({
            "optimizer": "Fine-Tuning",
            "config": {"model_size": "<3B", "quantize": True},
            "reason": "Cost-effective for high-frequency inference",
            "confidence": "Medium"
        })

    return recommendations

# Get recommendations
recommendations = recommend_optimizer(constraints)
print("\n=== Optimizer Recommendations ===")
for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec['optimizer']}")
    print(f"   Reason: {rec['reason']}")
    print(f"   Confidence: {rec['confidence']}")
    if 'config' in rec:
        print(f"   Suggested config: {rec['config']}")
```

## Use Case Analysis

### Use Case 1: Quick Prototype for Startup

**Scenario**: Building an MVP for a customer support bot

**Constraints**:
- Limited data (50 examples)
- Tight deadline (2 days)
- Moderate accuracy required (70%+)
- CPU inference only

**Recommendation**:
```python
optimizer = BootstrapFewShot(
    metric=answer_accuracy,
    max_bootstrapped_demos=8,
    max_labeled_demos=4
)

# Quick iteration cycle
prototype = optimizer.compile(SupportBot(), trainset=examples)
```

**Why**:
- Fast to implement and test
- Works with limited data
- Provides reasonable improvement quickly
- Easy to iterate and refine

### Use Case 2: Enterprise RAG System

**Scenario**: Large-scale document QA for legal firm

**Constraints**:
- Large dataset (10,000 examples)
- High accuracy required (95%+)
- Domain-specific (legal terminology)
- Inference cost matters

**Recommendation**:
```python
# Stage 1: Quick baseline
baseline = BootstrapFewShot(metric=f1_score).compile(
    LegalRAG(), trainset=trainset[:1000]
)

# Stage 2: Advanced optimization
optimizer = MIPRO(
    metric=weighted_metric,
    num_candidates=20,
    auto="heavy"
)
optimized = optimizer.compile(LegalRAG(), trainset=trainset)

# Stage 3: Fine-tune for cost efficiency
if inference_cost_high:
    fine_tuner = FineTuneForDomain()
    final_model = fine_tuner.fine_tune(optimized, domain_data)
```

**Why**:
- Start with BootstrapFewShot for quick baseline
- Use MIPRO for maximum performance
- Consider fine-tuning for long-term cost efficiency

### Use Case 3: Real-time Classification API

**Scenario**: Content moderation for social platform

**Constraints**:
- High throughput (1000+ requests/second)
- Low latency requirement (<100ms)
- Continuous learning (new content types)
- Good accuracy sufficient (85%+)

**Recommendation**:
```python
# KNNFewShot for adaptive context
optimizer = KNNFewShot(
    k=3,
    similarity_fn=semantic_similarity,
    cache_embeddings=True
)

classifier = optimizer.compile(
    ContentModerator(),
    trainset=moderation_examples
)

# Option: Fine-tune small model for deployment
if latency_critical:
    small_model = fine_tune_classifier(
        base_model="gemma-2b",
        training_data=examples,
        quantize=True
    )
```

**Why**:
- KNNFewShot provides context-aware classification
- Embedding caching improves speed
- Small fine-tuned model for production if needed

## Performance Comparison

### Benchmark Methodology

```python
import time
import pandas as pd

def benchmark_optimizers(program, trainset, testset, optimizers):
    """Compare optimizer performance."""
    results = []

    for name, optimizer_config in optimizers.items():
        print(f"\nTesting {name}...")

        # Record start time
        start_time = time.time()

        # Compile/prepare model
        if name == "Baseline":
            compiled = program
        else:
            optimizer = optimizer_config['optimizer']
            compiled = optimizer.compile(
                program,
                trainset=trainset,
                **optimizer_config.get('kwargs', {})
            )

        # Record compilation time
        compile_time = time.time() - start_time

        # Evaluate performance
        eval_start = time.time()
        accuracy = evaluate(compiled, testset)
        eval_time = time.time() - eval_start

        # Calculate inference speed
        speed_start = time.time()
        for example in testset[:10]:  # Sample for speed test
            _ = compiled(**example.inputs())
        avg_inference_time = (time.time() - speed_start) / 10

        results.append({
            'Optimizer': name,
            'Accuracy': accuracy,
            'Compilation Time (s)': compile_time,
            'Evaluation Time (s)': eval_time,
            'Avg Inference (ms)': avg_inference_time * 1000,
            'Parameters': str(optimizer_config.get('kwargs', {}))
        })

    return pd.DataFrame(results)

# Example benchmark
optimizers_to_test = {
    "Baseline": {},
    "BootstrapFewShot": {
        "optimizer": BootstrapFewShot(metric=accuracy_metric),
        "kwargs": {"max_bootstrapped_demos": 8}
    },
    "KNNFewShot": {
        "optimizer": KNNFewShot(k=5),
        "kwargs": {}
    },
    "MIPRO": {
        "optimizer": MIPRO(metric=accuracy_metric),
        "kwargs": {"num_candidates": 10, "auto": "medium"}
    }
}

results_df = benchmark_optimizers(
    my_program,
    trainset,
    testset,
    optimizers_to_test
)

print(results_df)
```

### Expected Performance Patterns

| Optimizer | Accuracy Gain | Compile Time | Inference Speed | Best For |
|-----------|---------------|--------------|-----------------|----------|
| Baseline | 0% | < 1s | Fastest | Quick testing |
| BootstrapFewShot | 5-15% | 1-5 min | Fast | Most tasks |
| KNNFewShot | 5-12% | 1-2 min | Medium | Context tasks |
| MIPRO | 10-25% | 5-30 min | Fast | Complex tasks |
| Fine-Tuning | 15-30% | 1-4 hrs | Fast-Medium | Production |

## Optimization Strategies

### Strategy 1: Progressive Optimization

```python
def progressive_optimization(program, trainset, valset):
    """Start simple and progressively add optimization."""
    stages = [
        {
            "name": "Baseline",
            "optimizer": None,
            "description": "No optimization"
        },
        {
            "name": "BootstrapFewShot",
            "optimizer": BootstrapFewShot(metric=accuracy_metric),
            "config": {"max_bootstrapped_demos": 4},
            "description": "Basic few-shot learning"
        },
        {
            "name": "KNNFewShot",
            "optimizer": KNNFewShot(k=3),
            "description": "Context-aware examples"
        },
        {
            "name": "MIPRO",
            "optimizer": MIPRO(metric=accuracy_metric, auto="medium"),
            "description": "Full optimization"
        }
    ]

    results = {}
    best_program = program
    best_score = 0

    for stage in stages:
        print(f"\n=== Stage: {stage['name']} ===")
        print(f"Description: {stage['description']}")

        if stage['optimizer']:
            compiled = stage['optimizer'].compile(
                best_program,
                trainset=trainset,
                **stage.get('config', {})
            )
        else:
            compiled = program

        # Evaluate
        score = evaluate(compiled, valset)
        results[stage['name']] = score

        print(f"Score: {score:.3f}")

        # Keep the best
        if score > best_score:
            best_score = score
            best_program = compiled
            print("✓ New best model!")

    return best_program, results

# Use progressive optimization
final_model, all_scores = progressive_optimization(
    my_program,
    trainset,
    valset
)
```

### Strategy 2: Ensemble Approaches

```python
class EnsembleOptimizer:
    """Combine multiple optimized programs."""
    def __init__(self):
        self.models = []
        self.weights = []

    def add_model(self, model, weight=1.0):
        self.models.append(model)
        self.weights.append(weight)

    def predict(self, **kwargs):
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model(**kwargs)
            predictions.append((pred, weight))

        # Weighted voting for classification
        if hasattr(predictions[0][0], 'answer'):
            answers = {}
            for pred, weight in predictions:
                answer = pred.answer
                answers[answer] = answers.get(answer, 0) + weight
            best_answer = max(answers, key=answers.get)
            return dspy.Prediction(answer=best_answer)

        return predictions[0][0]  # Return first for other cases

# Create ensemble
ensemble = EnsembleOptimizer()

# Add different optimized versions
ensemble.add_model(bootstrap_model, weight=0.3)
ensemble.add_model(knn_model, weight=0.3)
ensemble.add_model(mipro_model, weight=0.4)
```

### Strategy 3: Adaptive Optimization

```python
def adaptive_optimization(program, trainset, valset, target_accuracy):
    """Automatically select optimizer based on data characteristics."""
    # Analyze data
    num_examples = len(trainset)
    diversity_score = calculate_diversity(trainset)

    # Start with appropriate optimizer
    if num_examples < 20:
        optimizer = BootstrapFewShot(metric=accuracy_metric)
        print("Using BootstrapFewShot (small dataset)")
    elif diversity_score > 0.7:
        optimizer = KNNFewShot(k=5)
        print("Using KNNFewShot (high diversity)")
    else:
        optimizer = MIPRO(metric=accuracy_metric, auto="medium")
        print("Using MIPRO (general case)")

    # Initial optimization
    compiled = optimizer.compile(program, trainset=trainset)
    score = evaluate(compiled, valset)

    print(f"Initial score: {score:.3f}")

    # If still below target, try more advanced optimization
    if score < target_accuracy and num_examples > 50:
        print("Trying MIPRO for better performance...")
        mipro = MIPRO(metric=accuracy_metric, auto="heavy")
        compiled = mipro.compile(program, trainset=trainset)
        score = evaluate(compiled, valset)
        print(f"Final score: {score:.3f}")

    return compiled

# Use adaptive optimization
final_model = adaptive_optimization(
    my_program,
    trainset,
    valset,
    target_accuracy=0.85
)
```

## Cost-Benefit Analysis

### Optimization Cost Calculator

```python
def calculate_optimization_cost(optimizer_type, config, data_size):
    """Estimate time and compute cost of optimization."""
    costs = {
        "BootstrapFewShot": {
            "time_per_example": 0.5,  # seconds
            "base_time": 60,  # seconds
            "compute_multiplier": 1.0
        },
        "KNNFewShot": {
            "time_per_example": 0.2,
            "base_time": 30,
            "compute_multiplier": 1.2
        },
        "MIPRO": {
            "time_per_example": 5.0,
            "base_time": 300,
            "compute_multiplier": 3.0
        },
        "FineTuning": {
            "time_per_example": 10.0,
            "base_time": 1800,
            "compute_multiplier": 10.0
        }
    }

    if optimizer_type not in costs:
        return None

    cost_info = costs[optimizer_type]
    estimated_time = (
        cost_info["base_time"] +
        cost_info["time_per_example"] * data_size
    )

    # Adjust for configuration
    if optimizer_type == "MIPRO":
        candidates = config.get("num_candidates", 10)
        estimated_time *= candidates / 10

    return {
        "optimizer": optimizer_type,
        "estimated_time_minutes": estimated_time / 60,
        "estimated_time_hours": estimated_time / 3600,
        "compute_units": estimated_time * cost_info["compute_multiplier"] / 3600
    }

# Example usage
for optimizer in ["BootstrapFewShot", "KNNFewShot", "MIPRO", "FineTuning"]:
    cost = calculate_optimization_cost(optimizer, {}, 1000)
    print(f"\n{optimizer}:")
    print(f"  Time: {cost['estimated_time_minutes']:.1f} minutes")
    print(f"  Compute: {cost['compute_units']:.1f} units")
```

### ROI Analysis

```python
def analyze_roi(optimization_costs, performance_gains, inference_volume):
    """Analyze return on investment for optimization."""
    analysis = {}

    for optimizer, cost in optimization_costs.items():
        gain = performance_gains.get(optimizer, 0)
        monthly_savings = gain * inference_volume * 0.001  # Example value

        roi = {
            "optimizer": optimizer,
            "optimization_cost": cost["compute_units"] * 10,  # $10 per unit
            "monthly_savings": monthly_savings,
            "payback_period_days": (cost["compute_units"] * 10) / (monthly_savings / 30),
            "annual_roi": (monthly_savings * 12 - cost["compute_units"] * 10) / (cost["compute_units"] * 10)
        }

        analysis[optimizer] = roi

    return analysis

# Example ROI analysis
costs = {
    "BootstrapFewShot": calculate_optimization_cost("BootstrapFewShot", {}, 1000),
    "MIPRO": calculate_optimization_cost("MIPRO", {}, 1000)
}

gains = {
    "BootstrapFewShot": 0.08,  # 8% improvement
    "MIPRO": 0.15  # 15% improvement
}

roi_analysis = analyze_roi(costs, gains, inference_volume=100000)
for optimizer, analysis in roi_analysis.items():
    print(f"\n{optimizer} ROI:")
    print(f"  Payback period: {analysis['payback_period_days']:.1f} days")
    print(f"  Annual ROI: {analysis['annual_roi']:.1%}")
```

## Key Takeaways

1. **Data Size Matters**: More data enables more sophisticated optimization
2. **Task Complexity Drives Choice**: Complex tasks benefit from MIPRO
3. **Latency vs Accuracy Trade-off**: Consider your specific needs
4. **Progressive Approach Works**: Start simple, iterate to complex
5. **Cost-Benefit Analysis**: Not all optimization justifies the cost
6. **Ensemble Methods**: Can combine strengths of multiple optimizers

## Quick Decision Tree

```
Need optimization?
├─ No → Use baseline
└─ Yes
    ├─ < 20 examples?
    │  └─ BootstrapFewShot (k=4)
    ├─ < 100 examples?
    │  ├─ Need quick results?
    │  │  └─ BootstrapFewShot
    │  └─ Can wait longer?
    │     └─ MIPRO (light)
    ├─ > 100 examples?
    │  ├─ Context matters?
    │  │  └─ KNNFewShot
    │  └─ Maximum performance?
    │     └─ MIPRO (heavy)
    └─ Domain-specific & > 1000 examples?
       └─ Consider Fine-Tuning
```

## Next Steps

Now you have a comprehensive understanding of DSPy optimizers. In the exercises, you'll apply these concepts to real-world scenarios and learn to make informed optimization decisions.</think>