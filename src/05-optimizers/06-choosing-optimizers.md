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
| **[RPE](08-reflective-prompt-evolution.md)** | Complex reasoning, exploration | 30+ examples | Slow | Excellent | High |
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

        # Also suggest RPE for complex reasoning tasks
        if constraints.num_examples >= 30:
            recommendations.append({
                "optimizer": "ReflectivePromptEvolution",
                "config": {"population_size": 10, "generations": 5},
                "reason": "Evolutionary approach excellent for complex multi-step reasoning",
                "confidence": "Medium"
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

### Use Case 3: Complex Multi-hop Reasoning

**Scenario**: Question answering requiring multiple reasoning steps (e.g., HotpotQA, complex medical diagnosis)

**Constraints**:
- Multi-step reasoning required
- Complex problem decomposition needed
- Medium dataset size (50-500 examples)
- High accuracy critical (>90%)

**Recommendation**:
```python
# RPE for complex reasoning tasks
optimizer = ReflectivePromptEvolution(
    metric=multi_hop_accuracy,
    population_size=12,
    generations=6,
    mutation_rate=0.3,
    diversity_weight=0.4
)

reasoning_system = optimizer.compile(
    MultiHopReasoner(),
    trainset=complex_qa_examples,
    valset=val_examples
)

# Combine with Chain of Thought for best results
final_system = ChainOfThoughtEnhanced(reasoning_system)
```

**Why**:
- RPE excels at discovering novel reasoning patterns
- Evolutionary approach explores multiple solution paths
- Self-reflection improves reasoning quality over time
- Diversity maintenance prevents converging on suboptimal approaches

### Use Case 4: Real-time Classification API

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
    },
    "RPE": {
        "optimizer": ReflectivePromptEvolution(metric=accuracy_metric),
        "kwargs": {"population_size": 8, "generations": 4}
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
| RPE | 12-28% | 10-45 min | Fast | Complex reasoning |
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
        },
        {
            "name": "RPE",
            "optimizer": ReflectivePromptEvolution(metric=accuracy_metric),
            "config": {"population_size": 8, "generations": 4},
            "description": "Evolutionary optimization for complex reasoning"
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
        "RPE": {
            "time_per_example": 8.0,
            "base_time": 600,
            "compute_multiplier": 5.0
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
for optimizer in ["BootstrapFewShot", "KNNFewShot", "MIPRO", "RPE", "FineTuning"]:
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

## Optimization Order Effects

When combining multiple optimization strategies, the order of application significantly impacts final performance.

### Why Order Matters

Research on joint optimization demonstrates that **fine-tuning first, then prompt optimization** consistently outperforms the reverse order:

```python
# OPTIMAL ORDER
# Fine-tuning -> Prompt Optimization
# Improvement: 3.5x beyond individual approaches

# SUBOPTIMAL ORDER
# Prompt Optimization -> Fine-tuning
# Improvement: Only 1.8x (prompts don't transfer well)

def demonstrate_order_effects(program, trainset, testset, base_model):
    """Show impact of optimization order."""
    results = {}

    # Baseline
    results["baseline"] = evaluate(program, testset)

    # Order 1: Fine-tune first (RECOMMENDED)
    finetuned = finetune(base_model, trainset)
    dspy.settings.configure(lm=finetuned)
    optimizer = MIPRO(metric=accuracy, auto="medium")
    compiled = optimizer.compile(program, trainset=trainset)
    results["ft_then_po"] = evaluate(compiled, testset)

    # Order 2: Prompt optimize first (NOT RECOMMENDED)
    dspy.settings.configure(lm=base_model)
    compiled_base = optimizer.compile(program, trainset=trainset)
    finetuned_after = finetune(base_model, trainset)
    dspy.settings.configure(lm=finetuned_after)
    # Note: prompts optimized for base model may not work well
    results["po_then_ft"] = evaluate(compiled_base, testset)

    print(f"Baseline: {results['baseline']:.2%}")
    print(f"Fine-tune -> Prompt Opt: {results['ft_then_po']:.2%}")
    print(f"Prompt Opt -> Fine-tune: {results['po_then_ft']:.2%}")

    return results
```

### The Optimization Order Decision Tree

```
Starting optimization?
|
+-- Have compute for fine-tuning?
|   +-- Yes: Fine-tune first
|   |       |
|   |       +-- Need maximum performance?
|   |           +-- Yes: MIPRO or COPA
|   |           +-- No: BootstrapFewShot
|   |
|   +-- No: Skip to prompt optimization
|           |
|           +-- Complex task?
|               +-- Yes: MIPRO or RPE
|               +-- No: BootstrapFewShot or KNNFewShot
```

## Synergy Quantification

Combined optimization approaches achieve synergistic effects that exceed the sum of individual improvements.

### Measuring Synergy

```python
def calculate_synergy(baseline, ft_only, po_only, combined):
    """
    Calculate synergistic improvement from combined optimization.

    Synergy = Combined - (Baseline + FT_Improvement + PO_Improvement)

    A positive synergy indicates the approaches work better together
    than they would independently.
    """
    ft_improvement = ft_only - baseline
    po_improvement = po_only - baseline
    additive_expected = baseline + ft_improvement + po_improvement

    synergy = combined - additive_expected
    synergy_multiplier = combined / additive_expected if additive_expected > 0 else 0

    return {
        "baseline": baseline,
        "fine_tuning_only": ft_only,
        "prompt_opt_only": po_only,
        "combined": combined,
        "additive_expected": additive_expected,
        "synergy_absolute": synergy,
        "synergy_multiplier": synergy_multiplier
    }

# Example from research benchmarks:
# Baseline: 12%
# Fine-tuning only: 28% (+16%)
# Prompt optimization only: 20% (+8%)
# Combined: 45% (not 36%!)

synergy_result = calculate_synergy(
    baseline=0.12,
    ft_only=0.28,
    po_only=0.20,
    combined=0.45
)

print(f"Expected additive: {synergy_result['additive_expected']:.2%}")
print(f"Actual combined: {synergy_result['combined']:.2%}")
print(f"Synergy: {synergy_result['synergy_absolute']:.2%}")
print(f"Synergy multiplier: {synergy_result['synergy_multiplier']:.2f}x")
# Output: Synergy multiplier: 1.25x (25% better than additive)
```

### Benchmark Synergy Results

| Task | Baseline | FT Only | PO Only | Expected | Combined | Synergy |
|------|----------|---------|---------|----------|----------|---------|
| MultiHopQA | 12% | 28% | 20% | 36% | 45% | 3.5x |
| GSM8K Math | 11% | 32% | 22% | 43% | 55% | 2.8x |
| AQuA | 9% | 35% | 28% | 54% | 69% | 3.4x |
| Classification | 65% | 82% | 78% | 95%* | 91% | N/A |

*Note: Classification ceiling effects limit synergy measurement.

### When Synergy Is Highest

Synergy is most pronounced when:

1. **Task complexity is high**: Multi-step reasoning tasks
2. **Base model capability is low**: Smaller models (< 13B)
3. **Instructions are complex**: Multi-part requirements
4. **Domain is specialized**: Technical/domain-specific content

```python
def predict_synergy_potential(task_complexity, model_size, instruction_complexity):
    """
    Estimate potential synergy from combined optimization.

    Higher values indicate greater potential benefit.
    """
    # Empirical factors from research
    complexity_factor = {"simple": 1.0, "moderate": 1.5, "complex": 2.5}
    size_factor = {"<7B": 2.0, "7-13B": 1.5, ">13B": 1.0}
    instruction_factor = {"basic": 1.0, "detailed": 1.5, "multi_step": 2.0}

    synergy_potential = (
        complexity_factor.get(task_complexity, 1.0) *
        size_factor.get(model_size, 1.0) *
        instruction_factor.get(instruction_complexity, 1.0)
    )

    return synergy_potential
```

## Joint Optimization Limitations

While combined optimization offers powerful improvements, understanding its limitations helps set realistic expectations.

### Data Requirements

```python
JOINT_OPTIMIZATION_REQUIREMENTS = {
    "minimum_examples": 50,
    "recommended_examples": 100,
    "optimal_examples": 200,
    "warning_threshold": 30
}

def assess_data_sufficiency(num_examples):
    """Check if dataset is sufficient for joint optimization."""
    if num_examples < JOINT_OPTIMIZATION_REQUIREMENTS["warning_threshold"]:
        return {
            "sufficient": False,
            "recommendation": "Use prompt-only optimization (BootstrapFewShot)",
            "reason": "Insufficient data for fine-tuning"
        }
    elif num_examples < JOINT_OPTIMIZATION_REQUIREMENTS["minimum_examples"]:
        return {
            "sufficient": "marginal",
            "recommendation": "Consider lightweight fine-tuning or prompt-only",
            "reason": "Fine-tuning may overfit"
        }
    elif num_examples < JOINT_OPTIMIZATION_REQUIREMENTS["recommended_examples"]:
        return {
            "sufficient": True,
            "recommendation": "Joint optimization viable, use regularization",
            "reason": "Adequate but not ideal for fine-tuning"
        }
    else:
        return {
            "sufficient": True,
            "recommendation": "Full joint optimization recommended",
            "reason": "Sufficient data for robust optimization"
        }

# Example assessment
assessment = assess_data_sufficiency(75)
print(f"Sufficient: {assessment['sufficient']}")
print(f"Recommendation: {assessment['recommendation']}")
```

### Computational Cost Considerations

```python
def estimate_joint_optimization_cost(
    num_examples,
    model_size_b,
    optimization_strategy
):
    """
    Estimate computational requirements for joint optimization.

    Returns estimated GPU hours and API calls.
    """
    costs = {
        "fine_tuning_only": {
            "gpu_hours": model_size_b * num_examples / 5000,
            "api_calls": 0
        },
        "prompt_only_bootstrap": {
            "gpu_hours": 0,
            "api_calls": num_examples * 10
        },
        "prompt_only_mipro": {
            "gpu_hours": 0,
            "api_calls": num_examples * 25
        },
        "joint_bootstrap": {
            "gpu_hours": model_size_b * num_examples / 5000,
            "api_calls": num_examples * 10
        },
        "joint_mipro": {
            "gpu_hours": model_size_b * num_examples / 5000,
            "api_calls": num_examples * 25
        },
        "copa": {
            "gpu_hours": model_size_b * num_examples / 5000 * 1.2,
            "api_calls": num_examples * 30
        }
    }

    if optimization_strategy not in costs:
        return None

    cost = costs[optimization_strategy]

    # Rough cost estimate (adjust based on your infrastructure)
    estimated_cost = cost["gpu_hours"] * 2.0 + cost["api_calls"] * 0.002

    return {
        "strategy": optimization_strategy,
        "gpu_hours": cost["gpu_hours"],
        "api_calls": cost["api_calls"],
        "estimated_cost_usd": estimated_cost
    }

# Compare strategies
for strategy in ["prompt_only_bootstrap", "joint_bootstrap", "copa"]:
    cost = estimate_joint_optimization_cost(100, 7, strategy)
    print(f"{strategy}: ${cost['estimated_cost_usd']:.2f}")
```

### Scope Limitations and Mitigation

Joint optimization has inherent scope limitations:

| Limitation | Impact | Mitigation Strategy |
|------------|--------|---------------------|
| Domain shift | Fine-tuned model may not generalize | Include diverse training data |
| Prompt brittleness | Optimized prompts may not transfer | Test on held-out domains |
| Computational cost | Multiple optimization runs needed | Use progressive optimization |
| Data requirements | Need 50-100+ examples | Data augmentation techniques |
| Model lock-in | Fine-tuned weights are model-specific | Document and version models |

```python
def mitigate_scope_limitations(trainset, valset):
    """
    Apply mitigation strategies for joint optimization limitations.
    """
    mitigations = []

    # 1. Check domain diversity
    domains = set(getattr(ex, 'domain', 'unknown') for ex in trainset)
    if len(domains) < 3:
        mitigations.append({
            "issue": "Limited domain diversity",
            "action": "Add examples from related domains",
            "severity": "medium"
        })

    # 2. Check data size
    if len(trainset) < 50:
        mitigations.append({
            "issue": "Insufficient training data",
            "action": "Use data augmentation or reduce fine-tuning epochs",
            "severity": "high"
        })

    # 3. Recommend validation strategy
    if len(valset) < len(trainset) * 0.2:
        mitigations.append({
            "issue": "Small validation set",
            "action": "Use k-fold cross-validation",
            "severity": "medium"
        })

    return mitigations
```

## COPA: The Comprehensive Solution

For maximum performance with proper handling of optimization order and synergy, consider using COPA (Combined Optimization and Prompt Adaptation):

```python
from copa_optimizer import COPAOptimizer

# COPA automatically handles:
# 1. Proper optimization order (fine-tune first)
# 2. Synergistic combination of approaches
# 3. Data requirement checks
# 4. Computational budgeting

copa = COPAOptimizer(
    base_model_name="mistralai/Mistral-7B-v0.1",
    metric=your_metric,
    finetune_epochs=3,
    prompt_optimizer="mipro"
)

# Achieves 2-26x improvements on complex tasks
optimized, model = copa.optimize(
    program=YourProgram(),
    trainset=train_examples,
    valset=val_examples
)
```

See [COPA: Combined Fine-Tuning and Prompt Optimization](09-copa-optimizer.md) for complete documentation.

## Key Takeaways

1. **Data Size Matters**: More data enables more sophisticated optimization
2. **Task Complexity Drives Choice**: Complex tasks benefit from MIPRO
3. **Latency vs Accuracy Trade-off**: Consider your specific needs
4. **Progressive Approach Works**: Start simple, iterate to complex
5. **Cost-Benefit Analysis**: Not all optimization justifies the cost
6. **Ensemble Methods**: Can combine strengths of multiple optimizers
7. **Optimization Order**: Always fine-tune first, then apply prompt optimization
8. **Synergy Is Real**: Combined approaches achieve 2-3.5x better than additive
9. **Know Your Limits**: Joint optimization requires 50-100+ examples

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
    ├─ Complex multi-step reasoning?
    │  └─ RPE (evolutionary approach)
    ├─ > 100 examples?
    │  ├─ Context matters?
    │  │  └─ KNNFewShot
    │  └─ Maximum performance?
    │     └─ MIPRO (heavy) or RPE
    └─ Domain-specific & > 1000 examples?
       └─ Consider Fine-Tuning
```

## Next Steps

Now you have a comprehensive understanding of DSPy optimizers. In the exercises, you'll apply these concepts to real-world scenarios and learn to make informed optimization decisions.</think>