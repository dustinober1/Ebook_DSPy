# GEPA: Genetic-Pareto Optimization

## Overview

**GEPA (Genetic-Pareto)** is a cutting-edge DSPy optimizer that combines genetic algorithms with Pareto-based optimization for prompt engineering. Introduced in late 2025, GEPA represents a significant advancement in automated prompt optimization by leveraging natural language reflections and multi-objective optimization.

The key innovation of GEPA is its ability to:
- Optimize multiple conflicting objectives simultaneously
- Use natural language feedback to guide improvements
- Maintain a diverse set of high-quality solutions on the Pareto front
- Integrate seamlessly with DSPy's compilation framework

## Key Concepts

### Genetic-Pareto Algorithm

GEPA merges two powerful optimization paradigms:

1. **Genetic Algorithms**: Evolution-inspired optimization using:
   - Selection: Choosing high-performing prompts
   - Crossover: Combining parts of successful prompts
   - Mutation: Introducing controlled variations
   - Elitism: Preserving the best solutions

2. **Pareto Optimization**: Multi-objective optimization that:
   - Identifies non-dominated solutions
   - Maintains diversity in the solution space
   - Allows trade-offs between competing objectives
   - Produces a set of equally optimal solutions

### Natural Language Reflections

GEPA's distinguishing feature is its use of natural language reflections:
- Prompts are evaluated using qualitative feedback
- Reflections guide the evolutionary process
- Human-like reasoning informs prompt improvements
- Explanations help understand why prompts work

## GEPA in DSPy

### Basic Usage

```python
import dspy
from gepa import GEPAOptimizer

# Define your signature
class RCTRiskAssessment(dspy.Signature):
    """Assess risk of bias in a randomized controlled trial."""
    trial_text = dspy.InputField(desc="Full text of the RCT")
    risk_domain = dspy.InputField(desc="Specific bias domain to assess")
    risk_assessment = dspy.OutputField(desc="Detailed risk assessment")
    confidence_score = dspy.OutputField(desc="Confidence in assessment (0-1)")

# Create your program
program = dspy.ChainOfThought(RCTRiskAssessment)

# Configure GEPA
optimizer = GEPAOptimizer(
    population_size=20,
    generations=10,
    mutation_rate=0.2,
    crossover_rate=0.7,
    objectives=["accuracy", "clarity", "completeness"],
    reflection_model="gpt-4"
)

# Compile with GEPA
compiled = optimizer.compile(
    program=program,
    trainset=training_data,
    valset=validation_data
)
```

### Advanced Configuration

```python
# Configure multiple objectives
optimizer = GEPAOptimizer(
    objectives=[
        {"name": "accuracy", "weight": 0.5, "direction": "maximize"},
        {"name": "efficiency", "weight": 0.3, "direction": "minimize"},
        {"name": "interpretability", "weight": 0.2, "direction": "maximize"}
    ],
    genetic_params={
        "selection_strategy": "tournament",
        "tournament_size": 3,
        "crossover_type": "uniform",
        "mutation_types": ["substitution", "insertion", "deletion"]
    },
    pareto_params={
        "diversity_metric": "euclidean",
        "elitism_count": 5,
        "archive_size": 50
    }
)
```

## The GEPA Algorithm

### 1. Initialization

```python
def initialize_population(signature, base_prompt, population_size):
    """Create initial diverse population of prompts."""
    population = [base_prompt]  # Start with the original

    # Generate variations using different strategies
    for i in range(population_size - 1):
        if i < population_size // 3:
            # Simple variations
            prompt = vary_instructions(base_prompt)
        elif i < 2 * population_size // 3:
            # Domain-specific variations
            prompt = specialize_for_domain(base_prompt, signature)
        else:
            # Random variations
            prompt = random_variation(base_prompt)

        population.append(prompt)

    return population
```

### 2. Evaluation

```python
def evaluate_prompt(prompt, test_cases, objectives):
    """Evaluate a prompt against multiple objectives."""
    results = {}

    # Create temporary program with the prompt
    temp_program = create_program_with_prompt(prompt)

    # Evaluate on test cases
    predictions = []
    for case in test_cases:
        pred = temp_program(**case.inputs)
        predictions.append(pred)

    # Calculate metrics for each objective
    for obj in objectives:
        if obj["name"] == "accuracy":
            results["accuracy"] = calculate_accuracy(predictions, test_cases)
        elif obj["name"] == "efficiency":
            results["efficiency"] = measure_inference_time(predictions)
        elif obj["name"] == "clarity":
            results["clarity"] = assess_clarity(prompt)
        # ... other objectives

    return results
```

### 3. Natural Language Reflection

```python
def generate_reflection(prompt, performance, examples):
    """Generate natural language reflection on prompt performance."""
    reflection_prompt = f"""
    Analyze this prompt's performance:

    Prompt: {prompt}

    Performance: {performance}

    Examples:
    {format_examples(examples)}

    Provide a detailed reflection explaining:
    1. What makes this prompt effective or ineffective?
    2. Which specific components contribute to success/failure?
    3. How could the prompt be improved?
    4. What patterns emerge from the examples?

    Reflection:
    """

    reflection = dspy.Predict(reflection_prompt)
    return reflection.reflection
```

### 4. Genetic Operations

```python
def crossover(parent1, parent2, crossover_type="uniform"):
    """Combine two parent prompts."""
    if crossover_type == "uniform":
        # Exchange sections between parents
        sections1 = split_into_sections(parent1)
        sections2 = split_into_sections(parent2)

        child = []
        for i in range(max(len(sections1), len(sections2))):
            if i < len(sections1) and i < len(sections2):
                if random.random() < 0.5:
                    child.append(sections1[i])
                else:
                    child.append(sections2[i])
            elif i < len(sections1):
                child.append(sections1[i])
            else:
                child.append(sections2[i])

        return join_sections(child)

def mutate(prompt, mutation_rate):
    """Apply mutations to a prompt."""
    mutations = []

    for word in prompt.split():
        if random.random() < mutation_rate:
            mutation_type = random.choice([
                "substitute", "insert", "delete", "reorder"
            ])

            if mutation_type == "substitute":
                word = substitute_synonym(word)
            elif mutation_type == "insert":
                word = word + " " + get_contextual_word(word)
            elif mutation_type == "delete":
                word = ""
            elif mutation_type == "reorder":
                # Will be handled at sentence level
                mutations.append(word)
        else:
            mutations.append(word)

    return " ".join([w for w in mutations if w])
```

### 5. Pareto Front Selection

```python
def select_pareto_front(population, objectives):
    """Select non-dominated solutions from population."""
    pareto_front = []

    for individual in population:
        dominated = False

        for other in population:
            if dominates(other, individual, objectives):
                dominated = True
                break

        if not dominated:
            pareto_front.append(individual)

    # If too many solutions, apply diversity selection
    if len(pareto_front) > max_front_size:
        pareto_front = select_diverse(pareto_front, max_front_size)

    return pareto_front

def dominates(individual1, individual2, objectives):
    """Check if individual1 dominates individual2."""
    better_in_any = False

    for obj in objectives:
        val1 = individual1.performance[obj["name"]]
        val2 = individual2.performance[obj["name"]]

        if obj["direction"] == "maximize":
            if val1 < val2:
                return False
            elif val1 > val2:
                better_in_any = True
        else:  # minimize
            if val1 > val2:
                return False
            elif val1 < val2:
                better_in_any = True

    return better_in_any
```

## Practical Applications

### 1. Multi-Objective Classification

```python
class SentimentAnalysis(dspy.Signature):
    """Analyze sentiment with confidence and explanation."""
    text = dspy.InputField(desc="Text to analyze")
    sentiment = dspy.OutputField(desc="Positive/Negative/Neutral")
    confidence = dspy.OutputField(desc="Confidence score (0-1)")
    explanation = dspy.OutputField(desc="Brief explanation")

# Configure GEPA for multiple objectives
optimizer = GEPAOptimizer(
    objectives=[
        {"name": "accuracy", "direction": "maximize"},
        {"name": "confidence_calibration", "direction": "maximize"},
        {"name": "explanation_quality", "direction": "maximize"},
        {"name": "response_length", "direction": "minimize"}
    ]
)
```

### 2. Trade-off Visualization

```python
import matplotlib.pyplot as plt

def visualize_pareto_front(pareto_solutions):
    """Visualize the Pareto front with trade-offs."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each solution
    for solution in pareto_solutions:
        x = solution.performance["accuracy"]
        y = solution.performance["efficiency"]
        z = solution.performance["interpretability"]

        ax.scatter(x, y, z, s=100, alpha=0.6)
        ax.text(x, y, z, f"  {solution.id[:8]}", fontsize=8)

    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Efficiency (lower is better)")
    ax.set_zlabel("Interpretability")
    ax.set_title("Pareto Front of Optimized Prompts")

    plt.tight_layout()
    plt.show()
```

## GEPA vs. Other Optimizers

| Feature | GEPA | RPE | COPA | MIPRO |
|---------|------|-----|------|-------|
| **Multi-objective** | ✓ Native | ✗ Single | ✗ Single | ✗ Single |
| **Natural Language Feedback** | ✓ Core | ✓ Core | ✗ Limited | ✗ Limited |
| **Solution Diversity** | ✓ Maintained | ✗ Single best | ✗ Single best | ✗ Single best |
| **Evolutionary** | ✓ Genetic | ✓ Evolutionary | ✗ Gradient-based | ✗ Coordinate descent |
| **Pareto Optimization** | ✓ Native | ✗ N/A | ✗ N/A | ✗ N/A |
| **Explainability** | ✓ High | ✓ Medium | ✓ Low | ✓ Low |
| **Compute Cost** | Medium | Low | High | Medium |

## Best Practices

### 1. Objective Definition

```python
# Good: Clear, measurable objectives
objectives = [
    {
        "name": "factual_accuracy",
        "description": "Percentage of facts that are correct",
        "direction": "maximize",
        "weight": 0.4
    },
    {
        "name": "response_length",
        "description": "Average number of tokens",
        "direction": "minimize",
        "weight": 0.2
    }
]

# Bad: Vague objectives
objectives = [
    {"name": "quality", "direction": "maximize"},  # Too vague
    {"name": "speed", "direction": "minimize"}     # Not specific
]
```

### 2. Population Management

```python
# Start with diverse initial population
def create_diverse_population(base_prompt, size):
    strategies = [
        simplify_instructions,
        add_examples,
        specialize_domain,
        add_constraints,
        split_into_steps
    ]

    population = [base_prompt]
    for i in range(size - 1):
        strategy = random.choice(strategies)
        variant = strategy(base_prompt)
        population.append(variant)

    return population
```

### 3. Reflection Quality

```python
# High-quality reflection template
reflection_template = """
Critically analyze this prompt's performance:

**Prompt**: {prompt}

**Performance Metrics**:
{metrics}

**Success Examples**:
{successes}

**Failure Examples**:
{failures}

**Analysis**:
1. Identify specific patterns in successes vs failures
2. Determine which prompt components contribute to each
3. Explain the mechanism behind these effects
4. Suggest precise improvements with rationale

**Structured Reflection**:
{reflection}
"""
```

## Limitations and Considerations

1. **Computational Cost**: Evaluating multiple objectives increases cost
2. **Complexity**: More complex than single-objective optimizers
3. **Objective Balance**: Requires careful weighting of objectives
4. **Evaluation Metric Quality**: Depends on reliable multi-dimensional metrics

## Summary

GEPA represents a significant advancement in prompt optimization by:
- Combining genetic algorithms with Pareto optimization
- Using natural language reflections for intuitive improvements
- Maintaining diverse solutions for different use cases
- Optimizing multiple objectives simultaneously

This makes GEPA particularly valuable for applications where trade-offs between different performance metrics are important, such as in production systems balancing accuracy, efficiency, and interpretability.

## Exercises

1. **Multi-Objective Design**: Identify 3 conflicting objectives for your task and implement them in GEPA.

2. **Pareto Analysis**: Given a set of prompts, manually identify which ones belong to the Pareto front.

3. **Reflection Quality**: Write reflection prompts that would guide improvement for different types of tasks.

4. **Trade-off Visualization**: Create visualizations showing how different prompts balance competing objectives.

5. **Performance Comparison**: Compare GEPA results with single-objective optimizers on your task.