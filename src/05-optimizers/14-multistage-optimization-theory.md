# Multi-stage Optimization Theory

## Learning Objectives

By the end of this section, you will be able to:
- Understand the theoretical foundations of multi-stage language model optimization
- Apply mathematical frameworks for optimizing cascaded language model programs
- Analyze convergence properties and optimization landscapes
- Design optimization strategies that account for inter-stage dependencies
- Evaluate trade-offs in multi-stage optimization approaches

## Introduction

Multi-stage optimization addresses a fundamental challenge in language model programming: how to optimize programs that consist of multiple interconnected modules where each module's output becomes the input for subsequent stages. Unlike single-stage optimization, where we optimize a single prompt or set of demonstrations, multi-stage optimization must consider:

1. **Interdependencies between stages**: The optimal prompt for stage 2 depends on the output characteristics of stage 1
2. **Error propagation**: Mistakes in early stages compound and affect downstream performance
3. **Computational constraints**: Each stage adds computational overhead that must be balanced against performance gains
4. **Optimization complexity**: The parameter space grows exponentially with the number of stages

## Theoretical Foundations

### Formal Problem Definition

Consider a multi-stage language model program with K stages:

```
P(x; θ) = f_K(f_{K-1}(...f_1(x; θ_1)...; θ_{K-1}); θ_K)
```

Where:
- x = input
- θ = {θ_1, θ_2, ..., θ_K} = parameters for all stages
- f_i = i-th stage transformation (language model module)
- P = complete program

The optimization objective is:

```
θ* = argmax_θ E_{(x,y)~D}[M(P(x; θ), y)]
```

Subject to constraints:
- Computational budget: Σ_i C_i(θ_i) ≤ B
- Latency constraints: T(P(x; θ)) ≤ L_max
- Memory constraints: M(P(x; θ)) ≤ M_max

### Optimization Landscape Characteristics

#### Non-convexity and Local Optima
Multi-stage optimization landscapes exhibit:
- Multiple local optima due to discrete prompt parameters
- Plateaus where small parameter changes yield no performance difference
- Rugged terrain with sudden performance cliffs

#### Curse of Dimensionality
The parameter space dimensionality grows as:
```
dim(θ) = Σ_i dim(θ_i)
```

For K stages with average d parameters each:
- Parameter space grows as O(K^d)
- Exhaustive search becomes infeasible beyond K=3

#### Stage-wise Dependencies

Forward Dependency:
```
∂P/∂θ_i = (∂f_K/∂f_{K-1}) × ... × (∂f_{i+1}/∂f_i) × (∂f_i/∂θ_i)
```

This shows how early stage parameters affect the final output through all intermediate transformations.

### Theoretical Frameworks

#### 1. Decomposition Theory

Decomposition breaks multi-stage optimization into stage-wise subproblems:

```
θ_i* = argmax_θ_i E_{z~P_{i-1}}[M_i(f_i(z; θ_i), y_i)]
```

Where P_{i-1} is the distribution of outputs from previous stages.

**Strengths:**
- Reduces dimensionality
- Enables parallel optimization
- Simplifies optimization landscape

**Weaknesses:**
- Ignores cross-stage interactions
- May converge to suboptimal solutions
- Requires accurate modeling of intermediate distributions

#### 2. Coordinate Descent Framework

Sequentially optimize each stage while fixing others:

```python
def coordinate_descent_optimization(program, trainset, num_rounds=10):
    """Optimize multi-stage program using coordinate descent."""

    θ = initialize_parameters(program)

    for round in range(num_rounds):
        for stage in program.stages:
            # Fix all other stages
            for other_stage in program.stages:
                if other_stage != stage:
                    other_stage.freeze()

            # Optimize current stage
            stage_optimizer = create_stage_optimizer(stage)
            θ[stage] = stage_optimizer.optimize(
                stage,
                trainset,
                metric=stage_specific_metric(stage)
            )

            # Unfreeze all stages
            for s in program.stages:
                s.unfreeze()

    return θ
```

**Convergence Properties:**
- Guaranteed to converge to local optimum
- Convergence rate depends on condition number
- Can escape poor local optima through random restarts

#### 3. End-to-End Differentiable Optimization

When using differentiable components:

```
θ_{t+1} = θ_t - α ∇_θ L(P(x; θ), y)
```

Where gradients are computed through all stages using backpropagation.

**Applicability:**
- Works with soft prompts and adapter weights
- Enables gradient-based optimization
- Smooths optimization landscape

### Optimization Algorithms

#### 1. Hierarchical Bayesian Optimization

Treats optimization as hierarchical Bayesian inference:

```python
class HierarchicalBO:
    """Hierarchical Bayesian optimization for multi-stage programs."""

    def __init__(self, num_stages):
        self.num_stages = num_stages
        self.stage_optimizers = [BayesianOptimizer() for _ in range(num_stages)]
        self.global_optimizer = BayesianOptimizer()

    def optimize(self, program, trainset, budget):
        """Hierarchical optimization strategy."""

        # Stage 1: Independent optimization
        stage_params = {}
        for i, stage in enumerate(program.stages):
            stage_params[i] = self.stage_optimizers[i].optimize(
                stage, trainset, budget // (2 * self.num_stages)
            )

        # Stage 2: Coordinated refinement
        def evaluate_full_program(params):
            program.set_parameters(params)
            return evaluate(program, trainset)

        # Use stage-wise optima as initial points
        best_params = self.global_optimizer.optimize(
            evaluate_full_program,
            initial_points=[stage_params],
            budget=budget // 2
        )

        return best_params
```

#### 2. Multi-fidelity Optimization

Optimize using multiple fidelity levels:

```python
def multi_fidelity_optimize(program, trainset):
    """Optimize using progressive fidelity levels."""

    fidelity_levels = [
        {'subset': 0.1, 'max_demos': 1, 'max_length': 50},
        {'subset': 0.3, 'max_demos': 3, 'max_length': 100},
        {'subset': 0.6, 'max_demos': 5, 'max_length': 200},
        {'subset': 1.0, 'max_demos': 8, 'max_length': None}
    ]

    best_params = None
    best_score = -float('inf')

    for level in fidelity_levels:
        # Create low-fidelity evaluation
        subset = random.sample(trainset, int(len(trainset) * level['subset']))

        # Optimize at current fidelity
        params = optimize_at_fidelity(
            program,
            subset,
            max_demos=level['max_demos'],
            max_length=level['max_length']
        )

        # Evaluate on full validation set
        score = evaluate_full(program, params, validation_set)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params
```

#### 3. Evolutionary Multi-stage Optimization

```python
class EvolutionaryMultiStageOptimizer:
    """Evolutionary algorithm for multi-stage optimization."""

    def __init__(self, population_size=50, mutation_rate=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    def optimize(self, program, trainset, generations=100):
        """Evolve multi-stage programs."""

        # Initialize population
        population = self.initialize_population(program)

        for gen in range(generations):
            # Evaluate fitness
            fitness = []
            for individual in population:
                score = evaluate_program(individual, trainset)
                fitness.append(score)

            # Selection
            selected = self.tournament_selection(population, fitness)

            # Crossover and mutation
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child = self.crossover(selected[i], selected[i+1])
                    child = self.mutate(child)
                    offspring.append(child)

            # Replace population
            population = self.replace_population(population, offspring, fitness)

        # Return best individual
        final_fitness = [evaluate_program(p, trainset) for p in population]
        best_idx = np.argmax(final_fitness)
        return population[best_idx]
```

## Convergence Analysis

### Theoretical Guarantees

#### Convergence Conditions

For convergence to optimal solution θ*:
1. **Exploration sufficiency**: Algorithm must explore all relevant regions
2. **Exploitation balance**: Must refine promising regions adequately
3. **Stationarity**: Optimum must be stable point in parameter space

#### Convergence Rates

Different algorithms exhibit different convergence characteristics:

| Algorithm | Convergence Rate | Sample Efficiency | Parallelizability |
|-----------|------------------|-------------------|-------------------|
| Coordinate Descent | O(1/t) | Low | High |
| Bayesian Optimization | O(√(n)) | High | Low |
| Evolutionary | O(log n) | Medium | High |
| Gradient-based | O(1/t²) | Highest | Medium |

### Practical Considerations

#### Early Stopping Strategies

```python
def should_stop_optimization(scores, patience=5, min_delta=0.001):
    """Determine if optimization should stop."""

    if len(scores) < patience + 1:
        return False

    # Check for improvement
    recent_scores = scores[-patience-1:]
    best_recent = max(recent_scores[:-1])
    current = recent_scores[-1]

    # Stop if no significant improvement
    if current - best_recent < min_delta:
        return True

    return False
```

#### Learning Rate Scheduling

For differentiable optimization:

```python
def adaptive_lr_schedule(epoch, initial_lr, warmup_epochs=10):
    """Adaptive learning rate schedule."""

    if epoch < warmup_epochs:
        # Linear warmup
        return initial_lr * (epoch / warmup_epochs)
    else:
        # Cosine decay
        decay_epochs = epoch - warmup_epochs
        total_decay = 100 - warmup_epochs
        return initial_lr * 0.5 * (1 + np.cos(np.pi * decay_epochs / total_decay))
```

## Empirical Analysis

### Benchmarks and Evaluation

#### Standard Multi-stage Tasks

1. **Multi-hop Question Answering**:
   - Stage 1: Query decomposition
   - Stage 2: Information retrieval
   - Stage 3: Answer synthesis
   - Stage 4: Answer verification

2. **Code Generation with Refinement**:
   - Stage 1: Initial code generation
   - Stage 2: Error detection
   - Stage 3: Code refinement
   - Stage 4: Final validation

3. **Complex Reasoning**:
   - Stage 1: Problem understanding
   - Stage 2: Strategy planning
   - Stage 3: Step-by-step solution
   - Stage 4: Solution verification

#### Performance Metrics

Stage-wise metrics:
- Individual stage performance
- Error propagation analysis
- End-to-end accuracy

System metrics:
- Computational cost
- Latency
- Memory usage
- Scalability

### Case Study: HotpotQA Optimization

#### Baseline Performance
- Single-stage: 32.0 F1
- Unoptimized multi-stage: 28.5 F1

#### Optimized Multi-stage Results

| Method | Stage 1 | Stage 2 | Stage 3 | Overall F1 | Improvement |
|--------|---------|---------|---------|------------|-------------|
| Independent Optimization | 65.2 | 58.7 | 61.3 | 42.1 | +10.1 |
| Coordinate Descent | 67.8 | 62.1 | 64.5 | 45.8 | +13.8 |
| Joint Optimization | 70.3 | 65.9 | 68.2 | 49.6 | +17.6 |
| MIPRO (Multi-stage) | 72.1 | 68.4 | 70.8 | 52.3 | +20.3 |

#### Key Insights

1. **Stage interactions matter**: Joint optimization outperforms independent by 7.5 F1
2. **Error propagation critical**: Early stage improvements have outsized impact
3. **Computational trade-offs**: 2x computation for 20% performance gain

## Best Practices

### Design Principles

1. **Start Simple**: Begin with decomposition, progress to joint optimization
2. **Stage-wise Evaluation**: Monitor individual and overall performance
3. **Budget Allocation**: Allocate more optimization budget to critical stages
4. **Error Analysis**: Understand how errors propagate through stages

### Common Pitfalls

1. **Over-optimizing early stages**: Diminishing returns after certain point
2. **Ignoring computational costs**: Theoretical optimum may be impractical
3. **Local optima**: Multiple restarts often necessary
4. **Data leakage**: Validation data must not influence optimization

### Implementation Checklist

- [ ] Define clear stage-wise metrics
- [ ] Set computational budgets and constraints
- [ ] Choose appropriate optimization strategy
- [ ] Implement monitoring and logging
- [ ] Plan for multiple optimization runs
- [ ] Validate on held-out test set
- [ ] Document optimization decisions

## Summary

Multi-stage optimization theory provides a principled approach to optimizing complex language model programs. Key takeaways:

1. **Theoretical foundations** help understand optimization challenges
2. **Multiple frameworks** exist for different scenarios
3. **Convergence guarantees** guide algorithm selection
4. **Empirical validation** is essential for real-world performance
5. **Trade-offs** between performance and computation must be managed

The next sections will build upon this theoretical foundation to explore specific optimization strategies and techniques for multi-stage language model programs.