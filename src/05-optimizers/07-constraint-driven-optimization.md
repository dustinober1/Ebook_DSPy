# Constraint-Driven Optimization

## Prerequisites

- **Previous Section**: [Choosing Optimizers](./06-choosing-optimizers.md) - Understanding optimizer selection
- **Chapter 3**: Assertions Module - Familiarity with assertion concepts
- **Required Knowledge**: Optimization theory, constraint satisfaction
- **Difficulty Level**: Advanced
- **Estimated Reading Time**: 50 minutes

## Learning Objectives

By the end of this section, you will:
- Understand how to optimize DSPy programs with constraints
- Master the integration of assertions with optimization algorithms
- Learn to design constraint-aware optimization objectives
- Build robust systems that maintain quality during optimization
- Apply advanced techniques for constraint handling in large-scale systems

## Introduction to Constraint-Driven Optimization

Constraint-driven optimization extends DSPy's optimization framework to incorporate runtime constraints and validation directly into the optimization process. This ensures that optimized programs not only perform better on metrics but also satisfy critical requirements for correctness, format, and quality.

### Traditional vs Constraint-Driven Optimization

**Traditional Optimization:**
```python
# Only optimizes for metric performance
optimizer = dspy.BootstrapFewShot(metric=answer_f1_score)
optimized_program = optimizer.compile(
    student_program,
    trainset=trainset
)
# May sacrifice quality for metric gains
```

**Constraint-Driven Optimization:**
```python
# Optimizes while maintaining constraints
def constrained_metric(example, pred, trace):
    # First check constraints
    if not validate_format(pred):
        return 0.0  # Penalize constraint violations

    # Then calculate metric
    return answer_f1_score(example, pred)

optimizer = dspy.BootstrapFewShot(
    metric=constrained_metric,
    max_labeled_demos=3,
    max_bootstrapped_demos=3
)
optimized_program = optimizer.compile(
    program_with_assertions,
    trainset=trainset
)
# Optimizes within constraint boundaries
```

## Core Concepts

### 1. Constraint-Aware Metrics

Design metrics that incorporate constraint satisfaction:

```python
def constraint_aware_metric(gold, pred, trace=None):
    """Metric that balances performance with constraint satisfaction."""

    # Base performance score
    base_score = accuracy(gold, pred)

    # Constraint satisfaction weights
    format_weight = 0.3
    quality_weight = 0.2
    accuracy_weight = 0.5

    # Check constraints
    format_score = 1.0 if validate_format(pred) else 0.0
    quality_score = calculate_quality_score(pred)

    # Weighted combination
    total_score = (
        accuracy_weight * base_score +
        format_weight * format_score +
        quality_weight * quality_score
    )

    return total_score

def calculate_quality_score(pred):
    """Calculate overall quality score."""
    score = 1.0

    # Length requirements
    if hasattr(pred, 'output'):
        if len(pred.output) < 50:
            score -= 0.2
        elif len(pred.output) > 500:
            score -= 0.1

    # Structure requirements
    if hasattr(pred, 'sections'):
        if len(pred.sections) < 3:
            score -= 0.2

    return max(0.0, score)
```

### 2. Hard vs Soft Constraints

Differentiate between critical requirements and preferences:

```python
class OptimizationConstraints:
    """Define constraint types and their handling."""

    def __init__(self):
        self.hard_constraints = [
            self.validate_syntax,
            self.validate_required_fields,
            self.validate_output_type
        ]

        self.soft_constraints = [
            self.validate_length,
            self.validate_style,
            self.validate_completeness
        ]

    def validate_hard_constraints(self, pred):
        """Check critical constraints that must pass."""
        for constraint in self.hard_constraints:
            try:
                if not constraint(pred):
                    return False, f"Failed: {constraint.__name__}"
            except Exception as e:
                return False, f"Error in {constraint.__name__}: {e}"
        return True, None

    def validate_soft_constraints(self, pred):
        """Check preference constraints."""
        score = 1.0
        for constraint in self.soft_constraints:
            try:
                result = constraint(pred)
                score *= result if isinstance(result, float) else 1.0
            except:
                score *= 0.9  # Small penalty for errors
        return score
```

### 3. Progressive Constraint Enforcement

Gradually enforce stricter constraints during optimization:

```python
class ProgressiveOptimizer:
    """Optimizer with progressively stricter constraints."""

    def __init__(self, base_optimizer):
        self.optimizer = base_optimizer
        self.constraint_levels = [
            [],  # Level 0: No constraints
            [validate_format],  # Level 1: Basic format
            [validate_format, validate_length],  # Level 2: Format + length
            [validate_format, validate_length, validate_quality]  # Level 3: All
        ]

    def compile_with_progression(self, program, trainset):
        """Compile with progressively stricter constraints."""

        best_program = program
        best_score = 0.0

        for level, constraints in enumerate(self.constraint_levels):
            print(f"Optimization level {level}: {len(constraints)} constraints")

            # Create level-specific metric
            def level_metric(gold, pred, trace):
                # Check current level constraints
                for constraint in constraints:
                    if not constraint(pred):
                        return 0.0

                # Evaluate on validation set
                return evaluate_with_constraints(best_program, valset)

            # Compile at this level
            current_program = self.optimizer.compile(
                best_program,
                trainset=trainset,
                metric=level_metric
            )

            # Evaluate
            score = evaluate_with_constraints(current_program, valset)

            if score > best_score:
                best_program = current_program
                best_score = score
                print(f"  Improvement: {score:.3f}")
            else:
                print(f"  No improvement, keeping previous best")

        return best_program
```

## Optimization Strategies

### 1. Constraint-Guided Example Selection

Select training examples based on constraint satisfaction:

```python
class ConstraintGuidedOptimizer(dspy.BootstrapFewShot):
    """Optimizer that selects examples based on constraints."""

    def __init__(self, constraint_validator, **kwargs):
        super().__init__(**kwargs)
        self.constraint_validator = constraint_validator

    def generate_bootstrapped_demos(self, program, trainset):
        """Generate examples that satisfy constraints."""
        valid_examples = []

        # Filter training examples
        for example in trainset:
            # Generate prediction
            pred = program(**example.inputs())

            # Check constraints
            if self.constraint_validator(example, pred):
                valid_examples.append((example, pred))

        # Select diverse valid examples
        selected = self.select_diverse_examples(valid_examples)

        return selected

    def select_diverse_examples(self, examples, max_examples=5):
        """Select diverse examples from valid ones."""
        if len(examples) <= max_examples:
            return examples

        # Simple diversity: use different output lengths
        examples.sort(key=lambda x: len(x[1].output))

        # Select evenly spaced examples
        step = len(examples) // max_examples
        selected = [examples[i * step] for i in range(max_examples)]

        return selected
```

### 2. Multi-Objective Optimization

Optimize for multiple objectives simultaneously:

```python
class MultiObjectiveOptimizer:
    """Optimize for multiple objectives with constraints."""

    def __init__(self, objectives):
        self.objectives = objectives  # List of (name, weight, metric_func)

    def evaluate_program(self, program, testset):
        """Evaluate program on all objectives."""
        scores = {}

        for name, weight, metric_func in self.objectives:
            score = 0.0
            for example in testset:
                pred = program(**example.inputs())
                score += metric_func(example, pred)
            scores[name] = score / len(testset)

        # Calculate weighted sum
        total_score = sum(
            weight * scores[name]
            for name, weight, _ in self.objectives
        )

        return total_score, scores

    def optimize(self, program, trainset, valset, iterations=10):
        """Multi-objective optimization loop."""
        best_program = program
        best_score, best_scores = self.evaluate_program(program, valset)

        for i in range(iterations):
            # Create variation
            candidate = self.create_variation(best_program, trainset)

            # Evaluate
            score, scores = self.evaluate_program(candidate, valset)

            # Track best
            if score > best_score:
                best_program = candidate
                best_score = score
                best_scores = scores

                print(f"Iteration {i}: New best score {score:.3f}")
                for name, s in scores.items():
                    print(f"  {name}: {s:.3f}")

        return best_program
```

### 3. Constraint-Weighted Loss

Incorporate constraints directly into optimization loss:

```python
def constraint_weighted_loss(gold, pred, trace=None):
    """Loss function that includes constraint penalties."""

    # Base task loss
    task_loss = task_specific_loss(gold, pred)

    # Constraint penalties
    constraint_penalties = []

    # Format constraint
    if not validate_format(pred):
        constraint_penalties.append(1.0)
    else:
        constraint_penalties.append(0.0)

    # Quality constraint
    quality_score = calculate_quality(pred)
    constraint_penalties.append(1.0 - quality_score)

    # Length constraint
    length_violation = abs(pred.length - target_length) / target_length
    constraint_penalties.append(min(length_violation, 1.0))

    # Weighted combination
    constraint_loss = sum(constraint_penalties) / len(constraint_penalties)

    # Total loss with constraint weight
    total_loss = task_loss + 0.5 * constraint_loss

    return total_loss
```

## Advanced Techniques

### 1. Constraint-Aware Prompt Optimization

Optimize prompts while maintaining constraints:

```python
class ConstraintAwarePromptOptimizer:
    """Optimize prompts with constraint awareness."""

    def __init__(self, base_optimizer, constraints):
        self.base_optimizer = base_optimizer
        self.constraints = constraints

    def optimize_prompt(self, signature, trainset, initial_prompt=None):
        """Find optimal prompt under constraints."""

        if initial_prompt is None:
            initial_prompt = signature.instructions

        best_prompt = initial_prompt
        best_score = self.evaluate_prompt(initial_prompt, signature, trainset)

        # Generate prompt variations
        variations = self.generate_prompt_variations(initial_prompt)

        for variation in variations:
            # Check if variation satisfies constraints
            if self.prompt_satisfies_constraints(variation):
                # Evaluate
                score = self.evaluate_prompt(variation, signature, trainset)

                if score > best_score:
                    best_prompt = variation
                    best_score = score

        return best_prompt

    def prompt_satisfies_constraints(self, prompt):
        """Check if prompt meets constraints."""
        # Length constraint
        if len(prompt) > 500:
            return False

        # Must include constraint instructions
        required_phrases = ['format', 'ensure', 'must', 'required']
        if not any(phrase in prompt.lower() for phrase in required_phrases):
            return False

        return True

    def generate_prompt_variations(self, base_prompt):
        """Generate prompt variations while preserving constraints."""
        variations = []

        # Add constraint emphasis
        variation1 = base_prompt + "\n\nConstraints: Ensure all outputs are valid JSON."
        variations.append(variation1)

        # Add example format
        variation2 = base_prompt + "\n\nExample output format:\n{\n  \"field\": \"value\"\n}"
        variations.append(variation2)

        # Add validation reminder
        variation3 = base_prompt + "\n\nRemember to double-check your output for correctness."
        variations.append(variation3)

        return variations
```

### 2. Dynamic Constraint Adjustment

Adjust constraints based on optimization progress:

```python
class DynamicConstraintOptimizer:
    """Optimizer that adjusts constraints during training."""

    def __init__(self, initial_constraints):
        self.constraints = initial_constraints
        self.constraint_history = []

    def adjust_constraints(self, optimization_metrics):
        """Adjust constraints based on optimization performance."""

        # If constraint violations are high, relax constraints
        if optimization_metrics['violation_rate'] > 0.3:
            self.relax_constraints()

        # If performance is good, tighten constraints
        elif optimization_metrics['accuracy'] > 0.9:
            self.tighten_constraints()

        # Record adjustment
        self.constraint_history.append(self.constraints.copy())

    def relax_constraints(self):
        """Make constraints less strict."""
        # Increase length limits
        for constraint in self.constraints:
            if 'min_length' in constraint:
                constraint['min_length'] *= 0.8
            if 'max_length' in constraint:
                constraint['max_length'] *= 1.2

    def tighten_constraints(self):
        """Make constraints more strict."""
        # Decrease tolerance
        for constraint in self.constraints:
            if 'tolerance' in constraint:
                constraint['tolerance'] *= 0.9
```

### 3. Constraint Transfer Learning

Transfer constraints between related tasks:

```python
class ConstraintTransfer:
    """Transfer constraints between similar tasks."""

    def __init__(self):
        self.constraint_patterns = {}

    def learn_constraints(self, task_name, examples):
        """Learn common constraint patterns from examples."""
        patterns = {
            'format_patterns': self.extract_format_patterns(examples),
            'length_patterns': self.extract_length_patterns(examples),
            'structure_patterns': self.extract_structure_patterns(examples)
        }
        self.constraint_patterns[task_name] = patterns

    def transfer_constraints(self, source_task, target_task, examples):
        """Transfer constraints from source to target task."""
        if source_task not in self.constraint_patterns:
            return []

        source_patterns = self.constraint_patterns[source_task]
        transferred_constraints = []

        # Adapt format constraints
        for pattern in source_patterns['format_patterns']:
            if self.is_applicable(pattern, examples):
                adapted = self.adapt_constraint(pattern, target_task)
                transferred_constraints.append(adapted)

        return transferred_constraints

    def is_applicable(self, pattern, examples):
        """Check if constraint pattern applies to new task."""
        # Simple heuristic: check if pattern appears in some examples
        matches = sum(1 for ex in examples if pattern in str(ex))
        return matches / len(examples) > 0.1
```

## Practical Applications

### 1. RAG System Optimization

Optimize retrieval-augmented generation with constraints:

```python
class ConstrainedRAGOptimizer:
    """Optimize RAG systems with quality constraints."""

    def __init__(self, rag_program):
        self.rag_program = rag_program

    def optimize_with_constraints(self, trainset, valset):
        """Optimize RAG while maintaining answer quality."""

        # Define constraints
        constraints = {
            'min_evidence': 2,  # Must cite at least 2 sources
            'max_hallucination': 0.1,  # <10% hallucinated content
            'min_confidence': 0.7,  # Confidence threshold
            'max_length': 500  # Answer length limit
        }

        # Constraint-aware metric
        def rag_metric(gold, pred, trace=None):
            # Base accuracy
            accuracy = calculate_f1(gold.answer, pred.answer)

            # Constraint satisfaction
            constraint_score = 0.0

            # Check citations
            if hasattr(pred, 'citations') and len(pred.citations) >= constraints['min_evidence']:
                constraint_score += 0.25

            # Check confidence
            if hasattr(pred, 'confidence') and pred.confidence >= constraints['min_confidence']:
                constraint_score += 0.25

            # Check length
            if len(pred.answer) <= constraints['max_length']:
                constraint_score += 0.25

            # Check hallucination (using external model)
            hallucination_score = check_hallucination(pred.answer, pred.context)
            if hallucination_score >= (1 - constraints['max_hallucination']):
                constraint_score += 0.25

            # Combine scores
            return 0.6 * accuracy + 0.4 * constraint_score

        # Optimize
        optimizer = dspy.BootstrapFewShot(
            metric=rag_metric,
            max_labeled_demos=3,
            max_bootstrapped_demos=5
        )

        optimized_program = optimizer.compile(
            self.rag_program,
            trainset=trainset
        )

        return optimized_program
```

### 2. Code Generation with Constraints

Optimize code generation while maintaining correctness:

```python
class CodeConstraintOptimizer:
    """Optimize code generation with correctness constraints."""

    def __init__(self, code_generator):
        self.generator = code_generator

    def optimize_with_tests(self, trainset, test_cases):
        """Optimize while ensuring code passes tests."""

        def code_metric(gold, pred, trace=None):
            # Check if code is syntactically valid
            try:
                compile(pred.code, '<string>', 'exec')
                syntax_score = 1.0
            except:
                return 0.0  # Zero score for syntax errors

            # Run test cases
            test_score = 0.0
            passed = 0
            total = len(test_cases.get(gold.problem, []))

            for test in test_cases.get(gold.problem, []):
                try:
                    exec_globals = {}
                    exec(pred.code, exec_globals)

                    # Check if solution function exists
                    if 'solution' in exec_globals:
                        result = exec_globals['solution'](*test['input'])
                        if result == test['expected']:
                            passed += 1
                except:
                    pass  # Test failed

            test_score = passed / total if total > 0 else 0.0

            # Combine with style score
            style_score = self.check_code_style(pred.code)

            # Weighted combination
            return 0.4 * test_score + 0.3 * syntax_score + 0.3 * style_score

        # Optimize
        optimizer = dspy.MIPROv2(
            metric=code_metric,
            num_candidates=5,
            init_temperature=1.0
        )

        optimized = optimizer.compile(
            self.generator,
            trainset=trainset
        )

        return optimized
```

## Monitoring and Analysis

### 1. Constraint Violation Analysis

Track and analyze constraint violations during optimization:

```python
class ConstraintAnalyzer:
    """Analyze constraint violations in optimization."""

    def __init__(self):
        self.violations = []
        self.metrics = {
            'violation_rates': {},
            'common_violations': {},
            'optimization_progress': []
        }

    def record_violation(self, constraint_name, details):
        """Record a constraint violation."""
        self.violations.append({
            'constraint': constraint_name,
            'details': details,
            'timestamp': datetime.now()
        })

    def analyze_violations(self):
        """Analyze patterns in violations."""
        from collections import Counter

        # Count violations by constraint
        violation_counts = Counter(
            v['constraint'] for v in self.violations
        )

        # Calculate rates
        total = len(self.violations)
        for constraint, count in violation_counts.items():
            self.metrics['violation_rates'][constraint] = count / total

        # Most common violations
        self.metrics['common_violations'] = violation_counts.most_common(5)

        return self.metrics

    def generate_report(self):
        """Generate violation analysis report."""
        report = "Constraint Violation Analysis\n"
        report += "=" * 40 + "\n\n"

        # Violation rates
        report += "Violation Rates:\n"
        for constraint, rate in self.metrics['violation_rates'].items():
            report += f"  {constraint}: {rate:.1%}\n"

        # Common violations
        report += "\nMost Common Violations:\n"
        for constraint, count in self.metrics['common_violations']:
            report += f"  {constraint}: {count} occurrences\n"

        # Recommendations
        report += "\nRecommendations:\n"
        top_violation = self.metrics['common_violations'][0][0]
        if self.metrics['violation_rates'][top_violation] > 0.3:
            report += f"  - Consider relaxing {top_violation} constraint\n"
            report += f"  - Improve training examples for {top_violation}\n"

        return report
```

### 2. Optimization Progress Tracking

Track optimization progress with constraint awareness:

```python
class OptimizationTracker:
    """Track optimization progress with metrics."""

    def __init__(self):
        self.epochs = []
        self.current_epoch = 0

    def start_epoch(self):
        """Start a new optimization epoch."""
        self.current_epoch += 1
        self.epochs.append({
            'epoch': self.current_epoch,
            'metrics': {},
            'constraints': {},
            'improvements': []
        })

    def record_metrics(self, metrics):
        """Record optimization metrics."""
        self.epochs[-1]['metrics'].update(metrics)

    def record_constraints(self, constraint_stats):
        """Record constraint statistics."""
        self.epochs[-1]['constraints'].update(constraint_stats)

    def record_improvement(self, improvement_type, details):
        """Record an improvement."""
        self.epochs[-1]['improvements'].append({
            'type': improvement_type,
            'details': details
        })

    def plot_progress(self):
        """Plot optimization progress."""
        import matplotlib.pyplot as plt

        epochs = [e['epoch'] for e in self.epochs]
        scores = [e['metrics'].get('score', 0) for e in self.epochs]
        violations = [e['constraints'].get('violation_rate', 0)
                      for e in self.epochs]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Score progression
        ax1.plot(epochs, scores, 'b-', label='Optimization Score')
        ax1.set_ylabel('Score')
        ax1.set_title('Optimization Progress')
        ax1.legend()
        ax1.grid(True)

        # Violation rate
        ax2.plot(epochs, violations, 'r-', label='Constraint Violation Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Violation Rate')
        ax2.set_title('Constraint Satisfaction')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
```

## Summary

Constraint-driven optimization provides:

- **Balanced optimization** that considers both metrics and constraints
- **Flexible constraint handling** with hard and soft requirements
- **Progressive optimization** with gradually stricter requirements
- **Multi-objective support** for complex optimization scenarios
- **Monitoring and analysis** tools for understanding optimization behavior

### Key Takeaways

1. **Design constraint-aware metrics** that balance performance and requirements
2. **Use progressive enforcement** to guide optimization effectively
3. **Monitor violations** to understand and address optimization challenges
4. **Transfer constraints** between related tasks to speed up optimization
5. **Analyze results** comprehensively with both metric and constraint perspectives

## Next Steps

- [Self-Refining Pipelines](../07-advanced-topics/07-self-refining-pipelines.md) - Advanced constraint patterns
- [Assertion-Driven Applications](../08-case-studies/06-assertion-driven-applications.md) - Real implementations
- [Practical Examples](../../examples/chapter05/) - See optimization in action
- [Exercises](./07-exercises.md) - Practice constraint techniques

## Further Reading

- [Multi-Objective Optimization](https://en.wikipedia.org/wiki/Multi-objective_optimization) - Theoretical foundation
- [Constraint Satisfaction](https://en.wikipedia.org/wiki/Constraint_satisfaction) - Problem-solving paradigm
- [Bayesian Optimization](https://arxiv.org/abs/1206.2944) - Advanced optimization techniques