# Reflective Prompt Evolution (RPE): Evolutionary Optimization Without Gradients

## Introduction

Reflective Prompt Evolution (RPE) is a novel optimizer that brings evolutionary computation techniques to prompt optimization. Unlike gradient-based or heuristic-based methods, RPE uses a population-based approach with mutation and selection to evolve better prompts through self-reflection, making it particularly effective for complex reasoning tasks where traditional optimization methods may struggle.

## What Makes RPE Special?

### Evolutionary Approach
1. **Population-Based**: Maintains multiple candidate prompts simultaneously
2. **Self-Reflection**: Uses LM to critique and improve prompts
3. **Mutation Operations**: Applies structured mutations to evolve prompts
4. **Selection Pressure**: Keeps only the best-performing variants

### Key Innovation: Reflection-Guided Evolution
RPE's core insight is that language models can effectively critique their own prompts and suggest improvements. This self-reflection capability guides the evolutionary process, making mutations more intelligent than random perturbations.

## Basic RPE Usage

### Simple Example

```python
import dspy
from dspy.teleprompter import ReflectivePromptEvolution

# 1. Define your program
class ComplexReasoning(dspy.Module):
    def __init__(self):
        super().__init__()
        self.reason = dspy.ChainOfThought(
            "context, question -> reasoning, answer"
        )

    def forward(self, context, question):
        result = self.reason(context=context, question=question)
        return dspy.Prediction(
            answer=result.answer,
            reasoning=result.reasoning
        )

# 2. Define evaluation metric
def reasoning_accuracy(example, pred, trace=None):
    """Evaluate both answer correctness and reasoning quality."""
    answer_correct = example.answer.lower() == pred.answer.lower()
    reasoning_quality = evaluate_reasoning(pred.reasoning)
    return 0.7 * answer_correct + 0.3 * reasoning_quality

# 3. Prepare data
trainset = [
    dspy.Example(
        context="The company reported Q3 earnings...",
        question="What was the revenue growth?",
        answer="15% year-over-year"
    ),
    # ... more examples requiring complex reasoning
]

# 4. Create RPE optimizer
optimizer = ReflectivePromptEvolution(
    metric=reasoning_accuracy,
    population_size=10,        # Maintain 10 candidate prompts
    generations=5,             # Evolve for 5 generations
    mutation_rate=0.3,         # 30% chance of mutation per generation
    selection_pressure=0.5     # Keep top 50% each generation
)

# 5. Compile the program
compiled_reasoning = optimizer.compile(
    ComplexReasoning(),
    trainset=trainset,
    valset=valset  # Validation set for fitness evaluation
)

# 6. Use the evolved program
result = compiled_reasoning(
    context="In the latest shareholder meeting...",
    question="What are the strategic priorities?"
)
print(f"Answer: {result.answer}")
print(f"Reasoning: {result.reasoning}")
```

## The RPE Algorithm

### Three-Step Evolution Process

```python
class RPEEvolution:
    def __init__(self, population_size, generations):
        self.population_size = population_size
        self.generations = generations
        self.population = []

    def evolve(self, initial_program, trainset, metric):
        """Main evolution loop."""
        # Initialize population with variations
        self.population = self.initialize_population(initial_program)

        for generation in range(self.generations):
            print(f"Generation {generation + 1}/{self.generations}")

            # Step 1: Selection - evaluate fitness
            fitness_scores = self.evaluate_population(
                self.population, trainset, metric
            )

            # Step 2: Reflection - generate critiques
            reflections = self.generate_reflections(
                self.population, fitness_scores
            )

            # Step 3: Mutation - create offspring
            offspring = self.mutate_population(
                self.population, reflections
            )

            # Selection for next generation
            self.population = self.select_survivors(
                self.population + offspring,
                fitness_scores
            )

        # Return best individual
        best_idx = max(range(len(self.population)),
                      key=lambda i: fitness_scores[i])
        return self.population[best_idx]
```

### Selection Mechanisms

```python
def tournament_selection(population, fitness_scores, tournament_size=3):
    """Select individuals using tournament selection."""
    selected = []
    for _ in range(len(population) // 2):  # Select half
        # Random tournament participants
        tournament_idx = random.sample(
            range(len(population)),
            min(tournament_size, len(population))
        )

        # Select winner of tournament
        winner_idx = max(tournament_idx,
                        key=lambda i: fitness_scores[i])
        selected.append(population[winner_idx])

    return selected

def truncation_selection(population, fitness_scores, keep_ratio=0.5):
    """Select top-performing individuals."""
    # Sort by fitness
    sorted_pop = sorted(
        zip(population, fitness_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Keep top individuals
    num_keep = int(len(population) * keep_ratio)
    return [ind for ind, _ in sorted_pop[:num_keep]]
```

## Self-Reflection Mechanisms

### Generating Reflective Critiques

```python
def generate_reflection(program, performance_data):
    """Generate a reflective critique of the program."""
    reflection_prompt = f"""
    Analyze this prompt optimization task:

    Current Program: {program}
    Performance Data: {performance_data}

    Reflect on:
    1. What aspects of the prompt are causing errors?
    2. Which instructions are unclear or ambiguous?
    3. What type of reasoning is missing?
    4. How could the prompt structure be improved?

    Provide specific, actionable suggestions for improvement.
    """

    reflection = dspy.Predict(
        "program, performance -> critique, suggestions"
    )

    result = reflection(
        program=str(program),
        performance=performance_data
    )

    return {
        'critique': result.critique,
        'suggestions': result.suggestions.split('\n')
    }

def structured_reflection(program, examples, predictions):
    """Generate structured reflection focusing on specific aspects."""
    error_analysis = analyze_errors(examples, predictions)

    reflection_template = """
    Prompt Reflection Report:

    1. ERROR PATTERNS:
    - Most common error type: {error_type}
    - Frequency: {error_freq}%
    - Typical scenario: {error_scenario}

    2. PROMPT WEAKNESSES:
    - Missing instructions: {missing_instructions}
    - Ambiguous terms: {ambiguous_terms}
    - Insufficient context: {context_issues}

    3. IMPROVEMENT RECOMMENDATIONS:
    - Add specific guidance for: {additions}
    - Clarify ambiguous terms: {clarifications}
    - Restructure for better flow: {restructuring}
    """

    return reflection_prompt.format(**error_analysis)
```

### Using Reflections to Guide Mutations

```python
def reflection_guided_mutation(program, reflection):
    """Apply mutations based on reflection insights."""
    mutations = []

    # Parse reflection for specific suggestions
    for suggestion in reflection['suggestions']:
        if "add instruction" in suggestion.lower():
            mutations.append(('add_instruction', suggestion))
        elif "clarify" in suggestion.lower():
            mutations.append(('clarify_term', suggestion))
        elif "restructure" in suggestion.lower():
            mutations.append(('restructure', suggestion))

    # Apply mutations with probabilities
    mutated_program = program.copy()
    for mutation_type, mutation_detail in mutations:
        if random.random() < 0.3:  # 30% chance per suggestion
            mutated_program = apply_mutation(
                mutated_program,
                mutation_type,
                mutation_detail
            )

    return mutated_program
```

## Mutation Strategies

### Basic Mutation Operations

```python
class PromptMutator:
    def __init__(self, mutation_rate=0.3):
        self.mutation_rate = mutation_rate
        self.mutation_operators = [
            self.swap_instructions,
            self.reverse_order,
            self.random_replace,
            self.add_instruction,
            self.remove_instruction
        ]

    def swap_instructions(self, prompt):
        """Swap two instruction segments."""
        instructions = prompt.split('\n')
        if len(instructions) >= 2:
            i, j = random.sample(range(len(instructions)), 2)
            instructions[i], instructions[j] = instructions[j], instructions[i]
            return '\n'.join(instructions)
        return prompt

    def reverse_order(self, prompt):
        """Reverse the order of instructions."""
        instructions = prompt.split('\n')
        if len(instructions) > 1:
            return '\n'.join(reversed(instructions))
        return prompt

    def random_replace(self, prompt):
        """Replace a random instruction with a variation."""
        instructions = prompt.split('\n')
        if instructions:
            idx = random.randint(0, len(instructions) - 1)
            variations = [
                "Consider carefully:",
                "Think step by step:",
                "Analyze the following:",
                "Evaluate systematically:",
                "Examine in detail:"
            ]
            instructions[idx] = random.choice(variations)
            return '\n'.join(instructions)
        return prompt

    def add_instruction(self, prompt):
        """Add a new instruction at a random position."""
        new_instructions = [
            "Double-check your reasoning.",
            "Consider alternative perspectives.",
            "Ensure logical consistency.",
            "Verify all assumptions.",
            "Provide explicit justification."
        ]

        instructions = prompt.split('\n')
        insert_pos = random.randint(0, len(instructions))
        instructions.insert(insert_pos, random.choice(new_instructions))
        return '\n'.join(instructions)

    def remove_instruction(self, prompt):
        """Remove a random instruction."""
        instructions = prompt.split('\n')
        if len(instructions) > 1:
            idx = random.randint(0, len(instructions) - 1)
            instructions.pop(idx)
            return '\n'.join(instructions)
        return prompt
```

### Label Mutation for Demonstrations

```python
def mutate_labels(program, mutation_strength=0.2):
    """Mutate the labels in few-shot examples."""
    mutated_program = program.copy()

    for example in mutated_program.demos:
        # Get mutable fields
        for field_name, field_value in example.items():
            if isinstance(field_value, str):
                # Decide whether to mutate this field
                if random.random() < mutation_strength:
                    mutated_value = apply_label_mutation(field_value)
                    example[field_name] = mutated_value

    return mutated_program

def apply_label_mutation(text):
    """Apply specific mutation to a text label."""
    mutations = [
        lambda t: t.capitalize(),
        lambda t: t.lower(),
        lambda t: add_qualifier(t),
        lambda t: remove_qualifier(t),
        lambda t: rephrase(t)
    ]

    mutation_func = random.choice(mutations)
    return mutation_func(text)

def add_qualifier(text):
    """Add a qualifying phrase."""
    qualifiers = [
        "Clearly, ", "Obviously, ", "Typically, ",
        "Generally, ", "Usually, ", "Often "
    ]
    return random.choice(qualifiers) + text

def rephrase(text):
    """Simple rephrasing using synonyms."""
    # Simplified example - in practice, use LM for better rephrasing
    replacements = {
        "good": "excellent",
        "bad": "poor",
        "big": "large",
        "small": "tiny",
        "fast": "quick"
    }

    words = text.split()
    for i, word in enumerate(words):
        lower_word = word.lower().strip('.,!?')
        if lower_word in replacements:
            words[i] = word.replace(lower_word, replacements[lower_word])

    return ' '.join(words)
```

## Diversity Maintenance

### Cosine Similarity Thresholds

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_diversity(population):
    """Calculate diversity metrics for the population."""
    # Convert programs to embeddings
    embeddings = []
    for program in population:
        embedding = get_program_embedding(program)
        embeddings.append(embedding)

    embeddings = np.array(embeddings)

    # Calculate pairwise similarities
    similarities = cosine_similarity(embeddings)

    # Diversity metrics
    avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
    min_similarity = np.min(similarities[similarities > 0])
    max_similarity = np.max(similarities)

    return {
        'average_similarity': avg_similarity,
        'min_similarity': min_similarity,
        'max_similarity': max_similarity,
        'diversity_score': 1 - avg_similarity  # Higher = more diverse
    }

def enforce_diversity_constraint(population, threshold=0.9):
    """Remove programs that are too similar to others."""
    if len(population) <= 1:
        return population

    # Calculate all pairwise similarities
    embeddings = [get_program_embedding(p) for p in population]
    similarities = cosine_similarity(embeddings)

    # Find and remove similar programs
    to_remove = set()
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            if similarities[i][j] > threshold:
                # Remove the one with lower fitness (assume sorted)
                to_remove.add(j)

    # Return filtered population
    return [p for i, p in enumerate(population) if i not in to_remove]

def diversity_mutation(population, diversity_score, target_diversity=0.7):
    """Apply mutations to increase diversity if needed."""
    if diversity_score < target_diversity:
        # Population lacks diversity, apply more mutations
        mutated = []
        for program in population:
            if random.random() < 0.5:  # 50% chance
                mutated_program = apply_exploratory_mutation(program)
                mutated.append(mutated_program)
            else:
                mutated.append(program)
        return mutated
    return population

def apply_exploratory_mutation(program):
    """Apply more exploratory mutations for diversity."""
    mutator = PromptMutator(mutation_rate=0.5)  # Higher rate
    return mutator.mutate(program)
```

### Novelty-Based Selection

```python
def novelty_based_selection(population, fitness_scores, novelty_weight=0.3):
    """Select based on combination of fitness and novelty."""
    # Calculate novelty scores
    novelty_scores = calculate_novelty_scores(population)

    # Combine fitness and novelty
    combined_scores = []
    for i in range(len(population)):
        combined = (1 - novelty_weight) * fitness_scores[i] + \
                  novelty_weight * novelty_scores[i]
        combined_scores.append(combined)

    # Select based on combined scores
    selected_indices = sorted(
        range(len(population)),
        key=lambda i: combined_scores[i],
        reverse=True
    )[:len(population) // 2]

    return [population[i] for i in selected_indices]

def calculate_novelty_scores(population):
    """Calculate novelty score for each program."""
    embeddings = [get_program_embedding(p) for p in population]
    novelty_scores = []

    for i, embedding in enumerate(embeddings):
        # Average distance to all others
        distances = []
        for j, other_embedding in enumerate(embeddings):
            if i != j:
                dist = np.linalg.norm(embedding - other_embedding)
                distances.append(dist)

        novelty = np.mean(distances) if distances else 0
        novelty_scores.append(novelty)

    # Normalize to [0, 1]
    max_novelty = max(novelty_scores) if novelty_scores else 1
    return [n / max_novelty for n in novelty_scores]
```

## Integration Examples

### RPE with ChainOfThought

```python
class RPEChainOfThought(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(
            "question -> reasoning, answer"
        )

    def forward(self, question):
        return self.cot(question=question)

# Optimize Chain of Thought with RPE
optimizer = ReflectivePromptEvolution(
    metric=exact_match,
    population_size=8,
    generations=4,
    mutation_rate=0.4
)

optimized_cot = optimizer.compile(
    RPEChainOfThought(),
    trainset=math_problems,
    valset=math_problems_val
)

# The evolved Chain of Thought prompt might include:
# - Specific math problem-solving instructions
# - Step-by-step reasoning requirements
# - Error-checking procedures
# - Domain-specific guidance
```

### RPE for Complex Multi-hop Reasoning

```python
class MultiHopReasoning(dspy.Module):
    def __init__(self):
        super().__init__()
        self.hop1 = dspy.ChainOfThought("question -> intermediate_1")
        self.hop2 = dspy.ChainOfThought("question, intermediate_1 -> intermediate_2")
        self.hop3 = dspy.ChainOfThought("question, intermediate_1, intermediate_2 -> answer")

    def forward(self, question):
        result1 = self.hop1(question=question)
        result2 = self.hop2(question=question, intermediate_1=result1.intermediate_1)
        result3 = self.hop3(
            question=question,
            intermediate_1=result1.intermediate_1,
            intermediate_2=result2.intermediate_2
        )
        return dspy.Prediction(
            answer=result3.answer,
            reasoning_chain=[
                result1.reasoning,
                result2.reasoning,
                result3.reasoning
            ]
        )

# RPE optimization for multi-hop reasoning
optimizer = ReflectivePromptEvolution(
    metric=multi_hop_accuracy,
    population_size=12,
    generations=6,
    selection_pressure=0.4,  # More selective
    diversity_weight=0.4     # Emphasize diverse approaches
)

optimized_multihop = optimizer.compile(
    MultiHopReasoning(),
    trainset=complex_qa_pairs,
    valset=complex_qa_val
)
```

### Comparative Performance Analysis

```python
def compare_optimizers(task, trainset, testset):
    """Compare RPE with other optimizers on the same task."""
    results = {}

    # 1. Baseline
    baseline = task()
    results['baseline'] = evaluate(baseline, testset)

    # 2. BootstrapFewShot
    bootstrap_optimizer = BootstrapFewShot(metric=exact_match)
    bootstrap_compiled = bootstrap_optimizer.compile(task(), trainset=trainset)
    results['bootstrap'] = evaluate(bootstrap_compiled, testset)

    # 3. MIPRO
    mipro_optimizer = MIPRO(metric=exact_match, num_candidates=10)
    mipro_compiled = mipro_optimizer.compile(task(), trainset=trainset)
    results['mipro'] = evaluate(mipro_compiled, testset)

    # 4. RPE
    rpe_optimizer = ReflectivePromptEvolution(
        metric=exact_match,
        population_size=10,
        generations=5
    )
    rpe_compiled = rpe_optimizer.compile(task(), trainset=trainset)
    results['rpe'] = evaluate(rpe_compiled, testset)

    # Analyze results
    print("Optimizer Comparison Results:")
    for optimizer, score in results.items():
        improvement = ((score - results['baseline']) / results['baseline']) * 100
        print(f"{optimizer}: {score:.3f} ({improvement:+.1f}%)")

    return results

# Example comparison on a complex reasoning task
results = compare_optimizers(
    task=MultiHopReasoning,
    trainset=hotpotqa_train[:100],
    testset=hotpotqa_test[:50]
)
```

## Advanced Configuration

### Custom Mutation Operators

```python
class CustomMutationOperator:
    def __init__(self, domain_knowledge=None):
        self.domain_knowledge = domain_knowledge or {}

    def domain_specific_mutation(self, prompt, domain):
        """Apply domain-specific mutations."""
        if domain == "math":
            return self.add_math_guidance(prompt)
        elif domain == "code":
            return self.add_code_guidance(prompt)
        elif domain == "legal":
            return self.add_legal_guidance(prompt)
        return prompt

    def add_math_guidance(self, prompt):
        """Add mathematical reasoning guidance."""
        math_guidance = [
            "Show all calculations step by step.",
            "Verify your final answer makes sense.",
            "Consider edge cases and special conditions.",
            "Check for common mathematical errors."
        ]

        return prompt + "\n" + "\n".join(math_guidance)

    def add_code_guidance(self, prompt):
        """Add programming-specific guidance."""
        code_guidance = [
            "Consider time and space complexity.",
            "Handle edge cases and error conditions.",
            "Follow best practices for readability.",
            "Test with example inputs."
        ]

        return prompt + "\n" + "\n".join(code_guidance)

# Use custom mutations with RPE
custom_mutator = CustomMutationOperator(domain_knowledge={
    "math": ["algebra", "calculus", "statistics"],
    "code": ["python", "javascript", "sql"]
})

optimizer = ReflectivePromptEvolution(
    metric=accuracy_metric,
    population_size=10,
    generations=5,
    custom_mutator=custom_mutator
)
```

### Adaptive Evolution Parameters

```python
class AdaptiveRPE(ReflectivePromptEvolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.performance_history = []
        self.adaptive_params = {
            'mutation_rate': kwargs.get('mutation_rate', 0.3),
            'population_size': kwargs.get('population_size', 10),
            'selection_pressure': kwargs.get('selection_pressure', 0.5)
        }

    def adapt_parameters(self, generation, fitness_scores):
        """Adapt evolution parameters based on performance."""
        if len(self.performance_history) > 1:
            # Check if performance is stagnating
            recent_improvement = (
                self.performance_history[-1] -
                self.performance_history[-2]
            )

            if recent_improvement < 0.01:  # Stagnating
                # Increase mutation rate
                self.adaptive_params['mutation_rate'] = min(
                    0.7,
                    self.adaptive_params['mutation_rate'] * 1.2
                )
                print(f"Increasing mutation rate to {self.adaptive_params['mutation_rate']:.2f}")

            elif recent_improvement > 0.05:  # Rapid improvement
                # Decrease mutation rate to fine-tune
                self.adaptive_params['mutation_rate'] = max(
                    0.1,
                    self.adaptive_params['mutation_rate'] * 0.9
                )
                print(f"Decreasing mutation rate to {self.adaptive_params['mutation_rate']:.2f}")

    def evolve_generation(self, population, generation):
        """Evolve one generation with adaptive parameters."""
        # Record best fitness
        fitness_scores = self.evaluate_fitness(population)
        self.performance_history.append(max(fitness_scores))

        # Adapt parameters based on performance
        self.adapt_parameters(generation, fitness_scores)

        # Apply evolution with current parameters
        return super().evolve_generation(population, generation)

# Use adaptive RPE
adaptive_optimizer = AdaptiveRPE(
    metric=accuracy_metric,
    population_size=10,
    generations=10,
    mutation_rate=0.3
)
```

## Best Practices and Tips

### When to Use RPE

1. **Complex Reasoning Tasks**: Multi-step problems requiring sophisticated reasoning
2. **Limited Gradient Information**: When evaluation is expensive or non-differentiable
3. **Diverse Solution Space**: Problems with multiple valid approaches
4. **Exploratory Optimization**: When you want to discover novel prompt strategies

### RPE Configuration Guidelines

```python
# For small datasets (< 50 examples)
small_config = {
    "population_size": 5,
    "generations": 3,
    "mutation_rate": 0.5,  # Higher mutation due to less data
    "selection_pressure": 0.6
}

# For medium datasets (50-200 examples)
medium_config = {
    "population_size": 10,
    "generations": 5,
    "mutation_rate": 0.3,
    "selection_pressure": 0.5
}

# For large datasets (> 200 examples)
large_config = {
    "population_size": 15,
    "generations": 7,
    "mutation_rate": 0.2,  # Lower mutation, more exploitation
    "selection_pressure": 0.4
}

# For highly complex tasks
complex_config = {
    "population_size": 20,
    "generations": 10,
    "mutation_rate": 0.4,
    "selection_pressure": 0.3,  # Keep more diversity
    "diversity_weight": 0.4
}
```

### Common Pitfalls to Avoid

1. **Too Small Population**: Less than 5 individuals may not provide enough diversity
2. **Too High Mutation Rate**: Can destroy good solutions (> 0.7)
3. **Insufficient Generations**: Less than 3 generations may not converge
4. **Ignoring Diversity**: Can lead to premature convergence
5. **Poor Reflection Quality**: Ensure reflection prompts are specific and actionable

### Debugging RPE

```python
class DebugRPE(ReflectivePromptEvolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.debug_info = {
            'mutations_applied': [],
            'reflection_quality': [],
            'diversity_history': [],
            'fitness_progression': []
        }

    def log_mutation(self, program, mutation_type, result):
        """Log mutation for debugging."""
        self.debug_info['mutations_applied'].append({
            'generation': len(self.debug_info['fitness_progression']),
            'type': mutation_type,
            'before': program[:100],
            'after': result[:100]
        })

    def analyze_convergence(self):
        """Analyze optimization progress."""
        import matplotlib.pyplot as plt

        # Plot fitness progression
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(self.debug_info['fitness_progression'])
        plt.title('Fitness Progression')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')

        plt.subplot(1, 3, 2)
        plt.plot(self.debug_info['diversity_history'])
        plt.title('Population Diversity')
        plt.xlabel('Generation')
        plt.ylabel('Diversity Score')

        plt.subplot(1, 3, 3)
        mutation_counts = {}
        for mutation in self.debug_info['mutations_applied']:
            mutation_type = mutation['type']
            mutation_counts[mutation_type] = mutation_counts.get(mutation_type, 0) + 1

        plt.bar(mutation_counts.keys(), mutation_counts.values())
        plt.title('Mutation Types Applied')
        plt.xlabel('Mutation Type')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.show()

# Use debug RPE to analyze optimization
debug_optimizer = DebugRPE(
    metric=accuracy_metric,
    population_size=8,
    generations=5
)
```

## Summary

Reflective Prompt Evolution brings the power of evolutionary computation to prompt optimization:

1. **Self-Reflection**: Uses LM to intelligently guide mutations
2. **Population-Based**: Explores multiple solutions simultaneously
3. **Adaptive**: Adjusts to task complexity and data availability
4. **Diverse**: Maintains solution diversity through explicit mechanisms
5. **Principled**: Based on established evolutionary algorithms principles

RPE is particularly valuable for complex reasoning tasks where traditional optimizers may struggle, offering a novel approach to prompt optimization that combines the strengths of both evolutionary computation and language model self-reflection.

## Key Takeaways

1. **Evolution Without Gradients**: RPE doesn't require gradient information
2. **Reflection-Guided**: Self-reflection makes mutations more intelligent
3. **Diversity Matters**: Maintaining population diversity is crucial
4. **Adaptive Parameters**: RPE can adapt its strategy during optimization
5. **Best for Complex Tasks**: Excels at multi-step reasoning problems