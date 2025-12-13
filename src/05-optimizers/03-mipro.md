# MIPRO: Multi-step Instruction and Demonstration Optimization

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand MIPRO's dual-component optimization approach
- Implement meta-prompting for instruction generation
- Configure simulated annealing for efficient prompt search
- Apply module-specific demonstration selection strategies
- Optimize multi-stage pipelines effectively
- Interpret and replicate MIPRO benchmark results

## Introduction

MIPRO (Multi-step Instruction and demonstration PRompt Optimization) represents a significant advancement in automated prompt optimization for language model programs. Unlike simpler approaches that only optimize examples, MIPRO simultaneously optimizes both the instructions (prompts) and demonstrations (examples) for each module in a multi-stage pipeline.

Research demonstrates MIPRO's effectiveness across diverse benchmarks:
- **HotpotQA**: 52.3 F1 vs 32.0 F1 manual prompting (63% improvement)
- **GSM8K**: 33.8% vs 28.5% manual prompting (19% improvement)
- **CodeAlpaca**: 64.8% vs 63.1% manual prompting

These results highlight MIPRO's ability to discover optimized prompts that generalize better than hand-crafted alternatives, often achieving zero-shot superiority through optimized instructions alone.

## Core Architecture: Dual-Component Optimization

MIPRO's power comes from its dual-component approach that jointly optimizes two key elements:

### Component 1: Instruction Generation
MIPRO generates candidate instructions using **meta-prompting**, where a language model is prompted to create task-specific instructions conditioned on:
- The program's overall structure and purpose
- Individual module signatures and roles
- Relationships between pipeline stages
- Dataset characteristics and examples

### Component 2: Demonstration Selection
For few-shot learning, MIPRO selects demonstrations using:
- Data-driven selection from bootstrapped examples (via BootstrapFewShot)
- Module-specific demonstration counts
- Utility scoring based on validation performance
- Greedy selection algorithms

```
┌──────────────────────────────────────────────────────────────────┐
│                     MIPRO Optimization Loop                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐      ┌────────────────┐                      │
│  │  Meta-Prompt   │      │  Demonstration │                      │
│  │  Generation    │      │  Selection     │                      │
│  └───────┬────────┘      └───────┬────────┘                      │
│          │                       │                               │
│          ▼                       ▼                               │
│  ┌────────────────────────────────────────┐                      │
│  │        Candidate Configurations        │                      │
│  │   (Instruction + Demonstration Pairs)  │                      │
│  └───────────────────┬────────────────────┘                      │
│                      │                                           │
│                      ▼                                           │
│  ┌────────────────────────────────────────┐                      │
│  │      Simulated Annealing Search        │                      │
│  │   - Evaluate on validation set         │                      │
│  │   - Accept/reject based on temperature │                      │
│  │   - Gradually reduce temperature       │                      │
│  └───────────────────┬────────────────────┘                      │
│                      │                                           │
│                      ▼                                           │
│  ┌────────────────────────────────────────┐                      │
│  │         Best Configuration             │                      │
│  └────────────────────────────────────────┘                      │
└──────────────────────────────────────────────────────────────────┘
```

## Meta-Prompting for Instruction Generation

Meta-prompting is MIPRO's technique for generating candidate instructions automatically. Instead of requiring human-written prompts, MIPRO uses a language model to generate diverse, task-specific instructions.

### How Meta-Prompting Works

```python
class MIPROMetaPromptGenerator:
    """
    Generates candidate instructions using meta-prompting.

    Meta-prompts condition on:
    1. Program structure (modules and their connections)
    2. Dataset characteristics (input/output types, examples)
    3. Task description (what the program should accomplish)
    """

    def generate_instruction_candidates(
        self,
        module_signature: str,
        program_description: str,
        dataset_summary: str,
        num_candidates: int = 10,
        temperature: float = 0.7
    ) -> list[str]:
        """
        Generate diverse instruction candidates for a module.

        Args:
            module_signature: The module's input/output signature
            program_description: Overall program purpose
            dataset_summary: Summary of training data characteristics
            num_candidates: Number of candidates to generate
            temperature: Sampling temperature (0.7 recommended for diversity)

        Returns:
            List of candidate instruction strings
        """
        meta_prompt = f"""
You are designing instructions for a language model module in a larger program.

PROGRAM PURPOSE: {program_description}

MODULE SIGNATURE: {module_signature}

DATASET CHARACTERISTICS: {dataset_summary}

Generate {num_candidates} diverse instruction variations for this module.
Each instruction should:
1. Clearly specify the task
2. Guide the model toward high-quality outputs
3. Be self-contained and unambiguous
4. Vary in phrasing, structure, and emphasis

Instructions:
"""

        candidates = []
        for _ in range(num_candidates):
            # Temperature sampling enables diversity
            response = self.lm(meta_prompt, temperature=temperature)
            candidates.append(response)

        return candidates
```

### Temperature Sampling for Diversity

MIPRO uses temperature sampling (T=0.7 by default) to generate diverse instruction candidates:

```python
# Low temperature (T=0.3): More deterministic, similar instructions
# Medium temperature (T=0.7): Good diversity while maintaining quality
# High temperature (T=1.2): Maximum diversity, may reduce quality

optimizer = MIPRO(
    metric=your_metric,
    num_candidates=10,
    init_temperature=0.7  # Controls instruction generation diversity
)
```

### Self-Reflection for Instruction Refinement

MIPRO can optionally use self-reflection to refine generated instructions:

```python
class InstructionRefiner:
    """
    Refines candidate instructions through self-reflection.
    """

    def __init__(self):
        self.reflect = dspy.Predict(
            "instruction, task_description, failure_cases -> improved_instruction"
        )

    def refine(self, instruction: str, task_desc: str, failures: list) -> str:
        """
        Improve an instruction based on observed failures.
        """
        result = self.reflect(
            instruction=instruction,
            task_description=task_desc,
            failure_cases="\n".join(failures)
        )
        return result.improved_instruction
```

## Demonstration Selection Strategy

MIPRO's demonstration selection builds on BootstrapFewShot but adds sophisticated selection mechanisms.

### Data-Driven Selection

```python
class MIPRODemonstrationSelector:
    """
    Selects demonstrations using data-driven strategies.
    """

    def __init__(self, trainset, metric, max_demos_per_module=8):
        self.trainset = trainset
        self.metric = metric
        self.max_demos_per_module = max_demos_per_module

    def bootstrap_demonstrations(self, program):
        """
        Generate candidate demonstrations using BootstrapFewShot.
        """
        # First, bootstrap potential demonstrations
        bootstrap = BootstrapFewShot(
            metric=self.metric,
            max_bootstrapped_demos=self.max_demos_per_module * 2
        )
        bootstrapped = bootstrap.compile(program, trainset=self.trainset)
        return bootstrapped.demos

    def score_demonstration_utility(self, demo, module, valset):
        """
        Score a demonstration's utility for a specific module.

        Utility is measured by validation set performance improvement
        when the demonstration is included.
        """
        # Test with and without this demonstration
        score_with = self._evaluate_with_demo(module, demo, valset)
        score_without = self._evaluate_without_demo(module, demo, valset)

        return score_with - score_without

    def greedy_select(self, candidates, module, valset, k):
        """
        Greedy selection of top-k demonstrations.

        Args:
            candidates: List of candidate demonstrations
            module: The module to optimize
            valset: Validation set for scoring
            k: Number of demonstrations to select

        Returns:
            List of k selected demonstrations
        """
        selected = []
        remaining = candidates.copy()

        for _ in range(k):
            if not remaining:
                break

            # Score each remaining candidate
            scores = [
                (demo, self.score_demonstration_utility(demo, module, valset))
                for demo in remaining
            ]

            # Select the best
            best_demo, best_score = max(scores, key=lambda x: x[1])
            selected.append(best_demo)
            remaining.remove(best_demo)

        return selected
```

### Module-Specific Demonstration Counts

Different modules may benefit from different numbers of demonstrations:

```python
def determine_demo_count(module_type, context_budget=16000):
    """
    Determine optimal demonstration count per module.

    Args:
        module_type: Type of module (e.g., 'retrieval', 'reasoning', 'generation')
        context_budget: Available context window in tokens

    Returns:
        Recommended number of demonstrations
    """
    # Simple modules need fewer demos
    if module_type == 'classification':
        return min(4, context_budget // 500)

    # Reasoning tasks benefit from more demos
    elif module_type == 'reasoning':
        return min(8, context_budget // 1000)

    # Generation tasks need diverse examples
    elif module_type == 'generation':
        return min(6, context_budget // 800)

    # Default
    return min(5, context_budget // 600)
```

## Simulated Annealing for Prompt Search

MIPRO uses simulated annealing to efficiently search the space of possible prompt configurations.

### Why Simulated Annealing?

The prompt optimization landscape is:
- **High-dimensional**: Many modules, each with instruction and demonstration choices
- **Non-convex**: Local optima are common
- **Noisy**: Validation scores have variance

Simulated annealing handles these challenges by:
1. Starting with high temperature (accepting many changes)
2. Gradually cooling (becoming more selective)
3. Allowing occasional "uphill" moves to escape local optima

### Implementation

```python
import math
import random

class SimulatedAnnealingOptimizer:
    """
    Simulated annealing for prompt configuration search.
    """

    def __init__(
        self,
        init_temperature: float = 1.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
        max_iter: int = 100
    ):
        self.init_temperature = init_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iter = max_iter

    def optimize(
        self,
        initial_config,
        neighbor_fn,
        score_fn,
        valset
    ):
        """
        Find optimal configuration using simulated annealing.

        Args:
            initial_config: Starting configuration
            neighbor_fn: Function to generate neighboring configurations
            score_fn: Function to score a configuration
            valset: Validation set for evaluation

        Returns:
            Best configuration found
        """
        current_config = initial_config
        current_score = score_fn(current_config, valset)

        best_config = current_config
        best_score = current_score

        temperature = self.init_temperature

        for iteration in range(self.max_iter):
            # Generate neighbor configuration
            neighbor = neighbor_fn(current_config)
            neighbor_score = score_fn(neighbor, valset)

            # Calculate acceptance probability
            delta = neighbor_score - current_score

            if delta > 0:
                # Better solution: always accept
                accept_prob = 1.0
            else:
                # Worse solution: accept with probability
                accept_prob = math.exp(delta / temperature)

            # Accept or reject
            if random.random() < accept_prob:
                current_config = neighbor
                current_score = neighbor_score

                # Update best
                if current_score > best_score:
                    best_config = current_config
                    best_score = current_score

            # Cool down
            temperature = max(
                self.min_temperature,
                temperature * self.cooling_rate
            )

            # Optional: Log progress
            if iteration % 10 == 0:
                print(f"Iter {iteration}: Score={current_score:.3f}, "
                      f"Best={best_score:.3f}, T={temperature:.3f}")

        return best_config, best_score
```

### Configuration Neighbor Generation

```python
def generate_neighbor(current_config, instruction_pool, demo_pool):
    """
    Generate a neighboring configuration by making small changes.

    Possible mutations:
    1. Change instruction for one module
    2. Add/remove/swap demonstration for one module
    3. Adjust demonstration count
    """
    neighbor = copy.deepcopy(current_config)

    # Choose mutation type
    mutation_type = random.choice([
        'change_instruction',
        'swap_demo',
        'add_demo',
        'remove_demo'
    ])

    # Choose random module to mutate
    module_idx = random.randint(0, len(neighbor.modules) - 1)

    if mutation_type == 'change_instruction':
        # Select new instruction from pool
        new_instruction = random.choice(instruction_pool[module_idx])
        neighbor.modules[module_idx].instruction = new_instruction

    elif mutation_type == 'swap_demo':
        # Swap one demonstration
        if neighbor.modules[module_idx].demos:
            demo_idx = random.randint(0, len(neighbor.modules[module_idx].demos) - 1)
            new_demo = random.choice(demo_pool[module_idx])
            neighbor.modules[module_idx].demos[demo_idx] = new_demo

    elif mutation_type == 'add_demo':
        # Add a demonstration if under limit
        if len(neighbor.modules[module_idx].demos) < MAX_DEMOS:
            new_demo = random.choice(demo_pool[module_idx])
            neighbor.modules[module_idx].demos.append(new_demo)

    elif mutation_type == 'remove_demo':
        # Remove a demonstration if any exist
        if neighbor.modules[module_idx].demos:
            neighbor.modules[module_idx].demos.pop()

    return neighbor
```

## Multi-Stage Pipeline Optimization

MIPRO excels at optimizing multi-stage pipelines where modules depend on each other.

### Module Coupling Considerations

When optimizing pipelines, MIPRO considers how modules interact:

```python
class MultiStagePipelineOptimizer:
    """
    Optimizes multi-stage pipelines considering module dependencies.
    """

    def analyze_module_coupling(self, program):
        """
        Analyze how modules in a pipeline are coupled.

        Returns dependency graph and coupling strength estimates.
        """
        modules = program.modules
        coupling = {}

        for i, module in enumerate(modules):
            coupling[i] = {
                'inputs_from': [],
                'outputs_to': [],
                'coupling_strength': 0.0
            }

            # Analyze dataflow
            for j, other in enumerate(modules):
                if i != j:
                    if self._has_dataflow(module, other):
                        coupling[i]['outputs_to'].append(j)
                        coupling[j]['inputs_from'].append(i)

        return coupling

    def optimize_pipeline(self, program, trainset, valset):
        """
        Optimize a multi-stage pipeline.

        Strategy:
        1. Start with later stages (less dependent)
        2. Progressively optimize earlier stages
        3. Use frozen later stages when optimizing earlier ones
        """
        modules = program.modules
        coupling = self.analyze_module_coupling(program)

        # Order modules by dependency depth (later stages first)
        optimization_order = self._topological_sort_reverse(coupling)

        for module_idx in optimization_order:
            print(f"Optimizing module {module_idx}...")

            # Freeze downstream modules
            frozen_modules = [i for i in optimization_order
                           if i != module_idx and
                           module_idx in coupling[i]['inputs_from']]

            # Optimize this module
            self._optimize_module(
                program,
                module_idx,
                trainset,
                valset,
                frozen_modules
            )

        return program
```

### Generate-Retrieve-Generate Pipeline Example

```python
class GRGPipeline(dspy.Module):
    """
    Generate-Retrieve-Generate pipeline for complex QA.

    Stage 1 (Generate): Generate search queries from question
    Stage 2 (Retrieve): Retrieve relevant documents
    Stage 3 (Generate): Generate answer from retrieved context
    """

    def __init__(self):
        super().__init__()
        # Stage 1: Query generation
        self.generate_queries = dspy.Predict(
            "question -> search_queries"
        )

        # Stage 2: Retrieval
        self.retrieve = dspy.Retrieve(k=5)

        # Stage 3: Answer generation
        self.generate_answer = dspy.ChainOfThought(
            "question, context -> answer"
        )

    def forward(self, question):
        # Stage 1
        queries = self.generate_queries(question=question)

        # Stage 2
        all_passages = []
        for query in queries.search_queries.split('\n'):
            passages = self.retrieve(query=query.strip()).passages
            all_passages.extend(passages)

        # Stage 3
        context = '\n\n'.join(all_passages[:10])
        answer = self.generate_answer(
            question=question,
            context=context
        )

        return dspy.Prediction(
            answer=answer.answer,
            reasoning=answer.rationale,
            passages_used=len(all_passages)
        )

# Optimize with MIPRO
def optimize_grg_pipeline(trainset, valset):
    """
    Optimize GRG pipeline using MIPRO.
    """
    pipeline = GRGPipeline()

    def grg_metric(example, pred, trace=None):
        # Check answer correctness
        if hasattr(example, 'answer') and hasattr(pred, 'answer'):
            return example.answer.lower() in pred.answer.lower()
        return 0

    optimizer = MIPRO(
        metric=grg_metric,
        num_candidates=15,  # More candidates for multi-stage
        init_temperature=0.7,
        auto="medium"
    )

    optimized = optimizer.compile(
        pipeline,
        trainset=trainset,
        num_trials=5,
        max_bootstrapped_demos=6  # Per module
    )

    return optimized
```

### Zero-Shot vs Few-Shot Trade-offs

MIPRO research reveals important insights about zero-shot vs few-shot optimization:

```python
def analyze_zeroshot_vs_fewshot(program, trainset, valset, testset):
    """
    Analyze when zero-shot optimized prompts outperform few-shot.

    Key findings from MIPRO research:
    1. Optimized zero-shot can beat manual few-shot
    2. Context window savings enable more reasoning
    3. Generalization is often better with zero-shot
    """
    results = {}

    # Zero-shot optimization
    mipro_zeroshot = MIPRO(
        metric=metric,
        num_candidates=20,
        init_temperature=0.7
    )
    zeroshot_compiled = mipro_zeroshot.compile(
        program,
        trainset=trainset,
        max_bootstrapped_demos=0  # Zero demonstrations
    )
    results['zeroshot'] = evaluate(zeroshot_compiled, testset)

    # Few-shot optimization
    mipro_fewshot = MIPRO(
        metric=metric,
        num_candidates=20,
        init_temperature=0.7
    )
    fewshot_compiled = mipro_fewshot.compile(
        program,
        trainset=trainset,
        max_bootstrapped_demos=8  # Include demonstrations
    )
    results['fewshot'] = evaluate(fewshot_compiled, testset)

    # Manual few-shot baseline
    bootstrap = BootstrapFewShot(metric=metric, max_bootstrapped_demos=8)
    manual_fewshot = bootstrap.compile(program, trainset=trainset)
    results['manual_fewshot'] = evaluate(manual_fewshot, testset)

    # Analysis
    print("\nZero-Shot vs Few-Shot Analysis:")
    print(f"  MIPRO Zero-Shot: {results['zeroshot']:.1%}")
    print(f"  MIPRO Few-Shot:  {results['fewshot']:.1%}")
    print(f"  Manual Few-Shot: {results['manual_fewshot']:.1%}")

    if results['zeroshot'] > results['manual_fewshot']:
        print("\n  > Optimized zero-shot outperforms manual few-shot!")
        print("  > This indicates strong instruction optimization.")

    return results
```

## Hyperparameter Configuration

### Recommended Settings

Based on MIPRO research, here are recommended hyperparameter configurations:

```python
# Standard configuration
optimizer = MIPRO(
    metric=your_metric,
    num_candidates=10,          # 10-20 instruction candidates
    init_temperature=0.7,       # T=0.7 for diverse but quality instructions
    verbose=True
)

# Compile with appropriate settings
compiled = optimizer.compile(
    program,
    trainset=trainset,
    num_trials=3,              # 3-5 optimization trials
    max_bootstrapped_demos=8,  # Up to 8 demos per module
    max_labeled_demos=4,       # Up to 4 labeled demos
)

# Context window management (important for large pipelines)
# Total context budget: 16k tokens typical
# Reserve: ~4k for reasoning
# Remaining: ~12k for instructions + demonstrations
# With 8 demos at ~300 tokens each: 2.4k tokens
# Leaves: ~9.6k for instructions and output
```

### Configuration Table

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `num_candidates` | 10 | 5-30 | Instruction candidates per module |
| `init_temperature` | 1.0 | 0.5-1.5 | Meta-prompt sampling temperature |
| `num_trials` | 3 | 1-10 | Optimization iterations |
| `max_bootstrapped_demos` | 8 | 0-16 | Max demonstrations per module |
| `max_labeled_demos` | 4 | 0-8 | Max labeled (gold) demonstrations |
| `auto` | None | "light"/"medium"/"heavy" | Auto-configuration mode |

### Auto Mode Configurations

```python
# Light mode: Quick optimization for simple tasks
# Equivalent to: num_candidates=5, init_temperature=0.8
optimizer = MIPRO(auto="light")

# Medium mode: Balanced optimization (recommended default)
# Equivalent to: num_candidates=10, init_temperature=1.0
optimizer = MIPRO(auto="medium")

# Heavy mode: Extensive optimization for complex tasks
# Equivalent to: num_candidates=20, init_temperature=1.2
optimizer = MIPRO(auto="heavy")
```

## Performance Benchmarks

### Research Results

MIPRO has been extensively benchmarked across diverse tasks:

| Dataset | Task Type | Manual Prompt | MIPRO Optimized | Improvement |
|---------|-----------|---------------|-----------------|-------------|
| HotpotQA | Multi-hop QA | 32.0 F1 | 52.3 F1 | +63.4% |
| GSM8K | Math Reasoning | 28.5% | 33.8% | +18.6% |
| CodeAlpaca | Code Generation | 63.1% | 64.8% | +2.7% |
| FEVER | Fact Verification | 71.2% | 78.9% | +10.8% |
| Natural Questions | Open-domain QA | 45.3% | 54.7% | +20.8% |

### Reproducing Benchmarks

```python
def benchmark_mipro_hotpotqa(trainset, valset, testset):
    """
    Reproduce HotpotQA benchmark results.

    Expected: ~52.3 F1 with MIPRO optimization
    """
    # Define multi-hop QA program
    class HotpotQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.retrieve = dspy.Retrieve(k=5)
            self.hop1 = dspy.ChainOfThought("question, context -> intermediate_answer")
            self.hop2 = dspy.ChainOfThought(
                "question, intermediate_answer, context -> final_answer"
            )

        def forward(self, question):
            # First hop
            context1 = self.retrieve(question=question).passages
            hop1_result = self.hop1(
                question=question,
                context='\n'.join(context1)
            )

            # Second hop (refined query)
            refined_query = f"{question} {hop1_result.intermediate_answer}"
            context2 = self.retrieve(question=refined_query).passages

            final = self.hop2(
                question=question,
                intermediate_answer=hop1_result.intermediate_answer,
                context='\n'.join(context2)
            )

            return dspy.Prediction(
                answer=final.final_answer,
                reasoning=final.rationale
            )

    # F1 metric for evaluation
    def hotpot_f1(example, pred, trace=None):
        from collections import Counter

        pred_tokens = pred.answer.lower().split()
        gold_tokens = example.answer.lower().split()

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    # MIPRO optimization
    optimizer = MIPRO(
        metric=hotpot_f1,
        num_candidates=15,
        init_temperature=0.7,
        auto="medium"
    )

    optimized = optimizer.compile(
        HotpotQA(),
        trainset=trainset,
        num_trials=5,
        max_bootstrapped_demos=6
    )

    # Evaluate
    from dspy.evaluate import Evaluate
    evaluator = Evaluate(devset=testset, metric=hotpot_f1)
    score = evaluator(optimized)

    print(f"HotpotQA F1 Score: {score:.1f}")
    return optimized, score
```

## What Makes MIPRO Special?

### Dual Optimization
1. **Instruction Optimization**: Rewrites and refines natural language instructions using meta-prompting
2. **Demonstration Optimization**: Selects and generates optimal examples using utility-based scoring
3. **Joint Optimization**: Optimizes instructions and examples together using simulated annealing

### Multi-Step Process
MIPRO uses an iterative approach to progressively improve your program:
1. Generate diverse instruction candidates via meta-prompting
2. Bootstrap and score potential demonstrations
3. Use simulated annealing to search configuration space
4. Evaluate candidates on validation set
5. Select best configuration based on metric performance

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