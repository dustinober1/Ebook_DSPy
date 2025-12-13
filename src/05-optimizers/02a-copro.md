# COPRO: Chain-of-Thought Prompt Optimization

## Prerequisites

- **Previous Section**: [BootstrapFewShot](./02-bootstrapfewshot.md) - Understanding of few-shot optimization
- **Chapter 4**: Evaluation - Familiarity with metrics and evaluation
- **Required Knowledge**: Evolutionary algorithms basics (helpful but not required)
- **Difficulty Level**: Intermediate to Advanced
- **Estimated Reading Time**: 45 minutes

## Learning Objectives

By the end of this section, you will:
- Understand how COPRO uses evolutionary search for prompt optimization
- Master instruction generation and optimization techniques
- Learn the algorithm details and configuration options
- Compare COPRO with other DSPy optimizers
- Apply COPRO to complex reasoning tasks

## Introduction to COPRO

COPRO (Chain-of-thought PROmpt optimization) is an advanced DSPy optimizer that uses **evolutionary search** to discover and refine optimal instructions for your language model programs. Unlike BootstrapFewShot which focuses on selecting good demonstrations, COPRO specifically targets **instruction optimization** - finding the best way to describe your task to the language model.

### The Core Innovation

As described in the DSPy paper "Compiling Declarative Language Model Calls into Self-Improving Pipelines," COPRO addresses a fundamental challenge: **prompts that work well for humans may not work well for language models**. COPRO solves this by:

1. **Generating candidate instructions** using the LM itself
2. **Evaluating candidates** against your metric
3. **Evolving better instructions** through mutation and selection
4. **Converging on optimal prompts** without manual intervention

### COPRO as a Cost-Aware Optimization Framework

What makes COPRO particularly powerful is its **cost-aware approach** to optimization. Unlike naive prompt engineering methods that exhaustively test every possible variation, COPRO intelligently manages computational resources:

```
Budget Constraint → Selective Evaluation → Cost-Benefit Analysis → Optimal Resource Allocation
```

Key cost-aware features:
- **Adaptive Evaluation**: Spends more computation on promising candidates
- **Early Termination**: Stops unpromising searches to save resources
- **Budget Management**: Controls total optimization cost
- **Efficiency Metrics**: Tracks cost per improvement

#### Cost-Aware Search Strategies

1. **Progressive Deepening**
   - Start with shallow evaluations (few examples)
   - Deepen evaluation only for promising candidates
   - Reduces overall computation by 60-80%

2. **Resource-Reward Modeling**
   - Models expected improvement vs. computational cost
   - Selects candidates with highest improvement-per-cost ratio
   - Automatically balances exploration vs. exploitation

3. **Dynamic Budget Allocation**
   - Adjusts resource allocation based on early results
   - Shifts budget to more promising search regions
   - Maximizes improvements within fixed budget

## How COPRO Works

### The Evolutionary Search Algorithm

COPRO applies principles from evolutionary computation to prompt engineering:

```python
import dspy
from dspy.teleprompt import COPRO

# COPRO's internal process:
# 1. Initialize population of instruction candidates
# 2. For each generation:
#    a. Evaluate each candidate on training data
#    b. Select top performers
#    c. Generate mutations (variations)
#    d. Create new population
# 3. Return best instruction found

optimizer = COPRO(
    metric=your_metric,
    breadth=10,           # Number of candidates per generation
    depth=3,              # Number of generations
    init_temperature=1.4  # Creativity in generating candidates
)
```

### Step-by-Step Algorithm

#### Step 1: Candidate Generation

COPRO uses the language model to generate diverse instruction candidates:

```python
# COPRO generates candidates by asking the LM:
# "Given this task signature and these examples,
#  what are different ways to instruct an LM to perform this task?"

class TaskSignature(dspy.Signature):
    """Classify customer feedback into categories."""
    feedback: str = dspy.InputField()
    category: str = dspy.OutputField()

# COPRO might generate candidates like:
candidates = [
    "Analyze the customer feedback and determine its category.",
    "Read the feedback carefully and classify it into the most appropriate category based on content.",
    "As a customer service expert, categorize this feedback into one of the predefined categories.",
    "Identify the main topic and sentiment of this customer feedback to assign a category.",
    # ... more variations
]
```

#### Step 2: Evaluation

Each candidate is evaluated against your training data:

```python
def classification_metric(example, pred, trace=None):
    """Metric for evaluating classification accuracy."""
    return example.category.lower() == pred.category.lower()

# COPRO evaluates each candidate:
# Candidate 1: "Analyze..." -> 72% accuracy
# Candidate 2: "Read carefully..." -> 85% accuracy
# Candidate 3: "As an expert..." -> 78% accuracy
# etc.
```

#### Step 3: Evolution

Top-performing candidates are mutated to create new variations:

```python
# Best candidate: "Read the feedback carefully and classify it..."
# COPRO generates mutations:

mutations = [
    "Read the feedback carefully, consider the context, and classify it...",
    "Thoroughly read the feedback and classify it based on its primary concern...",
    "Read and understand the feedback deeply, then classify it...",
]
```

#### Step 4: Selection and Iteration

This process repeats for multiple generations:

```
Generation 1: Best = 85% accuracy
Generation 2: Best = 89% accuracy
Generation 3: Best = 92% accuracy
-> Final optimized instruction
```

## Basic Usage

### Simple Classification Optimization

```python
import dspy
from dspy.teleprompt import COPRO

# Configure LM
lm = dspy.LM(model="openai/gpt-4")
dspy.configure(lm=lm)

# Define signature
class SentimentClassifier(dspy.Signature):
    """Classify text sentiment."""
    text: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="positive, negative, or neutral")

# Create module
classifier = dspy.Predict(SentimentClassifier)

# Prepare training data
trainset = [
    dspy.Example(text="I love this product!", sentiment="positive"),
    dspy.Example(text="Terrible experience.", sentiment="negative"),
    dspy.Example(text="It's okay, nothing special.", sentiment="neutral"),
    # ... more examples (20-50 recommended)
]

# Define metric
def sentiment_accuracy(example, pred, trace=None):
    return example.sentiment.lower() == pred.sentiment.lower()

# Optimize with COPRO
copro = COPRO(
    metric=sentiment_accuracy,
    breadth=10,  # 10 candidates per generation
    depth=3      # 3 generations
)

optimized_classifier = copro.compile(classifier, trainset=trainset)

# Use the optimized classifier
result = optimized_classifier(text="This exceeded all my expectations!")
print(result.sentiment)  # "positive" with higher accuracy
```

### Complex Reasoning Optimization

```python
class MathReasoner(dspy.Signature):
    """Solve mathematical word problems step by step."""
    problem: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Step-by-step solution")
    answer: str = dspy.OutputField(desc="Final numerical answer")

# Use ChainOfThought for reasoning
reasoner = dspy.ChainOfThought(MathReasoner)

# Math problem training data
math_trainset = [
    dspy.Example(
        problem="If John has 3 apples and buys 5 more, how many does he have?",
        answer="8"
    ),
    dspy.Example(
        problem="A train travels 60 miles per hour for 2 hours. How far does it go?",
        answer="120"
    ),
    # ... more examples
]

def math_accuracy(example, pred, trace=None):
    # Extract numerical answer
    try:
        pred_num = float(pred.answer.strip())
        true_num = float(example.answer.strip())
        return abs(pred_num - true_num) < 0.01
    except:
        return example.answer.lower() in pred.answer.lower()

# Optimize for math reasoning
copro = COPRO(
    metric=math_accuracy,
    breadth=15,           # More candidates for complex task
    depth=4,              # More generations
    init_temperature=1.5  # Higher creativity
)

optimized_reasoner = copro.compile(reasoner, trainset=math_trainset)
```

## Advanced Configuration

### COPRO Parameters Explained

```python
copro = COPRO(
    # Core parameters
    metric=your_metric,           # Required: evaluation function

    # Search parameters
    breadth=10,                   # Candidates per generation (default: 10)
    depth=3,                      # Number of generations (default: 3)

    # Generation parameters
    init_temperature=1.4,         # Initial temperature for LM generation
    track_stats=True,             # Track optimization statistics
    verbose=True,                 # Print progress

    # Advanced options
    prompt_model=None,            # LM for generating prompts (default: same as main)
    metric_threshold=None         # Stop early if metric exceeds threshold
)
```

### Parameter Tuning Guidelines

| Task Type | Breadth | Depth | Temperature | Examples |
|-----------|---------|-------|-------------|----------|
| Simple classification | 8-10 | 2-3 | 1.0-1.2 | 20-50 |
| Complex reasoning | 12-15 | 3-5 | 1.3-1.6 | 30-100 |
| Creative generation | 15-20 | 4-6 | 1.5-2.0 | 50-150 |
| Domain-specific | 10-15 | 3-4 | 1.2-1.5 | 40-100 |

### Using Different Models for Prompt Generation

```python
# Use a stronger model to generate prompts
# but optimize for a smaller model

prompt_generator = dspy.LM(model="openai/gpt-4")
target_model = dspy.LM(model="openai/gpt-3.5-turbo")

dspy.configure(lm=target_model)  # Target model for optimization

copro = COPRO(
    metric=your_metric,
    breadth=12,
    depth=4,
    prompt_model=prompt_generator  # Use GPT-4 to generate prompt candidates
)

# Optimized prompts will work well with GPT-3.5
optimized_program = copro.compile(program, trainset=trainset)
```

## COPRO vs Other Optimizers

### Comparison Table

| Feature | COPRO | BootstrapFewShot | MIPRO |
|---------|-------|------------------|-------|
| **Focus** | Instructions | Demonstrations | Both |
| **Method** | Evolutionary | Bootstrap sampling | Bayesian + Bandit |
| **Speed** | Medium | Fast | Slow |
| **Best for** | Instruction-sensitive tasks | Few-shot learning | Maximum performance |
| **Data needs** | 20-100 examples | 10-100 examples | 50-200 examples |
| **Compute** | Medium | Low | High |

### When to Use COPRO

**COPRO excels at:**
- Tasks where **instruction wording matters** significantly
- **Reasoning tasks** that benefit from specific prompting strategies
- **Domain-specific tasks** requiring specialized language
- Scenarios with **limited demonstrations** but clear success criteria

**Consider alternatives when:**
- You have many high-quality demonstrations (use BootstrapFewShot)
- You need maximum performance regardless of cost (use MIPRO)
- Tasks are simple and instruction-independent

### Combining COPRO with Other Optimizers

```python
# Strategy: Use COPRO for instructions, then BootstrapFewShot for demos

# Step 1: Optimize instructions with COPRO
copro = COPRO(metric=your_metric, breadth=10, depth=3)
instruction_optimized = copro.compile(program, trainset=trainset)

# Step 2: Add optimized demonstrations with BootstrapFewShot
bootstrap = BootstrapFewShot(metric=your_metric, max_bootstrapped_demos=8)
fully_optimized = bootstrap.compile(instruction_optimized, trainset=trainset)

# This two-stage approach often outperforms either optimizer alone
```

## Real-World Applications

### Customer Support Classification

```python
class SupportTicketClassifier(dspy.Signature):
    """Classify customer support tickets for routing."""
    ticket_content: str = dspy.InputField(desc="Customer's support request")
    urgency: str = dspy.OutputField(desc="high, medium, or low")
    category: str = dspy.OutputField(desc="billing, technical, general, complaint")
    suggested_team: str = dspy.OutputField()

classifier = dspy.ChainOfThought(SupportTicketClassifier)

# Metric: weighted score for multi-output classification
def support_metric(example, pred, trace=None):
    score = 0
    if pred.urgency == example.urgency:
        score += 0.3
    if pred.category == example.category:
        score += 0.4
    if pred.suggested_team == example.suggested_team:
        score += 0.3
    return score

copro = COPRO(
    metric=support_metric,
    breadth=12,
    depth=4,
    init_temperature=1.3
)

optimized_classifier = copro.compile(classifier, trainset=support_tickets)
```

### Medical Triage Assistant

```python
class MedicalTriageSignature(dspy.Signature):
    """Assess medical symptoms for triage prioritization."""
    symptoms: str = dspy.InputField(desc="Patient reported symptoms")
    medical_history: str = dspy.InputField(desc="Relevant medical history")
    triage_level: str = dspy.OutputField(desc="emergency, urgent, standard, routine")
    reasoning: str = dspy.OutputField(desc="Clinical reasoning for triage decision")
    recommended_actions: str = dspy.OutputField()

triage = dspy.ChainOfThought(MedicalTriageSignature)

# Critical: penalize under-triaging emergencies
def triage_metric(example, pred, trace=None):
    correct = pred.triage_level == example.triage_level

    # Heavy penalty for under-triaging emergencies
    if example.triage_level == "emergency" and pred.triage_level != "emergency":
        return 0.0  # Critical failure

    # Moderate penalty for under-triaging urgent cases
    if example.triage_level == "urgent" and pred.triage_level in ["standard", "routine"]:
        return 0.3

    return 1.0 if correct else 0.5

copro = COPRO(
    metric=triage_metric,
    breadth=15,
    depth=5,           # More generations for critical task
    init_temperature=1.2  # Less wild variations for medical context
)
```

### Legal Document Analysis

```python
class LegalAnalysis(dspy.Signature):
    """Analyze legal documents for key provisions."""
    document: str = dspy.InputField(desc="Legal document text")
    document_type: str = dspy.OutputField(desc="contract, agreement, policy, other")
    key_provisions: str = dspy.OutputField(desc="List of important provisions")
    risks: str = dspy.OutputField(desc="Potential legal risks identified")
    recommendations: str = dspy.OutputField()

analyzer = dspy.ChainOfThought(LegalAnalysis)

# Domain-specific metric
def legal_metric(example, pred, trace=None):
    # Check document type accuracy
    type_score = 1.0 if pred.document_type == example.document_type else 0.0

    # Check provision coverage (simplified)
    expected_provisions = set(example.key_provisions.lower().split(','))
    pred_provisions = set(pred.key_provisions.lower().split(','))
    provision_overlap = len(expected_provisions & pred_provisions) / len(expected_provisions)

    return 0.3 * type_score + 0.7 * provision_overlap

# Use higher temperature for legal domain variety
copro = COPRO(
    metric=legal_metric,
    breadth=12,
    depth=4,
    init_temperature=1.4
)

optimized_analyzer = copro.compile(analyzer, trainset=legal_documents)
```

## Monitoring and Debugging COPRO

### Tracking Optimization Progress

```python
copro = COPRO(
    metric=your_metric,
    breadth=10,
    depth=4,
    track_stats=True,
    verbose=True
)

optimized = copro.compile(program, trainset=trainset)

# Access optimization statistics
if hasattr(copro, 'stats'):
    print("Optimization Statistics:")
    for gen, stats in enumerate(copro.stats):
        print(f"  Generation {gen + 1}:")
        print(f"    Best score: {stats['best_score']:.3f}")
        print(f"    Avg score: {stats['avg_score']:.3f}")
        print(f"    Best instruction: {stats['best_instruction'][:50]}...")
```

### Inspecting Generated Instructions

```python
# After optimization, inspect what COPRO discovered
def inspect_copro_results(optimized_program):
    """Inspect the instructions COPRO optimized."""

    # Get the optimized instructions from each module
    for name, module in optimized_program.named_predictors():
        print(f"\nModule: {name}")
        if hasattr(module, 'extended_signature'):
            sig = module.extended_signature
            print(f"  Optimized instructions: {sig.instructions[:200]}...")

# Usage
inspect_copro_results(optimized)
```

### Debugging Poor Performance

```python
# If COPRO isn't finding good instructions:

# 1. Check your metric
def debug_metric(example, pred, trace=None):
    score = your_original_metric(example, pred, trace)
    print(f"Example: {example}")
    print(f"Prediction: {pred}")
    print(f"Score: {score}")
    return score

# 2. Try more generations with higher breadth
copro_debug = COPRO(
    metric=debug_metric,
    breadth=20,      # More candidates
    depth=6,         # More generations
    init_temperature=1.8,  # More variation
    verbose=True
)

# 3. Ensure training data is diverse
print(f"Training set size: {len(trainset)}")
print(f"Unique examples: {len(set(str(e) for e in trainset))}")
```

## Best Practices

### 1. Provide Diverse Training Examples

```python
# Good: Diverse examples covering edge cases
diverse_trainset = [
    dspy.Example(text="Great product!", sentiment="positive"),
    dspy.Example(text="TERRIBLE SERVICE!!!", sentiment="negative"),
    dspy.Example(text="It works as expected", sentiment="neutral"),
    dspy.Example(text="Could be better, could be worse", sentiment="neutral"),
    dspy.Example(text="Absolutely phenomenal experience", sentiment="positive"),
    # Include edge cases, different formats, various lengths
]

# Bad: Homogeneous examples
bad_trainset = [
    dspy.Example(text="I like it", sentiment="positive"),
    dspy.Example(text="I love it", sentiment="positive"),
    dspy.Example(text="It's good", sentiment="positive"),
    # All similar -> COPRO won't learn to handle variety
]
```

### 2. Design Meaningful Metrics

```python
# Good: Metric captures what you actually care about
def good_metric(example, pred, trace=None):
    # Primary criterion: correctness
    correct = example.answer == pred.answer

    # Secondary: reasoning quality
    reasoning_present = len(pred.reasoning) > 50

    # Weighted combination
    return 0.8 * float(correct) + 0.2 * float(reasoning_present)

# Bad: Binary metric misses nuance
def bad_metric(example, pred, trace=None):
    return example.answer == pred.answer  # Only 0 or 1
```

### 3. Start Conservative, Then Expand

```python
# Phase 1: Quick exploration
copro_quick = COPRO(metric=metric, breadth=8, depth=2)
initial_result = copro_quick.compile(program, trainset=trainset[:20])

# Evaluate initial result
initial_score = evaluate(initial_result, valset)
print(f"Initial optimization: {initial_score:.2%}")

# Phase 2: Deep optimization if needed
if initial_score < target_score:
    copro_deep = COPRO(metric=metric, breadth=15, depth=5)
    final_result = copro_deep.compile(program, trainset=trainset)
```

## Advanced COPRO Techniques

### 1. Multi-Objective Optimization

Optimize for multiple criteria simultaneously:

```python
from dspy.teleprompt import COPRO
import numpy as np

class MultiObjectiveCOPRO:
    """COPRO with multi-objective optimization."""

    def __init__(self, objectives, weights=None):
        self.objectives = objectives  # List of (name, metric_fn) tuples
        self.weights = weights or [1.0] * len(objectives)
        self.pareto_front = []

    def combined_metric(self, example, pred, trace=None):
        """Combine multiple objectives into single score."""
        scores = []
        for (name, metric_fn), weight in zip(self.objectives, self.weights):
            score = metric_fn(example, pred, trace)
            scores.append(score * weight)

        # Weighted sum
        combined = sum(scores) / sum(self.weights)

        # Track individual scores for Pareto analysis
        pred.individual_scores = {
            name: metric_fn(example, pred, trace)
            for name, (metric_fn) in self.objectives
        }

        return combined

    def update_pareto_front(self, candidates):
        """Update Pareto front of non-dominated solutions."""
        for candidate in candidates:
            dominated = False

            # Check if candidate dominates any in front
            for i, existing in enumerate(self.pareto_front):
                if self.dominates(candidate, existing):
                    self.pareto_front[i] = candidate
                    dominated = True
                elif self.dominates(existing, candidate):
                    dominated = True
                    break

            if not dominated:
                self.pareto_front.append(candidate)

    def dominates(self, a, b):
        """Check if solution a dominates solution b."""
        a_scores = getattr(a, 'individual_scores', {})
        b_scores = getattr(b, 'individual_scores', {})

        better_in_all = True
        better_in_one = False

        for obj_name in a_scores:
            if a_scores[obj_name] < b_scores[obj_name]:
                better_in_all = False
            if a_scores[obj_name] > b_scores[obj_name]:
                better_in_one = True

        return better_in_all and better_in_one

# Example: Optimize for both accuracy and efficiency
def accuracy_metric(example, pred, trace=None):
    """Measure prediction accuracy."""
    return float(example.answer.lower() == pred.answer.lower())

def efficiency_metric(example, pred, trace=None):
    """Measure computational efficiency."""
    # Simulate efficiency based on response length
    return 1.0 / (1.0 + len(str(pred)) / 1000.0)

# Create multi-objective optimizer
multi_copro = COPRO(
    metric=MultiObjectiveCOPRO([
        ("accuracy", accuracy_metric),
        ("efficiency", efficiency_metric)
    ]).combined_metric,
    breadth=12,
    depth=4
)

optimized = multi_copro.compile(program, trainset=trainset)
```

### 2. Adaptive Search Strategies

Dynamically adjust search parameters based on progress:

```python
class AdaptiveCOPRO(COPRO):
    """COPRO with adaptive search strategies."""

    def __init__(self, metric, **kwargs):
        super().__init__(metric, **kwargs)
        self.performance_history = []
        self.adaptation_strategy = "progressive"

    def should_adapt(self, generation):
        """Determine if adaptation is needed."""
        if len(self.performance_history) < 3:
            return False

        # Check if performance is stagnating
        recent_scores = self.performance_history[-3:]
        improvement = max(recent_scores) - min(recent_scores)

        return improvement < 0.05  # Less than 5% improvement

    def adapt_parameters(self, generation):
        """Adapt search parameters based on performance."""
        if self.adaptation_strategy == "progressive":
            # Increase breadth if search is stuck
            self.breadth = min(self.breadth * 1.2, 20)

            # Adjust temperature based on diversity
            if self.measure_diversity() < 0.3:
                self.init_temperature = min(self.init_temperature * 1.1, 2.0)
            else:
                self.init_temperature = max(self.init_temperature * 0.9, 0.8)

        elif self.adaptation_strategy == "focused":
            # Focus search around best candidates
            self.breadth = 8  # Reduce breadth
            self.depth = min(self.depth + 1, 6)  # Increase depth

    def measure_diversity(self):
        """Measure diversity of current candidate pool."""
        # Simple diversity metric based on instruction similarity
        if not hasattr(self, 'current_candidates'):
            return 1.0

        instructions = [c.get('instruction', '') for c in self.current_candidates]

        # Calculate pairwise similarities (simplified)
        total_similarity = 0
        count = 0

        for i in range(len(instructions)):
            for j in range(i + 1, len(instructions)):
                # Simple word overlap similarity
                words_i = set(instructions[i].lower().split())
                words_j = set(instructions[j].lower().split())

                if len(words_i) > 0 and len(words_j) > 0:
                    similarity = len(words_i & words_j) / len(words_i | words_j)
                    total_similarity += similarity
                    count += 1

        if count == 0:
            return 1.0

        avg_similarity = total_similarity / count
        diversity = 1.0 - avg_similarity

        return diversity

# Usage
adaptive_copro = AdaptiveCOPRO(
    metric=your_metric,
    breadth=10,
    depth=3,
    adaptation_strategy="progressive"
)
```

### 3. Cost-Constrained Optimization

Optimize within strict budget constraints:

```python
class CostConstrainedCOPRO(COPRO):
    """COPRO with explicit cost constraints."""

    def __init__(self, metric, max_cost=100.0, cost_per_eval=0.01, **kwargs):
        super().__init__(metric, **kwargs)
        self.max_cost = max_cost
        self.cost_per_eval = cost_per_eval
        self.spent_cost = 0.0
        self.cost_history = []

    def compile(self, program, trainset, **kwargs):
        """Compile with cost tracking."""
        self.spent_cost = 0.0

        # Estimate total needed cost
        estimated_cost = self.estimate_optimization_cost(len(trainset))

        if estimated_cost > self.max_cost:
            print(f"Warning: Estimated cost (${estimated_cost:.2f}) exceeds budget (${self.max_cost:.2f})")
            self.adjust_for_budget()

        return super().compile(program, trainset, **kwargs)

    def estimate_optimization_cost(self, dataset_size):
        """Estimate total optimization cost."""
        total_evaluations = self.breadth * self.depth * dataset_size
        return total_evaluations * self.cost_per_eval

    def adjust_for_budget(self):
        """Adjust parameters to fit budget."""
        available_evaluations = self.max_cost / self.cost_per_eval

        # Adjust breadth and depth
        if self.breadth * self.depth > available_evaluations:
            # Prefer reducing breadth first
            self.breadth = max(5, int(available_evaluations ** 0.5))
            self.depth = max(2, int(available_evaluations / self.breadth))

            print(f"Adjusted to breadth={self.breadth}, depth={self.depth}")

    def evaluate_candidate(self, candidate, trainset):
        """Evaluate with cost tracking."""
        if self.spent_cost + self.cost_per_eval * len(trainset) > self.max_cost:
            raise RuntimeError("Budget exceeded!")

        # Record cost before evaluation
        eval_cost = self.cost_per_eval * len(trainset)

        # Evaluate candidate
        result = super().evaluate_candidate(candidate, trainset)

        # Update cost tracking
        self.spent_cost += eval_cost
        self.cost_history.append({
            'evaluation': len(self.cost_history),
            'cost': eval_cost,
            'total': self.spent_cost,
            'score': result.get('score', 0)
        })

        return result

    def get_cost_report(self):
        """Generate cost optimization report."""
        report = {
            'total_spent': self.spent_cost,
            'budget_used': self.spent_cost / self.max_cost,
            'evaluations': len(self.cost_history),
            'avg_cost_per_eval': np.mean([c['cost'] for c in self.cost_history]),
            'cost_efficiency': self.spent_cost / max(1, len(self.cost_history))
        }

        # Calculate improvement per dollar
        if len(self.cost_history) > 1:
            initial_score = self.cost_history[0]['score']
            final_score = self.cost_history[-1]['score']
            improvement = final_score - initial_score
            report['improvement_per_dollar'] = improvement / self.spent_cost

        return report

# Usage with budget constraints
budget_copro = CostConstrainedCOPRO(
    metric=your_metric,
    max_cost=50.0,  # $50 budget
    cost_per_eval=0.005,  # $0.005 per evaluation
    breadth=10,
    depth=3
)

optimized = budget_copro.compile(program, trainset=trainset)
cost_report = budget_copro.get_cost_report()
print(f"Optimization cost: ${cost_report['total_spent']:.2f}")
print(f"Budget used: {cost_report['budget_used']:.1%}")
```

### 4. Hierarchical COPRO

Apply COPRO at multiple levels of abstraction:

```python
class HierarchicalCOPRO:
    """Hierarchical COPRO for complex tasks."""

    def __init__(self, levels):
        """Initialize with optimization levels."""
        self.levels = levels  # List of (name, subprogram) tuples
        self.level_optimizers = {}
        self.global_instructions = None

    def optimize_hierarchically(self, program, trainset):
        """Optimize each level with COPRO."""
        results = {}

        # Level 1: Global instruction optimization
        global_optimizer = COPRO(
            metric=self.create_global_metric(),
            breadth=15,
            depth=4
        )

        self.global_instructions = global_optimizer.compile(
            program.global_module,
            trainset
        )

        results['global'] = self.global_instructions

        # Level 2: Sub-component optimization
        for name, subprogram in self.levels:
            sub_optimizer = COPRO(
                metric=self.create_component_metric(name),
                breadth=10,
                depth=3
            )

            # Use global instructions as context
            contextual_program = self.add_global_context(
                subprogram,
                self.global_instructions
            )

            optimized_sub = sub_optimizer.compile(
                contextual_program,
                trainset
            )

            results[name] = optimized_sub

        return self.reassemble_program(results)

    def create_global_metric(self):
        """Create metric for global optimization."""
        def global_metric(example, pred, trace=None):
            # Evaluate overall task performance
            score = self.evaluate_global_performance(example, pred)

            # Bonus for coherence across components
            coherence_bonus = self.evaluate_coherence(pred)

            return 0.8 * score + 0.2 * coherence_bonus

        return global_metric

    def create_component_metric(self, component_name):
        """Create metric for component optimization."""
        def component_metric(example, pred, trace=None):
            # Component-specific performance
            component_score = self.evaluate_component_performance(
                component_name, example, pred
            )

            # Compatibility with global instructions
            compatibility_score = self.evaluate_compatibility(
                pred, self.global_instructions
            )

            return 0.7 * component_score + 0.3 * compatibility_score

        return component_metric

    def add_global_context(self, subprogram, global_instructions):
        """Add global instruction context to subprogram."""
        # Create wrapper that includes global context
        class ContextualSubprogram(dspy.Module):
            def __init__(self, base_program, context):
                super().__init__()
                self.base_program = base_program
                self.context = context

            def forward(self, **kwargs):
                # Add context to inputs
                kwargs['global_context'] = self.context
                return self.base_program(**kwargs)

        return ContextualSubprogram(subprogram, global_instructions)

# Example: Hierarchical optimization for a QA system
class HierarchicalQA(dspy.Module):
    """Hierarchical QA system with multiple components."""

    def __init__(self):
        super().__init__()
        self.retriever = dspy.Predict("query -> context")
        self.reader = dspy.ChainOfThought("context, query -> answer")
        self.validator = dspy.Predict("query, answer -> confidence")

    def forward(self, query):
        # Get context
        context = self.retriever(query=query)

        # Generate answer
        answer = self.reader(context=context.context, query=query)

        # Validate
        confidence = self.validator(query=query, answer=answer.answer)

        return dspy.Prediction(
            answer=answer.answer,
            confidence=confidence.confidence,
            context=context.context
        )

# Optimize hierarchically
hierarchical_qa = HierarchicalQA()
hierarchical_optimizer = HierarchicalCOPRO([
    ('retriever', hierarchical_qa.retriever),
    ('reader', hierarchical_qa.reader),
    ('validator', hierarchical_qa.validator)
])

optimized_qa = hierarchical_optimizer.optimize_hierarchically(
    hierarchical_qa,
    trainset=qa_trainset
)
```

## Cost-Aware Best Practices

### 1. Budget Planning

```python
def plan_copro_budget(dataset_size, complexity="medium"):
    """Plan COPRO optimization budget."""
    complexity_multipliers = {
        "simple": 1.0,
        "medium": 2.0,
        "complex": 4.0
    }

    # Base cost estimates (in dollars)
    base_cost_per_eval = 0.01
    base_evaluations = dataset_size * 10  # Typical evaluations

    total_cost = (
        base_cost_per_eval *
        base_evaluations *
        complexity_multipliers[complexity]
    )

    return {
        'estimated_cost': total_cost,
        'recommended_breadth': min(15, max(5, int(dataset_size / 10))),
        'recommended_depth': 3 if complexity != "complex" else 4,
        'cost_saving_tips': [
            "Use progressive evaluation for large datasets",
            "Start with smaller breadth and increase if needed",
            "Set early stopping criteria to avoid wasted computation"
        ]
    }

budget_plan = plan_copro_budget(dataset_size=100, complexity="medium")
print(f"Estimated optimization cost: ${budget_plan['estimated_cost']:.2f}")
```

### 2. Efficiency Metrics

```python
class EfficiencyTracker:
    """Track COPRO optimization efficiency."""

    def __init__(self):
        self.metrics = {
            'improvements': [],
            'costs': [],
            'times': [],
            'iterations': []
        }

    def record_iteration(self, score, cost, time_taken):
        """Record metrics for an iteration."""
        self.metrics['improvements'].append(score)
        self.metrics['costs'].append(cost)
        self.metrics['times'].append(time_taken)
        self.metrics['iterations'].append(len(self.metrics['improvements']))

    def calculate_efficiency_metrics(self):
        """Calculate efficiency metrics."""
        if len(self.metrics['improvements']) < 2:
            return {}

        improvements = self.metrics['improvements']
        total_cost = sum(self.metrics['costs'])
        total_time = sum(self.metrics['times'])

        # Calculate metrics
        total_improvement = improvements[-1] - improvements[0]

        return {
            'improvement_per_dollar': total_improvement / max(total_cost, 0.01),
            'improvement_per_hour': total_improvement / max(total_time / 3600, 0.01),
            'cost_per_point': total_cost / max(total_improvement, 0.01),
            'time_per_point': total_time / max(total_improvement, 0.01),
            'efficiency_trend': self.calculate_efficiency_trend()
        }

    def calculate_efficiency_trend(self):
        """Calculate if efficiency is improving or declining."""
        if len(self.metrics['costs']) < 10:
            return "insufficient_data"

        # Compare recent efficiency to early efficiency
        early_improvement = (
            self.metrics['improvements'][4] - self.metrics['improvements'][0]
        ) / sum(self.metrics['costs'][:5])

        recent_improvement = (
            self.metrics['improvements'][-1] - self.metrics['improvements'][-5]
        ) / sum(self.metrics['costs'][-5:])

        if recent_improvement > early_improvement * 1.1:
            return "improving"
        elif recent_improvement < early_improvement * 0.9:
            return "declining"
        else:
            return "stable"

# Track optimization efficiency
tracker = EfficiencyTracker()

# During COPRO optimization
for iteration in range(num_iterations):
    start_time = time.time()
    score, cost = evaluate_candidate(candidate)
    time_taken = time.time() - start_time

    tracker.record_iteration(score, cost, time_taken)

# Get efficiency report
efficiency_metrics = tracker.calculate_efficiency_metrics()
print(f"Improvement per dollar: {efficiency_metrics.get('improvement_per_dollar', 0):.3f}")
print(f"Efficiency trend: {efficiency_metrics.get('efficiency_trend', 'unknown')}")
```

## Summary

COPRO is a powerful evolutionary optimizer for instruction optimization with advanced cost-aware features:

- **Evolutionary Search**: Uses LM-generated mutations and selection
- **Instruction Focus**: Optimizes how tasks are described to the model
- **Cost-Aware Optimization**: Intelligently manages computational resources
- **Multi-Objective Support**: Optimize for multiple criteria simultaneously
- **Adaptive Strategies**: Dynamically adjust search parameters
- **Budget Constraints**: Optimize within strict resource limits
- **Hierarchical Optimization**: Apply at multiple levels of abstraction

### Key Takeaways

1. **Use COPRO** when instruction wording significantly impacts performance
2. **Provide diverse training data** for robust optimization
3. **Design metrics** that capture what you care about
4. **Combine with BootstrapFewShot** for both instruction and demonstration optimization
5. **Monitor progress** and debug using verbose mode and statistics

## Next Steps

- [MIPRO](./03-mipro.md) - Multi-step instruction and demonstration optimization
- [KNNFewShot](./04-knnfewshot.md) - Similarity-based example selection
- [Choosing Optimizers](./06-choosing-optimizers.md) - Decision guide for optimizer selection
- [Exercises](./07-exercises.md) - Practice COPRO optimization

## Further Reading

- [DSPy Paper](https://arxiv.org/abs/2310.03714) - Original COPRO algorithm description
- [Evolutionary Optimization](https://en.wikipedia.org/wiki/Evolutionary_algorithm) - Background on evolutionary algorithms
- [DSPy Documentation: COPRO](https://dspy-docs.vercel.app/docs/deep-dive/copro)
