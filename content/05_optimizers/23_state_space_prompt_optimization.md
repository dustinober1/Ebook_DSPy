# State-Space Search for Prompt Optimization

## Overview

**State-Space Prompt Optimization** treats prompt optimization as a classical AI search problem where the prompt space is modeled as a graph. This approach, introduced by Taneja (2025), systematically explores prompt variations using defined transformation operators and search algorithms like beam search and random walk.

Unlike DSPy's demonstration-based approach, this method focuses on optimizing the instruction text itself through deliberate transformations, allowing us to quantify which prompt-engineering techniques consistently improve performance.

## Core Concepts

### The Prompt Space as a Graph

The state-space approach models prompts as nodes in a graph where:
- **States**: Individual prompt strings
- **Edges**: Transformation operations that modify prompts
- **Heuristic**: Performance score on a development set
- **Goal**: Find the prompt with maximum performance

```
Seed Prompt
    |
    | (apply transformations)
    v
Prompt A → Prompt B → Prompt C
    |         |         |
    v         v         v
Prompt D   Prompt E   Prompt F
```

### Key Components

1. **Prompt Operators**: Defined transformations that mutate prompts
2. **Search Algorithms**: Methods to explore the prompt space
3. **Evaluation Heuristics**: Functions to score prompt quality
4. **State Representation**: Data structures to track optimization paths

## Implementation in DSPy

### 1. PromptNode Structure

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class PromptNode:
    """Node in the prompt search graph."""
    prompt_text: str
    parent: Optional['PromptNode'] = None
    operator_used: Optional[str] = None
    score: Optional[float] = None
    children: List['PromptNode'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def add_child(self, child: 'PromptNode'):
        """Add a child node."""
        self.children.append(child)
        child.parent = self

    def get_path(self) -> List[str]:
        """Get the sequence of operators used to reach this node."""
        path = []
        node = self
        while node.parent is not None:
            path.append(node.operator_used)
            node = node.parent
        return list(reversed(path))

    def __str__(self):
        score_str = f" (score: {self.score:.3f})" if self.score is not None else ""
        return f"Prompt{score_str}"
```

### 2. Prompt Operators (Transformations)

```python
import dspy
from abc import ABC, abstractmethod

class PromptOperator(ABC):
    """Base class for prompt transformation operators."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def apply(self, prompt: str, context_examples: List[dspy.Example]) -> str:
        """Apply the transformation to a prompt."""
        pass

class MakeConciseOperator(PromptOperator):
    """Make prompt more concise and direct."""

    def __init__(self):
        super().__init__("make_concise")

    def apply(self, prompt: str, context_examples: List[dspy.Example]) -> str:
        rewrite_prompt = f"""
        Rewrite the following prompt to be more concise and direct:

        Original Prompt: {prompt}

        Requirements:
        - Preserve the exact same task objective
        - Remove unnecessary words and phrases
        - Keep only essential instructions
        - Maintain clarity

        Concise Prompt:
        """
        response = dspy.Predict(rewrite_prompt)
        return response.concise_prompt

class AddExamplesOperator(PromptOperator):
    """Add few-shot examples to the prompt."""

    def __init__(self):
        super().__init__("add_examples")

    def apply(self, prompt: str, context_examples: List[dspy.Example]) -> str:
        # Select 1-2 diverse examples
        examples_to_add = context_examples[:2]

        examples_text = "\n\nExamples:\n"
        for i, example in enumerate(examples_to_add, 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Input: {example.inputs()}\n"
            examples_text += f"Output: {example.outputs()}\n"

        return prompt + examples_text

class ReorderOperator(PromptOperator):
    """Reorganize prompt structure for better clarity."""

    def __init__(self):
        super().__init__("reorder")

    def apply(self, prompt: str, context_examples: List[dspy.Example]) -> str:
        rewrite_prompt = f"""
        Reorganize the following prompt to maximize clarity and flow:

        Original Prompt: {prompt}

        Common structure: Task → Requirements → Examples → Output Format

        Reorganized Prompt:
        """
        response = dspy.Predict(rewrite_prompt)
        return response.reorganized_prompt

class MakeVerboseOperator(PromptOperator):
    """Add more detail and explanation to the prompt."""

    def __init__(self):
        super().__init__("make_verbose")

    def apply(self, prompt: str, context_examples: List[dspy.Example]) -> str:
        rewrite_prompt = f"""
        Expand the following prompt with additional details and clarification:

        Original Prompt: {prompt}

        Add:
        - Detailed explanations
        - Step-by-step guidance
        - Clarification of edge cases
        - Explicit quality criteria

        Expanded Prompt:
        """
        response = dspy.Predict(rewrite_prompt)
        return response.expanded_prompt

# Collection of all operators
DEFAULT_OPERATORS = [
    MakeConciseOperator(),
    AddExamplesOperator(),
    ReorderOperator(),
    MakeVerboseOperator(),
]
```

### 3. Search Algorithms

#### Beam Search

```python
import heapq
from typing import List, Tuple

class BeamSearchOptimizer:
    """Beam search for prompt optimization."""

    def __init__(self,
                 beam_width: int = 2,
                 max_depth: int = 2,
                 operators: List[PromptOperator] = None):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.operators = operators or DEFAULT_OPERATORS

    def optimize(self,
                 seed_prompt: str,
                 train_set: List[dspy.Example],
                 dev_set: List[dspy.Example],
                 evaluator: dspy.Evaluate) -> PromptNode:
        """Optimize prompt using beam search."""

        # Create root node
        root = PromptNode(prompt_text=seed_prompt)
        root.score = evaluator(dev_set, metrics=None)

        # Initialize beam
        beam = [root]
        best_node = root

        for depth in range(self.max_depth):
            candidates = []

            # Expand all nodes in current beam
            for node in beam:
                for operator in self.operators:
                    # Apply transformation
                    new_prompt = operator.apply(node.prompt_text, train_set)

                    # Create child node
                    child = PromptNode(
                        prompt_text=new_prompt,
                        parent=node,
                        operator_used=operator.name
                    )

                    # Evaluate
                    child.score = evaluator(dev_set, metrics=None)
                    node.add_child(child)

                    # Add to candidates
                    candidates.append(child)

                    # Track best
                    if child.score > best_node.score:
                        best_node = child

            # Keep top-k candidates for next beam
            candidates.sort(key=lambda x: x.score, reverse=True)
            beam = candidates[:self.beam_width]

        return best_node
```

#### Random Walk

```python
import random

class RandomWalkOptimizer:
    """Random walk for prompt optimization."""

    def __init__(self,
                 num_steps: int = 5,
                 operators: List[PromptOperator] = None):
        self.num_steps = num_steps
        self.operators = operators or DEFAULT_OPERATORS

    def optimize(self,
                 seed_prompt: str,
                 train_set: List[dspy.Example],
                 dev_set: List[dspy.Example],
                 evaluator: dspy.Evaluate) -> PromptNode:
        """Optimize prompt using random walk."""

        current = PromptNode(prompt_text=seed_prompt)
        current.score = evaluator(dev_set, metrics=None)
        best = current

        for step in range(self.num_steps):
            # Choose random operator
            operator = random.choice(self.operators)

            # Apply transformation
            new_prompt = operator.apply(current.prompt_text, train_set)

            # Create new node
            child = PromptNode(
                prompt_text=new_prompt,
                parent=current,
                operator_used=operator.name
            )

            # Evaluate
            child.score = evaluator(dev_set, metrics=None)
            current.add_child(child)

            # Update if better
            if child.score > best.score:
                best = child

            # Continue from child (random walk)
            current = child

        return best
```

### 4. Evaluation Heuristics

```python
class StringMatchEvaluator:
    """Evaluator for tasks with discrete outputs."""

    def __init__(self, program: dspy.Module):
        self.program = program

    def evaluate(self, dev_set: List[dspy.Example]) -> float:
        """Evaluate using exact string matching."""
        correct = 0
        total = len(dev_set)

        for example in dev_set:
            prediction = self.program(**example.inputs())
            expected = example.outputs()

            # Check if prediction matches expected output
            if str(prediction) == str(expected):
                correct += 1

        return correct / total

class CriticLMEvaluator:
    """Evaluator using a stronger LM as critic."""

    def __init__(self,
                 program: dspy.Module,
                 critic_prompt_template: str = None):
        self.program = program
        self.critic_prompt = critic_prompt_template or self._default_critic_prompt()

    def _default_critic_prompt(self) -> str:
        return """
        Evaluate if the prediction correctly answers the question.

        Requirements for correctness:
        1. Contains all core meaning units from expected output
        2. Does not introduce major unrelated content
        3. Is not excessively longer than expected (max 3x)
        4. Matches expected output format

        Input: {input}
        Expected Output: {expected}
        Model Prediction: {prediction}

        Is this correct? (true/false)
        """

    def evaluate(self, dev_set: List[dspy.Example]) -> float:
        """Evaluate using a critic LM."""
        correct = 0
        total = len(dev_set)

        for example in dev_set:
            prediction = self.program(**example.inputs())
            expected = example.outputs()

            # Get critic judgment
            critic_prompt = self.critic_prompt.format(
                input=str(example.inputs()),
                expected=str(expected),
                prediction=str(prediction)
            )

            response = dspy.Predict(critic_prompt)
            if response.judgment.lower() == 'true':
                correct += 1

        return correct / total
```

### 5. Integrated Optimizer

```python
class StateSpaceOptimizer(dspy.Module):
    """Main state-space prompt optimizer for DSPy."""

    def __init__(self,
                 search_method: str = "beam",
                 beam_width: int = 2,
                 max_depth: int = 2,
                 num_steps: int = 5,
                 eval_type: str = "string_match"):
        super().__init__()

        self.search_method = search_method
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.num_steps = num_steps
        self.eval_type = eval_type

        # Initialize components
        self.operators = DEFAULT_OPERATORS

        if search_method == "beam":
            self.optimizer = BeamSearchOptimizer(beam_width, max_depth)
        elif search_method == "random":
            self.optimizer = RandomWalkOptimizer(num_steps)

    def forward(self,
                signature: dspy.Signature,
                train_set: List[dspy.Example],
                dev_set: List[dspy.Example]) -> Tuple[dspy.Module, dict]:
        """Optimize prompts for the given signature."""

        # Generate seed prompt
        seed_prompt = self._generate_seed_prompt(signature, train_set)

        # Create base program
        base_program = dspy.Predict(signature)

        # Set up evaluator
        if self.eval_type == "string_match":
            evaluator = StringMatchEvaluator(base_program)
        else:
            evaluator = CriticLMEvaluator(base_program)

        # Create evaluation function compatible with dspy.Evaluate
        def eval_function(dev_set, metrics=None):
            return evaluator.evaluate(dev_set)

        # Optimize
        best_node = self.optimizer.optimize(
            seed_prompt=seed_prompt,
            train_set=train_set,
            dev_set=dev_set,
            evaluator=eval_function
        )

        # Create optimized program with best prompt
        optimized_program = dspy.Predict(signature)
        optimized_program.prompt = best_node.prompt_text

        # Return optimization info
        optimization_info = {
            "seed_prompt": seed_prompt,
            "optimized_prompt": best_node.prompt_text,
            "optimization_path": best_node.get_path(),
            "seed_score": None,  # Could be tracked during optimization
            "optimized_score": best_node.score,
            "improvement": best_node.score  # Simplified
        }

        return optimized_program, optimization_info

    def _generate_seed_prompt(self,
                             signature: dspy.Signature,
                             examples: List[dspy.Example]) -> str:
        """Generate initial seed prompt from signature and examples."""
        # Extract signature information
        input_desc = str(signature.with_instructions())

        # Use a few examples
        sample_examples = examples[:3]

        prompt_template = f"""
        Generate a clear, concise instruction prompt for the following task:

        Task Description: {input_desc}

        Here are a few examples of the task:
        {self._format_examples(sample_examples)}

        Requirements for the prompt:
        - Clearly state what the model should do
        - Specify the expected output format
        - Do not include the examples in the prompt itself
        - Keep it concise and unambiguous

        Instruction Prompt:
        """

        response = dspy.Predict(prompt_template)
        return response.instruction_prompt

    def _format_examples(self, examples: List[dspy.Example]) -> str:
        """Format examples for seed prompt generation."""
        formatted = ""
        for i, example in enumerate(examples, 1):
            formatted += f"\nExample {i}:\n"
            formatted += f"Input: {str(example.inputs())}\n"
            formatted += f"Output: {str(example.outputs())}\n"
        return formatted
```

## Usage Examples

### 1. Basic Optimization

```python
import dspy

# Define your task
class SentimentAnalysis(dspy.Signature):
    """Classify text sentiment."""
    text = dspy.InputField(desc="Text to classify")
    sentiment = dspy.OutputField(desc="Positive, negative, or neutral")

# Create datasets
train_set = [dspy.Example(text="Great product!", sentiment="positive"), ...]
dev_set = [dspy.Example(text="Not worth it", sentiment="negative"), ...]

# Initialize optimizer
optimizer = StateSpaceOptimizer(
    search_method="beam",
    beam_width=3,
    max_depth=3
)

# Optimize
optimized_program, info = optimizer.forward(
    signature=SentimentAnalysis,
    train_set=train_set,
    dev_set=dev_set
)

# Use optimized program
result = optimized_program(text="This is amazing!")
print(result.sentiment)
print(f"Optimization path: {info['optimization_path']}")
```

### 2. Custom Operators

```python
class AddChainOfThoughtOperator(PromptOperator):
    """Add chain-of-thought instructions."""

    def __init__(self):
        super().__init__("add_cot")

    def apply(self, prompt: str, context_examples: List[dspy.Example]) -> str:
        cot_instruction = """

        Think step by step before giving your final answer.
        First, analyze what's being asked.
        Then, work through the reasoning.
        Finally, provide the final answer.
        """
        return prompt + cot_instruction

# Use custom operators
custom_optimizer = StateSpaceOptimizer(
    search_method="beam",
    beam_width=2,
    max_depth=2
)
custom_optimizer.operators = DEFAULT_OPERATORS + [AddChainOfThoughtOperator()]
```

### 3. Analyzing Optimization Results

```python
def analyze_optimization_path(best_node: PromptNode):
    """Analyze which operators were most useful."""

    # Count operator frequencies
    operator_counts = {}
    node = best_node

    while node.parent is not None:
        op = node.operator_used
        operator_counts[op] = operator_counts.get(op, 0) + 1
        node = node.parent

    # Print analysis
    print("Operator Usage in Optimization Path:")
    for op, count in sorted(operator_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op}: {count} times")

    print(f"\nTotal score improvement: {best_node.score:.3f}")
    print(f"Optimization depth: {len(best_node.get_path())}")

# Analyze results
analyze_optimization_path(best_node)
```

## Comparison with Other Approaches

| Method | Core Mechanism | Primary Focus | Strengths |
|--------|----------------|----------------|-----------|
| **DSPy** | Generate and refine demonstrations | Few-shot exemplars | Strong with limited data |
| **OPRO** | LLM-driven meta-optimization | Direct prompt rewriting | Leverages LLM understanding |
| **APE** | Sample large candidate sets | Instruction induction | Broad exploration |
| **State-Space** | Local graph search | Instruction refinement | Quantifies operator effectiveness |

## Best Practices

### 1. Search Configuration

```python
# For quick prototyping
quick_config = {
    "search_method": "beam",
    "beam_width": 2,
    "max_depth": 2
}

# For thorough optimization
thorough_config = {
    "search_method": "beam",
    "beam_width": 5,
    "max_depth": 5
}
```

### 2. Preventing Overfitting

```python
# Use cross-validation
def cross_validate_optimization(signature, train_set, k=3):
    """Perform k-fold cross-validation during optimization."""
    fold_size = len(train_set) // k
    scores = []

    for i in range(k):
        # Split data
        val_start = i * fold_size
        val_end = (i + 1) * fold_size

        val_set = train_set[val_start:val_end]
        train_subset = train_set[:val_start] + train_set[val_end:]

        # Optimize
        _, info = optimizer.forward(signature, train_subset, val_set)
        scores.append(info['optimized_score'])

    return sum(scores) / len(scores)
```

### 3. Operator Selection

```python
# Task-specific operator sets
reasoning_operators = [
    MakeConciseOperator(),
    AddExamplesOperator(),
    AddChainOfThoughtOperator(),
    ReorderOperator()
]

generation_operators = [
    MakeVerboseOperator(),
    AddExamplesOperator(),
    ReorderOperator(),
    AddConstraintsOperator()
]
```

## Limitations and Considerations

1. **Computational Cost**: Each evaluation requires LLM inference
2. **Overfitting Risk**: Small dev sets can lead to over-optimization
3. **Operator Quality**: Effectiveness depends on chosen transformations
4. **Evaluation Metrics**: String matching may be too strict for generative tasks

## Future Directions

1. **Learned Operators**: Discover transformations from data
2. **Adaptive Search**: Dynamically adjust search strategy
3. **Multi-objective Optimization**: Balance accuracy, efficiency, and interpretability
4. **Hierarchical Search**: Optimize sub-components independently

## Exercises

1. **Implement Custom Operator**: Create a new transformation operator for a specific task.

2. **Compare Search Strategies**: Run beam search vs. random walk on the same task and compare results.

3. **Operator Analysis**: Track which operators are most successful across different tasks.

4. **Prevent Overfitting**: Implement a regularization strategy to avoid overfitting to dev set.

5. **Multi-step Optimization**: Chain multiple optimizers, first using beam search then fine-tuning with random walk.