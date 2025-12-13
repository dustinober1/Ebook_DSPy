# CustomMIPROv2: Enhanced Multi-Stage Prompt Optimization

## Overview

**CustomMIPROv2** is an enhanced version of the MIPROv2 optimizer that addresses real-world production needs through a two-stage optimization process and explicit constraint handling. This optimizer was developed through extensive multi-use case studies and demonstrates significant improvements in complex tasks like routing agents, prompt evaluation, and code generation.

## Key Enhancements

1. **Two-Stage Instruction Generation**: Separates constraint extraction from instruction generation for better focus
2. **Explicit Constraint Handling**: Users can provide domain-specific constraints and optimization tips
3. **Mini-Batch Evaluation**: Efficient evaluation using representative subsets
4. **Context-Aware Optimization**: Better handling of long conversations and complex contexts
5. **Production-Ready Extraction**: Optimized prompts designed for extraction from DSPy framework

## Architecture

### 1. Two-Stage Optimization Process

```python
import dspy
from typing import List, Dict, Optional, Tuple
import random
from dataclasses import dataclass

@dataclass
class OptimizationConstraint:
    """Represents a constraint for prompt optimization."""
    name: str
    description: str
    priority: str  # HIGH, MEDIUM, LOW
    examples: List[str] = None

class CustomMIPROv2:
    """Enhanced MIPROv2 optimizer with two-stage optimization."""

    def __init__(self,
                 teacher_model: str = "gpt-4",
                 student_model: str = "gpt-4o-mini",
                 num_trials: int = 15,
                 mini_batch_size: int = 15,
                 temperature: float = 0.5):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.num_trials = num_trials
        self.mini_batch_size = mini_batch_size
        self.temperature = temperature

        # Optimization stages
        self.constraint_extractor = dspy.Predict(
            """Analyze the provided task demonstrations and extract key constraints
            and edge cases that the optimized instruction should handle.

            Task Demonstrations:
            {demonstrations}

            Program Context:
            {program_context}

            Extract:
            1. Critical constraints (must-follow rules)
            2. Edge cases to consider
            3. Common failure patterns
            4. Important contextual factors

            Constraints and Edge Cases:"""
        )

        self.instruction_generator = dspy.ChainOfThought(
            """Generate an optimized instruction based on constraints and examples.

            Task Description: {task_description}
            Constraints: {constraints}
            Edge Cases: {edge_cases}
            Tips/Guidance: {tips}

            Requirements:
            - Address all high-priority constraints
            - Handle identified edge cases
            - Follow provided tips when applicable
            - Keep instruction clear and concise
            - Maintain consistency with examples

            Optimized Instruction:"""
        )

    def compile(self,
                program: dspy.Module,
                trainset: List[dspy.Example],
                valset: List[dspy.Example],
                metric: callable,
                tips: Optional[List[str]] = None,
                constraints: Optional[List[OptimizationConstraint]] = None) -> dspy.Module:
        """Compile the program with enhanced optimization."""

        # Store references
        self.program = program
        self.trainset = trainset
        self.valset = valset
        self.metric = metric
        self.tips = tips or []
        self.constraints = constraints or []

        # Stage 1: Extract constraints from demonstrations
        print("Stage 1: Extracting constraints and edge cases...")
        extracted_constraints = self._extract_constraints_from_demos()

        # Combine user constraints with extracted ones
        all_constraints = self._combine_constraints(extracted_constraints, constraints)

        # Stage 2: Generate and evaluate optimized instructions
        print("Stage 2: Generating optimized instructions...")
        best_instruction = self._optimize_instructions(all_constraints, tips)

        # Create and return optimized program
        optimized_program = self._create_optimized_program(best_instruction)

        return optimized_program

    def _extract_constraints_from_demos(self) -> Dict:
        """Extract constraints from training demonstrations."""

        # Sample demonstrations for analysis
        demo_samples = random.sample(self.trainset, min(10, len(self.trainset)))

        # Extract program context
        program_context = self._analyze_program_structure()

        # Format demonstrations for analysis
        demo_text = "\n".join([
            f"Input: {demo.inputs()}\nOutput: {demo.outputs()}"
            for demo in demo_samples
        ])

        # Extract constraints
        extraction_result = self.constraint_extractor(
            demonstrations=demo_text,
            program_context=program_context
        )

        # Parse and structure the extraction
        constraints = self._parse_constraint_extraction(extraction_result)

        return constraints

    def _optimize_instructions(self,
                             constraints: Dict,
                             tips: List[str]) -> str:
        """Generate and evaluate multiple instruction candidates."""

        # Create mini-batches for evaluation
        mini_batches = self._create_mini_batches(self.valset, self.mini_batch_size)

        best_instruction = None
        best_score = 0.0

        for trial in range(self.num_trials):
            print(f"Trial {trial + 1}/{self.num_trials}")

            # Generate instruction candidate
            instruction_candidate = self._generate_instruction_candidate(
                constraints, tips, trial
            )

            # Evaluate on mini-batches
            avg_score = 0.0
            for batch_idx, batch in enumerate(mini_batches):
                # Create temporary program with new instruction
                temp_program = self._create_temp_program(instruction_candidate)

                # Evaluate on this batch
                batch_score = self._evaluate_on_batch(temp_program, batch)
                avg_score += batch_score

            avg_score /= len(mini_batches)

            print(f"  Score: {avg_score:.3f}")

            # Update best if improved
            if avg_score > best_score:
                best_score = avg_score
                best_instruction = instruction_candidate
                print(f"  New best instruction found!")

        print(f"\nOptimization complete. Best score: {best_score:.3f}")
        return best_instruction

    def _generate_instruction_candidate(self,
                                      constraints: Dict,
                                      tips: List[str],
                                      trial: int) -> str:
        """Generate a single instruction candidate."""

        # Select random tips for variety
        selected_tips = random.sample(
            tips + ["Focus on clarity and conciseness"],
            min(2, len(tips) + 1)
        )

        # Get random demonstration for reference
        demo = random.choice(self.trainset[:5])

        # Generate instruction
        result = self.instruction_generator(
            task_description=self._get_task_description(),
            constraints=self._format_constraints(constraints),
            edge_cases=constraints.get('edge_cases', []),
            tips="\n".join([f"- {tip}" for tip in selected_tips])
        )

        return result.optimized_instruction

    def _evaluate_on_batch(self, program: dspy.Module, batch: List[dspy.Example]) -> float:
        """Evaluate program on a mini-batch."""

        total_score = 0.0
        valid_examples = 0

        for example in batch:
            try:
                # Get prediction
                prediction = program(**example.inputs())

                # Evaluate
                score = self.metric(example, prediction, trace=None)
                total_score += score
                valid_examples += 1

            except Exception as e:
                print(f"    Error evaluating example: {e}")
                continue

        return total_score / valid_examples if valid_examples > 0 else 0.0

    def _create_optimized_program(self, best_instruction: str) -> dspy.Module:
        """Create the final optimized program."""

        # Clone the original program
        optimized_program = self._clone_program(self.program)

        # Update all Predict modules with optimized instruction
        for name, module in optimized_program.named_modules():
            if isinstance(module, dspy.Predict):
                module.update(instruction=best_instruction)

        return optimized_program
```

### 2. Routing Agent Optimization Example

```python
class RoutingAgentOptimizer:
    """Example: Optimizing a routing agent using CustomMIPROv2."""

    def __init__(self):
        # Define the routing signature
        class RouterSignature(dspy.Signature):
            """Read the conversation and select the next role from roles_list
            to play. Only return the role."""
            conversation = dspy.InputField(desc="Current conversation history")
            roles_list = dspy.InputField(desc="List of available roles")
            roles = dspy.InputField(desc="Role descriptions")
            selected_role = dspy.OutputField(
                desc="Selected role from the list"
            )

        self.signature = RouterSignature

        # Base program
        self.base_program = dspy.Predict(self.signature)

        # Training data (conversations with correct role selections)
        self.trainset = self._load_routing_examples("routing_train.json")
        self.valset = self._load_routing_examples("routing_val.json")

    def optimize_routing_agent(self) -> dspy.Module:
        """Optimize the routing agent for better performance."""

        # Define domain-specific constraints
        routing_constraints = [
            OptimizationConstraint(
                name="task_completion",
                description="If the last role didn't complete their task, they must be selected again",
                priority="HIGH"
            ),
            OptimizationConstraint(
                name="conversation_flow",
                description="Consider the flow and tone when selecting roles",
                priority="MEDIUM"
            ),
            OptimizationConstraint(
                name="role_availability",
                description="Only select from the provided roles list",
                priority="HIGH"
            )
        ]

        # Define optimization tips
        optimization_tips = [
            "The model should be aware of conversation context and tone",
            "Consider the current state of task completion",
            "Match role selection to conversation needs",
            "Maintain conversation coherence and flow"
        ]

        # Define evaluation metric
        def routing_metric(example, prediction, trace=None):
            """Evaluate routing accuracy."""
            expected_role = example.outputs()['selected_role']
            predicted_role = prediction.get('selected_role', '')

            return 1.0 if predicted_role == expected_role else 0.0

        # Initialize CustomMIPROv2
        optimizer = CustomMIPROv2(
            teacher_model="gpt-4",
            student_model="gpt-4o-mini",
            num_trials=12,
            mini_batch_size=15,
            temperature=0.5
        )

        # Compile optimized program
        optimized_program = optimizer.compile(
            program=self.base_program,
            trainset=self.trainset,
            valset=self.valset,
            metric=routing_metric,
            tips=optimization_tips,
            constraints=routing_constraints
        )

        return optimized_program

    def _load_routing_examples(self, file_path: str) -> List[dspy.Example]:
        """Load routing examples from file."""
        # Implementation depends on data format
        examples = []

        # Sample data structure
        sample_data = {
            "conversation": "User: I need help with the report\nAdmin: I'll help you with that",
            "roles": ["Human_Administrator", "Project_Manager", "Software_Engineer"],
            "selected_role": "Human_Administrator"
        }

        # Convert to DSPy examples
        for item in load_json(file_path):
            example = dspy.Example(
                conversation=item["conversation"],
                roles_list=", ".join(item["roles"]),
                roles=item["roles"]
            ).with_outputs(selected_role=item["selected_role"])
            examples.append(example)

        return examples
```

### 3. Prompt Evaluator Optimization

```python
class PromptEvaluatorOptimizer:
    """Example: Optimizing a prompt evaluator for contradiction detection."""

    def __init__(self):
        # Define evaluation signature
        class ContradictionSignature(dspy.Signature):
            """Evaluate the prompt on a scale from 0.0 (high contradiction)
            to 1.0 (no contradiction) based on internal consistency."""
            prompt = dspy.InputField(desc="The prompt to evaluate")
            score = dspy.OutputField(desc="Score between 0.0 and 1.0")
            explanation = dspy.OutputField(desc="Explanation of the score")

        self.signature = ContradictionSignature

        # Load contradiction detection dataset
        self.trainset = self._load_contradiction_examples("contradictions_train.json")
        self.valset = self._load_contradiction_examples("contradictions_val.json")

    def optimize_contradiction_detector(self) -> dspy.Module:
        """Optimize contradiction detection with specific constraints."""

        # Define contradiction-specific constraints
        contradiction_constraints = [
            OptimizationConstraint(
                name="format_contradiction",
                description="Check if output format conflicts with instructions",
                priority="HIGH",
                examples=["Instructions say 'no examples' but examples are provided"]
            ),
            OptimizationConstraint(
                name="instruction_contradiction",
                description="Check for conflicting instructions",
                priority="HIGH",
                examples=["Do X AND Don't do X"]
            ),
            OptimizationConstraint(
                name="example_contradiction",
                description="Check if examples don't follow instructions",
                priority="MEDIUM",
                examples=["Example shows different format than instructed"]
            )
        ]

        # Custom tip for contradiction detection
        contradiction_tips = [
            "Carefully examine all instruction pairs for conflicts",
            "Verify that examples strictly follow the instructions",
            "Check for ambiguous or conflicting requirements",
            "Score 0.0 if ANY contradiction is found"
        ]

        # Evaluation metric
        def contradiction_metric(example, prediction, trace=None):
            """Evaluate contradiction detection accuracy."""
            predicted_score = float(prediction.get('score', 0.5))
            expected_label = example.outputs()['has_contradiction']
            predicted_label = predicted_score < 0.6

            return 1.0 if predicted_label == expected_label else 0.0

        # Initialize optimizer
        optimizer = CustomMIPROv2(
            num_trials=10,
            mini_batch_size=10,
            temperature=0.5
        )

        # Optimize
        optimized_detector = optimizer.compile(
            program=dspy.Predict(self.signature),
            trainset=self.trainset,
            valset=self.valset,
            metric=contradiction_metric,
            tips=contradiction_tips,
            constraints=contradiction_constraints
        )

        return optimized_detector
```

### 4. Code Generation Optimization

```python
class CodeGenerationOptimizer:
    """Example: Optimizing code generation with CustomMIPROv2."""

    def __init__(self):
        # Code generation signature
        class CodeGenSignature(dspy.Signature):
            """Generate pandas code to answer the user's question."""
            question = dspy.InputField(desc="User's data analysis question")
            columns = dspy.InputField(desc="Available columns and sample values")
            code = dspy.OutputField(desc="Generated pandas code")

        self.signature = CodeGenSignature

        # Load code generation dataset
        self.trainset = self._load_code_examples("code_train.json")
        self.valset = self._load_code_examples("code_val.json")

    def optimize_code_generator(self) -> dspy.Module:
        """Optimize code generation with quality constraints."""

        # Code quality constraints
        code_constraints = [
            OptimizationConstraint(
                name="executable_code",
                description="Generated code must be syntactically correct and executable",
                priority="HIGH"
            ),
            OptimizationConstraint(
                name="efficiency",
                description="Code should be efficient and not overly complex",
                priority="MEDIUM"
            ),
            OptimizationConstraint(
                name="relevance",
                description="Code must directly address the user's question",
                priority="HIGH"
            ),
            OptimizationConstraint(
                name="readability",
                description="Include appropriate comments and clear structure",
                priority="LOW"
            )
        ]

        # Code generation tips
        code_tips = [
            "Use pandas built-in methods when possible",
            "Handle potential errors or edge cases",
            "Keep code concise but complete",
            "Add minimal but helpful comments"
        ]

        # LLM-as-a-Judge for code evaluation
        def code_quality_metric(example, prediction, trace=None):
            """Evaluate generated code quality using LLM judge."""

            code = prediction.get('code', '')
            question = example.inputs()['question']

            # Create judge prompt
            judge = dspy.ChainOfThought(
                """Evaluate the generated code for the given question.

                Question: {question}
                Generated Code: {code}

                Evaluate on:
                1. Correctness (0-1): Does it solve the problem correctly?
                2. Efficiency (0-1): Is it reasonably efficient?
                3. Readability (0-1): Is it well-structured?

                Overall Score (0-1):"""
            )

            result = judge(question=question, code=code)
            score = float(result.overall_score) if hasattr(result, 'overall_score') else 0.5

            return score

        # Initialize optimizer
        optimizer = CustomMIPROv2(
            num_trials=15,
            mini_batch_size=20,
            temperature=0.3
        )

        # Optimize
        optimized_generator = optimizer.compile(
            program=dspy.Predict(self.signature),
            trainset=self.trainset,
            valset=self.valset,
            metric=code_quality_metric,
            tips=code_tips,
            constraints=code_constraints
        )

        return optimized_generator
```

## Implementation Guide

### 1. Basic Usage

```python
# Define your DSPy program
class MyProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.predict(question=question)

# Create training and validation sets
trainset = [...]
valset = [...]

# Define evaluation metric
def my_metric(example, prediction, trace=None):
    # Your metric logic
    return 1.0 if prediction.answer == example.answer else 0.0

# Initialize CustomMIPROv2
optimizer = CustomMIPROv2(
    teacher_model="gpt-4",
    student_model="gpt-4o-mini",
    num_trials=20,
    mini_batch_size=15
)

# Optimize
optimized_program = optimizer.compile(
    program=MyProgram(),
    trainset=trainset,
    valset=valset,
    metric=my_metric
)
```

### 2. Using Constraints and Tips

```python
# Define constraints
constraints = [
    OptimizationConstraint(
        name="safety",
        description="Never provide harmful or unsafe content",
        priority="HIGH"
    ),
    OptimizationConstraint(
        name="clarity",
        description="Use clear and unambiguous language",
        priority="MEDIUM"
    )
]

# Define tips
tips = [
    "Consider safety implications before answering",
    "Provide structured responses when possible",
    "Acknowledge uncertainty when appropriate"
]

# Compile with constraints and tips
optimized = optimizer.compile(
    program=MyProgram(),
    trainset=trainset,
    valset=valset,
    metric=my_metric,
    tips=tips,
    constraints=constraints
)
```

### 3. Extracting and Using Optimized Prompts

```python
# Get the optimized instruction
optimized_instruction = optimized_program.predict.instruction

# Use outside DSPy (with caution)
def use_optimized_prompt_elsewhere():
    import openai

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": optimized_instruction},
            {"role": "user", "content": "Your input here"}
        ]
    )

    return response.choices[0].message.content

# Note: Performance may vary outside DSPy context
```

## Key Results from Paper

1. **Routing Agent**: Improved from 85.71% to 90.47% accuracy (5% absolute improvement)
2. **Prompt Evaluator**: Improved from 46.2% to 76.9% accuracy (30% absolute improvement)
3. **Code Generation**: Achieved 90% accuracy with optimized prompts
4. **Jailbreak Detection**: Maintained perfect recall while improving precision
5. **Hallucination Detection**: Up to 82% accuracy with optimized examples

## Best Practices

1. **Clear Constraints**: Define specific, actionable constraints with examples
2. **Domain-Specific Tips**: Provide tips that are relevant to your task domain
3. **Appropriate Mini-Batch Size**: Balance evaluation cost with accuracy
4. **Sufficient Trials**: Use enough trials to explore the instruction space
5. **Metric Design**: Ensure metrics capture important aspects of performance

## Advanced Features

### 1. Constraint Prioritization

```python
class PrioritizedCustomMIPROv2(CustomMIPROv2):
    """Enhanced version with constraint prioritization."""

    def _format_constraints(self, constraints: Dict) -> str:
        """Format constraints with priority information."""

        formatted = []
        for constraint in constraints.get('high_priority', []):
            formatted.append(f"MUST: {constraint}")

        for constraint in constraints.get('medium_priority', []):
            formatted.append(f"SHOULD: {constraint}")

        for constraint in constraints.get('low_priority', []):
            formatted.append(f"COULD: {constraint}")

        return "\n".join(formatted)
```

### 2. Adaptive Trial Management

```python
class AdaptiveCustomMIPROv2(CustomMIPROv2):
    """Version with adaptive trial management."""

    def optimize_instructions(self, constraints, tips):
        """Optimize with early stopping based on improvement."""

        best_score = 0.0
        no_improvement_count = 0
        max_no_improvement = 5

        for trial in range(self.num_trials):
            # Generate and evaluate candidate
            candidate = self._generate_instruction_candidate(constraints, tips, trial)
            score = self._evaluate_candidate(candidate)

            # Check for improvement
            if score > best_score:
                best_score = score
                best_instruction = candidate
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= max_no_improvement:
                print(f"Early stopping at trial {trial + 1}")
                break

        return best_instruction
```

## Limitations and Considerations

1. **Extraction Overhead**: Two-stage process increases optimization time
2. **Constraint Quality**: Poorly defined constraints can hurt performance
3. **Mini-Batch Representativeness**: Small batches may not represent full validation set
4. **Context Transfer**: Optimized prompts may perform differently outside DSPy
5. **Compute Cost**: Multiple trials increase API costs

## Conclusion

CustomMIPROv2 addresses practical challenges in prompt optimization for production systems. By separating constraint extraction from instruction generation and providing explicit control through constraints and tips, it enables more targeted and effective optimization. The framework demonstrates that systematic optimization can significantly improve performance across diverse tasks, from routing agents to code generation, making it a valuable tool for real-world DSPy applications.