# Structured Prompting for Robust Evaluation

## Overview

**Structured Prompting** is a systematic methodology for creating evaluation prompts that ensures consistency, reliability, and robustness in language model assessment. Introduced in late 2024, this approach addresses the variability and inconsistency issues that plague ad-hoc prompt engineering in evaluation scenarios.

The key innovation is the formalization of prompt creation into a structured process that:
- Standardizes prompt components
- Ensures comprehensive coverage of evaluation aspects
- Reduces ambiguity in task instructions
- Enables reproducible evaluation across different models and settings

## Why Structured Prompting Matters

### Problems with Ad-Hoc Prompting

Traditional ad-hoc prompting suffers from several issues:

1. **Inconsistency**: Different evaluators create wildly different prompts
2. **Ambiguity**: Unclear instructions lead to model confusion
3. **Coverage Gaps**: Important aspects of the task may be omitted
4. **Reproducibility**: Difficult to replicate results across setups
5. **Bias**: Unconscious biases in prompt formulation

### Benefits of Structured Prompting

```python
# Ad-hoc approach (problematic)
ad_hoc_prompt = "Tell me about the medical risks in this trial."

# Structured approach (robust)
structured_prompt = """
Task: Risk Assessment Evaluation

Context: You are evaluating a medical research paper for potential risks.
Please analyze the following randomized controlled trial (RCT).

Instructions:
1. Identify all potential risks mentioned
2. Categorize risks by severity (mild/moderate/severe)
3. Note the frequency of each risk
4. Assess if risks are adequately addressed
5. Provide a confidence score for your assessment

Format your response as:
- Risk Category: [Name] - Frequency - Severity
- Overall Assessment: [Summary]
- Confidence Score: [0-1]

Trial Text: {trial_text}
"""
```

## The Structured Prompting Framework

### Core Components

A structured prompt consists of five essential components:

1. **Task Definition**: Clear specification of what to evaluate
2. **Context Setting**: Background information and role definition
3. **Explicit Instructions**: Step-by-step guidance
4. **Output Format**: Precise formatting requirements
5. **Examples**: Demonstration of expected responses

### Implementation in DSPy

```python
import dspy
from typing import Dict, List, Optional

class StructuredPromptEvaluator(dspy.Module):
    """Base class for structured prompting evaluators."""

    def __init__(self, task_spec: Dict):
        super().__init__()
        self.task_spec = task_spec
        self.prompt_template = self._build_structured_prompt()

    def _build_structured_prompt(self) -> str:
        """Build a structured prompt from task specification."""
        components = []

        # Task Definition
        components.append(f"Task: {self.task_spec['task_name']}")
        components.append(f"Objective: {self.task_spec['objective']}")

        # Context Setting
        if 'context' in self.task_spec:
            components.append(f"Context: {self.task_spec['context']}")

        # Instructions
        components.append("\nInstructions:")
        for i, instruction in enumerate(self.task_spec['instructions'], 1):
            components.append(f"{i}. {instruction}")

        # Output Format
        components.append("\nOutput Format:")
        components.append(self.task_spec['output_format'])

        # Examples (if provided)
        if 'examples' in self.task_spec:
            components.append("\nExamples:")
            for example in self.task_spec['examples']:
                components.append(f"Input: {example['input']}")
                components.append(f"Output: {example['output']}\n")

        # Input placeholder
        components.append("\nInput: {input}")

        return "\n".join(components)

    def forward(self, **kwargs):
        """Execute the structured prompt."""
        prompt = self.prompt_template.format(**kwargs)
        return dspy.Predict(prompt)

# Example: Medical Risk Assessment
medical_risk_spec = {
    "task_name": "Medical Risk Assessment",
    "objective": "Evaluate potential risks in medical research papers",
    "context": "You are a medical safety officer reviewing clinical trials.",
    "instructions": [
        "Identify all potential risks and side effects mentioned",
        "Categorize each risk by severity (mild/moderate/severe)",
        "Note the frequency or percentage of each risk",
        "Assess if adequate monitoring is described",
        "Identify any missing safety considerations"
    ],
    "output_format": """
Risk Assessment Report:
{risk_summary}

Severity Breakdown:
- Mild: {mild_risks}
- Moderate: {moderate_risks}
- Severe: {severe_risks}

Safety Assessment: {safety_assessment}
Confidence Score: [0-1]
""",
    "examples": [
        {
            "input": "Trial reported headache in 15% of participants...",
            "output": """Risk Assessment Report:
- Headache: 15% - Mild
- Nausea: 8% - Mild
- Elevated liver enzymes: 2% - Moderate

Severity Breakdown:
- Mild: Headache, Nausea
- Moderate: Elevated liver enzymes
- Severe: None identified

Safety Assessment: Adequate monitoring described for liver enzymes
Confidence Score: 0.9"""
        }
    ]
}

evaluator = StructuredPromptEvaluator(medical_risk_spec)
```

## Advanced Structured Prompting Techniques

### 1. Template-Based Prompt Generation

```python
class PromptTemplate:
    """Template system for generating structured prompts."""

    def __init__(self, template_type: str):
        self.template_type = template_type
        self.templates = self._load_templates()

    def generate_prompt(self, task_config: Dict) -> str:
        """Generate a structured prompt from configuration."""
        template = self.templates[self.template_type]

        # Fill template with task-specific content
        prompt = template.format(**task_config)

        # Add task-specific adaptations
        if self.template_type == "classification":
            prompt = self._add_classification_guidelines(prompt, task_config)
        elif self.template_type == "generation":
            prompt = self._add_generation_constraints(prompt, task_config)

        return prompt

    def _add_classification_guidelines(self, prompt: str, config: Dict) -> str:
        """Add specific guidelines for classification tasks."""
        guidelines = "\n\nClassification Guidelines:\n"
        guidelines += "- Consider all possible categories\n"
        guidelines += "- Provide reasoning for your choice\n"
        guidelines += "- Assign confidence scores\n"

        if 'categories' in config:
            guidelines += "\nValid Categories:\n"
            for cat in config['categories']:
                guidelines += f"- {cat}: {cat['description']}\n"

        return prompt + guidelines

    def _add_generation_constraints(self, prompt: str, config: Dict) -> str:
        """Add specific constraints for generation tasks."""
        constraints = "\n\nGeneration Constraints:\n"

        if 'length' in config:
            constraints += f"- Length: {config['length']} words\n"

        if 'style' in config:
            constraints += f"- Style: {config['style']}\n"

        if 'include_elements' in config:
            constraints += "- Must include:\n"
            for element in config['include_elements']:
                constraints += f"  * {element}\n"

        return prompt + constraints

# Usage example
template_system = PromptTemplate("classification")

classification_config = {
    "task_name": "Sentiment Classification",
    "objective": "Classify text sentiment",
    "categories": [
        {"name": "positive", "description": "Expressing positive emotions"},
        {"name": "negative", "description": "Expressing negative emotions"},
        {"name": "neutral", "description": "No strong emotion expressed"}
    ],
    "input_text": "The product exceeded my expectations!"
}

prompt = template_system.generate_prompt(classification_config)
```

### 2. Modular Prompt Components

```python
class PromptComponent:
    """Base class for reusable prompt components."""

    def __init__(self, name: str):
        self.name = name

    def render(self, context: Dict) -> str:
        """Render the component with given context."""
        raise NotImplementedError

class TaskDefinition(PromptComponent):
    """Component for defining the evaluation task."""

    def __init__(self, task_name: str, objective: str):
        super().__init__("task_definition")
        self.task_name = task_name
        self.objective = objective

    def render(self, context: Dict) -> str:
        return f"""Task: {self.task_name}
Objective: {self.objective}"""

class InstructionBlock(PromptComponent):
    """Component for structured instructions."""

    def __init__(self, instructions: List[str]):
        super().__init__("instructions")
        self.instructions = instructions

    def render(self, context: Dict) -> str:
        instruction_text = "\n".join(
            f"{i+1}. {inst}" for i, inst in enumerate(self.instructions)
        )
        return f"Instructions:\n{instruction_text}"

class OutputFormat(PromptComponent):
    """Component for specifying output format."""

    def __init__(self, format_spec: str):
        super().__init__("output_format")
        self.format_spec = format_spec

    def render(self, context: Dict) -> str:
        return f"Output Format:\n{self.format_spec}"

class StructuredPromptBuilder:
    """Builder for assembling structured prompts from components."""

    def __init__(self):
        self.components = []

    def add_component(self, component: PromptComponent):
        """Add a component to the prompt."""
        self.components.append(component)
        return self

    def build(self, context: Optional[Dict] = None) -> str:
        """Build the complete structured prompt."""
        if context is None:
            context = {}

        parts = []
        for component in self.components:
            parts.append(component.render(context))

        return "\n\n".join(parts)

# Example: Building a complex evaluation prompt
builder = StructuredPromptBuilder()

builder.add_component(TaskDefinition(
    "Medical Literature Review",
    "Extract and categorize adverse events from clinical trials"
))

builder.add_component(InstructionBlock([
    "Read the entire trial report carefully",
    "Identify all mentioned adverse events",
    "Categorize by type (e.g., cardiovascular, neurological)",
    "Note severity and frequency for each event",
    "Highlight any unexpected or severe events"
]))

builder.add_component(OutputFormat("""
Adverse Event Summary:
- Event Name: [Type] - Frequency - Severity
- Total Events: [count]
- Most Common: [event]
- Most Severe: [event]

Assessment: [overall safety assessment]
"""))

prompt = builder.build()
```

## Structured Prompting for Different Evaluation Types

### 1. Classification Evaluation

```python
class ClassificationEvaluator(dspy.Module):
    """Structured evaluator for classification tasks."""

    def __init__(self, categories: List[str], description: str):
        super().__init__()
        self.categories = categories
        self.description = description
        self.evaluator = self._build_evaluator()

    def _build_evaluator(self):
        """Build the structured evaluation prompt."""
        prompt_template = f"""
Classification Task: {self.description}

Categories:
{self._format_categories()}

Evaluation Instructions:
1. Analyze the input text thoroughly
2. Consider each category carefully
3. Select the most appropriate category
4. Provide reasoning for your choice
5. Assign a confidence score (0-1)

Input: {{input}}

Output Format:
Category: [selected category]
Reasoning: [detailed explanation]
Confidence: [0-1]
"""
        return dspy.Predict(prompt_template)

    def _format_categories(self) -> str:
        """Format categories for display."""
        return "\n".join(f"- {cat}" for cat in self.categories)

    def forward(self, input_text: str):
        return self.evaluator(input=input_text)

# Usage
sentiment_evaluator = ClassificationEvaluator(
    categories=["positive", "negative", "neutral"],
    description="Classify the sentiment of the given text"
)
```

### 2. Generation Quality Evaluation

```python
class GenerationEvaluator(dspy.Module):
    """Structured evaluator for generated text quality."""

    def __init__(self, criteria: List[str]):
        super().__init__()
        self.criteria = criteria
        self.evaluator = self._build_evaluator()

    def _build_evaluator(self):
        """Build the structured evaluation prompt."""
        criteria_text = "\n".join(
            f"- {criterion}" for criterion in self.criteria
        )

        prompt_template = f"""
Text Quality Evaluation

Evaluation Criteria:
{criteria_text}

Instructions:
1. Read the original prompt and generated response
2. Evaluate the response against each criterion
3. Score each criterion (1-5, where 5 is excellent)
4. Provide specific feedback for improvement
5. Calculate overall quality score

Original Prompt: {{prompt}}
Generated Response: {{response}}

Evaluation Format:
Criterion Scores:
{self._criterion_format()}

Overall Score: [average of criteria]
Strengths: [list of positive aspects]
Improvements: [specific suggestions]
"""
        return dspy.Predict(prompt_template)

    def _criterion_format(self) -> str:
        """Generate criterion evaluation format."""
        return "\n".join(
            f"- {criterion}: [1-5] - [brief justification]"
            for criterion in self.criteria
        )

    def forward(self, prompt: str, response: str):
        return self.evaluator(prompt=prompt, response=response)

# Usage
quality_evaluator = GenerationEvaluator([
    "relevance", "coherence", "accuracy", "completeness", "clarity"
])
```

### 3. Comparison Evaluation

```python
class ComparisonEvaluator(dspy.Module):
    """Structured evaluator for comparing multiple outputs."""

    def __init__(self, comparison_aspects: List[str]):
        super().__init__()
        self.comparison_aspects = comparison_aspects
        self.evaluator = self._build_evaluator()

    def _build_evaluator(self):
        """Build the structured comparison prompt."""
        aspects_text = "\n".join(
            f"- {aspect}" for aspect in self.comparison_aspects
        )

        prompt_template = f"""
Response Comparison Analysis

Comparison Aspects:
{aspects_text}

Instructions:
1. Examine all responses carefully
2. Compare responses on each aspect
3. Identify strengths and weaknesses of each
4. Rank responses from best to worst
5. Provide justification for rankings

Original Prompt: {{prompt}}
Response A: {{response_a}}
Response B: {{response_b}}
Response C: {{response_c}}

Comparison Format:
Aspect-by-Aspect Analysis:
{self._comparison_format()}

Ranking:
1. [Response]: [justification]
2. [Response]: [justification]
3. [Response]: [justification]

Overall Recommendation: [which response to use]
"""
        return dspy.Predict(prompt_template)

    def _comparison_format(self) -> str:
        """Generate comparison analysis format."""
        return "\n".join(
            f"- {aspect}: A [score] vs B [score] vs C [score] - [analysis]"
            for aspect in self.comparison_aspects
        )

    def forward(self, prompt: str, responses: List[str]):
        # Ensure we have exactly 3 responses for the template
        while len(responses) < 3:
            responses.append("")

        return self.evaluator(
            prompt=prompt,
            response_a=responses[0],
            response_b=responses[1],
            response_c=responses[2]
        )
```

## Best Practices for Structured Prompting

### 1. Clear Task Decomposition

```python
# Good: Break down complex tasks
task_breakdown = {
    "main_task": "Evaluate medical paper quality",
    "subtasks": [
        "Check methodology soundness",
        "Verify statistical analysis",
        "Assess clinical significance",
        "Evaluate generalizability"
    ]
}

# Poor: Vague single instruction
vague_task = "Evaluate if the paper is good"
```

### 2. Explicit Output Specifications

```python
# Good: Precise formatting requirements
output_spec = """
Findings Report:
- Study Design: [type] - [quality score 1-5]
- Sample Size: [n] - [adequacy assessment]
- Statistical Methods: [methods] - [appropriateness]
- Bias Risk: [low/medium/high] - [justification]
- Overall Quality: [score 1-10] - [summary]
"""

# Poor: Unclear output expectations
vague_output = "Tell me about the study quality"
```

### 3. Comprehensive Coverage

```python
# Good: Check all important aspects
evaluation_aspects = [
    "methodological rigor",
    "statistical validity",
    "clinical relevance",
    "ethical considerations",
    "limitations and weaknesses",
    "conclusions justification"
]
```

### 4. Contextual Grounding

```python
# Good: Provide relevant context
context = """
You are an expert clinical trial reviewer with 15 years of experience.
Your role is to assess trial quality for publication in a top-tier journal.
Consider current standards in clinical research methodology.
"""
```

## Integration with DSPy Evaluation

### Structured Evaluation Metrics

```python
class StructuredMetric(dspy.Metric):
    """Custom metric for evaluating structured prompt outputs."""

    def __init__(self, structure_validator, content_evaluator):
        self.structure_validator = structure_validator
        self.content_evaluator = content_evaluator

    def __call__(self, example, pred, trace=None):
        """Evaluate both structure and content quality."""
        # Check if output follows required structure
        structure_score = self.structure_validator(pred.output)

        # Evaluate content quality
        content_score = self.content_evaluator(
            example=example,
            prediction=pred.output
        )

        # Combine scores
        total_score = 0.6 * structure_score + 0.4 * content_score
        return total_score

# Usage in evaluation
structured_metric = StructuredMetric(
    structure_validator=validate_output_format,
    content_evaluator=evaluate_content_quality
)

evaluate = dspy.Evaluate(
    devset=test_set,
    metric=structured_metric,
    num_threads=4
)
```

## Exercises

1. **Create a Structured Prompt**: Design a structured prompt for evaluating code quality. Include all five core components.

2. **Template System**: Implement a template-based system for generating structured prompts for different tasks (classification, generation, comparison).

3. **Component Reuse**: Create reusable prompt components that can be mixed and matched for different evaluation scenarios.

4. **Metric Integration**: Build a custom DSPy metric that evaluates both the structure and content of model responses to structured prompts.

5. **Comparative Analysis**: Compare evaluation results from structured vs. ad-hoc prompts on the same dataset to quantify the improvement.