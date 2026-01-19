# Assertions Module

## Prerequisites

- **Previous Section**: [Composing Modules](./06-composing-modules.md) - Understanding of module composition
- **Chapter 2**: Signatures - Strong familiarity with signature design
- **Required Knowledge**: Constraint validation, error handling patterns
- **Difficulty Level**: Advanced
- **Estimated Reading Time**: 60 minutes

## Learning Objectives

By the end of this section, you will:
- Master the `dspy.Assert` and `dspy.Suggest` constraint system
- Learn to implement runtime validation for AI outputs
- Build self-refining pipelines with automatic error recovery
- Understand the computational constraints framework
- Design robust AI applications with guaranteed output quality

## Introduction to Assertions

Assertions in DSPy provide a powerful mechanism for ensuring the quality and correctness of AI-generated outputs. They act as runtime validators that check if the model's output meets specified constraints, and can automatically trigger refinement when constraints are violated.

### Why Assertions Matter

**Without Assertions:**
```python
# Brittle - no validation
qa = dspy.Predict("question -> answer")
result = qa(question="What is 2+2?")
# Model might return "4", "Four", "The answer is 4", or even hallucinate
```

**With Assertions:**
```python
# Robust - guaranteed format and correctness
qa = dspy.Predict("question -> answer")

def validate_numeric_answer(example, pred, trace=None):
    # Check if answer is a number
    assert pred.answer.isdigit(), "Answer must be numeric"
    # Check if it's actually correct
    assert int(pred.answer) == 4, "Answer must be correct"
    return True

# Configure assertion
qa = dspy.Assert(
    qa,
    validation_fn=validate_numeric_answer,
    max_attempts=3
)

result = qa(question="What is 2+2?")
# Guaranteed: result.answer is "4"
```

## Core Assertion Types

### 1. dspy.Assert - Hard Constraints

`dspy.Assert` enforces strict constraints that must be satisfied. If a constraint fails, the system automatically retries with refined instructions.

```python
import dspy

class CodeGenerator(dspy.Signature):
    """Generate Python code for the given task."""
    task = dspy.InputField(desc="Programming task to implement", type=str)
    code = dspy.OutputField(desc="Valid Python code", type=str)

# Create the module
coder = dspy.ChainOfThought(CodeGenerator)

# Define assertion function
def validate_syntax(example, pred, trace=None):
    """Ensure generated code has valid Python syntax."""
    try:
        compile(pred.code, '<string>', 'exec')
        return True
    except SyntaxError as e:
        # Provide helpful error message
        raise AssertionError(f"Syntax error in generated code: {e}")

# Wrap with assertion
safe_coder = dspy.Assert(
    coder,
    validation_fn=validate_syntax,
    max_attempts=3,
    backtrack=True  # Try different approach on failure
)

# Use it
result = safe_coder(task="Create a function to calculate factorial")
print(result.code)  # Guaranteed to be syntactically valid
```

### 2. dspy.Suggest - Soft Constraints

`dspy.Suggest` provides gentle guidance for improving outputs without strict enforcement.

```python
class EssayWriter(dspy.Signature):
    """Write an essay on the given topic."""
    topic = dspy.InputField(desc="Essay topic", type=str)
    essay = dspy.OutputField(desc="Well-written essay", type=str)

writer = dspy.Predict(EssayWriter)

def suggest_improvements(example, pred, trace=None):
    """Suggest improvements for better essays."""
    suggestions = []

    if len(pred.essay.split()) < 200:
        suggestions.append("Essay should be at least 200 words")

    if not any(punc in pred.essay for punc in '.!?'):
        suggestions.append("Include proper punctuation")

    if len([s for s in pred.essay.split() if s[0].isupper()]) < 3:
        suggestions.append("Start sentences with capital letters")

    if suggestions:
        return False, f"Please improve: {'; '.join(suggestions)}"
    return True, None

# Wrap with suggestions
improved_writer = dspy.Suggest(
    writer,
    validation_fn=suggest_improvements,
    max_attempts=2,
    recovery_hint="Focus on clarity, grammar, and completeness"
)

result = improved_writer(topic="The importance of sleep")
```

### 3. Multiple Assertions

Chain multiple assertions for comprehensive validation:

```python
class DataProcessor(dspy.Signature):
    """Process and analyze data."""
    raw_data = dspy.InputField(desc="Raw input data", type=str)
    processed_data = dspy.OutputField(desc="Processed output", type=str)
    insights = dspy.OutputField(desc="Key insights from data", type=str)

processor = dspy.Predict(DataProcessor)

# Assertion 1: JSON format
def validate_json_format(example, pred, trace=None):
    import json
    try:
        json.loads(pred.processed_data)
        return True
    except:
        raise AssertionError("Processed data must be valid JSON")

# Assertion 2: Required fields
def validate_required_fields(example, pred, trace=None):
    import json
    data = json.loads(pred.processed_data)
    required = ['id', 'timestamp', 'value']
    missing = [f for f in required if f not in data]
    if missing:
        raise AssertionError(f"Missing required fields: {missing}")
    return True

# Assertion 3: Insights quality
def validate_insights(example, pred, trace=None):
    if len(pred.insights) < 50:
        raise AssertionError("Insights must be detailed (min 50 characters)")
    return True

# Chain all assertions
robust_processor = processor.with_assertions([
    validate_json_format,
    validate_required_fields,
    validate_insights
])
```

## Constraint Types

### 1. Format Constraints

Ensure outputs follow specific structural requirements:

```python
class APIResponse(dspy.Signature):
    """Generate API responses."""
    request = dspy.InputField(desc="API request details", type=str)
    response = dspy.OutputField(desc="JSON API response", type=str)

def validate_api_response(example, pred, trace=None):
    """Ensure valid API response format."""
    import json
    import re

    try:
        data = json.loads(pred.response)

        # Check required structure
        assert 'status' in data, "Missing 'status' field"
        assert 'data' in data, "Missing 'data' field"

        # Check status codes
        assert data['status'] in [200, 201, 400, 404, 500], \
               f"Invalid status code: {data['status']}"

        # Check data types
        assert isinstance(data['status'], int), "Status must be integer"
        assert isinstance(data['data'], (dict, list)), "Data must be object or array"

        return True

    except json.JSONDecodeError:
        raise AssertionError("Response must be valid JSON")

api_generator = dspy.Assert(
    dspy.Predict(APIResponse),
    validation_fn=validate_api_response,
    max_attempts=3
)
```

### 2. Semantic Constraints

Validate the meaning and correctness of outputs:

```python
class MathTutor(dspy.Signature):
    """Solve math problems with explanations."""
    problem = dspy.InputField(desc="Math problem to solve", type=str)
    solution = dspy.OutputField(desc="Step-by-step solution", type=str)
    answer = dspy.OutputField(desc="Final numerical answer", type=str)

def validate_math_solution(example, pred, trace=None):
    """Validate mathematical correctness."""
    import re
    import math

    # Extract numerical answer
    numbers = re.findall(r'-?\d+\.?\d*', pred.answer)
    if not numbers:
        raise AssertionError("Answer must contain a number")

    model_answer = float(numbers[-1])

    # Verify with actual calculation
    if "square root" in example.problem.lower():
        num = re.search(r'square root of (\d+)', example.problem.lower())
        if num:
            correct = math.sqrt(int(num.group(1)))
            if abs(model_answer - correct) > 0.01:
                raise AssertionError("Incorrect square root calculation")

    # Check if solution explains steps
    if len(pred.solution.split('\n')) < 2:
        raise AssertionError("Solution must show multiple steps")

    return True

math_tutor = dspy.Assert(
    dspy.Predict(MathTutor),
    validation_fn=validate_math_solution,
    max_attempts=3
)
```

### 3. Consistency Constraints

Ensure consistency between multiple outputs:

```python
class StoryGenerator(dspy.Signature):
    """Generate a coherent story."""
    prompt = dspy.InputField(desc="Story prompt", type=str)
    title = dspy.OutputField(desc="Story title", type=str)
    summary = dspy.OutputField(desc="Brief summary", type=str)
    content = dspy.OutputField(desc="Full story content", type=str)

def validate_story_consistency(example, pred, trace=None):
    """Ensure story elements are consistent."""

    # Title should reflect content
    title_words = set(pred.title.lower().split())
    content_words = set(pred.content.lower().split()[:50])  # First 50 words
    overlap = len(title_words.intersection(content_words))

    if overlap < 2:
        raise AssertionError("Title doesn't match story content")

    # Summary should match content
    if pred.summary not in pred.content:
        # Allow for paraphrasing by checking key concepts
        summary_concepts = pred.summary.lower().split()
        content_lower = pred.content.lower()

        for concept in summary_concepts:
            if len(concept) > 4 and concept not in content_lower:
                raise AssertionError(f"Summary mentions '{concept}' not in story")

    # Check story length
    if len(pred.content) < 500:
        raise AssertionError("Story too short (minimum 500 characters)")

    return True

story_generator = dspy.Assert(
    dspy.ChainOfThought(StoryGenerator),
    validation_fn=validate_story_consistency,
    max_attempts=2
)
```

## Advanced Assertion Patterns

### 1. Self-Refining Pipelines

Build pipelines that improve themselves based on assertion feedback:

```python
class SelfImprovingWriter(dspy.Module):
    """A writer that improves its output based on quality metrics."""

    def __init__(self):
        super().__init__()
        self.writer = dspy.ChainOfThought("topic -> draft")
        self.critic = dspy.ChainOfThought("draft, criteria -> critique")
        self.improver = dspy.ChainOfThought("draft, critique -> improved_draft")

    def forward(self, topic):
        # Initial draft
        draft = self.writer(topic=topic)

        # Quality criteria
        criteria = """
        1. Clarity: Is the writing clear and easy to understand?
        2. Completeness: Does it fully address the topic?
        3. Engagement: Is it interesting to read?
        4. Accuracy: Are all statements factual?
        """

        # Critique the draft
        critique = self.critic(draft=draft.draft, criteria=criteria)

        # Improve based on critique
        improved = self.improver(draft=draft.draft, critique=critique.critique)

        # Assert quality
        def validate_quality(example, pred, trace=None):
            word_count = len(pred.improved_draft.split())
            assert word_count > 100, "Draft too short"
            assert len(pred.improved_draft.split('\n')) > 3, "Add more paragraphs"
            return True

        # Apply assertion with self-refinement
        result = dspy.Assert(
            self,
            validation_fn=validate_quality,
            max_attempts=3
        )

        return dspy.Prediction(improved_draft=improved.improved_draft)

# Use the self-improving writer
writer = SelfImprovingWriter()
result = writer(topic="The benefits of renewable energy")
```

### 2. Contextual Assertions

Adapt validation based on input context:

```python
class AdaptiveValidator:
    """Validates outputs based on input context."""

    def __init__(self):
        self.rules = {
            'technical': self.validate_technical,
            'creative': self.validate_creative,
            'formal': self.validate_formal,
            'casual': self.validate_casual
        }

    def get_style(self, text):
        """Determine writing style from input."""
        text = text.lower()
        if any(word in text for word in ['code', 'algorithm', 'technical']):
            return 'technical'
        elif any(word in text for word in ['story', 'poem', 'creative']):
            return 'creative'
        elif any(word in text for word in ['report', 'formal', 'business']):
            return 'formal'
        else:
            return 'casual'

    def validate_technical(self, example, pred, trace=None):
        """Validate technical content."""
        assert '}' in pred.output or ';' in pred.output, \
               "Technical content should include code examples"
        assert any(word in pred.output.lower()
                  for word in ['implementation', 'example', 'function']), \
               "Include practical implementation details"
        return True

    def validate_creative(self, example, pred, trace=None):
        """Validate creative content."""
        assert len(pred.output) > 200, "Creative content should be substantial"
        sentences = pred.output.split('.')
        assert len(sentences) > 5, "Include multiple sentences"
        return True

    def validate_formal(self, example, pred, trace=None):
        """Validate formal content."""
        assert not any(word in pred.output.lower()
                      for word in ['hey', 'guys', 'awesome']), \
               "Avoid informal language in formal writing"
        return True

    def validate_casual(self, example, pred, trace=None):
        """Validate casual content."""
        return True  # No strict requirements

    def validate(self, example, pred, trace=None):
        """Route to appropriate validator based on context."""
        style = self.get_style(example.input)
        validator = self.rules.get(style, self.validate_casual)
        return validator(example, pred, trace)

# Use adaptive validation
validator = AdaptiveValidator()

adaptive_writer = dspy.Assert(
    dspy.Predict("input -> output"),
    validation_fn=validator.validate,
    max_attempts=2
)
```

### 3. Multi-Output Assertions

Validate relationships between multiple output fields:

```python
class MovieReview(dspy.Signature):
    """Generate a comprehensive movie review."""
    movie = dspy.InputField(desc="Movie title", type=str)
    rating = dspy.OutputField(desc="Rating 1-10", type=int)
    summary = dspy.OutputField(desc="Brief summary", type=str)
    detailed_review = dspy.OutputField(desc="Full review", type=str)

def validate_review_consistency(example, pred, trace=None):
    """Ensure all parts of the review are consistent."""

    # Rating must be in valid range
    assert 1 <= pred.rating <= 10, f"Rating {pred.rating} out of range"

    # High ratings should have positive content
    if pred.rating >= 7:
        positive_words = ['excellent', 'amazing', 'brilliant', 'outstanding']
        assert any(word in pred.detailed_review.lower()
                  for word in positive_words), \
               "High rating should include positive language"

    # Low ratings should include criticism
    if pred.rating <= 4:
        negative_words = ['disappointing', 'flawed', 'lacking', 'weak']
        assert any(word in pred.detailed_review.lower()
                  for word in negative_words), \
               "Low rating should include constructive criticism"

    # Summary should reflect rating
    if pred.rating >= 8 and 'not' in pred.summary:
        raise AssertionError("Summary conflicts with high rating")

    if pred.rating <= 3 and ('great' in pred.summary or 'excellent' in pred.summary):
        raise AssertionError("Summary conflicts with low rating")

    # Detailed review must be longer than summary
    assert len(pred.detailed_review) > len(pred.summary), \
           "Detailed review should be longer than summary"

    return True

review_generator = dspy.Assert(
    dspy.Predict(MovieReview),
    validation_fn=validate_review_consistency,
    max_attempts=3
)
```

## Integration with Existing Modules

### 1. Assertions with ChainOfThought

Add assertions to reasoning chains:

```python
class LogicalReasoning(dspy.Signature):
    """Solve logic puzzles with step-by-step reasoning."""
    puzzle = dspy.InputField(desc="Logic puzzle", type=str)
    reasoning = dspy.OutputField(desc="Step-by-step logical reasoning", type=str)
    conclusion = dspy.OutputField(desc="Final conclusion", type=str)
    confidence = dspy.OutputField(desc="Confidence level (1-10)", type=int)

reasoner = dspy.ChainOfThought(LogicalReasoning)

def validate_logical_reasoning(example, pred, trace=None):
    """Ensure reasoning is logically sound."""

    # Check for reasoning steps
    steps = pred.reasoning.split('\n')
    assert len(steps) >= 3, "Include at least 3 reasoning steps"

    # Look for logical connectors
    connectors = ['therefore', 'because', 'since', 'thus', 'hence']
    has_logic = any(connector in pred.reasoning.lower()
                   for connector in connectors)
    assert has_logic, "Use logical connectors in reasoning"

    # Conclusion should follow from reasoning
    if pred.confidence >= 8:
        assert len(pred.conclusion) > 20, \
               "High confidence conclusions should be well-justified"

    return True

logical_reasoner = dspy.Assert(
    reasoner,
    validation_fn=validate_logical_reasoning,
    max_attempts=3
)
```

### 2. Assertions with ReAct

Validate agent actions and observations:

```python
class ResearchAgent(dspy.Module):
    """An agent that performs research with validated findings."""

    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct("query -> findings")

    def forward(self, query):
        def validate_research(example, pred, trace=None):
            """Validate research quality."""

            # Must have taken some actions
            if trace and 'tool_calls' not in str(trace):
                raise AssertionError("Must use search tools for research")

            # Findings should be substantial
            assert len(pred.findings) > 100, "Research findings too brief"

            # Should include sources or evidence
            evidence_words = ['according', 'research shows', 'study', 'data']
            has_evidence = any(word in pred.findings.lower()
                             for word in evidence_words)
            assert has_evidence, "Include evidence or sources in findings"

            return True

        # Apply assertion
        validated_react = dspy.Assert(
            self.react,
            validation_fn=validate_research,
            max_attempts=3
        )

        return validated_react(query=query)

# Use the validated research agent
researcher = ResearchAgent()
result = researcher(query="Impact of AI on job markets")
```

### 3. Custom Assertion Handlers

Create specialized assertion handlers for complex scenarios:

```python
import datetime
import time

class AssertionHandler:
    """Custom handler for complex assertion scenarios."""

    def __init__(self):
        self.attempt_history = []

    def handle_assertion_failure(self, assertion_type, error_msg, attempt):
        """Custom logic for handling assertion failures."""
        self.attempt_history.append({
            'attempt': attempt,
            'type': assertion_type,
            'error': error_msg,
            'timestamp': datetime.now()
        })

        # Different recovery strategies based on error type
        if "format" in error_msg.lower():
            return "Please ensure strict adherence to the required format."
        elif "length" in error_msg.lower():
            return "Make your response more detailed and comprehensive."
        elif "accuracy" in error_msg.lower():
            return "Double-check your facts and calculations."
        else:
            return "Review your response for completeness and accuracy."

    def generate_recovery_prompt(self, original_input, failed_output, error_msg):
        """Generate a refined prompt for retry attempts."""
        recovery_instruction = self.handle_assertion_failure(
            "validation", error_msg, len(self.attempt_history)
        )

        return f"""
        Original task: {original_input}

        Your previous attempt: {failed_output}

        Error: {error_msg}

        {recovery_instruction}

        Please provide an improved response that addresses the issue.
        """

# Use custom handler
handler = AssertionHandler()

custom_assert = dspy.Assert(
    dspy.Predict("task -> result"),
    validation_fn=lambda ex, pred, tr: validate_output(ex, pred, tr),
    max_attempts=3,
    error_handler=handler.handle_assertion_failure
)
```

## Best Practices

### 1. Design Effective Validators

```python
# Good: Specific and actionable error messages
def validate_email(example, pred, trace=None):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, pred.email):
        raise AssertionError(
            f"'{pred.email}' is not a valid email. "
            f"Must follow format: user@domain.com"
        )
    return True

# Bad: Generic errors
def validate_email_bad(example, pred, trace=None):
    if '@' not in pred.email:
        raise AssertionError("Invalid email")  # Not helpful
    return True
```

### 2. Balance Strictness and Flexibility

```python
# Use suggestions for preferences, assertions for requirements
def generate_content(topic):
    # Hard requirement: must have title
    assert hasattr(pred, 'title'), "Content must have a title"

    # Soft suggestion: prefer subheadings (not mandatory)
    suggest_add_subheadings(pred.content)
```

### 3. Handle Edge Cases

```python
def robust_validator(example, pred, trace=None):
    try:
        # Main validation logic
        validate_main_logic(example, pred, trace)
        return True
    except AttributeError:
        raise AssertionError("Required field missing from output")
    except (TypeError, ValueError):
        raise AssertionError("Output has incorrect type or format")
    except Exception as e:
        raise AssertionError(f"Validation error: {str(e)}")
```

## Performance Considerations

### 1. Assertion Overhead

Each assertion adds computational overhead. Use judiciously:

```python
# Good: Critical assertions
validate_safety = dspy.Assert(safety_module, validate_safety_constraints)

# Consider: Performance-critical paths might use lighter validation
quick_validate = lambda ex, pred: len(pred.output) > 10  # Simple check
```

### 2. Caching Validation Results

Cache expensive validation operations:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_syntax_check(code_hash):
    """Cache syntax validation for identical code."""
    # Check syntax...
    pass
```

### 3. Progressive Validation

Validate in order of cost:

```python
def progressive_validate(example, pred, trace=None):
    # Fast checks first
    assert len(pred.output) > 0, "Empty output"

    # Medium checks
    assert pred.output.count('\n') > 2, "Need multiple paragraphs"

    # Expensive checks last
    validate_semantics(pred.output)  # Slow operation
    return True
```

## Debugging Assertions

### 1. Trace Inspection

Examine assertion failures for debugging:

```python
def debug_assertion(example, pred, trace=None):
    """Debug assertion with detailed information."""
    print(f"Input: {example}")
    print(f"Output: {pred}")
    print(f"Trace: {trace}")

    # Perform validation
    result = actual_validation(example, pred, trace)

    if not result:
        print("Validation failed!")
        # Analyze why...

    return result
```

### 2. Assertion Metrics

Track assertion performance:

```python
class AssertionMetrics:
    def __init__(self):
        self.stats = {
            'total_attempts': 0,
            'failures': 0,
            'retries': 0,
            'success_rate': 0
        }

    def record_attempt(self, success, retries):
        self.stats['total_attempts'] += 1
        if not success:
            self.stats['failures'] += 1
        self.stats['retries'] += retries
        self.stats['success_rate'] = (
            (self.stats['total_attempts'] - self.stats['failures']) /
            self.stats['total_attempts']
        )
```

## Advanced Assertion Patterns

### 1. Hierarchical Assertions

Multi-level validation with cascading constraints:

```python
from typing import TypeVar, Generic
from abc import ABC, abstractmethod

T = TypeVar('T')

class HierarchicalAssertion(Generic[T], ABC):
    """Base class for hierarchical assertion systems."""

    def __init__(self, name: str, level: int = 0):
        self.name = name
        self.level = level
        self.children = []
        self.parent = None

    def add_child(self, child: 'HierarchicalAssertion'):
        """Add child assertion."""
        child.parent = self
        child.level = self.level + 1
        self.children.append(child)

    def validate_hierarchy(self, example, pred, trace=None) -> Tuple[bool, List[str]]:
        """Validate entire hierarchy."""
        errors = []

        # Validate current level
        local_valid, local_errors = self.validate(example, pred, trace)
        if not local_valid:
            errors.extend([f"[{self.name}] {e}" for e in local_errors])

        # Validate children if current level passes
        if local_valid:
            for child in self.children:
                child_valid, child_errors = child.validate_hierarchy(
                    example, pred, trace
                )
                if not child_valid:
                    errors.extend(child_errors)

        return len(errors) == 0, errors

    @abstractmethod
    def validate(self, example, pred, trace=None) -> Tuple[bool, List[str]]:
        """Validate at this level."""
        pass

# Example: Document validation hierarchy
class DocumentAssertion(HierarchicalAssertion):
    """Top-level document validation."""

    def __init__(self):
        super().__init__("document", level=0)

        # Add child assertions
        self.add_child(StructureAssertion())
        self.add_child(ContentAssertion())
        self.add_child(FormatAssertion())

    def validate(self, example, pred, trace=None):
        """Validate document-level constraints."""
        errors = []

        # Basic document checks
        if not hasattr(pred, 'content'):
            return False, ["Missing content field"]

        if len(pred.content) < 100:
            errors.append("Document too short (minimum 100 characters)")

        if len(pred.content) > 10000:
            errors.append("Document too long (maximum 10000 characters)")

        return len(errors) == 0, errors

class StructureAssertion(HierarchicalAssertion):
    """Validate document structure."""

    def __init__(self):
        super().__init__("structure")

    def validate(self, example, pred, trace=None):
        """Validate structural elements."""
        errors = []
        content = pred.content

        # Check for sections
        if '#' not in content:
            errors.append("Document missing section headers")

        # Check for paragraphs
        paragraphs = content.split('\n\n')
        if len(paragraphs) < 3:
            errors.append("Document needs at least 3 paragraphs")

        # Check for flow
        if not self.has_logical_flow(content):
            errors.append("Document lacks logical flow")

        return len(errors) == 0, errors

    def has_logical_flow(self, content: str) -> bool:
        """Check if content has logical flow."""
        # Simple heuristic: look for transition words
        transitions = ['however', 'therefore', 'furthermore', 'consequently']
        return any(word in content.lower() for word in transitions)

# Use hierarchical assertions
doc_validator = DocumentAssertion()

# Wrap with hierarchical validation
class DocumentGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict("topic -> content")
        self.hierarchical_validator = doc_validator

    def forward(self, topic):
        result = self.generator(topic=topic)

        # Validate hierarchy
        is_valid, errors = self.hierarchical_validator.validate_hierarchy(
            example=None, pred=result
        )

        if not is_valid:
            # Refine based on hierarchical feedback
            refined_result = self.refine_hierarchically(
                result, errors, self.hierarchical_validator
            )
            return refined_result

        return result
```

### 2. Probabilistic Assertions

Assertions with confidence-based validation:

```python
from scipy import stats
import numpy as np

class ProbabilisticAssertion:
    """Assertions with probabilistic validation."""

    def __init__(self, confidence_threshold=0.95):
        self.confidence_threshold = confidence_threshold
        self.validation_history = []

    def validate_with_confidence(self, example, pred, trace=None) -> Tuple[bool, float, str]:
        """Validate with confidence scoring."""
        # Calculate confidence score
        confidence = self.calculate_confidence(example, pred, trace)

        # Determine if passes threshold
        passes = confidence >= self.confidence_threshold

        # Generate explanation
        explanation = self.generate_explanation(confidence, pred)

        # Record for learning
        self.validation_history.append({
            'confidence': confidence,
            'passed': passes,
            'explanation': explanation
        })

        return passes, confidence, explanation

    def calculate_confidence(self, example, pred, trace=None):
        """Calculate confidence score for validation."""
        confidence_factors = []

        # Factor 1: Structural consistency
        struct_confidence = self.check_structural_consistency(pred)
        confidence_factors.append(struct_confidence)

        # Factor 2: Semantic coherence
        semantic_confidence = self.check_semantic_coherence(pred)
        confidence_factors.append(semantic_confidence)

        # Factor 3: Historical performance
        history_confidence = self.get_historical_confidence()
        confidence_factors.append(history_confidence)

        # Combine factors (weighted average)
        weights = [0.4, 0.4, 0.2]  # Adjust as needed
        confidence = sum(w * c for w, c in zip(weights, confidence_factors))

        return confidence

    def check_structural_consistency(self, pred) -> float:
        """Check structural consistency of output."""
        score = 0.0

        # Check required fields
        required_fields = getattr(pred, '_required_fields', [])
        for field in required_fields:
            if hasattr(pred, field) and getattr(pred, field):
                score += 1.0 / len(required_fields)

        # Check field consistency
        if hasattr(pred, 'answer') and hasattr(pred, 'confidence'):
            # Higher confidence should correlate with longer answers
            if pred.confidence > 0.8 and len(pred.answer) < 10:
                score *= 0.5  # Penalize inconsistency

        return min(score, 1.0)

    def check_semantic_coherence(self, pred) -> float:
        """Check semantic coherence using NLP techniques."""
        # Simplified coherence check
        if not hasattr(pred, 'answer'):
            return 0.0

        answer = pred.answer

        # Check for repeated phrases
        words = answer.lower().split()
        unique_words = set(words)
        repetition_ratio = len(unique_words) / len(words) if words else 0

        # Check sentence structure
        sentences = answer.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s])

        # Combine factors
        coherence_score = 0.0
        coherence_score += repetition_ratio * 0.4
        coherence_score += min(avg_sentence_length / 15, 1.0) * 0.3
        coherence_score += 0.3 if 5 <= len(sentences) <= 10 else 0.1

        return coherence_score

    def get_historical_confidence(self) -> float:
        """Calculate confidence based on historical performance."""
        if not self.validation_history:
            return 0.5  # Neutral for no history

        # Recent performance more important
        recent_history = self.validation_history[-10:]
        success_rate = sum(1 for h in recent_history if h['passed']) / len(recent_history)

        return success_rate

class AdaptiveThreshold:
    """Adaptive confidence threshold based on context."""

    def __init__(self, initial_threshold=0.95):
        self.base_threshold = initial_threshold
        self.context_adjustments = {}
        self.performance_feedback = []

    def get_threshold(self, context: dict) -> float:
        """Get adjusted threshold for context."""
        threshold = self.base_threshold

        # Adjust based on context
        context_key = self.get_context_key(context)
        if context_key in self.context_adjustments:
            threshold *= self.context_adjustments[context_key]

        # Adjust based on recent performance
        if self.performance_feedback:
            recent_performance = np.mean(self.performance_feedback[-5:])
            if recent_performance < 0.8:
                threshold *= 0.9  # Lower threshold if struggling
            elif recent_performance > 0.95:
                threshold *= 1.1  # Raise threshold if doing well

        return min(max(threshold, 0.5), 0.99)  # Keep within bounds

    def update_adjustment(self, context: dict, adjustment: float):
        """Update context adjustment based on feedback."""
        context_key = self.get_context_key(context)
        self.context_adjustments[context_key] = adjustment

    def get_context_key(self, context: dict) -> str:
        """Generate key for context lookup."""
        # Simplified context key generation
        key_parts = []
        if 'domain' in context:
            key_parts.append(context['domain'])
        if 'complexity' in context:
            key_parts.append(f"complexity_{context['complexity']}")
        return "_".join(key_parts) or "default"

# Usage with probabilistic assertions
probabilistic_assert = ProbabilisticAssertion(confidence_threshold=0.9)
adaptive_threshold = AdaptiveThreshold()

class ProbabilisticValidator(dspy.Module):
    def __init__(self, base_module):
        super().__init__()
        self.base_module = base_module
        self.prob_assert = probabilistic_assert
        self.adaptive_threshold = adaptive_threshold

    def forward(self, **kwargs):
        # Get context
        context = {
            'domain': kwargs.get('domain', 'general'),
            'complexity': kwargs.get('complexity', 'medium')
        }

        # Get adaptive threshold
        threshold = self.adaptive_threshold.get_threshold(context)

        # Generate result
        result = self.base_module(**kwargs)

        # Validate with confidence
        passes, confidence, explanation = self.prob_assert.validate_with_confidence(
            example=None, pred=result
        )

        # Check against adaptive threshold
        if confidence < threshold:
            # Provide feedback for learning
            self.adaptive_threshold.update_adjustment(
                context,
                threshold / confidence  # Adjustment factor
            )

            # Try to improve
            improved = self.improve_result(result, explanation)
            if improved:
                result = improved

        return result
```

### 3. Distributed Assertions

Assertions across multiple model calls:

```python
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import asyncio

class DistributedAssertionSystem:
    """Manages assertions across distributed model calls."""

    def __init__(self, assertion_nodes: Dict[str, 'AssertionNode']):
        self.assertion_nodes = assertion_nodes
        self.communication_bus = AssertionCommunicationBus()
        self.coordinator = AssertionCoordinator(assertion_nodes)

    def validate_distributed(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate distributed validation."""
        # Create validation plan
        plan = self.coordinator.create_validation_plan(inputs)

        # Execute in parallel where possible
        results = self.execute_validation_plan(plan)

        # Aggregate results
        aggregated = self.coordinator.aggregate_results(results)

        # Resolve conflicts
        resolved = self.coordinator.resolve_conflicts(aggregated)

        return resolved

    def execute_validation_plan(self, plan: Dict) -> Dict:
        """Execute validation plan with parallel execution."""
        results = {}

        # Identify parallelizable tasks
        parallel_tasks = []
        sequential_tasks = []

        for task_id, task in plan.items():
            if task.get('parallelizable', False):
                parallel_tasks.append((task_id, task))
            else:
                sequential_tasks.append((task_id, task))

        # Execute parallel tasks
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_task = {
                executor.submit(self.execute_task, task): task_id
                for task_id, task in parallel_tasks
            }

            for future in concurrent.futures.as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    results[task_id] = future.result()
                except Exception as e:
                    results[task_id] = {'error': str(e)}

        # Execute sequential tasks
        for task_id, task in sequential_tasks:
            results[task_id] = self.execute_task(task)

        return results

class AssertionNode:
    """Individual assertion node in distributed system."""

    def __init__(self, node_id: str, assertions: List[dspy.Assert]):
        self.node_id = node_id
        self.assertions = assertions
        self.local_cache = {}

    def validate(self, data: Dict[str, Any], context: Dict = None) -> Dict:
        """Validate with local assertions."""
        results = {
            'node_id': self.node_id,
            'validations': [],
            'overall_status': 'passed',
            'metadata': {
                'validation_count': len(self.assertions),
                'execution_time': 0
            }
        }

        start_time = time.time()

        for assertion in self.assertions:
            try:
                # Check cache first
                cache_key = self.get_cache_key(data, assertion)
                if cache_key in self.local_cache:
                    validation_result = self.local_cache[cache_key]
                else:
                    # Execute assertion
                    validation_result = self.execute_assertion(
                        assertion, data, context
                    )
                    # Cache result
                    self.local_cache[cache_key] = validation_result

                results['validations'].append({
                    'assertion_id': id(assertion),
                    'result': validation_result,
                    'cached': cache_key in self.local_cache
                })

                if not validation_result['passed']:
                    results['overall_status'] = 'failed'

            except Exception as e:
                results['validations'].append({
                    'assertion_id': id(assertion),
                    'error': str(e),
                    'passed': False
                })
                results['overall_status'] = 'error'

        results['metadata']['execution_time'] = time.time() - start_time

        return results

# Example: Multi-modal validation system
class MultiModalValidationSystem:
    """Validates outputs across different modalities."""

    def __init__(self):
        # Create assertion nodes for each modality
        self.text_node = AssertionNode(
            'text_validation',
            [
                dspy.Assert(validate_text_coherence),
                dspy.Assert(validate_text_quality),
                dspy.Assert(validate_text_length)
            ]
        )

        self.image_node = AssertionNode(
            'image_validation',
            [
                dspy.Assert(validate_image_quality),
                dspy.Assert(validate_image_content),
                dspy.Assert(validate_image_style)
            ]
        )

        self.multimodal_node = AssertionNode(
            'multimodal_validation',
            [
                dspy.Assert(validate_text_image_consistency),
                dspy.Assert(validate_modality_balance)
            ]
        )

        # Create distributed system
        self.distributed_system = DistributedAssertionSystem({
            'text': self.text_node,
            'image': self.image_node,
            'multimodal': self.multimodal_node
        })

    def validate_multimodal_output(self, output: Dict[str, Any]):
        """Validate multimodal output."""
        # Prepare inputs for each node
        inputs = {
            'text': {'text_data': output.get('text', '')},
            'image': {'image_data': output.get('image', None)},
            'multimodal': {
                'text_data': output.get('text', ''),
                'image_data': output.get('image', None)
            }
        }

        # Execute distributed validation
        results = self.distributed_system.validate_distributed(inputs)

        # Generate comprehensive report
        report = self.generate_validation_report(results)

        return report

    def generate_validation_report(self, results: Dict) -> Dict:
        """Generate comprehensive validation report."""
        report = {
            'overall_status': 'passed',
            'modality_results': {},
            'cross_modality_issues': [],
            'recommendations': []
        }

        # Process individual modality results
        for modality, result in results.items():
            if 'error' in result:
                report['modality_results'][modality] = {
                    'status': 'error',
                    'message': result['error']
                }
                report['overall_status'] = 'failed'
            else:
                report['modality_results'][modality] = {
                    'status': result.get('overall_status', 'unknown'),
                    'validations_passed': sum(
                        1 for v in result.get('validations', [])
                        if v.get('result', {}).get('passed', False)
                    ),
                    'total_validations': len(result.get('validations', [])),
                    'execution_time': result.get('metadata', {}).get('execution_time', 0)
                }

                if result.get('overall_status') != 'passed':
                    report['overall_status'] = 'failed'

        # Cross-modality analysis
        if 'text' in results and 'image' in results:
            text_issues = self.extract_issues(results['text'])
            image_issues = self.extract_issues(results['image'])

            # Find related issues
            for text_issue in text_issues:
                for image_issue in image_issues:
                    if self.are_related_issues(text_issue, image_issue):
                        report['cross_modality_issues'].append({
                            'type': 'related',
                            'text_issue': text_issue,
                            'image_issue': image_issue,
                            'severity': 'high'
                        })

        # Generate recommendations
        report['recommendations'] = self.generate_recommendations(report)

        return report

# Usage
multimodal_validator = MultiModalValidationSystem()

# Validate multimodal output
output = {
    'text': 'A beautiful sunset over the mountains',
    'image': generated_image
}

validation_report = multimodal_validator.validate_multimodal_output(output)
print(f"Overall status: {validation_report['overall_status']}")
```

### 4. Learning Assertions

Assertions that improve over time:

```python
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

class LearningAssertion:
    """Assertions that learn from validation history."""

    def __init__(self, assertion_name: str, model_path: str = None):
        self.assertion_name = assertion_name
        self.model_path = model_path or f"models/{assertion_name}_model.pkl"
        self.model = self.load_or_create_model()
        self.training_data = []
        self.feature_extractor = AssertionFeatureExtractor()

    def load_or_create_model(self):
        """Load existing model or create new one."""
        if Path(self.model_path).exists():
            return joblib.load(self.model_path)
        else:
            return RandomForestClassifier(n_estimators=100, random_state=42)

    def validate_with_learning(self, example, pred, trace=None):
        """Validate using learned patterns."""
        # Extract features
        features = self.feature_extractor.extract(example, pred, trace)

        # Predict validation outcome
        prediction = self.model.predict([features])[0]
        confidence = self.model.predict_proba([features])[0].max()

        # Get feature importance
        feature_importance = self.get_feature_importance(features)

        return {
            'passed': bool(prediction),
            'confidence': float(confidence),
            'feature_importance': feature_importance,
            'learned': True
        }

    def learn_from_feedback(self, example, pred, actual_outcome, trace=None):
        """Learn from actual validation outcomes."""
        # Extract features
        features = self.feature_extractor.extract(example, pred, trace)

        # Add to training data
        self.training_data.append({
            'features': features,
            'outcome': actual_outcome
        })

        # Retrain if enough data
        if len(self.training_data) >= 50:
            self.retrain_model()

    def retrain_model(self):
        """Retrain the assertion model."""
        if not self.training_data:
            return

        # Prepare training data
        X = [d['features'] for d in self.training_data]
        y = [d['outcome'] for d in self.training_data]

        # Retrain
        self.model.fit(X, y)

        # Save model
        joblib.dump(self.model, self.model_path)

        # Clear training data to save memory
        self.training_data = []

    def get_feature_importance(self, features):
        """Get importance of each feature for this prediction."""
        if not hasattr(self.model, 'feature_importances_'):
            return {}

        feature_names = self.feature_extractor.get_feature_names()
        importances = self.model.feature_importances_

        return {
            name: float(imp)
            for name, imp in zip(feature_names, importances)
        }

class AssertionFeatureExtractor:
    """Extracts features for learning assertions."""

    def __init__(self):
        self.feature_cache = {}

    def extract(self, example, pred, trace=None):
        """Extract comprehensive features."""
        features = {}

        # Text features
        if hasattr(pred, 'answer'):
            text_features = self.extract_text_features(pred.answer)
            features.update({f"text_{k}": v for k, v in text_features.items()})

        # Structural features
        struct_features = self.extract_structural_features(pred)
        features.update({f"struct_{k}": v for k, v in struct_features.items()})

        # Context features
        if example:
            context_features = self.extract_context_features(example, pred)
            features.update({f"context_{k}": v for k, v in context_features.items()})

        # Trace features
        if trace:
            trace_features = self.extract_trace_features(trace)
            features.update({f"trace_{k}": v for k, v in trace_features.items()})

        return features

    def extract_text_features(self, text: str) -> Dict:
        """Extract text-based features."""
        features = {}

        # Basic statistics
        words = text.split()
        sentences = text.split('.')
        paragraphs = text.split('\n\n')

        features['word_count'] = len(words)
        features['sentence_count'] = len(sentences)
        features['paragraph_count'] = len(paragraphs)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences if s]) if sentences else 0

        # Vocabulary diversity
        unique_words = set(words)
        features['vocab_diversity'] = len(unique_words) / len(words) if words else 0

        # Punctuation patterns
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['comma_count'] = text.count(',')

        # Readability approximation
        features['readability_score'] = self.calculate_readability(text)

        return features

    def extract_structural_features(self, pred) -> Dict:
        """Extract structural features."""
        features = {}

        # Field presence
        all_fields = dir(pred)
        features['field_count'] = len(all_fields)
        features['has_confidence'] = hasattr(pred, 'confidence')
        features['has_reasoning'] = hasattr(pred, 'reasoning')

        # Field consistency
        if hasattr(pred, 'confidence') and hasattr(pred, 'answer'):
            # High confidence with short answer might be suspicious
            if pred.confidence > 0.9 and len(pred.answer) < 10:
                features['confidence_consistency'] = 0
            else:
                features['confidence_consistency'] = 1

        return features

    def calculate_readability(self, text: str) -> float:
        """Simple readability score."""
        # Simplified Flesch Reading Ease
        words = text.split()
        sentences = text.split('.')

        if not words or not sentences:
            return 0

        avg_sentence_length = len(words) / len(sentences)
        avg_syllables = np.mean([self.count_syllables(w) for w in words])

        readability = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables
        return max(0, min(100, readability))

    def count_syllables(self, word: str) -> int:
        """Approximate syllable count."""
        vowels = "aeiouy"
        word = word.lower()
        syllables = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllables += 1
            prev_was_vowel = is_vowel

        # Adjust for silent e
        if word.endswith('e') and syllables > 1:
            syllables -= 1

        return max(1, syllables)

# Usage with learning assertions
learning_assertion = LearningAssertion("answer_quality")

class AdaptiveQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("question -> answer")
        self.learning_assertion = learning_assertion

    def forward(self, question):
        result = self.qa(question=question)

        # Validate with learning
        validation = self.learning_assertion.validate_with_learning(
            example={'question': question},
            pred=result
        )

        if not validation['passed'] and validation['confidence'] > 0.8:
            # High confidence failure - likely an error
            print(f"Validation failed with high confidence: {validation['feature_importance']}")

            # Learn from this
            self.learning_assertion.learn_from_feedback(
                example={'question': question},
                pred=result,
                actual_outcome=False  # Failed
            )

            # Try again
            result = self.qa(question=question)

        return result

# Later, with human feedback
# learning_assertion.learn_from_feedback(
#     example=example,
#     pred=prediction,
#     actual_outcome=True  # Human confirmed it was good
# )
```

## Summary

DSPy Assertions provide:

- **Runtime validation** of model outputs
- **Automatic refinement** when constraints fail
- **Flexible constraint types** - hard and soft constraints
- **Self-improving systems** through iterative refinement
- **Production reliability** through guaranteed output quality
- **Hierarchical validation** for complex requirements
- **Probabilistic assertions** with confidence-based decisions
- **Distributed assertions** across multiple model calls
- **Learning assertions** that improve from experience

### Key Takeaways

1. **Use Assert for requirements** - Critical constraints that must pass
2. **Use Suggest for preferences** - Guidance for improving quality
3. **Write clear error messages** - Help the model understand failures
4. **Balance validation cost** - Consider performance implications
5. **Compose multiple assertions** - Build comprehensive validation

## Next Steps

- [Self-Refining Pipelines](../07-advanced-topics/07-self-refining-pipelines.md) - Learn advanced patterns
- [Constraint-Driven Optimization](../05-optimizers/07-constraint-driven-optimization.md) - Optimize with constraints
- [Assertion-Driven Applications](../08-case-studies/06-assertion-driven-applications.md) - Real-world examples
- [Exercises](./07-exercises.md) - Practice assertion techniques

## Further Reading

- [DSPy Documentation: Assertions](https://dspy-docs.vercel.app/docs/deep-dive/assertions)
- [Constraint Programming](https://en.wikipedia.org/wiki/Constraint_programming) - Theoretical foundation
- [Runtime Verification](https://en.wikipedia.org/wiki/Runtime_verification) - Validation techniques