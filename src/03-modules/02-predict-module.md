# The Predict Module

## Prerequisites

- **Previous Section**: [Module Basics](./01-module-basics.md) - Understanding of module concepts
- **Chapter 2**: Signatures - Familiarity with signature design
- **Required Knowledge**: Basic DSPy setup and configuration
- **Difficulty Level**: Beginner to Intermediate
- **Estimated Reading Time**: 35 minutes

## Learning Objectives

By the end of this section, you will:
- Master the `dspy.Predict` module - DSPy's most fundamental module
- Understand how to use Predict for various simple tasks
- Learn to configure Predict with examples and instructions
- Discover best practices for getting reliable results
- Know when to use Predict versus more complex modules

## Introduction to dspy.Predict

`dspy.Predict` is the simplest yet most versatile module in DSPy. It creates a direct mapping between inputs and outputs based on a signature, making it perfect for straightforward transformation tasks.

### Core Concept
```
Input(s) → [Predict Module] → Output(s)
```

Predict takes your signature, constructs an appropriate prompt, sends it to the LLM, and parses the response back into structured outputs according to your signature definition.

## Basic Usage

### Simple Example
```python
import dspy

# Define a signature
class BasicQA(dspy.Signature):
    """Answer a question based on provided context."""
    question = dspy.InputField(desc="Question to answer", type=str)
    context = dspy.InputField(desc="Relevant context", type=str, optional=True)
    answer = dspy.OutputField(desc="Answer to the question", type=str)

# Create a Predict module
qa = dspy.Predict(BasicQA)

# Use it
result = qa(
    question="What is the capital of France?",
    context="Paris is the capital city of France."
)

print(result.answer)  # "Paris"
```

### String Signature Shortcut
For simple cases, you can use string signatures directly:

```python
# Quick and simple
summarizer = dspy.Predict("long_text -> short_summary")

result = summarizer(
    long_text="A very long document that needs to be summarized..."
)

print(result.short_summary)
```

## Configuring Predict

### Adding Instructions
Customize how the module approaches the task:

```python
# With custom instructions
translator = dspy.Predict(
    "source_text, target_language -> translated_text",
    instructions="Translate accurately while preserving the original tone and meaning. "
                 "Consider cultural nuances and idiomatic expressions."
)

result = translator(
    source_text="It's raining cats and dogs!",
    target_language="Spanish"
)
```

### Adding Few-Shot Examples
Improve performance with examples:

```python
# Create examples
math_examples = [
    dspy.Example(
        problem="What is 15 + 27?",
        answer="42"
    ),
    dspy.Example(
        problem="What is 8 × 9?",
        answer="72"
    )
]

# Create module with examples
math_solver = dspy.Predict(
    "math_problem -> answer",
    demos=math_examples
)

result = math_solver(problem="What is 23 + 19?")
print(result.answer)  # 42
```

### Setting Temperature and Other Parameters
Control randomness and creativity:

```python
# For creative tasks
creative_writer = dspy.Predict(
    "prompt -> creative_response",
    temperature=0.9,
    max_tokens=500
)

# For precise tasks
classifier = dspy.Predict(
    "text -> category",
    temperature=0.1,
    max_tokens=10
)
```

## Real-World Examples

### 1. Text Classification
```python
import dspy
from typing import List

class EmailClassifier(dspy.Signature):
    """Classify emails into categories."""
    email_text = dspy.InputField(desc="Full email content", type=str)
    sender_info = dspy.InputField(desc="Information about sender", type=str, optional=True)
    category = dspy.OutputField(desc="Email category", type=str)
    urgency = dspy.OutputField(desc="Urgency level (1-5)", type=int)
    action_required = dspy.OutputField(desc="Whether action is needed", type=bool)

# Create with examples
email_examples = [
    dspy.Example(
        email_text="URGENT: Server is down! We need immediate assistance.",
        category="technical_support",
        urgency=5,
        action_required=True
    ),
    dspy.Example(
        email_text="Thank you for your purchase. Your order has been confirmed.",
        category="order_confirmation",
        urgency=1,
        action_required=False
    )
]

# Initialize classifier
classifier = dspy.Predict(EmailClassifier, demos=email_examples)

# Use it
result = classifier(
    email_text="Hi team, I'm having trouble accessing my account. Can you help?",
    sender_info="Customer from premium tier"
)

print(f"Category: {result.category}")
print(f"Urgency: {result.urgency}")
print(f"Action Needed: {result.action_required}")
```

### 2. Data Extraction
```python
class InformationExtractor(dspy.Signature):
    """Extract structured information from unstructured text."""
    document_text = dspy.InputField(desc="Text to extract from", type=str)
    entity_types = dspy.InputField(desc="Types of entities to find", type=List[str])
    entities = dspy.OutputField(desc="Extracted entities", type=List[dict])
    confidence = dspy.OutputField(desc="Confidence in extraction", type=float)

# Extract from business documents
extractor = dspy.Predict(
    InformationExtractor,
    instructions="Extract all specified entities with their locations in the text. "
                 "Assign confidence scores based on clarity of mention."
)

result = extractor(
    document_text="Apple Inc. announced today that CEO Tim Cook would present at the "
                 "Tech Conference 2024 in San Francisco next month.",
    entity_types=["organizations", "people", "events", "locations", "dates"]
)

for entity in result.entities:
    print(f"{entity['type']}: {entity['text']} (confidence: {entity['confidence']})")
```

### 3. Text Transformation
```python
class TextTransformer(dspy.Signature):
    """Transform text to different formats or styles."""
    original_text = dspy.InputField(desc="Text to transform", type=str)
    transformation_type = dspy.InputField(desc="Type of transformation", type=str)
    target_audience = dspy.InputField(desc="Target audience", type=str)
    transformed_text = dspy.OutputField(desc="Transformed text", type=str)
    changes_made = dspy.OutputField(desc="Summary of changes", type=List[str])

# Multiple transformations in one module
transformer = dspy.Predict(
    TextTransformer,
    temperature=0.3  # Keep changes consistent
)

# Simplify technical text
result = transformer(
    original_text="The implementation utilizes a RESTful API architecture with "
                 "asynchronous data processing capabilities.",
    transformation_type="simplify",
    target_audience="non-technical"
)

print(result.transformed_text)
print("Changes:", result.changes_made)
```

## Advanced Configuration

### Multiple Output Formats
```python
class FlexibleAnalyzer(dspy.Signature):
    """Analyze text with flexible output options."""
    text = dspy.InputField(desc="Text to analyze", type=str)
    analysis_type = dspy.InputField(desc="Type of analysis", type=str)
    output_format = dspy.InputField(desc="Desired output format", type=str)
    analysis = dspy.OutputField(desc="Analysis results", type=str)
    metadata = dspy.OutputField(desc="Analysis metadata", type=dict)

# Can output in different formats
analyzer = dspy.Predict(
    FlexibleAnalyzer,
    instructions="Adapt your analysis output based on the requested format."
)

# JSON output
json_result = analyzer(
    text="The product exceeded all expectations.",
    analysis_type="sentiment",
    output_format="json"
)

# Markdown output
md_result = analyzer(
    text="The product exceeded all expectations.",
    analysis_type="sentiment",
    output_format="markdown"
)
```

### Conditional Logic in Signatures
```python
class ConditionalProcessor(dspy.Signature):
    """Process data with conditional outputs."""
    input_data = dspy.InputField(desc="Data to process", type=str)
    processing_mode = dspy.InputField(desc="How to process", type=str)
    requires_escalation = dspy.InputField(desc="Whether escalation needed", type=bool, optional=True)
    standard_result = dspy.OutputField(desc="Standard processing result", type=str, optional=True)
    escalated_result = dspy.OutputField(desc="Escalated processing result", type=str, optional=True)
    escalation_reason = dspy.OutputField(desc="Why escalated", type=str, optional=True)

processor = dspy.Predict(ConditionalProcessor)

# Standard processing
standard = processor(
    input_data="Simple customer request",
    processing_mode="standard"
)

print(standard.standard_result)

# Escalated processing
escalated = processor(
    input_data="Complex issue requiring expert attention",
    processing_mode="escalate",
    requires_escalation=True
)

print(escalated.escalated_result)
print(escalated.escalation_reason)
```

## Performance Optimization

### Caching
```python
# Enable caching for repeated queries
cached_analyzer = dspy.Predict(
    "text -> analysis",
    cache=True  # Automatically cache results
)

# Or use a custom cache
import sqlite3

# Create persistent cache
cache_db = {}
persistent_analyzer = dspy.Predict(
    "text -> analysis",
    cache=cache_db
)
```

### Batch Processing
```python
# Process multiple items efficiently
class BatchProcessor(dspy.Signature):
    """Process multiple items in one call."""
    items = dspy.InputField(desc="List of items to process", type=list)
    processing_type = dspy.InputField(desc="How to process items", type=str)
    results = dspy.OutputField(desc="Processed results", type=list)
    summary = dspy.OutputField(desc="Processing summary", type=dict)

batch_processor = dspy.Predict(BatchProcessor)

# Process many emails at once
emails = ["email1 content", "email2 content", "email3 content"]
batch_result = batch_processor(
    items=emails,
    processing_type="classify"
)

# Returns structured results for all items
for item_result in batch_result.results:
    print(item_result)
```

### Token Optimization
```python
# For large inputs, use chunking
class ChunkedAnalyzer(dspy.Signature):
    """Analyze large documents in chunks."""
    document_chunk = dspy.InputField(desc="Chunk of document", type=str)
    chunk_number = dspy.InputField(desc="Chunk position", type=int)
    total_chunks = dspy.InputField(desc="Total number of chunks", type=int)
    chunk_analysis = dspy.OutputField(desc="Analysis of this chunk", type=str)

chunk_analyzer = dspy.Predict(
    ChunkedAnalyzer,
    max_tokens=1000  # Keep responses concise
)
```

## Common Use Cases

### 1. Content Moderation
```python
class ContentModerator(dspy.Signature):
    """Moderate user-generated content."""
    content = dspy.InputField(desc="Content to moderate", type=str)
    content_type = dspy.InputField(desc="Type of content", type=str)
    is_appropriate = dspy.OutputField(desc="Content appropriateness", type=bool)
    issues = dspy.OutputField(desc="Issues found", type=List[str])
    confidence = dspy.OutputField(desc="Moderation confidence", type=float)

moderator = dspy.Predict(
    ContentModerator,
    instructions="Be fair and consistent in moderation. Consider context and intent."
)
```

### 2. Data Validation
```python
class DataValidator(dspy.Signature):
    """Validate data against schema or rules."""
    data = dspy.InputField(desc="Data to validate", type=str)
    schema = dspy.InputField(desc="Validation rules", type=str)
    is_valid = dspy.OutputField(desc="Whether data is valid", type=bool)
    errors = dspy.OutputField(desc("Validation errors", type=List[str]))
    suggestions = dspy.OutputField(desc="How to fix errors", type=List[str])

validator = dspy.Predict(DataValidator)
```

### 3. Format Conversion
```python
class FormatConverter(dspy.Signature):
    """Convert data between formats."""
    source_data = dspy.InputField(desc="Data in source format", type=str)
    source_format = dspy.InputField(desc="Current format", type=str)
    target_format = dspy.InputField(desc="Desired format", type=str)
    converted_data = dspy.OutputField(desc="Data in target format", type=str)
    conversion_notes = dspy.OutputField(desc="Notes about conversion", type=List[str])

converter = dspy.Predict(FormatConverter)
```

## Best Practices

### 1. Keep Signatures Focused
```python
# Good: Single responsibility
sentiment_analyzer = dspy.Predict("text -> sentiment")

# Avoid: Multiple unrelated tasks
# bad_module = dspy.Predict("text -> sentiment, translation, summary, classification")
```

### 2. Use Clear Instructions
```python
# Specific instructions help
classifier = dspy.Predict(
    "resume_text -> job_category",
    instructions="Analyze the resume and assign the most appropriate job category. "
                 "Consider skills, experience, and industry keywords."
)
```

### 3. Validate Outputs
```python
def safe_predict(module, **kwargs):
    """Wrapper to validate outputs."""
    result = module(**kwargs)

    # Check required fields
    if hasattr(result, 'confidence'):
        if result.confidence < 0.5:
            print("Low confidence result!")

    return result
```

### 4. Handle Errors Gracefully
```python
try:
    result = module(input_data="test")
    # Process result
except Exception as e:
    print(f"Module failed: {e}")
    # Use fallback or try again
```

## When to Use Predict

### Use dspy.Predict when:

1. **Simple transformations** - Direct input → output mapping
2. **Classification tasks** - Categorizing text or data
3. **Extraction tasks** - Pulling specific information
4. **Format conversions** - Changing data formats
5. **Quick prototyping** - Fast iteration on ideas

### Consider other modules when:

1. **Complex reasoning needed** → Use ChainOfThought
2. **External tools required** → Use ReAct
3. **Multiple approaches to compare** → Use MultiChainComparison
4. **Step-by-step processing** → Use ProgramOfThought

## Troubleshooting

### Common Issues

1. **Incorrect output format**
   - Check signature field types
   - Add explicit formatting instructions
   - Use examples to demonstrate format

2. **Low confidence results**
   - Add more examples
   - Improve instructions
   - Increase temperature slightly

3. **Slow performance**
   - Enable caching
   - Use smaller model for simple tasks
   - Batch process when possible

4. **Inconsistent results**
   - Lower temperature
   - Add more consistent examples
   - Clarify instructions

### Debugging Predict

```python
# Enable tracing to see what's happening
import dspy
dspy.settings.configure(trace="all")

# Run module
result = module(input="test")

# Check the generated prompt
print("Prompt sent to LLM:")
print(module.lm.last_request_.prompt)

# Check the raw response
print("Raw response from LLM:")
print(module.lm.last_request_.response)
```

## Summary

`dspy.Predict` is the workhorse module of DSPy:

- **Simple to use** - Direct mapping from inputs to outputs
- **Highly configurable** - Instructions, examples, parameters
- **Versatile** - Handles many types of tasks
- **Performant** - Caching and optimization features
- **Reliable** - Structured outputs with validation

### Key Takeaways

1. **Start simple** with Predict, add complexity as needed
2. **Use examples** to improve performance significantly
3. **Configure carefully** for your specific use case
4. **Validate outputs** to ensure reliability
5. **Know when to upgrade** to more complex modules

## Next Steps

- [ChainOfThought Module](./03-chainofthought.md) - Add reasoning capabilities
- [Module Composition](./06-composing-modules.md) - Combine modules
- [Practical Examples](../examples/chapter03/) - See Predict in action
- [Exercises](./07-exercises.md) - Practice with hands-on exercises

## Further Reading

- [DSPy Documentation: Predict](https://dspy-docs.vercel.app/docs/deep-dive/predict)
- [Prompt Engineering Guide](https://dspy-docs.vercel.app/docs/tutorials/prompt-engineering)
- [Module Comparison](./01-module-basics.md) - Choose the right module