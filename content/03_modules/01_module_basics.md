# Module Basics

## Prerequisites

- **Chapter 1**: DSPy Fundamentals - Understanding of DSPy concepts and setup
- **Chapter 2**: Signatures - Complete understanding of signature design and types
- **Required Knowledge**: Basic programming concepts, understanding of classes and methods
- **Difficulty Level**: Intermediate
- **Estimated Reading Time**: 30 minutes

## Learning Objectives

By the end of this section, you will understand:
- What DSPy modules are and why they're essential
- The core module architecture and design patterns
- How modules interact with signatures to create powerful behaviors
- The basic module lifecycle and execution flow
- How to choose the right module for your task

## What are DSPy Modules?

DSPy modules are the fundamental building blocks that transform signatures into executable LLM programs. Think of modules as the "verbs" of DSPy - they define actions and behaviors, while signatures define the "nouns" - the data structures and interfaces.

### The Module-Signature Relationship

```
Signature  =  What to do (input/output contract)
Module     =  How to do it (processing behavior)
Program    =  Modules + Signatures = Complete application
```

Modules take signatures and add:
1. **Processing Logic** - How to transform inputs to outputs
2. **Prompt Engineering** - Automatic prompt generation
3. **LLM Integration** - Communication with language models
4. **Error Handling** - Robust error management
5. **Optimization Hooks** - Points for automatic improvement

### Core Module Features

Every DSPy module provides:
- **Signature Integration** - Seamless connection to defined signatures
- **Prompt Construction** - Automatic generation of effective prompts
- **LLM Abstraction** - Unified interface regardless of LLM provider
- **Structured Output** - Reliable parsing and validation of responses
- **Caching** - Intelligent caching for performance
- **Debugging Support** - Built-in debugging and tracing capabilities

## Module Architecture

### Basic Module Structure

```python
import dspy

class ModuleArchitecture:
    """Demonstrates the internal structure of DSPy modules."""

    def __init__(self, signature, **kwargs):
        # 1. Store the signature
        self.signature = signature

        # 2. Configure language model
        self.lm = kwargs.get('lm', dspy.settings.lm)

        # 3. Set up cache
        self.cache = kwargs.get('cache', {})

        # 4. Configure prompt templates
        self.demos = []  # Few-shot examples
        self.instructions = ""  # Instructions

    def __call__(self, **kwargs):
        # 1. Validate inputs against signature
        # 2. Construct prompt
        # 3. Call LLM
        # 4. Parse and validate outputs
        # 5. Cache results
        pass
```

### Module Types

DSPy provides several module types, each optimized for different tasks:

1. **Predict** - Direct prediction tasks
2. **ChainOfThought** - Multi-step reasoning
3. **ReAct** - Tool-using agents
4. **MultiChainComparison** - Comparing multiple reasoning paths
5. **ProgramOfThought** - Complex programmatic reasoning

## Module Lifecycle

Understanding how modules execute helps in debugging and optimization:

### 1. Initialization
```python
# Module is created with a signature
module = dspy.Predict("question -> answer")

# Internal setup:
# - Parses signature structure
# - Initializes prompt templates
# - Sets up LLM connection
# - Prepares cache and configuration
```

### 2. Input Validation
```python
# When called, module validates inputs
result = module(question="What is AI?")

# Validation checks:
# - All required inputs provided
# - Input types match signature expectations
# - No conflicting inputs
```

### 3. Prompt Construction
```python
# Module builds prompt internally:
# 1. Add instructions
# 2. Include few-shot examples
# 3. Format input fields
# 4. Add output format guidance
```

### 4. LLM Execution
```python
# Module calls LLM with constructed prompt:
# - Handles retries and error recovery
# - Manages token limits
# - Applies configuration settings
```

### 5. Output Processing
```python
# Module processes LLM response:
# - Parses structured outputs
# - Validates against signature
# - Returns typed results
```

## Why Use Modules Instead of Direct LLM Calls?

### Without Modules (Direct Prompting)
```python
import openai

def answer_question(question):
    prompt = f"Please answer this question: {question}"
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Problems:
# - No input validation
# - Unstructured output
# - No error handling
# - Hard to optimize
# - No caching
# - Not reusable
```

### With Modules
```python
import dspy

# Define signature
class QASignature(dspy.Signature):
    """Answer questions based on knowledge."""
    question = dspy.InputField(desc="Question to answer", type=str)
    context = dspy.InputField(desc="Relevant context", type=str, optional=True)
    answer = dspy.OutputField(desc="Answer to the question", type=str)
    confidence = dspy.OutputField(desc="Confidence in answer", type=float)

# Create module
qa_module = dspy.Predict(QASignature)

# Use with all benefits
result = qa_module(question="What is AI?", context="AI is artificial intelligence")
# Returns: result.answer, result.confidence

# Benefits:
# ✅ Input validation
# ✅ Structured output
# ✅ Error handling
# ✅ Automatic optimization
# ✅ Caching
# ✅ Reusability
```

## Module Configuration

### Language Model Configuration
```python
import dspy

# Configure default LM for all modules
dspy.settings.configure(
    lm=dspy.OpenAI(model="gpt-4", api_key="your-key"),
    rm=dspy.Retrieve(k=3)  # For retrieval-augmented modules
)

# Or configure per module
module = dspy.Predict(
    "question -> answer",
    lm=dspy.OpenAI(model="gpt-3.5-turbo")
)
```

### Cache Configuration
```python
# Enable caching for performance
module = dspy.Predict(
    "question -> answer",
    cache=True  # Automatically cache results
)

# Or use custom cache
custom_cache = {}
module = dspy.Predict(
    "question -> answer",
    cache=custom_cache
)
```

### Prompt Configuration
```python
# Add few-shot examples
examples = [
    dspy.Example(question="2+2", answer="4"),
    dspy.Example(question="5*3", answer="15")
]

module = dspy.Predict(
    "math_question -> math_answer",
    demos=examples  # Few-shot examples
)

# Add custom instructions
module = dspy.Predict(
    "task -> result",
    instructions="Think step by step and show your work."
)
```

## Module Selection Guide

Choose the right module based on your task complexity:

### Simple Direct Tasks → Use `dspy.Predict`
```python
# When you have a straightforward input → output mapping
translator = dspy.Predict("source_text, target_language -> translated_text")
classifier = dspy.Predict("email_text -> category, urgency")
summarizer = dspy.Predict("long_document -> short_summary")
```

### Reasoning Tasks → Use `dspy.ChainOfThought`
```python
# When the task requires step-by-step thinking
math_solver = dspy.ChainOfThought("math_problem -> solution, steps")
diagnostic_tool = dspy.ChainOfThought("symptoms -> diagnosis, reasoning")
planner = dspy.ChainOfThought("goal -> action_plan, alternatives")
```

### Tool-Using Tasks → Use `dspy.ReAct`
```python
# When the module needs to use external tools
researcher = dspy.ReAct("question -> research_answer, sources")
calculator = dspy.ReAct("calculation -> result, steps")
data_analyst = dspy.ReAct("dataset -> insights, visualizations")
```

### Comparison Tasks → Use `dspy.MultiChainComparison`
```python
# When comparing multiple approaches
decision_maker = dspy.MultiChainComparison("options -> recommendation, pros_cons")
evaluator = dspy.MultiChainComparison("solutions -> best_solution, criteria")
```

## Module Best Practices

### 1. Clear Signatures
```python
# Good - Clear, specific signature
class ProductReviewAnalyzer(dspy.Signature):
    review_text = dspy.InputField(desc="Customer review text", type=str)
    product_category = dspy.InputField(desc="Category of product", type=str)
    sentiment = dspy.OutputField(desc="Overall sentiment", type=str)
    key_points = dspy.OutputField(desc="Main feedback points", type=list)

# Avoid - Vague signature
class BadAnalyzer(dspy.Signature):
    text = dspy.InputField()
    output = dspy.OutputField()
```

### 2. Appropriate Module Selection
```python
# For simple classification
classifier = dspy.Predict("text -> category")  # Good
# Not: classifier = dspy.ReAct("text -> category")  # Unnecessary complexity

# For complex reasoning
reasoner = dspy.ChainOfThought("complex_problem -> solution")  # Good
# Not: reasoner = dspy.Predict("complex_problem -> solution")  # May fail
```

### 3. Proper Error Handling
```python
try:
    result = module(input_data="test")
    # Process result
except AttributeError as e:
    # Handle signature mismatches
    print(f"Invalid input: {e}")
except Exception as e:
    # Handle other errors
    print(f"Module error: {e}")
```

### 4. Performance Optimization
```python
# Enable caching for repeated operations
module = dspy.Predict(
    "input -> output",
    cache=True,
    lm=dspy.OpenAI(model="gpt-3.5-turbo")  # Use faster model for simple tasks
)

# Use batch processing when possible
batch_module = dspy.Predict("batch_input -> batch_output")
```

## Debugging Modules

### Enable Tracing
```python
import dspy

# Enable detailed tracing
dspy.settings.configure(trace="all")

# Run module
result = module(input="test")

# Access trace information
print(dspy.settings.trace)
```

### Inspect Prompts
```python
# Module stores the last prompt used
module = dspy.Predict("question -> answer")
result = module(question="What is AI?")

# View the generated prompt
print("Generated Prompt:")
print(module.last_request_.prompt)
```

### Check Examples
```python
# View few-shot examples being used
print("Module Examples:")
for example in module.demos:
    print(example)
```

## Summary

DSPy modules are powerful abstractions that:
- **Transform signatures** into executable programs
- **Handle complexity** of LLM interaction
- **Provide structure** for reliable applications
- **Enable optimization** through automatic prompt improvement
- **Support composition** for building complex systems

### Key Takeaways

1. **Modules are behavior**: They define how to process data
2. **Signatures are structure**: They define data flow
3. **Choose modules based on task complexity**
4. **Leverage built-in features** (caching, tracing, validation)
5. **Compose modules** to build sophisticated applications

## Next Steps

- [Next Section: Predict Module](./02-predict-module.md) - Learn the most fundamental module
- [ChainOfThought Module](./03-chainofthought.md) - Add reasoning capabilities
- [ReAct Agents](./04-react-agents.md) - Build tool-using agents
- [Custom Modules](./05-custom-modules.md) - Create your own module types

## Further Reading

- [DSPy Documentation: Modules](https://dspy-docs.vercel.app/docs/modules)
- [Module Examples](../examples/chapter03/) - Practical implementations
- [Advanced Patterns](../06-real-world-applications/) - Real-world applications