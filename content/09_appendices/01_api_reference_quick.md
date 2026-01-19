# API Reference Quick Guide

This is a concise reference for DSPy's most commonly used classes, methods, and functions. For complete documentation, visit the [official DSPy documentation](https://github.com/stanfordnlp/dspy).

## Table of Contents

- [Initialization](#initialization)
- [Signatures](#signatures)
- [Modules](#modules)
- [Language Models](#language-models)
- [Predictors](#predictors)
- [Evaluation](#evaluation)
- [Optimization](#optimization)
- [Utilities](#utilities)

## Initialization

### `dspy.configure()`

Configure DSPy with a language model and other settings.

```python
dspy.configure(
    default='openai',  # or 'anthropic', 'local', etc.
    api_key='...',
    model='gpt-4',
    temperature=0.7,
    max_tokens=1000
)
```

**Parameters:**
- `default` (str): Default LM to use
- `api_key` (str): API key for the service
- `model` (str): Model name/identifier
- `temperature` (float): Sampling temperature (0-1)
- `max_tokens` (int): Maximum output tokens

## Signatures

### `dspy.Signature`

Base class for defining input/output contracts using Python syntax.

```python
class QuestionAnswer(dspy.Signature):
    """Answer questions about documents."""

    context: str = dspy.InputField(desc="May contain relevant facts")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Often between 1-5 words")
```

**Common Fields:**
- `dspy.InputField(desc="...")` - Define input field with description
- `dspy.OutputField(desc="...")` - Define output field with description

### String Signatures

Simple signature syntax using strings:

```python
"context, question -> answer"
"input -> output"
"question, document -> answer, confidence"
```

**Format:** `input_fields -> output_fields`

## Modules

### `dspy.Predict`

Basic predictor for question-answering tasks.

```python
predictor = dspy.Predict("question -> answer")
result = predictor(question="What is DSPy?")
print(result.answer)
```

### `dspy.ChainOfThought`

Enhanced predictor with reasoning steps.

```python
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="What is 2 + 2?")
print(result.reasoning)
print(result.answer)
```

### `dspy.ReAct`

Agent module combining reasoning and tool use.

```python
react = dspy.ReAct(signature)
result = react(input=...)
```

### `dspy.Module`

Base class for creating custom modules.

```python
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("input -> output")

    def forward(self, input):
        return self.predictor(input=input)
```

### `dspy.InputField` and `dspy.OutputField`

Define fields within signatures.

```python
input_field = dspy.InputField(desc="Description of input")
output_field = dspy.OutputField(desc="Description of output")
```

## Language Models

### OpenAI Models

```python
lm = dspy.OpenAI(
    api_key="sk-...",
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)
dspy.configure(lm=lm)
```

### Anthropic (Claude) Models

```python
lm = dspy.Anthropic(
    api_key="sk-ant-...",
    model="claude-3-opus-20240229",
    max_tokens=1000
)
dspy.configure(lm=lm)
```

### Local Models

```python
lm = dspy.LocalModel(
    path="path/to/model",
    provider="ollama"  # or "vllm", "llamacpp"
)
dspy.configure(lm=lm)
```

## Predictors

### `dspy.Predict.forward()`

Execute a predictor.

```python
result = predictor.forward(question="What is AI?")
# or
result = predictor(question="What is AI?")
```

**Returns:** `Prediction` object with output fields

### Prediction Object

Access outputs from predictions:

```python
result = predictor(question="What is DSPy?")
print(result.answer)          # Access output
print(result[0])              # First output by position
print(dict(result))           # Convert to dictionary
```

## Evaluation

### `dspy.evaluate.Evaluate`

Core evaluation class.

```python
evaluator = dspy.evaluate.Evaluate(
    devset=dev_set,
    metric=metric_fn,
    num_threads=4,
    display_progress=True
)
score = evaluator(program)
```

### Metric Functions

Create custom metrics:

```python
def metric_fn(example, pred, trace=None):
    """
    Returns True if prediction is correct, False otherwise.
    """
    expected = example.expected_answer
    predicted = pred.answer
    return expected.lower() == predicted.lower()
```

### Common Metrics

```python
from dspy.evaluate import Metrics

# Exact match
em_metric = Metrics.exact_match

# Case-insensitive match
ci_metric = Metrics.case_insensitive_match

# F1 score (for span evaluation)
f1_metric = Metrics.f1
```

## Optimization

### `dspy.BootstrapFewShot`

Automatically find and use good in-context examples.

```python
optimizer = dspy.BootstrapFewShot(
    metric=metric_fn,
    max_bootstrapped_demos=4,
    max_rounds=10
)
optimized_program = optimizer.compile(
    student=program,
    trainset=train_set
)
```

### `dspy.MIPRO`

Instruction and demonstration optimization.

```python
optimizer = dspy.MIPRO(
    metric=metric_fn,
    num_candidates=10,
    infer_lr=0.1
)
optimized_program = optimizer.compile(
    student=program,
    trainset=train_set
)
```

### `dspy.KNNFewShot`

Use k-nearest neighbors to select examples.

```python
optimizer = dspy.KNNFewShot(
    k=3,
    metric=metric_fn
)
optimized_program = optimizer.compile(
    student=program,
    trainset=train_set
)
```

### `dspy.MIPROv2`

Advanced optimization combining multiple strategies.

```python
optimizer = dspy.MIPROv2(
    metric=metric_fn,
    num_candidates=10
)
optimized_program = optimizer.compile(
    student=program,
    trainset=train_set,
    valset=val_set
)
```

## Utilities

### Dataset Management

```python
# Create examples
example = dspy.Example(
    question="What is AI?",
    answer="Artificial Intelligence",
    context="AI is the field of creating intelligent machines."
)

# Create from dict
example = dspy.Example(**{
    'question': 'What is ML?',
    'answer': 'Machine Learning'
})

# Convert to/from dict
example_dict = dict(example)
```

### Tracing and Debugging

```python
# Enable tracing
dspy.settings.trace = True

# Run prediction with tracing
result = predictor(question="What is DSPy?")

# Access trace
if hasattr(result, '_trace'):
    print(result._trace)
```

### Caching

```python
# Enable caching
dspy.settings.cache = True

# Clear cache
dspy.settings.cache_clear()
```

## Common Patterns

### Question Answering

```python
qa = dspy.ChainOfThought("context, question -> answer")
result = qa(
    context="DSPy is a framework for LLM programs.",
    question="What is DSPy?"
)
```

### Classification

```python
classify = dspy.Predict("text -> label")
result = classify(text="This product is excellent!")
```

### Summarization

```python
summarize = dspy.Predict("document -> summary")
result = summarize(document=long_text)
```

### Multi-turn Conversation

```python
class ConversationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.dialogue = dspy.ChainOfThought("history, user_input -> response")

    def forward(self, history, user_input):
        return self.dialogue(history=history, user_input=user_input)
```

## Quick Reference

| Task | Code |
|------|------|
| Basic Q&A | `dspy.Predict("q -> a")` |
| Reasoning | `dspy.ChainOfThought("q -> a")` |
| Agent | `dspy.ReAct(signature)` |
| Configure LM | `dspy.configure(lm=...)` |
| Evaluate | `dspy.evaluate.Evaluate(...)` |
| Optimize | `dspy.BootstrapFewShot(...)` |
| Custom Module | `class MyModule(dspy.Module)` |

---

**Version Note:** This reference is based on DSPy 2.5+. Check the [official documentation](https://github.com/stanfordnlp/dspy) for the latest API changes.
