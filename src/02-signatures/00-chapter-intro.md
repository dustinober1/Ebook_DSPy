# Chapter 2: Signatures

Signatures are the foundation of structured LLM programming in DSPy. This chapter teaches you how to define clear input/output contracts that transform ambiguous prompts into reliable, composable, and optimizable specifications.

---

## What You'll Learn

By the end of this chapter, you will:

- Understand what signatures are and why they matter
- Master both string-based and class-based signature syntax
- Create typed signatures with field descriptions and constraints
- Design advanced multi-field signatures for complex tasks
- Apply signatures to real-world problems across multiple domains
- Build a library of reusable signature patterns

---

## Chapter Overview

This chapter covers everything you need to master DSPy signatures:

### [Understanding Signatures](01-understanding-signatures.md)
Learn what signatures are, why they're essential, and how they differ from traditional prompts.

### [Signature Syntax](02-signature-syntax.md)
Master the string-based syntax for quick signature definitions.

### [Typed Signatures](03-typed-signatures.md)
Create class-based signatures with type hints, descriptions, and validation.

### [Advanced Signatures](04-advanced-signatures.md)
Design complex signatures with multiple fields, optional inputs, and structured outputs.

### [Practical Examples](05-practical-examples.md)
See 10+ real-world signature patterns across different domains and use cases.

### [Exercises](06-exercises.md)
Practice your skills with 6 hands-on exercises.

---

## Prerequisites

Before starting this chapter, ensure you have:

- **Chapter 1**: DSPy Fundamentals completed
- **Working DSPy installation** with configured LM
- **Basic Python knowledge** (classes, type hints)
- **Understanding of function signatures** in programming

> **New to DSPy?** Complete [Chapter 1: DSPy Fundamentals](../01-fundamentals/00-chapter-intro.md) first.

---

## Difficulty Level

**Level**: ⭐⭐ Intermediate

This chapter builds on fundamental concepts and introduces more sophisticated patterns. You should be comfortable with basic DSPy operations before proceeding.

---

## Estimated Time

**Total time**: 4-5 hours

- Reading: 1.5-2 hours
- Running examples: 1-1.5 hours
- Exercises: 1.5-2 hours

---

## Why Signatures Matter

Signatures transform the way you interact with language models:

### Without Signatures
```python
# Ambiguous, hard to maintain
prompt = "Analyze this customer review and tell me if it's positive or negative"
response = llm.complete(prompt + review_text)
# What format is the response? How do you parse it?
```

### With Signatures
```python
import dspy

# Clear contract with structured output
class ReviewAnalysis(dspy.Signature):
    """Analyze customer review sentiment."""
    review: str = dspy.InputField(desc="Customer review text")
    sentiment: str = dspy.OutputField(desc="positive, negative, or neutral")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")
    key_points: list = dspy.OutputField(desc="Main points from review")

analyzer = dspy.Predict(ReviewAnalysis)
result = analyzer(review="Great product, fast shipping!")
# result.sentiment = "positive"
# result.confidence = 0.95
# result.key_points = ["quality", "shipping speed"]
```

---

## Key Concepts Preview

### 1. **Signatures as Contracts**
Signatures define explicit input/output agreements, making your LLM programs predictable and testable.

### 2. **Two Syntax Options**
- **String syntax**: Quick and concise - `"question -> answer"`
- **Class syntax**: Rich and typed - Full Python classes with metadata

### 3. **Field Descriptions**
Add context that helps the LLM understand exactly what you need:
```python
answer = dspy.OutputField(desc="A concise 2-3 sentence answer")
```

### 4. **Type Safety**
Specify expected types for validation and documentation:
```python
score: float = dspy.OutputField(desc="Score between 0 and 1")
```

### 5. **Composition**
Signatures can be chained for multi-step pipelines:
```
document -> summary
summary -> key_insights
key_insights -> action_items
```

---

## Chapter Outline

```
Chapter 2: Signatures
│
├── Understanding Signatures
│   ├── What is a signature?
│   ├── Signatures vs traditional prompts
│   └── Benefits and use cases
│
├── Signature Syntax
│   ├── String-based syntax
│   ├── Field naming conventions
│   └── Common patterns
│
├── Typed Signatures
│   ├── Class-based definitions
│   ├── InputField and OutputField
│   ├── Field descriptions
│   └── Type hints
│
├── Advanced Signatures
│   ├── Multiple inputs and outputs
│   ├── Optional fields
│   ├── Nested structures
│   └── Complex constraints
│
├── Practical Examples
│   ├── Text analysis
│   ├── Content generation
│   ├── Data extraction
│   └── Domain-specific applications
│
└── Exercises
    ├── 6 hands-on exercises
    ├── Progressive difficulty
    └── Complete solutions
```

---

## Code Examples

This chapter includes complete code examples in `examples/chapter02/`:

- `01_basic_signatures.py` - String and class-based basics
- `02_typed_signatures.py` - Type hints and descriptions
- `03_advanced_signatures.py` - Complex multi-field patterns
- `04_real_world_applications.py` - Domain-specific examples
- `05_signature_composition.py` - Building signature pipelines

All examples are tested and ready to run!

---

## Key Takeaways (Preview)

By chapter end, you'll understand:

1. **Signatures define contracts** - Clear input/output specifications
2. **Two syntax options** - String for simplicity, class for power
3. **Descriptions matter** - Field metadata improves LLM performance
4. **Types add safety** - Validation and documentation benefits
5. **Composition enables complexity** - Chain signatures for pipelines

---

## Learning Approach

This chapter emphasizes practical application:

1. **Understand the concept** - Clear explanations with analogies
2. **See the syntax** - Multiple examples of each pattern
3. **Apply to real problems** - Domain-specific use cases
4. **Practice with exercises** - Hands-on reinforcement

> **Tip**: Try modifying the examples to match your own use cases!

---

## Getting Help

As you work through this chapter:

- **Syntax confusion?** Refer to the syntax reference sections
- **Code not working?** Check examples in `examples/chapter02/`
- **Need more examples?** See the Practical Examples section
- **Conceptual questions?** Re-read Understanding Signatures

---

## Let's Begin!

Ready to master DSPy signatures? Start with [Understanding Signatures](01-understanding-signatures.md) to build a solid foundation.

**Remember**: Signatures are the backbone of DSPy programming. Time invested here pays dividends throughout the rest of your DSPy journey!
