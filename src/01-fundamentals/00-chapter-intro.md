# Chapter 1: DSPy Fundamentals

Welcome to your DSPy journey! This chapter introduces you to DSPy and gets you started building your first LM-powered applications.

---

## What You'll Learn

By the end of this chapter, you will:

- ✅ Understand what DSPy is and why it matters
- ✅ Grasp the paradigm shift from prompting to programming
- ✅ Install and configure DSPy in your development environment
- ✅ Write and run your first DSPy program
- ✅ Configure and work with different language models
- ✅ Build simple question-answering applications

---

## Chapter Overview

This chapter covers the essential foundations you need to start building with DSPy:

### [What is DSPy?](01-what-is-dspy.md)
Learn what DSPy is, why it was created, and how it differs from traditional prompt engineering approaches.

### [Programming vs. Prompting](02-programming-vs-prompting.md)
Understand the fundamental paradigm shift from manual prompt engineering to programmatic LM pipelines.

### [Installation and Setup](03-installation-setup.md)
Get DSPy installed and verify your environment is ready for development.

### [Your First DSPy Program](04-first-dspy-program.md)
Write and run a complete DSPy application from scratch.

### [Language Models](05-language-models.md)
Learn how to configure and work with different LM providers (OpenAI, Anthropic, local models).

### [Exercises](06-exercises.md)
Practice what you've learned with hands-on exercises.

---

## Prerequisites

Before starting this chapter, ensure you have:

- ✅ **Python 3.9+** installed
- ✅ **Basic Python knowledge** (functions, classes, imports)
- ✅ **Virtual environment** set up (from setup instructions)
- ✅ **API key** for at least one LM provider
- ✅ **Text editor or IDE** ready to use

> **Need help with prerequisites?** Review [Chapter 0: Prerequisites](../00-frontmatter/02-prerequisites.md)

---

## Difficulty Level

**Level**: ⭐ Beginner

This chapter is designed for complete beginners to DSPy. No prior experience with DSPy or advanced LLM concepts is required.

---

## Estimated Time

**Total time**: 3-4 hours

- Reading: 1-1.5 hours
- Running examples: 1 hour
- Exercises: 1-1.5 hours

Feel free to spread this over multiple sessions!

---

## What Makes DSPy Different?

Before diving in, here's a quick preview of what makes DSPy special:

### Traditional Prompting
```python
# Manual prompt engineering
prompt = """
You are a helpful assistant. Answer the question clearly.

Question: What is the capital of France?
Answer:
"""

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

**Problems**:
- Brittle and hard to maintain
- Doesn't compose well
- Manual tuning required
- No systematic optimization

### DSPy Approach
```python
import dspy

# Define the task signature
class QuestionAnswer(dspy.Signature):
    """Answer questions clearly."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Use it with automatic prompting
qa = dspy.Predict(QuestionAnswer)
response = qa(question="What is the capital of France?")
```

**Benefits**:
- Clean, modular code
- Easy to compose and reuse
- Automatically optimizable
- Systematic improvement

---

## Learning Approach

This chapter uses a hands-on approach:

1. **Concepts**: Clear explanations of core ideas
2. **Examples**: Working code you can run
3. **Practice**: Exercises to reinforce learning
4. **Experimentation**: Encouragement to modify and explore

> **Tip**: Don't just read—run every example and complete the exercises!

---

## Chapter Outline

```
Chapter 1: DSPy Fundamentals
│
├── What is DSPy?
│   ├── The problem with manual prompting
│   ├── DSPy's solution
│   └── Key concepts overview
│
├── Programming vs. Prompting
│   ├── The paradigm shift
│   ├── Declarative vs. imperative
│   └── Benefits of the DSPy approach
│
├── Installation and Setup
│   ├── Installing DSPy
│   ├── Verifying installation
│   └── Common issues
│
├── Your First DSPy Program
│   ├── Hello World in DSPy
│   ├── Breaking down the code
│   └── Running and testing
│
├── Language Models
│   ├── Configuring LM providers
│   ├── Model selection
│   └── Best practices
│
└── Exercises
    ├── 5 hands-on exercises
    ├── Progressive difficulty
    └── Solutions with explanations
```

---

## Code Examples

This chapter includes several complete code examples in the `examples/chapter01/` directory:

- `01_hello_dspy.py` - Your first DSPy program
- `02_basic_qa.py` - Simple question-answering
- `03_configure_lm.py` - Language model configuration
- Additional examples for experimentation

All examples are self-contained and runnable!

---

## Key Takeaways (Preview)

By the end of this chapter, you'll understand:

1. **DSPy is a framework** for programming (not prompting) LM-based applications
2. **Signatures define tasks** declaratively using input/output specifications
3. **Modules are composable** building blocks that can be optimized automatically
4. **LMs are configurable** and DSPy works with multiple providers
5. **Programming > Prompting** for building robust, maintainable applications

---

## Getting Help

As you work through this chapter:

- **Stuck on a concept?** Re-read the relevant section
- **Code not working?** Check the troubleshooting section
- **Need more examples?** Review the code in `examples/chapter01/`
- **Want deeper knowledge?** Check the additional resources at the end of each section

---

## Let's Begin!

Ready to learn DSPy? Start with [What is DSPy?](01-what-is-dspy.md) to understand the fundamentals.

**Remember**: Learning is a journey. Take your time, experiment freely, and don't hesitate to ask questions (in the community forums) when you need help.

Happy learning! 🚀
