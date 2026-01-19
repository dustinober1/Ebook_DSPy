# Chapter 3: Modules

Modules are the workhorses of DSPy - they transform your signatures into executable programs that interact with language models. This chapter teaches you how to use built-in modules and create custom ones for sophisticated AI applications.

---

## What You'll Learn

By the end of this chapter, you will:

- Understand DSPy's module architecture and design philosophy
- Master the `Predict` module for direct LLM interactions
- Use `ChainOfThought` for complex reasoning tasks
- Build intelligent agents with `ReAct`
- Create custom modules for specialized behaviors
- Compose modules into powerful multi-step pipelines

---

## Chapter Overview

This chapter covers the complete module ecosystem in DSPy:

### [Module Basics](01-module-basics.md)
Understand module architecture, lifecycle, and when to use each type.

### [Predict Module](02-predict-module.md)
Master the fundamental module for direct prediction tasks.

### [Chain of Thought](03-chainofthought.md)
Add step-by-step reasoning to improve complex task performance.

### [ReAct Agents](04-react-agents.md)
Build agents that can use tools and take actions in the world.

### [Custom Modules](05-custom-modules.md)
Create your own module types for specialized behaviors.

### [Composing Modules](06-composing-modules.md)
Combine modules into sophisticated multi-step pipelines.

### [Exercises](07-exercises.md)
Practice with 8 hands-on exercises covering all module types.

---

## Prerequisites

Before starting this chapter, ensure you have:

- **Chapter 1**: DSPy Fundamentals completed
- **Chapter 2**: Signatures - solid understanding of signature design
- **Working DSPy setup** with API keys configured
- **Python OOP knowledge** (classes, inheritance, methods)

> **Need signature review?** Complete [Chapter 2: Signatures](../02-signatures/00-chapter-intro.md) first.

---

## Difficulty Level

**Level**: ⭐⭐ Intermediate

This chapter requires understanding of signatures and introduces object-oriented patterns. The progression moves from simple to advanced module usage.

---

## Estimated Time

**Total time**: 5-6 hours

- Reading: 2-2.5 hours
- Running examples: 1.5-2 hours
- Exercises: 1.5-2 hours

---

## The Module Concept

Modules bridge the gap between your intent (signatures) and execution (LLM calls):

```
┌─────────────────────────────────────────────────────────────┐
│                     Your DSPy Application                   │
├─────────────────────────────────────────────────────────────┤
│  Signature:  "question, context -> answer, confidence"      │
│                           │                                 │
│                           ▼                                 │
│  Module:     dspy.ChainOfThought(signature)                 │
│              - Constructs prompt                            │
│              - Adds reasoning steps                         │
│              - Calls LLM                                    │
│              - Parses structured output                     │
│                           │                                 │
│                           ▼                                 │
│  Result:     answer="Paris", confidence=0.95                │
└─────────────────────────────────────────────────────────────┘
```

### Without Modules
```python
# Manual, brittle, hard to optimize
prompt = f"Answer this question step by step: {question}"
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
# Parse response manually... handle errors... no caching...
```

### With Modules
```python
import dspy

# Clean, optimizable, production-ready
qa = dspy.ChainOfThought("question -> answer")
result = qa(question="What is the capital of France?")
print(result.answer)  # "Paris"
print(result.reasoning)  # Step-by-step thought process
```

---

## Module Types at a Glance

| Module | Best For | Example Use Case |
|--------|----------|------------------|
| `Predict` | Direct transformations | Translation, classification |
| `ChainOfThought` | Complex reasoning | Math problems, analysis |
| `ReAct` | Tool-using agents | Research, data gathering |
| `ProgramOfThought` | Code-based reasoning | Calculations, data processing |
| Custom | Specialized behaviors | Domain-specific logic |

---

## Key Concepts Preview

### 1. **Modules as Building Blocks**
Each module encapsulates a specific behavior pattern that you can reuse and compose.

### 2. **Automatic Prompt Engineering**
Modules handle prompt construction, few-shot examples, and output parsing automatically.

### 3. **Optimization Ready**
All modules support DSPy's optimization framework - your programs can improve automatically.

### 4. **Composability**
Modules can be combined like LEGO blocks to build complex applications:

```python
class ResearchPipeline(dspy.Module):
    def __init__(self):
        self.search = dspy.ReAct("query -> findings")
        self.analyze = dspy.ChainOfThought("findings -> insights")
        self.summarize = dspy.Predict("insights -> summary")

    def forward(self, query):
        findings = self.search(query=query)
        insights = self.analyze(findings=findings.findings)
        return self.summarize(insights=insights.insights)
```

---

## Chapter Outline

```
Chapter 3: Modules
│
├── Module Basics
│   ├── What are modules?
│   ├── Module architecture
│   ├── Lifecycle and execution
│   └── Choosing the right module
│
├── Predict Module
│   ├── Basic usage
│   ├── Configuration options
│   ├── Few-shot examples
│   └── Best practices
│
├── Chain of Thought
│   ├── Reasoning patterns
│   ├── When to use CoT
│   ├── Accessing reasoning
│   └── Advanced techniques
│
├── ReAct Agents
│   ├── Agent architecture
│   ├── Tool integration
│   ├── Action-observation loops
│   └── Building effective agents
│
├── Custom Modules
│   ├── Module inheritance
│   ├── The forward() method
│   ├── State management
│   └── Integration patterns
│
├── Composing Modules
│   ├── Sequential pipelines
│   ├── Parallel processing
│   ├── Conditional logic
│   └── Complex architectures
│
└── Exercises
    ├── 8 progressive exercises
    ├── All module types covered
    └── Complete solutions
```

---

## Code Examples

This chapter includes comprehensive examples in `examples/chapter03/`:

- `01_basic_modules.py` - Predict and basic patterns
- `02_chain_of_thought.py` - Reasoning with CoT
- `03_react_agents.py` - Tool-using agent examples
- `04_custom_modules.py` - Building custom module types
- `05_module_composition.py` - Multi-step pipelines

All examples include detailed comments and are ready to run!

---

## Real-World Applications

Modules power real applications across domains:

### Customer Support
```python
# Classify and route support tickets
router = dspy.ChainOfThought("ticket -> category, priority, department")
```

### Content Creation
```python
# Generate and refine content
writer = dspy.Predict("topic, style -> draft")
editor = dspy.ChainOfThought("draft -> improved_draft, changes_made")
```

### Data Analysis
```python
# Analyze data with tool access
analyst = dspy.ReAct("dataset, question -> insights, visualizations")
```

### Research Assistant
```python
# Multi-step research pipeline
class Researcher(dspy.Module):
    def __init__(self):
        self.search = dspy.ReAct("query -> sources")
        self.synthesize = dspy.ChainOfThought("sources -> synthesis")
```

---

## Key Takeaways (Preview)

By chapter end, you'll understand:

1. **Modules execute signatures** - They add behavior to your contracts
2. **Choose modules by task complexity** - Match module type to requirements
3. **ChainOfThought adds reasoning** - Essential for complex tasks
4. **ReAct enables tools** - Build agents that take actions
5. **Composition creates power** - Combine modules for sophisticated systems

---

## Learning Approach

This chapter builds skills progressively:

1. **Start simple** - Master Predict before advancing
2. **Add reasoning** - Learn when and how to use ChainOfThought
3. **Build agents** - Understand ReAct patterns
4. **Go custom** - Create specialized modules
5. **Compose systems** - Build complete applications

> **Tip**: Run every example and experiment with modifications!

---

## Getting Help

As you work through this chapter:

- **Module choice unclear?** Review the Module Selection Guide in Module Basics
- **Code errors?** Check the examples in `examples/chapter03/`
- **Custom module issues?** See the patterns in Custom Modules section
- **Pipeline problems?** Reference Composing Modules

---

## Let's Begin!

Ready to master DSPy modules? Start with [Module Basics](01-module-basics.md) to understand the foundation.

**Remember**: Modules are where your DSPy programs come to life. Understanding them deeply unlocks the full power of the framework!
