# Chapter 9: Appendices - Practice Problems

## Overview

These practice problems help reinforce concepts from throughout the ebook using the reference materials in Chapter 9. They focus on applying the API reference, troubleshooting techniques, and resources effectively.

**Difficulty**: Beginner-Intermediate
**Time Estimate**: 1-2 hours total
**Prerequisites**: Basic familiarity with any previous chapter

---

## Problem 1: API Reference Navigation

**Objective**: Become comfortable using the API Reference Quick Guide to solve problems.

### Part A: Find the Right API

For each task description, identify which DSPy API you would use:

1. "I want to perform a simple prediction without showing reasoning steps"
   - Answer: `dspy.Predict`

2. "I need to show the model's reasoning steps to improve accuracy"
   - Answer: `dspy.ChainOfThought`

3. "I want my program to use tools to accomplish tasks"
   - Answer: `dspy.ReAct`

4. "I need to define what inputs and outputs my task requires"
   - Answer: `dspy.Signature` or `dspy.InputField`/`dspy.OutputField`

5. "I want to automatically find good examples for in-context learning"
   - Answer: `dspy.BootstrapFewShot`

### Part B: Read an Example

Study the API Reference for `dspy.ChainOfThought`. Answer:

1. What module does it extend?
2. What additional output field does it provide?
3. When would you use it instead of `Predict`?
4. Write a simple code example using it

**Expected Answers**:
- Extends `Predict`
- Provides `reasoning` field
- When task requires multiple reasoning steps
- See code example in reference

---

## Problem 2: Troubleshooting Scenarios

**Objective**: Apply the Troubleshooting Guide to diagnose and fix problems.

### Scenario 1: API Key Error

```python
import dspy

# This fails with: AuthenticationError: Invalid API key
predictor = dspy.Predict("question -> answer")
result = predictor(question="What is DSPy?")
```

**Questions**:
1. What section of the troubleshooting guide applies? (Answer: "API Keys and Authentication")
2. What is the root cause? (Answer: LM not configured before use)
3. How would you fix it? (Answer: Call `dspy.configure()` with valid API key)

**Fixed Code**:
```python
import os
from dotenv import load_dotenv

load_dotenv()
dspy.configure(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

predictor = dspy.Predict("question -> answer")
result = predictor(question="What is DSPy?")
```

### Scenario 2: Empty Responses

```python
# Predictions return empty strings
predictor = dspy.Predict("document -> summary")
for doc in large_documents:
    result = predictor(document=doc)
    if not result.summary:
        print(f"Empty summary for: {doc[:50]}...")
```

**Questions**:
1. What section applies? (Answer: "Language Model Issues" → "Empty or None responses")
2. What are the root causes? (Answer: max_tokens too low, document too long, signature unclear)
3. What are 3 solutions? (Answer: Increase max_tokens, summarize docs first, improve signature)

---

## Problem 3: Resource Discovery

**Objective**: Use the Resources guide to find answers to hypothetical questions.

For each question, identify which resource would be most helpful:

1. **Question**: "How does the Transformer architecture work?"
   - **Best Resource**: Attention Is All You Need paper or deep learning courses

2. **Question**: "I want to join the DSPy community and discuss ideas"
   - **Best Resource**: Stanford NLP Discord channel

3. **Question**: "I need official DSPy API documentation"
   - **Best Resource**: DSPy GitHub repository documentation

4. **Question**: "How do vector databases compare?"
   - **Best Resource**: Vector database documentation (Pinecone, Weaviate, etc.)

5. **Question**: "I want to learn about prompt engineering best practices"
   - **Best Resource**: Prompt Engineering Guide or DSPy papers

### Create Your Own Research Task

Design a research task and identify 3 resources you would consult:

**Example Task**: "Build a medical diagnosis assistant using DSPy"

**Resources**:
1. DSPy GitHub examples and Case Studies chapter
2. Healthcare NLP research papers
3. Medical knowledge base APIs or datasets

**Your Task**: [Design a task] → [Find 3 resources]

---

## Problem 4: Glossary Application

**Objective**: Use the Glossary to understand unfamiliar terms in context.

### Part A: Term Matching

Match each term with its correct definition from the Glossary:

1. **RAG** → A. Retrieval-Augmented Generation - combining documents with generation
2. **Signature** → B. Input/output contract specification for a task
3. **Module** → C. Composable unit performing a task in DSPy
4. **Metric** → D. Function measuring prediction correctness
5. **Optimization** → E. Improving program performance using training data

### Part B: Contextual Understanding

Read each scenario and define the underlined term from the glossary:

1. "Our **embedding** model converts sentences to 768-dimensional vectors"
   - Definition: Vector representation of text capturing semantic meaning

2. "We applied **fine-tuning** to adapt the model for our domain"
   - Definition: Training a pre-trained model on task-specific data

3. "The **hallucination** issue caused the model to generate false information"
   - Definition: When LLM generates plausible-sounding but false information

4. "Our **in-context learning** strategy uses examples in the prompt"
   - Definition: Learning from examples provided in the prompt without parameter updates

### Part C: Cross-Reference

Use multiple glossary entries to explain a concept:

**Concept**: "How does optimization work in DSPy?"

**Glossary Terms** (find and apply these):
- Module: What you're optimizing
- Metric: How you measure success
- Demonstration: What gets optimized
- Bootstrap: A specific optimization technique

**Your Explanation**: [Write 2-3 sentences using these terms]

---

## Problem 5: Integration Practice

**Objective**: Use multiple Chapter 9 resources together to solve a real problem.

### Scenario: Building a Production FAQ System

You're building a frequently-asked-questions system using DSPy. You encounter several challenges:

**Challenge 1: API Error**
```python
# Error: "ModuleNotFoundError: No module named 'dspy'"
```
- **Solution Path**: Check Installation section of Troubleshooting → Use pip install

**Challenge 2: Low Accuracy**
- **Research**: Check API Reference for optimization modules → Review Case Studies for similar problems
- **Solution**: Use BootstrapFewShot to improve demonstration selection

**Challenge 3: Slow Responses**
- **Research**: Check Advanced Topics chapter → Read Caching section in Troubleshooting
- **Solution**: Implement caching for frequently asked questions

**Challenge 4: Understanding a Concept**
- **Research**: Use Glossary to look up unfamiliar terms → Read API Reference for implementation details
- **Solution**: Better understand through definitions and code examples

### Your Turn: Real-World System

Describe a DSPy system you want to build:

1. **System Description**: [What does it do?]
2. **Potential Issues**: [What could go wrong?]
3. **Research Plan**: [What Chapter 9 resources would help?]
   - Troubleshooting section: _______________
   - API reference section: _______________
   - Glossary terms: _______________
   - External resources: _______________

---

## Problem 6: Comparative Analysis

**Objective**: Use API Reference to understand trade-offs between approaches.

### Part A: Module Comparison

Compare three predictor types using the API Reference:

| Aspect | Predict | ChainOfThought | ReAct |
|--------|---------|-----------------|-------|
| Output fields | answer | reasoning, answer | actions, final_answer |
| When to use | Direct Q&A | Complex reasoning | With tools |
| Speed | Fast | Slower | Slowest |
| Accuracy | Moderate | High | Highest |
| Complexity | Low | Medium | High |

### Part B: Optimizer Comparison

Use the API Reference to compare optimizers:

1. **BootstrapFewShot**:
   - Pros: Fast, automatic demonstration selection
   - Cons: Limited to example selection
   - Best for: Quick improvements on small datasets

2. **MIPRO**:
   - Pros: Optimizes instructions AND examples
   - Cons: More complex, slower
   - Best for: Maximum performance improvement

3. **KNNFewShot**:
   - Pros: Similarity-based, interpretable
   - Cons: Requires embedding space
   - Best for: Explicit example selection

**Question**: For your FAQ system, which optimizer would you choose and why?

---

## Problem 7: Documentation Navigation

**Objective**: Practice using official documentation efficiently.

### Information Scavenger Hunt

Using the Resources section, find answers to:

1. **DSPy Repository**: What is the current version number?
   - Go to: https://github.com/stanfordnlp/dspy
   - Look for: releases or version file

2. **API Stability**: Are any APIs marked as unstable/experimental?
   - Go to: DSPy documentation
   - Look for: API stability information

3. **Community**: When is the next meetup or community event?
   - Go to: Stanford NLP Discord
   - Look for: event announcements

4. **Benchmarks**: What performance benchmarks are reported in papers?
   - Go to: DSPy and MIPRO papers
   - Look for: results tables

---

## Problem 8: Synthesis Challenge

**Objective**: Combine all Chapter 9 resources to create a comprehensive reference guide.

### Create a Custom Quick Reference Card

Design a one-page quick reference specifically for YOUR use case:

**Instructions**:
1. Pick a DSPy use case (e.g., "Building a Customer Support Chatbot")
2. Extract relevant terms from the Glossary
3. Identify key APIs from the API Reference
4. List common issues from Troubleshooting
5. Note relevant external resources

**Example Format**:
```
┌─ Customer Support Chatbot - Quick Reference ─┐
│                                               │
│ Key Modules:                                  │
│ - dspy.ChainOfThought (for reasoning)        │
│ - dspy.Predict (for intent classification)   │
│                                               │
│ Key Terms:                                    │
│ - Intent: User's underlying goal             │
│ - Slot Filling: Extracting specific info     │
│                                               │
│ Common Issues:                                │
│ - Low accuracy → Use BootstrapFewShot        │
│ - Slow responses → Add caching               │
│                                               │
│ Resources:                                    │
│ - DSPy GitHub: examples/chatbot              │
│ - Paper: Chain-of-Thought Prompting          │
└─────────────────────────────────────────────┘
```

**Your Task**: Create a similar card for your use case

---

## Answer Key

Solutions and discussions available:

- **Problem 1**: See API Reference sections
- **Problem 2**: See Troubleshooting Guide sections
- **Problem 3**: See Resources section
- **Problem 4**: See Glossary entries
- **Problems 5-8**: Example solutions available in `solutions/` directory

## Learning Outcomes

After completing these practice problems, you should be able to:

✓ Navigate and use the API Reference effectively
✓ Apply troubleshooting techniques to common issues
✓ Find and use appropriate resources
✓ Understand DSPy terminology precisely
✓ Make informed decisions between different approaches
✓ Work with official documentation confidently

## Next Steps

1. **Review**: Revisit any Chapter 9 section that was unclear
2. **Practice**: Try the synthesis challenges with your own project ideas
3. **Contribute**: Share your custom quick reference cards with the community
4. **Build**: Apply Chapter 9 knowledge while building your DSPy system

---

**Difficulty Level**: Beginner-Intermediate
**Estimated Time**: 1-2 hours
**Best Used**: After completing earlier chapters or as reference while building
