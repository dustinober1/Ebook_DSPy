# What is DSPy?

DSPy (Declarative Self-improving Language Programs, yeah!) is a framework for programming—not prompting—foundation models like GPT-4, Claude, and others. It provides a systematic way to build LM-based applications that are modular, composable, and automatically optimizable.

---

## The Problem: Manual Prompt Engineering

Before understanding DSPy, let's look at the traditional approach to working with LLMs.

### Traditional Prompt Engineering

When you want an LLM to perform a task, you typically write a prompt:

```python
import openai

# Manual prompt for question answering
prompt = """
You are a knowledgeable assistant. Answer the following question accurately and concisely.

Question: What is the capital of France?

Provide your answer in a single sentence.
"""

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

This works for simple cases, but scaling this approach leads to significant problems.

---

## Problems with Manual Prompting

### 1. **Brittle and Hard to Maintain**

```python
# Prompt for sentiment analysis
sentiment_prompt = """
Analyze the sentiment of this text and classify it as positive, negative, or neutral.
Be careful to consider context and sarcasm.
Respond with only the sentiment label.

Text: {text}
Sentiment:
"""
```

**Issues**:
- What if the model doesn't follow the "only label" instruction?
- How do you handle edge cases consistently?
- Changes require manual testing of the entire prompt

### 2. **Doesn't Compose Well**

Suppose you want to chain multiple steps:

```python
# Step 1: Summarize
summary_prompt = f"Summarize this: {document}"
summary = call_llm(summary_prompt)

# Step 2: Extract entities
entity_prompt = f"Extract entities from: {summary}"
entities = call_llm(entity_prompt)

# Step 3: Classify
classification_prompt = f"Classify these entities: {entities}"
result = call_llm(classification_prompt)
```

**Issues**:
- Error propagation through the pipeline
- No systematic way to optimize the entire flow
- Debugging is a nightmare

### 3. **No Systematic Optimization**

How do you improve this?

```python
qa_prompt = """
Answer the question based on the context.

Context: {context}
Question: {question}
Answer:
"""
```

**Manual approach**:
- Try different phrasings
- Add examples manually
- Test each variation
- No guarantee of improvement

This is like trying to train a neural network by manually adjusting weights!

---

## The Solution: DSPy

DSPy changes the game by letting you **program** with language models instead of **prompting** them.

### Key Idea: Separate What from How

Instead of telling the model *how* to solve a task (via prompts), you tell it *what* to do (via signatures), and DSPy figures out *how*.

**Traditional prompting** (imperative):
```python
prompt = "You are an assistant. Answer questions. Question: {q}"
```

**DSPy** (declarative):
```python
class QuestionAnswer(dspy.Signature):
    """Answer questions accurately."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
```

DSPy automatically creates the prompts for you!

---

## What DSPy Provides

### 1. **Signatures**: Task Specifications

Signatures define *what* a task does, not *how*:

```python
import dspy

class Summarize(dspy.Signature):
    """Summarize the given text."""
    document: str = dspy.InputField()
    summary: str = dspy.OutputField(desc="concise summary in 2-3 sentences")
```

This is like a type signature in programming—it specifies inputs and outputs.

### 2. **Modules**: Composable Components

Modules are reusable components that use signatures:

```python
# Create a summarization module
summarizer = dspy.Predict(Summarize)

# Use it
result = summarizer(document="Long text here...")
print(result.summary)
```

Modules can be combined, extended, and optimized.

### 3. **Optimizers**: Automatic Improvement

This is where DSPy shines—you can automatically optimize your programs:

```python
# Define your program
class RAGPipeline(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.answer = dspy.ChainOfThought(QuestionAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)

# Optimize it automatically
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=your_metric)
optimized_rag = optimizer.compile(RAGPipeline(), trainset=your_data)
```

DSPy learns better prompts, better examples, and better module compositions!

---

## Core Concepts

### Signatures

Think of signatures as function declarations for LM tasks:

```python
# Input -> Output specification
class TranslateToFrench(dspy.Signature):
    english_text: str = dspy.InputField()
    french_text: str = dspy.OutputField()
```

### Modules

Pre-built and custom components:

- **`dspy.Predict`**: Basic prediction
- **`dspy.ChainOfThought`**: Step-by-step reasoning
- **`dspy.ReAct`**: Agent-style reasoning with tools
- **Custom**: Build your own!

### Teleprompters (Optimizers)

Automatically improve your program:

- **`BootstrapFewShot`**: Generate few-shot examples
- **`MIPRO`**: Optimize instructions and demonstrations
- **`KNNFewShot`**: Use similarity-based examples

---

## A Simple Example

Let's compare traditional prompting with DSPy:

### Traditional Approach

```python
import openai

def answer_question(question):
    prompt = f"""
    You are a helpful assistant. Answer this question accurately:

    Question: {question}

    Provide a clear, concise answer.
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# Use it
answer = answer_question("What is machine learning?")
print(answer)
```

### DSPy Approach

```python
import dspy

# Configure the language model
lm = dspy.LM(model="openai/gpt-4")
dspy.configure(lm=lm)

# Define the task
class QuestionAnswer(dspy.Signature):
    """Answer questions accurately."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Create the module
qa = dspy.Predict(QuestionAnswer)

# Use it
answer = qa(question="What is machine learning?")
print(answer.answer)
```

**Benefits of the DSPy version**:
- ✅ More modular and reusable
- ✅ Can be composed with other modules
- ✅ Can be automatically optimized
- ✅ Prompts are generated automatically
- ✅ Easier to maintain and test

---

## Why DSPy Matters

### 1. **Systematic Development**

DSPy brings software engineering practices to LM applications:
- Modularity and composition
- Abstraction and reusability
- Systematic testing and optimization

### 2. **Automatic Optimization**

Instead of manually tweaking prompts:
- DSPy learns from your data
- Generates optimal prompts
- Improves with more examples

### 3. **Scalability**

Build complex pipelines that:
- Chain multiple steps
- Handle errors gracefully
- Scale to production

### 4. **Research-Backed**

DSPy is developed by Stanford NLP and backed by research:
- Published at NeurIPS, NAACL, and other top venues
- Proven effectiveness across tasks
- Active research community

---

## Real-World Use Cases

DSPy excels at:

### Question Answering Systems
```python
# RAG-based QA
retriever = dspy.Retrieve(k=3)
qa = dspy.ChainOfThought("context, question -> answer")
```

### Multi-Step Reasoning
```python
# Complex analysis pipelines
class AnalysisPipeline(dspy.Module):
    def __init__(self):
        self.extract = dspy.Predict("text -> entities")
        self.classify = dspy.ChainOfThought("entities -> category")
        self.summarize = dspy.Predict("entities, category -> summary")
```

### Agents and Tools
```python
# ReAct-style agents
agent = dspy.ReAct("question -> answer", tools=[search, calculator])
```

---

## DSPy vs. Other Frameworks

### vs. LangChain

**LangChain**: Focuses on orchestration and integrations
**DSPy**: Focuses on optimization and systematic improvement

DSPy complements LangChain—you can use both together!

### vs. Guidance/LMQL

**Guidance/LMQL**: Template-based prompt control
**DSPy**: Automatic prompt generation and optimization

DSPy abstracts away the prompt engineering entirely.

### vs. Direct API Calls

**Direct APIs**: Maximum control, maximum effort
**DSPy**: Abstraction with automatic optimization

DSPy is higher-level but more powerful for complex tasks.

---

## When to Use DSPy

**DSPy is ideal when you**:
- ✅ Build complex LM pipelines with multiple steps
- ✅ Want to systematically improve performance
- ✅ Need modularity and reusability
- ✅ Have data for optimization
- ✅ Value maintainability over quick hacks

**Consider alternatives when you**:
- ❌ Need a simple one-off query
- ❌ Have zero data for optimization
- ❌ Need very specific prompt control
- ❌ Require guaranteed output formats (use Guidance/LMQL)

---

## The DSPy Philosophy

### Programming > Prompting

```
Traditional:  Human writes prompt → LM executes → Human tweaks prompt → Repeat
DSPy:         Human defines task → DSPy optimizes → LM executes → System improves
```

### Declarative > Imperative

```
Imperative:   "Here's how to answer: First read the context, then..."
Declarative:  "Given context and question, produce an answer"
```

### Optimizable > Static

```
Static:       Fixed prompts that require manual updates
Optimizable:  Programs that improve automatically from data
```

---

## Summary

**DSPy is**:
- A framework for programming foundation models
- Based on signatures (task specs) and modules (components)
- Designed for composition and optimization
- Research-backed and production-ready

**DSPy lets you**:
- Define *what* tasks do, not *how*
- Build modular, composable pipelines
- Automatically optimize from data
- Scale to complex applications

**Key Advantage**:
Instead of manually engineering prompts, you program at a higher level and let DSPy handle the prompt optimization automatically.

---

## Next Steps

Now that you understand what DSPy is, let's dive deeper into the paradigm shift it represents.

**Continue to**: [Programming vs. Prompting](02-programming-vs-prompting.md)

---

## Additional Resources

- **DSPy Paper**: [Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)
- **DSPy Website**: [https://dspy.ai](https://dspy.ai)
- **DSPy GitHub**: [https://github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
- **Blog Post**: [Intro to DSPy](https://dspy.ai/blog/)
