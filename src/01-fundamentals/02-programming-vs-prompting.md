# Programming vs. Prompting

The shift from **prompting** to **programming** language models is the core innovation of DSPy. Understanding this paradigm shift is essential to mastering the framework.

---

## The Traditional Approach: Prompting

### What is Prompting?

Prompting is the practice of crafting text instructions to guide a language model's behavior.

**Example**:
```python
prompt = """
You are an expert chef. Given a list of ingredients, suggest a recipe.
Be creative but practical. Include cooking time and difficulty level.

Ingredients: chicken, garlic, olive oil, lemon, thyme

Recipe:
"""
```

This approach has been the standard since GPT-3 launched in 2020.

### The Prompting Workflow

```
1. Write a prompt
2. Test with the model
3. Observe output
4. Tweak the prompt
5. Repeat steps 2-4 until satisfied
```

This is **manual prompt engineering**—an iterative, hands-on process.

---

## Problems with Prompting at Scale

While prompting works for simple cases, it breaks down as applications grow complex.

### Problem 1: Prompt Fragility

Small changes can dramatically affect results:

```python
# Version 1
prompt_v1 = "Summarize this article."

# Version 2
prompt_v2 = "Summarize this article concisely."

# Version 3
prompt_v3 = "Provide a concise summary of this article."
```

Each version may produce different quality results, and there's no systematic way to know which is best.

### Problem 2: No Composition

Chaining prompts is manual and error-prone:

```python
# Step 1: Extract entities
entities_prompt = f"Extract entities from: {text}"
entities = model(entities_prompt)

# Step 2: Classify entities
classification_prompt = f"Classify these entities: {entities}"
classification = model(classification_prompt)

# Step 3: Generate summary
summary_prompt = f"Summarize: {classification}"
summary = model(summary_prompt)
```

**Issues**:
- No abstraction or reusability
- Hard to test individual steps
- Difficult to optimize the pipeline
- Error handling is manual

### Problem 3: No Systematic Optimization

How do you improve this prompt?

```python
qa_prompt = """
Answer the question using the provided context.

Context: {context}
Question: {question}

Answer:
"""
```

Traditional approach:
- Try different wordings manually
- Add examples by hand
- Test each variation
- Hope for improvement

This doesn't scale to complex applications.

### Problem 4: Maintenance Nightmare

As your application grows:

```python
# You end up with dozens of prompts
SUMMARIZATION_PROMPT = "..."
CLASSIFICATION_PROMPT = "..."
ENTITY_EXTRACTION_PROMPT = "..."
SENTIMENT_ANALYSIS_PROMPT = "..."
QA_PROMPT = "..."
# ... and so on
```

Each prompt:
- Needs individual testing
- Requires manual updates
- May interact with others unpredictably
- Is hard to version and track

---

## The DSPy Approach: Programming

DSPy flips the paradigm: instead of writing prompts, you **program** what you want the LM to do.

### What is Programming with LMs?

Programming means writing **declarative specifications** of tasks, not imperative instructions.

**DSPy Example**:
```python
import dspy

class RecipeSuggestion(dspy.Signature):
    """Suggest a recipe based on ingredients."""

    ingredients: list[str] = dspy.InputField()
    recipe_name: str = dspy.OutputField()
    instructions: str = dspy.OutputField()
    cooking_time: str = dspy.OutputField()
    difficulty: str = dspy.OutputField(desc="easy, medium, or hard")
```

No manual prompt writing—DSPy generates the prompts automatically!

### The Programming Workflow

```
1. Define task signature (what to do)
2. Choose/create module (how to do it)
3. Optionally optimize (improve automatically)
4. Deploy and iterate
```

This is **declarative programming**—you specify outcomes, not implementation details.

---

## Key Differences

### Imperative vs. Declarative

**Prompting (Imperative)**:
```python
# You tell the model HOW to do it
prompt = """
First, read the context carefully.
Then, identify the key information.
Next, formulate an answer.
Finally, provide your response in one sentence.

Context: {context}
Question: {question}
"""
```

**DSPy (Declarative)**:
```python
# You tell the model WHAT to do
class AnswerQuestion(dspy.Signature):
    """Answer questions based on context."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="concise answer")
```

DSPy figures out the HOW!

### Manual vs. Automatic

**Prompting**: Manual optimization
```python
# Try different prompts manually
prompts = [
    "Answer: {question}",
    "Provide a clear answer to: {question}",
    "Question: {question}\nAnswer:",
]

for prompt in prompts:
    # Test and compare manually
    result = test(prompt)
```

**DSPy**: Automatic optimization
```python
# Define your program
program = dspy.ChainOfThought(AnswerQuestion)

# Optimize automatically
from dspy.teleprompt import BootstrapFewShot
optimizer = BootstrapFewShot(metric=accuracy)
optimized_program = optimizer.compile(program, trainset=data)
```

### Static vs. Composable

**Prompting**: Static, monolithic
```python
# One big prompt for the entire task
mega_prompt = """
Step 1: Extract entities from the text
Step 2: Classify each entity
Step 3: Summarize the entities
Step 4: Generate final output

Text: {text}
"""
```

**DSPy**: Modular, composable
```python
# Separate, reusable components
class Pipeline(dspy.Module):
    def __init__(self):
        self.extract = dspy.Predict("text -> entities")
        self.classify = dspy.Predict("entities -> categories")
        self.summarize = dspy.Predict("categories -> summary")

    def forward(self, text):
        entities = self.extract(text=text).entities
        categories = self.classify(entities=entities).categories
        summary = self.summarize(categories=categories).summary
        return summary
```

---

## The Paradigm Shift in Detail

### From Strings to Signatures

**Old way** (strings):
```python
# Prompt is a string you craft
prompt = "Translate '{text}' to French"
```

**New way** (signatures):
```python
# Signature is a type specification
class Translate(dspy.Signature):
    text: str = dspy.InputField()
    french_text: str = dspy.OutputField()
```

### From Templates to Types

**Old way** (templates):
```python
# Fill in template variables
template = "Context: {context}\nQuestion: {question}\nAnswer:"
filled = template.format(context=ctx, question=q)
```

**New way** (typed fields):
```python
# Define typed inputs and outputs
class QA(dspy.Signature):
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
```

### From Heuristics to Optimization

**Old way** (heuristics):
```python
# Add examples manually based on intuition
examples = [
    "Q: What is 2+2? A: 4",
    "Q: What is 3+3? A: 6",
]
prompt_with_examples = f"{examples}\n{prompt}"
```

**New way** (data-driven):
```python
# Learn examples automatically from data
optimizer = BootstrapFewShot(metric=accuracy)
optimized = optimizer.compile(program, trainset=training_data)
```

---

## Benefits of the Programming Paradigm

### 1. Modularity

Break complex tasks into simple components:

```python
# Each component is independent and testable
extract_entities = dspy.Predict("text -> entities")
classify_entities = dspy.Predict("entities -> categories")
generate_summary = dspy.Predict("categories -> summary")

# Combine them
def analyze(text):
    entities = extract_entities(text=text).entities
    categories = classify_entities(entities=entities).categories
    summary = generate_summary(categories=categories).summary
    return summary
```

### 2. Reusability

Create once, use everywhere:

```python
# Define a reusable QA signature
class QuestionAnswer(dspy.Signature):
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Use it in different contexts
basic_qa = dspy.Predict(QuestionAnswer)
reasoning_qa = dspy.ChainOfThought(QuestionAnswer)
verified_qa = dspy.MultiChainOfThought(QuestionAnswer)
```

### 3. Testability

Test components independently:

```python
# Test a single module
def test_entity_extraction():
    extractor = dspy.Predict("text -> entities")
    result = extractor(text="Apple released iPhone in 2007")
    assert "Apple" in result.entities
    assert "iPhone" in result.entities
```

### 4. Automatic Optimization

Improve systematically:

```python
# Define your metric
def accuracy_metric(example, prediction):
    return prediction.answer == example.answer

# Optimize automatically
optimizer = BootstrapFewShot(metric=accuracy_metric)
optimized_program = optimizer.compile(
    MyProgram(),
    trainset=training_examples
)
```

### 5. Maintainability

Changes are localized and manageable:

```python
# Change one signature
class ImprovedQA(dspy.Signature):
    """Better QA with sources."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    sources: list[str] = dspy.OutputField()  # Added field

# All modules using this signature automatically adapt
```

---

## Concrete Example: Building a QA System

Let's build the same QA system both ways to see the difference.

### Traditional Prompting Approach

```python
import openai

def answer_question(context, question):
    # Manually crafted prompt
    prompt = f"""
    You are a helpful assistant. Answer the question based only on the provided context.

    Context: {context}

    Question: {question}

    Provide a clear, accurate answer based on the context above.

    Answer:
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# Use it
context = "Paris is the capital of France. It has a population of 2.1 million."
question = "What is the capital of France?"
answer = answer_question(context, question)
```

**Issues**:
- Prompt is hardcoded
- No easy way to add reasoning
- No systematic optimization
- Hard to compose with other components

### DSPy Programming Approach

```python
import dspy

# Configure LM
lm = dspy.LM(model="openai/gpt-4")
dspy.configure(lm=lm)

# Define the task
class QuestionAnswer(dspy.Signature):
    """Answer questions based on provided context."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Create module (can easily upgrade to ChainOfThought!)
qa = dspy.Predict(QuestionAnswer)

# Use it
context = "Paris is the capital of France. It has a population of 2.1 million."
question = "What is the capital of France?"
answer = qa(context=context, question=question).answer
```

**Benefits**:
- Signature is declarative and reusable
- Easy to upgrade (change `Predict` to `ChainOfThought`)
- Can be optimized automatically
- Composes naturally with other modules

### Upgrading to Reasoning (DSPy Only!)

With traditional prompting, adding reasoning means rewriting the prompt. With DSPy:

```python
# Just change one line!
qa = dspy.ChainOfThought(QuestionAnswer)

# Now it reasons step-by-step automatically
answer = qa(context=context, question=question).answer
```

That's it! No prompt rewriting needed.

---

## The Learning Curve

### Traditional Prompting

```
Learn: Basic prompt structure → Practice trial and error → Build intuition
Time: Days to weeks
Scaling: Becomes harder with complexity
```

### DSPy Programming

```
Learn: Signatures → Modules → Optimization → Composition
Time: Days to weeks (similar initial investment)
Scaling: Becomes easier with complexity
```

**Key insight**: DSPy has a similar initial learning curve, but pays dividends as your application grows.

---

## When to Use Which Approach?

### Use Traditional Prompting When:

- ✅ One-off task or prototype
- ✅ Very simple, single-step operation
- ✅ You need specific prompt control
- ✅ No optimization needed

### Use DSPy When:

- ✅ Building a complex system
- ✅ Multiple steps or components
- ✅ Want systematic optimization
- ✅ Need maintainability and testability
- ✅ Have training data available

---

## Analogy: Assembly vs. High-Level Languages

The prompting → programming shift is like assembly → high-level languages:

### Assembly (Manual Prompting)

```assembly
; Direct, detailed control
MOV AX, 5
ADD AX, 3
MOV result, AX
```

- Maximum control
- Tedious for complex tasks
- Hard to maintain

### High-Level Language (DSPy)

```python
# Abstract, declarative
result = 5 + 3
```

- Easier to write and understand
- Better for complex systems
- Compiler handles optimization

Similarly, DSPy abstracts away prompt engineering!

---

## Summary

### The Paradigm Shift

| Aspect | Prompting | Programming (DSPy) |
|--------|-----------|-------------------|
| **Approach** | Imperative ("how") | Declarative ("what") |
| **Optimization** | Manual trial & error | Automatic from data |
| **Composition** | Difficult | Natural |
| **Maintainability** | Poor for complex | Good |
| **Scalability** | Struggles | Excels |
| **Learning curve** | Moderate | Moderate |
| **Best for** | Simple, one-off tasks | Complex, evolving systems |

### Key Takeaways

1. **Prompting** = Writing instructions for the model
2. **Programming** = Defining specifications for tasks
3. **DSPy generates prompts** automatically from signatures
4. **Composition and optimization** come naturally with programming
5. **Invest in learning DSPy** for long-term productivity

---

## Next Steps

Now that you understand the paradigm shift, let's get DSPy installed and configured.

**Continue to**: [Installation and Setup](03-installation-setup.md)

---

## Additional Resources

- **Blog**: [From Prompting to Programming](https://dspy.ai/blog/programming-vs-prompting)
- **Paper**: Section 2 of the [DSPy paper](https://arxiv.org/abs/2310.03714) discusses this paradigm shift
- **Tutorial**: [DSPy Tutorial on Programming Paradigm](https://dspy.ai/tutorials/programming)
