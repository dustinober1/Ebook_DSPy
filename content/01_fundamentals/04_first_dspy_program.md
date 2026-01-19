# Your First DSPy Program

Let's write your first DSPy program! This hands-on section will walk you through creating a simple question-answering application step by step.

---

## What We'll Build

We'll create a program that:
- Takes a question as input
- Uses a language model to generate an answer
- Returns the answer

This is the "Hello World" of DSPy!

---

## The Complete Program

Here's the full program. Don't worry if you don't understand everything yet—we'll break it down step by step.

**File**: `hello_dspy.py`

```python
"""
Your First DSPy Program
A simple question-answering application
"""

import os
from dotenv import load_dotenv
import dspy

# Load environment variables
load_dotenv()

def main():
    # Step 1: Configure the language model
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    dspy.configure(lm=lm)

    # Step 2: Define the task signature
    class QuestionAnswer(dspy.Signature):
        """Answer questions with factual information."""
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # Step 3: Create a predictor
    qa = dspy.Predict(QuestionAnswer)

    # Step 4: Use it!
    question = "What is the capital of France?"
    result = qa(question=question)

    # Step 5: Display the result
    print(f"Question: {question}")
    print(f"Answer: {result.answer}")

if __name__ == "__main__":
    main()
```

---

## Step-by-Step Breakdown

### Step 1: Configure the Language Model

```python
lm = dspy.LM(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)
dspy.configure(lm=lm)
```

**What's happening**:
1. `dspy.LM()` creates a language model instance
2. We specify which model to use (`gpt-4o-mini`)
3. We provide the API key from environment variables
4. `dspy.configure()` sets this as the default LM for DSPy

**Think of this as**: Setting up your "engine" that powers all DSPy operations.

### Step 2: Define the Task Signature

```python
class QuestionAnswer(dspy.Signature):
    """Answer questions with factual information."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
```

**What's happening**:
1. We create a class that inherits from `dspy.Signature`
2. The docstring describes what the task does
3. `question` is marked as an input field (what we provide)
4. `answer` is marked as an output field (what we want back)

**Think of this as**: A contract that says "Give me a question, I'll give you an answer."

### Step 3: Create a Predictor

```python
qa = dspy.Predict(QuestionAnswer)
```

**What's happening**:
1. `dspy.Predict` is a module that makes predictions
2. We pass our `QuestionAnswer` signature to it
3. This creates a predictor that can answer questions

**Think of this as**: Creating a function that implements our contract.

### Step 4: Use It!

```python
question = "What is the capital of France?"
result = qa(question=question)
```

**What's happening**:
1. We call our predictor like a function
2. We pass the question as a keyword argument
3. DSPy automatically generates a prompt, calls the LM, and returns the result

**Think of this as**: Just using the function we created!

### Step 5: Display the Result

```python
print(f"Answer: {result.answer}")
```

**What's happening**:
1. `result` is a prediction object
2. We access the `answer` field (from our signature)
3. DSPy has extracted this from the LM's response

**Think of this as**: Getting the output from our function.

---

## Running Your Program

### 1. Save the File

Save the code above as `hello_dspy.py`.

### 2. Ensure Your Environment is Ready

```bash
# Activate virtual environment
source venv/bin/activate

# Check .env file exists with API key
cat .env
```

### 3. Run It!

```bash
python hello_dspy.py
```

### 4. Expected Output

```
Question: What is the capital of France?
Answer: Paris
```

---

## What's Happening Behind the Scenes?

When you run this program, DSPy:

1. **Generates a prompt** based on your signature:
   ```
   Answer questions with factual information.

   ---

   Question: What is the capital of France?
   Answer:
   ```

2. **Calls the language model** with this prompt

3. **Parses the response** and extracts the answer

4. **Returns a structured result** with the answer field populated

You didn't write the prompt—DSPy did it for you!

---

## Experimenting

Try modifying the program to explore DSPy:

### Experiment 1: Different Questions

```python
questions = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "What is 25 multiplied by 4?",
]

for question in questions:
    result = qa(question=question)
    print(f"Q: {question}")
    print(f"A: {result.answer}\n")
```

### Experiment 2: Add Field Descriptions

```python
class QuestionAnswer(dspy.Signature):
    """Answer questions with factual information."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="concise answer in one sentence")
```

The description helps guide the model's response format!

### Experiment 3: Multiple Output Fields

```python
class DetailedQA(dspy.Signature):
    """Answer questions with details."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    confidence: str = dspy.OutputField(desc="high, medium, or low")
    explanation: str = dspy.OutputField(desc="brief reasoning")
```

### Experiment 4: Use Chain of Thought

```python
# Change one line!
qa = dspy.ChainOfThought(QuestionAnswer)

# Now it shows reasoning
result = qa(question="What is the capital of France?")
print(f"Reasoning: {result.rationale}")
print(f"Answer: {result.answer}")
```

---

## Common Issues and Fixes

### Issue: "API key not found"

**Fix**:
```python
# Debug: Print to see if key is loaded
import os
print(f"API Key present: {bool(os.getenv('OPENAI_API_KEY'))}")
```

Ensure `.env` file is in the same directory and `load_dotenv()` is called.

### Issue: "Module 'dspy' has no attribute 'LM'"

**Fix**: You might have an old DSPy version.
```bash
pip install --upgrade dspy-ai
```

### Issue: Response is empty or unexpected

**Fix**: Check your signature description and field names. Make them clear and descriptive.

---

## Understanding the Code Structure

### Typical DSPy Program Structure

```python
# 1. Imports
import dspy
from dotenv import load_dotenv

# 2. Configuration
load_dotenv()
lm = dspy.LM(...)
dspy.configure(lm=lm)

# 3. Signature Definition
class MyTask(dspy.Signature):
    input_field: str = dspy.InputField()
    output_field: str = dspy.OutputField()

# 4. Module Creation
module = dspy.Predict(MyTask)

# 5. Usage
result = module(input_field="...")
print(result.output_field)
```

This pattern will be consistent across all DSPy programs!

---

## Comparing to Traditional Approach

Let's see how this compares to traditional prompting:

### Traditional Prompting

```python
import openai

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": "What is the capital of France?"
    }]
)

answer = response.choices[0].message.content
print(answer)
```

**Problems**:
- No structure or reusability
- Hard to modify or extend
- Can't compose with other components
- No automatic optimization

### DSPy Approach

```python
import dspy

# Define once, use everywhere
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

qa = dspy.Predict(QA)
result = qa(question="What is the capital of France?")
```

**Benefits**:
- Structured and reusable
- Easy to modify (just change signature)
- Composable with other modules
- Can be automatically optimized

---

## Next Steps: Building on This

You can extend this basic program in many ways:

### Add Context

```python
class ContextualQA(dspy.Signature):
    """Answer questions based on context."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

qa = dspy.Predict(ContextualQA)
result = qa(
    context="Paris is the capital of France with 2.1M population.",
    question="What is the capital of France?"
)
```

### Add Multiple Steps

```python
# Step 1: Classify the question
classify = dspy.Predict("question -> category")
category = classify(question="What is the capital of France?").category

# Step 2: Answer based on category
qa = dspy.Predict("question, category -> answer")
result = qa(question=question, category=category)
```

### Create a Pipeline

```python
class QuestionPipeline(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("question -> category")
        self.answer = dspy.Predict("question, category -> answer")

    def forward(self, question):
        category = self.classify(question=question).category
        answer = self.answer(question=question, category=category).answer
        return answer
```

---

## Practice Exercise

Before moving on, try this exercise:

**Task**: Modify the program to create a simple translator.

**Requirements**:
1. Take English text as input
2. Translate to a target language
3. Return both translation and confidence level

**Starter code**:
```python
class Translate(dspy.Signature):
    # TODO: Define your signature
    pass

translator = dspy.Predict(Translate)
# TODO: Use the translator
```

**Solution**: Available in [Chapter 1 Exercises](06-exercises.md)

---

## Summary

**You've learned**:
1. ✅ How to configure DSPy with a language model
2. ✅ How to define a signature (task specification)
3. ✅ How to create a predictor with `dspy.Predict`
4. ✅ How to use the predictor to generate results
5. ✅ The basic structure of DSPy programs

**Key concepts**:
- **Signatures** define inputs and outputs
- **Modules** (like `Predict`) implement the behavior
- **Configuration** sets up the language model
- **Results** are structured objects with named fields

---

## Next Steps

Now that you can write basic DSPy programs, let's explore how to configure different language models.

**Continue to**: [Language Models](05-language-models.md)

---

## Code Example

The complete working example is available:
- **Location**: `examples/chapter01/01_hello_dspy.py`
- **Run it**: `python examples/chapter01/01_hello_dspy.py`

---

## Additional Resources

- **DSPy Quickstart**: [https://dspy.ai/learn/quick-start](https://dspy.ai/learn/quick-start)
- **Signatures Guide**: [https://dspy.ai/learn/programming/signatures](https://dspy.ai/learn/programming/signatures)
- **Module Reference**: [https://dspy.ai/api/modules](https://dspy.ai/api/modules)
