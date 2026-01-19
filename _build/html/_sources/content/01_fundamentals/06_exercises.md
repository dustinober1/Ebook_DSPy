# Chapter 1 Exercises

Practice what you've learned in this chapter with these hands-on exercises.

---

## Exercise Overview

| Exercise   | Difficulty      | Topics Covered               | Estimated Time |
| ---------- | --------------- | ---------------------------- | -------------- |
| Exercise 1 | ⭐ Beginner      | Installation verification    | 10-15 min      |
| Exercise 2 | ⭐ Beginner      | Basic signatures             | 15-20 min      |
| Exercise 3 | ⭐⭐ Intermediate | Language model configuration | 20-25 min      |
| Exercise 4 | ⭐⭐ Intermediate | Building a Q&A system        | 30-40 min      |
| Exercise 5 | ⭐⭐⭐ Advanced    | Multi-step pipeline          | 45-60 min      |

---

## Exercise 1: Verify Your DSPy Installation

**Difficulty**: ⭐ Beginner

### Objective

Confirm that DSPy is properly installed and configured in your environment.

### Requirements

Create a script that:

1. Imports DSPy successfully
2. Prints the DSPy version
3. Checks for an API key in environment variables
4. Creates and configures a language model
5. Runs a simple test prediction

### Success Criteria

- [ ] Script runs without errors
- [ ] DSPy version is displayed
- [ ] API key is detected
- [ ] Test prediction produces a valid response

### Starter Code

```python
"""
Exercise 1: Verify DSPy Installation
"""

import os
from dotenv import load_dotenv
import dspy

def main():
    # TODO: Load environment variables

    # TODO: Print DSPy version

    # TODO: Check for API key

    # TODO: Configure language model

    # TODO: Run a test prediction

    pass

if __name__ == "__main__":
    main()
```

### Hints

<details>
<summary>💡 Hint 1</summary>

Use `load_dotenv()` to load environment variables and `dspy.__version__` to get the version.

</details>

<details>
<summary>💡 Hint 2</summary>

Check for the API key with `os.getenv("OPENAI_API_KEY")` and verify it's not None.

</details>

<details>
<summary>💡 Hint 3</summary>

Create a simple signature like `question -> answer` and use `dspy.Predict` to test it.

</details>

### Expected Output

```
DSPy Installation Check
=======================
✓ DSPy version: 2.5.x
✓ API key found
✓ Language model configured
✓ Test prediction successful

Test question: What is 2+2?
Test answer: 4

All checks passed!
```

### Solution

See [exercises/chapter01/solutions/exercise01.py](../../exercises/chapter01/solutions/exercise01.py)

---

## Exercise 2: Create Custom Signatures

**Difficulty**: ⭐ Beginner

### Objective

Practice creating DSPy signatures for different tasks.

### Requirements

Create signatures for the following tasks:

1. **Translation**: Translate English text to Spanish
2. **Sentiment Analysis**: Classify text as positive, negative, or neutral
3. **Summarization**: Create a brief summary of text
4. **Entity Extraction**: Extract named entities from text

Each signature should:
- Have appropriate field names
- Include helpful descriptions
- Use correct input/output field types

### Starter Code

```python
"""
Exercise 2: Create Custom Signatures
"""

import dspy

# TODO: Create Translation signature
class Translate(dspy.Signature):
    pass

# TODO: Create Sentiment Analysis signature
class AnalyzeSentiment(dspy.Signature):
    pass

# TODO: Create Summarization signature
class Summarize(dspy.Signature):
    pass

# TODO: Create Entity Extraction signature
class ExtractEntities(dspy.Signature):
    pass

def test_signatures():
    """Test each signature"""
    # TODO: Test each signature with dspy.Predict
    pass

if __name__ == "__main__":
    test_signatures()
```

### Hints

<details>
<summary>💡 Hint 1</summary>

For translation, you'll need input fields for the text and target language, and an output field for the translated text.

</details>

<details>
<summary>💡 Hint 2</summary>

Add descriptions to output fields using `desc=` parameter to guide the model's responses.

</details>

<details>
<summary>💡 Hint 3</summary>

The docstring of each signature should clearly describe what the task does.

</details>

### Success Criteria

- [ ] All four signatures are properly defined
- [ ] Each signature has a clear docstring
- [ ] Field names are descriptive
- [ ] Output fields have helpful descriptions
- [ ] All signatures work with `dspy.Predict`

### Solution

See [exercises/chapter01/solutions/exercise02.py](../../exercises/chapter01/solutions/exercise02.py)

---

## Exercise 3: Configure Multiple Language Models

**Difficulty**: ⭐⭐ Intermediate

### Objective

Learn to configure and switch between different language models.

### Requirements

Create a program that:

1. Configures three different LMs:
   - A fast, cheap model (e.g., gpt-4o-mini)
   - A powerful model (e.g., gpt-4o)
   - A local model (if Ollama installed) OR another provider
2. Defines a simple Q&A signature
3. Tests the same question with all three models
4. Compares the responses
5. Measures and reports response time for each

> **Note**: If you only have access to one LM provider (e.g., OpenAI), you can simulate "different" models by:
> - Using different model sizes (e.g., `gpt-4o` vs `gpt-4o-mini`)
> - Varying the temperature (e.g., `temperature=0.0` vs `temperature=1.0`)
> - **Tip**: This simulation approach helps you understand how model choice affects output variability and quality, even without multiple providers.


### Starter Code

```python
"""
Exercise 3: Configure Multiple Language Models
"""

import os
import time
from dotenv import load_dotenv
import dspy

load_dotenv()

def test_model(lm, model_name, question):
    """Test a model and return response and time taken."""
    # TODO: Configure the model
    # TODO: Create predictor
    # TODO: Time the prediction
    # TODO: Return results
    pass

def main():
    # TODO: Define your LMs
    fast_lm = None
    smart_lm = None
    alt_lm = None

    # TODO: Define test question
    question = "Explain quantum computing in simple terms"

    # TODO: Test each model
    # TODO: Compare results

    pass

if __name__ == "__main__":
    main()
```

### Hints

<details>
<summary>💡 Hint 1</summary>

Use `time.time()` before and after the prediction to measure response time.

</details>

<details>
<summary>💡 Hint 2</summary>

Use `dspy.context(lm=...)` to temporarily switch models without changing the global configuration.

</details>

<details>
<summary>💡 Hint 3</summary>

Create a comparison table showing model name, response length, and time taken.

</details>

### Success Criteria

- [ ] Three different models are configured
- [ ] Same question is tested with all models
- [ ] Response times are measured and displayed
- [ ] Results are clearly presented for comparison
- [ ] Code handles errors gracefully

### Expected Output

```
Testing Multiple Language Models
=================================

Question: Explain quantum computing in simple terms

Model: gpt-4o-mini
Time: 1.2s
Response: Quantum computing uses quantum mechanics to process information...

Model: gpt-4o
Time: 2.1s
Response: Quantum computing is a revolutionary approach that leverages...

Model: ollama/llama3
Time: 3.5s
Response: Quantum computing is a type of computing that uses quantum bits...

Summary:
--------
Fastest: gpt-4o-mini (1.2s)
Most detailed: gpt-4o
```

### Solution

See [exercises/chapter01/solutions/exercise03.py](../../exercises/chapter01/solutions/exercise03.py)

---

## Exercise 4: Build a Simple Q&A System

**Difficulty**: ⭐⭐ Intermediate

### Objective

Build a complete question-answering system with context.

### Requirements

Create a Q&A system that:

1. Takes a context (paragraph of text) and a question
2. Uses DSPy to answer the question based only on the context
3. Provides a confidence level (high/medium/low)
4. Cites which part of the context was used
5. Handles cases where the answer isn't in the context

### Starter Code

```python
"""
Exercise 4: Build a Q&A System
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# TODO: Define your signature
class ContextualQA(dspy.Signature):
    pass

def create_qa_system():
    """Create and return a Q&A module."""
    # TODO: Configure LM
    # TODO: Create predictor
    pass

def test_qa_system():
    """Test the Q&A system with sample contexts."""
    # TODO: Define test contexts and questions
    # TODO: Test the system
    # TODO: Display results
    pass

if __name__ == "__main__":
    test_qa_system()
```

### Test Data

Use these test cases:

```python
test_cases = [
    {
        "context": "Paris is the capital of France. It has a population of about 2.1 million people. The city is known for the Eiffel Tower and the Louvre Museum.",
        "question": "What is the capital of France?"
    },
    {
        "context": "Python was created by Guido van Rossum and released in 1991. It emphasizes code readability and simplicity.",
        "question": "Who created Python?"
    },
    {
        "context": "The Great Wall of China is over 13,000 miles long. It was built over many centuries to protect against invasions.",
        "question": "What is the main programming language used in AI?"  # Not in context!
    }
]
```

### Hints

<details>
<summary>💡 Hint 1</summary>

Your signature should have `context` and `question` as inputs, and `answer` and `confidence` as outputs.

</details>

<details>
<summary>💡 Hint 2</summary>

Use field descriptions to guide the model. For example, confidence could be described as "high if certain, medium if somewhat sure, low if answer not in context".

</details>

<details>
<summary>💡 Hint 3</summary>

Consider using `dspy.ChainOfThought` instead of `dspy.Predict` for better reasoning about the context.

</details>

### Success Criteria

- [ ] System correctly answers questions from context
- [ ] Confidence levels are appropriate
- [ ] System indicates low confidence when answer not in context
- [ ] Code is well-structured and documented
- [ ] All test cases are handled properly

### Solution

See [exercises/chapter01/solutions/exercise04.py](../../exercises/chapter01/solutions/exercise04.py)

---

## Exercise 5: Multi-Step Classification Pipeline

**Difficulty**: ⭐⭐⭐ Advanced

### Objective

Build a multi-step pipeline that processes text through multiple stages.

### Requirements

Create a pipeline that:

1. **Step 1**: Extracts the main topic from input text
2. **Step 2**: Classifies the sentiment (positive/negative/neutral)
3. **Step 3**: Determines the intended audience (general/technical/academic)
4. **Step 4**: Generates a summary tailored to the audience
5. Returns all results in a structured format

### Starter Code

```python
"""
Exercise 5: Multi-Step Classification Pipeline
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# TODO: Define signatures for each step

class TextAnalysisPipeline(dspy.Module):
    """Multi-step text analysis pipeline."""

    def __init__(self):
        # TODO: Initialize modules for each step
        pass

    def forward(self, text):
        """Process text through the pipeline."""
        # TODO: Implement pipeline logic
        # TODO: Return structured results
        pass

def test_pipeline():
    """Test the pipeline with different texts."""
    test_texts = [
        "Machine learning models require large datasets and computational power. "
        "Recent advances in transformer architectures have revolutionized NLP tasks.",

        "I absolutely love this new restaurant! The food was amazing and the "
        "service was excellent. Can't wait to go back!",

        "The economic indicators suggest a potential downturn in the housing market. "
        "Analysts recommend caution in real estate investments."
    ]

    # TODO: Process each text
    # TODO: Display results
    pass

if __name__ == "__main__":
    test_pipeline()
```

### Hints

<details>
<summary>💡 Hint 1</summary>

Create separate signatures for each step of the pipeline. Each signature should have clear inputs and outputs.

</details>

<details>
<summary>💡 Hint 2</summary>

Use the output from one step as the input to the next step. Chain them together in the `forward` method.

</details>

<details>
<summary>💡 Hint 3</summary>

Return a dictionary or custom object with all the results (topic, sentiment, audience, summary) for easy access.

</details>

<details>
<summary>💡 Hint 4</summary>

Consider using different modules for different steps - maybe `ChainOfThought` for complex analysis and `Predict` for simple classification.

</details>

### Success Criteria

- [ ] Pipeline has four distinct stages
- [ ] Each stage uses a well-defined signature
- [ ] Results from one stage flow into the next
- [ ] Final output is structured and complete
- [ ] Pipeline works with different types of text
- [ ] Code is modular and follows DSPy best practices

### Expected Output

```
Processing: "Machine learning models require large datasets..."

Results:
========
Topic: Machine Learning and NLP
Sentiment: Neutral
Audience: Technical
Summary (for Technical audience):
  ML models need significant data and compute. Transformer
  architectures have significantly advanced NLP capabilities.

---

Processing: "I absolutely love this new restaurant..."

Results:
========
Topic: Restaurant Review
Sentiment: Positive
Audience: General
Summary (for General audience):
  A very positive review of a restaurant praising both
  food quality and service.
```

### Solution

See [exercises/chapter01/solutions/exercise05.py](../../exercises/chapter01/solutions/exercise05.py)

---

## Challenge Exercise (Optional)

**Difficulty**: ⭐⭐⭐⭐ Expert

### Build a Smart Document Analyzer

Create a complete application that:
1. Accepts a document (text) as input
2. Automatically determines the document type (article, email, report, etc.)
3. Extracts key information based on type
4. Generates appropriate outputs (summary, action items, key points)
5. Uses different models for different subtasks (optimization!)
6. Provides confidence scores for all outputs

This is an open-ended challenge with no starter code. Design your own architecture!

---

## Getting Help

- **Stuck?** Review the relevant chapter sections
- **Still stuck?** Check the hints progressively
- **Need code?** Look at the solutions, but try first!
- **Have questions?** See [Chapter 9: Appendices](../09-appendices/02-troubleshooting.md)

---

## Solutions

Complete solutions with detailed explanations are available in:
- **Code solutions**: [exercises/chapter01/solutions/](../../exercises/chapter01/solutions/)
- **Explanations**: [exercises/chapter01/solutions/solutions.md](../../exercises/chapter01/solutions/solutions.md)

**Important**: Try to solve exercises yourself before looking at solutions!

---

## Next Steps

Congratulations on completing Chapter 1! You now have a solid foundation in DSPy fundamentals.

**Continue to**: [Chapter 2: Signatures](../02-signatures/00-chapter-intro.md) to learn advanced signature techniques.

---

## Progress Tracker

Track your completion:

- [ ] Exercise 1: Installation verification
- [ ] Exercise 2: Custom signatures
- [ ] Exercise 3: Multiple models
- [ ] Exercise 4: Q&A system
- [ ] Exercise 5: Multi-step pipeline
- [ ] Challenge: Document analyzer (optional)

Once you've completed all exercises, you're ready for Chapter 2!
