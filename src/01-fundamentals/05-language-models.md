# Language Models

DSPy works with various language model providers. This section covers how to configure different LMs, choose the right model for your task, and follow best practices.

---

## Configuring Language Models

### The Basics

DSPy uses a consistent interface for all language models:

```python
import dspy

# Create an LM instance
lm = dspy.LM(model="provider/model-name", api_key="your-key")

# Set it as the default
dspy.configure(lm=lm)
```

Once configured, all DSPy modules will use this LM automatically.

---

## Supported Providers

### OpenAI

**Models available**:
- `gpt-4o` - Latest flagship model
- `gpt-4o-mini` - Fast, cost-effective
- `gpt-4-turbo` - Previous flagship
- `gpt-3.5-turbo` - Legacy, economical

**Configuration**:
```python
import dspy

lm = dspy.LM(
    model="openai/gpt-4o-mini",
    api_key="sk-your-key-here",
    temperature=0.7,
    max_tokens=500
)
dspy.configure(lm=lm)
```

**Best for**: General-purpose tasks, proven reliability

### Anthropic (Claude)

**Models available**:
- `claude-3-5-sonnet-20241022` - Latest, most capable
- `claude-3-5-haiku-20241022` - Fast, economical
- `claude-3-opus-20240229` - Maximum capability

**Configuration**:
```python
lm = dspy.LM(
    model="anthropic/claude-3-5-sonnet-20241022",
    api_key="your-anthropic-key",
    temperature=0.7,
    max_tokens=1000
)
dspy.configure(lm=lm)
```

**Best for**: Long contexts, detailed analysis, coding

### Local Models (Ollama)

**Models available**:
- `llama3`, `llama3.1`, `llama3.2` - Meta's open models
- `mistral`, `mixtral` - Mistral AI models
- `phi3` - Microsoft's small model
- Many others at [ollama.ai/library](https://ollama.ai/library)

**Configuration**:
```python
# No API key needed!
lm = dspy.LM(
    model="ollama/llama3",
    api_base="http://localhost:11434"
)
dspy.configure(lm=lm)
```

**Best for**: Privacy, no API costs, experimentation

### Other Providers

DSPy also supports:
- **Google (Gemini)**: `gemini/gemini-pro`
- **Cohere**: `cohere/command`
- **Together AI**: `together/model-name`
- **Anyscale**: `anyscale/model-name`

---

## Configuration Options

### Common Parameters

All providers support these parameters:

```python
lm = dspy.LM(
    model="provider/model-name",
    api_key="your-key",

    # Randomness (0.0 = deterministic, 2.0 = very random)
    temperature=0.7,

    # Maximum response length
    max_tokens=500,

    # API endpoint (for local/custom servers)
    api_base="http://localhost:11434",

    # Request timeout in seconds
    timeout=30
)
```

### Temperature Guide

**Temperature** controls output randomness:

| Value | Behavior | Use Case |
|-------|----------|----------|
| 0.0 - 0.3 | Deterministic, focused | Classification, extraction |
| 0.4 - 0.8 | Balanced | General Q&A, summaries |
| 0.9 - 1.5 | Creative, diverse | Creative writing, brainstorming |
| 1.6 - 2.0 | Very random | Experimental, exploration |

**Example**:
```python
# For factual tasks - low temperature
factual_lm = dspy.LM(model="openai/gpt-4o-mini", temperature=0.1)

# For creative tasks - higher temperature
creative_lm = dspy.LM(model="openai/gpt-4o-mini", temperature=1.2)
```

---

## Using Multiple Models

You can use different models for different tasks:

```python
import dspy

# Fast model for simple tasks
fast_lm = dspy.LM(model="openai/gpt-4o-mini")

# Powerful model for complex tasks
smart_lm = dspy.LM(model="openai/gpt-4o")

# Use specific models
class Pipeline(dspy.Module):
    def __init__(self):
        # Simple classification uses fast model
        self.classify = dspy.Predict("text -> category")

    def forward(self, text):
        # Switch to fast model for this step
        with dspy.context(lm=fast_lm):
            category = self.classify(text=text).category

        # Complex reasoning uses smart model
        with dspy.context(lm=smart_lm):
            # ... complex processing
            pass
```

---

## Model Selection Guide

### By Task Type

**Classification / Extraction**:
- OpenAI: `gpt-4o-mini`
- Anthropic: `claude-3-5-haiku-20241022`
- Local: `llama3`

**Question Answering**:
- OpenAI: `gpt-4o-mini` or `gpt-4o`
- Anthropic: `claude-3-5-sonnet-20241022`
- Local: `llama3.1`

**Complex Reasoning**:
- OpenAI: `gpt-4o`
- Anthropic: `claude-3-5-sonnet-20241022`
- Local: `llama3.1:70b` (if you have GPU)

**Long Context**:
- Anthropic: `claude-3-5-sonnet-20241022` (200K context)
- OpenAI: `gpt-4o` (128K context)

**Code Generation**:
- OpenAI: `gpt-4o`
- Anthropic: `claude-3-5-sonnet-20241022`
- Local: `codellama`

### By Budget

**Free / Low Cost**:
- Local models via Ollama (free, requires GPU)
- `gpt-4o-mini` (~$0.15 per 1M tokens)
- `claude-3-5-haiku-20241022` (~$0.25 per 1M tokens)

**Balanced**:
- `gpt-4o` (~$2.50 per 1M tokens)
- `claude-3-5-sonnet-20241022` (~$3 per 1M tokens)

**Maximum Capability** (cost is higher):
- `gpt-4o` (latest flagship)
- `claude-3-opus-20240229` (~$15 per 1M tokens)

---

## Best Practices

### 1. Start Small, Scale Up

```python
# Development: Use small, fast models
dev_lm = dspy.LM(model="openai/gpt-4o-mini")

# Production: Upgrade when needed
prod_lm = dspy.LM(model="openai/gpt-4o")

# Easy to switch!
lm = dev_lm if IS_DEVELOPMENT else prod_lm
dspy.configure(lm=lm)
```

### 2. Use Environment Variables

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Never hardcode API keys!
lm = dspy.LM(
    model="openai/gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

### 3. Set Appropriate Timeouts

```python
# Default timeout might be too short for complex tasks
lm = dspy.LM(
    model="openai/gpt-4o",
    timeout=60  # 60 seconds for complex reasoning
)
```

### 4. Cache Responses (Development)

```python
# DSPy has built-in caching
dspy.configure(lm=lm, cache=True)

# Speeds up development, saves costs
```

### 5. Handle Rate Limits

```python
import time

def call_with_retry(func, max_retries=3):
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "rate limit" in str(e).lower():
                wait_time = (2 ** i) * 1  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded")
```

---

## Common Configurations

### For Learning/Experimentation

```python
# Fast and cheap
lm = dspy.LM(
    model="openai/gpt-4o-mini",
    temperature=0.7,
    max_tokens=300,
    cache=True  # Save $ during development
)
```

### For Production

```python
# Reliable and capable
lm = dspy.LM(
    model="openai/gpt-4o",
    temperature=0.3,  # More deterministic
    max_tokens=1000,
    timeout=60,
    cache=False  # Fresh responses
)
```

### For Privacy-Sensitive Applications

```python
# Local model, no data leaves your machine
lm = dspy.LM(
    model="ollama/llama3",
    api_base="http://localhost:11434",
    temperature=0.7
)
```

---

## Switching Models

### Method 1: Global Configuration

```python
# Set globally
dspy.configure(lm=dspy.LM(model="openai/gpt-4o-mini"))

# All modules use this model
qa = dspy.Predict("question -> answer")
```

### Method 2: Context Manager

```python
# Default model
dspy.configure(lm=dspy.LM(model="openai/gpt-4o-mini"))

qa = dspy.Predict("question -> answer")

# Temporarily use a different model
with dspy.context(lm=dspy.LM(model="openai/gpt-4o")):
    result = qa(question="Complex question")
```

### Method 3: Per-Module

```python
class CustomPipeline(dspy.Module):
    def __init__(self):
        # Each module can have its own LM
        self.fast_step = dspy.Predict("input -> output")
        self.smart_step = dspy.ChainOfThought("input -> output")

    def forward(self, input_text):
        # Use fast model
        with dspy.context(lm=fast_lm):
            temp = self.fast_step(input=input_text).output

        # Use smart model
        with dspy.context(lm=smart_lm):
            result = self.smart_step(input=temp).output

        return result
```

---

## Troubleshooting

### Issue: "Rate limit exceeded"

**Solution**:
1. Reduce request frequency
2. Implement exponential backoff
3. Upgrade your API plan
4. Use a cheaper model for development

### Issue: "Connection timeout"

**Solution**:
```python
# Increase timeout
lm = dspy.LM(model="openai/gpt-4o", timeout=120)
```

### Issue: "Insufficient credits/quota"

**Solution**:
1. Check your billing on the provider's dashboard
2. Add payment method or increase limits
3. Switch to a local model temporarily

### Issue: Local model responses are poor quality

**Solution**:
1. Try a larger model (`llama3.1:70b` instead of `llama3`)
2. Adjust temperature
3. Provide more context in your signatures
4. Consider using a commercial API for better quality

---

## Cost Optimization Tips

### 1. Use Appropriate Models

```python
# Don't use gpt-4o for simple tasks!
# Use gpt-4o-mini instead
```

### 2. Limit Token Usage

```python
lm = dspy.LM(
    model="openai/gpt-4o-mini",
    max_tokens=200  # Shorter responses = lower cost
)
```

### 3. Cache During Development

```python
dspy.configure(lm=lm, cache=True)
# Repeated queries use cached results
```

### 4. Batch Similar Requests

```python
# Process multiple items together when possible
questions = ["Q1", "Q2", "Q3"]

# Instead of 3 separate calls, batch them
for q in questions:
    # DSPy handles this efficiently
    result = qa(question=q)
```

---

## Advanced: Custom LM Integration

You can integrate any LM that follows the DSPy interface:

```python
class CustomLM:
    def __call__(self, prompt, **kwargs):
        # Your custom LM logic here
        # Must return a string or list of strings
        response = your_custom_model(prompt)
        return response

# Use it
custom_lm = CustomLM()
dspy.configure(lm=custom_lm)
```

---

## Summary

**Key Concepts**:
- DSPy supports multiple LM providers (OpenAI, Anthropic, local, etc.)
- Configure once with `dspy.configure(lm=...)`
- Use `dspy.context()` to temporarily switch models
- Choose models based on task complexity and budget
- Start with smaller models, scale up as needed

**Best Practices**:
- Use environment variables for API keys
- Set appropriate timeouts and token limits
- Enable caching during development
- Choose the right model for each task
- Handle rate limits gracefully

---

## Next Steps

Now that you understand how to work with language models in DSPy, let's practice with some exercises!

**Continue to**: [Exercises](06-exercises.md)

---

## Quick Reference

### OpenAI
```python
lm = dspy.LM(model="openai/gpt-4o-mini", api_key=key)
```

### Anthropic
```python
lm = dspy.LM(model="anthropic/claude-3-5-sonnet-20241022", api_key=key)
```

### Ollama (Local)
```python
lm = dspy.LM(model="ollama/llama3", api_base="http://localhost:11434")
```

### Switch Models
```python
with dspy.context(lm=different_lm):
    result = module(input=data)
```

---

## Additional Resources

- **OpenAI Models**: [https://platform.openai.com/docs/models](https://platform.openai.com/docs/models)
- **Anthropic Models**: [https://docs.anthropic.com/claude/docs/models-overview](https://docs.anthropic.com/claude/docs/models-overview)
- **Ollama**: [https://ollama.ai](https://ollama.ai)
- **DSPy LM Docs**: [https://dspy.ai/api/language-models](https://dspy.ai/api/language-models)
