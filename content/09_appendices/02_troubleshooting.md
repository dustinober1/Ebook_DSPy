# Troubleshooting Guide

This guide covers common issues encountered when using DSPy, along with diagnostic steps and solutions.

## Table of Contents

- [Installation and Setup](#installation-and-setup)
- [API Keys and Authentication](#api-keys-and-authentication)
- [Language Model Issues](#language-model-issues)
- [Signature and Module Problems](#signature-and-module-problems)
- [Evaluation Issues](#evaluation-issues)
- [Optimization and Compilation](#optimization-and-compilation)
- [Performance and Caching](#performance-and-caching)
- [Debugging Tips](#debugging-tips)

## Installation and Setup

### Issue: `ModuleNotFoundError: No module named 'dspy'`

**Symptoms:** Python raises `ModuleNotFoundError` when importing DSPy.

**Solutions:**
1. Install DSPy: `pip install dspy-ai`
2. Verify installation: `python -c "import dspy; print(dspy.__version__)"`
3. Check Python version: DSPy requires Python 3.8+
4. If in a virtual environment, ensure it's activated
5. Try reinstalling: `pip install --upgrade --force-reinstall dspy-ai`

### Issue: Incompatible DSPy version

**Symptoms:** API or feature unavailable, examples don't work as shown.

**Solutions:**
1. Check current version: `pip show dspy-ai`
2. Check latest version: `pip index versions dspy-ai`
3. Upgrade to latest: `pip install --upgrade dspy-ai`
4. If downgrading needed: `pip install dspy-ai==2.5.0`

### Issue: Missing dependencies

**Symptoms:** Import errors for openai, anthropic, or other packages.

**Solutions:**
1. Install complete requirements: `pip install -r requirements.txt`
2. Install specific provider: `pip install openai anthropic google-cloud-aiplatform`
3. For local models: `pip install ollama` or `pip install vllm`

## API Keys and Authentication

### Issue: `AuthenticationError` or `Unauthorized` errors

**Symptoms:** Errors like "Invalid API key" or "Unauthorized" when making LM calls.

**Solutions:**

1. **OpenAI:**
   - Get key from https://platform.openai.com/api-keys
   - Never commit keys to git - use environment variables
   - Ensure key has credits available
   - Check key permissions/access level

2. **Anthropic (Claude):**
   - Get key from https://console.anthropic.com
   - Verify key format starts with `sk-ant-`
   - Check API usage in console
   - Ensure account has active billing

3. **Configuration:**
   ```python
   import os
   from dotenv import load_dotenv

   load_dotenv()
   api_key = os.getenv('OPENAI_API_KEY')
   dspy.configure(api_key=api_key, model='gpt-4')
   ```

### Issue: Rate limit errors

**Symptoms:** `RateLimitError` or 429 status codes.

**Solutions:**
1. Implement retry logic:
   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential

   @retry(stop=stop_after_attempt(3), wait=wait_exponential())
   def call_dspy(prompt):
       return dspy.Predict(signature)(input=prompt)
   ```
2. Add delays between requests: `time.sleep(1)`
3. Use caching to avoid duplicate requests
4. Reduce batch size or request frequency
5. Check pricing tier and request limits

## Language Model Issues

### Issue: Timeout errors when calling LM

**Symptoms:** `TimeoutError` or `ConnectionError` when making predictions.

**Solutions:**
1. Increase timeout:
   ```python
   lm = dspy.OpenAI(
       api_key="...",
       model="gpt-4",
       request_timeout=60
   )
   ```
2. Check internet connection
3. Try with a different model
4. Verify API service status
5. Check firewall/proxy settings

### Issue: Empty or `None` responses from LM

**Symptoms:** Predictions return empty strings or None values.

**Solutions:**
1. Check max_tokens setting:
   ```python
   dspy.configure(max_tokens=500)  # Increase if too small
   ```
2. Verify signature output fields are defined
3. Check model supports the requested format
4. Add explicit instructions in signature descriptions
5. Try with simpler input

### Issue: Inconsistent or low-quality outputs

**Symptoms:** Model responses are random, off-topic, or low quality.

**Solutions:**
1. **Improve signature clarity:**
   ```python
   class BetterSignature(dspy.Signature):
       """Answer factually and concisely."""
       question: str = dspy.InputField(desc="Clear, specific question")
       answer: str = dspy.OutputField(desc="Accurate answer in 1-2 sentences")
   ```

2. **Use ChainOfThought for reasoning:**
   ```python
   predictor = dspy.ChainOfThought(signature)
   ```

3. **Add examples via BootstrapFewShot:**
   ```python
   optimizer = dspy.BootstrapFewShot(metric=metric_fn)
   program = optimizer.compile(student=program, trainset=examples)
   ```

4. **Adjust temperature:**
   ```python
   dspy.configure(temperature=0.7)  # Lower for consistency, higher for creativity
   ```

## Signature and Module Problems

### Issue: `TypeError` in signature definition

**Symptoms:** Error like "Invalid field type" or attribute errors in Signature class.

**Solutions:**
1. Use proper field definitions:
   ```python
   class MySignature(dspy.Signature):
       input_field: str = dspy.InputField()  # Correct
       # NOT: input_field = "..."  # Wrong
   ```

2. Ensure type annotations are present:
   ```python
   question: str = dspy.InputField()
   answer: str = dspy.OutputField()
   ```

3. Use string signatures for simple cases:
   ```python
   "question -> answer"  # Simpler syntax
   ```

### Issue: Module forward() not called properly

**Symptoms:** Module produces no output or errors when called.

**Solutions:**
1. Ensure `forward()` is defined:
   ```python
   class MyModule(dspy.Module):
       def forward(self, **kwargs):  # Required method
           return self.predictor(**kwargs)
   ```

2. Call module correctly:
   ```python
   result = my_module(input_var="value")  # Calls forward()
   result = my_module.forward(input_var="value")  # Direct call
   ```

3. Check field names match signature:
   ```python
   # Signature expects 'question' and 'answer'
   result = predictor(question="What?")  # Must use correct field names
   ```

### Issue: Nested module composition errors

**Symptoms:** Errors in composite modules or pipelines.

**Solutions:**
1. Ensure sub-modules are initialized in `__init__`:
   ```python
   def __init__(self):
       super().__init__()
       self.step1 = dspy.Predict("input -> intermediate")
       self.step2 = dspy.Predict("intermediate -> output")
   ```

2. Pass outputs correctly between steps:
   ```python
   def forward(self, input):
       intermediate = self.step1(input=input).intermediate
       return self.step2(intermediate=intermediate)
   ```

3. Use consistent field naming across pipeline

## Evaluation Issues

### Issue: Evaluation hangs or is very slow

**Symptoms:** `Evaluate` runs for a long time without completing.

**Solutions:**
1. Use fewer threads initially:
   ```python
   evaluator = dspy.evaluate.Evaluate(
       devset=dev_set,
       metric=metric_fn,
       num_threads=1  # Start with 1
   )
   ```

2. Evaluate on subset first:
   ```python
   small_set = dev_set[:10]
   score = evaluator(program)
   ```

3. Increase timeout for slower models:
   ```python
   dspy.configure(request_timeout=120)
   ```

4. Check for infinite loops in metric function

### Issue: Metric function errors

**Symptoms:** `Evaluate` crashes with errors in the metric function.

**Solutions:**
1. Add error handling to metric:
   ```python
   def metric_fn(example, pred, trace=None):
       try:
           return example.answer == pred.answer
       except:
           return False
   ```

2. Verify metric receives correct objects:
   ```python
   def metric_fn(example, pred, trace=None):
       print(f"Example keys: {example.keys()}")
       print(f"Pred keys: {pred.keys()}")
       return True
   ```

3. Check Example objects have required fields

### Issue: Zero or unexpected metric scores

**Symptoms:** All predictions score 0, or all score 100%.

**Solutions:**
1. Debug metric function:
   ```python
   example = dev_set[0]
   pred = program(input=example.input)
   print(metric_fn(example, pred))  # Test single example
   ```

2. Check field names and types:
   ```python
   print(example.keys())
   print(pred.keys())
   ```

3. Verify metric logic is correct:
   ```python
   # Make sure comparison is meaningful
   def metric_fn(example, pred, trace=None):
       expected = str(example.answer).lower().strip()
       actual = str(pred.answer).lower().strip()
       return expected == actual
   ```

## Optimization and Compilation

### Issue: Optimizer doesn't improve performance

**Symptoms:** Compiled program performs same or worse than original.

**Solutions:**
1. **Ensure metric is correct:**
   - Test metric on known good/bad examples
   - Verify metric returns boolean

2. **Check trainset quality:**
   ```python
   # Verify trainset has good examples
   for ex in train_set[:5]:
       print(ex)
   ```

3. **Try different optimizer:**
   ```python
   # If BootstrapFewShot doesn't work, try MIPRO
   optimizer = dspy.MIPRO(metric=metric_fn)
   ```

4. **Increase training set size:**
   ```python
   optimizer = dspy.BootstrapFewShot(
       metric=metric_fn,
       max_bootstrapped_demos=8  # Increase from default
   )
   ```

### Issue: `BootstrapFewShot` hangs or takes very long

**Symptoms:** Optimizer runs indefinitely or very slowly.

**Solutions:**
1. Reduce max_rounds:
   ```python
   optimizer = dspy.BootstrapFewShot(
       metric=metric_fn,
       max_rounds=3  # Default is often higher
   )
   ```

2. Use smaller trainset:
   ```python
   small_train = train_set[:20]
   program = optimizer.compile(student=program, trainset=small_train)
   ```

3. Set max_bootstrapped_demos:
   ```python
   optimizer = dspy.BootstrapFewShot(
       metric=metric_fn,
       max_bootstrapped_demos=3
   )
   ```

### Issue: Optimized program crashes

**Symptoms:** `forward()` works but compiled program fails.

**Solutions:**
1. The program may have added demonstrations that cause issues
2. Check the optimized program's internal state:
   ```python
   optimized = optimizer.compile(student=program, trainset=train_set)
   # Inspect compiled demonstrations
   print(optimized.predictors[0].demos)
   ```

3. Manually set reasonable demonstrations instead of relying on optimization

## Performance and Caching

### Issue: Slow predictions or repeated API calls

**Symptoms:** Predictions take a long time, multiple identical requests to API.

**Solutions:**
1. **Enable caching:**
   ```python
   dspy.settings.cache = True
   ```

2. **Use local disk cache:**
   ```python
   import diskcache
   cache = diskcache.Cache('.dspy_cache')
   dspy.settings.cache = cache
   ```

3. **Batch requests:**
   ```python
   results = [predictor(q=q) for q in questions]  # Allows caching
   ```

### Issue: Memory usage grows over time

**Symptoms:** Program uses increasing memory, crashes after many predictions.

**Solutions:**
1. Clear cache periodically:
   ```python
   dspy.settings.cache_clear()
   ```

2. Limit cache size:
   ```python
   import diskcache
   cache = diskcache.Cache('.cache', size_limit=int(1e9))  # 1GB limit
   dspy.settings.cache = cache
   ```

3. Use generators for large datasets:
   ```python
   def predict_batch(items):
       for item in items:
           yield predictor(input=item)
   ```

## Debugging Tips

### Enable Detailed Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('dspy')
logger.setLevel(logging.DEBUG)
```

### Inspect Prediction Objects

```python
result = predictor(question="What is DSPy?")

# View all fields
print(dict(result))

# Check metadata
print(result.keys())

# Access specific field
print(result.answer)
```

### Trace LM Calls

```python
# Enable tracing
dspy.settings.trace = True

# Run prediction
result = predictor(question="Test")

# View trace (implementation varies by version)
if hasattr(result, '_trace'):
    print(result._trace)
```

### Test Signatures Independently

```python
# Test signature before using in module
class TestSig(dspy.Signature):
    input: str
    output: str

predictor = dspy.Predict(TestSig)
result = predictor(input="test")
print(result.output)
```

### Create Minimal Reproduction

```python
# If facing issues, create simplest possible example
dspy.configure(model='gpt-4')

sig = dspy.Predict("input -> output")
result = sig(input="test")
print(result)
```

---

**Not finding your issue?** Check the [official DSPy issues](https://github.com/stanfordnlp/dspy/issues) or ask in the [DSPy community](https://discord.gg/stanfordnlp).
