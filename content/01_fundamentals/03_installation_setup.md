# Installation and Setup

Before we can start building with DSPy, let's make sure your environment is properly configured.

> **Note**: If you've already completed the [Setup Instructions](../00-frontmatter/03-setup-instructions.md) from the front matter, you can skip ahead to [Verification](#verification) to confirm everything is working.

---

## Quick Setup Checklist

Ensure you have:

- [ ] Python 3.9 or higher installed
- [ ] Virtual environment created and activated
- [ ] DSPy installed (`pip install dspy-ai`)
- [ ] LM provider API key configured
- [ ] `python-dotenv` installed for environment variables

---

## Installation

### 1. Install DSPy

With your virtual environment activated:

```bash
pip install dspy-ai
```

This installs the latest stable version of DSPy.

### 2. Install Additional Dependencies

For the examples in this chapter:

```bash
pip install openai anthropic python-dotenv
```

**What these provide**:
- `openai`: OpenAI API client (for GPT models)
- `anthropic`: Anthropic API client (for Claude models)
- `python-dotenv`: Load environment variables from `.env` files

---

## Configuration

### Set Up Your API Key

Create a `.env` file in your project directory:

```bash
# Create .env file
touch .env
```

Add your API key:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

Or for Anthropic:

```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

> **Security**: Never commit `.env` files to version control! Add `.env` to your `.gitignore`.

---

## Verification

Let's verify DSPy is installed correctly.

### Check DSPy Version

```bash
python -c "import dspy; print(f'DSPy version: {dspy.__version__}')"
```

**Expected output**:
```
DSPy version: 2.5.x
```

### Quick Test

Create a file `test_dspy.py`:

```python
"""Quick test to verify DSPy installation."""

import os
from dotenv import load_dotenv
import dspy

# Load environment variables
load_dotenv()

def main():
    print("Testing DSPy installation...")
    print(f"DSPy version: {dspy.__version__}")

    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✓ API key found")

        # Try to configure a language model
        try:
            lm = dspy.LM(model="openai/gpt-4o-mini", api_key=api_key)
            dspy.configure(lm=lm)
            print("✓ Language model configured successfully")

            # Simple test
            class TestSignature(dspy.Signature):
                question: str = dspy.InputField()
                answer: str = dspy.OutputField()

            predictor = dspy.Predict(TestSignature)
            result = predictor(question="What is 1+1?")
            print(f"✓ Test prediction: {result.answer}")

            print("\n✅ DSPy is working correctly!")

        except Exception as e:
            print(f"✗ Error: {e}")
            print("\n⚠️  Check your API key and internet connection")

    else:
        print("✗ API key not found")
        print("Please set OPENAI_API_KEY in your .env file")

if __name__ == "__main__":
    main()
```

Run it:

```bash
python test_dspy.py
```

**Expected output**:
```
Testing DSPy installation...
DSPy version: 2.5.x
✓ API key found
✓ Language model configured successfully
✓ Test prediction: 2

✅ DSPy is working correctly!
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'dspy'`

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# Reinstall DSPy
pip install dspy-ai
```

### Issue: `API key not found`

**Solution**:
1. Check `.env` file exists in your project directory
2. Verify format: `OPENAI_API_KEY=sk-...`
3. Ensure `load_dotenv()` is called before using the key
4. Check for typos in the variable name

### Issue: `Connection error` or `Authentication failed`

**Solution**:
1. Verify your API key is valid (not expired/revoked)
2. Check internet connectivity
3. Ensure you have billing set up with your LM provider
4. Try the key in the provider's web interface to confirm it works

### Issue: Old DSPy version

**Solution**:
```bash
# Upgrade to latest version
pip install --upgrade dspy-ai
```

---

## IDE Setup (Optional)

### Visual Studio Code

Install recommended extensions:
1. **Python** (Microsoft)
2. **Pylance** (Microsoft)
3. **Python Indent**

Configure to use your virtual environment:
- `Cmd/Ctrl + Shift + P` → "Python: Select Interpreter"
- Choose the interpreter from your `venv` directory

### Jupyter Notebook (Optional)

If you prefer notebooks:

```bash
pip install jupyter ipykernel

# Add your virtual environment as a kernel
python -m ipykernel install --user --name=dspy-env
```

Start Jupyter:
```bash
jupyter notebook
```

---

## Environment Variables

### Using .env Files (Recommended)

```python
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Access variables
api_key = os.getenv("OPENAI_API_KEY")
```

### Using System Environment Variables

**macOS/Linux** (temporary):
```bash
export OPENAI_API_KEY="sk-your-key"
```

**Windows Command Prompt**:
```bash
set OPENAI_API_KEY=sk-your-key
```

**Windows PowerShell**:
```bash
$env:OPENAI_API_KEY="sk-your-key"
```

---

## Alternative: Using Local Models

Don't want to use API-based models? You can use local models with Ollama.

### Install Ollama

1. Visit [ollama.ai](https://ollama.ai)
2. Download and install for your OS
3. Pull a model:

```bash
ollama pull llama3
```

### Configure DSPy with Ollama

```python
import dspy

# No API key needed!
lm = dspy.LM(model="ollama/llama3", api_base="http://localhost:11434")
dspy.configure(lm=lm)
```

---

## Next Steps

Now that DSPy is installed and configured, you're ready to write your first program!

**Continue to**: [Your First DSPy Program](04-first-dspy-program.md)

---

## Quick Reference

### Check Version
```bash
python -c "import dspy; print(dspy.__version__)"
```

### Upgrade DSPy
```bash
pip install --upgrade dspy-ai
```

### Install with All Extras
```bash
pip install "dspy-ai[all]"
```

### Uninstall DSPy
```bash
pip uninstall dspy-ai
```

---

## Additional Help

- **Detailed setup**: [Front Matter Setup Instructions](../00-frontmatter/03-setup-instructions.md)
- **Prerequisites**: [Front Matter Prerequisites](../00-frontmatter/02-prerequisites.md)
- **DSPy docs**: [https://dspy.ai/learn/installation](https://dspy.ai/learn/installation)
- **Troubleshooting**: [Appendix Troubleshooting](../09-appendices/02-troubleshooting.md)
