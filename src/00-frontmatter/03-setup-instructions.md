# Setup Instructions

This chapter will guide you through setting up your DSPy development environment step by step. Follow these instructions carefully, and you'll be ready to start building with DSPy in about 15-30 minutes.

---

## Overview

We'll complete these steps:
1. ✅ Verify Python installation
2. ✅ Create a project directory
3. ✅ Set up a virtual environment
4. ✅ Install DSPy and dependencies
5. ✅ Configure API access
6. ✅ Run a test program
7. ✅ Clone the book's code examples (optional)

---

## Step 1: Verify Python Installation

First, confirm you have Python 3.9 or higher installed.

**Open your terminal** and run:

```bash
python3 --version
```

**Expected output** (your version may differ):
```
Python 3.11.5
```

> **Note**: On some systems, the command is `python` instead of `python3`. Use whichever works on your system.

**If Python is not installed or version is < 3.9**:
- Visit [Python.org](https://www.python.org/downloads/) to download and install the latest version
- After installation, close and reopen your terminal
- Verify the installation again

---

## Step 2: Create a Project Directory

Create a dedicated folder for your DSPy projects.

### On macOS/Linux

```bash
# Create a directory for DSPy projects
mkdir ~/dspy-learning
cd ~/dspy-learning
```

### On Windows (Command Prompt)

```cmd
# Create a directory for DSPy projects
mkdir %USERPROFILE%\dspy-learning
cd %USERPROFILE%\dspy-learning
```

### On Windows (PowerShell)

```powershell
# Create a directory for DSPy projects
New-Item -ItemType Directory -Path "$env:USERPROFILE\dspy-learning" -Force
Set-Location "$env:USERPROFILE\dspy-learning"
```

> **Tip**: You can create this directory anywhere you like. Just remember its location!

**Verify you're in the right directory**:

### macOS/Linux
```bash
pwd
```

**Expected output**:
```
/Users/yourname/dspy-learning
```

### Windows (Command Prompt)
```cmd
cd
```

**Expected output**:
```
C:\Users\yourname\dspy-learning
```

### Windows (PowerShell)
```powershell
Get-Location
```

**Expected output**:
```
C:\Users\yourname\dspy-learning
```

---

## Step 3: Set Up a Virtual Environment

Virtual environments isolate your project's dependencies from other Python projects.

### Create the Virtual Environment

```bash
python3 -m venv venv
```

This creates a folder named `venv` containing an isolated Python environment.

### Activate the Virtual Environment

**On macOS/Linux**:
```bash
source venv/bin/activate
```

**On Windows (Command Prompt)**:
```bash
venv\Scripts\activate
```

**On Windows (PowerShell)**:
```bash
venv\Scripts\Activate.ps1
```

**Expected result**:
Your terminal prompt should change to show `(venv)` at the beginning:

```
(venv) user@computer:~/dspy-learning$
```

> **Important**: Always activate your virtual environment before working on DSPy projects!

### Verify Virtual Environment

```bash
which python3
```

**Expected output** (path will vary):
```
/Users/yourname/dspy-learning/venv/bin/python3
```

The path should point to your `venv` directory.

---

## Step 4: Install DSPy and Dependencies

Now we'll install DSPy and the packages you'll need for this book.

### Upgrade pip (Recommended)

```bash
pip install --upgrade pip
```

### Install DSPy

```bash
pip install dspy-ai
```

**This will install**:
- DSPy framework
- Core dependencies

**Verify installation**:

```bash
python3 -c "import dspy; print(f'DSPy version: {dspy.__version__}')"
```

**Expected output**:
```
DSPy version: 2.5.x
```

### Install Additional Dependencies

For the examples in this book, install these packages:

```bash
pip install openai anthropic python-dotenv
```

**What these packages do**:
- `openai`: OpenAI API client
- `anthropic`: Anthropic (Claude) API client
- `python-dotenv`: Load API keys from `.env` files

### Optional: Install All Book Dependencies

If you've cloned the book's repository, install all dependencies at once:

```bash
pip install -r requirements.txt
```

---

## Step 5: Configure API Access

You'll need an API key to use language models with DSPy.

### Option 1: OpenAI API (Recommended for Beginners)

#### Get Your API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Click "Create new secret key"
5. Copy the key (it starts with `sk-`)

> **Warning**: Keep your API key secret! Never commit it to Git or share it publicly.

#### Configure the API Key

**Method 1: Environment File (Recommended)**

Create a `.env` file in your project directory:

```bash
# Create .env file
touch .env
```

Open `.env` in your text editor and add:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Method 2: Environment Variable**

**On macOS/Linux** (temporary, current session only):
```bash
export OPENAI_API_KEY="sk-your-actual-api-key-here"
```

**On Windows (Command Prompt)**:
```bash
set OPENAI_API_KEY=sk-your-actual-api-key-here
```

**On Windows (PowerShell)**:
```bash
$env:OPENAI_API_KEY="sk-your-actual-api-key-here"
```

### Option 2: Anthropic API (Claude)

#### Get Your API Key

1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to API Keys
4. Create a new key
5. Copy the key

#### Configure the API Key

Add to your `.env` file:

```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

### Option 3: Local Models (Ollama)

For free, local LLMs:

1. Install [Ollama](https://ollama.ai/)
2. Pull a model: `ollama pull llama3`
3. No API key needed!

---

## Step 6: Run a Test Program

Let's verify everything is working with a simple test.

### Create a Test Script

Create a file named `test_setup.py`:

```python
"""
Test script to verify DSPy installation and API access.
"""

import os
from dotenv import load_dotenv
import dspy

# Load environment variables
load_dotenv()

def test_dspy_installation():
    """Test DSPy installation."""
    print("=" * 60)
    print("DSPy Installation Test")
    print("=" * 60)
    print()

    print(f"✓ DSPy version: {dspy.__version__}")
    print()

def test_openai_connection():
    """Test OpenAI API connection."""
    print("Testing OpenAI API connection...")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("✗ OPENAI_API_KEY not found in environment")
        print("  Please set your API key in .env file")
        return False

    try:
        # Configure language model
        lm = dspy.LM(
            model="openai/gpt-4o-mini",
            api_key=api_key,
            temperature=0.7
        )
        dspy.configure(lm=lm)

        # Test with a simple prediction
        class SimpleQA(dspy.Signature):
            """Answer a question."""
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        predictor = dspy.Predict(SimpleQA)
        result = predictor(question="What is 2+2?")

        print(f"✓ OpenAI API connection successful")
        print(f"  Test question: What is 2+2?")
        print(f"  Answer: {result.answer}")
        return True

    except Exception as e:
        print(f"✗ Error connecting to OpenAI API:")
        print(f"  {e}")
        return False

def main():
    """Run all tests."""
    test_dspy_installation()

    print("Testing API connectivity...")
    print()

    success = test_openai_connection()

    print()
    print("=" * 60)
    if success:
        print("✓ Setup complete! You're ready to start learning DSPy.")
    else:
        print("⚠ Setup incomplete. Please check your API key configuration.")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

### Run the Test

```bash
python3 test_setup.py
```

**Expected output** (if successful):

```
============================================================
DSPy Installation Test
============================================================

✓ DSPy version: 2.5.x

Testing API connectivity...

Testing OpenAI API connection...
✓ OpenAI API connection successful
  Test question: What is 2+2?
  Answer: 4

============================================================
✓ Setup complete! You're ready to start learning DSPy.
============================================================
```

**If you see errors**:
- Check that your `.env` file has the correct API key
- Verify the API key is valid (not expired or revoked)
- Ensure you have internet connectivity
- Check that your virtual environment is activated

---

## Step 7: Clone the Book's Code Examples (Optional)

To access all the code examples from this book:

```bash
# Navigate to your projects directory
cd ~/dspy-learning

# Clone the repository
git clone https://github.com/dustinober1/Ebook_DSPy.git

# Navigate into the repository
cd Ebook_DSPy

# Install dependencies
pip install -r requirements.txt
```

**Repository structure**:
```
Ebook_DSPy/
├── examples/          # All code examples by chapter
├── exercises/         # Exercise starter code and solutions
├── assets/           # Datasets and templates
└── scripts/          # Build and utility scripts
```

---

## Common Issues and Solutions

### Issue: "command not found: python3"

**Solution**: Try `python` instead of `python3`, or install Python from [Python.org](https://www.python.org/).

### Issue: "No module named 'dspy'"

**Solution**:
1. Ensure your virtual environment is activated (you should see `(venv)` in your prompt)
2. Reinstall: `pip install dspy-ai`

### Issue: "API key not found"

**Solution**:
1. Check that `.env` file exists in your project directory
2. Verify the key format: `OPENAI_API_KEY=sk-...`
3. Ensure you're loading dotenv: `load_dotenv()` in your code
4. Check for typos in the key

### Issue: "Permission denied" when activating venv (Windows PowerShell)

**Solution**:
Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy RemoteSigned
```

### Issue: API calls failing with authentication errors

**Solution**:
1. Verify your API key is valid (try it in the provider's web interface)
2. Check if you have billing set up (OpenAI requires payment method after free credits)
3. Ensure the key hasn't expired

---

## Development Workflow

Now that you're set up, here's your typical workflow:

### Starting a New Session

```bash
# 1. Navigate to your project directory
cd ~/dspy-learning

# 2. Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows

# 3. Start coding!
```

### When You're Done

```bash
# Deactivate virtual environment
deactivate
```

---

## Editor Setup (Optional)

### Visual Studio Code

If using VS Code, install these extensions for the best experience:

1. **Python** (Microsoft) - Python language support
2. **Pylance** (Microsoft) - Fast Python language server
3. **Python Indent** - Correct Python indentation

**Configure VS Code to use your virtual environment**:
1. Open Command Palette (Cmd/Ctrl + Shift + P)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from your `venv` directory

### PyCharm

PyCharm automatically detects virtual environments. Just:
1. Open your project folder
2. PyCharm will prompt to use the detected venv
3. Click "OK"

---

## Next Steps

Congratulations! Your DSPy development environment is ready. 🎉

**You're now ready to**:
- Start Chapter 1: DSPy Fundamentals
- Run code examples from the book
- Experiment with DSPy modules
- Build your own LM-powered applications

### Recommended First Steps

1. **Read Chapter 1**: Learn DSPy fundamentals
2. **Run examples**: Try the code examples in `examples/chapter01/`
3. **Do exercises**: Practice with the chapter exercises
4. **Experiment**: Modify examples to see what happens

---

## Quick Reference

### Activate Virtual Environment

**macOS/Linux**:
```bash
source venv/bin/activate
```

**Windows**:
```bash
venv\Scripts\activate
```

### Install Package

```bash
pip install package-name
```

### Run Python Script

```bash
python3 script.py
```

### Deactivate Virtual Environment

```bash
deactivate
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check the appendices**: Chapter 9 has a troubleshooting guide
2. **DSPy Documentation**: [https://dspy.ai](https://dspy.ai)
3. **GitHub Issues**: [DSPy Repository](https://github.com/stanfordnlp/dspy/issues)
4. **Community**: [GitHub Discussions](https://github.com/stanfordnlp/dspy/discussions)

---

## You're All Set!

Your development environment is configured and tested. Time to start building with DSPy!

**Next**: Begin your learning journey with [Chapter 1: DSPy Fundamentals](../01-fundamentals/01-what-is-dspy.md)

Happy coding! 🚀
