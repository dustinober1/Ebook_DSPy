# Prerequisites

Before diving into DSPy, let's ensure you have the necessary background knowledge and tools. Don't worry if you're missing some prerequisites—we'll point you to resources to fill in any gaps.

---

## Required Knowledge

### 1. Python Programming (Required)

**What you need to know:**
- ✅ Basic syntax (variables, loops, conditionals, functions)
- ✅ Object-oriented programming (classes, inheritance, methods)
- ✅ Working with modules and packages (`import` statements)
- ✅ Basic error handling (`try/except`)
- ✅ Working with common data structures (lists, dicts, sets)

**Recommended experience**: 6+ months of Python programming

**Self-assessment**:
If you can understand and write code like this, you're ready:

```python
class DataProcessor:
    def __init__(self, data):
        self.data = data

    def process(self):
        results = []
        for item in self.data:
            try:
                result = self._transform(item)
                results.append(result)
            except ValueError as e:
                print(f"Skipping {item}: {e}")
        return results

    def _transform(self, item):
        return item.upper()
```

**Need to learn Python?**
- [Python.org Official Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/)
- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)

---

### 2. Command Line Basics (Required)

**What you need to know:**
- ✅ Navigate directories (`cd`, `ls`/`dir`)
- ✅ Run Python scripts (`python script.py`)
- ✅ Install packages (`pip install package`)
- ✅ Use a text editor or IDE

**Self-assessment**:
Can you execute these commands?

```bash
cd my-project
pip install -r requirements.txt
python my_script.py
```

**Need help?**
- [Command Line Crash Course](https://learnpythonthehardway.org/python3/appendixa.html)
- [The Linux Command Line](https://linuxcommand.org/tlcl.php) (also applies to macOS/Windows WSL)

---

### 3. Large Language Models Basics (Helpful, Not Required)

**Recommended knowledge:**
- Understanding what LLMs are (ChatGPT, GPT-4, Claude, etc.)
- Basic familiarity with prompting (asking LLMs questions)
- Awareness of API-based LLM usage

**Don't worry if you're new to LLMs!**

Chapter 1 covers everything you need to know about LLMs in the context of DSPy. However, if you want a head start:

**Recommended reading**:
- [OpenAI's GPT-4 Documentation](https://platform.openai.com/docs/)
- [Anthropic's Claude Documentation](https://docs.anthropic.com/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

## Optional Knowledge

These topics are helpful but not required. The book will introduce them as needed:

### Machine Learning Basics
- Understanding of training/testing splits
- Familiarity with metrics (accuracy, F1, etc.)
- Concept of optimization

### Natural Language Processing
- Text preprocessing concepts
- Understanding of embeddings (helpful for RAG chapters)

### Software Engineering
- Version control (Git)
- Testing practices
- API development (for deployment chapters)

---

## Technical Requirements

### 1. Operating System

DSPy works on:
- ✅ **macOS** (10.14+)
- ✅ **Linux** (Ubuntu 20.04+, or similar)
- ✅ **Windows** (10/11 with WSL2 recommended, or native Python)

> **Windows Users**: We recommend using Windows Subsystem for Linux (WSL2) for the best experience, though native Windows with Python 3.9+ also works.

---

### 2. Python Installation

**Required version**: Python 3.9 or higher

**Check your Python version**:

```bash
python3 --version
```

or

```bash
python --version
```

**Expected output** (version may vary):
```
Python 3.11.5
```

**Don't have Python 3.9+?**
- [Python.org Downloads](https://www.python.org/downloads/)
- [Anaconda Distribution](https://www.anaconda.com/products/distribution) (includes many scientific packages)

---

### 3. Package Manager (pip)

Python's package manager should be installed with Python.

**Verify pip installation**:

```bash
pip3 --version
```

or

```bash
pip --version
```

**Expected output**:
```
pip 23.2.1 from /usr/local/lib/python3.11/site-packages/pip (python 3.11)
```

---

### 4. Text Editor or IDE

You'll need a code editor. Popular choices:

**For Beginners**:
- [Visual Studio Code](https://code.visualstudio.com/) (Free, excellent Python support)
- [PyCharm Community Edition](https://www.jetbrains.com/pycharm/) (Free, Python-focused)

**For Advanced Users**:
- [Vim](https://www.vim.org/) / [Neovim](https://neovim.io/)
- [Emacs](https://www.gnu.org/software/emacs/)
- [Sublime Text](https://www.sublimetext.com/)

**Cloud-based (no installation)**:
- [Google Colab](https://colab.research.google.com/) (Free, includes Python environment)
- [Replit](https://replit.com/) (Free tier available)

---

### 5. Virtual Environment Tool (Recommended)

Virtual environments keep your project dependencies isolated.

**Options**:
- **venv** (built into Python 3.3+) - Recommended for most users
- **conda** (if using Anaconda)
- **poetry** (for advanced dependency management)

**We'll cover setup** in the next chapter.

---

## API Access Requirements

To use DSPy with LLM providers, you'll need API access to at least one:

### Primary Options (Choose One)

#### Option 1: OpenAI API (Recommended for Beginners)
- **Cost**: Pay-per-use (starts ~$0.002 per 1K tokens for GPT-4o-mini)
- **Sign up**: [OpenAI Platform](https://platform.openai.com/)
- **Free tier**: $5 credit for new accounts
- **Best for**: Experimenting and learning

#### Option 2: Anthropic API (Claude)
- **Cost**: Pay-per-use (pricing similar to OpenAI)
- **Sign up**: [Anthropic Console](https://console.anthropic.com/)
- **Best for**: Production applications, longer contexts

#### Option 3: Local Models (Free)
- **Options**: Ollama, LM Studio, LocalAI
- **Cost**: Free (requires local GPU/CPU resources)
- **Best for**: Privacy, experimentation without API costs
- **Note**: Performance may vary compared to commercial APIs

### Cost Expectations

For working through this book:
- **Estimated cost**: $5-$20 total
- **Per chapter**: ~$0.50-$2 depending on exercises
- **Cost-saving tips**:
  - Use cheaper models (e.g., GPT-4o-mini, GPT-3.5-turbo) for learning
  - Cache responses when experimenting
  - Use local models for initial development

---

## Hardware Requirements

### Minimum Requirements
- **Processor**: Any modern CPU (2+ GHz)
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 2 GB free space for Python packages and examples
- **Internet**: Required for API-based models

### For Local Models (Optional)
- **GPU**: NVIDIA GPU with 8+ GB VRAM (for larger models)
- **RAM**: 16 GB+ (32 GB for larger models)
- **Storage**: 10-50 GB depending on model size

> **Note**: You don't need powerful hardware to learn DSPy. API-based models run in the cloud—your computer just sends requests and receives responses.

---

## Time Commitment

Set realistic expectations for your learning journey:

### Complete Beginner Path
- **Total time**: 40-60 hours
- **Weekly commitment**: 5-10 hours over 6-8 weeks
- **Includes**: All chapters, exercises, 2-3 case studies

### Intermediate Developer Path
- **Total time**: 20-30 hours
- **Weekly commitment**: 5-10 hours over 3-4 weeks
- **Includes**: Core chapters, selected case studies

### Advanced/Reference Path
- **Total time**: 5-20 hours (variable)
- **Commitment**: As needed for specific topics

---

## Preparation Checklist

Before moving to the Setup Instructions, ensure you have:

- [ ] Python 3.9+ installed
- [ ] pip package manager available
- [ ] Text editor or IDE ready
- [ ] Basic Python knowledge (can write simple classes and functions)
- [ ] Command line basics (can navigate directories and run scripts)
- [ ] API key for at least one LLM provider (or plan to use local models)
- [ ] 1-2 hours available for initial setup and first examples

---

## Still Have Questions?

**Common concerns addressed**:

### "I'm not sure if I know enough Python..."

If you can:
- Write functions and classes
- Use loops and conditionals
- Import modules
- Handle basic errors

Then you're ready! The book includes detailed explanations for DSPy-specific concepts.

### "I've never used LLMs before..."

Perfect! Chapter 1 introduces everything you need to know about LLMs in the context of DSPy. No prior LLM experience required.

### "I don't have an OpenAI API key..."

You have several options:
1. Create an OpenAI account (gets $5 free credit)
2. Use Anthropic's Claude (similar setup)
3. Use local models (free, but requires setup)
4. Ask your organization if they provide API access

### "My Python version is older than 3.9..."

DSPy requires Python 3.9+ for modern features. We strongly recommend upgrading—it's worth it not just for DSPy, but for all modern Python development.

---

## Ready for Setup?

If you meet the prerequisites above (or know where to fill gaps), you're ready to proceed!

**Next**: [Setup Instructions](03-setup-instructions.md) will guide you through installing DSPy and configuring your environment.

Let's get your development environment ready! 🚀
