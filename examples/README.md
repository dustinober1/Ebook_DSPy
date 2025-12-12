# Code Examples

This directory contains all runnable code examples from the DSPy ebook, organized by chapter.

---

## 📂 Directory Structure

```
examples/
├── chapter01/          # Chapter 1: DSPy Fundamentals
├── chapter02/          # Chapter 2: Signatures
├── chapter03/          # Chapter 3: Modules
├── chapter04/          # Chapter 4: Evaluation
├── chapter05/          # Chapter 5: Optimizers
├── chapter06/          # Chapter 6: Real-World Applications
├── chapter07/          # Chapter 7: Advanced Topics
└── chapter08/          # Chapter 8: Case Studies
    ├── healthcare_clinical_notes/
    ├── finance_document_analysis/
    ├── legal_contract_review/
    ├── research_literature_review/
    ├── research_data_pipeline/
    ├── customer_support_automation/
    ├── enterprise_rag_system/
    ├── business_intelligence/
    └── complete_production_system/
```

---

## 🚀 Running Examples

### Prerequisites

1. **Python 3.9+ installed**
2. **Virtual environment activated**
3. **Dependencies installed**: `pip install -r requirements.txt`
4. **API key configured** (in `.env` file)

### Run a Specific Example

```bash
# Make sure you're in the project root directory
cd /path/to/Ebook_DSPy

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run an example
python examples/chapter01/01_hello_dspy.py
```

---

## 📋 Example Naming Convention

All example files follow this naming pattern:

```
XX_descriptive_name.py
```

- **XX**: Two-digit number (01, 02, 03, ...) indicating the order
- **descriptive_name**: Lowercase with underscores describing the example
- **Examples**:
  - `01_hello_dspy.py` - First example, simple intro
  - `02_basic_qa.py` - Second example, basic Q&A
  - `03_configure_lm.py` - Third example, LM configuration

---

## 💡 How to Use These Examples

### 1. Learn by Reading

Each example includes:
- **Docstring**: Explanation of what the example demonstrates
- **Comments**: Line-by-line explanations of key concepts
- **Sample output**: Expected results shown in comments

### 2. Learn by Running

```bash
# Run the example
python examples/chapter01/01_hello_dspy.py

# Observe the output
# Compare with expected output in comments
```

### 3. Learn by Modifying

```bash
# Copy an example to experiment
cp examples/chapter01/01_hello_dspy.py my_experiment.py

# Modify and run
python my_experiment.py
```

**Try these modifications**:
- Change the input data
- Modify the signature
- Adjust model parameters
- Add your own features

---

## 📖 Chapter-by-Chapter Guide

### Chapter 1: DSPy Fundamentals
**Examples**: Basic DSPy concepts, installation verification, first programs

**Key examples**:
- `01_hello_dspy.py` - Your first DSPy program
- `02_basic_qa.py` - Simple question answering
- `03_configure_lm.py` - Language model configuration

### Chapter 2: Signatures
**Examples**: Creating and using DSPy signatures

**Key examples**:
- String-based signatures
- Typed signatures with field descriptions
- Multi-field signatures
- Advanced signature patterns

### Chapter 3: Modules
**Examples**: Working with DSPy modules

**Key examples**:
- `dspy.Predict` for direct predictions
- `dspy.ChainOfThought` for reasoning
- `dspy.ReAct` for agent-based tasks
- Custom module development
- Module composition

### Chapter 4: Evaluation
**Examples**: Evaluation metrics and datasets

**Key examples**:
- Creating evaluation datasets
- Defining custom metrics
- Running evaluation loops
- Analyzing results

### Chapter 5: Optimizers
**Examples**: Program optimization with DSPy optimizers

**Key examples**:
- `BootstrapFewShot` for automatic examples
- `MIPRO` for instruction optimization
- `KNNFewShot` for similarity-based examples
- Optimizer comparison

### Chapter 6: Real-World Applications
**Examples**: Complete applications

**Key examples**:
- RAG system implementation
- Multi-hop question answering
- Classification pipelines
- Entity extraction systems
- Intelligent agents
- Code generation

### Chapter 7: Advanced Topics
**Examples**: Production patterns and advanced techniques

**Key examples**:
- Tool integration
- Caching strategies
- Async operations
- Deployment patterns

### Chapter 8: Case Studies
**Examples**: Complete, production-ready applications

**Projects include**:
- Healthcare clinical notes analysis
- Financial document analysis
- Legal contract review
- Research literature review
- Research data pipelines
- Customer support automation
- Enterprise RAG systems
- Business intelligence

Each case study includes:
- Complete source code
- Sample datasets
- Configuration files
- Deployment instructions
- README with detailed setup

---

## 🧪 Testing Examples

### Validate All Examples

```bash
# Run validation script
python scripts/validate_code.py

# With verbose output
python scripts/validate_code.py --verbose

# Check imports can be resolved
python scripts/validate_code.py --check-imports
```

### Manual Testing

```bash
# Test a specific example
python examples/chapter01/01_hello_dspy.py

# Verify it produces expected output
# Check for errors
```

---

## 🔧 Troubleshooting

### Common Issues

#### ImportError: No module named 'dspy'

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### API Key Not Found

**Solution**:
```bash
# Create .env file in project root
echo "OPENAI_API_KEY=your-key-here" > .env

# Or export environment variable
export OPENAI_API_KEY=your-key-here
```

#### Example Fails to Run

**Solution**:
1. Check that all prerequisites are met
2. Verify your API key is valid
3. Ensure you have internet connectivity
4. Check the error message for specific issues

---

## 📝 Example Template

All examples follow this structure:

```python
"""
Title: Description of the example
Chapter: XX - Chapter Name
Topic: Specific concept

Description:
Detailed explanation of what this example teaches.

Requirements:
- dspy-ai>=2.5.0
- Additional packages if needed

Usage:
python XX_example_name.py
"""

import dspy
from dotenv import load_dotenv

# Configuration
load_dotenv()

# Main code
def main():
    """Main function demonstrating the concept."""
    # Implementation
    pass

# Sample output (as comments)
"""
Expected Output:
--------------
Output shown here
"""

if __name__ == "__main__":
    main()
```

---

## 🤝 Contributing Examples

Have an example you'd like to share?

1. Follow the naming convention
2. Use the example template
3. Include clear comments
4. Add sample output
5. Test your example
6. Submit a pull request

---

## 📚 Additional Resources

- **Book chapters**: `src/` directory contains detailed explanations
- **Exercises**: `exercises/` directory has practice problems
- **Templates**: `assets/templates/example_template.py`

---

## ⚙️ Environment Setup

### Create .env File

```bash
# In the project root directory
touch .env
```

Add your API key(s):

```
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=your-key-here  # Optional
```

### Install Additional Dependencies

Some examples may require additional packages:

```bash
# For RAG examples
pip install chromadb

# For data processing
pip install pandas numpy

# For web scraping examples
pip install beautifulsoup4 requests
```

---

## 💬 Questions or Issues?

- **Found a bug in an example?** [Open an issue](https://github.com/dustinober1/Ebook_DSPy/issues)
- **Need help running an example?** Check the troubleshooting guide in Chapter 9
- **Have a question about the code?** See the relevant chapter for detailed explanations

---

**Happy coding!** 🚀

*All examples are tested with DSPy 2.5+ and Python 3.9+*
