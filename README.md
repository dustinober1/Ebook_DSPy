# DSPy: A Practical Guide

A comprehensive, hands-on tutorial for learning DSPy from fundamentals to production-ready applications.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DSPy](https://img.shields.io/badge/dspy-2.5+-orange.svg)](https://dspy.ai)

---

## 📚 About This Book

This ebook teaches you how to build robust, optimizable LLM-based applications using DSPy. Whether you're new to DSPy or looking to master advanced techniques, this book provides:

- **Progressive learning paths** for beginners through advanced practitioners
- **50+ hands-on code examples** you can run and modify
- **40+ exercises** with solutions to reinforce concepts
- **9 complete case studies** across healthcare, finance, legal, research, and business domains
- **Production-ready patterns** for deploying DSPy applications

---

## 🎯 Who This Book Is For

- **Complete beginners** who want to learn DSPy from scratch
- **Intermediate developers** familiar with LLMs who want to learn DSPy's framework
- **Advanced practitioners** looking for optimization techniques and production patterns

---

## 📖 Book Structure

The book is organized into five parts:

### Part I: Foundations (Beginner)
- Chapter 1: DSPy Fundamentals

### Part II: Core Concepts (Intermediate)
- Chapter 2: Signatures
- Chapter 3: Modules

### Part III: Evaluation & Optimization (Intermediate-Advanced)
- Chapter 4: Evaluation
- Chapter 5: Optimizers and Compilation

### Part IV: Real-World Applications (Advanced)
- Chapter 6: Building Real-World Applications
- Chapter 7: Advanced Topics

### Part V: Case Studies (Expert)
- Chapter 8: Domain-Specific Case Studies
  - Healthcare: Clinical Notes Analysis
  - Finance: Document Analysis
  - Legal: Contract Review
  - Research: Literature Review & Data Pipelines
  - Business/Enterprise: Customer Support, RAG Systems, BI

### Appendices
- Chapter 9: API Reference, Troubleshooting, Resources, Glossary

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **Basic Python knowledge** (classes, functions, modules)
- **Command line basics**
- **API key** for OpenAI, Anthropic, or local LLM setup

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dustinober1/Ebook_DSPy.git
   cd Ebook_DSPy
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key**:
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your-key-here" > .env
   ```

5. **Read the book**:
   - **Online**: Visit the [online version](#) (coming soon)
   - **Locally**: Open `src/00-frontmatter/00-preface.md` to start reading

---

## 📂 Repository Structure

```
Ebook_DSPy/
├── src/                          # Book content (Markdown)
│   ├── 00-frontmatter/           # Preface, prerequisites, setup
│   ├── 01-fundamentals/          # Chapter 1
│   ├── 02-signatures/            # Chapter 2
│   ├── 03-modules/               # Chapter 3
│   ├── 04-evaluation/            # Chapter 4
│   ├── 05-optimizers/            # Chapter 5
│   ├── 06-real-world-applications/  # Chapter 6
│   ├── 07-advanced-topics/       # Chapter 7
│   ├── 08-case-studies/          # Chapter 8
│   └── 09-appendices/            # Chapter 9
│
├── examples/                     # Code examples by chapter
│   ├── chapter01/                # Chapter 1 examples
│   ├── chapter02/                # Chapter 2 examples
│   └── ...
│
├── exercises/                    # Practice problems & solutions
│   ├── chapter01/                # Chapter 1 exercises
│   │   ├── problems.md
│   │   └── solutions/
│   └── ...
│
├── assets/                       # Supporting materials
│   ├── images/                   # Diagrams and screenshots
│   ├── datasets/                 # Sample data for exercises
│   └── templates/                # Content templates
│
├── scripts/                      # Build and utility scripts
│   ├── build.sh                  # Build the ebook
│   ├── serve.sh                  # Local development server
│   └── validate_code.py          # Validate code examples
│
├── book.toml                     # mdBook configuration
├── SUMMARY.md                    # Table of contents
└── requirements.txt              # Python dependencies
```

---

## 🛠️ Building the Book

This book is built using [mdBook](https://rust-lang.github.io/mdBook/).

### Install mdBook

```bash
# Using Homebrew (macOS/Linux)
brew install mdbook

# Or using Cargo (any platform)
cargo install mdbook
```

### Build HTML Version

```bash
# Quick build
mdbook build

# Output will be in `build/html/`
open build/html/index.html
```

### Local Development Server

```bash
# Start local server with live reload
mdbook serve

# Opens at http://localhost:3000 automatically
# Changes to markdown files reload in real-time
```

### Advanced Build Options

```bash
# Build with specific output directory
mdbook build --dest-dir ./output

# Clean build
rm -rf build && mdbook build

# Serve on custom port
mdbook serve --port 8080

# Serve with specific binding
mdbook serve --hostname 0.0.0.0
```

---

## 💻 Running Code Examples

All code examples are in the `examples/` directory.

### Run an Example

```bash
# Activate virtual environment
source venv/bin/activate

# Set your API key (if not in .env)
export OPENAI_API_KEY=your-key-here

# Run an example
python examples/chapter01/01_hello_dspy.py
```

### Validate All Examples

```bash
python scripts/validate_code.py
```

---

## 📝 Exercises

Each chapter includes exercises to reinforce learning.

### Find Exercises

Exercises are in `exercises/chapterXX/problems.md`

### Example

```bash
# View Chapter 1 exercises
cat exercises/chapter01/problems.md

# See solutions
ls exercises/chapter01/solutions/
```

---

## 🎓 Learning Paths

### Path 1: Complete Beginner (40-60 hours)
- Read all chapters sequentially (0-8)
- Complete all exercises
- Build 2-3 case studies

### Path 2: Intermediate Developer (20-30 hours)
- Skim Chapter 1, deep dive Chapters 2-3
- Study Chapters 5-7
- Complete 1-2 relevant case studies

### Path 3: Advanced/Reference (5-20 hours)
- Jump to relevant chapters
- Focus on Chapters 6-8
- Use as needed for specific topics

See [How to Use This Book](src/00-frontmatter/01-how-to-use-this-book.md) for detailed guidance.

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

- **Report issues**: Found an error or typo? [Open an issue](https://github.com/dustinober1/Ebook_DSPy/issues)
- **Suggest improvements**: Have ideas for additional content? Let us know!
- **Submit corrections**: Found a bug in code? Submit a pull request
- **Share examples**: Built something cool? Share it with the community

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines (coming soon).

---

## 📄 License

- **Content**: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)
- **Code**: MIT License

See [LICENSE](LICENSE) for details (coming soon).

---

## 🔗 Resources

### Official DSPy Resources
- [DSPy Website](https://dspy.ai)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [DSPy Documentation](https://dspy.ai/learn/)
- [DSPy Discussions](https://github.com/stanfordnlp/dspy/discussions)

### Related Materials
- [DSPy Research Paper](https://arxiv.org/abs/2310.03714)
- [Stanford NLP Group](https://nlp.stanford.edu/)

---

## 🙋 Getting Help

- **Book issues**: [GitHub Issues](https://github.com/dustinober1/Ebook_DSPy/issues)
- **DSPy questions**: [DSPy Discussions](https://github.com/stanfordnlp/dspy/discussions)
- **General questions**: See Chapter 9 Appendices

---

## 📊 Project Status

- **Overall Status**: ✅ Content Complete | 🚧 Polish Phase
- **Content Chapters**: ✅ 100% Complete (10 chapters)
  - ✅ Chapter 0: Front Matter
  - ✅ Chapter 1-3: Foundations & Core Concepts
  - ✅ Chapter 4-5: Evaluation & Optimization
  - ✅ Chapter 6-7: Real-World Applications & Advanced
  - ✅ Chapter 8: Case Studies (4 complete)
  - ✅ Chapter 9: Appendices (API Ref, Troubleshooting, Resources, Glossary)
- **Code Examples**: ✅ 25 examples (all syntax validated)
- **Build System**: ✅ mdBook configured and tested
- **Quality Assurance**: 🚧 In Progress

---

## 📧 Contact

- **Author**: Dustin Ober
- **Repository**: [https://github.com/dustinober1/Ebook_DSPy](https://github.com/dustinober1/Ebook_DSPy)

---

## ⭐ Support This Project

If you find this book helpful:
- ⭐ Star this repository
- 📢 Share it with others
- 🐛 Report issues to help improve it
- 🤝 Contribute examples or improvements

---

**Happy learning!** 🚀

*Built with [mdBook](https://rust-lang.github.io/mdBook/) and [DSPy](https://dspy.ai)*
