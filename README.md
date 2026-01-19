# DSPy: The Comprehensive Guide

Welcome to the **DSPy Ebook** repository. This project is a comprehensive, open-source guide designed to help you master **DSPy**, the framework for programming—rather than prompting—language models.

## 📖 About This Book

This ebook takes you from the absolute fundamentals of DSPy to advanced optimization techniques and real-world deployments. Whether you are an AI engineer, a researcher, or a developer looking to build robust LLM applications, this resource provides structured knowledge, practical code examples, and deep dives into the theory behind the framework.

## 📚 Table of Contents

The book is organized into the following chapters:

- **[Chapter 0: Frontmatter](content/00_frontmatter)** - Prerequisites and setup.
- **[Chapter 1: Fundamentals](content/01_fundamentals)** - Introduction to DSPy and the "Programming vs. Prompting" paradigm.
- **[Chapter 2: Signatures](content/02_signatures)** - Defining input/output behavior structurally.
- **[Chapter 3: Modules](content/03_modules)** - Building blocks like `Predict`, `ChainOfThought`, and custom modules.
- **[Chapter 4: Evaluation](content/04_evaluation)** - Metrics, datasets, and systematic evaluation loops.
- **[Chapter 5: Optimizers](content/05_optimizers)** - Compiling and optimizing your DSPy programs (BootstrapFewShot, MIPRO, etc.).
- **[Chapter 6: Real-World Applications](content/06_real_world_applications)** - RAG systems, agents, classification, and more.
- **[Chapter 7: Advanced Topics](content/07_advanced_topics)** - Caching, async execution, and deployment.
- **[Chapter 8: Case Studies](content/08_case_studies)** - Deep dives into enterprise and research use cases.
- **[Chapter 9: Appendices](content/09_appendices)** - API references, glossary, and resources.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A basic understanding of Python programming.
- API keys for LLM providers (e.g., OpenAI, Anthropic, or local models via Ollama).
- **DSPy Version**: This book is tested with `dspy-ai==2.5.43`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Ebook_DSPy
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Repository Structure

Ebook_DSPy/
├── content/        # Content (Notebooks & Markdown)
├── src/            # Python source code and utilities
├── tests/          # Test suite
├── planning/       # Planning and administrative documents
├── requirements.txt # Python dependencies (pinned versions)
├── _config.yml     # Jupyter Book configuration
├── _toc.yml        # Table of Contents
├── CONTRIBUTING.md # Guidelines for contributors
└── LICENSE         # Project license

## 🤝 Contributing

We welcome contributions! Whether fixing typos, adding new examples, or writing entire sections, please read our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

## 📄 License

This project is licensed under the terms found in the [LICENSE](LICENSE) file.
