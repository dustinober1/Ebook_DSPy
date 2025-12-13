# Summary

[Introduction](../README.md)

---

# Front Matter

- [Preface](00-frontmatter/00-preface.md)
- [How to Use This Book](00-frontmatter/01-how-to-use-this-book.md)
- [Prerequisites](00-frontmatter/02-prerequisites.md)
- [Setup Instructions](00-frontmatter/03-setup-instructions.md)

---

# Part I: Foundations

- [Chapter 1: DSPy Fundamentals](01-fundamentals/00-chapter-intro.md)
  - [What is DSPy?](01-fundamentals/01-what-is-dspy.md)
  - [Programming vs. Prompting](01-fundamentals/02-programming-vs-prompting.md)
  - [Installation and Setup](01-fundamentals/03-installation-setup.md)
  - [Your First DSPy Program](01-fundamentals/04-first-dspy-program.md)
  - [Language Models](01-fundamentals/05-language-models.md)
  - [Exercises](01-fundamentals/06-exercises.md)

---

# Part II: Core Concepts

- [Chapter 2: Signatures](02-signatures/00-chapter-intro.md)
  - [Understanding Signatures](02-signatures/01-understanding-signatures.md)
  - [Signature Syntax](02-signatures/02-signature-syntax.md)
  - [Typed Signatures](02-signatures/03-typed-signatures.md)
  - [Advanced Signatures](02-signatures/04-advanced-signatures.md)
  - [Practical Examples](02-signatures/05-practical-examples.md)
  - [Exercises](02-signatures/06-exercises.md)

- [Chapter 3: Modules](03-modules/00-chapter-intro.md)
  - [Module Basics](03-modules/01-module-basics.md)
  - [Predict Module](03-modules/02-predict-module.md)
  - [TypedPredictor](03-modules/02a-typed-predictor.md)
  - [Chain of Thought](03-modules/03-chainofthought.md)
  - [ReAct Agents](03-modules/04-react-agents.md)
  - [Custom Modules](03-modules/05-custom-modules.md)
  - [Composing Modules](03-modules/06-composing-modules.md)
  - [Assertions](03-modules/08-assertions.md)
  - [Exercises](03-modules/07-exercises.md)

---

# Part III: Evaluation and Optimization

- [Chapter 4: Evaluation](04-evaluation/00-chapter-intro.md)
  - [Why Evaluation Matters](04-evaluation/01-why-evaluation-matters.md)
  - [Creating Datasets](04-evaluation/02-creating-datasets.md)
  - [Defining Metrics](04-evaluation/03-defining-metrics.md)
  - [Evaluation Loops](04-evaluation/04-evaluation-loops.md)
  - [Best Practices](04-evaluation/05-best-practices.md)
  - [Exercises](04-evaluation/06-exercises.md)

- [Chapter 5: Optimizers and Compilation](05-optimizers/00-chapter-intro.md)
  - [The Compilation Concept](05-optimizers/01-compilation-concept.md)
  - [BootstrapFewShot](05-optimizers/02-bootstrapfewshot.md)
  - [COPRO: Cost-aware Prompt Optimization](05-optimizers/02a-copro.md)
  - [MIPRO](05-optimizers/03-mipro.md)
  - [KNNFewShot](05-optimizers/04-knnfewshot.md)
  - [Fine-tuning](05-optimizers/05-finetuning.md)
  - [Constraint-Driven Optimization](05-optimizers/07-constraint-driven-optimization.md)
  - [Reflective Prompt Evolution](05-optimizers/08-reflective-prompt-evolution.md)
  - [COPA Method](05-optimizers/09-copa-method.md)
  - [Joint Optimization](05-optimizers/10-joint-optimization.md)
  - [Monte Carlo Optimization](05-optimizers/11-monte-carlo-optimization.md)
  - [Bayesian Optimization](05-optimizers/12-bayesian-optimization.md)
  - [Comprehensive Examples](05-optimizers/13-comprehensive-examples.md)
  - [Choosing Optimizers](05-optimizers/06-choosing-optimizers.md)
  - [Multi-stage Optimization Theory](05-optimizers/14-multistage-optimization-theory.md)
  - [Instruction Tuning Frameworks](05-optimizers/15-instruction-tuning-frameworks.md)
  - [Demonstration Optimization](05-optimizers/16-demonstration-optimization.md)
  - [Multi-stage Program Architectures](05-optimizers/17-multistage-architectures.md)
  - [Complex Pipeline Optimization](05-optimizers/18-complex-pipeline-optimization.md)
  - [Instruction-Demonstration Interactions](05-optimizers/19-instruction-demonstration-interactions.md)
  - [Prompts as Auto-Optimized Hyperparameters](05-optimizers/20-prompts-as-hyperparameters.md)
  - [Minimal Data Training Pipelines](05-optimizers/21-minimal-data-pipelines.md)
  - [GEPA: Genetic-Pareto Optimization](05-optimizers/22-gepa-genetic-pareto-optimization.md)
  - [Exercises](05-optimizers/07-exercises.md)

---

# Part IV: Real-World Applications

- [Chapter 6: Building Real-World Applications](06-real-world-applications/00-chapter-intro.md)
  - [RAG Systems](06-real-world-applications/01-rag-systems.md)
  - [Multi-hop Search](06-real-world-applications/02-multi-hop-search.md)
  - [Classification Tasks](06-real-world-applications/03-classification-tasks.md)
  - [Entity Extraction](06-real-world-applications/04-entity-extraction.md)
  - [Intelligent Agents](06-real-world-applications/05-intelligent-agents.md)
  - [Code Generation](06-real-world-applications/06-code-generation.md)
  - [Perspective-Driven Research](06-real-world-applications/07-perspective-driven-research.md)
  - [Extreme Multi-Label Classification](06-real-world-applications/08-extreme-multilabel-classification.md)
  - [Long-Form Generation](06-real-world-applications/08-long-form-generation.md)
  - [Outline Generation](06-real-world-applications/09-outline-generation.md)
  - [Extreme Few-Shot Learning: Training with 10 Gold Labels](06-real-world-applications/11-extreme-few-shot-learning.md)
  - [IR Model Training from Scratch](06-real-world-applications/12-ir-model-training-scratch.md)
  - [Exercises](06-real-world-applications/07-exercises.md)

- [Chapter 7: Advanced Topics](07-advanced-topics/00-chapter-intro.md)
  - [Adapters and Tools](07-advanced-topics/01-adapters-tools.md)
  - [Caching and Performance](07-advanced-topics/02-caching-performance.md)
  - [Async and Streaming](07-advanced-topics/03-async-streaming.md)
  - [Debugging and Tracing](07-advanced-topics/04-debugging-tracing.md)
  - [Deployment Strategies](07-advanced-topics/05-deployment-strategies.md)
  - [Self-Refining Pipelines](07-advanced-topics/07-self-refining-pipelines.md)
  - [Declarative Compilation](07-advanced-topics/08-declarative-compilation.md)
  - [Exercises](07-advanced-topics/06-exercises.md)

---

# Part V: Case Studies

- [Chapter 8: Case Studies](08-case-studies/00-introduction.md)
  - [Enterprise RAG System](08-case-studies/01-enterprise-rag-system.md)
  - [Customer Support Chatbot](08-case-studies/02-customer-support-chatbot.md)
  - [AI Code Assistant](08-case-studies/03-ai-code-assistant.md)
  - [Automated Data Analysis](08-case-studies/04-automated-data-analysis.md)
  - [STORM Writing Assistant](08-case-studies/05-storm-writing-assistant.md)
  - [Assertion-Driven Applications](08-case-studies/06-assertion-driven-applications.md)
  - [Exercises](08-case-studies/05-exercises.md)

---

# Appendices

- [Chapter 9: Appendices](09-appendices/00-introduction.md)
  - [API Reference Quick Guide](09-appendices/01-api-reference-quick.md)
  - [Troubleshooting](09-appendices/02-troubleshooting.md)
  - [Additional Resources](09-appendices/03-resources.md)
  - [Glossary](09-appendices/04-glossary.md)

